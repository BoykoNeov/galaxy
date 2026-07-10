use galaxy_core::{DVec3, ForceSolver, State};
use rayon::prelude::*;

/// How the octree is constructed. The build was the largest serial fraction once
/// the force fill and potential were parallelized (see DESIGN M2/perf note).
///
/// Both modes yield a tree that produces **bit-identical** forces: topology is
/// pure geometry (deterministic octant tests; bbox min/max is associative+exact),
/// leaf occupancy is order-independent, and the aggregate COM/mass sums are done
/// in the same order (bodies ascending by original index, children in octant
/// order). `ParallelExact` therefore is not a tolerance trade — it is the serial
/// tree, built in parallel. A tolerance-only Morton bottom-up mode is deferred
/// pending a benchmark showing `ParallelExact` leaves speedup on the table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildMode {
    /// Sequential build. The reference/oracle; retained for single-thread runs
    /// and as the equivalence guard for `ParallelExact`.
    Serial,
    /// Partition bodies into disjoint subtrees (reusing the serial octant/child
    /// predicates), build each subtree in parallel into its own arena, then
    /// stitch. Sidesteps concurrent insertion into a shared tree.
    ParallelExact,
}

/// Below this particle count, `ParallelExact` falls back to the serial build:
/// the partition/stitch/rayon overhead is not worth it for tiny trees.
const PARALLEL_BUILD_MIN: usize = 512;

/// Barnes-Hut octree solver (monopole-only). An O(N log N) approximation to
/// direct summation, controlled by the opening angle `theta`: a node is used as
/// a single softened point mass when `node_size / distance < theta`, else its
/// children are opened. As `theta -> 0` it reproduces direct summation (same
/// Plummer-softened kernel) to roundoff. Quadrupole terms are a later refinement.
#[derive(Clone, Copy, Debug)]
pub struct BarnesHut {
    /// Gravitational constant.
    pub g: f64,
    /// Plummer softening length ε (use the same value as the oracle to compare).
    pub softening: f64,
    /// Opening angle θ. Smaller = more accurate, more work.
    pub theta: f64,
    /// Tree construction strategy. Defaults to `ParallelExact` via `new` — it is
    /// bit-exact to `Serial`, so this changes speed only, never results.
    pub build_mode: BuildMode,
}

impl BarnesHut {
    pub fn new(g: f64, softening: f64, theta: f64) -> Self {
        Self {
            g,
            softening,
            theta,
            // Parallel by default: bit-exact to serial, self-falls-back below
            // PARALLEL_BUILD_MIN, and consistent with the already-parallel force
            // fill. Opt back into Serial via `with_build_mode` for single-thread
            // debugging (it remains the equivalence oracle).
            build_mode: BuildMode::ParallelExact,
        }
    }

    /// Select the tree construction strategy (builder-style; `BarnesHut` is `Copy`).
    pub fn with_build_mode(mut self, mode: BuildMode) -> Self {
        self.build_mode = mode;
        self
    }
}

const SENTINEL: u32 = u32::MAX;
const LEAF_CAP: usize = 1;
/// Stop subdividing once a cell is this fraction of the root size; coincident or
/// near-coincident particles then share a bucket leaf (resolved by direct sum).
const MIN_HALF_FRAC: f64 = 1e-10;

struct Node {
    center: DVec3,
    half: f64,
    mass: f64,
    com: DVec3,
    /// Distance from the geometric center to the center of mass. The COM can sit
    /// near a cell edge ("detached"), so the nearest mass may be `delta` closer to
    /// a target than the COM is; the opening criterion subtracts it (Barnes 1994).
    delta: f64,
    children: [u32; 8],
    bodies: Vec<u32>,
    leaf: bool,
}

impl Node {
    fn leaf(center: DVec3, half: f64) -> Self {
        Node {
            center,
            half,
            mass: 0.0,
            com: DVec3::ZERO,
            delta: 0.0,
            children: [SENTINEL; 8],
            bodies: Vec::new(),
            leaf: true,
        }
    }
}

fn octant(center: DVec3, p: DVec3) -> usize {
    let mut o = 0;
    if p.x >= center.x {
        o |= 1;
    }
    if p.y >= center.y {
        o |= 2;
    }
    if p.z >= center.z {
        o |= 4;
    }
    o
}

fn child_center(center: DVec3, half: f64, oct: usize) -> DVec3 {
    let q = 0.5 * half;
    DVec3::new(
        center.x + if oct & 1 != 0 { q } else { -q },
        center.y + if oct & 2 != 0 { q } else { -q },
        center.z + if oct & 4 != 0 { q } else { -q },
    )
}

/// Cells with more than this many bodies fan their (up to 8) child builds out
/// across rayon; smaller cells build their children inline. Because dense regions
/// subdivide more, the split count adapts to the actual density — a clustered
/// collision IC produces many tasks in the core, not one giant serial subtree.
const BUILD_FANOUT_MIN: usize = 512;

/// Build the subtree for one cell (`center`, `half`) over `bodies` (which MUST be
/// ascending by original index) and return it as a self-contained arena whose
/// root — fully aggregated `(mass, com, delta)` — is at index 0.
///
/// This reproduces the serial build's tree exactly: it uses the same `octant`
/// split, the same subdivide rule (`len > LEAF_CAP && half > min_half`), keeps
/// bodies ascending within every bucket, and folds the aggregate bottom-up in
/// octant order (children 0..8) — so nothing the force traversal reads is ever
/// reassociated. Node *arena indices* differ from the serial layout, but the
/// reachable structure and every per-node value are bit-identical.
fn build_cell(
    center: DVec3,
    half: f64,
    min_half: f64,
    bodies: Vec<u32>,
    pos: &[DVec3],
    mass: &[f64],
) -> Vec<Node> {
    // Leaf: too few bodies to split, or the cell has shrunk to the coincidence
    // floor (a bucket of near-identical points, resolved later by direct sum).
    if bodies.len() <= LEAF_CAP || half <= min_half {
        let mut node = Node::leaf(center, half);
        let (m, c) = bodies.iter().fold((0.0, DVec3::ZERO), |(m, c), &b| {
            (m + mass[b as usize], c + pos[b as usize] * mass[b as usize])
        });
        node.mass = m;
        node.com = if m > 0.0 { c / m } else { center };
        node.delta = (node.com - center).length();
        node.bodies = bodies;
        return vec![node];
    }

    // Internal: partition into octants, preserving ascending body order per bucket.
    let mut groups: [Vec<u32>; 8] = Default::default();
    for b in bodies.iter().copied() {
        groups[octant(center, pos[b as usize])].push(b);
    }
    let child_half = 0.5 * half;
    // Occupied octants (in ascending order) paired with their bodies.
    let tasks: Vec<(usize, Vec<u32>)> = groups
        .into_iter()
        .enumerate()
        .filter(|(_, g)| !g.is_empty())
        .collect();

    // Build each occupied child's arena. Fan out only for large cells; small ones
    // build inline to avoid task overhead. Either way the results are reassembled
    // in ascending octant order below, so the arena layout is deterministic.
    let n_here: usize = tasks.iter().map(|(_, g)| g.len()).sum();
    let build = |(oct, g): (usize, Vec<u32>)| {
        let cc = child_center(center, half, oct);
        (oct, build_cell(cc, child_half, min_half, g, pos, mass))
    };
    let built: Vec<(usize, Vec<Node>)> = if n_here > BUILD_FANOUT_MIN {
        tasks.into_par_iter().map(build).collect()
    } else {
        tasks.into_iter().map(build).collect()
    };

    // Splice children into one arena and aggregate the root over them in octant
    // order — the exact summation order the serial `aggregate` uses.
    let mut nodes: Vec<Node> = vec![Node::leaf(center, half)];
    nodes[0].leaf = false;
    let mut m = 0.0;
    let mut c = DVec3::ZERO;
    for (oct, arena) in built {
        let base = nodes.len() as u32;
        let (cm, ccom) = (arena[0].mass, arena[0].com);
        for mut node in arena {
            for slot in node.children.iter_mut() {
                if *slot != SENTINEL {
                    *slot += base;
                }
            }
            nodes.push(node);
        }
        nodes[0].children[oct] = base; // child root landed at `base`
        m += cm;
        c += ccom * cm;
    }
    nodes[0].mass = m;
    nodes[0].com = if m > 0.0 { c / m } else { center };
    nodes[0].delta = (nodes[0].com - center).length();
    nodes
}

struct Octree {
    nodes: Vec<Node>,
    min_half: f64,
}

impl Octree {
    fn build(pos: &[DVec3], mass: &[f64], mode: BuildMode) -> Octree {
        match mode {
            BuildMode::Serial => Octree::build_serial(pos, mass),
            BuildMode::ParallelExact => Octree::build_parallel_exact(pos, mass),
        }
    }

    /// Parallel build that reproduces the serial tree bit-for-bit (topology,
    /// leaf body ordering, and aggregate summation order all preserved). See
    /// [`BuildMode::ParallelExact`].
    ///
    /// Strategy: the root bounding box is a rayon min/max reduction (associative
    /// and exact, so bit-identical to the serial fold). The tree is then built by
    /// [`build_cell`], which recursively partitions bodies by the *same* octant
    /// predicates the serial insert uses and assembles each cell bottom-up —
    /// fanning its 8 child builds out across rayon when the cell is large. Each
    /// subtree is built into its own arena (no shared-tree mutation, so no
    /// concurrent-insertion hazard) and spliced with a child-pointer offset remap.
    fn build_parallel_exact(pos: &[DVec3], mass: &[f64]) -> Octree {
        // Small trees don't amortize the partition/rayon overhead — the serial
        // build (which this must match exactly anyway) is faster below the cutoff.
        if pos.len() < PARALLEL_BUILD_MIN {
            return Octree::build_serial(pos, mass);
        }
        // Root bounding cube — identical to `build_serial`'s, since componentwise
        // min/max is associative + exact: the parallel reduction returns the same
        // lo/hi bits regardless of how rayon splits the range.
        let (lo, hi) = pos.par_iter().map(|&p| (p, p)).reduce(
            || (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)),
            |(alo, ahi), (blo, bhi)| (alo.min(blo), ahi.max(bhi)),
        );
        let center = (lo + hi) * 0.5;
        let half = (0.5 * (hi - lo).max_element()).max(1e-12) * (1.0 + 1e-9);
        let min_half = half * MIN_HALF_FRAC;
        let bodies: Vec<u32> = (0..pos.len() as u32).collect();
        let nodes = build_cell(center, half, min_half, bodies, pos, mass);
        Octree { nodes, min_half }
    }

    fn build_serial(pos: &[DVec3], mass: &[f64]) -> Octree {
        let mut lo = pos[0];
        let mut hi = pos[0];
        for &p in pos {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let center = (lo + hi) * 0.5;
        // Cube half-side that strictly contains every particle (slight padding).
        let half = (0.5 * (hi - lo).max_element()).max(1e-12) * (1.0 + 1e-9);
        let mut tree = Octree {
            nodes: vec![Node::leaf(center, half)],
            min_half: half * MIN_HALF_FRAC,
        };
        for (i, _) in pos.iter().enumerate() {
            tree.insert(0, i as u32, pos);
        }
        tree.aggregate(0, pos, mass);
        tree
    }

    fn insert(&mut self, node: usize, b: u32, pos: &[DVec3]) {
        if self.nodes[node].leaf {
            self.nodes[node].bodies.push(b);
            if self.nodes[node].bodies.len() > LEAF_CAP && self.nodes[node].half > self.min_half {
                self.subdivide(node, pos);
            }
        } else {
            let oct = octant(self.nodes[node].center, pos[b as usize]);
            let child = self.ensure_child(node, oct);
            self.insert(child, b, pos);
        }
    }

    fn subdivide(&mut self, node: usize, pos: &[DVec3]) {
        let bodies = std::mem::take(&mut self.nodes[node].bodies);
        self.nodes[node].leaf = false;
        for b in bodies {
            let oct = octant(self.nodes[node].center, pos[b as usize]);
            let child = self.ensure_child(node, oct);
            self.insert(child, b, pos);
        }
    }

    fn ensure_child(&mut self, node: usize, oct: usize) -> usize {
        let existing = self.nodes[node].children[oct];
        if existing != SENTINEL {
            return existing as usize;
        }
        let center = child_center(self.nodes[node].center, self.nodes[node].half, oct);
        let half = 0.5 * self.nodes[node].half;
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node::leaf(center, half));
        self.nodes[node].children[oct] = idx;
        idx as usize
    }

    fn aggregate(&mut self, node: usize, pos: &[DVec3], mass: &[f64]) {
        let mut m = 0.0;
        let mut c = DVec3::ZERO;
        if self.nodes[node].leaf {
            for &b in &self.nodes[node].bodies {
                m += mass[b as usize];
                c += pos[b as usize] * mass[b as usize];
            }
        } else {
            let children = self.nodes[node].children;
            for &ch in &children {
                if ch != SENTINEL {
                    self.aggregate(ch as usize, pos, mass);
                    m += self.nodes[ch as usize].mass;
                    c += self.nodes[ch as usize].com * self.nodes[ch as usize].mass;
                }
            }
        }
        self.nodes[node].mass = m;
        let com = if m > 0.0 {
            c / m
        } else {
            self.nodes[node].center
        };
        self.nodes[node].com = com;
        self.nodes[node].delta = (com - self.nodes[node].center).length();
    }

    fn accel_node(&self, node: usize, target: usize, q: &Query) -> DVec3 {
        let nd = &self.nodes[node];
        if nd.mass <= 0.0 {
            return DVec3::ZERO;
        }
        let xp = q.pos[target];
        if nd.leaf {
            let mut a = DVec3::ZERO;
            for &b in &nd.bodies {
                if b as usize == target {
                    continue;
                }
                let dx = q.pos[b as usize] - xp;
                let r2 = dx.length_squared() + q.eps2;
                a += dx * (q.mass[b as usize] / (r2 * r2.sqrt()));
            }
            return a;
        }
        // Never approximate a cell that contains the target as a single mass.
        let inside = (xp.x - nd.center.x).abs() <= nd.half
            && (xp.y - nd.center.y).abs() <= nd.half
            && (xp.z - nd.center.z).abs() <= nd.half;
        let dx = nd.com - xp;
        let d2 = dx.length_squared();
        let s = 2.0 * nd.half;
        // Barnes (1994) criterion: accept the multipole when the cell subtends a
        // small angle from the *nearest mass it could hold*, not just from the COM.
        // The nearest mass may be `delta` closer than the COM, so require
        //   s / (d − delta) ≤ θ   ⟺   s ≤ θ·(d − delta),
        // which reduces to the classic s/d ≤ θ for a centered COM (delta = 0) and
        // forces opening when a detached COM would otherwise hide a near particle.
        let d = d2.sqrt();
        if !inside && q.theta * (d - nd.delta) >= s {
            let r2 = d2 + q.eps2;
            return dx * (nd.mass / (r2 * r2.sqrt()));
        }
        let mut a = DVec3::ZERO;
        for &ch in &nd.children {
            if ch != SENTINEL {
                a += self.accel_node(ch as usize, target, q);
            }
        }
        a
    }
}

/// Read-only context threaded through the force traversal.
struct Query<'a> {
    pos: &'a [DVec3],
    mass: &'a [f64],
    theta: f64,
    eps2: f64,
}

impl BarnesHut {
    /// Serial reference fill: the sequential, single-threaded force evaluation.
    /// The parallel trait `accelerations` must reproduce this **bit-for-bit** —
    /// each target's traversal is independent, so parallelizing over targets only
    /// reorders *which* `acc[i]` is written when, never the ops inside one `acc[i]`.
    /// Retained as the equivalence-guard oracle (and for single-thread debugging).
    #[allow(clippy::needless_range_loop)]
    pub fn accelerations_serial(&self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        // Oracle path: always the serial build, regardless of `self.build_mode`.
        let tree = Octree::build(&state.pos, &state.mass, BuildMode::Serial);
        let q = Query {
            pos: &state.pos,
            mass: &state.mass,
            theta: self.theta,
            eps2: self.softening * self.softening,
        };
        for i in 0..n {
            acc[i] = tree.accel_node(0, i, &q) * self.g;
        }
    }
}

impl ForceSolver for BarnesHut {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        // Build the tree once, then only read it. `accel_node` is `&self` and each
        // target writes exactly its own `acc[i]`, so the fill is a pure map over
        // independent targets — parallelizing it is data-race-free and bit-exact to
        // the serial reference (no per-target sum is reassociated). The build itself
        // may also be parallelized (`build_mode`), and any `BuildMode` yields a
        // bit-identical tree, so the whole `accelerations` path stays bit-exact.
        let tree = Octree::build(&state.pos, &state.mass, self.build_mode);
        let q = Query {
            pos: &state.pos,
            mass: &state.mass,
            theta: self.theta,
            eps2: self.softening * self.softening,
        };
        let g = self.g;
        acc.par_iter_mut().enumerate().for_each(|(i, a)| {
            *a = tree.accel_node(0, i, &q) * g;
        });
    }

    fn potential_energy(&self, state: &State) -> f64 {
        // Exact softened potential (O(N²)); a global diagnostic, not tree-
        // accelerated, and identical to DirectSum's potential.
        crate::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}

impl BarnesHut {
    /// Serial reference for the exact softened potential — the tolerance oracle
    /// for the parallel reduction. Shares the kernel with DirectSum (`potential`
    /// module), so the two solvers' potentials stay identical by construction.
    pub fn potential_energy_serial(&self, state: &State) -> f64 {
        crate::potential::potential_energy_serial(state, self.g, self.softening)
    }
}

/// One node of a [`FlatTree`] — the Barnes-Hut octree linearized into a flat
/// array for **stackless** traversal (the GPU-friendly form; a GPU has no
/// recursion stack). Same geometry the recursive [`Octree`] node carries, plus a
/// `next` skip pointer and a leaf body range.
#[derive(Clone, Copy, Debug)]
pub struct FlatNode {
    /// Cell geometric center.
    pub center: DVec3,
    /// Cell half-side.
    pub half: f64,
    /// Aggregate center of mass.
    pub com: DVec3,
    /// Aggregate mass.
    pub mass: f64,
    /// |com − center| (the Barnes 1994 opening-criterion correction).
    pub delta: f64,
    /// **Skip pointer**: the index one past this node's entire subtree in DFS
    /// pre-order. Traversal that does NOT open this node jumps here; opening a
    /// node instead advances to `self_index + 1` (its first child, which — by the
    /// DFS layout — immediately follows it). Because `next > self_index` always
    /// and opening strictly increments, the flat walk's node index strictly
    /// increases every step, so it always terminates (no stack, no cycle).
    pub next: u32,
    /// For a leaf: start offset into [`FlatTree::leaf_bodies`]. Unused (0) for
    /// internal nodes.
    pub body_start: u32,
    /// For a leaf: number of bodies. **`body_count > 0` iff the node is a leaf** —
    /// every leaf holds ≥1 body and no internal node holds bodies directly, so this
    /// is the leaf test (no separate flag needed).
    pub body_count: u32,
}

/// The Barnes-Hut octree flattened into a DFS pre-order array with skip pointers —
/// a **stackless** representation a GPU kernel can walk with a single index and no
/// recursion. Precision-agnostic (f64 geometry); a GPU consumer narrows to f32.
///
/// The traversal contract mirrors the recursive [`Octree::accel_node`] exactly:
/// children are laid out in ascending octant order (0..8), so the flat walk
/// accumulates contributions in the *same* order the recursion does — hence
/// [`FlatTree::accel`] reproduces `accel_node` **bit-for-bit** in f64 (guarded by a
/// unit test). The GPU kernel is the f32 mirror of that same walk.
#[derive(Clone)]
pub struct FlatTree {
    /// DFS pre-order nodes; the root is index 0.
    pub nodes: Vec<FlatNode>,
    /// Concatenated leaf body indices (original particle indices), sliced per leaf
    /// by `body_start` / `body_count`.
    pub leaf_bodies: Vec<u32>,
}

impl FlatTree {
    /// Build the octree (bit-identical `ParallelExact` tree) and linearize it into
    /// DFS pre-order with skip pointers. Children are emitted in ascending octant
    /// order so the flat accumulation order matches [`Octree::accel_node`].
    ///
    /// `pos` must be non-empty (an empty system has no root cell — the caller
    /// handles N=0 before building, as the GPU solver does).
    pub fn build(pos: &[DVec3], mass: &[f64]) -> FlatTree {
        // Bit-identical `ParallelExact` tree, then a DFS pre-order linearization.
        let tree = Octree::build(pos, mass, BuildMode::ParallelExact);
        let mut nodes = Vec::with_capacity(tree.nodes.len());
        let mut leaf_bodies = Vec::new();
        flatten_node(&tree, 0, &mut nodes, &mut leaf_bodies);
        FlatTree { nodes, leaf_bodies }
    }

    /// f64 reference traversal: the acceleration on `target` from a stackless walk
    /// of the flat tree, using the Barnes (1994) opening criterion with opening
    /// angle `theta` and softening `eps2 = ε²`. Excludes the self term. This is the
    /// exact f64 analogue of the GPU kernel and is asserted bit-identical to
    /// [`Octree::accel_node`]; the returned value still needs `× g` by the caller,
    /// matching `accel_node`'s convention.
    pub fn accel(
        &self,
        target: usize,
        pos: &[DVec3],
        mass: &[f64],
        theta: f64,
        eps2: f64,
    ) -> DVec3 {
        let xp = pos[target];
        let mut a = DVec3::ZERO;
        let mut node = 0u32;
        let n_nodes = self.nodes.len() as u32;
        // Stackless walk: `node` strictly increases (open → node+1, otherwise →
        // next > node), so this terminates in ≤ n_nodes steps with no stack.
        while node < n_nodes {
            let nd = &self.nodes[node as usize];
            if nd.mass <= 0.0 {
                node = nd.next;
                continue;
            }
            if nd.body_count > 0 {
                // Leaf: exact direct sum over its bodies, excluding the self term.
                let end = nd.body_start + nd.body_count;
                for k in nd.body_start..end {
                    let b = self.leaf_bodies[k as usize] as usize;
                    if b == target {
                        continue;
                    }
                    let dx = pos[b] - xp;
                    let r2 = dx.length_squared() + eps2;
                    a += dx * (mass[b] / (r2 * r2.sqrt()));
                }
                node = nd.next;
            } else {
                // Internal: never approximate a cell that contains the target.
                let inside = (xp.x - nd.center.x).abs() <= nd.half
                    && (xp.y - nd.center.y).abs() <= nd.half
                    && (xp.z - nd.center.z).abs() <= nd.half;
                let dx = nd.com - xp;
                let d2 = dx.length_squared();
                let s = 2.0 * nd.half;
                let d = d2.sqrt();
                // Barnes (1994): accept the monopole when s ≤ θ·(d − delta).
                if !inside && theta * (d - nd.delta) >= s {
                    let r2 = d2 + eps2;
                    a += dx * (nd.mass / (r2 * r2.sqrt()));
                    node = nd.next;
                } else {
                    node += 1; // open: descend to the first child
                }
            }
        }
        a
    }
}

/// Recursively emit `tree`'s subtree rooted at `node` into DFS pre-order, filling
/// each [`FlatNode`]'s `next` skip pointer once its whole subtree has been laid down.
/// Children are visited in ascending octant order (0..8) — the same order
/// [`Octree::accel_node`] recurses — so [`FlatTree::accel`] accumulates identically.
fn flatten_node(tree: &Octree, node: usize, out: &mut Vec<FlatNode>, bodies: &mut Vec<u32>) {
    let nd = &tree.nodes[node];
    let me = out.len();
    out.push(FlatNode {
        center: nd.center,
        half: nd.half,
        com: nd.com,
        mass: nd.mass,
        delta: nd.delta,
        next: 0, // patched after the subtree is emitted
        body_start: 0,
        body_count: 0,
    });
    if nd.leaf {
        let start = bodies.len() as u32;
        bodies.extend_from_slice(&nd.bodies);
        out[me].body_start = start;
        out[me].body_count = nd.bodies.len() as u32;
    } else {
        // First child lands at `me + 1`; subsequent children follow in octant order.
        for &ch in &nd.children {
            if ch != SENTINEL {
                flatten_node(tree, ch as usize, out, bodies);
            }
        }
    }
    // The next node after this whole subtree — the skip target when not opened.
    out[me].next = out.len() as u32;
}

/// [`BarnesHut`] plus a cached [`FlatTree`] for the I-grav stale-tree active gravity
/// walk (`hydro+gravity` mode). [`BarnesHut`] is `Copy` and cannot hold the (owned,
/// `Vec`-backed) tree, so the cache lives in this wrapper. As a [`ForceSolver`] its
/// full `accelerations` delegates to the wrapped solver bit-for-bit (so any run that
/// only calls `accelerations` is byte-identical to using `BarnesHut` directly); the
/// stale-tree methods add the active-subset walk.
///
/// Lifecycle: the individual stepper calls [`rebuild_gravity_cache`](ForceSolver::rebuild_gravity_cache)
/// ONCE per base block (freezing the tree topology + cell multipoles at the block-start
/// positions), then [`gravity_active_cached`](ForceSolver::gravity_active_cached) every
/// fine tick at the CURRENT positions. Near-field leaf sources are read from the current
/// `state.pos` (drift-all keeps them exact); only the far cell COMs are stale — a
/// bounded, converging approximation validated by the stale-vs-rebuild convergence gate.
#[derive(Clone)]
pub struct TreeGravity {
    /// The wrapped monopole Barnes-Hut solver (g / softening / theta / build_mode).
    pub bh: BarnesHut,
    /// The tree frozen at the last `rebuild_gravity_cache` (block start). `None`
    /// until the first rebuild.
    cache: Option<FlatTree>,
}

impl TreeGravity {
    /// Wrap a [`BarnesHut`] with an (initially empty) stale-tree cache.
    pub fn new(bh: BarnesHut) -> Self {
        TreeGravity { bh, cache: None }
    }
}

impl ForceSolver for TreeGravity {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        // Full fresh pass — byte-identical to the wrapped BarnesHut.
        self.bh.accelerations(state, acc);
    }

    fn potential_energy(&self, state: &State) -> f64 {
        self.bh.potential_energy(state)
    }

    fn rebuild_gravity_cache(&mut self, state: &State) {
        // Freeze the tree (topology + cell multipoles) at the block-start positions.
        // An empty system has no root cell (FlatTree::build requires N>0) — clear the
        // cache so a later walk on a still-empty state is a harmless no-op.
        self.cache = if state.is_empty() {
            None
        } else {
            Some(FlatTree::build(&state.pos, &state.mass))
        };
    }

    fn gravity_active_cached(&mut self, state: &State, active: &[usize], acc: &mut [DVec3]) {
        assert_eq!(
            acc.len(),
            state.len(),
            "acc length must match particle count"
        );
        let Some(tree) = self.cache.as_ref() else {
            // No cache (never rebuilt, or empty state) — nothing to walk.
            return;
        };
        let (theta, eps2, g) = (
            self.bh.theta,
            self.bh.softening * self.bh.softening,
            self.bh.g,
        );
        // Walk the STALE tree at the CURRENT positions: near-field leaf sources read
        // `state.pos`/`state.mass` (exact — drift-all), far cells use the frozen COMs.
        // Each target writes only its own `acc[i]`, so the fill is a race-free pure map
        // and is bit-identical to `BarnesHut::accelerations` when the tree is fresh and
        // every target is active (FlatTree::accel ≡ Octree::accel_node).
        let contribs: Vec<(usize, DVec3)> = active
            .par_iter()
            .map(|&i| (i, tree.accel(i, &state.pos, &state.mass, theta, eps2) * g))
            .collect();
        for (i, a) in contribs {
            acc[i] = a;
        }
    }
}

#[cfg(test)]
mod build_tests {
    //! `ParallelExact` must build the *same* tree as the serial reference — not
    //! "close enough", but topology + per-node `(mass, com, delta)` bit-for-bit.
    //! Force-equivalence is tested at the integration level (`barnes_hut_parallel`);
    //! these unit tests can see the private `Octree` and pin the internals directly,
    //! so a stitching off-by-one (children offset remap across concatenated arenas)
    //! can't hide behind forces that happen to come out equal.
    //!
    //! The comparison is a lockstep traversal from the root in octant order — NOT
    //! an arena-index compare — because a parallel build legitimately assigns node
    //! indices in a different order; only the reachable structure must match.

    use super::*;

    /// Deterministic point cloud (LCG, no rand dep). `clustered` concentrates most
    /// bodies into a tight blob with a sparse halo, so the build's load balance /
    /// adaptive frontier is exercised against the non-uniform density that real
    /// galaxy-collision ICs produce.
    fn cloud(seed: u64, n: usize, clustered: bool) -> (Vec<DVec3>, Vec<f64>) {
        let mut s = seed | 1;
        let mut next = move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) // [0, 1)
        };
        let mut pos = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        for i in 0..n {
            let p = if clustered && i % 8 != 0 {
                // Tight core near the origin.
                DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 0.05
            } else {
                // Uniform spread (or the sparse halo of the clustered case).
                DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 3.0
            };
            pos.push(p);
            mass.push(0.1 + 0.9 * next());
        }
        (pos, mass)
    }

    /// Assert two trees have bit-identical structure by walking both from `node`
    /// in octant order. Compares leaf/internal status, leaf body lists, and the
    /// aggregated `(mass, com, delta)` to the last bit — the exact quantities the
    /// force traversal reads.
    fn assert_same_node(a: &Octree, an: usize, b: &Octree, bn: usize, path: &str) {
        let na = &a.nodes[an];
        let nb = &b.nodes[bn];
        assert_eq!(na.leaf, nb.leaf, "leaf flag differs at {path}");
        assert_eq!(na.bodies, nb.bodies, "leaf bodies differ at {path}");
        assert_eq!(
            na.mass.to_bits(),
            nb.mass.to_bits(),
            "mass not bit-exact at {path}"
        );
        assert_eq!(
            na.com.to_array().map(f64::to_bits),
            nb.com.to_array().map(f64::to_bits),
            "com not bit-exact at {path}"
        );
        assert_eq!(
            na.delta.to_bits(),
            nb.delta.to_bits(),
            "delta not bit-exact at {path}"
        );
        assert_eq!(
            na.half.to_bits(),
            nb.half.to_bits(),
            "half not bit-exact at {path}"
        );
        assert_eq!(
            na.center.to_array().map(f64::to_bits),
            nb.center.to_array().map(f64::to_bits),
            "center not bit-exact at {path}"
        );
        for oct in 0..8 {
            let ca = na.children[oct];
            let cb = nb.children[oct];
            assert_eq!(
                ca == SENTINEL,
                cb == SENTINEL,
                "child presence differs at {path}, octant {oct}"
            );
            if ca != SENTINEL {
                assert_same_node(a, ca as usize, b, cb as usize, &format!("{path}/{oct}"));
            }
        }
    }

    #[test]
    fn parallel_exact_build_is_bit_identical_to_serial() {
        // Sizes straddle PARALLEL_BUILD_MIN so both the small-N fallback and the
        // genuinely-partitioned path are covered.
        for &n in &[600usize, 1024, 4096] {
            for seed in 0..8u64 {
                for &clustered in &[false, true] {
                    let (pos, mass) = cloud(seed, n, clustered);
                    let ser = Octree::build(&pos, &mass, BuildMode::Serial);
                    let par = Octree::build(&pos, &mass, BuildMode::ParallelExact);
                    assert_eq!(
                        ser.nodes.len(),
                        par.nodes.len(),
                        "node count differs (n={n}, seed={seed}, clustered={clustered})"
                    );
                    assert_same_node(&ser, 0, &par, 0, "root");
                }
            }
        }
    }

    /// The stackless [`FlatTree`] walk must reproduce the recursive
    /// [`Octree::accel_node`] to reassociation precision — the flatten (DFS layout,
    /// `next` skip pointers, leaf `body_start`/`body_count` ranges, first-child-at
    /// `node+1`) is the genuinely new, off-by-one-prone logic, and the GPU f32 gates
    /// can't isolate it (threshold-straddle + cancellation noise hide a flatten bug
    /// behind tolerances).
    ///
    /// **Not bit-exact — reassociated, like `potential_energy_parallel`.**
    /// `accel_node` folds each *subtree* into its own accumulator then combines them
    /// (a tree reduction); the stackless walk keeps one running accumulator over the
    /// DFS scan. Both are valid summation orders and differ at the ULP level, so the
    /// force compare is a tight *relative* bound (≪ any topological error — a dropped
    /// or double-counted subtree is orders larger). Topology is separately pinned
    /// **exactly** by the structural checks: reachable node count matches, and the
    /// flattened leaf bodies are a permutation of `0..n` (every particle in exactly
    /// one leaf — catches a dropped/duplicated `body_start`/`count` range). Together
    /// these also foreclose the GPU-hang hazard: a strictly-increasing walk (open→
    /// node+1, accept→next>node over a correct flatten) cannot spin.
    ///
    /// Sampled targets (stride) per config — a flatten off-by-one perturbs many
    /// targets — over uniform + clustered clouds, sizes straddling the build cutoffs,
    /// and θ from the full-open (θ→0) regime through wide angles.
    #[test]
    fn flat_tree_traversal_matches_accel_node() {
        let eps2 = 0.05f64 * 0.05;
        let mut worst_rel = 0.0f64;
        for &n in &[1usize, 2, 5, 37, 120, 600, 1024, 4096] {
            for seed in 0..8u64 {
                for &clustered in &[false, true] {
                    let (pos, mass) = cloud(seed, n, clustered);
                    let tree = Octree::build(&pos, &mass, BuildMode::Serial);
                    let flat = FlatTree::build(&pos, &mass);

                    // Exact structural invariants (order-independent):
                    assert_eq!(
                        tree.nodes.len(),
                        flat.nodes.len(),
                        "flat node count differs (n={n}, seed={seed}, clustered={clustered})"
                    );
                    let mut seen = flat.leaf_bodies.clone();
                    seen.sort_unstable();
                    assert_eq!(
                        seen,
                        (0..n as u32).collect::<Vec<_>>(),
                        "leaf bodies must be a permutation of 0..n \
                         (n={n}, seed={seed}, clustered={clustered})"
                    );

                    let stride = (n / 64).max(1);
                    for &theta in &[1e-6f64, 0.3, 0.6, 1.0] {
                        let q = Query {
                            pos: &pos,
                            mass: &mass,
                            theta,
                            eps2,
                        };
                        for t in (0..n).step_by(stride) {
                            let want = tree.accel_node(0, t, &q);
                            let got = flat.accel(t, &pos, &mass, theta, eps2);
                            let rel = (got - want).length() / want.length().max(1e-300);
                            worst_rel = worst_rel.max(rel);
                            assert!(
                                rel < 1e-11,
                                "flat walk must match accel_node to reassociation precision \
                                 (n={n}, seed={seed}, clustered={clustered}, theta={theta}, \
                                 target={t}): rel {rel:e}"
                            );
                        }
                    }
                }
            }
        }
        // Observed worst relative reassociation gap — far under the 1e-11 bound and
        // astronomically under any topological error. Printed with --nocapture.
        println!("flat-vs-recursive worst relative reassociation gap: {worst_rel:e}");
    }

    /// Wall-clock build cost: serial vs `ParallelExact`, on both a uniform and a
    /// clustered cloud. Ignored (timing, machine-dependent) — run with
    /// `cargo test -p galaxy-solvers --release -- --ignored --nocapture bench_build`.
    ///
    /// This is the gate for the deferred tolerance/Morton mode: if `ParallelExact`
    /// already saturates the available speedup here, a second (lossy) build
    /// algorithm isn't worth the codepath. Best-of-k to damp scheduler noise.
    #[test]
    #[ignore = "timing benchmark; run explicitly with --ignored --nocapture"]
    fn bench_build() {
        use std::time::Instant;
        let best = |pos: &[DVec3], mass: &[f64], mode: BuildMode| {
            let mut b = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                let tree = Octree::build(pos, mass, mode);
                let dt = t.elapsed().as_secs_f64();
                std::hint::black_box(tree.nodes.len());
                b = b.min(dt);
            }
            b * 1e3 // ms
        };
        for &n in &[100_000usize, 500_000, 1_000_000] {
            for &clustered in &[false, true] {
                let (pos, mass) = cloud(0xB0A7, n, clustered);
                let s = best(&pos, &mass, BuildMode::Serial);
                let p = best(&pos, &mass, BuildMode::ParallelExact);
                let tag = if clustered { "clustered" } else { "uniform  " };
                println!(
                    "N={n:>8} {tag}  serial {s:8.2} ms   parallel {p:8.2} ms   speedup {:5.2}x",
                    s / p
                );
            }
        }
    }
}
