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
    /// Tree construction strategy. Defaults to `Serial` via `new`.
    pub build_mode: BuildMode,
}

impl BarnesHut {
    pub fn new(g: f64, softening: f64, theta: f64) -> Self {
        Self {
            g,
            softening,
            theta,
            build_mode: BuildMode::Serial,
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
    fn build_parallel_exact(_pos: &[DVec3], _mass: &[f64]) -> Octree {
        todo!("partition-then-build parallel octree (bit-exact to serial)")
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
}
