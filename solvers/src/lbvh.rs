//! [`Lbvh`]: a Barnes-Hut force solver built on a Morton-code **Linear BVH**
//! (Karras 2012 binary radix tree) instead of the octree — the CPU f64 reference
//! for the future **GPU-resident** Morton/LBVH build.
//!
//! ## Why this exists
//! [`crate::BarnesHut`] / [`crate::FlatTree`] build an *octree* on the CPU; the GPU
//! [`galaxy_gpu::GpuTree`](../../galaxy_gpu/index.html) only *traverses* it. The next
//! scaling step (DESIGN M4) is a build that runs entirely on the GPU. The GPU-shaped
//! build algorithm is the LBVH pipeline: bounding box → Morton codes → sort → Karras
//! binary radix tree → bottom-up aggregation. This module implements that pipeline in
//! **CPU f64**, exactly mirroring how [`crate::FlatTree`]'s CPU f64 walk is the oracle
//! for the GPU traversal: it is the algorithmic + numerical reference each future GPU
//! stage (Morton kernel, GPU sort, tree-build kernel, aggregation) is gated against.
//!
//! ## The tree is a *binary* radix tree, not an octree
//! Karras gives exactly `N` leaves (one per particle, in Morton-sorted order) and
//! `N-1` internal nodes = `2N-1` nodes total, each internal node with exactly two
//! children. A node is a leaf iff `body_count > 0` (always 1 here). Opening uses the
//! same Barnes (1994) form as the octree walk, but the cell size is the node's AABB
//! longest side `s = max(2·half_extents)` rather than an octree cube's `2·half`.
//!
//! ## Determinism
//! Morton ties (particles in the same 1024³ cell, or coincident) are broken by
//! original index in the sort *and* in the Karras `δ` prefix test, so the tree
//! topology — and therefore the forces — are a deterministic function of the input.

use galaxy_core::{DVec3, ForceSolver, State};

/// Morton grid resolution per axis: 10 bits ⇒ 1024³ cells ⇒ a 30-bit `u32` code.
/// 63-bit (2¹ per axis, two-word sort on the GPU) is the deferred resolution
/// refinement for the dense-core / large-coordinate regime.
const MORTON_BITS: u32 = 10;

/// Spread the low 10 bits of `v` so consecutive bits sit 3 apart (`b9…b0` →
/// `b9 0 0 b8 0 0 … b0`), the per-axis step of a 3D Morton interleave. Bits above
/// bit 9 are discarded by the initial mask.
fn expand10(v: u32) -> u32 {
    let mut x = v & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Interleave three 10-bit lane coordinates into one 30-bit Morton code:
/// `expand10(x) | expand10(y)<<1 | expand10(z)<<2`.
fn morton3(x: u32, y: u32, z: u32) -> u32 {
    expand10(x) | (expand10(y) << 1) | (expand10(z) << 2)
}

/// Quantize a position into its 3D Morton code. `bmin` is the cube's low corner and
/// `scale = cells / size` maps a coordinate into `[0, cells)`; each lane is floored
/// and clamped to `[0, cells-1]` (the `(1+1e-9)` bbox pad keeps interior points off
/// the upper edge, but degenerate/2D inputs can still land there — clamp guards it).
fn morton_code(p: DVec3, bmin: DVec3, scale: f64) -> u32 {
    let cells = 1u32 << MORTON_BITS;
    let q = |v: f64| -> u32 { (v.floor().max(0.0) as u32).min(cells - 1) };
    let x = q((p.x - bmin.x) * scale);
    let y = q((p.y - bmin.y) * scale);
    let z = q((p.z - bmin.z) * scale);
    morton3(x, y, z)
}

/// Augmented-key common-prefix length δ(a, b) over the **sorted** array (Karras 2012).
/// The augmented key is `(code, sorted_position)`: when two codes are equal the prefix
/// extends into the position bits (`32 + clz(a ^ b)`), so all keys are distinct and the
/// tree topology is well-defined even for coincident particles. Out-of-range ⇒ −1.
fn delta(codes: &[u32], a: i64, b: i64) -> i64 {
    let n = codes.len() as i64;
    if b < 0 || b >= n {
        return -1;
    }
    let (ca, cb) = (codes[a as usize], codes[b as usize]);
    if ca == cb {
        // Codes tie: extend into the (distinct) sorted positions.
        32 + (a as u32 ^ b as u32).leading_zeros() as i64
    } else {
        (ca ^ cb).leading_zeros() as i64
    }
}

/// A reference to a child in the Karras tree: a leaf (index into the sorted order) or
/// an internal node (index into the `internal` array).
#[derive(Clone, Copy)]
struct ChildRef {
    leaf: bool,
    idx: usize,
}

/// One Karras internal node: its two children, resolved during `karras_internal`.
#[derive(Clone, Copy)]
struct Internal {
    left: ChildRef,
    right: ChildRef,
}

/// Build the `N-1` Karras internal nodes from the sorted `codes` (Karras 2012,
/// *Maximizing Parallelism in the Construction of BVHs*, Algorithms 3–4). Internal
/// node `i` owns a contiguous range of leaves; `determineRange` finds its far end via
/// the δ direction + exponential/binary search, and the split (via δ over the range's
/// own prefix) partitions it into two children. Duplicate codes are handled by δ's
/// position extension, so every node splits deterministically.
fn karras_internal(codes: &[u32]) -> Vec<Internal> {
    let d0 = |a: i64, b: i64| delta(codes, a, b);
    (0..(codes.len() - 1))
        .map(|ii| {
            let i = ii as i64;
            // Direction of the range: toward the neighbour sharing the longer prefix.
            let dir: i64 = if d0(i, i + 1) > d0(i, i - 1) { 1 } else { -1 };
            let delta_min = d0(i, i - dir);
            // Exponential search for an upper bound on the range length.
            let mut l_max: i64 = 2;
            while d0(i, i + l_max * dir) > delta_min {
                l_max *= 2;
            }
            // Binary search for the exact far end.
            let mut l: i64 = 0;
            let mut t = l_max / 2;
            while t > 0 {
                if d0(i, i + (l + t) * dir) > delta_min {
                    l += t;
                }
                t /= 2;
            }
            let j = i + l * dir; // other end of the leaf range
                                 // Binary search for the split position within [i, j] (by the range prefix).
            let delta_node = d0(i, j);
            let mut s: i64 = 0;
            let mut t = l;
            loop {
                t = (t + 1) / 2; // ceil-halve
                if d0(i, i + (s + t) * dir) > delta_node {
                    s += t;
                }
                if t <= 1 {
                    break;
                }
            }
            let gamma = i + s * dir + dir.min(0); // split: last leaf of the left child
            let (lo, hi) = (i.min(j), i.max(j));
            let left = ChildRef {
                leaf: lo == gamma, // left child is a leaf iff it is a single leaf
                idx: gamma as usize,
            };
            let right = ChildRef {
                leaf: hi == gamma + 1,
                idx: (gamma + 1) as usize,
            };
            Internal { left, right }
        })
        .collect::<Vec<_>>()
}

/// Aggregate accumulated bottom-up for one subtree.
#[derive(Clone, Copy)]
struct Agg {
    mass: f64,
    com: DVec3,
    min: DVec3,
    max: DVec3,
}

/// One node of an [`LbvhFlat`] — the Karras binary tree linearized into a DFS
/// pre-order array with skip pointers, the same stackless form as [`crate::FlatNode`]
/// so a future GPU kernel is a direct mirror of the existing traversal.
#[derive(Clone, Copy, Debug)]
pub struct LbvhNode {
    /// AABB geometric center.
    pub center: DVec3,
    /// AABB half-extents (per axis). `2·half_extents` is the box size; the opening
    /// criterion uses its longest component as the cell size `s`.
    pub half_extents: DVec3,
    /// Aggregate center of mass.
    pub com: DVec3,
    /// Aggregate mass.
    pub mass: f64,
    /// |com − center| (Barnes 1994 opening-criterion correction).
    pub delta: f64,
    /// Skip pointer: one past this node's whole subtree in DFS pre-order. A node's
    /// first child is `self+1`; not opening jumps to `next`. `next > self` always,
    /// so the stackless walk strictly increases and terminates.
    pub next: u32,
    /// Leaf: start offset into [`LbvhFlat::leaf_bodies`]. Unused (0) for internal nodes.
    pub body_start: u32,
    /// Leaf body count — **`body_count > 0` iff leaf** (always 1 for an LBVH leaf).
    pub body_count: u32,
}

/// The Karras binary radix tree flattened into DFS pre-order with skip pointers —
/// the stackless representation a GPU kernel walks with a single index. f64 geometry
/// (a GPU consumer narrows to f32). The build is the CPU reference for the deferred
/// GPU-resident Morton/LBVH build; the walk is the exact f64 analogue of the GPU kernel.
pub struct LbvhFlat {
    /// DFS pre-order nodes; the root is index 0.
    pub nodes: Vec<LbvhNode>,
    /// Concatenated leaf body indices (original particle indices), sliced per leaf by
    /// `body_start` / `body_count`. For an LBVH this is a permutation of `0..N`.
    pub leaf_bodies: Vec<u32>,
}

impl LbvhFlat {
    /// Build the LBVH (Morton codes → sort by `(code, index)` → Karras binary radix
    /// tree → bottom-up aggregate) and linearize it to DFS pre-order with skip
    /// pointers. `pos` must be non-empty (the caller handles N=0 before building).
    pub fn build(pos: &[DVec3], mass: &[f64]) -> LbvhFlat {
        let n = pos.len();
        assert!(n > 0, "LbvhFlat::build requires a non-empty system");

        // Root bounding cube — same convention as `barnes_hut::Octree::build_serial`
        // (pad + floor) so quantization never lands an interior point on the top edge.
        let mut lo = pos[0];
        let mut hi = pos[0];
        for &p in pos {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let center = (lo + hi) * 0.5;
        let half = (0.5 * (hi - lo).max_element()).max(1e-12) * (1.0 + 1e-9);
        let bmin = center - DVec3::splat(half);
        let size = 2.0 * half;
        let scale = (1u32 << MORTON_BITS) as f64 / size;

        // Morton codes, then a deterministic sort by (code, original index).
        let codes: Vec<u32> = pos.iter().map(|&p| morton_code(p, bmin, scale)).collect();
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&i| (codes[i as usize], i));
        let sorted_codes: Vec<u32> = order.iter().map(|&i| codes[i as usize]).collect();

        // Karras internal nodes over the sorted order (none for a single leaf).
        let internal = if n > 1 {
            karras_internal(&sorted_codes)
        } else {
            Vec::new()
        };

        // Flatten to DFS pre-order with skip pointers, aggregating bottom-up. The root
        // is internal node 0 (or the single leaf when n == 1).
        let mut nodes: Vec<LbvhNode> = Vec::with_capacity(2 * n - 1);
        let mut leaf_bodies: Vec<u32> = Vec::with_capacity(n);
        let root = ChildRef {
            leaf: n == 1,
            idx: 0,
        };
        flatten(
            root,
            &internal,
            &order,
            pos,
            mass,
            &mut nodes,
            &mut leaf_bodies,
        );
        LbvhFlat { nodes, leaf_bodies }
    }

    /// f64 reference traversal: acceleration on `target` from a stackless walk of the
    /// flat tree, Barnes (1994) opening criterion with opening angle `theta` and
    /// softening `eps2 = ε²`. Excludes the self term. Returns a value needing `× g`,
    /// matching the [`crate::FlatTree::accel`] convention.
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
        // Stackless walk: `node` strictly increases (open → node+1, else → next > node),
        // so this terminates in ≤ n_nodes steps with no stack.
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
                // Internal: never approximate a cell that contains the target. The cell
                // size is the AABB's longest side (a binary node may be non-cubic).
                let he = nd.half_extents;
                let inside = (xp.x - nd.center.x).abs() <= he.x
                    && (xp.y - nd.center.y).abs() <= he.y
                    && (xp.z - nd.center.z).abs() <= he.z;
                let dx = nd.com - xp;
                let d2 = dx.length_squared();
                let s = 2.0 * he.max_element();
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

/// Recursively emit `child`'s subtree into DFS pre-order, filling each node's `next`
/// skip pointer once its subtree is laid down and returning the subtree's aggregate.
/// Children are emitted left-then-right, so a node's first child is `self+1` and its
/// right child is `nodes[self+1].next` — the strict-binary layout the walk relies on.
fn flatten(
    child: ChildRef,
    internal: &[Internal],
    order: &[u32],
    pos: &[DVec3],
    mass: &[f64],
    nodes: &mut Vec<LbvhNode>,
    leaf_bodies: &mut Vec<u32>,
) -> Agg {
    let me = nodes.len();
    if child.leaf {
        let orig = order[child.idx];
        let (p, m) = (pos[orig as usize], mass[orig as usize]);
        let body_start = leaf_bodies.len() as u32;
        leaf_bodies.push(orig);
        nodes.push(LbvhNode {
            center: p,
            half_extents: DVec3::ZERO,
            com: p,
            mass: m,
            delta: 0.0,
            next: me as u32 + 1,
            body_start,
            body_count: 1,
        });
        return Agg {
            mass: m,
            com: p,
            min: p,
            max: p,
        };
    }

    // Internal: reserve this node's slot, emit both children, then aggregate.
    nodes.push(LbvhNode {
        center: DVec3::ZERO,
        half_extents: DVec3::ZERO,
        com: DVec3::ZERO,
        mass: 0.0,
        delta: 0.0,
        next: 0, // patched below
        body_start: 0,
        body_count: 0,
    });
    let node = internal[child.idx];
    let la = flatten(node.left, internal, order, pos, mass, nodes, leaf_bodies);
    let ra = flatten(node.right, internal, order, pos, mass, nodes, leaf_bodies);

    // Fold children in fixed (left, right) order — deterministic sums.
    let mass_sum = la.mass + ra.mass;
    let com = if mass_sum > 0.0 {
        (la.com * la.mass + ra.com * ra.mass) / mass_sum
    } else {
        (la.min + la.max + ra.min + ra.max) * 0.25
    };
    let min = la.min.min(ra.min);
    let max = la.max.max(ra.max);
    let center = (min + max) * 0.5;
    nodes[me] = LbvhNode {
        center,
        half_extents: (max - min) * 0.5,
        com,
        mass: mass_sum,
        delta: (com - center).length(),
        next: nodes.len() as u32,
        body_start: 0,
        body_count: 0,
    };
    Agg {
        mass: mass_sum,
        com,
        min,
        max,
    }
}

/// Barnes-Hut force solver over a Morton-code Linear BVH. Same `(g, softening, theta)`
/// semantics and Plummer-softened kernel as [`crate::BarnesHut`], so it is directly
/// comparable — an O(N log N) monopole approximation that reproduces direct summation
/// as `theta → 0`. Pure CPU f64; the reference for the deferred GPU-resident build.
#[derive(Clone, Copy, Debug)]
pub struct Lbvh {
    /// Gravitational constant.
    pub g: f64,
    /// Plummer softening length ε.
    pub softening: f64,
    /// Opening angle θ. Smaller = more accurate, more work.
    pub theta: f64,
}

impl Lbvh {
    pub fn new(g: f64, softening: f64, theta: f64) -> Self {
        Self {
            g,
            softening,
            theta,
        }
    }
}

impl ForceSolver for Lbvh {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        let flat = LbvhFlat::build(&state.pos, &state.mass);
        let eps2 = self.softening * self.softening;
        // Per-target gather in a fixed skip-pointer order — deterministic (each acc[i]
        // is one independent accumulation; no cross-target reassociation).
        for (i, a) in acc.iter_mut().enumerate() {
            *a = flat.accel(i, &state.pos, &state.mass, self.theta, eps2) * self.g;
        }
    }

    /// Softened potential energy — delegated to the shared f64 reduction, identical to
    /// [`crate::BarnesHut`] (still O(N²); a periodic diagnostic, not the per-step path).
    fn potential_energy(&self, state: &State) -> f64 {
        crate::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}

#[cfg(test)]
mod morton_tests {
    //! Unit tests for the Morton primitives (private; the physics/structural gates
    //! live in `solvers/tests/lbvh.rs`). Hand-checkable bit spreading + monotonicity.
    use super::{expand10, morton3, MORTON_BITS};

    #[test]
    fn expand10_spreads_bits_by_three() {
        // Each set input bit lands 3 positions apart in the output.
        assert_eq!(expand10(0b0), 0b0);
        assert_eq!(expand10(0b1), 0b1);
        assert_eq!(expand10(0b11), 0b1001);
        assert_eq!(expand10(0b111), 0b1001001);
        // Top bit of the 10-bit lane → bit 27 of the 30-bit code.
        assert_eq!(expand10(1 << 9), 1 << 27);
        // Only the low 10 bits participate.
        assert_eq!(expand10(0xFFFF_FC00), 0);
    }

    #[test]
    fn morton3_places_lanes_on_interleaved_bits() {
        assert_eq!(morton3(0, 0, 0), 0);
        assert_eq!(morton3(1, 0, 0), 0b001); // x → bit 0
        assert_eq!(morton3(0, 1, 0), 0b010); // y → bit 1
        assert_eq!(morton3(0, 0, 1), 0b100); // z → bit 2
        assert_eq!(morton3(1, 1, 1), 0b111);
    }

    #[test]
    fn morton3_is_monotone_along_a_single_axis() {
        // With the other two lanes fixed at 0, increasing one lane must not decrease
        // the code (the space-filling curve is order-preserving along an axis).
        let max = 1u32 << MORTON_BITS;
        let mut prev = 0;
        for x in 0..max {
            let c = morton3(x, 0, 0);
            if x > 0 {
                assert!(c > prev, "morton3 not monotone in x at {x}");
            }
            prev = c;
        }
    }
}
