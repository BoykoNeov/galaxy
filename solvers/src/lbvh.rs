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
/// `b9 0 0 b8 0 0 … b0`), the per-axis step of a 3D Morton interleave.
fn expand10(v: u32) -> u32 {
    let _ = v;
    todo!("bit-spread the low 10 bits (part-by-two)")
}

/// Interleave three 10-bit lane coordinates into one 30-bit Morton code:
/// `expand10(x) | expand10(y)<<1 | expand10(z)<<2`.
fn morton3(x: u32, y: u32, z: u32) -> u32 {
    let _ = (x, y, z);
    todo!("interleave the three expanded lanes")
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
        let _ = (pos, mass);
        todo!("Morton + Karras build + aggregate + flatten")
    }

    /// f64 reference traversal: acceleration on `target` from a stackless walk of the
    /// flat tree, Barnes (1994) opening criterion with opening angle `theta` and
    /// softening `eps2 = ε²`. Excludes the self term. Returns a value needing `× g`,
    /// matching the [`crate::FlatTree::accel`] convention.
    pub fn accel(&self, target: usize, pos: &[DVec3], mass: &[f64], theta: f64, eps2: f64) -> DVec3 {
        let _ = (target, pos, mass, theta, eps2);
        todo!("stackless BVH walk")
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
        let _ = (state, &mut *acc);
        todo!("build LBVH, walk per target, scale by g")
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
