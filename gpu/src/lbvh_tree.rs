//! [`GpuLbvhBuilder`]: the GPU Karras tree-build kernel + atomic-flag bottom-up
//! aggregation — stage 3 of the GPU-resident LBVH build (DESIGN M4e).
//!
//! Given the Morton `sorted_codes` (the output of [`crate::GpuSorter`] / the CPU
//! [`galaxy_solvers::reference_sort`]) plus the per-leaf position + mass **pre-gathered
//! into that sorted order**, it builds — on the GPU (wgpu compute) — the binary radix
//! tree of [`galaxy_solvers::reference_karras`] (topology) and its bottom-up
//! [`galaxy_solvers::reference_aggregate`] (per-node AABB / centre-of-mass / mass).
//!
//! ## Two gates, because the stage is half integer and half f32
//! The Karras **topology** is a pure-integer function of the sorted codes
//! (`δ = clz(code_a ^ code_b)`, with a position tie-extension for equal codes), so the
//! GPU `(left, right, parent)` must equal the CPU reference **bit-for-bit** — the
//! load-bearing gate, exactly like the M4d sort. The **aggregation** runs in f32: its
//! AABB `min`/`max` folds never round and are order-independent (bit-exact vs an f32 CPU
//! fold), while `com`/`mass` are f32-lossy and gated on tolerance vs the f64 reference.
//!
//! This does **not** contradict the M4c f32-divergence note: that divergence lives
//! upstream in the Morton *codes*, not in this pure-integer tree step. Fed reference /
//! GPU-sorted codes (bit-identical per M4d), the topology is bit-exact.
//!
//! ## Determinism: an integer flag, never a float `atomicAdd`
//! The bottom-up aggregation launches one invocation per leaf; each walks up via parent
//! pointers, and at each internal node an **integer** `atomic<u32>` visit-counter gates
//! the fold so only the *second* child to arrive (both children final) combines the
//! parent — from its **stored** `left`/`right` in fixed order, so the result is
//! independent of which child won the race. Determinism is structural (the M3/M4
//! "gather, not scatter" discipline), gated same-device on topology **and** aggregation.
//!
//! ## Scope: the raw pointer tree, flatten deferred
//! This stage emits the raw pointer-based tree (parent + children + raw
//! `min`/`max`/`com`/`mass`), **not** the DFS skip-pointer `LbvhFlat` form — deriving
//! `center`/`half`/`delta` and the `next` skip pointer (a subtree-size prefix-sum /
//! Euler-tour) is the next stage, which lets the deferred `GpuLbvh` traverse the same
//! form the CPU `LbvhFlat::accel` walk uses. Reference-grade aggregation (one thread per
//! leaf walking up); a parallel multi-tile build is the named scale refinement.

use galaxy_core::DVec3;

use crate::GpuError;

/// The GPU-built Karras binary radix tree in the canonical unified node layout
/// ([`galaxy_solvers::KarrasTree`]): the `N` leaves occupy unified indices `[0, N)` (in
/// Morton-sorted order), the `N-1` internal nodes `[N, 2N-1)` (internal `i` at `N+i`;
/// root = internal 0 = unified `N`). A unified index `u` is a **leaf iff `u < N`**.
pub struct GpuLbvhTree {
    /// Number of leaves `N`.
    pub n: usize,
    /// Left child (unified index) of each of the `N-1` internal nodes.
    pub left: Vec<u32>,
    /// Right child (unified index) of each of the `N-1` internal nodes.
    pub right: Vec<u32>,
    /// Parent (unified index) of every node (`len 2N-1`); the root is
    /// [`galaxy_solvers::NO_PARENT`].
    pub parent: Vec<u32>,
    /// AABB low corner per unified node (`len 2N-1`, f32).
    pub aabb_min: Vec<[f32; 3]>,
    /// AABB high corner per unified node (`len 2N-1`, f32).
    pub aabb_max: Vec<[f32; 3]>,
    /// Aggregate centre of mass per unified node (`len 2N-1`, f32).
    pub com: Vec<[f32; 3]>,
    /// Aggregate mass per unified node (`len 2N-1`, f32).
    pub mass: Vec<f32>,
}

/// GPU Karras tree-build + atomic-flag aggregation stage. Holds a reusable wgpu compute
/// context built once and driven per [`build`](Self::build) call; storage buffers grow
/// lazily with N. Same bring-up idiom as [`crate::GpuSorter`].
pub struct GpuLbvhBuilder {
    // Fields are added in the implementation commit; the red-gate commit only needs the
    // API surface to compile and fail.
    _stub: (),
}

impl GpuLbvhBuilder {
    /// Bring up a headless wgpu compute device and build the tree-build + aggregation
    /// pipelines. Returns a typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        todo!("M4e: GPU Karras tree-build + atomic-flag aggregation stage")
    }

    /// Build the Karras tree over the **sorted** `sorted_codes` (as produced by the GPU
    /// sort), with `sorted_pos`/`sorted_mass` the leaf payload gathered into the same
    /// sorted order (leaf `k` → `sorted_pos[k]`). `sorted_codes` may be empty (yields an
    /// empty tree). Panics if the three input lengths disagree.
    pub fn build(
        &mut self,
        sorted_codes: &[u32],
        sorted_pos: &[DVec3],
        sorted_mass: &[f64],
    ) -> GpuLbvhTree {
        let _ = (sorted_codes, sorted_pos, sorted_mass);
        todo!("M4e: GPU Karras tree-build + atomic-flag aggregation stage")
    }
}
