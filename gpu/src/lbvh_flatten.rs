//! [`GpuLbvhFlattener`]: the GPU DFS skip-pointer flatten — stage 4 of the GPU-resident
//! LBVH build (DESIGN M4f). API surface only; the kernels land in the green commit.

use galaxy_core::DVec3;

use crate::GpuError;

/// The GPU-flattened LBVH in DFS pre-order with skip pointers — the SoA mirror of
/// [`galaxy_solvers::LbvhFlat`] the deferred `GpuLbvh` traversal walks. Every per-node
/// field has length `2N-1` (DFS slot order, root at slot 0); `leaf_bodies` has length `N`.
pub struct GpuLbvhFlat {
    /// Number of leaves `N`.
    pub n: usize,
    /// AABB geometric center per DFS slot (f32).
    pub center: Vec<[f32; 3]>,
    /// AABB half-extents per DFS slot (per axis, f32).
    pub half_extents: Vec<[f32; 3]>,
    /// Aggregate centre of mass per DFS slot (f32).
    pub com: Vec<[f32; 3]>,
    /// Aggregate mass per DFS slot (f32).
    pub mass: Vec<f32>,
    /// `|com − center|` per DFS slot (Barnes 1994 opening correction, f32).
    pub delta: Vec<f32>,
    /// Skip pointer per DFS slot: one past this node's subtree in pre-order.
    pub next: Vec<u32>,
    /// Leaf start offset into `leaf_bodies` (0 for internal nodes).
    pub body_start: Vec<u32>,
    /// Leaf body count — `> 0` iff leaf (always 1 for an LBVH leaf).
    pub body_count: Vec<u32>,
    /// Concatenated leaf body **original** particle indices, in DFS-leaf order.
    pub leaf_bodies: Vec<u32>,
}

/// GPU DFS skip-pointer flatten stage. API surface only for the red gate.
pub struct GpuLbvhFlattener {
    _private: (),
}

impl GpuLbvhFlattener {
    /// Bring up a headless wgpu compute device and the flatten pipelines.
    pub fn new() -> Result<Self, GpuError> {
        todo!("GPU DFS flatten kernels land in the M4f green commit")
    }

    /// Flatten the Karras pointer tree (built on the GPU from `sorted_codes` +
    /// `sorted_pos`/`sorted_mass`) into the DFS skip-pointer form. `order[k]` is the
    /// original index of the k-th sorted leaf (written into `leaf_bodies`).
    pub fn build_flat(
        &mut self,
        _sorted_codes: &[u32],
        _sorted_pos: &[DVec3],
        _sorted_mass: &[f64],
        _order: &[u32],
    ) -> GpuLbvhFlat {
        todo!("GPU DFS flatten kernels land in the M4f green commit")
    }
}
