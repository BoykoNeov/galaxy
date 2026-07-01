//! [`GpuLbvh`]: GPU Linear-BVH Barnes-Hut force solver (DESIGN M4g). API surface only;
//! the traversal kernel and the GPU pipeline wiring land in the green commit.

use galaxy_core::{DVec3, ForceSolver, State};

use crate::GpuError;

/// GPU Barnes-Hut force solver over a GPU-resident Morton Linear BVH. Same
/// `(g, softening, theta)` semantics as [`galaxy_solvers::Lbvh`], evaluated by an f32
/// stackless wgpu traversal of the M4f flat form built by the full GPU f32 chain.
pub struct GpuLbvh {
    _private: (),
}

impl GpuLbvh {
    /// Bring up the GPU LBVH pipeline (Morton → sort → build → flatten → traverse).
    pub fn new(_g: f64, _softening: f64, _theta: f64) -> Result<Self, GpuError> {
        todo!("GPU LBVH traversal + pipeline wiring land in the M4g green commit")
    }
}

impl ForceSolver for GpuLbvh {
    fn accelerations(&mut self, _state: &State, _acc: &mut [DVec3]) {
        todo!("GPU LBVH traversal lands in the M4g green commit")
    }

    fn potential_energy(&self, _state: &State) -> f64 {
        todo!("GPU LBVH potential energy lands in the M4g green commit")
    }
}
