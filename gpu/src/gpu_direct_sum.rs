//! [`GpuDirectSum`]: exact O(N²) Plummer-softened direct summation on the GPU.
//!
//! Holds a reusable wgpu compute context (adapter/device/queue + pipeline) built
//! once and driven per `accelerations` call; storage buffers grow lazily with N.
//! See the crate docs for the precision (f32-forced) and determinism (gather, not
//! scatter) rationale.

use galaxy_core::{DVec3, ForceSolver, State};

use crate::GpuError;

/// GPU direct-summation force solver. Same `(g, softening)` semantics as
/// [`galaxy_solvers::DirectSum`], evaluated in an f32 wgpu compute kernel.
pub struct GpuDirectSum {
    /// Gravitational constant.
    g: f64,
    /// Plummer softening length ε.
    softening: f64,
}

impl GpuDirectSum {
    /// Bring up a headless wgpu compute device and build the direct-sum pipeline.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new(g: f64, softening: f64) -> Result<Self, GpuError> {
        let _ = (g, softening);
        todo!("GPU direct-sum compute context + pipeline")
    }
}

impl ForceSolver for GpuDirectSum {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let _ = (state, acc);
        todo!("upload state, dispatch gather kernel, read back f32 accelerations")
    }

    fn potential_energy(&self, state: &State) -> f64 {
        let _ = state;
        todo!("delegate to the CPU f64 parallel softened potential")
    }
}
