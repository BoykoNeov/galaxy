//! [`GpuTree`]: Barnes-Hut O(N log N) tree force solver — CPU build, GPU traverse.
//!
//! The tree is built and linearized on the CPU (reusing the tested
//! [`galaxy_solvers::FlatTree`]), uploaded, and walked on the GPU by a **stackless**
//! gather kernel: one invocation per target follows skip pointers with a single
//! index. See the crate docs for precision (f32-forced) and determinism (gather).

// Stub for the [red] test commit — API surface only. Real implementation lands
// in the green commit.

use galaxy_core::{DVec3, ForceSolver, State};

use crate::GpuError;

/// GPU Barnes-Hut tree force solver. Same `(g, softening, theta)` semantics as
/// [`galaxy_solvers::BarnesHut`], evaluated in an f32 wgpu compute kernel.
pub struct GpuTree {
    #[allow(dead_code)]
    g: f64,
    #[allow(dead_code)]
    softening: f64,
    #[allow(dead_code)]
    theta: f64,
}

impl GpuTree {
    /// Bring up a headless wgpu compute device and build the tree-traversal
    /// pipeline. Returns a typed [`GpuError`] (never panics) when no adapter exists.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        let _ = (g, softening, theta);
        todo!("GpuTree::new")
    }
}

impl ForceSolver for GpuTree {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let _ = (state, acc);
        todo!("GpuTree::accelerations")
    }

    fn potential_energy(&self, state: &State) -> f64 {
        let _ = state;
        todo!("GpuTree::potential_energy")
    }
}
