use galaxy_core::{DVec3, ForceSolver, State};

/// Exact O(N²) Plummer-softened direct summation. The validation oracle and the
/// workhorse for small N. Uses the same softened kernel for force and potential
/// so that energy is conserved consistently (force = -∇U).
#[derive(Clone, Copy, Debug)]
pub struct DirectSum {
    /// Gravitational constant (choose units; e.g. G = 1).
    pub g: f64,
    /// Plummer softening length ε.
    pub softening: f64,
}

impl DirectSum {
    pub fn new(g: f64, softening: f64) -> Self {
        Self { g, softening }
    }
}

impl ForceSolver for DirectSum {
    fn accelerations(&mut self, _state: &State, _acc: &mut [DVec3]) {
        todo!()
    }

    fn potential_energy(&self, _state: &State) -> f64 {
        todo!()
    }
}
