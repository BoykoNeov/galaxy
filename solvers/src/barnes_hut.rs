use galaxy_core::{DVec3, ForceSolver, State};

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
}

impl BarnesHut {
    pub fn new(g: f64, softening: f64, theta: f64) -> Self {
        Self {
            g,
            softening,
            theta,
        }
    }
}

impl ForceSolver for BarnesHut {
    fn accelerations(&mut self, _state: &State, _acc: &mut [DVec3]) {
        todo!()
    }

    fn potential_energy(&self, _state: &State) -> f64 {
        todo!()
    }
}
