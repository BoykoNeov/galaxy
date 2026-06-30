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
    // Cross-indexed N-body double loop: the range index reads several arrays at
    // both i and j, so iterator adapters would not be clearer here.
    #[allow(clippy::needless_range_loop)]
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        let eps2 = self.softening * self.softening;
        for i in 0..n {
            let xi = state.pos[i];
            let mut ai = DVec3::ZERO;
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = state.pos[j] - xi;
                let r2 = dx.length_squared() + eps2;
                let inv_r3 = 1.0 / (r2 * r2.sqrt()); // (r² + ε²)^(-3/2)
                ai += dx * (state.mass[j] * inv_r3);
            }
            acc[i] = ai * self.g;
        }
    }

    #[allow(clippy::needless_range_loop)]
    fn potential_energy(&self, state: &State) -> f64 {
        let n = state.len();
        let eps2 = self.softening * self.softening;
        let mut u = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let dx = state.pos[j] - state.pos[i];
                let r = (dx.length_squared() + eps2).sqrt();
                u -= self.g * state.mass[i] * state.mass[j] / r;
            }
        }
        u
    }
}
