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
    // Newton's-third-law pairing: each pair (i, j) is evaluated once and its
    // force applied with opposite sign to both bodies. Halves the work versus a
    // full double loop and keeps a(i<-j) / a(j<-i) from the same shared term.
    #[allow(clippy::needless_range_loop)]
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        for a in acc.iter_mut() {
            *a = DVec3::ZERO;
        }
        let eps2 = self.softening * self.softening;
        let g = self.g;
        for i in 0..n {
            let xi = state.pos[i];
            let mi = state.mass[i];
            for j in (i + 1)..n {
                let dx = state.pos[j] - xi;
                let r2 = dx.length_squared() + eps2;
                let inv_r3 = 1.0 / (r2 * r2.sqrt()); // (r² + ε²)^(-3/2)
                let w = dx * (g * inv_r3); // shared per-pair term (G · dx / r³)
                acc[i] += w * state.mass[j];
                acc[j] -= w * mi;
            }
        }
    }

    fn potential_energy(&self, state: &State) -> f64 {
        crate::potential::potential_energy_parallel(state, self.g, self.softening)
    }
}

impl DirectSum {
    /// Serial reference for the exact softened potential — the tolerance oracle
    /// for the parallel reduction (see `potential` module).
    pub fn potential_energy_serial(&self, state: &State) -> f64 {
        crate::potential::potential_energy_serial(state, self.g, self.softening)
    }
}
