use crate::{Background, DVec3, ForceSolver, Integrator, State};

/// Kick-Drift-Kick leapfrog: symplectic, 2nd order, time-reversible, with
/// bounded (non-drifting) energy error. Caches accelerations between steps so
/// each step costs a single force evaluation after the first.
#[derive(Clone, Debug, Default)]
pub struct LeapfrogKdk {
    acc: Vec<DVec3>,
    primed: bool,
}

impl LeapfrogKdk {
    pub fn new() -> Self {
        Self {
            acc: Vec::new(),
            primed: false,
        }
    }

    /// Clear cached state so the next `step` re-primes from scratch. Call this
    /// before reusing one integrator on a different run / initial condition.
    pub fn reset(&mut self) {
        todo!()
    }

    /// Eagerly compute and cache accelerations at the current state, so the next
    /// `step` opens with a fresh (not stale) half-kick.
    pub fn prime(&mut self, _state: &State, _solver: &mut dyn ForceSolver) {
        todo!()
    }
}

impl Integrator for LeapfrogKdk {
    fn step(
        &mut self,
        state: &mut State,
        solver: &mut dyn ForceSolver,
        _bg: &dyn Background,
        dt: f64,
    ) {
        let n = state.len();
        if self.acc.len() != n {
            self.acc.clear();
            self.acc.resize(n, DVec3::ZERO);
            self.primed = false;
        }
        // Prime acceleration at the current positions on the first step (or
        // after a particle-count change); subsequent steps reuse the value left
        // by the previous step's closing kick.
        if !self.primed {
            solver.accelerations(state, &mut self.acc);
            self.primed = true;
        }

        let half = 0.5 * dt;
        // Kick (half) using a(xₙ).
        for (v, a) in state.vel.iter_mut().zip(&self.acc) {
            *v += *a * half;
        }
        // Drift.
        for (x, v) in state.pos.iter_mut().zip(&state.vel) {
            *x += *v * dt;
        }
        // Recompute a(xₙ₊₁), cached for the next step's opening kick.
        solver.accelerations(state, &mut self.acc);
        // Kick (half) using a(xₙ₊₁).
        for (v, a) in state.vel.iter_mut().zip(&self.acc) {
            *v += *a * half;
        }
        state.time += dt;
        // `_bg` is unused while a ≡ 1; comoving Hubble-drag terms attach here at
        // the cosmology milestone.
    }
}
