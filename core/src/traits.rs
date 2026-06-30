use crate::{DVec3, State};

/// Computes gravitational accelerations. Softening is a property of the
/// concrete solver. Implementations are swappable (direct-sum → Barnes-Hut →
/// PM/TreePM) without touching the integrator or callers.
pub trait ForceSolver {
    /// Fill `acc[i]` with the acceleration on particle `i`.
    /// Requires `acc.len() == state.len()`.
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]);

    /// Total gravitational potential energy, using the SAME softened kernel as
    /// `accelerations` so energy diagnostics stay consistent with the forces.
    fn potential_energy(&self, state: &State) -> f64;
}

/// Cosmological background. `StaticBackground` => a≡1, H≡0 (Newtonian).
/// A Friedmann background (later) supplies a(t) and the Hubble drag term, which
/// is where comoving integration attaches — the integrator interface is ready.
pub trait Background {
    /// Scale factor a(t).
    fn scale_factor(&self, t: f64) -> f64;
    /// Hubble parameter H = ȧ/a.
    fn hubble(&self, t: f64) -> f64;
}

/// Advances the state by one timestep `dt`.
pub trait Integrator {
    fn step(
        &mut self,
        state: &mut State,
        solver: &mut dyn ForceSolver,
        bg: &dyn Background,
        dt: f64,
    );
}
