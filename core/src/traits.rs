use crate::{DVec3, State};

/// Computes gravitational accelerations. Softening is a property of the
/// concrete solver. Implementations are swappable (direct-sum → Barnes-Hut →
/// PM/TreePM) without touching the integrator or callers.
pub trait ForceSolver {
    /// Fill `acc[i]` with the acceleration on particle `i`.
    /// Requires `acc.len() == state.len()`.
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]);

    /// Fused acceleration + thermal-derivative pass (E2a). Fills `acc` exactly
    /// as `accelerations` would AND fills `dudt[i]` with `du_i/dt` (zero for
    /// non-thermal particles/solvers). The default delegates to
    /// `accelerations` and zero-fills `dudt`, so every existing solver (pure
    /// gravity, GPU) gets `du/dt≡0` for free without touching its impl;
    /// `GravitySph` overrides this with a single fused SPH neighbor pass
    /// (accel + PdV work share the same loop).
    /// Requires `acc.len() == dudt.len() == state.len()`.
    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        self.accelerations(state, acc);
        dudt.fill(0.0);
    }

    /// Total gravitational potential energy, using the SAME softened kernel as
    /// `accelerations` so energy diagnostics stay consistent with the forces.
    fn potential_energy(&self, state: &State) -> f64;

    /// The CFL limit the solver's physics imposes at this state — the largest dt
    /// stable at Courant number 1 (for SPH, `min_i h_i / v_sig,i`). `+∞` when the
    /// solver imposes no timestep constraint (pure gravity has none in v1, so the
    /// default is `+∞`).
    ///
    /// This reports only the *physics* limit. The adaptive-dt loop applies its own
    /// Courant number and safety factor strictly BELOW this — timestep POLICY lives
    /// in the loop, never in the solver (mirroring how the pipeline `C_CFL` guard is
    /// a policy constant, not a solver property).
    fn max_stable_dt(&self, _state: &State) -> f64 {
        f64::INFINITY
    }

    /// The per-particle CFL limit (I1) — the state-indexed vector whose `min` is
    /// [`max_stable_dt`](Self::max_stable_dt). Individual timesteps bin these into
    /// power-of-two rungs (a particle's rung IS its `dt_i`). Gas rows carry the
    /// finite hydro bound; collisionless rows (and any solver with no hydro
    /// constraint) carry `+∞`. The default is `vec![+∞; len]`, consistent with the
    /// scalar default — a pure-gravity solver imposes no per-particle limit.
    ///
    /// Like the scalar, this reports only the *physics* limit; the Courant number
    /// and rung policy live in the individual-timestep loop, never in the solver.
    fn max_stable_dt_per_particle(&self, state: &State) -> Vec<f64> {
        vec![f64::INFINITY; state.len()]
    }
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
