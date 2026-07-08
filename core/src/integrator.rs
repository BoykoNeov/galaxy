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
        self.acc.clear();
        self.primed = false;
    }

    /// Eagerly compute and cache accelerations at the current state, so the next
    /// `step` opens with a fresh (not stale) half-kick.
    pub fn prime(&mut self, state: &State, solver: &mut dyn ForceSolver) {
        self.acc.clear();
        self.acc.resize(state.len(), DVec3::ZERO);
        solver.accelerations(state, &mut self.acc);
        self.primed = true;
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

/// Kick-Drift-Kick leapfrog for the adiabatic thermal path (E2b): identical KDK
/// structure to [`LeapfrogKdk`], but also kicks `state.u` alongside
/// `state.vel` at both half-kicks, using [`ForceSolver::accel_and_dudt`]'s
/// fused `(acc, du/dt)` output. A separate type (not a branch on
/// `LeapfrogKdk`) so the gravity-only/isothermal path keeps its exact current
/// bit-path untouched.
///
/// A positive internal-energy floor `u ← max(u, u_min)` (E4b) clamps `u` after
/// every half-kick — the energy formulation's one wart: aggressive PdV cooling
/// can drive `u` below zero (a NaN sound speed at the next force eval). The
/// clamp's injected energy is accumulated in [`u_floor_energy`] as a bounded,
/// reported non-conservation. The default `u_min = 0.0` makes the floor
/// provably inert on any run whose `u` stays positive (`max(positive, 0)` is
/// bit-identical, the leak sum empty); [`with_u_floor`] sets a small positive
/// value on the adaptive-adiabatic path.
///
/// [`u_floor_energy`]: LeapfrogKdkThermal::u_floor_energy
/// [`with_u_floor`]: LeapfrogKdkThermal::with_u_floor
#[derive(Clone, Debug, Default)]
pub struct LeapfrogKdkThermal {
    acc: Vec<DVec3>,
    dudt: Vec<f64>,
    primed: bool,
    /// Positive-`u` floor: `u ← max(u, u_min)` after each half-kick. `0.0` by
    /// default (inert on positive-`u` runs).
    u_min: f64,
    /// Accumulated energy injected by the floor: `Σ mᵢ·(u_min − u_raw)` over
    /// every clamp (≥ 0). The bounded, reported non-conservation.
    u_floor_energy: f64,
}

impl LeapfrogKdkThermal {
    pub fn new() -> Self {
        Self {
            acc: Vec::new(),
            dudt: Vec::new(),
            primed: false,
            u_min: 0.0,
            u_floor_energy: 0.0,
        }
    }

    /// Construct with a positive internal-energy floor `u_min`. `u` is clamped to
    /// `max(u, u_min)` after each half-kick and the injected energy accumulated
    /// (see [`u_floor_energy`](Self::u_floor_energy)).
    pub fn with_u_floor(u_min: f64) -> Self {
        Self {
            u_min,
            ..Self::new()
        }
    }

    /// Total energy injected by the `u`-floor so far: `Σ mᵢ·(u_min − u_raw)` over
    /// every clamped particle across every step (≥ 0, `0.0` if the floor never
    /// engaged). Cleared by [`reset`](Self::reset).
    pub fn u_floor_energy(&self) -> f64 {
        self.u_floor_energy
    }

    /// Clear cached state so the next `step` re-primes from scratch. Call this
    /// before reusing one integrator on a different run / initial condition.
    /// Also zeroes the accumulated `u`-floor leak (`u_min` is config, retained).
    pub fn reset(&mut self) {
        self.acc.clear();
        self.dudt.clear();
        self.primed = false;
        self.u_floor_energy = 0.0;
    }

    /// Eagerly compute and cache `(acc, du/dt)` at the current state, so the
    /// next `step` opens with a fresh (not stale) half-kick.
    pub fn prime(&mut self, state: &State, solver: &mut dyn ForceSolver) {
        self.acc.clear();
        self.acc.resize(state.len(), DVec3::ZERO);
        self.dudt.clear();
        self.dudt.resize(state.len(), 0.0);
        solver.accel_and_dudt(state, &mut self.acc, &mut self.dudt);
        self.primed = true;
    }
}

impl Integrator for LeapfrogKdkThermal {
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
            self.dudt.clear();
            self.dudt.resize(n, 0.0);
            self.primed = false;
        }
        // Prime (acc, du/dt) at the current positions on the first step (or after
        // a particle-count change); later steps reuse the values left by the
        // previous step's closing kick — one fused force evaluation per step.
        if !self.primed {
            solver.accel_and_dudt(state, &mut self.acc, &mut self.dudt);
            self.primed = true;
        }

        let half = 0.5 * dt;
        // Kick (half): velocity with a(xₙ), internal energy with (du/dt)(xₙ). Both
        // half-kicks straddle the drift symmetrically, so the u-integration is the
        // same 2nd-order symplectic-style update as the velocity.
        for (v, a) in state.vel.iter_mut().zip(&self.acc) {
            *v += *a * half;
        }
        for (u, d) in state.u.iter_mut().zip(&self.dudt) {
            *u += *d * half;
        }
        // Drift.
        for (x, v) in state.pos.iter_mut().zip(&state.vel) {
            *x += *v * dt;
        }
        // Recompute (a, du/dt)(xₙ₊₁), cached for the next step's opening kick.
        solver.accel_and_dudt(state, &mut self.acc, &mut self.dudt);
        // Kick (half) using the post-drift values.
        for (v, a) in state.vel.iter_mut().zip(&self.acc) {
            *v += *a * half;
        }
        for (u, d) in state.u.iter_mut().zip(&self.dudt) {
            *u += *d * half;
        }
        state.time += dt;
    }
}
