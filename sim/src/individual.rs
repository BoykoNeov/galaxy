//! Individual (per-particle power-of-two rung) timesteps on the SPH path
//! (plan: laddered-ember-cadence.md). The third stepping path, added BESIDE the
//! fixed-dt [`run`](crate::run) and global-adaptive [`run_adaptive`](crate::run_adaptive)
//! drivers — neither of those byte-paths is touched.
//!
//! Each gas particle sits on its own power-of-two rung below a base timestep
//! `dt_base` and is re-integrated only when its rung is due, so the diffuse
//! majority steps far less often than the shocked knot that pins the global
//! bound. This module hosts the rung POLICY (I2) — pure functions, unit-testable
//! without stepping — and (later) the `run_individual` driver + active-set
//! schedule. Rung policy lives here in the loop, never in the solver, mirroring
//! how [`plan_block`](crate::plan_block) owns the global-adaptive policy.

/// The sub-step for rung `r` below `dt_base`: `dt_base / 2^r`. Rung 0 is the
/// coarsest (the full base step); each higher rung halves it.
#[inline]
pub fn rung_step(dt_base: f64, r: u32) -> f64 {
    dt_base / (1u64 << r) as f64
}

/// The base (coarsest-rung) timestep for a block (I2): the courant-scaled step of
/// the COARSEST particle (largest finite `dt_i`), clamped by `cap`. Collisionless
/// `+∞` rows are ignored — they never set the base. Returns `cap` if no finite
/// `dt_i` exists (a gas-free block, which does not use the individual path).
pub fn base_dt(_dt_per_particle: &[f64], _courant: f64, _cap: f64) -> f64 {
    todo!("I2: base_dt")
}

/// Assign each particle to a power-of-two rung (I2): `r_i = clamp(⌈log2(dt_base /
/// (courant·dt_i))⌉, 0, r_max)`, so its sub-step `dt_base/2^r_i ≤ courant·dt_i`
/// (its safe step) and is the COARSEST rung that still fits. Computed by an exact
/// integer search (no float `log2` rounding at power-of-two boundaries).
///
/// A collisionless `dt_i = +∞` (or any `dt_i ≥ dt_base/courant`) maps to rung 0,
/// the coarsest. A `dt_i` too small to satisfy even at `r_max` clamps there
/// (bounded under-resolution). Output is state-length, aligned to the input.
pub fn assign_rungs(
    _dt_per_particle: &[f64],
    _dt_base: f64,
    _courant: f64,
    _r_max: u32,
) -> Vec<u32> {
    todo!("I2: assign_rungs")
}
