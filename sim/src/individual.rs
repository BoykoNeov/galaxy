//! Individual (per-particle power-of-two rung) timesteps on the SPH path
//! (plan: laddered-ember-cadence.md). The third stepping path, added BESIDE the
//! fixed-dt [`run`](crate::run) and global-adaptive [`run_adaptive`](crate::run_adaptive)
//! drivers — neither of those byte-paths is touched.
//!
//! Each gas particle sits on its own power-of-two rung below a base timestep
//! `dt_base` and is re-integrated only when its rung is due, so the diffuse
//! majority steps far less often than the shocked knot that pins the global
//! bound. This module hosts the rung POLICY (I2) — pure functions, unit-testable
//! without stepping — the active-set KDK stepper (I3, [`ActiveSetKdk`]), and
//! (later) the `run_individual` driver. Rung policy lives here in the loop, never
//! in the solver, mirroring how [`plan_block`](crate::plan_block) owns the
//! global-adaptive policy.

use galaxy_core::{Background, DVec3, ForceSolver, State};

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
pub fn base_dt(dt_per_particle: &[f64], courant: f64, cap: f64) -> f64 {
    let max_finite = dt_per_particle
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .fold(0.0_f64, f64::max);
    if max_finite > 0.0 {
        (courant * max_finite).min(cap)
    } else {
        cap
    }
}

/// Assign each particle to a power-of-two rung (I2): `r_i = clamp(⌈log2(dt_base /
/// (courant·dt_i))⌉, 0, r_max)`, so its sub-step `dt_base/2^r_i ≤ courant·dt_i`
/// (its safe step) and is the COARSEST rung that still fits. Computed by an exact
/// integer search (no float `log2` rounding at power-of-two boundaries).
///
/// A collisionless `dt_i = +∞` (or any `dt_i ≥ dt_base/courant`) maps to rung 0,
/// the coarsest. A `dt_i` too small to satisfy even at `r_max` clamps there
/// (bounded under-resolution). Output is state-length, aligned to the input.
pub fn assign_rungs(dt_per_particle: &[f64], dt_base: f64, courant: f64, r_max: u32) -> Vec<u32> {
    dt_per_particle
        .iter()
        .map(|&dt_i| assign_one(dt_i, dt_base, courant, r_max))
        .collect()
}

/// Rung of one particle: the SMALLEST `r ∈ [0, r_max]` whose sub-step
/// `dt_base/2^r` fits the safe step `courant·dt_i`. The integer search is exact at
/// power-of-two boundaries where `⌈log2(·)⌉` via a float `log2` could round either
/// way. `dt_i = +∞` (or any `dt_i` whose safe step already ≥ `dt_base`) never
/// enters the loop ⇒ rung 0; a `dt_i` too small to fit even at `r_max` clamps there.
fn assign_one(dt_i: f64, dt_base: f64, courant: f64, r_max: u32) -> u32 {
    let target = courant * dt_i; // the particle's safe sub-step (`+∞` for collisionless)
    let mut r = 0u32;
    while r < r_max && rung_step(dt_base, r) > target {
        r += 1;
    }
    r
}

/// Drift-predict a particle to a time offset `dt` from its last sync: `x + v·dt`
/// (I3 predictor). This is EXACT for KDK — acceleration enters only through the
/// kicks, so between an inactive particle's kicks its velocity is constant and the
/// linear extrapolation is the true position, NOT an approximation. (Adding a
/// `½a·Δt²` term would double-count acceleration — wrong for KDK.) I3 drifts every
/// particle at the fine cadence, so positions are already current; this pins the
/// predictor the I6 efficiency path will call to AVOID touching inactive neighbours.
#[inline]
pub fn predict_pos(x: DVec3, v: DVec3, dt: f64) -> DVec3 {
    todo!("I3 predictor: x + v·dt")
}

/// Active-set Kick-Drift-Kick stepper (I3): advances one base block by sub-cycling
/// a power-of-two rung hierarchy. Each fine tick drifts ALL particles and kicks
/// only the ACTIVE subset (those whose rung is due), so a rung-`r` particle takes
/// `2^r` sub-steps of `dt_base/2^r` per block while a rung-0 particle takes one. A
/// distinct type — NOT a branch on [`LeapfrogKdk`](galaxy_core::LeapfrogKdk) — because
/// per-particle rungs + an active mask do not fit `Integrator::step(dt)`.
///
/// When every particle is on rung 0 this reduces to `LeapfrogKdk` at `dt_base`
/// bit-for-bit (one fine tick, active set = all). Multi-rung it is a genuinely
/// different, correct integrator that converges to the true solution as rungs
/// refine — it is NOT bit-equal to any single-`dt` scheme.
///
/// Caches accelerations as scratch (the closing half-kick and the next block's
/// opening kick reuse the last force eval); nothing derived is stored in `State`
/// (the D2 discipline). Isothermal first (I3); the thermal `u`-kick arm is I5/I8.
#[derive(Clone, Debug, Default)]
pub struct ActiveSetKdk {
    /// Cached accelerations at the current positions (scratch). Reused across the
    /// block boundary as the next opening kick, exactly as `LeapfrogKdk` does.
    acc: Vec<DVec3>,
    primed: bool,
}

impl ActiveSetKdk {
    pub fn new() -> Self {
        Self {
            acc: Vec::new(),
            primed: false,
        }
    }

    /// Clear cached state so the next `step_block` re-primes from scratch. Call
    /// before reusing one stepper on a different run / initial condition.
    pub fn reset(&mut self) {
        self.acc.clear();
        self.primed = false;
    }

    /// Eagerly compute and cache accelerations at the current state, so the next
    /// `step_block` opens with a fresh (not stale) half-kick.
    pub fn prime(&mut self, state: &State, solver: &mut dyn ForceSolver) {
        todo!("I3: compute acc at current positions")
    }

    /// Advance one base block of size `dt_base`, each particle on rung `rungs[i]`
    /// (0 = coarsest = full base step; the finest present rung sets the fine-tick
    /// count `2^r_max`). Kicks only active particles each fine tick; drifts all.
    /// `rungs.len()` must equal `state.len()`. Synchronizes all rungs at the block
    /// boundary — the only place a snapshot may be emitted (I2/D3).
    pub fn step_block(
        &mut self,
        state: &mut State,
        solver: &mut dyn ForceSolver,
        bg: &dyn Background,
        dt_base: f64,
        rungs: &[u32],
    ) {
        todo!("I3: active-set KDK sub-cycle over one base block")
    }
}
