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

/// The Saitoh–Makino (2009) timestep limiter (I4b/I5) — CORRECTNESS, not a dial.
/// Raise (refine) any gas particle sitting more than `n_limit` rungs coarser than a
/// coupled neighbour, iterated to a fixpoint, so no coupled pair differs by more than
/// `n_limit` rungs. `pairs` are the force-coupled gas neighbours (global indices,
/// [`ForceSolver::coupled_pairs`](galaxy_core::ForceSolver::coupled_pairs)); the
/// limiter only ever INCREASES a rung, never coarsens, so it is monotone and bounded
/// above by the finest rung present ⇒ the fixpoint always converges.
///
/// Why it is load-bearing: a slow-rung particle in cold gas struck by a shock from a
/// fast-rung neighbour would, without this, step at its stale coarse `dt` straight
/// THROUGH the shock arrival and mis-integrate it (poisoning the shocked-merger gas
/// physics — and, on the adiabatic arm, the internal energy `u`). Forcing it within
/// `n_limit` rungs of its fastest neighbour wakes it early — many base blocks before
/// the shock physically reaches it, since the neighbour range (≈ 2h) far exceeds the
/// per-block signal travel (≈ courant·h). `n_limit` (typically 1) is the only dial.
///
/// `rungs` is state-length and mutated in place; `pairs` index into it. Non-gas rows
/// carry no pairs ⇒ are never touched. A no-op when the rung spread already satisfies
/// the constraint (e.g. `n_limit ≥` the spread), so a non-binding `n_limit` is free.
pub fn limit_rungs(rungs: &mut [u32], pairs: &[(usize, usize)], n_limit: u32) {
    // Sweep the pairs until a full pass makes no change (a fixpoint). Each pair raises
    // the coarser member to within `n_limit` of the finer; because rungs only ever go
    // UP and are bounded above by the finest rung present, the sweep terminates. A
    // single pass propagates fineness one hop; the fixpoint carries it across a chain.
    loop {
        let mut changed = false;
        for &(i, j) in pairs {
            let hi = rungs[i].max(rungs[j]);
            // `hi <= n_limit` ⇒ every rung is already within `n_limit` of `hi`
            // (nothing to raise, and it guards the `hi - n_limit` subtraction).
            if hi > n_limit {
                let floor = hi - n_limit;
                if rungs[i] < floor {
                    rungs[i] = floor;
                    changed = true;
                }
                if rungs[j] < floor {
                    rungs[j] = floor;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
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
    x + v * dt
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
        self.acc.clear();
        self.acc.resize(state.len(), DVec3::ZERO);
        solver.accelerations(state, &mut self.acc);
        self.primed = true;
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
        _bg: &dyn Background,
        dt_base: f64,
        rungs: &[u32],
    ) {
        let n = state.len();
        assert_eq!(rungs.len(), n, "rungs must be state-length");
        // Prime accelerations at the current positions on the first block (or after
        // a particle-count change); later blocks reuse the value left by the
        // previous block's closing kick — one fresh force eval per fine tick.
        if self.acc.len() != n {
            self.acc.clear();
            self.acc.resize(n, DVec3::ZERO);
            self.primed = false;
        }
        if !self.primed {
            solver.accelerations(state, &mut self.acc);
            self.primed = true;
        }

        // The finest rung present sets the fine-tick count: 2^r_max ticks of
        // `d = dt_base / 2^r_max`. A rung-r particle is active every 2^(r_max−r)
        // ticks (its step `dt_base/2^r` spans that many fine ticks). Both are exact
        // integer relations — no float `log2`.
        let r_max = rungs.iter().copied().max().unwrap_or(0);
        let n_fine: u64 = 1 << r_max;
        let d = dt_base / n_fine as f64;

        // Opening half-kick: every particle opens a step at the block start, kicked
        // by half its OWN step with the primed (block-start) acceleration. Index
        // loop: the body reads two parallel SoA columns (`rungs`, `acc`) and writes
        // a third (`vel`) at `i` — an `enumerate` over any one is no clearer.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n {
            let half_step = 0.5 * (dt_base / (1u64 << rungs[i]) as f64);
            state.vel[i] += self.acc[i] * half_step;
        }

        for k in 0..n_fine {
            // Drift ALL by the fine sub-step (positions stay exact — velocity is
            // constant between an inactive particle's kicks, so its many fine
            // sub-drifts sum to one full drift).
            for (x, v) in state.pos.iter_mut().zip(&state.vel) {
                *x += *v * d;
            }
            // Fresh force at the new positions (the caching POLICY — fresh vs a
            // stale tree — is the driver's choice at I4/I6, not this mechanic's:
            // it takes forces through the ForceSolver seam).
            solver.accelerations(state, &mut self.acc);

            let ticks_done = k + 1; // fine ticks completed this block, in units of d
            let block_end = ticks_done == n_fine;
            #[allow(clippy::needless_range_loop)] // parallel SoA columns, as above
            for i in 0..n {
                let period: u64 = 1 << (r_max - rungs[i]); // active every `period` ticks
                if ticks_done % period != 0 {
                    continue;
                }
                let step_i = dt_base / (1u64 << rungs[i]) as f64;
                if block_end {
                    // Closing half-kick: the particle's step ends at the block
                    // boundary; only the closing half remains (its next step opens
                    // in the following block).
                    state.vel[i] += self.acc[i] * (0.5 * step_i);
                } else {
                    // Interior boundary: the closing half of the ending step and the
                    // opening half of the next step share this time and force ⇒ one
                    // full-step kick (the standard KDK half-kick merge).
                    state.vel[i] += self.acc[i] * step_i;
                }
            }
        }
        state.time += dt_base;
    }
}

/// Active-set KDK stepper for the ADIABATIC thermal path (I5/I8): identical
/// active-set rung mechanic to [`ActiveSetKdk`], but also kicks `state.u`
/// alongside `state.vel` at every (opening / interior / closing) kick, using
/// [`ForceSolver::accel_and_dudt`](galaxy_core::ForceSolver::accel_and_dudt)'s
/// fused `(acc, du/dt)` output, and applies the positive-`u` floor (E4b) to the
/// just-kicked ACTIVE subset. A separate type — NOT a branch on `ActiveSetKdk`
/// — so the isothermal individual byte-path (I3/I4a/I4b bit-identity gates) is
/// never made to depend on the `accel_and_dudt`-fills-`acc`-like-`accelerations`
/// invariant.
///
/// When every particle lands on rung 0 this reduces to
/// [`LeapfrogKdkThermal`](galaxy_core::LeapfrogKdkThermal) at `dt_base`
/// bit-for-bit (one fine tick, active set = all): same fused solver call, same
/// KDK order, same floor placement (after each half-kick). Multi-rung it is a
/// genuinely different, correct integrator that converges to the true solution
/// as rungs refine.
///
/// The `u`-floor `u ← max(u, u_min)` clamps `u` after each active-subset kick so
/// the next force eval never builds pressure from a negative `u` (a NaN sound
/// speed). The injected energy `Σ mᵢ(u_min − u_raw)` is accumulated in
/// [`u_floor_energy`](Self::u_floor_energy) as bounded, reported
/// non-conservation. `u_min = 0.0` (the default) is provably inert on any run
/// whose `u` stays positive.
#[derive(Clone, Debug, Default)]
pub struct ActiveSetKdkThermal {
    /// Cached accelerations at the current positions (scratch), reused across
    /// the block boundary as the next opening kick.
    acc: Vec<DVec3>,
    /// Cached `du/dt` at the current positions (scratch), reused likewise.
    dudt: Vec<f64>,
    primed: bool,
    /// Positive-`u` floor `u ← max(u, u_min)` after each active-subset kick.
    /// `0.0` by default (inert on positive-`u` runs).
    u_min: f64,
    /// Accumulated energy injected by the floor: `Σ mᵢ(u_min − u_raw)` over
    /// every clamp (≥ 0). The bounded, reported non-conservation.
    u_floor_energy: f64,
}

impl ActiveSetKdkThermal {
    pub fn new() -> Self {
        Self {
            acc: Vec::new(),
            dudt: Vec::new(),
            primed: false,
            u_min: 0.0,
            u_floor_energy: 0.0,
        }
    }

    /// Construct with a positive internal-energy floor `u_min`, mirroring
    /// [`LeapfrogKdkThermal::with_u_floor`](galaxy_core::LeapfrogKdkThermal::with_u_floor).
    pub fn with_u_floor(u_min: f64) -> Self {
        Self {
            u_min,
            ..Self::new()
        }
    }

    /// Total energy injected by the `u`-floor so far (≥ 0, `0.0` if the floor
    /// never engaged). Cleared by [`reset`](Self::reset).
    pub fn u_floor_energy(&self) -> f64 {
        self.u_floor_energy
    }

    /// Clear cached state so the next `step_block` re-primes from scratch; also
    /// zeroes the accumulated `u`-floor leak (`u_min` is config, retained).
    pub fn reset(&mut self) {
        todo!("I5: clear acc/dudt/primed and zero u_floor_energy")
    }

    /// Eagerly compute and cache `(acc, du/dt)` at the current state, so the
    /// next `step_block` opens with a fresh (not stale) half-kick.
    pub fn prime(&mut self, state: &State, solver: &mut dyn ForceSolver) {
        let _ = (state, solver);
        todo!("I5: fused prime via accel_and_dudt")
    }

    /// Advance one base block of size `dt_base`, each particle on rung `rungs[i]`,
    /// kicking `vel` AND `u` on the active subset each fine tick and applying the
    /// `u`-floor to the just-kicked subset. Same active-set mechanic and
    /// synchronization as [`ActiveSetKdk::step_block`].
    pub fn step_block(
        &mut self,
        state: &mut State,
        solver: &mut dyn ForceSolver,
        bg: &dyn Background,
        dt_base: f64,
        rungs: &[u32],
    ) {
        let _ = (state, solver, bg, dt_base, rungs);
        todo!("I5: thermal active-set KDK — kick u + apply floor per active subset")
    }
}
