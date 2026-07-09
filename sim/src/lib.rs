//! `galaxy-sim`: the headless stepping engine.
//!
//! Drives `IC → (solver + integrator) stepping loop → snapshots`. The engine is
//! deliberately thin: it owns the time loop and the snapshot cadence and nothing
//! else. Force accuracy lives in the `ForceSolver`, the time discretization in the
//! `Integrator`, and the on-disk format in `galaxy-io` — all injected, so the
//! engine never needs to change when any of them is swapped (the 10^8 / cosmology
//! door stays open).
//!
//! Snapshots are delivered through a [`SnapshotSink`], so the same loop can write
//! numbered files ([`DirectorySink`]) in production or capture states in memory in
//! a test. Checkpoint/restart is a later milestone, not on the M2 path.

use std::path::{Path, PathBuf};

use galaxy_core::{Background, ForceSolver, Integrator, State};
use galaxy_io::Header;

pub mod individual;

/// Errors from a simulation run.
#[derive(thiserror::Error, Debug)]
pub enum SimError {
    /// A snapshot could not be written.
    #[error("snapshot error: {0}")]
    Snapshot(#[from] galaxy_io::SnapshotError),
    /// The run configuration is invalid.
    #[error("invalid simulation config: {0}")]
    Config(String),
}

/// Configuration for a run. Physics (G, θ) lives in the injected solver; this is
/// the time loop, the snapshot cadence, and the metadata stamped into headers.
#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// Timestep.
    pub dt: f64,
    /// Number of steps to integrate.
    pub n_steps: u64,
    /// Emit a snapshot every `snapshot_every` steps (must be ≥ 1). The initial
    /// conditions (step 0) and the final step are always emitted.
    pub snapshot_every: u64,
    /// Softening length to record in snapshot headers (must match the solver's).
    pub softening: f64,
    /// RNG seed that produced the IC, recorded in headers.
    pub rng_seed: u64,
    /// Scenario config hash, recorded in headers.
    pub config_hash: u64,
    /// Units tag, recorded in headers.
    pub units: String,
}

/// Summary returned by [`run`].
#[derive(Clone, Debug, PartialEq)]
pub struct RunSummary {
    /// Number of steps integrated.
    pub steps: u64,
    /// Simulation time after the final step.
    pub final_time: f64,
    /// Total number of snapshots emitted (including the IC and the final step).
    pub snapshots_emitted: u64,
}

/// A consumer of snapshots produced during a run.
pub trait SnapshotSink {
    /// Consume one snapshot. Called for the IC, at the configured cadence, and for
    /// the final step.
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError>;
}

/// A [`SnapshotSink`] that writes each snapshot to a numbered file in a directory,
/// named `snapshot_<step>.snap`.
pub struct DirectorySink {
    dir: PathBuf,
    written: u64,
}

impl DirectorySink {
    /// Create (or reuse) the output directory.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self, SimError> {
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| SimError::Snapshot(e.into()))?;
        Ok(Self { dir, written: 0 })
    }

    /// Number of files written so far.
    pub fn written(&self) -> u64 {
        self.written
    }
}

impl SnapshotSink for DirectorySink {
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError> {
        let path = self.dir.join(format!("snapshot_{:08}.snap", header.step));
        galaxy_io::write_file(path, header, state)?;
        self.written += 1;
        Ok(())
    }
}

/// Run the stepping loop: emit the IC, integrate `n_steps`, emitting a snapshot
/// every `snapshot_every` steps and always capturing the final step.
pub fn run(
    state: &mut State,
    solver: &mut dyn ForceSolver,
    integ: &mut dyn Integrator,
    bg: &dyn Background,
    config: &SimConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<RunSummary, SimError> {
    if config.snapshot_every == 0 {
        return Err(SimError::Config("snapshot_every must be >= 1".to_string()));
    }
    if !config.dt.is_finite() || config.dt <= 0.0 {
        return Err(SimError::Config(format!(
            "dt must be a positive finite number, got {}",
            config.dt
        )));
    }

    let mut emitted = 0u64;

    // Step 0: the initial conditions.
    emit_snapshot(state, 0, config, sink)?;
    emitted += 1;
    let mut last_emitted_step = 0u64;

    for step in 1..=config.n_steps {
        integ.step(state, solver, bg, config.dt);
        if step % config.snapshot_every == 0 {
            emit_snapshot(state, step, config, sink)?;
            emitted += 1;
            last_emitted_step = step;
        }
    }

    // Always capture the final step, unless it already landed on the cadence.
    if last_emitted_step != config.n_steps {
        emit_snapshot(state, config.n_steps, config, sink)?;
        emitted += 1;
    }

    Ok(RunSummary {
        steps: config.n_steps,
        final_time: state.time,
        snapshots_emitted: emitted,
    })
}

/// Stamp a header for the current state and hand it to the sink.
fn emit_snapshot(
    state: &State,
    step: u64,
    config: &SimConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<(), SimError> {
    let header = Header::for_state(
        state,
        step,
        config.softening,
        config.rng_seed,
        config.config_hash,
        config.units.as_str(),
    );
    sink.emit(&header, state)
}

// ---------------------------------------------------------------------------
// Global block-adaptive timestepping (plan: courant-quickening-cadence.md).
//
// Adaptive dt on the SPH path: the loop holds `dt` fixed across a BLOCK of ≤
// `block_steps` steps, recomputing it at each block boundary from the solver's CFL
// limit (`ForceSolver::max_stable_dt`). Snapshots land on a TIME cadence
// (`output_dt`), not a step count — variable dt breaks even step spacing (D3).
//
// The adaptive path deliberately forfeits leapfrog reversibility + energy
// oscillation (variable dt is not symplectic — D2); its gates are full-duration
// convergence to a fine-dt reference and contraction staleness (D2b), NOT the
// fixed-dt invariant gates, and there is NO energy gate (isothermal = heat bath).
// See the plan doc.
// ---------------------------------------------------------------------------

/// Policy + schedule for [`run_adaptive`]. `courant` and the safety controls are
/// timestep POLICY and live here (in the loop's config), never in the solver — the
/// solver reports only the physics CFL limit.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptiveConfig {
    /// Courant number applied to the solver's CFL limit: the block's target dt is
    /// `courant · max_stable_dt`. Must be in (0, 1); < 1 gives the stability margin
    /// that also absorbs mid-block bound tightening (D2b).
    pub courant: f64,
    /// Per-block dt growth cap: `dt_target ≤ max_growth · dt_prev` (must be ≥ 1).
    /// Bounds how fast dt ramps back up after a contraction — dt SHRINKS instantly
    /// (no cap), GROWS gradually (D2b). `dt_prev` tracks the CFL/growth target, not
    /// the landing-clamped dt, so a short final block does not throttle the next.
    pub max_growth: f64,
    /// Max steps held at one dt before re-querying the bound (the block size `B`;
    /// must be ≥ 1, and ≤ the GPU `MAX_BATCH` on the resident path). Bounded by D2b.
    pub block_steps: u64,
    /// Sim-time between emitted snapshots (must be > 0). Snapshots land exactly on
    /// integer multiples of this via a per-interval final-block clamp (D3).
    pub output_dt: f64,
    /// Number of output intervals; the run ends at `n_outputs · output_dt` (≥ 1).
    pub n_outputs: u64,
    /// Softening length recorded in headers (must match the solver's).
    pub softening: f64,
    /// RNG seed that produced the IC, recorded in headers.
    pub rng_seed: u64,
    /// Scenario config hash, recorded in headers.
    pub config_hash: u64,
    /// Units tag, recorded in headers.
    pub units: String,
}

/// One block's decision from [`plan_block`]: run `n_steps` at uniform `dt`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockPlan {
    /// Number of steps to run at `dt` this block (≥ 1).
    pub n_steps: u64,
    /// Uniform dt for these steps. `≤ dt_target` always; on the interval's final
    /// block it is `remaining / n_steps` so the block lands exactly on the output
    /// time (D3).
    pub dt: f64,
    /// The CFL/growth-limited target dt — the growth memory the caller carries into
    /// the next block as `dt_prev` (NOT `dt`, which may be clamped down to land).
    pub dt_target: f64,
    /// True iff this block reaches `remaining` exactly (the interval's final block).
    pub lands: bool,
}

/// Decide the next block: apply the Courant number + growth cap to the current CFL
/// `limit` (the `c_cfl = 1` value from [`ForceSolver::max_stable_dt`]), then size
/// the block to advance `remaining` sim-time — a full `block_steps` block at
/// `dt_target` if the interval does not fit, else a landing block of
/// `dt = remaining / n` (≤ `dt_target`) that reaches the output time exactly.
///
/// Pure (no stepping) so the D2b contraction-staleness property is unit-testable.
/// Requires `limit`, `prev_target`, `remaining` finite positive and a valid `cfg`.
pub fn plan_block(limit: f64, prev_target: f64, remaining: f64, cfg: &AdaptiveConfig) -> BlockPlan {
    // Courant number on the CFL limit, then the growth cap: dt SHRINKS instantly
    // (the `.min` with `courant·limit` wins on a tightening bound) but GROWS at most
    // `max_growth×` per block (D2b). The growth memory `prev_target` is the previous
    // block's `dt_target`, never its landing-clamped `dt`.
    let dt_target = (cfg.courant * limit).min(cfg.max_growth * prev_target);

    if remaining > dt_target * cfg.block_steps as f64 {
        // The interval does not fit in one block — advance a full block at dt_target
        // and re-query the bound next block (does not land on the output time).
        BlockPlan {
            n_steps: cfg.block_steps,
            dt: dt_target,
            dt_target,
            lands: false,
        }
    } else {
        // Final block of the interval: split `remaining` into whole steps of uniform
        // dt = remaining/n ≤ dt_target, landing exactly on the output time (D3). The
        // `max(1)` guards a huge bound (remaining/dt_target < 1 ⇒ a single step).
        let n = (remaining / dt_target).ceil().max(1.0) as u64;
        BlockPlan {
            n_steps: n,
            dt: remaining / n as f64,
            dt_target,
            lands: true,
        }
    }
}

/// Run the global block-adaptive stepping loop (plan: courant-quickening-cadence).
/// Emits the IC (time 0) then a snapshot at each output time `k · output_dt`
/// (`k = 1..=n_outputs`), holding dt fixed across blocks of ≤ `block_steps` and
/// recomputing it at block boundaries from `solver.max_stable_dt`. Header `step` is
/// the output index `k`; header/state time is the exact `k · output_dt`.
///
/// Errors on an invalid config, or if the initial CFL bound is not finite positive
/// (adaptive dt requires gas present — a gas-free state returns `+∞`, which has no
/// finite target; use the fixed-dt [`run`] for collisionless runs).
pub fn run_adaptive(
    state: &mut State,
    solver: &mut dyn ForceSolver,
    integ: &mut dyn Integrator,
    bg: &dyn Background,
    config: &AdaptiveConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<RunSummary, SimError> {
    // Validate the policy + schedule.
    if !(config.courant.is_finite() && config.courant > 0.0) {
        return Err(SimError::Config(format!(
            "courant must be a positive finite number, got {}",
            config.courant
        )));
    }
    if !(config.max_growth.is_finite() && config.max_growth >= 1.0) {
        return Err(SimError::Config(format!(
            "max_growth must be a finite number >= 1, got {}",
            config.max_growth
        )));
    }
    if config.block_steps == 0 {
        return Err(SimError::Config("block_steps must be >= 1".to_string()));
    }
    if !(config.output_dt.is_finite() && config.output_dt > 0.0) {
        return Err(SimError::Config(format!(
            "output_dt must be a positive finite number, got {}",
            config.output_dt
        )));
    }
    if config.n_outputs == 0 {
        return Err(SimError::Config("n_outputs must be >= 1".to_string()));
    }

    // Adaptive dt needs a finite CFL bound to size the step. A gas-free state returns
    // `+∞` (no hydro constraint) — use the fixed-dt `run` for collisionless runs.
    let limit0 = solver.max_stable_dt(state);
    if !(limit0.is_finite() && limit0 > 0.0) {
        return Err(SimError::Config(format!(
            "adaptive dt requires a finite positive CFL bound (gas present); \
             max_stable_dt = {limit0}"
        )));
    }

    // Step 0: the IC, stamped at time 0 (output index 0 ↔ time 0 is the cadence
    // contract). Emit before any integration, mirroring `run`.
    state.time = 0.0;
    emit_adaptive(state, 0, config, sink)?;
    let mut emitted = 1u64;

    // Seed the growth memory with the initial CFL target so the first block is
    // uncapped (`min(courant·limit0, max_growth·prev)` = `courant·limit0`).
    let mut prev_target = config.courant * limit0;
    let mut total_steps = 0u64;
    let mut t = 0.0_f64;

    for k in 1..=config.n_outputs {
        let t_target = k as f64 * config.output_dt;
        // Relative epsilon so FP wander cannot spawn a zero-length trailing block.
        let eps = 1e-12 * t_target.max(config.output_dt);
        while t < t_target - eps {
            let limit = solver.max_stable_dt(state);
            if !(limit.is_finite() && limit > 0.0) {
                return Err(SimError::Config(format!(
                    "CFL bound became non-finite mid-run (t = {t}); max_stable_dt = {limit}"
                )));
            }
            let plan = plan_block(limit, prev_target, t_target - t, config);
            // Hold `plan.dt` fixed across the block; the cached position-only accel
            // carries across the dt change (velocity-Verlet) — do NOT reprime.
            for _ in 0..plan.n_steps {
                integ.step(state, solver, bg, plan.dt);
            }
            t += plan.dt * plan.n_steps as f64;
            prev_target = plan.dt_target;
            total_steps += plan.n_steps;
        }
        // Land exactly on the output time (kill accumulated FP wander before emit).
        t = t_target;
        state.time = t_target;
        emit_adaptive(state, k, config, sink)?;
        emitted += 1;
    }

    Ok(RunSummary {
        steps: total_steps,
        final_time: config.n_outputs as f64 * config.output_dt,
        snapshots_emitted: emitted,
    })
}

/// Stamp a header for an adaptive-run snapshot (output index `k`, time already set
/// on `state`) and hand it to the sink.
fn emit_adaptive(
    state: &State,
    k: u64,
    config: &AdaptiveConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<(), SimError> {
    let header = Header::for_state(
        state,
        k,
        config.softening,
        config.rng_seed,
        config.config_hash,
        config.units.as_str(),
    );
    sink.emit(&header, state)
}

// ---------------------------------------------------------------------------
// Individual (per-particle rung) timestepping (plan: laddered-ember-cadence.md).
//
// The THIRD stepping path, beside fixed-dt `run` and global-adaptive
// `run_adaptive` — neither of those byte-paths is touched. Each gas particle sits
// on its own power-of-two rung below a per-block base dt and is re-integrated only
// when its rung is due (I1/I2 rung policy, I3 `ActiveSetKdk` mechanic). This driver
// owns the block loop: re-derive `dt_base` + rungs at each base-block boundary from
// the solver's per-particle CFL vector, sub-cycle the block via `ActiveSetKdk`, and
// emit snapshots on a TIME cadence (base dt varies block to block — a step count is
// meaningless, exactly as on the adaptive path, D3).
//
// I4a = `hydro-only` lever (a): gas CFL rungs, collisionless stars on the coarsest
// rung. Gates: full-DURATION convergence to a fine-courant reference (PRIMARY, on a
// testbed with a genuine ≥3-rung spread — else it is fixed-dt in disguise) and a
// momentum BOUNDED-DRIFT diagnostic (kick-active-only forfeits exact conservation —
// drift ∝ courant, so it shrinks as courant → 0; NOT a roundoff tripwire). No energy
// gate (isothermal heat bath + variable per-particle dt, D4/D2 carry over). The
// Saitoh–Makino timestep limiter + shock-wakeup gate are I4b; the thermal `u`-kick
// arm is I5; `[sim.individual].mode` scenario/snapshot wiring is owed before I6.
// ---------------------------------------------------------------------------

/// Policy + schedule for [`run_individual`]. Timestep POLICY (`courant`, the base-dt
/// cap, the rung depth) lives here in the loop's config, never in the solver — the
/// solver reports only the per-particle physics CFL vector.
#[derive(Clone, Debug, PartialEq)]
pub struct IndividualConfig {
    /// Courant number applied to each particle's CFL limit: a particle's safe
    /// sub-step is `courant · dt_i` and the base step is `courant · dt_coarsest`
    /// (capped). Must be in (0, 1]; < 1 gives the stability margin. Halving it
    /// halves every particle's step uniformly while the rung STRUCTURE is unchanged
    /// (as long as `dt_base_cap` does not bind) — the property the convergence gate
    /// leans on.
    pub courant: f64,
    /// Cap on the base (coarsest-rung) timestep: `dt_base = min(courant·dt_coarsest,
    /// dt_base_cap)`. A scenario ceiling that keeps the diffuse majority from an
    /// unbounded coarse step. Must be > 0. When it does NOT bind, the rung structure
    /// is courant-invariant (see `courant`); when it binds, finer courant pushes
    /// particles to finer rungs — keep it non-binding on convergence testbeds.
    pub dt_base_cap: f64,
    /// Maximum rung depth: the finest sub-step is `dt_base / 2^r_max`. A particle
    /// needing a finer step than `r_max` allows CLAMPS there and steps coarser than
    /// its CFL limit — bounded under-resolution with no red gate, so `r_max` must be
    /// chosen large enough for the run's dynamic range (a tracked I6 hazard; NOT what
    /// the I4b limiter covers — that is neighbour-rung coupling). Must be in [1, 60]
    /// (the rung set is tracked in a `u64` bitmask for the summary diagnostics).
    pub r_max: u32,
    /// Sim-time between emitted snapshots (must be > 0). Snapshots land exactly on
    /// integer multiples of this via a per-interval final-block clamp (D3) — the
    /// only place all rungs are synchronized (pos+vel consistent).
    pub output_dt: f64,
    /// Number of output intervals; the run ends at `n_outputs · output_dt` (≥ 1).
    pub n_outputs: u64,
    /// Softening length recorded in headers (must match the solver's).
    pub softening: f64,
    /// RNG seed that produced the IC, recorded in headers.
    pub rng_seed: u64,
    /// Scenario config hash, recorded in headers.
    pub config_hash: u64,
    /// Units tag, recorded in headers.
    pub units: String,
}

/// Summary returned by [`run_individual`]: the shared [`RunSummary`] plus the rung
/// DIAGNOSTICS the convergence gate needs to prove the run genuinely exercised the
/// multi-rung machinery (a single-rung run reduces to fixed-dt `LeapfrogKdk`, already
/// bit-pinned in I3, and would make the gates vacuous).
#[derive(Clone, Debug, PartialEq)]
pub struct IndividualSummary {
    /// The shared run summary (steps = base blocks integrated, final time, snapshots).
    pub run: RunSummary,
    /// The finest (largest) rung assigned to any particle over the whole run. A gate
    /// asserts this is `< r_max`, proving the reference run is not itself
    /// under-resolved (clamped) — which would make convergence meaningless.
    pub max_rung: u32,
    /// The number of DISTINCT rungs assigned over the whole run. A gate asserts this
    /// is ≥ 3 on the convergence/momentum testbed — genuine active-subset stepping,
    /// not fixed-dt in disguise.
    pub distinct_rungs: usize,
}

/// Run the individual-timestep stepping loop (plan: laddered-ember-cadence.md,
/// `hydro-only` mode / I4a). Emits the IC (time 0) then a snapshot at each output
/// time `k · output_dt` (`k = 1..=n_outputs`). Each base block re-derives `dt_base`
/// and the per-particle rungs from `solver.max_stable_dt_per_particle`, then
/// sub-cycles the block with an internally-owned [`individual::ActiveSetKdk`],
/// kicking only the active subset each fine tick. Header `step` is the output index
/// `k`; header/state time is the exact `k · output_dt`.
///
/// Errors on an invalid config, or if no gas is present (all per-particle CFL limits
/// are `+∞` — the individual path has no finite base dt to size; use the fixed-dt
/// [`run`] for collisionless runs, per Scope).
pub fn run_individual(
    state: &mut State,
    solver: &mut dyn ForceSolver,
    bg: &dyn Background,
    config: &IndividualConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<IndividualSummary, SimError> {
    let _ = (state, solver, bg, config, sink);
    todo!("I4a: run_individual driver")
}
