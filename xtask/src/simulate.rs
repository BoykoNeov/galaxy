//! `simulate_snapshots`: the movie pipeline's simulate step, extracted from
//! `run_movie` so the gas-gated solver choice is unit-testable (M7c, D6).
//!
//! It picks the force solver by gas presence: pure Barnes-Hut for a
//! collisionless scenario (byte-identical to the pre-M7c pipeline — literally
//! the same solver + sink + `run`), or `GravitySph` (Barnes-Hut gravity +
//! isothermal SPH) wrapped by a [`CflGuard`] snapshot sink when the scenario
//! carries gas. The guard validates the fixed global `dt` against the hydro CFL
//! bound of every emitted state *before* delegating to the real sink; since
//! `run` emits the t=0 IC as its first snapshot (before any integration step),
//! an over-large `dt` fails loud before the first snapshot is written — no
//! separate t=0 pre-check is needed. The single `Scenario::sound_speed` (already
//! baked into `state`'s pressure equilibrium) also drives the solver's
//! `HydroParams`, so IC and force law share one c_s and cannot diverge.

use std::path::Path;

use galaxy_core::{LeapfrogKdk, LeapfrogKdkThermal, State, StaticBackground};
use galaxy_gpu::GpuResidentLeapfrog;
use galaxy_io::Header;
use galaxy_sim::{
    plan_block, run, run_adaptive, run_individual, AdaptiveConfig, DirectorySink, IndividualConfig,
    RunSummary, SimConfig, SnapshotSink, ThermalArm,
};
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::{BarnesHut, TreeGravity};

use crate::cfl_guard::{CflGuard, C_CFL};
use crate::spec::{AdaptiveSpec, IndividualMode, IndividualSpec, Scenario};
use crate::{G, THETA};

/// Which force backend runs the gas-rich (`GravitySph`) branch of the simulate step.
///
/// The **gas-free** path is unaffected — it is always the pre-M7c CPU Barnes-Hut
/// pipeline regardless of `Backend` (the GPU-resident stepper is the SPH path, and
/// keeping gas-free on CPU preserves its byte-identity gate). `Backend` selects only
/// between the CPU composite [`GravitySph`] and the GPU-resident SPH stepper (G6) for
/// a scenario that carries gas.
///
/// The choice is an explicit parameter — never read from an env var *inside*
/// `simulate_snapshots` — so a single test can drive both paths and gate GPU-vs-CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// CPU composite `GravitySph` (Barnes-Hut gravity + isothermal SPH), guarded by
    /// [`CflGuard`]. The default — the movie pipeline is unchanged.
    #[default]
    Cpu,
    /// GPU-resident SPH on `GpuResidentLeapfrog` (G6). Gravity over all particles +
    /// hydro on the gas subset, resident across steps.
    Gpu,
}

/// Simulate `s` to `.snap` files under `snap_dir`, choosing Barnes-Hut (gas-free)
/// or `GravitySph` + `CflGuard` (gas-rich) by `s.sound_speed`. For a gas-rich
/// scenario, `backend` selects the CPU composite or the GPU-resident SPH stepper.
/// Errors on a CFL violation (gas path — caught by the guard at the t=0 IC emit
/// before any file is written) or any sink/`run` failure.
pub fn simulate_snapshots(
    s: &Scenario,
    snap_dir: &Path,
    backend: Backend,
) -> Result<RunSummary, Box<dyn std::error::Error>> {
    let mut state = s.state.clone();
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = SimConfig {
        dt: s.dt,
        n_steps: s.n_steps,
        snapshot_every: s.snapshot_every,
        softening: s.eps,
        rng_seed: s.seed,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    };

    match s.sound_speed {
        // Gas-rich: Barnes-Hut gravity + isothermal SPH. The one `sound_speed`
        // (already baked into `state`'s pressure equilibrium) also drives the
        // solver's `HydroParams`, so IC and force law share one c_s and cannot
        // diverge. `backend` picks the CPU composite (guarded by the CFL sentinel)
        // or the GPU-resident stepper (G6).
        Some(sound_speed) => {
            // Adiabatic gas (`[model.gas].gamma`, H5-C) is wired only on the CPU
            // block-adaptive path: the fixed-dt `CflGuard` calls
            // `HydroParams::sound_speed()` (panics on `Eos::Adiabatic`), the individual
            // arm hardcodes `ThermalArm::Isothermal`, and GPU adiabatic is deferred.
            // Reject the unsupported compositions loud — otherwise they run *silently
            // isothermal* (the IC seeds `u` but the isothermal integrator ignores it),
            // which is a black temperature render, not an error.
            if s.gamma.is_some() {
                if backend != Backend::Cpu {
                    return Err(
                        "adiabatic gas is CPU-only (the GPU adiabatic EOS is deferred)".into(),
                    );
                }
                if s.individual.is_some() {
                    return Err("adiabatic gas is not supported with [sim.individual] \
                                (the individual thermal arm is isothermal-only)"
                        .into());
                }
                if s.adaptive.is_none() {
                    return Err("adiabatic gas requires [sim.adaptive] (the fixed-dt CFL \
                                guard assumes a single isothermal c_s)"
                        .into());
                }
            }

            let hydro = HydroParams {
                eos: Eos::Isothermal { c_s: sound_speed },
                ..HydroParams::default()
            };
            let density_cfg = DensityConfig::default();

            // Individual (per-particle rung) timesteps take precedence when enabled at
            // `mode = hydro-only` (plan laddered-ember-cadence, lever a). CPU-only per
            // Scope; the parse-time mutual-exclusion keeps this from colliding with
            // `[sim.adaptive]`, and `mode = fixed-dt` drops through to the fixed-dt path
            // below (the layer under the toggle). No `CflGuard`: `run_individual` sizes
            // `dt_base` from the per-particle CFL bound each base block, so an over-large
            // fixed `dt` (now only the output cadence) can no longer make it unstable.
            if let Some(ind) = &s.individual {
                if matches!(
                    ind.mode,
                    IndividualMode::HydroOnly | IndividualMode::HydroGravity
                ) {
                    if backend != Backend::Cpu {
                        return Err("[sim.individual] timesteps are CPU-only; run the gas \
                                    path on the CPU backend"
                            .into());
                    }
                    let ind_cfg = build_individual_config(s, ind)?;
                    let bh = BarnesHut::new(G, s.eps, THETA);
                    let mut sink = DirectorySink::new(snap_dir)?;
                    // `hydro+gravity` subcycles gravity onto finite star rungs and MUST
                    // walk a cached stale tree (built once per base block): wrap Barnes-Hut
                    // in `TreeGravity` + enable the cached walk. `hydro-only` uses the
                    // FRESH walk (a fresh octree every fine tick) — tree-caching was
                    // REVERTED for hydro-only after the full-res re-measure showed the
                    // stale tree drives the merger core into a sustained finest-rung flood
                    // (6.4× deeper min-dt → 5.6× SLOWER, not the expected speedup); see
                    // docs/plans/laddered-ember-cadence.md (M-cache reverted).
                    let summary = if ind.mode == IndividualMode::HydroGravity {
                        let mut solver = GravitySph::new(TreeGravity::new(bh), hydro, density_cfg)
                            .with_gravity_cache(true);
                        run_individual(&mut state, &mut solver, &bg, &ind_cfg, &mut sink)?
                    } else {
                        let mut solver = GravitySph::new(bh, hydro, density_cfg);
                        run_individual(&mut state, &mut solver, &bg, &ind_cfg, &mut sink)?
                    };
                    return Ok(summary.run);
                }
                // `mode = fixed-dt`: fall through to the fixed-dt path below.
            }

            match (backend, s.adaptive) {
                // Fixed-dt CPU (adaptive off): the pre-A4 path. No separate t=0 pre-check
                // — `run` emits the t=0 IC first and `CflGuard` validates before
                // delegating, so an over-large `dt` fails loud before any file is written.
                (Backend::Cpu, None) => {
                    let gravity = BarnesHut::new(G, s.eps, THETA);
                    let mut solver = GravitySph::new(gravity, hydro, density_cfg.clone());
                    let inner = DirectorySink::new(snap_dir)?;
                    let mut sink = CflGuard::new(inner, hydro, density_cfg, cfg.dt, C_CFL);
                    Ok(run(
                        &mut state,
                        &mut solver,
                        &mut integ,
                        &bg,
                        &cfg,
                        &mut sink,
                    )?)
                }
                // Block-adaptive CPU (A4): the timestep is CFL-derived per block, so the
                // fixed-dt `CflGuard` is retired on this path — its premise (a human might
                // pick too large a `dt`) is void, since `plan_block` picks `dt ≤ courant·
                // limit` by construction; stability is structural + D2b-gated (D6).
                (Backend::Cpu, Some(a)) => {
                    let adaptive_cfg = build_adaptive_config(s, &a)?;
                    let gravity = BarnesHut::new(G, s.eps, THETA);
                    let mut sink = DirectorySink::new(snap_dir)?;
                    match s.gamma {
                        // Adiabatic (H5-C): evolved-`u` EOS `P=(γ−1)ρu` + the thermal
                        // integrator (E2/E3), sized by the per-particle adiabatic CFL arm
                        // (E4a) reading `c_s=√(γ(γ−1)u)`. The `with_u_floor` guards a
                        // rarefaction undershoot into negative `u`/NaN `c_s`.
                        Some(gamma) => {
                            let hydro_ad = HydroParams {
                                eos: Eos::Adiabatic { gamma },
                                ..hydro
                            };
                            let mut solver = GravitySph::new(gravity, hydro_ad, density_cfg);
                            let mut integ_t = LeapfrogKdkThermal::with_u_floor(s.u_floor);
                            Ok(run_adaptive(
                                &mut state,
                                &mut solver,
                                &mut integ_t,
                                &bg,
                                &adaptive_cfg,
                                &mut sink,
                            )?)
                        }
                        // Isothermal: the pre-H5-C path, byte-unchanged.
                        None => {
                            let mut solver = GravitySph::new(gravity, hydro, density_cfg);
                            Ok(run_adaptive(
                                &mut state,
                                &mut solver,
                                &mut integ,
                                &bg,
                                &adaptive_cfg,
                                &mut sink,
                            )?)
                        }
                    }
                }
                // Fixed-dt GPU-resident SPH (G6, adaptive off).
                (Backend::Gpu, None) => {
                    simulate_gas_gpu(&state, &cfg, hydro, density_cfg, snap_dir)
                }
                // Block-adaptive GPU-resident SPH (A3/A4): dt recomputed per block from
                // the on-device CFL bound.
                (Backend::Gpu, Some(a)) => {
                    let adaptive_cfg = build_adaptive_config(s, &a)?;
                    simulate_gas_gpu_adaptive(&state, &adaptive_cfg, hydro, density_cfg, snap_dir)
                }
            }
        }
        // Gas-free: the pre-M7c pipeline, byte-for-byte — plain Barnes-Hut, plain
        // DirectorySink, no CFL guard.
        None => {
            let mut solver = BarnesHut::new(G, s.eps, THETA);
            let mut sink = DirectorySink::new(snap_dir)?;
            Ok(run(
                &mut state,
                &mut solver,
                &mut integ,
                &bg,
                &cfg,
                &mut sink,
            )?)
        }
    }
}

/// Build the [`AdaptiveConfig`] for a gas-rich scenario's block-adaptive run. The output
/// cadence is derived so the adaptive run emits on the SAME time grid as the fixed-dt
/// path would (`output_dt = snapshot_every · dt`, `n_outputs = n_steps / snapshot_every`)
/// — the movie sees the identical snapshot times, only the substeps in between become
/// CFL-adaptive. Requires `n_steps` a clean multiple of `snapshot_every` (a whole output
/// grid); rejects otherwise so a mis-timed movie can't be produced silently.
fn build_adaptive_config(
    s: &Scenario,
    a: &AdaptiveSpec,
) -> Result<AdaptiveConfig, Box<dyn std::error::Error>> {
    if s.snapshot_every == 0 {
        return Err("snapshot_every must be >= 1".into());
    }
    if !s.n_steps.is_multiple_of(s.snapshot_every) {
        return Err(format!(
            "adaptive dt needs a whole output grid: n_steps ({}) must be a multiple of \
             snapshot_every ({})",
            s.n_steps, s.snapshot_every
        )
        .into());
    }
    Ok(AdaptiveConfig {
        courant: a.courant,
        max_growth: a.max_growth,
        block_steps: a.block_steps,
        output_dt: s.snapshot_every as f64 * s.dt,
        n_outputs: s.n_steps / s.snapshot_every,
        softening: s.eps,
        rng_seed: s.seed,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    })
}

/// Build the [`IndividualConfig`] for a gas-rich scenario's individual-timestep run.
/// Like [`build_adaptive_config`], the output cadence is derived so the run emits on the
/// SAME time grid the fixed-dt path would (`output_dt = snapshot_every · dt`,
/// `n_outputs = n_steps / snapshot_every`) — `dt` becomes purely the output cadence, the
/// sub-steps are CFL-derived rungs. Requires `n_steps` a clean multiple of
/// `snapshot_every` (a whole output grid); rejects otherwise.
///
/// The EOS arm is `Isothermal`: scenarios express only isothermal gas (a single global
/// `c_s`); the adiabatic scenario wiring is deferred (E-series), so the adiabatic
/// individual arm is reachable only from sim unit tests, not from a scenario.
fn build_individual_config(
    s: &Scenario,
    ind: &IndividualSpec,
) -> Result<IndividualConfig, Box<dyn std::error::Error>> {
    if s.snapshot_every == 0 {
        return Err("snapshot_every must be >= 1".into());
    }
    if !s.n_steps.is_multiple_of(s.snapshot_every) {
        return Err(format!(
            "individual dt needs a whole output grid: n_steps ({}) must be a multiple of \
             snapshot_every ({})",
            s.n_steps, s.snapshot_every
        )
        .into());
    }
    Ok(IndividualConfig {
        courant: ind.courant,
        dt_base_cap: ind.dt_base_cap,
        r_max: ind.r_max,
        n_limit: ind.n_limit,
        // `hydro+gravity` (I-grav): subcycle gravity on the stale tree, stars get
        // finite gravitational rungs. `grav_eta` scales the gravitational TIMESCALE
        // `√(ε/|a|)`; `courant` is then applied uniformly to hydro AND gravity (a
        // single global safety factor ⇒ the rung structure stays courant-invariant,
        // which the convergence gate needs). `1.0` ⇒ the grav safe step is
        // `courant·√(ε/|a|)`, matching the hydro `courant·h/v_sig`. A scenario knob is
        // deferred until a run needs to tune it.
        // Only `hydro+gravity` caches the gravity tree (built once per base block,
        // walked stale) — because subcycling gravity onto star rungs walks that cache.
        // `hydro-only` was REVERTED to the FRESH walk (a fresh octree every fine tick):
        // at full res the stale tree drove the merger core into a sustained finest-rung
        // flood (6.4× deeper min-dt → 5.6× SLOWER); the build-once saving is swamped by
        // the extra fine-tick work. See docs/plans/laddered-ember-cadence.md.
        cache_gravity_tree: ind.mode == IndividualMode::HydroGravity,
        subcycle_gravity: ind.mode == IndividualMode::HydroGravity,
        grav_eta: 1.0,
        eos: ThermalArm::Isothermal,
        output_dt: s.snapshot_every as f64 * s.dt,
        n_outputs: s.n_steps / s.snapshot_every,
        softening: s.eps,
        rng_seed: s.seed,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    })
}

/// The GPU-resident SPH simulate branch (G6): drive `GpuResidentLeapfrog` in gas mode
/// over the same fixed-`dt` cadence `run` uses, emitting the same `.snap` files.
///
/// Unlike the CPU path this cannot reuse [`galaxy_sim::run`]: the resident stepper owns
/// its step loop (positions/velocities live on the GPU across steps), so the loop is
/// hand-rolled — `upload → step_many(interval) → snapshot → emit` — mirroring `run`'s
/// snapshot schedule exactly (including the always-capture-final-step tail).
///
/// Two resident-specific hazards are handled explicitly:
/// - **Re-upload per snapshot interval.** G5b freezes the hydro gather radius at
///   `SUPPORT·h_max` at upload; contraction over-covers (safe) but *expansion*
///   under-covers → missing hydro pairs, silently, with no gate red. The QUICK gasrich
///   merger expands post-pericenter, so we re-upload each interval to recalibrate
///   `h_max`/`cell` (the sanctioned mitigation — the on-GPU gas-bbox reduction that
///   would make a single-upload long run safe is deferred). **Consequence:** this path
///   does NOT exercise the single-upload-long-run failure mode, so a green gate here
///   must not be read as "frozen `h_max` proven safe for one upload".
/// - **Column re-attach.** [`GpuResidentLeapfrog::snapshot`] rebuilds `State` via
///   `from_phase_space`, which resets `kind→Collisionless`, `progenitor→0`,
///   `id→sequential`. The stepper preserves upload index order, so we re-stamp the
///   uploaded `id`/`progenitor`/`kind` onto every snapshot — before both the emit (or
///   the gas subset and `sf_progenitors` coloring vanish) AND the re-upload (or the gas
///   map rebuilt from `kind` comes back empty).
///
/// The stepper resets its clock to 0 on each [`upload`](GpuResidentLeapfrog::upload), so
/// absolute time is tracked here (`step·dt`) and stamped onto each snapshot; otherwise
/// every snapshot after the first would be mis-timed.
fn simulate_gas_gpu(
    state: &State,
    cfg: &SimConfig,
    hydro: HydroParams,
    density_cfg: DensityConfig,
    snap_dir: &Path,
) -> Result<RunSummary, Box<dyn std::error::Error>> {
    if cfg.snapshot_every == 0 {
        return Err("snapshot_every must be >= 1".into());
    }
    if !cfg.dt.is_finite() || cfg.dt <= 0.0 {
        return Err(format!("dt must be a positive finite number, got {}", cfg.dt).into());
    }

    let mut stepper =
        GpuResidentLeapfrog::new_with_sph(G, cfg.softening, THETA, hydro, density_cfg)?;
    stepper.upload(state);

    // Fail-loud t=0 CFL check, mirroring the CPU path's `CflGuard`: reject an over-large
    // fixed `dt` against the hydro CFL bound of the IC before any `.snap` is written.
    let dt_max = stepper.min_stable_dt(C_CFL);
    if cfg.dt > dt_max {
        return Err(format!(
            "dt {} exceeds the t=0 hydro CFL bound {} (c_cfl = {})",
            cfg.dt, dt_max, C_CFL
        )
        .into());
    }

    let mut sink = DirectorySink::new(snap_dir)?;

    // The exact emit schedule `galaxy_sim::run` uses: step 0 (IC) always, then every
    // `snapshot_every`-th step, then the final step if it did not already land on cadence.
    let mut schedule: Vec<u64> = vec![0];
    for step in 1..=cfg.n_steps {
        if step % cfg.snapshot_every == 0 {
            schedule.push(step);
        }
    }
    if *schedule.last().expect("schedule always has step 0") != cfg.n_steps {
        schedule.push(cfg.n_steps);
    }
    let last_step = *schedule.last().expect("schedule non-empty");

    // Step 0: the IC is exactly the uploaded state (columns intact, time 0) — emit it
    // directly, bit-faithful to `run`'s first snapshot.
    emit_gpu_snapshot(&mut sink, state, 0, cfg)?;
    let mut emitted = 1u64;

    let mut prev = 0u64;
    for &target in &schedule[1..] {
        stepper.step_many(cfg.dt, target - prev);
        let mut snap = stepper.snapshot();
        reattach_columns(&mut snap, state);
        snap.time = target as f64 * cfg.dt; // absolute — the stepper clock reset on re-upload
        emit_gpu_snapshot(&mut sink, &snap, target, cfg)?;
        emitted += 1;

        // Re-upload to bound frozen-`h_max` staleness (the expansion landmine). Skip after
        // the final emit — nothing steps afterward, so its recalibration would be wasted.
        if target != last_step {
            stepper.upload(&snap);
        }
        prev = target;
    }

    Ok(RunSummary {
        steps: cfg.n_steps,
        final_time: cfg.n_steps as f64 * cfg.dt,
        snapshots_emitted: emitted,
    })
}

/// The GPU-resident **block-adaptive** SPH simulate branch (A3, plan
/// courant-quickening-cadence): like [`simulate_gas_gpu`] but the timestep is chosen
/// per block from the on-device CFL bound instead of a fixed `dt`. Mirrors the CPU
/// [`galaxy_sim::run_adaptive`] loop, sharing its [`plan_block`] block-sizing decision
/// (so both paths obey the same D2b safety analysis — NOT for a cross-path trajectory
/// gate, which D4 forbids). Snapshots land on the time grid `k · output_dt`.
///
/// The Courant number lives in `adaptive` (via `plan_block`), so `min_stable_dt` is
/// queried at the raw CFL limit (`c_cfl = 1.0`) — the same split the CPU path uses
/// (`max_stable_dt` reports the `c_cfl = 1` limit, the loop applies the Courant × cap).
///
/// The two resident hazards [`simulate_gas_gpu`] documents still apply and are handled
/// identically: **re-upload per output interval** (frozen-`h_max` recalibration — the
/// expansion landmine; the block re-query handles the *contraction* landmine D2b), and
/// **column re-attach** after each `snapshot` (`from_phase_space` drops kind/progenitor).
/// Absolute time is host-tracked (`k · output_dt`) since `upload` resets the stepper clock.
pub fn simulate_gas_gpu_adaptive(
    state: &State,
    adaptive: &AdaptiveConfig,
    hydro: HydroParams,
    density_cfg: DensityConfig,
    snap_dir: &Path,
) -> Result<RunSummary, Box<dyn std::error::Error>> {
    // Validate the policy + schedule (mirrors `galaxy_sim::run_adaptive`).
    if !(adaptive.courant.is_finite() && adaptive.courant > 0.0) {
        return Err(format!(
            "courant must be a positive finite number, got {}",
            adaptive.courant
        )
        .into());
    }
    if !(adaptive.max_growth.is_finite() && adaptive.max_growth >= 1.0) {
        return Err(format!(
            "max_growth must be a finite number >= 1, got {}",
            adaptive.max_growth
        )
        .into());
    }
    if adaptive.block_steps == 0 {
        return Err("block_steps must be >= 1".into());
    }
    if !(adaptive.output_dt.is_finite() && adaptive.output_dt > 0.0) {
        return Err(format!(
            "output_dt must be a positive finite number, got {}",
            adaptive.output_dt
        )
        .into());
    }
    if adaptive.n_outputs == 0 {
        return Err("n_outputs must be >= 1".into());
    }

    let mut stepper =
        GpuResidentLeapfrog::new_with_sph(G, adaptive.softening, THETA, hydro, density_cfg)?;
    stepper.upload(state);

    // Raw CFL limit (c_cfl = 1; the Courant number lives in `plan_block`, the same
    // physics/policy split the CPU path uses). Adaptive dt needs a finite bound — a
    // gas-free IC returns +∞, which has no finite target.
    let limit0 = stepper.min_stable_dt(1.0);
    if !(limit0.is_finite() && limit0 > 0.0) {
        return Err(format!(
            "adaptive dt requires a finite positive CFL bound (gas present); min_stable_dt = {limit0}"
        )
        .into());
    }

    let mut sink = DirectorySink::new(snap_dir)?;

    // Step 0: the IC (columns intact) at time 0.
    let mut ic = state.clone();
    ic.time = 0.0;
    emit_gpu_adaptive_snapshot(&mut sink, &ic, 0, adaptive)?;
    let mut emitted = 1u64;

    // Seed the growth memory with the initial CFL target so the first block is uncapped.
    let mut prev_target = adaptive.courant * limit0;
    let mut total_steps = 0u64;
    let mut t = 0.0_f64;

    for k in 1..=adaptive.n_outputs {
        let t_target = k as f64 * adaptive.output_dt;
        let eps = 1e-12 * t_target.max(adaptive.output_dt);
        while t < t_target - eps {
            let limit = stepper.min_stable_dt(1.0);
            if !(limit.is_finite() && limit > 0.0) {
                return Err(format!(
                    "CFL bound became non-finite mid-run (t = {t}); min_stable_dt = {limit}"
                )
                .into());
            }
            // Same block-sizing decision as the CPU path (shared D2b analysis): hold
            // `plan.dt` across the block, land the interval's final block exactly on
            // `t_target`. `step_many` runs the block at the single fixed dt (the
            // residency win) — the cached accel carries across the dt change.
            let plan = plan_block(limit, prev_target, t_target - t, adaptive);
            stepper.step_many(plan.dt, plan.n_steps);
            t += plan.dt * plan.n_steps as f64;
            prev_target = plan.dt_target;
            total_steps += plan.n_steps;
        }
        // Emit at the output time: read back, re-attach the columns `snapshot` dropped,
        // stamp the absolute time (the stepper clock reset on the last upload).
        let mut snap = stepper.snapshot();
        reattach_columns(&mut snap, state);
        snap.time = t_target;
        emit_gpu_adaptive_snapshot(&mut sink, &snap, k, adaptive)?;
        emitted += 1;

        // Re-upload to recalibrate the frozen `h_max` (the expansion landmine); the block
        // re-query already handles the contraction landmine (D2b). Skip after the final
        // emit — nothing steps afterward.
        if k != adaptive.n_outputs {
            stepper.upload(&snap);
        }
        t = t_target; // kill accumulated FP wander before the next interval
    }

    Ok(RunSummary {
        steps: total_steps,
        final_time: adaptive.n_outputs as f64 * adaptive.output_dt,
        snapshots_emitted: emitted,
    })
}

/// Stamp a header for an adaptive GPU snapshot (output index `k`, time already set on
/// `state`) and hand it to the sink.
fn emit_gpu_adaptive_snapshot(
    sink: &mut DirectorySink,
    state: &State,
    k: u64,
    adaptive: &AdaptiveConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = Header::for_state(
        state,
        k,
        adaptive.softening,
        adaptive.rng_seed,
        adaptive.config_hash,
        adaptive.units.as_str(),
    );
    sink.emit(&header, state)?;
    Ok(())
}

/// Re-stamp the identity columns (`id`, `progenitor`, `kind`) a GPU `snapshot` dropped,
/// from the uploaded state. The resident stepper preserves upload index order, so the
/// columns map 1:1 by index (Species does not change without star formation).
fn reattach_columns(snap: &mut State, src: &State) {
    snap.id.clone_from(&src.id);
    snap.progenitor.clone_from(&src.progenitor);
    snap.kind.clone_from(&src.kind);
}

/// Stamp a header for `state` at `step` and hand it to the sink — the GPU branch's
/// equivalent of `galaxy_sim::run`'s private `emit_snapshot`.
fn emit_gpu_snapshot(
    sink: &mut DirectorySink,
    state: &State,
    step: u64,
    cfg: &SimConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = Header::for_state(
        state,
        step,
        cfg.softening,
        cfg.rng_seed,
        cfg.config_hash,
        cfg.units.as_str(),
    );
    sink.emit(&header, state)?;
    Ok(())
}
