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

use galaxy_core::{LeapfrogKdk, StaticBackground};
use galaxy_sim::{run, DirectorySink, RunSummary, SimConfig};
use galaxy_solvers::sph::{DensityConfig, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

use crate::cfl_guard::{CflGuard, C_CFL};
use crate::spec::Scenario;
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
            let hydro = HydroParams {
                sound_speed,
                ..HydroParams::default()
            };
            let density_cfg = DensityConfig::default();
            match backend {
                Backend::Cpu => {
                    // No separate t=0 pre-check: `run` emits the t=0 IC as its first
                    // snapshot before any integration step, and `CflGuard` validates
                    // before delegating — so an over-large `dt` already fails loud at
                    // the IC emit, before a single `.snap` file is written.
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
                Backend::Gpu => simulate_gas_gpu(&state, &cfg, hydro, density_cfg, snap_dir),
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

/// The GPU-resident SPH simulate branch (G6): drive `GpuResidentLeapfrog` in gas mode
/// over the same fixed-`dt` cadence `run` uses, emitting the same `.snap` files.
///
/// Unlike the CPU path this cannot reuse [`galaxy_sim::run`]: the resident stepper owns
/// its step loop (positions/velocities live on the GPU across steps), so the loop is
/// hand-rolled — `upload → step_many(interval) → snapshot → emit` — mirroring `run`'s
/// snapshot schedule exactly (including the always-capture-final-step tail).
fn simulate_gas_gpu(
    _state: &galaxy_core::State,
    _cfg: &SimConfig,
    _hydro: HydroParams,
    _density_cfg: DensityConfig,
    _snap_dir: &Path,
) -> Result<RunSummary, Box<dyn std::error::Error>> {
    todo!("G6: GPU-resident SPH simulate branch")
}
