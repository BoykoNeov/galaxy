//! `simulate_snapshots`: the movie pipeline's simulate step, extracted from
//! `run_movie` so the gas-gated solver choice is unit-testable (M7c, D6).
//!
//! It picks the force solver by gas presence: pure Barnes-Hut for a
//! collisionless scenario (byte-identical to the pre-M7c pipeline — literally
//! the same solver + sink + `run`), or `GravitySph` (Barnes-Hut gravity +
//! isothermal SPH) wrapped by a [`CflGuard`] snapshot sink when the scenario
//! carries gas. On the gas path it validates the fixed global `dt` against the
//! hydro CFL bound at t=0 *before* any sink activity, so an over-large `dt`
//! fails loud before the first snapshot is written, not only at the first
//! emit (the `cfl_guard` contract). The single `Scenario::sound_speed` (already
//! baked into `state`'s pressure equilibrium) also drives the solver's
//! `HydroParams`, so IC and force law share one c_s and cannot diverge.

use std::path::Path;

use galaxy_core::{LeapfrogKdk, StaticBackground};
use galaxy_sim::{run, DirectorySink, RunSummary, SimConfig};
use galaxy_solvers::sph::{validate_dt, DensityConfig, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

use crate::cfl_guard::{CflGuard, C_CFL};
use crate::spec::Scenario;
use crate::{G, THETA};

/// Simulate `s` to `.snap` files under `snap_dir`, choosing Barnes-Hut (gas-free)
/// or `GravitySph` + `CflGuard` (gas-rich) by `s.sound_speed`. Errors on a t=0
/// CFL violation (gas path) or any sink/`run` failure.
pub fn simulate_snapshots(
    s: &Scenario,
    snap_dir: &Path,
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
        // Gas-rich: Barnes-Hut gravity + isothermal SPH, guarded by the CFL
        // sentinel. The one `sound_speed` (already baked into `state`'s pressure
        // equilibrium) also drives the solver's `HydroParams`, so IC and force
        // law share one c_s and cannot diverge.
        Some(sound_speed) => {
            let hydro = HydroParams {
                sound_speed,
                ..HydroParams::default()
            };
            let density_cfg = DensityConfig::default();
            // t=0 CFL check BEFORE any sink activity: fail loud before the first
            // snapshot is written, not only at the first emit (cfl_guard contract).
            validate_dt(&state, &hydro, &density_cfg, cfg.dt, C_CFL)
                .map_err(|v| format!("CFL violation at t=0: {v}"))?;
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
