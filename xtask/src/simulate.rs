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

use galaxy_sim::RunSummary;

use crate::spec::Scenario;

/// Simulate `s` to `.snap` files under `snap_dir`, choosing Barnes-Hut (gas-free)
/// or `GravitySph` + `CflGuard` (gas-rich) by `s.sound_speed`. Errors on a t=0
/// CFL violation (gas path) or any sink/`run` failure.
pub fn simulate_snapshots(
    s: &Scenario,
    snap_dir: &Path,
) -> Result<RunSummary, Box<dyn std::error::Error>> {
    let _ = (s, snap_dir);
    todo!("2c: gas-gated solver swap + t=0 CFL validate + CflGuard sink")
}
