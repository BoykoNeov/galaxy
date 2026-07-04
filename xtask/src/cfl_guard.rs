//! CFL timestep sentinel as a [`SnapshotSink`] decorator (DESIGN.md M7b, D6).
//!
//! Fixed global `dt` in v1; this decorator is the fail-loud guard that a chosen
//! `dt` stays within the hydro CFL bound as the gas flow evolves. It wraps the
//! real sink and, at snapshot cadence, checks the emitted state with
//! [`galaxy_solvers::sph::validate_dt`] before delegating — a violation returns
//! `SimError::Config` so `sim::run` dies loudly instead of integrating an unstable
//! flow. The error rides the existing snapshot-cadence hook, so neither the
//! `SnapshotSink` trait nor `sim::run` widen.
//!
//! It lives in xtask, not solvers: it bridges `sim::SnapshotSink` and
//! `solvers::sph::validate_dt`, and the engine crates stay decoupled (D6 —
//! `validate_dt` in solvers, the sink glue here). The caller also runs
//! `validate_dt` once at t=0 before `run`, so an over-large `dt` fails before the
//! first step, not only at the first snapshot.

use galaxy_core::State;
use galaxy_io::Header;
use galaxy_sim::{SimError, SnapshotSink};
use galaxy_solvers::sph::{validate_dt, DensityConfig, HydroParams};

/// The Courant number used by the sentinel at t=0 and at snapshot cadence. D6
/// fixes it at 0.25 — conservative, so the sentinel trips before the leapfrog
/// integrator visibly blows up. (Distinct from a *stability* study's own C_cfl;
/// this is the pipeline guard.)
pub const C_CFL: f64 = 0.25;

/// A [`SnapshotSink`] decorator that validates the fixed global `dt` against the
/// hydro CFL bound of every emitted state, then delegates to the inner sink.
pub struct CflGuard<S: SnapshotSink> {
    inner: S,
    hydro: HydroParams,
    density_cfg: DensityConfig,
    dt: f64,
    c_cfl: f64,
}

impl<S: SnapshotSink> CflGuard<S> {
    /// Wrap `inner`, validating each emitted state against `dt ≤ c_cfl · min_i h_i /
    /// v_sig,i` for the given hydro parameters and density (h) configuration.
    pub fn new(
        inner: S,
        hydro: HydroParams,
        density_cfg: DensityConfig,
        dt: f64,
        c_cfl: f64,
    ) -> Self {
        Self {
            inner,
            hydro,
            density_cfg,
            dt,
            c_cfl,
        }
    }
}

impl<S: SnapshotSink> SnapshotSink for CflGuard<S> {
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError> {
        let _ = (header, state);
        todo!("CflGuard::emit — validate_dt then delegate (M7c)")
    }
}
