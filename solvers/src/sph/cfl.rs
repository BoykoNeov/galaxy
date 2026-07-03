//! CFL timestep sentinel (DESIGN.md M7b, D6).
//!
//! Fixed global `dt` in v1; this is the fail-loud guard against picking one that
//! is too large. The stable bound is
//!
//! ```text
//! dt ≤ C_cfl · min_i h_i / v_sig,i
//! ```
//!
//! over the gas particles, with the Gadget-style projected signal velocity
//! `v_sig,i = max_j (2 c_s − 3 w_ij)` over approaching neighbors
//! (`w_ij = v_ij·r̂_ij < 0`), floored at `2 c_s` when nothing approaches. `h_i`
//! is the adaptive smoothing length (recomputed here, same routine the force
//! path uses). A pure-collisionless (no gas) state has no hydro CFL constraint,
//! so the bound is `+∞`.

use galaxy_core::State;

use super::density::DensityConfig;
use super::forces::HydroParams;

/// A CFL violation: the requested `dt` exceeds the stable bound `max_stable`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CflViolation {
    /// The requested timestep.
    pub dt: f64,
    /// The largest stable timestep for this state.
    pub max_stable: f64,
}

impl std::fmt::Display for CflViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CFL violation: dt = {} exceeds the stable bound {}",
            self.dt, self.max_stable
        )
    }
}

impl std::error::Error for CflViolation {}

/// The largest stable timestep `C_cfl · min_i h_i / v_sig,i` over the gas
/// particles, or `f64::INFINITY` if the state has no gas.
pub fn max_stable_dt(
    state: &State,
    params: &HydroParams,
    cfg: &DensityConfig,
    c_cfl: f64,
) -> f64 {
    let _ = (state, params, cfg, c_cfl);
    todo!("M7b: adaptive h + projected signal velocity → CFL bound")
}

/// Fail-loud check: `Ok(())` iff `dt ≤ max_stable_dt(...)`.
pub fn validate_dt(
    state: &State,
    params: &HydroParams,
    cfg: &DensityConfig,
    dt: f64,
    c_cfl: f64,
) -> Result<(), CflViolation> {
    let max_stable = max_stable_dt(state, params, cfg, c_cfl);
    if dt <= max_stable {
        Ok(())
    } else {
        Err(CflViolation { dt, max_stable })
    }
}
