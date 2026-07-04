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

use galaxy_core::{DVec3, Species, State};

use super::density::{density_adaptive, DensityConfig};
use super::forces::HydroParams;
use super::grid::HashGrid;
use super::kernel::SUPPORT;

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
pub fn max_stable_dt(state: &State, params: &HydroParams, cfg: &DensityConfig, c_cfl: f64) -> f64 {
    let gas: Vec<usize> = (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .collect();
    if gas.is_empty() {
        return f64::INFINITY;
    }
    let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
    let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
    let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
    let dens = density_adaptive(&gpos, &gmass, cfg, None);
    let h = &dens.h;

    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    let grid = HashGrid::build(&gpos, SUPPORT * h_max);
    let two_cs = 2.0 * params.sound_speed;

    let mut min_dt = f64::INFINITY;
    for i in 0..gpos.len() {
        // Gather at the GLOBAL max support and gate each pair on the real force
        // coupling range r < 2·max(h_i,h_j): the averaged kernel W̄ is nonzero
        // there, so a diffuse large-h_j neighbor drives particle i even when
        // 2·h_i < r. Querying only within 2·h_i would miss that approacher and
        // leave v_sig,i stuck at the 2c_s floor — overestimating the stable dt
        // (the force law it must track gathers at this same global radius). This
        // runs at snapshot cadence, not per step, so the wider gather is cheap.
        let ngb = grid.neighbours_within(&gpos, gpos[i], SUPPORT * h_max);
        // v_sig,i = max_j (2c_s − 3 w_ij) over APPROACHING neighbors
        // (w_ij = v_ij·r̂_ij < 0), floored at 2c_s.
        let mut v_sig = two_cs;
        for &j in &ngb {
            if j == i {
                continue;
            }
            let r_ij = gpos[i] - gpos[j];
            let r = r_ij.length();
            if r == 0.0 || r >= SUPPORT * h[i].max(h[j]) {
                continue; // outside the pair's force coupling range ⇒ no drive
            }
            let w = (gvel[i] - gvel[j]).dot(r_ij) / r; // projected relative velocity
            if w < 0.0 {
                v_sig = v_sig.max(two_cs - 3.0 * w);
            }
        }
        min_dt = min_dt.min(c_cfl * h[i] / v_sig);
    }
    min_dt
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
