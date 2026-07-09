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
//! `v_sig,i = max(2 c_s,i, max_j (c_s,i + c_s,j − 3 w_ij))` over neighbors
//! (`w_ij = v_ij·r̂_ij`; the `−3w` term contributes only on approach, `w<0`),
//! floored at the self-pair `2 c_s,i`. On the isothermal EOS `c_s,i ≡ c_s`, so
//! this collapses to `max_j (2 c_s − 3 w_ij)` floored at `2 c_s` — bit-identical
//! to the pre-E4a path (`isothermal_cfl_pins_pre_e4a_bits`). On the adiabatic
//! EOS `c_s,i = √(γ(γ−1)u_i)` is per-particle (E4a). `h_i` is the adaptive
//! smoothing length (recomputed here, same routine the force path uses). A
//! pure-collisionless (no gas) state has no hydro CFL constraint, so the bound
//! is `+∞`.

use galaxy_core::{DVec3, Species, State};

use super::density::{density_adaptive, DensityConfig};
use super::forces::{Eos, HydroParams};
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

    // EOS selects the sound speed feeding `v_sig`. The isothermal arm is kept
    // TEXTUALLY VERBATIM against the pre-E4a implementation (`two_cs` outside the
    // loop, per-pair `two_cs − 3w`) so the bound stays bit-identical — see
    // `isothermal_cfl_pins_pre_e4a_bits`. The adiabatic arm generalizes to the
    // per-particle `c_s,i = √(γ(γ−1)u_i)` (E4a).
    match params.eos {
        Eos::Isothermal { .. } => {
            let two_cs = 2.0 * params.sound_speed();
            let mut min_dt = f64::INFINITY;
            for i in 0..gpos.len() {
                // Gather at the GLOBAL max support and gate each pair on the real
                // force coupling range r < 2·max(h_i,h_j): the averaged kernel W̄ is
                // nonzero there, so a diffuse large-h_j neighbor drives particle i
                // even when 2·h_i < r. Querying only within 2·h_i would miss that
                // approacher and leave v_sig,i stuck at the 2c_s floor —
                // overestimating the stable dt (the force law it must track gathers
                // at this same global radius). This runs at snapshot cadence, not
                // per step, so the wider gather is cheap.
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
        Eos::Adiabatic { .. } => {
            // Per-particle c_s,i = √(γ(γ−1)u_i) via the shared EOS helper.
            // v_sig,i = max(2·c_s,i, max_j(c_s,i + c_s,j − 3·min(0,w_ij))): the
            // Gadget-2 signal velocity. The pair term c_s,i+c_s,j is taken over
            // ALL neighbors, not just approaching ones — a hot neighbor's sound
            // wave reaches a resting particle, so it must tighten dt even at
            // w=0. (For the isothermal arm above this generalization is a
            // provable no-op: c_s,i+c_s,j = 2c_s = the floor.) The 2·c_s,i floor
            // is the self-pair, matching the isothermal `two_cs`.
            let cs: Vec<f64> = gas
                .iter()
                .map(|&i| params.eos.sound_speed_of(state.u[i]))
                .collect();
            let mut min_dt = f64::INFINITY;
            for i in 0..gpos.len() {
                // Same global-support gather + real-coupling-range gate as the
                // isothermal arm and the force law (cross-support approacher).
                let ngb = grid.neighbours_within(&gpos, gpos[i], SUPPORT * h_max);
                let mut v_sig = 2.0 * cs[i];
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
                    let approach = if w < 0.0 { -3.0 * w } else { 0.0 };
                    v_sig = v_sig.max(cs[i] + cs[j] + approach);
                }
                min_dt = min_dt.min(c_cfl * h[i] / v_sig);
            }
            min_dt
        }
    }
}

/// The per-particle CFL vector `dt_i = c_cfl · h_i / v_sig,i` (I1) — the additive
/// generalization of [`max_stable_dt`]'s scalar `min`, for individual timesteps.
///
/// Full-length and state-indexed: gas rows carry the finite bound; collisionless
/// rows carry `+∞` (no hydro constraint). By construction the vector's `min`
/// equals the scalar [`max_stable_dt`] (asserted bit-for-bit in the I1 gate) — the
/// shipped scalar stays FROZEN (this is a parallel copy, not a refactor of it), so
/// its `isothermal_cfl_pins_pre_e4a_bits` guard is untouched. The individual-
/// timestep driver bins these into power-of-two rungs (I2).
pub fn max_stable_dt_per_particle(
    state: &State,
    params: &HydroParams,
    cfg: &DensityConfig,
    c_cfl: f64,
) -> Vec<f64> {
    // State-indexed output; collisionless rows stay `+∞` (no hydro constraint).
    // Gas rows are written at their GLOBAL index `gas[k]`. A gas-free state keeps
    // the all-`+∞` fill (its `min` = the scalar's gas-free `+∞`).
    let mut out = vec![f64::INFINITY; state.len()];
    let gas: Vec<usize> = (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .collect();
    if gas.is_empty() {
        return out;
    }
    let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
    let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
    let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
    let dens = density_adaptive(&gpos, &gmass, cfg, None);
    let h = &dens.h;

    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    let grid = HashGrid::build(&gpos, SUPPORT * h_max);

    // The per-particle body is TEXTUALLY VERBATIM against `max_stable_dt`'s inner
    // loop (same EOS split, same global-support gather + cross-support gate), with
    // the `min_dt.min(...)` fold replaced by a store at the global index — so
    // `out.iter().min()` reproduces the scalar bound bit-for-bit (the I1 gate). The
    // scalar itself is left FROZEN; this is a parallel copy, not a refactor of it.
    match params.eos {
        Eos::Isothermal { .. } => {
            let two_cs = 2.0 * params.sound_speed();
            for i in 0..gpos.len() {
                let ngb = grid.neighbours_within(&gpos, gpos[i], SUPPORT * h_max);
                let mut v_sig = two_cs;
                for &j in &ngb {
                    if j == i {
                        continue;
                    }
                    let r_ij = gpos[i] - gpos[j];
                    let r = r_ij.length();
                    if r == 0.0 || r >= SUPPORT * h[i].max(h[j]) {
                        continue;
                    }
                    let w = (gvel[i] - gvel[j]).dot(r_ij) / r;
                    if w < 0.0 {
                        v_sig = v_sig.max(two_cs - 3.0 * w);
                    }
                }
                out[gas[i]] = c_cfl * h[i] / v_sig;
            }
        }
        Eos::Adiabatic { .. } => {
            let cs: Vec<f64> = gas
                .iter()
                .map(|&i| params.eos.sound_speed_of(state.u[i]))
                .collect();
            for i in 0..gpos.len() {
                let ngb = grid.neighbours_within(&gpos, gpos[i], SUPPORT * h_max);
                let mut v_sig = 2.0 * cs[i];
                for &j in &ngb {
                    if j == i {
                        continue;
                    }
                    let r_ij = gpos[i] - gpos[j];
                    let r = r_ij.length();
                    if r == 0.0 || r >= SUPPORT * h[i].max(h[j]) {
                        continue;
                    }
                    let w = (gvel[i] - gvel[j]).dot(r_ij) / r;
                    let approach = if w < 0.0 { -3.0 * w } else { 0.0 };
                    v_sig = v_sig.max(cs[i] + cs[j] + approach);
                }
                out[gas[i]] = c_cfl * h[i] / v_sig;
            }
        }
    }
    out
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
