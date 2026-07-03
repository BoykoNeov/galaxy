//! Isothermal SPH hydrodynamic forces (DESIGN.md M7b).
//!
//! The momentum equation in the symmetric `P/ρ²` form, isothermal EOS
//! `P = c_s²ρ`:
//!
//! ```text
//! a_i = −Σ_j m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) ∇_i W̄_ij
//! ```
//!
//! with the **kernel average** symmetrization (D2) `W̄_ij = ½(W(r,h_i)+W(r,h_j))`,
//! so `∇_i W̄_ij = ½(∇W(r_ij,h_i)+∇W(r_ij,h_j))`. This gradient is exactly the
//! negation of `∇_j W̄_ji` and is parallel to `r_ij`, so the pairwise force is
//! antisymmetric (linear momentum) and central (angular momentum) — both
//! conserved to roundoff. `Π_ij` is the Monaghan (1992) artificial viscosity,
//! active only on approach.
//!
//! Like the density path, forces GATHER per target over neighbors in ascending
//! index so the sum associates in a fixed order — the rayon path is bit-exact
//! against the serial one. The grid is built at `SUPPORT·h_max` (global) so no
//! averaged-kernel neighbor (`r < 2·max(h_i,h_j)`) is ever missed.

use galaxy_core::DVec3;
use rayon::prelude::*;

use super::grid::HashGrid;
use super::kernel::{grad_w, SUPPORT};

/// Isothermal SPH force parameters.
#[derive(Clone, Copy, Debug)]
pub struct HydroParams {
    /// Isothermal sound speed `c_s` (EOS `P = c_s²ρ`).
    pub sound_speed: f64,
    /// Monaghan viscosity linear coefficient α (default 1.0).
    pub alpha: f64,
    /// Monaghan viscosity quadratic coefficient β (default 2.0).
    pub beta: f64,
    /// Regularization ε² in the `μ` denominator (`r² + ε²·h̄²`); default 0.01
    /// keeps `μ` finite for near-coincident approaching pairs.
    pub visc_eps2: f64,
}

impl Default for HydroParams {
    fn default() -> Self {
        HydroParams {
            sound_speed: 1.0,
            alpha: 1.0,
            beta: 2.0,
            visc_eps2: 0.01,
        }
    }
}

/// Hydro acceleration per particle, rayon over targets. `rho`/`h` are supplied
/// (the density pass ran first); every slice has length `pos.len()`.
pub fn hydro_accelerations(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    params: &HydroParams,
) -> Vec<DVec3> {
    hydro_impl(pos, vel, mass, rho, h, params, true)
}

/// Serial twin of [`hydro_accelerations`] for the parallel ≡ serial gate.
pub fn hydro_accelerations_serial(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    params: &HydroParams,
) -> Vec<DVec3> {
    hydro_impl(pos, vel, mass, rho, h, params, false)
}

fn hydro_impl(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    params: &HydroParams,
    parallel: bool,
) -> Vec<DVec3> {
    let n = pos.len();
    if n == 0 {
        return Vec::new();
    }
    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        h_max.is_finite() && h_max > 0.0,
        "hydro_accelerations needs positive finite smoothing lengths"
    );
    // Gather at the GLOBAL max support (not per-target 2·h_i): the averaged
    // kernel W̄ = ½(W(h_i)+W(h_j)) is nonzero for r < 2·max(h_i,h_j), so a pair
    // with 2h_i < r < 2h_j contributes force to BOTH i and j. Querying only
    // 2·h_i would give i's force to j but not j's to i — Newton's third law and
    // momentum conservation would break. So this radius is load-bearing, not
    // just a "don't miss a neighbor" convenience.
    //
    // PERF (flagged for M7c): under a wide adaptive-h range (a
    // centrally-concentrated gas disk/merger) h_max is a far-outskirts value, so
    // this global-radius gather goes quadratic — the same trap the M7d density
    // deposition hit. The bit-exactness-preserving fix is the M7d scatter-by-
    // plane template: gather at 2·h_i, then SCATTER each pair's contribution to
    // both i and j over the h_j-reach in ascending index. Deferred: the M7b
    // shock tube has near-uniform h (h_max ≈ h_typical), so it stays cheap here.
    let grid = HashGrid::build(pos, SUPPORT * h_max);
    let cs2 = params.sound_speed * params.sound_speed;

    // Acceleration on target `i`: gather neighbors in ascending index (fixed
    // order ⇒ parallel ≡ serial bit-exact), sum the symmetric pressure term and
    // Monaghan viscosity against the exactly-negated grad-average.
    let accel_one = |i: usize| -> DVec3 {
        let xi = pos[i];
        let term_i = cs2 / rho[i]; // P_i/ρ_i² for the isothermal EOS
        let ngb = grid.neighbours_within(pos, xi, SUPPORT * h_max);
        let mut a = DVec3::ZERO;
        for &j in &ngb {
            if j == i {
                continue;
            }
            let r_ij = xi - pos[j];
            let r = r_ij.length();
            // W̄ = ½(W(h_i)+W(h_j)); ∇_j W̄_ji is the exact negation of this.
            let grad_avg = (grad_w(r_ij, h[i]) + grad_w(r_ij, h[j])) * 0.5;
            let term_j = cs2 / rho[j];
            // Monaghan artificial viscosity, active only on approach.
            let v_ij = vel[i] - vel[j];
            let vr = v_ij.dot(r_ij);
            let visc = if vr < 0.0 {
                let h_bar = 0.5 * (h[i] + h[j]);
                let rho_bar = 0.5 * (rho[i] + rho[j]);
                let mu = h_bar * vr / (r * r + params.visc_eps2 * h_bar * h_bar);
                // Isothermal: c̄ = c_s (constant sound speed).
                (-params.alpha * params.sound_speed * mu + params.beta * mu * mu) / rho_bar
            } else {
                0.0
            };
            let coeff = term_i + term_j + visc;
            // a_i += −m_j·coeff·∇_i W̄. Structured so the equal-mass pair term is
            // the exact negation of particle j's (coeff bit-identical by
            // commutativity, grad_avg exactly negated).
            a += grad_avg * (-mass[j] * coeff);
        }
        a
    };

    if parallel {
        (0..n).into_par_iter().map(accel_one).collect()
    } else {
        (0..n).map(accel_one).collect()
    }
}
