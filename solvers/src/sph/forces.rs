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

/// Equation of state selecting the SPH pressure/sound-speed closure (E1b).
/// `Isothermal` is the M7b default; `Adiabatic` evolves the per-particle
/// internal energy `u` (see `State.u`, E1a) via `P=(γ−1)ρu`.
#[derive(Clone, Copy, Debug)]
pub enum Eos {
    /// `P = c_s²ρ`, constant sound speed.
    Isothermal {
        /// Isothermal sound speed `c_s`.
        c_s: f64,
    },
    /// Ideal-gas adiabatic EOS: `P=(γ−1)ρu`, `c_s=√(γ(γ−1)u)`.
    Adiabatic {
        /// Adiabatic index γ (e.g. 5/3 monatomic, 1.4 diatomic).
        gamma: f64,
    },
}

impl Eos {
    /// Per-particle sound speed for internal energy `u`. Isothermal ignores `u`
    /// (constant `c_s`); adiabatic is `√(γ(γ−1)u)`. Single source of the
    /// adiabatic `c_s` formula shared by the force loop (`forces.rs`) and the
    /// CFL path (`cfl.rs`) so the two cannot drift.
    #[inline]
    pub fn sound_speed_of(&self, u: f64) -> f64 {
        match *self {
            Eos::Isothermal { c_s } => c_s,
            Eos::Adiabatic { gamma } => (gamma * (gamma - 1.0) * u).sqrt(),
        }
    }
}

/// SPH force parameters.
#[derive(Clone, Copy, Debug)]
pub struct HydroParams {
    /// Equation of state.
    pub eos: Eos,
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
            eos: Eos::Isothermal { c_s: 1.0 },
            alpha: 1.0,
            beta: 2.0,
            visc_eps2: 0.01,
        }
    }
}

impl HydroParams {
    /// The isothermal sound speed. Isothermal-only consumers (`cfl.rs`, GPU
    /// src); per-particle adiabatic `c_s` in the CFL path lands in E4.
    pub fn sound_speed(&self) -> f64 {
        match self.eos {
            Eos::Isothermal { c_s } => c_s,
            Eos::Adiabatic { .. } => {
                panic!(
                    "HydroParams::sound_speed() called on Adiabatic EOS — per-particle c_s is E4"
                )
            }
        }
    }
}

/// Hydro acceleration per particle, rayon over targets. `rho`/`h` are supplied
/// (the density pass ran first); `u` is the per-particle internal energy
/// (ignored on the isothermal path); every slice has length `pos.len()`.
/// Thin wrapper over [`hydro_accel_and_dudt`] dropping `dudt` (E2a) — same
/// code path, so this output stays bit-identical to the pre-E2a function.
pub fn hydro_accelerations(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    u: &[f64],
    params: &HydroParams,
) -> Vec<DVec3> {
    hydro_accel_and_dudt_impl(pos, vel, mass, rho, h, u, params, true).0
}

/// Serial twin of [`hydro_accelerations`] for the parallel ≡ serial gate.
pub fn hydro_accelerations_serial(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    u: &[f64],
    params: &HydroParams,
) -> Vec<DVec3> {
    hydro_accel_and_dudt_impl(pos, vel, mass, rho, h, u, params, false).0
}

/// Fused acceleration + `du/dt` pass (E2a plumbing, E3 heating): the
/// thermodynamic partner of [`hydro_accelerations`], computed in the SAME
/// neighbor loop as the force.
/// `du_i/dt = Σ_j m_j (term_i + ½·Π_ij)(v_ij·∇_i W̄_ij)` — the PdV work
/// (`term_i` ALONE, the exact energy-conserving partner of the symmetric
/// `P/ρ²` momentum term) plus the Monaghan viscous-heating term `½·Π_ij` (E3),
/// the energy-conserving mate of the momentum viscosity and the entropy source.
/// Returns `(acc, dudt)`; `hydro_accelerations` wraps this and drops `dudt`, so
/// its acceleration output stays bit-identical.
#[allow(clippy::too_many_arguments)]
pub fn hydro_accel_and_dudt(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    u: &[f64],
    params: &HydroParams,
) -> (Vec<DVec3>, Vec<f64>) {
    hydro_accel_and_dudt_impl(pos, vel, mass, rho, h, u, params, true)
}

/// Serial twin of [`hydro_accel_and_dudt`] for the parallel ≡ serial gate.
#[allow(clippy::too_many_arguments)]
pub fn hydro_accel_and_dudt_serial(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    u: &[f64],
    params: &HydroParams,
) -> (Vec<DVec3>, Vec<f64>) {
    hydro_accel_and_dudt_impl(pos, vel, mass, rho, h, u, params, false)
}

#[allow(clippy::too_many_arguments)]
fn hydro_accel_and_dudt_impl(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
    u: &[f64],
    params: &HydroParams,
    parallel: bool,
) -> (Vec<DVec3>, Vec<f64>) {
    let n = pos.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
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

    // NOTE (E1b scope guard): the isothermal arm below is kept textually
    // VERBATIM against the pre-E1b implementation (same `cs2`, same per-pair
    // operation order, `c̄ = c_s`) so byte-identity is structural, not
    // incidental — see the frozen-bits regression test.
    match params.eos {
        Eos::Isothermal { c_s } => {
            let cs2 = c_s * c_s;

            // Acceleration + du/dt on target `i`: gather neighbors in ascending
            // index (fixed order ⇒ parallel ≡ serial bit-exact), sum the
            // symmetric pressure term and Monaghan viscosity against the
            // exactly-negated grad-average. `dudt_i` accumulates the PdV work
            // ALONE (`term_i`, not `term_i+term_j+visc`) — the exact
            // energy-conserving partner of the symmetric momentum term
            // (viscous heating is E3); reuses `v_ij`/`grad_avg` already
            // computed for the accel sum, so accel's ops/order are untouched.
            let one = |i: usize| -> (DVec3, f64) {
                let xi = pos[i];
                let term_i = cs2 / rho[i]; // P_i/ρ_i² for the isothermal EOS
                let ngb = grid.neighbours_within(pos, xi, SUPPORT * h_max);
                let mut a = DVec3::ZERO;
                let mut dudt_i = 0.0;
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
                        (-params.alpha * c_s * mu + params.beta * mu * mu) / rho_bar
                    } else {
                        0.0
                    };
                    let coeff = term_i + term_j + visc;
                    // a_i += −m_j·coeff·∇_i W̄. Structured so the equal-mass pair term is
                    // the exact negation of particle j's (coeff bit-identical by
                    // commutativity, grad_avg exactly negated).
                    a += grad_avg * (-mass[j] * coeff);
                    // du_i/dt = Σ_j m_j (term_i + ½·Π_ij)(v_ij·∇W̄): PdV work +
                    // the viscous-heating partner (E3). The ½ makes it the exact
                    // energy-conserving mate of the momentum viscosity (pairwise
                    // KE↔U cancellation, mod time integration) and it is ≥0 on
                    // approach — the entropy source. Accel path above is untouched
                    // (byte-identity of `hydro_accelerations` holds).
                    dudt_i += mass[j] * (term_i + 0.5 * visc) * v_ij.dot(grad_avg);
                }
                (a, dudt_i)
            };

            if parallel {
                (0..n).into_par_iter().map(one).unzip()
            } else {
                (0..n).map(one).unzip()
            }
        }
        Eos::Adiabatic { gamma } => {
            // Per-particle P_i=(γ−1)ρ_i u_i ⇒ term_i = P_i/ρ_i² = (γ−1)u_i/ρ_i.
            // Precomputed once (not per neighbor) since both i and j read it.
            // `sound_speed_of` is the shared adiabatic c_s formula (also used by
            // cfl.rs) — bit-identical to the inlined `√(γ(γ−1)u)` it replaced.
            let cs: Vec<f64> = (0..n).map(|k| params.eos.sound_speed_of(u[k])).collect();

            let one = |i: usize| -> (DVec3, f64) {
                let xi = pos[i];
                let term_i = (gamma - 1.0) * u[i] / rho[i];
                let ngb = grid.neighbours_within(pos, xi, SUPPORT * h_max);
                let mut a = DVec3::ZERO;
                let mut dudt_i = 0.0;
                for &j in &ngb {
                    if j == i {
                        continue;
                    }
                    let r_ij = xi - pos[j];
                    let r = r_ij.length();
                    let grad_avg = (grad_w(r_ij, h[i]) + grad_w(r_ij, h[j])) * 0.5;
                    let term_j = (gamma - 1.0) * u[j] / rho[j];
                    let v_ij = vel[i] - vel[j];
                    let vr = v_ij.dot(r_ij);
                    let visc = if vr < 0.0 {
                        let h_bar = 0.5 * (h[i] + h[j]);
                        let rho_bar = 0.5 * (rho[i] + rho[j]);
                        let mu = h_bar * vr / (r * r + params.visc_eps2 * h_bar * h_bar);
                        // Adiabatic: c̄ = ½(c_s,i+c_s,j) (pair-averaged, unlike the
                        // isothermal constant c_s).
                        let c_bar = 0.5 * (cs[i] + cs[j]);
                        (-params.alpha * c_bar * mu + params.beta * mu * mu) / rho_bar
                    } else {
                        0.0
                    };
                    let coeff = term_i + term_j + visc;
                    a += grad_avg * (-mass[j] * coeff);
                    // du_i/dt = Σ_j m_j (term_i + ½·Π_ij)(v_ij·∇W̄) — see the
                    // isothermal branch above; c̄=½(c_s,i+c_s,j) here.
                    dudt_i += mass[j] * (term_i + 0.5 * visc) * v_ij.dot(grad_avg);
                }
                (a, dudt_i)
            };

            if parallel {
                (0..n).into_par_iter().map(one).unzip()
            } else {
                (0..n).map(one).unzip()
            }
        }
    }
}
