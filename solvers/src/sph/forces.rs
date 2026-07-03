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
    let _ = (pos, vel, mass, rho, h, params);
    todo!("M7b: symmetric P/ρ² + Monaghan viscosity, gather-per-target (parallel)")
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
    let _ = (pos, vel, mass, rho, h, params);
    todo!("M7b: symmetric P/ρ² + Monaghan viscosity, gather-per-target (serial)")
}
