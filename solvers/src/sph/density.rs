//! SPH density summation with adaptive smoothing lengths.
//!
//! `h_i` solves `N_i(h) = n_ngb` by bisection, where the kernel-weighted
//! neighbor count is the smooth (deterministically root-findable) analogue of
//! "particles within the support radius":
//!
//! ```text
//! N_i(h) = (4π/3) · (SUPPORT·h)³ · Σ_j W(|x_i − x_j|, h)
//! ```
//!
//! (For a locally uniform cloud `Σ_j W ≈` number density, so `N_i` ≈ the count
//! inside `2h`.) Bisection runs to a fixed relative tolerance on `h` from a
//! deterministic bracket, so `h` is a pure function of the positions — a
//! warm-start (`h_init`) only seeds the bracket and cannot change the converged
//! value beyond that tolerance (gated: cold ≡ warm within `h_tol_rel`).
//!
//! Under-populated systems (too few particles for the target count — the
//! asymptote of `N_i` as h → ∞ is `(32/3)·n`, so no root exists below
//! `n ≈ 4.5` for the default target) clamp deterministically to the bracket
//! ceiling: finite `h`, finite `ρ`, no panic (gated).

use galaxy_core::DVec3;

/// Adaptive-h configuration.
#[derive(Clone, Debug)]
pub struct DensityConfig {
    /// Target kernel-weighted neighbor count. Default 48; the cubic spline
    /// pairs above ~57 (pairing instability), so keep below that.
    pub n_ngb: f64,
    /// Bisection convergence: relative tolerance on `h`.
    pub h_tol_rel: f64,
}

impl Default for DensityConfig {
    fn default() -> Self {
        DensityConfig {
            n_ngb: 48.0,
            h_tol_rel: 1e-3,
        }
    }
}

/// Densities and the smoothing lengths they were computed with.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityResult {
    pub rho: Vec<f64>,
    pub h: Vec<f64>,
}

/// Grid-accelerated density with CALLER-SUPPLIED smoothing lengths (the fixed-h
/// special case the unit gates use). Gathers neighbors in ascending index, so
/// the sum associates exactly like [`super::reference::reference_density`] —
/// gated bit-exact against it.
pub fn density_fixed(pos: &[DVec3], mass: &[f64], h: &[f64]) -> Vec<f64> {
    let _ = (pos, mass, h);
    todo!("M7a: grid-accelerated fixed-h density")
}

/// Adaptive-h density, rayon over targets (each target's neighbor sum has a
/// fixed gather order, so the result is bit-identical to
/// [`density_adaptive_serial`] — gated). `h_init`, if given, seeds the
/// per-particle bisection bracket (warm start).
pub fn density_adaptive(
    pos: &[DVec3],
    mass: &[f64],
    cfg: &DensityConfig,
    h_init: Option<&[f64]>,
) -> DensityResult {
    let _ = (pos, mass, cfg, h_init);
    todo!("M7a: adaptive-h density (parallel)")
}

/// Serial twin of [`density_adaptive`]: the same per-target computation without
/// the rayon dispatch, for the parallel ≡ serial bit-exactness gate.
pub fn density_adaptive_serial(
    pos: &[DVec3],
    mass: &[f64],
    cfg: &DensityConfig,
    h_init: Option<&[f64]>,
) -> DensityResult {
    let _ = (pos, mass, cfg, h_init);
    todo!("M7a: adaptive-h density (serial)")
}
