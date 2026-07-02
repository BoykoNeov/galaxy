//! Validation of the numerical Eddington-inversion DF machinery. Eddington's
//! formula recovers the unique isotropic distribution function f(ℰ) of a
//! spherical model from its density and potential alone:
//!
//!   f(ℰ) = 1/(√8 π²) · d/dℰ ∫₀^ℰ (dρ/dΨ) / √(ℰ − Ψ) dΨ,   Ψ = −Φ.
//!
//! It has no external oracle of its own, so it is validated against the two
//! models whose isotropic f(ℰ) IS known in closed form:
//!   - Hernquist: the machinery must reproduce `Hernquist::df` (itself validated
//!     independently by the density-recovery integral and the stability run in
//!     `hernquist_*`), pinning both shape AND absolute normalization.
//!   - Plummer: f(ℰ) ∝ ℰ^(7/2) exactly, so the machinery's output divided by
//!     ℰ^(7/2) must be constant — an independent check of the SHAPE across the
//!     whole energy range.
//!
//! Once it reproduces both, it is trustworthy for NFW, which has no closed form.

use galaxy_ic::eddington::{EddingtonDf, SphericalModel};
use galaxy_ic::{Hernquist, Plummer};

/// A model's deepest binding energy Ψ(0) = −Φ(0), the top of the f(ℰ) support.
fn psi_max<M: SphericalModel>(m: &M) -> f64 {
    m.relative_potential(0.0)
}

#[test]
fn eddington_reproduces_hernquist_df() {
    let model = Hernquist::new(1.0, 1.0, 1.0);
    let pmax = psi_max(&model);
    // Wide radius bracket: r_min resolves the cusp, r_max drives Ψ → 0.
    let df = EddingtonDf::build(&model, pmax, 1e-4, 1e4);
    // Compare across the bulk of the support; skip the very ends where the
    // finite-difference dI/dℰ and the interpolation are least accurate.
    for frac in [0.1_f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] {
        let e = frac * pmax;
        let num = df.f(e);
        let exact = model.df(e);
        assert!(
            (num - exact).abs() < 3e-2 * exact,
            "frac={frac}: numerical f={num} vs analytic f={exact}"
        );
    }
}

#[test]
fn eddington_reproduces_plummer_e72_shape() {
    let model = Plummer::new(1.0, 1.0, 1.0);
    let pmax = psi_max(&model);
    let df = EddingtonDf::build(&model, pmax, 1e-4, 1e4);
    // Plummer f(ℰ) ∝ ℰ^(7/2): the ratio f/ℰ^(7/2) must be constant. Compare each
    // sample to the mid-range reference rather than to an absolute constant.
    let reference = df.f(0.5 * pmax) / (0.5 * pmax).powf(3.5);
    for frac in [0.2_f64, 0.3, 0.4, 0.6, 0.7, 0.8] {
        let e = frac * pmax;
        let ratio = df.f(e) / e.powf(3.5);
        assert!(
            (ratio - reference).abs() < 3e-2 * reference,
            "frac={frac}: f/ℰ^3.5={ratio} vs reference={reference}"
        );
    }
}

#[test]
fn eddington_df_is_positive_on_support() {
    let model = Hernquist::new(1.3, 2.0, 0.8);
    let pmax = psi_max(&model);
    let df = EddingtonDf::build(&model, pmax, 1e-4, 1e4);
    for frac in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
        let f = df.f(frac * pmax);
        assert!(f > 0.0 && f.is_finite(), "f at frac={frac} = {f}");
    }
    assert_eq!(df.f(-1.0), 0.0, "f below support should be 0");
}
