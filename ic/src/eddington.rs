//! Numerical Eddington inversion: recover the isotropic distribution function
//! f(ℰ) of a spherically-symmetric model from its density and potential.
//!
//! For an isotropic system f depends only on the relative energy ℰ = Ψ − v²/2
//! (Ψ = −Φ ≥ 0), and Eddington's formula (Binney & Tremaine §4.3, eq. 4.46)
//! inverts the density–DF relation:
//!
//!   f(ℰ) = 1/(√8 π²) · d/dℰ ∫₀^ℰ (dρ/dΨ) / √(ℰ − Ψ) dΨ.
//!
//! Models with a closed-form f (Plummer, Hernquist) validate this machinery;
//! once it reproduces them it is used for models that have none (NFW). See the
//! `eddington` integration test for the validation ladder.
//!
//! Numerics:
//! - `dρ/dΨ(Ψ)` is tabulated on a log-spaced radius grid (its r-derivatives are
//!   analytic-quality central differences), then interpolated in Ψ. No per-point
//!   root inversion is needed.
//! - The `1/√(ℰ − Ψ)` endpoint singularity is removed by the substitution
//!   Ψ = ℰ − u², giving the smooth I(ℰ) = 2∫₀^√ℰ (dρ/dΨ)(ℰ − u²) du, evaluated
//!   by Gauss–Legendre quadrature.
//! - f(ℰ) = (1/√8 π²) dI/dℰ is taken by central difference on a smooth ℰ-grid
//!   and the resulting f(ℰ) table is interpolated for callers/samplers.

/// A spherically-symmetric model exposing the two profiles Eddington inversion
/// needs: the mass density and the **relative** potential Ψ(r) = −Φ(r) ≥ 0,
/// monotonically decreasing in r.
pub trait SphericalModel {
    /// Mass density ρ(r) ≥ 0.
    fn density(&self, r: f64) -> f64;
    /// Relative potential Ψ(r) = −Φ(r) ≥ 0.
    fn relative_potential(&self, r: f64) -> f64;
}

/// A tabulated isotropic distribution function f(ℰ) built by Eddington inversion.
pub struct EddingtonDf {
    /// Energy grid nodes ℰ_j (ascending, spanning (0, Ψ_max]).
    energies: Vec<f64>,
    /// f(ℰ_j) at the grid nodes.
    values: Vec<f64>,
    /// The deepest binding energy Ψ(0), the top of the support.
    psi_max: f64,
}

impl EddingtonDf {
    /// Build the DF table for `model` whose support tops out at `psi_max` = Ψ(0).
    /// `r_min`/`r_max` bracket the radius grid used to tabulate dρ/dΨ: `r_min`
    /// must resolve the center, `r_max` must be large enough that Ψ(r_max) ≈ 0.
    pub fn build<M: SphericalModel>(
        _model: &M,
        _psi_max: f64,
        _r_min: f64,
        _r_max: f64,
    ) -> Self {
        todo!()
    }

    /// The isotropic DF f(ℰ). Returns 0 for ℰ ≤ 0 and clamps to the tabulated
    /// support; interior values are interpolated from the table.
    pub fn f(&self, _energy: f64) -> f64 {
        todo!()
    }

    /// The deepest binding energy Ψ(0) = top of the f(ℰ) support.
    pub fn psi_max(&self) -> f64 {
        self.psi_max
    }
}

// Adapter impls so the analytic models can drive (and validate) the machinery.

impl SphericalModel for crate::Plummer {
    fn density(&self, r: f64) -> f64 {
        crate::Plummer::density(self, r)
    }
    fn relative_potential(&self, r: f64) -> f64 {
        -crate::Plummer::potential(self, r)
    }
}

impl SphericalModel for crate::Hernquist {
    fn density(&self, r: f64) -> f64 {
        crate::Hernquist::density(self, r)
    }
    fn relative_potential(&self, r: f64) -> f64 {
        -crate::Hernquist::potential(self, r)
    }
}
