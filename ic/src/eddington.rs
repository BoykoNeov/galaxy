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
    /// Energy grid nodes ℰ_j (ascending, spanning (0, Ψ_max)).
    energies: Vec<f64>,
    /// f(ℰ_j) at the grid nodes.
    values: Vec<f64>,
    /// The deepest binding energy Ψ(0), the top of the support.
    psi_max: f64,
}

/// Eddington's prefactor 1/(√8 π²) = 1/(2√2 π²).
const EDDINGTON_C: f64 =
    1.0 / (2.0 * std::f64::consts::SQRT_2 * std::f64::consts::PI * std::f64::consts::PI);

impl EddingtonDf {
    /// Build the DF table for `model` whose support tops out at `psi_max` = Ψ(0).
    /// `r_min`/`r_max` bracket the radius grid used to tabulate dρ/dΨ: `r_min`
    /// must resolve the center, `r_max` must be large enough that Ψ(r_max) ≈ 0.
    pub fn build<M: SphericalModel>(model: &M, psi_max: f64, r_min: f64, r_max: f64) -> Self {
        assert!(psi_max > 0.0 && r_min > 0.0 && r_max > r_min);

        // --- 1. Tabulate dρ/dΨ on a log-spaced radius grid, ascending in Ψ. ---
        // Ψ decreases with r, so iterating r downward gives ascending Ψ. Each
        // dρ/dΨ = ρ'(r)/Ψ'(r) uses analytic-quality central differences in r.
        const NR: usize = 4000;
        let ratio = (r_max / r_min).powf(1.0 / NR as f64);
        let mut psi_asc = Vec::with_capacity(NR + 1);
        let mut drho_dpsi_asc = Vec::with_capacity(NR + 1);
        for i in (0..=NR).rev() {
            let r = r_min * ratio.powi(i as i32);
            let h = r * 1e-6;
            let drho = model.density(r + h) - model.density(r - h);
            let dpsi = model.relative_potential(r + h) - model.relative_potential(r - h);
            // Ψ strictly decreasing ⇒ dpsi < 0; guard against a zero slope.
            let drho_dpsi = if dpsi.abs() > 0.0 { drho / dpsi } else { 0.0 };
            psi_asc.push(model.relative_potential(r));
            drho_dpsi_asc.push(drho_dpsi);
        }

        // Linear interpolation of dρ/dΨ(Ψ); 0 below the table (r > r_max, ρ ≈ 0),
        // clamped to the innermost value above it.
        let drho_dpsi_at = |psi: f64| -> f64 {
            if psi <= psi_asc[0] {
                return 0.0;
            }
            let last = psi_asc.len() - 1;
            if psi >= psi_asc[last] {
                return drho_dpsi_asc[last];
            }
            let j = psi_asc.partition_point(|&p| p < psi);
            let (p0, p1) = (psi_asc[j - 1], psi_asc[j]);
            let (d0, d1) = (drho_dpsi_asc[j - 1], drho_dpsi_asc[j]);
            let t = (psi - p0) / (p1 - p0);
            d0 + t * (d1 - d0)
        };

        // I(ℰ) = 2 ∫₀^√ℰ (dρ/dΨ)(ℰ − u²) du — the singularity-free form after
        // Ψ = ℰ − u². Composite Simpson over u; the integrand is smooth and
        // vanishes as u → √ℰ (Ψ → 0). Even panel count for Simpson.
        let integral = |e: f64| -> f64 {
            const NU: usize = 512;
            let upper = e.sqrt();
            let du = upper / NU as f64;
            let mut sum = drho_dpsi_at(e); // u = 0 ⇒ Ψ = ℰ
            for k in 1..NU {
                let u = k as f64 * du;
                let w = if k % 2 == 1 { 4.0 } else { 2.0 };
                sum += w * drho_dpsi_at(e - u * u);
            }
            sum += drho_dpsi_at(e - upper * upper); // u = √ℰ ⇒ Ψ = 0
            2.0 * sum * du / 3.0
        };

        // --- 2. f(ℰ) = C · dI/dℰ on an ascending ℰ-grid, by central difference. ---
        const NE: usize = 800;
        let e_lo = 1e-3 * psi_max;
        let e_hi = 0.999 * psi_max;
        let mut energies = Vec::with_capacity(NE + 1);
        let mut values = Vec::with_capacity(NE + 1);
        for j in 0..=NE {
            let e = e_lo + (e_hi - e_lo) * j as f64 / NE as f64;
            let delta = (1e-4 * e).min(0.5 * (e - e_lo).max(e_lo)).max(1e-9);
            let did_de = (integral(e + delta) - integral(e - delta)) / (2.0 * delta);
            energies.push(e);
            values.push((EDDINGTON_C * did_de).max(0.0));
        }

        Self {
            energies,
            values,
            psi_max,
        }
    }

    /// The isotropic DF f(ℰ). Returns 0 for ℰ ≤ 0 and clamps to the tabulated
    /// support; interior values are interpolated from the table.
    pub fn f(&self, energy: f64) -> f64 {
        if energy <= 0.0 {
            return 0.0;
        }
        let last = self.energies.len() - 1;
        if energy <= self.energies[0] {
            return self.values[0];
        }
        if energy >= self.energies[last] {
            return self.values[last];
        }
        let j = self.energies.partition_point(|&e| e < energy);
        let (e0, e1) = (self.energies[j - 1], self.energies[j]);
        let (f0, f1) = (self.values[j - 1], self.values[j]);
        let t = (energy - e0) / (e1 - e0);
        f0 + t * (f1 - f0)
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
