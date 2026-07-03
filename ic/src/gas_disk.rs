//! The gas option on [`ExponentialDisk`] (M7c, D7): a self-gravitating isothermal
//! SPH gas layer co-spatial with the stellar disk.
//!
//! **Model.** `gas_fraction` = f_gas = M_gas / M_disk *splits* the disk's total
//! mass: a fraction `f_gas` of the same truncated-exponential profile is tagged
//! `Species::Gas`, the rest stays `Progenitor(1)` stars. Because gas and stars
//! share the radial profile (`scale_length`, `r_max`), the *total* disk mass — and
//! therefore the circular-velocity curve every population orbits — is **unchanged**
//! from the gas-free disk of the same `disk_mass`. Gas differs from stars in two
//! physical ways, and only these two:
//!
//!  1. **Vertical structure.** Instead of the stars' geometric sech²(z/hz) layer,
//!     the gas sits in the *thermal* equilibrium of a self-gravitating isothermal
//!     sheet: ρ(z) ∝ sech²(z/z₀) with **z₀(R) = c_s² / (π G Σ_gas(R))** (Spitzer).
//!     z₀ *flares* outward (Σ_gas falls), so the gas disk is thicker than the stars
//!     and thickens with radius. Caveat, mirroring `disk.rs`'s σ_z note: z₀ is the
//!     gas's *own* self-gravity value; the halo adds vertical restoring force, so
//!     the sampled layer mildly over-estimates the settled thickness (it compresses
//!     a little under the halo). The stability gate's thickness tolerance is set
//!     from this approximation, not tighter.
//!
//!  2. **Pressure-corrected rotation.** Radial momentum balance for an isothermal
//!     fluid, v_φ,gas² = v_c² + (R/ρ) dP/dR = v_c² + c_s² R d ln ρ_mid/dR. With the
//!     self-gravity z₀, ρ_mid = Σ_gas/(2 z₀) ∝ Σ_gas², so d ln ρ_mid/dR =
//!     2 d ln Σ_gas/dR = −2R/Rd in the disk body (the exponential slope is exact;
//!     the truncation is a sampling cutoff, not a local density feature — same
//!     convention as `disk.rs`'s asymmetric-drift term). Hence
//!     **v_φ,gas² = v_c² − 2 c_s² R/Rd**, clamped ≥ 0 (`.max(0.0).sqrt()`): the
//!     correction is negative (pressure helps support the gas, so it rotates slower
//!     than v_c) and v_c² ~ R² vanishes faster than the linear pressure term near
//!     the center, so the raw v_φ,gas² dips negative there — the clamp keeps the IC
//!     NaN-free, exactly as [`ExponentialDisk::mean_azimuthal_velocity`] does.
//!
//! The gas is **fluid-cold**: v_r = v_z = 0 and a single deterministic v_φ,gas(R).
//! Unlike the *warm stellar* disk, there is NO particle velocity dispersion — the
//! gas's pressure support is supplied by the SPH sound speed during evolution, not
//! by a Box–Muller velocity spread at sampling time.
//!
//! **Toomre stability.** The gas criterion Q_gas = c_s κ / (π G Σ_gas) uses the
//! gas prefactor π (the stellar disk uses 3.36). Q_gas varies with R; a
//! gravitationally unstable gas disk (min over the body < 1) is rejected loudly so
//! the demo never launches a fragmenting IC.

use galaxy_core::State;

use crate::disk::ExponentialDisk;
use crate::SphericalHalo;

/// The isothermal gas component of an [`ExponentialDisk`]: the fraction of the
/// disk's total mass carried as SPH gas, and its sound speed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasParams {
    /// f_gas = M_gas / M_disk, the fraction of the total disk mass tagged as gas.
    /// In (0, 1); the remaining (1 − f_gas) is the stellar disk.
    pub fraction: f64,
    /// Isothermal sound speed c_s (EOS P = c_s² ρ). Must match the `HydroParams`
    /// sound speed the SPH solver evolves it with, or the pressure correction and
    /// the dynamics disagree.
    pub sound_speed: f64,
}

impl<H: SphericalHalo> ExponentialDisk<H> {
    /// Return a copy of this disk with an isothermal gas component: a fraction
    /// `gas_fraction` (= f_gas = M_gas/M_disk, in (0,1)) of the total disk mass is
    /// re-tagged as SPH gas with sound speed `sound_speed`. The stellar disk keeps
    /// the remaining mass; the total disk mass and its rotation curve are unchanged.
    ///
    /// Panics if `gas_fraction ∉ (0,1)` or `sound_speed ≤ 0`. A gravitationally
    /// unstable gas layer (min Q_gas over the disk body < 1) is also rejected —
    /// see [`min_gas_toomre_q`](Self::min_gas_toomre_q).
    pub fn with_gas(mut self, gas_fraction: f64, sound_speed: f64) -> Self {
        assert!(
            gas_fraction > 0.0 && gas_fraction < 1.0,
            "gas fraction f_gas must be in (0,1), got {gas_fraction}"
        );
        assert!(sound_speed > 0.0, "gas sound speed must be positive");
        self.gas = Some(GasParams {
            fraction: gas_fraction,
            sound_speed,
        });
        // (Q_gas ≥ 1 rejection lands with the physics in a later commit.)
        self
    }

    /// The gas component, or `None` for a purely stellar disk.
    pub fn gas_params(&self) -> Option<GasParams> {
        self.gas
    }

    /// The gas mass fraction f_gas = M_gas/M_disk, or `None` if gas-free.
    pub fn gas_fraction(&self) -> Option<f64> {
        self.gas.map(|g| g.fraction)
    }

    /// The isothermal sound speed c_s, or `None` if gas-free.
    pub fn sound_speed(&self) -> Option<f64> {
        self.gas.map(|g| g.sound_speed)
    }

    /// Total gas mass M_gas = f_gas · M_disk (0 for a gas-free disk).
    pub fn gas_mass(&self) -> f64 {
        self.gas.map_or(0.0, |g| g.fraction * self.disk_mass)
    }

    /// Central gas surface density Σ_gas,0 = f_gas · Σ₀ (the stellar-disk central
    /// value scaled by the gas fraction, since gas traces the same profile).
    pub fn gas_central_surface_density(&self) -> f64 {
        todo!("Σ_gas,0 = f_gas · central_surface_density()")
    }

    /// Gas surface density Σ_gas(R) = f_gas · Σ(R) for R ≤ r_max, else 0.
    pub fn gas_surface_density(&self, _r: f64) -> f64 {
        todo!("f_gas · surface_density(r)")
    }

    /// Gas mass enclosed within cylindrical radius R: f_gas · M_disk(<R).
    pub fn gas_enclosed_mass(&self, _r: f64) -> f64 {
        todo!("f_gas · disk_enclosed_mass(r)")
    }

    /// Isothermal self-gravity scale height z₀(R) = c_s² / (π G Σ_gas(R)) — the
    /// Spitzer sech² sheet thickness. Flares outward (Σ_gas falls with R). Panics
    /// on a gas-free disk (no c_s).
    pub fn gas_scale_height(&self, _r: f64) -> f64 {
        todo!("z0(R) = c_s^2 / (pi G Sigma_gas(R))")
    }

    /// Midplane gas density ρ_mid(R) = Σ_gas(R) / (2 z₀(R)) = π G Σ_gas(R)² /
    /// (2 c_s²). Panics on a gas-free disk.
    pub fn gas_midplane_density(&self, _r: f64) -> f64 {
        todo!("rho_mid = Sigma_gas / (2 z0)")
    }

    /// Pressure-corrected mean azimuthal velocity of the gas,
    /// v_φ,gas(R) = √max(0, v_c(R)² − 2 c_s² R/Rd). Clamped ≥ 0 near the center.
    /// Panics on a gas-free disk.
    pub fn gas_azimuthal_velocity(&self, _r: f64) -> f64 {
        todo!("sqrt(max(0, v_c^2 - 2 c_s^2 R / Rd))")
    }

    /// Gas Toomre parameter Q_gas(R) = c_s κ(R) / (π G Σ_gas(R)) (gas prefactor π).
    /// Panics on a gas-free disk.
    pub fn gas_toomre_q(&self, _r: f64) -> f64 {
        todo!("Q_gas = c_s kappa / (pi G Sigma_gas)")
    }

    /// Minimum Q_gas over the disk body (R in (0, r_max]) — the value the stability
    /// rejection keys on. Panics on a gas-free disk.
    pub fn min_gas_toomre_q(&self) -> f64 {
        todo!("min over the body of gas_toomre_q(r)")
    }

    /// Sample a gas-rich disk: `n_halo` halo particles (`Progenitor(0)`), `n_disk`
    /// stellar particles (`Progenitor(1)`), and `n_gas` SPH gas particles
    /// (`Progenitor(4)`, `Species::Gas`) drawn from the gas profile with the thermal
    /// sech² layer and pressure-corrected rotation. Deterministic in `seed`, zero
    /// net momentum / COM.
    ///
    /// The halo and stellar draws consume the SAME PRNG streams as
    /// [`ExponentialDisk::sample`] (halo `seed`, stellar positions `mix(seed)`,
    /// stellar velocity dispersion `mix²(seed)`), so the halo and the **stellar
    /// positions** are bit-identical to the gas-free IC at the same seed. Gas is
    /// drawn from a further, well-separated stream reserved whether or not
    /// `n_gas == 0`, keeping the stellar streams fixed. Panics on a gas-free disk.
    pub fn sample_gas(&self, _n_halo: usize, _n_disk: usize, _n_gas: usize, _seed: u64) -> State {
        todo!("halo + stars (shared streams) + gas (own stream), zero-COM")
    }
}
