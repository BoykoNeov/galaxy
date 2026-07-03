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

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};

use std::f64::consts::{PI, TAU};

use crate::disk::{mix_seed, ExponentialDisk, SplitMix64};
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
        // Reject a gravitationally unstable gas layer loudly: a min Q_gas < 1 over the
        // disk body means the isothermal gas fragments, and the demo must never launch
        // such an IC. (The message names "Q_gas" for the fail-loud gate.)
        let min_q = self.min_gas_toomre_q();
        assert!(
            min_q >= 1.0,
            "gas disk is gravitationally unstable: min Q_gas = {min_q} < 1 \
             (raise the sound speed or lower the gas fraction)"
        );
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
        self.gas_fraction()
            .expect("gas-free disk has no gas surface density")
            * self.central_surface_density()
    }

    /// Gas surface density Σ_gas(R) = f_gas · Σ(R) for R ≤ r_max, else 0.
    pub fn gas_surface_density(&self, r: f64) -> f64 {
        self.gas_fraction()
            .expect("gas-free disk has no gas surface density")
            * self.surface_density(r)
    }

    /// Gas mass enclosed within cylindrical radius R: f_gas · M_disk(<R).
    pub fn gas_enclosed_mass(&self, r: f64) -> f64 {
        self.gas_fraction().expect("gas-free disk has no gas mass") * self.disk_enclosed_mass(r)
    }

    /// Isothermal self-gravity scale height z₀(R) = c_s² / (π G Σ_gas(R)) — the
    /// Spitzer sech² sheet thickness. Flares outward (Σ_gas falls with R). Panics
    /// on a gas-free disk (no c_s).
    pub fn gas_scale_height(&self, r: f64) -> f64 {
        let cs = self
            .sound_speed()
            .expect("gas-free disk has no scale height");
        cs * cs / (PI * self.g * self.gas_surface_density(r))
    }

    /// Midplane gas density ρ_mid(R) = Σ_gas(R) / (2 z₀(R)) = π G Σ_gas(R)² /
    /// (2 c_s²). Panics on a gas-free disk.
    pub fn gas_midplane_density(&self, r: f64) -> f64 {
        self.gas_surface_density(r) / (2.0 * self.gas_scale_height(r))
    }

    /// Pressure-corrected mean azimuthal velocity of the gas,
    /// v_φ,gas(R) = √max(0, v_c(R)² − 2 c_s² R/Rd). Clamped ≥ 0 near the center.
    /// Panics on a gas-free disk.
    pub fn gas_azimuthal_velocity(&self, r: f64) -> f64 {
        let cs = self
            .sound_speed()
            .expect("gas-free disk has no gas rotation");
        let vc = self.circular_velocity(r);
        (vc * vc - 2.0 * cs * cs * r / self.scale_length)
            .max(0.0)
            .sqrt()
    }

    /// Gas Toomre parameter Q_gas(R) = c_s κ(R) / (π G Σ_gas(R)) (gas prefactor π).
    /// Panics on a gas-free disk.
    pub fn gas_toomre_q(&self, r: f64) -> f64 {
        let cs = self.sound_speed().expect("gas-free disk has no Toomre Q");
        cs * self.epicyclic_frequency(r) / (PI * self.g * self.gas_surface_density(r))
    }

    /// Minimum Q_gas over the disk body (R in (0, r_max]) — the value the stability
    /// rejection keys on. Panics on a gas-free disk.
    ///
    /// Grid scan: Q_gas → ∞ at the center (the epicyclic κ carries a 2πΣ/R term that
    /// diverges) and rises again toward `r_max` (Σ_gas → small), so the minimum sits
    /// in the disk body and a fine uniform scan over (0, r_max] captures it.
    pub fn min_gas_toomre_q(&self) -> f64 {
        self.sound_speed().expect("gas-free disk has no Toomre Q");
        const N: usize = 2000;
        let mut min = f64::INFINITY;
        for k in 1..=N {
            let r = self.r_max * k as f64 / N as f64;
            min = min.min(self.gas_toomre_q(r));
        }
        min
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
    pub fn sample_gas(&self, n_halo: usize, n_disk: usize, n_gas: usize, seed: u64) -> State {
        let f = self
            .gas_fraction()
            .expect("sample_gas called on a gas-free disk");

        // 1. Halo + stellar disk, BIT-IDENTICAL to the gas-free IC. `sample` ignores
        //    `self.gas`, so calling it on the gas-rich disk reproduces the gas-free
        //    realization exactly (full disk mass, recentered with full mass). The
        //    halo and stellar positions are now locked to the gas-free stream layout.
        let mut s = self.sample(n_halo, n_disk, seed);

        // 2. Re-tag the stellar mass down by (1 − f_gas): the gas carries that fraction
        //    of the total disk mass. Scaling mass AFTER the recenter cannot move any
        //    position, so the bit-identity of step 1 survives.
        for i in n_halo..n_halo + n_disk {
            s.mass[i] *= 1.0 - f;
        }

        if n_gas == 0 {
            return s;
        }

        // 3. Gas: its own PRNG stream, mix³(seed) — past the halo (seed), stellar-
        //    position (mix), and stellar-velocity (mix²) streams. Fluid-cold: v_r =
        //    v_z = 0, a single deterministic pressure-corrected v_φ,gas(R). The thermal
        //    sech²(z/z₀(R)) layer inverts its CDF as z = z₀·atanh(2Y−1).
        let gas_seed = mix_seed(mix_seed(mix_seed(seed)));
        let mut rng = SplitMix64::new(gas_seed);
        let m_gas_each = f * self.disk_mass / n_gas as f64;
        let base = s.len();
        for k in 0..n_gas {
            let r = self.sample_radius(rng.next_f64());
            let phi = TAU * rng.next_f64();
            let (sin_phi, cos_phi) = phi.sin_cos();
            let z0 = self.gas_scale_height(r);
            let t = (2.0 * rng.next_f64() - 1.0).clamp(-1.0 + 1e-12, 1.0 - 1e-12);
            let z = z0 * t.atanh();
            let vphi = self.gas_azimuthal_velocity(r);

            s.pos.push(DVec3::new(r * cos_phi, r * sin_phi, z));
            s.vel.push(DVec3::new(-vphi * sin_phi, vphi * cos_phi, 0.0));
            s.mass.push(m_gas_each);
            s.id.push(ParticleId((base + k) as u64));
            s.progenitor.push(Progenitor(4));
            s.kind.push(Species::Gas);
        }

        // 4. Zero the WHOLE-system COM and momentum by moving the GAS block only — the
        //    halo+star positions are locked to the gas-free IC (step 1), so the gas is
        //    the only block free to absorb the residual. `sample`'s combined recenter
        //    leaves each sub-block a finite-N COM of ±M_halo·mean_pos (not roundoff),
        //    and scaling the stars in step 2 unbalances it by −f·M_halo·mean_pos.
        //    Subtracting mean·(mtot/M_gas) from each gas particle drives the total to
        //    exactly zero: Σmᵢxᵢ − M_gas·(mtot/M_gas)·mean = mtot·mean − mtot·mean = 0.
        let m_gas_total = f * self.disk_mass;
        let mtot: f64 = s.mass.iter().sum();
        let kfac = mtot / m_gas_total;
        let mean_pos = (0..s.len()).fold(DVec3::ZERO, |a, i| a + s.pos[i] * s.mass[i]) / mtot;
        let mean_vel = (0..s.len()).fold(DVec3::ZERO, |a, i| a + s.vel[i] * s.mass[i]) / mtot;
        for i in base..base + n_gas {
            s.pos[i] -= mean_pos * kfac;
            s.vel[i] -= mean_vel * kfac;
        }

        s
    }
}
