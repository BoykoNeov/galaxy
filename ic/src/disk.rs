//! A rotating exponential-disk galaxy: a cold, low-mass stellar disk embedded in
//! a live Plummer halo/bulge that carries most of the mass. This is the IC that
//! unlocks the classic Toomre & Toomre tidal-tail visual — thin curved streams
//! come from *coherently rotating, dynamically cold disks*, which the isotropic
//! Plummer sphere cannot produce (see `DESIGN.md`, M3 accuracy note).
//!
//! Design (the "cold kinematic" model):
//! - The disk's surface density is exponential, Σ(R) = Σ₀ e^(−R/Rd), truncated at
//!   `r_max`; the vertical structure is a thin isothermal sheet ∝ sech²(z/hz).
//! - The disk is **submaximal**: its mass is a small fraction of the halo's, so
//!   the rotation is dominated by the smooth (spherical, analytic) halo potential
//!   and the cold self-gravitating disk stays close to its initial profile over an
//!   encounter rather than fragmenting. This is the stabilization mechanism, not a
//!   shortcut — a maximal cold disk has Toomre Q ≪ 1 and is *not* an equilibrium.
//! - Particles are placed on **near-circular orbits**: v_φ(R) = v_c(R) from the
//!   *combined* enclosed mass (spherical halo M(<r) + cylindrical disk M(<R)), with
//!   the disk spin along **+Z** so a face-on (+Z) camera sees it face-on and a
//!   coplanar collision (orbit in the x–y plane) is prograde. No random velocity
//!   dispersion is added (fully cold) — the money shot is the coherent stream.
//!
//! Why this is exactly checkable: the rotation curve comes from *enclosed mass*,
//! not the exponential disk's Bessel-function potential — so both the halo term
//! (analytic Plummer) and the disk term (closed-form truncated exponential) are
//! elementary, and the sampled kinematics can be compared to an independent
//! closed form (mirroring the Plummer virial check).
//!
//! Caveats (deliberate, for the follow-up "warm disk" milestone):
//! - The sech² vertical layer is a *geometric* profile, not a vertical
//!   equilibrium: particles are given no vertical velocity support (v_z = 0), so
//!   the sheet settles toward the midplane and phase-mixes thinner. Fine for the
//!   in-plane tidal-tail visual; it is not a self-consistent isothermal disk.
//! - The disk is fully cold (no random velocity dispersion, Toomre Q → 0). Over a
//!   single isolated orbit it holds, but across the several orbits of a collision
//!   a maximally-cold disk can fragment/clump before the tail develops. A small
//!   in-plane dispersion is the natural knob to add when the collision needs it.
//!
//! Disk particles are tagged `Progenitor(1)` (species: disk) and halo particles
//! `Progenitor(0)` (species: halo/bulge), so the renderer can color them apart and
//! tests can select the disk population robustly. `sample` returns the halo
//! particles first, then the disk particles.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};

use std::f64::consts::{PI, TAU};

use crate::{Plummer, SphericalHalo};

/// Toomre's stellar stability factor: Q = σ_R κ / (3.36 G Σ). The 3.36 is the
/// value for a collisionless (stellar) disk; π is the gas value.
const TOOMRE_FACTOR: f64 = 3.36;

/// A cold exponential disk of low-mass particles inside a live spherical halo/bulge.
///
/// The halo is any [`SphericalHalo`] — a cored [`Plummer`] (the default, and the
/// original M3 model) or a cuspy [`Nfw`](crate::Nfw)/[`Hernquist`](crate::Hernquist)/
/// [`TruncatedNfw`](crate::TruncatedNfw). Swapping it changes the disk's rotation
/// curve (a cuspy halo gives the realistic rising-to-flat curve) with no other code
/// change — the disk reads only the halo's closed-form `g`, total mass, ρ(r) and
/// M(<r), plus its particle sampler.
///
/// Choose units freely; the disk's `g` is taken from `halo.g()`. The disk mass
/// should be a small fraction of the halo mass (submaximal) for the cold disk to
/// hold together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExponentialDisk<H = Plummer> {
    /// Gravitational constant `G` (equals `halo.g()`).
    pub g: f64,
    /// Total disk mass `M_d` (should be ≪ the halo's total mass).
    pub disk_mass: f64,
    /// Radial exponential scale length `Rd`.
    pub scale_length: f64,
    /// Vertical sech² scale height `hz` (thin disk: hz ≪ Rd).
    pub scale_height: f64,
    /// Truncation radius (no disk particles beyond this cylindrical radius).
    pub r_max: f64,
    /// The live spherical halo/bulge the disk is embedded in.
    pub halo: H,
    /// Optional Toomre-Q warmth. `None` = the fully-cold kinematic disk (v_φ = v_c,
    /// zero dispersion). `Some(q)` gives the disk in-plane and vertical velocity
    /// dispersion targeting Toomre stability parameter `q`, with the mean azimuthal
    /// streaming reduced by asymmetric drift. Set via [`with_toomre_q`].
    toomre_q: Option<f64>,
}

impl<H: SphericalHalo> ExponentialDisk<H> {
    /// Construct a disk galaxy. All scalar parameters must be strictly positive,
    /// the disk's `g` is inherited from `halo.g()`, and `r_max` must exceed the
    /// scale length.
    pub fn new(disk_mass: f64, scale_length: f64, scale_height: f64, r_max: f64, halo: H) -> Self {
        assert!(disk_mass > 0.0, "disk mass must be positive");
        assert!(scale_length > 0.0, "scale length must be positive");
        assert!(scale_height > 0.0, "scale height must be positive");
        assert!(
            r_max > scale_length,
            "truncation radius ({r_max}) must exceed the scale length ({scale_length})"
        );
        Self {
            g: halo.g(),
            disk_mass,
            scale_length,
            scale_height,
            r_max,
            halo,
            toomre_q: None,
        }
    }

    /// Return a WARM copy of this disk targeting Toomre stability parameter `q`
    /// (typically 1.2–2.0). The warm disk carries radial, azimuthal, and vertical
    /// velocity dispersion set from `q` and the local epicyclic frequency, with the
    /// mean azimuthal streaming reduced by asymmetric drift so it stays near
    /// equilibrium. `q` must be strictly positive. Positions are unchanged — warmth
    /// perturbs only velocities.
    pub fn with_toomre_q(mut self, q: f64) -> Self {
        assert!(q > 0.0, "Toomre Q must be positive");
        self.toomre_q = Some(q);
        self
    }

    /// The target Toomre `Q`, or `None` for the fully-cold disk.
    pub fn toomre_q(&self) -> Option<f64> {
        self.toomre_q
    }

    /// Orbital (angular) frequency Ω(R) = v_c(R)/R. Zero at R = 0.
    pub fn orbital_frequency(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        self.circular_velocity(r) / r
    }

    /// Epicyclic frequency κ(R): the radial oscillation frequency of a near-circular
    /// orbit. Closed form κ² = Ω² + G M'(R)/R², where M'(R) = dM_enc/dR is the
    /// derivative of the same combined enclosed mass that sets v_c — the spherical
    /// halo's shell mass 4πR²ρ_halo(R) plus the disk's annulus mass 2πR Σ(R). Both
    /// densities are exact closed forms, so κ is exact (no numerical derivative).
    /// The identity follows from κ² = R dΩ²/dR + 4Ω² with Ω² = G M_enc/R³.
    pub fn epicyclic_frequency(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        let omega = self.orbital_frequency(r);
        let m_prime = 4.0 * PI * r * r * self.halo.density(r) + TAU * r * self.surface_density(r);
        (omega * omega + self.g * m_prime / (r * r)).sqrt()
    }

    /// Radial velocity dispersion σ_R(R) from the Toomre criterion:
    /// σ_R = Q · 3.36 · G Σ(R) / κ(R). Zero for the cold disk (`toomre_q == None`).
    pub fn radial_dispersion(&self, r: f64) -> f64 {
        match self.toomre_q {
            None => 0.0,
            Some(q) => {
                let kappa = self.epicyclic_frequency(r);
                if kappa <= 0.0 {
                    return 0.0;
                }
                q * TOOMRE_FACTOR * self.g * self.surface_density(r) / kappa
            }
        }
    }

    /// Azimuthal velocity dispersion σ_φ(R) = σ_R · κ/(2Ω), from the epicyclic
    /// relation σ_φ²/σ_R² = κ²/(4Ω²). Zero for the cold disk.
    pub fn azimuthal_dispersion(&self, r: f64) -> f64 {
        if self.toomre_q.is_none() {
            return 0.0;
        }
        let omega = self.orbital_frequency(r);
        if omega <= 0.0 {
            return 0.0;
        }
        self.radial_dispersion(r) * self.epicyclic_frequency(r) / (2.0 * omega)
    }

    /// Vertical velocity dispersion σ_z(R) = √(π G Σ(R) hz) — the isothermal value
    /// for a self-gravitating sech²(z/hz) sheet. Zero for the cold disk.
    ///
    /// Caveat (for the combined-potential refinement): this is the disk's *own*
    /// self-gravity value. Because the disk is submaximal and halo-dominated, the
    /// halo adds vertical restoring force, so this mildly *under*-supports the layer
    /// (it settles a little). Still strictly better than the cold v_z = 0.
    pub fn vertical_dispersion(&self, r: f64) -> f64 {
        if self.toomre_q.is_none() {
            return 0.0;
        }
        (PI * self.g * self.surface_density(r) * self.scale_height).sqrt()
    }

    /// Mean azimuthal streaming speed v̄_φ(R): the circular speed reduced by
    /// asymmetric drift (a warm disk needs less rotation because pressure helps
    /// support it). Binney & Tremaine (eq. 4.228), midplane, aligned ellipsoid:
    ///
    ///   v_c² − v̄_φ² = σ_R²·[σ_φ²/σ_R² − 1 − d ln(ν σ_R²)/d ln R].
    ///
    /// v̄_φ² is clamped to ≥ 0: near R → 0, v_c → 0 while the bracket stays finite,
    /// so the raw v̄_φ² can dip negative — the clamp keeps the IC free of NaNs.
    /// Equals v_c for the cold disk.
    pub fn mean_azimuthal_velocity(&self, r: f64) -> f64 {
        let vc = self.circular_velocity(r);
        if self.toomre_q.is_none() {
            return vc;
        }
        let sigma_r = self.radial_dispersion(r);
        let omega = self.orbital_frequency(r);
        if sigma_r <= 0.0 || omega <= 0.0 {
            return vc;
        }
        let kappa = self.epicyclic_frequency(r);
        let ratio2 = kappa * kappa / (4.0 * omega * omega); // σ_φ²/σ_R²
        let bracket = ratio2 - 1.0 - self.dlog_nu_sigma_r2(r);
        (vc * vc - sigma_r * sigma_r * bracket).max(0.0).sqrt()
    }

    /// d ln(ν σ_R²)/d ln R for the asymmetric-drift bracket. With constant scale
    /// height ν ∝ Σ(R), and σ_R² ∝ (Σ/κ)², so ν σ_R² ∝ Σ³/κ² and the log-derivative
    /// splits as 3·d lnΣ/d lnR − 2·d lnκ/d lnR. The exponential's slope is exact,
    /// d lnΣ/d lnR = −R/Rd (the truncation is a sampling cutoff, not a local density
    /// feature); only d lnκ/d lnR uses a central difference of the closed-form κ(R),
    /// confining the numerical derivative to this small drift correction.
    ///
    /// Near `r_max` the `+h` finite-difference point can cross the truncation, where
    /// `surface_density` drops the disk term from κ — a small kink in the drift there.
    /// Disk particles are sparse at `r_max` and v̄_φ is clamped ≥ 0, so the resulting
    /// drift near the edge is approximate; it does not affect the disk body.
    fn dlog_nu_sigma_r2(&self, r: f64) -> f64 {
        let dlog_sigma = -r / self.scale_length;
        let h = 1e-5_f64;
        let ln_kappa = |rr: f64| self.epicyclic_frequency(rr).ln();
        let dlog_kappa = (ln_kappa(r * h.exp()) - ln_kappa(r * (-h).exp())) / (2.0 * h);
        3.0 * dlog_sigma - 2.0 * dlog_kappa
    }

    /// The fraction of an *untruncated* exponential disk's mass that lies within
    /// `r_max`: 1 − (1 + u)e^(−u), u = r_max/Rd. The normalization denominator.
    fn truncation_fraction(&self) -> f64 {
        let u = self.r_max / self.scale_length;
        1.0 - (1.0 + u) * (-u).exp()
    }

    /// Central surface density Σ₀, fixed by normalizing the truncated exponential
    /// to the total disk mass: M_d = 2π Σ₀ Rd² · [1 − (1 + u)e^(−u)], u = r_max/Rd.
    pub fn central_surface_density(&self) -> f64 {
        let rd = self.scale_length;
        self.disk_mass / (TAU * rd * rd * self.truncation_fraction())
    }

    /// Surface density Σ(R) = Σ₀ e^(−R/Rd) for R ≤ r_max, else 0.
    pub fn surface_density(&self, r: f64) -> f64 {
        if r > self.r_max {
            0.0
        } else {
            self.central_surface_density() * (-r / self.scale_length).exp()
        }
    }

    /// Disk mass enclosed within cylindrical radius R:
    /// M_d(<R) = M_d · [1 − (1 + R/Rd)e^(−R/Rd)] / [1 − (1 + u)e^(−u)], u = r_max/Rd.
    /// Constant at M_d for R ≥ r_max (the disk is truncated).
    pub fn disk_enclosed_mass(&self, r: f64) -> f64 {
        let x = r.min(self.r_max) / self.scale_length;
        let interior = 1.0 - (1.0 + x) * (-x).exp();
        self.disk_mass * interior / self.truncation_fraction()
    }

    /// Circular speed v_c(R) = √(G · [M_halo(<R) + M_disk(<R)] / R) from the
    /// combined enclosed mass — the spherical Plummer term plus the cylindrical
    /// disk term. This is the target mean azimuthal speed of the cold disk.
    ///
    /// v_c(0) = 0 by symmetry (no enclosed mass at the center): the explicit guard
    /// keeps a particle sampled exactly at R=0 from producing a √(0/0)=NaN velocity.
    pub fn circular_velocity(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        let m_enc = self.halo.enclosed_mass(r) + self.disk_enclosed_mass(r);
        (self.g * m_enc / r).sqrt()
    }

    /// Total galaxy mass, disk + halo.
    pub fn total_mass(&self) -> f64 {
        self.disk_mass + self.halo.total_mass()
    }

    /// Sample the galaxy: `n_halo` Plummer halo particles (`Progenitor(0)`) drawn
    /// from the halo distribution function, followed by `n_disk` disk particles
    /// (`Progenitor(1)`) on near-circular orbits (cold, or warm with dispersion if a
    /// Toomre `Q` was set). Deterministic in `seed`, contiguous unique ids, delivered
    /// in the zero-COM / zero-momentum frame.
    ///
    /// Consumes THREE well-separated PRNG streams off `seed`: the halo (`seed`), the
    /// disk *positions* (`mix(seed)`), and the disk *velocity dispersion*
    /// (`mix²(seed)`, drawn only when warm). Decoupling positions from the velocity
    /// stream means a warm and a cold disk with the same seed share every particle
    /// position — warmth perturbs only velocities. (`DiskCollision` reserves all
    /// three streams per galaxy when spacing its second galaxy's seed.)
    pub fn sample(&self, n_halo: usize, n_disk: usize, seed: u64) -> State {
        // Halo: reuse the Plummer sampler on the primary seed stream. It returns a
        // zero-COM/zero-momentum sphere already tagged Progenitor(0).
        let halo = self.halo.sample(n_halo, seed);

        // Disk positions: an independent stream (one mix step off the halo's seed).
        // Disk velocity dispersion: a SECOND independent stream (two mix steps), so
        // the number of position draws is fixed at 3/particle regardless of warmth.
        let mut rng = SplitMix64::new(mix_seed(seed));
        let mut rng_v = SplitMix64::new(mix_seed(mix_seed(seed)));
        let hz = self.scale_height;
        let m_disk_each = self.disk_mass / n_disk as f64;

        let n = n_halo + n_disk;
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut progenitor = Vec::with_capacity(n);

        for i in 0..n_halo {
            pos.push(halo.pos[i]);
            vel.push(halo.vel[i]);
            mass.push(halo.mass[i]);
            progenitor.push(Progenitor(0));
        }

        for _ in 0..n_disk {
            // Radius: invert the truncated-exponential cumulative F(R) = X on
            // [0, r_max] by bisection (F is monotone). One RNG draw per particle.
            let x = rng.next_f64();
            let r = self.sample_radius(x);

            let phi = TAU * rng.next_f64();
            let (sin_phi, cos_phi) = phi.sin_cos();

            // Vertical: isothermal sheet ∝ sech²(z/hz). Its CDF is (1+tanh(z/hz))/2,
            // so z = hz·atanh(2Y−1); clamp the argument off ±1 to bound |z|.
            let y = rng.next_f64();
            let t = (2.0 * y - 1.0).clamp(-1.0 + 1e-12, 1.0 - 1e-12);
            let z = hz * t.atanh();

            pos.push(DVec3::new(r * cos_phi, r * sin_phi, z));

            // Velocity. Cylindrical basis: R̂ = (cosφ, sinφ, 0), φ̂ = (−sinφ, cosφ, 0),
            // ẑ. The cold disk is a pure azimuthal orbit v = v_c(R)·φ̂ (spin +Z). A
            // warm disk adds Gaussian dispersion (σ_R, σ_φ, σ_z) about the drifting
            // mean v̄_φ, drawn from the SEPARATE velocity stream (Box–Muller), so the
            // position stream is untouched and warm/cold positions match bit-for-bit.
            let (v_r, v_phi, v_z) = match self.toomre_q {
                None => (0.0, self.circular_velocity(r), 0.0),
                Some(_) => {
                    let (g_r, g_z) = box_muller(rng_v.next_f64(), rng_v.next_f64());
                    let (g_phi, _) = box_muller(rng_v.next_f64(), rng_v.next_f64());
                    (
                        self.radial_dispersion(r) * g_r,
                        self.mean_azimuthal_velocity(r) + self.azimuthal_dispersion(r) * g_phi,
                        self.vertical_dispersion(r) * g_z,
                    )
                }
            };
            vel.push(DVec3::new(
                v_r * cos_phi - v_phi * sin_phi,
                v_r * sin_phi + v_phi * cos_phi,
                v_z,
            ));

            mass.push(m_disk_each);
            progenitor.push(Progenitor(1));
        }

        // The halo is already zero-COM/zero-momentum, but the finite-N disk carries
        // a small residual COM offset and net momentum. A final mass-weighted
        // recenter delivers the whole galaxy in the zero-COM/zero-momentum frame.
        let mtot: f64 = mass.iter().sum();
        let mean_pos = pos
            .iter()
            .zip(&mass)
            .fold(DVec3::ZERO, |acc, (p, m)| acc + *p * *m)
            / mtot;
        let mean_vel = vel
            .iter()
            .zip(&mass)
            .fold(DVec3::ZERO, |acc, (v, m)| acc + *v * *m)
            / mtot;
        for p in &mut pos {
            *p -= mean_pos;
        }
        for v in &mut vel {
            *v -= mean_vel;
        }

        let id = (0..n as u64).map(ParticleId).collect();
        State {
            pos,
            vel,
            mass,
            id,
            progenitor,
            kind: vec![Species::Collisionless; n],
            time: 0.0,
            a: 1.0,
        }
    }

    /// Invert the disk's radial cumulative distribution: return the R ∈ [0, r_max]
    /// with M_d(<R)/M_d = `x` (x ∈ [0,1)). Bisection on the monotone CDF.
    fn sample_radius(&self, x: f64) -> f64 {
        let (mut lo, mut hi) = (0.0_f64, self.r_max);
        let target = x * self.disk_mass;
        // ~50 iterations halve [0, r_max] to well below f64 precision.
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.disk_enclosed_mass(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        0.5 * (lo + hi)
    }
}

/// SplitMix64: the same tiny deterministic PRNG the Plummer sampler uses (avoids
/// an external `rand` dependency). `next_f64` returns a value in [0, 1).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Box–Muller transform: two uniforms in [0, 1) → two independent standard normals
/// N(0, 1). `u1` is clamped off 0 so the ln is finite (`next_f64` can return 0).
fn box_muller(u1: f64, u2: f64) -> (f64, f64) {
    let radius = (-2.0 * u1.max(f64::MIN_POSITIVE).ln()).sqrt();
    let (sin, cos) = (TAU * u2).sin_cos();
    (radius * cos, radius * sin)
}

/// One SplitMix64 step, deriving the disk's PRNG seed from the halo's so the two
/// populations draw from well-separated streams. Mirrors `collision.rs`.
fn mix_seed(seed: u64) -> u64 {
    let z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
