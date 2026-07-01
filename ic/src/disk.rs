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
//! Disk particles are tagged `Progenitor(1)` (species: disk) and halo particles
//! `Progenitor(0)` (species: halo/bulge), so the renderer can color them apart and
//! tests can select the disk population robustly. `sample` returns the halo
//! particles first, then the disk particles.

use galaxy_core::{DVec3, ParticleId, Progenitor, State};

use std::f64::consts::TAU;

use crate::Plummer;

/// A cold exponential disk of low-mass particles inside a live Plummer halo/bulge.
///
/// Choose units freely; `g` must match the halo's `g`. The disk mass should be a
/// small fraction of the halo mass (submaximal) for the cold disk to hold together.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ExponentialDisk {
    /// Gravitational constant `G` (must equal `halo.g`).
    pub g: f64,
    /// Total disk mass `M_d` (should be ≪ `halo.total_mass`).
    pub disk_mass: f64,
    /// Radial exponential scale length `Rd`.
    pub scale_length: f64,
    /// Vertical sech² scale height `hz` (thin disk: hz ≪ Rd).
    pub scale_height: f64,
    /// Truncation radius (no disk particles beyond this cylindrical radius).
    pub r_max: f64,
    /// The live spherical halo/bulge the disk is embedded in.
    pub halo: Plummer,
}

impl ExponentialDisk {
    /// Construct a disk galaxy. All scalar parameters must be strictly positive,
    /// `g` must equal `halo.g`, and `r_max` must exceed the scale length.
    pub fn new(
        disk_mass: f64,
        scale_length: f64,
        scale_height: f64,
        r_max: f64,
        halo: Plummer,
    ) -> Self {
        assert!(disk_mass > 0.0, "disk mass must be positive");
        assert!(scale_length > 0.0, "scale length must be positive");
        assert!(scale_height > 0.0, "scale height must be positive");
        assert!(
            r_max > scale_length,
            "truncation radius ({r_max}) must exceed the scale length ({scale_length})"
        );
        Self {
            g: halo.g,
            disk_mass,
            scale_length,
            scale_height,
            r_max,
            halo,
        }
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
    pub fn circular_velocity(&self, r: f64) -> f64 {
        let m_enc = self.halo.enclosed_mass(r) + self.disk_enclosed_mass(r);
        (self.g * m_enc / r).sqrt()
    }

    /// Total galaxy mass, disk + halo.
    pub fn total_mass(&self) -> f64 {
        self.disk_mass + self.halo.total_mass
    }

    /// Sample the galaxy: `n_halo` Plummer halo particles (`Progenitor(0)`) drawn
    /// from the halo distribution function, followed by `n_disk` disk particles
    /// (`Progenitor(1)`) on cold near-circular orbits. Deterministic in `seed`,
    /// contiguous unique ids, delivered in the zero-COM / zero-momentum frame.
    pub fn sample(&self, n_halo: usize, n_disk: usize, seed: u64) -> State {
        // Halo: reuse the Plummer sampler on the primary seed stream. It returns a
        // zero-COM/zero-momentum sphere already tagged Progenitor(0).
        let halo = self.halo.sample(n_halo, seed);

        // Disk: an independent PRNG stream (one mix step off the halo's seed) so
        // the two populations never share draws.
        let mut rng = SplitMix64::new(mix_seed(seed));
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

            // Cold, purely azimuthal orbit with spin along +Z: v = v_c(R)·φ̂,
            // φ̂ = (−sinφ, cosφ, 0). No radial or vertical velocity (fully cold).
            let v_c = self.circular_velocity(r);
            vel.push(DVec3::new(-v_c * sin_phi, v_c * cos_phi, 0.0));

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

/// One SplitMix64 step, deriving the disk's PRNG seed from the halo's so the two
/// populations draw from well-separated streams. Mirrors `collision.rs`.
fn mix_seed(seed: u64) -> u64 {
    let z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
