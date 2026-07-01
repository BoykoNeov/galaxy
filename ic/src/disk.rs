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

use galaxy_core::State;

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

    /// Central surface density Σ₀, fixed by normalizing the truncated exponential
    /// to the total disk mass: M_d = 2π Σ₀ Rd² · [1 − (1 + u)e^(−u)], u = r_max/Rd.
    pub fn central_surface_density(&self) -> f64 {
        todo!("normalize the truncated exponential to disk_mass")
    }

    /// Surface density Σ(R) = Σ₀ e^(−R/Rd) for R ≤ r_max, else 0.
    pub fn surface_density(&self, _r: f64) -> f64 {
        todo!("evaluate the truncated exponential surface density")
    }

    /// Disk mass enclosed within cylindrical radius R:
    /// M_d(<R) = M_d · [1 − (1 + R/Rd)e^(−R/Rd)] / [1 − (1 + u)e^(−u)], u = r_max/Rd.
    pub fn disk_enclosed_mass(&self, _r: f64) -> f64 {
        todo!("closed-form truncated-exponential cylindrical enclosed mass")
    }

    /// Circular speed v_c(R) = √(G · [M_halo(<R) + M_disk(<R)] / R) from the
    /// combined enclosed mass — the spherical Plummer term plus the cylindrical
    /// disk term. This is the target mean azimuthal speed of the cold disk.
    pub fn circular_velocity(&self, _r: f64) -> f64 {
        todo!("combined-enclosed-mass circular velocity")
    }

    /// Total galaxy mass, disk + halo.
    pub fn total_mass(&self) -> f64 {
        todo!("disk_mass + halo.total_mass")
    }

    /// Sample the galaxy: `n_halo` Plummer halo particles (`Progenitor(0)`) drawn
    /// from the halo distribution function, followed by `n_disk` disk particles
    /// (`Progenitor(1)`) on cold near-circular orbits. Deterministic in `seed`,
    /// contiguous unique ids, delivered in the zero-COM / zero-momentum frame.
    pub fn sample(&self, _n_halo: usize, _n_disk: usize, _seed: u64) -> State {
        todo!("sample halo (Plummer) + cold exponential disk, recentered")
    }
}
