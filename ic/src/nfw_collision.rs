//! Two-galaxy **NFW–NFW collision** initial conditions — the demoable payoff of
//! the M5 cuspy-halo ladder.
//!
//! Builds a single `State` from two exponentially-truncated NFW halos
//! ([`TruncatedNfw`], M5d) set on a relative two-body Kepler encounter. Each halo's
//! center of mass is treated as a point mass of the halo's **full** (skirt-
//! inclusive) mass, and the two COMs are placed on the relative orbit (the conic is
//! set by eccentricity + pericenter; the starting point on it by the initial
//! separation). The canonical Toomre & Toomre tidal setup is the *parabolic*
//! encounter (e = 1).
//!
//! This is the direct analogue of the Plummer [`crate::Collision`]: an NFW halo is
//! spherical, isotropic and non-rotating, so — unlike the rotating-disk
//! [`crate::DiskCollision`] — there is no spin-orbit orientation and no multi-
//! species split. Both types delegate the orbital placement to the shared
//! [`crate::encounter`] module, so the one set of osculating-elements tests guards
//! the conic for all of them.
//!
//! The two halos are tagged with distinct `Progenitor` ids (0 and 1) — what lets
//! the renderer color the merging/stripped material by its source halo — and the
//! combined realization is delivered in the global zero-COM / zero-momentum frame
//! with contiguous, unique particle ids (galaxy 1's particles first).
//!
//! Why the *truncated* NFW and not the hard-cut M5c [`crate::Nfw`]: a collision is
//! exactly the regime the M5c caveat bites. M5c samples velocities from the
//! *untruncated* DF, so its outer halo re-virializes — and the outer halo is the
//! material tidally stripped into the bridges/debris the demo exists to show.
//! M5d's self-consistent (ρ, Ψ) makes those outskirts a genuine equilibrium before
//! the encounter perturbs them.
//!
//! NOTE on the physics: an NFW halo is *not* a point mass, so the inter-halo force
//! only approaches the point-mass Kepler force when the separation is much larger
//! than the scale radius, and the approximation degrades as the halos overlap near
//! pericenter. The Kepler setup therefore fixes the *initial* COM phase-space
//! coordinates exactly; the subsequent many-body evolution is the simulation's job,
//! not a closed form.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};

use crate::encounter;
use crate::TruncatedNfw;

/// A two-body Kepler encounter between two exponentially-truncated NFW halos.
///
/// Both halos must share the same gravitational constant `G`. The relative orbit of
/// their centers of mass is the conic with the given `eccentricity` and
/// `pericenter` separation; the halos start on the *incoming* branch at the given
/// COM–COM `separation` (which must be ≥ `pericenter`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NfwCollision {
    /// The first (primary) halo — tagged `Progenitor(0)`.
    pub galaxy1: TruncatedNfw,
    /// The second halo — tagged `Progenitor(1)`.
    pub galaxy2: TruncatedNfw,
    /// Orbital eccentricity e of the relative orbit. e=1 is parabolic (the classic
    /// Toomre encounter), e<1 bound, e>1 hyperbolic. Must be strictly positive.
    pub eccentricity: f64,
    /// Pericenter separation r_peri of the relative orbit (closest approach of the
    /// two COMs on the conic). Must be strictly positive.
    pub pericenter: f64,
    /// Initial COM–COM separation r0 at which the halos start. Must be ≥
    /// `pericenter` (and, for a bound orbit, ≤ the apocenter).
    pub separation: f64,
}

impl NfwCollision {
    /// Construct an encounter. Both halos must share the same `G`; the eccentricity
    /// and pericenter must be strictly positive and the initial separation at least
    /// the pericenter (and at most the apocenter for a bound orbit).
    pub fn new(
        galaxy1: TruncatedNfw,
        galaxy2: TruncatedNfw,
        eccentricity: f64,
        pericenter: f64,
        separation: f64,
    ) -> Self {
        assert_eq!(
            galaxy1.base.g, galaxy2.base.g,
            "both halos must share the same gravitational constant G"
        );
        encounter::validate_orbit(eccentricity, pericenter, separation);
        Self {
            galaxy1,
            galaxy2,
            eccentricity,
            pericenter,
            separation,
        }
    }

    /// The shared gravitational constant `G`.
    pub fn g(&self) -> f64 {
        self.galaxy1.base.g
    }

    /// Relative position and velocity of the two COMs, `(r_rel, v_rel)` with
    /// `r_rel = r2 − r1` and `v_rel = v2 − v1`, on the incoming branch of the Kepler
    /// orbit. Pericenter lies along +x; the orbit is in the x–y plane.
    ///
    /// The two-body mass is the **combined full** mass `total_mass()` of each halo
    /// (virial + exponential skirt) — the same mass `sample` actually places — not
    /// `M_vir`.
    pub fn relative_state(&self) -> (DVec3, DVec3) {
        let mu = self.g() * (self.galaxy1.total_mass() + self.galaxy2.total_mass());
        encounter::relative_state(mu, self.eccentricity, self.pericenter, self.separation)
    }

    /// The two COMs' `(position, velocity)` in the global zero-COM / zero-momentum
    /// frame: `((r1, v1), (r2, v2))`. By construction `m1·r1 + m2·r2 = 0` and
    /// `m1·v1 + m2·v2 = 0`, and `r2 − r1 = r_rel`, `v2 − v1 = v_rel`.
    pub fn com_states(&self) -> ((DVec3, DVec3), (DVec3, DVec3)) {
        let (r_rel, v_rel) = self.relative_state();
        encounter::com_states(
            self.galaxy1.total_mass(),
            self.galaxy2.total_mass(),
            r_rel,
            v_rel,
        )
    }

    /// Sample the full collision: `n1` particles for halo 1 (tagged `Progenitor(0)`)
    /// and `n2` for halo 2 (tagged `Progenitor(1)`), each drawn from its truncated-
    /// NFW distribution, rigidly placed at its COM orbital state, then concatenated.
    /// The result is deterministic in `seed`, carries contiguous unique ids
    /// `0..n1+n2`, and sits in the zero-COM/zero-momentum frame.
    pub fn sample(&self, n1: usize, n2: usize, seed: u64) -> State {
        let ((r1, v1), (r2, v2)) = self.com_states();

        // The two halos draw from well-separated PRNG streams (one SplitMix64 step
        // apart) so an equal-model collision still yields distinct draws.
        // `TruncatedNfw::sample` consumes a single stream (unlike the disk's three),
        // so one mix step suffices to keep them disjoint.
        let s1 = self.galaxy1.sample(n1, seed);
        let s2 = self.galaxy2.sample(n2, mix_seed(seed));

        let n = n1 + n2;
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut progenitor = Vec::with_capacity(n);

        // Each halo is sampled in its own zero-COM/zero-momentum frame, then rigidly
        // translated to its COM position and boosted by its COM velocity.
        for i in 0..n1 {
            pos.push(s1.pos[i] + r1);
            vel.push(s1.vel[i] + v1);
            mass.push(s1.mass[i]);
            progenitor.push(Progenitor(0));
        }
        for i in 0..n2 {
            pos.push(s2.pos[i] + r2);
            vel.push(s2.vel[i] + v2);
            mass.push(s2.mass[i]);
            progenitor.push(Progenitor(1));
        }

        // The COM split makes the barycenter vanish analytically; a final mass-
        // weighted recenter removes the O(machine-ε) finite-N residual so the
        // realization is delivered in the zero-COM/zero-momentum frame to roundoff.
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
            u: vec![0.0; n],
            time: 0.0,
            a: 1.0,
        }
    }
}

/// One SplitMix64 step, used to derive the second halo's seed from the first so the
/// two draw from well-separated, independent PRNG streams. Mirrors `collision.rs`.
fn mix_seed(seed: u64) -> u64 {
    let z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
