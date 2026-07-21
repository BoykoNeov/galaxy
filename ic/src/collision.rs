//! Two-galaxy collision initial conditions.
//!
//! Builds a single `State` from two Plummer galaxies set on a Kepler encounter.
//! Each galaxy's center of mass is treated as a point mass of the galaxy's total
//! mass, and the two COMs are placed on a relative two-body orbit (the conic is
//! set by eccentricity + pericenter; the starting point on it by the initial
//! separation). The canonical Toomre & Toomre tidal-tail setup is the *parabolic*
//! encounter (e = 1).
//!
//! The two galaxies are tagged with distinct `Progenitor` ids (0 and 1) — this is
//! what lets the renderer color tidal tails by their source galaxy — and the
//! combined realization is delivered in the global zero-COM / zero-momentum frame
//! with contiguous, unique particle ids.
//!
//! NOTE on the physics: a Plummer sphere is *not* a point mass (its density has
//! infinite extent), so the inter-galaxy force only approaches the point-mass
//! Kepler force when the separation is much larger than the scale radius, and the
//! approximation degrades as the galaxies overlap near pericenter. The Kepler
//! setup therefore fixes the *initial* COM phase-space coordinates exactly; the
//! subsequent many-body evolution is the simulation's job, not a closed form.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};

use crate::encounter;
use crate::Plummer;

/// A two-body Kepler encounter between two Plummer galaxies.
///
/// Both galaxies must share the same gravitational constant `G`. The relative
/// orbit of their centers of mass is the conic with the given `eccentricity` and
/// `pericenter` separation; the galaxies start on the *incoming* branch at the
/// given COM-COM `separation` (which must be ≥ `pericenter`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Collision {
    /// The first (primary) galaxy — tagged `Progenitor(0)`.
    pub galaxy1: Plummer,
    /// The second galaxy — tagged `Progenitor(1)`.
    pub galaxy2: Plummer,
    /// Orbital eccentricity e of the relative orbit. e=1 is parabolic (the
    /// classic Toomre tidal-tail encounter), e<1 bound, e>1 hyperbolic. Must be
    /// strictly positive.
    pub eccentricity: f64,
    /// Pericenter separation r_peri of the relative orbit (closest approach of
    /// the two COMs on the conic). Must be strictly positive.
    pub pericenter: f64,
    /// Initial COM-COM separation r0 at which the galaxies start. Must be
    /// ≥ `pericenter` (and, for a bound orbit, ≤ the apocenter).
    pub separation: f64,
}

impl Collision {
    /// Construct an encounter. Both galaxies must share the same `G`; the
    /// eccentricity and pericenter must be strictly positive and the initial
    /// separation must be at least the pericenter.
    pub fn new(
        galaxy1: Plummer,
        galaxy2: Plummer,
        eccentricity: f64,
        pericenter: f64,
        separation: f64,
    ) -> Self {
        assert_eq!(
            galaxy1.g, galaxy2.g,
            "both galaxies must share the same gravitational constant G"
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
        self.galaxy1.g
    }

    /// Relative position and velocity of the two COMs, `(r_rel, v_rel)` with
    /// `r_rel = r2 − r1` and `v_rel = v2 − v1`, on the incoming branch of the
    /// Kepler orbit. Pericenter lies along +x; the orbit is in the x–y plane.
    pub fn relative_state(&self) -> (DVec3, DVec3) {
        let mu = self.g() * (self.galaxy1.total_mass + self.galaxy2.total_mass);
        encounter::relative_state(mu, self.eccentricity, self.pericenter, self.separation)
    }

    /// The two COMs' `(position, velocity)` in the global zero-COM /
    /// zero-momentum frame: `((r1, v1), (r2, v2))`. By construction
    /// `m1·r1 + m2·r2 = 0` and `m1·v1 + m2·v2 = 0`, and `r2 − r1 = r_rel`,
    /// `v2 − v1 = v_rel`.
    pub fn com_states(&self) -> ((DVec3, DVec3), (DVec3, DVec3)) {
        let (r_rel, v_rel) = self.relative_state();
        encounter::com_states(
            self.galaxy1.total_mass,
            self.galaxy2.total_mass,
            r_rel,
            v_rel,
        )
    }

    /// Sample the full collision: `n1` particles for galaxy 1 (tagged
    /// `Progenitor(0)`) and `n2` for galaxy 2 (tagged `Progenitor(1)`), each drawn
    /// from its Plummer distribution function, rigidly placed at its COM orbital
    /// state, then concatenated. The result is deterministic in `seed`, carries
    /// contiguous unique ids `0..n1+n2`, and sits in the zero-COM/zero-momentum
    /// frame.
    pub fn sample(&self, n1: usize, n2: usize, seed: u64) -> State {
        let ((r1, v1), (r2, v2)) = self.com_states();

        // The two galaxies draw from well-separated PRNG streams (one SplitMix64
        // step apart) so an equal-model collision still yields distinct draws.
        let s1 = self.galaxy1.sample(n1, seed);
        let s2 = self.galaxy2.sample(n2, mix_seed(seed));

        let n = n1 + n2;
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut progenitor = Vec::with_capacity(n);

        // Each galaxy is sampled in its own zero-COM/zero-momentum frame, then
        // rigidly translated to its COM position and boosted by its COM velocity.
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

        // The COM split makes the barycenter vanish analytically; a final
        // mass-weighted recenter removes the O(machine-ε) finite-N residual so the
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
            formation_time: vec![State::PRIMORDIAL; n],
            time: 0.0,
            a: 1.0,
        }
    }
}

/// One SplitMix64 step, used to derive the second galaxy's seed from the first so
/// the two galaxies draw from well-separated, independent PRNG streams. Mirrors
/// the generator in `plummer.rs`.
fn mix_seed(seed: u64) -> u64 {
    let z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
