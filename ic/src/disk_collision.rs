//! Two-galaxy collision of *rotating disk* galaxies — the IC that turns the
//! disk-tail physics (`ExponentialDisk`) into an actual encounter, and the last
//! wiring step before the two-disk tidal-tail movie.
//!
//! It is the disk analogue of [`crate::Collision`] (two Plummer spheres) and
//! reuses the same two-body Kepler placement (`encounter`), so the orbital setup
//! is guarded by the very same osculating-elements tests. The two additions over
//! the Plummer case are:
//!
//!  1. **Orientation.** Each disk carries an [`Orientation`] that sets its spin
//!     axis relative to the orbital plane (prograde / retrograde / inclined) — the
//!     Toomre spin-orbit geometry. The default (both `prograde`, spin +Z, orbit in
//!     x–y) is the coplanar prograde passage that makes the cleanest tails.
//!  2. **Four species.** A disk galaxy has two populations (halo, disk), so a
//!     disk-disk encounter has four. They are tagged so the renderer can color the
//!     two *disks* (the tails) apart from the two halos:
//!       - galaxy 1: halo `Progenitor(0)`, disk `Progenitor(1)`
//!       - galaxy 2: halo `Progenitor(2)`, disk `Progenitor(3)`
//!
//! The combined realization is delivered in the global zero-COM / zero-momentum
//! frame with contiguous unique ids, galaxy 1's particles first.

use galaxy_core::{Progenitor, State};

use crate::{ExponentialDisk, Orientation};

/// A two-body Kepler encounter between two rotating disk galaxies, each with its
/// own spin-orbit [`Orientation`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskCollision {
    /// The first galaxy — halo `Progenitor(0)`, disk `Progenitor(1)`.
    pub galaxy1: ExponentialDisk,
    /// The second galaxy — halo `Progenitor(2)`, disk `Progenitor(3)`.
    pub galaxy2: ExponentialDisk,
    /// Spin-orbit orientation of galaxy 1 (default `prograde`).
    pub orient1: Orientation,
    /// Spin-orbit orientation of galaxy 2 (default `prograde`).
    pub orient2: Orientation,
    /// Orbital eccentricity e (e=1 parabolic — the classic tidal case).
    pub eccentricity: f64,
    /// Pericenter separation of the relative orbit.
    pub pericenter: f64,
    /// Initial COM–COM separation (≥ pericenter; ≤ apocenter if bound).
    pub separation: f64,
}

impl DiskCollision {
    /// Construct a disk-disk encounter. Both galaxies must share the same `G`; the
    /// eccentricity and pericenter must be strictly positive and the initial
    /// separation must be at least the pericenter (and at most the apocenter for a
    /// bound orbit). Both disks start `prograde` (coplanar, spin +Z); set
    /// [`orient1`](Self::orient1) / [`orient2`](Self::orient2) for other geometries.
    pub fn new(
        galaxy1: ExponentialDisk,
        galaxy2: ExponentialDisk,
        eccentricity: f64,
        pericenter: f64,
        separation: f64,
    ) -> Self {
        let _ = (galaxy1, galaxy2, eccentricity, pericenter, separation);
        todo!()
    }

    /// The shared gravitational constant `G`.
    pub fn g(&self) -> f64 {
        todo!()
    }

    /// Relative position/velocity `(r_rel, v_rel)` of the two COMs on the incoming
    /// branch of the Kepler orbit (see [`crate::Collision::relative_state`]).
    pub fn relative_state(&self) -> (galaxy_core::DVec3, galaxy_core::DVec3) {
        todo!()
    }

    /// Per-galaxy COM `(position, velocity)` in the global zero-COM / zero-momentum
    /// frame: `((r1, v1), (r2, v2))`.
    pub fn com_states(
        &self,
    ) -> (
        (galaxy_core::DVec3, galaxy_core::DVec3),
        (galaxy_core::DVec3, galaxy_core::DVec3),
    ) {
        todo!()
    }

    /// Sample the full encounter: galaxy 1 gets `n_halo1` halo + `n_disk1` disk
    /// particles, galaxy 2 gets `n_halo2` + `n_disk2`. Each galaxy is sampled in
    /// its body frame, rotated by its [`Orientation`], then rigidly placed at its
    /// COM orbital state; the two are concatenated (galaxy 1 first), tagged into the
    /// four progenitors, given contiguous unique ids, and recentered to the global
    /// zero-COM / zero-momentum frame. Deterministic in `seed`.
    pub fn sample(
        &self,
        n_halo1: usize,
        n_disk1: usize,
        n_halo2: usize,
        n_disk2: usize,
        seed: u64,
    ) -> State {
        let _ = (n_halo1, n_disk1, n_halo2, n_disk2, seed);
        let _ = Progenitor(0);
        todo!()
    }
}
