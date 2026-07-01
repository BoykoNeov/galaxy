//! Disk orientation for a galaxy collision: a rigid rotation that sets a disk's
//! spin axis relative to the encounter's orbital plane.
//!
//! The encounter fixes its geometry once (`encounter.rs`): the orbit lies in the
//! **x–y plane**, so its angular momentum points along **+Z**. A freshly-sampled
//! disk (`ExponentialDisk`) spins about **+Z** in its own body frame. An
//! `Orientation` rotates that body frame into the encounter frame *before* the
//! galaxy is placed on its orbit, which is exactly what distinguishes the classic
//! Toomre passage geometries:
//!
//! - **prograde** (identity): spin stays +Z, co-rotating with the orbit — the
//!   resonant, tail-making case.
//! - **retrograde**: spin flipped to −Z, counter-rotating.
//! - **inclined(i)**: spin tilted by `i` out of the orbital plane about the line
//!   of nodes.
//!
//! The public surface is **physical angles** (inclination + argument of the node,
//! the two Toomre parameters); the quaternion is an implementation detail. A
//! rotation is rigid (length-preserving) and maps a zero-mean point set to a
//! zero-mean point set, so applying one never disturbs a galaxy's internal
//! structure or its zero-COM / zero-momentum framing.

use galaxy_core::{DQuat, DVec3};

/// A rigid orientation of a galaxy relative to the encounter's orbital plane.
///
/// Construct one with [`Orientation::prograde`], [`Orientation::retrograde`],
/// [`Orientation::inclined`], or the general [`Orientation::from_angles`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Orientation {
    /// Rotation carrying the galaxy body frame into the encounter frame.
    rot: DQuat,
}

impl Orientation {
    /// Prograde, coplanar: the identity. The disk spin stays along +Z, co-rotating
    /// with the orbital angular momentum — the tail-making resonant passage.
    pub fn prograde() -> Self {
        todo!()
    }

    /// Retrograde, coplanar: the disk spin is flipped to −Z (a π rotation about the
    /// line of nodes, +x). The disk still lies in the orbital plane but counter-
    /// rotates. Equivalent to `inclined(π)`.
    pub fn retrograde() -> Self {
        todo!()
    }

    /// Tilt the disk by `inclination` (radians) about the line of nodes (+x): the
    /// spin axis moves from +Z to angle `inclination` off +Z. `inclined(0)` is
    /// prograde; `inclined(π)` is retrograde.
    pub fn inclined(inclination: f64) -> Self {
        todo!()
    }

    /// The general orientation from the two Toomre angles: `inclination` (tilt of
    /// the disk plane out of the orbital plane) about the line of nodes, whose
    /// azimuth in the orbital plane is `argument`. Composed as Rz(argument)·Rx(incl),
    /// so the spin axis +Z maps to (sin i·sin ω, −sin i·cos ω, cos i) — a tilt of
    /// exactly `inclination` off +Z, with the node line rotated by `argument`.
    pub fn from_angles(inclination: f64, argument: f64) -> Self {
        let _ = (inclination, argument);
        todo!()
    }

    /// Apply the orientation to a body-frame vector (position or velocity).
    pub fn apply(&self, v: DVec3) -> DVec3 {
        let _ = v;
        todo!()
    }
}
