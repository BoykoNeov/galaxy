//! Camera rig (DESIGN.md M6d): a **time-parameterized** camera for the movie
//! pipeline — smoothed framing envelope + eased orbit/tilt paths — replacing the
//! single static framing. Pure CPU math; xtask wires a path per scenario.
//!
//! Conventions (pinned by the gates in `tests/rig.rs`):
//!
//! * **Angles** are spherical about the collision's orbital-plane normal (+Z).
//!   The camera sits along `r̂(θ, φ) = (sin φ·cos θ, sin φ·sin θ, cos φ)` from the
//!   target and looks back along `−r̂`. `φ` (tilt) is the polar angle from +Z:
//!   `0` = face-on, `π/2` = edge-on. `θ` (azimuth) orbits in the orbital plane.
//!   Screen-up is `−φ̂ = (−cos φ·cos θ, −cos φ·sin θ, sin φ)` — exactly orthogonal
//!   to the view axis for every `(θ, φ)`, so the basis is never degenerate; at
//!   `θ = −π/2, φ = 0` it reproduces the historical face-on `+Y`-up orientation.
//! * **Framing radius** is sampled from a per-frame radii track by *linear*
//!   interpolation in raw `u` (the movie timeline), NOT in eased time — the
//!   envelope is already time-aligned to the action; easing applies to angles
//!   only. `half_extent.y = r·(1 + margin)`, widened to the image aspect exactly
//!   like [`Camera::frame_bounds`].

use glam::Vec3;

use crate::camera::Camera;

/// Errors constructing a [`CameraPath`].
#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum RigError {
    /// The orbit/tilt path needs at least one framing radius to sample.
    #[error("orbit/tilt path needs at least one framing radius")]
    EmptyRadii,
    /// A numeric parameter is outside its documented domain (non-finite, or
    /// non-positive where positivity is required). The string names the offender.
    #[error("invalid orbit/tilt parameter: {0}")]
    InvalidParam(&'static str),
}

/// Quintic smootherstep ease-in-out: `6u⁵ − 15u⁴ + 10u³`, with `u` clamped to
/// `[0, 1]`. Endpoints are exact (`ease(0) = 0`, `ease(1) = 1`) and the first
/// derivative vanishes at both ends — an animated camera starts and stops at
/// rest instead of jerking. Peak slope is `15/8` at `u = ½` (the smoothness
/// gates budget against it).
pub fn ease_in_out(u: f32) -> f32 {
    let _ = u;
    todo!("M6d")
}

/// Temporally smoothed **upper envelope** of a per-frame framing-radius track:
/// a moving max over `±window` frames, then a truncated Gaussian (σ = window/3,
/// kernel radius = window, edge indices clamped). Because the kernel never
/// reaches past the moving-max half-window, the result is a convex combination
/// of maxima that each cover frame `i` — so `out[i] ≥ raw[i]` holds exactly
/// (enforced against fp rounding by a final elementwise max). The camera zooms
/// out *before* a transient needs the room (the moving max looks ahead) and
/// never crops tighter than the raw per-frame requirement.
///
/// `window = 0` (or an empty `raw`) returns the input unchanged.
pub fn smooth_envelope(raw: &[f32], window: usize) -> Vec<f32> {
    let _ = (raw, window);
    todo!("M6d")
}

/// A time-parameterized camera over the movie's unit timeline `u ∈ [0, 1]`.
#[derive(Clone, Debug)]
pub struct CameraPath(PathKind);

#[derive(Clone, Debug)]
enum PathKind {
    /// The back-compat default: one static camera for every frame.
    Fixed(Camera),
    /// Eased azimuth/tilt sweep + breathing framing radius about a fixed target.
    OrbitTilt {
        target: Vec3,
        /// (start, end) azimuth θ in radians, eased over the timeline.
        azimuth: (f32, f32),
        /// (start, end) polar tilt φ in radians from face-on (+Z), eased.
        tilt: (f32, f32),
        /// Per-frame framing radii (world units), sampled by linear interp in u.
        radii: Vec<f32>,
        /// Fractional framing margin (as [`crate::camera::DEFAULT_MARGIN`]).
        margin: f32,
        /// Image aspect (width : height) the half-extent is widened to.
        aspect: f32,
    },
}

impl CameraPath {
    /// The static path: every `camera_at(u)` returns `camera` unchanged —
    /// bit-exact back-compat with the pre-M6d single-framing pipeline.
    pub fn fixed(camera: Camera) -> Self {
        let _ = camera;
        todo!("M6d")
    }

    /// An eased orbit/tilt sweep about `target` with a breathing framing radius.
    ///
    /// Validation (fail loudly, per house style): `radii` non-empty with every
    /// entry finite and `> 0`; `margin` finite and `≥ 0`; `aspect` finite and
    /// `> 0`; all four angles finite. Angles are otherwise unrestricted (multi-
    /// turn orbits are legitimate).
    pub fn orbit_tilt(
        target: Vec3,
        azimuth: (f32, f32),
        tilt: (f32, f32),
        radii: Vec<f32>,
        margin: f32,
        aspect: f32,
    ) -> Result<Self, RigError> {
        let _ = (target, azimuth, tilt, radii, margin, aspect);
        todo!("M6d")
    }

    /// The camera at timeline position `u` (clamped to `[0, 1]`). Deterministic
    /// pure function of the path — same `u`, same camera, bit-for-bit.
    pub fn camera_at(&self, u: f32) -> Camera {
        let _ = u;
        todo!("M6d")
    }
}
