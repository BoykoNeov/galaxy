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

use glam::{Vec2, Vec3};

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
    // Evaluated in f64: the quintic in f32 wobbles by ~5e-7 near u = 1, enough
    // to break monotonicity between adjacent samples; f64 error (~1e-16) is far
    // below any frame-to-frame increment, and the downcast is monotone.
    let u = f64::from(u.clamp(0.0, 1.0));
    (u * u * u * (u * (6.0 * u - 15.0) + 10.0)) as f32
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
    if raw.is_empty() || window == 0 {
        return raw.to_vec();
    }
    let n = raw.len();
    let w = window as isize;

    // Moving max over ±window frames.
    let moving_max: Vec<f32> = (0..n)
        .map(|i| {
            let lo = (i as isize - w).max(0) as usize;
            let hi = ((i as isize + w) as usize).min(n - 1);
            raw[lo..=hi]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max)
        })
        .collect();

    // Truncated Gaussian, radius = window (never past the moving-max half-window,
    // which is what makes the convex combination ≥ raw[i]), σ = window/3.
    let sigma = window as f32 / 3.0;
    let weights: Vec<f32> = (-w..=w)
        .map(|k| (-0.5 * (k as f32 / sigma).powi(2)).exp())
        .collect();
    let weight_sum: f32 = weights.iter().sum();

    (0..n)
        .map(|i| {
            let blurred: f32 = (-w..=w)
                .zip(&weights)
                .map(|(k, &g)| {
                    let j = (i as isize + k).clamp(0, n as isize - 1) as usize;
                    g * moving_max[j]
                })
                .sum::<f32>()
                / weight_sum;
            // ≥ raw holds in exact arithmetic; the max makes it hold in f32 too.
            blurred.max(raw[i])
        })
        .collect()
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
        CameraPath(PathKind::Fixed(camera))
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
        if radii.is_empty() {
            return Err(RigError::EmptyRadii);
        }
        if !radii.iter().all(|r| r.is_finite() && *r > 0.0) {
            return Err(RigError::InvalidParam(
                "framing radii must be finite and positive",
            ));
        }
        if !(margin.is_finite() && margin >= 0.0) {
            return Err(RigError::InvalidParam(
                "margin must be finite and non-negative",
            ));
        }
        if !(aspect.is_finite() && aspect > 0.0) {
            return Err(RigError::InvalidParam("aspect must be finite and positive"));
        }
        if ![azimuth.0, azimuth.1, tilt.0, tilt.1]
            .iter()
            .all(|a| a.is_finite())
        {
            return Err(RigError::InvalidParam("angles must be finite"));
        }
        Ok(CameraPath(PathKind::OrbitTilt {
            target,
            azimuth,
            tilt,
            radii,
            margin,
            aspect,
        }))
    }

    /// The camera at timeline position `u` (clamped to `[0, 1]`). Deterministic
    /// pure function of the path — same `u`, same camera, bit-for-bit.
    pub fn camera_at(&self, u: f32) -> Camera {
        match &self.0 {
            PathKind::Fixed(camera) => *camera,
            PathKind::OrbitTilt {
                target,
                azimuth,
                tilt,
                radii,
                margin,
                aspect,
            } => {
                let u = u.clamp(0.0, 1.0);
                let e = ease_in_out(u);
                let theta = lerp(azimuth.0, azimuth.1, e);
                let phi = lerp(tilt.0, tilt.1, e);
                let r = sample_track(radii, u);

                // The documented spherical basis: camera along r̂, looking back,
                // screen-up −φ̂ (unit, exactly ⊥ the view axis — never degenerate).
                let (sin_t, cos_t) = theta.sin_cos();
                let (sin_p, cos_p) = phi.sin_cos();
                let r_hat = Vec3::new(sin_p * cos_t, sin_p * sin_t, cos_p);
                let up_hint = Vec3::new(-cos_p * cos_t, -cos_p * sin_t, sin_p);

                // r·(1+margin) on the short axis, widened to the image aspect —
                // the same aspect law as Camera::frame_bounds.
                let e_short = r * (1.0 + margin);
                let half_extent = if *aspect >= 1.0 {
                    Vec2::new(e_short * aspect, e_short)
                } else {
                    Vec2::new(e_short, e_short / aspect)
                };
                Camera::orthographic(*target, -r_hat, up_hint, half_extent)
            }
        }
    }
}

/// Two-product lerp `(1−t)·a + t·b`: exact at both endpoints (the house form,
/// per the M6c attribute-lerp precedent).
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    (1.0 - t) * a + t * b
}

/// Sample a per-frame track at timeline position `u ∈ [0, 1]` by linear
/// interpolation between adjacent entries (endpoints land exactly on the first
/// and last entries).
fn sample_track(track: &[f32], u: f32) -> f32 {
    let n = track.len();
    if n == 1 {
        return track[0];
    }
    let t = u * (n - 1) as f32;
    let i = (t.floor() as usize).min(n - 2);
    lerp(track[i], track[i + 1], t - i as f32)
}
