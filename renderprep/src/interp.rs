//! Hermite temporal upsampling (DESIGN.md M6c): physically-informed in-between
//! frames from adjacent snapshots, at zero simulation cost.
//!
//! Snapshots store full phase space (`pos` *and* `vel`), so a cubic Hermite
//! interpolant between two adjacent snapshots matches position and velocity at
//! both ends — the in-betweens follow the local dynamics instead of straight
//! chords. This is a **view-side** concern (it manufactures frames, not physics),
//! so it lives here in renderprep, not in core.
//!
//! Attribute strategy (the M6c decision): full [`prepare`](crate::prepare) —
//! including the O(N²) k-NN density pass — runs only on the two *endpoint*
//! snapshots; each subframe takes Hermite positions plus linearly interpolated
//! color/brightness/size. Density evolves on the snapshot timescale, and
//! re-running k-NN per subframe would multiply prep cost by the upsampling
//! factor for no visible gain.

use galaxy_core::{DVec3, Species, State};
use glam::Vec3;

use crate::frame::FrameData;

/// Errors from constructing a Hermite span over a snapshot pair.
#[derive(thiserror::Error, Debug)]
pub enum InterpError {
    /// The two snapshots hold different particle counts.
    #[error("snapshot particle counts differ: {n0} vs {n1}")]
    LengthMismatch { n0: usize, n1: usize },
    /// The particle identity streams disagree — interpolating would pair up
    /// unrelated particles and scramble every in-between frame.
    #[error("particle id mismatch at index {index}: {id0} vs {id1} (snapshots are not order-consistent)")]
    IdMismatch { index: usize, id0: u64, id1: u64 },
    /// `s1.time` must be strictly after `s0.time` (the cubic divides by Δt).
    #[error("snapshot times not increasing: t0 = {t0}, t1 = {t1}")]
    NonIncreasingTime { t0: f64, t1: f64 },
}

/// A validated snapshot pair ready for cubic Hermite sampling on `u ∈ [0, 1]`
/// (u = 0 is `s0`, u = 1 is `s1`). Construction checks the defensive gates
/// (matching lengths, identical `id` streams, strictly increasing time) once;
/// sampling is then infallible.
pub struct HermiteSpan<'a> {
    s0: &'a State,
    s1: &'a State,
    dt: f64,
}

impl<'a> HermiteSpan<'a> {
    /// Validate `(s0, s1)` as an interpolation span. See [`InterpError`].
    pub fn new(s0: &'a State, s1: &'a State) -> Result<Self, InterpError> {
        if s0.len() != s1.len() {
            return Err(InterpError::LengthMismatch {
                n0: s0.len(),
                n1: s1.len(),
            });
        }
        // Order stability is *expected* from the in-place integrator, but a silent
        // mismatch would pair unrelated particles — check every id, fail with the
        // first offender.
        for (index, (a, b)) in s0.id.iter().zip(&s1.id).enumerate() {
            if a != b {
                return Err(InterpError::IdMismatch {
                    index,
                    id0: a.0,
                    id1: b.0,
                });
            }
        }
        let dt = s1.time - s0.time;
        // `is_finite` also rejects NaN times, which `dt <= 0.0` alone would let through.
        if !dt.is_finite() || dt <= 0.0 {
            return Err(InterpError::NonIncreasingTime {
                t0: s0.time,
                t1: s1.time,
            });
        }
        Ok(HermiteSpan { s0, s1, dt })
    }

    /// The span's time extent `s1.time - s0.time` (strictly positive).
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Positions and velocities at normalized time `u` (0 ⇒ `s0`, 1 ⇒ `s1`,
    /// both reproduced bit-exact). Velocities are the cubic's analytic
    /// derivative — C¹ at the joins, and the later Doppler-coloring input.
    pub fn sample(&self, u: f64) -> (Vec<DVec3>, Vec<DVec3>) {
        // Hermite basis on [0,1]. These forms hit the endpoint values EXACTLY
        // (h00(0)=1, h01(1)=1, g10(0)=1, g11(1)=1, all others 0), which is what
        // makes u=0/u=1 reproduce the snapshots bit-for-bit: the sums degenerate
        // to `x·1 + 0 + 0 + 0`.
        let u2 = u * u;
        let u3 = u2 * u;
        let h00 = 2.0 * u3 - 3.0 * u2 + 1.0;
        let h10 = u3 - 2.0 * u2 + u;
        let h01 = 3.0 * u2 - 2.0 * u3;
        let h11 = u3 - u2;
        // Basis derivatives d/du; velocity picks up the 1/Δt chain-rule factor on
        // the POSITION terms only (the v0/v1 terms carry Δt·(1/Δt) = 1, kept
        // unscaled so endpoint velocities come back bit-exact, not v·Δt/Δt).
        let g00 = 6.0 * u2 - 6.0 * u;
        let g10 = 3.0 * u2 - 4.0 * u + 1.0;
        let g01 = 6.0 * u - 6.0 * u2;
        let g11 = 3.0 * u2 - 2.0 * u;

        let n = self.s0.len();
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        for i in 0..n {
            let (p0, p1) = (self.s0.pos[i], self.s1.pos[i]);
            let (v0, v1) = (self.s0.vel[i], self.s1.vel[i]);
            pos.push(p0 * h00 + p1 * h01 + (v0 * h10 + v1 * h11) * self.dt);
            vel.push((p0 * g00 + p1 * g01) / self.dt + v0 * g10 + v1 * g11);
        }
        (pos, vel)
    }
}

/// Assemble one in-between frame: Hermite positions from `span` at `u`, visual
/// attributes (color / brightness / size) linearly blended between the two
/// *prepared* endpoint frames. `u = 0` reproduces `f0` bit-exact, `u = 1`
/// reproduces `f1`.
///
/// The frames may be full-length (one splat per span particle) or
/// species-routed (M7d: `prepare`'s default filters `Species::Gas` out of the
/// splat list) — a filtered frame's rows pair with the span's collisionless
/// particles, in order.
///
/// Panics if `f0`/`f1` particle counts match neither the span's length nor its
/// collisionless count — a caller contract violation (the frames must be
/// `prepare`d from the span's own endpoint snapshots), not a data condition.
pub fn subframe(span: &HermiteSpan, f0: &FrameData, f1: &FrameData, u: f64) -> FrameData {
    let n = span.s0.len();
    let n_star = span
        .s0
        .kind
        .iter()
        .filter(|k| **k == Species::Collisionless)
        .count();
    let filtered = f0.len() == n_star && n_star != n;
    if !filtered {
        assert_eq!(f0.len(), n, "f0 is not a prepared frame of the span's s0");
    }
    assert_eq!(
        f1.len(),
        f0.len(),
        "f1 is not a prepared frame of the span's s1"
    );

    let (hpos, _) = span.sample(u);
    let pos: Vec<Vec3> = if filtered {
        hpos.iter()
            .zip(&span.s0.kind)
            .filter(|(_, k)| **k == Species::Collisionless)
            .map(|(p, _)| p.as_vec3())
            .collect()
    } else {
        hpos.iter().map(|p| p.as_vec3()).collect()
    };

    // Two-product lerp (1-w)·a + w·b, NOT a + w·(b-a): the former is bit-exact at
    // BOTH endpoints (w=0 ⇒ 1·a + 0·b, w=1 ⇒ 0·a + 1·b), which is what lets
    // u=0/u=1 reproduce the prepared frames exactly.
    let w1 = u as f32;
    let w0 = 1.0 - w1;
    let lerp = |a: f32, b: f32| w0 * a + w1 * b;

    let color = f0
        .color
        .iter()
        .zip(&f1.color)
        .map(|(c0, c1)| [lerp(c0[0], c1[0]), lerp(c0[1], c1[1]), lerp(c0[2], c1[2])])
        .collect();
    let brightness = f0
        .brightness
        .iter()
        .zip(&f1.brightness)
        .map(|(&a, &b)| lerp(a, b))
        .collect();
    let size = f0
        .size
        .iter()
        .zip(&f1.size)
        .map(|(&a, &b)| lerp(a, b))
        .collect();

    FrameData {
        pos,
        color,
        brightness,
        size,
    }
}
