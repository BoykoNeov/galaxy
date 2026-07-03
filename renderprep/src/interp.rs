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

use galaxy_core::{DVec3, State};
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
        let _ = (s0, s1);
        todo!("M6c red phase")
    }

    /// The span's time extent `s1.time - s0.time` (strictly positive).
    pub fn dt(&self) -> f64 {
        self.dt
    }

    /// Positions and velocities at normalized time `u` (0 ⇒ `s0`, 1 ⇒ `s1`,
    /// both reproduced bit-exact). Velocities are the cubic's analytic
    /// derivative — C¹ at the joins, and the later Doppler-coloring input.
    pub fn sample(&self, u: f64) -> (Vec<DVec3>, Vec<DVec3>) {
        let _ = u;
        todo!("M6c red phase")
    }
}

/// Assemble one in-between frame: Hermite positions from `span` at `u`, visual
/// attributes (color / brightness / size) linearly blended between the two
/// *prepared* endpoint frames. `u = 0` reproduces `f0` bit-exact, `u = 1`
/// reproduces `f1`.
///
/// Panics if `f0`/`f1` particle counts disagree with the span — that is a
/// caller contract violation (the frames must be `prepare`d from the span's own
/// endpoint snapshots), not a data condition.
pub fn subframe(span: &HermiteSpan, f0: &FrameData, f1: &FrameData, u: f64) -> FrameData {
    let _ = (span, f0, f1, u, Vec3::ZERO);
    todo!("M6c red phase")
}
