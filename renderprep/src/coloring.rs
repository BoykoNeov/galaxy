//! Coloring modes v2 (DESIGN.md M6e): pure color maps that diversify what the
//! movie's colors *mean*, beyond the flat progenitor palette.
//!
//! Three maps live here, all pure functions gated by exact/invariant tests:
//!
//! * [`initial_radius_colors`] — the **frozen-at-t=0 initial-radius ramp**: each
//!   particle is colored once, from its radius about its progenitor's mass-weighted
//!   COM in the *initial* snapshot, and keeps that color for the whole movie (keyed
//!   by particle index). Tidal tails then carry a visible gradient showing where
//!   their material came from.
//! * [`dispersion_colors`] — a σ_v ramp: dynamically cold material toward one color,
//!   dynamically hot toward another (the deferred refinement named in `prepare`).
//! * [`compression_colors`] — the **star-formation proxy**: a hue shift from the
//!   base color toward a "young population" blue-white, keyed on density
//!   *compression* `ρ(t)/ρ(0)` rather than absolute ρ, so only tidally-compressed
//!   material (bridges, shocked overlap, ring waves) lights up while pre-existing
//!   dense cores keep their old-population color. Honest caveat (DESIGN M6e): the
//!   sim is collisionless — stellar density stands in for gas compression; this is
//!   a standard visualization proxy, not physics.
//!
//! All ramps use the two-product lerp `(1−t)·a + t·b`, which is bit-exact at both
//! endpoints — the same discipline as `interp::subframe` — so "exactly the base
//! color when the effect is off" is a hard guarantee, not a tolerance.

use galaxy_core::State;

/// Per-progenitor `(inner, outer)` color ramp for [`initial_radius_colors`].
/// A progenitor id out of range wraps modulo the ramp count; an empty ramp list
/// falls back to white — exactly the palette conventions of `prepare`.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialRamp {
    /// `(inner, outer)` linear-RGB endpoints, indexed by `progenitor % len`.
    pub ramps: Vec<([f32; 3], [f32; 3])>,
}

/// Frozen-at-t=0 initial-radius colors, one per particle (keyed by index).
///
/// Per progenitor: the mass-weighted COM and the **half-mass radius** `r_half`
/// (the smallest particle radius enclosing ≥ half the progenitor's mass) are
/// computed from `state`; each particle maps through `t = r / (r + r_half)` —
/// monotone, bounded in `[0, 1)`, exactly `0` at the COM and exactly `½` at
/// `r = r_half`, so the *median-mass* particle sits at the ramp midpoint and the
/// normalization is per-progenitor scale-free. A degenerate progenitor
/// (`r_half = 0`: single particle, or all coincident) gets the inner color.
pub fn initial_radius_colors(state: &State, ramp: &RadialRamp) -> Vec<[f32; 3]> {
    let _ = (state, ramp);
    todo!("M6e: initial-radius ramp colors")
}

/// σ_v color ramp: `t = σ / (σ + σ_ref)` from `cold` toward `hot`, with `σ_ref`
/// the mean over the *positive* dispersions. Monotone, bounded in `[0, 1)`,
/// exactly `cold` at `σ = 0` (including the degenerate-neighbourhood sentinel)
/// and exactly the midpoint mix at `σ = σ_ref`. All-zero dispersions (no
/// reference) yield all-`cold`.
pub fn dispersion_colors(sigma: &[f64], cold: [f32; 3], hot: [f32; 3]) -> Vec<[f32; 3]> {
    let _ = (sigma, cold, hot);
    todo!("M6e: velocity-dispersion ramp colors")
}

/// Star-formation-proxy hue shift (compression-triggered): lerp each base color
/// toward `young` by `t = clamp(strength, 0, 1) · (1 − ρ0/max(ρ, ρ0))`.
///
/// The trigger is the density **increment** over the same particle's t=0
/// neighbourhood: `ρ ≤ ρ0` (uncompressed or rarefied) keeps the base color
/// *bit-exactly*, `ρ → ∞` saturates at `strength` of the way to `young` —
/// bounded on the `[base, young]` segment and monotone in `ρ`. The `0.0`
/// density sentinel on either side means "no neighbourhood", not a real void,
/// and keeps the base color. `strength = 0` is the identity.
///
/// Panics if the column lengths disagree — a caller contract violation
/// (`rho`/`rho0` must be estimates for these same particles), not a data
/// condition.
pub fn compression_colors(
    base: &[[f32; 3]],
    rho: &[f64],
    rho0: &[f64],
    young: [f32; 3],
    strength: f32,
) -> Vec<[f32; 3]> {
    let _ = (base, rho, rho0, young, strength);
    todo!("M6e: compression-triggered star-formation hue")
}
