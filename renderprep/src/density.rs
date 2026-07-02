//! Local **number-density** estimation via the k-th nearest neighbour, and the
//! non-dimming brightness boost it drives (DESIGN.md M3: density-aware coloring,
//! the deferred pass beyond the pure progenitor map).
//!
//! The estimator is the standard k-NN density: a particle sitting in a region
//! where its k-th nearest neighbour is close is in a dense neighbourhood. In 3-D,
//!
//! ```text
//!   ρ_i = k / ( (4/3) π d_{k,i}³ )
//! ```
//!
//! where `d_{k,i}` is the distance from particle `i` to its k-th nearest neighbour
//! (**self excluded**). This is a *number* density (particles per unit volume); for
//! the equal-mass disk it is proportional to mass density, and mass-weighting is a
//! documented later refinement. The k-th NN distance is floored at a `softening`
//! length **before** cubing, so coincident / near-coincident particles (collision
//! cores, the Plummer centre) yield a finite large density rather than `+∞ → NaN`
//! that would poison the whole frame.
//!
//! `knn_density` is a brute-force **O(N²)** reference — the oracle. A grid/tree
//! acceleration is the deferred follow-up, to be gated bit-for-bit against this,
//! exactly as the GPU solvers are gated against their CPU references.
//!
//! The brightness mapping ([`density_boost`]) is deliberately **non-dimming**: it
//! only ever *brightens* overdense regions (cores, tidal bridges) and leaves
//! underdense regions (the diffuse tidal tails — the feature of interest) at full
//! brightness. A naive "denser → brighter, sparser → dimmer" power law would darken
//! the very streams the render is meant to reveal, because the halo dominates the
//! density field.

use galaxy_core::DVec3;

use std::f64::consts::PI;

/// Density-aware brightness modulation for [`crate::prepare`]. Denser regions are
/// brightened by up to a factor `1 + strength`; underdense regions are left exactly
/// at full brightness (the boost never dims). Off by default
/// (`PrepConfig.density == None`), in which case `prepare` is a bit-for-bit pure map.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DensityColoring {
    /// Neighbour count for the k-th-NN density estimate (a typical choice is 8–32:
    /// small enough to resolve local structure, large enough to smooth shot noise).
    pub k: usize,
    /// Floor on the k-th-NN distance (length units) applied **before** cubing. Guards
    /// coincident particles against an infinite density; must be `> 0` for that guard
    /// to bite. Pick it a touch below the smallest resolved separation.
    pub softening: f64,
    /// Boost saturation: the densest regions are brightened by up to `1 + strength`.
    /// `0.0` is the identity (the frame is unchanged — equivalent to `density: None`).
    pub strength: f32,
}

/// k-th nearest-neighbour local **number density** for every particle:
/// `ρ_i = k / ((4/3)π d_{k,i}³)`, with `d_{k,i}` the distance from `i` to its k-th
/// nearest neighbour (self excluded) floored at `softening` before cubing.
///
/// Brute-force O(N²) — the reference oracle. Degenerate inputs are defined, not
/// panics: with `N ≤ k` (or `k == 0`) there is no k-th neighbour, so every density
/// is `0.0` — a sentinel the boost maps to "no brightening".
pub fn knn_density(positions: &[DVec3], k: usize, softening: f64) -> Vec<f64> {
    todo!("k-th NN number density (self-excluded, softened); reference O(N^2)")
}

/// Per-particle brightness multiplier from local density — **mean-referenced and
/// non-dimming**:
///
/// ```text
///   boost_i = 1 + strength · ( 1 − ρ_ref / max(ρ_i, ρ_ref) )
/// ```
///
/// where `ρ_ref` is the mean over the *positive* densities (a `0.0` estimate means
/// "no neighbourhood", not a real void, so it is excluded from the reference and
/// receives boost `1`). The result is bounded in `[1, 1 + strength]`, monotone
/// non-decreasing in `ρ_i`, and exactly `1` wherever `ρ_i ≤ ρ_ref` — so the tidal
/// tails keep full brightness while cores and bridges glow. Returns all-`1.0`
/// (identity) when `strength == 0` or no density is positive.
pub fn density_boost(density: &[f64], strength: f32) -> Vec<f32> {
    todo!("non-dimming mean-referenced brightness boost")
}
