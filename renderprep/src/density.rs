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

/// Density-driven splat sizing for [`crate::prepare`] (M6e "size-by-density"):
/// dense regions get tight small splats, diffuse regions soft large ones, via
/// `size = base · clamp((ρ_ref/ρ)^{1/3}, min_frac, max_frac)` — the inverse
/// cube-root is the natural inter-particle-spacing law (an SPH-style adaptive
/// smoothing length). Carries its own kNN parameterization so `prepare` can share
/// one O(N²) pass between all consumers that agree on `(k, softening)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SizeByDensity {
    /// Neighbour count for the k-th-NN density estimate (see [`DensityColoring::k`]).
    pub k: usize,
    /// kNN distance floor (see [`DensityColoring::softening`]).
    pub softening: f64,
    /// Lower clamp on the size fraction (densest splats), `0 < min_frac ≤ 1`.
    pub min_frac: f32,
    /// Upper clamp on the size fraction (sparsest splats), `max_frac ≥ 1`.
    pub max_frac: f32,
}

/// k-th nearest-neighbour local **number density** for every particle:
/// `ρ_i = k / ((4/3)π d_{k,i}³)`, with `d_{k,i}` the distance from `i` to its k-th
/// nearest neighbour (self excluded) floored at `softening` before cubing.
///
/// Brute-force O(N²) — the reference oracle. Degenerate inputs are defined, not
/// panics: with `N ≤ k` (or `k == 0`) there is no k-th neighbour, so every density
/// is `0.0` — a sentinel the boost maps to "no brightening".
pub fn knn_density(positions: &[DVec3], k: usize, softening: f64) -> Vec<f64> {
    let n = positions.len();
    // A k-th neighbour needs k *other* particles; with N ≤ k (or k == 0) there is
    // none, so the density is the defined 0.0 sentinel (the boost reads it as "no
    // neighbourhood" → no brightening).
    if k == 0 || n <= k {
        return vec![0.0; n];
    }
    // ρ = k / ((4/3)π d³) = k · (3 / 4π) / d³ — fold the constant once.
    let coeff = k as f64 * 3.0 / (4.0 * PI);
    (0..n)
        .map(|i| {
            let pi = positions[i];
            // Squared distances to every *other* particle (self excluded).
            let mut d2: Vec<f64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| pi.distance_squared(positions[j]))
                .collect();
            // Only the k-th order statistic is needed — partial select, not a full
            // sort. `d2` has n-1 ≥ k entries, so index k-1 is in range. `total_cmp`
            // gives a total order (distances are finite non-negative, no NaN).
            let (_, kth, _) = d2.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
            let d_k = kth.sqrt().max(softening); // floor before cubing
            coeff / (d_k * d_k * d_k)
        })
        .collect()
}

/// k-NN **neighbourhood**: the densities of [`knn_density`] (bit-for-bit — gated)
/// *plus* the k nearest-neighbour **indices** per particle (self excluded,
/// unordered; any valid set on exact distance ties). The indices feed the
/// velocity-dispersion coloring (M6e), which needs the neighbour *members*, not
/// just the k-th distance.
///
/// Degenerate inputs mirror [`knn_density`]: with `N ≤ k` (or `k == 0`) every
/// density is the `0.0` sentinel and every neighbour list is empty.
pub fn knn_neighbourhood(
    positions: &[DVec3],
    k: usize,
    softening: f64,
) -> (Vec<f64>, Vec<Vec<usize>>) {
    let n = positions.len();
    if k == 0 || n <= k {
        return (vec![0.0; n], vec![Vec::new(); n]);
    }
    let coeff = k as f64 * 3.0 / (4.0 * PI);
    let mut density = Vec::with_capacity(n);
    let mut neighbours = Vec::with_capacity(n);
    for i in 0..n {
        let pi = positions[i];
        // Squared distance + index, self excluded — the same candidates as
        // knn_density, just carrying identity alongside.
        let mut d2: Vec<(f64, usize)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| (pi.distance_squared(positions[j]), j))
            .collect();
        // Partial select on the k-th order statistic: the left partition plus the
        // pivot are exactly the k nearest (unordered; any valid set on ties). The
        // pivot's distance is the same d_k value knn_density computes, and the
        // density arithmetic below is kept operation-for-operation identical so
        // the two estimators agree bit-for-bit (gated).
        let (left, kth, _) = d2.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
        let d_k = kth.0.sqrt().max(softening); // floor before cubing
        let mut nbrs: Vec<usize> = left.iter().map(|&(_, j)| j).collect();
        nbrs.push(kth.1);
        density.push(coeff / (d_k * d_k * d_k));
        neighbours.push(nbrs);
    }
    (density, neighbours)
}

/// Local velocity dispersion `σ_v` per particle over its neighbourhood set
/// `{self} ∪ neighbours[i]`: the root-mean-square deviation from the
/// neighbourhood's mean velocity,
///
/// ```text
///   σ_i = sqrt( (1/m) Σ_{j ∈ members} |v_j − v̄|² ),   v̄ = (1/m) Σ v_j
/// ```
///
/// An empty neighbour list (the degenerate-kNN sentinel) yields `σ = 0.0` —
/// "no neighbourhood", which the color ramp maps to the cold end.
pub fn velocity_dispersion(vel: &[DVec3], neighbours: &[Vec<usize>]) -> Vec<f64> {
    neighbours
        .iter()
        .enumerate()
        .map(|(i, nbrs)| {
            if nbrs.is_empty() {
                return 0.0; // degenerate kNN — "no neighbourhood"
            }
            let m = (nbrs.len() + 1) as f64; // members: self + the k neighbours
            let sum = nbrs.iter().fold(vel[i], |s, &j| s + vel[j]);
            let mean = sum / m;
            let var = nbrs.iter().fold(vel[i].distance_squared(mean), |s, &j| {
                s + vel[j].distance_squared(mean)
            }) / m;
            var.sqrt()
        })
        .collect()
}

/// Per-particle splat sizes from local density — the [`SizeByDensity`] map:
/// `base · clamp((ρ_ref/ρ_i)^{1/3}, min_frac, max_frac)`, with `ρ_ref` the mean
/// over the *positive* densities (the same reference discipline as
/// [`density_boost`]). `ρ_i = ρ_ref` gives exactly `base`; the `0.0` sentinel
/// ("no neighbourhood") gets exactly `base` too — sizing only acts where the
/// estimate is real. Returns all-`base` when no density is positive.
///
/// Panics if the clamp band is invalid (`min_frac`/`max_frac` non-finite,
/// `min_frac ≤ 0`, or `min_frac > max_frac`) — a config bug, not a data condition.
pub fn density_sizes(density: &[f64], base: f32, min_frac: f32, max_frac: f32) -> Vec<f32> {
    assert!(
        min_frac.is_finite() && max_frac.is_finite() && min_frac > 0.0 && min_frac <= max_frac,
        "invalid size clamp band [{min_frac}, {max_frac}]"
    );
    // Reference density: the mean over the positive estimates, exactly as in
    // density_boost — a 0.0 is the "no neighbourhood" sentinel and gets base size.
    let (sum, count) = density
        .iter()
        .filter(|&&d| d > 0.0)
        .fold((0.0, 0usize), |(s, c), &d| (s + d, c + 1));
    if count == 0 {
        return vec![base; density.len()];
    }
    let rho_ref = sum / count as f64;
    density
        .iter()
        .map(|&d| {
            if d > 0.0 {
                // (ρ_ref/ρ)^⅓ is the inter-particle-spacing ratio: exactly 1 at the
                // reference (cbrt(1) = 1, so ρ = ρ_ref sizes exactly `base`).
                let frac = ((rho_ref / d).cbrt() as f32).clamp(min_frac, max_frac);
                base * frac
            } else {
                base // sentinel: no estimate, no resizing
            }
        })
        .collect()
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
    if strength == 0.0 {
        return vec![1.0; density.len()];
    }
    // Reference density: the mean over the *positive* estimates. A 0.0 is a "no
    // neighbourhood" sentinel (degenerate / boundary), not a real void, so it is
    // excluded from the reference and receives boost 1 below.
    let (sum, count) = density
        .iter()
        .filter(|&&d| d > 0.0)
        .fold((0.0, 0usize), |(s, c), &d| (s + d, c + 1));
    if count == 0 {
        return vec![1.0; density.len()];
    }
    let rho_ref = sum / count as f64;
    density
        .iter()
        .map(|&d| {
            // 1 + strength·(1 − ρ_ref / max(d, ρ_ref)): exactly 1 for d ≤ ρ_ref
            // (including the d = 0 sentinel), rising monotonically toward 1+strength
            // as d → ∞. Bounded, and never below 1 — the boost never dims.
            let frac = 1.0 - rho_ref / d.max(rho_ref);
            1.0 + strength * frac as f32
        })
        .collect()
}
