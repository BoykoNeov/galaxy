//! k-NN local **number-density** estimator and its non-dimming brightness boost
//! (DESIGN.md M3 density-aware coloring).
//!
//! Expectations are hand-derived, never read back from the function under test:
//! for a known geometry the k-th nearest-neighbour distance `d_k` is exact, so
//! `ρ = k / ((4/3)π d_k³)` is a closed-form oracle. The estimator is the reference
//! (brute-force O(N²)); a spatial acceleration is the deferred follow-up, gated
//! against these same values.

use galaxy_core::DVec3;
use galaxy_renderprep::{density_boost, knn_density};

use std::f64::consts::PI;

/// The k-NN number-density closed form, for hand oracles.
fn rho(k: usize, d_k: f64) -> f64 {
    k as f64 / ((4.0 / 3.0) * PI * d_k.powi(3))
}

/// Assert two f64 are equal to a relative tolerance (0 handled exactly).
fn close(a: f64, b: f64, rel: f64) {
    let scale = a.abs().max(b.abs()).max(1.0);
    assert!((a - b).abs() <= rel * scale, "{a} vs {b} (rel {rel})");
}

// --------------------------------------------------------------------------
// Estimator: exact hand oracles
// --------------------------------------------------------------------------

#[test]
fn two_particles_k1_is_the_pair_distance() {
    // Each particle's only (k=1) neighbour is the other, at distance 2.
    let pos = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0)];
    let d = knn_density(&pos, 1, 0.0);
    let expect = rho(1, 2.0); // 3 / (32π)
    close(d[0], expect, 1e-12);
    close(d[1], expect, 1e-12);
}

#[test]
fn lattice_kth_neighbour_distance_is_exact() {
    // 1-D unit lattice x_i = i, i = 0..=10. For the interior point at x=5, sorted
    // neighbour distances are 1,1,2,2,3,3,...; the k=3rd is 2, so d_3 = 2.
    let pos: Vec<DVec3> = (0..=10).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
    let d = knn_density(&pos, 3, 0.0);
    close(d[5], rho(3, 2.0), 1e-12);
}

#[test]
fn neighbour_search_excludes_self() {
    // 0, 1, 5 on a line. Particle at x=0, k=1: the nearest *other* is x=1 (distance
    // 1). If self (distance 0) were counted, ρ would blow up — so d[0] == ρ(1, 1).
    let pos = [
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(5.0, 0.0, 0.0),
    ];
    let d = knn_density(&pos, 1, 0.0);
    close(d[0], rho(1, 1.0), 1e-12); // 3 / (4π)
}

#[test]
fn coincident_particles_are_finite_via_softening() {
    // Two particles at the SAME point: raw d_1 = 0. With softening 0.5 the distance
    // is floored to 0.5 before cubing, so ρ = 1/((4/3)π·0.5³) = 6/π — finite.
    let pos = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(0.0, 0.0, 0.0)];
    let d = knn_density(&pos, 1, 0.5);
    let expect = rho(1, 0.5); // 6/π
    assert!(
        d[0].is_finite() && d[1].is_finite(),
        "softening must bound ρ"
    );
    close(d[0], expect, 1e-12);
    close(d[1], expect, 1e-12);
}

#[test]
fn density_obeys_the_inverse_cube_scaling_law() {
    // ρ(λx) = λ⁻³ ρ(x): scaling all positions by λ scales every k-th NN distance by
    // λ, hence every density by λ⁻³. Tolerance, not bit-exact — float cubing.
    let base = [
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.3, -0.4, 0.2),
        DVec3::new(-0.7, 0.9, 1.1),
        DVec3::new(2.1, 0.3, -0.5),
        DVec3::new(-1.2, -1.4, 0.6),
        DVec3::new(0.5, 2.0, -1.3),
    ];
    let lambda = 3.0;
    let scaled: Vec<DVec3> = base.iter().map(|p| *p * lambda).collect();
    let d0 = knn_density(&base, 2, 0.0);
    let d1 = knn_density(&scaled, 2, 0.0);
    for (a, b) in d0.iter().zip(&d1) {
        close(*b, a / lambda.powi(3), 1e-9);
    }
}

// --------------------------------------------------------------------------
// Estimator: degenerate inputs and structural invariants
// --------------------------------------------------------------------------

#[test]
fn fewer_than_k_plus_one_particles_yields_zero() {
    // No k-th neighbour exists → defined sentinel 0.0 (not a panic, not ∞).
    let two = [DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)];
    assert_eq!(knn_density(&two, 5, 1e-6), vec![0.0, 0.0]);
    assert_eq!(knn_density(&[], 3, 1e-6), Vec::<f64>::new());
    assert_eq!(knn_density(&[DVec3::ZERO], 1, 1e-6), vec![0.0]);
    // k == 0 has no meaningful neighbour either.
    let three = [
        DVec3::ZERO,
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.0),
    ];
    assert_eq!(knn_density(&three, 0, 1e-6), vec![0.0, 0.0, 0.0]);
}

#[test]
fn density_is_permutation_equivariant() {
    let pos = [
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(0.1, 0.0, 0.0),
        DVec3::new(0.0, 0.2, 0.0),
        DVec3::new(5.0, 5.0, 5.0),
        DVec3::new(-3.0, 4.0, 0.0),
    ];
    let d = knn_density(&pos, 2, 0.0);
    // Reverse the particle order; densities must reverse identically.
    let mut rev = pos.to_vec();
    rev.reverse();
    let dr = knn_density(&rev, 2, 0.0);
    for i in 0..pos.len() {
        assert_eq!(d[i], dr[pos.len() - 1 - i]);
    }
}

#[test]
fn density_is_deterministic() {
    let pos: Vec<DVec3> = (0..20)
        .map(|i| {
            DVec3::new(
                (i as f64 * 0.7).sin(),
                (i as f64 * 1.1).cos(),
                i as f64 * 0.05,
            )
        })
        .collect();
    assert_eq!(knn_density(&pos, 4, 1e-6), knn_density(&pos, 4, 1e-6));
}

// --------------------------------------------------------------------------
// Brightness boost: non-dimming, bounded, mean-referenced
// --------------------------------------------------------------------------

#[test]
fn boost_hand_values() {
    // density [1, 3], strength 1 → mean ρ_ref = 2.
    //   ρ=1 ≤ ρ_ref → boost 1;  ρ=3 → 1 + (1 − 2/3) = 4/3.
    let b = density_boost(&[1.0, 3.0], 1.0);
    assert_eq!(b[0], 1.0);
    assert!((b[1] - 4.0 / 3.0).abs() < 1e-6, "got {}", b[1]);
}

#[test]
fn boost_never_dims_underdense_particles() {
    // Everything at or below the mean stays exactly at boost 1 (never < 1).
    let b = density_boost(&[0.5, 1.0, 4.0], 0.5);
    assert_eq!(b[0], 1.0);
    assert_eq!(b[1], 1.0);
    assert!(
        b[2] > 1.0,
        "the overdense one should be boosted, got {}",
        b[2]
    );
}

#[test]
fn boost_is_bounded_and_monotone() {
    let density = [0.1, 1.0, 2.0, 10.0, 250.0];
    let strength = 0.7;
    let b = density_boost(&density, strength);
    // Bounded in [1, 1 + strength].
    for g in &b {
        assert!(
            *g >= 1.0 && *g <= 1.0 + strength + 1e-6,
            "out of range: {g}"
        );
    }
    // Ascending density ⇒ non-decreasing boost.
    for w in b.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "boost not monotone: {} then {}",
            w[0],
            w[1]
        );
    }
}

#[test]
fn boost_strength_zero_is_identity() {
    let b = density_boost(&[1.0, 5.0, 0.1], 0.0);
    assert_eq!(b, vec![1.0, 1.0, 1.0]);
}

#[test]
fn boost_all_zero_density_is_identity() {
    // No positive density → no reference → every particle keeps boost 1.
    let b = density_boost(&[0.0, 0.0, 0.0], 1.0);
    assert_eq!(b, vec![1.0, 1.0, 1.0]);
}
