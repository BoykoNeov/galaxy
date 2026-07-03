//! k-NN local **number-density** estimator and its non-dimming brightness boost
//! (DESIGN.md M3 density-aware coloring).
//!
//! Expectations are hand-derived, never read back from the function under test:
//! for a known geometry the k-th nearest-neighbour distance `d_k` is exact, so
//! `ρ = k / ((4/3)π d_k³)` is a closed-form oracle. The estimator is the reference
//! (brute-force O(N²)); a spatial acceleration is the deferred follow-up, gated
//! against these same values.

use galaxy_core::DVec3;
use galaxy_renderprep::{
    density_boost, density_sizes, knn_density, knn_neighbourhood, velocity_dispersion,
};

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

// --------------------------------------------------------------------------
// k-NN neighbourhood (M6e): densities bit-identical to knn_density + indices
// --------------------------------------------------------------------------

/// A small irregular 3-D cloud (no symmetries, no distance ties).
fn cloud(n: usize) -> Vec<DVec3> {
    (0..n)
        .map(|i| {
            let x = i as f64;
            DVec3::new(
                (x * 0.7).sin() * 3.0 + x * 0.11,
                (x * 1.3).cos() * 2.0,
                (x * 0.4).sin() - x * 0.07,
            )
        })
        .collect()
}

#[test]
fn neighbourhood_densities_match_knn_density_bit_for_bit() {
    // The whole point of extending rather than replacing the estimator: the
    // density outputs must be EXACTLY today's, for every (k, softening) probed.
    let pos = cloud(12);
    for k in [1, 3, 5] {
        for softening in [0.0, 0.3] {
            let (density, _) = knn_neighbourhood(&pos, k, softening);
            assert_eq!(density, knn_density(&pos, k, softening), "k={k}");
        }
    }
}

#[test]
fn neighbourhood_indices_hand_oracle() {
    // Line at x = 0, 1, 3, 7, 12 (all pairwise gaps distinct — no ties). k = 2
    // nearest (self excluded), hand-derived per particle.
    let pos: Vec<DVec3> = [0.0, 1.0, 3.0, 7.0, 12.0]
        .iter()
        .map(|&x| DVec3::new(x, 0.0, 0.0))
        .collect();
    let (_, neighbours) = knn_neighbourhood(&pos, 2, 0.0);
    let mut sorted: Vec<Vec<usize>> = neighbours;
    for n in &mut sorted {
        n.sort_unstable();
    }
    assert_eq!(
        sorted,
        vec![
            vec![1, 2], // x=0:  dists 1, 3, 7, 12
            vec![0, 2], // x=1:  dists 1, 2, 6, 11
            vec![0, 1], // x=3:  dists 3, 2, 4, 9
            vec![2, 4], // x=7:  dists 7, 6, 4, 5
            vec![2, 3], // x=12: dists 12, 11, 9, 5
        ]
    );
}

#[test]
fn neighbourhood_kth_distance_is_consistent_with_the_density() {
    // Structural invariant tying the two outputs together: for every particle the
    // FARTHEST returned neighbour must sit at exactly the d_k the density encodes,
    // i.e. ρ = k/((4/3)π max(d_max, softening)³); the list holds k distinct
    // others, never self.
    let pos = cloud(15);
    let k = 4;
    for softening in [0.0, 0.5] {
        let (density, neighbours) = knn_neighbourhood(&pos, k, softening);
        for i in 0..pos.len() {
            let nbrs = &neighbours[i];
            assert_eq!(nbrs.len(), k, "particle {i}: wrong neighbour count");
            assert!(!nbrs.contains(&i), "particle {i}: self in neighbours");
            let mut dedup = nbrs.clone();
            dedup.sort_unstable();
            dedup.dedup();
            assert_eq!(dedup.len(), k, "particle {i}: duplicate neighbours");
            let d_max = nbrs
                .iter()
                .map(|&j| pos[i].distance(pos[j]))
                .fold(0.0f64, f64::max);
            close(density[i], rho(k, d_max.max(softening)), 1e-12);
        }
    }
}

#[test]
fn neighbourhood_degenerate_inputs_yield_zeros_and_empty_lists() {
    // Mirrors knn_density's sentinel rules: N ≤ k or k = 0 → density 0.0 and no
    // neighbour list to act on.
    let two = [DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)];
    let (d, n) = knn_neighbourhood(&two, 5, 1e-6);
    assert_eq!(d, vec![0.0, 0.0]);
    assert_eq!(n, vec![Vec::<usize>::new(), Vec::new()]);
    let (d, n) = knn_neighbourhood(&two, 0, 1e-6);
    assert_eq!(d, vec![0.0, 0.0]);
    assert_eq!(n, vec![Vec::<usize>::new(), Vec::new()]);
    let (d, n) = knn_neighbourhood(&[], 3, 1e-6);
    assert!(d.is_empty() && n.is_empty());
}

#[test]
fn neighbourhood_is_deterministic() {
    let pos = cloud(20);
    assert_eq!(
        knn_neighbourhood(&pos, 4, 1e-6),
        knn_neighbourhood(&pos, 4, 1e-6)
    );
}

// --------------------------------------------------------------------------
// Velocity dispersion over kNN neighbourhoods (M6e)
// --------------------------------------------------------------------------

#[test]
fn dispersion_pair_hand_value() {
    // Two mutual neighbours with velocities 0 and 2x̂: the member set is {v0, v1}
    // for both, mean = x̂, deviations = 1 each → σ = 1.
    let vel = [DVec3::ZERO, DVec3::new(2.0, 0.0, 0.0)];
    let neighbours = vec![vec![1], vec![0]];
    let sigma = velocity_dispersion(&vel, &neighbours);
    close(sigma[0], 1.0, 1e-12);
    close(sigma[1], 1.0, 1e-12);
}

#[test]
fn dispersion_of_a_cold_population_is_exactly_zero() {
    // Identical velocities → zero deviation, exactly (components chosen so the
    // 3-member mean is fp-exact).
    let v = DVec3::new(1.5, -2.0, 0.25);
    let vel = [v, v, v];
    let neighbours = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    assert_eq!(velocity_dispersion(&vel, &neighbours), vec![0.0, 0.0, 0.0]);
}

#[test]
fn dispersion_lattice_hand_value() {
    // Velocities 0, 1, 2 (x̂), every neighbourhood = all three: mean 1,
    // σ² = (1 + 0 + 1)/3 = 2/3 → σ = √(2/3).
    let vel = [
        DVec3::ZERO,
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.0),
    ];
    let neighbours = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    let sigma = velocity_dispersion(&vel, &neighbours);
    let expect = (2.0f64 / 3.0).sqrt();
    for s in sigma {
        close(s, expect, 1e-12);
    }
}

#[test]
fn dispersion_separates_two_populations() {
    // The M6e two-population oracle: a cold pair (equal velocities) and a hot pair
    // (opposite ±3ŷ), neighbourhoods internal to each pair → σ = [0, 0, 3, 3].
    let vel = [
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 3.0, 0.0),
        DVec3::new(0.0, -3.0, 0.0),
    ];
    let neighbours = vec![vec![1], vec![0], vec![3], vec![2]];
    let sigma = velocity_dispersion(&vel, &neighbours);
    assert_eq!(sigma[0], 0.0);
    assert_eq!(sigma[1], 0.0);
    close(sigma[2], 3.0, 1e-12);
    close(sigma[3], 3.0, 1e-12);
}

#[test]
fn dispersion_empty_neighbourhood_is_the_zero_sentinel() {
    // Degenerate kNN (no neighbours) → σ = 0.0, defined not panicking.
    let vel = [DVec3::new(5.0, 0.0, 0.0)];
    assert_eq!(velocity_dispersion(&vel, &[vec![]]), vec![0.0]);
}

#[test]
fn dispersion_is_bulk_velocity_invariant() {
    // Adding a constant bulk velocity must not change σ (mean-subtracted): the
    // physics gate — a fast-moving cold stream is still cold.
    let vel: Vec<DVec3> = (0..6)
        .map(|i| {
            let x = i as f64;
            DVec3::new((x * 0.9).sin(), (x * 1.7).cos(), x * 0.3)
        })
        .collect();
    let neighbours: Vec<Vec<usize>> = (0..6)
        .map(|i| (0..6).filter(|&j| j != i).take(3).collect())
        .collect();
    let bulk = DVec3::new(10.0, -5.0, 2.0);
    let shifted: Vec<DVec3> = vel.iter().map(|v| *v + bulk).collect();
    let a = velocity_dispersion(&vel, &neighbours);
    let b = velocity_dispersion(&shifted, &neighbours);
    for (x, y) in a.iter().zip(&b) {
        close(*x, *y, 1e-9);
    }
}

#[test]
fn dispersion_is_deterministic() {
    let vel = [
        DVec3::new(0.3, 1.0, -0.2),
        DVec3::new(-1.1, 0.4, 0.9),
        DVec3::new(2.0, -0.6, 0.1),
    ];
    let neighbours = vec![vec![1, 2], vec![0, 2], vec![0, 1]];
    assert_eq!(
        velocity_dispersion(&vel, &neighbours),
        velocity_dispersion(&vel, &neighbours)
    );
}

// --------------------------------------------------------------------------
// Size-by-density (M6e): base · clamp((ρ_ref/ρ)^⅓, min, max)
// --------------------------------------------------------------------------

#[test]
fn sizes_hand_values() {
    // Densities [4, 4, 16, 40]: ρ_ref = mean = 16. Fractions are (16/ρ)^⅓ —
    // hand-derived: (4)^⅓ = 1.5874…, 1 exactly at ρ = ρ_ref, (0.4)^⅓ = 0.7368… —
    // none clamped by the [0.5, 2] band. Base 2.
    let sizes = density_sizes(&[4.0, 4.0, 16.0, 40.0], 2.0, 0.5, 2.0);
    let close32 = |a: f32, b: f64| assert!((f64::from(a) - b).abs() < 1e-6, "{a} vs {b}");
    close32(sizes[0], 2.0 * 4.0f64.cbrt());
    close32(sizes[1], 2.0 * 4.0f64.cbrt());
    assert_eq!(sizes[2], 2.0, "ρ = ρ_ref must give exactly the base size");
    close32(sizes[3], 2.0 * 0.4f64.cbrt());
}

#[test]
fn sizes_are_clamped_to_the_band() {
    // Extreme under/over-densities pin the clamp ends exactly.
    let sizes = density_sizes(&[1e-30, 1.0, 1e30], 1.0, 0.8, 1.5);
    assert_eq!(sizes[0], 1.5, "sparse extreme → max_frac·base");
    assert_eq!(sizes[1], 1.5, "well under ρ_ref → still max-clamped here");
    assert_eq!(sizes[2], 0.8, "dense extreme → min_frac·base");
}

#[test]
fn sizes_zero_sentinel_gets_exactly_the_base() {
    // 0.0 densities are "no neighbourhood", not voids: they keep the base size and
    // are excluded from the reference (ρ_ref = mean(2, 8) = 5 here).
    let base = 1.25;
    let sizes = density_sizes(&[0.0, 2.0, 8.0], base, 0.5, 2.0);
    assert_eq!(sizes[0], base);
    assert!(sizes[1] > base, "underdense → softer/larger splat");
    assert!(sizes[2] < base, "overdense → tighter/smaller splat");
    // All-zero input: no reference → everyone at base.
    assert_eq!(density_sizes(&[0.0, 0.0], base, 0.5, 2.0), vec![base, base]);
}

#[test]
fn sizes_are_monotone_nonincreasing_in_density() {
    let sizes = density_sizes(&[0.5, 1.0, 2.0, 4.0, 8.0, 64.0], 1.0, 0.25, 4.0);
    for w in sizes.windows(2) {
        assert!(w[1] <= w[0], "denser must never splat larger: {sizes:?}");
    }
}

#[test]
#[should_panic]
fn sizes_reject_an_inverted_clamp_band() {
    // min_frac > max_frac is a config bug — fail fast, not silent nonsense.
    density_sizes(&[1.0, 2.0], 1.0, 2.0, 0.5);
}

#[test]
fn sizes_are_deterministic() {
    let d = [0.7, 3.0, 0.0, 12.5];
    assert_eq!(
        density_sizes(&d, 1.5, 0.5, 2.0),
        density_sizes(&d, 1.5, 0.5, 2.0)
    );
}
