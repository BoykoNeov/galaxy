//! SPH density gates (DESIGN.md M7a): the grid-accelerated summation is
//! bit-exact against the O(N²) oracle (same ascending gather order), a uniform
//! lattice recovers the analytic density, adaptive h hits the target neighbor
//! count and obeys the scaling law, and the whole pipeline is deterministic
//! with parallel ≡ serial bit-exactness.

use galaxy_core::DVec3;
use galaxy_solvers::sph::{
    density_adaptive, density_adaptive_serial, density_fixed, reference_density, w,
    DensityConfig, SUPPORT,
};

const PI: f64 = std::f64::consts::PI;

fn random_points(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| DVec3::new(next(), next(), next()) * scale)
        .collect()
}

/// The kernel-weighted neighbor count the adaptive solve targets, recomputed
/// independently by brute force: N_i(h) = (4π/3)·(2h)³·Σ_j W(|x_i−x_j|, h).
fn weighted_count(pos: &[DVec3], i: usize, h: f64) -> f64 {
    let sum: f64 = pos.iter().map(|&p| w((pos[i] - p).length(), h)).sum();
    (4.0 * PI / 3.0) * (SUPPORT * h).powi(3) * sum
}

#[test]
fn fixed_h_density_is_bit_exact_against_the_oracle() {
    // Same neighbor sets, same ascending gather order ⇒ the sums must associate
    // identically: exact f64 equality, not a tolerance.
    let pos = random_points(11, 400, 4.0);
    let mass: Vec<f64> = (0..pos.len()).map(|i| 0.5 + (i % 7) as f64 * 0.1).collect();
    let h = vec![0.35; pos.len()];
    assert_eq!(
        density_fixed(&pos, &mass, &h),
        reference_density(&pos, &mass, &h),
        "grid-accelerated density must be bit-exact vs brute force"
    );
}

#[test]
fn uniform_lattice_recovers_analytic_density() {
    // A cubic lattice of spacing s and particle mass m has ρ = m/s³ exactly.
    // The kernel lattice-sum approximates it with an error set by h/s: at
    // h = 1.25s the M4 cubic spline's lattice quadrature error is well under a
    // percent (the kernel is C¹ and the sum is a 4th-order-accurate midpoint
    // rule on it); 2% gives headroom without letting a wrong normalization
    // (π factors, support convention) through — those miss by ≥ 25%.
    let (nx, s, m) = (10usize, 1.0f64, 2.5f64);
    let mut pos = Vec::new();
    for x in 0..nx {
        for y in 0..nx {
            for z in 0..nx {
                pos.push(DVec3::new(x as f64, y as f64, z as f64) * s);
            }
        }
    }
    let h = 1.25 * s;
    let mass = vec![m; pos.len()];
    let rho = density_fixed(&pos, &mass, &vec![h; pos.len()]);

    let expect = m / (s * s * s);
    let margin = SUPPORT * h; // interior = farther than the support from every face
    let hi = (nx - 1) as f64 * s;
    let mut checked = 0;
    for (i, p) in pos.iter().enumerate() {
        let interior = [p.x, p.y, p.z]
            .iter()
            .all(|&c| c > margin && c < hi - margin);
        if interior {
            let rel = (rho[i] - expect).abs() / expect;
            assert!(rel < 0.02, "interior ρ = {} vs analytic {expect}", rho[i]);
            checked += 1;
        }
    }
    assert!(checked > 0, "lattice too small: no interior particles checked");
}

#[test]
fn adaptive_h_recovers_the_target_neighbor_count() {
    // Bisection to h_tol_rel on h ⇒ |N(h) − n_ngb| ≲ 3·n_ngb·h_tol_rel ≈ 0.15
    // (dN/dh ≈ 3N/h near the root). Gate at ±1.0 for slack; clamped particles
    // would miss by tens, so the gate still bites.
    let pos = random_points(21, 2000, 1.0);
    let mass = vec![1.0; pos.len()];
    let cfg = DensityConfig::default();
    let result = density_adaptive(&pos, &mass, &cfg, None);

    assert_eq!(result.h.len(), pos.len());
    assert_eq!(result.rho.len(), pos.len());
    for i in (0..pos.len()).step_by(97) {
        let n = weighted_count(&pos, i, result.h[i]);
        assert!(
            (n - cfg.n_ngb).abs() <= 1.0,
            "particle {i}: N(h) = {n}, want {} ± 1", cfg.n_ngb
        );
        assert!(result.rho[i] > 0.0 && result.rho[i].is_finite());
    }
}

#[test]
fn adaptive_density_obeys_the_scaling_law() {
    // ρ(λx) = ρ(x)/λ³ with h(λx) = λ·h(x). The bisection tolerance (1e-3 on h)
    // maps to ~3e-3 on ρ; gate at 1% / 0.5% with that justification.
    let pos = random_points(5, 800, 1.0);
    let mass = vec![1.0; pos.len()];
    let cfg = DensityConfig::default();
    let lambda = 4.2;
    let scaled: Vec<DVec3> = pos.iter().map(|&p| p * lambda).collect();

    let base = density_adaptive(&pos, &mass, &cfg, None);
    let big = density_adaptive(&scaled, &mass, &cfg, None);
    for i in (0..pos.len()).step_by(53) {
        let rho_rel = (big.rho[i] - base.rho[i] / lambda.powi(3)).abs()
            / (base.rho[i] / lambda.powi(3));
        let h_rel = (big.h[i] - lambda * base.h[i]).abs() / (lambda * base.h[i]);
        assert!(rho_rel < 1e-2, "ρ scaling broken at {i}: rel {rho_rel}");
        assert!(h_rel < 5e-3, "h scaling broken at {i}: rel {h_rel}");
    }
}

#[test]
fn adaptive_is_deterministic_and_parallel_equals_serial() {
    let pos = random_points(33, 1500, 2.0);
    let mass: Vec<f64> = (0..pos.len()).map(|i| 1.0 + (i % 3) as f64).collect();
    let cfg = DensityConfig::default();

    let a = density_adaptive(&pos, &mass, &cfg, None);
    let b = density_adaptive(&pos, &mass, &cfg, None);
    assert_eq!(a, b, "two identical calls must be bit-identical");

    let s = density_adaptive_serial(&pos, &mass, &cfg, None);
    assert_eq!(a, s, "rayon and serial paths must be bit-identical");
}

#[test]
fn warm_start_cannot_move_the_converged_h() {
    // h_init only seeds the bracket; the root is position-determined. Cold and
    // (deliberately bad) warm starts must agree to the bisection tolerance,
    // and ρ to the ~3× amplified equivalent.
    let pos = random_points(77, 1000, 1.5);
    let mass = vec![0.7; pos.len()];
    let cfg = DensityConfig::default();

    let cold = density_adaptive(&pos, &mass, &cfg, None);
    let bad_seed: Vec<f64> = cold.h.iter().map(|&h| h * 1.5).collect();
    let warm = density_adaptive(&pos, &mass, &cfg, Some(&bad_seed));
    for i in 0..pos.len() {
        let h_rel = (warm.h[i] - cold.h[i]).abs() / cold.h[i];
        assert!(h_rel <= 2.0 * cfg.h_tol_rel, "h moved by {h_rel} at {i}");
        let rho_rel = (warm.rho[i] - cold.rho[i]).abs() / cold.rho[i];
        assert!(rho_rel <= 8.0 * cfg.h_tol_rel, "ρ moved by {rho_rel} at {i}");
    }
}

#[test]
fn underpopulated_systems_clamp_without_panicking() {
    // Below ~5 particles the weighted count's h → ∞ asymptote (32/3·n) sits
    // under the default target: no root exists. The solve must clamp h
    // deterministically and return finite positive values — never panic, hang,
    // or emit NaN. (Physics never runs SPH at n < N_ngb; robustness does.)
    let cfg = DensityConfig::default();
    for n in [1usize, 2, 3] {
        let pos = random_points(n as u64 + 1, n, 1.0);
        let mass = vec![1.0; n];
        let a = density_adaptive(&pos, &mass, &cfg, None);
        let b = density_adaptive(&pos, &mass, &cfg, None);
        assert_eq!(a, b, "clamped solve must stay deterministic (n={n})");
        for i in 0..n {
            assert!(a.h[i].is_finite() && a.h[i] > 0.0, "h[{i}] bad (n={n})");
            assert!(a.rho[i].is_finite() && a.rho[i] > 0.0, "ρ[{i}] bad (n={n})");
        }
    }
}

#[test]
fn empty_input_yields_empty_output() {
    let cfg = DensityConfig::default();
    let out = density_adaptive(&[], &[], &cfg, None);
    assert!(out.rho.is_empty() && out.h.is_empty());
    assert!(density_fixed(&[], &[], &[]).is_empty());
}
