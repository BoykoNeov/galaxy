//! Cubic-spline (M4) kernel gates (DESIGN.md M7a): normalization, hand values,
//! compact support, analytic gradient vs central differences, and the h-scaling
//! law. Expectations are hand-derived from the Monaghan (1992) form, never read
//! back from the function under test.

use galaxy_core::DVec3;
use galaxy_solvers::sph::{grad_w, w, SUPPORT};

const PI: f64 = std::f64::consts::PI;

#[test]
fn support_radius_is_two_h() {
    assert_eq!(SUPPORT, 2.0, "M4 cubic spline has compact support at 2h");
}

#[test]
fn w_at_origin_is_the_hand_value() {
    // W(0, h) = 1/(π h³): the polynomial factor is exactly 1 at q = 0.
    for h in [0.5, 1.0, 2.0, 7.25] {
        let expect = 1.0 / (PI * h * h * h);
        let got = w(0.0, h);
        let rel = (got - expect).abs() / expect;
        // 1e-14: both sides are a handful of f64 ops on exact inputs.
        assert!(rel < 1e-14, "W(0,{h}) = {got}, want {expect}");
    }
}

#[test]
fn w_vanishes_at_and_beyond_support() {
    for h in [0.5, 1.0, 3.0] {
        assert_eq!(w(SUPPORT * h, h), 0.0, "W must be exactly 0 at r = 2h");
        assert_eq!(w(SUPPORT * h * 1.0001, h), 0.0);
        assert_eq!(w(10.0 * h, h), 0.0);
        assert!(
            w(SUPPORT * h * 0.999, h) > 0.0,
            "W must be positive just inside the support"
        );
    }
}

#[test]
fn w_is_continuous_at_the_knots() {
    // The M4 spline is C¹: value continuous at q = 1 and q = 2. Lipschitz bound:
    // |W′(q=1)| = (3/4)/(π h³)·(1/h), so a ±δ straddle differs by ≲ 2δ·|W′|.
    let h = 1.3;
    let norm = 1.0 / (PI * h * h * h);
    let delta = 1e-8 * h;
    for knot in [1.0 * h, 2.0 * h] {
        let below = w(knot - delta, h);
        let above = w(knot + delta, h);
        assert!(
            (below - above).abs() < 3.0 * delta * norm / h,
            "W discontinuous at r = {knot}: {below} vs {above}"
        );
    }
}

#[test]
fn w_integrates_to_one() {
    // 4π ∫₀^{2h} W(r) r² dr = 1. Composite Simpson per polynomial piece
    // ([0,h] and [h,2h], so the integrand is a single quintic on each panel):
    // error per panel ≤ (b−a)·Δ⁴/180·max|f⁗| = O(n⁻⁴); n = 2000 pushes that
    // far below the 1e-9 gate, which then only allows f64 accumulation noise.
    let h = 0.9;
    let simpson = |a: f64, b: f64, n: usize| -> f64 {
        // n even
        let dx = (b - a) / n as f64;
        let f = |r: f64| w(r, h) * r * r;
        let mut s = f(a) + f(b);
        for k in 1..n {
            let x = a + k as f64 * dx;
            s += f(x) * if k % 2 == 1 { 4.0 } else { 2.0 };
        }
        s * dx / 3.0
    };
    let integral = 4.0 * PI * (simpson(0.0, h, 2000) + simpson(h, 2.0 * h, 2000));
    assert!(
        (integral - 1.0).abs() < 1e-9,
        "∫W dV = {integral}, want 1 (kernel not normalized)"
    );
}

#[test]
fn grad_w_matches_central_differences() {
    // ∇W is analytic; central differences on the scalar field x ↦ W(|x|, h) have
    // O(δ²·f‴) error. With δ = 1e-6·h and |f‴| = O(norm/h³), the absolute error
    // is ~1e-12·norm/h; gate at 1e-9·norm/h for slack. Probe all three kernel
    // regimes (q<1, 1<q<2, q>2) off-axis so every component is exercised.
    let h = 0.8;
    let norm = 1.0 / (PI * h * h * h);
    let delta = 1e-6 * h;
    let probes = [
        DVec3::new(0.3, -0.2, 0.1) * h, // q ≈ 0.37
        DVec3::new(0.9, 0.4, -0.3) * h, // q ≈ 1.03
        DVec3::new(-1.2, 0.8, 0.6) * h, // q ≈ 1.56
        DVec3::new(2.0, 1.0, -1.5) * h, // q ≈ 2.7 → exactly zero
    ];
    for r_ij in probes {
        let analytic = grad_w(r_ij, h);
        let mut numeric = DVec3::ZERO;
        for axis in 0..3 {
            let mut e = DVec3::ZERO;
            e[axis] = delta;
            numeric[axis] = (w((r_ij + e).length(), h) - w((r_ij - e).length(), h)) / (2.0 * delta);
        }
        let err = (analytic - numeric).length();
        assert!(
            err < 1e-9 * norm / h,
            "grad mismatch at {r_ij:?}: analytic {analytic:?} vs numeric {numeric:?}"
        );
    }
}

#[test]
fn grad_w_is_zero_at_the_origin_and_outside_support() {
    let h = 1.1;
    assert_eq!(grad_w(DVec3::ZERO, h), DVec3::ZERO);
    assert_eq!(grad_w(DVec3::new(3.0 * h, 0.0, 0.0), h), DVec3::ZERO);
}

#[test]
fn grad_w_points_downhill() {
    // W decreases monotonically in r, so ∇_i W must point from j toward i's
    // far side: opposite the separation vector (attractive-looking sign).
    let h = 1.0;
    let r_ij = DVec3::new(0.7, 0.2, -0.4);
    let g = grad_w(r_ij, h);
    assert!(
        g.dot(r_ij) < 0.0,
        "∇W must be anti-parallel to the separation, got {g:?}"
    );
}

#[test]
fn w_and_grad_obey_the_scaling_law() {
    // W(λr, λh) = W(r, h)/λ³ and ∇W(λr, λh) = ∇W(r, h)/λ⁴ (dimensional
    // analysis of the 3-D normalization). 1e-13 rel: pure f64 rounding.
    let (h, lambda) = (0.6, 3.7);
    for r in [0.1, 0.5, 0.9, 1.4] {
        let base = w(r * h, h);
        let scaled = w(lambda * r * h, lambda * h);
        let rel = (scaled - base / lambda.powi(3)).abs() / base.max(1e-300);
        assert!(rel < 1e-13, "W scaling broken at q={r}");
    }
    let r_ij = DVec3::new(0.5, -0.3, 0.2);
    let g_base = grad_w(r_ij, h) / lambda.powi(4);
    let g_scaled = grad_w(r_ij * lambda, lambda * h);
    assert!(
        (g_base - g_scaled).length() < 1e-13 * g_base.length(),
        "∇W scaling broken"
    );
}
