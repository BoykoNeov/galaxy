//! Tone-curve + sRGB grading math (DESIGN.md M3, asinh added in M6a). Pure CPU.
//! Expectations are hand-derived from the operator definitions, not read back from
//! the code:
//!   - Reinhard(1) = 1/(1+1) = 0.5.
//!   - ACES-approx (Narkowicz) at 0.5 = 0.6425/1.0425 ≈ 0.616307.
//!   - sRGB(0.5) = 1.055·0.5^(1/2.4) − 0.055 ≈ 0.735357; sRGB is linear (×12.92)
//!     below 0.0031308.
//!   - Asinh (Lupton stretch) f(x; β) = β·asinh(x/β) clamped to [0, 1]:
//!     asinh(1) = ln(1+√2) ≈ 0.8813736, so f(1; 1) ≈ 0.8813736 and
//!     f(2; 0.25) = 0.25·asinh(8) = 0.25·ln(8+√65) ≈ 0.694118.

use galaxy_grade::{linear_to_srgb, tone_curve, tonemap, GradeConfig, ToneMap};

const EPS: f32 = 1e-4;

#[test]
fn reinhard_curve_is_x_over_one_plus_x() {
    let t = tone_curve([1.0, 1.0, 1.0], ToneMap::Reinhard);
    for c in t {
        assert!((c - 0.5).abs() < EPS, "reinhard(1) should be 0.5, got {c}");
    }
    assert_eq!(tone_curve([0.0; 3], ToneMap::Reinhard), [0.0; 3]);
}

#[test]
fn aces_curve_matches_narkowicz() {
    let t = tone_curve([0.5, 0.5, 0.5], ToneMap::AcesApprox);
    for c in t {
        assert!((c - 0.616307).abs() < EPS, "aces(0.5) ≈ 0.616307, got {c}");
    }
    assert_eq!(tone_curve([0.0; 3], ToneMap::AcesApprox), [0.0; 3]);
}

#[test]
fn tone_curves_are_monotonic_and_clamped() {
    for op in [ToneMap::AcesApprox, ToneMap::Reinhard] {
        let lo = tone_curve([0.3; 3], op)[0];
        let hi = tone_curve([0.6; 3], op)[0];
        assert!(hi >= lo, "{op:?} not monotonic: {lo} !<= {hi}");
        // A huge input saturates at 1.0, never above.
        let big = tone_curve([1.0e6; 3], op)[0];
        assert!((0.0..=1.0).contains(&big), "{op:?} out of range: {big}");
        assert!(
            big > 0.99,
            "{op:?} should approach 1.0 for huge input: {big}"
        );
    }
}

// --- Asinh (Lupton stretch), M6a ---------------------------------------------

#[test]
fn asinh_matches_hand_values() {
    // f(x; β) = β·asinh(x/β). asinh(1) = ln(1+√2) ≈ 0.8813736.
    let t = tone_curve([1.0; 3], ToneMap::Asinh { beta: 1.0 });
    for c in t {
        assert!(
            (c - 0.881_374).abs() < EPS,
            "asinh(1;β=1) ≈ 0.881374, got {c}"
        );
    }
    // f(2; 0.25) = 0.25·asinh(8) = 0.25·ln(8+√65) ≈ 0.694118.
    let t = tone_curve([2.0; 3], ToneMap::Asinh { beta: 0.25 });
    for c in t {
        assert!(
            (c - 0.694_118).abs() < EPS,
            "asinh(2;β=0.25) ≈ 0.694118, got {c}"
        );
    }
    // Zero maps to zero exactly.
    assert_eq!(tone_curve([0.0; 3], ToneMap::Asinh { beta: 0.5 }), [0.0; 3]);
}

#[test]
fn asinh_is_monotonic_and_clamped() {
    for beta in [0.05f32, 0.2, 1.0] {
        let op = ToneMap::Asinh { beta };
        let mut prev = -1.0f32;
        for i in 0..=60 {
            // Geometric sweep from deep shadow into far HDR highlight.
            let x = 1.0e-4 * 1.35f32.powi(i);
            let y = tone_curve([x; 3], op)[0];
            assert!(
                (0.0..=1.0).contains(&y),
                "asinh(β={beta}) out of range at x={x}: {y}"
            );
            assert!(
                y >= prev,
                "asinh(β={beta}) not monotonic at x={x}: {prev} !<= {y}"
            );
            prev = y;
        }
        // A huge input must saturate at exactly 1.0 (clamped), never above.
        assert_eq!(tone_curve([1.0e9; 3], op)[0], 1.0);
    }
}

#[test]
fn asinh_large_beta_recovers_linear_at_small_x() {
    // asinh(u) = u − u³/6 + …, so β·asinh(x/β) → x as β → ∞: error ~ x³/(6β²),
    // ≤ 1.5e-8 for x ≤ 0.9 at β = 100 — far below the 1e-4 gate (f32 noise floor).
    let op = ToneMap::Asinh { beta: 100.0 };
    for x in [0.1f32, 0.5, 0.9] {
        let y = tone_curve([x; 3], op)[0];
        assert!(
            (y - x).abs() < EPS,
            "asinh(β=100) should be ~linear at x={x}, got {y}"
        );
    }
}

#[test]
fn asinh_compresses_highlights_harder_than_reinhard() {
    // At x = 100: Reinhard → 100/101 ≈ 0.990, asinh(β=0.1) → 0.1·asinh(1000)
    // ≈ 0.1·ln(2000) ≈ 0.760. The log regime holds highlights far below
    // Reinhard's asymptote — that headroom is what lets exposure be pushed to
    // reveal the faint tails without the cores flat-lining first.
    let a = tone_curve([100.0; 3], ToneMap::Asinh { beta: 0.1 })[0];
    let r = tone_curve([100.0; 3], ToneMap::Reinhard)[0];
    assert!(
        a < r,
        "asinh(β=0.1) should compress highlights below Reinhard at x=100: {a} !< {r}"
    );
    assert!(
        (a - 0.760_090).abs() < EPS,
        "asinh(100;β=0.1) ≈ 0.760090, got {a}"
    );
}

#[test]
fn asinh_degenerate_beta_stays_total() {
    // β = 0 would make asinh(x/β) = ∞ and 0·∞ = NaN; the curve floors β at
    // f32::MIN_POSITIVE so the output stays a defined value in [0, 1].
    for x in [0.0f32, 0.5, 1.0e6] {
        let y = tone_curve([x; 3], ToneMap::Asinh { beta: 0.0 })[0];
        assert!(
            y.is_finite() && (0.0..=1.0).contains(&y),
            "asinh(β=0) must stay total: x={x} → {y}"
        );
    }
}

#[test]
fn srgb_oetf_endpoints_and_midtone() {
    assert!((linear_to_srgb(0.0)).abs() < EPS);
    assert!((linear_to_srgb(1.0) - 1.0).abs() < EPS);
    assert!((linear_to_srgb(0.5) - 0.735357).abs() < EPS);
    // Linear segment below the knee: 12.92 · x.
    assert!((linear_to_srgb(0.002) - 12.92 * 0.002).abs() < EPS);
}

#[test]
fn tonemap_black_is_zero_and_hdr_saturates_white() {
    let cfg = GradeConfig::default();
    assert_eq!(tonemap([0.0; 3], &cfg), [0; 3]);
    let white = tonemap([1.0e6; 3], &cfg);
    assert_eq!(
        white,
        [u16::MAX; 3],
        "huge HDR should quantize to full white"
    );
}

#[test]
fn exposure_scales_before_the_curve() {
    // tonemap(c, exposure=2) must equal tonemap(2c, exposure=1): exposure is a
    // linear pre-multiplier.
    let c = [0.1, 0.2, 0.35];
    let doubled = [0.2, 0.4, 0.70];
    let a = tonemap(
        c,
        &GradeConfig {
            exposure: 2.0,
            tonemap: ToneMap::AcesApprox,
            bloom: None,
            ..GradeConfig::default()
        },
    );
    let b = tonemap(
        doubled,
        &GradeConfig {
            exposure: 1.0,
            tonemap: ToneMap::AcesApprox,
            bloom: None,
            ..GradeConfig::default()
        },
    );
    assert_eq!(a, b);
}
