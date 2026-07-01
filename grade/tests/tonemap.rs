//! Tone-curve + sRGB grading math (DESIGN.md M3). Pure CPU. Expectations are
//! hand-derived from the operator definitions, not read back from the code:
//!   - Reinhard(1) = 1/(1+1) = 0.5.
//!   - ACES-approx (Narkowicz) at 0.5 = 0.6425/1.0425 ≈ 0.616307.
//!   - sRGB(0.5) = 1.055·0.5^(1/2.4) − 0.055 ≈ 0.735357; sRGB is linear (×12.92)
//!     below 0.0031308.

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
        },
    );
    let b = tonemap(
        doubled,
        &GradeConfig {
            exposure: 1.0,
            tonemap: ToneMap::AcesApprox,
        },
    );
    assert_eq!(a, b);
}
