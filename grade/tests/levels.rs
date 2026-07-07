//! Levels stage (black point / white point / gamma) added to the grade after
//! the tone curve and before the sRGB OETF (galaxy-render "more controls" pass).
//!
//! Semantics (hand-derived, not read back from the code):
//!   levels(x) = clamp((x − black) / (white − black), 0, 1) ^ (1/gamma)
//!   applied on the DISPLAY-referred tone-curve output x ∈ [0, 1].
//!   - black point lifts shadows to true black (contrast + reveals faint stars
//!     out of a scatter haze);
//!   - white point sets where the signal clips;
//!   - gamma is the Photoshop midtone slider: gamma > 1 brightens mids
//!     (out = n^(1/gamma), n^0.5 > n), gamma < 1 darkens/crushes mids.
//!
//! The neutral triple (0, 1, 1) is the EXACT identity — bit-for-bit the
//! pre-levels grade (the codebase's neutral-gate convention). This is enforced
//! by special-casing the neutral values, since `x.powf(1.0)` is not guaranteed
//! to be bit-identical to `x`.

use galaxy_grade::{apply_levels, tonemap, GradeConfig, ToneMap};

const EPS: f32 = 1e-6;

// ---------- apply_levels: the pure curve ----------

/// Neutral (0, 1, 1) is the exact identity for every input, including the
/// awkward values where `powf(1.0)` could perturb the last bit.
#[test]
fn neutral_levels_are_bit_identical() {
    for i in 0..=1000 {
        let x = i as f32 / 1000.0;
        let y = apply_levels(x, 0.0, 1.0, 1.0);
        assert_eq!(
            y.to_bits(),
            x.to_bits(),
            "neutral levels must be bit-identical at x={x}: got {y}"
        );
    }
    // Awkward non-round values too.
    for x in [0.123_456_79_f32, 0.618_034, 0.999_999, 1.0 / 3.0] {
        assert_eq!(apply_levels(x, 0.0, 1.0, 1.0).to_bits(), x.to_bits());
    }
}

/// A lifted black point maps everything at/below `black` to 0 and rescales the
/// rest: (x − black)/(1 − black), gamma = 1.
#[test]
fn black_point_crushes_shadows() {
    let black = 0.2;
    // Below and at the black point → exactly 0 (true black).
    assert_eq!(apply_levels(0.1, black, 1.0, 1.0), 0.0);
    assert_eq!(apply_levels(black, black, 1.0, 1.0), 0.0);
    // White stays white.
    assert!((apply_levels(1.0, black, 1.0, 1.0) - 1.0).abs() < EPS);
    // A midtone rescales: (0.6 − 0.2)/0.8 = 0.5.
    assert!((apply_levels(0.6, black, 1.0, 1.0) - 0.5).abs() < EPS);
}

/// A pulled-in white point maps `white` to 1.0 and clamps above it.
#[test]
fn white_point_sets_the_clip() {
    let white = 0.5;
    assert!((apply_levels(0.5, 0.0, white, 1.0) - 1.0).abs() < EPS);
    // (0.25 − 0)/0.5 = 0.5.
    assert!((apply_levels(0.25, 0.0, white, 1.0) - 0.5).abs() < EPS);
    // Above the white point clamps at 1.0, never above.
    assert_eq!(apply_levels(0.8, 0.0, white, 1.0), 1.0);
}

/// Gamma is the midtone control: out = n^(1/gamma) on the black/white-normalized
/// n. gamma = 2 brightens the mid (0.25 → 0.5), gamma = 0.5 crushes it
/// (0.5 → 0.25) — the haze-suppression direction.
#[test]
fn gamma_moves_the_midtone() {
    // n = 0.25, gamma = 2 → 0.25^(1/2) = 0.5.
    assert!((apply_levels(0.25, 0.0, 1.0, 2.0) - 0.5).abs() < EPS);
    // n = 0.5, gamma = 0.5 → 0.5^(2) = 0.25 (crush mids toward black).
    assert!((apply_levels(0.5, 0.0, 1.0, 0.5) - 0.25).abs() < EPS);
    // Endpoints are gamma-invariant: 0^p = 0, 1^p = 1.
    assert_eq!(apply_levels(0.0, 0.0, 1.0, 0.3), 0.0);
    assert!((apply_levels(1.0, 0.0, 1.0, 0.3) - 1.0).abs() < EPS);
}

/// Monotone non-decreasing and clamped to [0, 1] for a spread of settings —
/// levels must never invert or leave gamut.
#[test]
fn levels_are_monotonic_and_in_gamut() {
    for &(b, w, g) in &[
        (0.0f32, 1.0f32, 1.0f32),
        (0.15, 0.9, 0.6),
        (0.0, 0.5, 2.2),
        (0.3, 1.0, 1.0),
    ] {
        let mut prev = -1.0f32;
        for i in 0..=200 {
            let x = i as f32 / 200.0;
            let y = apply_levels(x, b, w, g);
            assert!(
                (0.0..=1.0).contains(&y),
                "levels({b},{w},{g}) out of gamut at x={x}: {y}"
            );
            assert!(
                y >= prev - EPS,
                "levels({b},{w},{g}) not monotone at x={x}: {prev} !<= {y}"
            );
            prev = y;
        }
    }
}

// ---------- tonemap integration: levels run after the tone curve ----------

/// Levels are applied to the tone-curve OUTPUT (display-referred), before sRGB.
/// Reinhard(1.0) = 0.5; a black point at 0.5 crushes that to pure black.
#[test]
fn tonemap_applies_levels_after_the_curve() {
    let cfg = GradeConfig {
        exposure: 1.0,
        tonemap: ToneMap::Reinhard,
        bloom: None,
        black_point: 0.5,
        white_point: 1.0,
        gamma: 1.0,
        local: None,
    };
    // tone_curve → 0.5, levels((0.5−0.5)/0.5)=0 → sRGB(0)=0 → black.
    assert_eq!(tonemap([1.0; 3], &cfg), [0; 3]);

    // A white point at the curve output 0.5 pushes it to full white.
    let cfg_white = GradeConfig {
        white_point: 0.5,
        ..cfg
    };
    assert_eq!(tonemap([1.0; 3], &cfg_white), [u16::MAX; 3]);
}

/// The neutral levels triple leaves tonemap bit-identical to the pre-levels
/// grade: black stays black, huge HDR still saturates white.
#[test]
fn neutral_levels_preserve_tonemap() {
    let cfg = GradeConfig::default();
    assert_eq!(cfg.black_point, 0.0);
    assert_eq!(cfg.white_point, 1.0);
    assert_eq!(cfg.gamma, 1.0);
    assert_eq!(tonemap([0.0; 3], &cfg), [0; 3]);
    assert_eq!(tonemap([1.0e6; 3], &cfg), [u16::MAX; 3]);
}

// ---------- validation ----------

/// A degenerate levels window (black ≥ white) or a non-positive gamma is a
/// config error, caught before a frame is processed.
#[test]
fn invalid_levels_are_rejected() {
    // black == white: zero-width window (division by zero).
    assert!(GradeConfig {
        black_point: 0.5,
        white_point: 0.5,
        ..GradeConfig::default()
    }
    .validate()
    .is_err());
    // black > white: inverted window.
    assert!(GradeConfig {
        black_point: 0.8,
        white_point: 0.2,
        ..GradeConfig::default()
    }
    .validate()
    .is_err());
    // gamma <= 0: undefined power / division.
    assert!(GradeConfig {
        gamma: 0.0,
        ..GradeConfig::default()
    }
    .validate()
    .is_err());
    assert!(GradeConfig {
        gamma: -1.0,
        ..GradeConfig::default()
    }
    .validate()
    .is_err());
    // NaN/Inf are rejected.
    assert!(GradeConfig {
        black_point: f32::NAN,
        ..GradeConfig::default()
    }
    .validate()
    .is_err());
    // The neutral default validates.
    assert!(GradeConfig::default().validate().is_ok());
}
