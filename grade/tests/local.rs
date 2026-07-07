//! Local (spatially-adaptive) tone compression — the "white-blob" fix (render
//! "more controls" pass, tonemap lever 3). Where additive star splats pile into
//! one region the *global* tone curve saturates the whole core to a flat white
//! and internal sub-cores vanish; a global curve cannot recover them. The local
//! operator pulls exposure down *where the surround is bright*, dropping the blob
//! back into the tone curve's responsive range so its structure survives.
//!
//! Semantics (hand-derived from the operator definition, not read back from the
//! code): `g(V) = max(floor, 1 / (1 + strength·V))` where `V` is a Gaussian
//! low-pass of the *exposed* luminance. The linear RGB is multiplied by the
//! scalar `g`, so hue is preserved exactly and the operator only ever darkens.
//!
//! Key properties gated here:
//!   - `strength = 0` is a bit-exact no-op (the neutral-knob convention);
//!   - `g` is monotone *decreasing* in `V` and lives in `[floor, 1]` — more
//!     surround never brightens (the safety property);
//!   - a scalar gain preserves chroma ratios (hue);
//!   - **structure survival**: two pixels the global curve crushes to the same
//!     saturated white become distinguishable again under the local operator.
//!     That last one is the whole point of the feature.

use galaxy_grade::{
    apply_local_tonemap, local_gain, tonemap, GradeConfig, LocalToneConfig, ToneMap,
};

const EPS: f32 = 1e-4;

/// A uniform bright floor with one brighter core pixel at the centre.
fn blob_image(w: usize, h: usize, floor: f32, core: f32) -> Vec<[f32; 3]> {
    let mut px = vec![[floor; 3]; w * h];
    px[(h / 2) * w + w / 2] = [core; 3];
    px
}

// ---------- local_gain: the pure pointwise operator ----------

/// `strength = 0` yields exactly `1.0` for any surround — the identity gain that
/// makes the whole feature a bit-exact no-op when disabled.
#[test]
fn zero_strength_gain_is_exactly_one() {
    let cfg = LocalToneConfig {
        strength: 0.0,
        radius: 8.0,
        floor: 0.0,
    };
    for v in [0.0f32, 0.5, 8.0, 1.0e6] {
        assert_eq!(
            local_gain(v, &cfg),
            1.0,
            "zero strength must be unit gain at V={v}"
        );
    }
}

/// Hand values: `g = max(floor, 1/(1 + k·V))`.
#[test]
fn gain_matches_hand_values() {
    // k=2, V=0.5 → 1/(1+1) = 0.5.
    let g = local_gain(
        0.5,
        &LocalToneConfig {
            strength: 2.0,
            radius: 4.0,
            floor: 0.0,
        },
    );
    assert!((g - 0.5).abs() < EPS, "g(V=0.5;k=2) = 0.5, got {g}");

    // k=1, V=8 → 1/9 ≈ 0.11111.
    let g = local_gain(
        8.0,
        &LocalToneConfig {
            strength: 1.0,
            radius: 4.0,
            floor: 0.0,
        },
    );
    assert!((g - 1.0 / 9.0).abs() < EPS, "g(V=8;k=1) = 1/9, got {g}");

    // k=10, V=1 → 1/11 ≈ 0.0909 < floor 0.25 → clamps up to 0.25.
    let g = local_gain(
        1.0,
        &LocalToneConfig {
            strength: 10.0,
            radius: 4.0,
            floor: 0.25,
        },
    );
    assert!(
        (g - 0.25).abs() < EPS,
        "g below floor must clamp to 0.25, got {g}"
    );
}

/// `g` is monotone *decreasing* in the surround `V` (more surround never raises
/// the gain — "never brightens") and always in `[floor, 1]`.
#[test]
fn gain_is_monotone_decreasing_and_bounded() {
    for &(k, floor) in &[(0.5f32, 0.0f32), (1.0, 0.1), (4.0, 0.3), (20.0, 0.5)] {
        let cfg = LocalToneConfig {
            strength: k,
            radius: 8.0,
            floor,
        };
        let mut prev = f32::INFINITY;
        for i in 0..=200 {
            // Geometric sweep of the surround from deep shadow into HDR.
            let v = 1.0e-4 * 1.2f32.powi(i);
            let g = local_gain(v, &cfg);
            assert!(
                (floor..=1.0).contains(&g),
                "g(k={k},floor={floor}) out of [floor,1] at V={v}: {g}"
            );
            assert!(
                g <= prev + EPS,
                "g(k={k}) not monotone-decreasing at V={v}: {prev} !>= {g}"
            );
            prev = g;
        }
    }
}

// ---------- apply_local_tonemap: the image-space op ----------

/// `strength = 0` returns a bit-exact copy of the input image (the neutral-knob
/// convention, matching bloom's `strength = 0` no-op).
#[test]
fn zero_strength_apply_is_bit_identical() {
    let img = blob_image(16, 16, 8.0, 30.0);
    let out = apply_local_tonemap(
        &img,
        16,
        16,
        2.5,
        &LocalToneConfig {
            strength: 0.0,
            radius: 8.0,
            floor: 0.0,
        },
    );
    for (o, i) in out.iter().zip(&img) {
        for c in 0..3 {
            assert_eq!(
                o[c].to_bits(),
                i[c].to_bits(),
                "zero strength must be bit-identical: {o:?} vs {i:?}"
            );
        }
    }
}

/// The gain is a single scalar per pixel, so chroma ratios are preserved (hue)
/// and the operator only darkens. A uniform-colour image stays uniform.
#[test]
fn apply_preserves_hue_and_only_darkens() {
    let color = [0.6f32, 0.3, 0.1];
    let img = vec![color; 16 * 16];
    let out = apply_local_tonemap(
        &img,
        16,
        16,
        1.0,
        &LocalToneConfig {
            strength: 1.0,
            radius: 4.0,
            floor: 0.0,
        },
    );
    let first = out[0];
    for p in &out {
        // Uniform input → uniform output.
        for c in 0..3 {
            assert!(
                (p[c] - first[c]).abs() < EPS,
                "output not uniform: {p:?} vs {first:?}"
            );
            // Never brightens.
            assert!(
                p[c] <= color[c] + EPS,
                "channel {c} brightened: {} > {}",
                p[c],
                color[c]
            );
        }
        // Chroma ratios preserved: p0·C1 == p1·C0, p0·C2 == p2·C0.
        assert!(
            (p[0] * color[1] - p[1] * color[0]).abs() < EPS,
            "hue drift R:G in {p:?}"
        );
        assert!(
            (p[0] * color[2] - p[2] * color[0]).abs() < EPS,
            "hue drift R:B in {p:?}"
        );
    }
    // And it actually darkened (g<1 for a nonzero surround).
    assert!(
        first[0] < color[0] - EPS,
        "expected darkening, got {first:?}"
    );
}

/// Every output channel is ≤ the input (`g ≤ 1` everywhere), for an arbitrary
/// bright-blob image — the operator never brightens a pixel.
#[test]
fn apply_never_brightens_any_pixel() {
    let img = blob_image(24, 24, 5.0, 40.0);
    let out = apply_local_tonemap(
        &img,
        24,
        24,
        1.0,
        &LocalToneConfig {
            strength: 2.0,
            radius: 6.0,
            floor: 0.05,
        },
    );
    for (o, i) in out.iter().zip(&img) {
        for c in 0..3 {
            assert!(
                o[c] <= i[c] + EPS,
                "brightened a pixel: {} > {}",
                o[c],
                i[c]
            );
        }
    }
}

/// Wrong-length buffer is a programmer error (matches bloom's contract).
#[test]
#[should_panic]
fn apply_panics_on_size_mismatch() {
    let img = vec![[1.0f32; 3]; 10];
    let _ = apply_local_tonemap(
        &img,
        4,
        4,
        1.0,
        &LocalToneConfig {
            strength: 1.0,
            radius: 2.0,
            floor: 0.0,
        },
    );
}

// ---------- THE gate: structure survives inside the blob ----------

/// The whole point. A bright blob floor and a brighter sub-core both saturate the
/// global ACES curve to the SAME flat white (structure lost). The local operator
/// pulls the region down so the two become distinguishable again — and the floor
/// itself drops out of pure white (blob relieved).
#[test]
fn local_operator_recovers_blob_structure() {
    let (w, h) = (32, 32);
    let floor = 8.0f32; // ACES(8) = 1.003… → clamps to white.
    let core = 30.0f32; // ACES(30) → white too.
    let img = blob_image(w, h, floor, core);
    let ci = (h / 2) * w + w / 2; // core index
    let fi = 0; // a floor pixel, far from the core

    // --- Global grade: both saturate to full white, indistinguishable. ---
    let global = GradeConfig {
        tonemap: ToneMap::AcesApprox,
        ..GradeConfig::default()
    };
    let g_core = tonemap(img[ci], &global);
    let g_floor = tonemap(img[fi], &global);
    assert_eq!(
        g_core,
        [u16::MAX; 3],
        "global: core should blow out to white"
    );
    assert_eq!(
        g_floor,
        [u16::MAX; 3],
        "global: floor should blow out to white"
    );
    assert_eq!(
        g_core, g_floor,
        "global: blob structure is lost (the problem)"
    );

    // --- Local grade: apply the spatial gain, then the SAME global curve. ---
    let local_cfg = LocalToneConfig {
        strength: 1.0,
        radius: 8.0,
        floor: 0.0,
    };
    let relieved = apply_local_tonemap(&img, w, h, global.exposure, &local_cfg);
    let l_core = tonemap(relieved[ci], &global);
    let l_floor = tonemap(relieved[fi], &global);

    // Structure recovered: the core is now meaningfully brighter than the floor.
    assert!(
        l_core[0] > l_floor[0] + 1000,
        "local: core must separate from floor, got core={} floor={}",
        l_core[0],
        l_floor[0]
    );
    // Blob relieved: the floor is no longer pinned at pure white.
    assert!(
        l_floor[0] < u16::MAX,
        "local: floor should drop out of pure white, got {}",
        l_floor[0]
    );
}

// ---------- config validation ----------

/// A malformed local config (negative strength, floor outside [0,1], non-finite)
/// is rejected before a frame is touched; a valid one and `None` both pass.
#[test]
fn invalid_local_config_is_rejected() {
    let with = |local| GradeConfig {
        local: Some(local),
        ..GradeConfig::default()
    };
    let base = LocalToneConfig {
        strength: 1.0,
        radius: 8.0,
        floor: 0.1,
    };
    // Negative strength would invert the operator (brighten bright regions).
    assert!(with(LocalToneConfig {
        strength: -1.0,
        ..base
    })
    .validate()
    .is_err());
    // Floor outside [0, 1] — >1 would brighten, <0 is meaningless.
    assert!(with(LocalToneConfig { floor: 1.5, ..base })
        .validate()
        .is_err());
    assert!(with(LocalToneConfig {
        floor: -0.1,
        ..base
    })
    .validate()
    .is_err());
    // Non-finite knobs.
    assert!(with(LocalToneConfig {
        strength: f32::NAN,
        ..base
    })
    .validate()
    .is_err());
    assert!(with(LocalToneConfig {
        radius: f32::INFINITY,
        ..base
    })
    .validate()
    .is_err());
    // A valid local config, and no local at all, both pass.
    assert!(with(base).validate().is_ok());
    assert!(GradeConfig::default().validate().is_ok());
}
