//! Pinprick-starfield gates (docs/plans/pinprick-starfield.md): `max_splat_px`
//! caps the on-screen splat **half-extent in pixels** — the screen-space PSF of
//! a point source, so stars stay point-like at any zoom — while conserving
//! integrated flux: clamping DOWN boosts emission by (true/clamped)², the exact
//! mirror of the `min_splat_px` dimming. The cap must bite under BOTH
//! projections: the orbit-tilt rig is orthographic, and that branch carried no
//! clamps before this feature. Off (the `INFINITY` default) is bit-identical —
//! the M6g ortho golden keeps pinning the default path.
//!
//! GPU-gated like `vertex_path.rs`: needs a wgpu adapter, fails loudly without.

use galaxy_render::camera::Camera;
use galaxy_render::render::{HdrImage, RenderConfig, Renderer};
use galaxy_render::RenderError;
use galaxy_renderprep::FrameData;
use glam::{Vec2, Vec3};

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for splat-cap gates")
}

/// The reference perspective rig shared with `tests/vertex_path.rs`: target at
/// the origin, looking down −Z with +Y up, half_extent (1,1) at the target
/// plane, eye at (0, 0, +4), near 0.01 — hand-derivable geometry.
fn persp_cam() -> Camera {
    Camera::perspective(
        Vec3::ZERO,
        Vec3::NEG_Z,
        Vec3::Y,
        Vec2::new(1.0, 1.0),
        4.0,
        0.01,
    )
}

fn ortho_cam() -> Camera {
    Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0))
}

/// A single white splat.
fn one(pos: Vec3, size: f32) -> (Vec3, [f32; 3], f32, f32) {
    (pos, [1.0, 1.0, 1.0], size, 1.0)
}

fn frame_of(parts: &[(Vec3, [f32; 3], f32, f32)]) -> FrameData {
    FrameData {
        pos: parts.iter().map(|p| p.0).collect(),
        color: parts.iter().map(|p| p.1).collect(),
        size: parts.iter().map(|p| p.2).collect(),
        brightness: parts.iter().map(|p| p.3).collect(),
    }
}

/// The same deterministic trig-varied scene as `vertex_path.rs` (no RNG).
fn scene(n: usize) -> FrameData {
    let mut pos = Vec::new();
    let mut color = Vec::new();
    let mut size = Vec::new();
    let mut brightness = Vec::new();
    for i in 0..n {
        let t = i as f32;
        pos.push(Vec3::new(0.8 * (t * 1.3).cos(), 0.8 * (t * 0.7).sin(), 0.0));
        color.push([
            0.5 + 0.5 * t.sin(),
            0.5 + 0.5 * (t * 1.1).cos(),
            0.5 + 0.5 * (t * 0.3).sin(),
        ]);
        size.push(0.05 + 0.03 * (i % 3) as f32);
        brightness.push(1.0 + (i % 5) as f32);
    }
    FrameData {
        pos,
        color,
        size,
        brightness,
    }
}

/// Total RGB flux in the left / right half of the image (one splat per half).
fn half_fluxes(img: &HdrImage) -> (f64, f64) {
    let (mut l, mut r) = (0.0f64, 0.0f64);
    for y in 0..img.height {
        for x in 0..img.width {
            let p = img.pixel(x, y);
            let s = p[0] as f64 + p[1] as f64 + p[2] as f64;
            if x < img.width / 2 {
                l += s;
            } else {
                r += s;
            }
        }
    }
    (l, r)
}

/// Peak RGB value over an image half (`left` selects x < width/2).
fn half_peak(img: &HdrImage, left: bool) -> f32 {
    let mut m = 0.0f32;
    for y in 0..img.height {
        for x in 0..img.width {
            if (x < img.width / 2) == left {
                let p = img.pixel(x, y);
                m = m.max(p[0]).max(p[1]).max(p[2]);
            }
        }
    }
    m
}

/// Peak RGB value over the whole image.
fn peak(img: &HdrImage) -> f32 {
    half_peak(img, true).max(half_peak(img, false))
}

/// Chebyshev half-width in pixels of the lit region (any RGB channel > 0)
/// about the image center — the rendered splat's on-screen extent.
fn lit_half_width(img: &HdrImage) -> f32 {
    let cx = (img.width as f32 - 1.0) / 2.0;
    let cy = (img.height as f32 - 1.0) / 2.0;
    let mut r = 0.0f32;
    for y in 0..img.height {
        for x in 0..img.width {
            let p = img.pixel(x, y);
            if p[0] > 0.0 || p[1] > 0.0 || p[2] > 0.0 {
                r = r.max((x as f32 - cx).abs().max((y as f32 - cy).abs()));
            }
        }
    }
    r
}

// --- off is off -----------------------------------------------------------------

/// A finite cap that never bites (1e6 px) must be bit-identical to the off
/// default under BOTH projections — the cap is a taken-branch only when it
/// bites, so "enabled but idle" changes no arithmetic. (The M6g ortho golden in
/// `vertex_path.rs` keeps pinning the fully-off default path.)
#[test]
fn idle_cap_is_bit_identical() {
    let r = renderer();
    let base = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let capped = RenderConfig {
        max_splat_px: 1e6,
        ..base
    };
    let frame = scene(40);

    let a = r.render_frame(&frame, &ortho_cam(), &base).unwrap();
    let b = r.render_frame(&frame, &ortho_cam(), &capped).unwrap();
    assert_eq!(a, b, "idle cap changed the orthographic render");

    let frame_p = frame_of(&[
        one(Vec3::new(-0.5, 0.0, 0.0), 0.15),
        one(Vec3::new(1.0, 0.0, -4.0), 0.15),
    ]);
    let a = r.render_frame(&frame_p, &persp_cam(), &base).unwrap();
    let b = r.render_frame(&frame_p, &persp_cam(), &capped).unwrap();
    assert_eq!(a, b, "idle cap changed the perspective render");
}

// --- the cap reshapes the PSF at constant flux (orthographic) --------------------

/// One well-resolved splat (0.3 NDC = 38.4 px half-extent at 256²), capped to
/// 9.6 px: the lit footprint must shrink to the cap, while total flux stays
/// equal to the uncapped render within 2% — the same tolerance as the
/// inverse-square gate, since both footprints are well-resolved and the
/// Gaussian's quad-edge truncation is scale-relative (it cancels in the ratio).
#[test]
fn ortho_cap_shrinks_footprint_at_constant_flux() {
    let r = renderer();
    let base = RenderConfig {
        width: 256,
        height: 256,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let capped = RenderConfig {
        max_splat_px: 9.6,
        ..base
    };
    let frame = frame_of(&[one(Vec3::ZERO, 0.3)]);

    let full = r.render_frame(&frame, &ortho_cam(), &base).unwrap();
    let tight = r.render_frame(&frame, &ortho_cam(), &capped).unwrap();

    // The footprint actually shrank (this is what fails before the WGSL lands).
    let w_full = lit_half_width(&full);
    let w_tight = lit_half_width(&tight);
    assert!(
        w_full > 30.0,
        "uncapped splat should span ~38 px, got {w_full}"
    );
    assert!(
        w_tight <= 12.0,
        "capped splat must fit the 9.6 px cap (+ raster margin), got {w_tight}"
    );

    // Photometry is untouched: the cap reshapes, never dims or brightens.
    let (ff, tf) = (full.total_flux(), tight.total_flux());
    for c in 0..3 {
        assert!(ff[c] > 0.0, "uncapped channel {c} rendered black");
        let rel = (tf[c] - ff[c]).abs() / ff[c];
        assert!(
            rel < 0.02,
            "cap changed channel {c} flux: {} vs {} (rel {rel})",
            tf[c],
            ff[c]
        );
    }
}

/// Same pair: conserving flux while shrinking the footprint means the peak
/// brightens by (true/clamped)² = (38.4/9.6)² = 16 — the point-source regime
/// (zooming into a star concentrates its flux). The splat is centered on a
/// pixel CENTER (NDC (0.5/128, 0.5/128) at 256²), so the peak pixel samples
/// the Gaussian at r = 0 in both renders and the ratio is the boost factor
/// exactly (at NDC 0 the center falls on a pixel corner and the 0.5√2 px
/// diagonal offset skews the ratio to 15.52). 1% tolerance.
#[test]
fn ortho_cap_concentrates_peak_by_flux_ratio() {
    let r = renderer();
    let base = RenderConfig {
        width: 256,
        height: 256,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let capped = RenderConfig {
        max_splat_px: 9.6,
        ..base
    };
    let c = 0.5 / 128.0;
    let frame = frame_of(&[one(Vec3::new(c, c, 0.0), 0.3)]);

    let full = r.render_frame(&frame, &ortho_cam(), &base).unwrap();
    let tight = r.render_frame(&frame, &ortho_cam(), &capped).unwrap();

    let ratio = peak(&tight) / peak(&full);
    assert!(
        (ratio - 16.0).abs() / 16.0 < 0.01,
        "cap must boost peak by (38.4/9.6)² = 16, got ×{ratio}"
    );
}

// --- the flux law survives under perspective --------------------------------------

/// The inverse-square pair from `vertex_path.rs` with the near splat capped
/// (19.2 px → 12 px) and the far splat untouched (9.6 px < cap): the flux
/// ratio must still be (8/4)² = 4 within 2% — the cap coexists with the 1/d²
/// law it was built to preserve — and the near peak must show the
/// (19.2/12)² = 2.56 concentration against the uncapped reference (3% tol,
/// same 0.5 px center-offset argument as the ortho peak gate).
#[test]
fn perspective_inverse_square_survives_cap() {
    let r = renderer();
    let base = RenderConfig {
        width: 256,
        height: 256,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let capped = RenderConfig {
        max_splat_px: 12.0,
        ..base
    };
    let frame = frame_of(&[
        one(Vec3::new(-0.5, 0.0, 0.0), 0.15),
        one(Vec3::new(1.0, 0.0, -4.0), 0.15),
    ]);

    let reference = r.render_frame(&frame, &persp_cam(), &base).unwrap();
    let img = r.render_frame(&frame, &persp_cam(), &capped).unwrap();

    let (near_flux, far_flux) = half_fluxes(&img);
    assert!(near_flux > 0.0 && far_flux > 0.0, "a splat rendered black");
    let ratio = near_flux / far_flux;
    assert!(
        (ratio - 4.0).abs() / 4.0 < 0.02,
        "cap broke inverse-square: flux ratio {ratio}, expected 4"
    );

    let boost = half_peak(&img, true) / half_peak(&reference, true);
    let want = (19.2f32 / 12.0).powi(2);
    assert!(
        (boost - want).abs() / want < 0.03,
        "capped near splat must concentrate by (19.2/12)² = {want}, got ×{boost}"
    );
}

/// The full clamp window under perspective: `min_splat_px = 8` dims a sub-pixel
/// splat UP (depth 40, 1.92 px → 8 px) while `max_splat_px = 12` boosts an
/// oversized one DOWN (depth 4, 19.2 px → 12 px) — both flux-conserving, so the
/// integrated ratio is still (40/4)² = 100 within 3% (the min-clamp gate's
/// tolerance). Kills a saturating (non-boosting) cap impl, which would read 39.
#[test]
fn perspective_window_composes_min_and_cap() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 256,
        height: 256,
        falloff: 6.0,
        min_splat_px: 8.0,
        max_splat_px: 12.0,
        ..RenderConfig::default()
    };
    // A at NDC (−0.5, 0), depth 4, 19.2 px → capped to 12. B at depth 40
    // (z = −36), lateral 5 → NDC +0.5, 1.92 px → min-clamped to 8.
    let frame = frame_of(&[
        one(Vec3::new(-0.5, 0.0, 0.0), 0.15),
        one(Vec3::new(5.0, 0.0, -36.0), 0.15),
    ]);
    let img = r.render_frame(&frame, &persp_cam(), &cfg).unwrap();
    let (near_flux, far_flux) = half_fluxes(&img);
    assert!(near_flux > 0.0 && far_flux > 0.0, "a splat rendered black");
    let ratio = near_flux / far_flux;
    assert!(
        (ratio - 100.0).abs() / 100.0 < 0.03,
        "clamp window broke inverse-square: flux ratio {ratio}, expected 100"
    );
}

// --- validation (fail loudly) ------------------------------------------------------

/// A finite cap must be positive: 0, negative, and NaN are configuration
/// errors under BOTH projections (INFINITY is the documented off value).
#[test]
fn nonpositive_or_nan_cap_errors() {
    let r = renderer();
    let frame = frame_of(&[one(Vec3::ZERO, 0.15)]);
    for bad in [0.0f32, -3.0, f32::NAN] {
        let cfg = RenderConfig {
            width: 64,
            height: 64,
            falloff: 6.0,
            max_splat_px: bad,
            ..RenderConfig::default()
        };
        for cam in [ortho_cam(), persp_cam()] {
            let err = r
                .render_frame(&frame, &cam, &cfg)
                .expect_err(&format!("max_splat_px = {bad} must be rejected"));
            assert!(
                matches!(err, RenderError::Config(_)),
                "wrong error kind for max_splat_px = {bad}: {err:?}"
            );
        }
    }
}

/// Under perspective a finite cap below `min_splat_px` is a crossed clamp
/// window (WGSL clamp() UB) — an error. Orthographic ignores `min_splat_px`,
/// so the same config is legitimate there.
#[test]
fn perspective_cap_below_min_errors() {
    let r = renderer();
    let frame = frame_of(&[one(Vec3::ZERO, 0.15)]);
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        falloff: 6.0,
        min_splat_px: 8.0,
        max_splat_px: 4.0,
        ..RenderConfig::default()
    };
    let err = r
        .render_frame(&frame, &persp_cam(), &cfg)
        .expect_err("cap < min_splat_px must be rejected under perspective");
    assert!(
        matches!(err, RenderError::Config(_)),
        "wrong error kind: {err:?}"
    );
    r.render_frame(&frame, &ortho_cam(), &cfg)
        .expect("ortho ignores min_splat_px — the same window is valid there");
}
