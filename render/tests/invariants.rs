//! Render invariants (DESIGN.md M3). These are the *testable* properties of the
//! splat renderer — a GPU renderer can't be pure-TDD'd pixel-for-pixel, so we pin
//! the physics-apt invariants instead, in the project's invariant-over-example style:
//!
//!   - **Additive commutativity** — DESIGN's order-independence claim, made a test:
//!     permuting particle order changes the float summation order only, so the buffer
//!     matches within a *relative* tolerance (absolute epsilon false-fails at bright
//!     cores where many splats sum).
//!   - **Flux linearity** — doubling every brightness doubles the total accumulated
//!     flux (the Gaussian integral need not be predicted analytically).
//!   - **32F headroom** — piling bright splats on one pixel exceeds 1.0 without
//!     clamping (the reason DESIGN rejects a 16F accumulation buffer).
//!   - **Centered-splat symmetry** — a splat at the view center is left/right and
//!     up/down symmetric (odd image dims so the center lands on a pixel, not a seam).
//!
//! GPU-gated: these need a wgpu adapter. On a box without one, `Renderer::new()`
//! returns `NoAdapter` and these tests fail loudly (by design — see DESIGN's M3 note).

use galaxy_render::camera::Camera;
use galaxy_render::render::{HdrImage, RenderConfig, Renderer};
use galaxy_renderprep::FrameData;
use glam::{Vec2, Vec3};

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for render invariant tests")
}

/// A deterministic spread of splats in the z=0 plane (no RNG — trig-varied).
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

/// A square camera centered on the origin, view box `[-1, 1]²` in the z=0 plane.
fn centered_camera() -> Camera {
    Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0))
}

fn reversed(frame: &FrameData) -> FrameData {
    let mut f = frame.clone();
    f.pos.reverse();
    f.color.reverse();
    f.size.reverse();
    f.brightness.reverse();
    f
}

#[test]
fn additive_blend_is_order_independent() {
    let r = renderer();
    let cam = centered_camera();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };

    let frame = scene(40);
    let a = r.render_frame(&frame, &cam, &cfg).unwrap();
    let b = r.render_frame(&reversed(&frame), &cam, &cfg).unwrap();

    // Relative to the brightest pixel: reordering the sum perturbs only float rounding.
    let peak = a
        .pixels
        .iter()
        .flat_map(|p| p[..3].iter())
        .fold(0.0f32, |m, &c| m.max(c.abs()));
    let max_diff = a
        .pixels
        .iter()
        .zip(&b.pixels)
        .flat_map(|(x, y)| (0..3).map(move |c| (x[c] - y[c]).abs()))
        .fold(0.0f32, f32::max);
    assert!(peak > 0.0, "scene rendered black");
    assert!(
        max_diff / peak < 1e-3,
        "additive blend not order-independent: max_diff {max_diff}, peak {peak}"
    );
}

#[test]
fn total_flux_scales_linearly_with_brightness() {
    let r = renderer();
    let cam = centered_camera();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };

    let frame = scene(25);
    let base = r.render_frame(&frame, &cam, &cfg).unwrap();

    let mut doubled = frame.clone();
    for b in &mut doubled.brightness {
        *b *= 2.0;
    }
    let bright = r.render_frame(&doubled, &cam, &cfg).unwrap();

    let f1 = base.total_flux();
    let f2 = bright.total_flux();
    for c in 0..3 {
        assert!(f1[c] > 0.0, "channel {c} flux was zero");
        let rel = (f2[c] - 2.0 * f1[c]).abs() / f1[c];
        assert!(rel < 1e-3, "channel {c}: flux not linear, rel err {rel}");
    }
}

#[test]
fn overlapping_splats_exceed_one_without_clamping() {
    let r = renderer();
    let cam = centered_camera();
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        falloff: 6.0,
        ..RenderConfig::default()
    };

    // 20 white unit-brightness splats stacked at the center: the center pixel must
    // accumulate well past 1.0 and stay finite (32F headroom, no 16F-style clamp).
    let n = 20;
    let frame = FrameData {
        pos: vec![Vec3::ZERO; n],
        color: vec![[1.0, 1.0, 1.0]; n],
        size: vec![0.3; n],
        brightness: vec![1.0; n],
    };
    let img = r.render_frame(&frame, &cam, &cfg).unwrap();
    let center = img.pixel(32, 32);
    assert!(center[0] > 1.0, "center did not exceed 1.0: {center:?}");
    assert!(
        center.iter().all(|c| c.is_finite()),
        "non-finite: {center:?}"
    );
}

#[test]
fn centered_splat_is_symmetric() {
    let r = renderer();
    let cam = centered_camera();
    // Odd dims so the origin lands on pixel 64, not a pixel seam.
    let cfg = RenderConfig {
        width: 129,
        height: 129,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let frame = FrameData {
        pos: vec![Vec3::ZERO],
        color: vec![[1.0, 1.0, 1.0]],
        size: vec![0.25],
        brightness: vec![1.0],
    };
    let img: HdrImage = r.render_frame(&frame, &cam, &cfg).unwrap();
    let c = 64u32;
    let center = img.pixel(c, c)[0];
    assert!(center > 0.0, "center pixel is dark: {center}");

    for k in 1..=10u32 {
        let (l, rgt) = (img.pixel(c - k, c)[0], img.pixel(c + k, c)[0]);
        let (up, dn) = (img.pixel(c, c - k)[0], img.pixel(c, c + k)[0]);
        let tol = 1e-3 * center;
        assert!((l - rgt).abs() <= tol, "horiz asym at k={k}: {l} vs {rgt}");
        assert!((up - dn).abs() <= tol, "vert asym at k={k}: {up} vs {dn}");
        assert!(
            l <= center + tol,
            "off-center brighter than center at k={k}"
        );
    }
}
