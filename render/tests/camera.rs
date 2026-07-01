//! Orthographic camera math (DESIGN.md M3). Pure CPU — no GPU needed.
//!
//! Expectations are hand-derived from the projection definition, not read back from
//! the code under test: a point at `target` is the NDC origin; a point one
//! `half_extent` along `right` is NDC x = 1; auto-framing encloses every AABB corner
//! with margin headroom and no aspect distortion.

use galaxy_render::camera::{Camera, DEFAULT_MARGIN};
use glam::{Vec2, Vec3};

const EPS: f32 = 1e-5;

#[test]
fn face_on_uses_z_view_and_standard_screen_axes() {
    let c = Camera::face_on(Vec3::new(-2.0, -1.0, -3.0), Vec3::new(2.0, 1.0, 3.0), 1.0);
    // Looking down -Z with +Y up ⇒ right = +X, up = +Y, forward = -Z.
    assert!(
        c.forward.abs_diff_eq(Vec3::new(0.0, 0.0, -1.0), EPS),
        "{:?}",
        c.forward
    );
    assert!(c.right.abs_diff_eq(Vec3::X, EPS), "{:?}", c.right);
    assert!(c.up.abs_diff_eq(Vec3::Y, EPS), "{:?}", c.up);
}

#[test]
fn target_projects_to_ndc_origin() {
    let target = Vec3::new(1.0, 2.0, 3.0);
    let c = Camera::orthographic(target, Vec3::NEG_Z, Vec3::Y, Vec2::new(2.0, 1.0));
    assert!(c.project(target).abs_diff_eq(Vec2::ZERO, EPS));
}

#[test]
fn one_half_extent_along_axes_maps_to_ndc_edge() {
    let c = Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(2.0, 1.0));
    // target + right * half.x  →  ndc x = 1
    assert!(c
        .project(Vec3::new(2.0, 0.0, 0.0))
        .abs_diff_eq(Vec2::new(1.0, 0.0), EPS));
    // target + up * half.y  →  ndc y = 1
    assert!(c
        .project(Vec3::new(0.0, 1.0, 0.0))
        .abs_diff_eq(Vec2::new(0.0, 1.0), EPS));
}

#[test]
fn splat_world_radius_scales_by_half_extent() {
    let c = Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(2.0, 1.0));
    // radius 0.5 world units → (0.5/2, 0.5/1) NDC.
    assert!(c.splat_ndc(0.5).abs_diff_eq(Vec2::new(0.25, 0.5), EPS));
}

#[test]
fn frame_bounds_encloses_all_corners_with_margin_headroom() {
    let min = Vec3::new(-2.0, -1.0, -0.5);
    let max = Vec3::new(2.0, 1.0, 0.5);
    let c = Camera::face_on(min, max, 1.0);

    let mut max_mag = 0.0f32;
    for &x in &[min.x, max.x] {
        for &y in &[min.y, max.y] {
            for &z in &[min.z, max.z] {
                let ndc = c.project(Vec3::new(x, y, z));
                max_mag = max_mag.max(ndc.x.abs()).max(ndc.y.abs());
            }
        }
    }
    // Every corner is inside NDC, and the tightest one sits exactly at the margin
    // boundary 1/(1+margin) < 1 (headroom, nothing clipped at the frame edge).
    let expected = 1.0 / (1.0 + DEFAULT_MARGIN);
    assert!(
        max_mag < 1.0,
        "corners must be strictly inside NDC, got {max_mag}"
    );
    assert!(
        (max_mag - expected).abs() < 1e-4,
        "got {max_mag}, expected {expected}"
    );
}

#[test]
fn aspect_ratio_is_applied_without_distortion() {
    // A square scene rendered into a 2:1 image must widen the view box to 2:1 so a
    // world circle stays a screen circle (equal world-units-per-pixel on both axes).
    let c = Camera::face_on(Vec3::new(-1.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 0.0), 2.0);
    let ratio = c.half_extent.x / c.half_extent.y;
    assert!(
        (ratio - 2.0).abs() < 1e-4,
        "half_extent ratio {ratio} should equal aspect 2.0"
    );
}
