//! Camera math (DESIGN.md M3 ortho, M6g perspective). Pure CPU — no GPU needed.
//!
//! Expectations are hand-derived from the projection definition, not read back from
//! the code under test: a point at `target` is the NDC origin; a point one
//! `half_extent` along `right` is NDC x = 1; auto-framing encloses every AABB corner
//! with margin headroom and no aspect distortion. Perspective (M6g): the target
//! plane reproduces the ortho projection exactly (shared framing parameterization),
//! lateral offsets and splat extents shrink ∝ 1/view-depth from the pinhole
//! similar-triangles law, and view depth is signed (≤ 0 behind the eye ⇒ cull).

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

// ---------------------------------------------------------------------------
// M6g perspective gates. Reference geometry throughout: target at the origin,
// looking down −Z with +Y up, half_extent (1, 1) at the target plane, eye at
// (0, 0, +4) (distance 4), near 0.01. All expectations are similar-triangles
// arithmetic done by hand, never read back from the implementation.
// ---------------------------------------------------------------------------

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

#[test]
fn perspective_basis_matches_orthographic() {
    let c = persp_cam();
    assert!(c.forward.abs_diff_eq(Vec3::NEG_Z, EPS), "{:?}", c.forward);
    assert!(c.right.abs_diff_eq(Vec3::X, EPS), "{:?}", c.right);
    assert!(c.up.abs_diff_eq(Vec3::Y, EPS), "{:?}", c.up);
    assert_eq!(c.target, Vec3::ZERO);
    assert_eq!(c.half_extent, Vec2::new(1.0, 1.0));
}

#[test]
fn perspective_target_projects_to_ndc_origin() {
    let c = persp_cam();
    assert!(c.project(Vec3::ZERO).abs_diff_eq(Vec2::ZERO, EPS));
}

#[test]
fn perspective_matches_ortho_at_target_plane() {
    // Shared framing parameterization: in the z = 0 target plane the pinhole and
    // the ortho window agree exactly — the rig can swap projection in place.
    let p = persp_cam();
    let o = Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0));
    for &w in &[
        Vec3::new(0.3, -0.2, 0.0),
        Vec3::new(-0.9, 0.7, 0.0),
        Vec3::new(1.0, 0.0, 0.0), // exactly the window edge → NDC x = 1
    ] {
        assert!(
            p.project(w).abs_diff_eq(o.project(w), EPS),
            "target-plane mismatch at {w:?}: {:?} vs {:?}",
            p.project(w),
            o.project(w)
        );
    }
}

#[test]
fn perspective_foreshortens_inverse_with_depth() {
    // Eye at z = +4. A lateral offset of 0.5 at the target plane (depth 4) is
    // NDC 0.5; the same offset one target-distance further (z = −4, depth 8)
    // subtends half the angle → NDC 0.25. Similar triangles, by hand.
    let c = persp_cam();
    assert!(c
        .project(Vec3::new(0.5, 0.0, 0.0))
        .abs_diff_eq(Vec2::new(0.5, 0.0), EPS));
    assert!(c
        .project(Vec3::new(0.5, 0.0, -4.0))
        .abs_diff_eq(Vec2::new(0.25, 0.0), EPS));
    // And a nearer point (z = +2, depth 2) doubles: NDC 0.5 from offset 0.25.
    assert!(c
        .project(Vec3::new(0.25, 0.0, 2.0))
        .abs_diff_eq(Vec2::new(0.5, 0.0), EPS));
}

#[test]
fn perspective_respects_anisotropic_half_extent() {
    // half_extent (2, 1): a point at (0.5, 0.25, 0) → NDC (0.5/2, 0.25/1) at the
    // target plane, both axes divided by depth ratio further out.
    let c = Camera::perspective(
        Vec3::ZERO,
        Vec3::NEG_Z,
        Vec3::Y,
        Vec2::new(2.0, 1.0),
        4.0,
        0.01,
    );
    assert!(c
        .project(Vec3::new(0.5, 0.25, 0.0))
        .abs_diff_eq(Vec2::new(0.25, 0.25), EPS));
    assert!(c
        .project(Vec3::new(0.5, 0.25, -4.0))
        .abs_diff_eq(Vec2::new(0.125, 0.125), EPS));
}

#[test]
fn view_depth_is_measured_from_the_eye_and_signed() {
    let c = persp_cam();
    // Target plane is 4 in front of the eye, laterally invariant.
    assert!((c.view_depth(Vec3::ZERO) - 4.0).abs() < EPS);
    assert!((c.view_depth(Vec3::new(0.7, -0.3, 0.0)) - 4.0).abs() < EPS);
    // One unit behind the eye → −1 (culled by the renderer).
    assert!((c.view_depth(Vec3::new(0.0, 0.0, 5.0)) + 1.0).abs() < EPS);
    // At the eye plane → exactly 0 (also culled: ≤ near).
    assert!(c.view_depth(Vec3::new(0.0, 0.0, 4.0)).abs() < EPS);
}

#[test]
fn perspective_splat_extent_shrinks_inverse_with_depth() {
    let c = persp_cam();
    // Radius 0.2 at the target plane (depth 4) → NDC 0.2 (matches splat_ndc);
    // at depth 8 → NDC 0.1. Peak intensity is untouched by design (surface
    // brightness is distance-invariant), so flux falls as 1/d² automatically.
    assert!(c
        .splat_extent(Vec3::ZERO, 0.2)
        .abs_diff_eq(Vec2::new(0.2, 0.2), EPS));
    assert!(c
        .splat_extent(Vec3::ZERO, 0.2)
        .abs_diff_eq(c.splat_ndc(0.2), EPS));
    assert!(c
        .splat_extent(Vec3::new(0.0, 0.0, -4.0), 0.2)
        .abs_diff_eq(Vec2::new(0.1, 0.1), EPS));
}

#[test]
fn ortho_splat_extent_is_position_independent() {
    let c = Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(2.0, 1.0));
    for &w in &[Vec3::ZERO, Vec3::new(0.5, -0.3, 7.0)] {
        assert!(
            c.splat_extent(w, 0.5).abs_diff_eq(c.splat_ndc(0.5), EPS),
            "ortho splat_extent must equal splat_ndc at {w:?}"
        );
    }
}

#[test]
#[should_panic(expected = "near")]
fn perspective_rejects_near_at_or_past_distance() {
    let _ = Camera::perspective(
        Vec3::ZERO,
        Vec3::NEG_Z,
        Vec3::Y,
        Vec2::new(1.0, 1.0),
        4.0,
        4.0,
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
