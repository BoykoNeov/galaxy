//! Camera-rig gates (DESIGN.md M6d orbit/tilt, M6g dolly). Expectations are
//! hand-derived from the documented definitions in `rig.rs` — the spherical
//! basis convention, the σ = window/3 truncated-Gaussian envelope, the quintic
//! smootherstep — never read back from the code under test.

use galaxy_render::camera::{Camera, Projection, DEFAULT_MARGIN};
use galaxy_render::rig::{ease_in_out, smooth_envelope, CameraPath, RigError};
use glam::{Vec2, Vec3};
use proptest::prelude::*;

const EPS: f32 = 1e-5;

// --- ease_in_out -------------------------------------------------------------

#[test]
fn ease_endpoints_are_exact() {
    assert_eq!(ease_in_out(0.0), 0.0);
    assert_eq!(ease_in_out(1.0), 1.0);
}

#[test]
fn ease_midpoint_is_exactly_half() {
    // 6/32 − 15/16 + 10/8 = 0.5, and every term is a power-of-two multiple in
    // f32, so the quintic evaluates bit-exactly.
    assert_eq!(ease_in_out(0.5), 0.5);
}

#[test]
fn ease_clamps_outside_the_unit_interval() {
    assert_eq!(ease_in_out(-0.5), 0.0);
    assert_eq!(ease_in_out(1.5), 1.0);
}

#[test]
fn ease_end_derivatives_vanish() {
    // Forward/backward difference at the ends. The quintic gives
    // ease(h) ≈ 10h³, so the difference quotient is ~10h² — gate well above
    // that but far below any curve with a nonzero end slope.
    let h = 1e-3f32;
    assert!(
        ease_in_out(h) / h < 1e-3,
        "start not at rest: {}",
        ease_in_out(h) / h
    );
    assert!(
        (1.0 - ease_in_out(1.0 - h)) / h < 1e-3,
        "end not at rest: {}",
        (1.0 - ease_in_out(1.0 - h)) / h
    );
}

#[test]
fn ease_is_monotone_nondecreasing() {
    let mut prev = ease_in_out(0.0);
    for i in 1..=1000 {
        let v = ease_in_out(i as f32 / 1000.0);
        assert!(v >= prev, "decreasing at i={i}: {v} < {prev}");
        prev = v;
    }
}

#[test]
fn ease_is_symmetric_about_the_midpoint() {
    for i in 0..=100 {
        let u = i as f32 / 100.0;
        let s = ease_in_out(u) + ease_in_out(1.0 - u);
        assert!((s - 1.0).abs() < 1e-6, "asymmetric at u={u}: {s}");
    }
}

// --- smooth_envelope ----------------------------------------------------------

#[test]
fn envelope_of_empty_is_empty() {
    assert!(smooth_envelope(&[], 5).is_empty());
}

#[test]
fn envelope_with_zero_window_is_the_identity() {
    let raw = [3.0f32, 1.0, 4.0, 1.0, 5.0];
    assert_eq!(smooth_envelope(&raw, 0), raw.to_vec());
}

#[test]
fn envelope_of_a_constant_is_that_constant() {
    let raw = vec![7.25f32; 40];
    let env = smooth_envelope(&raw, 6);
    for (i, &e) in env.iter().enumerate() {
        // ≥ is exact by contract; ≤ allows only fp rounding headroom.
        assert!(e >= 7.25, "dipped below the constant at {i}: {e}");
        assert!(
            e <= 7.25 * (1.0 + 1e-5),
            "grew past the constant at {i}: {e}"
        );
    }
}

#[test]
fn envelope_matches_the_hand_computed_kernel() {
    // window = 1 ⇒ σ = 1/3, kernel taps k ∈ {−1, 0, 1} with weights
    // {e^{−4.5}, 1, e^{−4.5}} normalized. raw = [10,0,0,0,10] has moving max
    // [10,10,0,10,10]; index 2 mixes (g·10 + 1·0 + g·10)/(1+2g).
    let g = (-4.5f32).exp();
    let expected_mid = 20.0 * g / (1.0 + 2.0 * g);
    let env = smooth_envelope(&[10.0, 0.0, 0.0, 0.0, 10.0], 1);
    assert!(
        (env[2] - expected_mid).abs() < 1e-3,
        "index 2: got {}, hand value {expected_mid}",
        env[2]
    );
    // Ends: every tap (after edge clamping) sees the moving-max value 10.
    assert!((env[0] - 10.0).abs() < 1e-3, "index 0: {}", env[0]);
    assert!((env[4] - 10.0).abs() < 1e-3, "index 4: {}", env[4]);
}

#[test]
fn envelope_step_response_is_smooth() {
    // A 0→H step. The moving max shifts the step; the truncated Gaussian then
    // spreads it, so no per-frame jump can exceed H·max(kernel weight). The
    // test recomputes the documented kernel (σ = w/3, radius w) independently.
    let (h, w) = (100.0f32, 6usize);
    let sigma = w as f32 / 3.0;
    let weights: Vec<f32> = (-(w as i32)..=w as i32)
        .map(|k| (-0.5 * (k as f32 / sigma).powi(2)).exp())
        .collect();
    let g_max = weights.iter().cloned().fold(0.0f32, f32::max) / weights.iter().sum::<f32>();

    let mut raw = vec![0.0f32; 60];
    for r in raw.iter_mut().skip(30) {
        *r = h;
    }
    let env = smooth_envelope(&raw, w);
    for i in 1..env.len() {
        let step = (env[i] - env[i - 1]).abs();
        assert!(
            step <= h * g_max * 1.01 + 1e-4,
            "jump {step} at {i} exceeds H·g_max = {}",
            h * g_max
        );
    }
}

proptest! {
    #[test]
    fn envelope_never_crops_tighter_than_raw(
        raw in prop::collection::vec(0.0f32..1e6, 1..200),
        window in 0usize..20,
    ) {
        let env = smooth_envelope(&raw, window);
        prop_assert_eq!(env.len(), raw.len());
        let max = raw.iter().cloned().fold(0.0f32, f32::max);
        for (i, (&e, &r)) in env.iter().zip(&raw).enumerate() {
            // The envelope is a framing REQUIREMENT: ≥ raw exactly, and it never
            // exceeds the largest requirement it was built from (convexity).
            prop_assert!(e >= r, "cropped tighter than raw at {}: {} < {}", i, e, r);
            prop_assert!(e <= max * (1.0 + 1e-5), "overshot the track max at {}: {}", i, e);
        }
    }

    #[test]
    fn envelope_is_deterministic(
        raw in prop::collection::vec(0.0f32..1e6, 1..64),
        window in 0usize..12,
    ) {
        prop_assert_eq!(smooth_envelope(&raw, window), smooth_envelope(&raw, window));
    }
}

// --- CameraPath::fixed ---------------------------------------------------------

#[test]
fn fixed_path_is_bit_exactly_todays_camera() {
    // The pre-M6d pipeline framing, wrapped: every u must return it unchanged.
    let cam = Camera::face_on(Vec3::splat(-3.7), Vec3::splat(3.7), 16.0 / 9.0);
    let path = CameraPath::fixed(cam);
    for u in [0.0, 0.37, 0.5, 1.0] {
        assert_eq!(path.camera_at(u), cam, "static path drifted at u={u}");
    }
}

// --- CameraPath::orbit_tilt -----------------------------------------------------

/// The documented camera direction r̂(θ, φ) — re-derived here, independent of rig.rs.
fn r_hat(theta: f32, phi: f32) -> Vec3 {
    Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
}

fn test_path(radii: Vec<f32>, aspect: f32, margin: f32) -> CameraPath {
    CameraPath::orbit_tilt(
        Vec3::ZERO,
        (-90f32.to_radians(), 60f32.to_radians()),
        (55f32.to_radians(), 25f32.to_radians()),
        radii,
        margin,
        aspect,
    )
    .unwrap()
}

#[test]
fn orbit_tilt_rejects_invalid_parameters() {
    let ok_radii = || vec![1.0f32, 2.0];
    let build = |az: (f32, f32), tilt: (f32, f32), radii: Vec<f32>, margin: f32, aspect: f32| {
        CameraPath::orbit_tilt(Vec3::ZERO, az, tilt, radii, margin, aspect)
    };
    let a = (0.0, 1.0);
    assert_eq!(
        build(a, a, vec![], 0.05, 1.0).unwrap_err(),
        RigError::EmptyRadii
    );
    for (radii, why) in [
        (vec![1.0, 0.0], "zero radius"),
        (vec![-1.0], "negative radius"),
        (vec![f32::NAN], "NaN radius"),
        (vec![f32::INFINITY], "infinite radius"),
    ] {
        assert!(
            matches!(
                build(a, a, radii.clone(), 0.05, 1.0),
                Err(RigError::InvalidParam(_))
            ),
            "should reject {why}"
        );
    }
    for (margin, aspect, why) in [
        (-0.1, 1.0, "negative margin"),
        (f32::NAN, 1.0, "NaN margin"),
        (0.05, 0.0, "zero aspect"),
        (0.05, -1.0, "negative aspect"),
        (0.05, f32::NAN, "NaN aspect"),
    ] {
        assert!(
            matches!(
                build(a, a, ok_radii(), margin, aspect),
                Err(RigError::InvalidParam(_))
            ),
            "should reject {why}"
        );
    }
    for (az, tilt, why) in [
        ((f32::NAN, 1.0), a, "NaN azimuth start"),
        ((0.0, f32::INFINITY), a, "infinite azimuth end"),
        (a, (f32::NAN, 1.0), "NaN tilt start"),
        (a, (0.0, f32::NEG_INFINITY), "infinite tilt end"),
    ] {
        assert!(
            matches!(
                build(az, tilt, ok_radii(), 0.05, 1.0),
                Err(RigError::InvalidParam(_))
            ),
            "should reject {why}"
        );
    }
}

#[test]
fn face_on_pose_reproduces_the_historical_orientation() {
    // θ = −π/2, φ = 0 is documented to reproduce the face_on convention:
    // forward = −Z, right = +X, up = +Y (the +Y-up face-on movie framing).
    let path = CameraPath::orbit_tilt(
        Vec3::ZERO,
        (-90f32.to_radians(), -90f32.to_radians()),
        (0.0, 0.0),
        vec![2.0],
        DEFAULT_MARGIN,
        16.0 / 9.0,
    )
    .unwrap();
    let c = path.camera_at(0.5);
    assert!(c.forward.abs_diff_eq(Vec3::NEG_Z, EPS), "{:?}", c.forward);
    assert!(c.right.abs_diff_eq(Vec3::X, EPS), "{:?}", c.right);
    assert!(c.up.abs_diff_eq(Vec3::Y, EPS), "{:?}", c.up);
    // half_extent: r·(1+margin) on the short (y) axis, widened by aspect on x.
    let eu = 2.0 * (1.0 + DEFAULT_MARGIN);
    assert!((c.half_extent.y - eu).abs() < 1e-4, "{:?}", c.half_extent);
    assert!(
        (c.half_extent.x - eu * 16.0 / 9.0).abs() < 1e-4,
        "{:?}",
        c.half_extent
    );
    assert_eq!(c.target, Vec3::ZERO);
}

#[test]
fn orbit_tilt_endpoints_hit_the_requested_angles() {
    // Ease is exact at the endpoints, so the view axis at u = 0 / u = 1 must be
    // −r̂ of the requested (θ, φ) — hand-derived from the documented convention.
    let path = test_path(vec![3.0], 1.0, 0.0);
    let (th0, ph0) = (-90f32.to_radians(), 55f32.to_radians());
    let (th1, ph1) = (60f32.to_radians(), 25f32.to_radians());
    assert!(
        path.camera_at(0.0)
            .forward
            .abs_diff_eq(-r_hat(th0, ph0), EPS),
        "u=0: {:?}",
        path.camera_at(0.0).forward
    );
    assert!(
        path.camera_at(1.0)
            .forward
            .abs_diff_eq(-r_hat(th1, ph1), EPS),
        "u=1: {:?}",
        path.camera_at(1.0).forward
    );
}

#[test]
fn radius_track_is_sampled_by_linear_interpolation_in_raw_u() {
    // radii [1,3,2] over u ∈ [0,1], aspect 1, margin 0: half_extent.y IS the
    // sampled radius. Quarter points land exactly between track entries — and
    // easing must NOT apply to the radius (only angles ease), so u = 0.25
    // samples lerp(1,3,0.5) = 2, not lerp at ease(0.25) = 0.103515625.
    let path = test_path(vec![1.0, 3.0, 2.0], 1.0, 0.0);
    for (u, expected) in [(0.0, 1.0), (0.25, 2.0), (0.5, 3.0), (0.75, 2.5), (1.0, 2.0)] {
        let hy = path.camera_at(u).half_extent.y;
        assert!(
            (hy - expected).abs() < 1e-6,
            "u={u}: half_extent.y {hy}, expected {expected}"
        );
    }
    // A single-entry track is a constant radius.
    let single = test_path(vec![4.0], 1.0, 0.0);
    assert!((single.camera_at(0.7).half_extent.y - 4.0).abs() < 1e-6);
}

#[test]
fn per_frame_steps_stay_under_the_smoothness_budget() {
    // n samples of an eased sweep: the per-step angular change is bounded by
    // (|Δθ| + |Δφ|)·max ease slope/(n−1), with slope 15/8 for the quintic. The
    // radius track (m entries, linear in u) is bounded by max|Δradii|·(m−1)/(n−1).
    let radii = vec![5.0f32, 6.0, 4.0, 5.0];
    let path = test_path(radii, 1.0, 0.0);
    let (d_theta, d_phi) = (150f32.to_radians(), 30f32.to_radians());
    let n = 97usize;
    let angle_budget = (d_theta + d_phi) * (15.0 / 8.0) / (n - 1) as f32;
    let radius_budget = 2.0 * 3.0 / (n - 1) as f32;

    let mut prev = path.camera_at(0.0);
    for i in 1..n {
        let u = i as f32 / (n - 1) as f32;
        let c = path.camera_at(u);
        let dot = f64::from(prev.forward.x) * f64::from(c.forward.x)
            + f64::from(prev.forward.y) * f64::from(c.forward.y)
            + f64::from(prev.forward.z) * f64::from(c.forward.z);
        let angle = dot.clamp(-1.0, 1.0).acos() as f32;
        assert!(
            angle <= angle_budget * 1.02 + 2e-3,
            "view axis jumped {angle} rad at sample {i} (budget {angle_budget})"
        );
        let dr = (c.half_extent.y - prev.half_extent.y).abs();
        assert!(
            dr <= radius_budget * 1.02 + 1e-5,
            "framing radius jumped {dr} at sample {i} (budget {radius_budget})"
        );
        prev = c;
    }
}

#[test]
fn orbit_tilt_is_deterministic() {
    let a = test_path(vec![1.0, 2.0, 1.5], 16.0 / 9.0, 0.05);
    let b = test_path(vec![1.0, 2.0, 1.5], 16.0 / 9.0, 0.05);
    for i in 0..=50 {
        let u = i as f32 / 50.0;
        assert_eq!(a.camera_at(u), b.camera_at(u), "differs at u={u}");
    }
}

proptest! {
    #[test]
    fn orbit_tilt_basis_is_orthonormal_everywhere(
        th0 in -10.0f32..10.0, th1 in -10.0f32..10.0,
        ph0 in 0.0f32..std::f32::consts::PI, ph1 in 0.0f32..std::f32::consts::PI,
        u in 0.0f32..1.0,
    ) {
        let path = CameraPath::orbit_tilt(
            Vec3::new(1.0, -2.0, 0.5),
            (th0, th1),
            (ph0, ph1),
            vec![1.0, 2.0, 3.0],
            0.05,
            16.0 / 9.0,
        ).unwrap();
        let c = path.camera_at(u);
        for (v, name) in [(c.right, "right"), (c.up, "up"), (c.forward, "forward")] {
            prop_assert!((v.length() - 1.0).abs() < EPS, "{} not unit: {:?}", name, v);
        }
        prop_assert!(c.right.dot(c.up).abs() < EPS);
        prop_assert!(c.right.dot(c.forward).abs() < EPS);
        prop_assert!(c.up.dot(c.forward).abs() < EPS);
        // Handedness matches Camera::orthographic's own convention
        // (right = forward × up ⇒ right × up = −forward).
        prop_assert!(c.right.cross(c.up).abs_diff_eq(-c.forward, 1e-4));
        // Aspect is preserved at every sample — a world circle stays a screen circle.
        let ratio = c.half_extent.x / c.half_extent.y;
        prop_assert!((ratio - 16.0 / 9.0).abs() < 1e-4, "aspect drifted: {}", ratio);
        // The view axis always points back at the target from r̂(θ(u), φ(u)).
        prop_assert!(c.target.abs_diff_eq(Vec3::new(1.0, -2.0, 0.5), EPS));
    }
}

#[test]
fn splat_isotropy_under_aspect() {
    // With the aspect-correct half-extent, a world-space radius maps to an NDC
    // extent whose x:y ratio is 1/aspect — isotropic on screen after the
    // viewport's pixel aspect. Same law the static camera obeys.
    let path = test_path(vec![2.0], 2.0, 0.0);
    let s: Vec2 = path.camera_at(0.5).splat_ndc(0.5);
    assert!((s.y / s.x - 2.0).abs() < 1e-4, "splat anisotropic: {s:?}");
}

// --- dolly (M6g) ---------------------------------------------------------------
//
// A fixed-direction perspective dolly: the eye approaches the target along
// r̂(θ, φ) (the same spherical convention as orbit/tilt), the eye distance eases
// from start to end, and the vertical field of view is CONSTANT — a physical
// camera move, not a zoom, so `half_extent.y = distance·tan(fov_y/2)` at every u.
// Gates use fov_y = π/2 (tan = 1: half_extent.y == distance, hand-checkable).

fn dolly_path() -> CameraPath {
    CameraPath::dolly(
        Vec3::ZERO,
        -std::f32::consts::FRAC_PI_2,
        0.0,
        (10.0, 2.0),
        std::f32::consts::FRAC_PI_2,
        0.1,
        2.0,
    )
    .unwrap()
}

/// The perspective parameters of a path's camera at `u` (distance, near),
/// panicking if the camera is not perspective — every dolly camera must be.
fn persp_params(path: &CameraPath, u: f32) -> (f32, f32) {
    match path.camera_at(u).projection {
        Projection::Perspective { distance, near } => (distance, near),
        other => panic!("dolly camera at u={u} is not perspective: {other:?}"),
    }
}

#[test]
fn dolly_endpoints_hit_the_requested_distances() {
    let path = dolly_path();
    let (d0, n0) = persp_params(&path, 0.0);
    let (d1, n1) = persp_params(&path, 1.0);
    assert!((d0 - 10.0).abs() < EPS, "start distance {d0}");
    assert!((d1 - 2.0).abs() < EPS, "end distance {d1}");
    // With fov_y = π/2, half_extent.y == distance exactly.
    assert!(path
        .camera_at(0.0)
        .half_extent
        .abs_diff_eq(Vec2::new(20.0, 10.0), 1e-3));
    assert!(path
        .camera_at(1.0)
        .half_extent
        .abs_diff_eq(Vec2::new(4.0, 2.0), 1e-3));
    // The near plane rides along unchanged.
    assert!((n0 - 0.1).abs() < EPS && (n1 - 0.1).abs() < EPS);
}

#[test]
fn dolly_field_of_view_is_constant() {
    // A dolly is a camera MOVE: fov fixed, so half_extent.y/distance =
    // tan(fov_y/2) = 1 at every sample (this is what separates it from a zoom).
    let path = dolly_path();
    for u in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] {
        let c = path.camera_at(u);
        let (d, _) = persp_params(&path, u);
        assert!(
            (c.half_extent.y / d - 1.0).abs() < 1e-4,
            "fov drifted at u={u}: he.y {} vs distance {d}",
            c.half_extent.y
        );
        assert!(
            (c.half_extent.x / c.half_extent.y - 2.0).abs() < 1e-4,
            "aspect drifted at u={u}"
        );
    }
}

#[test]
fn dolly_distance_eases_like_the_angles() {
    // Same quintic ease as orbit/tilt: midpoint is the exact arithmetic mean
    // (ease(1/2) = 1/2 bit-exactly), approach is monotone for start > end.
    let path = dolly_path();
    let (dm, _) = persp_params(&path, 0.5);
    assert!((dm - 6.0).abs() < 1e-4, "midpoint distance {dm}");
    let mut prev = persp_params(&path, 0.0).0;
    for i in 1..=100 {
        let (d, _) = persp_params(&path, i as f32 / 100.0);
        assert!(d <= prev + EPS, "approach not monotone at i={i}");
        prev = d;
    }
}

#[test]
fn dolly_basis_matches_the_orbit_convention() {
    // The dolly direction uses the SAME documented spherical basis as
    // orbit/tilt (which is pinned against hand-derived poses above): a dolly
    // at fixed (θ, φ) must produce the identical camera orientation as a
    // degenerate orbit parked at those angles.
    let target = Vec3::new(1.0, 2.0, 3.0);
    let d = CameraPath::dolly(target, 0.7, 1.1, (8.0, 3.0), 1.0, 0.05, 1.5).unwrap();
    let o = CameraPath::orbit_tilt(target, (0.7, 0.7), (1.1, 1.1), vec![1.0], 0.0, 1.5).unwrap();
    for u in [0.0, 0.3, 0.5, 1.0] {
        let cd = d.camera_at(u);
        let co = o.camera_at(u);
        assert!(cd.right.abs_diff_eq(co.right, EPS), "right at u={u}");
        assert!(cd.up.abs_diff_eq(co.up, EPS), "up at u={u}");
        assert!(cd.forward.abs_diff_eq(co.forward, EPS), "forward at u={u}");
        assert!(cd.target.abs_diff_eq(target, EPS), "target at u={u}");
    }
}

#[test]
fn dolly_face_on_pose_reproduces_the_historical_orientation() {
    // θ = −π/2, φ = 0 is the pinned face-on pose: right +X, up +Y, forward −Z.
    let c = dolly_path().camera_at(0.0);
    assert!(c.right.abs_diff_eq(Vec3::X, EPS), "{:?}", c.right);
    assert!(c.up.abs_diff_eq(Vec3::Y, EPS), "{:?}", c.up);
    assert!(c.forward.abs_diff_eq(Vec3::NEG_Z, EPS), "{:?}", c.forward);
}

#[test]
fn dolly_rejects_invalid_parameters() {
    use std::f32::consts::PI;
    let ok = (Vec3::ZERO, -1.0f32, 0.5f32);
    for (dist, fov, near, aspect, why) in [
        ((0.0, 2.0), 1.0, 0.1, 1.0, "zero start distance"),
        ((10.0, -2.0), 1.0, 0.1, 1.0, "negative end distance"),
        ((10.0, 2.0), 0.0, 0.1, 1.0, "zero fov"),
        ((10.0, 2.0), PI, 0.1, 1.0, "fov at π (tan blows up)"),
        ((10.0, 2.0), 1.0, 2.0, 1.0, "near at the closest approach"),
        ((10.0, 2.0), 1.0, -0.1, 1.0, "negative near"),
        ((10.0, 2.0), 1.0, 0.1, 0.0, "zero aspect"),
        ((f32::NAN, 2.0), 1.0, 0.1, 1.0, "non-finite distance"),
    ] {
        let r = CameraPath::dolly(ok.0, ok.1, ok.2, dist, fov, near, aspect);
        assert!(r.is_err(), "should reject: {why}");
    }
    assert!(
        CameraPath::dolly(ok.0, f32::INFINITY, ok.2, (10.0, 2.0), 1.0, 0.1, 1.0).is_err(),
        "should reject: non-finite azimuth"
    );
    // Dolly OUT (end > start) is legitimate — retreat shots are real shots.
    assert!(CameraPath::dolly(ok.0, ok.1, ok.2, (2.0, 10.0), 1.0, 0.1, 1.0).is_ok());
}
