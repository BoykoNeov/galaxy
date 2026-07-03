//! M6g swap gates: the world-space vertex-shader projection path must reproduce
//! the CPU-projected ortho renders, and the new perspective projection must obey
//! hand-derivable physics (inverse-square flux, behind-camera culling, finite
//! degenerate-depth output).
//!
//! GPU-gated like `invariants.rs`: needs a wgpu adapter, fails loudly without one.

use galaxy_render::camera::Camera;
use galaxy_render::render::{HdrImage, RenderConfig, Renderer};
use galaxy_renderprep::FrameData;
use glam::{Vec2, Vec3};

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for vertex-path gates")
}

/// The reference perspective rig for the physics gates: target at the origin,
/// looking down −Z with +Y up, half_extent (1,1) at the target plane, eye at
/// (0, 0, +4), near 0.01 — same hand-derivable geometry as `tests/camera.rs`.
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

/// Total RGB flux in the left / right half of the image (for two-particle
/// ratio gates: one splat per half, no overlap).
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

/// The same deterministic trig-varied scene as `invariants.rs` (no RNG).
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

fn centered_camera() -> Camera {
    Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0))
}

/// THE swap-equivalence gate: the world-space vertex-shader path must reproduce
/// the CPU-projected renderer on the deterministic ortho scene. The golden
/// numbers below were captured from the pre-M6g CPU-projection renderer (run
/// `capture_golden` below) — total flux to 1e-3 relative, probe pixels (the six
/// brightest, spread apart) to 1e-3 relative with a 1e-4 absolute floor. Every
/// pre-M6g movie must be reproducible; this pins it.
#[test]
fn ortho_vertex_path_matches_cpu_projected_golden() {
    const GOLDEN_FLUX: [f64; 3] = [954.6582147401833, 857.9963445594467, 882.8485137405329];
    const GOLDEN_PROBES: [(u32, u32, [f32; 4]); 6] = [
        (104, 82, [4.8903437, 0.11558055, 0.31550068, 0.98268366]),
        (110, 21, [4.8431454, 3.6236181, 0.5870012, 0.9864867]),
        (113, 113, [3.5628026, 4.5565963, 0.69929844, 0.93201035]),
        (87, 46, [0.55983, 1.5944897, 4.447469, 0.9207825]),
        (115, 12, [0.7877806, 4.41498, 3.8947272, 0.9368132]),
        (77, 29, [3.4902692, 4.29747, 4.2775927, 1.5346284]),
    ];

    let r = renderer();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let img = r
        .render_frame(&scene(40), &centered_camera(), &cfg)
        .unwrap();

    let flux = img.total_flux();
    for c in 0..3 {
        let rel = (flux[c] - GOLDEN_FLUX[c]).abs() / GOLDEN_FLUX[c];
        assert!(
            rel < 1e-3,
            "channel {c} flux drifted from CPU-projected golden: {} vs {} (rel {rel})",
            flux[c],
            GOLDEN_FLUX[c]
        );
    }
    for &(x, y, want) in &GOLDEN_PROBES {
        let got = img.pixel(x, y);
        for c in 0..4 {
            let tol = (1e-3 * want[c].abs()).max(1e-4);
            assert!(
                (got[c] - want[c]).abs() <= tol,
                "probe ({x},{y}) channel {c}: {} vs golden {}",
                got[c],
                want[c]
            );
        }
    }
}

/// Apparent flux follows the inverse-square law: two identical particles, one at
/// the target plane (depth 4), one a target-distance further (depth 8), placed
/// in opposite image halves. Peak surface intensity is fixed and screen size
/// shrinks ∝ 1/d, so integrated flux ratio = (8/4)² = 4 — real optics, not a
/// tuned attenuation factor. Both splats are well-resolved (≈19 px and ≈10 px
/// half-extents at 256², above the default sub-pixel clamp), so discretization
/// error is small: 2% tolerance.
#[test]
fn perspective_flux_follows_inverse_square() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 256,
        height: 256,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    // A at NDC (−0.5, 0), depth 4; B at NDC (+0.5, 0): lateral 1.0 at z = −4
    // projects to 1.0·4/8 = 0.5.
    let frame = frame_of(&[
        one(Vec3::new(-0.5, 0.0, 0.0), 0.15),
        one(Vec3::new(1.0, 0.0, -4.0), 0.15),
    ]);
    let img = r.render_frame(&frame, &persp_cam(), &cfg).unwrap();
    let (near_flux, far_flux) = half_fluxes(&img);
    assert!(near_flux > 0.0 && far_flux > 0.0, "a splat rendered black");
    let ratio = near_flux / far_flux;
    assert!(
        (ratio - 4.0).abs() / 4.0 < 0.02,
        "inverse-square violated: flux ratio {ratio}, expected 4"
    );
}

/// A particle behind the eye contributes nothing — the whole quad is culled, and
/// adding it changes the image by exactly zero flux.
#[test]
fn behind_camera_particle_contributes_nothing() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let front = frame_of(&[one(Vec3::new(0.2, -0.1, 0.0), 0.2)]);
    let with_behind = frame_of(&[
        one(Vec3::new(0.2, -0.1, 0.0), 0.2),
        one(Vec3::new(0.0, 0.0, 8.0), 0.2), // view depth −4: behind the eye
    ]);
    let a = r.render_frame(&front, &persp_cam(), &cfg).unwrap();
    let b = r.render_frame(&with_behind, &persp_cam(), &cfg).unwrap();
    let (fa, fb) = (a.total_flux(), b.total_flux());
    for c in 0..3 {
        assert!(fa[c] > 0.0, "front particle rendered black");
        let rel = (fb[c] - fa[c]).abs() / fa[c];
        assert!(
            rel < 1e-6,
            "behind-camera particle leaked flux: channel {c} {} vs {}",
            fb[c],
            fa[c]
        );
    }
}

/// Degenerate depths — at the eye plane (depth 0, a 1/z pole), just inside the
/// near plane, and exactly at near — must be culled cleanly: zero flux, every
/// pixel finite. No NaN/Inf may ever reach the accumulation buffer.
#[test]
fn near_and_degenerate_depths_are_culled_and_finite() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let frame = frame_of(&[
        one(Vec3::new(0.0, 0.0, 4.0), 0.2), // depth 0: at the eye (1/z pole)
        one(Vec3::new(0.0, 0.0, 3.995), 0.2), // depth 0.005 < near
        one(Vec3::new(0.0, 0.0, 3.99), 0.2), // depth 0.01 == near (≤ culls)
    ]);
    let img = r.render_frame(&frame, &persp_cam(), &cfg).unwrap();
    let flux = img.total_flux();
    assert!(
        flux.iter().all(|f| *f == 0.0),
        "near-culled particles leaked flux: {flux:?}"
    );
    assert!(
        img.pixels.iter().flatten().all(|c| c.is_finite()),
        "non-finite pixel from degenerate depth"
    );
}

/// The sub-pixel clamp preserves the inverse-square law: a splat whose projected
/// half-extent falls below `min_splat_px` is drawn at the clamp size with its
/// emission dimmed by (true/clamped)², so integrated flux stays physical — the
/// point-source regime. Reference at depth 4 (unclamped) vs a particle at depth
/// 40 whose ≈1.9 px extent clamps up to 8 px: flux ratio must still be
/// (40/4)² = 100, within 3%.
#[test]
fn subpixel_clamp_preserves_inverse_square_flux() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 256,
        height: 256,
        falloff: 6.0,
        min_splat_px: 8.0,
        ..RenderConfig::default()
    };
    // A at NDC (−0.5, 0), depth 4, extent 0.15 NDC = 19.2 px (unclamped).
    // B at depth 40 (z = −36), lateral 5.0 → NDC 5·4/40 = +0.5; extent
    // 0.15·4/40 = 0.015 NDC = 1.92 px < 8 px → clamped + dimmed.
    let frame = frame_of(&[
        one(Vec3::new(-0.5, 0.0, 0.0), 0.15),
        one(Vec3::new(5.0, 0.0, -36.0), 0.15),
    ]);
    let img = r.render_frame(&frame, &persp_cam(), &cfg).unwrap();
    let (near_flux, far_flux) = half_fluxes(&img);
    assert!(near_flux > 0.0, "reference splat rendered black");
    assert!(
        far_flux > 0.0,
        "clamped splat rendered black — clamp culled instead of dimming"
    );
    let ratio = near_flux / far_flux;
    assert!(
        (ratio - 100.0).abs() / 100.0 < 0.03,
        "clamp broke inverse-square: flux ratio {ratio}, expected 100"
    );
}

/// One-off capture harness: prints the golden numbers for the equivalence gate
/// from the CURRENT (pre-swap) CPU-projection renderer. Run with
/// `cargo test -p galaxy-render --test vertex_path -- --ignored --nocapture`.
#[test]
#[ignore]
fn capture_golden() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let img = r
        .render_frame(&scene(40), &centered_camera(), &cfg)
        .unwrap();
    println!("total_flux = {:?}", img.total_flux());
    // The 6 brightest pixels (by max RGB), spread out by a 8-px exclusion radius.
    let mut ranked: Vec<(u32, u32, f32)> = (0..cfg.height)
        .flat_map(|y| (0..cfg.width).map(move |x| (x, y)))
        .map(|(x, y)| {
            let p = img.pixel(x, y);
            (x, y, p[0].max(p[1]).max(p[2]))
        })
        .collect();
    ranked.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut picked: Vec<(u32, u32)> = Vec::new();
    for (x, y, _) in ranked {
        if picked
            .iter()
            .all(|&(px, py)| (px.abs_diff(x)).max(py.abs_diff(y)) > 8)
        {
            picked.push((x, y));
            if picked.len() == 6 {
                break;
            }
        }
    }
    for (x, y) in picked {
        println!("probe ({x:3},{y:3}) = {:?}", img.pixel(x, y));
    }
}
