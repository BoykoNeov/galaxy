//! M7e gates (plan D9): volumetric gas raymarch + full star attenuation.
//!
//! CPU gates pin the march rules (`render::volume`) against closed-form
//! oracles — uniform-slab transmittance `T = exp(−κρL)`, uniform-slab radiance
//! `(j/κ)(1−e^{−κρL})`, hand-derived corner-pixel rays. GPU gates hold the
//! shaders to the CPU mirror within f32 tolerance and pin the load-bearing
//! regression: **gas-off is bit-compatible with the landed M6g golden**.
//!
//! GPU tests need a wgpu adapter and fail loudly without one (the
//! `invariants.rs` convention).

use galaxy_render::camera::Camera;
use galaxy_render::render::{HdrImage, RenderConfig, Renderer};
use galaxy_render::volume::{
    march_gas, ray_for_pixel, render_gas_cpu, star_transmittance, step_size, GasFrame, GasLook,
    EXIT_TRANSMITTANCE,
};
use galaxy_renderprep::{FrameData, GasGrid};
use glam::{Vec2, Vec3};

// ---------- helpers ----------

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for volume gates")
}

/// A uniform-density grid: the analytic-slab workhorse.
fn uniform_grid(rho: f32, bounds_min: Vec3, bounds_max: Vec3, dims: [u32; 3]) -> GasGrid {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    GasGrid {
        dims,
        bounds_min,
        bounds_max,
        data: vec![rho; n],
    }
}

/// A deterministic non-uniform grid (trig pattern, strictly positive).
fn pattern_grid(bounds_min: Vec3, bounds_max: Vec3, dims: [u32; 3], phase: f32) -> GasGrid {
    let mut data = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let v = 0.45
                    + 0.35
                        * (0.7 * ix as f32 + 1.3 * iy as f32 + phase).sin()
                        * (0.5 * iz as f32 + 0.4).cos();
                data.push(v);
            }
        }
    }
    GasGrid {
        dims,
        bounds_min,
        bounds_max,
        data,
    }
}

/// The reference absorption slab: z ∈ [−1, 0], generous lateral extent, uniform
/// ρ = 0.5. With κ = 0.6 a full crossing has optical depth τ = κρL = 0.3.
const SLAB_RHO: f32 = 0.5;
const SLAB_KAPPA: f32 = 0.6;
fn slab_grid() -> GasGrid {
    uniform_grid(
        SLAB_RHO,
        Vec3::new(-2.0, -2.0, -1.0),
        Vec3::new(2.0, 2.0, 0.0),
        [16, 16, 32],
    )
}

fn absorb_look(kappa: f32) -> GasLook {
    GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 0.0,
        opacity: kappa,
        scatter: None,
    }
}

/// Ortho camera at the scene center looking down −Z (+Z side), ±1 view window —
/// the golden-gate framing.
fn centered_camera() -> Camera {
    Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0))
}

/// The reference perspective rig from `tests/vertex_path.rs`: eye at (0,0,+4).
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

/// The deterministic trig-varied golden scene from `tests/vertex_path.rs`.
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

/// Total RGB flux in the top / bottom half of the image (two-star ordering
/// gates place one star per half).
fn half_fluxes_vertical(img: &HdrImage) -> (f64, f64) {
    let (mut top, mut bot) = (0.0f64, 0.0f64);
    for y in 0..img.height {
        for x in 0..img.width {
            let p = img.pixel(x, y);
            let s = p[0] as f64 + p[1] as f64 + p[2] as f64;
            if y < img.height / 2 {
                top += s;
            } else {
                bot += s;
            }
        }
    }
    (top, bot)
}

fn assert_vec3_close(got: Vec3, want: Vec3, tol: f32, what: &str) {
    assert!(
        (got - want).length() <= tol,
        "{what}: got {got:?}, want {want:?}"
    );
}

// ---------- CPU gates: ray generation hand oracles ----------

/// Orthographic ray generation at hand-derived pixels: pixel centers, top-left
/// origin, NDC y up, origin on the target plane, direction = forward. Also a
/// non-square case (8×4 image, half_extent (2,1)) so the two axes can't be
/// swapped silently.
#[test]
fn ray_for_pixel_ortho_hand_oracle() {
    let cam = centered_camera();
    // 4×4, half_extent (1,1): pixel (0,0) center (0.5,0.5) → NDC (−0.75, +0.75).
    let (o, d) = ray_for_pixel(&cam, 4, 4, 0, 0);
    assert_vec3_close(o, Vec3::new(-0.75, 0.75, 0.0), 1e-6, "ortho (0,0) origin");
    assert_vec3_close(d, Vec3::new(0.0, 0.0, -1.0), 1e-6, "ortho (0,0) dir");
    // Opposite corner (3,3) → NDC (+0.75, −0.75).
    let (o, d) = ray_for_pixel(&cam, 4, 4, 3, 3);
    assert_vec3_close(o, Vec3::new(0.75, -0.75, 0.0), 1e-6, "ortho (3,3) origin");
    assert_vec3_close(d, Vec3::new(0.0, 0.0, -1.0), 1e-6, "ortho (3,3) dir");
    // Interior pixel (1,2) center (1.5, 2.5) → NDC (−0.25, −0.25).
    let (o, _) = ray_for_pixel(&cam, 4, 4, 1, 2);
    assert_vec3_close(o, Vec3::new(-0.25, -0.25, 0.0), 1e-6, "ortho (1,2) origin");

    // Non-square: 8×4 image, half_extent (2,1). Pixel (0,0) center (0.5,0.5):
    // NDC (0.5/4−1, 1−0.5/2) = (−0.875, +0.75) → world (−1.75, +0.75, 0).
    let wide = Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(2.0, 1.0));
    let (o, _) = ray_for_pixel(&wide, 8, 4, 0, 0);
    assert_vec3_close(o, Vec3::new(-1.75, 0.75, 0.0), 1e-6, "wide ortho origin");
}

/// Perspective ray generation: origin at the eye, direction through the
/// pixel's point on the target plane (hand-built and normalized independently).
#[test]
fn ray_for_pixel_perspective_hand_oracle() {
    let cam = persp_cam(); // eye (0,0,+4), target plane z = 0, half_extent (1,1)
    let eye = Vec3::new(0.0, 0.0, 4.0);
    // 4×4: pixel (0,0) → target-plane point (−0.75, +0.75, 0).
    let (o, d) = ray_for_pixel(&cam, 4, 4, 0, 0);
    assert_vec3_close(o, eye, 1e-6, "persp (0,0) origin");
    let want = (Vec3::new(-0.75, 0.75, 0.0) - eye).normalize();
    assert_vec3_close(d, want, 1e-6, "persp (0,0) dir");
    assert!((d.length() - 1.0).abs() < 1e-6, "persp dir must be unit");
    // Pixel (3,0) → (+0.75, +0.75, 0).
    let (_, d) = ray_for_pixel(&cam, 4, 4, 3, 0);
    let want = (Vec3::new(0.75, 0.75, 0.0) - eye).normalize();
    assert_vec3_close(d, want, 1e-6, "persp (3,0) dir");
}

// ---------- CPU gates: analytic uniform slab ----------

/// Star transmittance through the uniform slab equals the closed form
/// exp(−κρL): behind the slab L = 1, inside it L = the remaining depth, on the
/// camera side exactly 1.0. Constant ρ makes the step sum exact up to f32
/// accumulation (≲ n·ulp), so the tolerance is 1e-5 relative.
#[test]
fn star_transmittance_uniform_slab_analytic() {
    let g = slab_grid();
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look: absorb_look(SLAB_KAPPA),
    };
    let tau_full = SLAB_KAPPA * SLAB_RHO; // L = 1

    // Ortho camera on the +Z side: a star behind the slab crosses all of it.
    let cam = centered_camera();
    let t = star_transmittance(&gas, &cam, Vec3::new(0.2, 0.1, -2.0));
    let want = (-tau_full).exp();
    assert!(
        (t - want).abs() / want < 1e-5,
        "behind-slab T {t} vs analytic {want}"
    );
    // A star on the camera side of the slab is unattenuated — exactly 1.
    let t = star_transmittance(&gas, &cam, Vec3::new(0.2, 0.1, 0.5));
    assert_eq!(t, 1.0, "camera-side star must have T exactly 1");
    // A star inside the slab crosses only the depth in front of it (0.25).
    let t = star_transmittance(&gas, &cam, Vec3::new(0.0, 0.0, -0.25));
    let want = (-SLAB_KAPPA * SLAB_RHO * 0.25).exp();
    assert!(
        (t - want).abs() / want < 1e-5,
        "inside-slab T {t} vs analytic {want}"
    );

    // Perspective (eye at (0,0,+4)): same crossings along the star→eye segment.
    let cam = persp_cam();
    let t = star_transmittance(&gas, &cam, Vec3::new(0.0, 0.0, -2.0));
    let want = (-tau_full).exp();
    assert!(
        (t - want).abs() / want < 1e-5,
        "persp behind-slab T {t} vs analytic {want}"
    );
    let t = star_transmittance(&gas, &cam, Vec3::new(0.5, -0.3, 1.5));
    assert_eq!(t, 1.0, "persp camera-side star must have T exactly 1");
}

/// Gas radiance through the uniform slab equals (j/κ)(1−e^{−κρL})·color within
/// the first-order quadrature bound. Slab depth L = 1 at 32 cells → step
/// Δs = 1/64, per-step optical depth dτ = κρΔs ≈ 0.0047; the emit-then-
/// attenuate rule overshoots by dτ/(1−e^{−dτ})−1 ≈ 0.23%, so 0.5% relative is
/// a 2× margin. The returned transmittance has no quadrature error for
/// constant ρ (τ sums exactly) — 1e-5.
#[test]
fn march_uniform_slab_radiance_analytic() {
    let g = slab_grid();
    let look = GasLook {
        color: [1.0, 0.5, 0.25],
        emissivity: 0.9,
        opacity: SLAB_KAPPA,
        scatter: None,
    };
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look,
    };
    assert!(
        (step_size(&g, &g) - 1.0 / 64.0).abs() < 1e-9,
        "slab step must be half the smallest cell edge (1/64)"
    );

    let (c, t) = march_gas(
        &gas,
        None,
        Vec3::new(0.3, -0.2, 5.0),
        Vec3::new(0.0, 0.0, -1.0),
        f32::NEG_INFINITY,
    );
    let tau = SLAB_KAPPA * SLAB_RHO; // L = 1
    let base = look.emissivity / look.opacity * (1.0 - (-tau).exp());
    for (k, &ck) in c.iter().enumerate() {
        let want = base * look.color[k];
        assert!(
            (ck - want).abs() / want < 5e-3,
            "channel {k} radiance {ck} vs analytic {want}"
        );
    }
    let want_t = (-tau).exp();
    assert!(
        (t - want_t).abs() / want_t < 1e-5,
        "slab transmittance {t} vs analytic {want_t}"
    );
}

/// κ = 0 is the emission-only limit: T stays exactly 1 and the radiance is the
/// plain path integral j·ρ·L·color — exact for constant ρ (no exponentials in
/// play), 1e-5 relative for f32 accumulation.
#[test]
fn march_emission_only_kappa_zero_exact() {
    let g = slab_grid();
    let look = GasLook {
        color: [0.8, 1.0, 0.6],
        emissivity: 0.9,
        opacity: 0.0,
        scatter: None,
    };
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look,
    };
    let (c, t) = march_gas(
        &gas,
        None,
        Vec3::new(0.0, 0.0, 5.0),
        Vec3::new(0.0, 0.0, -1.0),
        f32::NEG_INFINITY,
    );
    assert_eq!(t, 1.0, "κ = 0 must leave transmittance exactly 1");
    for (k, &ck) in c.iter().enumerate() {
        let want = look.emissivity * SLAB_RHO * 1.0 * look.color[k];
        assert!(
            (ck - want).abs() / want < 1e-5,
            "channel {k} emission {ck} vs exact {want}"
        );
    }
}

/// High optical depth exercises the early exit: τ = 15 across the box, so the
/// march crosses T < EXIT_TRANSMITTANCE ≈ e^{−9.2} well inside. The combined
/// error bound is documented: quadrature dτ/(1−e^{−dτ})−1 ≈ 0.73% at
/// dτ = 15/1024, plus the exit truncation ≤ EXIT_TRANSMITTANCE·(j/κ)
/// (relative 1e-4) — gate at 1.5% relative, ≈ 2× margin.
#[test]
fn march_high_tau_early_exit_bounded() {
    let g = uniform_grid(
        1.0,
        Vec3::new(-1.0, -1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0),
        [1, 1, 512],
    );
    let look = GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 2.0,
        opacity: 15.0,
        scatter: None,
    };
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look,
    };
    let (c, t) = march_gas(
        &gas,
        None,
        Vec3::new(0.1, 0.2, -3.0),
        Vec3::new(0.0, 0.0, 1.0),
        f32::NEG_INFINITY,
    );
    assert!(
        t <= EXIT_TRANSMITTANCE,
        "τ = 15 march must end below the exit threshold, got T = {t}"
    );
    let want = look.emissivity / look.opacity * (1.0 - (-15.0f32).exp());
    for (k, &ck) in c.iter().enumerate() {
        assert!(
            (ck - want).abs() / want < 1.5e-2,
            "channel {k} saturated radiance {ck} vs analytic {want}"
        );
    }
}

/// Endpoint-mix plumbing: at u = 0 the march must be bit-identical whatever
/// grid1 holds (and symmetrically at u = 1) — the two-product lerp identity
/// `sample_mix` already pins, carried through the whole march. Same-bounds
/// grids so the union AABB and step count cannot differ.
#[test]
fn march_mix_endpoints_bit_exact() {
    let bmin = Vec3::splat(-1.0);
    let bmax = Vec3::splat(1.0);
    let a = pattern_grid(bmin, bmax, [6, 6, 6], 0.0);
    let b = pattern_grid(bmin, bmax, [6, 6, 6], 2.1);
    let look = GasLook {
        color: [1.0, 0.9, 0.8],
        emissivity: 1.3,
        opacity: 0.7,
        scatter: None,
    };
    let origin = Vec3::new(0.1, 0.05, 3.0);
    let dir = Vec3::new(0.4, -0.3, -1.0).normalize();

    let run = |g0: &GasGrid, g1: &GasGrid, u: f32| {
        march_gas(
            &GasFrame {
                grid0: g0,
                grid1: g1,
                mix: u,
                lights: &[],
                look,
            },
            None,
            origin,
            dir,
            f32::NEG_INFINITY,
        )
    };
    assert_eq!(
        run(&a, &b, 0.0),
        run(&a, &a, 0.0),
        "u = 0 must ignore grid1"
    );
    assert_eq!(
        run(&a, &b, 1.0),
        run(&b, &b, 1.0),
        "u = 1 must ignore grid0"
    );
}

/// Doubling the emissivity doubles every radiance channel EXACTLY (scaling by
/// 2 is exact in binary floating point and κ is untouched), and leaves the
/// transmittance bit-identical — emission linearity at the strongest possible
/// tolerance.
#[test]
fn march_emissivity_linear_exact() {
    let g = pattern_grid(Vec3::splat(-1.0), Vec3::splat(1.0), [6, 6, 6], 0.0);
    let origin = Vec3::new(-0.2, 0.15, 3.0);
    let dir = Vec3::new(-0.1, 0.25, -1.0).normalize();
    let mk = |e: f32| GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look: GasLook {
            color: [0.9, 0.5, 0.3],
            emissivity: e,
            opacity: 0.8,
            scatter: None,
        },
    };
    let (c1, t1) = march_gas(&mk(0.7), None, origin, dir, f32::NEG_INFINITY);
    let (c2, t2) = march_gas(&mk(1.4), None, origin, dir, f32::NEG_INFINITY);
    assert!(c1.iter().all(|&v| v > 0.0), "ray must cross the grid");
    for k in 0..3 {
        assert_eq!(
            c2[k],
            2.0 * c1[k],
            "channel {k}: 2× emissivity must double radiance exactly"
        );
    }
    assert_eq!(t1, t2, "transmittance must not depend on emissivity");
}

/// A 1×1×1 grid is the degenerate sampling case (trilinear collapses to the
/// single cell value everywhere inside). The transmittance is still exact
/// (constant ρ); the radiance quadrature is deliberately coarse here — two
/// steps at dτ = 0.4 overshoot by dτ/(1−e^{−dτ})−1 ≈ 21% — so the gate only
/// brackets it (documented; real grids are far finer than their support).
#[test]
fn march_single_cell_grid() {
    let g = uniform_grid(2.0, Vec3::splat(-0.5), Vec3::splat(0.5), [1, 1, 1]);
    let look = GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 1.0,
        opacity: 0.4,
        scatter: None,
    };
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look,
    };
    let (c, t) = march_gas(
        &gas,
        None,
        Vec3::new(0.0, 0.0, 2.0),
        Vec3::new(0.0, 0.0, -1.0),
        f32::NEG_INFINITY,
    );
    let tau: f32 = 0.4 * 2.0; // κρL, L = 1
    let want_t = (-tau).exp();
    assert!(
        (t - want_t).abs() / want_t < 1e-5,
        "single-cell T {t} vs analytic {want_t}"
    );
    let exact = 1.0 / 0.4 * (1.0 - (-tau).exp());
    assert!(
        c[0] >= exact && c[0] < exact * 1.25,
        "single-cell radiance {} outside documented coarse bracket of {exact}",
        c[0]
    );
}

// ---------- GPU gates ----------

/// THE regression gate: gas-off must be bit-compatible with the landed M6g
/// golden — `render_frame_with_gas(None)` AND a present-but-inert gas frame
/// (κ = 0, emissivity = 0: star T ≡ exp(0) = 1, ×1 is bit-exact; the gas pass
/// adds exact zeros) both reproduce the CPU-projection golden flux and probe
/// pixels captured before M6g.
#[test]
fn gas_off_matches_m6g_golden() {
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
    let frame = scene(40);
    let inert_grid = uniform_grid(0.7, Vec3::splat(-1.5), Vec3::splat(1.5), [8, 8, 8]);
    let inert = GasFrame {
        grid0: &inert_grid,
        grid1: &inert_grid,
        mix: 0.0,
        lights: &[],
        look: GasLook {
            color: [1.0, 1.0, 1.0],
            emissivity: 0.0,
            opacity: 0.0,
            scatter: None,
        },
    };

    for (what, gas) in [("gas=None", None), ("κ=0, j=0", Some(&inert))] {
        let img = r
            .render_frame_with_gas(&frame, gas, &centered_camera(), &cfg)
            .unwrap();
        let flux = img.total_flux();
        for c in 0..3 {
            let rel = (flux[c] - GOLDEN_FLUX[c]).abs() / GOLDEN_FLUX[c];
            assert!(
                rel < 1e-3,
                "{what}: channel {c} flux drifted from golden: {} vs {} (rel {rel})",
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
                    "{what}: probe ({x},{y}) channel {c}: {} vs golden {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

/// GPU star attenuation through the uniform slab: with emissivity 0 the gas
/// adds no light, so the total-flux ratio (with gas / without) IS the per-star
/// transmittance exp(−κρL). The whole image scales by one T value, so the
/// ratio is tight: 1e-4 relative (GPU exp + f32 chord clipping; an off-by-a-
/// cell path-length bug would show at the ~1% level).
#[test]
fn gpu_star_attenuation_matches_slab_transmittance() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let frame = frame_of(&[one(Vec3::new(0.0, 0.0, -2.0), 0.2)]);
    let g = slab_grid();
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look: absorb_look(SLAB_KAPPA),
    };
    let cam = centered_camera();
    let base = r.render_frame_with_gas(&frame, None, &cam, &cfg).unwrap();
    let dimmed = r
        .render_frame_with_gas(&frame, Some(&gas), &cam, &cfg)
        .unwrap();
    let want = (-(SLAB_KAPPA * SLAB_RHO) as f64).exp(); // τ = κρL, L = 1
    let (f0, f1) = (base.total_flux(), dimmed.total_flux());
    for c in 0..3 {
        assert!(f0[c] > 0.0, "unattenuated star rendered black");
        let ratio = f1[c] / f0[c];
        assert!(
            (ratio - want).abs() / want < 1e-4,
            "channel {c}: flux ratio {ratio} vs analytic transmittance {want}"
        );
    }
}

/// Two-star depth ordering (the plan gate verbatim): the star on the camera
/// side of the slab is unattenuated, the star behind it is dimmed by the full
/// slab — and looking from the other side swaps the roles. Stars are separated
/// vertically (the up axis survives the camera flip; the right axis doesn't).
#[test]
fn gpu_two_star_depth_ordering_swaps_with_camera() {
    let r = renderer();
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    // Top star in front of the slab (z = +0.5), bottom star behind it (z = −2).
    let frame = frame_of(&[
        one(Vec3::new(0.0, 0.5, 0.5), 0.15),
        one(Vec3::new(0.0, -0.5, -2.0), 0.15),
    ]);
    let g = slab_grid();
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &[],
        look: absorb_look(SLAB_KAPPA),
    };
    let t_slab = (-(SLAB_KAPPA * SLAB_RHO) as f64).exp();

    let check = |cam: &Camera, dim_top: bool, what: &str| {
        let base = r.render_frame_with_gas(&frame, None, cam, &cfg).unwrap();
        let with = r
            .render_frame_with_gas(&frame, Some(&gas), cam, &cfg)
            .unwrap();
        let (t0, b0) = half_fluxes_vertical(&base);
        let (t1, b1) = half_fluxes_vertical(&with);
        assert!(t0 > 0.0 && b0 > 0.0, "{what}: a star rendered black");
        let (dimmed, clear) = if dim_top {
            (t1 / t0, b1 / b0)
        } else {
            (b1 / b0, t1 / t0)
        };
        assert!(
            (clear - 1.0).abs() < 1e-6,
            "{what}: camera-side star attenuated: ratio {clear}"
        );
        assert!(
            (dimmed - t_slab).abs() / t_slab < 1e-4,
            "{what}: far star ratio {dimmed} vs analytic {t_slab}"
        );
    };

    // From +Z the bottom star (z = −2) sits behind the slab.
    check(&centered_camera(), false, "camera on +Z");
    // From −Z the roles swap: now the top star (z = +0.5) is behind the slab.
    let flipped = Camera::orthographic(Vec3::ZERO, Vec3::Z, Vec3::Y, Vec2::new(1.0, 1.0));
    check(&flipped, true, "camera on -Z");
}

/// GPU march ≡ CPU reference, orthographic: two different pattern grids with
/// DIFFERENT bounds (the union-AABB and per-grid clamp/zero paths all
/// exercised), an oblique view axis, a non-trivial mix. Per-pixel, all four
/// channels (alpha carries 1−T): 1e-3 relative + 1e-5 absolute — f32
/// arithmetic-order differences over a few hundred steps, no more.
#[test]
fn gpu_march_matches_cpu_reference_ortho() {
    let r = renderer();
    let g0 = pattern_grid(
        Vec3::new(-1.2, -1.0, -0.8),
        Vec3::new(1.0, 1.1, 0.9),
        [12, 10, 9],
        0.0,
    );
    let g1 = pattern_grid(
        Vec3::new(-0.9, -1.1, -1.0),
        Vec3::new(1.2, 0.9, 1.1),
        [8, 14, 11],
        1.7,
    );
    let gas = GasFrame {
        grid0: &g0,
        grid1: &g1,
        mix: 0.37,
        lights: &[],
        look: GasLook {
            color: [0.9, 0.5, 0.3],
            emissivity: 1.7,
            opacity: 2.1,
            scatter: None,
        },
    };
    let cam = Camera::orthographic(
        Vec3::new(0.1, -0.05, 0.0),
        Vec3::new(0.3, -0.2, -1.0),
        Vec3::Y,
        Vec2::new(1.4, 1.05),
    );
    let cfg = RenderConfig {
        width: 64,
        height: 48,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let gpu = r
        .render_frame_with_gas(&FrameData::default(), Some(&gas), &cam, &cfg)
        .unwrap();
    let cpu = render_gas_cpu(&gas, &cam, cfg.width, cfg.height);
    let mut nonzero = false;
    for y in 0..cfg.height {
        for x in 0..cfg.width {
            let (g, c) = (gpu.pixel(x, y), cpu.pixel(x, y));
            nonzero |= c[0] > 0.0;
            for k in 0..4 {
                let tol = 1e-3 * c[k].abs() + 1e-5;
                assert!(
                    (g[k] - c[k]).abs() <= tol,
                    "pixel ({x},{y}) channel {k}: GPU {} vs CPU {}",
                    g[k],
                    c[k]
                );
            }
        }
    }
    assert!(nonzero, "reference image is all black — degenerate gate");
}

/// GPU march ≡ CPU reference, perspective: eye rays through an off-axis
/// camera, rays that miss the grids entirely, and the t ≥ 0 (nothing behind
/// the eye) clamp — same tolerance as the ortho twin.
#[test]
fn gpu_march_matches_cpu_reference_perspective() {
    let r = renderer();
    let g0 = pattern_grid(
        Vec3::new(-1.2, -1.0, -0.8),
        Vec3::new(1.0, 1.1, 0.9),
        [12, 10, 9],
        0.0,
    );
    let g1 = pattern_grid(
        Vec3::new(-0.9, -1.1, -1.0),
        Vec3::new(1.2, 0.9, 1.1),
        [8, 14, 11],
        1.7,
    );
    let gas = GasFrame {
        grid0: &g0,
        grid1: &g1,
        mix: 0.37,
        lights: &[],
        look: GasLook {
            color: [0.9, 0.5, 0.3],
            emissivity: 1.7,
            opacity: 2.1,
            scatter: None,
        },
    };
    let cam = Camera::perspective(
        Vec3::ZERO,
        Vec3::new(0.25, 0.15, -1.0),
        Vec3::Y,
        Vec2::new(1.2, 0.9),
        3.5,
        0.05,
    );
    let cfg = RenderConfig {
        width: 64,
        height: 48,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let gpu = r
        .render_frame_with_gas(&FrameData::default(), Some(&gas), &cam, &cfg)
        .unwrap();
    let cpu = render_gas_cpu(&gas, &cam, cfg.width, cfg.height);
    let mut nonzero = false;
    for y in 0..cfg.height {
        for x in 0..cfg.width {
            let (g, c) = (gpu.pixel(x, y), cpu.pixel(x, y));
            nonzero |= c[0] > 0.0;
            for k in 0..4 {
                let tol = 1e-3 * c[k].abs() + 1e-5;
                assert!(
                    (g[k] - c[k]).abs() <= tol,
                    "pixel ({x},{y}) channel {k}: GPU {} vs CPU {}",
                    g[k],
                    c[k]
                );
            }
        }
    }
    assert!(nonzero, "reference image is all black — degenerate gate");
}

/// Emission linearity on the GPU: 2× emissivity ⇒ exactly 2× RGB flux
/// (doubling is exact in f32 and the blend is additive).
#[test]
fn gpu_emission_linearity_exact() {
    let r = renderer();
    let g = slab_grid();
    let mk = |e: f32| GasLook {
        color: [1.0, 0.7, 0.4],
        emissivity: e,
        opacity: 0.9,
        scatter: None,
    };
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let cam = centered_camera();
    let img = |e: f32| {
        let gas = GasFrame {
            grid0: &g,
            grid1: &g,
            mix: 0.0,
            lights: &[],
            look: mk(e),
        };
        r.render_frame_with_gas(&FrameData::default(), Some(&gas), &cam, &cfg)
            .unwrap()
            .total_flux()
    };
    let (f1, f2) = (img(0.8), img(1.6));
    for c in 0..3 {
        assert!(f1[c] > 0.0, "gas rendered black");
        let ratio = f2[c] / f1[c];
        assert!(
            (ratio - 2.0).abs() < 1e-7,
            "channel {c}: flux ratio {ratio} must be exactly 2"
        );
    }
}

/// Endpoint grids on the GPU: at u = 0 the image is bit-identical whatever
/// grid1 holds; at u = 1, whatever grid0 holds (same-bounds grids so the march
/// geometry is common — the mix identity is what's under test).
#[test]
fn gpu_mix_endpoint_grids_bit_exact() {
    let r = renderer();
    let bmin = Vec3::splat(-1.0);
    let bmax = Vec3::splat(1.0);
    let a = pattern_grid(bmin, bmax, [8, 8, 8], 0.0);
    let b = pattern_grid(bmin, bmax, [8, 8, 8], 2.1);
    let c = pattern_grid(bmin, bmax, [8, 8, 8], 4.4);
    let look = GasLook {
        color: [1.0, 0.8, 0.6],
        emissivity: 1.2,
        opacity: 0.9,
        scatter: None,
    };
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let cam = centered_camera();
    let img = |g0: &GasGrid, g1: &GasGrid, u: f32| {
        r.render_frame_with_gas(
            &FrameData::default(),
            Some(&GasFrame {
                grid0: g0,
                grid1: g1,
                mix: u,
                lights: &[],
                look,
            }),
            &cam,
            &cfg,
        )
        .unwrap()
    };
    assert_eq!(
        img(&a, &b, 0.0).pixels,
        img(&a, &c, 0.0).pixels,
        "u = 0 image must not depend on grid1"
    );
    assert_eq!(
        img(&b, &a, 1.0).pixels,
        img(&c, &a, 1.0).pixels,
        "u = 1 image must not depend on grid0"
    );
}

/// Same-device determinism: the full composite (stars + prepass + gas) twice
/// over is bit-identical — the fixed-step march and fixed gather order leave
/// no nondeterminism on one device.
#[test]
fn gpu_same_scene_bit_identical() {
    let r = renderer();
    let g0 = pattern_grid(
        Vec3::new(-1.2, -1.0, -0.8),
        Vec3::new(1.0, 1.1, 0.9),
        [12, 10, 9],
        0.0,
    );
    let g1 = pattern_grid(
        Vec3::new(-0.9, -1.1, -1.0),
        Vec3::new(1.2, 0.9, 1.1),
        [8, 14, 11],
        1.7,
    );
    let gas = GasFrame {
        grid0: &g0,
        grid1: &g1,
        mix: 0.37,
        lights: &[],
        look: GasLook {
            color: [0.9, 0.5, 0.3],
            emissivity: 1.1,
            opacity: 1.4,
            scatter: None,
        },
    };
    let cfg = RenderConfig {
        width: 128,
        height: 128,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let cam = centered_camera();
    let frame = scene(40);
    let a = r
        .render_frame_with_gas(&frame, Some(&gas), &cam, &cfg)
        .unwrap();
    let b = r
        .render_frame_with_gas(&frame, Some(&gas), &cam, &cfg)
        .unwrap();
    assert_eq!(a.pixels, b.pixels, "same scene must render bit-identically");
}

/// Gas the camera cannot see adds nothing: a grid entirely outside the ortho
/// view window, and a grid entirely BEHIND a perspective eye (the t ≥ 0
/// clamp), both render exactly black — with aggressive look values that would
/// glow on any clipping bug.
#[test]
fn gpu_gas_outside_view_adds_nothing() {
    let r = renderer();
    let look = GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 5.0,
        opacity: 1.0,
        scatter: None,
    };
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        falloff: 6.0,
        ..RenderConfig::default()
    };

    // Ortho: the ±1 window's rays run down −Z at |x|,|y| ≤ 1; a box at 5..7
    // is never intersected.
    let far = uniform_grid(3.0, Vec3::splat(5.0), Vec3::splat(7.0), [4, 4, 4]);
    let gas = GasFrame {
        grid0: &far,
        grid1: &far,
        mix: 0.0,
        lights: &[],
        look,
    };
    let img = r
        .render_frame_with_gas(&FrameData::default(), Some(&gas), &centered_camera(), &cfg)
        .unwrap();
    assert!(
        img.pixels.iter().flatten().all(|&v| v == 0.0),
        "off-view gas leaked into the ortho image"
    );

    // Perspective: eye at (0,0,+4) looking −Z; a box at z ∈ [4.5, 5.5] is
    // behind the eye and must be clamped away, not marched with negative t.
    let behind = uniform_grid(
        3.0,
        Vec3::new(-0.5, -0.5, 4.5),
        Vec3::new(0.5, 0.5, 5.5),
        [4, 4, 4],
    );
    let gas = GasFrame {
        grid0: &behind,
        grid1: &behind,
        mix: 0.0,
        lights: &[],
        look,
    };
    let img = r
        .render_frame_with_gas(&FrameData::default(), Some(&gas), &persp_cam(), &cfg)
        .unwrap();
    assert!(
        img.pixels.iter().flatten().all(|&v| v == 0.0),
        "behind-eye gas leaked into the perspective image"
    );
}
