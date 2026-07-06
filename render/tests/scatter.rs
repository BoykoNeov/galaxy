//! Single-scatter starlight gates (plan scattered-starlit-veil): the gas march
//! gains `j_scat = σ_s·ρ·Σ_k p_HG(cosθ_k)·L_k/(4π·(d_k²+r_k²))` from clustered
//! stellar point lights, OPTIONAL and bit-compatible when off.
//!
//! CPU gates pin the phase function (normalization by quadrature, hand values),
//! the light clustering (power conservation, weighted centroids — hand
//! oracles), and the scattered radiance against closed forms in the far-field
//! limit where the incident intensity is constant along the chord. GPU gates
//! hold the WGSL march to the CPU mirror and pin the off-path bit-identity.
//!
//! GPU tests need a wgpu adapter and fail loudly without one (the
//! `invariants.rs` convention).

use galaxy_render::camera::Camera;
use galaxy_render::render::{RenderConfig, Renderer};
use galaxy_render::volume::{
    cluster_lights, hg_phase, march_gas, render_gas_cpu, GasFrame, GasLook, Light, ScatterLook,
    LIGHT_BINS,
};
use galaxy_renderprep::{FrameData, GasGrid};
use glam::{Vec2, Vec3};

// ---------- helpers (the volume.rs test fixtures, scatter-aware) ----------

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for scatter gates")
}

fn uniform_grid(rho: f32, bounds_min: Vec3, bounds_max: Vec3, dims: [u32; 3]) -> GasGrid {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    GasGrid {
        dims,
        bounds_min,
        bounds_max,
        data: vec![rho; n],
    }
}

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

/// The volume.rs reference slab: z ∈ [−1, 0], generous lateral extent,
/// uniform ρ = 0.5, cell edge 1/32 along z.
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

fn centered_camera() -> Camera {
    Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0))
}

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

const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

// ---------- CPU gates: the phase function ----------

/// HG normalization by quadrature: ∫ p dΩ = 2π ∫₋₁¹ p(μ) dμ = 1 for every g
/// (midpoint rule, 200k panels — the integrand is smooth, so the quadrature
/// error is far below the gate). g = 0 must be exactly isotropic: 1/4π at
/// every angle.
#[test]
fn hg_phase_normalizes_over_the_sphere() {
    for g in [0.0f32, 0.4, -0.7, 0.9] {
        let n = 200_000;
        let mut integral = 0.0f64;
        for i in 0..n {
            let mu = -1.0 + (i as f64 + 0.5) * (2.0 / n as f64);
            integral += hg_phase(mu as f32, g) as f64 * (2.0 / n as f64);
        }
        integral *= 2.0 * std::f64::consts::PI;
        assert!(
            (integral - 1.0).abs() < 1e-4,
            "g = {g}: ∫p dΩ = {integral}, must be 1"
        );
    }
    let iso = (1.0 / FOUR_PI) as f32;
    for mu in [-1.0f32, -0.3, 0.0, 0.7, 1.0] {
        let p = hg_phase(mu, 0.0);
        assert!(
            (p - iso).abs() < 1e-9,
            "g = 0 must be isotropic: p({mu}) = {p} vs 1/4π = {iso}"
        );
    }
}

/// Hand values of the HG phase at the geometry the slab gates use (f64
/// arithmetic, independent of the implementation): transverse (μ = 0),
/// forward (μ = 1) and backward (μ = −1) at g = 0.6, where
/// p(±1) = (1−g²)/(4π(1∓g)³) gives the exact 64× forward/backward ratio.
#[test]
fn hg_phase_hand_values() {
    let hand = |mu: f64, g: f64| (1.0 - g * g) / (FOUR_PI * (1.0 + g * g - 2.0 * g * mu).powf(1.5));
    for (mu, g) in [(0.0f32, 0.6f32), (1.0, 0.6), (-1.0, 0.6), (0.25, -0.35)] {
        let want = hand(mu as f64, g as f64) as f32;
        let got = hg_phase(mu, g);
        assert!(
            (got - want).abs() / want.abs() < 1e-5,
            "p({mu}, {g}) = {got} vs hand {want}"
        );
    }
    // The exact forward/backward ratio at g = 0.6: ((1+g)/(1−g))³ = 4³ = 64.
    let ratio = hg_phase(1.0, 0.6) / hg_phase(-1.0, 0.6);
    assert!(
        (ratio - 64.0).abs() < 1e-3,
        "forward/backward ratio {ratio} vs exact 64"
    );
}

// ---------- CPU gates: light clustering ----------

/// Hand-built clustering oracle: two near stars merge into one light at their
/// luminance-weighted centroid with summed RGB power; a far star stays its own
/// light; total power is conserved exactly; a zero-brightness star neither
/// lights nor stretches the binning AABB.
#[test]
fn cluster_lights_hand_oracle() {
    // Stars A (origin, red, w=2) and B (+0.1x, green, w=1) share the corner
    // bin of the [0,10]³ AABB; C at the opposite corner is alone in its bin.
    let frame = FrameData {
        pos: vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
            Vec3::new(10.0, 10.0, 10.0),
        ],
        color: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        size: vec![0.1; 3],
        brightness: vec![2.0, 1.0, 4.0],
    };
    let lights = cluster_lights(&frame);
    assert_eq!(lights.len(), 2, "two occupied bins ⇒ two lights");

    // Power conservation: Σ rgb over lights = Σ color·brightness over stars.
    let mut total = [0.0f32; 3];
    for l in &lights {
        for (acc, v) in total.iter_mut().zip(l.rgb) {
            *acc += v;
        }
    }
    for (c, want) in [2.0f32, 1.0, 4.0].iter().enumerate() {
        assert!(
            (total[c] - want).abs() < 1e-6 * want,
            "channel {c}: total power {} vs Σ emissive {want}",
            total[c]
        );
    }

    // The merged light: luminance weights w_A = 2, w_B = 1 ⇒ centroid at
    // x = (2·0 + 1·0.1)/3, rgb = [2, 1, 0]. Bin cell = (10/8)³ ⇒ softening
    // radius = half its diagonal.
    let merged = lights
        .iter()
        .find(|l| l.rgb[0] > 0.0)
        .expect("the A+B light exists");
    assert!(
        (merged.pos - Vec3::new(0.1 / 3.0, 0.0, 0.0)).length() < 1e-6,
        "merged centroid {:?} vs hand (0.0333, 0, 0)",
        merged.pos
    );
    assert!((merged.rgb[0] - 2.0).abs() < 1e-6 && (merged.rgb[1] - 1.0).abs() < 1e-6);
    let cell = 10.0 / LIGHT_BINS as f32;
    let want_r = 0.5 * (3.0f32).sqrt() * cell;
    assert!(
        (merged.radius - want_r).abs() < 1e-5,
        "softening radius {} vs half bin diagonal {want_r}",
        merged.radius
    );

    // The lone light: C's own position, color, brightness.
    let lone = lights
        .iter()
        .find(|l| l.rgb[2] > 0.0)
        .expect("the C light exists");
    assert!((lone.pos - Vec3::splat(10.0)).length() < 1e-6);
    assert!((lone.rgb[2] - 4.0).abs() < 1e-6);

    // A dark star far outside must change nothing (it would otherwise stretch
    // the AABB and move every bin boundary).
    let mut with_dark = frame.clone();
    with_dark.pos.push(Vec3::new(100.0, 0.0, 0.0));
    with_dark.color.push([1.0, 1.0, 1.0]);
    with_dark.size.push(0.1);
    with_dark.brightness.push(0.0);
    assert_eq!(
        cluster_lights(&with_dark),
        lights,
        "a zero-brightness star must not affect the clustering"
    );

    // All-dark frame ⇒ no lights; empty frame ⇒ no lights.
    let dark = FrameData {
        pos: vec![Vec3::ZERO],
        color: vec![[1.0, 1.0, 1.0]],
        size: vec![0.1],
        brightness: vec![0.0],
    };
    assert!(cluster_lights(&dark).is_empty());
    assert!(cluster_lights(&FrameData::default()).is_empty());
}

/// The degenerate AABB: every luminous star at one point collapses to a single
/// light exactly there, with zero softening radius (and no 0/0 NaNs).
#[test]
fn cluster_lights_degenerate_aabb() {
    let p = Vec3::new(1.5, -2.0, 0.25);
    let frame = FrameData {
        pos: vec![p, p, p],
        color: vec![[1.0, 0.5, 0.2]; 3],
        size: vec![0.1; 3],
        brightness: vec![1.0, 2.0, 3.0],
    };
    let lights = cluster_lights(&frame);
    assert_eq!(lights.len(), 1);
    assert_eq!(lights[0].pos, p, "degenerate AABB centroid is the point");
    assert_eq!(lights[0].radius, 0.0, "degenerate AABB has zero cell size");
    for (c, base) in [1.0f32, 0.5, 0.2].iter().enumerate() {
        let want = base * 6.0; // brightness 1+2+3
        assert!(
            (lights[0].rgb[c] - want).abs() < 1e-6 * want,
            "channel {c}: {} vs {want}",
            lights[0].rgb[c]
        );
    }
}

// ---------- CPU gates: scattered radiance closed forms ----------

/// Far-field limit: one light at D = 200 ≫ slab, transverse to the ray
/// (cosθ ≈ 0), so the incident intensity I = L/(4πD²) and the phase value are
/// constant along the chord to ≪ 1% and the scattered term is exactly an
/// emission slab with j_eff = σ_s·p·I per channel:
/// C = (σ_s·p·I/κ)(1 − e^{−κρL}). Hand-computed p in f64 for g = 0 and
/// g = 0.6; 1% relative covers quadrature (0.23%) + far-field (≲ 0.4%).
#[test]
fn scatter_far_field_slab_analytic() {
    let g = slab_grid();
    let light = Light {
        pos: Vec3::new(200.0, 0.0, -0.5),
        radius: 0.0,
        rgb: [4.0e5, 2.0e5, 1.0e5],
    };
    let origin = Vec3::new(0.3, -0.2, 5.0);
    let dir = Vec3::new(0.0, 0.0, -1.0);
    let strength = 1.7f32;

    for aniso in [0.0f64, 0.6] {
        let gas = GasFrame {
            grid0: &g,
            grid1: &g,
            mix: 0.0,
            lights: std::slice::from_ref(&light),
            look: GasLook {
                color: [1.0, 1.0, 1.0],
                emissivity: 0.0, // isolate the scattered term
                opacity: SLAB_KAPPA,
                scatter: Some(ScatterLook {
                    strength,
                    anisotropy: aniso as f32,
                }),
            },
        };
        let (c, t) = march_gas(&gas, origin, dir, f32::NEG_INFINITY);

        // Independent f64 expectation at the chord center (z = −0.5).
        let d2 = (200.0f64 - 0.3).powi(2) + 0.2f64.powi(2);
        let p = (1.0 - aniso * aniso) / (FOUR_PI * (1.0 + aniso * aniso).powf(1.5)); // μ = 0
        let tau = (SLAB_KAPPA * SLAB_RHO) as f64; // L = 1
        for (k, &lk) in light.rgb.iter().enumerate() {
            let intensity = lk as f64 / (FOUR_PI * d2);
            let want = strength as f64 * p * intensity / SLAB_KAPPA as f64 * (1.0 - (-tau).exp());
            assert!(
                (c[k] as f64 - want).abs() / want < 1e-2,
                "g = {aniso}, channel {k}: scattered {} vs analytic {want}",
                c[k]
            );
        }
        // Scattering must not touch the transmittance.
        let want_t = (-tau).exp() as f32;
        assert!(
            (t - want_t).abs() / want_t < 1e-5,
            "g = {aniso}: T {t} vs {want_t}"
        );
    }
}

/// Inverse-square law: the same transverse light at D = 50 vs D = 100 (radius
/// 0) scatters exactly 4× the radiance, up to the shared ±0.5 chord offset
/// (relative ≤ 1e-4 at D = 50).
#[test]
fn scatter_inverse_square() {
    let g = slab_grid();
    let run = |d: f32| {
        let light = Light {
            pos: Vec3::new(d, 0.0, -0.5),
            radius: 0.0,
            rgb: [1.0e4; 3],
        };
        let gas = GasFrame {
            grid0: &g,
            grid1: &g,
            mix: 0.0,
            lights: std::slice::from_ref(&light),
            look: GasLook {
                color: [1.0, 1.0, 1.0],
                emissivity: 0.0,
                opacity: SLAB_KAPPA,
                scatter: Some(ScatterLook {
                    strength: 1.0,
                    anisotropy: 0.0,
                }),
            },
        };
        march_gas(
            &gas,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            f32::NEG_INFINITY,
        )
        .0
    };
    let (near, far) = (run(50.0), run(100.0));
    for k in 0..3 {
        assert!(near[k] > 0.0, "channel {k} scattered nothing");
        let ratio = near[k] / far[k];
        assert!(
            (ratio - 4.0).abs() < 4.0 * 1e-3,
            "channel {k}: D vs 2D ratio {ratio} must be 4 (inverse-square)"
        );
    }
}

/// Off is off, bitwise: `scatter: None` (lights present), `strength = 0`
/// (lights present), and `strength > 0` with NO lights all produce the
/// bit-identical march — the compatibility contract that lets a scenario
/// disable the feature.
#[test]
fn scatter_off_is_bit_identical() {
    let g = pattern_grid(Vec3::splat(-1.0), Vec3::splat(1.0), [6, 6, 6], 0.0);
    let light = Light {
        pos: Vec3::new(0.5, 0.2, 3.0),
        radius: 0.1,
        rgb: [7.0, 5.0, 3.0],
    };
    let origin = Vec3::new(0.1, 0.05, 3.0);
    let dir = Vec3::new(0.4, -0.3, -1.0).normalize();
    let run = |scatter: Option<ScatterLook>, lights: &[Light]| {
        march_gas(
            &GasFrame {
                grid0: &g,
                grid1: &g,
                mix: 0.0,
                lights,
                look: GasLook {
                    color: [1.0, 0.9, 0.8],
                    emissivity: 1.3,
                    opacity: 0.7,
                    scatter,
                },
            },
            origin,
            dir,
            f32::NEG_INFINITY,
        )
    };
    let base = run(None, std::slice::from_ref(&light));
    assert_eq!(
        run(
            Some(ScatterLook {
                strength: 0.0,
                anisotropy: 0.5,
            }),
            std::slice::from_ref(&light),
        ),
        base,
        "strength = 0 must be bit-identical to scatter: None"
    );
    assert_eq!(
        run(
            Some(ScatterLook {
                strength: 2.0,
                anisotropy: 0.5,
            }),
            &[],
        ),
        base,
        "no lights must be bit-identical to scatter: None"
    );
}

/// Linearity: 2× strength ⇒ exactly 2× scattered radiance per channel
/// (doubling is exact in binary fp and distributes over the additive
/// accumulation), with the transmittance bit-identical.
#[test]
fn scatter_strength_linear_exact() {
    let g = pattern_grid(Vec3::splat(-1.0), Vec3::splat(1.0), [6, 6, 6], 1.3);
    let lights = [
        Light {
            pos: Vec3::new(0.5, 0.2, 2.0),
            radius: 0.2,
            rgb: [7.0, 5.0, 3.0],
        },
        Light {
            pos: Vec3::new(-1.5, 0.0, -0.5),
            radius: 0.0,
            rgb: [2.0, 4.0, 6.0],
        },
    ];
    let run = |strength: f32| {
        march_gas(
            &GasFrame {
                grid0: &g,
                grid1: &g,
                mix: 0.0,
                lights: &lights,
                look: GasLook {
                    color: [1.0, 1.0, 1.0],
                    emissivity: 0.0, // scattered term only
                    opacity: 0.8,
                    scatter: Some(ScatterLook {
                        strength,
                        anisotropy: 0.3,
                    }),
                },
            },
            Vec3::new(-0.2, 0.15, 3.0),
            Vec3::new(-0.1, 0.25, -1.0).normalize(),
            f32::NEG_INFINITY,
        )
    };
    let (c1, t1) = run(0.7);
    let (c2, t2) = run(1.4);
    assert!(c1.iter().all(|&v| v > 0.0), "ray must scatter something");
    for k in 0..3 {
        assert_eq!(
            c2[k],
            2.0 * c1[k],
            "channel {k}: 2× strength must double the radiance exactly"
        );
    }
    assert_eq!(t1, t2, "transmittance must not depend on scattering");
}

/// Forward scattering is directional: with g = 0.6 a BACKLIT light (behind the
/// slab, μ = +1) out-scatters the mirrored front-lit geometry (μ = −1) by the
/// exact HG ratio ((1+g)/(1−g))³ = 64 — κ = 0 keeps the two chords' distance
/// sets identical, so the phase ratio is the only asymmetry. At g = 0 the two
/// geometries are equal (isotropic).
#[test]
fn scatter_forward_anisotropy_backlights() {
    let g = slab_grid();
    // Light on the ray axis, 5 units from the slab center (z = −0.5): behind
    // at z = −5.5 (ω_in = +Z = toward the camera, μ = +1), in front at
    // z = +4.5 (μ = −1).
    let run = |light_z: f32, aniso: f32| {
        let light = Light {
            pos: Vec3::new(0.0, 0.0, light_z),
            radius: 0.0,
            rgb: [1.0e3; 3],
        };
        let gas = GasFrame {
            grid0: &g,
            grid1: &g,
            mix: 0.0,
            lights: std::slice::from_ref(&light),
            look: GasLook {
                color: [1.0, 1.0, 1.0],
                emissivity: 0.0,
                opacity: 0.0, // T ≡ 1: distance sets are mirror-identical
                scatter: Some(ScatterLook {
                    strength: 1.0,
                    anisotropy: aniso,
                }),
            },
        };
        march_gas(
            &gas,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            f32::NEG_INFINITY,
        )
        .0[0]
    };
    // Isotropic: back and front are equal (same 1/d² set, summed in reverse).
    let (back0, front0) = (run(-5.5, 0.0), run(4.5, 0.0));
    assert!(back0 > 0.0, "backlit ray must scatter");
    assert!(
        (back0 / front0 - 1.0).abs() < 1e-4,
        "g = 0: backlit/frontlit = {} must be 1",
        back0 / front0
    );
    // Forward-peaked: the exact 64× HG ratio survives the (identical) 1/d²
    // weighting.
    let ratio = run(-5.5, 0.6) / run(4.5, 0.6);
    assert!(
        (ratio - 64.0).abs() < 0.1,
        "g = 0.6: backlit/frontlit = {ratio} vs exact 64"
    );
}

// ---------- GPU gates ----------

/// GPU scatter march ≡ CPU reference, orthographic: different-bounds pattern
/// grids, a non-trivial mix, three hand-built lights (one inside each grid,
/// one outside, mixed radii), forward anisotropy — per-pixel, all four
/// channels, at the volume.rs tolerance (1e-3 relative + 1e-5 absolute).
#[test]
fn gpu_scatter_matches_cpu_reference_ortho() {
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
    let lights = [
        Light {
            pos: Vec3::new(0.5, 0.3, 0.2),
            radius: 0.15,
            rgb: [8.0, 5.0, 3.0],
        },
        Light {
            pos: Vec3::new(-0.7, -0.4, 0.5),
            radius: 0.3,
            rgb: [2.0, 6.0, 4.0],
        },
        Light {
            pos: Vec3::new(0.1, 2.0, -0.6),
            radius: 0.0,
            rgb: [5.0, 5.0, 9.0],
        },
    ];
    let gas = GasFrame {
        grid0: &g0,
        grid1: &g1,
        mix: 0.37,
        lights: &lights,
        look: GasLook {
            color: [0.9, 0.5, 0.3],
            emissivity: 1.7,
            opacity: 2.1,
            scatter: Some(ScatterLook {
                strength: 1.3,
                anisotropy: 0.4,
            }),
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

/// GPU scatter march ≡ CPU reference, perspective: the same scene through eye
/// rays (per-pixel ω_out varies — the phase angle actually changes across the
/// image, unlike ortho), same tolerance.
#[test]
fn gpu_scatter_matches_cpu_reference_perspective() {
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
    let lights = [
        Light {
            pos: Vec3::new(0.5, 0.3, 0.2),
            radius: 0.15,
            rgb: [8.0, 5.0, 3.0],
        },
        Light {
            pos: Vec3::new(-0.7, -0.4, 0.5),
            radius: 0.3,
            rgb: [2.0, 6.0, 4.0],
        },
    ];
    let gas = GasFrame {
        grid0: &g0,
        grid1: &g1,
        mix: 0.37,
        lights: &lights,
        look: GasLook {
            color: [0.9, 0.5, 0.3],
            emissivity: 1.7,
            opacity: 2.1,
            scatter: Some(ScatterLook {
                strength: 1.3,
                anisotropy: -0.5,
            }),
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

/// The GPU off-path contract: the full composite (stars + prepass + gas) with
/// `scatter: None`, with `strength = 0` (lights present), and with lights
/// absent (strength > 0) are all bit-identical images — flipping the scenario
/// knob off restores today's output exactly.
#[test]
fn gpu_scatter_off_bit_identical() {
    let r = renderer();
    let g = slab_grid();
    let lights = [Light {
        pos: Vec3::new(0.2, 0.1, 0.5),
        radius: 0.1,
        rgb: [5.0, 4.0, 3.0],
    }];
    let cfg = RenderConfig {
        width: 96,
        height: 96,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let cam = centered_camera();
    let frame = scene(40);
    let img = |scatter: Option<ScatterLook>, lights: &[Light]| {
        let gas = GasFrame {
            grid0: &g,
            grid1: &g,
            mix: 0.0,
            lights,
            look: GasLook {
                color: [0.6, 0.7, 1.0],
                emissivity: 0.8,
                opacity: SLAB_KAPPA,
                scatter,
            },
        };
        r.render_frame_with_gas(&frame, Some(&gas), &cam, &cfg)
            .unwrap()
            .pixels
    };
    let base = img(None, &[]);
    assert_eq!(
        img(None, &lights),
        base,
        "lights without a scatter look must change nothing"
    );
    assert_eq!(
        img(
            Some(ScatterLook {
                strength: 0.0,
                anisotropy: 0.7,
            }),
            &lights,
        ),
        base,
        "strength = 0 must be bit-identical to scatter: None"
    );
    assert_eq!(
        img(
            Some(ScatterLook {
                strength: 2.0,
                anisotropy: 0.7,
            }),
            &[],
        ),
        base,
        "no lights must be bit-identical to scatter: None"
    );
}

/// GPU linearity: 2× strength ⇒ exactly 2× flux (scatter-only: emissivity 0).
#[test]
fn gpu_scatter_strength_linear_exact() {
    let r = renderer();
    let g = slab_grid();
    let lights = [Light {
        pos: Vec3::new(0.0, 0.0, 2.0),
        radius: 0.2,
        rgb: [6.0, 5.0, 4.0],
    }];
    let cfg = RenderConfig {
        width: 64,
        height: 64,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    let cam = centered_camera();
    let flux = |strength: f32| {
        let gas = GasFrame {
            grid0: &g,
            grid1: &g,
            mix: 0.0,
            lights: &lights,
            look: GasLook {
                color: [1.0, 1.0, 1.0],
                emissivity: 0.0,
                opacity: 0.9,
                scatter: Some(ScatterLook {
                    strength,
                    anisotropy: 0.3,
                }),
            },
        };
        r.render_frame_with_gas(&FrameData::default(), Some(&gas), &cam, &cfg)
            .unwrap()
            .total_flux()
    };
    let (f1, f2) = (flux(0.8), flux(1.6));
    for c in 0..3 {
        assert!(f1[c] > 0.0, "gas scattered nothing");
        let ratio = f2[c] / f1[c];
        assert!(
            (ratio - 2.0).abs() < 1e-7,
            "channel {c}: flux ratio {ratio} must be exactly 2"
        );
    }
    // Sanity: perspective path accepts scatter too (no per-pixel oracle here,
    // the perspective GPU ≡ CPU gate holds it; this pins it renders non-black).
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: GasLook {
            color: [1.0, 1.0, 1.0],
            emissivity: 0.0,
            opacity: 0.9,
            scatter: Some(ScatterLook {
                strength: 1.0,
                anisotropy: 0.3,
            }),
        },
    };
    let img = r
        .render_frame_with_gas(&FrameData::default(), Some(&gas), &persp_cam(), &cfg)
        .unwrap();
    assert!(
        img.total_flux()[0] > 0.0,
        "perspective scatter rendered black"
    );
}
