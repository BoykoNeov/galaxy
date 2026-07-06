//! Per-light shadow-volume gates (plan umbral-lantern-lattice): the scatter
//! term's incident intensity gains a baked light→sample transmittance
//! `T_k(p) = exp(−∫ κ·ρ_mix ds)` per light, tabulated on a `SHADOW_RES`³
//! lattice over the union AABB and sampled trilinearly inside the march —
//! OPTIONAL and bit-compatible when off.
//!
//! CPU gates pin the bake against hand-computed chords through the uniform
//! slab, the trilinear sampler against a linear field (which trilinear
//! interpolation reproduces exactly), and the shadowed march against the
//! backlit-slab closed form where the camera-side and light-side exponentials
//! compose to a constant. GPU gates hold the on-device bake + march to the CPU
//! mirror per pixel and pin the off paths bitwise.
//!
//! GPU tests need a wgpu adapter and fail loudly without one (the
//! `invariants.rs` convention).

use galaxy_render::camera::Camera;
use galaxy_render::render::{RenderConfig, Renderer};
use galaxy_render::volume::{
    bake_shadows, march_gas, render_gas_cpu, GasFrame, GasLook, Light, ScatterLook, ShadowVolumes,
    SHADOW_RES,
};
use galaxy_renderprep::{FrameData, GasGrid};
use glam::{DVec3, Vec2, Vec3};

// ---------- helpers (the scatter.rs fixtures, shadow-aware) ----------

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for shadow gates")
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
/// uniform ρ = 0.5, cell edge 1/32 along z (nominal march step 1/64).
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

/// The HG phase evaluated independently in f64 (the scatter.rs hand formula).
fn hg_hand(mu: f64, g: f64) -> f64 {
    (1.0 - g * g) / (FOUR_PI * (1.0 + g * g - 2.0 * g * mu).powf(1.5))
}

/// A hand-built [`ShadowVolumes`] whose per-voxel values follow the LINEAR
/// field `5 + ix + 2·iy + 3·iz` (+1000 per light): trilinear interpolation
/// reproduces a linear field exactly, so the expected sample anywhere is the
/// same linear formula in fractional lattice coordinates. The +5 offset keeps
/// the clamped corner distinguishable from a zero-outside bug.
fn linear_volumes(bounds_min: Vec3, bounds_max: Vec3, count: usize) -> ShadowVolumes {
    let r = SHADOW_RES as usize;
    let mut data = vec![0.0f32; count * r * r * r];
    for k in 0..count {
        for iz in 0..r {
            for iy in 0..r {
                for ix in 0..r {
                    data[k * r * r * r + (iz * r + iy) * r + ix] =
                        5.0 + ix as f32 + 2.0 * iy as f32 + 3.0 * iz as f32 + 1000.0 * k as f32;
                }
            }
        }
    }
    ShadowVolumes {
        bounds_min,
        bounds_max,
        count,
        data,
    }
}

// ---------- CPU gates: the bake ----------

/// The baked transmittance equals the hand chord `T = exp(−κρ·len)` through
/// the uniform slab, where `len` is the light→voxel segment's portion inside
/// the slab, computed independently in f64 (entry through the top face at
/// z = 0, truncated at the voxel center). Inside the slab the trilinear
/// density is exactly the uniform ρ, so midpoint quadrature is exact and the
/// only slack is f32 clip/summation noise.
#[test]
fn bake_shadows_matches_hand_chords_on_the_uniform_slab() {
    let g = slab_grid();
    let lights = [Light {
        pos: Vec3::new(0.0, 0.0, 10.0),
        radius: 0.3, // must NOT soften the shadow segment (intensity-only)
        rgb: [1.0, 1.0, 1.0],
    }];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: GasLook {
            color: [1.0, 1.0, 1.0],
            emissivity: 0.0,
            opacity: SLAB_KAPPA,
            scatter: Some(ScatterLook {
                strength: 1.0,
                anisotropy: 0.0,
                shadows: true,
                tint: [1.0; 3],
                softening: None,
            }),
        },
    };
    let sv = bake_shadows(&gas);
    let r = SHADOW_RES as usize;
    assert_eq!(sv.count, 1, "one light ⇒ one volume");
    assert_eq!(sv.data.len(), r * r * r);
    assert_eq!(sv.bounds_min, g.bounds_min, "lattice spans the union AABB");
    assert_eq!(sv.bounds_max, g.bounds_max);

    let bmin = g.bounds_min.as_dvec3();
    let cell = (g.bounds_max.as_dvec3() - bmin) / SHADOW_RES as f64;
    let l = DVec3::new(0.0, 0.0, 10.0);
    // Voxels chosen off the light axis (oblique chords) and near the top face
    // (a nearly-empty chord): none of their segments exits a side face first.
    for (ix, iy, iz) in [(16usize, 16usize, 4usize), (5, 25, 20), (0, 0, 31)] {
        let vc = bmin + (DVec3::new(ix as f64, iy as f64, iz as f64) + 0.5) * cell;
        let d = vc - l;
        // The segment enters the slab through z = 0 at parameter t_top and
        // ends at the voxel (t = 1): in-gas length = |d|·(1 − t_top).
        let t_top = (0.0 - l.z) / d.z;
        assert!(
            (0.0..1.0).contains(&t_top),
            "fixture: light outside, voxel inside"
        );
        let tau = (SLAB_KAPPA * SLAB_RHO) as f64 * d.length() * (1.0 - t_top);
        let want = (-tau).exp();
        let got = sv.data[(iz * r + iy) * r + ix] as f64;
        assert!(
            (got - want).abs() / want < 1e-3,
            "voxel ({ix},{iy},{iz}): baked T {got} vs hand {want}"
        );
    }
}

/// Exact-one cases: κ = 0 bakes every voxel to exactly 1 (τ is a sum of exact
/// zeros), and a light sitting exactly on a voxel center bakes that voxel to
/// exactly 1 (an empty segment — the coincident-light guard, no NaNs).
#[test]
fn bake_shadows_exact_ones() {
    let g = slab_grid();
    let r = SHADOW_RES as usize;
    let cell = (g.bounds_max - g.bounds_min) / SHADOW_RES as f32;
    // Voxel (10, 20, 7)'s center, computed exactly as the bake does.
    let (ix, iy, iz) = (10usize, 20usize, 7usize);
    let center = g.bounds_min + (Vec3::new(ix as f32, iy as f32, iz as f32) + 0.5) * cell;
    let lights = [
        Light {
            pos: Vec3::new(0.0, 0.0, 10.0),
            radius: 0.0,
            rgb: [1.0, 1.0, 1.0],
        },
        Light {
            pos: center,
            radius: 0.1,
            rgb: [1.0, 1.0, 1.0],
        },
    ];
    let look = |kappa: f32| GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 0.0,
        opacity: kappa,
        scatter: Some(ScatterLook {
            strength: 1.0,
            anisotropy: 0.0,
            shadows: true,
            tint: [1.0; 3],
            softening: None,
        }),
    };
    // κ = 0: all ones, both volumes, exactly.
    let gas0 = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(0.0),
    };
    let sv0 = bake_shadows(&gas0);
    assert_eq!(sv0.count, 2);
    assert!(
        sv0.data.iter().all(|&t| t == 1.0),
        "κ = 0 must bake exactly 1 everywhere"
    );
    // κ > 0: the light-coincident voxel of volume 1 is exactly 1.
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(SLAB_KAPPA),
    };
    let sv = bake_shadows(&gas);
    let got = sv.data[r * r * r + (iz * r + iy) * r + ix];
    assert_eq!(
        got, 1.0,
        "a voxel coincident with its light must be exactly 1"
    );
    // …and a deep voxel of the OUTSIDE light's volume is strictly shadowed
    // (the two volumes are independent — pins the light-major layout).
    let deep = sv.data[(2 * r + 16) * r + 16];
    assert!(deep < 1.0, "deep voxel vs outside light must be shadowed");
}

// ---------- CPU gates: the trilinear sampler ----------

/// Trilinear reproduction of a linear field: exact voxel values at voxel
/// centers, the linear formula at arbitrary interior points, clamp-to-edge
/// (NOT zero) outside the bounds, and independent per-light volumes.
#[test]
fn shadow_sample_trilinear_oracle() {
    let bmin = Vec3::new(-1.0, -2.0, 0.5);
    let bmax = Vec3::new(3.0, 2.0, 4.5);
    let sv = linear_volumes(bmin, bmax, 2);
    let cell = (bmax - bmin) / SHADOW_RES as f32;
    let value = |c: Vec3, k: usize| 5.0 + c.x + 2.0 * c.y + 3.0 * c.z + 1000.0 * k as f32;

    // Exact at a voxel center.
    let c = Vec3::new(3.0, 4.0, 5.0);
    let p = bmin + (c + 0.5) * cell;
    assert_eq!(sv.sample(0, p), value(c, 0), "voxel center must be exact");
    assert_eq!(sv.sample(1, p), value(c, 1), "light-major layout");

    // The linear formula at a fractional interior point.
    let c = Vec3::new(10.25, 3.5, 7.75);
    let p = bmin + (c + 0.5) * cell;
    let got = sv.sample(0, p);
    let want = value(c, 0);
    assert!(
        (got - want).abs() / want < 1e-5,
        "interior trilinear: {got} vs linear field {want}"
    );

    // Clamp-to-edge beyond +x: the x index pins at R−1, y/z still interpolate.
    let p = Vec3::new(10.0, 0.0, 1.0); // far outside +x, inside y/z
    let cy = (p.y - bmin.y) / cell.y - 0.5;
    let cz = (p.z - bmin.z) / cell.z - 0.5;
    let want = value(Vec3::new((SHADOW_RES - 1) as f32, cy, cz), 0);
    let got = sv.sample(0, p);
    assert!(
        (got - want).abs() / want < 1e-5,
        "+x overflow must clamp to the edge sheet: {got} vs {want}"
    );

    // Below bmin on ALL axes: the (0,0,0) corner value — NOT zero (the
    // zero-outside rule of GasGrid::sample would punch dark rims here).
    let got = sv.sample(0, bmin - Vec3::splat(5.0));
    assert_eq!(got, 5.0, "corner clamp must return the corner voxel, not 0");
    let got = sv.sample(1, bmin - Vec3::splat(5.0));
    assert_eq!(got, 1005.0, "corner clamp of the second volume");
}

// ---------- CPU gates: the shadowed march ----------

/// Backlit-slab closed form: light on the ray axis far BEHIND the slab
/// (μ = +1 exactly), so the camera-side and light-side optical depths sum to
/// the constant full-slab τ at every sample and the scattered radiance is
///
/// ```text
/// C_k = σ_s·ρ·p(1, g)·e^{−τ}·(L_k/4π)·∫ dz/(z+D)²,   ∫ = 1/(199·200)
/// ```
///
/// computed independently in f64. Tolerance 2% covers the emit-then-attenuate
/// quadrature bias (e^{κρΔs/2} ≈ 1.0023) and the 32³ trilinear bake.
#[test]
fn shadowed_march_matches_backlit_slab_closed_form() {
    let g = slab_grid();
    for aniso in [0.0f64, 0.6] {
        let light = Light {
            pos: Vec3::new(0.0, 0.0, -200.0),
            radius: 0.0,
            rgb: [4.0e5, 2.0e5, 1.0e5],
        };
        let strength = 1.7f32;
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
                    shadows: true,
                    tint: [1.0; 3],
                    softening: None,
                }),
            },
        };
        let sv = bake_shadows(&gas);
        let (c, t) = march_gas(
            &gas,
            Some(&sv),
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            f32::NEG_INFINITY,
        );

        let tau = (SLAB_KAPPA * SLAB_RHO) as f64; // L = 1
        let p1 = hg_hand(1.0, aniso);
        let integral = 1.0 / (199.0 * 200.0); // ∫_{−1}^{0} dz/(z+200)²
        for (k, &lk) in light.rgb.iter().enumerate() {
            let want = strength as f64 * SLAB_RHO as f64 * p1 * (-tau).exp() * lk as f64 / FOUR_PI
                * integral;
            assert!(
                (c[k] as f64 - want).abs() / want < 2e-2,
                "g = {aniso}, channel {k}: shadowed scatter {} vs closed form {want}",
                c[k]
            );
        }
        // Shadowing must not touch the camera-path transmittance.
        let want_t = (-tau).exp() as f32;
        assert!(
            (t - want_t).abs() / want_t < 1e-5,
            "g = {aniso}: T {t} vs {want_t}"
        );
    }
}

/// Shadowing only removes light: with strictly positive density everywhere,
/// the shadowed march is strictly below the unshadowed one on every channel,
/// and the camera-path transmittance is bit-identical.
#[test]
fn shadowed_march_is_strictly_below_unshadowed() {
    let g = pattern_grid(Vec3::splat(-1.0), Vec3::splat(1.0), [10, 10, 10], 0.7);
    let lights = [
        Light {
            pos: Vec3::new(0.3, 0.2, 0.4),
            radius: 0.1,
            rgb: [7.0, 5.0, 3.0],
        },
        Light {
            pos: Vec3::new(0.0, 0.0, 3.0),
            radius: 0.0,
            rgb: [2.0, 4.0, 6.0],
        },
    ];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: GasLook {
            color: [1.0, 1.0, 1.0],
            emissivity: 0.0,
            opacity: 2.0,
            scatter: Some(ScatterLook {
                strength: 1.5,
                anisotropy: 0.3,
                shadows: true,
                tint: [1.0; 3],
                softening: None,
            }),
        },
    };
    let origin = Vec3::new(0.1, 0.05, 3.0);
    let dir = Vec3::new(0.05, -0.1, -1.0).normalize();
    let sv = bake_shadows(&gas);
    let (shadowed, t_s) = march_gas(&gas, Some(&sv), origin, dir, f32::NEG_INFINITY);
    let (plain, t_p) = march_gas(&gas, None, origin, dir, f32::NEG_INFINITY);
    for k in 0..3 {
        assert!(plain[k] > 0.0, "channel {k}: ray must scatter something");
        assert!(
            shadowed[k] < plain[k],
            "channel {k}: shadowed {} must be strictly below unshadowed {}",
            shadowed[k],
            plain[k]
        );
    }
    assert_eq!(t_s, t_p, "shadows must not touch the camera-path T");
}

/// Off is off, bitwise: the oracle keys on the `shadows` ARGUMENT — the look
/// flag alone changes nothing — and an inactive scatter term (strength 0, or
/// no lights) ignores even a present shadow argument.
#[test]
fn shadowed_march_off_is_bit_identical() {
    let g = pattern_grid(Vec3::splat(-1.0), Vec3::splat(1.0), [6, 6, 6], 1.3);
    let lights = [Light {
        pos: Vec3::new(0.5, 0.2, 2.0),
        radius: 0.2,
        rgb: [7.0, 5.0, 3.0],
    }];
    let origin = Vec3::new(-0.2, 0.15, 3.0);
    let dir = Vec3::new(-0.1, 0.25, -1.0).normalize();
    let run = |scatter: Option<ScatterLook>, lights: &[Light], sv: Option<&ShadowVolumes>| {
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
            sv,
            origin,
            dir,
            f32::NEG_INFINITY,
        )
    };
    let hand = linear_volumes(g.bounds_min, g.bounds_max, 1);
    let on = ScatterLook {
        strength: 2.0,
        anisotropy: 0.5,
        shadows: true,
        tint: [1.0; 3],
        softening: None,
    };
    let off = ScatterLook {
        shadows: false,
        tint: [1.0; 3],
        softening: None,
        ..on
    };
    // The flag alone (no argument) is the v1 unshadowed march, bitwise.
    assert_eq!(
        run(Some(on), &lights, None),
        run(Some(off), &lights, None),
        "shadows flag without a shadow argument must not change the oracle"
    );
    // An inactive scatter term ignores a present (non-trivial) argument.
    let base = run(None, &lights, None);
    assert_eq!(
        run(
            Some(ScatterLook {
                strength: 0.0,
                ..on
            }),
            &lights,
            Some(&hand),
        ),
        base,
        "strength = 0 must ignore the shadow argument"
    );
    assert_eq!(
        run(Some(on), &[], Some(&hand)),
        base,
        "no lights must ignore the shadow argument"
    );
}

// ---------- GPU gates ----------

/// GPU shadowed march ≡ CPU reference, orthographic: the scatter.rs scene
/// (different-bounds pattern grids, non-trivial mix, lights inside AND outside
/// the domain) with `shadows: true` — the on-device bake + trilinear + march
/// against `bake_shadows` + `sample` + `march_gas` end-to-end, per pixel, all
/// four channels, at the volume.rs tolerance (1e-3 relative + 1e-5 absolute).
#[test]
fn gpu_shadowed_scatter_matches_cpu_reference_ortho() {
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
                shadows: true,
                tint: [1.0; 3],
                softening: None,
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

/// GPU shadowed march ≡ CPU reference, perspective: eye rays (per-pixel ω_out
/// varies), same scene and tolerance.
#[test]
fn gpu_shadowed_scatter_matches_cpu_reference_perspective() {
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
                shadows: true,
                tint: [1.0; 3],
                softening: None,
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

/// The GPU off paths stay bitwise: with the shadow flag SET but the scatter
/// term inactive (strength 0, or no lights), the full composite (stars +
/// prepass + gas) is bit-identical to `scatter: None` — the new shadow
/// binding and uniform flag must not disturb the off-path arithmetic.
#[test]
fn gpu_shadow_flag_with_inactive_scatter_stays_bit_identical() {
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
        img(
            Some(ScatterLook {
                strength: 0.0,
                anisotropy: 0.7,
                shadows: true,
                tint: [1.0; 3],
                softening: None,
            }),
            &lights,
        ),
        base,
        "strength = 0 with the shadow flag must be bit-identical to scatter: None"
    );
    assert_eq!(
        img(
            Some(ScatterLook {
                strength: 2.0,
                anisotropy: 0.7,
                shadows: true,
                tint: [1.0; 3],
                softening: None,
            }),
            &[],
        ),
        base,
        "no lights with the shadow flag must be bit-identical to scatter: None"
    );
}

/// GPU linearity survives shadowing (T_k is σ_s-independent): 2× strength ⇒
/// exactly 2× flux with shadows on — and shadows actually bite: the shadowed
/// flux is strictly below the unshadowed flux for the same scene.
#[test]
fn gpu_shadowed_strength_linear_and_shadows_bite() {
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
    let flux = |strength: f32, shadows: bool| {
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
                    shadows,
                    tint: [1.0; 3],
                    softening: None,
                }),
            },
        };
        r.render_frame_with_gas(&FrameData::default(), Some(&gas), &cam, &cfg)
            .unwrap()
            .total_flux()
    };
    let (f1, f2) = (flux(0.8, true), flux(1.6, true));
    for c in 0..3 {
        assert!(f1[c] > 0.0, "gas scattered nothing under shadows");
        let ratio = f2[c] / f1[c];
        assert!(
            (ratio - 2.0).abs() < 1e-7,
            "channel {c}: shadowed flux ratio {ratio} must be exactly 2"
        );
    }
    let plain = flux(0.8, false);
    for c in 0..3 {
        assert!(
            f1[c] < plain[c],
            "channel {c}: shadowed flux {} must be strictly below unshadowed {}",
            f1[c],
            plain[c]
        );
    }
}
