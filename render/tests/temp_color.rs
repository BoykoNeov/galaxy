//! Temperature-dependent gas color gates (plan `incandescent-nebular-veil`, H2).
//!
//! The march colors gas emission by `ū = mix(N)/mix(ρ)` (the SPH mass-weighted
//! specific internal energy, `T ∝ u`) through a fixed cold→hot colormap, instead
//! of the flat [`GasLook::color`]. The oracles are: the colormap endpoints/clamp
//! are exact lerps; a CONSTANT colormap reduces the temperature march to the
//! flat march bit-for-bit (the load-bearing regression — `temperature = None`
//! and a constant map are both bit-compatible with the pre-temperature renderer);
//! a uniform temperature field renders as a flat tint of the mapped color; and a
//! temperature gradient colors hot and cold regions differently.

use galaxy_render::camera::Camera;
use galaxy_render::render::{RenderConfig, Renderer};
use galaxy_render::volume::{
    march_gas, render_gas_cpu, temperature_color, GasFrame, GasLook, TempColor,
};
use galaxy_renderprep::{FrameData, GasGrid};
use glam::{Vec2, Vec3};

// ---------- helpers ----------

/// A uniform grid over `[-1,1]³` at the given constant value.
fn uniform(value: f32, dims: [u32; 3]) -> GasGrid {
    let n = (dims[0] * dims[1] * dims[2]) as usize;
    GasGrid {
        dims,
        bounds_min: Vec3::splat(-1.0),
        bounds_max: Vec3::splat(1.0),
        data: vec![value; n],
    }
}

/// Pure-emission look (κ = 0 ⇒ T ≡ 1, so the march is a clean sum of per-step
/// emission) with the given flat tint.
fn emit_look(color: [f32; 3]) -> GasLook {
    GasLook {
        color,
        emissivity: 1.0,
        opacity: 0.0,
        scatter: None,
    }
}

/// March straight down −z through the grid at lateral `(x, 0)` — the ray stays
/// in one x-half, so a left/right-split grid is sampled on one side only.
fn march_column(gas: &GasFrame, x: f32) -> ([f32; 3], f32) {
    march_gas(
        gas,
        None,
        Vec3::new(x, 0.0, 2.0),
        Vec3::new(0.0, 0.0, -1.0),
        0.0,
    )
}

// ---------- colormap unit oracle ----------

#[test]
fn temperature_color_ramp_endpoints_and_clamp() {
    let tc = TempColor {
        moment0: &uniform(0.0, [1, 1, 1]),
        moment1: &uniform(0.0, [1, 1, 1]),
        cold: [0.1, 0.2, 0.9],
        hot: [1.0, 0.3, 0.05],
        u_lo: 2.0,
        u_hi: 6.0,
    };
    // At/below u_lo ⇒ cold exactly (t = 0 ⇒ (1−0)·cold + 0·hot); at/above u_hi
    // ⇒ hot exactly (t = 1).
    assert_eq!(temperature_color(&tc, 2.0), tc.cold);
    assert_eq!(
        temperature_color(&tc, 0.0),
        tc.cold,
        "below band not clamped"
    );
    assert_eq!(temperature_color(&tc, 6.0), tc.hot);
    assert_eq!(
        temperature_color(&tc, 9.0),
        tc.hot,
        "above band not clamped"
    );
    // Midpoint ū = 4 ⇒ t = 0.5 ⇒ the two-product lerp per channel.
    let mid = temperature_color(&tc, 4.0);
    for (ch, &got) in mid.iter().enumerate() {
        let want = 0.5 * tc.cold[ch] + 0.5 * tc.hot[ch];
        assert_eq!(got, want, "channel {ch} midpoint");
    }
    // Degenerate band (u_hi ≤ u_lo) ⇒ everything maps to cold (no div-by-zero).
    let deg = TempColor {
        u_lo: 5.0,
        u_hi: 5.0,
        ..tc
    };
    assert_eq!(temperature_color(&deg, 5.0), deg.cold);
    assert_eq!(temperature_color(&deg, 99.0), deg.cold);
}

// ---------- reduction to the flat march (bit-identity regression) ----------

#[test]
fn constant_colormap_reduces_to_the_flat_march_bit_for_bit() {
    // cold == hot ⇒ the colormap is the constant look.color for every ū, so the
    // temperature march must be byte-identical to the flat (temperature = None)
    // march — the pre-temperature renderer's arithmetic, unchanged.
    let rho = uniform(0.7, [4, 4, 8]);
    let mom = uniform(3.5, [4, 4, 8]); // arbitrary N; ū is irrelevant when cold==hot
    let tint = [0.3, 0.6, 0.9];

    let flat = GasFrame {
        grid0: &rho,
        grid1: &rho,
        mix: 0.0,
        look: emit_look(tint),
        lights: &[],
        temperature: None,
    };
    let tempered = GasFrame {
        temperature: Some(TempColor {
            moment0: &mom,
            moment1: &mom,
            cold: tint,
            hot: tint,
            u_lo: 0.0,
            u_hi: 10.0,
        }),
        ..flat
    };

    for &x in &[-0.5f32, 0.0, 0.5] {
        assert_eq!(
            march_column(&tempered, x),
            march_column(&flat, x),
            "constant colormap diverged from the flat march at x={x}"
        );
    }
}

#[test]
fn uniform_temperature_renders_as_a_flat_tint_of_the_mapped_color() {
    // Uniform ρ and N = 2·ρ ⇒ ū = 2.0 at every sample (2·ρ/ρ = 2.0 exactly). With
    // the band below ū the color pins to `hot` exactly, so the temperature march
    // equals a flat march whose tint IS `hot`, bit-for-bit.
    let rho = uniform(0.5, [4, 4, 8]);
    let mom = uniform(1.0, [4, 4, 8]); // 2·ρ ⇒ ū = 2.0
    let hot = [0.9, 0.4, 0.2];

    let flat = GasFrame {
        grid0: &rho,
        grid1: &rho,
        mix: 0.0,
        look: emit_look(hot),
        lights: &[],
        temperature: None,
    };
    let tempered = GasFrame {
        look: emit_look([0.05, 0.05, 0.05]), // look.color is unused under a temp map
        temperature: Some(TempColor {
            moment0: &mom,
            moment1: &mom,
            cold: [0.0, 0.1, 0.7],
            hot,
            u_lo: 0.0,
            u_hi: 1.0, // ū = 2.0 ≥ u_hi ⇒ hot exactly
        }),
        ..flat
    };

    assert_eq!(
        march_column(&tempered, 0.0),
        march_column(&flat, 0.0),
        "uniform temperature did not reduce to a flat tint of the mapped color"
    );
}

// ---------- spatial behavior ----------

#[test]
fn temperature_gradient_colors_hot_and_cold_regions_differently() {
    // ρ uniform; N splits left (cold) / right (hot) across x = 0. A blue→red map
    // must make the +x column read red-dominant and the −x column blue-dominant.
    let dims = [2u32, 1, 8];
    let rho = uniform(1.0, dims);
    // N: ix=0 (x<0) cold ⇒ ū=0; ix=1 (x>0) hot ⇒ ū=8. x-fastest, so alternate.
    let mut ndata = Vec::new();
    for _ in 0..(dims[1] * dims[2]) {
        ndata.push(0.0); // ix = 0
        ndata.push(8.0); // ix = 1
    }
    let mom = GasGrid {
        dims,
        bounds_min: Vec3::splat(-1.0),
        bounds_max: Vec3::splat(1.0),
        data: ndata,
    };
    let gas = GasFrame {
        grid0: &rho,
        grid1: &rho,
        mix: 0.0,
        look: emit_look([1.0, 1.0, 1.0]),
        lights: &[],
        temperature: Some(TempColor {
            moment0: &mom,
            moment1: &mom,
            cold: [0.0, 0.0, 1.0], // blue
            hot: [1.0, 0.0, 0.0],  // red
            u_lo: 0.0,
            u_hi: 8.0,
        }),
    };

    let (hot_c, _) = march_column(&gas, 0.5); // +x ⇒ hot half
    let (cold_c, _) = march_column(&gas, -0.5); // −x ⇒ cold half
    assert!(
        hot_c[0] > hot_c[2] && hot_c[0] > 0.0,
        "hot column not red-dominant: {hot_c:?}"
    );
    assert!(
        cold_c[2] > cold_c[0] && cold_c[2] > 0.0,
        "cold column not blue-dominant: {cold_c:?}"
    );
}

#[test]
fn empty_cells_are_guarded_against_zero_over_zero() {
    // ρ = 0 and N = 0 in every cell: ū = 0/floor = 0 (finite), emission e ∝ ρ = 0,
    // so the march returns exactly no radiance with no NaN from the division.
    let rho = uniform(0.0, [4, 4, 4]);
    let mom = uniform(0.0, [4, 4, 4]);
    let gas = GasFrame {
        grid0: &rho,
        grid1: &rho,
        mix: 0.0,
        look: emit_look([1.0, 1.0, 1.0]),
        lights: &[],
        temperature: Some(TempColor {
            moment0: &mom,
            moment1: &mom,
            cold: [0.2, 0.4, 0.9],
            hot: [1.0, 0.5, 0.1],
            u_lo: 1.0,
            u_hi: 5.0,
        }),
    };
    let (c, t) = march_column(&gas, 0.0);
    assert!(c.iter().all(|v| v.is_finite()), "NaN/Inf radiance: {c:?}");
    assert_eq!(c, [0.0, 0.0, 0.0], "zero gas emitted radiance");
    assert_eq!(t, 1.0, "zero gas attenuated light");
}

// ---------- GPU ≡ CPU parity (plan H3) ----------
//
// The WGSL march must mirror temperature_color op-for-op: sample the moment
// grids, divide by ρ, and lerp cold→hot — matching the CPU reference within the
// established volume tolerance (1e-3 rel + 1e-5 abs per channel). The off path
// (temperature = None) is already pinned bit-identical by the M6g golden gate.

fn renderer() -> Renderer {
    Renderer::new().expect("wgpu adapter required for temperature GPU gates")
}

/// A deterministic strictly-positive pattern grid (mirrors volume.rs's helper),
/// used for both ρ and — scaled — the internal-energy moment N.
fn pattern(bounds_min: Vec3, bounds_max: Vec3, dims: [u32; 3], phase: f32, scale: f32) -> GasGrid {
    let mut data = Vec::with_capacity((dims[0] * dims[1] * dims[2]) as usize);
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let v = 0.45
                    + 0.35
                        * (0.7 * ix as f32 + 1.3 * iy as f32 + phase).sin()
                        * (0.5 * iz as f32 + 0.4).cos();
                data.push(scale * v);
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

/// The parity scene: two different-bounds ρ grids, CO-REGISTERED moment grids (a
/// distinct pattern per endpoint so ū = N/ρ varies spatially), an oblique view,
/// a non-trivial mix, and a colormap whose cold/hot differ from look.color (so a
/// GPU that ignored temperature — flat look.color — fails the parity).
fn parity_gas<'a>(
    g0: &'a GasGrid,
    g1: &'a GasGrid,
    m0: &'a GasGrid,
    m1: &'a GasGrid,
) -> GasFrame<'a> {
    GasFrame {
        grid0: g0,
        grid1: g1,
        mix: 0.37,
        look: GasLook {
            color: [0.2, 0.8, 0.3], // deliberately unlike cold/hot
            emissivity: 1.7,
            opacity: 2.1,
            scatter: None,
        },
        lights: &[],
        temperature: Some(TempColor {
            moment0: m0,
            moment1: m1,
            cold: [0.1, 0.2, 0.9],
            hot: [1.0, 0.35, 0.05],
            u_lo: 0.4,
            u_hi: 1.6,
        }),
    }
}

fn assert_gpu_matches_cpu(gas: &GasFrame, cam: &Camera, cfg: &RenderConfig) {
    let r = renderer();
    let gpu = r
        .render_frame_with_gas(&FrameData::default(), Some(gas), cam, cfg)
        .unwrap();
    let cpu = render_gas_cpu(gas, cam, cfg.width, cfg.height);
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

#[test]
fn gpu_temperature_matches_cpu_reference_ortho() {
    let g0 = pattern(
        Vec3::new(-1.2, -1.0, -0.8),
        Vec3::new(1.0, 1.1, 0.9),
        [12, 10, 9],
        0.0,
        1.0,
    );
    let g1 = pattern(
        Vec3::new(-0.9, -1.1, -1.0),
        Vec3::new(1.2, 0.9, 1.1),
        [8, 14, 11],
        1.7,
        1.0,
    );
    // Moment grids: co-registered with their ρ endpoint, scaled ⇒ ū ≈ scale·(N/ρ)
    // sweeps across [u_lo, u_hi] and beyond, exercising both clamp arms.
    let m0 = pattern(g0.bounds_min, g0.bounds_max, g0.dims, 0.9, 1.3);
    let m1 = pattern(g1.bounds_min, g1.bounds_max, g1.dims, 2.4, 0.7);
    let gas = parity_gas(&g0, &g1, &m0, &m1);
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
    assert_gpu_matches_cpu(&gas, &cam, &cfg);
}

#[test]
fn gpu_temperature_matches_cpu_reference_perspective() {
    let g0 = pattern(
        Vec3::new(-1.2, -1.0, -0.8),
        Vec3::new(1.0, 1.1, 0.9),
        [12, 10, 9],
        0.0,
        1.0,
    );
    let g1 = pattern(
        Vec3::new(-0.9, -1.1, -1.0),
        Vec3::new(1.2, 0.9, 1.1),
        [8, 14, 11],
        1.7,
        1.0,
    );
    let m0 = pattern(g0.bounds_min, g0.bounds_max, g0.dims, 0.9, 1.3);
    let m1 = pattern(g1.bounds_min, g1.bounds_max, g1.dims, 2.4, 0.7);
    let gas = parity_gas(&g0, &g1, &m0, &m1);
    let cam = Camera::perspective(
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.2, 0.1, -1.0),
        Vec3::Y,
        Vec2::new(1.2, 0.9),
        4.0,
        0.1,
    );
    let cfg = RenderConfig {
        width: 64,
        height: 48,
        falloff: 6.0,
        ..RenderConfig::default()
    };
    assert_gpu_matches_cpu(&gas, &cam, &cfg);
}
