//! Scatter-softening decoupling gates (galaxy-render "more controls" pass):
//! `ScatterLook::softening: Option<f32>`.
//!
//! The single-scatter incident term softens the 1/d² pole with a length that,
//! in the v1 path, is each light cluster's OWN cell radius `r_k`
//! (`d² + r_k²`). Because the octree cut refines with `REFINE_TOL`, that radius
//! — and hence the scattered brightness near the dense cores — silently tracks
//! the tolerance: a hidden brightness knob. `softening: Some(ε)` replaces `r_k`
//! with a single fixed ε (floored at the gas voxel scale), so the INTEGRATED
//! scattered energy is invariant to the refinement tolerance. `None` is the v1
//! radius softening, bit-identical to the shipped path.
//!
//! Gates (CPU oracle; the WGSL mirror is held to `render_gas_cpu` by the
//! existing GPU≡CPU scatter suite):
//!   1. **Tol-invariance** (headline): under `Some(ε)`, total scattered energy
//!      from a fixed star clump is invariant across two refinement tolerances.
//!   2. **Coupling control**: under `None`, the SAME two tolerances give
//!      materially different totals — the bug the fix removes.
//!   3. **Voxel floor**: a sub-voxel ε floors to the voxel scale; ε actually
//!      changes the result (distinct ε → distinct render).
//!   4. **Single-light bridge**: for one light, `Some(r)` (r ≥ floor) reduces to
//!      the `None` arithmetic `d² + r²` bit-for-bit; a different ε does not.

use galaxy_render::camera::Camera;
use galaxy_render::volume::{
    cluster_lights_with, march_gas, render_gas_cpu, GasFrame, GasLook, Light, ScatterLook,
    MAX_LIGHTS,
};
use galaxy_renderprep::{FrameData, GasGrid};
use glam::{Vec2, Vec3};

// ---------- fixtures ----------

/// Deterministic LCG in [0, 1) — the clump and the octree cut must reproduce.
struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
    /// Approx N(0, 1): sum of 3 uniforms, centered (Irwin–Hall).
    fn gauss(&mut self) -> f32 {
        self.unit() + self.unit() + self.unit() - 1.5
    }
}

/// A cube of uniform gas centered on the origin.
const CUBE_HALF: f32 = 1.0;
const CUBE_DIM: u32 = 24;
const CUBE_RHO: f32 = 0.5;
const CUBE_KAPPA: f32 = 0.6;
/// The gas voxel edge — the softening floor the `Some(ε)` path clamps to.
const VOXEL_EDGE: f32 = 2.0 * CUBE_HALF / CUBE_DIM as f32; // 1/12 ≈ 0.0833

fn cube_grid() -> GasGrid {
    let n = (CUBE_DIM * CUBE_DIM * CUBE_DIM) as usize;
    GasGrid {
        dims: [CUBE_DIM; 3],
        bounds_min: Vec3::splat(-CUBE_HALF),
        bounds_max: Vec3::splat(CUBE_HALF),
        data: vec![CUBE_RHO; n],
    }
}

/// A compact stellar clump near the origin: 400 warm stars, σ ≈ 0.12 (tighter
/// than the voxel floor's reach, so a fixed ε ≥ floor is smooth over it — the
/// regime where the tol-invariance is sharp). Deterministic.
fn clump() -> FrameData {
    let mut rng = Lcg(0xC0FFEE_u64);
    let n = 400;
    let (mut pos, mut color, mut brightness) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..n {
        pos.push(Vec3::new(
            0.12 * rng.gauss(),
            0.12 * rng.gauss(),
            0.12 * rng.gauss(),
        ));
        color.push([1.0, 0.6 + 0.2 * rng.unit(), 0.3]);
        brightness.push(1.0 + 3.0 * rng.unit());
    }
    FrameData {
        pos,
        color,
        size: vec![0.05; n],
        brightness,
    }
}

fn look(scatter: Option<ScatterLook>) -> GasLook {
    GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 0.0, // isolate the scattered term
        opacity: CUBE_KAPPA,
        scatter,
    }
}

/// Total scattered radiance the camera collects from the clump — Σ over pixels
/// of RGB. Emission is off, so this is purely the single-scatter term weighted
/// by the (tol-independent) camera-path transmittance.
fn total_scatter(grid: &GasGrid, softening: Option<f32>, aniso: f32, lights: &[Light]) -> f64 {
    let gas = GasFrame {
        grid0: grid,
        grid1: grid,
        mix: 0.0,
        lights,
        look: look(Some(ScatterLook {
            strength: 1.0,
            anisotropy: aniso,
            shadows: false,
            tint: [1.0; 3],
            softening,
        })),
    };
    let cam = Camera::orthographic(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, Vec2::new(1.0, 1.0));
    let img = render_gas_cpu(&gas, &cam, 20, 20);
    img.pixels
        .iter()
        .map(|p| p[0] as f64 + p[1] as f64 + p[2] as f64)
        .sum()
}

const TOL_COARSE: f64 = 3e-1;
const TOL_FINE: f64 = 3e-3;

// ---------- gate 1: tol-invariance under fixed ε ----------

#[test]
fn fixed_epsilon_scatter_is_refinement_invariant() {
    let frame = clump();
    let grid = cube_grid();
    let coarse = cluster_lights_with(&frame, TOL_COARSE, MAX_LIGHTS);
    let fine = cluster_lights_with(&frame, TOL_FINE, MAX_LIGHTS);
    assert!(
        fine.len() > coarse.len(),
        "fixture must actually refine: coarse {} vs fine {}",
        coarse.len(),
        fine.len()
    );

    // ε ≈ 0.5 ≫ the σ≈0.12 clump, so the incident kernel is smooth over the
    // cluster extent and the luminosity-weighted centroid carries it: the total
    // is invariant to how finely the clump is cut. (Measured coupling under the
    // v1 radius softening on this same fixture is ~12% — see the control gate.)
    let eps = Some(6.0 * VOXEL_EDGE);
    let e_coarse = total_scatter(&grid, eps, 0.0, &coarse);
    let e_fine = total_scatter(&grid, eps, 0.0, &fine);
    let rel = (e_fine - e_coarse).abs() / e_coarse;
    assert!(
        rel < 0.05,
        "fixed ε must be refinement-invariant: coarse {e_coarse} vs fine {e_fine} (rel {rel})"
    );
}

// ---------- gate 2: the coupling the fix removes (control) ----------

#[test]
fn radius_softening_scatter_tracks_refinement() {
    let frame = clump();
    let grid = cube_grid();
    let coarse = cluster_lights_with(&frame, TOL_COARSE, MAX_LIGHTS);
    let fine = cluster_lights_with(&frame, TOL_FINE, MAX_LIGHTS);

    // None = v1 per-cluster radius softening: the SAME clump renders materially
    // brighter as it refines (smaller radii ⇒ less softening near the core).
    let e_coarse = total_scatter(&grid, None, 0.0, &coarse);
    let e_fine = total_scatter(&grid, None, 0.0, &fine);
    let rel = (e_fine - e_coarse).abs() / e_coarse;
    assert!(
        rel > 0.08,
        "radius softening must visibly track REFINE_TOL (the coupling): \
         coarse {e_coarse} vs fine {e_fine} (rel {rel})"
    );
}

// ---------- gate 3: voxel-scale floor + ε is live ----------

#[test]
fn epsilon_floors_at_voxel_scale_and_is_live() {
    let frame = clump();
    let grid = cube_grid();
    let lights = cluster_lights_with(&frame, TOL_FINE, MAX_LIGHTS);

    // A sub-voxel ε floors to the voxel edge: identical to Some(VOXEL_EDGE).
    let sub = total_scatter(&grid, Some(1e-9), 0.0, &lights);
    let floored = total_scatter(&grid, Some(VOXEL_EDGE), 0.0, &lights);
    assert_eq!(
        sub.to_bits(),
        floored.to_bits(),
        "sub-voxel ε must floor to the voxel scale: {sub} vs {floored}"
    );

    // ε is a real knob: a larger ε spreads/softens ⇒ a different (lower) total.
    let wide = total_scatter(&grid, Some(9.0 * VOXEL_EDGE), 0.0, &lights);
    assert!(
        (wide - floored).abs() / floored > 0.05,
        "distinct ε must change the render: floored {floored} vs wide {wide}"
    );
}

// ---------- gate 4: single-light bridge to the None arithmetic ----------

#[test]
fn single_light_epsilon_matches_radius_arithmetic() {
    let grid = cube_grid();
    let r = 0.15_f32; // ≥ VOXEL_EDGE, so no flooring
    let light = Light {
        pos: Vec3::new(0.1, -0.05, 0.0),
        radius: r,
        rgb: [3.0e2, 2.0e2, 1.0e2],
    };
    let origin = Vec3::new(0.05, 0.0, 5.0);
    let dir = Vec3::new(0.0, 0.0, -1.0);
    let march = |softening: Option<f32>| {
        let gas = GasFrame {
            grid0: &grid,
            grid1: &grid,
            mix: 0.0,
            lights: std::slice::from_ref(&light),
            look: look(Some(ScatterLook {
                strength: 1.3,
                anisotropy: 0.0,
                shadows: false,
                tint: [1.0; 3],
                softening,
            })),
        };
        march_gas(&gas, None, origin, dir, f32::NEG_INFINITY).0
    };

    // Some(r) with r = the light's own radius reduces to d² + r² — bit-identical
    // to the None path for this single light.
    let none = march(None);
    let eps_r = march(Some(r));
    for k in 0..3 {
        assert_eq!(
            none[k].to_bits(),
            eps_r[k].to_bits(),
            "Some(r) must equal None for a single light, channel {k}: {} vs {}",
            none[k],
            eps_r[k]
        );
    }
    // A different ε must NOT collapse to the same arithmetic.
    let eps_2r = march(Some(2.0 * r));
    assert!(
        (0..3).any(|k| eps_2r[k].to_bits() != none[k].to_bits()),
        "a different ε must change the scattered radiance: {eps_2r:?} vs {none:?}"
    );
}
