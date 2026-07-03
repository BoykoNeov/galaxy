//! Volumetric gas rendering (M7e, plan D9): the shared march rules and their
//! CPU reference implementations.
//!
//! The frame the renderer produces is
//!
//! ```text
//! L(pixel) = Σ_stars E·T(camera→star)  +  ∫ j(ρ)·T(camera→s) ds
//! ```
//!
//! — both terms ADDITIVE once each carries its own attenuation, so the
//! order-independent `Rgba32Float` additive target survives intact. This module
//! defines the exact numerical rules (ray generation, AABB clipping, step size,
//! quadrature, early exit) and implements them on the CPU; the WGSL shaders in
//! `render.rs` mirror them operation-for-operation, and the GPU gates in
//! `tests/volume.rs` hold the two within f32 tolerance. Density sampling
//! delegates to [`GasGrid::sample`] / [`sample_mix`] — the renderprep functions
//! documented as the M7e shader oracle.
//!
//! **March rule** (one definition, used by every path):
//! - Domain: the ray is clipped against the UNION AABB of both endpoint grids.
//!   Orthographic rays march the full chord (the splat path draws at all
//!   depths — the ortho camera sits at infinity, nothing is "behind" it);
//!   perspective rays start no earlier than the eye (`t ≥ 0`).
//! - Steps: `Δs_nominal =` [`step_size`] `= ½·min cell edge over both grids`;
//!   the chord `[t0, t1]` is divided into `n = ceil((t1−t0)/Δs_nominal)` equal
//!   steps (capped at [`MAX_STEPS`]) and ρ is sampled at step MIDPOINTS.
//! - Quadrature (gas pass): per step, emit THEN attenuate:
//!   `C += T·(emissivity·ρ)·Δs·color;  T *= exp(−κ·ρ·Δs)` — first-order in the
//!   emission/absorption coupling (relative error ≈ ½·κρΔs per step, gated
//!   against the analytic uniform slab), while the accumulated optical depth
//!   itself is midpoint-exact. Early exit when `T <` [`EXIT_TRANSMITTANCE`]
//!   (truncation error ≤ EXIT_TRANSMITTANCE·(emissivity/κ), gated).
//! - Star transmittance: the same clip + step rule over the star→camera
//!   segment, but pure optical depth `τ = Σ κ·ρ·Δs` summed then exponentiated
//!   once (`T = exp(−τ)`) — no emission, no early exit.
//!
//! Emission is `j(ρ) = emissivity·ρ` per unit length, tinted by `color`;
//! absorption is `κ·ρ` per unit length. Both are LOOK uniforms ([`GasLook`]),
//! not baked into the grid — the look iterates at re-render cost (plan D8).

use glam::Vec3;

use galaxy_renderprep::{sample_mix, GasGrid};

use crate::camera::{Camera, Projection};
use crate::render::HdrImage;

/// Early-exit threshold for the gas march: once transmittance falls below this,
/// everything behind contributes less than this fraction of the saturated
/// radiance `emissivity/κ` — the march stops. Shared verbatim by the WGSL
/// fragment shader (injected into the source) and the CPU mirror.
pub const EXIT_TRANSMITTANCE: f32 = 1e-4;

/// Hard cap on march steps per ray — a backstop against degenerate step/chord
/// ratios (never reached by sane grids: a 128³ diagonal at half-cell steps is
/// ~443 steps). Shared by shader and CPU mirror so both truncate identically.
pub const MAX_STEPS: u32 = 1 << 20;

/// Gas look uniforms (plan D8: the grid carries ρ only; everything visual lives
/// here and iterates at re-render cost).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasLook {
    /// Linear RGB tint of the gas emission.
    pub color: [f32; 3],
    /// Emissivity `j`: emitted radiance per unit (ρ · path length).
    pub emissivity: f32,
    /// Opacity `κ`: extinction per unit (ρ · path length). `0` disables
    /// absorption entirely (transmittance ≡ 1 — the emission-only mode, and the
    /// bit-compat limit the gas-off golden gate pins).
    pub opacity: f32,
}

impl Default for GasLook {
    fn default() -> Self {
        GasLook {
            color: [1.0, 1.0, 1.0],
            emissivity: 1.0,
            opacity: 1.0,
        }
    }
}

/// One frame's gas input: the two snapshot-endpoint density grids and the
/// subframe mix `u` (M6c endpoint argument: grids are deposited ONLY at
/// snapshot endpoints; in-betweens blend the two, `ρ = (1−u)·ρ₀ + u·ρ₁`,
/// exactly [`sample_mix`]). A static frame passes the same grid twice with
/// `mix = 0`.
#[derive(Clone, Copy, Debug)]
pub struct GasFrame<'a> {
    /// Density grid at the earlier snapshot endpoint.
    pub grid0: &'a GasGrid,
    /// Density grid at the later snapshot endpoint.
    pub grid1: &'a GasGrid,
    /// Endpoint blend factor `u ∈ [0, 1]`: `0` = `grid0`, `1` = `grid1`.
    pub mix: f32,
    /// Emission/absorption look knobs.
    pub look: GasLook,
}

/// The shared nominal step: half the smallest cell edge over BOTH endpoint
/// grids — fine enough that the grid's own band-limit (the deposition kernel)
/// dominates the quadrature error, coarse enough that a full-frame march stays
/// trivial GPU work.
pub fn step_size(grid0: &GasGrid, grid1: &GasGrid) -> f32 {
    let min_edge = |g: &GasGrid| {
        let c = g.cell_size();
        c.x.min(c.y).min(c.z)
    };
    (0.5 * min_edge(grid0).min(min_edge(grid1))) as f32
}

/// The march domain: the union AABB of both endpoint grids (each grid's own
/// sample function zeroes outside its own bounds, so the union over-covers
/// harmlessly and one clip serves both).
fn union_bounds(gas: &GasFrame) -> (Vec3, Vec3) {
    (
        gas.grid0.bounds_min.min(gas.grid1.bounds_min),
        gas.grid0.bounds_max.max(gas.grid1.bounds_max),
    )
}

/// Slab-clip the ray `origin + t·dir` against `[bmin, bmax]`: `Some((t0, t1))`
/// for a non-empty chord, `None` for a miss. Axes where `|dir| < 1e-12` are
/// resolved by an inside test instead of dividing (no 0·∞ NaNs). Mirrored
/// operation-for-operation by the WGSL `clip_aabb` (same ±1e30 sentinels).
fn clip_aabb(origin: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let mut t0 = -1e30_f32;
    let mut t1 = 1e30_f32;
    for a in 0..3 {
        if dir[a].abs() < 1e-12 {
            if origin[a] < bmin[a] || origin[a] > bmax[a] {
                return None;
            }
        } else {
            let ta = (bmin[a] - origin[a]) / dir[a];
            let tb = (bmax[a] - origin[a]) / dir[a];
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
        }
    }
    (t0 < t1).then_some((t0, t1))
}

/// The mixed density the march samples: exactly [`sample_mix`], the renderprep
/// CPU reference for the shader's two-texture blend.
fn density_at(gas: &GasFrame, p: Vec3) -> f32 {
    sample_mix(gas.grid0, gas.grid1, gas.mix, p)
}

/// Step count and effective step for a chord `[t0, t1]`: `n` equal steps of
/// the nominal size rounded up, capped at [`MAX_STEPS`].
fn steps(t0: f32, t1: f32, ds_nominal: f32) -> (u32, f32) {
    let n = (((t1 - t0) / ds_nominal).ceil() as u32).clamp(1, MAX_STEPS);
    (n, (t1 - t0) / n as f32)
}

/// The camera ray through the CENTER of pixel `(px, py)` of a `width × height`
/// image (top-left origin, matching [`HdrImage`]): returns `(origin, dir)`,
/// `dir` unit length.
///
/// Orthographic: origin on the target plane at the pixel's world position,
/// direction `forward` (all rays parallel). Perspective: origin at the eye
/// (`target − forward·distance`), direction through the pixel's point on the
/// target plane. NDC convention: pixel centers at `(px+½, py+½)`, `x` right,
/// `y` UP (row 0 is NDC y = +1) — exactly the splat vertex path's mapping, so
/// gas and stars agree per pixel. Pinned by hand oracles at corner pixels.
pub fn ray_for_pixel(camera: &Camera, width: u32, height: u32, px: u32, py: u32) -> (Vec3, Vec3) {
    let ndc_x = (px as f32 + 0.5) / (width as f32 / 2.0) - 1.0;
    let ndc_y = 1.0 - (py as f32 + 0.5) / (height as f32 / 2.0);
    let lateral =
        camera.right * (ndc_x * camera.half_extent.x) + camera.up * (ndc_y * camera.half_extent.y);
    match camera.projection {
        Projection::Orthographic => (camera.target + lateral, camera.forward),
        Projection::Perspective { distance, .. } => {
            let eye = camera.target - camera.forward * distance;
            (eye, (camera.target + lateral - eye).normalize())
        }
    }
}

/// CPU reference for the gas fragment march along one ray (module-doc march
/// rule verbatim): returns `(accumulated RGB radiance, final transmittance)`.
///
/// `t_min` clamps the chord start: perspective passes `0.0` (nothing behind
/// the eye), orthographic passes `f32::NEG_INFINITY` (the full chord). A ray
/// that misses both grids returns `([0,0,0], 1.0)`.
pub fn march_gas(gas: &GasFrame, origin: Vec3, dir: Vec3, t_min: f32) -> ([f32; 3], f32) {
    let (bmin, bmax) = union_bounds(gas);
    let Some((t0_raw, t1)) = clip_aabb(origin, dir, bmin, bmax) else {
        return ([0.0; 3], 1.0);
    };
    let t0 = t0_raw.max(t_min);
    if t0 >= t1 {
        return ([0.0; 3], 1.0);
    }
    let (n, ds) = steps(t0, t1, step_size(gas.grid0, gas.grid1));

    let mut t = 1.0_f32;
    let mut c = [0.0_f32; 3];
    for i in 0..n {
        let s = t0 + (i as f32 + 0.5) * ds;
        let rho = density_at(gas, origin + dir * s);
        // Emit THEN attenuate (module-doc quadrature rule), the exact operation
        // order of the WGSL march.
        let e = t * gas.look.emissivity * rho * ds;
        c[0] += e * gas.look.color[0];
        c[1] += e * gas.look.color[1];
        c[2] += e * gas.look.color[2];
        t *= (-(gas.look.opacity * rho * ds)).exp();
        if t < EXIT_TRANSMITTANCE {
            break;
        }
    }
    (c, t)
}

/// CPU reference for the transmittance compute prepass: `T = exp(−τ)` with
/// `τ = ∫ κ·ρ_mix ds` over the segment from `star` toward the camera (to the
/// eye for perspective, to the grid exit along `−forward` for orthographic),
/// clipped against the union grid AABB, same step rule as [`march_gas`].
/// A star with no gas in front returns exactly `1.0`.
pub fn star_transmittance(gas: &GasFrame, camera: &Camera, star: Vec3) -> f32 {
    let (dir, t_max) = match camera.projection {
        Projection::Orthographic => (-camera.forward, f32::INFINITY),
        Projection::Perspective { distance, .. } => {
            let eye = camera.target - camera.forward * distance;
            let d = eye - star;
            let dist = d.length();
            if dist == 0.0 {
                return 1.0; // star at the eye: zero path, unattenuated
            }
            (d / dist, dist)
        }
    };
    let (bmin, bmax) = union_bounds(gas);
    let Some((t0_raw, t1_raw)) = clip_aabb(star, dir, bmin, bmax) else {
        return 1.0;
    };
    // Only gas IN FRONT of the star (toward the camera, and no farther than
    // the eye) attenuates it.
    let t0 = t0_raw.max(0.0);
    let t1 = t1_raw.min(t_max);
    if t0 >= t1 {
        return 1.0;
    }
    let (n, ds) = steps(t0, t1, step_size(gas.grid0, gas.grid1));

    // Pure optical depth: sum τ, exponentiate once (no emission, no early
    // exit) — the exact operation order of the WGSL compute prepass.
    let mut tau = 0.0_f32;
    for i in 0..n {
        let s = t0 + (i as f32 + 0.5) * ds;
        tau += gas.look.opacity * density_at(gas, star + dir * s) * ds;
    }
    (-tau).exp()
}

/// Render the gas pass alone on the CPU: [`ray_for_pixel`] + [`march_gas`] per
/// pixel, `RGB = radiance`, `alpha = 1 − transmittance` (the per-pixel gas
/// opacity) — exactly what the GPU fullscreen pass additively blends into the
/// cleared target. This is the oracle image for the GPU ≡ CPU gates (small
/// resolutions only; it is a reference, not a fast path).
pub fn render_gas_cpu(gas: &GasFrame, camera: &Camera, width: u32, height: u32) -> HdrImage {
    let t_min = match camera.projection {
        Projection::Orthographic => f32::NEG_INFINITY, // the full chord
        Projection::Perspective { .. } => 0.0,         // nothing behind the eye
    };
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for py in 0..height {
        for px in 0..width {
            let (origin, dir) = ray_for_pixel(camera, width, height, px, py);
            let (c, t) = march_gas(gas, origin, dir, t_min);
            pixels.push([c[0], c[1], c[2], 1.0 - t]);
        }
    }
    HdrImage {
        width,
        height,
        pixels,
    }
}
