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
    todo!("M7e: ½·min cell edge over both grids")
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
    todo!("M7e: per-pixel ray generation, ortho + perspective")
}

/// CPU reference for the gas fragment march along one ray (module-doc march
/// rule verbatim): returns `(accumulated RGB radiance, final transmittance)`.
///
/// `t_min` clamps the chord start: perspective passes `0.0` (nothing behind
/// the eye), orthographic passes `f32::NEG_INFINITY` (the full chord). A ray
/// that misses both grids returns `([0,0,0], 1.0)`.
pub fn march_gas(gas: &GasFrame, origin: Vec3, dir: Vec3, t_min: f32) -> ([f32; 3], f32) {
    todo!("M7e: CPU mirror of the fragment-shader gas march")
}

/// CPU reference for the transmittance compute prepass: `T = exp(−τ)` with
/// `τ = ∫ κ·ρ_mix ds` over the segment from `star` toward the camera (to the
/// eye for perspective, to the grid exit along `−forward` for orthographic),
/// clipped against the union grid AABB, same step rule as [`march_gas`].
/// A star with no gas in front returns exactly `1.0`.
pub fn star_transmittance(gas: &GasFrame, camera: &Camera, star: Vec3) -> f32 {
    todo!("M7e: CPU mirror of the per-star transmittance prepass")
}

/// Render the gas pass alone on the CPU: [`ray_for_pixel`] + [`march_gas`] per
/// pixel, `RGB = radiance`, `alpha = 1 − transmittance` (the per-pixel gas
/// opacity) — exactly what the GPU fullscreen pass additively blends into the
/// cleared target. This is the oracle image for the GPU ≡ CPU gates (small
/// resolutions only; it is a reference, not a fast path).
pub fn render_gas_cpu(gas: &GasFrame, camera: &Camera, width: u32, height: u32) -> HdrImage {
    todo!("M7e: CPU reference image of the gas pass")
}
