//! Gas voxelization (DESIGN.md M7d, plan D8): SPH kernel deposition of the gas
//! population onto a single-channel density grid, the payload of the frame-data
//! v2 gas block and the input of the M7e volumetric raymarcher.
//!
//! The deposition is the SPH density estimate evaluated at cell centers:
//!
//! ```text
//! ρ(x_cell) = Σ_j m_j · W(|x_cell − x_j|, h_j)
//! ```
//!
//! with the shared `galaxy_solvers::sph` cubic-spline kernel — the kernel is the
//! correct band-limit for SPH data (NOT cloud-in-cell), and reusing it keeps one
//! source of truth for all smoothing math. Smoothing lengths are DERIVED per
//! snapshot via the shared adaptive-h routine (plan D2: `h` is never stored).
//!
//! Determinism: cells GATHER from a hash grid over the particles (neighbor lists
//! ascending, one independent sum per cell), so the result is bit-identical
//! serial vs parallel — the same discipline as `density_adaptive`.
//!
//! The grid carries ρ only: emission/absorption/color are renderer uniforms, so
//! the look iterates at re-render cost, not re-prep (plan D8).

use galaxy_core::{DVec3, State};
use glam::Vec3;

use galaxy_solvers::sph::DensityConfig;

/// A single-channel gas density voxel grid.
///
/// `data` holds one f32 density per cell, x-fastest
/// (`data[ix + dims[0]·(iy + dims[1]·iz)]`) — the layout a wgpu 3D texture
/// upload consumes row-by-row, slice-by-slice. Values live at CELL CENTERS:
/// cell `(ix, iy, iz)` is centered at `bounds_min + (i + ½)·cell_size`.
#[derive(Clone, Debug, PartialEq)]
pub struct GasGrid {
    /// Cell counts per axis (all ≥ 1).
    pub dims: [u32; 3],
    /// World-space minimum corner of the grid volume.
    pub bounds_min: Vec3,
    /// World-space maximum corner of the grid volume.
    pub bounds_max: Vec3,
    /// Densities at cell centers, x-fastest; `dims[0]·dims[1]·dims[2]` values.
    pub data: Vec<f32>,
}

impl GasGrid {
    /// Total number of cells (`dims[0]·dims[1]·dims[2]`).
    pub fn cell_count(&self) -> usize {
        todo!()
    }

    /// Flat index of cell `(ix, iy, iz)` in `data` (x-fastest).
    pub fn index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        todo!()
    }

    /// World-space cell edge lengths (extent / dims), in f64 — the deposition
    /// and sampling both derive cell centers from this one definition.
    pub fn cell_size(&self) -> DVec3 {
        todo!()
    }

    /// World-space center of cell `(ix, iy, iz)`.
    pub fn cell_center(&self, ix: u32, iy: u32, iz: u32) -> DVec3 {
        todo!()
    }

    /// Trilinear density sample at world point `p`: exact cell values at cell
    /// centers, edge-clamped within the outer half-cell ring (the GPU
    /// clamp-to-edge convention), exactly `0.0` outside `[bounds_min, bounds_max]`.
    ///
    /// The interpolation uses the two-product lerp `(1−t)·a + t·b`, so a sample
    /// AT a cell center returns that cell's value bit-exactly — this function is
    /// the CPU reference for the M7e shader's texture sampling.
    pub fn sample(&self, p: Vec3) -> f32 {
        todo!()
    }
}

/// The endpoint-grid mix the M7e shader performs, as a CPU reference:
/// `(1−u)·g0.sample(p) + u·g1.sample(p)` (two-product lerp, so `u = 0` returns
/// `g0`'s sample and `u = 1` returns `g1`'s sample bit-exactly).
pub fn sample_mix(g0: &GasGrid, g1: &GasGrid, u: f32, p: Vec3) -> f32 {
    let _ = (g0, g1, u, p);
    todo!()
}

/// Configuration for [`deposit_gas`].
#[derive(Clone, Debug)]
pub struct GasGridConfig {
    /// Cell counts per axis. Default 128³ (QUICK paths use 64³).
    pub dims: [u32; 3],
    /// The grid volume is a cube centered on the gas centroid whose half-edge is
    /// the `percentile` radius of the gas population (about the centroid) padded
    /// by the full kernel support `2·h_max` — camera-independent, and every
    /// particle inside the percentile radius deposits its whole kernel inside
    /// the box. Clamped to `[0, 1]`; default 0.99.
    pub percentile: f64,
    /// Adaptive-h configuration for the shared `solvers::sph` smoothing-length
    /// solve (plan D2: h is recomputed per snapshot, never stored).
    pub density: DensityConfig,
}

impl Default for GasGridConfig {
    fn default() -> Self {
        GasGridConfig {
            dims: [128; 3],
            percentile: 0.99,
            density: DensityConfig::default(),
        }
    }
}

/// Deposit gas density with CALLER-SUPPLIED smoothing lengths onto the given
/// grid geometry (the fixed-h primitive the unit gates pin; [`deposit_gas`] is
/// the adaptive-h state-level wrapper). Parallel over cells; bit-identical to
/// [`deposit_fixed_serial`] (gated).
///
/// Panics on malformed inputs (zero dims, non-positive bounds extent,
/// non-finite/non-positive `h`, mismatched lengths) — caller contract, not a
/// data condition.
pub fn deposit_fixed(
    pos: &[DVec3],
    mass: &[f64],
    h: &[f64],
    dims: [u32; 3],
    bounds_min: Vec3,
    bounds_max: Vec3,
) -> GasGrid {
    let _ = (pos, mass, h, dims, bounds_min, bounds_max);
    todo!()
}

/// Serial twin of [`deposit_fixed`]: the same per-cell computation without the
/// rayon dispatch, for the parallel ≡ serial bit-exactness gate.
pub fn deposit_fixed_serial(
    pos: &[DVec3],
    mass: &[f64],
    h: &[f64],
    dims: [u32; 3],
    bounds_min: Vec3,
    bounds_max: Vec3,
) -> GasGrid {
    let _ = (pos, mass, h, dims, bounds_min, bounds_max);
    todo!()
}

/// Voxelize the gas population of `state`: select `Species::Gas` rows, derive
/// adaptive smoothing lengths via the shared `solvers::sph` routine, choose
/// cubic bounds (percentile radius about the gas centroid + `2·h_max` pad, see
/// [`GasGridConfig`]), and kernel-deposit onto `cfg.dims` cells.
///
/// Returns `None` when the state holds no gas — a gas-free run has no grid and
/// its frame-data v2 carries no gas block.
pub fn deposit_gas(state: &State, cfg: &GasGridConfig) -> Option<GasGrid> {
    let _ = (state, cfg);
    todo!()
}
