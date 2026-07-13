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
//! Determinism: deposition scatters by z-plane — each plane walks the particles
//! in ascending index and adds each kernel over the cell box its support covers,
//! so every cell's sum associates in ascending particle order and planes are
//! disjoint units of parallel work. The result is bit-identical serial vs
//! parallel — the same discipline as `density_adaptive`.
//!
//! The grid carries ρ only: emission/absorption/color are renderer uniforms, so
//! the look iterates at re-render cost, not re-prep (plan D8).

use galaxy_core::{DVec3, Species, State};
use glam::Vec3;
use rayon::prelude::*;

use galaxy_solvers::sph::{density_adaptive, w, DensityConfig, SUPPORT};

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
        self.dims.iter().map(|&d| d as usize).product()
    }

    /// Flat index of cell `(ix, iy, iz)` in `data` (x-fastest).
    pub fn index(&self, ix: u32, iy: u32, iz: u32) -> usize {
        debug_assert!(ix < self.dims[0] && iy < self.dims[1] && iz < self.dims[2]);
        ix as usize + self.dims[0] as usize * (iy as usize + self.dims[1] as usize * iz as usize)
    }

    /// World-space cell edge lengths (extent / dims), in f64 — the deposition
    /// and sampling both derive cell centers from this one definition.
    pub fn cell_size(&self) -> DVec3 {
        let extent = self.bounds_max.as_dvec3() - self.bounds_min.as_dvec3();
        DVec3::new(
            extent.x / self.dims[0] as f64,
            extent.y / self.dims[1] as f64,
            extent.z / self.dims[2] as f64,
        )
    }

    /// World-space center of cell `(ix, iy, iz)`.
    pub fn cell_center(&self, ix: u32, iy: u32, iz: u32) -> DVec3 {
        let cell = self.cell_size();
        self.bounds_min.as_dvec3()
            + DVec3::new(
                (ix as f64 + 0.5) * cell.x,
                (iy as f64 + 0.5) * cell.y,
                (iz as f64 + 0.5) * cell.z,
            )
    }

    /// Trilinear density sample at world point `p`: exact cell values at cell
    /// centers, edge-clamped within the outer half-cell ring (the GPU
    /// clamp-to-edge convention), exactly `0.0` outside `[bounds_min, bounds_max]`.
    ///
    /// The interpolation uses the two-product lerp `(1−t)·a + t·b`, so a sample
    /// AT a cell center returns that cell's value bit-exactly — this function is
    /// the CPU reference for the M7e shader's texture sampling.
    pub fn sample(&self, p: Vec3) -> f32 {
        if p.cmplt(self.bounds_min).any() || p.cmpgt(self.bounds_max).any() {
            return 0.0;
        }
        let cell = self.cell_size();
        let q = p.as_dvec3() - self.bounds_min.as_dvec3();
        // Continuous cell coordinate: cell center i sits at coordinate i.
        let cx = q.x / cell.x - 0.5;
        let cy = q.y / cell.y - 0.5;
        let cz = q.z / cell.z - 0.5;

        // Per-axis floor index + fraction, edge-clamped (clamp-to-edge): within
        // the outer half-cell ring the fraction pins to the boundary cell.
        let axis = |c: f64, d: u32| -> (u32, u32, f32) {
            let max = (d - 1) as f64;
            let c = c.clamp(0.0, max);
            let i0 = c.floor().min(max - 1.0).max(0.0) as u32; // d = 1 ⇒ i0 = 0
            let i1 = (i0 + 1).min(d - 1);
            let t = (c - i0 as f64) as f32;
            (i0, i1, t)
        };
        let (x0, x1, tx) = axis(cx, self.dims[0]);
        let (y0, y1, ty) = axis(cy, self.dims[1]);
        let (z0, z1, tz) = axis(cz, self.dims[2]);

        // Two-product lerp: bit-exact at t = 0 and t = 1 (1·a + 0·b).
        let lerp = |a: f32, b: f32, t: f32| (1.0 - t) * a + t * b;
        let at = |ix: u32, iy: u32, iz: u32| self.data[self.index(ix, iy, iz)];

        let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), tx);
        let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), tx);
        let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), tx);
        let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), tx);
        let c0 = lerp(c00, c10, ty);
        let c1 = lerp(c01, c11, ty);
        lerp(c0, c1, tz)
    }
}

/// The endpoint-grid mix the M7e shader performs, as a CPU reference:
/// `(1−u)·g0.sample(p) + u·g1.sample(p)` (two-product lerp, so `u = 0` returns
/// `g0`'s sample and `u = 1` returns `g1`'s sample bit-exactly).
pub fn sample_mix(g0: &GasGrid, g1: &GasGrid, u: f32, p: Vec3) -> f32 {
    (1.0 - u) * g0.sample(p) + u * g1.sample(p)
}

/// Configuration for [`deposit_gas`].
#[derive(Clone, Debug)]
pub struct GasGridConfig {
    /// Cell counts per axis. Default 128³ (QUICK paths use 64³).
    pub dims: [u32; 3],
    /// The grid volume is a cube centered on the gas centroid whose half-edge is
    /// the `percentile` radius of the gas population (about the centroid) padded
    /// by the median kernel support `2·h_med` — camera-independent, and robust
    /// to the huge adaptive `h` of sparse outskirt/tidal particles (a `h_max`
    /// pad let one such particle inflate the cube and dilute the gas). Clamped
    /// to `[0, 1]`; default 0.99.
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
/// the adaptive-h state-level wrapper). Parallel over z-planes; bit-identical
/// to [`deposit_fixed_serial`] (gated).
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
    deposit_impl(pos, mass, h, dims, bounds_min, bounds_max, true)
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
    deposit_impl(pos, mass, h, dims, bounds_min, bounds_max, false)
}

/// Deposit the internal-energy MOMENT `N = Σ_j (m_j·u_j)·W(|x_cell − x_j|, h_j)`
/// — the identical scatter-by-plane deposition as [`deposit_fixed`] but with the
/// per-particle weight `m_j·u_j` instead of `m_j`. Divided by the co-registered
/// density grid, `ū = N/ρ` is the SPH mass-weighted specific internal energy at
/// each cell — the temperature field (`T ∝ u`) the raymarcher colors by (plan
/// `incandescent-nebular-veil`). The result is a `GasGrid` (a scalar field at
/// cell centers, sampled by the same [`GasGrid::sample`]); only its physical
/// meaning differs from a density grid.
///
/// Bit-exact `parallel ≡ serial` holds for `N` by the same `+0.0`
/// scatter-by-plane argument as ρ. Panics on the same malformed inputs as
/// [`deposit_fixed`], plus a `u` length mismatch.
pub fn deposit_moment_fixed(
    pos: &[DVec3],
    mass: &[f64],
    u: &[f64],
    h: &[f64],
    dims: [u32; 3],
    bounds_min: Vec3,
    bounds_max: Vec3,
) -> GasGrid {
    let _ = (pos, mass, u, h, dims, bounds_min, bounds_max);
    todo!("H1: deposit N = Σ (m_j·u_j)·W via the weighted deposit_impl")
}

/// Serial twin of [`deposit_moment_fixed`] for the parallel ≡ serial gate.
pub fn deposit_moment_fixed_serial(
    pos: &[DVec3],
    mass: &[f64],
    u: &[f64],
    h: &[f64],
    dims: [u32; 3],
    bounds_min: Vec3,
    bounds_max: Vec3,
) -> GasGrid {
    let _ = (pos, mass, u, h, dims, bounds_min, bounds_max);
    todo!("H1: serial N deposition")
}

#[allow(clippy::too_many_arguments)]
fn deposit_impl(
    pos: &[DVec3],
    weight: &[f64],
    h: &[f64],
    dims: [u32; 3],
    bounds_min: Vec3,
    bounds_max: Vec3,
    parallel: bool,
) -> GasGrid {
    assert_eq!(weight.len(), pos.len(), "weight length must match pos");
    assert_eq!(h.len(), pos.len(), "h length must match pos");
    assert!(
        dims.iter().all(|&d| d >= 1),
        "grid dims must be ≥ 1, got {dims:?}"
    );
    assert!(
        bounds_max.cmpgt(bounds_min).all(),
        "grid bounds must have positive extent: {bounds_min:?}..{bounds_max:?}"
    );

    let mut grid = GasGrid {
        dims,
        bounds_min,
        bounds_max,
        data: Vec::new(),
    };
    let n_cells = grid.cell_count();
    if pos.is_empty() {
        grid.data = vec![0.0; n_cells];
        return grid;
    }

    assert!(
        h.iter().all(|&x| x.is_finite() && x > 0.0),
        "deposition needs positive finite smoothing lengths"
    );

    // Scatter-by-plane: each z-plane walks the particles in ASCENDING index and
    // adds every particle's kernel over the cell box its support covers. Per
    // cell that is the same ascending-index association order as a gather —
    // terms a global gather would also visit but that lie outside this
    // particle's own support are exact `+0.0`s and cannot change the partial
    // sums — while doing only O(support volume) kernel work per particle
    // instead of O(2·h_max-ball candidates) per cell (adaptive h makes h_max a
    // far-outskirts value, which turned the naive gather quadratic in dense
    // regions). Planes are disjoint, so rayon over planes is race-free and
    // bit-identical to the serial loop.
    let cell = grid.cell_size();
    let bmin = bounds_min.as_dvec3();
    let (dx, dy) = (dims[0] as usize, dims[1] as usize);

    // Inclusive index range of cell centers within `±r` of `x` on one axis,
    // padded by one cell against fp boundary rounding (an extra cell is a
    // kernel evaluation at q ≥ 2 → exact 0.0). `None` when the support misses
    // the axis entirely.
    let axis_range = |x: f64, r: f64, min: f64, step: f64, n: u32| -> Option<(u32, u32)> {
        let lo = (((x - r - min) / step - 0.5).ceil() as i64 - 1).max(0);
        let hi = (((x + r - min) / step - 0.5).floor() as i64 + 1).min(n as i64 - 1);
        (lo <= hi).then_some((lo as u32, hi as u32))
    };

    let plane = |iz: u32| -> Vec<f64> {
        let mut acc = vec![0.0_f64; dx * dy];
        let cz = bmin.z + (iz as f64 + 0.5) * cell.z;
        for j in 0..pos.len() {
            let r = SUPPORT * h[j];
            if (cz - pos[j].z).abs() > r + cell.z {
                continue; // whole plane outside this particle's support (+pad)
            }
            let Some((x_lo, x_hi)) = axis_range(pos[j].x, r, bmin.x, cell.x, dims[0]) else {
                continue;
            };
            let Some((y_lo, y_hi)) = axis_range(pos[j].y, r, bmin.y, cell.y, dims[1]) else {
                continue;
            };
            for iy in y_lo..=y_hi {
                let cy = bmin.y + (iy as f64 + 0.5) * cell.y;
                let row = iy as usize * dx;
                for ix in x_lo..=x_hi {
                    let cx = bmin.x + (ix as f64 + 0.5) * cell.x;
                    let d = DVec3::new(cx, cy, cz) - pos[j];
                    acc[row + ix as usize] += weight[j] * w(d.length(), h[j]);
                }
            }
        }
        acc
    };

    let planes: Vec<Vec<f64>> = if parallel {
        (0..dims[2]).into_par_iter().map(plane).collect()
    } else {
        (0..dims[2]).map(plane).collect()
    };
    let mut data = Vec::with_capacity(n_cells);
    data.extend(planes.iter().flatten().map(|&v| v as f32));
    grid.data = data;
    grid
}

/// Voxelize the gas population of `state`: select `Species::Gas` rows, derive
/// adaptive smoothing lengths via the shared `solvers::sph` routine, choose
/// cubic bounds (percentile radius about the gas centroid + `2·h_med` pad, see
/// [`GasGridConfig`]), and kernel-deposit onto `cfg.dims` cells.
///
/// Returns `None` when the state holds no gas — a gas-free run has no grid and
/// its frame-data v2 carries no gas block.
pub fn deposit_gas(state: &State, cfg: &GasGridConfig) -> Option<GasGrid> {
    let mut pos = Vec::new();
    let mut mass = Vec::new();
    for i in 0..state.len() {
        if state.kind[i] == Species::Gas {
            pos.push(state.pos[i]);
            mass.push(state.mass[i]);
        }
    }
    if pos.is_empty() {
        return None;
    }

    // h is a pure function of the gas positions (plan D2) — the same shared
    // routine the force path uses, so render and physics smooth identically.
    let h = density_adaptive(&pos, &mass, &cfg.density, None).h;

    // Robust pad scale: the MEDIAN smoothing length, not the max. Adaptive h in
    // the sparse outskirts (and, mid-merger, in tidal debris) runs orders of
    // magnitude above the bulk, so a SUPPORT·h_max pad let one isolated particle
    // inflate the cube several-fold and dilute the gas below one cell per scale
    // length (the M7c demo finding). The median tracks the bulk that carries the
    // visible signal; particles whose support pokes past the box lose only their
    // own outer kernel — a `+0.0` clip the M7d mass gate already sanctions for
    // the dropped far tail.
    let mut hs = h.clone();
    hs.sort_by(|a, b| a.total_cmp(b));
    let h_med = hs[hs.len() / 2];

    // Cubic bounds: percentile radius about the (unweighted) gas centroid,
    // padded by the full kernel support of the median particle. f32 rounding of
    // the corners is harmless — the pad dwarfs the f32 ulp at scene scale.
    let centroid = pos.iter().fold(DVec3::ZERO, |a, &p| a + p) / pos.len() as f64;
    let mut d: Vec<f64> = pos.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    let idx = ((d.len() - 1) as f64 * cfg.percentile.clamp(0.0, 1.0)).round() as usize;
    let half = d[idx] + SUPPORT * h_med;
    let bounds_min = (centroid - DVec3::splat(half)).as_vec3();
    let bounds_max = (centroid + DVec3::splat(half)).as_vec3();

    Some(deposit_fixed(
        &pos, &mass, &h, cfg.dims, bounds_min, bounds_max,
    ))
}

/// Voxelize BOTH the gas density ρ and the internal-energy moment
/// `N = Σ (m_j·u_j)·W` with ONE shared adaptive-h solve and identical geometry,
/// returning `(rho, moment)`. The two grids are co-registered (same `dims`,
/// `bounds`, and per-particle `h`), so `moment.sample(p) / rho.sample(p)` is the
/// SPH mass-weighted specific internal energy `ū` at `p` — the temperature field
/// the raymarcher colors by. `rho` is bit-identical to [`deposit_gas`]'s grid
/// (same inputs, same deposition). `None` when the state holds no gas.
pub fn deposit_gas_with_temperature(
    state: &State,
    cfg: &GasGridConfig,
) -> Option<(GasGrid, GasGrid)> {
    let _ = (state, cfg);
    todo!("H1: co-registered (rho, u-moment) deposition sharing one h-solve")
}
