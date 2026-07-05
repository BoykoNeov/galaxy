//! [`GpuCfl`]: GPU isothermal SPH CFL reduction (GPU-SPH G4).
//!
//! Per-gas-particle stable step `dt_i = C_cfl · h_i / v_sig,i`, with the Gadget-style
//! projected signal velocity `v_sig,i = max_j (2c_s − 3 w_ij)` over APPROACHING
//! neighbors (`w_ij = (v_i−v_j)·r̂_ij < 0`), floored at `2c_s`. The milestone bound is
//! `min_i dt_i` — D6's per-batch adaptive-dt substrate. Gated against the CPU oracle
//! [`galaxy_solvers::sph::max_stable_dt`].
//!
//! STUB (red): API surface only; the compute path is [`todo!`] until the green pass.

use galaxy_solvers::sph::HydroParams;

use galaxy_core::DVec3;

use crate::GpuError;

/// GPU isothermal CFL reduction. Reusable wgpu compute context (mirrors [`crate::GpuHydro`]).
pub struct GpuCfl {
    _priv: (),
}

impl GpuCfl {
    /// Bring up a headless wgpu compute device and the build + CFL pipelines.
    pub fn new() -> Result<Self, GpuError> {
        todo!("G4 green: bring up the CFL wgpu context")
    }

    /// Per-target stable step `dt_i = C_cfl · h_i / v_sig,i`, gathering per target over
    /// the global `SUPPORT·h_max`. `h` is supplied (the density pass ran first); every
    /// slice must have length `pos.len()`.
    pub fn per_target_dt(
        &mut self,
        _pos: &[DVec3],
        _vel: &[DVec3],
        _h: &[f64],
        _params: &HydroParams,
        _c_cfl: f64,
    ) -> Vec<f64> {
        todo!("G4 green: GPU per-target CFL dt")
    }

    /// The stable-dt bound `min_i dt_i`, or `f64::INFINITY` for no gas (empty input).
    pub fn max_stable_dt(
        &mut self,
        _pos: &[DVec3],
        _vel: &[DVec3],
        _h: &[f64],
        _params: &HydroParams,
        _c_cfl: f64,
    ) -> f64 {
        todo!("G4 green: GPU CFL reduction")
    }
}
