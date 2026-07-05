//! [`GpuHydro`]: GPU isothermal SPH hydro force (GPU-SPH G3).
//!
//! RED stub — API surface only (`todo!()` bodies) so the workspace builds while the
//! G3 gates fail. The green implementation follows [`crate::GpuDensity`]'s shape:
//! reuse [`crate::sph_grid::GRID_HELPERS_WGSL`] for the counting-sort hash, gather per
//! target over the GLOBAL `SUPPORT·h_max` radius (never per-target — that breaks
//! Newton's third law), and sum the symmetric `P/ρ²` pressure term + Monaghan
//! viscosity against the exactly-negated grad-average, matching
//! [`galaxy_solvers::sph::hydro_accelerations`].

use galaxy_core::DVec3;
use galaxy_solvers::sph::HydroParams;

use crate::GpuError;

/// GPU isothermal hydro force. Reusable wgpu compute context built once
/// ([`new`](Self::new)) and driven per call; storage grows lazily with N — the same
/// bring-up idiom as [`crate::GpuDensity`].
pub struct GpuHydro {
    _priv: (),
}

impl GpuHydro {
    /// Bring up a headless wgpu compute device and the build + hydro pipelines.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        todo!("G3 green: bring up the wgpu device + build/hydro pipelines")
    }

    /// Per-gas-particle acceleration `a_i = −Σ_j m_j (P_i/ρ_i² + P_j/ρ_j² + Π_ij) ∇_i W̄_ij`
    /// (isothermal EOS `P = c_s²ρ`, kernel-average `W̄`, Monaghan `Π`), gathering per
    /// target over the global `SUPPORT·h_max`. `rho`/`h` are supplied (the density pass
    /// ran first); every slice has length `pos.len()`.
    pub fn accelerations(
        &mut self,
        _pos: &[DVec3],
        _vel: &[DVec3],
        _mass: &[f64],
        _rho: &[f64],
        _h: &[f64],
        _params: &HydroParams,
    ) -> Vec<DVec3> {
        todo!("G3 green: gather-per-target hydro force on the GPU")
    }
}
