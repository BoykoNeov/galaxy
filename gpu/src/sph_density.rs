//! [`GpuDensity`]: GPU adaptive-h SPH density (GPU-SPH G2).
//!
//! Per-gas-particle smoothing length `h` and density `ρ`, computed on the GPU by
//! the same bisection root-find as the CPU oracle [`galaxy_solvers::sph::density_adaptive`]:
//! `h` solves `N_i(h) = n_ngb` where the kernel-weighted count
//! `N_i(h) = (4π/3)(SUPPORT·h)³ · Σ_j W(|x_i − x_j|, h)` is monotone in `h`, then
//! `ρ_i = Σ_j m_j W(|x_i − x_j|, h_i)`. It reuses G1's spatial-hash grid
//! ([`crate::GpuNeighborGrid`] shares the cell/hash math) and walks the buckets
//! **GPU-side** per target — no host CSR round-trip.
//!
//! This is the RED stub: API surface only (`todo!()` bodies) so the gates compile
//! and fail. The green kernel lands next.

use galaxy_core::DVec3;

use crate::GpuError;

/// Per-particle SPH density and the smoothing length it was computed at. `f32`
/// because the device computes in f32 (the CPU oracle is f64 and the gate is an
/// f32-tolerance comparison — D1/D5, never bit-exact).
pub struct DensityField {
    /// Density `ρ_i` per particle.
    pub rho: Vec<f32>,
    /// Adaptive smoothing length `h_i` per particle.
    pub h: Vec<f32>,
}

/// GPU adaptive-h density. Reusable wgpu compute context (green); mirrors the
/// [`crate::GpuNeighborGrid`] bring-up + lazy-capacity idiom.
pub struct GpuDensity {
    // (green) wgpu device/queue, the build + density pipelines, and lazily-grown
    // storage buffers. RED carries no state — `new` is `todo!()`.
    _stub: (),
}

impl GpuDensity {
    /// Bring up a headless wgpu compute device and the density pipelines. Returns a
    /// typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        todo!("G2 green: bring up wgpu context + build/density pipelines")
    }

    /// Adaptive-h density: per particle, bisect `h` to hit `n_ngb`, then sum `ρ`.
    /// Returns `(ρ, h)`. Gated f32-tolerance against
    /// [`galaxy_solvers::sph::density_adaptive`] on a wide-h rooted cloud.
    pub fn densities(
        &mut self,
        _pos: &[DVec3],
        _mass: &[f64],
        _n_ngb: f64,
        _h_tol_rel: f64,
    ) -> DensityField {
        todo!("G2 green: GPU-side per-target bisection + density sum")
    }

    /// Fixed-`h` density summation (the decoupled summation gate): `ρ_i = Σ_j m_j
    /// W(r_ij, h_i)` at the CALLER's `h`, no root-find. Gated f32-tolerance against
    /// [`galaxy_solvers::sph::density_fixed`].
    pub fn densities_at(&mut self, _pos: &[DVec3], _mass: &[f64], _h: &[f64]) -> Vec<f32> {
        todo!("G2 green: GPU-side per-target density sum at given h")
    }
}
