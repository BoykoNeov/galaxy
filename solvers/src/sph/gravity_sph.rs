//! `GravitySph<G>`: the composite gravity + isothermal-SPH `ForceSolver` (D4).
//!
//! Gravity acts on ALL particles (gas is just mass to the wrapped gravity
//! solver `G`, sharing its Plummer softening — softening and smoothing are
//! deliberately decoupled in v1). Hydrodynamic acceleration is added to the gas
//! rows only; density and smoothing lengths are recomputed over the gas subset
//! internally each call — exactly once per KDK step, at post-drift positions,
//! which is what `&mut self` on the trait buys. Viscosity uses the velocities
//! present at the call (v_{n+1/2}); the pairwise force stays antisymmetric, so
//! momentum gates are blind to that first-order viscous-term timing.
//!
//! `gravity: None` is the pure-hydro mode (shock tube, gas-ball tests): the
//! IDENTICAL hydro path runs, only the gravity add is skipped.

use galaxy_core::{DVec3, ForceSolver, State};

use super::density::DensityConfig;
use super::forces::HydroParams;

/// Gravity + SPH composite solver. `G` is the wrapped gravity solver (any
/// `ForceSolver`); `None` disables gravity for pure-hydro runs.
pub struct GravitySph<G: ForceSolver> {
    /// Wrapped gravity solver, or `None` for gravity-off.
    pub gravity: Option<G>,
    /// Isothermal SPH force parameters.
    pub params: HydroParams,
    /// Adaptive-h density configuration.
    pub density_cfg: DensityConfig,
    /// Warm-start smoothing lengths for the gas subset (bracket hint only; the
    /// converged h is position-determined). `None` on the first call, or when
    /// the gas count changes.
    h_hint: Option<Vec<f64>>,
}

impl<G: ForceSolver> GravitySph<G> {
    /// Gravity + SPH.
    pub fn new(gravity: G, params: HydroParams, density_cfg: DensityConfig) -> Self {
        GravitySph {
            gravity: Some(gravity),
            params,
            density_cfg,
            h_hint: None,
        }
    }

    /// Pure hydro (gravity off): same hydro path, gravity add skipped.
    pub fn hydro_only(params: HydroParams, density_cfg: DensityConfig) -> Self {
        GravitySph {
            gravity: None,
            params,
            density_cfg,
            h_hint: None,
        }
    }
}

impl<G: ForceSolver> ForceSolver for GravitySph<G> {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let _ = (state, acc);
        todo!("M7b: gravity over all + hydro over gas subset (adaptive ρ/h, warm-start h)")
    }

    fn potential_energy(&self, state: &State) -> f64 {
        match &self.gravity {
            Some(g) => g.potential_energy(state),
            None => 0.0,
        }
    }
}
