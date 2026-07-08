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

use galaxy_core::{DVec3, ForceSolver, Species, State};

use super::density::{density_adaptive, DensityConfig};
use super::forces::{hydro_accelerations, HydroParams};

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
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");

        // Gravity over ALL particles (gas is just mass to the gravity solver);
        // zero the buffer ourselves for the gravity-off path.
        match &mut self.gravity {
            Some(g) => g.accelerations(state, acc),
            None => acc.iter_mut().for_each(|a| *a = DVec3::ZERO),
        }

        // Hydro over the gas subset only: recompute ρ/h internally (once per KDK
        // step, at post-drift positions), warm-starting h across steps.
        let gas: Vec<usize> = (0..n).filter(|&i| state.kind[i] == Species::Gas).collect();
        if gas.is_empty() {
            self.h_hint = None;
            return;
        }
        let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
        let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
        let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
        let gu: Vec<f64> = gas.iter().map(|&i| state.u[i]).collect();

        // Warm-start hint is a bracket seed only (the converged h is
        // position-determined); drop it if the gas count changed.
        let hint = self
            .h_hint
            .as_ref()
            .filter(|hh| hh.len() == gas.len())
            .map(Vec::as_slice);
        let dens = density_adaptive(&gpos, &gmass, &self.density_cfg, hint);
        let a_hydro =
            hydro_accelerations(&gpos, &gvel, &gmass, &dens.rho, &dens.h, &gu, &self.params);
        for (k, &i) in gas.iter().enumerate() {
            acc[i] += a_hydro[k];
        }
        self.h_hint = Some(dens.h);
    }

    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        let _ = (state, acc, dudt);
        todo!("E2a: GravitySph fused accel_and_dudt (gravity acc, fused hydro over gas subset, dudt=0 for non-gas)")
    }

    fn potential_energy(&self, state: &State) -> f64 {
        match &self.gravity {
            Some(g) => g.potential_energy(state),
            None => 0.0,
        }
    }

    fn max_stable_dt(&self, state: &State) -> f64 {
        // The CFL limit at Courant number 1 (the raw `min_i h_i / v_sig,i`
        // timescale); the adaptive loop applies its own Courant × safety below it.
        // Reuses the gas-subset CFL reduction verbatim (`c_cfl = 1.0`), so a
        // gas-free state returns `+∞` exactly as the free function does.
        super::cfl::max_stable_dt(state, &self.params, &self.density_cfg, 1.0)
    }
}
