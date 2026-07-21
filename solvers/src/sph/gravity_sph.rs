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

use galaxy_core::{DVec3, ForceSolver, SfFields, Species, State};

use super::density::{density_adaptive, density_adaptive_active, DensityConfig};
use super::divergence::velocity_divergence;
use super::forces::{
    hydro_accel_and_dudt, hydro_accel_and_dudt_active, hydro_accelerations, HydroParams,
};

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
    /// the gas count changes. On the I7 active path this doubles as the persistent
    /// `h` scratch: active targets overwrite their entry each fine tick, inactive
    /// ones keep their last-active value (read as stale neighbour `h`).
    h_hint: Option<Vec<f64>>,
    /// Persistent gas-subset density scratch for the I7 active path (`accelerations_active`
    /// / `accel_and_dudt_active`): active targets refresh their entry each fine tick,
    /// inactive neighbours are read stale. Paired with `h_hint` as the (ρ, h) scratch.
    /// `None` until the first active call (or after a gas-count change) initializes it
    /// with a full over-all-gas refresh. Untouched by the full `accelerations` path.
    rho_scratch: Option<Vec<f64>>,
    /// When `true`, the active path walks gravity on the STALE cached tree
    /// (`gravity_active_cached`, active subset) instead of the all-N fresh
    /// `accelerations`. The driver MUST rebuild the cache once per base block
    /// (`rebuild_gravity_cache`) for this to be correct; requires `G` to actually cache
    /// (e.g. `TreeGravity`) — with a non-caching `G` the trait defaults fall back to a
    /// full fresh walk (correct, unreduced). This is a property of the WALK ONLY, shared
    /// by BOTH cached modes: `hydro+gravity` (I-grav, stars also get finite gravitational
    /// rungs — the driver's `subcycle_gravity`) AND `hydro-only` with tree caching (stars
    /// stay on rung 0, gravity walked stale once/block). `false` = the fresh path
    /// (gravity all-N every fine tick, byte-identical to a bare `BarnesHut`).
    cached_gravity_walk: bool,
}

impl<G: ForceSolver> GravitySph<G> {
    /// Gravity + SPH.
    pub fn new(gravity: G, params: HydroParams, density_cfg: DensityConfig) -> Self {
        GravitySph {
            gravity: Some(gravity),
            params,
            density_cfg,
            h_hint: None,
            rho_scratch: None,
            cached_gravity_walk: false,
        }
    }

    /// Enable the stale cached-tree gravity walk: the active path walks gravity on the
    /// tree the driver rebuilds once per base block instead of all-N fresh every fine
    /// tick. Shared by both cached modes (`hydro+gravity` subcycling AND `hydro-only`
    /// with tree caching); the driver decides rung-folding independently. Requires the
    /// driver to rebuild the cache each base block and `G` to cache (`TreeGravity`).
    /// Builder-style.
    pub fn with_gravity_cache(mut self, on: bool) -> Self {
        self.cached_gravity_walk = on;
        self
    }

    /// Pure hydro (gravity off): same hydro path, gravity add skipped.
    pub fn hydro_only(params: HydroParams, density_cfg: DensityConfig) -> Self {
        GravitySph {
            gravity: None,
            params,
            density_cfg,
            h_hint: None,
            rho_scratch: None,
            cached_gravity_walk: false,
        }
    }

    /// Shared setup for the I7 active paths ([`accelerations_active`](ForceSolver::accelerations_active)
    /// / [`accel_and_dudt_active`](ForceSolver::accel_and_dudt_active)): extract the
    /// gas subset and its position/velocity/mass/`u` columns (positions are exact —
    /// the stepper drifts every particle at the fine cadence), size the persistent
    /// `(ρ, h)` scratch, and map the global `active` indices to gas-local ones.
    /// Returns `None` (clearing the scratch) when the state has no gas.
    ///
    /// Warm-start discipline — the load-bearing bit for the I3 collapsed
    /// bit-identity gate: the density solve warm-starts each target's bracket from
    /// `h_hint`. An already-sized `h_hint` (left by a prior full `accelerations`
    /// prime, or a previous active tick) is KEPT — zeroing it would cold-start the
    /// bisection to a within-tolerance-but-different `h`, breaking bit-identity vs
    /// the full path. Only a missing/mis-sized `h_hint` is (re)allocated to zeros
    /// (⇒ the occupancy seed, matching the full path's `h_init = None` first call).
    /// Whenever EITHER scratch is (re)allocated the returned `active_local` is FORCED
    /// to all gas, so the full over-all refresh populates every ρ entry (valid for
    /// later stale-neighbour reads) — warm-started from the preserved `h_hint`.
    #[allow(clippy::type_complexity)]
    fn active_gas(
        &mut self,
        state: &State,
        active: &[usize],
    ) -> Option<(
        Vec<usize>,
        Vec<DVec3>,
        Vec<DVec3>,
        Vec<f64>,
        Vec<f64>,
        Vec<usize>,
    )> {
        let n = state.len();
        let gas: Vec<usize> = (0..n).filter(|&i| state.kind[i] == Species::Gas).collect();
        if gas.is_empty() {
            self.h_hint = None;
            self.rho_scratch = None;
            return None;
        }
        let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
        let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
        let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
        let gu: Vec<f64> = gas.iter().map(|&i| state.u[i]).collect();

        let h_needs_init = self.h_hint.as_ref().is_none_or(|hh| hh.len() != gas.len());
        let rho_needs_init = self
            .rho_scratch
            .as_ref()
            .is_none_or(|r| r.len() != gas.len());
        if h_needs_init {
            self.h_hint = Some(vec![0.0; gas.len()]);
        }
        if rho_needs_init {
            self.rho_scratch = Some(vec![0.0; gas.len()]);
        }

        let active_local: Vec<usize> = if h_needs_init || rho_needs_init {
            // A (re)allocated scratch must be fully populated before any stale read.
            (0..gas.len()).collect()
        } else {
            // Global → gas-local map; non-gas `active` entries carry no hydro rung
            // and are dropped (their gas-local index is absent).
            let mut g2l = vec![usize::MAX; n];
            for (loc, &g) in gas.iter().enumerate() {
                g2l[g] = loc;
            }
            active
                .iter()
                .filter_map(|&i| {
                    let loc = g2l[i];
                    (loc != usize::MAX).then_some(loc)
                })
                .collect()
        };
        Some((gas, gpos, gvel, gmass, gu, active_local))
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
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        assert_eq!(dudt.len(), n, "dudt length must match particle count");

        // Gravity over ALL particles, exactly as `accelerations` does.
        match &mut self.gravity {
            Some(g) => g.accelerations(state, acc),
            None => acc.iter_mut().for_each(|a| *a = DVec3::ZERO),
        }
        dudt.iter_mut().for_each(|d| *d = 0.0);

        // Fused hydro over the gas subset only: recompute ρ/h internally, same
        // warm-start discipline as `accelerations`.
        let gas: Vec<usize> = (0..n).filter(|&i| state.kind[i] == Species::Gas).collect();
        if gas.is_empty() {
            self.h_hint = None;
            return;
        }
        let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
        let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
        let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
        let gu: Vec<f64> = gas.iter().map(|&i| state.u[i]).collect();

        let hint = self
            .h_hint
            .as_ref()
            .filter(|hh| hh.len() == gas.len())
            .map(Vec::as_slice);
        let dens = density_adaptive(&gpos, &gmass, &self.density_cfg, hint);
        let (a_hydro, dudt_hydro) =
            hydro_accel_and_dudt(&gpos, &gvel, &gmass, &dens.rho, &dens.h, &gu, &self.params);
        for (k, &i) in gas.iter().enumerate() {
            acc[i] += a_hydro[k];
            dudt[i] = dudt_hydro[k];
        }
        self.h_hint = Some(dens.h);
    }

    fn accelerations_active(&mut self, state: &State, active: &[usize], acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        // Gravity fill. Fresh path: all-N fresh every fine tick (the unreduced non-rung
        // fraction). Cached path (`hydro-only` tree caching OR `hydro+gravity`): the
        // ACTIVE-subset walk on the STALE cached tree (built once per base block by the
        // driver) at current positions — writes only `active` (the stepper reads only
        // those).
        let cached = self.cached_gravity_walk;
        match &mut self.gravity {
            Some(g) if cached => g.gravity_active_cached(state, active, acc),
            Some(g) => g.accelerations(state, acc),
            None => acc.iter_mut().for_each(|a| *a = DVec3::ZERO),
        }
        let Some((gas, gpos, gvel, gmass, gu, active_local)) = self.active_gas(state, active)
        else {
            return;
        };
        // Own the (Clone) config / (Copy) params so the persistent scratch can be
        // borrowed mutably below without a self-field aliasing conflict.
        let cfg = self.density_cfg.clone();
        let params = self.params;
        let rho = self.rho_scratch.as_mut().unwrap();
        let h = self.h_hint.as_mut().unwrap();
        // PASS 1: refresh (ρ, h) for the active gas targets into the persistent
        // scratch (inactive entries keep their last-active value).
        density_adaptive_active(&gpos, &gmass, &cfg, &active_local, rho, h);
        // PASS 2: hydro accel on the active targets, reading neighbour ρ/h from the
        // scratch (inactive neighbours stale — the sole bounded I7 approximation;
        // positions are exact). Add to the gravity already in `acc`.
        let contribs =
            hydro_accel_and_dudt_active(&gpos, &gvel, &gmass, rho, h, &gu, &params, &active_local);
        for (k, &loc) in active_local.iter().enumerate() {
            acc[gas[loc]] += contribs[k].0;
        }
    }

    fn accel_and_dudt_active(
        &mut self,
        state: &State,
        active: &[usize],
        acc: &mut [DVec3],
        dudt: &mut [f64],
    ) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        assert_eq!(dudt.len(), n, "dudt length must match particle count");
        let cached = self.cached_gravity_walk;
        match &mut self.gravity {
            Some(g) if cached => g.gravity_active_cached(state, active, acc),
            Some(g) => g.accelerations(state, acc),
            None => acc.iter_mut().for_each(|a| *a = DVec3::ZERO),
        }
        // Zero-fill `dudt` exactly as `accel_and_dudt` does (non-gas rows stay 0);
        // active gas rows are overwritten with their PdV+heating rate below.
        dudt.iter_mut().for_each(|d| *d = 0.0);
        let Some((gas, gpos, gvel, gmass, gu, active_local)) = self.active_gas(state, active)
        else {
            return;
        };
        let cfg = self.density_cfg.clone();
        let params = self.params;
        let rho = self.rho_scratch.as_mut().unwrap();
        let h = self.h_hint.as_mut().unwrap();
        density_adaptive_active(&gpos, &gmass, &cfg, &active_local, rho, h);
        let contribs =
            hydro_accel_and_dudt_active(&gpos, &gvel, &gmass, rho, h, &gu, &params, &active_local);
        for (k, &loc) in active_local.iter().enumerate() {
            acc[gas[loc]] += contribs[k].0;
            dudt[gas[loc]] = contribs[k].1;
        }
    }

    fn rebuild_gravity_cache(&mut self, state: &State) {
        // Delegate to the wrapped gravity solver (I-grav: the driver calls this once
        // per base block; `TreeGravity` freezes its tree, other solvers no-op).
        if let Some(g) = &mut self.gravity {
            g.rebuild_gravity_cache(state);
        }
    }

    fn gravity_active_cached(&mut self, state: &State, active: &[usize], acc: &mut [DVec3]) {
        // The gravity-ONLY active walk on the cached tree (no hydro added). The driver
        // walks this over ALL particles to get |a_grav| for the combined rung dt, at
        // the SAME tree/θ/ε the fine-tick force uses (rung–force consistency by
        // construction). Delegates to the wrapped solver; gravity-off zeroes `active`.
        match &mut self.gravity {
            Some(g) => g.gravity_active_cached(state, active, acc),
            None => {
                for &i in active {
                    acc[i] = DVec3::ZERO;
                }
            }
        }
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

    fn max_stable_dt_per_particle(&self, state: &State) -> Vec<f64> {
        // The per-particle CFL vector (I1) at Courant number 1, state-indexed:
        // gas rows carry `h_i / v_sig,i`, collisionless rows `+∞`. Its `min`
        // equals `max_stable_dt` (the I1 bit-identity gate). Same reuse as the
        // scalar — the rung policy sits in the individual-timestep loop.
        super::cfl::max_stable_dt_per_particle(state, &self.params, &self.density_cfg, 1.0)
    }

    fn coupled_pairs(&self, state: &State) -> Vec<(usize, usize)> {
        // The gas pairs the force law couples (r < SUPPORT·max(h_i,h_j)), for the
        // I4b timestep limiter. Same density/grid machinery as the CFL vector.
        super::cfl::coupled_pairs(state, &self.density_cfg)
    }

    fn sf_fields(&self, state: &State) -> SfFields {
        // The SPH fields the star-formation recipe reads (F5): gas ρ (reuse the
        // adaptive-h density solve) and ∇·v (the divergence gather). Extract the
        // gas subset, solve over it, scatter back to global indices; non-gas rows
        // stay (0, 0). Transient — nothing stored (D2-clean).
        let n = state.len();
        let mut rho = vec![0.0; n];
        let mut div_v = vec![0.0; n];
        let gas: Vec<usize> = (0..n).filter(|&i| state.kind[i] == Species::Gas).collect();
        if gas.is_empty() {
            return SfFields { rho, div_v };
        }
        let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
        let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
        let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
        // COLD-START density (h_init: None) — NOT the warm-started h_hint. ρ must
        // be a pure function of positions so it is bit-reproducible by a fresh
        // `density_adaptive(..., None)` (gated bit-exact against reference_density);
        // a stale-h warm start would shift the converged h within bisection
        // tolerance and break that. sf_fields runs once per snapshot interval (cold
        // path), so the warm-start perf win is nil.
        let dens = density_adaptive(&gpos, &gmass, &self.density_cfg, None);
        let dv = velocity_divergence(&gpos, &gvel, &gmass, &dens.rho, &dens.h);
        for (k, &i) in gas.iter().enumerate() {
            rho[i] = dens.rho[k];
            div_v[i] = dv[k];
        }
        SfFields { rho, div_v }
    }
}
