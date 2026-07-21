use crate::{DVec3, State};

/// The SPH fields the star-formation recipe reads (plan `natal-ember-forge.md`,
/// F5): per-particle gas density `ρ` and velocity divergence `∇·v`. Returned
/// transiently by [`ForceSolver::sf_fields`] — computed fresh, never STORED on
/// `State` (D2: the trait takes `&State`, so a stored ρ column could not be kept
/// fresh). Both are length `n` (parallel to the SoA columns); collisionless rows
/// (and every row under a pure-gravity solver) carry `ρ = 0`, `∇·v = 0`, which
/// can never pass the recipe's `ρ >= rho_thresh` gate, so the zero-default is
/// inert for SF.
#[derive(Clone, Debug, PartialEq)]
pub struct SfFields {
    /// Per-particle SPH gas density `ρ_i` (0 on collisionless rows).
    pub rho: Vec<f64>,
    /// Per-particle SPH velocity divergence `∇·v_i` (0 on collisionless rows).
    /// Negative ⇒ locally converging (collapsing) flow — the "converging" half
    /// of the two-part SF criterion.
    pub div_v: Vec<f64>,
}

/// Computes gravitational accelerations. Softening is a property of the
/// concrete solver. Implementations are swappable (direct-sum → Barnes-Hut →
/// PM/TreePM) without touching the integrator or callers.
pub trait ForceSolver {
    /// Fill `acc[i]` with the acceleration on particle `i`.
    /// Requires `acc.len() == state.len()`.
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]);

    /// Fused acceleration + thermal-derivative pass (E2a). Fills `acc` exactly
    /// as `accelerations` would AND fills `dudt[i]` with `du_i/dt` (zero for
    /// non-thermal particles/solvers). The default delegates to
    /// `accelerations` and zero-fills `dudt`, so every existing solver (pure
    /// gravity, GPU) gets `du/dt≡0` for free without touching its impl;
    /// `GravitySph` overrides this with a single fused SPH neighbor pass
    /// (accel + PdV work share the same loop).
    /// Requires `acc.len() == dudt.len() == state.len()`.
    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        self.accelerations(state, acc);
        dudt.fill(0.0);
    }

    /// Active-subset acceleration pass (I7 — the individual-timestep efficiency
    /// path). Fills `acc[i]` for the `active` targets (global indices) with the
    /// same acceleration `accelerations` would produce; `acc` entries outside
    /// `active` are left unspecified (the individual stepper only reads the ones
    /// it is about to kick). A solver MAY reduce its per-call cost to the active
    /// subset (SPH gathers only the active gas, reading persistent ρ/h for the
    /// inactive neighbours it drifted). The default computes the FULL pass and
    /// ignores `active`, so every existing solver is correct-but-unaccelerated for
    /// free; `GravitySph` overrides it to gather hydro on the active gas only.
    /// Requires `acc.len() == state.len()`.
    fn accelerations_active(&mut self, state: &State, _active: &[usize], acc: &mut [DVec3]) {
        self.accelerations(state, acc);
    }

    /// Fused active-subset acceleration + `du/dt` pass (I7, thermal arm): the
    /// [`accel_and_dudt`](Self::accel_and_dudt) analogue of
    /// [`accelerations_active`](Self::accelerations_active). Fills `acc[i]`/`dudt[i]`
    /// for the `active` targets; entries outside `active` are unspecified. The
    /// default computes the full fused pass and ignores `active`; `GravitySph`
    /// overrides it to gather on the active gas only.
    /// Requires `acc.len() == dudt.len() == state.len()`.
    fn accel_and_dudt_active(
        &mut self,
        state: &State,
        _active: &[usize],
        acc: &mut [DVec3],
        dudt: &mut [f64],
    ) {
        self.accel_and_dudt(state, acc, dudt);
    }

    /// Rebuild the cached spatial structure for the stale-tree active gravity walk
    /// (I-grav, `hydro+gravity` mode). The individual stepper calls this ONCE at each
    /// BASE-BLOCK start; [`gravity_active_cached`](Self::gravity_active_cached) then
    /// walks the cached structure on every fine tick. The default is a no-op — a
    /// solver with no cacheable structure ignores it and walks all-N fresh each tick.
    fn rebuild_gravity_cache(&mut self, _state: &State) {}

    /// Active-subset gravity walk against the structure cached by the last
    /// [`rebuild_gravity_cache`](Self::rebuild_gravity_cache), evaluated at the
    /// CURRENT positions in `state` (I-grav). Fills `acc[i]` for `i` in `active`;
    /// entries outside `active` are unspecified. Because the individual stepper
    /// drifts ALL particles every fine tick, the near-field (opened-leaf) sources
    /// read from `state.pos` are current — only the far-field cell multipoles are
    /// stale (a bounded, converging approximation). The default computes the FULL
    /// fresh pass and ignores `active` (correct but unreduced); `TreeGravity`
    /// overrides it to walk the cached `FlatTree`.
    fn gravity_active_cached(&mut self, state: &State, _active: &[usize], acc: &mut [DVec3]) {
        self.accelerations(state, acc);
    }

    /// Total gravitational potential energy, using the SAME softened kernel as
    /// `accelerations` so energy diagnostics stay consistent with the forces.
    fn potential_energy(&self, state: &State) -> f64;

    /// The CFL limit the solver's physics imposes at this state — the largest dt
    /// stable at Courant number 1 (for SPH, `min_i h_i / v_sig,i`). `+∞` when the
    /// solver imposes no timestep constraint (pure gravity has none in v1, so the
    /// default is `+∞`).
    ///
    /// This reports only the *physics* limit. The adaptive-dt loop applies its own
    /// Courant number and safety factor strictly BELOW this — timestep POLICY lives
    /// in the loop, never in the solver (mirroring how the pipeline `C_CFL` guard is
    /// a policy constant, not a solver property).
    fn max_stable_dt(&self, _state: &State) -> f64 {
        f64::INFINITY
    }

    /// The per-particle CFL limit (I1) — the state-indexed vector whose `min` is
    /// [`max_stable_dt`](Self::max_stable_dt). Individual timesteps bin these into
    /// power-of-two rungs (a particle's rung IS its `dt_i`). Gas rows carry the
    /// finite hydro bound; collisionless rows (and any solver with no hydro
    /// constraint) carry `+∞`. The default is `vec![+∞; len]`, consistent with the
    /// scalar default — a pure-gravity solver imposes no per-particle limit.
    ///
    /// Like the scalar, this reports only the *physics* limit; the Courant number
    /// and rung policy live in the individual-timestep loop, never in the solver.
    fn max_stable_dt_per_particle(&self, state: &State) -> Vec<f64> {
        vec![f64::INFINITY; state.len()]
    }

    /// Gas neighbour pairs coupled by the force law (I4b): unordered global-index
    /// pairs `(i, j)` with `i < j` and `r_ij < SUPPORT·max(h_i, h_j)` — the SAME
    /// coupling range the SPH force (`forces.rs`) and CFL (`cfl.rs`) paths gather
    /// over, so the individual-timestep limiter constrains exactly the particles the
    /// force actually couples. The Saitoh–Makino limiter consumes these to keep no
    /// gas particle more than `n_limit` rungs coarser than a coupled neighbour (rung
    /// POLICY lives in the loop; this reports only the *physics* adjacency).
    ///
    /// Default empty: a solver with no hydro coupling (pure gravity) constrains no
    /// rungs, so the limiter is a no-op there.
    fn coupled_pairs(&self, _state: &State) -> Vec<(usize, usize)> {
        Vec::new()
    }

    /// The SPH fields the star-formation recipe consumes (F5): per-particle gas
    /// density `ρ` and velocity divergence `∇·v`, both length `state.len()`.
    /// A transient accessor (fresh `Vec`s, nothing stored — D2-clean, exactly
    /// like [`max_stable_dt_per_particle`](Self::max_stable_dt_per_particle) and
    /// [`coupled_pairs`](Self::coupled_pairs)); the SF driver calls it once per
    /// snapshot interval at the output-cadence sync site (S4).
    ///
    /// Default returns all-zeros: a pure-gravity solver has no gas, so it forms
    /// no stars (a `ρ = 0` row can never clear `ρ >= rho_thresh`). `GravitySph`
    /// overrides it, reusing the adaptive-h density solve for `ρ` and the SPH
    /// divergence gather for `∇·v`; collisionless rows stay `(0, 0)`.
    fn sf_fields(&self, state: &State) -> SfFields {
        let n = state.len();
        SfFields {
            rho: vec![0.0; n],
            div_v: vec![0.0; n],
        }
    }
}

/// Cosmological background. `StaticBackground` => a≡1, H≡0 (Newtonian).
/// A Friedmann background (later) supplies a(t) and the Hubble drag term, which
/// is where comoving integration attaches — the integrator interface is ready.
pub trait Background {
    /// Scale factor a(t).
    fn scale_factor(&self, t: f64) -> f64;
    /// Hubble parameter H = ȧ/a.
    fn hubble(&self, t: f64) -> f64;
}

/// Advances the state by one timestep `dt`.
pub trait Integrator {
    fn step(
        &mut self,
        state: &mut State,
        solver: &mut dyn ForceSolver,
        bg: &dyn Background,
        dt: f64,
    );
}
