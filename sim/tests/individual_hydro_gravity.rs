//! I-grav (milestone 12) — `run_individual` in `hydro+gravity` mode, the driver-level
//! correctness gates for gravity subcycling.
//!
//! `hydro-only` gives collisionless stars hydro `dt = +∞` (rung 0, gravity walked
//! all-N). `hydro+gravity` folds the gravitational criterion `η·√(ε/|a|)` into the
//! per-particle `dt` (so stars get FINITE rungs) and subcycles gravity on a cached
//! stale tree the driver rebuilds once per base block.
//!
//! GATE DESIGN (advisor-vetted 2026-07-10):
//!   * NO collapsed-rung-0 ≡ LeapfrogKdk bit-identity exists here — even all-rung-0,
//!     the one fine tick drifts to p1 then walks the BLOCK-START tree (stale by a full
//!     base step), which is not fresh all-N gravity. So run-level correctness is
//!     CONVERGENCE only (the solver-level fresh-cache anchor lives in tree_gravity.rs).
//!   * SUBCYCLE-ENGAGED (non-vacuous): on a dense gas core + stars testbed, subcycling
//!     ON gives stars finite rungs FINER than the gas, so `max_rung` strictly exceeds
//!     the `hydro-only` value — proof the stars actually subcycled (else both gates
//!     could pass while testing nothing).
//!   * CONVERGENCE: coarse run → fine as courant ↓ (halving courant halves every
//!     hydro AND gravity safe step ⇒ courant-invariant rungs, same discipline as I4a).
//!   * MOMENTUM (fork b): kick-active-only PLUS the stale tree breaking pairwise
//!     antisymmetry both add drift ∝ courant; the honest gate is drift SHRINKS with
//!     courant, not a widened bound.

use galaxy_core::{diagnostics, DVec3, Species, State, StaticBackground};
use galaxy_io::Header;
use galaxy_sim::{
    run_individual, IndividualConfig, IndividualSummary, SimError, SnapshotSink, ThermalArm,
};
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::{BarnesHut, TreeGravity};

struct NullSink;
impl SnapshotSink for NullSink {
    fn emit(&mut self, _h: &Header, _s: &State) -> Result<(), SimError> {
        Ok(())
    }
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn ball(rng: &mut impl FnMut() -> f64, n: usize, radius: f64) -> Vec<DVec3> {
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    pos
}

/// A DENSE gas core (strong gravity well + hydro pressure) surrounded by STARS at a
/// range of radii. Stars near the core feel a large |a| ⇒ a fine gravitational rung;
/// far stars feel a small |a| ⇒ a coarse rung — a genuine star grav-rung SPREAD that
/// only `hydro+gravity` resolves (under `hydro-only` every star is rung 0). Gas has
/// zero initial velocity (pressure/gravity drive the CFL-moving dynamics); stars get a
/// tangential orbital velocity so the system evolves.
fn core_and_stars(seed: u64) -> State {
    let mut rng = lcg(seed);
    let gas = ball(&mut rng, 400, 0.1); // dense gas core
    let n_gas = gas.len();
    let mut pos = gas;
    let mut vel = vec![DVec3::ZERO; n_gas];
    // Stars sprinkled over radii 0.12..1.5 (some hugging the core ⇒ fine grav rungs).
    let n_star = 300;
    for _ in 0..n_star {
        let r = 0.12 + rng() * 1.38;
        let dir = {
            let v = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5);
            v / v.length().max(1e-9)
        };
        let p = dir * r;
        pos.push(p);
        // A mild circular-ish velocity in the x-y plane.
        vel.push(DVec3::new(-p.y, p.x, 0.0) * 0.3);
    }
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0 / n as f64; n]);
    for (i, kind) in s.kind.iter_mut().enumerate() {
        *kind = if i < n_gas {
            Species::Gas
        } else {
            Species::Collisionless
        };
    }
    s
}

fn solver(subcycle: bool) -> GravitySph<TreeGravity> {
    let params = HydroParams {
        eos: Eos::Isothermal { c_s: 0.3 },
        ..HydroParams::default()
    };
    let bh = BarnesHut::new(1.0, 0.05, 0.5);
    GravitySph::new(TreeGravity::new(bh), params, DensityConfig::default())
        .with_gravity_cache(subcycle)
}

fn cfg(courant: f64, subcycle: bool, output_dt: f64, n_outputs: u64) -> IndividualConfig {
    IndividualConfig {
        courant,
        dt_base_cap: f64::INFINITY, // non-binding ⇒ rung structure is courant-invariant
        r_max: 14,
        n_limit: 14, // == r_max ⇒ limiter non-binding (pure CFL/grav rungs)
        cache_gravity_tree: subcycle, // subcycling requires the cached tree
        subcycle_gravity: subcycle,
        grav_eta: 1.0, // gravity shares the hydro courant (courant·√(ε/|a|))
        eos: ThermalArm::Isothermal,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0x91A7,
        config_hash: 0,
        units: "nbody-G1".to_string(),
        sf: None,
    }
}

fn run(subcycle: bool, courant: f64, output_dt: f64, n_outputs: u64) -> (State, IndividualSummary) {
    let mut state = core_and_stars(7);
    let mut solv = solver(subcycle);
    let bg = StaticBackground;
    let summary = run_individual(
        &mut state,
        &mut solv,
        &bg,
        &cfg(courant, subcycle, output_dt, n_outputs),
        &mut NullSink,
    )
    .expect("hydro+gravity run must complete");
    (state, summary)
}

// --------------------------------------------------------------------------
// SUBCYCLE-ENGAGED — the non-vacuous check: stars actually get finite (finer) rungs.
// --------------------------------------------------------------------------

#[test]
fn subcycling_gives_stars_finer_rungs_than_hydro_only() {
    let (_s_off, off) = run(false, 0.2, 0.25, 1);
    let (_s_on, on) = run(true, 0.2, 0.25, 1);

    // Both must stay UNDER r_max (reference not clamped/under-resolved).
    assert!(on.max_rung < 14, "hydro+gravity max_rung clamped at r_max");
    // Subcycling must reach a STRICTLY finer rung — the stars hugging the dense core
    // feel a large |a| and land below the finest gas rung, which hydro-only (stars on
    // rung 0) cannot produce. This is the proof the gravity criterion engaged.
    assert!(
        on.max_rung > off.max_rung,
        "gravity subcycling must give stars finer rungs than hydro-only: \
         on.max_rung {} !> off.max_rung {}",
        on.max_rung,
        off.max_rung
    );
    assert!(
        on.distinct_rungs >= 3,
        "the testbed must span a real rung spread, got {}",
        on.distinct_rungs
    );
}

// --------------------------------------------------------------------------
// CONVERGENCE — coarse → fine as courant halves (the only run-level correctness gate).
// --------------------------------------------------------------------------

#[test]
fn hydro_gravity_converges_as_courant_halves() {
    // Enough output span to accumulate a measurable trajectory difference.
    let (output_dt, n_outputs) = (0.3, 2);
    let reference = run(true, 0.05, output_dt, n_outputs).0;
    let err = |s: &State| -> f64 {
        s.pos
            .iter()
            .zip(&reference.pos)
            .map(|(p, r)| (*p - *r).length())
            .fold(0.0_f64, f64::max)
    };
    let coarse = run(true, 0.2, output_dt, n_outputs).0;
    let fine = run(true, 0.1, output_dt, n_outputs).0;
    let (e_coarse, e_fine) = (err(&coarse), err(&fine));
    assert!(
        e_fine < e_coarse,
        "halving courant must reduce the error toward the fine reference: \
         err(0.1) = {e_fine:.3e} !< err(0.2) = {e_coarse:.3e}"
    );
}

// --------------------------------------------------------------------------
// MOMENTUM — bounded drift that SHRINKS with courant (fork b: two error sources,
// kick-active-only + stale-tree antisymmetry break, both ∝ courant).
// --------------------------------------------------------------------------

#[test]
fn hydro_gravity_momentum_drift_shrinks_with_courant() {
    let (output_dt, n_outputs) = (0.3, 2);
    let p0 = diagnostics::total_momentum(&core_and_stars(7));
    let drift = |courant: f64| -> f64 {
        let (s, _) = run(true, courant, output_dt, n_outputs);
        (diagnostics::total_momentum(&s) - p0).length()
    };
    let d_coarse = drift(0.2);
    let d_fine = drift(0.05);
    assert!(
        d_fine < d_coarse,
        "momentum drift must shrink as courant → 0 (kick-active-only + stale-tree \
         antisymmetry, both ∝ courant): drift(0.05) = {d_fine:.3e} !< drift(0.2) = {d_coarse:.3e}"
    );
}
