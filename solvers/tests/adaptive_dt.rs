//! Adaptive-dt gates (plan: courant-quickening-cadence.md).
//!
//! A1: the `ForceSolver::max_stable_dt` trait method reports the solver's CFL
//! limit at Courant number 1 (the raw `min_i h_i / v_sig,i` timescale) so the
//! adaptive loop can pick a dt below it. `GravitySph` overrides it via the
//! gas-subset CFL reduction; a pure-gravity solver (or a gas-free state) imposes
//! no hydro constraint and returns `+∞`.
//!
//! NOTE (D2): these gate the *bound query*, not an adaptive trajectory. The
//! adaptive path deliberately forfeits leapfrog time-reversibility and
//! energy-oscillation (variable dt is not symplectic); its correctness gates are
//! convergence-to-a-fine-dt-reference and bounded drift, NOT the fixed-dt
//! invariant gates — see the plan doc, D2.

use galaxy_core::{DVec3, ForceSolver, Species, State};
use galaxy_solvers::sph::{max_stable_dt as cfl_limit, DensityConfig, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

fn random_points(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * scale)
        .collect()
}

fn gas_state(pos: Vec<DVec3>, vel: Vec<DVec3>) -> State {
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for k in s.kind.iter_mut() {
        *k = Species::Gas;
    }
    s
}

/// The `GravitySph` trait method returns the CFL limit at Courant number 1 — the
/// raw `min_i h_i / v_sig,i` — matching the free `max_stable_dt(..., c_cfl = 1.0)`
/// verbatim (the loop applies its own Courant × safety below it).
#[test]
fn gravity_sph_reports_the_cfl_limit_at_courant_one() {
    let pos = random_points(101, 700, 2.5);
    let vel = random_points(102, 700, 1.2); // live signal velocity
    let state = gas_state(pos, vel);
    let params = HydroParams::default();
    let cfg = DensityConfig::default();

    // Solver bound (Courant-number policy lives OUTSIDE the solver, so the method
    // reports the c_cfl = 1 limit).
    let solver = GravitySph::new(BarnesHut::new(1.0, 0.05, 0.5), params, cfg.clone());
    let got = solver.max_stable_dt(&state);

    let expect = cfl_limit(&state, &params, &cfg, 1.0);
    assert!(
        expect.is_finite() && expect > 0.0,
        "the gas cloud must have a finite positive CFL limit"
    );
    let rel = (got - expect).abs() / expect;
    assert!(
        rel < 1e-12,
        "GravitySph::max_stable_dt = {got}, want the c_cfl=1 CFL limit {expect}"
    );
}

/// A gas-free state carries no hydro CFL constraint, so even the gas-capable
/// `GravitySph` reports `+∞` (the gravity-only physics imposes no v1 dt limit).
#[test]
fn gravity_sph_over_gas_free_state_is_unconstrained() {
    let pos = random_points(200, 300, 3.0);
    let state = State::from_phase_space(pos, vec![DVec3::ZERO; 300], vec![1.0; 300]);
    let solver = GravitySph::new(
        BarnesHut::new(1.0, 0.05, 0.5),
        HydroParams::default(),
        DensityConfig::default(),
    );
    assert_eq!(solver.max_stable_dt(&state), f64::INFINITY);
}

/// A pure-gravity solver inherits the trait default: no timestep constraint.
#[test]
fn bare_gravity_solver_inherits_infinite_bound() {
    let pos = random_points(300, 400, 3.0);
    let state = gas_state(pos.clone(), vec![DVec3::ZERO; pos.len()]);
    let solver = BarnesHut::new(1.0, 0.05, 0.5);
    // Even over a gas state, plain gravity imposes no hydro CFL limit (v1).
    assert_eq!(solver.max_stable_dt(&state), f64::INFINITY);
}
