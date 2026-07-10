//! I-grav (milestone 9) — the gravitational per-particle timestep criterion and its
//! merge with the hydro CFL vector, the rung POLICY for `hydro+gravity` mode.
//!
//! Collisionless stars have hydro `dt = +∞` (no CFL constraint) ⇒ under `hydro-only`
//! they never subcycle and the gravity walk stays all-N. Giving them a FINITE
//! gravitational rung `dt_i = η·√(ε/|a_i|)` is what lets the gravity walk reduce to an
//! active subset (lever b). This file gates the pure criterion + the min-merge with
//! the hydro vector — independent of all the stale-tree / active-walk machinery.
//!
//! GATE DESIGN: hand-derived values (`η·√(ε/|a|)`, independent of the fn's own
//! output), the `+∞` force-free case (inverted vs the hydro `+∞`), the min-merge
//! semantics (gas takes the tighter of hydro/grav; a star takes grav alone; a
//! force-free star stays `+∞`), and monotonicity in `|a|`.

use galaxy_sim::individual::{combined_particle_dt, grav_rung_dt};

const INF: f64 = f64::INFINITY;

#[test]
fn grav_rung_dt_is_the_softened_criterion() {
    // dt = η·√(ε/|a|), hand-derived independent of the function.
    let (eps, eta) = (0.05, 0.3);
    for &a in &[0.1_f64, 1.0, 7.5, 123.4] {
        let want = eta * (eps / a).sqrt();
        assert!(
            (grav_rung_dt(a, eps, eta) - want).abs() <= 1e-15 * want.max(1.0),
            "grav_rung_dt({a}) != {want}"
        );
    }
}

#[test]
fn grav_rung_dt_is_infinite_for_a_force_free_particle() {
    // |a| = 0 ⇒ dt = +∞ = the COARSEST rung (best case), inverted vs the hydro +∞.
    assert_eq!(grav_rung_dt(0.0, 0.05, 0.3), INF);
    assert_eq!(grav_rung_dt(-0.0, 0.05, 0.3), INF);
}

#[test]
fn grav_rung_dt_is_monotone_decreasing_in_accel() {
    // Stronger pull ⇒ shorter safe step (finer rung).
    let (eps, eta) = (0.05, 0.3);
    let mut prev = INF;
    for &a in &[0.01_f64, 0.1, 1.0, 10.0, 100.0] {
        let dt = grav_rung_dt(a, eps, eta);
        assert!(
            dt < prev,
            "dt must strictly decrease as |a| grows: {dt} !< {prev}"
        );
        prev = dt;
    }
}

#[test]
fn combined_takes_the_tighter_of_hydro_and_gravity_per_particle() {
    let (eps, eta) = (0.05, 0.3);
    // Row 0: gas whose HYDRO step is tighter than its gravitational step.
    // Row 1: gas whose GRAVITATIONAL step is tighter than its hydro step.
    // Row 2: a star (hydro +∞) with a finite pull ⇒ takes gravity alone.
    // Row 3: a force-free star (hydro +∞, |a| = 0) ⇒ stays +∞ (rung 0).
    let a_small = 0.1; // grav dt = 0.3·√(0.05/0.1) ≈ 0.2121
    let a_big = 100.0; // grav dt = 0.3·√(0.05/100) ≈ 0.006708
    let hydro = vec![0.001, 0.5, INF, INF];
    let grav_mag = vec![a_small, a_big, a_small, 0.0];

    let got = combined_particle_dt(&hydro, &grav_mag, eps, eta);
    assert_eq!(got.len(), 4, "output is state-length");
    assert_eq!(got[0], 0.001, "gas keeps its tighter hydro step");
    assert_eq!(
        got[1],
        grav_rung_dt(a_big, eps, eta),
        "gas takes its tighter gravitational step"
    );
    assert_eq!(
        got[2],
        grav_rung_dt(a_small, eps, eta),
        "a star (hydro +∞) takes its gravitational step"
    );
    assert_eq!(got[3], INF, "a force-free star stays +∞ (coarsest rung)");
}
