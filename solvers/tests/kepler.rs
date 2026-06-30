//! Eccentric two-body validation. The circular binary in `physics.rs` is the
//! degenerate, constant-force case; an eccentric orbit exercises the varying-
//! force regime near perihelion where fixed-step integrators are stressed.
//!
//! For a central force, KDK leapfrog conserves energy (bounded) and angular
//! momentum (exactly), and preserves the orbit *shape* (semi-major axis a and
//! eccentricity e). It is NOT orbit-orientation preserving: symplectic leapfrog
//! precesses the apsides at O(dt^2). We therefore assert the precession is small
//! and bounded, not zero.

use galaxy_core::{diagnostics, DVec3, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_solvers::DirectSum;

const PI: f64 = std::f64::consts::PI;

/// Osculating (a, e, eccentricity-vector) of a relative two-body orbit.
fn elements(r_rel: DVec3, v_rel: DVec3, mu: f64) -> (f64, f64, DVec3) {
    let r = r_rel.length();
    let energy = 0.5 * v_rel.length_squared() - mu / r;
    let a = -mu / (2.0 * energy);
    let h = r_rel.cross(v_rel);
    let e_vec = v_rel.cross(h) / mu - r_rel / r;
    (a, e_vec.length(), e_vec)
}

#[test]
fn eccentric_two_body_conserves_shape_and_barely_precesses() {
    let (g, m1, m2) = (1.0_f64, 1.0, 0.5);
    let mtot = m1 + m2;
    let mu = g * mtot;
    let (a0, ecc0) = (1.0_f64, 0.6);

    // Start at apoapsis on +x, relative velocity tangential (+y).
    let r_apo = a0 * (1.0 + ecc0);
    let v_apo = (mu * (1.0 - ecc0) / (a0 * (1.0 + ecc0))).sqrt();
    let r_rel0 = DVec3::new(r_apo, 0.0, 0.0);
    let v_rel0 = DVec3::new(0.0, v_apo, 0.0);

    // Place in the center-of-mass frame.
    let mut s = State::from_phase_space(
        vec![r_rel0 * (-m2 / mtot), r_rel0 * (m1 / mtot)],
        vec![v_rel0 * (-m2 / mtot), v_rel0 * (m1 / mtot)],
        vec![m1, m2],
    );

    let period = 2.0 * PI * (a0.powi(3) / mu).sqrt();
    let mut solver = DirectSum::new(g, 1e-6); // eps << perihelion r = a(1-e) = 0.4
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let steps_per_orbit = 8000;
    let orbits = 5;
    let dt = period / steps_per_orbit as f64;

    let rel = |s: &State| (s.pos[1] - s.pos[0], s.vel[1] - s.vel[0]);
    let (r, v) = rel(&s);
    let (a_init, e_init, evec_init) = elements(r, v, mu);
    let ang_init = evec_init.y.atan2(evec_init.x);
    let energy0 = diagnostics::total_energy(&s, &solver);
    let l0 = diagnostics::total_angular_momentum(&s);

    let mut max_e_err = 0.0_f64;
    let mut max_l_err = 0.0_f64;
    let mut max_da = 0.0_f64;
    let mut max_de = 0.0_f64;
    for _ in 0..(orbits * steps_per_orbit) {
        integ.step(&mut s, &mut solver, &bg, dt);
        let energy = diagnostics::total_energy(&s, &solver);
        max_e_err = max_e_err.max(((energy - energy0) / energy0).abs());
        max_l_err = max_l_err.max((diagnostics::total_angular_momentum(&s) - l0).length());
        let (r, v) = rel(&s);
        let (a, e, _) = elements(r, v, mu);
        max_da = max_da.max((a - a_init).abs() / a_init);
        max_de = max_de.max((e - e_init).abs());
    }

    let (r, v) = rel(&s);
    let (_, _, evec_f) = elements(r, v, mu);
    let precession = (evec_f.y.atan2(evec_f.x) - ang_init).abs();

    assert!(
        e_init > 0.55 && e_init < 0.65,
        "setup wrong: e_init = {e_init}"
    );
    assert!(max_e_err < 1e-3, "energy not conserved: {max_e_err:e}");
    assert!(
        max_l_err < 1e-8,
        "angular momentum not conserved (central force): {max_l_err:e}"
    );
    assert!(max_da < 1e-3, "semi-major axis drifted: {max_da:e}");
    assert!(max_de < 1e-3, "eccentricity drifted: {max_de:e}");
    assert!(
        precession < 0.02,
        "apsidal precession too large over {orbits} orbits: {precession:e} rad"
    );
}
