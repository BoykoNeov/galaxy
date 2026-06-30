//! Physics validation: direct-sum forces vs Newton, plus the conservation and
//! time-reversibility invariants of leapfrog. These combine solver + integrator
//! so they live as integration tests here.

use galaxy_core::{
    diagnostics, DVec3, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground,
};
use galaxy_solvers::DirectSum;
use proptest::prelude::*;

const PI: f64 = std::f64::consts::PI;

/// Deterministic pseudo-random cluster via a small LCG (no external rand dep).
fn random_cluster(seed: u64, n: usize, pos_scale: f64, vel_scale: f64) -> State {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) // in [0, 1)
    };
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    for _ in 0..n {
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * pos_scale);
        vel.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * vel_scale);
        mass.push(0.1 + 0.9 * next());
    }
    State::from_phase_space(pos, vel, mass)
}

#[test]
fn direct_sum_force_matches_newton() {
    // Two unit masses at x = ±1 (separation 2), G = 1, negligible softening.
    let s = State::from_phase_space(
        vec![DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
        vec![DVec3::ZERO, DVec3::ZERO],
        vec![1.0, 1.0],
    );
    let mut solver = DirectSum::new(1.0, 1e-9);
    let mut acc = vec![DVec3::ZERO; 2];
    solver.accelerations(&s, &mut acc);
    // |a| = G m / r^2 = 1/4; body0 pulled toward +x, body1 toward -x.
    assert!((acc[0] - DVec3::new(0.25, 0.0, 0.0)).length() < 1e-9, "acc0 = {:?}", acc[0]);
    assert!((acc[1] - DVec3::new(-0.25, 0.0, 0.0)).length() < 1e-9, "acc1 = {:?}", acc[1]);
}

/// Equal-mass circular binary about the origin: each mass at radius d/2 with
/// speed v = sqrt(G m / (2 d)); period T = 2*pi*(d/2)/v.
fn circular_binary(g: f64, m: f64, d: f64) -> (State, f64) {
    let r = d / 2.0;
    let v = (g * m / (2.0 * d)).sqrt();
    let s = State::from_phase_space(
        vec![DVec3::new(-r, 0.0, 0.0), DVec3::new(r, 0.0, 0.0)],
        vec![DVec3::new(0.0, -v, 0.0), DVec3::new(0.0, v, 0.0)],
        vec![m, m],
    );
    (s, 2.0 * PI * r / v)
}

#[test]
fn circular_binary_conserves_energy_and_closes_orbit() {
    let (g, m, d) = (1.0, 1.0, 2.0);
    let (mut s, period) = circular_binary(g, m, d);
    let s0 = s.clone();
    let mut solver = DirectSum::new(g, 1e-4); // epsilon << d
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let steps = 4000;
    let dt = period / steps as f64;
    let e0 = diagnostics::total_energy(&s, &solver);
    let mut max_e_err = 0.0_f64;
    for _ in 0..steps {
        integ.step(&mut s, &mut solver, &bg, dt);
        let e = diagnostics::total_energy(&s, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
    }
    assert!(max_e_err < 1e-3, "energy drift too large: {max_e_err:e}");
    // Orbit closes after exactly one period.
    let c0 = (s.pos[0] - s0.pos[0]).length();
    let c1 = (s.pos[1] - s0.pos[1]).length();
    assert!(c0 < 0.05 * d && c1 < 0.05 * d, "orbit did not close: {c0}, {c1}");
    // Center of mass stays put.
    assert!(diagnostics::center_of_mass(&s).length() < 1e-6, "COM drifted");
}

#[test]
fn cluster_conserves_energy_momentum_and_angular_momentum() {
    let mut s = random_cluster(0xC0FFEE, 50, 2.0, 0.2);
    let mut solver = DirectSum::new(1.0, 0.05); // softening suppresses close-encounter scattering
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let e0 = diagnostics::total_energy(&s, &solver);
    let p0 = diagnostics::total_momentum(&s);
    let l0 = diagnostics::total_angular_momentum(&s);
    let dt = 1e-3;
    let mut max_e_err = 0.0_f64;
    for _ in 0..2000 {
        integ.step(&mut s, &mut solver, &bg, dt);
        let e = diagnostics::total_energy(&s, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
    }
    assert!(max_e_err < 1e-2, "energy not conserved: {max_e_err:e}");
    assert!(
        (diagnostics::total_momentum(&s) - p0).length() < 1e-8,
        "linear momentum not conserved"
    );
    assert!(
        (diagnostics::total_angular_momentum(&s) - l0).length() < 1e-8,
        "angular momentum not conserved"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// KDK leapfrog is time-reversible: integrate forward, flip velocities,
    /// integrate the same number of steps, flip back => recover initial state.
    #[test]
    fn leapfrog_is_time_reversible(
        seed in any::<u64>(),
        dt in 1e-4f64..2e-3,
        steps in 5usize..40,
    ) {
        let mut s = random_cluster(seed, 12, 2.0, 0.1);
        let s0 = s.clone();
        let mut solver = DirectSum::new(1.0, 0.1);
        let bg = StaticBackground;

        let mut fwd = LeapfrogKdk::new();
        for _ in 0..steps { fwd.step(&mut s, &mut solver, &bg, dt); }

        for v in s.vel.iter_mut() { *v = -*v; }
        let mut rev = LeapfrogKdk::new();
        for _ in 0..steps { rev.step(&mut s, &mut solver, &bg, dt); }
        for v in s.vel.iter_mut() { *v = -*v; }

        for i in 0..s.len() {
            prop_assert!((s.pos[i] - s0.pos[i]).length() < 1e-6, "pos drift at {i}");
            prop_assert!((s.vel[i] - s0.vel[i]).length() < 1e-6, "vel drift at {i}");
        }
    }
}
