//! Barnes-Hut validated against the direct-sum oracle. The key structural check
//! is θ→0: the tree must then reproduce direct summation to roundoff, so a single
//! mis-bucketed particle shows up as a large worst-case error. At finite θ the
//! worst-case error is bounded and grows with θ.
//!
//! Errors are normalized by the RMS force, not per-particle magnitude, so a
//! particle near a force null (where |a| ≈ 0) does not blow up the metric while
//! a genuinely mis-computed particle still does.

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::{BarnesHut, DirectSum};
use proptest::prelude::*;

/// Deterministic pseudo-random cluster (LCG; no external rand dep).
fn cluster(seed: u64, n: usize) -> State {
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
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 3.0);
        vel.push(DVec3::ZERO);
        mass.push(0.1 + 0.9 * next());
    }
    State::from_phase_space(pos, vel, mass)
}

fn accel(solver: &mut dyn ForceSolver, s: &State) -> Vec<DVec3> {
    let mut a = vec![DVec3::ZERO; s.len()];
    solver.accelerations(s, &mut a);
    a
}

/// Worst-case per-particle error, normalized by the RMS acceleration.
fn worst_rel_err(approx: &[DVec3], exact: &[DVec3]) -> f64 {
    let n = exact.len() as f64;
    let ms = exact.iter().map(|a| a.length_squared()).sum::<f64>() / n;
    let rms = ms.sqrt().max(1e-300);
    approx
        .iter()
        .zip(exact)
        .map(|(b, d)| (*b - *d).length() / rms)
        .fold(0.0_f64, f64::max)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn barnes_hut_matches_direct_sum(seed in any::<u64>()) {
        const N: usize = 120;
        let (g, eps) = (1.0, 0.05);
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(g, eps), &s);

        // θ→0 must reproduce direct summation to roundoff (catches mis-bucketing).
        let near0 = accel(&mut BarnesHut::new(g, eps, 1e-6), &s);
        let e0 = worst_rel_err(&near0, &exact);
        prop_assert!(e0 < 1e-9, "theta->0 must match oracle: worst rel err {e0:e}");

        // Finite θ: worst-case bounded and growing with θ.
        let e_lo = worst_rel_err(&accel(&mut BarnesHut::new(g, eps, 0.3), &s), &exact);
        let e_hi = worst_rel_err(&accel(&mut BarnesHut::new(g, eps, 0.6), &s), &exact);
        prop_assert!(e_lo < 0.05, "theta=0.3 worst rel err {e_lo:e}");
        prop_assert!(e_hi < 0.20, "theta=0.6 worst rel err {e_hi:e}");
    }
}
