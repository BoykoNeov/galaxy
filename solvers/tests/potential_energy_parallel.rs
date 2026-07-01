//! Parallel softened-potential-energy equivalence guard.
//!
//! `potential_energy` is an O(N²) reduction and stays O(N²) even under
//! Barnes-Hut, so at scale it dominates whenever the energy diagnostic runs —
//! worth parallelizing. Unlike the force fill, a parallel *reduction* reassociates
//! the floating-point sum (rayon splits the index range and folds sub-ranges),
//! so the result is NOT bit-identical to the serial nested loop. The contract is
//! therefore equality to a tight *relative tolerance*, not to the last bit.
//!
//! Bound: 1e-12 relative. Double precision carries ~15-16 digits; an O(N)
//! reduction of same-sign terms (the potential is strictly negative here) loses
//! at most a few ULP-scale digits to reassociation, far inside 1e-12. This pins
//! "the parallel reduction computes the same physical quantity" while tolerating
//! the reassociation that a bit-exact assert would wrongly flag.

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::{BarnesHut, DirectSum};

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

const SIZES: [usize; 4] = [1, 137, 1024, 4096];

fn assert_close(par: f64, ser: f64, ctx: &str) {
    assert!(
        par.is_finite(),
        "parallel potential not finite ({ctx}): {par}"
    );
    let tol = 1e-12 * ser.abs().max(1e-300);
    assert!(
        (par - ser).abs() <= tol,
        "parallel != serial potential ({ctx}): par={par:e} ser={ser:e} \
         |Δ|={:e} > tol={tol:e}",
        (par - ser).abs()
    );
}

#[test]
fn direct_sum_parallel_potential_matches_serial() {
    let ds = DirectSum::new(1.0, 0.05);
    for &n in &SIZES {
        for seed in 0..16u64 {
            let s = cluster(seed, n);
            let par = ds.potential_energy(&s);
            let ser = ds.potential_energy_serial(&s);
            assert_close(par, ser, &format!("DirectSum n={n} seed={seed}"));
        }
    }
}

#[test]
fn barnes_hut_parallel_potential_matches_serial() {
    let bh = BarnesHut::new(1.0, 0.05, 0.5);
    for &n in &SIZES {
        for seed in 0..16u64 {
            let s = cluster(seed, n);
            let par = bh.potential_energy(&s);
            let ser = bh.potential_energy_serial(&s);
            assert_close(par, ser, &format!("BarnesHut n={n} seed={seed}"));
        }
    }
}
