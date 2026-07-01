//! Parallel Barnes-Hut equivalence guard.
//!
//! Parallelizing the force loop is a *perf* change; its correctness contract is
//! "results are unchanged". The physics accuracy is already pinned by the oracle
//! suite (`barnes_hut.rs`: θ→0 reproduces direct sum, finite-θ error bounds).
//! What parallelism newly introduces — and what these tests pin — is:
//!
//!  1. **Bit-exact equivalence:** the parallel `accelerations` must equal the
//!     serial `accelerations_serial` to the *last bit*. This is achievable (not
//!     just to-tolerance) because the loop parallelizes over independent targets:
//!     each `acc[i]` is written by exactly one task via the same fixed traversal,
//!     so no floating-point accumulation is ever reassociated. (Reductions like
//!     `potential_energy` do NOT get this guarantee and are excluded here.)
//!  2. **Determinism:** repeated runs yield identical output, regardless of how
//!     rayon happens to schedule/steal work across threads.
//!
//! A fixed seed set keeps the suite fully deterministic (see `barnes_hut.rs` for
//! why entropy-seeded proptest was abandoned for this solver).

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::BarnesHut;

/// Deterministic pseudo-random cluster (LCG; no external rand dep). Mirrors the
/// generator used by the oracle suite so both exercise the same geometries.
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

fn parallel_accel(bh: &BarnesHut, s: &State) -> Vec<DVec3> {
    let mut a = vec![DVec3::ZERO; s.len()];
    let mut bh = *bh;
    bh.accelerations(s, &mut a);
    a
}

fn serial_accel(bh: &BarnesHut, s: &State) -> Vec<DVec3> {
    let mut a = vec![DVec3::ZERO; s.len()];
    bh.accelerations_serial(s, &mut a);
    a
}

/// A spread of sizes so the tree is sometimes shallow, sometimes deep, and always
/// large enough that rayon splits the target range across worker threads.
const SIZES: [usize; 4] = [1, 137, 1024, 4096];

#[test]
fn parallel_matches_serial_bit_exact() {
    let bh = BarnesHut::new(1.0, 0.05, 0.5);
    for &n in &SIZES {
        for seed in 0..16u64 {
            let s = cluster(seed, n);
            let par = parallel_accel(&bh, &s);
            let ser = serial_accel(&bh, &s);
            for i in 0..n {
                // Exact equality: same ops, same order — differences would signal a
                // data race or an accidental reassociation of the per-target sum.
                assert_eq!(
                    par[i].to_array(),
                    ser[i].to_array(),
                    "parallel != serial at particle {i} (n={n}, seed={seed})"
                );
            }
        }
    }
}

#[test]
fn parallel_is_deterministic_across_runs() {
    let bh = BarnesHut::new(1.0, 0.05, 0.5);
    let s = cluster(0xBEEF, 4096);
    let a = parallel_accel(&bh, &s);
    // Re-run several times; rayon's scheduling/work-stealing must not perturb the
    // result. Any drift here means a nondeterministic reduction crept in.
    for run in 0..8 {
        let b = parallel_accel(&bh, &s);
        for i in 0..s.len() {
            assert_eq!(
                a[i].to_array(),
                b[i].to_array(),
                "nondeterministic parallel output at particle {i} (run {run})"
            );
        }
    }
}
