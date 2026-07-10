//! I7 — the active-subset gather WIRED into the stepper (efficiency, not physics).
//!
//! The physics of the active path (bit-identity to the full gather, partial-active
//! correctness) is gated at the solver level in `solvers/tests/sph_active_gather.rs`.
//! Its downstream CORRECTNESS (the stale-neighbour bounded approximation still
//! converges) is gated by the driver convergence tests (`individual_driver.rs`,
//! I4a), which — once the stepper calls `accelerations_active` — now exercise the
//! active path automatically and must stay green.
//!
//! THIS file pins the EFFICIENCY claim the whole milestone exists for: the stepper
//! must ask the solver for forces on only the ACTIVE subset each fine tick, so the
//! total number of per-target force evaluations over one base block is `Σ_i 2^r_i`
//! (each rung-`r` particle is active `2^r` times per block) — strictly fewer than
//! the `N · 2^r_max` a recompute-everything-every-tick loop (the I3 stepper) costs.
//! A counting mock solver records exactly which targets it is asked for.

use galaxy_core::{DVec3, ForceSolver, State, StaticBackground};
use galaxy_sim::individual::ActiveSetKdk;

/// A trivial constant-acceleration solver that COUNTS how many per-target force
/// evaluations it is asked to do. `accelerations` (full) charges `N`;
/// `accelerations_active` (the I7 path) charges `active.len()`. Whichever the
/// stepper calls, `evals` reflects the true gather cost.
struct CountingSolver {
    evals: usize,
    accel: DVec3,
}

impl ForceSolver for CountingSolver {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        self.evals += state.len();
        for a in acc.iter_mut() {
            *a = self.accel;
        }
    }
    fn accelerations_active(&mut self, _state: &State, active: &[usize], acc: &mut [DVec3]) {
        self.evals += active.len();
        for &i in active {
            acc[i] = self.accel;
        }
    }
    fn potential_energy(&self, _state: &State) -> f64 {
        0.0
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

#[test]
fn active_path_reduces_force_evals_to_sum_of_rungs() {
    // A hand-made rung spread (three distinct rungs, finest r_max=3).
    let rungs: Vec<u32> = vec![0, 1, 2, 3, 3, 2, 1, 0];
    let n = rungs.len();
    let r_max = *rungs.iter().max().unwrap();
    let n_fine: u64 = 1 << r_max;

    let mut rng = lcg(99);
    let pos: Vec<DVec3> = (0..n)
        .map(|_| DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5))
        .collect();
    let vel: Vec<DVec3> = (0..n)
        .map(|_| DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 0.1)
        .collect();
    let mut state = State::from_phase_space(pos, vel, vec![1.0; n]);

    let mut solver = CountingSolver {
        evals: 0,
        accel: DVec3::new(0.0, -1.0, 0.0),
    };
    let bg = StaticBackground;
    let mut stepper = ActiveSetKdk::new();

    // Prime OUTSIDE the measured window (the opening kick reuses a full block-start
    // force eval — that cost is shared by both the full and active stepper and is
    // not what I7 reduces), then zero the counter and measure ONE base block.
    stepper.prime(&state, &mut solver);
    solver.evals = 0;
    stepper.step_block(&mut state, &mut solver, &bg, 0.1, &rungs);

    // Each rung-r particle is active (needs a fresh force) 2^r times per block.
    let expected: usize = rungs.iter().map(|&r| 1usize << r).sum();
    let full_baseline = n * n_fine as usize; // recompute-everything-every-tick (I3)

    assert!(
        solver.evals < full_baseline,
        "active gather must cost fewer force evals than recompute-all: {} vs {}",
        solver.evals,
        full_baseline
    );
    assert_eq!(
        solver.evals, expected,
        "active gather must cost exactly Σ_i 2^r_i force evals per block"
    );
}
