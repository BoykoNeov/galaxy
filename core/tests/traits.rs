//! Default `ForceSolver::accel_and_dudt` (E2a): pure-force solvers get
//! `du/dt≡0` for free by delegating to `accelerations`, so no existing solver
//! (pure gravity, GPU) needs to change to gain the fused-pass trait surface.

use galaxy_core::{DVec3, ForceSolver, State};

struct Harmonic;
impl ForceSolver for Harmonic {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        for (a, x) in acc.iter_mut().zip(&state.pos) {
            *a = -*x;
        }
    }
    fn potential_energy(&self, state: &State) -> f64 {
        0.5 * state.pos.iter().map(|x| x.length_squared()).sum::<f64>()
    }
}

fn ic() -> State {
    State::from_phase_space(
        vec![
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-0.5, -0.5, 0.2),
        ],
        vec![DVec3::ZERO; 3],
        vec![1.0, 1.0, 1.0],
    )
}

#[test]
fn default_accel_and_dudt_delegates_to_accelerations_and_zeros_dudt() {
    let state = ic();
    let mut solver = Harmonic;
    let n = state.len();

    let mut acc_direct = vec![DVec3::ZERO; n];
    solver.accelerations(&state, &mut acc_direct);

    // Poison both buffers first so a no-op default couldn't accidentally pass.
    let mut acc_fused = vec![DVec3::new(9.0, 9.0, 9.0); n];
    let mut dudt = vec![7.0; n];
    solver.accel_and_dudt(&state, &mut acc_fused, &mut dudt);

    assert_eq!(
        acc_fused, acc_direct,
        "default accel_and_dudt must match accelerations()"
    );
    assert!(
        dudt.iter().all(|&d| d == 0.0),
        "default accel_and_dudt must zero-fill dudt for a non-thermal solver"
    );
}
