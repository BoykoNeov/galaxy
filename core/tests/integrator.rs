//! Integrator-reuse safety. KDK leapfrog caches accelerations across steps, so
//! reusing one integrator on a new initial condition (same particle count)
//! without resetting would open the first half-kick with a *stale* acceleration
//! from the previous run. `reset()` / `prime()` make reuse safe.

use galaxy_core::{
    DVec3, ForceSolver, Integrator, LeapfrogKdk, LeapfrogKdkThermal, State, StaticBackground,
};

/// Position-dependent toy force (harmonic, a = -x) so that a stale cached
/// acceleration from a different state is observably wrong.
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

fn ic_a() -> State {
    State::from_phase_space(
        vec![DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 1.0, 0.0)],
        vec![DVec3::new(0.0, 0.5, 0.0), DVec3::new(-0.5, 0.0, 0.0)],
        vec![1.0, 1.0],
    )
}

fn ic_b() -> State {
    State::from_phase_space(
        vec![DVec3::new(-2.0, 1.0, 0.5), DVec3::new(1.5, -1.0, 0.0)],
        vec![DVec3::new(0.1, 0.0, -0.2), DVec3::new(0.0, 0.3, 0.0)],
        vec![1.0, 1.0],
    )
}

fn run(mut s: State, integ: &mut LeapfrogKdk, steps: usize, dt: f64) -> State {
    let mut solver = Harmonic;
    let bg = StaticBackground;
    for _ in 0..steps {
        integ.step(&mut s, &mut solver, &bg, dt);
    }
    s
}

#[test]
fn reset_makes_integrator_reuse_match_a_fresh_instance() {
    let (steps, dt) = (200, 1e-2);

    // Reference: a fresh integrator on IC-A.
    let mut fresh = LeapfrogKdk::new();
    let reference = run(ic_a(), &mut fresh, steps, dt);

    // Reuse: same integrator runs IC-B first (dirtying the cache), then reset().
    let mut reused = LeapfrogKdk::new();
    let _ = run(ic_b(), &mut reused, steps, dt);
    reused.reset();
    let after_reset = run(ic_a(), &mut reused, steps, dt);
    assert_eq!(
        after_reset, reference,
        "reset() must clear the stale acceleration cache"
    );

    // Footgun demonstrated: without reset(), reuse diverges from the reference.
    let mut dirty = LeapfrogKdk::new();
    let _ = run(ic_b(), &mut dirty, steps, dt);
    let without_reset = run(ic_a(), &mut dirty, steps, dt);
    assert_ne!(
        without_reset, reference,
        "reuse without reset() should be observably wrong"
    );
}

#[test]
fn prime_matches_lazy_first_step() {
    let (steps, dt) = (50, 1e-2);

    // Lazy auto-prime on a fresh integrator.
    let mut lazy = LeapfrogKdk::new();
    let lazy_out = run(ic_a(), &mut lazy, steps, dt);

    // Explicit prime before the first step must produce the identical trajectory.
    let mut eager = LeapfrogKdk::new();
    {
        let mut solver = Harmonic;
        eager.prime(&ic_a(), &mut solver);
    }
    let eager_out = run(ic_a(), &mut eager, steps, dt);
    assert_eq!(
        eager_out, lazy_out,
        "prime() must match the lazy first-step priming"
    );
}

/// Toy thermal solver (E2b plumbing gate): harmonic accel (same as
/// `Harmonic`) plus a `du/dt` that is a DIFFERENT function of position
/// (`|x|²`, not derived from `acc`) so a bug that reuses or skips one of the
/// two per-step force evaluations shows up in `u`, independent of any
/// acceleration check.
struct ThermalToy;
impl ForceSolver for ThermalToy {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        for (a, x) in acc.iter_mut().zip(&state.pos) {
            *a = -*x;
        }
    }
    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        self.accelerations(state, acc);
        for (d, x) in dudt.iter_mut().zip(&state.pos) {
            *d = x.length_squared();
        }
    }
    fn potential_energy(&self, state: &State) -> f64 {
        0.5 * state.pos.iter().map(|x| x.length_squared()).sum::<f64>()
    }
}

#[test]
fn thermal_kick_matches_the_hand_oracle() {
    // One particle, one step: hand-derive both v and u through the KDK's two
    // force evaluations (x_n then x_{n+1}) to pin the u-kick's timing/ordering
    // — the highest-risk detail per the E2b plan (a mismatch would show up
    // only as energy drift in the dynamical SPH gate, not a failed accel
    // check).
    let x0 = DVec3::new(1.0, 0.0, 0.0);
    let v0 = DVec3::new(0.0, 0.5, 0.0);
    let mut state = State::from_phase_space(vec![x0], vec![v0], vec![1.0]);
    let mut solver = ThermalToy;
    let bg = StaticBackground;
    let mut integ = LeapfrogKdkThermal::new();
    let dt = 0.1;
    integ.step(&mut state, &mut solver, &bg, dt);

    let half = 0.5 * dt;
    let acc0 = -x0;
    let dudt0 = x0.length_squared();
    let v_half = v0 + acc0 * half;
    let u_half = dudt0 * half; // u0 = 0
    let x1 = x0 + v_half * dt;
    let acc1 = -x1;
    let dudt1 = x1.length_squared();
    let v1 = v_half + acc1 * half;
    let u1 = u_half + dudt1 * half;

    assert!(
        (state.pos[0] - x1).length() < 1e-14,
        "pos = {:?}",
        state.pos[0]
    );
    assert!(
        (state.vel[0] - v1).length() < 1e-14,
        "vel = {:?}",
        state.vel[0]
    );
    assert!(
        (state.u[0] - u1).abs() < 1e-14,
        "u = {} vs oracle {u1}",
        state.u[0]
    );
    assert_eq!(state.time, dt);
}
