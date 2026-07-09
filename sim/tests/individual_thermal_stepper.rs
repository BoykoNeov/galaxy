//! I5 — the active-set KDK stepper on the ADIABATIC thermal arm.
//!
//! `ActiveSetKdkThermal` mirrors the isothermal `ActiveSetKdk` (I3) but also
//! evolves the per-particle internal energy `u`: it kicks `u` via `du/dt` from
//! `accel_and_dudt` wherever it kicks `vel`, and applies the positive-`u` floor
//! (E4b) to the just-kicked ACTIVE subset. It is a DISTINCT type (not a branch
//! on `ActiveSetKdk`) so the frozen isothermal individual byte-path never has to
//! trust the `accel_and_dudt`-fills-`acc`-like-`accelerations` invariant.
//!
//! GATE DESIGN (advisor-vetted, 2026-07-10 — the plan's I5 "reduces to global-
//! adaptive thermal to tolerance" wording is superseded by bit-identity, exactly
//! as the I3 revision superseded it for the isothermal arm: `run_adaptive`'s
//! growth limiter diverges the dt sequence, so individual-vs-adaptive is apples-
//! to-oranges):
//!   * COLLAPSED → BIT-IDENTICAL. All-rung-0 ⇒ one fine tick, active set = all,
//!     sub-step = dt_base, floor after each half-kick ⇒ bit-for-bit equal to
//!     `LeapfrogKdkThermal` at dt_base on pos/vel/`u`/time. Run at `u_min = 0`
//!     (floor inert) so ordering — not the clamp — is what the gate tests. The
//!     load-bearing pin (dictates the kick/floor/drift ordering).
//!   * U-FLOOR LEAK is checked as an EQUALITY (the strongest form): the collapsed
//!     stepper with `u_min > 0` over a hard-cooling toy must report the SAME leak
//!     `Σ mᵢ(u_min − u_raw)` and land on the SAME `u` as `LeapfrogKdkThermal`
//!     with the same floor — apples-to-apples, not a hand tolerance. A separate
//!     multi-rung gate confirms the floor holds `u ≥ u_min` at the synchronized
//!     block boundary (per-active flooring reaches everyone by their closing kick).
//!   * PER-RUNG du/dt CONVERGENCE. With a STATE-COUPLED `dudt = |x|²` on an SHM
//!     field (the u-analog of the oscillator — `dudt` varies over the block, so
//!     WHEN a rung samples it matters; a constant `dudt` would integrate exactly
//!     per-rung and test nothing), a finer rung tracks the analytic `u(t)` more
//!     closely and the coarse-rung `u`-error falls ~2nd order under dt_base
//!     refinement — the "converges to truth as rungs refine" property, u-channel.

use galaxy_core::{
    Background, DVec3, ForceSolver, Integrator, LeapfrogKdkThermal, Species, State,
    StaticBackground,
};
use galaxy_sim::individual::ActiveSetKdkThermal;
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

const GAMMA: f64 = 1.4;

// --------------------------------------------------------------------------
// Mock solvers with a fused, state-coupled `accel_and_dudt`.
// --------------------------------------------------------------------------

/// SHM acceleration `a = −ω²x` with internal-energy source `dudt = |x|²`
/// (the E-series `ThermalToy` shape). Because `dudt` tracks the oscillating
/// position it VARIES over a base block, so per-rung sampling of `u` is a real
/// test of the u-kick timing (a constant `dudt` would be rung-exact and vacuous).
struct ShmThermal {
    omega2: f64,
}
impl ForceSolver for ShmThermal {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        for (a, x) in acc.iter_mut().zip(&state.pos) {
            *a = *x * -self.omega2;
        }
    }
    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        self.accelerations(state, acc);
        for (d, x) in dudt.iter_mut().zip(&state.pos) {
            *d = x.length_squared();
        }
    }
    fn potential_energy(&self, _state: &State) -> f64 {
        0.0
    }
}

/// Uniform CONSTANT acceleration `g` AND constant `du/dt = c` on every particle.
/// Leapfrog integrates constant fields EXACTLY at any step, so the block-end
/// pos/vel/`u` are closed-form regardless of rung — the sharpest pin on the kick
/// bookkeeping, and the ONLY gate that exercises the interior full-kick branch
/// (`n_fine > 1`, floor inert) without the floor clamping the result away.
struct ConstAccelThermal {
    g: DVec3,
    c: f64,
}
impl ForceSolver for ConstAccelThermal {
    fn accelerations(&mut self, _state: &State, acc: &mut [DVec3]) {
        for a in acc.iter_mut() {
            *a = self.g;
        }
    }
    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        self.accelerations(state, acc);
        for d in dudt.iter_mut() {
            *d = self.c;
        }
    }
    fn potential_energy(&self, _state: &State) -> f64 {
        0.0
    }
}

/// Zero acceleration + a large CONSTANT negative `du/dt` (E4b `CoolingToy`):
/// drives `u` below any positive floor within a step, isolating the clamp +
/// leak accounting from any dynamics.
struct CoolingToy {
    rate: f64,
}
impl ForceSolver for CoolingToy {
    fn accelerations(&mut self, _state: &State, acc: &mut [DVec3]) {
        for a in acc.iter_mut() {
            *a = DVec3::ZERO;
        }
    }
    fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
        self.accelerations(state, acc);
        for d in dudt.iter_mut() {
            *d = -self.rate;
        }
    }
    fn potential_energy(&self, _state: &State) -> f64 {
        0.0
    }
}

// --------------------------------------------------------------------------
// A small adiabatic gas ball for the real-solver bit-identity gate.
// --------------------------------------------------------------------------

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Gas ball with a mild swirl and POSITIVE internal energy (adiabatic needs
/// `u > 0` for a real sound speed). `u` varies particle-to-particle so the
/// `dudt` field is non-trivial.
fn gas_blob(seed: u64, n: usize, radius: f64) -> State {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    let vel: Vec<DVec3> = pos
        .iter()
        .map(|&p| DVec3::new(-p.y, p.x, 0.0) * 0.3)
        .collect();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    // Warm, positive, non-uniform internal energy.
    s.u = (0..n).map(|i| 0.5 + 0.1 * (i % 7) as f64).collect();
    s
}

fn adiabatic_solver() -> GravitySph<BarnesHut> {
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma: GAMMA },
        ..HydroParams::default()
    };
    GravitySph::<BarnesHut>::hydro_only(params, DensityConfig::default())
}

fn state_of(pos: Vec<DVec3>, vel: Vec<DVec3>, u: Vec<f64>) -> State {
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    s.u = u;
    s
}

// --------------------------------------------------------------------------
// GATE 1 — COLLAPSED → BIT-IDENTICAL to LeapfrogKdkThermal (u_min = 0).
// --------------------------------------------------------------------------

#[test]
fn collapsed_rungs_are_bit_identical_to_leapfrog_kdk_thermal() {
    let dt_base = 0.01;
    let n = 120;
    let bg = StaticBackground;

    // Reference: whole-state adiabatic KDK at dt_base (u_min = 0 ⇒ floor inert).
    let mut s_ref = gas_blob(0x7DE12A, n, 1.0);
    let mut solver_ref = adiabatic_solver();
    let mut kdk = LeapfrogKdkThermal::new();

    // Under test: active-set thermal stepper, EVERY particle on rung 0.
    let mut s_ind = gas_blob(0x7DE12A, n, 1.0);
    let mut solver_ind = adiabatic_solver();
    let mut stepper = ActiveSetKdkThermal::new();
    let rungs = vec![0u32; n];

    // Multiple blocks pins the cross-block cached-(acc,dudt) reuse.
    for block in 0..4 {
        kdk.step(&mut s_ref, &mut solver_ref, &bg, dt_base);
        stepper.step_block(&mut s_ind, &mut solver_ind, &bg, dt_base, &rungs);

        assert_eq!(
            s_ind.pos, s_ref.pos,
            "block {block}: positions must match LeapfrogKdkThermal bit-for-bit"
        );
        assert_eq!(
            s_ind.vel, s_ref.vel,
            "block {block}: velocities must match LeapfrogKdkThermal bit-for-bit"
        );
        assert_eq!(
            s_ind.u, s_ref.u,
            "block {block}: internal energy u must match LeapfrogKdkThermal bit-for-bit"
        );
        assert_eq!(s_ind.time, s_ref.time, "block {block}: time must match");
    }
    // No floor engaged on either path (u_min = 0, u stayed positive).
    assert_eq!(
        stepper.u_floor_energy(),
        0.0,
        "u_min = 0 floor must be inert on a positive-u run"
    );
}

// --------------------------------------------------------------------------
// GATE 1b — INTERIOR full-kick exactness under constant fields (u-channel).
// --------------------------------------------------------------------------

#[test]
fn two_rung_block_is_exact_under_constant_accel_and_dudt() {
    // Particle A rung 0 (full base step), particle B rung 2 (four fine sub-steps
    // ⇒ exercises the interior full-kick branch the collapsed gate never reaches).
    // Constant g and c ⇒ leapfrog is exact at any step, so BOTH land on the closed
    // form regardless of rung. B's opening + 3 interior full-kicks + closing sum to
    // exactly c·dt_base on u (and g·dt_base on v) — a wrong interior multiplier
    // (half instead of full, stale force, missed tick) throws B off this mark.
    let g = DVec3::new(0.3, -0.2, 0.1);
    let c = 0.7;
    let dt_base = 0.2;
    let x0 = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(5.0, 1.0, -2.0)];
    let v0 = [DVec3::new(1.0, 0.0, 0.5), DVec3::new(-0.4, 0.7, 0.0)];
    let u0 = [1.5, 0.4];

    let mut s = state_of(x0.to_vec(), v0.to_vec(), u0.to_vec());
    let mut solver = ConstAccelThermal { g, c };
    let mut stepper = ActiveSetKdkThermal::new(); // u_min = 0 ⇒ floor inert (c > 0)
    let bg = StaticBackground;
    let rungs = vec![0u32, 2u32];

    stepper.step_block(&mut s, &mut solver, &bg, dt_base, &rungs);

    for i in 0..2 {
        let want_x = x0[i] + v0[i] * dt_base + g * (0.5 * dt_base * dt_base);
        let want_v = v0[i] + g * dt_base;
        let want_u = u0[i] + c * dt_base;
        assert!(
            (s.pos[i] - want_x).length() < 1e-12,
            "particle {i} (rung {}) pos {:?} != analytic {:?}",
            rungs[i],
            s.pos[i],
            want_x
        );
        assert!(
            (s.vel[i] - want_v).length() < 1e-12,
            "particle {i} (rung {}) vel {:?} != analytic {:?}",
            rungs[i],
            s.vel[i],
            want_v
        );
        assert!(
            (s.u[i] - want_u).abs() < 1e-12,
            "particle {i} (rung {}) u {} != analytic {want_u}",
            rungs[i],
            s.u[i]
        );
    }
    assert_eq!(
        stepper.u_floor_energy(),
        0.0,
        "u_min = 0 floor must stay inert"
    );
}

// --------------------------------------------------------------------------
// GATE 2 — U-FLOOR LEAK EQUALITY (collapsed ≡ LeapfrogKdkThermal) + multi-rung hold.
// --------------------------------------------------------------------------

#[test]
fn collapsed_u_floor_leak_equals_leapfrog_kdk_thermal() {
    // Hard cooling drives u below a positive floor at both half-kicks. Collapsed
    // (all rung 0) the active-set stepper must inject the SAME energy and land on
    // the SAME u as LeapfrogKdkThermal with the same floor — the strongest form
    // of "u-floor leak reported": an equality, not a hand tolerance.
    let u_min = 0.1;
    let dt_base = 0.02;
    let rate = 50.0;
    let n = 2;
    let bg = StaticBackground;

    let pos = vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)];
    let vel = vec![DVec3::ZERO, DVec3::ZERO];
    let u0 = vec![0.5, 0.3];
    let mass = vec![2.0, 1.0];

    let mut s_ref = state_of(pos.clone(), vel.clone(), u0.clone());
    s_ref.mass = mass.clone();
    let mut integ = LeapfrogKdkThermal::with_u_floor(u_min);
    let mut solver_ref = CoolingToy { rate };

    let mut s_ind = state_of(pos, vel, u0);
    s_ind.mass = mass;
    let mut stepper = ActiveSetKdkThermal::with_u_floor(u_min);
    let mut solver_ind = CoolingToy { rate };
    let rungs = vec![0u32; n];

    for _ in 0..5 {
        integ.step(&mut s_ref, &mut solver_ref, &bg, dt_base);
        stepper.step_block(&mut s_ind, &mut solver_ind, &bg, dt_base, &rungs);
    }

    assert_eq!(
        s_ind.u, s_ref.u,
        "collapsed floored u must match LeapfrogKdkThermal bit-for-bit"
    );
    assert_eq!(
        stepper.u_floor_energy(),
        integ.u_floor_energy(),
        "collapsed floor leak must EQUAL LeapfrogKdkThermal's: {} vs {}",
        stepper.u_floor_energy(),
        integ.u_floor_energy()
    );
    assert!(
        stepper.u_floor_energy() > 0.0,
        "sanity: the floor must have engaged (non-vacuous)"
    );
}

#[test]
fn multi_rung_floor_holds_u_at_the_block_boundary() {
    // A genuinely multi-rung block under hard cooling: the floor is applied to
    // the just-kicked ACTIVE subset each fine tick, so by the synchronized block
    // boundary EVERY particle has taken its closing (floored) kick ⇒ u ≥ u_min
    // holds for all, and the injected energy is accounted (> 0).
    let u_min = 0.1;
    let dt_base = 0.02;
    let rate = 50.0;
    let bg = StaticBackground;

    let pos = vec![
        DVec3::ZERO,
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(2.0, 0.0, 0.0),
    ];
    let vel = vec![DVec3::ZERO; 3];
    let u0 = vec![0.5, 0.3, 0.4];
    let mut s = state_of(pos, vel, u0);
    s.mass = vec![2.0, 1.0, 1.5];

    let mut stepper = ActiveSetKdkThermal::with_u_floor(u_min);
    let mut solver = CoolingToy { rate };
    let rungs = vec![0u32, 1u32, 2u32]; // distinct rungs — a real active-set block

    stepper.step_block(&mut s, &mut solver, &bg, dt_base, &rungs);

    for (i, &u) in s.u.iter().enumerate() {
        assert!(
            u >= u_min - 1e-15,
            "particle {i} (rung {}) u = {u} fell below the floor {u_min}",
            rungs[i]
        );
    }
    assert!(
        stepper.u_floor_energy() > 0.0,
        "the floor must have injected energy under this hard cooling"
    );
}

// --------------------------------------------------------------------------
// GATE 3 — PER-RUNG du/dt CONVERGENCE (state-coupled dudt = |x|²).
// --------------------------------------------------------------------------

/// Run `n_blocks` base blocks of the SHM+thermal system with the given rungs,
/// returning each particle's max `u`-error vs the analytic `u(t)` at the
/// synchronized block boundaries. IC: x=(1,0,0), v=0, ω²=1, u=0 ⇒
/// x(t)=cos t, dudt=cos²t, u(t)=∫₀ᵗcos²s ds = t/2 + sin(2t)/4.
fn shm_thermal_u_errors(dt_base: f64, n_blocks: usize, rungs: &[u32]) -> Vec<f64> {
    let n = rungs.len();
    let mut s = state_of(
        vec![DVec3::new(1.0, 0.0, 0.0); n],
        vec![DVec3::ZERO; n],
        vec![0.0; n],
    );
    let mut solver = ShmThermal { omega2: 1.0 };
    let mut stepper = ActiveSetKdkThermal::new();
    let bg: &dyn Background = &StaticBackground;

    let mut max_err = vec![0.0_f64; n];
    for b in 1..=n_blocks {
        stepper.step_block(&mut s, &mut solver, bg, dt_base, rungs);
        let t = b as f64 * dt_base;
        let truth_u = 0.5 * t + 0.25 * (2.0 * t).sin();
        for (i, e) in max_err.iter_mut().enumerate() {
            *e = e.max((s.u[i] - truth_u).abs());
        }
    }
    max_err
}

#[test]
fn finer_rung_tracks_the_analytic_u_more_closely() {
    // Particle A rung 0 (eff dt = dt_base), particle B rung 2 (eff dt = dt_base/4).
    // Both integrate the same u-source; the finer rung samples the varying dudt
    // more often and sits much closer to the analytic u(t).
    let errs = shm_thermal_u_errors(0.2, 10, &[0, 2]);
    let (e_a, e_b) = (errs[0], errs[1]);
    assert!(
        e_a > 0.0 && e_b > 0.0,
        "both must have nonzero finite u-error (else dudt was constant/vacuous)"
    );
    assert!(
        e_b < 0.25 * e_a,
        "finer rung must track u closer: e_B = {e_b:e} !< 0.25·e_A = {:e}",
        0.25 * e_a
    );
}

#[test]
fn coarse_rung_u_error_falls_second_order_under_base_dt_refinement() {
    // The rung-0 particle integrates its u as fixed-dt KDK-thermal at dt_base
    // (its u is kicked only at its own boundaries; dudt is sampled at the two
    // force evals straddling the drift). Halving dt_base must cut its u-error ~4×.
    let coarse = shm_thermal_u_errors(0.2, 10, &[0, 2])[0];
    let fine = shm_thermal_u_errors(0.1, 20, &[0, 2])[0]; // same horizon T = 2.0
    assert!(
        fine < coarse,
        "refinement must reduce u-error: {fine:e} !< {coarse:e}"
    );
    let ratio = coarse / fine;
    assert!(
        ratio > 3.0,
        "coarse-rung u-error must fall ~2nd order (ratio ≈ 4): got {ratio:.2}"
    );
}
