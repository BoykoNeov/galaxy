//! I3 — the active-set KDK stepper + drift predictor (ISOTHERMAL first).
//!
//! The individual-timestep mechanic: advance one base block by sub-cycling a
//! power-of-two rung hierarchy, kicking only the ACTIVE subset each fine tick and
//! drifting all particles at the fine cadence. This is the piece that does not fit
//! `Integrator::step(dt)` (per-particle rungs, active mask), so it is a distinct
//! type — NOT a branch on `LeapfrogKdk`. The `run_individual` DRIVER (block loop,
//! rung re-assignment, limiter, momentum diagnostic) is I4 and deliberately absent
//! here.
//!
//! GATE DESIGN (advisor-vetted, 2026-07-09 — supersedes the plan's I3 "reduces to
//! global-adaptive to tolerance" wording, which was the weaker/wronger framing):
//!   * COLLAPSED → BIT-IDENTICAL. When every particle lands on rung 0 the active
//!     set is everyone every tick, the predictor never fires, and the sub-step IS
//!     `dt_base`: same solver call, same KDK order ⇒ bit-for-bit equal to
//!     `LeapfrogKdk` stepped once at `dt_base`. Compared integrator-vs-integrator
//!     (NOT vs `run_adaptive`, whose growth limiter diverges the dt sequence for
//!     reasons unrelated to this mechanic). This is the load-bearing pin.
//!   * PREDICTOR is drift-only and EXACT for KDK — acceleration enters only via
//!     kicks, so between an inactive particle's kicks its velocity is constant and
//!     `x + v·Δt` is the true position, not an approximation. Adding `½a·Δt²` would
//!     double-count acceleration (wrong for KDK). Pinned by a hand-value unit test.
//!   * MULTI-RUNG is a genuinely DIFFERENT, correct integrator — it converges to
//!     the TRUE solution as rungs refine, NOT to the global-fixed-dt answer, so it
//!     is NOT bit-compared to anything. Gated two ways: (a) exact under constant
//!     acceleration (leapfrog is exact there for any step ⇒ pins the open/interior/
//!     close kick bookkeeping to roundoff), (b) convergence to the analytic
//!     oscillator (finer rung tracks closer; coarse-rung error falls ~2nd order
//!     under base-dt refinement). Momentum bounded-drift is an I4 (driver) gate.

use galaxy_core::{DVec3, ForceSolver, Integrator, LeapfrogKdk, Species, State, StaticBackground};
use galaxy_sim::individual::{predict_pos, ActiveSetKdk};
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

// --------------------------------------------------------------------------
// Mock solvers: analytic force fields with hand-derivable exact trajectories.
// --------------------------------------------------------------------------

/// A uniform, constant acceleration on every particle. Leapfrog integrates a
/// constant field EXACTLY at any step, so the block-end state is closed-form —
/// the sharpest pin on the kick bookkeeping (a mis-applied interior full-kick or
/// half-vs-full slip shows up immediately on the fine-rung particle).
struct ConstAccel(DVec3);
impl ForceSolver for ConstAccel {
    fn accelerations(&mut self, _state: &State, acc: &mut [DVec3]) {
        for a in acc.iter_mut() {
            *a = self.0;
        }
    }
    fn potential_energy(&self, _state: &State) -> f64 {
        0.0
    }
}

/// Independent simple-harmonic oscillators: `a_i = −ω²·x_i` per particle, no
/// coupling. Analytic solution `x(t) = x₀cos(ωt) + (v₀/ω)sin(ωt)`, so the
/// per-particle error vs truth is measurable — the convergence testbed.
struct Shm {
    omega2: f64,
}
impl ForceSolver for Shm {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        for (a, x) in acc.iter_mut().zip(&state.pos) {
            *a = *x * -self.omega2;
        }
    }
    fn potential_energy(&self, _state: &State) -> f64 {
        0.0
    }
}

fn state_of(pos: Vec<DVec3>, vel: Vec<DVec3>) -> State {
    let n = pos.len();
    State::from_phase_space(pos, vel, vec![1.0; n])
}

// A small gas ball for the real-solver bit-identity gate (isothermal SPH+gravity).
fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn gas_blob(seed: u64, n: usize, radius: f64) -> State {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    // A mild swirl so velocities are non-trivial (exercises the drift arithmetic).
    let vel: Vec<DVec3> = pos
        .iter()
        .map(|&p| DVec3::new(-p.y, p.x, 0.0) * 0.3)
        .collect();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    s
}

fn hydro_solver(c_s: f64) -> GravitySph<BarnesHut> {
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };
    GravitySph::<BarnesHut>::hydro_only(params, DensityConfig::default())
}

// --------------------------------------------------------------------------
// PREDICTOR — drift-only, exact.
// --------------------------------------------------------------------------

#[test]
fn predictor_is_exact_drift_only() {
    // Hand values (NOT `x + v·dt` recomputed in the test — that would pass a wrong
    // impl that also happens to add v·dt). Drift-only ⇒ NO acceleration term: the
    // signature has no `a`, so `½a·Δt²` is structurally impossible, and these pins
    // confirm the linear extrapolation.
    let got = predict_pos(DVec3::new(1.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0), 0.5);
    assert_eq!(got, DVec3::new(2.0, 0.0, 0.0), "1 + 2·0.5 = 2");

    let got = predict_pos(
        DVec3::new(-3.0, 4.0, 10.0),
        DVec3::new(1.0, -2.0, 0.0),
        0.25,
    );
    assert_eq!(
        got,
        DVec3::new(-2.75, 3.5, 10.0),
        "componentwise x + v·0.25"
    );

    // Δt = 0 is the identity (a synced particle predicts to itself).
    let x = DVec3::new(7.0, -1.5, 2.0);
    assert_eq!(predict_pos(x, DVec3::new(9.0, 9.0, 9.0), 0.0), x);
}

// --------------------------------------------------------------------------
// COLLAPSED → BIT-IDENTICAL to LeapfrogKdk (the load-bearing pin).
// --------------------------------------------------------------------------

#[test]
fn collapsed_rungs_are_bit_identical_to_leapfrog_kdk() {
    let dt_base = 0.01;
    let n = 120;

    // Reference: whole-state KDK at dt_base.
    let mut s_ref = gas_blob(0xB17B17, n, 1.0);
    let mut solver_ref = hydro_solver(1.0);
    let mut kdk = LeapfrogKdk::new();
    let bg = StaticBackground;

    // Under test: active-set stepper with EVERY particle on rung 0 ⇒ one fine tick,
    // active set = all, sub-step = dt_base. Must reproduce KDK bit-for-bit.
    let mut s_ind = gas_blob(0xB17B17, n, 1.0);
    let mut solver_ind = hydro_solver(1.0);
    let mut stepper = ActiveSetKdk::new();
    let rungs = vec![0u32; n];

    // Multiple blocks: also pins the cross-block cached-acc reuse (both reuse the
    // closing-kick acceleration as the next opening kick — no re-prime).
    for block in 0..4 {
        kdk.step(&mut s_ref, &mut solver_ref, &bg, dt_base);
        stepper.step_block(&mut s_ind, &mut solver_ind, &bg, dt_base, &rungs);

        assert_eq!(
            s_ind.pos, s_ref.pos,
            "block {block}: positions must match LeapfrogKdk bit-for-bit"
        );
        assert_eq!(
            s_ind.vel, s_ref.vel,
            "block {block}: velocities must match LeapfrogKdk bit-for-bit"
        );
        assert_eq!(s_ind.time, s_ref.time, "block {block}: time must match");
    }
}

// --------------------------------------------------------------------------
// MULTI-RUNG (a) — exact under constant acceleration.
// --------------------------------------------------------------------------

#[test]
fn two_rung_block_is_exact_under_constant_acceleration() {
    // Particle A on rung 0 (full base step), particle B on rung 2 (four fine
    // sub-steps). Constant accel ⇒ leapfrog is exact at any step, so BOTH land on
    // the closed form x₀ + v₀·D + ½g·D² regardless of rung. A wrong interior
    // full-kick (half instead of full, stale force, or missed tick) throws B off
    // this analytic mark.
    let g = DVec3::new(0.3, -0.2, 0.1);
    let dt_base = 0.2;
    let x0 = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(5.0, 1.0, -2.0)];
    let v0 = [DVec3::new(1.0, 0.0, 0.5), DVec3::new(-0.4, 0.7, 0.0)];

    let mut s = state_of(x0.to_vec(), v0.to_vec());
    let mut solver = ConstAccel(g);
    let mut stepper = ActiveSetKdk::new();
    let bg = StaticBackground;
    let rungs = vec![0u32, 2u32];

    stepper.step_block(&mut s, &mut solver, &bg, dt_base, &rungs);

    for i in 0..2 {
        let want_x = x0[i] + v0[i] * dt_base + g * (0.5 * dt_base * dt_base);
        let want_v = v0[i] + g * dt_base;
        assert!(
            (s.pos[i] - want_x).length() < 1e-12,
            "particle {i} (rung {}) position {:?} != analytic {:?}",
            rungs[i],
            s.pos[i],
            want_x
        );
        assert!(
            (s.vel[i] - want_v).length() < 1e-12,
            "particle {i} (rung {}) velocity {:?} != analytic {:?}",
            rungs[i],
            s.vel[i],
            want_v
        );
    }
    assert!(
        (s.time - dt_base).abs() < 1e-15,
        "block advances time by dt_base"
    );
}

// --------------------------------------------------------------------------
// MULTI-RUNG (b) — convergence to the true oscillator.
// --------------------------------------------------------------------------

/// Run `n_blocks` base blocks of an SHM system with the given per-particle rungs,
/// returning each particle's max position error vs the analytic oscillator over
/// the block boundaries (where all rungs are synchronized).
fn shm_run_errors(omega2: f64, dt_base: f64, n_blocks: usize, rungs: &[u32]) -> Vec<f64> {
    let n = rungs.len();
    // All particles: same IC x=(1,0,0), v=0 ⇒ x(t) = cos(ωt) on the x-axis.
    let mut s = state_of(vec![DVec3::new(1.0, 0.0, 0.0); n], vec![DVec3::ZERO; n]);
    let mut solver = Shm { omega2 };
    let mut stepper = ActiveSetKdk::new();
    let bg = StaticBackground;
    let omega = omega2.sqrt();

    let mut max_err = vec![0.0_f64; n];
    for b in 1..=n_blocks {
        stepper.step_block(&mut s, &mut solver, &bg, dt_base, rungs);
        let t = b as f64 * dt_base;
        let truth = DVec3::new((omega * t).cos(), 0.0, 0.0);
        for (i, e) in max_err.iter_mut().enumerate() {
            *e = e.max((s.pos[i] - truth).length());
        }
    }
    max_err
}

#[test]
fn finer_rung_tracks_the_true_oscillator_more_closely() {
    // Particle A rung 0 (eff dt = dt_base), particle B rung 2 (eff dt = dt_base/4).
    // Both are correct integrators of the SAME oscillator; the finer rung sits much
    // closer to truth (leapfrog error ∝ dt² ⇒ B's bound ≈ A's/16).
    let errs = shm_run_errors(1.0, 0.2, 10, &[0, 2]);
    let (e_a, e_b) = (errs[0], errs[1]);
    assert!(
        e_a > 0.0 && e_b > 0.0,
        "both must have nonzero finite error"
    );
    assert!(
        e_b < 0.25 * e_a,
        "finer rung must track closer: e_B = {e_b:e} !< 0.25·e_A = {:e}",
        0.25 * e_a
    );
}

#[test]
fn coarse_rung_error_falls_second_order_under_base_dt_refinement() {
    // A coarse (rung-0) particle sharing a mixed block with a rung-2 particle still
    // integrates as fixed-dt leapfrog at dt_base (its sub-drifts sum to one drift,
    // velocity constant between its kicks). Halving dt_base must cut its error ~4×
    // (2nd order) — the "converges to truth as rungs refine" property, measured on
    // the coarse particle where the base step is the whole error.
    let coarse = shm_run_errors(1.0, 0.2, 10, &[0, 2])[0];
    let fine = shm_run_errors(1.0, 0.1, 20, &[0, 2])[0]; // same horizon T = 2.0
    assert!(
        fine < coarse,
        "refinement must reduce error: {fine:e} !< {coarse:e}"
    );
    let ratio = coarse / fine;
    assert!(
        ratio > 3.0,
        "coarse-rung error must fall ~2nd order (ratio ≈ 4): got {ratio:.2}"
    );
}
