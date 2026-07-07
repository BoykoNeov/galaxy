//! Adaptive-dt engine gates (plan: courant-quickening-cadence.md, milestone A2).
//!
//! The global block-adaptive loop holds `dt` fixed across a block of ≤ `block_steps`
//! steps and recomputes it at each block boundary from the solver's CFL limit,
//! emitting snapshots on a TIME cadence (`output_dt`).
//!
//! GATE INTENT (D2/D4) — the adaptive path deliberately forfeits leapfrog
//! time-reversibility and energy-oscillation (variable dt is not symplectic), so it
//! is NOT gated by the fixed-dt invariant tests, and there is NO energy gate on it
//! (isothermal SPH is an implicit heat bath — DESIGN.md 1582–1583 — so total energy
//! legitimately changes even at fixed dt; there is no flat baseline to measure
//! spurious drift against). A future session must NOT add an energy-conservation or
//! reversibility gate here. The correct gates are:
//!   * PRIMARY — full-DURATION convergence to a fine-dt reference as courant → 0
//!     (asserted as monotone error decrease + a generous absolute cap, NOT a numeric
//!     order factor: variable-dt leapfrog is between 1st and 2nd order). The testbed
//!     COMPRESSES so the CFL bound actually moves (else it tests fixed-dt in disguise).
//!   * SECOND — contraction staleness (D2b): the realized block dt never exceeds the
//!     end-of-block CFL limit over a converging flow.
//!   * TRIPWIRE — momentum conservation: a cheap regression guard, conserved by
//!     construction for a global scheme, NOT billed as a correctness gate.

use galaxy_core::{
    diagnostics, DVec3, ForceSolver, Integrator, LeapfrogKdk, Species, State, StaticBackground,
};
use galaxy_io::Header;
use galaxy_sim::{plan_block, run_adaptive, AdaptiveConfig, SimError, SnapshotSink};
use galaxy_solvers::sph::{DensityConfig, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

/// In-memory sink keeping a full f64 copy of every snapshot (conservation is judged
/// on the simulated state, not the lossy f32 on-disk mass).
#[derive(Default)]
struct CollectingSink {
    snaps: Vec<(Header, State)>,
}
impl SnapshotSink for CollectingSink {
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError> {
        self.snaps.push((header.clone(), state.clone()));
        Ok(())
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

/// A gas ball with a radially CONVERGING velocity field `v = −k·x`, so it compresses
/// and the CFL bound (∝ h/v_sig) genuinely moves across the run — the testbed the
/// convergence + staleness gates require.
fn converging_gas_blob(seed: u64, n: usize, radius: f64, k: f64) -> State {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    let vel: Vec<DVec3> = pos.iter().map(|&p| -k * p).collect();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    s
}

fn hydro_solver(c_s: f64) -> GravitySph<BarnesHut> {
    let params = HydroParams {
        sound_speed: c_s,
        ..HydroParams::default()
    };
    GravitySph::<BarnesHut>::hydro_only(params, DensityConfig::default())
}

fn cfg(courant: f64, block_steps: u64, output_dt: f64, n_outputs: u64) -> AdaptiveConfig {
    AdaptiveConfig {
        courant,
        max_growth: 1.25,
        block_steps,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0xADA9_7175,
        config_hash: 0xC0FFEE,
        units: "nbody-G1".to_string(),
    }
}

// --------------------------------------------------------------------------
// plan_block: the pure block-sizing decision (Courant × growth cap, land-exact).
// --------------------------------------------------------------------------

#[test]
fn plan_block_landing_block_lands_and_stays_within_target() {
    let c = cfg(0.25, 16, 0.05, 1);
    // Interval fits: remaining ≤ dt_target · block_steps ⇒ a landing block.
    let p = plan_block(1.0, 0.25, 0.1, &c);
    assert!(p.lands, "a fitting interval must be a landing block");
    assert!(p.n_steps >= 1);
    // Lands exactly: n·dt == remaining.
    assert!(
        (p.dt * p.n_steps as f64 - 0.1).abs() < 1e-15,
        "must land on remaining"
    );
    // Never steps past the CFL/growth target.
    assert!(
        p.dt <= p.dt_target + 1e-15,
        "landing dt must stay ≤ dt_target"
    );
}

#[test]
fn plan_block_full_block_when_interval_does_not_fit() {
    let c = cfg(0.25, 16, 0.05, 1);
    // dt_target = 0.25, block_steps = 16 ⇒ fits ≤ 4.0; a 10.0 remaining does not.
    let p = plan_block(1.0, 0.25, 10.0, &c);
    assert!(!p.lands, "an over-long interval must NOT land this block");
    assert_eq!(p.n_steps, 16, "a non-fitting interval runs a full block");
    assert!((p.dt - 0.25).abs() < 1e-15, "full block runs at dt_target");
}

#[test]
fn plan_block_growth_cap_and_instant_shrink() {
    let c = cfg(0.25, 16, 0.05, 1);
    // Growth capped: cfl = 0.25 but prev_target 0.1 ⇒ dt_target ≤ 1.25·0.1 = 0.125.
    let grow = plan_block(1.0, 0.1, 10.0, &c);
    assert!(
        (grow.dt_target - 0.125).abs() < 1e-15,
        "growth cap: dt_target = {} want 0.125",
        grow.dt_target
    );
    // Instant shrink: the bound tightened (limit 0.1) ⇒ dt_target = 0.25·0.1 = 0.025,
    // NOT capped up toward the large prev_target.
    let shrink = plan_block(0.1, 1.0, 10.0, &c);
    assert!(
        (shrink.dt_target - 0.025).abs() < 1e-15,
        "instant shrink: dt_target = {} want 0.025",
        shrink.dt_target
    );
}

// --------------------------------------------------------------------------
// PRIMARY gate: full-duration convergence to a fine-dt reference (D2 i).
// --------------------------------------------------------------------------

#[test]
fn adaptive_converges_to_fine_reference_as_courant_shrinks() {
    let ic = converging_gas_blob(0xB10B, 350, 1.0, 0.6);
    let c_s = 1.0;
    let bg = StaticBackground;
    let output_dt = 0.05;
    let n_outputs = 8; // full horizon T = 0.4 — full-duration, not a prefix.

    // Run the adaptive loop at a given Courant number, return the final gas positions.
    let run_at = |courant: f64| -> Vec<DVec3> {
        let mut s = ic.clone();
        let mut solver = hydro_solver(c_s);
        let mut integ = LeapfrogKdk::new();
        let mut sink = CollectingSink::default();
        run_adaptive(
            &mut s,
            &mut solver,
            &mut integ,
            &bg,
            &cfg(courant, 16, output_dt, n_outputs),
            &mut sink,
        )
        .unwrap();
        sink.snaps.last().unwrap().1.pos.clone()
    };

    let reference = run_at(0.02); // fine ≈ fixed-tiny-dt truth
    let err = |a: &[DVec3]| {
        a.iter()
            .zip(&reference)
            .map(|(p, r)| (*p - *r).length())
            .fold(0.0_f64, f64::max)
    };

    let e_coarse = err(&run_at(0.2));
    let e_fine = err(&run_at(0.1));

    // Monotone decrease as courant halves (NOT a numeric order factor — variable-dt
    // leapfrog is between 1st and 2nd order), plus a generous absolute cap.
    assert!(
        e_fine < e_coarse,
        "halving courant must reduce the error toward the reference: \
         err(0.1) = {e_fine:e} !< err(0.2) = {e_coarse:e}"
    );
    assert!(
        e_coarse < 0.1,
        "even the coarse adaptive run must track the reference within a blob radius: \
         err(0.2) = {e_coarse:e}"
    );
}

// --------------------------------------------------------------------------
// SECOND gate: contraction staleness (D2b) — realized dt ≤ end-of-block bound.
// --------------------------------------------------------------------------

#[test]
fn block_dt_never_exceeds_end_of_block_cfl_limit_under_contraction() {
    let mut s = converging_gas_blob(0xC0117AC7, 400, 1.0, 1.0); // strong compression
    let c_s = 1.0;
    let mut solver = hydro_solver(c_s);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let c = cfg(0.25, 8, 0.05, 1); // block_steps = 8, courant = 0.25

    let t_end = c.output_dt * c.n_outputs as f64;
    let mut t = 0.0;
    // Seed the growth memory with the initial CFL target (first block uncapped).
    let mut prev_target = c.courant * solver.max_stable_dt(&s);
    assert!(prev_target.is_finite() && prev_target > 0.0);

    while t < t_end - 1e-12 * t_end.max(c.output_dt) {
        let limit = solver.max_stable_dt(&s);
        let plan = plan_block(limit, prev_target, t_end - t, &c);
        for _ in 0..plan.n_steps {
            integ.step(&mut s, &mut solver, &bg, plan.dt);
        }
        t += plan.dt * plan.n_steps as f64;
        prev_target = plan.dt_target;

        // The bound tightens WITHIN the block under compression; the realized dt must
        // still sit under the end-of-block CFL limit (else the block over-stepped a
        // bound it never re-checked — the D2b instability).
        let limit_end = solver.max_stable_dt(&s);
        assert!(
            plan.dt <= limit_end * (1.0 + 1e-9),
            "block dt {} exceeded the end-of-block CFL limit {} — contraction staleness",
            plan.dt,
            limit_end
        );
    }
}

// --------------------------------------------------------------------------
// TRIPWIRE: momentum conservation (cheap regression guard, NOT a correctness gate).
// --------------------------------------------------------------------------

#[test]
fn momentum_is_conserved_across_the_adaptive_run() {
    let ic = converging_gas_blob(0x3110, 300, 1.0, 0.5);
    let mut s = ic.clone();
    let mut solver = hydro_solver(1.0);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let mut sink = CollectingSink::default();
    run_adaptive(
        &mut s,
        &mut solver,
        &mut integ,
        &bg,
        &cfg(0.25, 16, 0.05, 6),
        &mut sink,
    )
    .unwrap();

    let p0 = diagnostics::total_momentum(&ic);
    let max_drift = sink
        .snaps
        .iter()
        .map(|(_, st)| (diagnostics::total_momentum(st) - p0).length())
        .fold(0.0_f64, f64::max);
    assert!(
        max_drift < 1e-9,
        "momentum drift {max_drift:e} (pairwise-antisym floor)"
    );
}

// --------------------------------------------------------------------------
// Cadence + headers (D3): snapshots land on the TIME grid, header step = output idx.
// --------------------------------------------------------------------------

#[test]
fn snapshots_land_on_the_output_time_grid() {
    let mut s = converging_gas_blob(0xCADE, 250, 1.0, 0.5);
    let mut solver = hydro_solver(1.0);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let c = cfg(0.25, 16, 0.05, 5);

    let mut sink = CollectingSink::default();
    let summary = run_adaptive(&mut s, &mut solver, &mut integ, &bg, &c, &mut sink).unwrap();

    // IC (index 0) + one snapshot per output interval.
    let steps: Vec<u64> = sink.snaps.iter().map(|(h, _)| h.step).collect();
    assert_eq!(
        steps,
        vec![0, 1, 2, 3, 4, 5],
        "output indices wrong: {steps:?}"
    );
    assert_eq!(summary.snapshots_emitted, 6);
    for (h, st) in &sink.snaps {
        let want = h.step as f64 * c.output_dt;
        assert!(
            (h.time - want).abs() < 1e-12,
            "header time {} != output index {} · output_dt",
            h.time,
            h.step
        );
        assert_eq!(h.time, st.time, "header/state time disagree");
        assert_eq!(h.softening, c.softening);
    }
    assert!((summary.final_time - 5.0 * c.output_dt).abs() < 1e-12);
}

// --------------------------------------------------------------------------
// Config validation: gas-free (no finite bound) and degenerate schedule reject.
// --------------------------------------------------------------------------

#[test]
fn rejects_gas_free_state_and_invalid_schedule() {
    // Gas-free: max_stable_dt = +∞, no finite adaptive target ⇒ Config error.
    let mut rng = lcg(7);
    let pos: Vec<DVec3> = (0..100)
        .map(|_| DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 2.0)
        .collect();
    let mut gas_free = State::from_phase_space(pos, vec![DVec3::ZERO; 100], vec![1.0; 100]);
    let mut solver = hydro_solver(1.0);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let mut sink = CollectingSink::default();
    assert!(matches!(
        run_adaptive(
            &mut gas_free,
            &mut solver,
            &mut integ,
            &bg,
            &cfg(0.25, 16, 0.05, 4),
            &mut sink
        ),
        Err(SimError::Config(_))
    ));

    // Degenerate schedule: output_dt = 0.
    let mut s = converging_gas_blob(1, 100, 1.0, 0.5);
    let mut solver = hydro_solver(1.0);
    let mut sink = CollectingSink::default();
    assert!(matches!(
        run_adaptive(
            &mut s,
            &mut solver,
            &mut integ,
            &bg,
            &cfg(0.25, 16, 0.0, 4),
            &mut sink
        ),
        Err(SimError::Config(_))
    ));
}
