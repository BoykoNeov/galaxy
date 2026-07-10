//! I4a — the `run_individual` driver (`hydro-only` mode, ISOTHERMAL first).
//!
//! The individual-timestep DRIVER: the block-over-block loop that re-derives
//! `dt_base` + the per-particle rungs (I1 CFL vector → I2 rung policy) at each base
//! boundary and sub-cycles the block with the I3 `ActiveSetKdk` mechanic, emitting
//! snapshots on a TIME cadence. This is the third stepping path, beside fixed-dt
//! `run` and global-adaptive `run_adaptive` — neither of those byte-paths is touched.
//!
//! GATE INTENT (advisor-vetted, 2026-07-09) — the load-bearing risk in I4a is that
//! BOTH gates can pass while testing nothing: active-subset kicking only differs from
//! a full kick when MULTIPLE rungs are present, so a uniformly-dense testbed (one
//! rung ⇒ everyone active every tick) reduces to fixed-dt `LeapfrogKdk` (already
//! bit-pinned in I3) and makes the gates vacuous. Every gate here therefore runs on a
//! CENTRALLY-CONCENTRATED core+halo IC (a density gradient ⇒ an h gradient ⇒ a real
//! dt spread) and SELF-CHECKS, on the driver's actual behaviour, that the run spanned
//! ≥ 3 distinct rungs and that the finest rung stayed `< r_max` (so the reference is
//! not itself under-resolved). The gates:
//!   * PRIMARY — full-DURATION convergence to a fine-courant reference as courant ↓
//!     (monotone error decrease + a generous absolute cap, NOT a numeric order
//!     factor: variable-dt leapfrog is between 1st and 2nd order). Self-reference at
//!     fine courant is the reference, matching the sibling adaptive gate; the I3
//!     gates already pin "converges to TRUTH" at the mechanic level.
//!   * DIAGNOSTIC — momentum BOUNDED DRIFT: kick-active-only forfeits exact
//!     conservation (the equal-and-opposite reaction on an inactive neighbour is
//!     deferred), so momentum drifts; the drift is ∝ courant and must SHRINK as
//!     courant → 0. NOT a roundoff tripwire (that is the global path's gate, where
//!     one dt kicks all particles ⇒ Σmᵢaᵢ = 0 exactly).
//!
//! There is NO energy gate (isothermal heat bath + variable per-particle dt — D4/D2
//! carry over and worsen); convergence subsumes it. The Saitoh–Makino limiter +
//! shock-wakeup gate are I4b; the thermal `u`-kick arm is I5.

use galaxy_core::{diagnostics, DVec3, Species, State, StaticBackground};
use galaxy_io::Header;
use galaxy_sim::{
    run_individual, IndividualConfig, IndividualSummary, SimError, SnapshotSink, ThermalArm,
};
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

/// In-memory sink keeping a full f64 copy of every snapshot.
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

/// Sample `n` points uniformly inside a ball of the given radius (rejection).
fn ball(rng: &mut impl FnMut() -> f64, n: usize, radius: f64) -> Vec<DVec3> {
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    pos
}

/// A CENTRALLY-CONCENTRATED gas cloud: a dense core (`n_core` particles in a tiny
/// radius) inside a diffuse halo (`n_halo` particles in a large radius). The steep
/// density contrast gives a steep `h` contrast ⇒ a steep per-particle `dt = c·h/v_sig`
/// contrast ⇒ a genuine SPREAD of power-of-two rungs — the testbed that makes the
/// active-subset machinery actually engage (a uniform blob is one rung = fixed-dt in
/// disguise). Zero initial velocity: SPH pressure alone drives the (converging then
/// expanding) dynamics the convergence gate measures. Pure gas.
fn core_halo_cloud(seed: u64) -> State {
    let mut rng = lcg(seed);
    let mut pos = ball(&mut rng, 500, 0.1); // dense core
    pos.extend(ball(&mut rng, 100, 1.0)); // diffuse halo
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vec![DVec3::ZERO; n], vec![1.0; n]);
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

/// Config with the base-dt cap set NON-BINDING (`f64::INFINITY`) and `output_dt`
/// large enough that a full base block fits inside an output interval even at the
/// coarsest courant — so `dt_base = courant·dt_coarsest` and the rung STRUCTURE is
/// courant-invariant (the property the convergence gate leans on: halving courant
/// halves every particle's step uniformly, same rungs, so the three runs are
/// comparable and self-reference at fine courant is valid).
fn cfg(courant: f64, output_dt: f64, n_outputs: u64) -> IndividualConfig {
    IndividualConfig {
        courant,
        dt_base_cap: f64::INFINITY,
        r_max: 10,
        n_limit: 10,             // == r_max ⇒ limiter non-binding (I4a is pure CFL rungs)
        subcycle_gravity: false, // hydro-only (I4a gates)
        grav_eta: 0.3,
        eos: ThermalArm::Isothermal, // I4a driver gates are the isothermal byte-path
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0x1AD_DE4,
        config_hash: 0xC0FFEE,
        units: "nbody-G1".to_string(),
    }
}

// --------------------------------------------------------------------------
// PRIMARY gate: full-duration convergence to a fine-courant reference, on a
// genuinely multi-rung testbed.
// --------------------------------------------------------------------------

#[test]
fn individual_converges_to_fine_reference_as_courant_shrinks() {
    let ic = core_halo_cloud(0xC0FFEE);
    let c_s = 1.0;
    let bg = StaticBackground;
    let output_dt = 0.15;
    let n_outputs = 2; // full horizon T = 0.3.

    // Run the individual loop at a given courant, returning the final gas positions
    // and the rung diagnostics (so the gate can prove the run was multi-rung).
    let run_at = |courant: f64| -> (Vec<DVec3>, IndividualSummary) {
        let mut s = ic.clone();
        let mut solver = hydro_solver(c_s);
        let mut sink = CollectingSink::default();
        let summary = run_individual(
            &mut s,
            &mut solver,
            &bg,
            &cfg(courant, output_dt, n_outputs),
            &mut sink,
        )
        .unwrap();
        (sink.snaps.last().unwrap().1.pos.clone(), summary)
    };

    let (reference, ref_summary) = run_at(0.02); // fine ≈ fixed-tiny-dt truth

    // SELF-CHECK (load-bearing): the run must genuinely exercise the multi-rung
    // machinery, else convergence is a vacuous fixed-dt test in disguise. The rung
    // structure is courant-invariant here (non-binding cap), so the reference's
    // diagnostics stand for all three runs.
    assert!(
        ref_summary.distinct_rungs >= 3,
        "testbed must span ≥3 distinct rungs (got {}) — else this is fixed-dt in disguise",
        ref_summary.distinct_rungs
    );
    assert!(
        ref_summary.max_rung < 10,
        "finest rung {} must stay < r_max=10 — the reference must not be under-resolved (clamped)",
        ref_summary.max_rung
    );

    let err = |a: &[DVec3]| {
        a.iter()
            .zip(&reference)
            .map(|(p, r)| (*p - *r).length())
            .fold(0.0_f64, f64::max)
    };

    let e_coarse = err(&run_at(0.2).0);
    let e_fine = err(&run_at(0.1).0);

    // Monotone decrease as courant halves (NOT an order factor — variable-dt leapfrog
    // is between 1st and 2nd order), plus a generous absolute cap (a blob radius).
    assert!(
        e_fine < e_coarse,
        "halving courant must reduce the error toward the reference: \
         err(0.1) = {e_fine:e} !< err(0.2) = {e_coarse:e}"
    );
    assert!(
        e_coarse < 0.1,
        "even the coarse individual run must track the reference within a blob radius: \
         err(0.2) = {e_coarse:e}"
    );
}

// --------------------------------------------------------------------------
// DIAGNOSTIC gate: momentum bounded drift — kick-active-only drifts, ∝ courant.
// --------------------------------------------------------------------------

#[test]
fn momentum_drift_is_bounded_and_shrinks_with_courant() {
    let ic = core_halo_cloud(0x3110);
    let c_s = 1.0;
    let bg = StaticBackground;
    let output_dt = 0.15;
    let n_outputs = 2;

    // Max |Σ mᵢvᵢ − p₀| over the run's snapshots at a given courant. p₀ = 0 (zero
    // initial velocity), so this is the net momentum manufactured by deferring the
    // reaction on inactive neighbours.
    let drift_at = |courant: f64| -> f64 {
        let mut s = ic.clone();
        let mut solver = hydro_solver(c_s);
        let mut sink = CollectingSink::default();
        let summary = run_individual(
            &mut s,
            &mut solver,
            &bg,
            &cfg(courant, output_dt, n_outputs),
            &mut sink,
        )
        .unwrap();
        assert!(
            summary.distinct_rungs >= 3,
            "momentum drift is only meaningful on a multi-rung run (got {} rungs)",
            summary.distinct_rungs
        );
        let p0 = diagnostics::total_momentum(&ic);
        sink.snaps
            .iter()
            .map(|(_, st)| (diagnostics::total_momentum(st) - p0).length())
            .fold(0.0_f64, f64::max)
    };

    // A gross momentum SCALE from the fine run, to size the "generous cap" relative
    // to the actual dynamics (net drift must be a small fraction of the gross flux).
    let gross_scale = {
        let mut s = ic.clone();
        let mut solver = hydro_solver(c_s);
        let mut sink = CollectingSink::default();
        run_individual(
            &mut s,
            &mut solver,
            &bg,
            &cfg(0.1, output_dt, n_outputs),
            &mut sink,
        )
        .unwrap();
        let (_, last) = sink.snaps.last().unwrap();
        last.mass
            .iter()
            .zip(&last.vel)
            .map(|(m, v)| m * v.length())
            .sum::<f64>()
    };

    let d_coarse = drift_at(0.2);
    let d_fine = drift_at(0.05); // 4× finer ⇒ ≈4× smaller drift (drift ∝ courant).

    // The drift must SHRINK as courant → 0 (the theoretical ∝ courant behaviour) —
    // the real signal that it is bounded and controlled, not a roundoff tripwire.
    assert!(
        d_fine < d_coarse,
        "momentum drift must shrink as courant → 0 (∝ courant): \
         drift(0.05) = {d_fine:e} !< drift(0.2) = {d_coarse:e}"
    );
    // Generous absolute cap: even the coarse run's net drift stays a small fraction
    // of the gross momentum flux (a genuinely bounded, not runaway, error).
    assert!(
        d_coarse < 0.05 * gross_scale,
        "momentum drift {d_coarse:e} must stay < 5% of the gross flux {:e}",
        0.05 * gross_scale
    );
}

// --------------------------------------------------------------------------
// Cadence + headers (D3): snapshots land on the TIME grid, header step = output idx.
// --------------------------------------------------------------------------

#[test]
fn snapshots_land_on_the_output_time_grid() {
    let mut s = core_halo_cloud(0xCADE);
    let mut solver = hydro_solver(1.0);
    let bg = StaticBackground;
    let c = cfg(0.2, 0.15, 3);

    let mut sink = CollectingSink::default();
    let summary = run_individual(&mut s, &mut solver, &bg, &c, &mut sink).unwrap();

    // IC (index 0) + one snapshot per output interval.
    let steps: Vec<u64> = sink.snaps.iter().map(|(h, _)| h.step).collect();
    assert_eq!(steps, vec![0, 1, 2, 3], "output indices wrong: {steps:?}");
    assert_eq!(summary.run.snapshots_emitted, 4);
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
    assert!((summary.run.final_time - 3.0 * c.output_dt).abs() < 1e-12);
}

// --------------------------------------------------------------------------
// Config validation: gas-free (no finite bound) and degenerate schedule reject.
// --------------------------------------------------------------------------

#[test]
fn rejects_gas_free_state_and_invalid_config() {
    let bg = StaticBackground;

    // Gas-free: every per-particle CFL limit is +∞, no finite base dt ⇒ Config error
    // (Scope: collisionless runs stay on the fixed-dt `run`, not the individual path).
    let mut rng = lcg(7);
    let pos = ball(&mut rng, 100, 1.0);
    let mut gas_free = State::from_phase_space(pos, vec![DVec3::ZERO; 100], vec![1.0; 100]);
    let mut solver = hydro_solver(1.0);
    let mut sink = CollectingSink::default();
    assert!(matches!(
        run_individual(
            &mut gas_free,
            &mut solver,
            &bg,
            &cfg(0.2, 0.15, 2),
            &mut sink
        ),
        Err(SimError::Config(_))
    ));

    // Degenerate schedule: output_dt = 0, courant = 0, r_max = 0, n_outputs = 0 each reject.
    let bad = [
        cfg(0.2, 0.0, 2),  // output_dt = 0
        cfg(0.0, 0.15, 2), // courant = 0
        IndividualConfig {
            r_max: 0,
            ..cfg(0.2, 0.15, 2)
        },
        cfg(0.2, 0.15, 0), // n_outputs = 0
    ];
    for c in &bad {
        let mut s = core_halo_cloud(1);
        let mut solver = hydro_solver(1.0);
        let mut sink = CollectingSink::default();
        assert!(
            matches!(
                run_individual(&mut s, &mut solver, &bg, c, &mut sink),
                Err(SimError::Config(_))
            ),
            "config must reject: {c:?}"
        );
    }
}
