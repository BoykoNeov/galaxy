//! End-to-end engine validation (DESIGN.md M2): run a small two-galaxy collision
//! through the stepping loop and assert the conservation laws plus the snapshot
//! bookkeeping.
//!
//! The conservation half uses the EXACT `DirectSum` oracle (so any drift is the
//! integrator's, not a force approximation) over a genuinely dynamical encounter:
//! total energy is a bounded symplectic oscillation, and — because softened
//! gravity is a central pairwise force — total linear and angular momentum are
//! conserved to roundoff. This is the always-on "conservation" deliverable; the
//! independent-integrator (REBOUND) cross-check is a separate, manually-run
//! harness.
//!
//! The wiring half checks the engine's contract with `galaxy-io`: the IC and the
//! final step are always captured, the cadence is honored, header step/time are
//! correct, and every emitted file is a valid, re-readable snapshot.

use std::path::PathBuf;

use galaxy_core::{diagnostics, LeapfrogKdk, State, StaticBackground};
use galaxy_ic::{Collision, Plummer};
use galaxy_io::Header;
use galaxy_sim::{run, DirectorySink, SimConfig, SimError, SnapshotSink};

/// In-memory sink that keeps a full f64 copy of every snapshot, so conservation is
/// judged on the simulated state, not the lossy f32-stored mass.
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

const G: f64 = 1.0;
const EPS: f64 = 0.05;

fn small_collision() -> (Collision, State) {
    let g1 = Plummer::new(G, 1.0, 1.0);
    let g2 = Plummer::new(G, 0.7, 0.8);
    // Parabolic grazing encounter starting at r0=6 with pericenter 1.5.
    let c = Collision::new(g1, g2, 1.0, 1.5, 6.0);
    let s = c.sample(240, 160, 0xBADC_0DE);
    (c, s)
}

fn config(n_steps: u64, snapshot_every: u64) -> SimConfig {
    SimConfig {
        dt: 0.02,
        n_steps,
        snapshot_every,
        softening: EPS,
        rng_seed: 0xBADC_0DE,
        config_hash: 0x1234,
        units: "nbody-G1".to_string(),
    }
}

#[test]
fn collision_run_conserves_energy_momentum_and_angular_momentum() {
    let (_, mut s) = small_collision();
    let mut solver = galaxy_solvers::DirectSum::new(G, EPS);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = config(500, 50);

    let mut sink = CollectingSink::default();
    let summary = run(&mut s, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();

    assert_eq!(summary.steps, 500);
    assert!(
        summary.snapshots_emitted >= 2,
        "must capture at least IC + final"
    );
    assert!(sink.snaps.len() >= 2);

    // Reference conserved quantities from the IC snapshot.
    let (_, ref s0) = &sink.snaps[0];
    let e0 = diagnostics::total_energy(s0, &solver);
    let p0 = diagnostics::total_momentum(s0);
    let l0 = diagnostics::total_angular_momentum(s0);
    assert!(
        e0 < 0.0,
        "a bound/interacting system should have negative or small E"
    );

    let mut max_e_err = 0.0_f64;
    let mut max_p_err = 0.0_f64;
    let mut max_l_err = 0.0_f64;
    for (_, st) in &sink.snaps {
        let e = diagnostics::total_energy(st, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
        max_p_err = max_p_err.max((diagnostics::total_momentum(st) - p0).length());
        max_l_err = max_l_err.max((diagnostics::total_angular_momentum(st) - l0).length());
        // The barycenter does not wander (zero net momentum, started at origin).
        assert!(
            diagnostics::center_of_mass(st).length() < 1e-6,
            "COM drifted"
        );
    }

    assert!(max_e_err < 1e-3, "energy not conserved: {max_e_err:e}");
    assert!(
        max_p_err < 1e-8,
        "linear momentum not conserved: {max_p_err:e}"
    );
    assert!(
        max_l_err < 1e-8,
        "angular momentum not conserved: {max_l_err:e}"
    );
}

#[test]
fn snapshot_cadence_and_headers_are_correct() {
    let (_, mut s) = small_collision();
    let mut solver = galaxy_solvers::DirectSum::new(G, EPS);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = config(100, 25);

    let mut sink = CollectingSink::default();
    let summary = run(&mut s, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();

    // IC (0) + steps 25,50,75,100. The final step coincides with the cadence, so
    // there should be no duplicate final snapshot.
    let steps: Vec<u64> = sink.snaps.iter().map(|(h, _)| h.step).collect();
    assert_eq!(steps, vec![0, 25, 50, 75, 100], "cadence wrong: {steps:?}");
    assert_eq!(summary.snapshots_emitted, 5);

    // Header time tracks step*dt; metadata is stamped through.
    for (h, st) in &sink.snaps {
        assert!(
            (h.time - h.step as f64 * cfg.dt).abs() < 1e-12,
            "header time {} != step {} * dt",
            h.time,
            h.step
        );
        assert_eq!(h.time, st.time, "header/state time disagree");
        assert_eq!(h.softening, EPS);
        assert_eq!(h.rng_seed, cfg.rng_seed);
        assert_eq!(h.units, cfg.units);
        assert_eq!(h.n_particles, st.len() as u64);
    }
    // final_time is t0 + n_steps*dt.
    assert!((summary.final_time - 100.0 * cfg.dt).abs() < 1e-12);
}

#[test]
fn final_step_is_always_captured_off_cadence() {
    let (_, mut s) = small_collision();
    let mut solver = galaxy_solvers::DirectSum::new(G, EPS);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    // 70 is not a multiple of 30: cadence emits 0,30,60; the final (70) must still
    // be captured exactly once.
    let cfg = config(70, 30);

    let mut sink = CollectingSink::default();
    run(&mut s, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    let steps: Vec<u64> = sink.snaps.iter().map(|(h, _)| h.step).collect();
    assert_eq!(
        steps,
        vec![0, 30, 60, 70],
        "final step not captured: {steps:?}"
    );
}

#[test]
fn directory_sink_writes_valid_readable_snapshots() {
    let (_, mut s) = small_collision();
    let mut solver = galaxy_solvers::DirectSum::new(G, EPS);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = config(40, 20);

    let mut dir: PathBuf = std::env::temp_dir();
    dir.push(format!("galaxy_sim_run_{}", std::process::id()));

    let mut sink = DirectorySink::new(&dir).unwrap();
    let summary = run(&mut s, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    assert_eq!(sink.written(), summary.snapshots_emitted);

    // Each emitted file must be a valid snapshot whose data matches what ran. We
    // check the IC file against the freshly-sampled IC (positions are exact f64).
    let (_, ic) = small_collision();
    let ic_path = dir.join("snapshot_00000000.snap");
    let (h0, s0) = galaxy_io::read_file(&ic_path).unwrap();
    assert_eq!(h0.step, 0);
    assert_eq!(s0.len(), ic.len());
    assert_eq!(
        s0.pos, ic.pos,
        "IC positions did not round-trip through the run"
    );

    // The final-step file exists and is readable.
    let last_path = dir.join("snapshot_00000040.snap");
    let (h_last, _) = galaxy_io::read_file(&last_path).unwrap();
    assert_eq!(h_last.step, 40);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rejects_invalid_config() {
    let (_, mut s) = small_collision();
    let mut solver = galaxy_solvers::DirectSum::new(G, EPS);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let mut sink = CollectingSink::default();

    let mut bad = config(10, 0); // snapshot_every = 0
    assert!(matches!(
        run(&mut s, &mut solver, &mut integ, &bg, &bad, &mut sink),
        Err(SimError::Config(_))
    ));

    bad = config(10, 5);
    bad.dt = 0.0; // non-positive dt
    assert!(matches!(
        run(&mut s, &mut solver, &mut integ, &bg, &bad, &mut sink),
        Err(SimError::Config(_))
    ));
}
