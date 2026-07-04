//! Gates for `simulate_snapshots` (M7c, D6): the movie pipeline's simulate step,
//! which chooses the force solver by gas presence.
//!
//! The one genuinely new contract is the gas path: a gas-rich scenario runs
//! Barnes-Hut gravity + isothermal SPH under a CFL sentinel that validates the
//! fixed global `dt` at t=0 *before* any snapshot is written, and the gas rows
//! actually feel the hydro force (i.e. `GravitySph`, not bare Barnes-Hut). The
//! gas-free path must stay the pre-M7c Barnes-Hut + `DirectorySink` + `run`
//! pipeline byte-for-byte — held here as a drift change-detector (the load-
//! bearing gas-free identity is the full pixel re-render against the retained
//! control, not this unit test).

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use galaxy_core::{LeapfrogKdk, Species, State, StaticBackground};
use galaxy_io::Header;
use galaxy_sim::{run, DirectorySink, SimConfig, SimError, SnapshotSink};
use galaxy_solvers::BarnesHut;
use galaxy_xtask::simulate::simulate_snapshots;
use galaxy_xtask::spec::{build_scenario, parse_scenario_toml, Scenario};
use galaxy_xtask::{G, THETA};

// --- a small gas-rich scenario, built through the real front-end ----------------

/// A valid gas-rich `disk-plummer` scenario with QUICK counts small enough for a
/// few sim steps to be cheap (≥48 gas particles per galaxy for the adaptive-h
/// density pass). f = 0.3, c_s = 0.1 keeps min Q_gas comfortably above 1.
const GAS_TOML: &str = r#"
name = "gassim"
seed = 7

[model]
kind = "disk-plummer"

[model.gas]
fraction = 0.3
sound_speed = 0.1

[model.galaxy1]
disk_mass = 0.15
scale_length = 0.5
hz_frac = 0.1
rmax_frac = 4.0
toomre_q = 1.5
halo = { mass = 1.0, scale = 1.0 }

[model.galaxy2]
disk_mass = 0.1
scale_length = 0.45
hz_frac = 0.1
rmax_frac = 4.0
toomre_q = 1.5
halo = { mass = 0.7, scale = 0.9 }

[model.counts.full]
halo1 = 800
disk1 = 600
halo2 = 600
disk2 = 500
gas1 = 400
gas2 = 300

[model.counts.quick]
halo1 = 200
disk1 = 150
halo2 = 150
disk2 = 120
gas1 = 150
gas2 = 120

[orbit]
eccentricity = 1.0
pericenter = 1.5
separation = 8.0

[sim]
dt = 0.01
n_steps = 100
snapshot_every = 10
eps = 0.05

[look]
splat_size = 0.12
frame_percentile = 0.98
palette = [[0.05, 0.035, 0.025], [1.0, 0.5, 0.25], [0.025, 0.035, 0.05], [0.35, 0.6, 1.0]]
sf_progenitors = [1, 3]

[[look.ramps]]
inner = [0.05, 0.035, 0.025]
outer = [0.05, 0.035, 0.025]

[[look.ramps]]
inner = [1.0, 0.35, 0.1]
outer = [0.55, 0.75, 1.0]

[[look.ramps]]
inner = [0.025, 0.035, 0.05]
outer = [0.025, 0.035, 0.05]

[[look.ramps]]
inner = [1.0, 0.3, 0.45]
outer = [0.4, 0.9, 0.9]

[rig]
kind = "static"
"#;

/// The gas scenario at QUICK size, then re-timed for a short, stable run:
/// `dt` well below the isolated-disk CFL bound, a handful of steps, snapshots
/// every step.
fn gas_scenario() -> Scenario {
    let mut s = build_scenario(&parse_scenario_toml(GAS_TOML).expect("gas toml parses"), true);
    s.dt = 1e-4;
    s.n_steps = 4;
    s.snapshot_every = 1;
    s
}

/// The `disk` preset (gas-free), re-timed to a couple of cheap steps.
fn gas_free_scenario() -> Scenario {
    let disk = parse_scenario_toml(galaxy_xtask::spec::preset("disk").unwrap()).unwrap();
    let mut s = build_scenario(&disk, true);
    s.n_steps = 2;
    s.snapshot_every = 1;
    s
}

fn snap_paths(dir: &Path) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "snap"))
        .collect();
    v.sort();
    v
}

/// Run a state through `n_steps` of plain Barnes-Hut gravity (no hydro), returning
/// the final state — the gravity-only reference for the "hydro actually acts" gate.
fn gravity_only_final(s: &Scenario) -> State {
    let mut state = s.state.clone();
    let mut solver = BarnesHut::new(G, s.eps, THETA);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = SimConfig {
        dt: s.dt,
        n_steps: s.n_steps,
        snapshot_every: s.snapshot_every,
        softening: s.eps,
        rng_seed: s.seed,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    };
    let last = Rc::new(RefCell::new(None));
    let mut sink = CaptureLast {
        last: last.clone(),
    };
    run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    let final_state = last.borrow_mut().take();
    final_state.expect("at least one emit")
}

/// A sink that keeps a clone of the most recently emitted state.
struct CaptureLast {
    last: Rc<RefCell<Option<State>>>,
}

impl SnapshotSink for CaptureLast {
    fn emit(&mut self, _header: &Header, state: &State) -> Result<(), SimError> {
        *self.last.borrow_mut() = Some(state.clone());
        Ok(())
    }
}

fn gas_velocities(state: &State) -> Vec<galaxy_core::DVec3> {
    (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .map(|i| state.vel[i])
        .collect()
}

// --- the gas path: t=0 CFL gate, snapshots, and hydro actually acting -----------

#[test]
fn gas_scenario_rejects_over_large_dt_before_writing_any_snapshot() {
    let tmp = tempdir();
    let snap_dir = tmp.join("snapshots");
    std::fs::create_dir_all(&snap_dir).unwrap();

    let mut s = gas_scenario();
    // A huge finite dt passes `run`'s own dt check but blows the hydro CFL bound
    // by orders of magnitude — the t=0 validate must reject it.
    s.dt = 1e6;

    let err = simulate_snapshots(&s, &snap_dir);
    assert!(err.is_err(), "over-large dt must be rejected at t=0");
    assert!(
        snap_paths(&snap_dir).is_empty(),
        "the t=0 CFL reject must precede any snapshot write"
    );
}

#[test]
fn gas_scenario_at_stable_dt_writes_the_expected_snapshots() {
    let tmp = tempdir();
    let snap_dir = tmp.join("snapshots");
    std::fs::create_dir_all(&snap_dir).unwrap();

    let s = gas_scenario();
    let summary = simulate_snapshots(&s, &snap_dir).expect("stable dt must run");

    // 4 steps, snapshot every step → step 0 + steps 1..=4 = 5 snapshots.
    assert_eq!(summary.snapshots_emitted, 5);
    assert_eq!(snap_paths(&snap_dir).len(), 5);
}

#[test]
fn gas_path_applies_hydro_not_bare_gravity() {
    // Route the gas scenario through `simulate_snapshots` (→ GravitySph), read the
    // final gas velocities, and compare against the SAME IC evolved by plain
    // Barnes-Hut gravity for the same steps. Pressure gradients are nonzero at
    // t=0, so hydro must push the gas onto a different velocity field.
    let tmp = tempdir();
    let snap_dir = tmp.join("snapshots");
    std::fs::create_dir_all(&snap_dir).unwrap();

    let s = gas_scenario();
    simulate_snapshots(&s, &snap_dir).expect("stable dt must run");

    let last_snap = snap_paths(&snap_dir).pop().expect("a final snapshot");
    let (_, sph_final) = galaxy_io::read_file(&last_snap).unwrap();
    let v_sph = gas_velocities(&sph_final);
    let v_grav = gas_velocities(&gravity_only_final(&s));

    assert_eq!(v_sph.len(), v_grav.len(), "same gas population");
    assert!(
        v_sph
            .iter()
            .zip(&v_grav)
            .any(|(a, b)| (*a - *b).length() > 1e-9),
        "gas velocities must diverge from gravity-only — hydro did not act"
    );
}

// --- the gas-free path is the bare Barnes-Hut pipeline, unchanged ---------------

#[test]
fn gas_free_path_matches_the_bare_barnes_hut_pipeline() {
    let s = gas_free_scenario();

    let tmp_a = tempdir();
    let dir_a = tmp_a.join("snapshots");
    std::fs::create_dir_all(&dir_a).unwrap();
    simulate_snapshots(&s, &dir_a).expect("gas-free run");

    // The reference: the exact pre-M7c inline pipeline — plain Barnes-Hut, plain
    // DirectorySink, no CFL guard.
    let tmp_b = tempdir();
    let dir_b = tmp_b.join("snapshots");
    std::fs::create_dir_all(&dir_b).unwrap();
    {
        let mut state = s.state.clone();
        let mut solver = BarnesHut::new(G, s.eps, THETA);
        let mut integ = LeapfrogKdk::new();
        let bg = StaticBackground;
        let cfg = SimConfig {
            dt: s.dt,
            n_steps: s.n_steps,
            snapshot_every: s.snapshot_every,
            softening: s.eps,
            rng_seed: s.seed,
            config_hash: 0,
            units: "nbody-G1".to_string(),
        };
        let mut sink = DirectorySink::new(&dir_b).unwrap();
        run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    }

    let (pa, pb) = (snap_paths(&dir_a), snap_paths(&dir_b));
    assert_eq!(pa.len(), pb.len(), "same snapshot count");
    assert!(!pa.is_empty(), "the run must emit snapshots");
    for (a, b) in pa.iter().zip(&pb) {
        assert_eq!(
            a.file_name(),
            b.file_name(),
            "snapshot filenames (step cadence) must match"
        );
        assert_eq!(
            std::fs::read(a).unwrap(),
            std::fs::read(b).unwrap(),
            "gas-free snapshot bytes must be identical to the bare pipeline: {}",
            a.display()
        );
    }
}

/// A unique-enough temp directory for one test, under the repo-configured temp
/// root. Uniqueness comes from the OS thread id + the dir name the caller joins.
fn tempdir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "sim_snap_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}
