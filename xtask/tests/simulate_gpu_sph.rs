//! GPU-SPH **G6** gate: the GPU-resident SPH branch of `simulate_snapshots`.
//!
//! G1–G5 brought the SPH stages up standalone (G1–G4) then wired them into
//! `GpuResidentLeapfrog` (G5a–c). G6 is the last landing: a GPU branch in the movie
//! pipeline's simulate step, selectable alongside the CPU `GravitySph` branch, that
//! drives the resident stepper over the same fixed-`dt` snapshot cadence and emits the
//! same `.snap` files.
//!
//! ## The gate is NOT a trajectory match (D5)
//! A self-gravitating merger is chaotic: an f32-vs-f64 force difference e-folds over a
//! fraction of a dynamical time, so GPU and CPU trajectories diverge macroscopically
//! over a long run *by construction*. A per-particle tolerance over the full run would
//! fail on a correct port. Instead the sharp, tolerance-robust gate is a **two-sided
//! bracket at early snapshots**: the GPU gas must
//!   (a) **differ from gravity-only** — hydro acted at all; and
//!   (b) **agree with CPU-SPH** within a measured f32 tolerance — hydro acted correctly.
//! It is about *which reference the GPU is closer to*, so it catches both wiring-bug
//! signatures directly: an empty gas subset / no-hydro looks like gravity-only, and a
//! wrong scatter / collapsed-stale grid diverges from *both*.
//!
//! Backing it are cheap unit guards, each pinning a named failure the plan calls out:
//! the output gas subset non-empty and progenitor-tagged (the `from_phase_space`
//! column-drop), the snapshot cadence identical to the CPU path, total gas mass exact,
//! and bounded total-momentum drift (a D5 long-run invariant; no blowup/NaN).
//!
//! GPU-gated: needs a wgpu adapter; without one the GPU branch returns an error and the
//! tests fail loudly (never silently skipped).

use std::path::Path;

use galaxy_core::{DVec3, LeapfrogKdk, Species, State, StaticBackground};
use galaxy_sim::{run, SimConfig, SnapshotSink};
use galaxy_solvers::BarnesHut;
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{build_scenario, parse_scenario_toml, Scenario};
use galaxy_xtask::{G, THETA};

// --- a small gas-rich scenario, built through the real front-end ----------------

/// A valid gas-rich `disk-plummer` scenario with QUICK counts small enough for a few
/// GPU steps to be cheap (≥48 gas particles per galaxy for the adaptive-h density
/// pass). f = 0.3, c_s = 0.1 keeps min Q_gas comfortably above 1. Two galaxies ⇒
/// several progenitor tags, so the progenitor re-attach guard has something to check.
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

/// The gas scenario at QUICK size, re-timed to a short, stable run: `dt` well below the
/// isolated-disk CFL bound, a handful of steps, snapshots every step. Kept early enough
/// that f32-vs-f64 chaos has not yet e-folded, so the two-sided bracket's per-particle
/// side (b) is meaningful.
fn gas_scenario() -> Scenario {
    let mut s = build_scenario(
        &parse_scenario_toml(GAS_TOML).expect("gas toml parses"),
        true,
    );
    s.dt = 1e-4;
    s.n_steps = 4;
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

/// Gas-particle velocities in index order (the resident stepper preserves the upload
/// order, so GPU and CPU gas rows line up 1:1 by this index for the bracket's side b).
fn gas_velocities(state: &State) -> Vec<DVec3> {
    (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .map(|i| state.vel[i])
        .collect()
}

fn gas_count(state: &State) -> usize {
    state.kind.iter().filter(|&&k| k == Species::Gas).count()
}

/// Σ m over the gas subset — exact under both backends (mass is host-tracked, never
/// evolved), so it must match to the bit between GPU and CPU snapshots.
fn gas_mass(state: &State) -> f64 {
    (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .map(|i| state.mass[i])
        .sum()
}

/// Total-system linear momentum Σ m·v — conserved to f32-force/integration roundoff
/// over a few steps (gravity + hydro are internal), the cheap D5 no-blowup invariant.
fn total_momentum(state: &State) -> DVec3 {
    (0..state.len())
        .map(|i| state.vel[i] * state.mass[i])
        .fold(DVec3::ZERO, |a, b| a + b)
}

/// Run the same IC through plain Barnes-Hut gravity (no hydro) for the scenario's steps,
/// returning the final state — the gravity-only reference for bracket side (a).
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
    let last = std::rc::Rc::new(std::cell::RefCell::new(None));
    let mut sink = CaptureLast { last: last.clone() };
    run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    let final_state = last.borrow_mut().take();
    final_state.expect("at least one emit")
}

struct CaptureLast {
    last: std::rc::Rc<std::cell::RefCell<Option<State>>>,
}
impl SnapshotSink for CaptureLast {
    fn emit(&mut self, _h: &galaxy_io::Header, state: &State) -> Result<(), galaxy_sim::SimError> {
        *self.last.borrow_mut() = Some(state.clone());
        Ok(())
    }
}

/// Run `s` through both backends into fresh dirs and read back the last snapshot of each
/// plus the gravity-only final — the shared setup for the bracket + guards.
struct AbRun {
    gpu_last: State,
    cpu_last: State,
    grav_last: State,
    gpu_paths: Vec<std::path::PathBuf>,
    cpu_paths: Vec<std::path::PathBuf>,
}

fn run_ab(s: &Scenario) -> AbRun {
    let tmp = tempdir();
    let dir_gpu = tmp.join("gpu");
    let dir_cpu = tmp.join("cpu");
    std::fs::create_dir_all(&dir_gpu).unwrap();
    std::fs::create_dir_all(&dir_cpu).unwrap();

    simulate_snapshots(s, &dir_gpu, Backend::Gpu).expect("gpu run (needs a wgpu adapter)");
    simulate_snapshots(s, &dir_cpu, Backend::Cpu).expect("cpu run");

    let gpu_paths = snap_paths(&dir_gpu);
    let cpu_paths = snap_paths(&dir_cpu);
    let (_, gpu_last) = galaxy_io::read_file(gpu_paths.last().expect("gpu snapshot")).unwrap();
    let (_, cpu_last) = galaxy_io::read_file(cpu_paths.last().expect("cpu snapshot")).unwrap();
    AbRun {
        gpu_last,
        cpu_last,
        grav_last: gravity_only_final(s),
        gpu_paths,
        cpu_paths,
    }
}

// --- the two-sided bracket: hydro acted (a) AND acted correctly (b) --------------

#[test]
fn gpu_gas_brackets_between_gravity_only_and_cpu_sph() {
    let s = gas_scenario();
    let ab = run_ab(&s);

    let v_gpu = gas_velocities(&ab.gpu_last);
    let v_cpu = gas_velocities(&ab.cpu_last);
    let v_grav = gas_velocities(&ab.grav_last);
    assert_eq!(v_gpu.len(), v_cpu.len(), "same gas population (gpu vs cpu)");
    assert_eq!(
        v_gpu.len(),
        v_grav.len(),
        "same gas population (gpu vs grav)"
    );
    assert!(!v_gpu.is_empty(), "the scenario must carry gas");

    // (a) Hydro acted: the GPU gas velocity field is NOT the gravity-only field.
    let max_vs_grav = v_gpu
        .iter()
        .zip(&v_grav)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0_f64, f64::max);
    // (b) Hydro acted CORRECTLY: the GPU gas field tracks the CPU-SPH field to an f32
    //     tolerance at this early snapshot (before chaos e-folds).
    let max_vs_cpu = v_gpu
        .iter()
        .zip(&v_cpu)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0_f64, f64::max);
    eprintln!("G6 bracket: max|v_gpu - v_grav| = {max_vs_grav:.3e} (must be large), max|v_gpu - v_cpu| = {max_vs_cpu:.3e} (must be small)");

    assert!(
        max_vs_grav > 1e-6,
        "GPU gas is indistinguishable from gravity-only — hydro did not act (max Δ = {max_vs_grav:.3e})"
    );
    // Measured-then-tightened placeholder bound; refined once the green run prints it.
    assert!(
        max_vs_cpu < 1e-4,
        "GPU gas diverged from CPU-SPH beyond the f32 bracket (max Δ = {max_vs_cpu:.3e})"
    );
    // The bracket is only meaningful if the correct side is far tighter than the wrong one.
    assert!(
        max_vs_cpu < 0.1 * max_vs_grav,
        "GPU is not decisively closer to CPU-SPH than to gravity-only"
    );
}

// --- cheap unit guards, each pinning a named failure -----------------------------

#[test]
fn gpu_snapshot_reattaches_gas_and_progenitor_columns() {
    // `snapshot()` rebuilds State via `from_phase_space`, which resets kind→Collisionless,
    // progenitor→0, id→sequential. The GPU branch must re-stamp the uploaded columns, or
    // the gas subset comes back empty and the movie's sf_progenitors coloring is lost.
    let s = gas_scenario();
    let ab = run_ab(&s);

    let ic_gas = gas_count(&s.state);
    assert!(ic_gas > 0, "the scenario must carry gas");
    assert_eq!(
        gas_count(&ab.gpu_last),
        ic_gas,
        "GPU snapshot lost its gas subset — kind not re-attached"
    );
    // The IC spans multiple progenitors; at least one non-zero tag must survive.
    assert!(
        ab.gpu_last.progenitor.iter().any(|p| p.0 != 0),
        "GPU snapshot lost progenitor tags — from_phase_space zeroed them and they were not re-stamped"
    );
}

#[test]
fn gpu_snapshot_cadence_matches_cpu() {
    let s = gas_scenario();
    let ab = run_ab(&s);
    assert_eq!(
        ab.gpu_paths.len(),
        ab.cpu_paths.len(),
        "GPU and CPU must emit the same number of snapshots"
    );
    assert!(!ab.gpu_paths.is_empty(), "the run must emit snapshots");
    for (g, c) in ab.gpu_paths.iter().zip(&ab.cpu_paths) {
        assert_eq!(
            g.file_name(),
            c.file_name(),
            "GPU snapshot step cadence must match the CPU path"
        );
    }
}

#[test]
fn gpu_conserves_gas_mass_and_bounds_momentum_drift() {
    let s = gas_scenario();
    let ab = run_ab(&s);

    // Mass is host-tracked, never evolved ⇒ exact GPU==CPU.
    assert_eq!(
        gas_mass(&ab.gpu_last).to_bits(),
        gas_mass(&ab.cpu_last).to_bits(),
        "total gas mass must be bit-identical across backends"
    );

    // Total linear momentum drifts only by f32-force / integration roundoff over a few
    // steps (gravity + hydro are internal) — a cheap D5 no-blowup invariant.
    let p0 = total_momentum(&s.state);
    let p_gpu = total_momentum(&ab.gpu_last);
    let drift = (p_gpu - p0).length();
    eprintln!(
        "G6 total-momentum drift over {} steps = {drift:.3e}",
        s.n_steps
    );
    assert!(
        p_gpu.x.is_finite() && p_gpu.y.is_finite() && p_gpu.z.is_finite(),
        "GPU momentum went non-finite — blowup/NaN"
    );
    assert!(
        drift < 1e-3,
        "total-momentum drift {drift:.3e} exceeds the bounded-invariant threshold"
    );
}

/// A unique-enough temp directory for one test, under the repo-configured temp root.
fn tempdir() -> std::path::PathBuf {
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("sim_gpu_sph_{:?}", std::thread::current().id()));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}
