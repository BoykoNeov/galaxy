//! A4 — wiring block-adaptive dt into the movie pipeline (plan
//! courant-quickening-cadence). `[sim.adaptive]` on a gas-rich scenario routes the
//! simulate step through the adaptive driver; a gas-free scenario ignores it.
//!
//! The headline gate is PRODUCIBILITY: a gas run whose fixed `dt` exceeds the hydro
//! CFL bound (so the fixed-dt path fails loud via `CflGuard`) instead COMPLETES under
//! adaptive dt — the exact failure (`settling-cinder-vigil` Finding A) this feature
//! exists to fix. Plus: the shipped `gasrich` preset now enables adaptive, gas-free
//! keeps its fixed-dt byte-identity, and the adaptive output lands on the same time grid.

use galaxy_solvers::sph::{max_stable_dt, DensityConfig, HydroParams};
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{build_scenario, parse_scenario_toml, preset, AdaptiveSpec, Scenario};

/// A compact gas-rich scenario (QUICK counts) — enough gas for the adaptive-h density
/// pass, small enough for a few CPU steps to be cheap.
const GAS_TOML: &str = r#"
name = "adaptwire"
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
halo1 = 120
disk1 = 100
halo2 = 100
disk2 = 80
gas1 = 90
gas2 = 80
[orbit]
eccentricity = 1.0
pericenter = 1.5
separation = 8.0
[sim]
dt = 0.005
n_steps = 8
snapshot_every = 2
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

/// A tiny gas-FREE scenario (the `disk` shape, minimal counts) for the byte-identity gate.
const GAS_FREE_TOML: &str = r#"
name = "adaptwire_free"
seed = 3
[model]
kind = "disk-plummer"
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
halo1 = 200
disk1 = 150
halo2 = 150
disk2 = 120
[model.counts.quick]
halo1 = 60
disk1 = 40
halo2 = 40
disk2 = 30
[orbit]
eccentricity = 1.0
pericenter = 1.5
separation = 8.0
[sim]
dt = 0.01
n_steps = 6
snapshot_every = 2
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

fn gas_scenario() -> Scenario {
    build_scenario(
        &parse_scenario_toml(GAS_TOML).expect("gas toml parses"),
        true,
    )
}

fn snap_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "snap"))
        .collect();
    v.sort();
    v
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "adaptive_wiring_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// The shipped `gasrich` preset now enables adaptive dt at the shipped defaults.
#[test]
fn gasrich_preset_enables_adaptive() {
    let s = build_scenario(
        &parse_scenario_toml(preset("gasrich").unwrap()).unwrap(),
        true,
    );
    let a = s.adaptive.expect("gasrich must enable [sim.adaptive]");
    assert_eq!(a.courant, 0.25, "shipped Courant default");
    assert_eq!(a.block_steps, 16);
    assert_eq!(a.max_growth, 1.25);
}

/// HEADLINE (producibility, Finding A): a gas run whose fixed `dt` exceeds the CFL bound
/// FAILS loud on the fixed-dt path but COMPLETES under adaptive dt.
#[test]
fn adaptive_gas_completes_where_fixed_dt_aborts() {
    let mut s = gas_scenario();
    let c_s = s.sound_speed.expect("gas scenario has c_s");
    let params = HydroParams {
        sound_speed: c_s,
        ..HydroParams::default()
    };
    // Pick a dt comfortably ABOVE the CFL sentinel's threshold (C_cfl = 0.25) so the
    // fixed-dt guard is guaranteed to trip.
    let threshold = max_stable_dt(&s.state, &params, &DensityConfig::default(), 0.25);
    assert!(threshold.is_finite() && threshold > 0.0);
    s.dt = 2.5 * threshold;
    // n_steps / snapshot_every = 4 output intervals ⇒ IC + 4 = 5 snapshots.

    // Fixed-dt path (adaptive disabled): the CFL sentinel aborts before any file.
    let mut fixed = s.clone();
    fixed.adaptive = None;
    let dir_fixed = tempdir("fixed_abort");
    let r = simulate_snapshots(&fixed, &dir_fixed, Backend::Cpu);
    assert!(
        r.is_err(),
        "fixed dt = 2.5x the CFL threshold must trip the sentinel"
    );

    // Adaptive path: same over-large dt (now only the output cadence) — completes.
    let mut adaptive = s.clone();
    adaptive.adaptive = Some(AdaptiveSpec {
        courant: 0.25,
        max_growth: 1.25,
        block_steps: 16,
    });
    let dir_adapt = tempdir("adaptive_ok");
    let summary = simulate_snapshots(&adaptive, &dir_adapt, Backend::Cpu)
        .expect("adaptive gas run must complete where fixed dt aborts");
    assert_eq!(summary.snapshots_emitted, 5, "IC + 4 output intervals");

    // Snapshots land on the time grid k · output_dt (output_dt = snapshot_every · dt).
    let output_dt = adaptive.snapshot_every as f64 * adaptive.dt;
    let files = snap_files(&dir_adapt);
    assert_eq!(files.len(), 5);
    for (k, f) in files.iter().enumerate() {
        let (h, _) = galaxy_io::read_file(f).unwrap();
        let want = k as f64 * output_dt;
        assert!(
            (h.time - want).abs() < 1e-9,
            "snapshot {k} time {} != {want}",
            h.time
        );
    }
}

/// A gas-FREE scenario ignores `[sim.adaptive]` — the fixed-dt Barnes-Hut path is taken
/// either way, byte-for-byte (preserving the pre-adaptive byte-identity gate).
#[test]
fn gas_free_ignores_adaptive_flag_byte_identical() {
    let base = build_scenario(
        &parse_scenario_toml(GAS_FREE_TOML).expect("gas-free toml parses"),
        true,
    );
    assert!(base.sound_speed.is_none(), "scenario must be gas-free");

    let mut without = base.clone();
    without.adaptive = None;
    let dir_a = tempdir("free_noadapt");
    simulate_snapshots(&without, &dir_a, Backend::Cpu).unwrap();

    let mut with = base.clone();
    with.adaptive = Some(AdaptiveSpec {
        courant: 0.25,
        max_growth: 1.25,
        block_steps: 16,
    });
    let dir_b = tempdir("free_adapt");
    simulate_snapshots(&with, &dir_b, Backend::Cpu).unwrap();

    let fa = snap_files(&dir_a);
    let fb = snap_files(&dir_b);
    assert_eq!(fa.len(), fb.len(), "same snapshot count");
    assert!(!fa.is_empty());
    for (a, b) in fa.iter().zip(&fb) {
        let ba = std::fs::read(a).unwrap();
        let bb = std::fs::read(b).unwrap();
        assert_eq!(
            ba, bb,
            "gas-free snapshots must be byte-identical with/without the adaptive flag"
        );
    }
}
