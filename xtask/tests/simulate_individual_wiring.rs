//! I8/I6 pre-work — wiring individual (per-particle rung) timesteps into the movie
//! pipeline (plan laddered-ember-cadence). `[sim.individual].mode = "hydro-only"` on a
//! gas-rich scenario routes the simulate step through `run_individual` (CPU only); a
//! gas-free scenario ignores it, and the toggle is mutually exclusive with
//! `[sim.adaptive]`.
//!
//! Gates (mirroring the `[sim.adaptive]` wiring, plus the toggle's reject surface):
//!   * PRODUCIBILITY (headline): a gas run whose fixed `dt` exceeds the hydro CFL bound
//!     — so the fixed-dt path fails loud via `CflGuard` — instead COMPLETES under
//!     individual timesteps, which size `dt_base` from the CFL bound per base block.
//!   * REJECTS: `[sim.adaptive]` + `[sim.individual]` together (ambiguous gas driver);
//!     `mode = "hydro+gravity"` (the I-grav layer is unbuilt); `Backend::Gpu` (the
//!     individual path is CPU-only).
//!   * GAS-FREE IGNORE: a collisionless scenario takes the fixed-dt Barnes-Hut path
//!     either way, byte-for-byte (the toggle is a gas-path feature).
//!   * DEFAULTS: an empty `[sim.individual]` table enables `hydro-only` at the shipped
//!     knob defaults (courant 0.25, r_max 10, the binding limiter n_limit 1).

use galaxy_solvers::sph::{max_stable_dt, DensityConfig, Eos, HydroParams};
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{
    build_scenario, parse_scenario_toml, IndividualMode, IndividualSpec, Scenario,
};

/// A compact gas-rich scenario (QUICK counts). The trailing `EXTRA` is spliced in just
/// before `[look]` so a test can inject `[sim.adaptive]` / `[sim.individual]` sections.
fn gas_toml(extra: &str) -> String {
    format!(
        r#"
name = "indwire"
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
halo = {{ mass = 1.0, scale = 1.0 }}
[model.galaxy2]
disk_mass = 0.1
scale_length = 0.45
hz_frac = 0.1
rmax_frac = 4.0
toomre_q = 1.5
halo = {{ mass = 0.7, scale = 0.9 }}
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
{extra}
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
"#
    )
}

/// A tiny gas-FREE scenario (the `disk` shape) for the byte-identity gate.
const GAS_FREE_TOML: &str = r#"
name = "indwire_free"
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

fn gas_scenario(extra: &str) -> Scenario {
    build_scenario(
        &parse_scenario_toml(&gas_toml(extra)).expect("gas toml parses"),
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
        "individual_wiring_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

// --------------------------------------------------------------------------
// HEADLINE — producibility: individual timesteps complete where fixed dt aborts.
// --------------------------------------------------------------------------

#[test]
fn individual_gas_completes_where_fixed_dt_aborts() {
    let mut s = gas_scenario("");
    let c_s = s.sound_speed.expect("gas scenario has c_s");
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };
    // A dt comfortably ABOVE the CFL sentinel threshold (C_cfl = 0.25) ⇒ the fixed-dt
    // guard is guaranteed to trip.
    let threshold = max_stable_dt(&s.state, &params, &DensityConfig::default(), 0.25);
    assert!(threshold.is_finite() && threshold > 0.0);
    s.dt = 2.5 * threshold;

    // Fixed-dt path (no toggle): the CFL sentinel aborts before any file.
    let mut fixed = s.clone();
    fixed.individual = None;
    let dir_fixed = tempdir("fixed_abort");
    assert!(
        simulate_snapshots(&fixed, &dir_fixed, Backend::Cpu).is_err(),
        "fixed dt = 2.5x the CFL threshold must trip the sentinel"
    );

    // Individual path (hydro-only): same over-large dt (now only the output cadence) —
    // completes, sizing dt_base from the CFL bound per base block.
    let mut ind = s.clone();
    ind.individual = Some(IndividualSpec {
        mode: IndividualMode::HydroOnly,
        courant: 0.25,
        r_max: 10,
        n_limit: 1,
        dt_base_cap: f64::INFINITY,
    });
    let dir_ind = tempdir("individual_ok");
    let summary = simulate_snapshots(&ind, &dir_ind, Backend::Cpu)
        .expect("individual gas run must complete where fixed dt aborts");
    assert_eq!(summary.snapshots_emitted, 5, "IC + 4 output intervals");

    // Snapshots land on the time grid k · output_dt (output_dt = snapshot_every · dt).
    let output_dt = ind.snapshot_every as f64 * ind.dt;
    let files = snap_files(&dir_ind);
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

// --------------------------------------------------------------------------
// REJECTS — spec-level (parse) and runtime (backend).
// --------------------------------------------------------------------------

#[test]
fn adaptive_and_individual_together_are_rejected() {
    let both = gas_toml("[sim.adaptive]\n[sim.individual]\nmode = \"hydro-only\"\n");
    let err = parse_scenario_toml(&both).expect_err("declaring both gas drivers must reject");
    assert!(
        err.contains("adaptive") && err.contains("individual"),
        "reject message should name both toggles, got: {err}"
    );
}

#[test]
fn hydro_plus_gravity_mode_is_rejected_as_unbuilt() {
    let toml = gas_toml("[sim.individual]\nmode = \"hydro+gravity\"\n");
    let err = parse_scenario_toml(&toml).expect_err("hydro+gravity is unbuilt and must reject");
    assert!(
        err.contains("hydro+gravity") || err.contains("gravity"),
        "reject message should point at the unbuilt gravity layer, got: {err}"
    );
}

#[test]
fn individual_on_gpu_backend_is_rejected() {
    let mut s = gas_scenario("");
    s.individual = Some(IndividualSpec {
        mode: IndividualMode::HydroOnly,
        courant: 0.25,
        r_max: 10,
        n_limit: 1,
        dt_base_cap: f64::INFINITY,
    });
    let dir = tempdir("gpu_reject");
    let err = simulate_snapshots(&s, &dir, Backend::Gpu)
        .expect_err("individual timesteps are CPU-only — GPU backend must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("CPU") || msg.contains("individual"),
        "GPU reject message should explain the CPU-only constraint, got: {msg}"
    );
}

// --------------------------------------------------------------------------
// GAS-FREE IGNORE — the toggle is a gas-path feature (byte-identical either way).
// --------------------------------------------------------------------------

#[test]
fn gas_free_ignores_individual_flag_byte_identical() {
    let base = build_scenario(
        &parse_scenario_toml(GAS_FREE_TOML).expect("gas-free toml parses"),
        true,
    );
    assert!(base.sound_speed.is_none(), "scenario must be gas-free");

    let mut without = base.clone();
    without.individual = None;
    let dir_a = tempdir("free_noind");
    simulate_snapshots(&without, &dir_a, Backend::Cpu).unwrap();

    let mut with = base.clone();
    with.individual = Some(IndividualSpec {
        mode: IndividualMode::HydroOnly,
        courant: 0.25,
        r_max: 10,
        n_limit: 1,
        dt_base_cap: f64::INFINITY,
    });
    let dir_b = tempdir("free_ind");
    simulate_snapshots(&with, &dir_b, Backend::Cpu).unwrap();

    let (fa, fb) = (snap_files(&dir_a), snap_files(&dir_b));
    assert_eq!(fa.len(), fb.len(), "same snapshot count");
    assert!(!fa.is_empty());
    for (a, b) in fa.iter().zip(&fb) {
        assert_eq!(
            std::fs::read(a).unwrap(),
            std::fs::read(b).unwrap(),
            "gas-free snapshots must be byte-identical with/without the individual flag"
        );
    }
}

// --------------------------------------------------------------------------
// DEFAULTS — an empty [sim.individual] table enables hydro-only at shipped defaults.
// --------------------------------------------------------------------------

#[test]
fn empty_individual_table_enables_hydro_only_at_defaults() {
    let s = gas_scenario("[sim.individual]\n");
    let ind = s
        .individual
        .expect("an empty [sim.individual] table enables the toggle");
    assert_eq!(ind.mode, IndividualMode::HydroOnly, "default mode");
    assert_eq!(ind.courant, 0.25, "default courant");
    assert_eq!(ind.r_max, 10, "default r_max");
    assert_eq!(ind.n_limit, 1, "default binding limiter");
    assert!(
        ind.dt_base_cap.is_infinite(),
        "default non-binding base-dt cap"
    );
}
