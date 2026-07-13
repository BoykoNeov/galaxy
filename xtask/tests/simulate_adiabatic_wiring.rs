//! H5-C2 — wiring the adiabatic gas EOS into the movie pipeline
//! (incandescent-nebular-veil). `[model.gas].gamma` switches the gas to the ideal-gas
//! adiabatic EOS `P=(γ−1)ρu` and routes `simulate_snapshots` through
//! `LeapfrogKdkThermal` + `Eos::Adiabatic` on the CPU block-adaptive path — the one
//! path whose CFL arm reads the per-particle adiabatic sound speed `√(γ(γ−1)u)` (E4a).
//!
//! Two gates:
//!  - **Rejection**: the unsupported compositions fail loud rather than silently running
//!    isothermal — adiabatic requires `[sim.adaptive]` (the fixed-dt `CflGuard` assumes a
//!    single isothermal `c_s` and would panic on `HydroParams::sound_speed()`), is CPU-only
//!    (GPU adiabatic deferred), and is incompatible with `[sim.individual]` (isothermal arm).
//!  - **Evolution**: an adiabatic run COMPLETES and its gas `u` actually evolves — the IC
//!    seeds a uniform `u = c_s²/(γ−1)`, and the thermal integrator develops a spread
//!    (proving the adiabatic branch ran, not a silent isothermal `LeapfrogKdk` that never
//!    touches `u`). Every `u` stays finite and positive.

use galaxy_core::Species;
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{
    build_scenario, parse_scenario_toml, IndividualMode, IndividualSpec, Scenario,
};

/// A compact adiabatic gas-rich scenario (QUICK counts + `[sim.adaptive]` + γ = 5/3).
const ADIA_TOML: &str = r#"
name = "adiawire"
seed = 7
[model]
kind = "disk-plummer"
[model.gas]
fraction = 0.3
sound_speed = 0.1
gamma = 1.6666666666666667
u_floor = 1e-8
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
n_steps = 12
snapshot_every = 3
eps = 0.05
[sim.adaptive]
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

fn adia_scenario() -> Scenario {
    build_scenario(
        &parse_scenario_toml(ADIA_TOML).expect("adiabatic toml parses"),
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
        "adiabatic_wiring_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// Gas-only internal energies in a snapshot, in file order.
fn gas_u(path: &std::path::Path) -> Vec<f64> {
    let (_, st) = galaxy_io::read_file(path).unwrap();
    st.u.iter()
        .zip(&st.kind)
        .filter(|(_, k)| **k == Species::Gas)
        .map(|(&u, _)| u)
        .collect()
}

/// Adiabatic gas without `[sim.adaptive]` is rejected — the fixed-dt `CflGuard` assumes a
/// single isothermal `c_s` (and would panic on `sound_speed()`), so running silently
/// isothermal is not allowed.
#[test]
fn adiabatic_without_adaptive_is_rejected() {
    let mut s = adia_scenario();
    assert!(s.gamma.is_some(), "scenario must be adiabatic");
    s.adaptive = None;
    let dir = tempdir("no_adaptive");
    let err = simulate_snapshots(&s, &dir, Backend::Cpu)
        .expect_err("adiabatic + fixed-dt must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("adaptive"),
        "error should point at the missing [sim.adaptive]: {msg}"
    );
}

/// Adiabatic gas on the GPU backend is rejected (GPU adiabatic EOS is deferred).
#[test]
fn adiabatic_on_gpu_is_rejected() {
    let s = adia_scenario();
    let dir = tempdir("gpu");
    let err = simulate_snapshots(&s, &dir, Backend::Gpu)
        .expect_err("adiabatic + GPU must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("CPU"),
        "error should say adiabatic is CPU-only: {msg}"
    );
}

/// Adiabatic gas with `[sim.individual]` is rejected — the individual thermal arm is
/// isothermal-only.
#[test]
fn adiabatic_with_individual_is_rejected() {
    let mut s = adia_scenario();
    s.adaptive = None;
    s.individual = Some(IndividualSpec {
        mode: IndividualMode::HydroOnly,
        courant: 0.25,
        r_max: 10,
        n_limit: 1,
        dt_base_cap: f64::INFINITY,
    });
    let dir = tempdir("individual");
    let err = simulate_snapshots(&s, &dir, Backend::Cpu)
        .expect_err("adiabatic + individual must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("individual"),
        "error should point at [sim.individual]: {msg}"
    );
}

/// An adiabatic adaptive run COMPLETES and its gas `u` EVOLVES: the IC seeds a uniform
/// `u = c_s²/(γ−1)`, and by the final snapshot the thermal integrator has developed a
/// spread (a silent isothermal `LeapfrogKdk` would leave `u` bit-uniform). All `u` stay
/// finite and positive.
#[test]
fn adiabatic_adaptive_completes_and_evolves_u() {
    let s = adia_scenario();
    let c_s = s.sound_speed.expect("gas-rich");
    let gamma = s.gamma.expect("adiabatic");
    let u_init = c_s * c_s / (gamma - 1.0);

    let dir = tempdir("evolve");
    let summary =
        simulate_snapshots(&s, &dir, Backend::Cpu).expect("adiabatic adaptive run completes");
    assert_eq!(summary.snapshots_emitted, 5, "IC + 4 output intervals");

    let files = snap_files(&dir);
    assert_eq!(files.len(), 5);

    // Step 0 (the IC): gas u is the uniform seed c_s²/(γ−1).
    let u0 = gas_u(&files[0]);
    assert!(!u0.is_empty(), "scenario carries gas");
    for u in &u0 {
        assert!(
            (u - u_init).abs() < 1e-12,
            "IC gas u must be the uniform seed {u_init}, got {u}"
        );
    }

    // Final snapshot: u evolved (spread developed), every value finite and positive.
    let u_last = gas_u(files.last().unwrap());
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &u in &u_last {
        assert!(u.is_finite(), "gas u must stay finite, got {u}");
        assert!(u > 0.0, "gas u must stay positive, got {u}");
        min = min.min(u);
        max = max.max(u);
    }
    assert!(
        max - min > 0.0,
        "adiabatic thermal integrator must evolve u into a spread (min {min}, max {max}); \
         a bit-uniform field means the isothermal integrator ran instead"
    );
}
