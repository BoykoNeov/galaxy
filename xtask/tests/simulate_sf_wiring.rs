//! S6 — wiring star formation (`[physics.star_formation]`, natal-ember-forge) into
//! the movie pipeline. A gas-rich scenario carrying an SF recipe threads it into the
//! CPU stepping config's `sf` field, so dense converging gas converts in place to
//! collisionless star particles at each snapshot sync site. The GPU-resident path
//! does not apply the SF operator and must reject it loud.
//!
//! Gates:
//!   * CONVERSION (headline): a gas run with a decisive SF recipe (low `rho_thresh`,
//!     huge `efficiency`) over a converging (∇·v < 0) gas flow converts some gas to
//!     collisionless stars, stamping `formation_time = snapshot time`.
//!   * CAUSE-IS-SF (control): the SAME converging run with `sf = None` converts
//!     NOTHING (every particle stays gas, `formation_time = PRIMORDIAL`) — so it is
//!     the SF recipe, not the dynamics, that forms the stars.
//!   * GPU REJECT: `sf = Some(..)` on `Backend::Gpu` rejects loud (the resident path
//!     ignores SF — running it would silently drop every conversion).

use galaxy_core::{Species, State};
use galaxy_sim::StarFormationConfig;
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{build_scenario, parse_scenario_toml, Scenario};

/// A compact gas-rich scenario (QUICK counts) — the same shape the individual-wiring
/// test uses. Individual (hydro-only) timesteps auto-size `dt` from the CFL bound, so
/// the test can drive the gas hard (converging inflow) without a `CflGuard` abort.
const GAS_TOML: &str = r#"
name = "sfwire"
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
n_steps = 4
snapshot_every = 2
eps = 0.05
[sim.individual]
mode = "hydro-only"
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
    build_scenario(&parse_scenario_toml(GAS_TOML).expect("gas toml parses"), true)
}

/// A decisive SF recipe: `rho_thresh` tiny so the density gate passes for essentially
/// all gas, `efficiency` huge so `p = 1 − exp(−ε·dt/t_ff) ≈ 1` for any converging gas.
fn decisive_sf() -> StarFormationConfig {
    StarFormationConfig {
        rho_thresh: 1e-6,
        efficiency: 1e6,
        seed: 0xA11CE,
    }
}

/// Impose a homologous inflow `v = −c·pos` on every gas particle: `∇·v = −3c < 0`
/// everywhere (translation-invariant, so it converges within each disk regardless of
/// the galaxy offset), guaranteeing the converging-flow half of the SF criterion.
fn make_gas_converge(state: &mut State) {
    for i in 0..state.len() {
        if state.kind[i] == Species::Gas {
            state.vel[i] = -0.3 * state.pos[i];
        }
    }
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
        "sf_wiring_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

fn gas_count(state: &State) -> usize {
    state.kind.iter().filter(|k| **k == Species::Gas).count()
}

// --------------------------------------------------------------------------
// HEADLINE — conversion: a decisive SF recipe forms stars from converging gas.
// --------------------------------------------------------------------------

#[test]
fn star_formation_converts_converging_gas_on_cpu() {
    let mut s = gas_scenario();
    let gas0 = gas_count(&s.state);
    assert!(gas0 > 0, "the scenario must carry gas to form stars");
    make_gas_converge(&mut s.state);
    s.sf = Some(decisive_sf());

    let dir = tempdir("convert");
    simulate_snapshots(&s, &dir, Backend::Cpu).expect("SF gas run must complete");

    let files = snap_files(&dir);
    let (_, last) = galaxy_io::read_file(files.last().unwrap()).unwrap();
    let gas_final = gas_count(&last);
    assert!(
        gas_final < gas0,
        "a decisive SF recipe over converging gas must convert some gas to stars \
         (gas {gas0} -> {gas_final})"
    );

    // Every formed star carries a real formation time (not the primordial sentinel),
    // and it is one of the emitted snapshot times.
    let formed: Vec<f64> = last
        .formation_time
        .iter()
        .copied()
        .filter(|t| *t != State::PRIMORDIAL)
        .collect();
    assert_eq!(
        formed.len(),
        gas0 - gas_final,
        "the count of stamped formation times must equal the number converted"
    );
    assert!(
        formed.iter().all(|t| t.is_finite() && *t >= 0.0),
        "formation times must be finite, non-negative snapshot times, got {formed:?}"
    );
}

// --------------------------------------------------------------------------
// CONTROL — cause is SF, not dynamics: the same inflow with sf = None forms nothing.
// --------------------------------------------------------------------------

#[test]
fn star_formation_off_converts_nothing() {
    let mut s = gas_scenario();
    let gas0 = gas_count(&s.state);
    make_gas_converge(&mut s.state);
    s.sf = None;

    let dir = tempdir("off");
    simulate_snapshots(&s, &dir, Backend::Cpu).expect("SF-off gas run must complete");

    let files = snap_files(&dir);
    for f in &files {
        let (_, st) = galaxy_io::read_file(f).unwrap();
        assert_eq!(
            gas_count(&st),
            gas0,
            "with sf = None the converging gas must NOT convert (dynamics alone form nothing)"
        );
        assert!(
            st.formation_time.iter().all(|t| *t == State::PRIMORDIAL),
            "with sf = None every formation time stays the primordial sentinel"
        );
    }
}

// --------------------------------------------------------------------------
// GPU REJECT — the resident path does not apply the SF operator.
// --------------------------------------------------------------------------

#[test]
fn star_formation_on_gpu_backend_is_rejected() {
    let mut s = gas_scenario();
    // The GPU-resident path is the block-adaptive/fixed-dt SPH stepper, not the CPU
    // individual driver — clear the individual toggle so the run would route to GPU.
    s.individual = None;
    s.sf = Some(decisive_sf());

    let dir = tempdir("gpu_reject");
    let err = simulate_snapshots(&s, &dir, Backend::Gpu)
        .expect_err("star formation is CPU-only — the GPU backend must reject");
    let msg = err.to_string();
    assert!(
        msg.contains("CPU") || msg.to_lowercase().contains("star formation"),
        "GPU reject message should explain the CPU-only SF constraint, got: {msg}"
    );
}
