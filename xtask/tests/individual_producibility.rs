//! I6 — producibility + speedup validation for the individual (per-particle rung)
//! `hydro-only` path (plan laddered-ember-cadence). The "real done" for lever (a):
//! the gasrich showpiece runs through `run_individual` in `hydro-only` mode,
//! COMPLETES, CONVERGES to a finer-courant reference, and the measured wall-clock
//! speedup over the global-adaptive path (A5) justifies the individual machinery.
//!
//! Mirrors A5 (`adaptive_producibility.rs`): a cheap always-on QUICK smoke plus a
//! deferred `#[ignore]` full-res run. I7 (the active-subset gather) is what makes
//! this a speedup and not just a completes — before it the fine-tick loop
//! recomputed the whole gas set every tick (same eval count as global adaptive).
//!
//! Two gates:
//!   * `gasrich_quick_individual_completes` (always-on): the REAL gasrich preset at
//!     QUICK size, adaptive dropped and `[sim.individual]` hydro-only enabled,
//!     completes a bounded run landing on the output grid — the real-preset smoke
//!     that the I7-wired individual path works end-to-end.
//!   * `full_res_gasrich_individual_completes_and_converges` (`#[ignore]`, the I6
//!     run — `--release --ignored`): the full-res showpiece to completion, a
//!     short-prefix convergence check to a finer-courant reference, the realized
//!     CFL dynamic range across the 61 snapshots, and the wall-clock speedup vs the
//!     A5 adaptive baseline. Snapshots are RETAINED (env `GALAXY_I6_OUT`, default
//!     `M:\claud_projects\temp\i6_individual`) for the FULL grav-rung-spread record
//!     gate that precedes any I-grav work.

use galaxy_io::read_file;
use galaxy_solvers::sph::{max_stable_dt, DensityConfig, Eos, HydroParams};
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{
    build_scenario, parse_scenario_toml, preset, IndividualMode, IndividualSpec, Scenario,
};

/// The shipped gasrich preset with the global-adaptive toggle dropped and the
/// individual `hydro-only` toggle enabled at a generous `r_max` (the QUICK
/// pericenter peaked at rung 8; 14 gives headroom for the denser FULL knots — a
/// silent clamp would step coarser than CFL, and the convergence gate is the
/// backstop). `courant` is overridable for the convergence sweep.
fn gasrich_individual(quick: bool, courant: f64, r_max: u32) -> Scenario {
    let mut s = build_scenario(
        &parse_scenario_toml(preset("gasrich").expect("gasrich preset")).expect("gasrich parses"),
        quick,
    );
    s.adaptive = None; // mutually exclusive with individual
    s.individual = Some(IndividualSpec {
        mode: IndividualMode::HydroOnly,
        courant,
        r_max,
        n_limit: 1,
        dt_base_cap: f64::INFINITY,
    });
    s
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
        "individual_prod_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// The REAL gasrich preset, individual hydro-only path: completes a bounded run
/// landing on the output grid. Real-preset smoke that the I7-wired individual
/// wiring (GravitySph + active-subset stepper) works end-to-end.
#[test]
fn gasrich_quick_individual_completes() {
    let mut s = gasrich_individual(true, 0.25, 12); // QUICK
    assert!(s.individual.is_some(), "individual hydro-only enabled");
    assert!(
        s.adaptive.is_none(),
        "adaptive dropped (mutually exclusive)"
    );
    assert!(s.sound_speed.is_some(), "gasrich carries gas");
    // One output interval (snapshot_every = 100 ⇒ n_steps = 100): IC + 1 = 2 snapshots.
    s.n_steps = s.snapshot_every;

    let dir = tempdir("gasrich_ind");
    let summary = simulate_snapshots(&s, &dir, Backend::Cpu)
        .expect("individual gasrich (real preset) must complete a bounded run");
    assert_eq!(summary.snapshots_emitted, 2, "IC + 1 output interval");
    let output_dt = s.snapshot_every as f64 * s.dt;
    for (k, f) in snap_files(&dir).iter().enumerate() {
        let (h, _) = read_file(f).unwrap();
        assert!(
            (h.time - k as f64 * output_dt).abs() < 1e-9,
            "snapshot {k} off the output grid"
        );
    }
}

/// I6 (run: `cargo test -p galaxy-xtask --release -- --ignored
/// full_res_gasrich_individual`). The full-res showpiece end-to-end on the
/// individual hydro-only path: (1) COMPLETES with the full 61 snapshots; (2)
/// short-prefix convergence to a finer-courant reference (chaotic showpiece — no
/// full-trajectory match, D5); (3) records the realized CFL dynamic range; (4)
/// reports the wall-clock vs the A5 adaptive baseline (~47.8 min = 2868 s, same
/// scenario/seed) — the "did the win survive at FULL" number. Snapshots RETAINED
/// under `GALAXY_I6_OUT` for the FULL grav-rung-spread record gate.
#[test]
#[ignore = "I6: full-res gasrich individual is ~45 min; run with --release --ignored"]
fn full_res_gasrich_individual_completes_and_converges() {
    // A5 adaptive baseline (documented, same scenario/seed): the speedup reference.
    const A5_ADAPTIVE_SECS: f64 = 2868.0;

    // Retained output dir (honours the temp-artifacts convention; survives for the
    // follow-up grav-rung-spread record gate).
    let out = std::env::var("GALAXY_I6_OUT")
        .unwrap_or_else(|_| r"M:\claud_projects\temp\i6_individual".to_string());
    let out = std::path::PathBuf::from(out);
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();

    // (1) COMPLETES — full-res, full horizon, individual hydro-only at r_max=14.
    let full = gasrich_individual(false, 0.25, 14);
    assert!(full.individual.is_some() && full.adaptive.is_none());
    let n_gas = full
        .state
        .kind
        .iter()
        .filter(|k| matches!(k, galaxy_core::Species::Gas))
        .count();
    let n_outputs = full.n_steps / full.snapshot_every; // 60 ⇒ 61 snapshots
    let t0 = std::time::Instant::now();
    let summary = simulate_snapshots(&full, &out, Backend::Cpu)
        .expect("full-res gasrich individual hydro-only must COMPLETE");
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "=== I6: full-res gasrich INDIVIDUAL hydro-only COMPLETED in {secs:.1} s \
         ({} snapshots, {} particles / {n_gas} gas) ===",
        summary.snapshots_emitted,
        full.state.len()
    );
    eprintln!(
        "  wall-clock speedup vs A5 adaptive baseline ({:.0} s): {:.2}x",
        A5_ADAPTIVE_SECS,
        A5_ADAPTIVE_SECS / secs
    );
    assert_eq!(summary.snapshots_emitted, n_outputs + 1);

    // (3) Realized CFL dynamic range across the run (post-hoc, no instrumentation).
    let params = HydroParams {
        eos: Eos::Isothermal {
            c_s: full.sound_speed.unwrap(),
        },
        ..HydroParams::default()
    };
    let cfg = DensityConfig::default();
    let bounds: Vec<f64> = snap_files(&out)
        .iter()
        .map(|f| {
            let (_, st) = read_file(f).unwrap();
            max_stable_dt(&st, &params, &cfg, 1.0)
        })
        .filter(|b| b.is_finite() && *b > 0.0)
        .collect();
    let (bmin, bmax) = bounds
        .iter()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &b| {
            (lo.min(b), hi.max(b))
        });
    eprintln!(
        "  CFL bound (c_cfl=1) across {} snapshots: min {bmin:.4e}  max {bmax:.4e}  dynamic range {:.1}x",
        bounds.len(),
        bmax / bmin
    );
    eprintln!(
        "  snapshots RETAINED at {} (for the FULL grav-rung-spread record gate)",
        out.display()
    );

    // (2) Short-prefix convergence to a finer-courant reference (bounded prefix —
    // the chaotic showpiece does not admit a full-trajectory match).
    let prefix = |courant: f64, tag: &str| -> Vec<[f64; 3]> {
        let mut s = gasrich_individual(false, courant, 14);
        s.n_steps = 3 * s.snapshot_every; // 3 output intervals
        let d = tempdir(tag);
        simulate_snapshots(&s, &d, Backend::Cpu).expect("prefix run completes");
        let (_, last) = read_file(snap_files(&d).last().unwrap()).unwrap();
        last.pos.iter().map(|p| [p.x, p.y, p.z]).collect()
    };
    let reference = prefix(0.0625, "prefix_ref");
    let err = |a: &[[f64; 3]]| {
        a.iter()
            .zip(&reference)
            .map(|(p, r)| {
                let d = [p[0] - r[0], p[1] - r[1], p[2] - r[2]];
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .fold(0.0_f64, f64::max)
    };
    let e_coarse = err(&prefix(0.25, "prefix_coarse"));
    let e_fine = err(&prefix(0.125, "prefix_fine"));
    eprintln!("  prefix convergence: err(0.25) = {e_coarse:.3e}, err(0.125) = {e_fine:.3e}");
    assert!(
        e_fine < e_coarse,
        "halving courant must reduce the error toward the finer reference"
    );
}
