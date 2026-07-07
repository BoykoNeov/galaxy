//! A5 — producibility validation for block-adaptive dt (plan
//! courant-quickening-cadence). The "real done": the gasrich showpiece, which is
//! UNPRODUCIBLE at any verified fixed dt (`settling-cinder-vigil` Finding A — the
//! merger-wide-minimum CFL bound is unknown a priori and declines <0.002
//! unpredictably), COMPLETES under adaptive dt because the run tracks the bound down
//! automatically and never CFL-aborts.
//!
//! Two gates:
//!   * `gasrich_quick_adaptive_completes` (always-on): on the REAL gasrich preset at
//!     QUICK size, the adaptive path completes a bounded run (an output interval of
//!     CFL-safe substeps) landing on the output grid — the real-preset smoke that the
//!     adaptive wiring (GravitySph + the CFL substep loop) works end-to-end. The
//!     fixed-dt abort CONTRAST is not asserted here: this seed's CFL bound only dips
//!     below the shipped dt=0.005 near pericenter (~step 250, `settling-cinder-vigil`),
//!     past a gate-cheap bounded run — so the producibility contrast lives in A4's cheap
//!     synthetic gate and in the deferred full-res test below.
//!   * `full_res_gasrich_adaptive_completes_and_converges` (#[ignore], the deferred A5
//!     run — `--release --ignored`): the full-res showpiece to completion (Finding A
//!     discharged), the fixed-dt abort contrast (the fixed path CFL-aborts before
//!     pericenter), a short-prefix convergence check to a finer-courant reference (the
//!     chaotic-showpiece analogue of D5 — no full-trajectory match), and a post-hoc
//!     measurement of the CFL bound's realized dynamic range across the 61 snapshots
//!     (the "size the win" number, measured not estimated; calibrates courant/block/growth).

use galaxy_io::read_file;
use galaxy_solvers::sph::{max_stable_dt, DensityConfig, HydroParams};
use galaxy_xtask::simulate::{simulate_snapshots, Backend};
use galaxy_xtask::spec::{build_scenario, parse_scenario_toml, preset, Scenario};

fn gasrich(quick: bool) -> Scenario {
    build_scenario(
        &parse_scenario_toml(preset("gasrich").expect("gasrich preset")).expect("gasrich parses"),
        quick,
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
        "adaptive_prod_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// The REAL gasrich preset, adaptive path: completes a bounded run landing on the output
/// grid. Real-preset smoke that the adaptive wiring works end-to-end (a synthetic scenario
/// proves the fixed-abort contrast cheaply in A4; this pins the shipped preset itself).
#[test]
fn gasrich_quick_adaptive_completes() {
    let mut s = gasrich(true); // QUICK
    assert!(s.adaptive.is_some(), "gasrich ships adaptive-enabled");
    assert!(s.sound_speed.is_some(), "gasrich carries gas");
    // One output interval (snapshot_every = 100 ⇒ n_steps = 100): IC + 1 = 2 snapshots,
    // but the interval integrates ~a hundred CFL-safe adaptive substeps.
    s.n_steps = s.snapshot_every;

    let dir = tempdir("gasrich_adapt");
    let summary = simulate_snapshots(&s, &dir, Backend::Cpu)
        .expect("adaptive gasrich (real preset) must complete a bounded run");
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

/// DEFERRED A5 (run: `cargo test -p galaxy-xtask --release -- --ignored
/// full_res_gasrich_adaptive`). The full-res showpiece end-to-end: (1) COMPLETES — the
/// producibility proof Finding A demands (>30 min under any completing dt; the fixed-dt
/// preset CFL-aborts before pericenter); (2) short-prefix convergence to a finer-courant
/// reference (the chaotic-showpiece gate — no full-trajectory match, D5); (3) records the
/// CFL bound's realized dynamic range across the snapshots (measured, not estimated).
///
/// ALSO fold into the deferred A5 session (advisor #2): drive a QUICK `run_movie` on the
/// flipped gasrich preset end-to-end (`cargo run -p galaxy-xtask -- movie gasrich <out>`
/// with `GALAXY_MOVIE_QUICK=1`) — the preset flip is a shipped behavior change, and the
/// renderprep→render pipeline is exercised by no unit test (only `simulate_snapshots` is).
/// Low risk (snapshots consumed generically by sorted-glob + time), but it is the real
/// "the shipped showpiece still renders" verification.
#[test]
#[ignore = "deferred A5: full-res gasrich adaptive is >30 min; run with --release --ignored"]
fn full_res_gasrich_adaptive_completes_and_converges() {
    // (0) CONTRAST: the fixed-dt full-res path CFL-aborts before pericenter (Finding A —
    // its shipped dt = 0.005 exceeds the bound by ~step 250). This is what adaptive fixes.
    let mut fixed = gasrich(false);
    fixed.adaptive = None;
    let dir_fixed = tempdir("fullres_fixed_abort");
    assert!(
        simulate_snapshots(&fixed, &dir_fixed, Backend::Cpu).is_err(),
        "fixed-dt full-res gasrich must CFL-abort (the unproducibility adaptive fixes)"
    );

    // (1) COMPLETES — the whole point. Full-res, full horizon, shipped courant.
    let full = gasrich(false);
    assert!(full.adaptive.is_some());
    let n_outputs = full.n_steps / full.snapshot_every; // 60 ⇒ 61 snapshots
    let dir = tempdir("fullres_complete");
    let t0 = std::time::Instant::now();
    let summary = simulate_snapshots(&full, &dir, Backend::Cpu)
        .expect("full-res gasrich adaptive must COMPLETE (Finding A discharged)");
    eprintln!(
        "=== A5: full-res gasrich adaptive COMPLETED in {:.1} s ({} snapshots) ===",
        t0.elapsed().as_secs_f64(),
        summary.snapshots_emitted
    );
    assert_eq!(summary.snapshots_emitted, n_outputs + 1);

    // (3) Realized dynamic range of the CFL bound across the run: post-hoc, no loop
    // instrumentation — recompute the c_cfl=1 limit at each emitted snapshot.
    let params = HydroParams {
        sound_speed: full.sound_speed.unwrap(),
        ..HydroParams::default()
    };
    let cfg = DensityConfig::default();
    let bounds: Vec<f64> = snap_files(&dir)
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

    // (2) Short-prefix convergence to a finer-courant reference (chaotic showpiece: a
    // bounded prefix, not the full horizon). Coarse must track the finer reference.
    let prefix = |courant: f64, tag: &str| -> Vec<[f64; 3]> {
        let mut s = gasrich(false);
        s.n_steps = 3 * s.snapshot_every; // 3 output intervals
        if let Some(a) = s.adaptive.as_mut() {
            a.courant = courant;
        }
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
