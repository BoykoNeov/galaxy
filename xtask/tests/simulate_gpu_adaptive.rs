//! GPU-SPH adaptive-dt gate (plan: courant-quickening-cadence.md, milestone A3).
//!
//! [`simulate_gas_gpu_adaptive`] drives the resident stepper block-adaptively: the
//! timestep is recomputed per block from the on-device CFL bound (`min_stable_dt`)
//! instead of a fixed `dt`, sharing the CPU path's [`plan_block`] block-sizing.
//!
//! GATE INTENT (D4) — adaptive dt BREAKS the GPU-vs-CPU trajectory oracle (the f32 GPU
//! bound and f64 CPU bound pick different dt each block → divergent trajectories), so
//! this milestone does NOT compare adaptive trajectories across paths. The three
//! decoupled gates are:
//!   * (a) bound-agreement CPU `max_stable_dt` vs GPU `min_stable_dt` at a FIXED state —
//!     already gated by G4/G5c (`gpu/tests/sph_cfl.rs`); not duplicated here.
//!   * (b) the fixed-dt GPU-vs-CPU force/step oracle stays green (G6, unchanged).
//!   * (c) adaptivity correctness is PER-PATH: the GPU adaptive run converges to a
//!     FINER-courant GPU reference (GPU-vs-GPU, so the common-mode f32 error cancels and
//!     the difference isolates the dt/adaptivity error). Asserted as monotone error
//!     decrease + a generous absolute cap, NOT a numeric order factor (variable-dt
//!     leapfrog is between 1st and 2nd order). The blob COMPRESSES so the CFL bound moves.
//!
//! GPU-gated: needs a wgpu adapter; without one the branch errors and these fail loudly.

use galaxy_core::{DVec3, Species, State};
use galaxy_io::read_file;
use galaxy_sim::AdaptiveConfig;
use galaxy_solvers::sph::{DensityConfig, Eos, HydroParams};
use galaxy_xtask::simulate::simulate_gas_gpu_adaptive;

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// An all-gas ball with a radially CONVERGING velocity field `v = −k·x`, so it
/// compresses and the CFL bound genuinely moves across the run — the testbed the
/// per-path convergence gate needs (a static blob would test fixed-dt in disguise).
fn converging_gas_blob(seed: u64, n: usize, radius: f64, k: f64) -> State {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    let vel: Vec<DVec3> = pos.iter().map(|&p| -k * p).collect();
    let mut s = State::from_phase_space(pos, vel, vec![0.01; n]); // light ⇒ hydro-led, tame gravity
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    s
}

fn adaptive_cfg(courant: f64, output_dt: f64, n_outputs: u64) -> AdaptiveConfig {
    AdaptiveConfig {
        courant,
        max_growth: 1.25,
        block_steps: 16,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0xADA9_67D0,
        config_hash: 0xC0FFEE,
        units: "nbody-G1".to_string(),
    }
}

fn snap_paths(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "snap"))
        .collect();
    v.sort();
    v
}

fn gas_positions(state: &State) -> Vec<DVec3> {
    (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .map(|i| state.pos[i])
        .collect()
}

fn tempdir(tag: &str) -> std::path::PathBuf {
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "sim_gpu_adaptive_{tag}_{:?}",
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&base).unwrap();
    base
}

/// D4(c): the GPU adaptive run converges to a finer-courant GPU reference as courant
/// shrinks — the per-path adaptivity-correctness gate.
#[test]
fn gpu_adaptive_converges_to_finer_courant_reference() {
    let ic = converging_gas_blob(0xB10B, 300, 1.0, 0.6);
    let output_dt = 0.02;
    let n_outputs = 6; // full horizon, not a prefix

    let final_gas = |courant: f64, tag: &str| -> Vec<DVec3> {
        let dir = tempdir(tag);
        simulate_gas_gpu_adaptive(
            &ic,
            &adaptive_cfg(courant, output_dt, n_outputs),
            HydroParams {
                eos: Eos::Isothermal { c_s: 1.0 },
                ..HydroParams::default()
            },
            DensityConfig::default(),
            &dir,
        )
        .expect("gpu adaptive run (needs a wgpu adapter)");
        let (_, last) = read_file(snap_paths(&dir).last().expect("a snapshot")).unwrap();
        gas_positions(&last)
    };

    let reference = final_gas(0.02, "ref");
    let err = |a: &[DVec3]| {
        a.iter()
            .zip(&reference)
            .map(|(p, r)| (*p - *r).length())
            .fold(0.0_f64, f64::max)
    };
    let e_coarse = err(&final_gas(0.2, "coarse"));
    let e_fine = err(&final_gas(0.1, "fine"));
    eprintln!("A3 GPU per-path convergence: err(0.2) = {e_coarse:.3e}, err(0.1) = {e_fine:.3e}");

    assert!(
        e_fine < e_coarse,
        "halving courant must reduce the GPU error toward the finer reference: \
         err(0.1) = {e_fine:e} !< err(0.2) = {e_coarse:e}"
    );
    assert!(
        e_coarse < 0.1,
        "even the coarse GPU adaptive run must track the reference within a blob radius: \
         err(0.2) = {e_coarse:e}"
    );
}

/// Snapshots land on the output time grid; header step = output index, time = k·output_dt.
#[test]
fn gpu_adaptive_snapshots_land_on_the_output_time_grid() {
    let ic = converging_gas_blob(0xCADE, 250, 1.0, 0.5);
    let dir = tempdir("cadence");
    let c = adaptive_cfg(0.25, 0.02, 5);
    let summary = simulate_gas_gpu_adaptive(
        &ic,
        &c,
        HydroParams {
            eos: Eos::Isothermal { c_s: 1.0 },
            ..HydroParams::default()
        },
        DensityConfig::default(),
        &dir,
    )
    .expect("gpu adaptive run (needs a wgpu adapter)");

    assert_eq!(summary.snapshots_emitted, 6, "IC + one per output interval");
    let paths = snap_paths(&dir);
    assert_eq!(paths.len(), 6);
    for (k, p) in paths.iter().enumerate() {
        let (h, st) = read_file(p).unwrap();
        assert_eq!(h.step, k as u64, "snapshot step must be the output index");
        let want = k as f64 * c.output_dt;
        assert!(
            (h.time - want).abs() < 1e-9,
            "header time {} != output index {} · output_dt",
            h.time,
            k
        );
        assert_eq!(h.time, st.time, "header/state time disagree");
        // Column re-attach survived (gas subset non-empty).
        assert!(
            st.kind.contains(&Species::Gas),
            "gas subset lost — kind not re-attached after snapshot()"
        );
    }
    assert!((summary.final_time - 5.0 * c.output_dt).abs() < 1e-9);
}

/// A gas-free IC has no finite CFL bound (`min_stable_dt = +∞`) ⇒ the adaptive branch
/// errors rather than stepping with an undefined dt.
#[test]
fn gpu_adaptive_rejects_gas_free_state() {
    let mut rng = lcg(9);
    let pos: Vec<DVec3> = (0..64)
        .map(|_| DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 2.0)
        .collect();
    let gas_free = State::from_phase_space(pos, vec![DVec3::ZERO; 64], vec![1.0; 64]); // all Collisionless
    let dir = tempdir("gasfree");
    let r = simulate_gas_gpu_adaptive(
        &gas_free,
        &adaptive_cfg(0.25, 0.02, 3),
        HydroParams::default(),
        DensityConfig::default(),
        &dir,
    );
    assert!(
        r.is_err(),
        "gas-free adaptive run must error (no finite CFL bound)"
    );
}
