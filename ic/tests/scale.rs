//! Scale smoke-test for the Barnes-Hut workhorse (DESIGN.md M1: "10^5–10^6,
//! CPU Barnes-Hut"). The correctness tests run at N≤512, where a tree is *slower*
//! than direct summation — so they never exercise the one property BH exists for:
//! O(N log N) force evaluation that beats O(N²) at galaxy scale.
//!
//! This test, on a realistic centrally-concentrated Plummer sphere at N≈30k,
//! asserts the two things no N=512 unit test can:
//!   1. BarnesHut(θ=0.5) forces still match the exact oracle to the RMS bound
//!      *at scale* (the deep, unbalanced tree a clustered sphere builds is the
//!      real stress on the opening criterion);
//!   2. BarnesHut is actually FASTER than DirectSum here — the crossover has been
//!      passed, so the per-node work (incl. the Barnes-1994 criterion's sqrt) and
//!      the per-step tree rebuild have not eaten the asymptotic win.
//!
//! Ignored by default (it is a perf check and DirectSum is O(N²)). Run with:
//!   cargo test -p galaxy-ic --release --test scale -- --ignored --nocapture

use std::time::Instant;

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_ic::Plummer;
use galaxy_solvers::{BarnesHut, DirectSum};

fn accel(solver: &mut dyn ForceSolver, s: &State) -> Vec<DVec3> {
    let mut a = vec![DVec3::ZERO; s.len()];
    solver.accelerations(s, &mut a);
    a
}

/// RMS per-particle force error, normalized by the RMS acceleration.
fn rms_rel_err(approx: &[DVec3], exact: &[DVec3]) -> f64 {
    let n = exact.len() as f64;
    let rms = (exact.iter().map(|a| a.length_squared()).sum::<f64>() / n)
        .sqrt()
        .max(1e-300);
    let err_ms = approx
        .iter()
        .zip(exact)
        .map(|(b, d)| (*b - *d).length_squared())
        .sum::<f64>()
        / n;
    err_ms.sqrt() / rms
}

#[test]
#[ignore = "perf/scale smoke-test (O(N^2) oracle); run with --release -- --ignored"]
fn barnes_hut_beats_oracle_and_matches_it_at_scale() {
    let model = Plummer::new(1.0, 1.0, 1.0);
    let n = 30_000;
    let eps = 0.05 * model.scale_radius;
    let s = model.sample(n, 0xA11CE5);

    let mut ds = DirectSum::new(model.g, eps);
    let t0 = Instant::now();
    let exact = accel(&mut ds, &s);
    let ds_time = t0.elapsed();

    let mut bh = BarnesHut::new(model.g, eps, 0.5);
    let t1 = Instant::now();
    let approx = accel(&mut bh, &s);
    let bh_time = t1.elapsed();

    let rms = rms_rel_err(&approx, &exact);
    let speedup = ds_time.as_secs_f64() / bh_time.as_secs_f64();
    eprintln!(
        "N={n}: DirectSum {ds_time:?}, BarnesHut {bh_time:?} -> {speedup:.1}x faster, RMS force err {rms:e}"
    );

    // Correctness at scale: the same θ=0.5 budget as the equilibrium test holds on
    // the larger, deeper, more clustered tree — measured ≈1.7e-3 RMS, comfortably
    // under 1% (the monopole O(θ²) error; θ=0.5 sits between the θ=0.3≈0.3% and
    // θ=0.6≈1.5% RMS of the N=120 sweep). Bound carries ~6× headroom.
    assert!(rms < 0.01, "BH RMS force error at N={n}: {rms:e}");

    // The reason BarnesHut exists: at this scale the tree must beat direct sum.
    // (Use a clear margin, not just <, so a near-tie still flags a perf regression
    // in the per-node criterion or the per-step rebuild.)
    assert!(
        bh_time.as_secs_f64() * 2.0 < ds_time.as_secs_f64(),
        "BarnesHut ({bh_time:?}) must be comfortably faster than DirectSum ({ds_time:?}) at N={n}"
    );
}
