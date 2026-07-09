//! I2 — power-of-two rung assignment (pure, no stepping).
//!
//! Each particle sits on a rung `r_i` below a base timestep `dt_base`, with
//! sub-step `dt_base / 2^r_i ≤ courant · dt_i` (its safe step). `r_i = clamp(⌈log2(
//! dt_base/(courant·dt_i))⌉, 0, r_max)`, computed by an exact integer search (no
//! float `log2` rounding at power-of-two boundaries). Collisionless `dt = +∞` →
//! the coarsest rung 0. All rungs synchronize at each `dt_base` boundary.

use galaxy_sim::individual::{assign_rungs, base_dt, rung_step};

const COURANT: f64 = 0.25;

#[test]
fn uniform_cfl_maps_every_particle_to_one_rung() {
    // A uniform-CFL state: every dt_i identical ⇒ every particle on the SAME rung.
    let dt = vec![0.37_f64; 64];
    let base = base_dt(&dt, COURANT, f64::INFINITY);
    let rungs = assign_rungs(&dt, base, COURANT, 20);
    assert!(
        rungs.iter().all(|&r| r == rungs[0]),
        "uniform ⇒ one rung: {rungs:?}"
    );
    // With dt_base = courant·max(dt) = courant·dt, the coarsest particle IS every
    // particle, so that single rung is 0.
    assert_eq!(rungs[0], 0, "uniform coarsest ⇒ rung 0");
}

#[test]
fn rung_is_monotone_nondecreasing_in_inverse_dt() {
    // Smaller dt_i (needs a finer step) ⇒ higher-or-equal rung. Feed a strictly
    // DEcreasing dt sequence and assert the rungs are non-decreasing.
    let dt: Vec<f64> = (1..=200).rev().map(|k| k as f64 * 0.01).collect(); // 2.0 → 0.01
    let base = base_dt(&dt, COURANT, f64::INFINITY);
    let rungs = assign_rungs(&dt, base, COURANT, 30);
    for w in rungs.windows(2) {
        assert!(
            w[1] >= w[0],
            "rungs must be monotone in 1/dt: {} then {}",
            w[0],
            w[1]
        );
    }
    // The spread is real (not all one rung) over a 200× dt range.
    assert!(
        *rungs.last().unwrap() > rungs[0],
        "a 200× dt range must span rungs"
    );
}

#[test]
fn rungs_are_clamped_to_zero_and_r_max() {
    let r_max = 8;
    let base = 1.0;
    // dt_i ≥ dt_base/courant ⇒ safe step ≥ dt_base ⇒ rung 0 (can't go coarser).
    // dt_i → 0 ⇒ rung clamps at r_max. +∞ (collisionless) ⇒ rung 0.
    let dt = vec![
        10.0,          // huge ⇒ rung 0
        f64::INFINITY, // collisionless ⇒ coarsest rung 0
        1e-9,          // tiny ⇒ clamps at r_max
    ];
    let rungs = assign_rungs(&dt, base, COURANT, r_max);
    assert_eq!(rungs[0], 0, "huge dt ⇒ rung 0");
    assert_eq!(rungs[1], 0, "collisionless +∞ ⇒ coarsest rung 0");
    assert_eq!(rungs[2], r_max, "tiny dt ⇒ clamped at r_max");
}

#[test]
fn rung_values_match_hand_derived_ceil_log2() {
    // Pin the exact ceil-log2 binning (incl. the power-of-two boundary, where a
    // float log2 could land on either side). dt_base = 1, courant = 1 ⇒ target_i =
    // dt_i, rung = smallest r with 1/2^r ≤ dt_i.
    let base = 1.0;
    let c = 1.0;
    let cases = [
        (1.0, 0u32), // 1/1 = 1 ≤ 1 at r=0
        (0.6, 1),    // r=0: 1>0.6; r=1: 0.5≤0.6
        (0.5, 1),    // exact boundary: 0.5 ≤ 0.5 at r=1 (NOT r=0: 1>0.5)
        (0.4, 2),    // r=1: 0.5>0.4; r=2: 0.25≤0.4
        (0.25, 2),   // exact boundary: 0.25 ≤ 0.25 at r=2
        (0.26, 2),   // 0.25 ≤ 0.26 at r=2
        (2.0, 0),    // safe step ≥ base ⇒ rung 0
    ];
    for (dt_i, want) in cases {
        let got = assign_rungs(&[dt_i], base, c, 30)[0];
        assert_eq!(got, want, "dt_i={dt_i}: rung {got}, want {want}");
    }

    // courant genuinely shifts the ladder: at courant=0.5 the safe step halves, so
    // dt_i=1.0 needs 0.5 ≤ 0.5 ⇒ rung 1 (vs rung 0 at courant=1).
    assert_eq!(
        assign_rungs(&[1.0], 1.0, 0.5, 30)[0],
        1,
        "courant 0.5 shifts dt_i=1 to rung 1"
    );
}

#[test]
fn every_finite_rung_fits_and_is_tight() {
    // The defining property: for each finite dt_i the sub-step FITS
    // (dt_base/2^r ≤ courant·dt_i) and, unless clamped at r_max or already rung 0,
    // is TIGHT (one rung coarser would overshoot: dt_base/2^(r-1) > courant·dt_i).
    // This characterizes the exact ceil-log2 without invoking log2 in the test.
    let dt: Vec<f64> = (1..=300).map(|k| k as f64 * 0.013).collect();
    let r_max = 30;
    let base = base_dt(&dt, COURANT, f64::INFINITY);
    let rungs = assign_rungs(&dt, base, COURANT, r_max);
    for (i, (&dt_i, &r)) in dt.iter().zip(&rungs).enumerate() {
        let target = COURANT * dt_i;
        let step = rung_step(base, r);
        assert!(
            step <= target * (1.0 + 1e-12),
            "i={i}: step {step} must fit target {target}"
        );
        if r > 0 && r < r_max {
            let coarser = rung_step(base, r - 1);
            assert!(
                coarser > target,
                "i={i}: rung {r} not tight — coarser step {coarser} also fits target {target}"
            );
        }
    }
}

#[test]
fn base_dt_is_courant_scaled_coarsest_capped() {
    // dt_base is set by the COARSEST particle (largest finite dt), courant-scaled,
    // then capped. +∞ rows are ignored (collisionless don't set the base).
    let dt = vec![0.2, f64::INFINITY, 0.8, 0.5];
    // Uncapped: courant · max_finite = 0.25 · 0.8 = 0.2.
    let uncapped = base_dt(&dt, COURANT, f64::INFINITY);
    assert!(
        (uncapped - 0.2).abs() < 1e-15,
        "base_dt {uncapped}, want 0.2"
    );
    // A tighter cap wins.
    let capped = base_dt(&dt, COURANT, 0.1);
    assert!(
        (capped - 0.1).abs() < 1e-15,
        "cap must clamp base_dt: {capped}"
    );
}
