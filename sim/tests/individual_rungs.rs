//! I2 — power-of-two rung assignment (pure, no stepping).
//!
//! Each particle sits on a rung `r_i` below a base timestep `dt_base`, with
//! sub-step `dt_base / 2^r_i ≤ courant · dt_i` (its safe step). `r_i = clamp(⌈log2(
//! dt_base/(courant·dt_i))⌉, 0, r_max)`, computed by an exact integer search (no
//! float `log2` rounding at power-of-two boundaries). Collisionless `dt = +∞` →
//! the coarsest rung 0. All rungs synchronize at each `dt_base` boundary.

use galaxy_sim::individual::{assign_rungs, base_dt, limit_rungs, rung_step};

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

// --------------------------------------------------------------------------
// I4b — the Saitoh–Makino rung limiter (pure fixpoint, no stepping). After CFL
// assignment, no coupled pair may differ by more than `n_limit` rungs; the coarser
// particle is refined (raised). Monotone (never coarsens) ⇒ the fixpoint converges.
// --------------------------------------------------------------------------

#[test]
fn non_binding_n_limit_leaves_rungs_untouched() {
    // A wide spread with n_limit ≥ the spread: every pair already satisfies the
    // constraint ⇒ no particle is refined (the I4a / fixed-dt-disguise setting).
    let mut rungs = vec![0, 2, 4, 7];
    let pairs = [(0, 1), (1, 2), (2, 3), (0, 3)];
    let before = rungs.clone();
    limit_rungs(&mut rungs, &pairs, 7);
    assert_eq!(rungs, before, "n_limit ≥ spread must be a no-op");
}

#[test]
fn one_hop_refines_the_coarse_neighbour() {
    // A lone coarse particle (rung 0) coupled to a fine one (rung 5), n_limit = 1 ⇒
    // the coarse is raised to within 1 rung (4); the fine is never coarsened.
    let mut rungs = vec![0, 5];
    limit_rungs(&mut rungs, &[(0, 1)], 1);
    assert_eq!(
        rungs,
        vec![4, 5],
        "coarse neighbour must wake to r_max−n_limit"
    );
}

#[test]
fn refinement_propagates_along_a_chain_to_a_fixpoint() {
    // The load-bearing multi-hop case: a fine spike at one end of a coupling chain
    // must grade DOWN across the whole chain (each hop at most n_limit apart), which
    // only a fixpoint (not a single pass) achieves. Chain 0-1-2-3, rung 5 at node 3,
    // n_limit = 1 ⇒ [2,3,4,5] (hand-derived: r_i ≥ 5 − (3−i)).
    let mut rungs = vec![0, 0, 0, 5];
    let pairs = [(0, 1), (1, 2), (2, 3)];
    limit_rungs(&mut rungs, &pairs, 1);
    assert_eq!(
        rungs,
        vec![2, 3, 4, 5],
        "fineness must grade along the chain"
    );
}

#[test]
fn limiter_only_ever_refines_never_coarsens() {
    // Every output rung is ≥ its input (the monotonicity that guarantees convergence
    // and means the limiter can only make steps SAFER, never coarser).
    let input = vec![3, 1, 6, 0, 2, 5];
    let pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)];
    let mut rungs = input.clone();
    limit_rungs(&mut rungs, &pairs, 1);
    for (i, (&out, &inp)) in rungs.iter().zip(&input).enumerate() {
        assert!(out >= inp, "i={i}: limiter coarsened {inp}→{out}");
    }
    // And the constraint now holds on every pair.
    for &(i, j) in &pairs {
        let d = rungs[i].abs_diff(rungs[j]);
        assert!(d <= 1, "pair ({i},{j}) still violates n_limit=1: |Δ|={d}");
    }
}

#[test]
fn pair_order_is_symmetric() {
    // (i,j) and (j,i) impose the same constraint — the result must not depend on
    // which way a coupled pair is listed.
    let mut a = vec![0, 5];
    let mut b = vec![0, 5];
    limit_rungs(&mut a, &[(0, 1)], 1);
    limit_rungs(&mut b, &[(1, 0)], 1);
    assert_eq!(a, b, "limiter must be symmetric in pair order");
}

#[test]
fn n_limit_zero_forces_a_connected_component_to_one_rung() {
    // n_limit = 0 ⇒ coupled particles must share a rung; a connected component
    // collapses to its MAX (raise-only). Two disjoint components stay independent.
    let mut rungs = vec![0, 3, 0, /* isolated */ 1, 7];
    // Component A: 0-1-2 (max 3). Component B: 3-4 (max 7).
    let pairs = [(0, 1), (1, 2), (3, 4)];
    limit_rungs(&mut rungs, &pairs, 0);
    assert_eq!(
        rungs,
        vec![3, 3, 3, 7, 7],
        "n_limit=0 ⇒ component → its max"
    );
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
