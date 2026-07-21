//! S2 (RED) — the pure `form_stars` operator and its recipe (plan
//! `natal-ember-forge.md`, F2/F3). These gates pin the whole-particle in-place
//! conversion: SF-off / empty-gas make no conversions; mass & N are conserved
//! exactly; the draw is deterministic and order-independent; conversion is
//! one-way; the two-part threshold + converging-flow criterion selects
//! correctly; and a uniform box converts at the analytic
//! `p = 1 − exp(−ε·dt/t_ff)`, `t_ff = √(3π/32Gρ)`, to sampling tolerance.

use galaxy_core::{DVec3, Progenitor, Species, State};
use galaxy_sim::star_formation::{form_stars, StarFormationConfig};

/// Build a state of `n` gas particles at the origin with sequential ids,
/// distinct nonzero `u`, and a stamped `progenitor` tag so we can assert those
/// columns are untouched by conversion. `time` is set so a formed star's
/// `formation_time` is a recognizable non-sentinel value.
fn gas_state(n: usize, time: f64) -> State {
    let pos = (0..n).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
    let vel = (0..n).map(|i| DVec3::new(0.0, i as f64, 0.0)).collect();
    let mass = (0..n).map(|i| 1.0 + i as f64).collect();
    let mut s = State::from_phase_space(pos, vel, mass);
    for i in 0..n {
        s.kind[i] = Species::Gas;
        s.u[i] = 10.0 + i as f64; // nonzero, so we can see it zeroed on conversion
        s.progenitor[i] = Progenitor(4); // gas provenance tag — must survive
    }
    s.time = time;
    s
}

/// The recipe's free-fall time and per-particle conversion probability,
/// hand-derived here (independent of the operator) for the calibration gate.
fn t_ff(rho: f64) -> f64 {
    (3.0 * std::f64::consts::PI / (32.0 * rho)).sqrt() // G = 1
}
fn p_convert(eff: f64, dt: f64, rho: f64) -> f64 {
    1.0 - (-eff * dt / t_ff(rho)).exp()
}

/// Ids that flipped from Gas (`before`) to Collisionless (`after`).
fn converted_ids(before: &State, after: &State) -> Vec<u64> {
    (0..after.len())
        .filter(|&i| before.kind[i] == Species::Gas && after.kind[i] == Species::Collisionless)
        .map(|i| after.id[i].0)
        .collect()
}

#[test]
fn efficiency_zero_makes_no_conversions() {
    let mut s = gas_state(100, 5.0);
    let rho = vec![1e6; 100]; // far above threshold
    let div_v = vec![-1.0; 100]; // converging
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 0.0, // p = 1 − exp(0) = 0
        seed: 0xABCD,
    };
    let summary = form_stars(&mut s, &rho, &div_v, 1.0, &cfg, 0);
    assert_eq!(summary.n_formed, 0, "efficiency 0 must form no stars");
    assert_eq!(summary.mass_formed, 0.0);
    assert!(s.kind.iter().all(|&k| k == Species::Gas), "no gas converted");
}

#[test]
fn empty_gas_makes_no_conversions() {
    // All-collisionless input: nothing is a candidate regardless of ρ / ∇·v.
    let mut s = State::from_phase_space(
        vec![DVec3::ZERO; 50],
        vec![DVec3::ZERO; 50],
        vec![2.0; 50],
    );
    let rho = vec![1e6; 50];
    let div_v = vec![-1.0; 50];
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 1.0,
        seed: 7,
    };
    let summary = form_stars(&mut s, &rho, &div_v, 1.0, &cfg, 0);
    assert_eq!(summary.n_formed, 0, "no gas ⇒ no formation");
}

#[test]
fn conversion_conserves_mass_and_count() {
    let n = 400;
    let mut s = gas_state(n, 12.5);
    let before = s.clone();
    let rho = vec![5.0; n]; // above threshold
    let div_v = vec![-1.0; n]; // all converging
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 3.0, // large ⇒ a healthy fraction converts, but not all
        seed: 0x1234_5678,
    };
    let total_mass_before: f64 = before.mass.iter().sum();

    let summary = form_stars(&mut s, &rho, &div_v, 0.2, &cfg, 0);

    // Some but not all — otherwise the "conserves" claim is vacuous.
    assert!(summary.n_formed > 0 && summary.n_formed < n, "a proper fraction converts");

    // N and total mass are conserved EXACTLY (whole-particle in-place flip).
    assert_eq!(s.len(), n, "particle count unchanged");
    let total_mass_after: f64 = s.mass.iter().sum();
    assert_eq!(total_mass_after, total_mass_before, "total mass conserved exactly");

    // Species bookkeeping: gas lost == stars gained == n_formed.
    let stars_after = s.kind.iter().filter(|&&k| k == Species::Collisionless).count();
    assert_eq!(stars_after, summary.n_formed, "star count == n_formed");
    let gas_after = s.kind.iter().filter(|&&k| k == Species::Gas).count();
    assert_eq!(gas_after, n - summary.n_formed, "gas count dropped by n_formed");

    // mass_formed is the sum over the converted set.
    let expected_mass: f64 = (0..n)
        .filter(|&i| s.kind[i] == Species::Collisionless)
        .map(|i| s.mass[i])
        .sum();
    assert_eq!(summary.mass_formed, expected_mass, "mass_formed == Σ converted mass");

    // Per-particle: conversion touches EXACTLY kind, formation_time, u.
    for i in 0..n {
        // pos / vel / mass / id / progenitor untouched for everyone.
        assert_eq!(s.pos[i], before.pos[i], "pos untouched at {i}");
        assert_eq!(s.vel[i], before.vel[i], "vel untouched at {i}");
        assert_eq!(s.mass[i], before.mass[i], "mass untouched at {i}");
        assert_eq!(s.id[i], before.id[i], "id untouched at {i}");
        assert_eq!(s.progenitor[i], before.progenitor[i], "progenitor tag survives at {i}");
        if s.kind[i] == Species::Collisionless {
            assert_eq!(s.formation_time[i], s.time, "formed star stamped with state.time at {i}");
            assert_eq!(s.u[i], 0.0, "converted row's u zeroed at {i}");
        } else {
            assert_eq!(s.formation_time[i], State::PRIMORDIAL, "unconverted keeps sentinel at {i}");
            assert_eq!(s.u[i], before.u[i], "unconverted u untouched at {i}");
        }
    }
}

#[test]
fn determinism_same_seed_order_independent() {
    let n = 300;
    let base = gas_state(n, 3.0);
    let rho: Vec<f64> = (0..n).map(|i| 2.0 + (i % 5) as f64).collect();
    let div_v = vec![-0.5; n];
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 2.0,
        seed: 0xDEAD_BEEF,
    };

    // Run 1: natural order.
    let mut s1 = base.clone();
    let b1 = s1.clone();
    form_stars(&mut s1, &rho, &div_v, 0.15, &cfg, 0);
    let mut ids1 = converted_ids(&b1, &s1);
    ids1.sort_unstable();

    // Run 2: REVERSED order — every SoA column AND the parallel ρ / ∇·v arrays
    // co-permuted, so index i still lines up. The draw is id-keyed, so the
    // converted id-set must be identical despite the different iteration order.
    let mut s2 = base.clone();
    s2.pos.reverse();
    s2.vel.reverse();
    s2.mass.reverse();
    s2.id.reverse();
    s2.progenitor.reverse();
    s2.kind.reverse();
    s2.u.reverse();
    s2.formation_time.reverse();
    let mut rho2 = rho.clone();
    rho2.reverse();
    let mut div2 = div_v.clone();
    div2.reverse();
    let b2 = s2.clone();
    form_stars(&mut s2, &rho2, &div2, 0.15, &cfg, 0);
    let mut ids2 = converted_ids(&b2, &s2);
    ids2.sort_unstable();

    assert_eq!(ids1, ids2, "same seed ⇒ same converted id-set, order-independent");
    assert!(!ids1.is_empty(), "the test converts a nonempty set (else vacuous)");
}

#[test]
fn different_epoch_draws_independent_stream() {
    // Same seed, different epoch ⇒ (generally) a different converted set, so
    // successive SF calls are not locked to the same particles.
    let n = 500;
    let base = gas_state(n, 1.0);
    let rho = vec![3.0; n];
    let div_v = vec![-1.0; n];
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 1.0,
        seed: 42,
    };
    let mut a = base.clone();
    let ba = a.clone();
    form_stars(&mut a, &rho, &div_v, 0.1, &cfg, 0);
    let mut b = base.clone();
    let bb = b.clone();
    form_stars(&mut b, &rho, &div_v, 0.1, &cfg, 1);
    assert_ne!(
        converted_ids(&ba, &a),
        converted_ids(&bb, &b),
        "different epoch ⇒ independent draw substream"
    );
}

#[test]
fn one_way_monotonicity_no_star_reverts() {
    // Repeated calls: star count only grows and no existing star reverts to gas.
    let n = 400;
    let mut s = gas_state(n, 0.0);
    let rho = vec![4.0; n];
    let div_v = vec![-1.0; n];
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 1.5,
        seed: 99,
    };
    let mut prev_stars = 0usize;
    for epoch in 0..6u64 {
        // Stars already present must remain stars across the call.
        let stars_before: Vec<bool> = s.kind.iter().map(|&k| k == Species::Collisionless).collect();
        s.time = epoch as f64;
        form_stars(&mut s, &rho, &div_v, 0.2, &cfg, epoch);
        for i in 0..n {
            if stars_before[i] {
                assert_eq!(s.kind[i], Species::Collisionless, "no star reverts to gas at {i}");
            }
        }
        let stars_now = s.kind.iter().filter(|&&k| k == Species::Collisionless).count();
        assert!(stars_now >= prev_stars, "star count non-decreasing");
        prev_stars = stars_now;
    }
    assert!(prev_stars > 0, "some stars formed over the run");
}

#[test]
fn below_threshold_never_converts() {
    let n = 200;
    let mut s = gas_state(n, 1.0);
    // Half above threshold, half strictly below — only the above-threshold half
    // may ever convert.
    let rho: Vec<f64> = (0..n).map(|i| if i < n / 2 { 5.0 } else { 0.5 }).collect();
    let div_v = vec![-1.0; n];
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 5.0, // huge ⇒ the eligible half converts heavily
        seed: 3,
    };
    form_stars(&mut s, &rho, &div_v, 0.5, &cfg, 0);
    for i in n / 2..n {
        assert_eq!(s.kind[i], Species::Gas, "ρ < ρ_thresh never converts at {i}");
    }
    assert!(
        (0..n / 2).any(|i| s.kind[i] == Species::Collisionless),
        "the above-threshold half does convert"
    );
}

#[test]
fn diverging_flow_never_converts() {
    let n = 200;
    let mut s = gas_state(n, 1.0);
    let rho = vec![5.0; n]; // all dense
    // Half converging (∇·v < 0), half diverging/at-rest (∇·v ≥ 0).
    let div_v: Vec<f64> = (0..n).map(|i| if i < n / 2 { -1.0 } else { 0.0 }).collect();
    let cfg = StarFormationConfig {
        rho_thresh: 1.0,
        efficiency: 5.0,
        seed: 5,
    };
    form_stars(&mut s, &rho, &div_v, 0.5, &cfg, 0);
    for i in n / 2..n {
        assert_eq!(s.kind[i], Species::Gas, "∇·v ≥ 0 never converts at {i}");
    }
    assert!(
        (0..n / 2).any(|i| s.kind[i] == Species::Collisionless),
        "the converging half does convert"
    );
}

#[test]
fn statistical_calibration_matches_analytic_probability() {
    // Uniform-ρ, all-converging box ⇒ every gas particle has the SAME analytic
    // conversion probability p. The empirical converted fraction must match p
    // (hand-derived, independent of the operator) to binomial sampling tol.
    let n = 20_000;
    let rho_val = 1.0;
    let eff = 1.0;
    let dt = 0.2772; // chosen so p ≈ 0.40 (see below)
    let mut s = gas_state(n, 1.0);
    let rho = vec![rho_val; n];
    let div_v = vec![-1.0; n];
    let cfg = StarFormationConfig {
        rho_thresh: 0.5,
        efficiency: eff,
        seed: 0x5EED_0F5F,
    };

    let summary = form_stars(&mut s, &rho, &div_v, dt, &cfg, 0);

    let p = p_convert(eff, dt, rho_val);
    assert!((p - 0.40).abs() < 0.02, "sanity: chosen p is ≈ 0.40, got {p}");
    let frac = summary.n_formed as f64 / n as f64;
    let tol = 3.0 * (p * (1.0 - p) / n as f64).sqrt(); // 3σ binomial
    assert!(
        (frac - p).abs() < tol,
        "converted fraction {frac} within 3σ ({tol:.5}) of analytic p = {p}"
    );
}
