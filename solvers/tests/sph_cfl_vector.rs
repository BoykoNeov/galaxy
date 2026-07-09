//! I1 — the per-particle CFL vector (`max_stable_dt_per_particle`).
//!
//! Individual timesteps need `dt_i = c_cfl · h_i / v_sig,i` per gas particle,
//! not the scalar `min`. This is the additive VECTOR variant: the shipped scalar
//! `max_stable_dt` stays frozen (its `isothermal_cfl_pins_pre_e4a_bits` gate is
//! untouched), and the vector's `min` is asserted equal to it as a copy-drift
//! guard — with the caveat (advisor-flagged) that `min ≡ scalar` only guards the
//! MINIMAL particle, so the non-minimal entries are pinned independently below.
//!
//! Full-length, state-indexed: gas rows carry `c_cfl·h_i/v_sig,i`, collisionless
//! rows carry `+∞` (never hydro-rung-limited). The `+∞` rows make the vector a
//! strict generalization: `min` over it equals the gas-only scalar bound.

use galaxy_core::{DVec3, ForceSolver, Species, State};
use galaxy_solvers::sph::{
    density_adaptive, max_stable_dt, max_stable_dt_per_particle, DensityConfig, Eos, GravitySph,
    HydroParams,
};
use galaxy_solvers::BarnesHut;

fn random_points(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * scale)
        .collect()
}

fn gas_state(pos: Vec<DVec3>, vel: Vec<DVec3>) -> State {
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for k in s.kind.iter_mut() {
        *k = Species::Gas;
    }
    s
}

fn gas_state_u(pos: Vec<DVec3>, vel: Vec<DVec3>, u: Vec<f64>) -> State {
    let mut s = gas_state(pos, vel);
    s.u = u;
    s
}

const C_CFL: f64 = 0.25;

fn vec_min(v: &[f64]) -> f64 {
    v.iter().copied().fold(f64::INFINITY, f64::min)
}

#[test]
fn isothermal_vector_min_equals_scalar_bit_for_bit() {
    // The core I1 gate: the vector is a strict generalization of the shipped
    // scalar bound, so its `min` must equal `max_stable_dt` BIT-for-bit on the
    // same moving cloud that `isothermal_cfl_pins_pre_e4a_bits` freezes. This
    // rides the frozen isothermal arm and is the copy-drift guard for it.
    let pos = random_points(21, 600, 2.5);
    let vel = random_points(22, 600, 1.5);
    let state = gas_state(pos, vel);
    let cfg = DensityConfig::default();
    let params = HydroParams::default();

    let scalar = max_stable_dt(&state, &params, &cfg, C_CFL);
    let vector = max_stable_dt_per_particle(&state, &params, &cfg, C_CFL);
    assert_eq!(vector.len(), state.len(), "vector is full state-length");
    assert_eq!(
        vec_min(&vector).to_bits(),
        scalar.to_bits(),
        "isothermal vector min {} (0x{:016x}) must equal scalar {} bit-for-bit",
        vec_min(&vector),
        vec_min(&vector).to_bits(),
        scalar
    );
}

#[test]
fn adiabatic_vector_min_equals_scalar_bit_for_bit() {
    // Same guard for the adiabatic arm (the second copied EOS branch — a distinct
    // copy-drift surface). Moving cloud with non-uniform `u` so `c_s,i` varies and
    // the `−3w` approach term is live.
    let gamma = 1.4_f64;
    let pos = random_points(31, 600, 2.5);
    let vel = random_points(32, 600, 1.5);
    // Spread of internal energies so c_s,i differs particle to particle.
    let u: Vec<f64> = (0..pos.len()).map(|k| 0.5 + (k % 7) as f64 * 0.3).collect();
    let state = gas_state_u(pos, vel, u);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma },
        ..HydroParams::default()
    };

    let scalar = max_stable_dt(&state, &params, &cfg, C_CFL);
    let vector = max_stable_dt_per_particle(&state, &params, &cfg, C_CFL);
    assert_eq!(
        vec_min(&vector).to_bits(),
        scalar.to_bits(),
        "adiabatic vector min {} must equal scalar {} bit-for-bit",
        vec_min(&vector),
        scalar
    );
}

#[test]
fn collisionless_rows_are_infinite() {
    // Mixed gas + collisionless: every collisionless row is `+∞` (never
    // hydro-rung-limited), every gas row is finite. A pure-collisionless state is
    // all `+∞` (its `min` recovers the scalar's `+∞` gas-free bound).
    let mut pos = random_points(44, 300, 3.0);
    let n_gas = pos.len();
    pos.extend(random_points(45, 200, 3.0)); // 200 collisionless
    let vel = vec![DVec3::ZERO; pos.len()];
    let mut state = State::from_phase_space(pos, vel, vec![1.0; 500]);
    for i in 0..n_gas {
        state.kind[i] = Species::Gas;
    }
    // indices n_gas..500 stay Collisionless (the from_phase_space default).

    let cfg = DensityConfig::default();
    let params = HydroParams::default();
    let vector = max_stable_dt_per_particle(&state, &params, &cfg, C_CFL);

    for (i, &dt) in vector.iter().enumerate() {
        match state.kind[i] {
            Species::Gas => assert!(
                dt.is_finite() && dt > 0.0,
                "gas row {i} must be finite positive, got {dt}"
            ),
            _ => assert_eq!(
                dt,
                f64::INFINITY,
                "collisionless row {i} must be +∞, got {dt}"
            ),
        }
    }

    // Pure collisionless ⇒ all +∞, min recovers the gas-free scalar bound.
    let pure = State::from_phase_space(
        random_points(46, 100, 3.0),
        vec![DVec3::ZERO; 100],
        vec![1.0; 100],
    );
    let pv = max_stable_dt_per_particle(&pure, &params, &cfg, C_CFL);
    assert!(pv.iter().all(|&d| d == f64::INFINITY), "all rows +∞");
    assert_eq!(vec_min(&pv), max_stable_dt(&pure, &params, &cfg, C_CFL));
}

#[test]
fn static_cloud_pins_the_full_vector_closed_form() {
    // Teeth on ALL entries (not just the min): a static (v ≡ 0) isothermal cloud
    // has v_sig,i = 2·c_s for EVERY particle, so dt_i = C_cfl·h_i/(2c_s) with h_i
    // recovered independently. This pins the whole per-particle mapping — the
    // gas-subset→global index placement and the 2c_s floor — everywhere, catching
    // a copy-drift that leaves the min unchanged but corrupts a non-minimal rung.
    let c_s = 2.0;
    let pos = random_points(11, 800, 3.0);
    let state = gas_state(pos.clone(), vec![DVec3::ZERO; pos.len()]);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };

    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let vector = max_stable_dt_per_particle(&state, &params, &cfg, C_CFL);

    for (k, &dt) in vector.iter().enumerate() {
        let expect = C_CFL * dens.h[k] / (2.0 * c_s);
        let rel = (dt - expect).abs() / expect;
        assert!(
            rel < 1e-9,
            "static entry {k}: got {dt}, want {expect} (= C_cfl·h/(2c_s))"
        );
    }
}

#[test]
fn non_minimal_approacher_entry_pins_the_minus_3w_term() {
    // Advisor's explicit teeth: a `−3w` copy-drift in a NON-minimal entry sails
    // through `min ≡ scalar`. Reuse the cross-support geometry (a tight static
    // clump + one lone diffuse approacher at z = D moving at −V). The binding MIN
    // is a clump particle; the LONE APPROACHER (last index) is strictly
    // non-minimal (large h_dist) yet its own v_sig is driven by the same approach
    // (w ≈ −V), so its entry is C_cfl·h_dist/(2c_s + 3V). Pin that entry — a wrong
    // approach coefficient there goes red without touching the min.
    let c_s = 1.0;
    let big_v = 100.0;
    let d = 5.0;

    let clump = random_points(77, 200, 0.03);
    let mut pos = clump.clone();
    pos.push(DVec3::new(0.0, 0.0, d));
    let mut vel = vec![DVec3::ZERO; clump.len()];
    vel.push(DVec3::new(0.0, 0.0, -big_v));

    let state = gas_state(pos.clone(), vel);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };

    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let h_dist = *dens.h.last().unwrap();
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    // The geometry must sit in the cross-support regime (else the approacher would
    // not see the clump at all).
    assert!(2.0 * h_min < d, "clump must not reach the approacher");
    assert!(2.0 * h_dist > d, "approacher must reach the clump");

    let vector = max_stable_dt_per_particle(&state, &params, &cfg, C_CFL);
    let last = state.len() - 1;

    // The approacher is non-minimal (h_dist ≫ h_min ⇒ larger dt than the clump).
    assert!(
        vector[last] > vec_min(&vector),
        "approacher entry {} must be non-minimal (min {})",
        vector[last],
        vec_min(&vector)
    );
    // ... and its value pins the −3w approach term.
    let v_sig = 2.0 * c_s + 3.0 * big_v;
    let expect = C_CFL * h_dist / v_sig;
    let rel = (vector[last] - expect).abs() / expect;
    assert!(
        rel < 5e-2,
        "approacher entry {}: want ≈ {expect} (C_cfl·h_dist/(2c_s+3V)); \
         dropping −3w gives the static floor {} ({}× too large)",
        vector[last],
        C_CFL * h_dist / (2.0 * c_s),
        v_sig / (2.0 * c_s)
    );
}

#[test]
fn gravitysph_trait_per_particle_min_equals_scalar() {
    // The trait plumbing (I3/I4 reach the vector through `&mut dyn ForceSolver`):
    // GravitySph::max_stable_dt_per_particle must apply c_cfl = 1.0 exactly as its
    // scalar max_stable_dt does, so the vector's min equals the scalar bound.
    let pos = random_points(21, 600, 2.5);
    let vel = random_points(22, 600, 1.5);
    let state = gas_state(pos, vel);
    let params = HydroParams::default();
    let cfg = DensityConfig::default();
    let solver = GravitySph::new(BarnesHut::new(1.0, 0.05, 0.5), params, cfg);

    let scalar = solver.max_stable_dt(&state);
    let vector = solver.max_stable_dt_per_particle(&state);
    assert_eq!(vector.len(), state.len());
    assert_eq!(
        vec_min(&vector).to_bits(),
        scalar.to_bits(),
        "GravitySph per-particle min {} must equal scalar {} bit-for-bit",
        vec_min(&vector),
        scalar
    );
}
