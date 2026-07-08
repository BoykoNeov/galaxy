//! CFL sentinel gates (DESIGN.md M7b, D6): the stable-dt bound scales as
//! `h_min / v_sig`, `validate_dt` trips on a deliberately over-large dt and
//! passes a safe one, and a gas-free state carries no hydro CFL constraint.

use galaxy_core::{DVec3, Species, State};
use galaxy_solvers::sph::{
    density_adaptive, max_stable_dt, validate_dt, DensityConfig, Eos, HydroParams,
};

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

const C_CFL: f64 = 0.25;

#[test]
fn bound_is_positive_and_scales_like_h_over_signal_speed() {
    // For a static (v = 0) gas cloud the signal velocity is 2·c_s everywhere, so
    // the bound is C_cfl · h_min / (2 c_s). Recompute h_min independently and
    // check the closed form.
    let pos = random_points(11, 800, 3.0);
    let state = gas_state(pos.clone(), vec![DVec3::ZERO; pos.len()]);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Isothermal { c_s: 2.0 },
        ..HydroParams::default()
    };

    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    let expect = C_CFL * h_min / (2.0 * params.sound_speed());

    let got = max_stable_dt(&state, &params, &cfg, C_CFL);
    let rel = (got - expect).abs() / expect;
    assert!(
        rel < 1e-9,
        "max_stable_dt = {got}, want {expect} (static ⇒ v_sig = 2c_s)"
    );
}

#[test]
fn validate_trips_on_over_large_dt_and_passes_a_safe_one() {
    let pos = random_points(21, 600, 2.5);
    // Give the cloud some relative velocity so viscosity/signal speed is live.
    let vel = random_points(22, 600, 1.5);
    let state = gas_state(pos, vel);
    let cfg = DensityConfig::default();
    let params = HydroParams::default();

    let bound = max_stable_dt(&state, &params, &cfg, C_CFL);
    assert!(
        bound.is_finite() && bound > 0.0,
        "bound must be finite positive"
    );

    // Safe: half the bound passes; over-large: twice the bound trips.
    assert!(validate_dt(&state, &params, &cfg, 0.5 * bound, C_CFL).is_ok());
    let err = validate_dt(&state, &params, &cfg, 2.0 * bound, C_CFL)
        .expect_err("2× the CFL bound must trip the sentinel");
    assert_eq!(err.dt, 2.0 * bound);
    assert!((err.max_stable - bound).abs() < 1e-12 * bound);
}

#[test]
fn moving_toward_neighbors_shrinks_the_bound() {
    // A strongly converging flow raises v_sig (the −3 w_ij term), so the stable
    // dt must be strictly smaller than the static (v = 0) bound at the same
    // positions.
    let pos = random_points(33, 500, 2.0);
    let cfg = DensityConfig::default();
    let params = HydroParams::default();

    let static_bound = max_stable_dt(
        &gas_state(pos.clone(), vec![DVec3::ZERO; pos.len()]),
        &params,
        &cfg,
        C_CFL,
    );
    // Radially converging velocity field v = −k·x (everything falls inward).
    let conv: Vec<DVec3> = pos.iter().map(|&p| -3.0 * p).collect();
    let moving_bound = max_stable_dt(&gas_state(pos, conv), &params, &cfg, C_CFL);
    assert!(
        moving_bound < static_bound,
        "converging flow bound {moving_bound} must be < static {static_bound}"
    );
}

#[test]
fn cross_support_approacher_tightens_the_bound() {
    // The force law couples a pair out to 2·max(h_i,h_j) (the averaged kernel
    // W̄ is nonzero there), so a SMALL-h particle can be driven by a DIFFUSE
    // large-h neighbor whose support reaches it even though the small particle's
    // own 2·h_i ball does not reach back. The CFL signal velocity must see that
    // approach: gathering v_sig,i only within 2·h_i misses it and overestimates
    // the stable dt.
    //
    // Construct exactly that regime: a tight static clump (→ small h) plus one
    // isolated particle a distance D away, moving straight at the clump at
    // V ≫ c_s (→ large h, since it must reach the clump to find neighbors). Then
    // 2·h_clump < D < 2·h_dist, so the clump particle only "sees" the approacher
    // through the neighbor's larger support.
    let c_s = 1.0;
    let big_v = 100.0;
    let d = 5.0;

    let clump = random_points(77, 200, 0.03); // tight ⇒ small h, all static
    let mut pos = clump.clone();
    pos.push(DVec3::new(0.0, 0.0, d)); // lone diffuse approacher
    let mut vel = vec![DVec3::ZERO; clump.len()];
    vel.push(DVec3::new(0.0, 0.0, -big_v)); // heading straight at the clump

    let state = gas_state(pos.clone(), vel);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };

    // Recover h independently (same routine the CFL path uses) to hand-derive
    // the bound and to assert the geometry actually sits in the cross-support
    // regime we mean to test.
    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let h_dist = *dens.h.last().unwrap();
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        2.0 * h_min < d,
        "clump 2·h_min = {} must NOT reach the approacher at D = {d}",
        2.0 * h_min
    );
    assert!(
        2.0 * h_dist > d,
        "approacher 2·h_dist = {} must reach the clump at D = {d}",
        2.0 * h_dist
    );

    // The binding particle is the min-h clump member; it is approached at w ≈ −V
    // (the clump is tiny beside D), so v_sig = 2c_s − 3w = 2c_s + 3V.
    let v_sig = 2.0 * c_s + 3.0 * big_v;
    let expect = C_CFL * h_min / v_sig;

    let got = max_stable_dt(&state, &params, &cfg, C_CFL);
    let rel = (got - expect).abs() / expect;
    assert!(
        rel < 5e-2,
        "max_stable_dt = {got}, want ≈ {expect} (cross-support approacher seen). \
         Gathering only within 2·h_i would return the static floor \
         {} — 2c_s/v_sig = {}× too large.",
        C_CFL * h_min / (2.0 * c_s),
        v_sig / (2.0 * c_s),
    );
}

// ---------------------------------------------------------------------------
// E4a — per-particle adiabatic CFL. `v_sig,i = max(2·c_s,i, max_j(c_s,i+c_s,j
// − 3·min(0,w_ij)))` with `c_s,i = √(γ(γ−1)u_i)`; the isothermal arm stays
// bit-identical (pair term `c_s,i+c_s,j = 2c_s = floor` is a provable no-op).
// ---------------------------------------------------------------------------

/// Build a gas state with per-particle internal energy `u`.
fn gas_state_u(pos: Vec<DVec3>, vel: Vec<DVec3>, u: Vec<f64>) -> State {
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for k in s.kind.iter_mut() {
        *k = Species::Gas;
    }
    s.u = u;
    s
}

#[test]
fn isothermal_cfl_pins_pre_e4a_bits() {
    // Byte-identity guard: the isothermal CFL bound feeds the shipped gasrich
    // adaptive movie, an out-of-band A/B control (NOT a `cargo test` assertion),
    // so a 1-ULP slip from the E4a `match`/per-particle refactor turns no other
    // gate red but silently shifts the shipped trajectory. Pin the exact f64
    // bits of the current (pre-E4a) isothermal bound on a fixed moving cloud
    // (the `−3w` branch is live). Captured against the pre-E4a code.
    let pos = random_points(21, 600, 2.5);
    let vel = random_points(22, 600, 1.5);
    let state = gas_state(pos, vel);
    let cfg = DensityConfig::default();
    let params = HydroParams::default();

    let got = max_stable_dt(&state, &params, &cfg, C_CFL);
    let frozen = f64::from_bits(0x3f80c12ff76329e9);
    assert_eq!(
        got.to_bits(),
        frozen.to_bits(),
        "isothermal CFL bound must stay bit-identical across the E4a refactor: \
         got {got:e} (0x{:016x}), want {frozen:e}",
        got.to_bits()
    );
}

#[test]
fn adiabatic_static_bound_scales_like_h_over_2cs() {
    // Adiabatic twin of `bound_is_positive_and_scales_like_h_over_signal_speed`:
    // uniform-`u` gas at rest ⇒ c_s,i = √(γ(γ−1)u) uniform, every pair term is
    // 2c_s (= floor), so v_sig = 2c_s everywhere and the bound is
    // C_cfl · h_min / (2 c_s). Recompute h_min independently.
    let gamma = 1.4_f64;
    let u0 = 2.0_f64;
    let c_s = (gamma * (gamma - 1.0) * u0).sqrt();

    let pos = random_points(11, 800, 3.0);
    let n = pos.len();
    let state = gas_state_u(pos.clone(), vec![DVec3::ZERO; n], vec![u0; n]);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma },
        ..HydroParams::default()
    };

    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    let expect = C_CFL * h_min / (2.0 * c_s);

    let got = max_stable_dt(&state, &params, &cfg, C_CFL);
    let rel = (got - expect).abs() / expect;
    assert!(
        rel < 1e-9,
        "adiabatic max_stable_dt = {got}, want {expect} (uniform-u static ⇒ v_sig = 2c_s)"
    );
}

#[test]
fn adiabatic_hot_neighbor_raises_vsig_at_rest() {
    // The per-particle pair term `c_s,i + c_s,j` must apply to NON-approaching
    // neighbors too (a hot neighbor's sound wave reaches a resting cold
    // particle). Same cold lattice at rest, twice: once uniformly cold, once
    // with a hot slab welded onto one face. The cold particles bordering the
    // hot slab get v_sig = c_s,cold + c_s,hot > 2c_s,cold, so the hot-neighbor
    // bound must be STRICTLY smaller even though nothing moves (w ≡ 0).
    let gamma = 1.4;
    let u_cold = 1.0;
    let u_hot = 25.0; // c_s,hot = 5× c_s,cold

    // A cold cubic lattice spanning x∈[0,1].
    let mut cold = Vec::new();
    let m = 8;
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                let s = 1.0 / m as f64;
                cold.push(DVec3::new(a as f64 * s, b as f64 * s, c as f64 * s));
            }
        }
    }
    let n_cold = cold.len();

    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma },
        ..HydroParams::default()
    };

    // All-cold reference.
    let all_cold = gas_state_u(
        cold.clone(),
        vec![DVec3::ZERO; n_cold],
        vec![u_cold; n_cold],
    );
    let cold_bound = max_stable_dt(&all_cold, &params, &cfg, C_CFL);

    // Hot slab welded onto the +x face (x∈[1+s, 2]), so the x≈1 cold layer has
    // hot neighbors within the coupling range. Everything at rest.
    let mut pos = cold.clone();
    let mut u = vec![u_cold; n_cold];
    let s = 1.0 / m as f64;
    for a in 0..m {
        for b in 0..m {
            for c in 0..m {
                pos.push(DVec3::new(
                    1.0 + s + a as f64 * s,
                    b as f64 * s,
                    c as f64 * s,
                ));
                u.push(u_hot);
            }
        }
    }
    let n = pos.len();
    let hot_slab = gas_state_u(pos, vec![DVec3::ZERO; n], u);
    let hot_bound = max_stable_dt(&hot_slab, &params, &cfg, C_CFL);

    assert!(
        hot_bound < cold_bound,
        "a resting hot neighbor must raise v_sig via the pair term and shrink \
         the bound: hot-slab {hot_bound} should be < all-cold {cold_bound}"
    );
}

#[test]
fn adiabatic_approaching_pair_pins_the_minus_3w_term() {
    // Coverage backfill (advisor-flagged): the two adiabatic tests above hold
    // every particle at rest (w ≡ 0), so they exercise only the `2·c_s,i` floor
    // and the `c_s,i + c_s,j` pair branch. The adiabatic arm's APPROACH term
    // `−3·min(0,w_ij)` is textually separate from the (bit-pinned) isothermal
    // arm and was executed by no value-pinning adiabatic test. Mirror the
    // isothermal `cross_support_approacher_tightens_the_bound` in adiabatic form
    // with a HOT approacher so the exact binding bound
    //   C_cfl · h_min / (c_s,clump + c_s,hot − 3w),   w ≈ −V,
    // pins all three terms (floor sound speed, cross-species pair, `−3w`) at
    // once — and a wrong coefficient (e.g. `−2w`) goes red.
    let gamma = 1.4_f64;
    let u_clump = 1.0_f64;
    let u_hot = 25.0_f64; // c_s,hot = 5× c_s,clump
    let cs_clump = (gamma * (gamma - 1.0) * u_clump).sqrt();
    let cs_hot = (gamma * (gamma - 1.0) * u_hot).sqrt();
    let big_v = 100.0;
    let d = 5.0;

    let clump = random_points(77, 200, 0.03); // tight ⇒ small h, all static & cold
    let mut pos = clump.clone();
    pos.push(DVec3::new(0.0, 0.0, d)); // lone diffuse HOT approacher
    let mut vel = vec![DVec3::ZERO; clump.len()];
    vel.push(DVec3::new(0.0, 0.0, -big_v)); // heading straight at the clump
    let mut u = vec![u_clump; clump.len()];
    u.push(u_hot);

    let state = gas_state_u(pos.clone(), vel, u);
    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma },
        ..HydroParams::default()
    };

    // Recover h independently and assert the cross-support geometry (same as the
    // isothermal twin): 2·h_clump < D < 2·h_approacher.
    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let h_dist = *dens.h.last().unwrap();
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        2.0 * h_min < d,
        "clump 2·h_min = {} must NOT reach the approacher at D = {d}",
        2.0 * h_min
    );
    assert!(
        2.0 * h_dist > d,
        "approacher 2·h_dist = {} must reach the clump at D = {d}",
        2.0 * h_dist
    );

    // Binding particle = min-h clump member, approached at w ≈ −V, so
    // v_sig = c_s,clump + c_s,hot − 3w = c_s,clump + c_s,hot + 3V. The `−3w`
    // approach term dominates (3V = 300 ≫ the pair sound speeds), which is
    // exactly the line this test exists to pin.
    let v_sig = cs_clump + cs_hot + 3.0 * big_v;
    let expect = C_CFL * h_min / v_sig;

    let got = max_stable_dt(&state, &params, &cfg, C_CFL);
    let rel = (got - expect).abs() / expect;
    assert!(
        rel < 5e-2,
        "adiabatic max_stable_dt = {got}, want ≈ {expect} (approach term −3w seen). \
         Dropping `−3w` would return the static floor {} — {}× too large.",
        C_CFL * h_min / (cs_clump + cs_hot),
        v_sig / (cs_clump + cs_hot),
    );
}

#[test]
fn gas_free_state_has_no_hydro_cfl_constraint() {
    // Pure collisionless state ⇒ no SPH CFL bound (returns +∞, any dt validates).
    let pos = random_points(44, 100, 3.0);
    let state = State::from_phase_space(pos, vec![DVec3::ZERO; 100], vec![1.0; 100]);
    let cfg = DensityConfig::default();
    let params = HydroParams::default();
    assert_eq!(max_stable_dt(&state, &params, &cfg, C_CFL), f64::INFINITY);
    assert!(validate_dt(&state, &params, &cfg, 1e9, C_CFL).is_ok());
}
