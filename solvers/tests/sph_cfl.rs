//! CFL sentinel gates (DESIGN.md M7b, D6): the stable-dt bound scales as
//! `h_min / v_sig`, `validate_dt` trips on a deliberately over-large dt and
//! passes a safe one, and a gas-free state carries no hydro CFL constraint.

use galaxy_core::{DVec3, Species, State};
use galaxy_solvers::sph::{
    density_adaptive, max_stable_dt, validate_dt, DensityConfig, HydroParams,
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
        sound_speed: 2.0,
        ..HydroParams::default()
    };

    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    let expect = C_CFL * h_min / (2.0 * params.sound_speed);

    let got = max_stable_dt(&state, &params, &cfg, C_CFL);
    let rel = (got - expect).abs() / expect;
    assert!(rel < 1e-9, "max_stable_dt = {got}, want {expect} (static ⇒ v_sig = 2c_s)");
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
    assert!(bound.is_finite() && bound > 0.0, "bound must be finite positive");

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

    let static_bound = max_stable_dt(&gas_state(pos.clone(), vec![DVec3::ZERO; pos.len()]), &params, &cfg, C_CFL);
    // Radially converging velocity field v = −k·x (everything falls inward).
    let conv: Vec<DVec3> = pos.iter().map(|&p| -3.0 * p).collect();
    let moving_bound = max_stable_dt(&gas_state(pos, conv), &params, &cfg, C_CFL);
    assert!(
        moving_bound < static_bound,
        "converging flow bound {moving_bound} must be < static {static_bound}"
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
