//! I7 — the active-subset gather (the individual-timestep efficiency path).
//!
//! Under `hydro-only` individual timesteps only a handful of gas particles are
//! ACTIVE on each fine tick; recomputing density + hydro force over the WHOLE gas
//! set every tick (as the I3 stepper does) does the same number of force evals as
//! global adaptive ⇒ zero speedup. I7 reduces the *gather* to the active subset
//! while keeping positions exact (the stepper drifts all) and rebuilding the grid
//! fresh each tick (build is ~367× cheaper than the gather — measured).
//!
//! GATE DESIGN (advisor-vetted, 2026-07-10):
//!   * ANCHOR — active-over-ALL ≡ full, BIT-IDENTICAL. The grid and per-particle
//!     bracket seeds are computed over all gas regardless of the active set, and
//!     each target's solve/force is the SAME per-target computation, independent
//!     of which other targets are active. So with `active = 0..n` the active pass
//!     reproduces the full pass exactly — byte-identity by construction, not a
//!     tolerance. This is the load-bearing pin (it also protects I3's collapsed
//!     bit-identity gate once the stepper is wired to the active path).
//!   * PARTIAL — a subset gather on a FRESH (all-refreshed) scratch equals the
//!     full pass at exactly the active indices. This is the sharp check that the
//!     active pass picks the right targets and reads neighbour ρ/h correctly; the
//!     STALE-neighbour bounded approximation (inactive ρ/h) is covered downstream
//!     by the driver convergence gates (I4a) once the stepper is wired.
//!   * GravitySph anchor is checked TWICE per solver (call, then call again) so the
//!     internal warm-start scratch (ρ/h) evolution is proven to match the full
//!     path too, not just the first acceleration — without a private-field accessor.

use galaxy_core::{DVec3, ForceSolver, Species, State};
use galaxy_solvers::sph::{
    density_adaptive, density_adaptive_active, hydro_accel_and_dudt, hydro_accel_and_dudt_active,
    DensityConfig, Eos, GravitySph, HydroParams,
};
use galaxy_solvers::BarnesHut;

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A compact gas cloud with a mild swirl (non-trivial velocities exercise the
/// viscosity / du/dt terms), spatially spread so h is non-uniform (a real gather).
fn gas_cloud(seed: u64, n: usize, radius: f64) -> (Vec<DVec3>, Vec<DVec3>, Vec<f64>, Vec<f64>) {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    let vel: Vec<DVec3> = pos
        .iter()
        .map(|&p| DVec3::new(-p.y, p.x, 0.0) * 0.3)
        .collect();
    let mass = vec![1.0; n];
    let u = vec![0.5; n]; // used only by the adiabatic arm
    (pos, vel, mass, u)
}

fn gas_state(seed: u64, n: usize, radius: f64) -> State {
    let (pos, vel, mass, u) = gas_cloud(seed, n, radius);
    let mut s = State::from_phase_space(pos, vel, mass);
    for (su, uu) in s.u.iter_mut().zip(&u) {
        *su = *uu;
    }
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    s
}

fn dcfg() -> DensityConfig {
    DensityConfig::default()
}

fn hparams() -> HydroParams {
    HydroParams {
        eos: Eos::Isothermal { c_s: 1.1 },
        ..HydroParams::default()
    }
}

// --------------------------------------------------------------------------
// Free-function ANCHOR: active-over-all ≡ full (bit-identical).
// --------------------------------------------------------------------------

#[test]
fn density_active_all_equals_full_bit_identical() {
    let (pos, _vel, mass, _u) = gas_cloud(7, 200, 3.0);
    let cfg = dcfg();
    let full = density_adaptive(&pos, &mass, &cfg, None);

    // Active over EVERY index, fresh (zero) scratch ⇒ the h[i]=0 hint falls back to
    // the occupancy seed, reproducing the `h_init = None` full path exactly.
    let all: Vec<usize> = (0..pos.len()).collect();
    let mut rho = vec![0.0; pos.len()];
    let mut h = vec![0.0; pos.len()];
    density_adaptive_active(&pos, &mass, &cfg, &all, &mut rho, &mut h);

    assert_eq!(
        rho, full.rho,
        "active-all density ρ must be bit-identical to full"
    );
    assert_eq!(
        h, full.h,
        "active-all density h must be bit-identical to full"
    );
}

#[test]
fn forces_active_all_equals_full_bit_identical() {
    let (pos, vel, mass, u) = gas_cloud(9, 200, 3.0);
    let cfg = dcfg();
    let params = hparams();
    let dens = density_adaptive(&pos, &mass, &cfg, None);
    let (acc_full, dudt_full) =
        hydro_accel_and_dudt(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &params);

    let all: Vec<usize> = (0..pos.len()).collect();
    let contribs =
        hydro_accel_and_dudt_active(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &params, &all);

    assert_eq!(contribs.len(), pos.len());
    for i in 0..pos.len() {
        assert_eq!(
            contribs[i].0, acc_full[i],
            "active-all accel differs at {i}"
        );
        assert_eq!(
            contribs[i].1, dudt_full[i],
            "active-all du/dt differs at {i}"
        );
    }
}

// --------------------------------------------------------------------------
// PARTIAL: subset gather on a fresh scratch equals full at the active indices.
// --------------------------------------------------------------------------

#[test]
fn density_active_subset_matches_full_at_active_indices() {
    let (pos, _vel, mass, _u) = gas_cloud(13, 200, 3.0);
    let cfg = dcfg();
    let full = density_adaptive(&pos, &mass, &cfg, None);

    // Prime the scratch with a full refresh, then refresh only every 3rd index.
    let all: Vec<usize> = (0..pos.len()).collect();
    let mut rho = vec![0.0; pos.len()];
    let mut h = vec![0.0; pos.len()];
    density_adaptive_active(&pos, &mass, &cfg, &all, &mut rho, &mut h);

    let subset: Vec<usize> = (0..pos.len()).step_by(3).collect();
    // Corrupt the active entries so a no-op would be caught; the refresh must
    // restore each to its full value bit-for-bit.
    for &i in &subset {
        rho[i] = -1.0;
        h[i] = -1.0;
    }
    density_adaptive_active(&pos, &mass, &cfg, &subset, &mut rho, &mut h);
    for &i in &subset {
        assert_eq!(
            rho[i], full.rho[i],
            "subset density ρ differs at active {i}"
        );
        assert_eq!(h[i], full.h[i], "subset density h differs at active {i}");
    }
}

#[test]
fn forces_active_subset_matches_full_at_active_indices() {
    let (pos, vel, mass, u) = gas_cloud(17, 200, 3.0);
    let cfg = dcfg();
    let params = hparams();
    // Fresh ρ/h everywhere (all-fresh scratch): a subset gather then equals the
    // full force at the active indices bit-for-bit (same grid, same neighbour ρ/h).
    let dens = density_adaptive(&pos, &mass, &cfg, None);
    let (acc_full, dudt_full) =
        hydro_accel_and_dudt(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &params);

    let subset: Vec<usize> = (0..pos.len()).step_by(4).collect();
    let contribs =
        hydro_accel_and_dudt_active(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &params, &subset);
    assert_eq!(contribs.len(), subset.len());
    for (k, &i) in subset.iter().enumerate() {
        assert_eq!(
            contribs[k].0, acc_full[i],
            "subset accel differs at active {i}"
        );
        assert_eq!(
            contribs[k].1, dudt_full[i],
            "subset du/dt differs at active {i}"
        );
    }
}

// --------------------------------------------------------------------------
// GravitySph ANCHOR: accelerations_active(all) ≡ accelerations, TWICE
// (so the warm-start ρ/h scratch evolution matches the full path too).
// --------------------------------------------------------------------------

fn hydro_solver() -> GravitySph<BarnesHut> {
    GravitySph::<BarnesHut>::hydro_only(hparams(), dcfg())
}

#[test]
fn gravity_sph_accelerations_active_all_equals_full_twice() {
    let state = gas_state(21, 300, 2.0);
    let n = state.len();
    let all: Vec<usize> = (0..n).collect();

    let mut full = hydro_solver();
    let mut active = hydro_solver();

    for round in 0..2 {
        let mut a_full = vec![DVec3::ZERO; n];
        let mut a_active = vec![DVec3::ZERO; n];
        full.accelerations(&state, &mut a_full);
        active.accelerations_active(&state, &all, &mut a_active);
        assert_eq!(
            a_active, a_full,
            "accelerations_active(all) must equal accelerations (round {round})"
        );
    }
}

#[test]
fn gravity_sph_accel_and_dudt_active_all_equals_full_twice() {
    // Adiabatic arm so du/dt is non-trivial (isothermal du/dt is also fine but the
    // adiabatic EOS exercises the per-particle sound-speed path in the gather).
    let state = gas_state(23, 300, 2.0);
    let n = state.len();
    let all: Vec<usize> = (0..n).collect();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma: 1.4 },
        ..HydroParams::default()
    };
    let mut full = GravitySph::<BarnesHut>::hydro_only(params, dcfg());
    let mut active = GravitySph::<BarnesHut>::hydro_only(params, dcfg());

    for round in 0..2 {
        let (mut a_full, mut d_full) = (vec![DVec3::ZERO; n], vec![0.0; n]);
        let (mut a_active, mut d_active) = (vec![DVec3::ZERO; n], vec![0.0; n]);
        full.accel_and_dudt(&state, &mut a_full, &mut d_full);
        active.accel_and_dudt_active(&state, &all, &mut a_active, &mut d_active);
        assert_eq!(
            a_active, a_full,
            "active fused accel differs (round {round})"
        );
        assert_eq!(
            d_active, d_full,
            "active fused du/dt differs (round {round})"
        );
    }
}
