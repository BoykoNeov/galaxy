//! Adiabatic homologous-compression gate (E2b): the physics validation of
//! `LeapfrogKdkThermal`. Gravity OFF (`GravitySph::hydro_only`) and viscosity
//! OFF (alpha=0, beta=0) isolate pure PdV heating; a uniform-ρ/uniform-u
//! lattice with an imposed linear velocity field `v_i = -k(pos_i-center)`
//! deforms homologously (`s(t)=1-kt`), giving a closed-form, code-independent
//! reference: `ρ(t)=ρ0/s(t)³`, `u(t)=u0·s(t)^{-3(γ-1)}` (integrated adiabatic
//! first law `du/dt = P·(dρ/dt)/ρ²` with `P=(γ-1)ρu`, i.e. `PV^γ=const`).

use galaxy_core::{
    diagnostics, DVec3, ForceSolver, Integrator, LeapfrogKdkThermal, Species, State,
    StaticBackground,
};
use galaxy_solvers::sph::{density_adaptive, DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::DirectSum;

/// Centered cubic lattice, spacing `s`, `nx` per side, uniform mass/u, all
/// tagged `Gas`, with an imposed homologous compression velocity
/// `v_i = -k(pos_i-center)`. Returns the state and the geometric center.
fn compression_lattice(nx: usize, s: f64, k: f64, u0: f64, rho0: f64) -> (State, DVec3) {
    let half = (nx - 1) as f64 * s * 0.5;
    let center = DVec3::splat(half);
    let mut pos = Vec::new();
    for ix in 0..nx {
        for iy in 0..nx {
            for iz in 0..nx {
                pos.push(DVec3::new(ix as f64, iy as f64, iz as f64) * s);
            }
        }
    }
    let n = pos.len();
    let m = rho0 * s * s * s;
    let vel: Vec<DVec3> = pos.iter().map(|&p| -k * (p - center)).collect();
    let mut state = State::from_phase_space(pos, vel, vec![m; n]);
    for kind in state.kind.iter_mut() {
        *kind = Species::Gas;
    }
    for u in state.u.iter_mut() {
        *u = u0;
    }
    (state, center)
}

/// Fast smoke gate (non-ignored): a 5³ lattice over 10 short steps must (a)
/// heat under compression, (b) conserve total energy to a symplectic-oscillation
/// bound, (c) leave total momentum untouched. Catches gross wiring bugs (u-kick
/// dropped, sign flipped, energy accounting broken) without the cost of the
/// self-similar convergence run.
#[test]
fn thermal_integrator_heats_a_compressing_lattice_and_conserves_energy() {
    let (nx, s, k, u0, rho0, gamma) = (5usize, 1.0, 0.1, 1.0, 1.0, 5.0 / 3.0);
    let (mut state, _center) = compression_lattice(nx, s, k, u0, rho0);
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma },
        alpha: 0.0,
        beta: 0.0,
        ..HydroParams::default()
    };
    let cfg = DensityConfig::default();
    let mut solver = GravitySph::<DirectSum>::hydro_only(params, cfg);
    let mut integ = LeapfrogKdkThermal::new();
    let bg = StaticBackground;

    let u0_total = diagnostics::thermal_energy(&state);
    let e0 = diagnostics::total_energy(&state, &solver);
    let p0 = diagnostics::total_momentum(&state);
    let (dt, n_steps) = (0.01, 10);
    let mut max_e_err = 0.0_f64;
    for _ in 0..n_steps {
        integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
        let e = diagnostics::total_energy(&state, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
    }
    // 5% is a loose gross-wiring bound; the ignored self-similar test pins the
    // real (much tighter) oscillation floor empirically.
    assert!(max_e_err < 5e-2, "energy drift too large: {max_e_err:e}");
    assert!(
        (diagnostics::total_momentum(&state) - p0).length() < 1e-8,
        "momentum not conserved (thermal integrator must not perturb it)"
    );
    assert!(
        diagnostics::thermal_energy(&state) > u0_total,
        "adiabatic compression must heat the gas (u increased)"
    );
}

/// Dynamical self-similar gate (ignored — run `--release --ignored --nocapture`):
/// an 11³ = 1331-particle lattice compressed over 67 steps must track the
/// closed-form homologous solution `u(t)=u0·s(t)^{-3(γ-1)}`, `ρ(t)=ρ0/s(t)³` in
/// the interior (particles ≥ a sound-crossing margin from the boundary, where
/// the free-surface rarefaction wave contaminates the uniform-field assumption).
/// The margin is sized from the solver's ACTUAL runtime `h_max` and the PEAK
/// sound speed (which occurs at `t_end`, since `u` grows under compression).
#[test]
#[ignore = "dynamical validation: 1331-particle homologous-compression SPH run over 67 steps (run --release --ignored)"]
fn adiabatic_compression_tracks_the_self_similar_solution() {
    let (nx, s, k, u0, rho0, gamma) = (11usize, 1.0, 0.1, 1.0, 1.0, 5.0 / 3.0);
    let (state0, _center) = compression_lattice(nx, s, k, u0, rho0);
    let cfg = DensityConfig::default();

    // Runtime-derived h_max on the IC (advisor: size the margin from the actual
    // smoothing length the solver will use, not a hardcoded guess).
    let dens0 = density_adaptive(&state0.pos, &state0.mass, &cfg, None);
    let h_max = dens0.h.iter().cloned().fold(0.0_f64, f64::max);

    let (dt, n_steps) = (0.01, 67);
    let t_end = dt * n_steps as f64;
    let s_end = 1.0 - k * t_end;
    let u_end_peak = u0 * s_end.powf(-3.0 * (gamma - 1.0));
    let cs_peak = (gamma * (gamma - 1.0) * u_end_peak).sqrt();
    let margin = (2.0 * h_max).max(cs_peak * t_end);

    let hi = (nx - 1) as f64 * s;
    let interior: Vec<usize> = (0..state0.len())
        .filter(|&i| {
            let p = state0.pos[i];
            [p.x, p.y, p.z]
                .iter()
                .all(|&c| c > margin && c < hi - margin)
        })
        .collect();
    assert!(
        interior.len() > 20,
        "too few interior particles: {} (margin {margin:.3}, h_max {h_max:.3})",
        interior.len()
    );

    let params = HydroParams {
        eos: Eos::Adiabatic { gamma },
        alpha: 0.0,
        beta: 0.0,
        ..HydroParams::default()
    };
    let mut solver = GravitySph::<DirectSum>::hydro_only(params, cfg.clone());
    let mut integ = LeapfrogKdkThermal::new();
    let bg = StaticBackground;
    let mut state = state0;

    let e0 = diagnostics::total_energy(&state, &solver);
    let p0 = diagnostics::total_momentum(&state);
    let mut max_e_err = 0.0_f64;
    let checkpoints = [20usize, 40, 67];
    let mut cp_idx = 0;
    // PLACEHOLDERS — calibrated empirically from the --nocapture run (see plan
    // E2b). Set above the observed L1/oscillation floor once, then documented.
    const TOL: f64 = 0.05;
    const E_TOL: f64 = 0.05;
    for step in 1..=n_steps {
        integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
        let e = diagnostics::total_energy(&state, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());

        if cp_idx < checkpoints.len() && step == checkpoints[cp_idx] {
            let t = state.time;
            let s_t = 1.0 - k * t;
            let u_ref = u0 * s_t.powf(-3.0 * (gamma - 1.0));
            let rho_ref = rho0 / s_t.powi(3);

            let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);
            let mut u_err = 0.0;
            let mut rho_err = 0.0;
            for &i in &interior {
                u_err += (state.u[i] - u_ref).abs() / u_ref;
                rho_err += (dens.rho[i] - rho_ref).abs() / rho_ref;
            }
            u_err /= interior.len() as f64;
            rho_err /= interior.len() as f64;
            println!(
                "step={step} t={t:.3} s={s_t:.4} L1(u)={u_err:.5} (ref {u_ref:.5}) \
                 L1(rho)={rho_err:.5} (ref {rho_ref:.5})"
            );
            assert!(u_err < TOL, "L1(u) = {u_err} exceeds {TOL} at step {step}");
            assert!(
                rho_err < TOL,
                "L1(rho) = {rho_err} exceeds {TOL} at step {step}"
            );
            cp_idx += 1;
        }
    }
    assert_eq!(cp_idx, checkpoints.len(), "not all checkpoints reached");
    assert!(
        max_e_err < E_TOL,
        "energy oscillation too large: {max_e_err:e}"
    );
    assert!(
        (diagnostics::total_momentum(&state) - p0).length() < 1e-8,
        "momentum not conserved"
    );
}
