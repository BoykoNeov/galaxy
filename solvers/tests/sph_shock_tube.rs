//! Isothermal SPH shock tube (DESIGN.md M7b): the flagship physics gate. A
//! 4:1 density jump at rest relaxes into a left rarefaction + right shock; the
//! SPH density and velocity profiles must match the exact isothermal Riemann
//! solution within a loose L1 bound set by resolution and ~2–3h shock smearing.
//!
//! Geometry (advisor-vetted, D-plan "padded free ends"): free surfaces
//! everywhere, transverse widened to ±4 so a central measurement column stays
//! farther than 2h from every transverse face for the whole window; the
//! longitudinal ends at ±4 are far enough that their rarefactions do not reach
//! the shock region (|x| ≲ 2.1) by t_meas = 1.5. Driven by the stock
//! `LeapfrogKdk` + `GravitySph::hydro_only` (gravity off) — the real force code
//! the merger will use, no periodic-BC machinery.

use galaxy_core::{DVec3, ForceSolver, Integrator, LeapfrogKdk, Species, State, StaticBackground};
use galaxy_solvers::sph::{density_adaptive, DensityConfig, GravitySph, HydroParams};

const CS: f64 = 1.0;
const RHO_L: f64 = 4.0;
const RHO_R: f64 = 1.0;

// --- exact isothermal Riemann oracle --------------------------------------

/// The intermediate density ρ*: root of `ln(ρ_L/ρ*) = (ρ*−ρ_R)/√(ρ_R·ρ*)`
/// (rarefaction velocity == shock velocity), bisected in (ρ_R, ρ_L).
fn rho_star() -> f64 {
    let f = |rs: f64| (RHO_L / rs).ln() - (rs - RHO_R) / (RHO_R * rs).sqrt();
    let (mut lo, mut hi) = (RHO_R, RHO_L);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Analytic self-similar isothermal Riemann state `(ρ, u)` at `ξ = x/t`
/// (u_L = u_R = 0, ρ_L > ρ_R → left rarefaction fan, right shock).
fn riemann(xi: f64, rho_s: f64, u_s: f64, s_shock: f64) -> (f64, f64) {
    let head = -CS; // rarefaction head
    let tail = u_s - CS; // rarefaction tail
    if xi <= head {
        (RHO_L, 0.0) // undisturbed left
    } else if xi < tail {
        // rarefaction fan: u = ξ + c_s, ρ = ρ_L·exp(−1 − ξ/c_s)
        (RHO_L * (-1.0 - xi / CS).exp(), xi + CS)
    } else if xi < s_shock {
        (rho_s, u_s) // star region
    } else {
        (RHO_R, 0.0) // undisturbed right
    }
}

// --- IC ---------------------------------------------------------------------

/// Two glued cubic-lattice blocks, equal particle mass, densities set by
/// spacing. Left: dense (ρ_L), right: sparse (ρ_R). Transverse ±HALF_T, all Gas,
/// at rest.
fn shock_tube_ic() -> State {
    const HALF_T: f64 = 4.0;
    const X_END: f64 = 4.0;
    let s_l = 0.5_f64;
    let m = RHO_L * s_l * s_l * s_l; // ρ_L = m/s_l³
    let s_r = s_l * (RHO_L / RHO_R).cbrt(); // ρ_R = m/s_r³

    let mut pos = Vec::new();
    // Transverse axis values for a given spacing, symmetric about 0.
    let axis = |s: f64| -> Vec<f64> {
        let n = (HALF_T / s).floor() as i64;
        (-n..=n).map(|k| k as f64 * s).collect()
    };
    // Left block: x in [-X_END, -s_l].
    let ys_l = axis(s_l);
    let nx_l = (X_END / s_l).floor() as i64;
    for ix in 1..=nx_l {
        let x = -(ix as f64) * s_l;
        for &y in &ys_l {
            for &z in &ys_l {
                pos.push(DVec3::new(x, y, z));
            }
        }
    }
    // Right block: x in [0, X_END].
    let ys_r = axis(s_r);
    let nx_r = (X_END / s_r).floor() as i64;
    for ix in 0..=nx_r {
        let x = ix as f64 * s_r;
        for &y in &ys_r {
            for &z in &ys_r {
                pos.push(DVec3::new(x, y, z));
            }
        }
    }
    let n = pos.len();
    let mut state = State::from_phase_space(pos, vec![DVec3::ZERO; n], vec![m; n]);
    for k in state.kind.iter_mut() {
        *k = Species::Gas;
    }
    state
}

#[test]
fn isothermal_riemann_oracle_matches_hand_values() {
    // Independent spot-check of the oracle before trusting any profile failure
    // (advisor hand values for ρ_L=4, ρ_R=1, c_s=1).
    let rs = rho_star();
    let us = CS * (RHO_L / rs).ln();
    let s = CS * (rs / RHO_R).sqrt();
    assert!((rs - 1.985).abs() < 0.01, "ρ* = {rs}, want ≈1.985");
    assert!((us - 0.700).abs() < 0.01, "u* = {us}, want ≈0.700");
    assert!((s - 1.409).abs() < 0.01, "S = {s}, want ≈1.409");
    // Continuity: rarefaction tail meets the star state.
    let (rho_tail, u_tail) = riemann(us - CS + 1e-9, rs, us, s);
    assert!((rho_tail - rs).abs() < 1e-3 && (u_tail - us).abs() < 1e-3);
}

#[test]
#[ignore = "dynamical validation: ~3000-particle SPH shock tube over 75 steps (run --release --ignored)"]
fn sph_shock_tube_matches_the_isothermal_riemann_solution() {
    let mut state = shock_tube_ic();
    let params = HydroParams {
        sound_speed: CS,
        ..HydroParams::default()
    };
    let cfg = DensityConfig::default();
    let mut solver = GravitySph::<galaxy_solvers::DirectSum>::hydro_only(params, cfg.clone());
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let dt = 0.02;
    let n_steps = 75; // t_meas = 1.5
    for _ in 0..n_steps {
        integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
    }
    let t = state.time;

    // Oracle constants.
    let rs = rho_star();
    let us = CS * (RHO_L / rs).ln();
    let s_shock = CS * (rs / RHO_R).sqrt();

    // SPH density at the particles (what the sim actually integrates against).
    let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);

    // Central column, away from the longitudinal ends (free-surface deficit).
    let mut l1_rho = 0.0;
    let mut l1_u = 0.0;
    let mut star_rho_sum = 0.0;
    let mut n_meas = 0usize;
    let mut n_star = 0usize;
    for i in 0..state.len() {
        let p = state.pos[i];
        // Central column, ≥ 3 from the ±4 transverse faces (farther than
        // 2h ≈ 1.8 → no kernel deficit; the transverse rarefaction (c_s·t ≈ 1.5)
        // hasn't reached it either). ASYMMETRIC x-window: the left holds only
        // the compact rarefaction fan (x ∈ [−1.5, −0.45]), so x > −2 clears the
        // left free-end rarefaction (it reaches x ≈ −2.5 by t=1.5); the right
        // must reach past the shock (x ≈ 2.1) to sample the post-shock ρ_R.
        if p.y.abs() > 1.0 || p.z.abs() > 1.0 || p.x < -2.0 || p.x > 3.0 {
            continue;
        }
        let xi = p.x / t;
        let (rho_ref, u_ref) = riemann(xi, rs, us, s_shock);
        l1_rho += (dens.rho[i] - rho_ref).abs() / rho_ref;
        l1_u += (state.vel[i].x - u_ref).abs();
        n_meas += 1;
        // Star-region plateau: strictly between rarefaction tail and shock.
        if xi > us - CS + 0.15 && xi < s_shock - 0.15 {
            star_rho_sum += dens.rho[i];
            n_star += 1;
        }
    }
    assert!(n_meas > 100, "too few measured particles: {n_meas}");
    l1_rho /= n_meas as f64;
    l1_u /= n_meas as f64;
    let star_rho = star_rho_sum / n_star.max(1) as f64;
    println!(
        "t={t:.3} n_meas={n_meas} n_star={n_star} L1(rho)={l1_rho:.4} L1(u)={l1_u:.4} \
         star_rho={star_rho:.4} (ρ*={rs:.4})"
    );

    // Loose L1: 2–3h shock smearing spread over the measured span dominates the
    // error; a wrong shock speed or ρ* misses by ≫ 0.3.
    assert!(l1_rho < 0.15, "L1(ρ) = {l1_rho} exceeds 0.15");
    assert!(l1_u < 0.15 * CS, "L1(u) = {l1_u} exceeds 0.15 c_s");

    // Intermediate density plateau pins ρ* (and hence the shock speed) directly.
    assert!(n_star > 20, "star region under-sampled: {n_star}");
    assert!(
        (star_rho - rs).abs() / rs < 0.08,
        "star-region ρ = {star_rho}, want ρ* = {rs} within 8%"
    );
}
