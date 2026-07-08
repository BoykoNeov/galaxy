//! Adiabatic (γ=1.4) Sod shock tube (E3b): the flagship gate for the evolved
//! internal-energy path. The canonical Sod IC (ρ_L=1,P_L=1 / ρ_R=0.125,P_R=0.1,
//! at rest) relaxes into a left rarefaction fan + contact + right shock. The SPH
//! run must match the exact adiabatic Riemann solution and — the point of this
//! milestone — obey the two thermodynamic statements the isothermal path never
//! could: total energy is oscillation-bounded (the sharp validator of the E3a
//! ½·Π viscous-heating term) and the global entropy `Σ mᵢ sᵢ` is monotonically
//! non-decreasing (2nd law; viscous heating ≥0 ⇒ ↑, rarefaction isentropic ⇒ ~0).
//!
//! Resolution note (advisor-vetted): the 8:1 density jump forces `s_r = 2·s_l`,
//! so the right (post-shock) gas is coarse and the ~2–3h shock smearing is WIDER
//! than the whole contact→shock star region — it will NOT resolve into a clean
//! plateau. That is expected, not a bug: the discriminating profile weight is on
//! the well-resolved LEFT rarefaction fan + the two undisturbed states, the
//! contact→shock band is excluded from the tight L1, and energy+entropy are the
//! sharp validators. Refining is not worth 8× the particles for E3b.
//!
//! Driven by the stock `LeapfrogKdkThermal` + `GravitySph::hydro_only` (gravity
//! off) with viscosity ON (defaults) — the real force+heating code, no periodic
//! BCs; padded free ends keep the boundary rarefactions out of the window.

use galaxy_core::{
    diagnostics, DVec3, ForceSolver, Integrator, LeapfrogKdkThermal, Species, State,
    StaticBackground,
};
use galaxy_solvers::sph::{density_adaptive, DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::DirectSum;

const GAMMA: f64 = 1.4;
const RHO_L: f64 = 1.0;
const P_L: f64 = 1.0;
const RHO_R: f64 = 0.125;
const P_R: f64 = 0.1;

fn sound(p: f64, rho: f64) -> f64 {
    (GAMMA * p / rho).sqrt()
}

// --- exact adiabatic Riemann oracle (Toro, ch. 4) -------------------------

/// Toro pressure function `f_K(p)` for side K: shock branch if `p > p_K`,
/// rarefaction branch otherwise. `p*` solves `f_L(p)+f_R(p)+(v_R−v_L)=0`.
fn f_k(p: f64, p_k: f64, rho_k: f64) -> f64 {
    let c_k = sound(p_k, rho_k);
    if p > p_k {
        // shock
        let a = 2.0 / ((GAMMA + 1.0) * rho_k);
        let b = (GAMMA - 1.0) / (GAMMA + 1.0) * p_k;
        (p - p_k) * (a / (p + b)).sqrt()
    } else {
        // rarefaction
        (2.0 * c_k / (GAMMA - 1.0)) * ((p / p_k).powf((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)
    }
}

/// Star pressure `p*`, bisected on `f_L(p)+f_R(p)=0` (v_L=v_R=0).
fn p_star() -> f64 {
    let f = |p: f64| f_k(p, P_L, RHO_L) + f_k(p, P_R, RHO_R);
    let (mut lo, mut hi) = (1e-9, 10.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Star velocity `v* = ½(f_R(p*) − f_L(p*))` (v_L=v_R=0).
fn v_star(ps: f64) -> f64 {
    0.5 * (f_k(ps, P_R, RHO_R) - f_k(ps, P_L, RHO_L))
}

/// Right-shock speed (into the undisturbed ρ_R gas at rest).
fn shock_speed(ps: f64) -> f64 {
    let c_r = sound(P_R, RHO_R);
    c_r * ((GAMMA + 1.0) / (2.0 * GAMMA) * ps / P_R + (GAMMA - 1.0) / (2.0 * GAMMA)).sqrt()
}

/// Self-similar Sod state `(ρ, v, P)` at `ξ = x/t`: undisturbed L | left
/// rarefaction fan | star-L | contact | star-R | right shock | undisturbed R.
/// (v_L=v_R=0, ρ_L>ρ_R ⇒ left rarefaction, right shock.)
fn sample(xi: f64, ps: f64, vs: f64) -> (f64, f64, f64) {
    let c_l = sound(P_L, RHO_L);
    if xi <= vs {
        // left of the contact — left rarefaction (p* < p_L)
        let s_head = -c_l;
        if xi <= s_head {
            return (RHO_L, 0.0, P_L); // undisturbed left
        }
        let cs_l = c_l * (ps / P_L).powf((GAMMA - 1.0) / (2.0 * GAMMA)); // sound speed behind fan
        let s_tail = vs - cs_l;
        if xi >= s_tail {
            // star-left: isentropic decompression of the left gas
            let rho = RHO_L * (ps / P_L).powf(1.0 / GAMMA);
            (rho, vs, ps)
        } else {
            // inside the fan (v_L=0)
            let v = 2.0 / (GAMMA + 1.0) * (c_l + xi);
            let c = 2.0 / (GAMMA + 1.0) * (c_l - (GAMMA - 1.0) / 2.0 * xi);
            let rho = RHO_L * (c / c_l).powf(2.0 / (GAMMA - 1.0));
            let p = P_L * (c / c_l).powf(2.0 * GAMMA / (GAMMA - 1.0));
            (rho, v, p)
        }
    } else {
        // right of the contact — right shock (p* > p_R)
        let s_shock = shock_speed(ps);
        if xi >= s_shock {
            (RHO_R, 0.0, P_R) // undisturbed right
        } else {
            // star-right: post-shock (Rankine–Hugoniot density)
            let beta = (GAMMA - 1.0) / (GAMMA + 1.0);
            let rho = RHO_R * (ps / P_R + beta) / (beta * ps / P_R + 1.0);
            (rho, vs, ps)
        }
    }
}

// --- oracle self-check ------------------------------------------------------

#[test]
fn sod_riemann_oracle_matches_hand_values() {
    // Canonical Sod star state (Toro Table 4.x / any published Sod reference) —
    // independent of the SPH solver, so a profile failure can't hide behind a
    // wrong oracle (isothermal-oracle discipline).
    let ps = p_star();
    let vs = v_star(ps);
    let rho_star_l = RHO_L * (ps / P_L).powf(1.0 / GAMMA);
    let beta = (GAMMA - 1.0) / (GAMMA + 1.0);
    let rho_star_r = RHO_R * (ps / P_R + beta) / (beta * ps / P_R + 1.0);
    assert!((ps - 0.30313).abs() < 1e-4, "p* = {ps}, want ≈0.30313");
    assert!((vs - 0.92745).abs() < 1e-4, "v* = {vs}, want ≈0.92745");
    assert!(
        (rho_star_l - 0.42632).abs() < 1e-4,
        "ρ*_L = {rho_star_l}, want ≈0.42632"
    );
    assert!(
        (rho_star_r - 0.26557).abs() < 1e-4,
        "ρ*_R = {rho_star_r}, want ≈0.26557"
    );

    // Fan-tail → star-left continuity: sampling just inside the tail must give
    // the star-left state (catches a fan/star branch-boundary bug).
    let cs_l = sound(P_L, RHO_L) * (ps / P_L).powf((GAMMA - 1.0) / (2.0 * GAMMA));
    let s_tail = vs - cs_l;
    let (rho_t, v_t, p_t) = sample(s_tail + 1e-9, ps, vs);
    assert!(
        (rho_t - rho_star_l).abs() < 1e-3 && (v_t - vs).abs() < 1e-3 && (p_t - ps).abs() < 1e-3,
        "fan tail does not meet star-left: ρ={rho_t} v={v_t} p={p_t}"
    );

    // Rankine–Hugoniot shock jump: mass flux across the (moving) shock must be
    // continuous, ρ_post(S−v_post)=ρ_pre(S−v_pre) (catches a shock branch bug).
    let s_shock = shock_speed(ps);
    let (rho_pre, v_pre, p_pre) = sample(s_shock + 1e-9, ps, vs); // undisturbed R
    let (rho_post, v_post, p_post) = sample(s_shock - 1e-9, ps, vs); // post-shock
    assert!(
        (rho_pre - RHO_R).abs() < 1e-12 && (p_pre - P_R).abs() < 1e-12 && v_pre.abs() < 1e-12,
        "pre-shock is not the undisturbed R state"
    );
    assert!(
        (rho_post - rho_star_r).abs() < 1e-3
            && (p_post - ps).abs() < 1e-3
            && (v_post - vs).abs() < 1e-3,
        "post-shock is not the star-R state"
    );
    let flux_pre = rho_pre * (s_shock - v_pre);
    let flux_post = rho_post * (s_shock - v_post);
    assert!(
        (flux_post - flux_pre).abs() / flux_pre < 1e-3,
        "Rankine–Hugoniot mass flux mismatch: pre={flux_pre} post={flux_post}"
    );
}

// --- IC ---------------------------------------------------------------------

/// Two glued equal-mass cubic-lattice blocks, ρ jump via spacing (8:1 ⇒ 2:1
/// spacing), per-particle `u` set for the 10:1 pressure jump
/// (`u = P/((γ−1)ρ)` ⇒ `u_L=2.5`, `u_R=2.0`). Transverse ±HALF_T, at rest, Gas.
fn sod_ic() -> State {
    const HALF_T: f64 = 4.0;
    const X_END: f64 = 4.0;
    let s_l = 0.5_f64;
    let m = RHO_L * s_l * s_l * s_l; // ρ_L = m/s_l³
    let s_r = s_l * (RHO_L / RHO_R).cbrt(); // ρ_R = m/s_r³ ⇒ s_r = 2·s_l = 1.0
    let u_l = P_L / ((GAMMA - 1.0) * RHO_L); // 2.5
    let u_r = P_R / ((GAMMA - 1.0) * RHO_R); // 2.0

    let mut pos = Vec::new();
    let mut us = Vec::new();
    let axis = |s: f64| -> Vec<f64> {
        let n = (HALF_T / s).floor() as i64;
        (-n..=n).map(|k| k as f64 * s).collect()
    };
    // Left (dense) block: x ∈ [−X_END, −s_l].
    let ys_l = axis(s_l);
    let nx_l = (X_END / s_l).floor() as i64;
    for ix in 1..=nx_l {
        let x = -(ix as f64) * s_l;
        for &y in &ys_l {
            for &z in &ys_l {
                pos.push(DVec3::new(x, y, z));
                us.push(u_l);
            }
        }
    }
    // Right (sparse) block: x ∈ [0, X_END].
    let ys_r = axis(s_r);
    let nx_r = (X_END / s_r).floor() as i64;
    for ix in 0..=nx_r {
        let x = ix as f64 * s_r;
        for &y in &ys_r {
            for &z in &ys_r {
                pos.push(DVec3::new(x, y, z));
                us.push(u_r);
            }
        }
    }
    let n = pos.len();
    let mut state = State::from_phase_space(pos, vec![DVec3::ZERO; n], vec![m; n]);
    for k in state.kind.iter_mut() {
        *k = Species::Gas;
    }
    for (slot, &val) in state.u.iter_mut().zip(us.iter()) {
        *slot = val;
    }
    state
}

/// Global entropy function `Σ mᵢ sᵢ`, `sᵢ = (γ−1)uᵢ/ρᵢ^{γ−1}` (= P/ρ^γ, the
/// adiabatic invariant). Isentropic in smooth flow, strictly ↑ across shocks —
/// per-particle `dsᵢ/dt = (γ−1)ρᵢ^{1−γ}·(viscous heating) ≥ 0` exactly, so the
/// extensive sum is the clean 2nd-law statement.
fn total_entropy(state: &State, rho: &[f64]) -> f64 {
    (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .map(|i| state.mass[i] * (GAMMA - 1.0) * state.u[i] / rho[i].powf(GAMMA - 1.0))
        .sum()
}

// --- gates ------------------------------------------------------------------

/// Fast smoke gate (non-ignored): a coarse Sod over a few steps must (a) heat
/// (global entropy rises — viscosity dissipating at the forming shock), (b)
/// conserve total energy to an oscillation bound, (c) leave total momentum at
/// ~0 (internal pairwise-antisymmetric forces). Catches gross wiring bugs
/// (heating dropped/sign-flipped, EOS mis-wired) without the full run cost.
#[test]
fn sod_shock_tube_heats_and_conserves_energy() {
    let mut state = sod_ic();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma: GAMMA },
        ..HydroParams::default() // viscosity ON
    };
    let cfg = DensityConfig::default();
    let mut solver = GravitySph::<DirectSum>::hydro_only(params, cfg.clone());
    let mut integ = LeapfrogKdkThermal::new();
    let bg = StaticBackground;

    let dens0 = density_adaptive(&state.pos, &state.mass, &cfg, None);
    let s0 = total_entropy(&state, &dens0.rho);
    let e0 = diagnostics::total_energy(&state, &solver);
    let p0 = diagnostics::total_momentum(&state);

    let (dt, n_steps) = (0.02, 15);
    let mut max_e_err = 0.0_f64;
    for _ in 0..n_steps {
        integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
        let e = diagnostics::total_energy(&state, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
    }
    let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);
    let s1 = total_entropy(&state, &dens.rho);

    assert!(max_e_err < 5e-2, "energy drift too large: {max_e_err:e}");
    assert!(
        (diagnostics::total_momentum(&state) - p0).length() < 1e-6,
        "total momentum not conserved (internal forces): {}",
        (diagnostics::total_momentum(&state) - p0).length()
    );
    assert!(
        s1 > s0,
        "shock heating must raise the global entropy: {s0} → {s1}"
    );
}

/// Dynamical gate (ignored — run `--release --ignored --nocapture`): a
/// ~2700-particle Sod tube over 50 steps (t≈1.0) must match the exact Riemann
/// profiles on the resolved LEFT (rarefaction + undisturbed states), keep total
/// energy oscillation-bounded, and drive the global entropy monotonically up.
#[test]
#[ignore = "dynamical validation: ~2700-particle adiabatic Sod SPH run over 50 steps (run --release --ignored)"]
fn sod_shock_tube_matches_the_exact_riemann_solution() {
    let mut state = sod_ic();
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma: GAMMA },
        ..HydroParams::default() // viscosity ON
    };
    let cfg = DensityConfig::default();
    let mut solver = GravitySph::<DirectSum>::hydro_only(params, cfg.clone());
    let mut integ = LeapfrogKdkThermal::new();
    let bg = StaticBackground;

    let ps = p_star();
    let vs = v_star(ps);
    let s_shock = shock_speed(ps);

    // Tolerances calibrated from the --release --ignored --nocapture run (E2b
    // discipline), each ~1.5× the observed floor so this stays a regression gate:
    //   L1(rho) 0.130 → 0.20   (resolved: fan + undisturbed states)
    //   L1(v)   0.113 → 0.18
    //   L1(P)   0.200 → 0.30   (P varies steeply through the fan → higher floor)
    //   max_e_err 3.9e-3 → 1.2e-2 (bounded oscillation — the SHARP validator of
    //           the E3a ½·Π coefficient: a wrong factor DRIFTS, not oscillates)
    // NO star-plateau pin: at 8:1 the shock's ±2h smearing footprint engulfs the
    // whole contact→shock star region, so neither ρ* nor p* forms a clean plateau
    // (star_p reads ~25% low — a resolution bias, not a bug; advisor-predicted).
    // Pinning to the sim's own smeared value would violate "compare to independent
    // expectations". p*/shock physics is instead validated by energy (½Π) + the
    // 2nd-law entropy gate; star_p and the s*_R jump are PRINTED for diagnostics.
    const L1_RHO_TOL: f64 = 0.20;
    const L1_V_TOL: f64 = 0.18;
    const L1_P_TOL: f64 = 0.30;
    const E_TOL: f64 = 1.2e-2;
    const S_EPS: f64 = 1e-3; // entropy monotonicity slack (discrete leapfrog wobble)

    // Baseline entropy/energy from AFTER the first step (the sharp IC smooths in
    // one step; advisor: absorb that transient rather than baselining at t=0).
    let (dt, n_steps) = (0.02, 50);
    integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
    let e0 = diagnostics::total_energy(&state, &solver);
    let p0 = diagnostics::total_momentum(&state);
    let dens1 = density_adaptive(&state.pos, &state.mass, &cfg, None);
    let mut s_prev = total_entropy(&state, &dens1.rho);
    let mut max_e_err = 0.0_f64;

    // Entropy sampled every few steps: monotonic non-decreasing within slack.
    let s_sample_every = 10usize;
    for step in 2..=n_steps {
        integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
        let e = diagnostics::total_energy(&state, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
        if step % s_sample_every == 0 {
            let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);
            let s_now = total_entropy(&state, &dens.rho);
            assert!(
                s_now >= s_prev - S_EPS * s_prev.abs(),
                "entropy decreased at step {step}: {s_prev} → {s_now}"
            );
            s_prev = s_now;
        }
    }

    let t = state.time;
    let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);

    // Profile L1 over the central column, away from transverse/longitudinal free
    // ends. Discriminating (tight) mask = the resolved regions: left of contact
    // (undisturbed-L + fan + star-L, up to a small pre-contact guard) OR past the
    // shock (undisturbed-R). Excluded band = contact→shock, where the coarse
    // right gas smears the star region wider than the plateau itself.
    let (mut l1_rho, mut l1_v, mut l1_p) = (0.0, 0.0, 0.0);
    let mut n_tight = 0usize;
    // Smeared star-region mean pressure — PRINTED for diagnostics only (reads
    // ~25% low: the shock ramp, not just the contact, engulfs the whole region).
    let mut star_p_sum = 0.0;
    let mut n_star = 0usize;
    let vs_guard = vs - 0.1; // just left of the contact
    let shock_guard = s_shock + 0.25; // just right of the smeared shock
    let cs_l = sound(P_L, RHO_L) * (ps / P_L).powf((GAMMA - 1.0) / (2.0 * GAMMA));
    let s_tail = vs - cs_l;
    for i in 0..state.len() {
        let p = state.pos[i];
        // Central column, ≥3 from ±4 transverse faces (>2h ⇒ no kernel deficit);
        // x-window clears both free-end rarefactions by t≈1.0.
        if p.y.abs() > 1.0 || p.z.abs() > 1.0 || p.x < -2.6 || p.x > 2.7 {
            continue;
        }
        let xi = p.x / t;
        let (rho_ref, v_ref, p_ref) = sample(xi, ps, vs);
        let p_sim = (GAMMA - 1.0) * dens.rho[i] * state.u[i];
        // Tight mask = resolved regions only: left of the contact (undisturbed-L
        // + fan + star-L) or past the smeared shock (undisturbed-R). The
        // contact→shock band is excluded (unresolvable at 8:1).
        if xi < vs_guard || xi > shock_guard {
            l1_rho += (dens.rho[i] - rho_ref).abs() / rho_ref;
            l1_v += (state.vel[i].x - v_ref).abs();
            l1_p += (p_sim - p_ref).abs() / p_ref;
            n_tight += 1;
        }
        if xi > s_tail + 0.15 && xi < s_shock - 0.15 {
            star_p_sum += p_sim;
            n_star += 1;
        }
    }
    assert!(n_tight > 100, "too few tight-mask particles: {n_tight}");
    l1_rho /= n_tight as f64;
    l1_v /= n_tight as f64;
    l1_p /= n_tight as f64;
    let star_p = star_p_sum / n_star.max(1) as f64;
    // Secondary (PRINTED, not gated — "contact-blip-muddied", plan E3b): the
    // Rankine–Hugoniot entropy jump s*_R/s_R across the shock (~5.5%).
    let beta = (GAMMA - 1.0) / (GAMMA + 1.0);
    let rho_star_r = RHO_R * (ps / P_R + beta) / (beta * ps / P_R + 1.0);
    let s_jump = (ps / rho_star_r.powf(GAMMA)) / (P_R / RHO_R.powf(GAMMA));
    println!(
        "t={t:.3} n_tight={n_tight} n_star={n_star} L1(rho)={l1_rho:.4} \
         L1(v)={l1_v:.4} L1(P)={l1_p:.4} star_p={star_p:.4}(smeared, p*={ps:.4}) \
         s*_R/s_R={s_jump:.4} max_e_err={max_e_err:e} s_final={s_prev:.5}"
    );

    assert!(l1_rho < L1_RHO_TOL, "L1(ρ) = {l1_rho} exceeds {L1_RHO_TOL}");
    assert!(l1_v < L1_V_TOL, "L1(v) = {l1_v} exceeds {L1_V_TOL}");
    assert!(l1_p < L1_P_TOL, "L1(P) = {l1_p} exceeds {L1_P_TOL}");
    assert!(
        max_e_err < E_TOL,
        "energy oscillation too large: {max_e_err:e}"
    );
    assert!(
        (diagnostics::total_momentum(&state) - p0).length() < 1e-6,
        "total momentum not conserved: {}",
        (diagnostics::total_momentum(&state) - p0).length()
    );
}
