//! Export an isothermal SPH shock-tube profile for the `plot_shock.py` overlay
//! (the M7b demo, sibling of the REBOUND cross-check). Runs the exact same IC /
//! solver as the `sph_shock_tube` gate, then dumps the central-column particle
//! profile plus the analytic Riemann constants as JSON.
//!
//! Usage:
//!   cargo run -p galaxy-validate --release --example export_shock_tube -- <out.json>
//!   python validate/sph/plot_shock.py <out.json>

use std::io::Write;

use galaxy_core::{DVec3, ForceSolver, Integrator, LeapfrogKdk, Species, State, StaticBackground};
use galaxy_solvers::sph::{density_adaptive, DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::DirectSum;

const CS: f64 = 1.0;
const RHO_L: f64 = 4.0;
const RHO_R: f64 = 1.0;

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

fn shock_tube_ic() -> State {
    const HALF_T: f64 = 4.0;
    const X_END: f64 = 4.0;
    let s_l = 0.5_f64;
    let m = RHO_L * s_l * s_l * s_l;
    let s_r = s_l * (RHO_L / RHO_R).cbrt();
    let axis = |s: f64| -> Vec<f64> {
        let n = (HALF_T / s).floor() as i64;
        (-n..=n).map(|k| k as f64 * s).collect()
    };
    let mut pos = Vec::new();
    let ys_l = axis(s_l);
    for ix in 1..=(X_END / s_l).floor() as i64 {
        let x = -(ix as f64) * s_l;
        for &y in &ys_l {
            for &z in &ys_l {
                pos.push(DVec3::new(x, y, z));
            }
        }
    }
    let ys_r = axis(s_r);
    for ix in 0..=(X_END / s_r).floor() as i64 {
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

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: export_shock_tube <out.json>");

    let mut state = shock_tube_ic();
    let params = HydroParams {
        eos: Eos::Isothermal { c_s: CS },
        ..HydroParams::default()
    };
    let cfg = DensityConfig::default();
    let mut solver = GravitySph::<DirectSum>::hydro_only(params, cfg.clone());
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let (dt, n_steps) = (0.02, 75u32);
    for _ in 0..n_steps {
        integ.step(&mut state, &mut solver as &mut dyn ForceSolver, &bg, dt);
    }
    let t = state.time;
    let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);

    // Central column, away from the longitudinal ends.
    let (mut xs, mut rhos, mut us) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..state.len() {
        let p = state.pos[i];
        if p.y.abs() > 1.0 || p.z.abs() > 1.0 || p.x.abs() > 3.5 {
            continue;
        }
        xs.push(p.x);
        rhos.push(dens.rho[i]);
        us.push(state.vel[i].x);
    }

    let rs = rho_star();
    let us_star = CS * (RHO_L / rs).ln();
    let s_shock = CS * (rs / RHO_R).sqrt();

    let arr = |v: &[f64]| -> String {
        v.iter()
            .map(|x| format!("{x:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let json = format!(
        "{{\"cs\":{CS},\"rho_l\":{RHO_L},\"rho_r\":{RHO_R},\"t\":{t:.6},\
         \"rho_star\":{rs:.6},\"u_star\":{us_star:.6},\"s_shock\":{s_shock:.6},\
         \"x\":[{}],\"rho\":[{}],\"u\":[{}]}}",
        arr(&xs),
        arr(&rhos),
        arr(&us)
    );
    let mut f = std::fs::File::create(&out).expect("create output");
    f.write_all(json.as_bytes()).expect("write json");
    println!(
        "wrote {} ({} central-column particles, t={t:.3}); ρ*={rs:.4} u*={us_star:.4} S={s_shock:.4}",
        out,
        xs.len()
    );
}
