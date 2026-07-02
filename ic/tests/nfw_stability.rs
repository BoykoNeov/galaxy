//! Integration-level equilibrium check for the assembled NFW IC: a halo sampled
//! from the numerically Eddington-inverted DF (positions truncated at r_vir) and
//! evolved under self-gravity must hold together — no gross expansion or
//! collapse of the bound structure.
//!
//! Division of labour. The velocity distribution's SHAPE is validated tightly
//! elsewhere and cheaply: the Eddington machinery reproduces the closed-form
//! Hernquist and Plummer DFs to <3% (`eddington` test), and `nfw_sampling`'s
//! density-recovery integral confirms the SAME machinery is self-consistent with
//! the NFW density. This test then confirms the *assembled* IC — truncated
//! positions + those velocities + recentering — is a working equilibrium, which
//! the piecewise checks cannot.
//!
//! Caveats budgeted in the tolerances: (i) the ρ ∝ r⁻¹ cusp is unresolved by
//! Plummer softening, so structure is judged at the half-mass radius (≈3.6 r_s ≫
//! ε), not the center; (ii) the hard truncation at r_vir perturbs the outermost
//! shells, so the 90% radius (near the edge) is a loose guard only; (iii) the
//! velocities come from the *untruncated* DF, so a small initial re-virialization
//! of the outer halo is expected and allowed.

use galaxy_core::{diagnostics, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_ic::Nfw;
use galaxy_solvers::{BarnesHut, DirectSum};

fn lagrangian_radius(s: &State, frac: f64) -> f64 {
    let com = diagnostics::center_of_mass(s);
    let mut r: Vec<f64> = s.pos.iter().map(|p| (*p - com).length()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((frac * r.len() as f64).round() as usize).clamp(1, r.len());
    r[k - 1]
}

struct Measured {
    max_e_err: f64,
    rh_drift: f64,
    rh_band: f64,
    max_r90_dev: f64,
}

fn evolve_and_measure(model: &Nfw, mut s: State, solver: &mut dyn ForceSolver) -> Measured {
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let t_dyn = model.dynamical_time();
    let dt = 0.025 * t_dyn;
    let steps = (16.0 * t_dyn / dt).round() as usize;

    let e0 = diagnostics::total_energy(&s, solver);
    let r90_0 = lagrangian_radius(&s, 0.9);

    let mut m = Measured {
        max_e_err: 0.0,
        rh_drift: 0.0,
        rh_band: 0.0,
        max_r90_dev: 0.0,
    };
    let mut rh_series = Vec::new();

    let sample_every = (steps / 40).max(1);
    for step in 1..=steps {
        integ.step(&mut s, solver, &bg, dt);
        if step % sample_every != 0 {
            continue;
        }
        let e = diagnostics::total_energy(&s, solver);
        m.max_e_err = m.max_e_err.max(((e - e0) / e0).abs());
        rh_series.push(lagrangian_radius(&s, 0.5));
        m.max_r90_dev = m
            .max_r90_dev
            .max(((lagrangian_radius(&s, 0.9) - r90_0) / r90_0).abs());
    }

    let k = rh_series.len();
    let mean_rh = rh_series.iter().sum::<f64>() / k as f64;
    let third = (k / 3).max(1);
    let early = rh_series[..third].iter().sum::<f64>() / third as f64;
    let late = rh_series[k - third..].iter().sum::<f64>() / third as f64;
    m.rh_drift = ((late - early) / mean_rh).abs();
    m.rh_band = rh_series
        .iter()
        .map(|r| ((r - mean_rh) / mean_rh).abs())
        .fold(0.0_f64, f64::max);
    m
}

const N: usize = 512;
const SEED: u64 = 0x4F00;
/// Softening a few % of r_s: resolves the half-mass radius and the outer profile
/// while taming two-body scattering in the dense cusp.
const EPS_FRAC: f64 = 0.05;

fn assert_in_equilibrium(m: &Measured) {
    assert!(
        m.max_e_err < 2e-3,
        "energy not conserved: {:e}",
        m.max_e_err
    );
    assert!(
        m.rh_drift < 0.10,
        "half-mass radius drifting: {}",
        m.rh_drift
    );
    assert!(
        m.rh_band < 0.15,
        "half-mass radius oscillation too large: {}",
        m.rh_band
    );
    assert!(
        m.max_r90_dev < 0.25,
        "90% Lagrangian radius drifted grossly: {}",
        m.max_r90_dev
    );
}

#[test]
fn isolated_nfw_stays_in_equilibrium_direct_sum() {
    let model = Nfw::new(1.0, 1.0, 1.0, 10.0);
    let s = model.sample(N, SEED);
    let mut solver = DirectSum::new(model.g, EPS_FRAC * model.scale_radius);
    assert_in_equilibrium(&evolve_and_measure(&model, s, &mut solver));
}

#[test]
fn isolated_nfw_stays_in_equilibrium_barnes_hut() {
    let model = Nfw::new(1.0, 1.0, 1.0, 10.0);
    let s = model.sample(N, SEED);
    let mut solver = BarnesHut::new(model.g, EPS_FRAC * model.scale_radius, 0.5);
    assert_in_equilibrium(&evolve_and_measure(&model, s, &mut solver));
}
