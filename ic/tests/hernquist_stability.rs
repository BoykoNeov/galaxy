//! The DF-shape validation for the Hernquist model: an isolated sphere sampled
//! from its closed-form isotropic f(ℰ) and evolved under self-gravity must STAY
//! in equilibrium. The `hernquist_sampling` tests pin the DF's normalization and
//! the mass profile; only *evolution* can confirm the velocity distribution has
//! the right SHAPE (a mis-shaped f with the right total kinetic energy still
//! passes a single-snapshot virial check). Over many dynamical times we require:
//!   - total energy conserved (bounded oscillation, no drift),
//!   - virial ratio 2T/|W| stays ≈ 1,
//!   - the half-mass radius is stable about its own time-mean.
//!
//! Cusp caveat: Hernquist has a ρ ∝ r⁻¹ central cusp that Plummer softening
//! cannot resolve, so the softened force departs from the exact Hernquist force
//! for r ≲ ε and the core re-virializes slightly. We therefore judge STRUCTURE
//! at the half-mass radius (r_h ≈ 2.414a ≫ ε), treat the inner Lagrangian radius
//! as a loose gross-collapse guard only, and budget that mild core adjustment in
//! the tolerances — which still sit far from a mis-scaled DF (a dropped √2 in the
//! escape speed drives 2T/|W| to ≈0.5, nowhere near these gates).

use galaxy_core::{diagnostics, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_ic::Hernquist;
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
    max_virial_dev: f64,
    rh_drift: f64,
    rh_band: f64,
    max_r90_dev: f64,
}

fn evolve_and_measure(model: &Hernquist, mut s: State, solver: &mut dyn ForceSolver) -> Measured {
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let t_dyn = model.dynamical_time();
    let dt = 0.02 * t_dyn;
    let steps = (12.0 * t_dyn / dt).round() as usize;

    let e0 = diagnostics::total_energy(&s, solver);
    let r90_0 = lagrangian_radius(&s, 0.9);

    let mut m = Measured {
        max_e_err: 0.0,
        max_virial_dev: 0.0,
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

        let t = diagnostics::kinetic_energy(&s);
        let w = solver.potential_energy(&s);
        m.max_virial_dev = m.max_virial_dev.max((2.0 * t / w.abs() - 1.0).abs());

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
const SEED: u64 = 0x4E57;
/// ε a few % of a: small enough that r_h and the outer profile track the exact
/// Hernquist potential, large enough to tame two-body scattering in the cusp.
const EPS_FRAC: f64 = 0.05;

fn assert_in_equilibrium(m: &Measured) {
    assert!(
        m.max_e_err < 2e-3,
        "energy not conserved: {:e}",
        m.max_e_err
    );
    assert!(
        m.max_virial_dev < 0.10,
        "virial ratio wandered from 1: {}",
        m.max_virial_dev
    );
    assert!(
        m.rh_drift < 0.08,
        "half-mass radius drifting: {}",
        m.rh_drift
    );
    assert!(
        m.rh_band < 0.12,
        "half-mass radius oscillation too large: {}",
        m.rh_band
    );
    // Outer (sparse r⁻⁴ halo) radius: loose gross expansion/collapse guard only.
    assert!(
        m.max_r90_dev < 0.20,
        "90% Lagrangian radius drifted grossly: {}",
        m.max_r90_dev
    );
}

#[test]
fn isolated_hernquist_stays_in_equilibrium_direct_sum() {
    let model = Hernquist::new(1.0, 1.0, 1.0);
    let s = model.sample(N, SEED);
    let mut solver = DirectSum::new(model.g, EPS_FRAC * model.scale_radius);
    assert_in_equilibrium(&evolve_and_measure(&model, s, &mut solver));
}

#[test]
fn isolated_hernquist_stays_in_equilibrium_barnes_hut() {
    let model = Hernquist::new(1.0, 1.0, 1.0);
    let s = model.sample(N, SEED);
    let mut solver = BarnesHut::new(model.g, EPS_FRAC * model.scale_radius, 0.5);
    assert_in_equilibrium(&evolve_and_measure(&model, s, &mut solver));
}
