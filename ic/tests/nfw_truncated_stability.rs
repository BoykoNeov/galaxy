//! Integration-level equilibrium check for the assembled exponentially-truncated
//! NFW IC. Like `nfw_stability` for the M5c hard-truncated halo, but with a key
//! difference that TIGHTENS the expectation: this IC is **self-consistent** —
//! positions and velocities share one (truncated) potential, so the M5c caveat
//! "velocities come from the untruncated DF ⇒ a small outer re-virialization is
//! expected" no longer applies. The smooth skirt also removes the hard r_vir edge
//! that perturbed the outermost shells. So the halo should hold together at least
//! as well as M5c, and we can hold the 90% radius to a tighter band.

use galaxy_core::{diagnostics, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_ic::{Nfw, TruncatedNfw};
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

fn evolve_and_measure(
    model: &TruncatedNfw,
    mut s: State,
    solver: &mut dyn ForceSolver,
) -> Measured {
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
    // Tighter than M5c's 0.25: the self-consistent DF + smooth skirt should not
    // shed/settle the outer 90% shell the way the untruncated-DF hard-cut IC did.
    assert!(
        m.max_r90_dev < 0.18,
        "90% Lagrangian radius drifted grossly: {}",
        m.max_r90_dev
    );
}

fn model() -> TruncatedNfw {
    TruncatedNfw::new(Nfw::new(1.0, 1.0, 1.0, 10.0), 3.0)
}

#[test]
fn isolated_truncated_nfw_stays_in_equilibrium_direct_sum() {
    let t = model();
    let s = t.sample(N, SEED);
    let mut solver = DirectSum::new(t.base.g, EPS_FRAC * t.base.scale_radius);
    assert_in_equilibrium(&evolve_and_measure(&t, s, &mut solver));
}

#[test]
fn isolated_truncated_nfw_stays_in_equilibrium_barnes_hut() {
    let t = model();
    let s = t.sample(N, SEED);
    let mut solver = BarnesHut::new(t.base.g, EPS_FRAC * t.base.scale_radius, 0.5);
    assert_in_equilibrium(&evolve_and_measure(&t, s, &mut solver));
}
