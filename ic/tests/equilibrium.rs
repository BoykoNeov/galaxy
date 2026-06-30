//! The headline IC validation (DESIGN.md M1): an isolated Plummer sphere,
//! sampled from its distribution function and evolved under self-gravity, must
//! STAY in equilibrium. Concretely, over many dynamical times:
//!   - total energy is conserved (oscillates within a bound, does not drift),
//!   - the virial ratio 2T/|W| stays ≈ 1,
//!   - the mass profile is stable — the half-mass radius neither drifts nor
//!     oscillates beyond a small band (the robust structural gate).
//!
//! This jointly exercises IC sampling (positions AND velocity scaling) and the
//! solver+integrator. A mis-scaled DF (e.g. a dropped √2 in the escape speed)
//! would show up immediately: the virial ratio would sit far from 1 and the
//! sphere would expand or collapse monotonically.
//!
//! Run under BOTH force solvers (M1 is "BarnesHut + equilibrium test"):
//!   - `DirectSum` is the exact softened oracle: it confirms the IC itself is an
//!     equilibrium of the *exact* force, so any failure is the sampler's fault.
//!   - `BarnesHut` is the workhorse M1 is named for: it confirms the tree's
//!     O(θ²) force approximation is accurate enough to hold the galaxy together,
//!     not just reproduce a single snapshot's accelerations. At θ=0.5 it meets
//!     the very same gates as the exact oracle (see `assert_in_equilibrium`), so
//!     the M1 claim is the strong one — the tree holds the galaxy as well as
//!     direct summation, not merely "well enough".
//!
//! Sizing notes:
//!   - N=512 keeps debug-mode summation to a few seconds (this runs on every
//!     `cargo test`). The fixed seed passes the gates with comfortable margin.
//!   - ~12 dynamical times is long enough to expose a non-equilibrium IC, yet
//!     stays below the two-body relaxation time (~N/(8 ln N) crossing times) so
//!     we are testing the IC, not collisional relaxation.
//!   - Stability is judged against the half-mass radius's OWN time-mean, not its
//!     initial single-snapshot value (which carries the same finite-N noise).
//!     Symmetric breathing about the mean is fine — exactly like the energy.

use galaxy_core::{diagnostics, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_ic::Plummer;
use galaxy_solvers::{BarnesHut, DirectSum};

/// Radius enclosing fraction `frac` of the particles, measured about the
/// *instantaneous* center of mass so that any bulk COM drift cannot masquerade
/// as a change in the internal structure.
fn lagrangian_radius(s: &State, frac: f64) -> f64 {
    let com = diagnostics::center_of_mass(s);
    let mut r: Vec<f64> = s.pos.iter().map(|p| (*p - com).length()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ((frac * r.len() as f64).round() as usize).clamp(1, r.len());
    r[k - 1]
}

/// The worst-case deviations seen over the run, against which each solver's gates
/// are asserted.
struct Measured {
    max_e_err: f64,
    max_virial_dev: f64,
    rh_drift: f64,
    rh_band: f64,
    max_r10_dev: f64,
    max_r90_dev: f64,
}

/// Evolve a freshly-sampled sphere for ~12 dynamical times under the given solver
/// and return the worst-case excursions of the equilibrium diagnostics.
fn evolve_and_measure(model: &Plummer, mut s: State, solver: &mut dyn ForceSolver) -> Measured {
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let t_dyn = model.dynamical_time();
    let dt = 0.02 * t_dyn;
    let steps = (12.0 * t_dyn / dt).round() as usize;

    let e0 = diagnostics::total_energy(&s, solver);
    let r10_0 = lagrangian_radius(&s, 0.1);
    let r90_0 = lagrangian_radius(&s, 0.9);

    let mut m = Measured {
        max_e_err: 0.0,
        max_virial_dev: 0.0,
        rh_drift: 0.0,
        rh_band: 0.0,
        max_r10_dev: 0.0,
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
        m.max_r10_dev = m
            .max_r10_dev
            .max(((lagrangian_radius(&s, 0.1) - r10_0) / r10_0).abs());
        m.max_r90_dev = m
            .max_r90_dev
            .max(((lagrangian_radius(&s, 0.9) - r90_0) / r90_0).abs());
    }

    // Half-mass radius judged against its own time-mean: no monotonic drift
    // (early-window mean ≈ late-window mean) and bounded oscillation.
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
const SEED: u64 = 0xA11CE5;
/// ε a few % of a: small enough that the softened force tracks the exact Plummer
/// potential (so the DF really is an equilibrium of the integrated force), large
/// enough to tame discrete two-body scattering.
const EPS_FRAC: f64 = 0.05;

/// The equilibrium gates. The SAME bounds apply to both solvers: BarnesHut at
/// θ=0.5 meets the exact oracle's tolerances, so M1's claim is the strong one —
/// the tree workhorse holds the galaxy together as well as direct summation, not
/// merely "well enough". Tolerances are the deterministic single-seed excursions
/// with comfortable headroom; energy is a bounded symplectic oscillation (the
/// tree force is non-conservative in principle, but at θ=0.5 its effect on energy
/// stays below the integrator's own bound over this baseline).
fn assert_in_equilibrium(m: &Measured) {
    assert!(
        m.max_e_err < 1e-3,
        "energy not conserved: {:e}",
        m.max_e_err
    );
    assert!(
        m.max_virial_dev < 0.06,
        "virial ratio wandered from 1: {}",
        m.max_virial_dev
    );
    assert!(
        m.rh_drift < 0.06,
        "half-mass radius drifting: {}",
        m.rh_drift
    );
    assert!(
        m.rh_band < 0.085,
        "half-mass radius oscillation too large: {}",
        m.rh_band
    );
    // Inner/outer Lagrangian radii: loose gross-collapse/expansion guards only —
    // the 10% radius is the noisiest, the 90% radius samples the sparse halo.
    assert!(
        m.max_r10_dev < 0.20,
        "10% Lagrangian radius drifted grossly: {}",
        m.max_r10_dev
    );
    assert!(
        m.max_r90_dev < 0.15,
        "90% Lagrangian radius drifted grossly: {}",
        m.max_r90_dev
    );
}

#[test]
fn isolated_plummer_stays_in_equilibrium_direct_sum() {
    // The exact softened oracle: any failure here is the IC sampler's fault.
    let model = Plummer::new(1.0, 1.0, 1.0);
    let s = model.sample(N, SEED);
    let mut solver = DirectSum::new(model.g, EPS_FRAC * model.scale_radius);
    assert_in_equilibrium(&evolve_and_measure(&model, s, &mut solver));
}

#[test]
fn isolated_plummer_stays_in_equilibrium_barnes_hut() {
    // The M1 workhorse. θ=0.5 is the usual production accuracy/speed trade; its
    // RMS force error is well under 1%, and the sphere stays in equilibrium to
    // the very same gates as the exact oracle.
    let model = Plummer::new(1.0, 1.0, 1.0);
    let s = model.sample(N, SEED);
    let mut solver = BarnesHut::new(model.g, EPS_FRAC * model.scale_radius, 0.5);
    assert_in_equilibrium(&evolve_and_measure(&model, s, &mut solver));
}
