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
//! Sizing notes:
//!   - N=512 keeps debug-mode O(N²) direct summation to a few seconds (this
//!     test runs on every `cargo test`). The fixed, representative seed below
//!     passes the gates with 1.4–2.5× margin; a seed survey confirms the gates
//!     hold for the large majority of draws, so they are not tuned to one seed.
//!   - ~12 dynamical times is long enough to expose a non-equilibrium IC, yet
//!     stays below the two-body relaxation time (~N/(8 ln N) crossing times) so
//!     we are testing the IC, not collisional relaxation.
//!   - Stability is judged against the half-mass radius's OWN time-mean, not its
//!     initial single-snapshot value (which carries the same finite-N noise).
//!     Symmetric breathing about the mean is fine — exactly like the energy.

use galaxy_core::{diagnostics, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_ic::Plummer;
use galaxy_solvers::DirectSum;

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

#[test]
fn isolated_plummer_stays_in_equilibrium() {
    let model = Plummer::new(1.0, 1.0, 1.0);
    let n = 512;
    let mut s = model.sample(n, 0xA11CE5);

    // ε a few % of a: small enough that the softened force tracks the exact
    // Plummer potential (so the DF really is an equilibrium of the integrated
    // force), large enough to tame discrete two-body scattering.
    let eps = 0.05 * model.scale_radius;
    let mut solver = DirectSum::new(model.g, eps);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let t_dyn = model.dynamical_time();
    let dt = 0.02 * t_dyn;
    let n_tdyn = 12.0;
    let steps = (n_tdyn * t_dyn / dt).round() as usize;

    let e0 = diagnostics::total_energy(&s, &solver);
    let r10_0 = lagrangian_radius(&s, 0.1);
    let r90_0 = lagrangian_radius(&s, 0.9);

    let mut max_e_err = 0.0_f64;
    let mut max_virial_dev = 0.0_f64;
    let mut max_r10_dev = 0.0_f64;
    let mut max_r90_dev = 0.0_f64;
    let mut rh_series = Vec::new();

    let sample_every = (steps / 40).max(1);
    for step in 1..=steps {
        integ.step(&mut s, &mut solver, &bg, dt);
        if step % sample_every != 0 {
            continue;
        }
        let e = diagnostics::total_energy(&s, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());

        let t = diagnostics::kinetic_energy(&s);
        let w = solver.potential_energy(&s);
        max_virial_dev = max_virial_dev.max((2.0 * t / w.abs() - 1.0).abs());

        rh_series.push(lagrangian_radius(&s, 0.5));
        max_r10_dev = max_r10_dev.max(((lagrangian_radius(&s, 0.1) - r10_0) / r10_0).abs());
        max_r90_dev = max_r90_dev.max(((lagrangian_radius(&s, 0.9) - r90_0) / r90_0).abs());
    }

    // Energy: bounded oscillation, not drift (symplectic leapfrog).
    assert!(max_e_err < 1e-3, "energy not conserved: {max_e_err:e}");

    // Virial equilibrium maintained throughout: 2T/|W| ≈ 1.
    assert!(
        max_virial_dev < 0.06,
        "virial ratio wandered from 1: {max_virial_dev}"
    );

    // Half-mass radius — the robust structural gate, judged against its own
    // time-mean: no monotonic drift (early-window mean ≈ late-window mean) and
    // bounded oscillation (every sample within a band of the mean).
    let k = rh_series.len();
    let mean_rh = rh_series.iter().sum::<f64>() / k as f64;
    let third = (k / 3).max(1);
    let early = rh_series[..third].iter().sum::<f64>() / third as f64;
    let late = rh_series[k - third..].iter().sum::<f64>() / third as f64;
    let drift = ((late - early) / mean_rh).abs();
    let band = rh_series
        .iter()
        .map(|r| ((r - mean_rh) / mean_rh).abs())
        .fold(0.0_f64, f64::max);
    assert!(drift < 0.06, "half-mass radius drifting one way: {drift}");
    assert!(
        band < 0.085,
        "half-mass radius oscillation too large: {band}"
    );

    // Inner/outer Lagrangian radii: loose diagnostics only. The 10% radius is
    // the noisiest and first to feel relaxation; the 90% radius samples the
    // sparse halo. They guard against gross collapse/expansion, not fine detail.
    assert!(
        max_r10_dev < 0.20,
        "10% Lagrangian radius drifted grossly: {max_r10_dev}"
    );
    assert!(
        max_r90_dev < 0.15,
        "90% Lagrangian radius drifted grossly: {max_r90_dev}"
    );
}
