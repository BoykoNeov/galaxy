//! The headline IC validation (DESIGN.md M1): an isolated Plummer sphere,
//! sampled from its distribution function and evolved under self-gravity, must
//! STAY in equilibrium. Concretely, over many dynamical times:
//!   - total energy is conserved (oscillates within a bound, does not drift),
//!   - the virial ratio 2T/|W| stays ≈ 1,
//!   - the mass profile is stable — the half-mass radius neither drifts nor
//!     jumps (the robust, low-noise, relaxation-insensitive structural gate).
//!
//! This jointly exercises IC sampling (positions AND velocity scaling) and the
//! solver+integrator. A mis-scaled DF (e.g. a missing √2 in the escape speed)
//! would show up immediately as the sphere expanding or collapsing.
//!
//! Sizing note: at the N affordable for debug-mode O(N²) direct summation, the
//! two-body relaxation time (~N/(8 ln N) crossing times) is only a few times
//! the run length, so we deliberately keep the run to ~12 dynamical times and
//! gate tightly only on relaxation-robust quantities.

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
    let n_tdyn = 12.0; // long enough to expose a bad IC, short enough to stay
                       // below the N≈512 relaxation time.
    let steps = (n_tdyn * t_dyn / dt).round() as usize;

    let e0 = diagnostics::total_energy(&s, &solver);
    let rh0 = lagrangian_radius(&s, 0.5);
    let r10_0 = lagrangian_radius(&s, 0.1);
    let r90_0 = lagrangian_radius(&s, 0.9);

    let mut max_e_err = 0.0_f64;
    let mut max_rh_dev = 0.0_f64;
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

        let rh = lagrangian_radius(&s, 0.5);
        rh_series.push(rh);
        max_rh_dev = max_rh_dev.max(((rh - rh0) / rh0).abs());
        max_r10_dev = max_r10_dev.max(((lagrangian_radius(&s, 0.1) - r10_0) / r10_0).abs());
        max_r90_dev = max_r90_dev.max(((lagrangian_radius(&s, 0.9) - r90_0) / r90_0).abs());
    }

    // Energy: bounded oscillation, not drift (symplectic leapfrog).
    assert!(max_e_err < 2e-2, "energy not conserved: {max_e_err:e}");
    // Virial equilibrium maintained throughout.
    assert!(
        max_virial_dev < 0.10,
        "virial ratio wandered: {max_virial_dev}"
    );
    // Half-mass radius — the robust structural gate.
    assert!(max_rh_dev < 0.06, "half-mass radius deviated: {max_rh_dev}");
    // No *monotonic* drift: mean of the early window ≈ mean of the late window
    // (symmetric oscillation about rh0 is fine; one-way creep is not).
    let k = rh_series.len();
    let third = (k / 3).max(1);
    let early = rh_series[..third].iter().sum::<f64>() / third as f64;
    let late = rh_series[k - third..].iter().sum::<f64>() / third as f64;
    assert!(
        ((late - early) / rh0).abs() < 0.04,
        "half-mass radius drifting one way: {early} -> {late}"
    );
    // Inner/outer radii: loose diagnostics only (noisier and relaxation-prone).
    assert!(
        max_r10_dev < 0.25,
        "10% Lagrangian radius drifted: {max_r10_dev}"
    );
    assert!(
        max_r90_dev < 0.25,
        "90% Lagrangian radius drifted: {max_r90_dev}"
    );
}
