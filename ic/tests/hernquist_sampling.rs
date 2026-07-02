//! Statistical validation of Hernquist IC sampling. The analytic profile
//! functions must be mutually consistent, the closed-form distribution function
//! must be self-consistent with the density (the integral that recovers ρ from
//! f pins both the DF's *shape* and its *normalization* without trusting any
//! external constant), and a realization drawn from f(ℰ) must reproduce the
//! analytic mass profile, be properly recentered, and satisfy the virial
//! theorem. Every expectation is an independent closed form, not the code's own
//! output.

use galaxy_core::{diagnostics, State};
use galaxy_ic::Hernquist;
use proptest::prelude::*;

const PI: f64 = std::f64::consts::PI;

fn sorted_radii(s: &State) -> Vec<f64> {
    let mut r: Vec<f64> = s.pos.iter().map(|p| p.length()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r
}

// ---------- analytic profile self-consistency (no sampling) ----------

#[test]
fn density_and_mass_derivative_agree() {
    let p = Hernquist::new(1.0, 3.0, 2.0);
    // ρ(r) = M a / (2π r (r+a)³): a central r⁻¹ cusp, r⁻⁴ envelope.
    let (m, a) = (p.total_mass, p.scale_radius);
    for &r in &[0.3_f64, 1.0, 2.5, 5.0] {
        let rho = m * a / (2.0 * PI * r * (r + a).powi(3));
        assert!(
            (p.density(r) - rho).abs() < 1e-12 * rho,
            "ρ({r}) closed form wrong"
        );
        // The shell mass 4π r² ρ(r) must equal dM(<r)/dr.
        let h = 1e-6;
        let dmdr = (p.enclosed_mass(r + h) - p.enclosed_mass(r - h)) / (2.0 * h);
        let shell = 4.0 * PI * r * r * p.density(r);
        assert!(
            (dmdr - shell).abs() < 1e-4 * shell,
            "r={r}: dM/dr={dmdr} vs 4πr²ρ={shell}"
        );
    }
}

#[test]
fn enclosed_mass_limits_and_half_mass_radius() {
    let p = Hernquist::new(2.0, 5.0, 1.5);
    assert!(p.enclosed_mass(0.0).abs() < 1e-12, "M(<0) should be 0");
    assert!(
        (p.enclosed_mass(1e6 * p.scale_radius) - p.total_mass).abs() < 1e-3 * p.total_mass,
        "M(<∞) should approach total mass"
    );
    // r_h = (1 + √2) a ≈ 2.41421 a and must enclose exactly half the mass.
    let rh = p.half_mass_radius();
    assert!(
        (rh / p.scale_radius - (1.0 + 2.0_f64.sqrt())).abs() < 1e-9,
        "r_h/a = {}",
        rh / p.scale_radius
    );
    assert!(
        (p.enclosed_mass(rh) - 0.5 * p.total_mass).abs() < 1e-9 * p.total_mass,
        "r_h does not enclose half the mass"
    );
}

#[test]
fn potential_and_energies_match_closed_forms() {
    let (g, m, a) = (1.3, 2.7, 0.9);
    let p = Hernquist::new(g, m, a);
    // Φ(0) = −GM/a.
    assert!((p.potential(0.0) + g * m / a).abs() < 1e-12, "Φ(0) wrong");
    assert!(
        (p.potential(a) + g * m / (2.0 * a)).abs() < 1e-12,
        "Φ(a) = −GM/2a wrong"
    );
    // W = −G M² / (6a); virial T = −W/2; t_dyn = √(a³/GM).
    let w = -g * m * m / (6.0 * a);
    assert!(
        (p.potential_energy() - w).abs() < 1e-12 * w.abs(),
        "W wrong: {} vs {w}",
        p.potential_energy()
    );
    assert!(
        (p.kinetic_energy() + 0.5 * w).abs() < 1e-12 * w.abs(),
        "T should be −W/2"
    );
    assert!(
        (p.dynamical_time() - (a * a * a / (g * m)).sqrt()).abs() < 1e-12,
        "t_dyn wrong"
    );
}

// ---------- distribution-function self-consistency ----------

/// The isotropic DF and the density must satisfy
///   ρ(r) = 4π ∫₀^Ψ f(ℰ) √(2(Ψ−ℰ)) dℰ,   Ψ(r) = −Φ(r).
/// Recovering ρ from f is the oracle that pins the DF's normalization AND shape
/// with no external constant to trust — density is independently checked above.
#[test]
fn df_integrates_to_density() {
    let (g, m, a) = (1.0, 1.0, 1.0);
    let p = Hernquist::new(g, m, a);
    // Radii chosen so Ψ(r) = GM/(r+a) stays clear of the ℰ→GM/a divergence.
    for &r in &[0.5_f64, 1.0, 2.0, 4.0] {
        let psi = -p.potential(r); // relative potential Ψ = GM/(r+a) > 0
                                   // Midpoint rule over ℰ ∈ (0, Ψ); integrand vanishes at both ends.
        let n = 20_000;
        let de = psi / n as f64;
        let mut acc = 0.0;
        for k in 0..n {
            let e = (k as f64 + 0.5) * de;
            acc += p.df(e) * (2.0 * (psi - e)).sqrt();
        }
        let recovered = 4.0 * PI * acc * de;
        let rho = p.density(r);
        assert!(
            (recovered - rho).abs() < 3e-3 * rho,
            "r={r}: ∫f recovered ρ={recovered} vs analytic ρ={rho}"
        );
    }
}

#[test]
fn df_is_nonnegative_and_vanishes_off_support() {
    let p = Hernquist::new(1.0, 1.0, 1.0);
    let psi_max = -p.potential(0.0); // GM/a, the deepest binding energy
                                     // Zero (or numerically non-positive contribution) outside the support.
    assert_eq!(p.df(0.0), 0.0, "f(0) should be 0");
    assert_eq!(p.df(-0.3), 0.0, "f(ℰ<0) should be 0");
    // Strictly positive and finite on the interior of the support.
    for frac in [0.1_f64, 0.3, 0.5, 0.7, 0.9] {
        let f = p.df(frac * psi_max);
        assert!(f > 0.0 && f.is_finite(), "f at frac={frac} = {f}");
    }
}

// ---------- sampled-realization statistics ----------

#[test]
fn realized_mass_profile_matches_analytic_cdf() {
    let p = Hernquist::new(1.0, 1.0, 1.0);
    let n = 40_000;
    let s = p.sample(n, 0x5EED);
    let r = sorted_radii(&s);
    for &rr in &[0.5_f64, 1.0, 2.414, 4.0, 8.0] {
        let frac = r.partition_point(|&x| x <= rr) as f64 / n as f64;
        let expected = p.enclosed_mass(rr) / p.total_mass;
        let tol = 4.0 * (expected * (1.0 - expected) / n as f64).sqrt() + 1e-3;
        assert!(
            (frac - expected).abs() < tol,
            "r={rr}: empirical={frac} expected={expected} tol={tol}"
        );
    }
}

#[test]
fn realized_half_mass_radius_matches_analytic() {
    let p = Hernquist::new(1.0, 1.0, 1.0);
    let n = 40_000;
    let s = p.sample(n, 0xBEEF);
    let median = sorted_radii(&s)[n / 2];
    assert!(
        (median - p.half_mass_radius()).abs() < 0.03 * p.half_mass_radius(),
        "median radius {median} vs r_h {}",
        p.half_mass_radius()
    );
}

#[test]
fn sample_is_recentered_equal_mass_and_deterministic() {
    let p = Hernquist::new(1.0, 1.0, 1.0);
    let n = 5000;
    let s = p.sample(n, 42);
    assert!(
        diagnostics::center_of_mass(&s).length() < 1e-9,
        "COM not zeroed"
    );
    assert!(
        diagnostics::total_momentum(&s).length() < 1e-9,
        "net momentum not zeroed"
    );
    let mtot: f64 = s.mass.iter().sum();
    assert!((mtot - p.total_mass).abs() < 1e-12, "masses don't sum to M");
    let m_each = p.total_mass / n as f64;
    assert!(
        s.mass.iter().all(|&m| (m - m_each).abs() < 1e-15),
        "masses not equal"
    );
    assert_eq!(s.len(), n);
    let s2 = p.sample(n, 42);
    assert_eq!(s.pos, s2.pos, "not deterministic (pos)");
    assert_eq!(s.vel, s2.vel, "not deterministic (vel)");
    let s3 = p.sample(n, 43);
    assert!(s3.pos != s.pos, "seed had no effect");
}

#[test]
fn realized_virial_ratio_is_unity() {
    let p = Hernquist::new(1.0, 1.0, 1.0);
    let n = 20_000;
    let s = p.sample(n, 0xC0FFEE);
    let t = diagnostics::kinetic_energy(&s);
    let w = p.potential_energy(); // analytic |W|
    let virial = 2.0 * t / w.abs();
    // Velocities drawn from f(ℰ) make ⟨2T/|W|⟩ → 1; fluctuation is O(1/√N).
    assert!((virial - 1.0).abs() < 0.05, "2T/|W| = {virial}");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(12))]

    /// Recentering and virial equilibrium hold for any seed.
    #[test]
    fn virial_and_recentering_hold_across_seeds(seed in any::<u64>()) {
        let p = Hernquist::new(1.0, 1.0, 1.0);
        let s = p.sample(8000, seed);
        prop_assert!(diagnostics::center_of_mass(&s).length() < 1e-9, "COM not zeroed");
        let virial = 2.0 * diagnostics::kinetic_energy(&s) / p.potential_energy().abs();
        prop_assert!((virial - 1.0).abs() < 0.08, "2T/|W| = {virial}");
    }
}
