//! Statistical validation of NFW IC sampling. The analytic profile functions
//! must be mutually consistent; the numerically Eddington-inverted DF must be
//! self-consistent with the NFW density (the density-recovery integral, now
//! exercising the (b) machinery on a model with NO closed-form DF); and a
//! realization must reproduce the *truncated* mass profile and be properly
//! recentered. Expectations are independent closed forms, not the code's output.

use galaxy_core::{diagnostics, State};
use galaxy_ic::eddington::EddingtonDf;
use galaxy_ic::Nfw;

const PI: f64 = std::f64::consts::PI;

fn sorted_radii(s: &State) -> Vec<f64> {
    let mut r: Vec<f64> = s.pos.iter().map(|p| p.length()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r
}

// ---------- analytic profile self-consistency (no sampling) ----------

#[test]
fn characteristic_quantities_and_virial_normalization() {
    let (g, mvir, rs, c) = (1.0, 1.0, 1.0, 10.0);
    let p = Nfw::new(g, mvir, rs, c);
    // m(c) = ln(1+c) − c/(1+c); M_s = M_vir/m(c); ρ_s = M_s/(4π r_s³).
    let mc = (1.0 + c).ln() - c / (1.0 + c);
    assert!((p.mass_function() - mc).abs() < 1e-12, "m(c) wrong");
    let ms = mvir / mc;
    assert!(
        (p.characteristic_mass() - ms).abs() < 1e-12 * ms,
        "M_s wrong"
    );
    let rho_s = ms / (4.0 * PI * rs.powi(3));
    assert!(
        (p.characteristic_density() - rho_s).abs() < 1e-12 * rho_s,
        "ρ_s wrong"
    );
    // Virial radius and the defining normalization M(<r_vir) = M_vir.
    assert!((p.virial_radius() - c * rs).abs() < 1e-12, "r_vir wrong");
    assert!(
        (p.enclosed_mass(p.virial_radius()) - mvir).abs() < 1e-9 * mvir,
        "M(<r_vir) must equal M_vir"
    );
    assert!(p.enclosed_mass(0.0).abs() < 1e-12, "M(<0) should be 0");
}

#[test]
fn density_and_mass_derivative_agree() {
    let p = Nfw::new(1.0, 2.0, 1.5, 8.0);
    let (rho_s, rs) = (p.characteristic_density(), p.scale_radius);
    for &r in &[0.3_f64, 1.0, 3.0, 9.0] {
        let x = r / rs;
        let rho = rho_s / (x * (1.0 + x).powi(2));
        assert!(
            (p.density(r) - rho).abs() < 1e-12 * rho,
            "ρ({r}) closed form wrong"
        );
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
fn potential_closed_form_and_central_limit() {
    let (g, mvir, rs, c) = (1.3, 2.7, 0.9, 12.0);
    let p = Nfw::new(g, mvir, rs, c);
    let ms = p.characteristic_mass();
    // Φ(0) = −G M_s / r_s (the ln(1+x)/x → 1 limit).
    assert!(
        (p.potential(0.0) + g * ms / rs).abs() < 1e-9 * (g * ms / rs),
        "Φ(0) = −G M_s/r_s wrong"
    );
    for &r in &[0.5_f64, 2.0, 7.0] {
        let x = r / rs;
        let phi = -g * ms * (1.0 + x).ln() / r;
        assert!(
            (p.potential(r) - phi).abs() < 1e-12 * phi.abs(),
            "Φ({r}) closed form wrong"
        );
    }
}

// ---------- Eddington DF ↔ density self-consistency (the (b) machinery on NFW) ----------

#[test]
fn eddington_df_recovers_nfw_density() {
    let p = Nfw::new(1.0, 1.0, 1.0, 10.0);
    let psi_max = -p.potential(0.0);
    // Untruncated model, wide bracket so Ψ(r_max) ≈ 0.
    let df = EddingtonDf::build(&p, psi_max, 1e-3, 1e4);
    // ρ(r) = 4π ∫₀^Ψ f(ℰ) √(2(Ψ−ℰ)) dℰ at radii well inside r_vir = 10.
    for &r in &[0.5_f64, 1.0, 2.0, 5.0] {
        let psi = -p.potential(r);
        let n = 20_000;
        let de = psi / n as f64;
        let mut acc = 0.0;
        for k in 0..n {
            let e = (k as f64 + 0.5) * de;
            acc += df.f(e) * (2.0 * (psi - e)).sqrt();
        }
        let recovered = 4.0 * PI * acc * de;
        let rho = p.density(r);
        assert!(
            (recovered - rho).abs() < 5e-2 * rho,
            "r={r}: DF-recovered ρ={recovered} vs analytic ρ={rho}"
        );
    }
}

// ---------- sampled-realization statistics ----------

#[test]
fn realized_mass_profile_matches_truncated_cdf() {
    let p = Nfw::new(1.0, 1.0, 1.0, 10.0);
    let n = 40_000;
    let s = p.sample(n, 0x5EED);
    let r = sorted_radii(&s);
    let r_vir = p.virial_radius();
    // All particles lie within r_vir; the CDF is normalized to M_vir = M(<r_vir).
    assert!(
        *r.last().unwrap() <= r_vir + 1e-9,
        "a particle escaped the virial radius"
    );
    for &rr in &[1.0_f64, 2.0, 5.0, 8.0] {
        let frac = r.partition_point(|&x| x <= rr) as f64 / n as f64;
        let expected = p.enclosed_mass(rr) / p.virial_mass;
        let tol = 4.0 * (expected * (1.0 - expected) / n as f64).sqrt() + 3e-3;
        assert!(
            (frac - expected).abs() < tol,
            "r={rr}: empirical={frac} expected={expected} tol={tol}"
        );
    }
}

#[test]
fn sample_is_recentered_equal_mass_and_deterministic() {
    let p = Nfw::new(1.0, 1.0, 1.0, 10.0);
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
    assert!((mtot - p.virial_mass).abs() < 1e-12, "masses don't sum to M_vir");
    let m_each = p.virial_mass / n as f64;
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
