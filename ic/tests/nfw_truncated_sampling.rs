//! Statistical validation of the exponentially-truncated NFW IC (Springel &
//! White 1999), the self-consistent (Path A) sibling of the hard-truncated M5c
//! `Nfw`. The distinguishing properties over plain NFW:
//!   (i)   the density AND its log-slope are continuous at r_vir (the exponent ε
//!         is fixed by that continuity — the load-bearing formula);
//!   (ii)  the exponential skirt makes the total mass finite (> M_vir);
//!   (iii) velocities come from the DF of the *truncated* (ρ, Ψ) pair, so the
//!         model is a self-consistent equilibrium — the Eddington machinery is
//!         exercised on the truncated potential (which has no closed form).
//! Expectations are independent closed forms / hand integrals, not code output.

use galaxy_core::{diagnostics, State};
use galaxy_ic::eddington::{EddingtonDf, SphericalModel};
use galaxy_ic::{Nfw, TruncatedNfw};

const PI: f64 = std::f64::consts::PI;

/// Canonical test halo: M_vir = 1, r_s = 1, c = 10 (so r_vir = 10), with a gentle
/// decay length r_d = 0.3 r_vir (positivity-safe for the Eddington inversion).
fn model() -> TruncatedNfw {
    TruncatedNfw::new(Nfw::new(1.0, 1.0, 1.0, 10.0), 3.0)
}

fn sorted_radii(s: &State) -> Vec<f64> {
    let mut r: Vec<f64> = s.pos.iter().map(|p| p.length()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r
}

// ---------- truncation formula + profile self-consistency (no sampling) ----------

#[test]
fn epsilon_is_slope_continuity_value() {
    let t = model();
    let (rs, rvir, rd) = (t.base.scale_radius, t.base.virial_radius(), t.decay_length);
    // ε = r_vir/r_d − (r_s + 3 r_vir)/(r_s + r_vir), the value that matches the
    // truncated log-slope (ε − r_vir/r_d) to the NFW log-slope at r_vir.
    let eps = rvir / rd - (rs + 3.0 * rvir) / (rs + rvir);
    assert!(
        (t.epsilon() - eps).abs() < 1e-12,
        "ε wrong: got {} want {eps}",
        t.epsilon()
    );
}

#[test]
fn density_and_log_slope_continuous_at_truncation() {
    let t = model();
    let rvir = t.base.virial_radius();
    // ρ is continuous: the inner branch at r_vir equals the NFW value, and the
    // outer branch approaches the same value from above.
    let rho_v = t.base.density(rvir);
    assert!(
        (t.density(rvir) - rho_v).abs() < 1e-12 * rho_v,
        "ρ(r_vir) must equal the NFW value"
    );
    let just_out = t.density(rvir * (1.0 + 1e-6));
    assert!(
        (just_out - rho_v).abs() < 1e-4 * rho_v,
        "ρ discontinuous across r_vir: {just_out} vs {rho_v}"
    );
    // Log-slope continuity: one-sided numerical d ln ρ / d ln r straddling r_vir
    // must agree and both equal the NFW value −(1+3c)/(1+c).
    let c = t.base.concentration;
    let want = -(1.0 + 3.0 * c) / (1.0 + c);
    let h = 1e-4;
    let ln = |r: f64| t.density(r).ln();
    let left = (ln(rvir) - ln(rvir * (1.0 - h))) / h;
    let right = (ln(rvir * (1.0 + h)) - ln(rvir)) / h;
    assert!(
        (left - right).abs() < 2e-2,
        "log-slope discontinuous: left {left} right {right}"
    );
    assert!(
        (left - want).abs() < 2e-2 && (right - want).abs() < 2e-2,
        "log-slope at r_vir wrong: left {left} right {right} want {want}"
    );
}

#[test]
fn skirt_density_matches_closed_form() {
    let t = model();
    let (rvir, rd, eps) = (t.base.virial_radius(), t.decay_length, t.epsilon());
    let a = t.base.density(rvir); // ρ_NFW(r_vir), the skirt amplitude
    for &r in &[12.0_f64, 15.0, 20.0, 30.0] {
        let want = a * (r / rvir).powf(eps) * (-(r - rvir) / rd).exp();
        assert!(
            (t.density(r) - want).abs() < 1e-12 * want.max(1e-300),
            "skirt ρ({r}) = {} vs closed form {want}",
            t.density(r)
        );
    }
}

#[test]
fn density_and_mass_derivative_agree() {
    let t = model();
    // dM/dr = 4π r² ρ across BOTH regions (inside r_vir and out in the skirt).
    for &r in &[0.3_f64, 1.0, 3.0, 9.0, 12.0, 16.0, 22.0] {
        let h = 1e-5 * r;
        let dmdr = (t.enclosed_mass(r + h) - t.enclosed_mass(r - h)) / (2.0 * h);
        let shell = 4.0 * PI * r * r * t.density(r);
        assert!(
            (dmdr - shell).abs() < 5e-3 * shell,
            "r={r}: dM/dr={dmdr} vs 4πr²ρ={shell}"
        );
    }
}

#[test]
fn total_mass_finite_and_captures_skirt() {
    let t = model();
    let mvir = t.base.virial_mass;
    // M(<r_vir) is still exactly M_vir (skirt lives beyond r_vir).
    assert!(
        (t.enclosed_mass(t.base.virial_radius()) - mvir).abs() < 1e-9 * mvir,
        "M(<r_vir) must equal M_vir"
    );
    // The exponential cutoff makes the total mass finite and strictly larger than
    // M_vir (the skirt adds mass), but not wildly so.
    let mtot = t.total_mass();
    assert!(mtot.is_finite(), "total mass not finite");
    assert!(
        mtot > mvir && mtot < 2.0 * mvir,
        "total mass {mtot} should exceed M_vir={mvir} by a modest skirt"
    );
    // total_mass == M(<R) for R far beyond the skirt (the mass integral converged).
    let r_far = t.base.virial_radius() + 60.0 * t.decay_length;
    assert!(
        (t.enclosed_mass(r_far) - mtot).abs() < 1e-3 * mtot,
        "M(<r_far)={} disagrees with total_mass={mtot}",
        t.enclosed_mass(r_far)
    );
}

// ---------- self-consistent Eddington DF on the TRUNCATED (ρ, Ψ) ----------

#[test]
fn eddington_df_recovers_truncated_density() {
    let t = model();
    // Build the DF from the truncated model's OWN (ρ, Ψ) — the Path A step. Its
    // potential is numerical (no closed form), so this exercises the M5b machinery
    // on a self-consistently truncated potential.
    let psi_max = t.relative_potential(0.0);
    assert!(psi_max.is_finite() && psi_max > 0.0, "Ψ(0) must be finite > 0");
    let df = EddingtonDf::build(&t, psi_max, 1e-3, 1e4);
    // ρ(r) = 4π ∫₀^Ψ f(ℰ) √(2(Ψ−ℰ)) dℰ at radii well inside r_vir, using the
    // TRUNCATED Ψ(r). Self-consistency of the (ρ, Ψ, f) triple.
    for &r in &[0.5_f64, 1.0, 2.0, 5.0] {
        let psi = t.relative_potential(r);
        let n = 20_000;
        let de = psi / n as f64;
        let mut acc = 0.0;
        for k in 0..n {
            let e = (k as f64 + 0.5) * de;
            acc += df.f(e) * (2.0 * (psi - e)).sqrt();
        }
        let recovered = 4.0 * PI * acc * de;
        let rho = t.density(r);
        assert!(
            (recovered - rho).abs() < 6e-2 * rho,
            "r={r}: DF-recovered ρ={recovered} vs truncated ρ={rho}"
        );
    }
}

#[test]
fn truncated_potential_differs_from_untruncated_nfw() {
    // Path A fingerprint: the potential is built from the TRUNCATED profile, not
    // reused from the untruncated NFW closed form. Ψ(0) = 4πG ∫₀^∞ ρ s ds is a
    // convergent integral even for NFW; beyond r_vir the exp skirt falls strictly
    // below the r⁻³ NFW tail (equal slope at r_vir, then steepening as −1/r_d), so
    // it carries LESS ∫ρ s ds than the tail it replaces. The truncated central
    // potential is therefore measurably SHALLOWER than the untruncated NFW value
    // (Path B, which reuses the closed form, would show zero difference).
    let t = model();
    let psi_nfw_0 = -t.base.potential(0.0); // G M_s / r_s (closed form)
    let psi_t_0 = t.relative_potential(0.0);
    assert!(
        psi_t_0 < psi_nfw_0 && (psi_nfw_0 - psi_t_0) > 0.02 * psi_nfw_0,
        "Ψ_t(0)={psi_t_0} should be measurably shallower than Ψ_NFW(0)={psi_nfw_0}"
    );
}

#[test]
fn df_is_positive_without_hitting_the_clamp() {
    // The advisor's gate: too-sharp a decay drives Eddington's f(ℰ) negative,
    // which the ≥0 clamp silently hides. A gentle r_d must yield a STRICTLY
    // positive DF across the interior energies we actually sample.
    let t = model();
    let psi_max = t.relative_potential(0.0);
    let df = EddingtonDf::build(&t, psi_max, 1e-3, 1e4);
    for k in 1..20 {
        let e = psi_max * k as f64 / 20.0;
        assert!(
            df.f(e) > 0.0,
            "DF hit the clamp (f≤0) at ℰ={e} (ℰ/Ψ_max={})",
            k as f64 / 20.0
        );
    }
}

// ---------- sampled-realization statistics ----------

#[test]
fn realized_mass_profile_matches_truncated_cdf() {
    let t = model();
    let n = 40_000;
    let s = t.sample(n, 0x5EED);
    let r = sorted_radii(&s);
    let mtot = t.total_mass();
    // Positions sample the FULL truncated profile, so the extent runs into the
    // skirt (well beyond r_vir) but is bounded by the exponential cutoff.
    let r_far = t.base.virial_radius() + 60.0 * t.decay_length;
    assert!(
        *r.last().unwrap() <= r_far,
        "max radius {} beyond the skirt cutoff {r_far}",
        r.last().unwrap()
    );
    // Empirical CDF vs M(<r)/M_tot at radii inside and out in the skirt.
    for &rr in &[2.0_f64, 5.0, 10.0, 15.0, 22.0] {
        let frac = r.partition_point(|&x| x <= rr) as f64 / n as f64;
        let expected = t.enclosed_mass(rr) / mtot;
        let tol = 4.0 * (expected * (1.0 - expected) / n as f64).sqrt() + 3e-3;
        assert!(
            (frac - expected).abs() < tol,
            "r={rr}: empirical={frac} expected={expected} tol={tol}"
        );
    }
}

/// Independent oracle for the velocities: the isotropic Jeans equation fixes the
/// radial dispersion from the (truncated) density and potential alone:
///   σ_r²(r) = (1/ρ(r)) ∫_r^∞ ρ(r') G M(<r')/r'² dr',   ⟨v²⟩ = 3 σ_r².
/// Because Path A samples velocities from the DF of the SAME truncated potential,
/// the realized ⟨v²⟩ must match the truncated-Jeans prediction tightly (no
/// untruncated-DF re-virialization slack, unlike M5c).
#[test]
fn realized_velocity_dispersion_matches_jeans() {
    let t = model();
    let n = 120_000;
    let s = t.sample(n, 0x1EE7);

    let sigma_r2 = |r: f64| -> f64 {
        // Integrate to well past the skirt (integrand → 0 there).
        let r_hi = t.base.virial_radius() + 80.0 * t.decay_length;
        let nq = 20_000;
        let ratio = (r_hi / r).powf(1.0 / nq as f64);
        let mut acc = 0.0;
        let mut r0 = r;
        for _ in 0..nq {
            let r1 = r0 * ratio;
            let rm = (r0 * r1).sqrt();
            acc += t.density(rm) * t.base.g * t.enclosed_mass(rm) / (rm * rm) * (r1 - r0);
            r0 = r1;
        }
        acc / t.density(r)
    };

    for &(rlo, rhi) in &[(0.8_f64, 1.2), (1.7, 2.3), (3.5, 4.5)] {
        let mut sum_v2 = 0.0;
        let mut cnt = 0usize;
        for i in 0..s.len() {
            let r = s.pos[i].length();
            if r >= rlo && r < rhi {
                sum_v2 += s.vel[i].length_squared();
                cnt += 1;
            }
        }
        let realized = sum_v2 / cnt as f64;

        let (mut num, mut den) = (0.0, 0.0);
        let sub = 200;
        for k in 0..sub {
            let r = rlo + (rhi - rlo) * (k as f64 + 0.5) / sub as f64;
            let w = t.density(r) * r * r;
            num += w * 3.0 * sigma_r2(r);
            den += w;
        }
        let expected = num / den;

        assert!(
            (realized - expected).abs() < 0.06 * expected,
            "shell [{rlo},{rhi}): realized ⟨v²⟩={realized} vs 3σ_r²(Jeans)={expected} (n={cnt})"
        );
    }
}

#[test]
fn sample_is_recentered_equal_mass_and_deterministic() {
    let t = model();
    let n = 5000;
    let s = t.sample(n, 42);
    assert!(
        diagnostics::center_of_mass(&s).length() < 1e-9,
        "COM not zeroed"
    );
    assert!(
        diagnostics::total_momentum(&s).length() < 1e-9,
        "net momentum not zeroed"
    );
    // Equal-mass particles summing to the FULL (truncated) mass, not just M_vir.
    let mtot: f64 = s.mass.iter().sum();
    assert!(
        (mtot - t.total_mass()).abs() < 1e-9 * t.total_mass(),
        "masses must sum to total_mass()"
    );
    let m_each = t.total_mass() / n as f64;
    assert!(
        s.mass.iter().all(|&m| (m - m_each).abs() < 1e-12 * m_each),
        "masses not equal"
    );
    assert_eq!(s.len(), n);
    let s2 = t.sample(n, 42);
    assert_eq!(s.pos, s2.pos, "not deterministic (pos)");
    assert_eq!(s.vel, s2.vel, "not deterministic (vel)");
    let s3 = t.sample(n, 43);
    assert!(s3.pos != s.pos, "seed had no effect");
}
