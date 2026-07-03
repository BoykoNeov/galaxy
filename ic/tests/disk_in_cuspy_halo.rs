//! Validation of the cuspy-halo exponential disk: [`ExponentialDisk<H>`] with the
//! halo abstracted behind [`SphericalHalo`], so the same cold-disk sampler works in
//! a cuspy [`Hernquist`]/[`TruncatedNfw`] halo instead of only the cored [`Plummer`].
//!
//! The physical point of a cuspy halo is its enclosed-mass profile: M(<r) rises
//! steeply from the very center (ρ ∝ r⁻¹ or steeper), so the disk's rotation curve
//! *rises to a flat plateau* — the realistic CDM-galaxy shape — rather than turning
//! over the way it does in a Plummer core. These gates pin that:
//!
//!   1. Analytic self-consistency (no sampling): v_c(R) must equal
//!      √(G·[M_halo(<R) + M_disk(<R)]/R) against **independently hand-derived** cuspy
//!      enclosed masses — Hernquist M(<r)=M r²/(r+a)² and the NFW closed form. A
//!      dropped disk term, a wrong √, or the wrong halo mass profile would miss it.
//!   2. Statistical validation of a realization: disk particles sit on that analytic
//!      v_c, are tagged/ordered halo-then-disk, carry coherent +Z spin, and the whole
//!      galaxy is delivered in the zero-COM / zero-momentum frame.
//!
//! Scope is the COLD disk (no Toomre warmth): a cold disk on circular orbits is in
//! equilibrium by construction in *any* spherical potential, so it is the honest
//! first increment. The warm path leans on ρ(r), which diverges at a cusp, and is a
//! deliberate follow-up. Expectations are independent closed forms written inline.

use galaxy_core::{DVec3, Progenitor, State};
use galaxy_ic::{ExponentialDisk, Hernquist, Nfw, TruncatedNfw};

const PI: f64 = std::f64::consts::PI;

// ---------- fiducial galaxies ----------

/// A submaximal cold disk (10% of the halo mass) in a unit Hernquist halo. The
/// bare `-> ExponentialDisk` type would default to `<Plummer>`, so the cuspy halo
/// type must be named explicitly.
fn fiducial_hernquist() -> ExponentialDisk<Hernquist> {
    let halo = Hernquist::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}

/// The same disk in an exponentially-truncated NFW halo (M_vir 1, r_s 1, c 10 ⇒
/// r_vir 10). Every disk radius (r_max = 2) sits well inside r_vir, where the
/// truncated halo's enclosed mass is exactly the NFW closed form.
fn fiducial_nfw() -> ExponentialDisk<TruncatedNfw> {
    let halo = TruncatedNfw::new(Nfw::new(1.0, 1.0, 1.0, 10.0), 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}

// ---------- independent closed forms ----------

/// Hernquist enclosed mass, closed form, independent of the library:
/// M(<r) = M · r² / (r + a)².
fn hernquist_enclosed(m: f64, a: f64, r: f64) -> f64 {
    m * r * r / ((r + a) * (r + a))
}

/// NFW enclosed mass, closed form, independent: M(<r) = M_s·[ln(1+x) − x/(1+x)],
/// x = r/r_s, with the characteristic mass M_s = M_vir / [ln(1+c) − c/(1+c)].
/// Valid here because every test radius is < r_vir (no truncation skirt).
fn nfw_enclosed(m_vir: f64, r_s: f64, c: f64, r: f64) -> f64 {
    let m_c = (1.0 + c).ln() - c / (1.0 + c);
    let m_s = m_vir / m_c;
    let x = r / r_s;
    m_s * ((1.0 + x).ln() - x / (1.0 + x))
}

/// Truncated-exponential cylindrical disk enclosed mass, closed form, independent:
/// M_d(<R) = M_d · [1 − (1+R/Rd)e^(−R/Rd)] / [1 − (1+u)e^(−u)], u = r_max/Rd.
fn disk_enclosed(md: f64, rd: f64, r_max: f64, r: f64) -> f64 {
    let f = |x: f64| 1.0 - (1.0 + x / rd) * (-x / rd).exp();
    md * f(r.min(r_max)) / f(r_max)
}

// ---------- 1. analytic self-consistency (the discriminating gate) ----------

#[test]
fn hernquist_circular_velocity_matches_combined_enclosed_mass() {
    let d = fiducial_hernquist();
    let (g, a, mh) = (d.g, d.halo.scale_radius, d.halo.total_mass);
    for &r in &[0.1_f64, 0.3, 0.5, 1.0, 1.5, 2.0] {
        // Independent expectation: v_c² = G·[M_Hernquist(<R) + M_disk,cyl(<R)]/R.
        let m_enc =
            hernquist_enclosed(mh, a, r) + disk_enclosed(d.disk_mass, d.scale_length, d.r_max, r);
        let want = (g * m_enc / r).sqrt();
        let got = d.circular_velocity(r);
        assert!(
            (got - want).abs() < 1e-12 * want,
            "Hernquist v_c({r}) = {got} vs expected {want}"
        );
    }
}

#[test]
fn nfw_circular_velocity_matches_combined_enclosed_mass() {
    let d = fiducial_nfw();
    let g = d.g;
    let base = d.halo.base;
    let (mv, rs, c) = (base.virial_mass, base.scale_radius, base.concentration);
    for &r in &[0.1_f64, 0.3, 0.5, 1.0, 1.5, 2.0] {
        // Independent expectation with the NFW closed form (r < r_vir, so the
        // truncated halo's mass IS the untruncated NFW closed form there).
        let m_enc =
            nfw_enclosed(mv, rs, c, r) + disk_enclosed(d.disk_mass, d.scale_length, d.r_max, r);
        let want = (g * m_enc / r).sqrt();
        let got = d.circular_velocity(r);
        assert!(
            (got - want).abs() < 1e-12 * want,
            "NFW v_c({r}) = {got} vs expected {want}"
        );
    }
}

#[test]
fn cuspy_rotation_curve_rises_to_a_plateau() {
    // The cuspy signature vs a Plummer core: v_c climbs steeply near the center and
    // flattens outward (dv_c/dR → small), rather than peaking and turning over.
    let d = fiducial_hernquist();
    let v = |r: f64| d.circular_velocity(r);
    // Rises across the inner disk...
    assert!(v(0.5) > v(0.2) && v(1.0) > v(0.5), "v_c should rise inward-out");
    // ...and the outer slope is much gentler than the inner slope (plateau-like).
    let inner_slope = (v(0.5) - v(0.2)) / 0.3;
    let outer_slope = (v(2.0) - v(1.5)) / 0.5;
    assert!(
        outer_slope.abs() < 0.5 * inner_slope.abs(),
        "outer slope {outer_slope} not flatter than inner {inner_slope}"
    );
}

#[test]
fn total_mass_is_disk_plus_halo() {
    let dh = fiducial_hernquist();
    assert!((dh.total_mass() - (dh.disk_mass + dh.halo.total_mass)).abs() < 1e-12);
    // The untruncated NFW mass diverges, so the disk uses the truncated total.
    let dn = fiducial_nfw();
    assert!((dn.total_mass() - (dn.disk_mass + dn.halo.total_mass())).abs() < 1e-12);
}

// ---------- 2. statistical validation of a realization ----------

const N_HALO: usize = 3000;
const N_DISK: usize = 3000;
const SEED: u64 = 0xC5B7;

/// (cylindrical radius, azimuthal speed v_φ) of a particle.
fn cylindrical(p: DVec3, v: DVec3) -> (f64, f64) {
    let r = (p.x * p.x + p.y * p.y).sqrt();
    let v_phi = (p.x * v.y - p.y * v.x) / r;
    (r, v_phi)
}

fn disk_indices(s: &State) -> Vec<usize> {
    (0..s.len())
        .filter(|&i| s.progenitor[i] == Progenitor(1))
        .collect()
}

#[test]
fn disk_particles_are_tagged_and_ordered_after_halo() {
    let d = fiducial_hernquist();
    let s = d.sample(N_HALO, N_DISK, SEED);
    assert_eq!(s.len(), N_HALO + N_DISK);
    assert!((0..N_HALO).all(|i| s.progenitor[i] == Progenitor(0)));
    assert!((N_HALO..N_HALO + N_DISK).all(|i| s.progenitor[i] == Progenitor(1)));
    assert_eq!(disk_indices(&s).len(), N_DISK);
}

#[test]
fn disk_rotation_curve_matches_analytic_vc() {
    let d = fiducial_hernquist();
    let s = d.sample(N_HALO, N_DISK, SEED);
    let idx = disk_indices(&s);

    // Bin by cylindrical R; mean v_φ per bin sits on the analytic v_c(⟨R⟩). Cold
    // disk ⇒ no asymmetric drift; compare at the density-weighted mean radius to
    // leave only a small Jensen gap where v_c is steep.
    let edges = [0.2_f64, 0.4, 0.6, 0.8, 1.0, 1.4];
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let (mut sum_vphi, mut sum_r, mut n) = (0.0, 0.0, 0usize);
        for &i in &idx {
            let (r, v_phi) = cylindrical(s.pos[i], s.vel[i]);
            if r >= lo && r < hi {
                sum_vphi += v_phi;
                sum_r += r;
                n += 1;
            }
        }
        assert!(n > 40, "bin [{lo},{hi}) underpopulated: {n}");
        let mean_vphi = sum_vphi / n as f64;
        let vc = d.circular_velocity(sum_r / n as f64);
        assert!(
            (mean_vphi - vc).abs() < 0.05 * vc,
            "bin [{lo},{hi}): ⟨v_φ⟩={mean_vphi} vs v_c={vc}"
        );
    }
}

#[test]
fn disk_has_coherent_spin_and_galaxy_is_zero_com() {
    let d = fiducial_hernquist();
    let s = d.sample(N_HALO, N_DISK, SEED);

    // Coherent +Z spin is the disk's invariant (measured over the disk population;
    // the halo is non-rotating and only contributes finite-N shot noise).
    let mut l_disk = DVec3::ZERO;
    for &i in &disk_indices(&s) {
        l_disk += s.pos[i].cross(s.vel[i]) * s.mass[i];
    }
    assert!(l_disk.z > 0.0, "disk spin must be along +Z: {}", l_disk.z);
    assert!(
        l_disk.z > 20.0 * l_disk.x.abs().max(l_disk.y.abs()),
        "disk spin not coherently axial: {l_disk:?}"
    );

    // Zero-COM / zero-momentum is a property of the whole galaxy.
    let (mut mom, mut com, mut mtot) = (DVec3::ZERO, DVec3::ZERO, 0.0);
    for i in 0..s.len() {
        mom += s.vel[i] * s.mass[i];
        com += s.pos[i] * s.mass[i];
        mtot += s.mass[i];
    }
    assert!(mom.length() < 1e-9 * mtot.max(1.0), "net momentum: {mom:?}");
    assert!((com / mtot).length() < 1e-9, "COM off origin: {:?}", com / mtot);
}
