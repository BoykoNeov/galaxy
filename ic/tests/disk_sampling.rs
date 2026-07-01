//! Validation of the rotating exponential-disk IC (`ExponentialDisk`), the cold
//! kinematic model. Two layers, mirroring the Plummer sampling tests:
//!
//!   1. Analytic self-consistency (no sampling): the surface density, its
//!      normalization, the cylindrical enclosed mass, and the combined-mass
//!      circular-velocity curve must all agree with independently hand-derived
//!      closed forms.
//!   2. Statistical validation of a realization: disk particles must reproduce the
//!      analytic radial CDF, sit on the analytic rotation curve (⟨v_φ⟩(R) ≈ v_c),
//!      and — the invariant that DISTINGUISHES a disk from an isotropic Plummer —
//!      carry a coherent net angular momentum along +Z with zero net momentum/COM.
//!
//! Expectations are independent closed forms written inline here, not the code's
//! own output. A mis-set rotation curve (e.g. forgetting the disk's own mass, or a
//! dropped √) would make ⟨v_φ⟩ miss v_c; a mis-normalized profile would fail the
//! enclosed-fraction gate.

use galaxy_core::{DVec3, Progenitor, State};
use galaxy_ic::{ExponentialDisk, Plummer};

const PI: f64 = std::f64::consts::PI;

// ---------- fiducial galaxy shared by the tests ----------

/// A submaximal cold disk (10% of the halo mass) in a unit Plummer halo.
fn fiducial() -> ExponentialDisk {
    let halo = Plummer::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}

/// Plummer enclosed mass, closed form, written independently of the library.
fn plummer_enclosed(m: f64, a: f64, r: f64) -> f64 {
    m * r * r * r / (r * r + a * a).powf(1.5)
}

/// Truncated-exponential cylindrical enclosed mass, closed form, independent:
/// M_d(<R) = M_d · [1 − (1+R/Rd)e^(−R/Rd)] / [1 − (1+u)e^(−u)], u = r_max/Rd.
fn disk_enclosed(md: f64, rd: f64, r_max: f64, r: f64) -> f64 {
    let f = |x: f64| 1.0 - (1.0 + x / rd) * (-x / rd).exp();
    let norm = f(r_max);
    md * f(r.min(r_max)) / norm
}

// ---------- 1. analytic self-consistency ----------

#[test]
fn central_surface_density_normalizes_to_disk_mass() {
    let d = fiducial();
    let (md, rd, u) = (d.disk_mass, d.scale_length, d.r_max / d.scale_length);
    // Σ₀ = M_d / (2π Rd² · [1 − (1+u)e^(−u)]).
    let sigma0 = md / (2.0 * PI * rd * rd * (1.0 - (1.0 + u) * (-u).exp()));
    assert!(
        (d.central_surface_density() - sigma0).abs() < 1e-12 * sigma0,
        "Σ₀ = {} vs expected {sigma0}",
        d.central_surface_density()
    );
    // Σ(0) is the central value; Σ falls as e^(−R/Rd) and truncates past r_max.
    assert!((d.surface_density(0.0) - sigma0).abs() < 1e-12 * sigma0);
    for &r in &[0.3_f64, 1.0, 1.9] {
        let want = sigma0 * (-r / rd).exp();
        assert!(
            (d.surface_density(r) - want).abs() < 1e-12 * want,
            "Σ({r}) = {} vs {want}",
            d.surface_density(r)
        );
    }
    assert_eq!(
        d.surface_density(d.r_max + 1e-9),
        0.0,
        "Σ must vanish past r_max"
    );
}

#[test]
fn disk_enclosed_mass_limits_and_density_derivative_agree() {
    let d = fiducial();
    assert!(
        d.disk_enclosed_mass(0.0).abs() < 1e-12,
        "M_d(<0) should be 0"
    );
    assert!(
        (d.disk_enclosed_mass(d.r_max) - d.disk_mass).abs() < 1e-12 * d.disk_mass,
        "M_d(<r_max) should equal the total disk mass"
    );
    // The annulus mass 2π R Σ(R) must equal dM_d(<R)/dR — density and enclosed
    // mass describe the same disk (mirrors the Plummer dM/dr = 4πr²ρ test).
    for &r in &[0.2_f64, 0.5, 1.0, 1.5] {
        let h = 1e-6;
        let dmdr = (d.disk_enclosed_mass(r + h) - d.disk_enclosed_mass(r - h)) / (2.0 * h);
        let annulus = 2.0 * PI * r * d.surface_density(r);
        assert!(
            (dmdr - annulus).abs() < 1e-4 * annulus,
            "r={r}: dM/dR={dmdr} vs 2πRΣ={annulus}"
        );
    }
}

#[test]
fn circular_velocity_matches_combined_enclosed_mass() {
    let d = fiducial();
    let g = d.g;
    for &r in &[0.2_f64, 0.5, 1.0, 1.5, 2.0] {
        // Independent expectation: v_c² = G·[M_halo,sph(<R) + M_disk,cyl(<R)]/R.
        let m_enc = plummer_enclosed(d.halo.total_mass, d.halo.scale_radius, r)
            + disk_enclosed(d.disk_mass, d.scale_length, d.r_max, r);
        let want = (g * m_enc / r).sqrt();
        let got = d.circular_velocity(r);
        assert!(
            (got - want).abs() < 1e-12 * want,
            "v_c({r}) = {got} vs expected {want}"
        );
    }
}

#[test]
fn total_mass_is_disk_plus_halo() {
    let d = fiducial();
    assert!((d.total_mass() - (d.disk_mass + d.halo.total_mass)).abs() < 1e-12);
}

// ---------- 2. statistical validation of a realization ----------

const N_HALO: usize = 4000;
const N_DISK: usize = 4000;
const SEED: u64 = 0xD15C0;

/// (cylindrical radius, azimuthal speed v_φ, radial speed v_R, z) of a particle.
fn cylindrical(p: DVec3, v: DVec3) -> (f64, f64, f64, f64) {
    let r = (p.x * p.x + p.y * p.y).sqrt();
    let v_phi = (p.x * v.y - p.y * v.x) / r;
    let v_r = (p.x * v.x + p.y * v.y) / r;
    (r, v_phi, v_r, p.z)
}

/// Indices of the disk particles (tagged `Progenitor(1)`).
fn disk_indices(s: &State) -> Vec<usize> {
    (0..s.len())
        .filter(|&i| s.progenitor[i] == Progenitor(1))
        .collect()
}

#[test]
fn disk_particles_are_tagged_and_ordered_after_halo() {
    let d = fiducial();
    let s = d.sample(N_HALO, N_DISK, SEED);
    assert_eq!(s.len(), N_HALO + N_DISK);
    // Halo first (Progenitor 0), then disk (Progenitor 1).
    assert!((0..N_HALO).all(|i| s.progenitor[i] == Progenitor(0)));
    assert!((N_HALO..N_HALO + N_DISK).all(|i| s.progenitor[i] == Progenitor(1)));
    assert_eq!(disk_indices(&s).len(), N_DISK);
}

#[test]
fn disk_radial_profile_matches_analytic_cdf() {
    let d = fiducial();
    let s = d.sample(N_HALO, N_DISK, SEED);
    let idx = disk_indices(&s);
    let radii: Vec<f64> = idx
        .iter()
        .map(|&i| cylindrical(s.pos[i], s.vel[i]).0)
        .collect();

    // Fraction of disk particles within cylindrical R must match M_d(<R)/M_d.
    for &r in &[0.25_f64, 0.5, 1.0, 1.5] {
        let frac_measured = radii.iter().filter(|&&x| x <= r).count() as f64 / N_DISK as f64;
        let frac_analytic = disk_enclosed(d.disk_mass, d.scale_length, d.r_max, r) / d.disk_mass;
        assert!(
            (frac_measured - frac_analytic).abs() < 0.03,
            "enclosed fraction within R={r}: measured {frac_measured} vs analytic {frac_analytic}"
        );
    }
    // No particle beyond the truncation radius.
    assert!(
        radii.iter().all(|&x| x <= d.r_max + 1e-9),
        "disk truncated at r_max"
    );
}

#[test]
fn disk_rotation_curve_matches_analytic_vc() {
    let d = fiducial();
    let s = d.sample(N_HALO, N_DISK, SEED);
    let idx = disk_indices(&s);

    // Bin by cylindrical R; mean v_φ per bin must sit on the analytic v_c(R).
    // The disk is cold (no dispersion) so asymmetric drift is negligible.
    // Compare ⟨v_φ⟩ to v_c at the bin's MEAN radius (not the geometric midpoint):
    // where v_c has a steep gradient the density-weighted mean radius is the fair
    // comparison point, leaving only a small Jensen gap.
    let edges = [0.2_f64, 0.4, 0.6, 0.8, 1.0, 1.4];
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let mut sum_vphi = 0.0;
        let mut sum_r = 0.0;
        let mut n = 0usize;
        for &i in &idx {
            let (r, v_phi, _, _) = cylindrical(s.pos[i], s.vel[i]);
            if r >= lo && r < hi {
                sum_vphi += v_phi;
                sum_r += r;
                n += 1;
            }
        }
        assert!(n > 50, "bin [{lo},{hi}) underpopulated: {n}");
        let mean_vphi = sum_vphi / n as f64;
        let mean_r = sum_r / n as f64;
        let vc = d.circular_velocity(mean_r);
        assert!(
            (mean_vphi - vc).abs() < 0.05 * vc,
            "bin [{lo},{hi}): ⟨v_φ⟩={mean_vphi} vs v_c(⟨R⟩={mean_r})={vc}"
        );
    }
}

#[test]
fn disk_has_coherent_spin_and_zero_net_momentum() {
    let d = fiducial();
    let s = d.sample(N_HALO, N_DISK, SEED);

    // Axial-spin coherence is the DISK's invariant: the halo is intentionally
    // non-rotating, and its finite-N angular-momentum shot noise (~M⟨rv⟩/√N) is
    // not the disk's spin — so the L_z ≫ |L_x|,|L_y| ratio is measured over the
    // disk population, where every particle contributes m·R·v_c to L_z.
    let mut l_disk = DVec3::ZERO;
    for &i in &disk_indices(&s) {
        l_disk += s.pos[i].cross(s.vel[i]) * s.mass[i];
    }
    assert!(
        l_disk.z > 0.0,
        "disk spin must be along +Z: L_z = {}",
        l_disk.z
    );
    assert!(
        l_disk.z > 20.0 * l_disk.x.abs().max(l_disk.y.abs()),
        "disk spin not coherently axial: L = {l_disk:?}"
    );

    // Zero-COM / zero-momentum frame is a property of the WHOLE galaxy.
    let mut mom = DVec3::ZERO;
    let mut com = DVec3::ZERO;
    let mut mtot = 0.0;
    for i in 0..s.len() {
        mom += s.vel[i] * s.mass[i];
        com += s.pos[i] * s.mass[i];
        mtot += s.mass[i];
    }
    assert!(
        mom.length() < 1e-9 * mtot.max(1.0),
        "net momentum not zero: {mom:?}"
    );
    assert!(
        (com / mtot).length() < 1e-9,
        "COM not at origin: {:?}",
        com / mtot
    );
}

#[test]
fn disk_is_flattened_and_not_expanding() {
    let d = fiducial();
    let s = d.sample(N_HALO, N_DISK, SEED);
    let idx = disk_indices(&s);

    // RMS |z| ≪ RMS cylindrical R: the disk is thin (hz ≪ Rd by construction).
    let mut sz2 = 0.0;
    let mut sr2 = 0.0;
    let mut svr = 0.0;
    let mut svz = 0.0;
    for &i in &idx {
        let (r, _, v_r, z) = cylindrical(s.pos[i], s.vel[i]);
        sz2 += z * z;
        sr2 += r * r;
        svr += v_r;
        svz += s.vel[i].z;
    }
    let n = idx.len() as f64;
    let (rms_z, rms_r) = ((sz2 / n).sqrt(), (sr2 / n).sqrt());
    assert!(
        rms_z < 0.2 * rms_r,
        "disk not thin: RMS z {rms_z} vs RMS R {rms_r}"
    );
    // Cold disk: no net radial or vertical streaming.
    assert!((svr / n).abs() < 0.02, "net radial streaming: {}", svr / n);
    assert!(
        (svz / n).abs() < 0.02,
        "net vertical streaming: {}",
        svz / n
    );
}
