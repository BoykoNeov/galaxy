//! Validation of the WARM exponential-disk IC — the milestone that adds in-plane
//! and vertical velocity dispersion to the cold kinematic disk so it survives the
//! several orbits of a collision without fragmenting (the cold Q→0 disk clumps;
//! see `DESIGN.md` M3.5). Warmth is opt-in via `with_toomre_q(q)`; the default
//! `new(...)` disk stays fully cold and every cold gate in `disk_sampling.rs` /
//! `disk_stability.rs` is untouched.
//!
//! Two layers, mirroring the cold sampling tests:
//!   1. Analytic self-consistency (no sampling): the epicyclic frequency, the
//!      Toomre-Q radial dispersion, the epicyclic σ_φ/σ_R ratio, the isothermal
//!      σ_z, and the asymmetric-drift lag must each agree with an independently
//!      hand-derived closed form (NOT the code's own output).
//!   2. Statistical validation of a realization: per-radius sample dispersions must
//!      recover the input Q, ⟨v_φ⟩ must lag v_c (asymmetric drift), and the warm
//!      disk must still be delivered zero-COM / zero-momentum.
//!
//! Independence of the checks:
//!   - κ is validated by the *definitional* form κ² = R dΩ²/dR + 4Ω² (a central
//!     difference of the code's circular_velocity) — a different expression and a
//!     different code path (enclosed-mass derivative) than the production
//!     κ² = Ω² + G M'(R)/R² (halo density + disk surface density). Agreement pins
//!     both.
//!   - Q is *recovered* from σ_R (Q = σ_R κ / (3.36 G Σ)) — a dropped √ or a
//!     forgotten factor would miss the input Q.
//!   - the asymmetric-drift sign is checked directly: v̄_φ < v_c in the disk body,
//!     the lag scales as O(σ_R²/v_c), and v̄_φ is clamped ≥ 0 near the center.

use galaxy_core::{DVec3, Progenitor, State};
use galaxy_ic::{ExponentialDisk, Plummer};

const G_NEWTON: f64 = 1.0;
const PI: f64 = std::f64::consts::PI;

/// Toomre stellar factor: Q = σ_R κ / (3.36 G Σ). The 3.36 (not π) is the stellar
/// value; π is the gas value.
const TOOMRE_FACTOR: f64 = 3.36;

/// The fiducial galaxy shared with the cold tests: a submaximal disk (10% of the
/// halo mass) in a unit Plummer halo. `Q` picks the warmth.
fn fiducial_cold() -> ExponentialDisk {
    let halo = Plummer::new(G_NEWTON, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}

fn fiducial_warm(q: f64) -> ExponentialDisk {
    fiducial_cold().with_toomre_q(q)
}

// ---------- independent closed forms (hand-derived, not the library's) ----------

/// Plummer spherical enclosed mass M(<r) = M r³/(r²+a²)^{3/2}.
fn plummer_enclosed(m: f64, a: f64, r: f64) -> f64 {
    m * r * r * r / (r * r + a * a).powf(1.5)
}

/// Truncated-exponential cylindrical enclosed mass.
fn disk_enclosed(md: f64, rd: f64, r_max: f64, r: f64) -> f64 {
    let f = |x: f64| 1.0 - (1.0 + x / rd) * (-x / rd).exp();
    md * f(r.min(r_max)) / f(r_max)
}

/// Independent circular velocity from combined enclosed mass.
fn v_c_indep(d: &ExponentialDisk, r: f64) -> f64 {
    let m = plummer_enclosed(d.halo.total_mass, d.halo.scale_radius, r)
        + disk_enclosed(d.disk_mass, d.scale_length, d.r_max, r);
    (d.g * m / r).sqrt()
}

/// Independent surface density Σ(R) = Σ₀ e^{−R/Rd}, Σ₀ = M_d/(2π Rd²·norm).
fn sigma_indep(d: &ExponentialDisk, r: f64) -> f64 {
    let rd = d.scale_length;
    let u = d.r_max / rd;
    let sigma0 = d.disk_mass / (2.0 * PI * rd * rd * (1.0 - (1.0 + u) * (-u).exp()));
    if r > d.r_max {
        0.0
    } else {
        sigma0 * (-r / rd).exp()
    }
}

/// Independent epicyclic frequency via the DEFINITIONAL form κ² = R dΩ²/dR + 4Ω²,
/// with Ω² central-differenced from the independent v_c (Ω = v_c/R).
fn kappa_indep(d: &ExponentialDisk, r: f64) -> f64 {
    let omega2 = |rr: f64| {
        let vc = v_c_indep(d, rr);
        (vc / rr) * (vc / rr)
    };
    let h = 1e-5 * r;
    let domega2_dr = (omega2(r + h) - omega2(r - h)) / (2.0 * h);
    (r * domega2_dr + 4.0 * omega2(r)).sqrt()
}

// ============================================================================
// 1. analytic self-consistency
// ============================================================================

#[test]
fn epicyclic_frequency_matches_definitional_form() {
    let d = fiducial_warm(1.5);
    for &r in &[0.2_f64, 0.5, 1.0, 1.5] {
        let want = kappa_indep(&d, r);
        let got = d.epicyclic_frequency(r);
        assert!(
            (got - want).abs() < 1e-3 * want,
            "κ({r}) = {got} vs definitional {want}"
        );
        // Sanity: κ lies between Ω (Keplerian) and 2Ω (solid-body) for a rising
        // rotation curve of this kind.
        let omega = d.orbital_frequency(r);
        assert!(
            got > 0.99 * omega && got < 2.01 * omega,
            "κ({r})={got} outside [Ω,2Ω]=[{omega},{}]",
            2.0 * omega
        );
    }
}

#[test]
fn radial_dispersion_recovers_input_toomre_q() {
    let q = 1.4;
    let d = fiducial_warm(q);
    for &r in &[0.3_f64, 0.6, 1.0, 1.4] {
        let sigma_r = d.radial_dispersion(r);
        // Invert the Toomre definition with INDEPENDENT Σ and κ.
        let sigma = sigma_indep(&d, r);
        let kappa = kappa_indep(&d, r);
        let q_recovered = sigma_r * kappa / (TOOMRE_FACTOR * d.g * sigma);
        assert!(
            (q_recovered - q).abs() < 1e-3 * q,
            "recovered Q({r}) = {q_recovered} vs input {q}"
        );
    }
}

#[test]
fn azimuthal_dispersion_follows_epicyclic_ratio() {
    let d = fiducial_warm(1.5);
    for &r in &[0.3_f64, 0.6, 1.0, 1.4] {
        let sigma_r = d.radial_dispersion(r);
        let sigma_phi = d.azimuthal_dispersion(r);
        // Independent epicyclic ratio (σ_φ/σ_R)² = κ²/(4Ω²).
        let kappa = kappa_indep(&d, r);
        let omega = v_c_indep(&d, r) / r;
        let want_ratio2 = kappa * kappa / (4.0 * omega * omega);
        let got_ratio2 = (sigma_phi / sigma_r).powi(2);
        assert!(
            (got_ratio2 - want_ratio2).abs() < 2e-3 * want_ratio2,
            "σ_φ²/σ_R²({r}) = {got_ratio2} vs κ²/4Ω² = {want_ratio2}"
        );
    }
}

#[test]
fn vertical_dispersion_matches_isothermal_sheet() {
    let d = fiducial_warm(1.5);
    for &r in &[0.2_f64, 0.5, 1.0, 1.5] {
        // Self-gravitating sech²(z/hz) sheet: σ_z² = π G Σ(R) hz.
        let want = (PI * d.g * sigma_indep(&d, r) * d.scale_height).sqrt();
        let got = d.vertical_dispersion(r);
        assert!(
            (got - want).abs() < 1e-6 * want.max(1e-12),
            "σ_z({r}) = {got} vs √(πGΣhz) = {want}"
        );
    }
}

#[test]
fn asymmetric_drift_lags_v_c_and_scales_with_dispersion() {
    let d = fiducial_warm(1.5);
    for &r in &[0.4_f64, 0.7, 1.0, 1.3] {
        let vc = v_c_indep(&d, r);
        let vbar = d.mean_azimuthal_velocity(r);
        // Correct sign: the warm disk lags the circular speed (pressure support).
        assert!(vbar < vc, "v̄_φ({r})={vbar} should lag v_c={vc}");
        assert!(vbar > 0.0, "v̄_φ({r})={vbar} must stay positive");
        // Magnitude: the FRACTIONAL lag (v_c−v̄_φ)/v_c ≈ (v_c²−v̄_φ²)/(2v_c²) is of
        // order (σ_R/v_c)² — the asymmetric-drift equation v_c²−v̄_φ² = σ_R²·B with
        // the O(1–10) density-gradient bracket B. Bound it by 10·(σ_R/v_c)² (ample
        // margin on B), which pins the lag to the dispersion scale without asserting
        // B's exact value.
        let frac_lag = (vc - vbar) / vc;
        let sigma_ratio2 = (d.radial_dispersion(r) / vc).powi(2);
        assert!(frac_lag > 0.0, "lag must be positive at r={r}");
        assert!(
            frac_lag < 10.0 * sigma_ratio2,
            "lag {frac_lag} at r={r} not O(σ_R²/v_c²)={sigma_ratio2}"
        );
    }
}

#[test]
fn mean_azimuthal_velocity_is_clamped_nonnegative_near_center() {
    // Near R→0 v_c→0 but the drift correction stays finite, so v̄_φ² can go
    // negative without a guard → NaN. It must clamp to a real, non-negative value.
    let d = fiducial_warm(2.0);
    for &r in &[1e-4_f64, 1e-3, 1e-2] {
        let vbar = d.mean_azimuthal_velocity(r);
        assert!(vbar.is_finite(), "v̄_φ({r}) = {vbar} is not finite");
        assert!(vbar >= 0.0, "v̄_φ({r}) = {vbar} went negative");
    }
}

#[test]
fn cold_disk_has_no_dispersion_and_no_drift() {
    // The default (Q = None) disk is fully cold: zero dispersion, v̄_φ = v_c.
    let d = fiducial_cold();
    assert_eq!(d.toomre_q(), None);
    for &r in &[0.3_f64, 0.7, 1.2] {
        assert_eq!(d.radial_dispersion(r), 0.0);
        assert_eq!(d.azimuthal_dispersion(r), 0.0);
        assert_eq!(d.vertical_dispersion(r), 0.0);
        let vc = v_c_indep(&d, r);
        assert!(
            (d.mean_azimuthal_velocity(r) - vc).abs() < 1e-12 * vc,
            "cold v̄_φ must equal v_c"
        );
    }
}

#[test]
fn with_toomre_q_records_the_warmth() {
    let d = fiducial_warm(1.7);
    assert_eq!(d.toomre_q(), Some(1.7));
}

// ============================================================================
// 2. statistical validation of a realization
// ============================================================================

const N_HALO: usize = 6000;
const N_DISK: usize = 6000;

/// (cylindrical radius, azimuthal speed, radial speed, z-speed) of a particle.
fn cylindrical(p: DVec3, v: DVec3) -> (f64, f64, f64, f64) {
    let r = (p.x * p.x + p.y * p.y).sqrt();
    let v_phi = (p.x * v.y - p.y * v.x) / r;
    let v_r = (p.x * v.x + p.y * v.y) / r;
    (r, v_phi, v_r, v.z)
}

fn disk_indices(s: &State) -> Vec<usize> {
    (0..s.len())
        .filter(|&i| s.progenitor[i] == Progenitor(1))
        .collect()
}

#[test]
fn warm_realization_recovers_input_q_per_radius() {
    let q = 1.4;
    let d = fiducial_warm(q);
    let s = d.sample(N_HALO, N_DISK, 0xC0FFEE);
    let idx = disk_indices(&s);

    // Bin by cylindrical R; the sample radial-velocity dispersion in each bin,
    // fed back through the Toomre definition with independent Σ and κ, must recover
    // the input Q. This ties the REALIZATION (sampler + kinematics) to the input.
    let edges = [0.3_f64, 0.5, 0.7, 0.9, 1.1];
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let (mut s_vr, mut s_vr2, mut s_r, mut n) = (0.0, 0.0, 0.0, 0usize);
        for &i in &idx {
            let (r, _, v_r, _) = cylindrical(s.pos[i], s.vel[i]);
            if r >= lo && r < hi {
                s_vr += v_r;
                s_vr2 += v_r * v_r;
                s_r += r;
                n += 1;
            }
        }
        assert!(n > 100, "bin [{lo},{hi}) underpopulated: {n}");
        let mean = s_vr / n as f64;
        let var = s_vr2 / n as f64 - mean * mean;
        let sigma_r_meas = var.max(0.0).sqrt();
        let mean_r = s_r / n as f64;
        let q_meas = sigma_r_meas * kappa_indep(&d, mean_r)
            / (TOOMRE_FACTOR * d.g * sigma_indep(&d, mean_r));
        assert!(
            (q_meas - q).abs() < 0.15 * q,
            "bin [{lo},{hi}): measured Q={q_meas} vs input {q}"
        );
    }
}

#[test]
fn warm_realization_vertical_dispersion_matches() {
    let d = fiducial_warm(1.5);
    let s = d.sample(N_HALO, N_DISK, 0xC0FFEE);
    let idx = disk_indices(&s);
    let edges = [0.3_f64, 0.6, 0.9, 1.2];
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let (mut s_vz2, mut s_r, mut n) = (0.0, 0.0, 0usize);
        for &i in &idx {
            let (r, _, _, v_z) = cylindrical(s.pos[i], s.vel[i]);
            if r >= lo && r < hi {
                s_vz2 += v_z * v_z;
                s_r += r;
                n += 1;
            }
        }
        assert!(n > 80, "bin [{lo},{hi}) underpopulated: {n}");
        let sigma_z_meas = (s_vz2 / n as f64).sqrt();
        let mean_r = s_r / n as f64;
        let want = (PI * d.g * sigma_indep(&d, mean_r) * d.scale_height).sqrt();
        assert!(
            (sigma_z_meas - want).abs() < 0.15 * want,
            "bin [{lo},{hi}): σ_z meas={sigma_z_meas} vs {want}"
        );
    }
}

#[test]
fn warm_realization_mean_vphi_lags_circular_speed() {
    let d = fiducial_warm(1.5);
    let s = d.sample(N_HALO, N_DISK, 0xC0FFEE);
    let idx = disk_indices(&s);
    let edges = [0.4_f64, 0.7, 1.0, 1.3];
    for w in edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let (mut s_vphi, mut s_r, mut n) = (0.0, 0.0, 0usize);
        for &i in &idx {
            let (r, v_phi, _, _) = cylindrical(s.pos[i], s.vel[i]);
            if r >= lo && r < hi {
                s_vphi += v_phi;
                s_r += r;
                n += 1;
            }
        }
        assert!(n > 80, "bin [{lo},{hi}) underpopulated: {n}");
        let mean_vphi = s_vphi / n as f64;
        let mean_r = s_r / n as f64;
        let vc = v_c_indep(&d, mean_r);
        // Asymmetric drift: the mean azimuthal streaming lags the circular speed,
        // and matches the analytic v̄_φ.
        assert!(
            mean_vphi < vc,
            "bin [{lo},{hi}): ⟨v_φ⟩={mean_vphi} !< v_c={vc}"
        );
        let vbar = d.mean_azimuthal_velocity(mean_r);
        assert!(
            (mean_vphi - vbar).abs() < 0.06 * vc,
            "bin [{lo},{hi}): ⟨v_φ⟩={mean_vphi} vs analytic v̄_φ={vbar}"
        );
    }
}

#[test]
fn warm_disk_stays_zero_com_zero_momentum() {
    let d = fiducial_warm(1.5);
    let s = d.sample(N_HALO, N_DISK, 0xC0FFEE);
    let (mut mom, mut com, mut mtot) = (DVec3::ZERO, DVec3::ZERO, 0.0);
    for i in 0..s.len() {
        mom += s.vel[i] * s.mass[i];
        com += s.pos[i] * s.mass[i];
        mtot += s.mass[i];
    }
    assert!(mom.length() < 1e-9 * mtot, "net momentum not zero: {mom:?}");
    assert!((com / mtot).length() < 1e-9, "COM not at origin");
}

#[test]
fn warm_positions_match_cold_but_velocities_differ() {
    // Warmth must perturb only velocities: the position draws are identical, so a
    // warm and a cold disk with the same seed share every particle position but
    // differ in velocity. (This also pins that the cold path stays bit-identical.)
    let cold = fiducial_cold();
    let warm = fiducial_warm(1.5);
    let sc = cold.sample(N_HALO, N_DISK, 0xC0FFEE);
    let sw = warm.sample(N_HALO, N_DISK, 0xC0FFEE);
    let mut any_vel_diff = false;
    for i in 0..sc.len() {
        assert!(
            (sc.pos[i] - sw.pos[i]).length() < 1e-12,
            "position {i} diverged between cold and warm"
        );
        if (sc.vel[i] - sw.vel[i]).length() > 1e-9 {
            any_vel_diff = true;
        }
    }
    assert!(any_vel_diff, "warm disk velocities should differ from cold");
}

#[test]
fn warm_disk_is_deterministic_in_seed() {
    let d = fiducial_warm(1.5);
    let a = d.sample(2000, 2000, 0x1234);
    let b = d.sample(2000, 2000, 0x1234);
    for i in 0..a.len() {
        assert!((a.pos[i] - b.pos[i]).length() < 1e-15);
        assert!((a.vel[i] - b.vel[i]).length() < 1e-15);
    }
}
