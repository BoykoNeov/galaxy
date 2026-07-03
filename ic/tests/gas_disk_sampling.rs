//! Validation of the isothermal gas option on `ExponentialDisk` (M7c, D7). Two
//! layers, mirroring `disk_sampling.rs`:
//!
//!   1. Analytic self-consistency (no sampling): the gas surface density and its
//!      normalization, the cylindrical gas enclosed mass, the self-gravity scale
//!      height z₀(R), the pressure-corrected rotation v_φ,gas(R), and the gas Toomre
//!      parameter Q_gas(R) must all match independently hand-derived closed forms.
//!   2. Statistical validation of a realization: gas particles must reproduce the
//!      analytic radial CDF, sit on the pressure-corrected rotation curve at bin
//!      means, and realize the flaring sech² scale height z₀(R) statistically; the
//!      gas must be tagged `Species::Gas`/`Progenitor(4)`, the whole galaxy must
//!      carry zero net momentum/COM, and — the stream-spacing invariant — the halo
//!      and the STELLAR POSITIONS must be bit-identical to the gas-free IC.
//!
//! Expectations are independent closed forms written inline here, not the code's own
//! output. The gas model: f_gas splits the total disk mass, so gas traces the same
//! truncated exponential as the stars; the NEW physics is the thermal layer
//! z₀ = c_s²/(πGΣ_gas), the pressure correction v_φ,gas² = v_c² − 2c_s²R/Rd, and the
//! π-prefactor Toomre Q_gas = c_s κ/(πGΣ_gas).

use galaxy_core::{DVec3, Progenitor, Species, State};
use galaxy_ic::{ExponentialDisk, Plummer};

const PI: f64 = std::f64::consts::PI;
const G: f64 = 1.0;

// ---------- fiducial galaxies shared by the tests ----------

/// The gas-free reference disk (identical to `disk_sampling::fiducial`).
fn fiducial_stellar() -> ExponentialDisk {
    let halo = Plummer::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}

/// The gas-rich fiducial: half the disk mass as gas, sound speed 0.08 — a band
/// where Q_gas ≥ 1 (stable) and the pressure correction stays a modest fraction of
/// v_c across the disk body (not a pressure blob). Tune with the measured stability
/// numbers when the gates go green.
fn fiducial_gas() -> ExponentialDisk {
    fiducial_stellar().with_gas(0.5, 0.08)
}

/// Truncated-exponential central surface density Σ₀, independent closed form:
/// Σ₀ = M / (2π Rd² · [1 − (1+u)e^(−u)]), u = r_max/Rd.
fn central_sigma(m: f64, rd: f64, r_max: f64) -> f64 {
    let u = r_max / rd;
    m / (2.0 * PI * rd * rd * (1.0 - (1.0 + u) * (-u).exp()))
}

/// Truncated-exponential cylindrical enclosed mass, independent closed form.
fn trunc_enclosed(m: f64, rd: f64, r_max: f64, r: f64) -> f64 {
    let f = |x: f64| 1.0 - (1.0 + x / rd) * (-x / rd).exp();
    m * f(r.min(r_max)) / f(r_max)
}

// ---------- 1. analytic self-consistency ----------

#[test]
fn gas_surface_density_normalizes_to_gas_mass() {
    let d = fiducial_gas();
    let (rd, r_max) = (d.scale_length, d.r_max);
    let m_gas = 0.5 * d.disk_mass; // f_gas = 0.5
    let sigma_gas0 = central_sigma(m_gas, rd, r_max);

    assert!(
        (d.gas_mass() - m_gas).abs() < 1e-12 * m_gas,
        "M_gas = {} vs expected {m_gas}",
        d.gas_mass()
    );
    assert!(
        (d.gas_central_surface_density() - sigma_gas0).abs() < 1e-12 * sigma_gas0,
        "Σ_gas,0 = {} vs expected {sigma_gas0}",
        d.gas_central_surface_density()
    );
    // Exponential falloff, truncating past r_max.
    for &r in &[0.0_f64, 0.3, 1.0, 1.9] {
        let want = sigma_gas0 * (-r / rd).exp();
        assert!(
            (d.gas_surface_density(r) - want).abs() < 1e-12 * sigma_gas0,
            "Σ_gas({r}) = {} vs expected {want}",
            d.gas_surface_density(r)
        );
    }
    assert_eq!(
        d.gas_surface_density(r_max * 1.01),
        0.0,
        "no gas past r_max"
    );
}

#[test]
fn gas_enclosed_mass_matches_closed_form() {
    let d = fiducial_gas();
    let m_gas = 0.5 * d.disk_mass;
    for &r in &[0.1_f64, 0.5, 1.0, 2.0, 3.0] {
        let want = trunc_enclosed(m_gas, d.scale_length, d.r_max, r);
        assert!(
            (d.gas_enclosed_mass(r) - want).abs() < 1e-12 * m_gas,
            "M_gas(<{r}) = {} vs expected {want}",
            d.gas_enclosed_mass(r)
        );
    }
    // Self-consistency: the whole gas mass is enclosed by r_max.
    assert!(
        (d.gas_enclosed_mass(d.r_max) - m_gas).abs() < 1e-12 * m_gas,
        "gas enclosed at r_max must equal M_gas"
    );
}

#[test]
fn gas_scale_height_matches_spitzer_hand_values() {
    let d = fiducial_gas();
    let cs = 0.08_f64;
    let m_gas = 0.5 * d.disk_mass;
    let sigma_gas0 = central_sigma(m_gas, d.scale_length, d.r_max);
    // z₀(R) = c_s² / (π G Σ_gas(R)); Σ_gas(R) = Σ_gas,0 e^(−R/Rd).
    for &r in &[0.3_f64, 1.0, 1.5] {
        let sigma = sigma_gas0 * (-r / d.scale_length).exp();
        let want = cs * cs / (PI * G * sigma);
        assert!(
            (d.gas_scale_height(r) - want).abs() < 1e-10 * want,
            "z0({r}) = {} vs expected {want}",
            d.gas_scale_height(r)
        );
    }
    // Flares outward: z₀ strictly increases with R (Σ_gas falls).
    assert!(
        d.gas_scale_height(1.0) > d.gas_scale_height(0.3),
        "gas scale height must flare outward"
    );
}

#[test]
fn gas_azimuthal_velocity_is_pressure_corrected() {
    let d = fiducial_gas();
    let cs = 0.08_f64;
    // In the disk body, v_φ,gas² = v_c² − 2 c_s² R/Rd (independent of the sampler).
    for &r in &[0.4_f64, 0.6, 1.0] {
        let vc = d.circular_velocity(r);
        let want2 = vc * vc - 2.0 * cs * cs * r / d.scale_length;
        assert!(
            want2 > 0.0,
            "test radius {r} should be in the supported regime"
        );
        let want = want2.sqrt();
        assert!(
            (d.gas_azimuthal_velocity(r) - want).abs() < 1e-10 * vc,
            "v_phi,gas({r}) = {} vs expected {want}",
            d.gas_azimuthal_velocity(r)
        );
        // The correction is negative: gas rotates slower than the stars' v_c.
        assert!(d.gas_azimuthal_velocity(r) < vc, "gas must lag v_c");
    }
}

#[test]
fn gas_azimuthal_velocity_clamps_to_zero_near_center() {
    // Near R→0, v_c² ~ R² vanishes faster than the linear pressure term, so the raw
    // v_φ,gas² goes negative — the clamp must return 0, never a NaN.
    let d = fiducial_gas();
    for &r in &[1e-6_f64, 1e-4, 1e-3] {
        let v = d.gas_azimuthal_velocity(r);
        assert!(
            v.is_finite() && v >= 0.0,
            "v_phi,gas({r}) = {v} must be finite ≥ 0"
        );
    }
}

#[test]
fn gas_toomre_q_uses_the_gas_prefactor_pi() {
    let d = fiducial_gas();
    let cs = 0.08_f64;
    let m_gas = 0.5 * d.disk_mass;
    let sigma_gas0 = central_sigma(m_gas, d.scale_length, d.r_max);
    // Q_gas = c_s κ / (π G Σ_gas). κ is pinned independently by the warm-disk gates,
    // so reuse it; the NEW content gated here is the π prefactor and Σ_gas.
    for &r in &[0.5_f64, 1.0] {
        let sigma = sigma_gas0 * (-r / d.scale_length).exp();
        let want = cs * d.epicyclic_frequency(r) / (PI * G * sigma);
        assert!(
            (d.gas_toomre_q(r) - want).abs() < 1e-10 * want,
            "Q_gas({r}) = {} vs expected {want}",
            d.gas_toomre_q(r)
        );
    }
    // The fiducial must be stable (min over the body ≥ 1).
    assert!(
        d.min_gas_toomre_q() >= 1.0,
        "fiducial gas disk must be Toomre-stable: min Q_gas = {}",
        d.min_gas_toomre_q()
    );
}

#[test]
#[should_panic(expected = "Q_gas")]
fn unstable_gas_disk_is_rejected_loudly() {
    // A tiny sound speed makes Q_gas = c_s κ/(π G Σ_gas) ≪ 1 across the disk: a
    // gravitationally unstable, fragmenting gas layer. Constructing it must panic
    // rather than silently launch a doomed IC. (c_s and f_gas are individually in
    // range, so only the Q_gas guard can trip.)
    let _ = fiducial_stellar().with_gas(0.5, 0.005);
}

// ---------- 2. statistical validation of a realization ----------

const N_HALO: usize = 2000;
const N_DISK: usize = 1000;
const N_GAS: usize = 2000;
const SEED: u64 = 0x6A5;

/// Indices of the gas particles (`Species::Gas`).
fn gas_indices(s: &State) -> Vec<usize> {
    (0..s.len())
        .filter(|&i| s.kind[i] == Species::Gas)
        .collect()
}

#[test]
fn sampled_gas_is_tagged_species_gas_and_progenitor_four() {
    let d = fiducial_gas();
    let s = d.sample_gas(N_HALO, N_DISK, N_GAS, SEED);
    assert_eq!(s.len(), N_HALO + N_DISK + N_GAS);

    let gas = gas_indices(&s);
    assert_eq!(gas.len(), N_GAS, "exactly n_gas particles are gas");
    for &i in &gas {
        assert_eq!(s.progenitor[i], Progenitor(4), "gas tagged Progenitor(4)");
    }
    // Halo and stars stay collisionless.
    let collisionless = (0..s.len())
        .filter(|&i| s.kind[i] == Species::Collisionless)
        .count();
    assert_eq!(collisionless, N_HALO + N_DISK);
}

#[test]
fn gas_rich_disk_has_zero_net_momentum_and_com() {
    let d = fiducial_gas();
    let s = d.sample_gas(N_HALO, N_DISK, N_GAS, SEED);
    let mtot: f64 = s.mass.iter().sum();
    let com = (0..s.len()).fold(DVec3::ZERO, |a, i| a + s.pos[i] * s.mass[i]) / mtot;
    let mom = (0..s.len()).fold(DVec3::ZERO, |a, i| a + s.vel[i] * s.mass[i]);
    let scale = d.r_max;
    assert!(com.length() < 1e-10 * scale, "COM not centered: {com:?}");
    assert!(mom.length() < 1e-10, "net momentum not zero: {mom:?}");
}

#[test]
fn stellar_positions_bit_identical_to_gas_free_ic() {
    // The stream-spacing invariant (positions only — gas re-tags disk mass, so the
    // stellar particle MASSES legitimately shrink; the shared PRNG streams keep the
    // halo and the stellar POSITIONS bit-for-bit identical to the gas-free IC).
    let stellar = fiducial_stellar();
    let gassy = fiducial_gas();
    let s0 = stellar.sample(N_HALO, N_DISK, SEED);
    let sg = gassy.sample_gas(N_HALO, N_DISK, N_GAS, SEED);

    // Halo: first N_HALO particles, fully bit-identical (drawn from stream 0 first).
    for i in 0..N_HALO {
        assert_eq!(sg.pos[i], s0.pos[i], "halo position {i} drifted");
        assert_eq!(sg.vel[i], s0.vel[i], "halo velocity {i} drifted");
    }
    // Stellar POSITIONS: next N_DISK particles, positions bit-identical.
    for i in N_HALO..N_HALO + N_DISK {
        assert_eq!(
            sg.pos[i], s0.pos[i],
            "stellar position {i} drifted with gas on"
        );
    }
}

#[test]
fn zero_gas_reproduces_gas_free_stellar_positions() {
    // The degenerate n_gas = 0 case: reserving the gas stream must not shift the
    // stellar draws. Halo + stellar positions match the gas-free IC exactly.
    let stellar = fiducial_stellar();
    let gassy = fiducial_gas();
    let s0 = stellar.sample(N_HALO, N_DISK, SEED);
    let sg = gassy.sample_gas(N_HALO, N_DISK, 0, SEED);
    assert_eq!(sg.len(), N_HALO + N_DISK);
    for i in 0..N_HALO + N_DISK {
        assert_eq!(
            sg.pos[i], s0.pos[i],
            "position {i} shifted by reserving the gas stream"
        );
    }
}

#[test]
fn realization_reproduces_gas_radial_cdf() {
    // Gas radii must follow the truncated-exponential CDF (same shape as the stars).
    let d = fiducial_gas();
    let s = d.sample_gas(N_HALO, N_DISK, N_GAS, SEED);
    let m_gas = 0.5 * d.disk_mass;
    let mut radii: Vec<f64> = gas_indices(&s)
        .iter()
        .map(|&i| (s.pos[i].x * s.pos[i].x + s.pos[i].y * s.pos[i].y).sqrt())
        .collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    // Compare the empirical CDF to M_gas(<R)/M_gas at several quantiles.
    for &frac in &[0.25_f64, 0.5, 0.75] {
        let r = radii[(frac * radii.len() as f64) as usize];
        let cdf = trunc_enclosed(m_gas, d.scale_length, d.r_max, r) / m_gas;
        assert!(
            (cdf - frac).abs() < 0.05,
            "gas radial CDF off at frac {frac}: empirical R={r}, model CDF={cdf}"
        );
    }
}

#[test]
fn realization_recovers_flaring_sech2_scale_height() {
    // Bin gas by radius; the RMS thickness in each bin must track z₀(R). For a
    // sech²(z/z₀) layer, RMS(z) = z₀·π/√12 ≈ 0.9069·z₀. The gate is statistical
    // (finite N), and z₀ flares, so require the OUTER bin thicker than the INNER.
    let d = fiducial_gas();
    let s = d.sample_gas(N_HALO, N_DISK, N_GAS, SEED);
    let gas = gas_indices(&s);
    let rms_z_in_annulus = |lo: f64, hi: f64| -> (f64, f64) {
        let mut sz2 = 0.0;
        let mut r_sum = 0.0;
        let mut n = 0usize;
        for &i in &gas {
            let r = (s.pos[i].x * s.pos[i].x + s.pos[i].y * s.pos[i].y).sqrt();
            if r >= lo && r < hi {
                sz2 += s.pos[i].z * s.pos[i].z;
                r_sum += r;
                n += 1;
            }
        }
        assert!(n > 50, "annulus [{lo},{hi}) too sparse: {n}");
        ((sz2 / n as f64).sqrt(), r_sum / n as f64)
    };
    let (rms_in, r_in) = rms_z_in_annulus(0.2, 0.5);
    let (rms_out, r_out) = rms_z_in_annulus(0.8, 1.2);
    // Flaring: the outer annulus is thicker.
    assert!(
        rms_out > rms_in,
        "gas layer must flare: rms_in={rms_in}, rms_out={rms_out}"
    );
    // Magnitude: each bin near 0.9069·z₀ at its mean radius (loose, finite N).
    for &(rms, r) in &[(rms_in, r_in), (rms_out, r_out)] {
        let want = 0.9069 * d.gas_scale_height(r);
        assert!(
            (rms - want).abs() < 0.25 * want,
            "RMS-z at R≈{r}: {rms} vs expected ≈{want}"
        );
    }
}

#[test]
fn gas_mean_azimuthal_matches_pressure_corrected_curve_at_bin_means() {
    // Bin gas by radius; the mean azimuthal speed in each bin must sit on the
    // pressure-corrected v_φ,gas(R) at the bin's mean radius (not the stellar v_c).
    let d = fiducial_gas();
    let s = d.sample_gas(N_HALO, N_DISK, N_GAS, SEED);
    let gas = gas_indices(&s);
    for &(lo, hi) in &[(0.4_f64, 0.6), (0.7, 0.9)] {
        let mut vphi_sum = 0.0;
        let mut r_sum = 0.0;
        let mut n = 0usize;
        for &i in &gas {
            let (x, y) = (s.pos[i].x, s.pos[i].y);
            let r = (x * x + y * y).sqrt();
            if r >= lo && r < hi {
                // Azimuthal component v·φ̂, φ̂ = (−y, x, 0)/R.
                vphi_sum += (-s.vel[i].x * y + s.vel[i].y * x) / r;
                r_sum += r;
                n += 1;
            }
        }
        assert!(n > 50, "annulus [{lo},{hi}) too sparse: {n}");
        let (vphi_mean, r_mean) = (vphi_sum / n as f64, r_sum / n as f64);
        let want = d.gas_azimuthal_velocity(r_mean);
        assert!(
            (vphi_mean - want).abs() < 0.05 * want,
            "⟨v_φ,gas⟩ at R≈{r_mean}: {vphi_mean} vs pressure-corrected {want}"
        );
    }
}
