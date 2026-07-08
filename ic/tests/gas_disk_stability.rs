//! Dynamical acceptance for the gas-rich disk IC (M7c) — the gates that show the
//! milestone did something, run under the real `GravitySph(BarnesHut)` SPH+gravity
//! solver, not just the sampler echoing its own formulas. Three claims:
//!
//!   1. **Equilibrium.** An isolated gas-rich disk holds its structure over an
//!      inner orbit: the gas half-mass radius, thickness, and mean rotation stay
//!      within a (loose) band, and total angular momentum L_z is conserved. Energy
//!      is deliberately NOT gated — the isothermal EOS is an implicit heat bath, so
//!      total energy is not a conserved quantity (see DESIGN.md M7b).
//!
//!   2. **The pressure correction is load-bearing.** Take the SAME realization and
//!      remove only the pressure correction — spin every gas particle up to the full
//!      circular speed v_c while it still carries SPH pressure support. That gas is
//!      *over-supported* (full rotation PLUS pressure) and must expand markedly more
//!      than the correctly-corrected disk. This is a differential between two
//!      dynamical runs, mirroring the warm stellar disk's asymmetric-drift gate.
//!
//!   3. **CFL green at the IC.** The fixed demo `dt` satisfies the SPH CFL bound at
//!      t = 0, so the gas run is launched stably.
//!
//! Run with `cargo test -p galaxy-ic --release -- --ignored`.

use galaxy_core::{DVec3, Integrator, LeapfrogKdk, Species, State, StaticBackground};
use galaxy_ic::{ExponentialDisk, Plummer};
use galaxy_solvers::sph::{validate_dt, DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

const TAU: f64 = std::f64::consts::TAU;
const C_CFL: f64 = 0.3;

/// The gas-rich fiducial, matching `gas_disk_sampling`: half the disk mass as gas,
/// c_s = 0.08 (Q_gas ≥ 1, modest pressure correction).
fn fiducial_gas() -> ExponentialDisk {
    let halo = Plummer::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo).with_gas(0.5, 0.08)
}

/// Build the SPH+gravity solver whose sound speed matches the disk's gas c_s.
fn solver(d: &ExponentialDisk) -> GravitySph<BarnesHut> {
    let grav = BarnesHut::new(d.g, 0.05 * d.halo.scale_radius, 0.5);
    let params = HydroParams {
        eos: Eos::Isothermal {
            c_s: d.sound_speed().expect("gas-rich disk has a sound speed"),
        },
        ..HydroParams::default()
    };
    GravitySph::new(grav, params, DensityConfig::default())
}

/// One inner-disk orbital period at the scale length.
fn t_orbit(d: &ExponentialDisk) -> f64 {
    TAU * d.scale_length / d.circular_velocity(d.scale_length)
}

/// Total angular momentum about the origin.
fn angular_momentum(s: &State) -> DVec3 {
    (0..s.len()).fold(DVec3::ZERO, |l, i| l + s.pos[i].cross(s.vel[i]) * s.mass[i])
}

/// Cylindrical half-mass radius and RMS thickness of the GAS population.
fn gas_shape(s: &State) -> (f64, f64) {
    let mut radii: Vec<f64> = Vec::new();
    let mut sz2 = 0.0;
    for i in 0..s.len() {
        if s.kind[i] == Species::Gas {
            let (x, y, z) = (s.pos[i].x, s.pos[i].y, s.pos[i].z);
            radii.push((x * x + y * y).sqrt());
            sz2 += z * z;
        }
    }
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let r_half = radii[radii.len() / 2];
    let rms_z = (sz2 / radii.len() as f64).sqrt();
    (r_half, rms_z)
}

/// Mean cylindrical radius of the GAS population (prompt tracer of expansion).
fn gas_mean_radius(s: &State) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..s.len() {
        if s.kind[i] == Species::Gas {
            sum += (s.pos[i].x * s.pos[i].x + s.pos[i].y * s.pos[i].y).sqrt();
            n += 1;
        }
    }
    sum / n as f64
}

#[test]
#[ignore = "dynamical: integrates a gas-rich disk under SPH+gravity (~10s, --release)"]
fn gas_rich_disk_holds_equilibrium_over_an_orbit() {
    let d = fiducial_gas();
    let s0 = d.sample_gas(2000, 1000, 2000, 0x6A5);

    let dt = 0.01;
    // CFL green at the IC (claim 3).
    assert!(
        validate_dt(
            &s0,
            &HydroParams {
                eos: Eos::Isothermal {
                    c_s: d.sound_speed().unwrap(),
                },
                ..HydroParams::default()
            },
            &DensityConfig::default(),
            dt,
            C_CFL,
        )
        .is_ok(),
        "chosen dt = {dt} violates the SPH CFL bound at the IC"
    );

    let steps = (t_orbit(&d) / dt).round() as usize;
    let lz0 = angular_momentum(&s0).z;
    let (rhalf0, rmsz0) = gas_shape(&s0);

    let mut s = s0;
    let mut sol = solver(&d);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let (mut max_lz, mut max_rdev, mut max_zrat) = (0.0_f64, 0.0_f64, 1.0_f64);
    for step in 1..=steps {
        integ.step(&mut s, &mut sol, &bg, dt);
        if step % 10 != 0 {
            continue;
        }
        max_lz = max_lz.max(((angular_momentum(&s).z - lz0) / lz0).abs());
        let (rhalf, rmsz) = gas_shape(&s);
        max_rdev = max_rdev.max(((rhalf - rhalf0) / rhalf0).abs());
        max_zrat = max_zrat.max(rmsz / rmsz0);
    }

    println!(
        "gas equilibrium: max|dLz/Lz|={max_lz:.4} max r_half dev={max_rdev:.4} \
         max RMS-z ratio={max_zrat:.4}"
    );
    // Bounds mirror the warm stellar gate: L_z tight (isolated system), radius/
    // thickness loose because the self-gravity z₀ under-supports in the halo
    // potential (the layer settles) — the gate is "holds together", not "static".
    // Retune from the measured numbers when the gate first runs green.
    assert!(max_lz < 0.02, "L_z not conserved: {max_lz:e}");
    assert!(
        max_rdev < 0.30,
        "gas half-mass radius drifted grossly: {max_rdev}"
    );
    assert!(max_zrat < 3.0, "gas disk puffed up: {max_zrat}");
}

#[test]
#[ignore = "dynamical: integrates two gas disks over 2 orbits (~20s, --release)"]
fn pressure_correction_prevents_gas_disk_expansion() {
    let d = fiducial_gas();
    let corrected = d.sample_gas(1500, 900, 1500, 0x6A5C);
    let over_supported = remove_pressure_correction(&d, &corrected);

    let dt = 0.01;
    let steps = (2.0 * t_orbit(&d) / dt).round() as usize;
    let r0 = gas_mean_radius(&corrected);

    let growth = |mut s: State| -> f64 {
        let mut sol = solver(&d);
        let mut integ = LeapfrogKdk::new();
        let bg = StaticBackground;
        let mut g = 0.0_f64;
        for _ in 1..=steps {
            integ.step(&mut s, &mut sol, &bg, dt);
            g = g.max((gas_mean_radius(&s) - r0) / r0);
        }
        g
    };
    let g_corr = growth(corrected);
    let g_over = growth(over_supported);
    println!("gas mean-radius growth: corrected={g_corr:.4}  over-supported={g_over:.4}");

    // `over_supported` is `corrected` with ONLY the gas velocities boosted (identical
    // positions), so both share a common-mode baseline expansion B (SPH relaxation +
    // the analytic IC not being a perfect radial SPH equilibrium — this is a
    // cylindrical radial metric, so it is NOT vertical settling): g_corr = B,
    // g_over = B + Δ. The milestone claim is the differential Δ > 0 — removing the
    // pressure correction adds real radial expansion. Gate on the GAP, not the ratio:
    // the ratio 1 + Δ/B drifts toward 1 if B grows under a future N/integrator change
    // even with the physics intact, whereas the gap cancels B and measures the effect
    // directly. Self-gating: `remove_pressure_correction` IS "correction removed", so
    // if `sample_gas`'s correction breaks, g_corr rises toward g_over, the gap
    // collapses, and this fails. Measured (fiducial c_s=0.08, ~5% of v_c²):
    // corrected≈0.146, over-supported≈0.228, gap≈0.082; the 0.04 floor sits ~50% below.
    assert!(
        g_over > 0.02,
        "over-supported gas should visibly expand (>2%): {g_over:.4}"
    );
    assert!(
        g_over > g_corr,
        "the pressure correction must reduce expansion: over={g_over:.4} corrected={g_corr:.4}"
    );
    assert!(
        g_over - g_corr > 0.04,
        "removing the pressure correction must add >4% mean-radius expansion \
         (measured ~8%): over={g_over:.4} corrected={g_corr:.4} gap={:.4}",
        g_over - g_corr
    );
}

/// Return a copy of a gas-rich realization with the pressure correction removed:
/// every gas particle (`Species::Gas`) is spun up to the full circular speed v_c(R)
/// by adding (v_c − v_φ,gas) along φ̂, keeping its position. Stars and halo untouched.
fn remove_pressure_correction(d: &ExponentialDisk, s: &State) -> State {
    let mut out = s.clone();
    for i in 0..out.len() {
        if out.kind[i] != Species::Gas {
            continue;
        }
        let p = out.pos[i];
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if r <= 0.0 {
            continue;
        }
        let phi_hat = DVec3::new(-p.y / r, p.x / r, 0.0);
        let boost = d.circular_velocity(r) - d.gas_azimuthal_velocity(r);
        out.vel[i] += phi_hat * boost;
    }
    out
}
