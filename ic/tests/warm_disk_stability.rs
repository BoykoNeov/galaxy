//! Dynamical acceptance for the WARM disk — the test that shows the milestone did
//! something, not just that the sampler echoes its own dispersions. Two dynamical
//! claims, run under the BarnesHut workhorse:
//!
//!   1. Equilibrium: a warm disk (Toomre Q ≈ 1.5, dispersion balanced by the
//!      asymmetric-drift rotation lag) holds together over an inner orbit — energy
//!      and L_z conserved, half-mass radius and thickness bounded. The cold-disk
//!      gate in `disk_stability.rs` proves the cold limit; this proves warmth did
//!      not break equilibrium.
//!
//!   2. The drift is load-bearing: take the SAME warm realization and remove only
//!      the asymmetric-drift correction — spin every disk particle back up to the
//!      full circular speed v_c while keeping its dispersion. That disk is
//!      *over-supported* (full rotation PLUS pressure) and must expand markedly more
//!      than the correctly-drifted warm disk. This isolates the drift: if
//!      `mean_azimuthal_velocity` returned v_c (no drift), the two runs would be
//!      identical and the gap would vanish.
//!
//! The second test is the real content — a differential between two dynamical runs,
//! not a comparison of a realization to the formula that made it.

use galaxy_core::{
    diagnostics, DVec3, Integrator, LeapfrogKdk, Progenitor, State, StaticBackground,
};
use galaxy_ic::{ExponentialDisk, Plummer};
use galaxy_solvers::BarnesHut;

const TAU: f64 = std::f64::consts::TAU;

fn fiducial_warm(q: f64) -> ExponentialDisk {
    let halo = Plummer::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo).with_toomre_q(q)
}

/// Total angular momentum about the origin.
fn angular_momentum(s: &State) -> DVec3 {
    (0..s.len()).fold(DVec3::ZERO, |l, i| l + s.pos[i].cross(s.vel[i]) * s.mass[i])
}

/// Axial angular momentum L_z of the DISK population only (the halo is
/// non-rotating, so its shot noise would swamp the disk's small drift signal).
fn disk_lz(s: &State) -> f64 {
    (0..s.len())
        .filter(|&i| s.progenitor[i] == Progenitor(1))
        .fold(0.0, |lz, i| {
            lz + (s.pos[i].x * s.vel[i].y - s.pos[i].y * s.vel[i].x) * s.mass[i]
        })
}

/// Mean cylindrical radius of the disk population. More responsive than the median
/// to a coherent outward shift: an over-supported particle swings to a larger
/// apocenter within half an epicyclic period, so ⟨R⟩ tracks expansion promptly.
fn disk_mean_radius(s: &State) -> f64 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for i in 0..s.len() {
        if s.progenitor[i] == Progenitor(1) {
            sum += (s.pos[i].x * s.pos[i].x + s.pos[i].y * s.pos[i].y).sqrt();
            n += 1;
        }
    }
    sum / n as f64
}

/// Cylindrical half-mass radius and RMS thickness of the disk population.
fn disk_shape(s: &State) -> (f64, f64) {
    let mut radii: Vec<f64> = Vec::new();
    let mut sz2 = 0.0;
    for i in 0..s.len() {
        if s.progenitor[i] == Progenitor(1) {
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

/// One inner-disk orbital period at the scale length.
fn t_orbit(d: &ExponentialDisk) -> f64 {
    TAU * d.scale_length / d.circular_velocity(d.scale_length)
}

fn integrate(mut s: State, d: &ExponentialDisk, dt: f64, steps: usize) -> Vec<State> {
    let mut solver = BarnesHut::new(d.g, 0.05 * d.halo.scale_radius, 0.5);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let mut out = vec![s.clone()];
    for step in 1..=steps {
        integ.step(&mut s, &mut solver, &bg, dt);
        if step % 10 == 0 {
            out.push(s.clone());
        }
    }
    out
}

#[test]
fn warm_disk_holds_equilibrium_over_an_orbit() {
    let d = fiducial_warm(1.5);
    let s0 = d.sample(2000, 1000, 0x5EED);

    let dt = 0.02;
    let steps = (t_orbit(&d) / dt).round() as usize;

    let mut solver = BarnesHut::new(d.g, 0.05 * d.halo.scale_radius, 0.5);
    let e0 = diagnostics::total_energy(&s0, &solver);
    let lz0 = angular_momentum(&s0).z;
    let (rhalf0, rmsz0) = disk_shape(&s0);

    let mut s = s0;
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let (mut max_e, mut max_lz, mut max_rdev, mut max_zrat) = (0.0_f64, 0.0_f64, 0.0_f64, 1.0_f64);
    for step in 1..=steps {
        integ.step(&mut s, &mut solver, &bg, dt);
        if step % 10 != 0 {
            continue;
        }
        let e = diagnostics::total_energy(&s, &solver);
        max_e = max_e.max(((e - e0) / e0).abs());
        max_lz = max_lz.max(((angular_momentum(&s).z - lz0) / lz0).abs());
        let (rhalf, rmsz) = disk_shape(&s);
        max_rdev = max_rdev.max(((rhalf - rhalf0) / rhalf0).abs());
        max_zrat = max_zrat.max(rmsz / rmsz0);
    }

    println!(
        "warm equilibrium: max|dE/E|={max_e:.4} max|dLz/Lz|={max_lz:.4} \
         max r_half dev={max_rdev:.4} max RMS-z ratio={max_zrat:.4}"
    );
    // Measured (seed 0x5EED): dE/E≈2e-4, dLz/Lz≈1e-3 (both excellent), r_half
    // dev≈0.22, RMS-z ratio≈2.0. The bounds are deliberately looser than the cold
    // gate's (0.20 / 3.0) because a WARM disk legitimately breathes more: σ_R puts
    // particles on epicyclic orbits (radial breathing → larger r_half swing), and
    // the sech² layer is a geometric profile, not a vertical equilibrium in the
    // combined disk+halo potential, so it puffs toward its true scale height (the
    // same mechanism the cold gate already tolerates; σ_z here is the disk's
    // self-gravity value, a documented under-support — see `disk.rs`). The gate is
    // "holds together" (no collapse, no fly-apart), not "static".
    assert!(max_e < 0.02, "energy not conserved: {max_e:e}");
    assert!(max_lz < 0.02, "L_z not conserved: {max_lz:e}");
    assert!(
        max_rdev < 0.30,
        "warm disk half-mass radius drifted grossly: {max_rdev}"
    );
    assert!(max_zrat < 3.0, "warm disk puffed up: {max_zrat}");
}

#[test]
#[ignore = "dynamical demonstration: integrates two disks over 2 orbits (~10s)"]
fn asymmetric_drift_prevents_disk_expansion() {
    // A disk warm enough (Q≈3) for the drift to be a ~10% velocity correction, so
    // the over-supported case separates cleanly. Same realization, two velocity
    // fields: WITH drift (as sampled) and WITHOUT (disk spun up to the full circular
    // speed, keeping the dispersion → over-supported: full rotation PLUS pressure).
    let d = fiducial_warm(3.0);
    let drifted = d.sample(1500, 900, 0xACCE1);
    let over_supported = remove_drift(&d, &drifted);

    // Sanity: the two states differ ONLY in disk azimuthal velocity, and the
    // over-supported disk carries more disk azimuthal streaming (higher disk L_z).
    assert!(
        disk_lz(&over_supported) > disk_lz(&drifted) * 1.01,
        "removing the drift must raise the disk's azimuthal streaming: \
         over={} vs drift={}",
        disk_lz(&over_supported),
        disk_lz(&drifted)
    );

    let dt = 0.02;
    let steps = (2.0 * t_orbit(&d) / dt).round() as usize; // two inner orbits

    let r0 = disk_mean_radius(&drifted);
    let series_drift = integrate(drifted, &d, dt, steps);
    let series_over = integrate(over_supported, &d, dt, steps);

    let growth = |series: &[State]| -> f64 {
        series
            .iter()
            .map(|s| (disk_mean_radius(s) - r0) / r0)
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let g_drift = growth(&series_drift);
    let g_over = growth(&series_over);
    println!("mean-radius growth: drifted={g_drift:.4}  over-supported={g_over:.4}");

    // The correctly-drifted warm disk stays close to equilibrium; the over-supported
    // one expands. The differential is the milestone claim: the drift is what keeps
    // the warm disk from flinging outward.
    assert!(
        g_over > 0.02,
        "over-supported disk should visibly expand (>2%): {g_over:.4}"
    );
    assert!(
        g_over > 3.0 * g_drift,
        "over-supported disk must expand several times more than the near-\
         equilibrium drifted one: over={g_over:.4} vs drift={g_drift:.4}"
    );
}

/// Return a copy of a warm-disk realization with the asymmetric-drift correction
/// removed: every disk particle (`Progenitor(1)`) is spun up to the full circular
/// speed v_c(R) by adding (v_c − v̄_φ) along φ̂, keeping its position and dispersion.
/// The halo is untouched.
fn remove_drift(d: &ExponentialDisk, s: &State) -> State {
    let mut out = s.clone();
    for i in 0..out.len() {
        if out.progenitor[i] != Progenitor(1) {
            continue;
        }
        let p = out.pos[i];
        let r = (p.x * p.x + p.y * p.y).sqrt();
        if r <= 0.0 {
            continue;
        }
        let phi_hat = DVec3::new(-p.y / r, p.x / r, 0.0);
        let boost = d.circular_velocity(r) - d.mean_azimuthal_velocity(r);
        out.vel[i] += phi_hat * boost;
    }
    out
}
