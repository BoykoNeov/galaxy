//! Dynamical sanity gate for the cold disk in a CUSPY halo, mirroring
//! `disk_stability.rs` (the Plummer version). A cold disk on circular orbits is in
//! equilibrium by construction in any spherical potential, so over ~one inner-disk
//! orbit under the BarnesHut workhorse it must:
//!
//!   - conserve total energy (bounded symplectic oscillation, not drift);
//!   - conserve total angular momentum L_z (isolated system);
//!   - keep its cylindrical half-mass radius within a loose band — a mis-set cuspy
//!     rotation curve (wrong M(<r)) would make the disk fall in or fling out;
//!   - stay thin (RMS |z| does not blow up).
//!
//! This is the dynamical counterpart to the analytic/statistical gates in
//! `disk_in_cuspy_halo.rs`; the Hernquist cusp is the interesting case because its
//! steep central force is exactly what a wrong rotation curve would expose.

use galaxy_core::{DVec3, Integrator, LeapfrogKdk, Progenitor, State, StaticBackground};
use galaxy_ic::{ExponentialDisk, Hernquist};
use galaxy_solvers::BarnesHut;

const N_HALO: usize = 1000;
const N_DISK: usize = 500;
const SEED: u64 = 0x5AB1E;
const EPS_FRAC: f64 = 0.05;

fn fiducial() -> ExponentialDisk<Hernquist> {
    let halo = Hernquist::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}

fn angular_momentum(s: &State) -> DVec3 {
    let mut l = DVec3::ZERO;
    for i in 0..s.len() {
        l += s.pos[i].cross(s.vel[i]) * s.mass[i];
    }
    l
}

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

#[test]
fn cold_disk_holds_together_in_cuspy_halo() {
    let d = fiducial();
    let mut s = d.sample(N_HALO, N_DISK, SEED);
    let mut solver = BarnesHut::new(d.g, EPS_FRAC * d.halo.scale_radius, 0.5);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;

    let t_orbit = std::f64::consts::TAU * d.scale_length / d.circular_velocity(d.scale_length);
    let dt = 0.02;
    let steps = (t_orbit / dt).round() as usize;

    let e0 = galaxy_core::diagnostics::total_energy(&s, &solver);
    let lz0 = angular_momentum(&s).z;
    let (rhalf0, rmsz0) = disk_shape(&s);

    let mut max_e_err = 0.0_f64;
    let mut max_lz_err = 0.0_f64;
    let mut max_rhalf_dev = 0.0_f64;
    let mut max_rmsz_ratio = 1.0_f64;

    for step in 1..=steps {
        integ.step(&mut s, &mut solver, &bg, dt);
        if step % 10 != 0 {
            continue;
        }
        let e = galaxy_core::diagnostics::total_energy(&s, &solver);
        max_e_err = max_e_err.max(((e - e0) / e0).abs());
        let lz = angular_momentum(&s).z;
        max_lz_err = max_lz_err.max(((lz - lz0) / lz0).abs());
        let (rhalf, rmsz) = disk_shape(&s);
        max_rhalf_dev = max_rhalf_dev.max(((rhalf - rhalf0) / rhalf0).abs());
        max_rmsz_ratio = max_rmsz_ratio.max(rmsz / rmsz0);
    }

    assert!(max_e_err < 0.02, "energy not conserved: {max_e_err:e}");
    assert!(
        max_lz_err < 0.02,
        "angular momentum L_z not conserved: {max_lz_err:e}"
    );
    assert!(
        max_rhalf_dev < 0.20,
        "disk half-mass radius drifted grossly: {max_rhalf_dev}"
    );
    assert!(
        max_rmsz_ratio < 3.0,
        "disk puffed up (RMS thickness ratio): {max_rmsz_ratio}"
    );
}
