//! Dynamical sanity gate for the cold disk in a CUSPY halo, the dynamical
//! counterpart to the analytic/statistical gates in `disk_in_cuspy_halo.rs`
//! (which pin the *sampling*). It mirrors `disk_stability.rs` (the Plummer version)
//! but at deliberately HIGHER resolution — and that difference is the physics.
//!
//! A cold disk on circular orbits is in equilibrium in the *smooth* combined
//! potential the analytic v_c is built from. But the disk here sits in a **live
//! N-body halo**, and a cusp (ρ ∝ r⁻¹) is far harder to reproduce than a Plummer
//! core: at the cored-halo resolution (N_halo=1000, ε=0.05·r_s) the N-body inward
//! force in the inner cusp falls several-fold below the analytic G·M(<r)/r², so the
//! disk — placed on the analytic curve — grossly over-rotates and flies apart
//! (r_half drifts ~80% over one orbit). This is a resolution/softening artifact of
//! the live halo, NOT an IC defect: the sampling is exact (analytic gates), the disk
//! is simply reading a smooth force the under-resolved cusp doesn't deliver. So the
//! cuspy disk needs a **smaller softening fraction and more halo particles** than the
//! cored case for the live halo to reproduce the force the disk sits on. This gate
//! runs at that resolution and checks, over one inner-disk orbit:
//!
//!   - total energy conserved (bounded symplectic oscillation, not drift);
//!   - total angular momentum L_z conserved (isolated system);
//!   - the disk's cylindrical half-mass radius AND its 90% Lagrangian radius both
//!     stay within a loose band — the two together are the meaningful gate: r_half
//!     catches a bulk fall-in/fling-out, and r90 catches a minority of inner
//!     (least-resolved) particles blowing outward that a median would mask;
//!   - the disk stays a disk (RMS thickness does not blow up) — a LOOSE sanity
//!     bound only: the cold sheet has v_z = 0 (a geometric layer, not a vertical
//!     equilibrium; see `disk.rs`), so it settles/phase-mixes vertically, and that
//!     settling is stronger in the steeper cuspy vertical field. The radial gates,
//!     not this one, are what would fail on a mis-set rotation curve.

use galaxy_core::{DVec3, Integrator, LeapfrogKdk, Progenitor, State, StaticBackground};
use galaxy_ic::{ExponentialDisk, Hernquist};
use galaxy_solvers::BarnesHut;

// Resolution chosen so the live N-body cusp reproduces the analytic force the disk
// is placed on (see the module docstring): a cored Plummer holds at N_halo=1000,
// ε=0.05; the cusp needs ~6× the particles and 5× smaller softening.
const N_HALO: usize = 6000;
const N_DISK: usize = 1500;
const SEED: u64 = 0x5AB1E;
const EPS_FRAC: f64 = 0.01;

fn angular_momentum(s: &State) -> DVec3 {
    let mut l = DVec3::ZERO;
    for i in 0..s.len() {
        l += s.pos[i].cross(s.vel[i]) * s.mass[i];
    }
    l
}

/// (cylindrical half-mass radius, 90% Lagrangian radius, RMS thickness) of the disk
/// population. r90 catches a minority of inner particles blowing outward — the
/// failure mode the median r_half would mask.
fn disk_shape(s: &State) -> (f64, f64, f64) {
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
    let r90 = radii[(radii.len() * 9) / 10];
    let rms_z = (sz2 / radii.len() as f64).sqrt();
    (r_half, r90, rms_z)
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
    let (rhalf0, r90_0, rmsz0) = disk_shape(&s);

    let mut max_e_err = 0.0_f64;
    let mut max_lz_err = 0.0_f64;
    let mut max_rhalf_dev = 0.0_f64;
    let mut max_r90_dev = 0.0_f64;
    let mut max_rmsz_ratio = 1.0_f64;

    // The cheap O(N) structure/momentum checks run every 10 steps; the O(N²) energy
    // diagnostic runs only at ~5 checkpoints to keep the (higher-N) gate fast.
    let energy_every = steps / 5;
    for step in 1..=steps {
        integ.step(&mut s, &mut solver, &bg, dt);
        if step % 10 == 0 {
            let lz = angular_momentum(&s).z;
            max_lz_err = max_lz_err.max(((lz - lz0) / lz0).abs());
            let (rhalf, r90, rmsz) = disk_shape(&s);
            max_rhalf_dev = max_rhalf_dev.max(((rhalf - rhalf0) / rhalf0).abs());
            max_r90_dev = max_r90_dev.max(((r90 - r90_0) / r90_0).abs());
            max_rmsz_ratio = max_rmsz_ratio.max(rmsz / rmsz0);
        }
        if step % energy_every == 0 {
            let e = galaxy_core::diagnostics::total_energy(&s, &solver);
            max_e_err = max_e_err.max(((e - e0) / e0).abs());
        }
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
        max_r90_dev < 0.20,
        "disk 90% Lagrangian radius drifted grossly (inner blowout?): {max_r90_dev}"
    );
    // Loose sanity bound only (see module docstring): the cold v_z=0 sheet settles
    // vertically, more so in a cusp. Observed ratio ~1.7–2.6 across seeds.
    assert!(
        max_rmsz_ratio < 3.5,
        "disk puffed up grossly (RMS thickness ratio): {max_rmsz_ratio}"
    );
}

fn fiducial() -> ExponentialDisk<Hernquist> {
    let halo = Hernquist::new(1.0, 1.0, 1.0);
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, halo)
}
