//! GPU-SPH **G5a** gate: resident gas density on [`GpuResidentLeapfrog`].
//!
//! G1–G4 brought up the SPH stages as standalone, host-upload/readback passes. G5 wires
//! them into the GPU-resident stepper so gas hydro runs *without leaving the device*
//! between steps (the residency win M4i unlocked). G5a is the first landing: the
//! adaptive-h **density** stage, gathered onto the resident `bodies` buffer over the gas
//! subset and left resident as (ρ, h) for the hydro force (G5b) to consume.
//!
//! ## The gate: resident ρ/h vs the CPU oracle, over the gas subset
//! Gravity acts on ALL particles; hydro (and its density prerequisite) on the **gas
//! subset only** — exactly as the CPU composite [`galaxy_solvers::sph::GravitySph`]. So
//! the density gate extracts the gas rows and compares the resident device's (ρ, h)
//! against [`galaxy_solvers::sph::density_adaptive`] on that same subset, at the G2
//! f32-tolerance (never bit-exact — D1/D5).
//!
//! **Why the CPU oracle and not the standalone `GpuDensity`:** resident-vs-standalone
//! would be near bit-exact (both run the same WGSL), so it could not catch a bug the two
//! share. The f64 CPU path is the independent reference.
//!
//! GPU-gated: needs a wgpu adapter; without one `new_with_sph` returns `NoAdapter` and
//! the tests fail loudly (never silently skipped).

use galaxy_core::{DVec3, Species, State};
use galaxy_gpu::GpuResidentLeapfrog;
use galaxy_solvers::sph::{density_adaptive, DensityConfig, HydroParams};

const G: f64 = 1.0;
const EPS: f64 = 0.05;
const THETA: f64 = 0.5;

/// A gravity+SPH mixed cloud: a dense gas blob (so every gas particle is rooted at
/// `n_ngb = 48`) embedded in a wider, sparser star field. Gas sits at INTERLEAVED
/// indices (every 3rd particle) so the resident gas-index map is non-trivial — an
/// identity map (`gas_idx[k] == k`) would give the wrong neighbors and fail the gate.
fn gas_star_mix(seed: u64, n_gas: usize, n_star: usize) -> State {
    let mut s = seed | 1;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };

    // Build gas and star pools first, then interleave 1 gas : (ratio) stars by index.
    let mut gas_pos = Vec::with_capacity(n_gas);
    let mut gas_vel = Vec::with_capacity(n_gas);
    let mut gas_mass = Vec::with_capacity(n_gas);
    for _ in 0..n_gas {
        // Dense blob, radius ~1 — enough overlap for a 48-neighbor root.
        gas_pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 2.0);
        gas_vel.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 0.1);
        gas_mass.push(0.5 + 0.5 * next());
    }
    let mut star_pos = Vec::with_capacity(n_star);
    let mut star_vel = Vec::with_capacity(n_star);
    let mut star_mass = Vec::with_capacity(n_star);
    for _ in 0..n_star {
        star_pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 6.0);
        star_vel.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 0.1);
        star_mass.push(0.1 + 0.9 * next());
    }

    let mut pos = Vec::new();
    let mut vel = Vec::new();
    let mut mass = Vec::new();
    let mut kind = Vec::new();
    let (mut gi, mut si) = (0usize, 0usize);
    while gi < n_gas || si < n_star {
        // one gas, then as many stars as keep the ratio ~ n_star/n_gas
        if gi < n_gas {
            pos.push(gas_pos[gi]);
            vel.push(gas_vel[gi]);
            mass.push(gas_mass[gi]);
            kind.push(Species::Gas);
            gi += 1;
        }
        let take = (n_star / n_gas.max(1)).max(1);
        for _ in 0..take {
            if si < n_star {
                pos.push(star_pos[si]);
                vel.push(star_vel[si]);
                mass.push(star_mass[si]);
                kind.push(Species::Collisionless);
                si += 1;
            }
        }
    }

    let mut state = State::from_phase_space(pos, vel, mass);
    state.kind = kind;
    state.assert_consistent();
    state
}

/// Gas-subset (ascending index) positions/masses — the exact arrays the CPU oracle and
/// the resident gas map both consume.
fn gas_subset(state: &State) -> (Vec<usize>, Vec<DVec3>, Vec<f64>) {
    let idx: Vec<usize> = (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .collect();
    let pos = idx.iter().map(|&i| state.pos[i]).collect();
    let mass = idx.iter().map(|&i| state.mass[i]).collect();
    (idx, pos, mass)
}

/// f32-narrow a coordinate the way the GPU does (so the oracle sees the same positions
/// the device does — the density gate is f32-tolerance, not a narrowing test).
fn narrow_state(state: &mut State) {
    for p in state.pos.iter_mut() {
        *p = DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64);
    }
}

/// Max relative error between paired scalar fields (denominator floored so near-zero
/// entries don't blow the ratio up).
fn max_rel_err(a: &[f32], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "field length mismatch");
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - y).abs() / y.abs().max(1e-30))
        .fold(0.0_f64, f64::max)
}

/// **G5a primary gate.** One resident force evaluation (the prime in `upload`) must leave
/// resident gas (ρ, h) matching `density_adaptive` over the gas subset at the G2
/// f32-tolerance. This exercises the whole resident-density plumbing: the gas gather off
/// `bodies`, the gas-only grid build, and the GPU root-find.
#[test]
fn resident_gas_density_matches_cpu_oracle() {
    let mut state = gas_star_mix(0xA5F0, 120, 240);
    narrow_state(&mut state);
    let (_gas_idx, gpos, gmass) = gas_subset(&state);

    let dcfg = DensityConfig::default(); // n_ngb = 48, h_tol_rel = 1e-3
    let reference = density_adaptive(&gpos, &gmass, &dcfg, None);

    let mut stepper =
        GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, HydroParams::default(), dcfg)
            .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state); // primes a(x₀) = gravity(all) + hydro(gas); density runs here
    let gd = stepper.snapshot_gas_density();

    // The resident gas map must BE the gas subset, in ascending global index.
    assert_eq!(gd.gas_idx, _gas_idx, "resident gas map != gas subset");

    let h_err = max_rel_err(&gd.h, &reference.h);
    let rho_err = max_rel_err(&gd.rho, &reference.rho);
    // Measured on the Vulkan test adapter: h 7.5e-4, ρ 9.1e-4 — in line with the standalone
    // G2 gate (worst h 9e-4 / ρ 1.3e-3 vs this same oracle), confirming the resident path
    // (which fixes the bracket seed at upload, vs GpuDensity's fresh seed) tracks G2 because
    // the root is seed-independent. Gates are measure-then-tighten, with headroom for
    // cross-adapter f32 variation.
    assert!(h_err < 1e-3, "resident h rel err {h_err:e} exceeds 1e-3");
    assert!(
        rho_err < 1.3e-3,
        "resident ρ rel err {rho_err:e} exceeds 1.3e-3"
    );
}

/// The resident gas map must survive a re-upload of a DIFFERENT gas/star split without
/// carrying stale indices — a sharp check that `upload` rebuilds the map from
/// `state.kind` each time (not once).
#[test]
fn resident_gas_map_rebuilt_on_reupload() {
    let mut stepper = GpuResidentLeapfrog::new_with_sph(
        G,
        EPS,
        THETA,
        HydroParams::default(),
        DensityConfig::default(),
    )
    .expect("wgpu adapter required for GPU-SPH resident gates");

    let mut a = gas_star_mix(0x1111, 80, 160);
    narrow_state(&mut a);
    let (idx_a, _, _) = gas_subset(&a);
    stepper.upload(&a);
    assert_eq!(stepper.snapshot_gas_density().gas_idx, idx_a);

    let mut b = gas_star_mix(0x2222, 100, 100);
    narrow_state(&mut b);
    let (idx_b, _, _) = gas_subset(&b);
    stepper.upload(&b);
    assert_eq!(stepper.snapshot_gas_density().gas_idx, idx_b);
}
