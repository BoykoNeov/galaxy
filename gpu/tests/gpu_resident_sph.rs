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
use galaxy_solvers::sph::{
    density_adaptive, hydro_accelerations, max_stable_dt, DensityConfig, HashGrid, HydroParams,
    SUPPORT,
};

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

/// f32-narrow positions AND velocities the way the GPU does at upload (so the oracle
/// sees the same phase-space the device does — the gates are f32-tolerance, not
/// narrowing tests). G5a used only positions; the G5b viscosity term reads velocity, so
/// the velocity narrowing must match too (a no-op for the density-only G5a gates).
fn narrow_state(state: &mut State) {
    let narrow = |v: DVec3| DVec3::new(v.x as f32 as f64, v.y as f32 as f64, v.z as f32 as f64);
    for p in state.pos.iter_mut() {
        *p = narrow(*p);
    }
    for v in state.vel.iter_mut() {
        *v = narrow(*v);
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

// ===========================================================================
// GPU-SPH **G5b**: resident hydro force + scatter-add.
//
// G5a left resident gas (ρ, h) on the device each force evaluation. G5b consumes them:
// the symmetric-P/ρ² + Monaghan hydro force over the gas subset → a resident `gas_acc`,
// scatter-added into `accel`'s gas rows AFTER the gravity traverse (unique gas indices ⇒
// no race). The hydro WGSL is the G3 `sph_hydro` text VERBATIM (one source of truth);
// the resident-only pieces are the gas-velocity gather, the [mass,ρ,h] scalar pack, and
// the scatter kernel.
//
// ## The gates (mirroring G3, plus the two residency-specific probes)
//   1. `resident_hydro_accel_matches_cpu_oracle` — isolated hydro-force accuracy: the
//      resident `gas_acc` (read BEFORE the scatter) vs `hydro_accelerations` fed the SAME
//      GPU-computed (ρ, h) read back, so only the force f32 error is measured (density
//      error, already G2/G5a-gated, is factored out). VARYING gas mass (catches a
//      mass[i]/mass[j] swap); mixed velocities (viscosity branch live on both sides).
//   2. `resident_hydro_momentum_drift_is_roundoff` — the sharp antisymmetry detector on a
//      DEDICATED cloud: EQUAL gas mass (so each pair's two contributions are exact f32
//      negatives → net drift is reduction roundoff only) + spatially non-uniform (so
//      asymmetric-coupling pairs exist and the global-`SUPPORT·h_max` gather invariant is
//      actually tested). `gas_star_mix`'s VARYING mass would break this floor — hence a
//      separate cloud.
//   3. `resident_hydro_scatter_composition` — the scatter-add lands in the right rows:
//      the SAME cloud through a gravity-only stepper and a gas-mode stepper. Gravity is
//      bit-identical across the two (same positions/pipeline), so star rows must match and
//      gas rows must differ by EXACTLY the resident `gas_acc`. GPU-vs-GPU (no CPU-gravity
//      oracle), deterministic, chaos-free.
//   4. `resident_hydro_matches_cpu_over_stepped_run` — the caveat-2 staleness probe. The
//      G5a gate never steps, so per-step behaviour (density gathered off DRIFTED bodies;
//      the hydro grid cell + global radius FROZEN at upload's `h_max` vs evolved positions)
//      was assumed, not tested. Step a self-gravitating (contracting) blob, then compare
//      the resident (ρ, h, gas_acc) against the CPU oracle AT THE SNAPSHOT POSITIONS — a
//      deterministic function of position, so chaos-immune (D5). Contraction is the
//      frozen-grid SAFE direction (over-cover ⇒ every real pair captured, extras add 0);
//      an expanding blob would under-cover and is deliberately not built. Viscosity is off
//      so `gas_acc` is pure pressure (position + ρ/h determined), sidestepping the
//      v_{n+1/2}-vs-v_{n+1} half-kick velocity-timing mismatch (viscosity is validated in
//      gate 1, where no step is taken and velocities match exactly).
//
// Shock tube: NOT re-run here (advisor call, option b). The resident stepper always has
// gravity (FusedCore) whereas the shock tube is pure hydro, so a resident end-to-end run
// needs a gravity-off mode (deferred). The resident hydro force is transitively validated:
// resident `gas_acc` ≈ CPU `hydro_accelerations` (gate 1) and CPU hydro ≈ the analytic
// Riemann solution (the standalone CPU/`GpuHydro` shock tube, whose WGSL is byte-identical
// to the resident hydro text). Recorded in the plan.
// ===========================================================================

/// Deterministic pseudo-random `[0, 1)` stream (the LCG used across the GPU tests).
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// An all-gas, centrally-concentrated cloud (dense core + diffuse tail, radius `radius`):
/// every particle `Species::Gas`, so the adaptive `h` spans the measured wide range. Gas
/// mass and velocities are caller-supplied. Gravity acts on the whole cloud, but the
/// hydro gates read the pre-scatter `gas_acc` (pure hydro), so gravity is irrelevant to
/// them; the stepped gate DOES want the gravity (it drives the contraction).
fn gas_cloud(pos_seed: u64, mass: Vec<f64>, vel: Vec<DVec3>, radius: f64) -> State {
    let n = mass.len();
    assert_eq!(vel.len(), n);
    let mut next = rng(pos_seed);
    let pos: Vec<DVec3> = (0..n)
        .map(|_| {
            let r = radius * next();
            let z = 2.0 * next() - 1.0;
            let phi = std::f64::consts::TAU * next();
            let s = (1.0 - z * z).max(0.0).sqrt();
            DVec3::new(s * phi.cos(), s * phi.sin(), z) * r
        })
        .collect();
    let mut st = State::from_phase_space(pos, vel, mass);
    st.kind = vec![Species::Gas; n];
    st.assert_consistent();
    st
}

/// A mixed velocity field, each component uniform in `[−scale, scale]` — uncorrelated
/// with position, so a coupled pair approaches (`vr < 0`) or recedes with ~50/50 odds
/// (exercises both sides of the `vr ≥ 0 → Π = 0` viscosity branch).
fn random_velocities(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut next = rng(seed);
    (0..n)
        .map(|_| {
            DVec3::new(
                scale * (2.0 * next() - 1.0),
                scale * (2.0 * next() - 1.0),
                scale * (2.0 * next() - 1.0),
            )
        })
        .collect()
}

/// RMS acceleration over the set — the scale that normalizes relative errors so a
/// near-null particle can't blow up the metric (mirrors `sph_hydro` / `gpu_direct_sum`).
fn rms_accel(a: &[DVec3]) -> f64 {
    let n = a.len().max(1) as f64;
    (a.iter().map(|v| v.length_squared()).sum::<f64>() / n)
        .sqrt()
        .max(1e-300)
}

/// RMS of the per-particle errors, normalized by the RMS acceleration.
fn rms_rel_err(approx: &[DVec3], exact: &[DVec3]) -> f64 {
    assert_eq!(approx.len(), exact.len());
    let rms = rms_accel(exact);
    let n = exact.len().max(1) as f64;
    let err_ms = approx
        .iter()
        .zip(exact)
        .map(|(b, d)| (*b - *d).length_squared())
        .sum::<f64>()
        / n;
    err_ms.sqrt() / rms
}

/// Worst-case per-particle error, normalized by the RMS acceleration.
fn worst_rel_err(approx: &[DVec3], exact: &[DVec3]) -> f64 {
    let rms = rms_accel(exact);
    approx
        .iter()
        .zip(exact)
        .map(|(b, d)| (*b - *d).length() / rms)
        .fold(0.0_f64, f64::max)
}

/// (approaching, receding) counts over coupled pairs `0 < r < SUPPORT·max(h_i,h_j)`,
/// split by `vr = (v_i−v_j)·(x_i−x_j)`. O(N²), fine at test scale.
fn approach_split(pos: &[DVec3], vel: &[DVec3], h: &[f64]) -> (usize, usize) {
    let n = pos.len();
    let (mut approaching, mut receding) = (0usize, 0usize);
    for i in 0..n {
        for j in (i + 1)..n {
            let r_ij = pos[i] - pos[j];
            let r = r_ij.length();
            let coupling = SUPPORT * h[i].max(h[j]);
            if r > 0.0 && r < coupling {
                if (vel[i] - vel[j]).dot(r_ij) < 0.0 {
                    approaching += 1;
                } else {
                    receding += 1;
                }
            }
        }
    }
    (approaching, receding)
}

/// Count ASYMMETRIC-coupling pairs `SUPPORT·min(h_i,h_j) ≤ r < SUPPORT·max(h_i,h_j)` —
/// exactly where one particle's per-target radius reaches the other but not vice-versa.
/// A material count makes the momentum-drift gate a real test of the global-`h_max`
/// gather invariant (a per-target-radius bug drops one half of each such pair).
fn asymmetric_coupling_count(pos: &[DVec3], h: &[f64]) -> usize {
    let n = pos.len();
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let r = (pos[i] - pos[j]).length();
            let lo = SUPPORT * h[i].min(h[j]);
            let hi = SUPPORT * h[i].max(h[j]);
            if r >= lo && r < hi {
                count += 1;
            }
        }
    }
    count
}

/// Total momentum change rate `|Σ_i m_i a_i|`, normalized by `Σ_i m_i|a_i|` (the drift
/// metric — for equal mass and exact pairwise antisymmetry this is reduction roundoff).
fn momentum_drift(acc: &[DVec3], mass: &[f64]) -> f64 {
    let mut net = DVec3::ZERO;
    let mut scale = 0.0;
    for (a, &m) in acc.iter().zip(mass) {
        net += *a * m;
        scale += a.length() * m;
    }
    net.length() / scale.max(1e-300)
}

/// RMS distance of the gas particles from their centroid — the contraction diagnostic
/// for the stepped gate (the frozen grid is only tested if the cloud actually evolves).
fn gas_rms_radius(state: &State) -> f64 {
    let gas: Vec<DVec3> = (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .map(|i| state.pos[i])
        .collect();
    let n = gas.len().max(1) as f64;
    let c: DVec3 = gas.iter().copied().sum::<DVec3>() / n;
    (gas.iter().map(|p| (*p - c).length_squared()).sum::<f64>() / n).sqrt()
}

/// **G5b accuracy gate.** Resident `gas_acc` (read BEFORE the scatter) vs the CPU
/// `hydro_accelerations` oracle fed the SAME GPU-computed (ρ, h). Isolates the hydro
/// force's f32 error from the density root-find (already G2/G5a-gated). VARYING mass
/// catches a mass-index swap; a mixed velocity field keeps the viscosity branch live.
#[test]
fn resident_hydro_accel_matches_cpu_oracle() {
    let n = 3000;
    let mass: Vec<f64> = (0..n).map(|j| 1.0 + (j % 7) as f64).collect();
    let vel = random_velocities(0x5EED, n, 1.0);
    let mut state = gas_cloud(0xC0FFEE, mass, vel, 5.0);
    narrow_state(&mut state);
    let (_gas_idx, gpos, gmass) = gas_subset(&state);
    let gvel: Vec<DVec3> = _gas_idx.iter().map(|&i| state.vel[i]).collect();
    let params = HydroParams::default();
    let dcfg = DensityConfig::default();

    let mut stepper = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);
    let gd = stepper.snapshot_gas_density();
    let gpu_acc = stepper.snapshot_gas_accel();

    // Feed the CPU oracle the GPU's (ρ, h) so only the force f32 error is measured. The
    // mixed velocity field keeps the viscosity branch live on both sides of vr=0; that the
    // branch is materially exercised is guaranteed transitively — the resident hydro WGSL is
    // byte-identical to the standalone G3, whose `gpu_hydro_viscosity_is_exercised` asserts it.
    let rho: Vec<f64> = gd.rho.iter().map(|&x| x as f64).collect();
    let h: Vec<f64> = gd.h.iter().map(|&x| x as f64).collect();
    let cpu = hydro_accelerations(&gpos, &gvel, &gmass, &rho, &h, &params);

    assert_eq!(gpu_acc.len(), cpu.len());
    for (i, a) in gpu_acc.iter().enumerate() {
        assert!(a.is_finite(), "NaN/inf gas accel at {i}");
    }
    let rms = rms_rel_err(&gpu_acc, &cpu);
    let worst = worst_rel_err(&gpu_acc, &cpu);
    // Measured on the Vulkan test adapter: rms 1.6e-7, worst 3.9e-6 — in line with the
    // standalone G3 gate (rms 2.9e-7 / worst 1.2e-5), confirming the resident hydro force
    // (reusing the G3 WGSL verbatim) reproduces the CPU oracle to f32 roundoff. Bounds match
    // G3 (~600×/~250× headroom for cross-adapter f32 variation; a wrong kernel/coeff/index
    // bug is ≫1%).
    assert!(rms < 1.0e-4, "resident hydro RMS rel err {rms:.3e}");
    assert!(worst < 1.0e-3, "resident hydro worst rel err {worst:.3e}");
}

/// **G5b momentum gate.** On a dedicated EQUAL-mass, non-uniform cloud the resident
/// `gas_acc` must conserve momentum to reduction roundoff — the sharp detector for a
/// gather-radius leak / grad-sign / asymmetric-coeff bug in the resident hydro pass.
#[test]
fn resident_hydro_momentum_drift_is_roundoff() {
    let n = 2000;
    let mass = vec![1.0; n]; // EQUAL mass ⇒ exact pairwise f32 antisymmetry
    let vel = random_velocities(0x1CE, n, 1.0);
    let mut state = gas_cloud(0xBEEF, mass, vel, 5.0);
    narrow_state(&mut state);
    let (_gas_idx, gpos, gmass) = gas_subset(&state);
    let params = HydroParams::default();
    let dcfg = DensityConfig::default();

    // Guard: the cloud must contain a material fraction of asymmetric-coupling pairs, else
    // the global-`h_max` gather invariant isn't exercised (a per-target-radius bug would
    // pass trivially). Uses CPU h (≈ GPU h) for the geometry check.
    let dens = density_adaptive(&gpos, &gmass, &dcfg, None);
    let asym = asymmetric_coupling_count(&gpos, &dens.h);
    let coupled = {
        let (a, r) = approach_split(&gpos, &vel_of(&state, &_gas_idx), &dens.h);
        a + r
    };
    assert!(
        asym > coupled / 100,
        "cloud must have a material fraction of asymmetric-coupling pairs \
         (got {asym} of {coupled} coupled) for the drift gate to test the global radius"
    );

    let mut stepper = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);
    let gpu_acc = stepper.snapshot_gas_accel();

    let rel = momentum_drift(&gpu_acc, &gmass);
    // Exact per-pair f32 antisymmetry ⇒ only reduction roundoff survives: measured 1.6e-8
    // (standalone G3 saw 2.1e-9; the resident path's extra gather/pack stages shuffle the
    // reduction order but keep it roundoff-scale). Measured 29036 of 55462 coupled pairs
    // (52%) are asymmetric-coupling. A radius leak / grad-sign / asymmetric-coeff bug produces
    // O(1e-2–1) drift, so the 1e-5 bound (~600× over the floor) is sharp.
    assert!(rel < 1.0e-5, "resident hydro net momentum drift {rel:.3e}");
}

/// **G5b scatter gate.** The hydro force must be scatter-added into the GAS rows of
/// `accel` and nothing else. The same cloud is run through a gravity-only stepper and a
/// gas-mode stepper: gravity is bit-identical across the two (same positions/masses/
/// pipeline), so star rows must be unchanged and gas rows must differ by EXACTLY the
/// resident `gas_acc`. GPU-vs-GPU, deterministic, chaos-free.
#[test]
fn resident_hydro_scatter_composition() {
    let mut state = gas_star_mix(0x5CA7, 120, 240);
    narrow_state(&mut state);
    let (gas_idx, _, _) = gas_subset(&state);
    let params = HydroParams::default();
    let dcfg = DensityConfig::default();

    let mut grav = GpuResidentLeapfrog::new(G, EPS, THETA)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    grav.upload(&state);
    let g_only = grav.snapshot_accel();

    let mut gas = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    gas.upload(&state);
    let g_gas = gas.snapshot_accel();
    let gas_acc = gas.snapshot_gas_accel();

    assert_eq!(g_only.len(), state.len());
    assert_eq!(g_gas.len(), state.len());
    let mut kgas = 0usize;
    for i in 0..state.len() {
        if state.kind[i] == Species::Gas {
            let expected = g_only[i] + gas_acc[kgas];
            let err = (g_gas[i] - expected).length() / expected.length().max(1e-30);
            assert!(err < 1.0e-4, "gas row {i}: scatter-add mismatch {err:.3e}");
            kgas += 1;
        } else {
            // Star: gravity identical across steppers, no hydro added.
            let err = (g_gas[i] - g_only[i]).length() / g_only[i].length().max(1e-30);
            assert!(err < 1.0e-5, "star row {i} perturbed by gas mode {err:.3e}");
        }
    }
    assert_eq!(kgas, gas_idx.len(), "gas row count mismatch");
}

/// **G5b staleness gate (caveat 2).** Step a self-gravitating (contracting) blob, then
/// compare the resident (ρ, h, gas_acc) against the CPU oracle AT THE SNAPSHOT POSITIONS.
/// Deterministic in position ⇒ chaos-immune (D5). Contraction is the frozen-grid safe
/// direction; viscosity off so `gas_acc` is pure pressure (no velocity-timing subtlety).
#[test]
fn resident_hydro_matches_cpu_over_stepped_run() {
    let n = 1500;
    let mass = vec![1.0; n];
    let vel = vec![DVec3::ZERO; n]; // cold ⇒ self-gravity drives the contraction
    let mut state = gas_cloud(0xC017, mass, vel, 3.0);
    narrow_state(&mut state);
    let inviscid = HydroParams {
        alpha: 0.0,
        beta: 0.0,
        ..HydroParams::default()
    };
    let dcfg = DensityConfig::default();
    let r0 = gas_rms_radius(&state);

    let mut stepper = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, inviscid, dcfg.clone())
        .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);
    let dt = 0.002;
    for _ in 0..30 {
        stepper.step(dt);
    }
    // snapshot() rebuilds State via from_phase_space, which defaults every species to
    // Collisionless; the resident stepper preserves particle order, so restore the kinds
    // from the original state before any gas filtering. (G6's simulate branch will do the
    // same re-attach when it threads Species through the GPU path.)
    let mut evolved = stepper.snapshot();
    evolved.kind = state.kind.clone();
    let r1 = gas_rms_radius(&evolved);
    assert!(
        r1 < 0.98 * r0,
        "blob must contract to exercise frozen-grid staleness (r0={r0:.4}, r1={r1:.4})"
    );

    let gd = stepper.snapshot_gas_density();
    let gpu_acc = stepper.snapshot_gas_accel();
    let (gas_idx, gpos, gmass) = gas_subset(&evolved);
    let gvel: Vec<DVec3> = gas_idx.iter().map(|&i| evolved.vel[i]).collect();

    // CPU density at the EVOLVED positions (fresh grid) — the staleness reference.
    let dens = density_adaptive(&gpos, &gmass, &dcfg, None);
    let rho_err = max_rel_err(&gd.rho, &dens.rho);
    let h_err = max_rel_err(&gd.h, &dens.h);
    // Measured after a 21% contraction (r0 1.74 → r1 1.37): h 8.7e-4, ρ 1.4e-3 — barely above
    // the no-step G5a floor (h 7.5e-4 / ρ 9.1e-4), confirming contraction is the frozen-grid
    // SAFE direction (the frozen density cell over-covers, so per-step density stays accurate
    // off the drifted `bodies`). Bounds ~2× the measured values: tight enough that an
    // expansion-clip staleness blow-up (≫1e-2) fails hard, loose enough for cross-adapter
    // trajectory variation.
    assert!(h_err < 2.0e-3, "stepped resident h rel err {h_err:.3e}");
    assert!(rho_err < 3.0e-3, "stepped resident ρ rel err {rho_err:.3e}");

    // CPU hydro (inviscid) fed the GPU (ρ, h): isolates the frozen HYDRO-grid staleness.
    let rho: Vec<f64> = gd.rho.iter().map(|&x| x as f64).collect();
    let h: Vec<f64> = gd.h.iter().map(|&x| x as f64).collect();
    let cpu_acc = hydro_accelerations(&gpos, &gvel, &gmass, &rho, &h, &inviscid);
    let rms = rms_rel_err(&gpu_acc, &cpu_acc);
    // Measured 3.2e-7 (pure pressure ⇒ position + GPU-(ρ,h) determined): the frozen hydro grid
    // (cell = radius = SUPPORT·h_max at upload) stays accurate under contraction. Bound 1e-4
    // (~300× headroom), matching the no-step accuracy gate's style.
    assert!(rms < 1.0e-4, "stepped resident hydro RMS rel err {rms:.3e}");
}

/// Gas-subset velocities in ascending global index (helper for the momentum guard).
fn vel_of(state: &State, gas_idx: &[usize]) -> Vec<DVec3> {
    gas_idx.iter().map(|&i| state.vel[i]).collect()
}

// ===========================================================================
// GPU-SPH **G5c**: resident CFL stable-dt + on-device no-readback min reduction.
//
// G5c ports the G4 CFL per-target `dt_i = C_cfl·h_i/v_sig,i` onto the resident stepper —
// running on the gas (pos/vel/h) the last force evaluation left resident, WITHOUT leaving
// the device — and adds the one piece G4 deferred: the milestone bound `min_i dt_i` is
// reduced ON the GPU, so only that single scalar crosses the bus (not the per-gas vector
// G4 folded host-side). The CFL kernel WGSL is the G4 `sph_cfl` text VERBATIM (one source
// of truth); the resident-only piece is the min-reduction kernel. This is compute only —
// NO dt policy (a block-adaptive stepper writing `min_stable_dt` into the step uniform is
// the deferred D6/G6 follow-up).
//
// ## Why min-reduction is SAFE where the drift two-sum was not
// The M4j drift accumulator needed runtime-uniform XOR barriers because IEEE
// non-associativity let the f32 compiler collapse the compensated two-sum to zero. The
// CFL min has NO such trap: `min` merely SELECTS one of its inputs, so `min(min(a,b),c) ==
// min(a,min(b,c))` bit-for-bit for any tree order — and `v_sig ≥ 2c_s > 0`, `h > 0` mean
// `dt_i` is finite positive (no NaN to make `min` order-dependent). The tree reduction is
// therefore bit-deterministic with no barrier, and the on-GPU min equals a host fold of
// the same vector EXACTLY (gate `resident_min_dt_is_device_reduction_of_vector`).
//
// ## The gates (all NO-STEP — resident v = narrow(v₀) matches the oracle)
//   1. `resident_min_stable_dt_matches_cpu_oracle` — full chain: the device-reduced
//      `min_stable_dt` vs the trusted `max_stable_dt` oracle. Density-limited (the oracle
//      recomputes `h`), so ~1e-3 (the G2 `h` tol), not G4's isolated 1e-5.
//   2. `resident_gas_dt_matches_cpu_per_target` — the SHARP per-target vector, oracle fed
//      the GPU `h` (isolates CFL from density), with the asymmetric-approaching precondition
//      that makes the global-`SUPPORT·h_max` gather invariant observable (a per-target-`h_i`
//      radius bug on ANY particle fails it — the scalar min would only move if the minimizer
//      is hit).
//   3. `resident_min_dt_is_device_reduction_of_vector` — the no-readback proof: the
//      scalar path (which copies back ONLY the 1-element reduced buffer, never the vector)
//      equals the host fold of the vector path BIT-FOR-BIT. Because `min_stable_dt` never
//      transfers the N values, it structurally CANNOT have folded them host-side.
//   4. `resident_cfl_is_deterministic` — same input ⇒ identical vector and scalar.
//   5. `resident_min_stable_dt_no_gas_is_infinity` — the catastrophic-`0` contract: a
//      gas-mode stepper over an all-star state ⇒ `+∞` (no hydro CFL constraint), NOT 0.
// ===========================================================================

/// CFL number ≠ 1 so a "forgot ×C_cfl" bug can't hide (it factors out of the ratio) — the
/// same choice as the standalone G4 gate.
const C_CFL: f64 = 0.25;

/// Per-target signal velocity `v_sig,i = max_j (2c_s − 3 w_ij)` over approaching neighbors,
/// floored at `2c_s` — an independent hand-transcription of the oracle's inner loop
/// (`galaxy_solvers::sph::cfl::max_stable_dt`), gathering at the GLOBAL `SUPPORT·h_max` and
/// rejecting each pair outside its own `SUPPORT·max(h_i,h_j)`. Copied from the G4 gate.
fn per_target_v_sig(pos: &[DVec3], vel: &[DVec3], h: &[f64], cs: f64) -> Vec<f64> {
    let n = pos.len();
    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    let grid = HashGrid::build(pos, SUPPORT * h_max);
    let two_cs = 2.0 * cs;
    (0..n)
        .map(|i| {
            let ngb = grid.neighbours_within(pos, pos[i], SUPPORT * h_max);
            let mut v_sig = two_cs;
            for &j in &ngb {
                if j == i {
                    continue;
                }
                let r_ij = pos[i] - pos[j];
                let r = r_ij.length();
                if r == 0.0 || r >= SUPPORT * h[i].max(h[j]) {
                    continue;
                }
                let w = (vel[i] - vel[j]).dot(r_ij) / r;
                if w < 0.0 {
                    v_sig = v_sig.max(two_cs - 3.0 * w);
                }
            }
            v_sig
        })
        .collect()
}

/// Per-target stable step `dt_i = C_cfl · h_i / v_sig,i` — the vector the resident device
/// must match element-wise (its min is `max_stable_dt`).
fn cpu_per_target_dt(pos: &[DVec3], vel: &[DVec3], h: &[f64], cs: f64, c_cfl: f64) -> Vec<f64> {
    let v_sig = per_target_v_sig(pos, vel, h, cs);
    (0..pos.len()).map(|i| c_cfl * h[i] / v_sig[i]).collect()
}

/// Worst per-element relative error. `dt_i > 0` bounded away from zero (`v_sig ≥ 2c_s`,
/// `h > 0`), so a plain per-element relative error is clean (no RMS normalization needed).
fn worst_rel_dt(gpu: &[f64], cpu: &[f64]) -> f64 {
    assert_eq!(gpu.len(), cpu.len());
    gpu.iter()
        .zip(cpu)
        .map(|(g, c)| (g - c).abs() / c.abs())
        .fold(0.0_f64, f64::max)
}

/// Count ASYMMETRIC-coupling pairs that are ALSO APPROACHING:
/// `SUPPORT·min(h_i,h_j) ≤ r < SUPPORT·max(h_i,h_j)` and `w_ij < 0`. These make a
/// per-target-radius bug OBSERVABLE in the vector gate (the diffuse particle's support
/// reaches the compact one but not vice-versa, AND the approach drives `v_sig`). Copied
/// from the G4 gate.
fn asymmetric_approaching_count(pos: &[DVec3], vel: &[DVec3], h: &[f64]) -> usize {
    let n = pos.len();
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let r_ij = pos[i] - pos[j];
            let r = r_ij.length();
            let lo = SUPPORT * h[i].min(h[j]);
            let hi = SUPPORT * h[i].max(h[j]);
            if r >= lo && r < hi && (vel[i] - vel[j]).dot(r_ij) < 0.0 {
                count += 1;
            }
        }
    }
    count
}

/// **G5c full-chain gate.** The device-reduced `min_stable_dt` must match the trusted CPU
/// `max_stable_dt` oracle on an all-gas cloud with a mixed velocity field. The oracle
/// recomputes `h` internally, so this is density-limited (the G2 `h` tolerance), not G4's
/// isolated per-target tolerance — it validates the WHOLE resident chain (gather → density
/// root-find → CFL per-target → on-device min) end-to-end against the trusted number.
#[test]
fn resident_min_stable_dt_matches_cpu_oracle() {
    let n = 3000;
    let mass = vec![1.0; n];
    let vel = random_velocities(0x5EED, n, 1.0);
    let mut state = gas_cloud(0xC0FFEE, mass, vel, 5.0);
    narrow_state(&mut state);
    let params = HydroParams::default();
    let dcfg = DensityConfig::default();

    // Precondition: the minimizer must be driven above the 2c_s floor, else max_stable_dt is
    // just C_cfl·h_min/(2c_s) and a broken approach loop would be invisible to the scalar gate.
    let (_gas_idx, gpos, gmass) = gas_subset(&state);
    let gvel: Vec<DVec3> = _gas_idx.iter().map(|&i| state.vel[i]).collect();
    let dens = density_adaptive(&gpos, &gmass, &dcfg, None);
    let cs = params.sound_speed;
    let v_sig = per_target_v_sig(&gpos, &gvel, &dens.h, cs);
    let dt = cpu_per_target_dt(&gpos, &gvel, &dens.h, cs, C_CFL);
    let argmin = (0..dt.len())
        .min_by(|&a, &b| dt[a].total_cmp(&dt[b]))
        .expect("non-empty");
    assert!(
        v_sig[argmin] > 2.0 * cs,
        "minimizer {argmin} sits at the v_sig floor — pick a livelier field"
    );

    let cpu = max_stable_dt(&state, &params, &dcfg, C_CFL);

    let mut stepper = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);
    let gpu = stepper.min_stable_dt(C_CFL);

    assert!(
        gpu.is_finite() && gpu > 0.0,
        "resident min_stable_dt must be finite positive: {gpu}"
    );
    let rel = (gpu - cpu).abs() / cpu;
    // Density-limited: the oracle recomputes h from positions while the device uses its own
    // root-find h, so the bound tracks the G2 h tolerance (~1e-3), not G4's isolated 1e-5.
    // Measure-then-tighten with cross-adapter headroom.
    assert!(
        rel < 2.0e-3,
        "resident min_stable_dt {gpu} vs oracle {cpu} (rel {rel:.3e})"
    );
}

/// **G5c sharp per-target gate.** The resident per-gas `dt_i` vector vs the CPU per-target
/// oracle fed the SAME GPU-computed `h` (isolates the CFL kernel from the density root-find,
/// already G2/G5a-gated). Fails on a per-target radius bug on ANY particle — not only the
/// minimizer. The precondition asserts the cloud actually contains asymmetric-coupling
/// approaching pairs, where that bug is observable.
#[test]
fn resident_gas_dt_matches_cpu_per_target() {
    let n = 3000;
    let mass = vec![1.0; n];
    let vel = random_velocities(0x5EED, n, 1.0);
    let mut state = gas_cloud(0xC0FFEE, mass, vel, 5.0);
    narrow_state(&mut state);
    let params = HydroParams::default();
    let dcfg = DensityConfig::default();
    let (_gas_idx, gpos, _gmass) = gas_subset(&state);
    let gvel: Vec<DVec3> = _gas_idx.iter().map(|&i| state.vel[i]).collect();

    let mut stepper = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);
    // GPU h isolates the CFL kernel; GPU dt is the vector under test (SAME resident h).
    let gd = stepper.snapshot_gas_density();
    let gc = stepper.snapshot_gas_dt(C_CFL);
    let h: Vec<f64> = gd.h.iter().map(|&x| x as f64).collect();

    assert_eq!(gc.gas_idx, _gas_idx, "resident gas map != gas subset");

    // Precondition (on the GPU h, the same geometry the device sees): asymmetric-coupling
    // APPROACHING pairs must exist, else a per-target-h_i gather bug is invisible.
    let asym = asymmetric_approaching_count(&gpos, &gvel, &h);
    assert!(
        asym > 100,
        "cloud/field must contain asymmetric-coupling APPROACHING pairs (got {asym})"
    );

    let cpu = cpu_per_target_dt(&gpos, &gvel, &h, params.sound_speed, C_CFL);
    for (i, &d) in gc.dt.iter().enumerate() {
        assert!(
            d.is_finite() && d > 0.0,
            "dt_i must be finite positive at {i}: {d}"
        );
    }
    let worst = worst_rel_dt(&gc.dt, &cpu);
    // Isolated (GPU h fed in): dt_i is one max + one divide, no accumulation ⇒ tight f32,
    // matching G4's per-target gate (worst ≈ 1e-6). A real bug (coupling cutoff, per-target
    // radius, or w/r² instead of w/r) is ≫1%; the 1e-5 bound fails hard on it.
    assert!(
        worst < 1.0e-5,
        "resident CFL per-target worst rel err {worst:.3e}"
    );
}

/// **G5c no-readback proof.** The device-reduced scalar `min_stable_dt` (which copies back
/// ONLY the 1-element min buffer — it never transfers the per-gas vector) must equal the
/// host fold of the vector `snapshot_gas_dt` returns, BIT-FOR-BIT. `min` selects one input
/// and f32→f64 promotion is exact, so `==` is legitimate; because the scalar path never sees
/// the N values, it structurally cannot have folded them host-side — this is what pins the
/// reduction ONTO the device. (Ties two submits via determinism, established independently
/// by `resident_cfl_is_deterministic`.)
#[test]
fn resident_min_dt_is_device_reduction_of_vector() {
    let n = 2000;
    let mass = vec![1.0; n];
    let vel = random_velocities(0x1CE, n, 1.0);
    let mut state = gas_cloud(0xBEEF, mass, vel, 5.0);
    narrow_state(&mut state);

    let mut stepper = GpuResidentLeapfrog::new_with_sph(
        G,
        EPS,
        THETA,
        HydroParams::default(),
        DensityConfig::default(),
    )
    .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);

    let gc = stepper.snapshot_gas_dt(C_CFL);
    let device_min = stepper.min_stable_dt(C_CFL);
    let host_min = gc.dt.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    assert_eq!(
        device_min, host_min,
        "device-reduced min must equal the host fold of the per-gas vector bit-for-bit"
    );
}

/// **G5c determinism gate.** Same input ⇒ identical per-gas vector AND scalar min on a
/// given device (each CFL thread owns its own `dt_out` slot; the min tree is
/// order-independent — no scatter race, no reassociation).
#[test]
fn resident_cfl_is_deterministic() {
    let n = 1500;
    let mass = vec![1.0; n];
    let vel = random_velocities(0xFEED, n, 1.0);
    let mut state = gas_cloud(0xD1CE, mass, vel, 4.0);
    narrow_state(&mut state);
    let params = HydroParams::default();
    let dcfg = DensityConfig::default();

    let mut a = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg.clone())
        .expect("wgpu adapter required for GPU-SPH resident gates");
    a.upload(&state);
    let av = a.snapshot_gas_dt(C_CFL);
    let am = a.min_stable_dt(C_CFL);

    let mut b = GpuResidentLeapfrog::new_with_sph(G, EPS, THETA, params, dcfg)
        .expect("wgpu adapter required for GPU-SPH resident gates");
    b.upload(&state);
    let bv = b.snapshot_gas_dt(C_CFL);
    let bm = b.min_stable_dt(C_CFL);

    assert_eq!(
        av.dt, bv.dt,
        "per-target CFL dt must be run-to-run identical"
    );
    assert_eq!(am, bm, "reduced min dt must be run-to-run identical");
}

/// **G5c catastrophic-`0` contract.** A gas-mode stepper over an all-star (no gas) state
/// has no hydro CFL constraint, so `min_stable_dt` must be `+∞`, NOT `0` (a `0` would
/// falsely report that every dt is too large). Mirrors the standalone G4 empty gate.
#[test]
fn resident_min_stable_dt_no_gas_is_infinity() {
    // All Collisionless (from_phase_space's default kind) ⇒ empty gas map even in gas mode.
    let mut next = rng(0x57A5);
    let n = 200;
    let pos: Vec<DVec3> = (0..n)
        .map(|_| DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 4.0)
        .collect();
    let vel = random_velocities(0xA11, n, 0.1);
    let mut state = State::from_phase_space(pos, vel, vec![1.0; n]);
    narrow_state(&mut state);
    assert!(
        state.kind.iter().all(|&k| k != Species::Gas),
        "this gate needs a gas-free state"
    );

    let mut stepper = GpuResidentLeapfrog::new_with_sph(
        G,
        EPS,
        THETA,
        HydroParams::default(),
        DensityConfig::default(),
    )
    .expect("wgpu adapter required for GPU-SPH resident gates");
    stepper.upload(&state);
    assert_eq!(
        stepper.min_stable_dt(C_CFL),
        f64::INFINITY,
        "no gas ⇒ +∞ (a 0 would falsely say every dt is too large)"
    );
}
