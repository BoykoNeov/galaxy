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
    density_adaptive, hydro_accelerations, DensityConfig, HydroParams, SUPPORT,
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

    // Feed the CPU oracle the GPU's (ρ, h) so only the force f32 error is measured.
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
