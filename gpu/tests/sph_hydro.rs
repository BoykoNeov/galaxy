//! GPU-SPH G3 — the GPU isothermal hydro force ([`GpuHydro`]) validated against the
//! CPU oracle [`galaxy_solvers::sph::hydro_accelerations`].
//!
//! ## What is (and is not) gated (D1/D5)
//! GPU forces are f32 (D1), so this is an f32-TOLERANCE gate, never bit-exact: the
//! GPU gathers per target in grid-walk order, the CPU in ascending index, so the sums
//! differ at the ulp by construction. Both force paths are fed ONE CPU
//! [`density_adaptive`] `(ρ, h)` so the force computation is isolated from the density
//! root-find (already gated in G2 — `h` is an INPUT here, bit-identical to both paths,
//! so G3 does NOT inherit G2's all-rooted precondition; a clamp divergence cannot
//! occur). The house force metrics (`rms_rel_err` / `worst_rel_err`, normalized by the
//! RMS acceleration) mirror `gpu_direct_sum`.
//!
//! ## The three things the gates must actually exercise
//!   1. **Gather radius = global `SUPPORT·h_max`, never per-target `SUPPORT·h_i`.**
//!      The averaged kernel `W̄ = ½(W(h_i)+W(h_j))` is nonzero for `r < 2·max(h_i,h_j)`,
//!      so a pair with `2h_i < r < 2h_j` gives force to BOTH i and j; a per-target
//!      radius would give i's force to j but not j's to i, breaking Newton's third law.
//!      `gpu_hydro_momentum_drift_bounded` is the sharp detector: per-pair antisymmetry
//!      is EXACT in f32 (`grad_w(−r)=−grad_w(r)` exactly; `coeff` commutative-equal;
//!      equal mass closes it), so the net drift is reduction roundoff ONLY. An O(1)
//!      drift means a radius leak / grad-sign / asymmetric-coeff bug, not roundoff.
//!   2. **The viscosity branch (`vr ≥ 0 → Π = 0`) on BOTH sides of `vr = 0`.** A pure
//!      converging field makes every pair approach, so a "compute Π for all pairs" GPU
//!      bug would hide (both paths compute-for-all and agree). The accuracy gate uses
//!      a MIXED (random) field; `gpu_hydro_viscosity_is_exercised` asserts the
//!      approach/recede SPLIT is material (the analog of the G2 uniform-mass gap).
//!   3. **The `m_j` factor with the RIGHT index.** The accuracy gate uses VARYING mass
//!      so a `mass[i]`-vs-`mass[j]` swap is caught (equal mass would hide it). The
//!      momentum gate uses EQUAL mass so the exact-antisymmetry floor holds.
//!
//! GPU-gated: these need a wgpu adapter; without one `GpuHydro::new` returns
//! `NoAdapter` and the tests fail loudly (the M3/M4 GPU-invariants convention).

use galaxy_core::DVec3;
use galaxy_gpu::GpuHydro;
use galaxy_solvers::sph::{
    density_adaptive, hydro_accelerations, DensityConfig, HydroParams, SUPPORT,
};

fn hydro() -> GpuHydro {
    GpuHydro::new().expect("wgpu adapter required for GPU SPH hydro tests")
}

/// Deterministic pseudo-random `[0, 1)` stream (the same LCG as the other GPU tests).
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Centrally-concentrated, origin-centered cloud (radius `R·u`, u uniform): a dense
/// core and diffuse tail so the adaptive `h` spans a wide range (the measured gas-disk
/// regime), while `h_min` stays large enough that f32 is clean. Same generator as the
/// G2 density gates.
fn concentrated_cloud(seed: u64, n: usize, radius: f64) -> Vec<DVec3> {
    let mut next = rng(seed);
    (0..n)
        .map(|_| {
            let r = radius * next();
            let z = 2.0 * next() - 1.0;
            let phi = std::f64::consts::TAU * next();
            let s = (1.0 - z * z).max(0.0).sqrt();
            DVec3::new(s * phi.cos(), s * phi.sin(), z) * r
        })
        .collect()
}

/// A MIXED velocity field: each component uniform in `[−scale, scale]`. Uncorrelated
/// with position, so a neighbor pair approaches (`vr < 0`) or recedes (`vr ≥ 0`) with
/// ~50/50 odds — this is what exercises BOTH sides of the viscosity branch condition
/// (a converging field would make every pair approach and hide a branch bug).
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

/// RMS acceleration over the system — the scale that normalizes relative errors so a
/// particle near a force null does not blow up the metric (mirrors `gpu_direct_sum`).
fn rms_accel(a: &[DVec3]) -> f64 {
    let n = a.len() as f64;
    (a.iter().map(|v| v.length_squared()).sum::<f64>() / n)
        .sqrt()
        .max(1e-300)
}

/// RMS of the per-particle errors, normalized by the RMS acceleration.
fn rms_rel_err(approx: &[DVec3], exact: &[DVec3]) -> f64 {
    let rms = rms_accel(exact);
    let n = exact.len() as f64;
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

/// Count coupled neighbor pairs (`0 < r < SUPPORT·max(h_i,h_j)`, the averaged-kernel
/// range) split by approach sign `vr = (v_i−v_j)·(x_i−x_j)`: (approaching, receding).
/// O(N²) — fine at test scale, and the honest definition of "coupled" (the same range
/// the force couples over).
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

/// Count ASYMMETRIC-coupling pairs: `SUPPORT·min(h_i,h_j) ≤ r < SUPPORT·max(h_i,h_j)`.
/// These are exactly the pairs where one particle's per-target radius would reach the
/// other but not vice-versa — so they make the momentum-drift gate a real test of the
/// "gather at the GLOBAL h_max" invariant (a per-target-radius bug would drop one half
/// of each such pair and blow up the drift). A cloud with none of these would keep the
/// drift at roundoff no matter what — the gate would pass while testing nothing.
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

// ---------------------------------------------------------------------------
// CPU-only precondition guard (advisor refinement 1): the viscosity branch is
// exercised on BOTH sides of vr = 0, and it materially changes the force.
// ---------------------------------------------------------------------------

/// The accuracy gate is only a real viscosity test if the field puts a MATERIAL
/// fraction of coupled pairs on each side of `vr = 0` (else the `vr ≥ 0 → Π = 0`
/// branch is dead and a "compute Π for all pairs" bug hides) AND viscosity materially
/// changes the acceleration (else `Π ≈ 0` and the whole term is untested). CPU-only,
/// so green from the red commit on — a standing guard on the test's velocity field.
#[test]
fn gpu_hydro_viscosity_is_exercised() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 1500, 5.0);
    let vel = random_velocities(0x5EED, pos.len(), 1.0);
    let mass = vec![1.0; pos.len()];
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    let (approaching, receding) = approach_split(&pos, &vel, &dens.h);
    let total = (approaching + receding) as f64;
    assert!(
        total > 0.0,
        "no coupled pairs — cloud/velocity field is degenerate"
    );
    let approach_frac = approaching as f64 / total;
    let recede_frac = receding as f64 / total;
    assert!(
        approach_frac > 0.2 && recede_frac > 0.2,
        "viscosity branch must be exercised on BOTH sides: approaching {approach_frac:.2}, \
         receding {recede_frac:.2} of {total} coupled pairs"
    );

    // Viscosity ON vs OFF (α=β=0) must differ materially — else Π contributes nothing
    // and the accuracy gate is silently a pressure-only test.
    let visc = HydroParams::default();
    let inviscid = HydroParams {
        alpha: 0.0,
        beta: 0.0,
        ..HydroParams::default()
    };
    let u = vec![0.0; pos.len()];
    let a_visc = hydro_accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &visc);
    let a_inv = hydro_accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &inviscid);
    let rel = rms_rel_err(&a_inv, &a_visc); // ‖Δ‖_rms / rms(a_visc)
    assert!(
        rel > 0.05,
        "viscosity must change the force materially (rms rel diff {rel:.3e} > 0.05)"
    );
}

// ---------------------------------------------------------------------------
// main accuracy gate: GPU hydro accel vs hydro_accelerations (VARYING mass).
// ---------------------------------------------------------------------------

/// GPU `accelerations` vs the CPU `hydro_accelerations` oracle on the concentrated
/// cloud with a mixed velocity field. VARYING mass (`1 + j%7`) so a `mass[i]`-vs-
/// `mass[j]` index bug in the pair term is caught (equal mass would hide it); the same
/// `(ρ, h)` from `density_adaptive` feeds both paths. f32-tolerance, house metric.
#[test]
fn gpu_hydro_matches_cpu() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 3000, 5.0);
    let vel = random_velocities(0x5EED, pos.len(), 1.0);
    let mass: Vec<f64> = (0..pos.len()).map(|j| 1.0 + (j % 7) as f64).collect();
    let params = HydroParams::default();
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    let u = vec![0.0; pos.len()];
    let cpu = hydro_accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &u, &params);
    let gpu = hydro().accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &params);

    assert_eq!(gpu.len(), cpu.len());
    for (i, a) in gpu.iter().enumerate() {
        assert!(a.is_finite(), "NaN/inf acceleration at particle {i}");
    }
    let rms = rms_rel_err(&gpu, &cpu);
    let worst = worst_rel_err(&gpu, &cpu);
    // Pure f32 summation over the same neighbor set with no catastrophic cancellation
    // (moderate coordinates, smooth kernel) → measured rms ≈ 2.9e-7, worst ≈ 1.2e-5.
    // Bounds sit ~300×/~80× above the measured values: ample cross-device f32 headroom
    // while still failing hard on any wrong kernel/coeff/index bug (those are ≫1%).
    assert!(rms < 1.0e-4, "GPU hydro RMS rel err {rms:.3e}");
    assert!(worst < 1.0e-3, "GPU hydro worst rel err {worst:.3e}");
}

// ---------------------------------------------------------------------------
// momentum drift (sharp antisymmetry detector, EQUAL mass).
// ---------------------------------------------------------------------------

/// Total momentum change rate `Σ_i m_i a_i` must vanish to reduction roundoff. With
/// EQUAL mass, each pair's two contributions are EXACT f32 negatives (`grad_w(−r) =
/// −grad_w(r)` exactly; `coeff` commutative-equal), so the only residual is the order
/// in which the per-thread partial sums combine. An O(1) drift means the gather radius
/// leaked to per-target (i sees j but not vice-versa), a grad sign flip, or an
/// asymmetric `coeff` — the bugs this gate exists to catch.
#[test]
fn gpu_hydro_momentum_drift_bounded() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xBEEF, 2000, 5.0);
    let vel = random_velocities(0x1CE, pos.len(), 1.0);
    let mass = vec![1.0; pos.len()];
    let params = HydroParams::default();
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    // Precondition (advisor): the drift only tests the global-h_max invariant if the
    // cloud HAS asymmetric-coupling pairs (one particle's radius reaches the other but
    // not vice-versa). Assert a material count so a future narrow-h cloud change can't
    // silently reduce this to a roundoff-passes-trivially gate.
    let asym = asymmetric_coupling_count(&pos, &dens.h);
    let coupled = {
        let (a, r) = approach_split(&pos, &vel, &dens.h);
        a + r
    };
    // Measured: ≈29k of ≈55k coupled pairs (52%) are asymmetric on this wide-h cloud,
    // so >1% is a comfortable, robust floor that still fails if a future narrow-h cloud
    // change drops them to ~zero (which would gut this detector).
    assert!(
        asym > coupled / 100,
        "cloud must contain a material fraction of asymmetric-coupling pairs \
         (got {asym} of {coupled} coupled) for the drift gate to test the global radius"
    );

    let gpu = hydro().accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &params);

    let mut net = DVec3::ZERO;
    let mut scale = 0.0;
    for (a, &m) in gpu.iter().zip(&mass) {
        net += *a * m;
        scale += a.length() * m;
    }
    let rel = net.length() / scale.max(1e-300);
    // Exact per-pair f32 antisymmetry ⇒ only reduction roundoff survives: measured
    // ≈ 2.1e-9. A broken antisymmetry (radius leak / grad sign / asymmetric coeff)
    // produces O(1e-2–1) drift — many orders above this bound, which sits ~5000× over
    // the measured roundoff floor with cross-device margin.
    assert!(rel < 1.0e-5, "GPU hydro net momentum drift {rel:.3e}");
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run) & edge cases.
// ---------------------------------------------------------------------------

/// Same input ⇒ bit-identical acceleration on a given device (each thread owns its own
/// output slot; the walk is a gather, no scatter race — same discipline as G1/G2).
#[test]
fn gpu_hydro_deterministic() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xD1CE, 1500, 4.0);
    let vel = random_velocities(0xFEED, pos.len(), 1.0);
    let mass = vec![1.0; pos.len()];
    let params = HydroParams::default();
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    let mut g = hydro();
    let a = g.accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &params);
    let b = g.accelerations(&pos, &vel, &mass, &dens.rho, &dens.h, &params);
    assert_eq!(a, b, "acceleration must be run-to-run identical");
}

/// Empty input ⇒ empty output, no panic.
#[test]
fn gpu_hydro_empty() {
    let a = hydro().accelerations(&[], &[], &[], &[], &[], &HydroParams::default());
    assert!(a.is_empty());
}

/// A lone particle has no neighbors, and the self term is skipped (`grad_w(0) = 0`
/// anyway), so its force is exactly zero — the G2 single-particle edge parity, and a
/// clean check that the walk finds nothing to add for an isolated body.
#[test]
fn gpu_hydro_single_particle_zero() {
    let pos = [DVec3::new(0.5, -0.5, 2.0)];
    let vel = [DVec3::new(0.1, 0.2, -0.3)];
    let mass = [1.0];
    let rho = [1.0];
    let h = [0.3];
    let a = hydro().accelerations(&pos, &vel, &mass, &rho, &h, &HydroParams::default());
    assert_eq!(a, vec![DVec3::ZERO], "a lone particle feels no hydro force");
}
