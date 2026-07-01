//! `GpuDirectSum` validated against the CPU f64 `DirectSum` oracle.
//!
//! The GPU kernel runs in **f32** (wgpu/naga has no portable f64 compute — see the
//! crate docs), so — unlike the θ→0 Barnes-Hut gate that reproduces the oracle to
//! f64 roundoff (1e-9) — these gates are pinned at **f32-precision** tolerances,
//! *derived analytically* from the accumulation, not fit to the observed output.
//!
//! Tolerance model (mirrors the BH test's "bound from the method's order"):
//!   * ε_f32 ≈ 1.19e-7 (machine epsilon, 2⁻²³).
//!   * Summing N random-sign terms into one f32 accumulator → error grows like
//!     √N · ε_f32 (random walk); the worst case is N · ε_f32.
//!   * Close pairs (small r vs the coordinate scale) add catastrophic cancellation
//!     in `xᵢ − xⱼ`: relative error ≈ (coordinate magnitude / separation) · ε_f32.
//!
//! So the **unit-box** gate (coords O(1), √128·ε ≈ 1.4e-6 baseline, ×~100 close-pair
//! conditioning) is bounded well under 3e-4 RMS; the **large-coordinate** gate
//! (offset 5000, cancellation factor ~5000·ε ≈ 6e-4) is deliberately looser AND is
//! asserted to be *worse* than the unit-box case — proving the f32 cancellation the
//! GPU path must own is real (the analogue of "BH error grows with θ").
//!
//! GPU-gated: these need a wgpu adapter. On a box without one, `GpuDirectSum::new`
//! returns `NoAdapter` and these tests fail loudly (by design — matches the M3
//! render-invariants convention).

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_gpu::GpuDirectSum;
use galaxy_solvers::DirectSum;

const G: f64 = 1.0;
const EPS: f64 = 0.05;

/// f32 machine epsilon (2⁻²³) — the unit that scales every tolerance below.
const EPS_F32: f64 = 1.192_092_9e-7;

fn gpu(g: f64, eps: f64) -> GpuDirectSum {
    GpuDirectSum::new(g, eps).expect("wgpu adapter required for GPU solver tests")
}

/// Deterministic pseudo-random cluster (same LCG as the Barnes-Hut oracle test),
/// optionally rigidly offset by `shift` so separations are preserved but the
/// coordinate magnitude (hence f32 cancellation) is controlled.
fn cluster(seed: u64, n: usize, shift: DVec3) -> State {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) // in [0, 1)
    };
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    for _ in 0..n {
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 3.0 + shift);
        vel.push(DVec3::ZERO);
        mass.push(0.1 + 0.9 * next());
    }
    State::from_phase_space(pos, vel, mass)
}

fn accel(solver: &mut dyn ForceSolver, s: &State) -> Vec<DVec3> {
    let mut a = vec![DVec3::ZERO; s.len()];
    solver.accelerations(s, &mut a);
    a
}

/// RMS acceleration over the system — the scale that normalizes relative errors so a
/// particle near a force null does not blow up the metric.
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

/// The unit-box GPU forces reproduce the f64 oracle to f32 precision. RMS is the
/// method-meaningful bound (√N·ε with close-pair conditioning); worst-case carries a
/// loose gross-error guard (a mis-indexed particle would blow both up by orders).
#[test]
fn gpu_matches_oracle_unit_box() {
    const N: usize = 128;
    let mut solver = gpu(G, EPS);
    for seed in 0..64u64 {
        let s = cluster(seed, N, DVec3::ZERO);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);
        let got = accel(&mut solver, &s);

        let rms = rms_rel_err(&got, &exact);
        let worst = worst_rel_err(&got, &exact);
        // √128·ε ≈ 1.4e-6 baseline, ×~100 close-pair conditioning ⇒ ≪ 3e-4.
        assert!(rms < 3e-4, "unit-box RMS rel err {rms:e} (seed {seed})");
        assert!(worst < 5e-2, "unit-box worst rel err {worst:e} (seed {seed})");
    }
}

/// Rigidly translating the cluster far from the origin (positions ~5000, separations
/// unchanged) leaves the *physics* identical — the f64 oracle is translation-invariant
/// to roundoff — but forces the GPU's f32 `xᵢ − xⱼ` to lose bits. The GPU still tracks
/// the oracle within a looser, analytically-derived bound, AND its error is strictly
/// worse than the unit-box case: proof the f32 cancellation is real and coordinate-scale
/// driven (the analogue of BH error growing with θ).
#[test]
fn gpu_large_coordinate_cancellation_is_real_and_bounded() {
    const N: usize = 128;
    const SHIFT: f64 = 5000.0;
    let mut solver = gpu(G, EPS);
    for seed in 0..32u64 {
        let s_near = cluster(seed, N, DVec3::ZERO);
        let s_far = cluster(seed, N, DVec3::splat(SHIFT));

        let exact_near = accel(&mut DirectSum::new(G, EPS), &s_near);
        let exact_far = accel(&mut DirectSum::new(G, EPS), &s_far);
        let got_far = accel(&mut solver, &s_far);
        let got_near = accel(&mut solver, &s_near);

        let rms_far = rms_rel_err(&got_far, &exact_far);
        let rms_near = rms_rel_err(&got_near, &exact_near);

        // Cancellation factor ≈ SHIFT·ε ≈ 6e-4 ⇒ bounded well under 5e-3.
        assert!(rms_far < 5e-3, "large-coord RMS rel err {rms_far:e} (seed {seed})");
        // The whole point of the GPU path's precision caveat: coordinate magnitude
        // degrades f32. Far must be materially worse than near.
        assert!(
            rms_far > rms_near && rms_far > 1e-4,
            "large-coord error must exceed unit-box (seed {seed}): near {rms_near:e} -> far {rms_far:e}"
        );
    }
}

/// The gather kernel writes each `acc[i]` from a single private accumulator with a
/// fixed loop order, so repeated dispatches on the SAME device are bit-identical
/// (no float `atomicAdd` reassociation). Cross-device equality is NOT claimed.
#[test]
fn gpu_is_bit_deterministic_same_device() {
    const N: usize = 300;
    let mut solver = gpu(G, EPS);
    let s = cluster(7, N, DVec3::ZERO);
    let a1 = accel(&mut solver, &s);
    let a2 = accel(&mut solver, &s);
    assert_eq!(a1, a2, "same-device GPU dispatch must be bit-deterministic");
}

/// Newton's third law made a field invariant: the internal forces sum to zero, so
/// Σ mᵢ aᵢ = 0 analytically. The gather kernel computes each aᵢ independently (it does
/// NOT reuse the CPU's pairwise antisymmetry), so in f32 the cancellation is only
/// approximate — but the normalized net force must stay at the f32 floor, not O(1).
/// A kernel that failed to exclude self-interaction or mis-signed `dx` blows this up.
#[test]
fn gpu_conserves_total_momentum_flux() {
    const N: usize = 256;
    let mut solver = gpu(G, EPS);
    for seed in 0..16u64 {
        let s = cluster(seed, N, DVec3::ZERO);
        let a = accel(&mut solver, &s);

        let mut net = DVec3::ZERO;
        let mut scale = 0.0;
        for (ai, &mi) in a.iter().zip(&s.mass) {
            net += *ai * mi;
            scale += ai.length() * mi;
        }
        let rel = net.length() / scale.max(1e-300);
        assert!(rel < 1e-4, "net internal force {rel:e} (seed {seed})");
    }
}

/// Degenerate sizes must not panic and must be physically trivial: an empty system
/// yields no accelerations; a lone particle feels no force (its only "pair" is the
/// self term, whose zero separation contributes exactly zero).
#[test]
fn gpu_handles_empty_and_single() {
    let mut solver = gpu(G, EPS);

    let empty = State::from_phase_space(vec![], vec![], vec![]);
    let a = accel(&mut solver, &empty);
    assert!(a.is_empty());

    let one = State::from_phase_space(vec![DVec3::new(1.0, 2.0, 3.0)], vec![DVec3::ZERO], vec![1.0]);
    let a = accel(&mut solver, &one);
    assert_eq!(a, vec![DVec3::ZERO], "a lone particle feels no force");
}
