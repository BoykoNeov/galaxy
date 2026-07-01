//! `GpuTree` (GPU Barnes-Hut) validated against the f64 `DirectSum` oracle and the
//! CPU `BarnesHut` it mirrors.
//!
//! The kernel runs in **f32** (like `GpuDirectSum` — wgpu/naga has no portable f64
//! compute), so tolerances are f32-precision bounds *derived analytically*, not fit
//! to output. The tree adds one axis over direct-sum: the opening decision is f32 on
//! the GPU vs f64 on the CPU, so a handful of near-threshold nodes flip open/closed
//! — a discrete O(θ²) swing for those particles. That noise is why the "vs CPU BH
//! same-θ" gate bounds **RMS only** (few particles straddle) and the *clean*
//! traversal-isolation gate is **θ→0** (full open = direct sum, no straddling), which
//! together with the f64 bit-exact flatten test (`galaxy-solvers`) pins the walk.
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuTree::new` returns
//! `NoAdapter` and the tests fail loudly (matches the M3 render-invariants convention).

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_gpu::GpuTree;
use galaxy_solvers::{BarnesHut, DirectSum};

const G: f64 = 1.0;
const EPS: f64 = 0.05;

fn gpu(g: f64, eps: f64, theta: f64) -> GpuTree {
    GpuTree::new(g, eps, theta).expect("wgpu adapter required for GPU solver tests")
}

/// Deterministic pseudo-random cluster (same LCG as the Barnes-Hut oracle test).
fn cluster(seed: u64, n: usize) -> State {
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
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 3.0);
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

/// RMS acceleration over the system — normalizes relative errors so a particle near
/// a force null does not blow up the metric.
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

/// θ→0 opens every node down to its leaves, so the tree walk *is* direct summation
/// — the clean traversal-isolation gate (no opening-threshold straddling), pinned at
/// the same f32 precision as the `GpuDirectSum` unit-box gate. A flattening or
/// skip-pointer bug that dropped or double-counted a subtree blows this far past f32.
#[test]
fn gpu_tree_theta_to_zero_matches_oracle() {
    const N: usize = 128;
    let mut solver = gpu(G, EPS, 1e-6);
    for seed in 0..64u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);
        let got = accel(&mut solver, &s);

        let rms = rms_rel_err(&got, &exact);
        let worst = worst_rel_err(&got, &exact);
        // Full-open = direct sum in f32: √128·ε ≈ 1.4e-6 baseline × close-pair
        // conditioning ⇒ ≪ 3e-4 (same bound as the GpuDirectSum unit-box gate).
        assert!(rms < 3e-4, "theta->0 RMS rel err {rms:e} (seed {seed})");
        assert!(
            worst < 5e-2,
            "theta->0 worst rel err {worst:e} (seed {seed})"
        );
    }
}

/// At finite θ the monopole approximation drops the quadrupole, so the error is
/// O(θ²) and grows with θ — the GPU tree reproduces the CPU BarnesHut's accuracy
/// trade (bounds loosened only slightly over the CPU-BH gate for the f32 floor).
#[test]
fn gpu_tree_finite_theta_bounded_and_grows() {
    const N: usize = 128;
    for seed in 0..32u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);

        let lo = accel(&mut gpu(G, EPS, 0.3), &s);
        let hi = accel(&mut gpu(G, EPS, 0.6), &s);

        let rms_lo = rms_rel_err(&lo, &exact);
        let rms_hi = rms_rel_err(&hi, &exact);

        // CPU-BH RMS bounds (0.005 / 0.03) + f32 slack; O(θ²) truncation dominates
        // the f32 noise, so it stays well under.
        assert!(rms_lo < 0.01, "theta=0.3 RMS err {rms_lo:e} (seed {seed})");
        assert!(rms_hi < 0.05, "theta=0.6 RMS err {rms_hi:e} (seed {seed})");
        assert!(
            rms_hi > rms_lo,
            "RMS error should grow with theta (seed {seed}): {rms_lo:e} -> {rms_hi:e}"
        );
    }
}

/// The GPU tree must reproduce the CPU BarnesHut it mirrors at the *same* θ far more
/// faithfully than either tracks the exact oracle — this isolates the traversal from
/// BH truncation. **RMS only, coarse**: the opening test is f32 on the GPU vs f64 on
/// the CPU, so near-threshold nodes flip and a few particles take a discrete O(θ²)
/// swing (worst-case is BH-scale, not f32-scale — deliberately not asserted). A
/// broken traversal instead gives an O(1) disagreement, which this catches.
#[test]
fn gpu_tree_matches_cpu_bh_same_theta() {
    const N: usize = 256;
    const THETA: f64 = 0.5;
    let mut solver = gpu(G, EPS, THETA);
    for seed in 0..16u64 {
        let s = cluster(seed, N);
        let bh = accel(&mut BarnesHut::new(G, EPS, THETA), &s);
        let got = accel(&mut solver, &s);
        let rms = rms_rel_err(&got, &bh);
        assert!(
            rms < 2e-2,
            "GPU tree must track CPU BH at same theta (seed {seed}): RMS {rms:e}"
        );
    }
}

/// The stackless walk follows a fixed skip-pointer order and writes each `acc[i]`
/// from one private accumulator, so repeated dispatches on the SAME device are
/// bit-identical (no float `atomicAdd`). Cross-device equality is NOT claimed.
#[test]
fn gpu_tree_is_bit_deterministic_same_device() {
    const N: usize = 300;
    let mut solver = gpu(G, EPS, 0.5);
    let s = cluster(7, N);
    let a1 = accel(&mut solver, &s);
    let a2 = accel(&mut solver, &s);
    assert_eq!(
        a1, a2,
        "same-device GPU tree dispatch must be bit-deterministic"
    );
}

/// At θ→0 the tree is exact direct summation, so Σ mᵢ aᵢ = 0 to the f32 floor (the
/// pairwise antisymmetry is recovered when every node opens to its leaves). At finite
/// θ the monopole approximation breaks Newton's third law at O(θ²), so this invariant
/// is asserted in the θ→0 regime — a kernel that failed to exclude self-interaction or
/// mis-signed `dx` blows it up to O(1) regardless of θ.
#[test]
fn gpu_tree_conserves_total_momentum_flux_at_theta_zero() {
    const N: usize = 256;
    let mut solver = gpu(G, EPS, 1e-6);
    for seed in 0..16u64 {
        let s = cluster(seed, N);
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
/// yields no accelerations; a lone particle feels no force (its only leaf holds just
/// itself, excluded as the self term).
#[test]
fn gpu_tree_handles_empty_and_single() {
    let mut solver = gpu(G, EPS, 0.5);

    let empty = State::from_phase_space(vec![], vec![], vec![]);
    let a = accel(&mut solver, &empty);
    assert!(a.is_empty());

    let one = State::from_phase_space(
        vec![DVec3::new(1.0, 2.0, 3.0)],
        vec![DVec3::ZERO],
        vec![1.0],
    );
    let a = accel(&mut solver, &one);
    assert_eq!(a, vec![DVec3::ZERO], "a lone particle feels no force");
}
