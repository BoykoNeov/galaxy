//! The M4h **faithful-refactor** gate: `GpuLbvhFused` (single device, one submit) reproduces
//! the reference-grade `GpuLbvh` (M4g; five devices, host round-trips between stages) — the
//! direct evidence that fusing the pipeline onto one device changed *nothing but where the
//! intermediate buffers live*.
//!
//! ## Why this can be exact
//! The fuse runs the **same f32 WGSL** as the reference chain (the traversal kernel is reused
//! verbatim; only the trivial `gather` and geometry-repack kernels are new, and they compute
//! identical values). On a **given device** an f32 op sequence is deterministic, and neither
//! path uses a float `atomicAdd` or any order-dependent reduction, so the two should agree
//! bit-for-bit. This is a *stronger* claim than the existing same-device determinism gates
//! (which compare a solver to itself); here two *different* device/pipeline setups on the same
//! adapter are compared. The assertion form below (bit-exact vs a tight documented tolerance)
//! was set by measurement — mirroring the project's "bounds set from measurement" precedent.
//!
//! Same-device only: both solvers request the `HighPerformance` adapter in the same process, so
//! they resolve to the same GPU. Cross-device (different adapter/driver/FMA) equality is NOT
//! claimed — the same caveat every GPU stage carries.
//!
//! GPU-gated: needs a wgpu adapter; without one both constructors return `NoAdapter` and the
//! tests fail loudly.

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_gpu::{GpuLbvh, GpuLbvhFused};

const G: f64 = 1.0;
const EPS: f64 = 0.05;

/// Deterministic pseudo-random cluster (same LCG as the other GPU-LBVH tests).
fn cluster(seed: u64, n: usize) -> State {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
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

/// Max per-component absolute difference between two acceleration fields.
fn max_abs_diff(a: &[DVec3], b: &[DVec3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(u, v)| {
            (u.x - v.x)
                .abs()
                .max((u.y - v.y).abs())
                .max((u.z - v.z).abs())
        })
        .fold(0.0_f64, f64::max)
}

/// `GpuLbvhFused` == the reference `GpuLbvh`, **bit-for-bit**, on the same device — at θ→0
/// (every node opens: exercises the full leaf-direct-sum path) and at finite θ (exercises the
/// monopole-acceptance path, where both build the *same* f32 tree so even the opening decisions
/// coincide). Across N spanning the small θ→0 regime up to the straddle regime.
///
/// Measured result: the fuse is exact — `max |Δ| == 0` on every (seed, N, θ) below. That is the
/// strongest possible statement that the fuse is a lossless refactor: same kernels, same inputs,
/// same device ⇒ same bits, differing only in that no data round-trips through host memory.
#[test]
fn gpu_lbvh_fused_matches_reference_bit_for_bit() {
    for &theta in &[1e-6_f64, 0.5] {
        let mut fused =
            GpuLbvhFused::new(G, EPS, theta).expect("wgpu adapter required for GPU solver tests");
        let mut reference =
            GpuLbvh::new(G, EPS, theta).expect("wgpu adapter required for GPU solver tests");
        for &n in &[128usize, 256, 2000] {
            for seed in 0..8u64 {
                let s = cluster(seed, n);
                let got = accel(&mut fused, &s);
                let want = accel(&mut reference, &s);
                let d = max_abs_diff(&got, &want);
                assert!(
                    d == 0.0,
                    "fused must reproduce reference GpuLbvh bit-for-bit on the same device \
                     (θ={theta}, N={n}, seed={seed}): max |Δ| = {d:e}"
                );
            }
        }
    }
}
