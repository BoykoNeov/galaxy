//! `GpuLbvhFused` (single-device fused GPU Linear-BVH, DESIGN M4h) validated against the f64
//! `DirectSum` oracle and the CPU `Lbvh` it mirrors — the M4h fuse of the whole M4c–M4g build
//! pipeline onto **one wgpu device in one submit** (Morton → sort → gather → Karras build +
//! aggregate → DFS skip-pointer flatten → traverse, all GPU-resident between stages; one
//! upload of `bodies`, one readback of `accel`).
//!
//! ## Same semantics as `GpuLbvh` ⇒ the same physics suite
//! `GpuLbvhFused` uses the **same f32 kernels** as `GpuLbvh` (the traversal kernel is reused
//! verbatim); only *where the intermediate buffers live* changes. So it must pass exactly the
//! M4g gates: θ→0 reproduces `DirectSum` to the f32 floor (topology-insensitive, so it is the
//! clean end-to-end gate that still catches a dropped/double-counted subtree or a bad skip
//! pointer); finite-θ error is bounded and grows with θ; it tracks the CPU `Lbvh` at the same
//! θ; momentum flux vanishes at θ→0; a repeated dispatch is bit-deterministic on a given
//! device; and the empty/single edges are trivial. The real-topology-straddle survival gate
//! (θ→0 stays exact even when the f32 tree genuinely differs from the f64 tree) is replicated
//! too, since the fuse builds the same f32 tree internally.
//!
//! The dedicated *faithful-refactor* gate — `GpuLbvhFused` reproduces the reference `GpuLbvh`
//! forces on the same device — lives in `gpu_lbvh_fused_refactor.rs` (added after the fuse
//! lands, its assertion form set by measurement, mirroring the M4g straddle "prove" commit).
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuLbvhFused::new` returns `NoAdapter`
//! and the tests fail loudly (matches the M3/M4 convention).

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_gpu::{GpuLbvhFused, GpuMortonBuilder, GpuSorter};
use galaxy_solvers::{reference_karras, reference_morton, reference_sort, DirectSum, Lbvh};

const G: f64 = 1.0;
const EPS: f64 = 0.05;

fn gpu(g: f64, eps: f64, theta: f64) -> GpuLbvhFused {
    GpuLbvhFused::new(g, eps, theta).expect("wgpu adapter required for GPU solver tests")
}

/// Deterministic pseudo-random cluster (same LCG as the Barnes-Hut / gpu-lbvh tests).
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

/// RMS acceleration over the system — normalizes relative errors so a particle near a
/// force null does not blow up the metric.
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

/// θ→0 opens every node down to its leaves, so the fused chain's tree walk *is* direct
/// summation — the clean end-to-end gate (topology-insensitive; no opening straddle), pinned
/// at the same f32 precision as the `GpuLbvh`/`GpuDirectSum`/`GpuTree` θ→0 gates. A flatten or
/// skip-pointer bug that dropped or double-counted a subtree blows this far past f32.
#[test]
fn gpu_lbvh_fused_theta_to_zero_matches_oracle() {
    const N: usize = 128;
    let mut solver = gpu(G, EPS, 1e-6);
    for seed in 0..64u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);
        let got = accel(&mut solver, &s);

        let rms = rms_rel_err(&got, &exact);
        let worst = worst_rel_err(&got, &exact);
        assert!(rms < 3e-4, "theta->0 RMS rel err {rms:e} (seed {seed})");
        assert!(
            worst < 5e-2,
            "theta->0 worst rel err {worst:e} (seed {seed})"
        );
    }
}

/// At finite θ the monopole approximation drops the quadrupole ⇒ O(θ²) error that grows with
/// θ. Same bounds as the M4g `GpuLbvh` gate (the fuse runs the same f32 kernels): max over 32
/// seeds ≈ 6.8e-3 at θ=0.3, 3.3e-2 at θ=0.6. A broken traversal gives O(1).
#[test]
fn gpu_lbvh_fused_finite_theta_bounded_and_grows() {
    const N: usize = 128;
    let mut lo_solver = gpu(G, EPS, 0.3);
    let mut hi_solver = gpu(G, EPS, 0.6);
    for seed in 0..32u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);

        let lo = accel(&mut lo_solver, &s);
        let hi = accel(&mut hi_solver, &s);

        let rms_lo = rms_rel_err(&lo, &exact);
        let rms_hi = rms_rel_err(&hi, &exact);

        assert!(rms_lo < 0.012, "theta=0.3 RMS err {rms_lo:e} (seed {seed})");
        assert!(rms_hi < 0.05, "theta=0.6 RMS err {rms_hi:e} (seed {seed})");
        assert!(
            rms_hi > rms_lo,
            "RMS error should grow with theta (seed {seed}): {rms_lo:e} -> {rms_hi:e}"
        );
    }
}

/// The fused GPU LBVH tracks the CPU `Lbvh` it mirrors at the *same* θ. Coarse RMS bound: the
/// two build independent trees (f32 vs f64 Morton), so beyond opening flips whole cells can be
/// approximated differently — the disagreement is BH-scale, not f32-scale. A broken traversal
/// gives an O(1) disagreement, which this catches.
#[test]
fn gpu_lbvh_fused_tracks_cpu_lbvh_same_theta() {
    const N: usize = 256;
    const THETA: f64 = 0.5;
    let mut solver = gpu(G, EPS, THETA);
    for seed in 0..16u64 {
        let s = cluster(seed, N);
        let cpu = accel(&mut Lbvh::new(G, EPS, THETA), &s);
        let got = accel(&mut solver, &s);
        let rms = rms_rel_err(&got, &cpu);
        assert!(
            rms < 5e-2,
            "fused GPU LBVH must track CPU Lbvh at same theta (seed {seed}): RMS {rms:e}"
        );
    }
}

/// The whole fused build+traverse is deterministic on a given device (no float `atomicAdd`
/// anywhere in the chain; every stage is single-invocation or race-free, and the traversal
/// writes each `acc[i]` from one private accumulator over a fixed skip-pointer order), so
/// repeated dispatches are bit-identical. Cross-device equality is NOT claimed.
#[test]
fn gpu_lbvh_fused_is_bit_deterministic_same_device() {
    const N: usize = 300;
    let mut solver = gpu(G, EPS, 0.5);
    let s = cluster(7, N);
    let a1 = accel(&mut solver, &s);
    let a2 = accel(&mut solver, &s);
    assert_eq!(
        a1, a2,
        "same-device fused GPU LBVH dispatch must be bit-deterministic"
    );
}

/// At θ→0 the tree is exact direct summation, so Σ mᵢ aᵢ = 0 to the f32 floor (pairwise
/// antisymmetry recovered when every node opens). A kernel that failed to exclude the self
/// term or mis-signed `dx` blows this to O(1) regardless of θ.
#[test]
fn gpu_lbvh_fused_conserves_total_momentum_flux_at_theta_zero() {
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

/// Degenerate sizes must not panic and must be physically trivial: an empty system yields no
/// accelerations; a lone particle feels no force (its only leaf holds just itself).
#[test]
fn gpu_lbvh_fused_handles_empty_and_single() {
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

/// A cube of half-width `r` centered at the origin — larger than [`cluster`]'s box so the
/// f32-vs-f64 Morton straddle actually triggers (it never does at the θ→0 test's N=128/r=1.5
/// scale; it needs enough near-boundary particles).
fn cube(seed: u64, n: usize, r: f64) -> State {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut pos = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    for _ in 0..n {
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * (2.0 * r));
        mass.push(0.1 + 0.9 * next());
    }
    let vel = vec![DVec3::ZERO; n];
    State::from_phase_space(pos, vel, mass)
}

/// **The straddle made provable, not merely possible.** The θ→0-vs-DirectSum gate is
/// topology-*insensitive*, so on its own it passes whether or not the f32 and f64 Morton
/// topologies ever actually diverged. This test closes that for the fused solver: at a scale
/// where the straddle *does* fire, it (1) **proves** the GPU f32 tree topology differs from
/// the f64 `reference_karras` topology on ≥1 seed, then (2) shows `GpuLbvhFused` at θ→0
/// **still** reproduces the exact `DirectSum` forces on exactly those diverged seeds — so the
/// end-to-end f32 topology straddle is exercised *and* survived by the fused pipeline too.
#[test]
fn gpu_lbvh_fused_theta_zero_survives_a_real_topology_straddle() {
    const N: usize = 20_000; // large enough that some particle lands within an f32 ulp of a
    const R: f64 = 3.0; // cell boundary; small coords keep θ→0 at the f32 floor.
    let mut morton = GpuMortonBuilder::new().expect("wgpu adapter required");
    let mut sorter = GpuSorter::new().expect("wgpu adapter required");

    // Pass 1: find seeds whose GPU f32 tree topology differs from the f64 reference tree.
    let mut diverged: Vec<u64> = Vec::new();
    for seed in 0..96u64 {
        let s = cube(seed, N, R);
        let gpu_codes = morton.compute(&s.pos).codes;
        let f64_codes = reference_morton(&s.pos).codes;
        let g_order = sorter.sort(&gpu_codes).order;
        let g_sorted: Vec<u32> = g_order.iter().map(|&i| gpu_codes[i as usize]).collect();
        let f_order = reference_sort(&f64_codes);
        let f_sorted: Vec<u32> = f_order.iter().map(|&i| f64_codes[i as usize]).collect();
        let gt = reference_karras(&g_sorted);
        let ft = reference_karras(&f_sorted);
        if gt.left != ft.left || gt.right != ft.right || gt.parent != ft.parent {
            diverged.push(seed);
        }
    }
    assert!(
        !diverged.is_empty(),
        "expected the f32 Morton topology straddle to fire on >=1 of 96 seeds — else the \
         end-to-end straddle is never exercised and the θ→0 gate is vacuous"
    );

    // Pass 2: on the diverged seeds, θ→0 GpuLbvhFused must still match DirectSum.
    let mut solver = gpu(G, EPS, 1e-6);
    let mut ds = DirectSum::new(G, EPS);
    for &seed in diverged.iter().take(3) {
        let s = cube(seed, N, R);
        let exact = accel(&mut ds, &s);
        let got = accel(&mut solver, &s);
        let rms = rms_rel_err(&got, &exact);
        let worst = worst_rel_err(&got, &exact);
        assert!(
            rms < 3e-4,
            "θ→0 must survive a real topology straddle (seed {seed}): RMS {rms:e}"
        );
        assert!(
            worst < 5e-2,
            "θ→0 worst must survive a real topology straddle (seed {seed}): {worst:e}"
        );
    }
}
