//! `GpuLbvh` (GPU Linear-BVH Barnes-Hut, DESIGN M4g) validated against the f64 `DirectSum`
//! oracle and the CPU `Lbvh` it mirrors — the fifth and final stage of the GPU-resident
//! LBVH build: an f32 stackless **traversal** of the M4f flat form, fed by the full GPU f32
//! chain (`GpuMorton` → `GpuSort` → `GpuLbvhBuilder` → `GpuLbvhFlattener` → traverse).
//!
//! ## θ→0 is where the end-to-end f32 topology straddle is finally checked
//! Every earlier stage was gated against its CPU reference in isolation. This is the first
//! gate on the **whole f32 pipeline run end-to-end**. The subtlety: θ→0 opens every node to
//! its leaves, so the walk *is* direct summation **regardless of tree topology** — which is
//! exactly why it is the clean gate. The f32 Morton codes can quantize differently from the
//! f64 codes (the M4c divergence), so the GPU tree's *topology* may differ from the CPU
//! `Lbvh`'s; θ→0 is **insensitive** to that straddle, yet still catches any dropped or
//! double-counted subtree or bad skip pointer. So it does not *assert* the topology matches
//! — it shows the f32 pipeline runs end-to-end and *still* yields exact forces despite a
//! possibly-different topology. A flatten/traversal bug blows this past the f32 floor.
//!
//! At finite θ the bound is **looser** than `GpuTree`'s same-θ gate: `GpuTree` traverses the
//! very octree it is compared against (only f32 opening flips differ), whereas `GpuLbvh`
//! builds its **own** tree, so whole cells can be approximated differently, not just a few
//! near-threshold nodes. The finite-θ gate is therefore RMS-only and coarse (bounded + grows
//! with θ); θ→0 stays the tight gate.
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuLbvh::new` returns `NoAdapter` and
//! the tests fail loudly (matches the M3/M4 convention).

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_gpu::GpuLbvh;
use galaxy_solvers::{DirectSum, Lbvh};

const G: f64 = 1.0;
const EPS: f64 = 0.05;

fn gpu(g: f64, eps: f64, theta: f64) -> GpuLbvh {
    GpuLbvh::new(g, eps, theta).expect("wgpu adapter required for GPU solver tests")
}

/// Deterministic pseudo-random cluster (same LCG as the Barnes-Hut / gpu-tree tests).
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

/// θ→0 opens every node down to its leaves, so the whole GPU chain's tree walk *is* direct
/// summation — the clean end-to-end gate (topology-insensitive; no opening straddle), pinned
/// at the same f32 precision as the `GpuDirectSum`/`GpuTree` θ→0 gates. A flatten or
/// skip-pointer bug that dropped or double-counted a subtree blows this far past f32.
#[test]
fn gpu_lbvh_theta_to_zero_matches_oracle() {
    const N: usize = 128;
    let mut solver = gpu(G, EPS, 1e-6);
    for seed in 0..64u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);
        let got = accel(&mut solver, &s);

        let rms = rms_rel_err(&got, &exact);
        let worst = worst_rel_err(&got, &exact);
        // Full-open = direct sum in f32: same bound as the GpuDirectSum/GpuTree θ→0 gate.
        assert!(rms < 3e-4, "theta->0 RMS rel err {rms:e} (seed {seed})");
        assert!(
            worst < 5e-2,
            "theta->0 worst rel err {worst:e} (seed {seed})"
        );
    }
}

/// At finite θ the monopole approximation drops the quadrupole ⇒ O(θ²) error that grows with
/// θ. Bounds set from measurement (max over 32 seeds ≈ 6.8e-3 at θ=0.3, 3.3e-2 at θ=0.6),
/// with modest headroom: the O(θ²) BH truncation dominates the topology straddle at these θ,
/// so these land near `GpuTree`'s vs-DirectSum figures — a broken traversal gives O(1).
#[test]
fn gpu_lbvh_finite_theta_bounded_and_grows() {
    const N: usize = 128;
    // Solvers are hoisted out of the seed loop: each GpuLbvh brings up the whole GPU chain
    // (several devices), so per-seed construction would dominate wall-clock.
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

/// The GPU LBVH tracks the CPU `Lbvh` it mirrors at the *same* θ. Coarse RMS bound: the two
/// build independent trees (f32 vs f64 Morton), so beyond opening flips whole cells can be
/// approximated differently — the disagreement is BH-scale, not f32-scale. A broken traversal
/// gives an O(1) disagreement, which this catches.
#[test]
fn gpu_lbvh_tracks_cpu_lbvh_same_theta() {
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
            "GPU LBVH must track CPU Lbvh at same theta (seed {seed}): RMS {rms:e}"
        );
    }
}

/// The stackless walk writes each `acc[i]` from one private accumulator over a fixed
/// skip-pointer order, and the whole GPU build is deterministic on a given device (no float
/// `atomicAdd` anywhere in the chain), so repeated dispatches are bit-identical. Cross-device
/// equality is NOT claimed.
#[test]
fn gpu_lbvh_is_bit_deterministic_same_device() {
    const N: usize = 300;
    let mut solver = gpu(G, EPS, 0.5);
    let s = cluster(7, N);
    let a1 = accel(&mut solver, &s);
    let a2 = accel(&mut solver, &s);
    assert_eq!(
        a1, a2,
        "same-device GPU LBVH dispatch must be bit-deterministic"
    );
}

/// At θ→0 the tree is exact direct summation, so Σ mᵢ aᵢ = 0 to the f32 floor (pairwise
/// antisymmetry recovered when every node opens). A kernel that failed to exclude the self
/// term or mis-signed `dx` blows this to O(1) regardless of θ.
#[test]
fn gpu_lbvh_conserves_total_momentum_flux_at_theta_zero() {
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
fn gpu_lbvh_handles_empty_and_single() {
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
