//! GPU-SPH G2 — the GPU adaptive-h density ([`GpuDensity`]) validated against the
//! CPU oracles [`galaxy_solvers::sph::density_adaptive`] / `density_fixed`.
//!
//! ## Gate design (D5): tolerance, not bit-exact; decoupled root-find vs summation
//! GPU forces/fields are f32 (D1), so this is an f32-TOLERANCE gate, never a
//! bit-exact one — the GPU walk order differs from the CPU ascending-index gather,
//! so the sums differ at the ulp by construction. Following the advisor's split, the
//! two halves are gated SEPARATELY so a bug localizes:
//!   * `gpu_density_fixed_h_matches_cpu` — the SUMMATION (`ρ` at a caller-supplied
//!     `h`) vs `density_fixed`. A neighbor-walk / kernel bug shows here.
//!   * `gpu_density_adaptive_matches_cpu` — the ROOT-FIND (`h`, then `ρ`) vs
//!     `density_adaptive`. A bisection/bracket bug shows here.
//!
//! ## The all-rooted precondition (advisor trap 1)
//! Clamped (rootless) particles get a seed-dependent `h` (`hi`=cap or `lo`=floor), so
//! GPU-vs-CPU would diverge on them unless the seed+cap are replicated bit-for-bit.
//! The diffuse tail of a wide-h cloud is exactly where under-population clamps — so
//! the main gate ASSERTS its cloud is fully rooted (`|N_i(h_i) − n_ngb| < ε` for all
//! `i`; a rooted particle hits the target by construction, a clamped one misses by
//! several-to-tens). `main_gate_cloud_fully_rooted` pins that precondition as a
//! standing CPU-only guard. Genuinely under-populated inputs (the single-particle
//! edge, which ALWAYS clamps at `N = 32/3 < n_ngb`) are gated STRUCTURALLY (finite,
//! positive `ρ`/`h`, no NaN), not against the CPU's clamped `h`.
//!
//! GPU-gated: these need a wgpu adapter; without one `GpuDensity::new` returns
//! `NoAdapter` and the tests fail loudly (the M3/M4 GPU-invariants convention).

use galaxy_core::DVec3;
use galaxy_gpu::GpuDensity;
use galaxy_solvers::sph::{density_adaptive, density_fixed, w, DensityConfig, SUPPORT};

const PI: f64 = std::f64::consts::PI;

fn density() -> GpuDensity {
    GpuDensity::new().expect("wgpu adapter required for GPU SPH density tests")
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

/// Centrally-concentrated, origin-centered cloud: radius `R·u` (u uniform) gives a
/// dense core and a diffuse tail, so the ADAPTIVE `h` (density-driven) spans a wide
/// range — robustly ~48× (p0.5/p99.5) at `n = 4000`, matching the measured gas-disk
/// regime (34×+) without a near-coincident-core singularity that would wreck f32
/// (`h_min ≈ 0.0075` here). Every particle stays rooted (asserted).
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

/// The kernel-weighted count `N_i(h)` the bisection targets. Rooted ⇔ `≈ n_ngb`.
fn count_of(pos: &[DVec3], i: usize, h: f64) -> f64 {
    let mut sum = 0.0;
    for &pj in pos {
        sum += w((pos[i] - pj).length(), h);
    }
    (4.0 * PI / 3.0) * (SUPPORT * h).powi(3) * sum
}

/// Number of particles NOT rooted: `|N_i(h_i) − n_ngb| ≥ tol`. The bisection resolves
/// `N` to ~`3·h_tol_rel·n_ngb ≈ 0.14`, so `tol = 0.5` cleanly separates rooted
/// (converged) from clamped (misses the root by several-to-tens).
fn clamped_count(pos: &[DVec3], h: &[f64], n_ngb: f64, tol: f64) -> usize {
    (0..pos.len())
        .filter(|&i| (count_of(pos, i, h[i]) - n_ngb).abs() >= tol)
        .count()
}

fn robust_ratio(h: &[f64]) -> f64 {
    let mut s = h.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    s[(n as f64 * 0.995) as usize] / s[(n as f64 * 0.005) as usize]
}

// ---------------------------------------------------------------------------
// CPU-only precondition guard (advisor trap 1): the main-gate cloud is rooted.
// ---------------------------------------------------------------------------

/// The main-gate cloud must be BOTH wide-h (so it exercises the multi-cell walk +
/// the density dynamic range) AND fully rooted (so GPU-vs-CPU `h` is seed-independent
/// and the tolerance gate is meaningful). This runs against the CPU oracle only, so
/// it is green from the red commit on — a standing regression guard on the cloud.
#[test]
fn main_gate_cloud_fully_rooted() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 4000, 5.0);
    let mass = vec![1.0; pos.len()];
    let res = density_adaptive(&pos, &mass, &cfg, None);
    let clamped = clamped_count(&pos, &res.h, cfg.n_ngb, 0.5);
    assert_eq!(
        clamped, 0,
        "main-gate cloud must be fully rooted (no seed-dependent clamps); {clamped} clamped"
    );
    let ratio = robust_ratio(&res.h);
    assert!(
        ratio > 20.0,
        "main-gate cloud must span a wide h range (robust ratio {ratio:.1}× > 20×)"
    );
}

// ---------------------------------------------------------------------------
// summation gate (decoupled): ρ at caller-supplied h vs density_fixed.
// ---------------------------------------------------------------------------

/// GPU `densities_at` (pure summation, no root-find) vs the CPU `density_fixed`
/// oracle, on the concentrated cloud with a smooth synthetic `h` that varies with
/// radius (mild range, every particle well-populated → nonzero `ρ`). Isolates the
/// neighbor-walk + kernel evaluation from the bisection. f32-tolerance.
#[test]
fn gpu_density_fixed_h_matches_cpu() {
    let pos = concentrated_cloud(0xF15ED, 3000, 5.0);
    // VARYING mass so the `m_j` factor is actually exercised: with uniform mass this
    // gate would pass identically for `Σ W`, `Σ m_i W` (wrong index), or no mass at
    // all. Per-`j` mass distinguishes all three (the project is equal-mass by
    // invariant, but G3's force consumes this ρ — catch a mass-indexing bug here).
    let mass: Vec<f64> = (0..pos.len()).map(|j| 1.0 + (j % 7) as f64).collect();
    let h: Vec<f64> = pos
        .iter()
        .map(|p| 0.1 + 0.15 * (p.length() / 5.0))
        .collect();

    let cpu = density_fixed(&pos, &mass, &h);
    let gpu = density().densities_at(&pos, &mass, &h);

    assert_eq!(gpu.len(), cpu.len());
    let mut worst = 0.0f64;
    for i in 0..cpu.len() {
        let rel = (gpu[i] as f64 - cpu[i]).abs() / cpu[i].max(1e-12);
        worst = worst.max(rel);
    }
    // Pure summation over the same neighbor set → essentially f32-roundoff (measured
    // worst ≈ 1e-6, only the sum order differs). 1e-3 keeps ~1000× headroom for
    // cross-device f32 while still failing hard on any wrong kernel/neighbor bug.
    assert!(
        worst < 1.0e-3,
        "GPU fixed-h ρ must match density_fixed to f32 tolerance; worst rel = {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// root-find gate (main): (ρ, h) vs density_adaptive on the wide-h rooted cloud.
// ---------------------------------------------------------------------------

/// The main gate. GPU adaptive `(ρ, h)` vs `density_adaptive` on the wide-h cloud,
/// per-particle f32-tolerance. Re-asserts all-rooted inline so a clamp can never
/// slip through and silently drag the tolerance.
#[test]
fn gpu_density_adaptive_matches_cpu() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 4000, 5.0);
    let mass = vec![1.0; pos.len()];

    let cpu = density_adaptive(&pos, &mass, &cfg, None);
    assert_eq!(
        clamped_count(&pos, &cpu.h, cfg.n_ngb, 0.5),
        0,
        "precondition: cloud must be fully rooted"
    );

    let gpu = density().densities(&pos, &mass, cfg.n_ngb, cfg.h_tol_rel);
    assert_eq!(gpu.h.len(), pos.len());
    assert_eq!(gpu.rho.len(), pos.len());

    let mut worst_h = 0.0f64;
    let mut worst_rho = 0.0f64;
    for i in 0..pos.len() {
        assert!(gpu.h[i].is_finite() && gpu.rho[i].is_finite(), "NaN at {i}");
        worst_h = worst_h.max((gpu.h[i] as f64 - cpu.h[i]).abs() / cpu.h[i]);
        worst_rho = worst_rho.max((gpu.rho[i] as f64 - cpu.rho[i]).abs() / cpu.rho[i].max(1e-12));
    }
    // h: two independent bisections (±h_tol_rel = ±1e-3 each) to the same unique root
    // + f32 noise → theoretical floor ~2·h_tol_rel = 2e-3; measured worst ≈ 9e-4.
    // 5e-3 sits above the floor with ~5× margin over the measured value.
    assert!(
        worst_h < 5.0e-3,
        "GPU adaptive h must match to f32+bisection tolerance; worst rel = {worst_h:.3e}"
    );
    // ρ inherits d ln ρ/d ln h ≈ O(1–3) × the h error → measured worst ≈ 1.3e-3.
    // 1e-2 gives ~8× margin while still catching a real (≫1%) regression.
    assert!(
        worst_rho < 1.0e-2,
        "GPU adaptive ρ must match to f32 tolerance; worst rel = {worst_rho:.3e}"
    );
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run)
// ---------------------------------------------------------------------------

/// Same input ⇒ bit-identical `(ρ, h)` on a given device (each thread owns its own
/// output slot; the walk is a gather, no scatter race — same discipline as G1).
#[test]
fn gpu_density_deterministic() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xBEEF, 1500, 4.0);
    let mass = vec![1.0; pos.len()];
    let mut g = density();
    let a = g.densities(&pos, &mass, cfg.n_ngb, cfg.h_tol_rel);
    let b = g.densities(&pos, &mass, cfg.n_ngb, cfg.h_tol_rel);
    assert_eq!(a.rho, b.rho, "ρ must be run-to-run identical");
    assert_eq!(a.h, b.h, "h must be run-to-run identical");
}

// ---------------------------------------------------------------------------
// edge cases
// ---------------------------------------------------------------------------

/// Empty input ⇒ empty fields, no panic.
#[test]
fn gpu_density_empty() {
    let field = density().densities(&[], &[], 48.0, 1e-3);
    assert!(field.rho.is_empty());
    assert!(field.h.is_empty());
}

/// A single particle is always under-populated (`N = 32/3 ≈ 10.7 < 48` at every
/// `h`), so it clamps — gate STRUCTURALLY (finite, positive `ρ` and `h`, no NaN),
/// not against the CPU's seed-dependent clamped `h`.
#[test]
fn gpu_density_single_particle_structural() {
    let pos = [DVec3::new(0.5, -0.5, 2.0)];
    let mass = [1.0];
    let field = density().densities(&pos, &mass, 48.0, 1e-3);
    assert_eq!(field.rho.len(), 1);
    assert_eq!(field.h.len(), 1);
    assert!(
        field.h[0].is_finite() && field.h[0] > 0.0,
        "h must be finite and positive, got {}",
        field.h[0]
    );
    assert!(
        field.rho[0].is_finite() && field.rho[0] > 0.0,
        "ρ must be finite and positive (self-term ρ = m/(π h³)), got {}",
        field.rho[0]
    );
}
