//! GPU-SPH G1 — the GPU spatial-hash neighbor search ([`GpuNeighborGrid`])
//! validated against the CPU [`galaxy_solvers::sph::HashGrid`].
//!
//! ## The gate is the FILTERED pair set, not raw candidates (D4)
//! The load-bearing gate is equality of the **filtered pair set** — unordered
//! pairs `(i, j)`, `i ≠ j`, with `r_ij < SUPPORT·max(h_i, h_j)` (the true
//! averaged-kernel coupling range, `W̄` nonzero out to `2·max(h_i,h_j)`) — between
//! the GPU grid and `HashGrid`, on synthetic clouds with a synthetic per-particle
//! `h` (real `h` is G2's output; G1 needs only *a* filter to gate against). This
//! is deliberately radius-policy-invariant: fork(a) here over-gathers at the global
//! `SUPPORT·h_max`, the fork(b)/LBVH endpoint will gather at per-particle
//! `SUPPORT·h_i` + prune — DIFFERENT raw candidate sets, the SAME filtered set. So
//! this gate survives the eventual grid→LBVH swap unchanged.
//!
//! A raw-candidate equality check is included too, but ONLY as a fork(a) sanity
//! test and marked throwaway — it pins the global over-gather radius and MUST be
//! deleted (not "fixed") when the LBVH range query lands, or it will false-fail on
//! a correct implementation.
//!
//! GPU-gated: these need a wgpu adapter. Without one `GpuNeighborGrid::new` returns
//! `NoAdapter` and the tests fail loudly (the M3/M4 GPU-invariants convention).

use std::collections::HashSet;

use galaxy_core::DVec3;
use galaxy_gpu::GpuNeighborGrid;
use galaxy_solvers::sph::{HashGrid, SUPPORT};

fn grid() -> GpuNeighborGrid {
    GpuNeighborGrid::new().expect("wgpu adapter required for GPU SPH neighbor-grid tests")
}

/// Deterministic pseudo-random `[0, 1)` stream (the same LCG as the other
/// solver/GPU tests).
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Uniform pseudo-random positions in a cube of half-width `r` centered at
/// `center`.
fn uniform_cloud(seed: u64, n: usize, r: f64, center: DVec3) -> Vec<DVec3> {
    let mut next = rng(seed);
    (0..n)
        .map(|_| center + DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * (2.0 * r))
        .collect()
}

/// Centrally-concentrated cloud: radius `R·u²` (u uniform) packs points toward the
/// center, giving the intrinsically wide local-spacing (hence wide-`h`) range the
/// real gas disk has — the case that actually exercises the `ceil(r/cell)`
/// neighborhood walk. Returns positions and a matching synthetic per-particle `h`
/// that grows with radius (dense small-`h` core, diffuse large-`h` halo).
fn concentrated_cloud(seed: u64, n: usize, radius: f64) -> (Vec<DVec3>, Vec<f64>) {
    let mut next = rng(seed);
    let mut pos = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    for _ in 0..n {
        let u = next();
        let r = radius * u * u;
        // Random direction on the sphere.
        let z = 2.0 * next() - 1.0;
        let phi = std::f64::consts::TAU * next();
        let s = (1.0 - z * z).max(0.0).sqrt();
        pos.push(DVec3::new(s * phi.cos(), s * phi.sin(), z) * r);
        // h spans ~0.02 (core) to ~radius (halo): a wide, `r ≫ cell` range.
        h.push(0.02 + r);
    }
    (pos, h)
}

/// A constant synthetic `h` for the uniform-cloud gate (fixed coupling radius).
fn constant_h(n: usize, h: f64) -> Vec<f64> {
    vec![h; n]
}

fn h_max_of(h: &[f64]) -> f64 {
    h.iter().fold(0.0_f64, |a, &b| a.max(b))
}

/// The filtered pair set built from a candidate-list provider: unordered `(i, j)`
/// with `i < j` and `r_ij < SUPPORT·max(h_i, h_j)`. Candidates for `i` may include
/// `i` and duplicates; both are handled (self dropped, set dedups).
fn filtered_pairs(
    pos: &[DVec3],
    h: &[f64],
    mut candidates_of: impl FnMut(usize) -> Vec<usize>,
) -> HashSet<(usize, usize)> {
    let mut set = HashSet::new();
    for i in 0..pos.len() {
        for j in candidates_of(i) {
            if j == i {
                continue;
            }
            let coupling = SUPPORT * h[i].max(h[j]);
            if (pos[i] - pos[j]).length() < coupling {
                set.insert((i.min(j), i.max(j)));
            }
        }
    }
    set
}

/// CPU reference filtered pair set via `HashGrid`, over-gathering at the global
/// `SUPPORT·h_max` (the fork(a) / `forces.rs` radius) so no coupling pair is missed.
fn cpu_filtered_pairs(pos: &[DVec3], h: &[f64]) -> HashSet<(usize, usize)> {
    if pos.is_empty() {
        return HashSet::new();
    }
    let r = SUPPORT * h_max_of(h);
    let g = HashGrid::build(pos, r);
    filtered_pairs(pos, h, |i| g.neighbours_within(pos, pos[i], r))
}

/// GPU filtered pair set: query every particle at the global `SUPPORT·h_max` with
/// the given `cell` (decoupled from the radius so `cell < radius` stresses the
/// multi-cell walk), then apply the identical coupling filter.
fn gpu_filtered_pairs(
    g: &mut GpuNeighborGrid,
    pos: &[DVec3],
    h: &[f64],
    cell: f64,
) -> HashSet<(usize, usize)> {
    if pos.is_empty() {
        return HashSet::new();
    }
    let r = SUPPORT * h_max_of(h);
    let ngb = g.query_all(pos, cell, r);
    filtered_pairs(pos, h, |i| {
        ngb.neighbours(i).iter().map(|&j| j as usize).collect()
    })
}

/// Core gate: GPU filtered pair set must equal `HashGrid`'s, exactly (set equality,
/// order-independent).
fn assert_filtered_pairs_match(pos: &[DVec3], h: &[f64], cell: f64) {
    let cpu = cpu_filtered_pairs(pos, h);
    let gpu = gpu_filtered_pairs(&mut grid(), pos, h, cell);
    assert_eq!(
        gpu,
        cpu,
        "GPU filtered pair set ({} pairs) must equal HashGrid's ({} pairs)",
        gpu.len(),
        cpu.len()
    );
}

// ---------------------------------------------------------------------------
// core: filtered-pair-set agreement with the CPU HashGrid
// ---------------------------------------------------------------------------

/// Uniform cloud, constant `h` (cell ≈ radius → 27-cell walk). The baseline gate.
#[test]
fn gpu_grid_matches_hashgrid_uniform() {
    for seed in [1u64, 7, 42] {
        let pos = uniform_cloud(seed, 3000, 3.0, DVec3::ZERO);
        let h = constant_h(pos.len(), 0.35);
        let cell = SUPPORT * 0.35;
        assert_filtered_pairs_match(&pos, &h, cell);
    }
}

/// Centrally-concentrated cloud with a wide synthetic `h` range, queried with a
/// SMALL cell (`cell = SUPPORT·h_min ≪ SUPPORT·h_max = radius`) — the wide-`h` gas
/// disk regime a uniform-cloud test cannot reach. `GpuNeighborGrid` internally CAPS
/// the bucket edge at `max(cell, radius/4)`, so the raw `cell ≪ radius` here does NOT
/// literally drive a `ceil(r/cell) ≈ 250`-cell walk — that workload is infeasible on
/// a uniform grid and is exactly what the LBVH endpoint exists for (D4). The cap is
/// correctness-neutral (a coarser bucket only enlarges buckets), so this still
/// exercises the genuine multi-cell walk + hash-collision + cell-match-dedup path
/// (≤ 9³ cells) against the widest-`h` cloud — the assertion is unchanged.
#[test]
fn gpu_grid_matches_hashgrid_wide_h_small_cell() {
    for seed in [2u64, 13, 99] {
        let (pos, h) = concentrated_cloud(seed, 2000, 5.0);
        let h_min = h.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let cell = SUPPORT * h_min; // ≪ SUPPORT·h_max (grid caps the walk; see D4)
        assert_filtered_pairs_match(&pos, &h, cell);
    }
}

/// Far outliers alongside a dense core: distinct cell coords (dense-core cells and
/// distant-flier cells) collide in the fixed hash table, so the query must reject
/// out-of-radius bucket-mates by TRUE distance and never double-count a particle
/// reached via two colliding cells. Set equality catches both.
#[test]
fn gpu_grid_survives_hash_collisions_from_far_debris() {
    let mut pos = uniform_cloud(0x5EED, 1500, 1.0, DVec3::ZERO);
    // A handful of distant fliers (merger debris), each in its own far-off cell.
    let mut next = rng(0xDEB1);
    for _ in 0..40 {
        let d = 200.0 + 800.0 * next();
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * (2.0 * d));
    }
    let h = constant_h(pos.len(), 0.2);
    let cell = SUPPORT * 0.2;
    assert_filtered_pairs_match(&pos, &h, cell);
}

/// Negative coordinates and points landing exactly on cell walls must bin
/// consistently between GPU (`floor`) and `HashGrid` (`floor`) — an off-center
/// cloud straddling the origin exercises the sign/boundary path.
#[test]
fn gpu_grid_matches_hashgrid_off_origin() {
    let pos = uniform_cloud(0xA11CE, 2500, 2.0, DVec3::new(-3.7, 1.3, -0.9));
    let h = constant_h(pos.len(), 0.3);
    assert_filtered_pairs_match(&pos, &h, SUPPORT * 0.15); // cell < radius
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run)
// ---------------------------------------------------------------------------

/// Same input ⇒ bit-identical CSR on a given device. Near-vacuous for a gather
/// (no scatter race), but a regression guard for the deferred parallel counting
/// sort where bucket-scatter ordering could reintroduce nondeterminism.
#[test]
fn gpu_grid_query_is_deterministic() {
    let (pos, h) = concentrated_cloud(0x0FF, 1200, 4.0);
    let cell = SUPPORT * 0.1;
    let r = SUPPORT * h_max_of(&h);
    let mut g = grid();
    let a = g.query_all(&pos, cell, r);
    let b = g.query_all(&pos, cell, r);
    for i in 0..pos.len() {
        let mut ai = a.neighbours(i).to_vec();
        let mut bi = b.neighbours(i).to_vec();
        ai.sort_unstable();
        bi.sort_unstable();
        assert_eq!(
            ai, bi,
            "query for particle {i} must be run-to-run identical"
        );
    }
}

// ---------------------------------------------------------------------------
// edge cases
// ---------------------------------------------------------------------------

/// Empty input ⇒ empty result, no panic.
#[test]
fn gpu_grid_empty() {
    let ngb = grid().query_all(&[], 1.0, 1.0);
    assert!(ngb.is_empty());
    assert_eq!(ngb.len(), 0);
}

/// A single particle is its own only neighbor (distance 0 ≤ radius).
#[test]
fn gpu_grid_single() {
    let pos = [DVec3::new(0.5, -0.5, 2.0)];
    let ngb = grid().query_all(&pos, 1.0, 1.0);
    assert_eq!(ngb.len(), 1);
    assert_eq!(ngb.neighbours(0), &[0]);
}

/// Two particles inside the radius see each other and themselves.
#[test]
fn gpu_grid_pair_within_radius() {
    let pos = [DVec3::ZERO, DVec3::new(0.5, 0.0, 0.0)];
    let ngb = grid().query_all(&pos, 1.0, 1.0);
    let mut n0 = ngb.neighbours(0).to_vec();
    let mut n1 = ngb.neighbours(1).to_vec();
    n0.sort_unstable();
    n1.sort_unstable();
    assert_eq!(n0, vec![0, 1]);
    assert_eq!(n1, vec![0, 1]);
}

/// Two particles beyond the radius see only themselves (true-distance rejection
/// even when they share, or neighbor, a cell).
#[test]
fn gpu_grid_pair_beyond_radius() {
    let pos = [DVec3::ZERO, DVec3::new(5.0, 0.0, 0.0)];
    let ngb = grid().query_all(&pos, 1.0, 1.0);
    assert_eq!(ngb.neighbours(0), &[0]);
    assert_eq!(ngb.neighbours(1), &[1]);
}

// ---------------------------------------------------------------------------
// THROWAWAY fork(a) sanity: raw candidate set == HashGrid at the same radius.
// DELETE this test (do NOT "fix" it) when the LBVH range query lands — it pins the
// global over-gather radius, so it WILL false-fail on the per-particle-radius LBVH
// even when that implementation is correct. The filtered-pair-set gates above are
// the swap-stable ones.
// ---------------------------------------------------------------------------

/// fork(a) only: at a common radius the GPU grid's raw candidate SET (per particle)
/// equals `HashGrid::neighbours_within`. Sanity that the bucket walk enumerates the
/// same particles; not swap-stable.
#[test]
fn throwaway_forka_raw_candidates_match_hashgrid() {
    let pos = uniform_cloud(0xCA11, 2000, 2.0, DVec3::ZERO);
    let radius = 0.5;
    let cell = radius; // fork(a): cell == radius
    let g_cpu = HashGrid::build(&pos, radius);
    let ngb = grid().query_all(&pos, cell, radius);
    for i in 0..pos.len() {
        let cpu: HashSet<usize> = g_cpu
            .neighbours_within(&pos, pos[i], radius)
            .into_iter()
            .collect();
        let gpu: HashSet<usize> = ngb.neighbours(i).iter().map(|&j| j as usize).collect();
        assert_eq!(
            gpu, cpu,
            "raw candidate set for particle {i} must match (fork(a) sanity)"
        );
    }
}
