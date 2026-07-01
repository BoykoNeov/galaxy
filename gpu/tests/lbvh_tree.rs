//! GPU Karras tree-build + atomic-flag bottom-up aggregation (DESIGN M4e) validated
//! against the CPU references [`galaxy_solvers::reference_karras`] (topology) and
//! [`galaxy_solvers::reference_aggregate`] (fold).
//!
//! Two gates, because the stage is half integer and half f32:
//!
//! * **Topology is pure integer ⇒ bit-exact.** Given the same `sorted_codes`, the GPU
//!   `(left, right, parent)` must equal `reference_karras` bit-for-bit — the load-bearing
//!   gate, like the M4d sort. The **all-equal-codes** case is the sharpest: it forces
//!   every node onto the `32 + clz(a^b)` position tie-break, where a `u32` (instead of
//!   signed `i32`) port of the δ search — the #1 silent-corruption risk — goes wrong.
//! * **Aggregation is f32 ⇒ split gate.** AABB `min`/`max` folds never round and are
//!   order-independent, so they are **bit-exact** vs the f64 reference run over the same
//!   f32-narrowed positions (min/max just select a leaf coordinate, exact under
//!   widening). `com`/`mass` are f32-lossy → tolerance.
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuLbvhBuilder::new` returns
//! `NoAdapter` and the tests fail loudly (matches the M3/M4 GPU-invariants convention).

use galaxy_core::DVec3;
use galaxy_gpu::{GpuLbvhBuilder, GpuLbvhTree};
use galaxy_solvers::{
    reference_aggregate, reference_karras, reference_morton, reference_sort, KarrasTree, NO_PARENT,
};

fn builder() -> GpuLbvhBuilder {
    GpuLbvhBuilder::new().expect("wgpu adapter required for GPU tree-build tests")
}

/// Deterministic pseudo-random positions in a cube of half-width `r` centered at
/// `center` (same LCG as the other solver/GPU tests).
fn cloud(seed: u64, n: usize, r: f64, center: DVec3) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) // in [0, 1)
    };
    (0..n)
        .map(|_| center + DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * (2.0 * r))
        .collect()
}

/// Deterministic pseudo-random positive masses in `[0.5, 1.5)` — varied so the
/// mass-weighted centre of mass is a real (f32-lossy) fold, not an exact integer sum.
fn masses(seed: u64, n: usize) -> Vec<f64> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.5 + ((state >> 11) as f64) / ((1u64 << 53) as f64)
        })
        .collect()
}

/// Gather positions/masses/codes into Morton-sorted order — the exact pipeline prefix
/// (`reference_morton` → `reference_sort` → gather) the GPU sort produces before this
/// stage. Returns `(sorted_codes, sorted_pos, sorted_mass)`.
fn sorted_inputs(pos: &[DVec3], mass: &[f64]) -> (Vec<u32>, Vec<DVec3>, Vec<f64>) {
    let codes = reference_morton(pos).codes;
    let order = reference_sort(&codes);
    let sorted_codes = order.iter().map(|&i| codes[i as usize]).collect();
    let sorted_pos = order.iter().map(|&i| pos[i as usize]).collect();
    let sorted_mass = order.iter().map(|&i| mass[i as usize]).collect();
    (sorted_codes, sorted_pos, sorted_mass)
}

/// Round a f64 vector through f32 and back — the exact leaf value the GPU sees
/// (`p.x as f32`), so the f64 reference folds over identical inputs.
fn narrow(pos: &[DVec3]) -> Vec<DVec3> {
    pos.iter()
        .map(|p| DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64))
        .collect()
}

// ---------------------------------------------------------------------------
// topology — bit-exact vs reference_karras (pure integer)
// ---------------------------------------------------------------------------

/// Feed `sorted_codes` (with throwaway positions/masses) and assert the GPU topology
/// matches `reference_karras` bit-for-bit and is structurally well-formed.
fn assert_topology(sorted_codes: &[u32]) {
    let n = sorted_codes.len();
    // Any leaf payload works for a topology check; use a cheap deterministic one.
    let pos: Vec<DVec3> = (0..n).map(|i| DVec3::splat(i as f64)).collect();
    let mass = vec![1.0; n];
    let gpu = builder().build(sorted_codes, &pos, &mass);
    let refr = reference_karras(sorted_codes);

    assert_eq!(gpu.n, refr.n, "N mismatch");
    assert_eq!(
        gpu.left, refr.left,
        "left children must bit-match reference_karras (pure integer — no tolerance)"
    );
    assert_eq!(
        gpu.right, refr.right,
        "right children must bit-match reference_karras"
    );
    assert_eq!(
        gpu.parent, refr.parent,
        "parents must bit-match reference_karras"
    );
    assert_structural(&gpu);
}

/// Real Morton codes from position clouds build the reference tree bit-identically.
#[test]
fn gpu_tree_topology_matches_reference_on_morton_codes() {
    for seed in [1u64, 7, 42, 1000] {
        let pos = cloud(seed, 6000, 3.0, DVec3::ZERO);
        let (sorted_codes, ..) = sorted_inputs(&pos, &vec![1.0; pos.len()]);
        assert_topology(&sorted_codes);
    }
}

/// **All-equal codes** — every internal node falls onto the `32 + clz(a^b)` sorted-
/// position tie-break. This is the case that exposes a `u32`-instead-of-`i32` δ search
/// (where the out-of-range −1 sentinel becomes `0xFFFFFFFF` and wins every comparison)
/// or a `countLeadingZeros(0u) != 32u` mismatch. Builds a fully left-leaning tie tree.
#[test]
fn gpu_tree_topology_all_equal_codes() {
    let sorted_codes = vec![42u32; 4096];
    assert_topology(&sorted_codes);
}

/// Heavy-duplicate codes (blocks of 500 equal values, still non-decreasing) mix real
/// code splits with position tie-breaks within each block.
#[test]
fn gpu_tree_topology_heavy_duplicates() {
    let sorted_codes: Vec<u32> = (0..5000u32).map(|i| i / 500).collect();
    assert_topology(&sorted_codes);
}

/// A dense monotone chain of distinct codes (`0,1,2,…`) — the deepest, most degenerate
/// (right-leaning) split pattern, exercising the range/split search at full depth.
#[test]
fn gpu_tree_topology_monotone_chain() {
    let sorted_codes: Vec<u32> = (0..4096u32).collect();
    assert_topology(&sorted_codes);
}

/// Large N exercises the tree-build kernel across many workgroups (full 30-bit range).
/// Kept at 2^16 to stay well under the GPU driver watchdog.
#[test]
fn gpu_tree_topology_large_n() {
    let pos = cloud(0xBEEF, 65_536, 5.0, DVec3::ZERO);
    let (sorted_codes, ..) = sorted_inputs(&pos, &vec![1.0; pos.len()]);
    assert_topology(&sorted_codes);
}

// ---------------------------------------------------------------------------
// structural self-consistency (recomputed from the read-back tree)
// ---------------------------------------------------------------------------

/// `2N-1` nodes; leaves are exactly `[0, N)`; every child's `parent` points back; every
/// non-root node has exactly one parent; the root has [`NO_PARENT`].
fn assert_structural(gpu: &GpuLbvhTree) {
    let n = gpu.n;
    assert!(n >= 1);
    let total = 2 * n - 1;
    assert_eq!(gpu.left.len(), n.saturating_sub(1), "left len = N-1");
    assert_eq!(gpu.right.len(), n.saturating_sub(1));
    assert_eq!(gpu.parent.len(), total, "parent len = 2N-1");
    assert_eq!(gpu.aabb_min.len(), total);
    assert_eq!(gpu.aabb_max.len(), total);
    assert_eq!(gpu.com.len(), total);
    assert_eq!(gpu.mass.len(), total);

    let root = if n == 1 { 0 } else { n }; // internal 0 = unified N (or the lone leaf)
    let mut child_count = vec![0u32; total];
    for i in 0..n.saturating_sub(1) {
        let me = (n + i) as u32;
        let (l, r) = (gpu.left[i], gpu.right[i]);
        assert!(
            (l as usize) < total && (r as usize) < total,
            "child index in range"
        );
        assert_ne!(l, r, "internal node's two children must differ");
        assert_eq!(
            gpu.parent[l as usize], me,
            "left child's parent must point back"
        );
        assert_eq!(
            gpu.parent[r as usize], me,
            "right child's parent must point back"
        );
        child_count[l as usize] += 1;
        child_count[r as usize] += 1;
    }
    for (u, &cc) in child_count.iter().enumerate() {
        if u == root {
            assert_eq!(cc, 0, "root must not be any node's child");
            assert_eq!(gpu.parent[u], NO_PARENT, "root parent must be NO_PARENT");
        } else {
            assert_eq!(cc, 1, "every non-root node has exactly one parent");
        }
    }
}

// ---------------------------------------------------------------------------
// aggregation — AABB bit-exact, com/mass tolerance
// ---------------------------------------------------------------------------

/// Full check: topology bit-exact, AABB bit-exact vs the f64 reference over the same
/// f32-narrowed leaves, com/mass within an f32 tolerance, child AABB ⊆ parent, root AABB
/// contains all inputs. `r` is the position box half-width (the com tolerance scale).
fn assert_full(pos: &[DVec3], mass: &[f64], r: f64) {
    let (sorted_codes, sorted_pos, sorted_mass) = sorted_inputs(pos, mass);
    assert_full_with_codes(&sorted_codes, &sorted_pos, &sorted_mass, r);
}

/// Like [`assert_full`] but with the sorted codes supplied directly — lets a test drive
/// a chosen *topology* (e.g. a degenerate monotone chain) with arbitrary leaf payload,
/// since [`reference_aggregate`] folds `(tree, pos, mass)` independently of the codes.
fn assert_full_with_codes(sorted_codes: &[u32], sorted_pos: &[DVec3], sorted_mass: &[f64], r: f64) {
    let n = sorted_codes.len();
    let gpu = builder().build(sorted_codes, sorted_pos, sorted_mass);
    let refr: KarrasTree = reference_karras(sorted_codes);

    // Topology bit-exact.
    assert_eq!(gpu.left, refr.left);
    assert_eq!(gpu.right, refr.right);
    assert_eq!(gpu.parent, refr.parent);
    assert_structural(&gpu);

    // Reference aggregate over the SAME f32-narrowed leaves the GPU folds.
    let pos_f = narrow(sorted_pos);
    let mass_f: Vec<f64> = sorted_mass.iter().map(|&m| m as f32 as f64).collect();
    let agg = reference_aggregate(&refr, &pos_f, &mass_f);

    let total = 2 * n - 1;
    // Loose f32 tolerances: com error scales with the box size and fold depth (~log N);
    // mass is a fixed-order f32 sum. Both are far tighter than an O(box) topology bug.
    let com_tol = 1e-3 * r + 1e-5;
    for u in 0..total {
        let gmin = gpu.aabb_min[u];
        let gmax = gpu.aabb_max[u];
        let rmin = agg.aabb_min[u].to_array();
        let rmax = agg.aabb_max[u].to_array();
        let gcom = gpu.com[u];
        let rcom = agg.com[u].to_array();
        for a in 0..3 {
            // AABB min/max are exact under widening (min/max just picks a leaf coord).
            assert_eq!(
                gmin[a] as f64, rmin[a],
                "aabb_min bit-exact @node {u} axis {a}"
            );
            assert_eq!(
                gmax[a] as f64, rmax[a],
                "aabb_max bit-exact @node {u} axis {a}"
            );
            assert!(
                (gcom[a] as f64 - rcom[a]).abs() <= com_tol,
                "com within f32 tolerance @node {u} axis {a}: {} vs {}",
                gcom[a],
                rcom[a]
            );
        }
        assert!(
            (gpu.mass[u] as f64 - agg.mass[u]).abs() <= 1e-4 * agg.mass[u].max(1.0),
            "mass within f32 tolerance @node {u}"
        );
    }

    // Child AABB ⊆ parent AABB (in the GPU's own f32 space).
    for i in 0..n.saturating_sub(1) {
        let parent = n + i;
        for &c in &[gpu.left[i] as usize, gpu.right[i] as usize] {
            for a in 0..3 {
                assert!(
                    gpu.aabb_min[c][a] >= gpu.aabb_min[parent][a],
                    "child min inside parent @axis {a}"
                );
                assert!(
                    gpu.aabb_max[c][a] <= gpu.aabb_max[parent][a],
                    "child max inside parent @axis {a}"
                );
            }
        }
    }

    // Root AABB contains every input position (f32-narrowed).
    let root = if n == 1 { 0 } else { n };
    for p in &pos_f {
        for (a, &pv) in p.to_array().iter().enumerate() {
            assert!(gpu.aabb_min[root][a] as f64 <= pv, "root min bounds all");
            assert!(gpu.aabb_max[root][a] as f64 >= pv, "root max bounds all");
        }
    }
}

/// Aggregation over seeded clouds with varied positive masses.
#[test]
fn gpu_tree_aggregation_matches_reference() {
    for seed in [3u64, 17, 271] {
        let pos = cloud(seed, 4000, 3.0, DVec3::ZERO);
        let mass = masses(seed ^ 0xABCD, pos.len());
        assert_full(&pos, &mass, 3.0);
    }
}

/// Coincident leaves (all at one point) → zero-extent AABBs everywhere; the degenerate
/// fold must still bit-match (min == max == the point) and com == the point.
#[test]
fn gpu_tree_aggregation_coincident() {
    let pos = vec![DVec3::new(1.5, -2.0, 0.25); 512];
    let mass = masses(99, pos.len());
    assert_full(&pos, &mass, 2.0);
}

/// Aggregation over a **monotone-chain** topology — the deepest, right-leaning tree,
/// where the single-invocation aggregation walk performs up to N-1 folds in one leaf's
/// cascade (the path unique to this design's serial fold). Synthetic distinct codes
/// force the chain; the leaf payload is an arbitrary cloud (positions are independent of
/// the codes for `reference_aggregate`), so this gates the deep cascade's *values*, not
/// just its topology.
#[test]
fn gpu_tree_aggregation_monotone_chain() {
    let n = 2048;
    let sorted_codes: Vec<u32> = (0..n as u32).collect();
    let pos = cloud(0xC0FFEE, n, 4.0, DVec3::ZERO);
    let mass = masses(0xBEE, n);
    assert_full_with_codes(&sorted_codes, &pos, &mass, 4.0);
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run) — topology AND aggregation
// ---------------------------------------------------------------------------

/// Same input ⇒ bit-identical topology *and* aggregation on a given device. With the
/// current single-invocation aggregation this is near-vacuous (a serial kernel is
/// deterministic regardless of fold order) — its real job is a **regression guard for
/// the deferred parallel atomic-flag refit**, where the cross-workgroup fold ordering
/// could reintroduce nondeterminism (the same role the M4d sort's determinism gate plays).
#[test]
fn gpu_tree_is_bit_deterministic() {
    let pos = cloud(0x50, 5000, 3.0, DVec3::ZERO);
    let mass = masses(0x51, pos.len());
    let (sorted_codes, sorted_pos, sorted_mass) = sorted_inputs(&pos, &mass);
    let mut b = builder();
    let a = b.build(&sorted_codes, &sorted_pos, &sorted_mass);
    let c = b.build(&sorted_codes, &sorted_pos, &sorted_mass);
    assert_eq!(a.left, c.left);
    assert_eq!(a.right, c.right);
    assert_eq!(a.parent, c.parent);
    assert_eq!(a.aabb_min, c.aabb_min, "aabb_min run-to-run deterministic");
    assert_eq!(a.aabb_max, c.aabb_max);
    assert_eq!(a.com, c.com, "com run-to-run deterministic");
    assert_eq!(a.mass, c.mass);
}

// ---------------------------------------------------------------------------
// edge cases
// ---------------------------------------------------------------------------

/// N=1: a single leaf is the whole tree — no internal nodes, `NO_PARENT`, AABB = its
/// position, com = its position, mass = its mass.
#[test]
fn gpu_tree_single() {
    let pos = vec![DVec3::new(2.0, -1.0, 0.5)];
    let mass = vec![1.25];
    let (sorted_codes, sorted_pos, sorted_mass) = sorted_inputs(&pos, &mass);
    let gpu = builder().build(&sorted_codes, &sorted_pos, &sorted_mass);
    assert_eq!(gpu.n, 1);
    assert!(gpu.left.is_empty() && gpu.right.is_empty());
    assert_eq!(gpu.parent, vec![NO_PARENT]);
    let p = [2.0f32, -1.0, 0.5];
    assert_eq!(gpu.aabb_min[0], p);
    assert_eq!(gpu.aabb_max[0], p);
    assert_eq!(gpu.com[0], p);
    assert_eq!(gpu.mass[0], 1.25);
}

/// N=2: exactly one internal (the root, unified index 2) over two leaves.
#[test]
fn gpu_tree_two() {
    let pos = vec![DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)];
    let mass = vec![1.0, 3.0];
    assert_full(&pos, &mass, 1.0);
    let (sc, sp, sm) = sorted_inputs(&pos, &mass);
    let gpu = builder().build(&sc, &sp, &sm);
    assert_eq!(gpu.n, 2);
    assert_eq!(gpu.left.len(), 1);
    assert_eq!(gpu.parent.len(), 3);
    // Root (unified 2) is the parent of both leaves.
    assert_eq!(gpu.parent[0], 2);
    assert_eq!(gpu.parent[1], 2);
    assert_eq!(gpu.parent[2], NO_PARENT);
}

/// N=0: empty input yields an empty tree and does not dispatch or panic.
#[test]
fn gpu_tree_empty() {
    let gpu = builder().build(&[], &[], &[]);
    assert_eq!(gpu.n, 0);
    assert!(gpu.left.is_empty());
    assert!(gpu.right.is_empty());
    assert!(gpu.parent.is_empty());
    assert!(gpu.aabb_min.is_empty());
    assert!(gpu.mass.is_empty());
}
