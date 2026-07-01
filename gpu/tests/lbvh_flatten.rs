//! GPU DFS skip-pointer flatten (DESIGN M4f) validated against the CPU reference
//! [`galaxy_solvers::reference_flatten`] — the fourth GPU-resident LBVH-build stage,
//! turning the M4e Karras pointer tree into the stackless DFS pre-order skip-pointer form
//! [`galaxy_solvers::LbvhFlat`] that the deferred `GpuLbvh` traversal (M4g) will walk.
//!
//! Two gates, split exactly like M4e's build/aggregate:
//!
//! * **Structure is a pure-integer function of the topology ⇒ bit-exact.** Given the same
//!   `sorted_codes`, the DFS pre-order is deterministic, so per-slot `next` / `body_start` /
//!   `body_count` and the whole `leaf_bodies` permutation must equal `reference_flatten`
//!   **bit-for-bit**. This is the load-bearing gate — a dropped/double-counted subtree or a
//!   skip-pointer off-by-one shows up here, not in the geometry.
//! * **Geometry is f32 ⇒ tolerance.** `center`/`half_extents` = `(min±max)/2` round in f32
//!   (the AABB min/max are exact under widening, but the halving sum is not), `com`/`mass`
//!   are f32-lossy folds, and `delta = |com − center|` is an f32-lossy sqrt — all toleranced
//!   vs the f64 reference run over the SAME f32-narrowed leaves.
//!
//! Plus topology-free invariants that catch a bad flatten without leaning on the reference:
//! a full-open skip-pointer walk (the θ→0 traversal's structural core) must visit each body
//! exactly once; skip pointers strictly increase; leaves/internals count `N`/`N-1`.
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuLbvhFlattener::new` returns
//! `NoAdapter` and the tests fail loudly (matches the M4c/M4d/M4e convention).

use galaxy_core::DVec3;
use galaxy_gpu::{GpuLbvhFlat, GpuLbvhFlattener};
use galaxy_solvers::{
    reference_aggregate, reference_flatten, reference_karras, reference_morton, reference_sort,
    LbvhFlat,
};

fn flattener() -> GpuLbvhFlattener {
    GpuLbvhFlattener::new().expect("wgpu adapter required for GPU flatten tests")
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
/// mass-weighted centre of mass is a real (f32-lossy) fold.
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

/// Gather positions/masses/codes into Morton-sorted order and keep the permutation —
/// the exact pipeline prefix (`reference_morton` → `reference_sort` → gather) the GPU
/// sort produces before the build/aggregate/flatten stages. `order[k]` is the original
/// index of the k-th sorted leaf (what the flatten writes into `leaf_bodies`).
fn sorted_inputs(pos: &[DVec3], mass: &[f64]) -> (Vec<u32>, Vec<DVec3>, Vec<f64>, Vec<u32>) {
    let codes = reference_morton(pos).codes;
    let order = reference_sort(&codes);
    let sorted_codes = order.iter().map(|&i| codes[i as usize]).collect();
    let sorted_pos = order.iter().map(|&i| pos[i as usize]).collect();
    let sorted_mass = order.iter().map(|&i| mass[i as usize]).collect();
    (sorted_codes, sorted_pos, sorted_mass, order)
}

/// Round a f64 vector through f32 and back — the exact leaf value the GPU sees
/// (`p.x as f32`), so the f64 reference folds over identical inputs.
fn narrow(pos: &[DVec3]) -> Vec<DVec3> {
    pos.iter()
        .map(|p| DVec3::new(p.x as f32 as f64, p.y as f32 as f64, p.z as f32 as f64))
        .collect()
}

/// The f64 reference flat form over the SAME f32-narrowed leaves the GPU flattens —
/// staged exactly as `LbvhFlat::build` (karras → aggregate → flatten).
fn reference(
    sorted_codes: &[u32],
    sorted_pos: &[DVec3],
    sorted_mass: &[f64],
    order: &[u32],
) -> LbvhFlat {
    let tree = reference_karras(sorted_codes);
    let pos_f = narrow(sorted_pos);
    let mass_f: Vec<f64> = sorted_mass.iter().map(|&m| m as f32 as f64).collect();
    let agg = reference_aggregate(&tree, &pos_f, &mass_f);
    reference_flatten(&tree, &agg, order)
}

/// Full-open skip-pointer walk of the GPU flat — the structural core of the θ→0
/// traversal (every internal opens to `node+1`; a leaf emits its bodies then jumps to
/// `next`). Returns the leaf bodies in visitation order. A dropped or double-counted
/// subtree, or a skip pointer that doesn't strictly increase, corrupts this immediately.
fn walk_leaves(flat: &GpuLbvhFlat) -> Vec<u32> {
    let total = flat.next.len() as u32;
    let mut out = Vec::new();
    let mut node = 0u32;
    let mut steps = 0u32;
    while node < total {
        assert!(
            steps <= 2 * total + 2,
            "walk did not terminate (bad skip pointers)"
        );
        steps += 1;
        let ni = node as usize;
        if flat.body_count[ni] > 0 {
            let s = flat.body_start[ni] as usize;
            for k in s..s + flat.body_count[ni] as usize {
                out.push(flat.leaf_bodies[k]);
            }
            node = flat.next[ni];
        } else {
            node += 1; // open
        }
    }
    out
}

/// Structure bit-exact vs `reference_flatten`, geometry within f32 tolerance, plus the
/// topology-free invariants. `r` is the position box half-width (the geometry tol scale).
fn assert_flat(pos: &[DVec3], mass: &[f64], r: f64) {
    let (sc, sp, sm, order) = sorted_inputs(pos, mass);
    let n = sc.len();
    let total = 2 * n - 1;
    let gpu = flattener().build_flat(&sc, &sp, &sm, &order);
    let refr = reference(&sc, &sp, &sm, &order);

    assert_eq!(gpu.n, n, "N mismatch");
    assert_eq!(gpu.next.len(), total, "next len = 2N-1");
    assert_eq!(gpu.center.len(), total);
    assert_eq!(gpu.half_extents.len(), total);
    assert_eq!(gpu.com.len(), total);
    assert_eq!(gpu.mass.len(), total);
    assert_eq!(gpu.delta.len(), total);
    assert_eq!(gpu.body_start.len(), total);
    assert_eq!(gpu.body_count.len(), total);
    assert_eq!(gpu.leaf_bodies.len(), n, "leaf_bodies len = N");

    // --- structure: pure-integer DFS layout, bit-exact ---
    for d in 0..total {
        assert_eq!(
            gpu.next[d], refr.nodes[d].next,
            "next skip pointer bit-exact @slot {d}"
        );
        assert_eq!(
            gpu.body_start[d], refr.nodes[d].body_start,
            "body_start bit-exact @slot {d}"
        );
        assert_eq!(
            gpu.body_count[d], refr.nodes[d].body_count,
            "body_count bit-exact @slot {d}"
        );
    }
    assert_eq!(
        gpu.leaf_bodies, refr.leaf_bodies,
        "leaf_bodies permutation bit-exact vs reference_flatten"
    );

    // --- geometry: f32 tolerance vs the f64 reference over the same narrowed leaves ---
    let tol = 1e-3 * r + 1e-5;
    for d in 0..total {
        let rc = refr.nodes[d].center.to_array();
        let rh = refr.nodes[d].half_extents.to_array();
        let rcom = refr.nodes[d].com.to_array();
        for a in 0..3 {
            assert!(
                (gpu.center[d][a] as f64 - rc[a]).abs() <= tol,
                "center within f32 tol @slot {d} axis {a}: {} vs {}",
                gpu.center[d][a],
                rc[a]
            );
            assert!(
                (gpu.half_extents[d][a] as f64 - rh[a]).abs() <= tol,
                "half_extents within f32 tol @slot {d} axis {a}: {} vs {}",
                gpu.half_extents[d][a],
                rh[a]
            );
            assert!(
                (gpu.com[d][a] as f64 - rcom[a]).abs() <= tol,
                "com within f32 tol @slot {d} axis {a}: {} vs {}",
                gpu.com[d][a],
                rcom[a]
            );
        }
        assert!(
            (gpu.mass[d] as f64 - refr.nodes[d].mass).abs() <= 1e-4 * refr.nodes[d].mass.max(1.0),
            "mass within f32 tol @slot {d}"
        );
        assert!(
            (gpu.delta[d] as f64 - refr.nodes[d].delta).abs() <= tol,
            "delta within f32 tol @slot {d}: {} vs {}",
            gpu.delta[d],
            refr.nodes[d].delta
        );
    }

    // --- topology-free invariants ---
    // skip pointers strictly increase and never overrun.
    for d in 0..total {
        assert!(
            gpu.next[d] as usize > d && gpu.next[d] as usize <= total,
            "skip pointer must strictly increase and stay in range @slot {d}: {}",
            gpu.next[d]
        );
    }
    assert_eq!(
        gpu.next[0] as usize, total,
        "root's subtree spans the whole tree"
    );
    // leaf/internal split.
    let leaves = gpu.body_count.iter().filter(|&&c| c == 1).count();
    let internals = gpu.body_count.iter().filter(|&&c| c == 0).count();
    assert_eq!(leaves, n, "exactly N leaves");
    assert_eq!(internals, n - 1, "exactly N-1 internal nodes");
    for d in 0..total {
        assert!(
            gpu.body_count[d] <= 1,
            "an LBVH leaf holds exactly one body"
        );
    }
    // a full-open walk visits each body exactly once.
    let mut visited = walk_leaves(&gpu);
    assert_eq!(visited.len(), n, "full-open walk visits N bodies");
    assert_eq!(
        visited, gpu.leaf_bodies,
        "walk order == leaf_bodies (DFS order)"
    );
    visited.sort_unstable();
    assert_eq!(
        visited,
        (0..n as u32).collect::<Vec<_>>(),
        "leaf bodies are a permutation of 0..N"
    );
}

// ---------------------------------------------------------------------------
// clouds — structure + geometry over real Morton topologies
// ---------------------------------------------------------------------------

/// Seeded clouds with varied positive masses: the common case.
#[test]
fn gpu_flatten_matches_reference() {
    for seed in [3u64, 17, 271] {
        let pos = cloud(seed, 4000, 3.0, DVec3::ZERO);
        let mass = masses(seed ^ 0xABCD, pos.len());
        assert_flat(&pos, &mass, 3.0);
    }
}

/// **Monotone-chain topology** — synthetic distinct codes force the deepest, most
/// degenerate right-leaning tree (depth N-1). This is the case that overflows any
/// fixed-size DFS stack, so it gates the serial flatten's subtree-size prefix approach
/// at full depth; the leaf payload is an arbitrary cloud.
#[test]
fn gpu_flatten_monotone_chain() {
    let n = 2048;
    let sc: Vec<u32> = (0..n as u32).collect();
    let pos = cloud(0xC0FFEE, n, 4.0, DVec3::ZERO);
    let mass = masses(0xBEE, n);
    // Codes are already sorted & distinct; order is identity (no gather permutation).
    let order: Vec<u32> = (0..n as u32).collect();
    let gpu = flattener().build_flat(&sc, &pos, &mass, &order);
    let refr = reference(&sc, &pos, &mass, &order);
    assert_eq!(gpu.leaf_bodies, refr.leaf_bodies);
    for d in 0..(2 * n - 1) {
        assert_eq!(gpu.next[d], refr.nodes[d].next, "chain next @slot {d}");
        assert_eq!(gpu.body_count[d], refr.nodes[d].body_count);
    }
    let mut visited = walk_leaves(&gpu);
    visited.sort_unstable();
    assert_eq!(visited, (0..n as u32).collect::<Vec<_>>());
}

/// All-equal codes — every node on the `32 + clz` position tie-break (fully left-leaning
/// tie tree): a different deep degeneracy from the monotone chain.
#[test]
fn gpu_flatten_all_equal_codes() {
    let n = 1500;
    let sc = vec![42u32; n];
    let pos = cloud(0x5A, n, 2.0, DVec3::ZERO);
    let mass = masses(0x5B, n);
    let order: Vec<u32> = (0..n as u32).collect();
    let gpu = flattener().build_flat(&sc, &pos, &mass, &order);
    let refr = reference(&sc, &pos, &mass, &order);
    assert_eq!(gpu.leaf_bodies, refr.leaf_bodies);
    for d in 0..(2 * n - 1) {
        assert_eq!(gpu.next[d], refr.nodes[d].next, "tie next @slot {d}");
    }
    let mut visited = walk_leaves(&gpu);
    visited.sort_unstable();
    assert_eq!(visited, (0..n as u32).collect::<Vec<_>>());
}

/// Coincident leaves (all at one point) → zero-extent AABBs; center == com == the point,
/// half == 0, delta == 0 everywhere.
#[test]
fn gpu_flatten_coincident() {
    let pos = vec![DVec3::new(1.5, -2.0, 0.25); 512];
    let mass = masses(99, pos.len());
    assert_flat(&pos, &mass, 2.0);
}

/// Large N exercises the flatten across the full DFS with many nodes.
#[test]
fn gpu_flatten_large_n() {
    let pos = cloud(0xBEEF, 40_000, 5.0, DVec3::ZERO);
    let mass = masses(0xF00D, pos.len());
    assert_flat(&pos, &mass, 5.0);
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run)
// ---------------------------------------------------------------------------

/// Same input ⇒ bit-identical flat form on a given device. With the current
/// single-invocation flatten this is near-vacuous (serial ⇒ order-free), but it is the
/// regression guard for the deferred parallel Euler-tour refit (same role as M4d/M4e).
#[test]
fn gpu_flatten_is_bit_deterministic() {
    let pos = cloud(0x50, 5000, 3.0, DVec3::ZERO);
    let mass = masses(0x51, pos.len());
    let (sc, sp, sm, order) = sorted_inputs(&pos, &mass);
    let mut f = flattener();
    let a = f.build_flat(&sc, &sp, &sm, &order);
    let b = f.build_flat(&sc, &sp, &sm, &order);
    assert_eq!(a.next, b.next);
    assert_eq!(a.body_start, b.body_start);
    assert_eq!(a.body_count, b.body_count);
    assert_eq!(a.leaf_bodies, b.leaf_bodies);
    assert_eq!(a.center, b.center);
    assert_eq!(a.half_extents, b.half_extents);
    assert_eq!(a.com, b.com);
    assert_eq!(a.mass, b.mass);
    assert_eq!(a.delta, b.delta);
}

// ---------------------------------------------------------------------------
// edge cases
// ---------------------------------------------------------------------------

/// N=1: a single leaf is the whole flat tree — one node, `next = 1`, one body, geometry
/// = its position, delta 0.
#[test]
fn gpu_flatten_single() {
    let pos = vec![DVec3::new(2.0, -1.0, 0.5)];
    let mass = vec![1.25];
    let (sc, sp, sm, order) = sorted_inputs(&pos, &mass);
    let gpu = flattener().build_flat(&sc, &sp, &sm, &order);
    assert_eq!(gpu.n, 1);
    assert_eq!(gpu.next, vec![1]);
    assert_eq!(gpu.body_start, vec![0]);
    assert_eq!(gpu.body_count, vec![1]);
    assert_eq!(gpu.leaf_bodies, vec![order[0]]);
    let p = [2.0f32, -1.0, 0.5];
    assert_eq!(gpu.center[0], p);
    assert_eq!(gpu.half_extents[0], [0.0, 0.0, 0.0]);
    assert_eq!(gpu.com[0], p);
    assert_eq!(gpu.mass[0], 1.25);
    assert_eq!(gpu.delta[0], 0.0);
}

/// N=2: root (slot 0) spans both leaves (slots 1, 2); leaves have `next = self+1`.
#[test]
fn gpu_flatten_two() {
    let pos = vec![DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)];
    let mass = vec![1.0, 3.0];
    assert_flat(&pos, &mass, 1.0);
    let (sc, sp, sm, order) = sorted_inputs(&pos, &mass);
    let gpu = flattener().build_flat(&sc, &sp, &sm, &order);
    assert_eq!(gpu.n, 2);
    assert_eq!(
        gpu.next,
        vec![3, 2, 3],
        "root spans all; leaves point to self+1"
    );
    assert_eq!(gpu.body_count, vec![0, 1, 1]);
}

/// N=0: empty input yields an empty flat form and does not dispatch or panic.
#[test]
fn gpu_flatten_empty() {
    let gpu = flattener().build_flat(&[], &[], &[], &[]);
    assert_eq!(gpu.n, 0);
    assert!(gpu.next.is_empty());
    assert!(gpu.leaf_bodies.is_empty());
    assert!(gpu.center.is_empty());
    assert!(gpu.mass.is_empty());
}
