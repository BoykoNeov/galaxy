//! CPU LBVH reference solver (`Lbvh`) validated against the `DirectSum` oracle.
//!
//! `Lbvh` is a Barnes-Hut monopole solver over a Morton-code Linear BVH (Karras
//! 2012 binary radix tree) — the CPU f64 reference for the deferred GPU-resident
//! Morton/LBVH build. It is a different tree shape from the octree `BarnesHut`, so
//! the physics gates that pin it are the **topology-independent** ones:
//!
//!  1. θ→0 reproduces direct summation to f64 roundoff — a structural invariant that
//!     catches a dropped or double-counted subtree, holds for *any* correct BVH.
//!  2. At finite θ the monopole approximation drops the quadrupole, so the RMS error
//!     is O(θ²): bounded and growing with θ. Bounds are looser than the octree gate
//!     because the LBVH cell size is the AABB longest side (a binary node can be more
//!     elongated than a cubic octree cell), so `s/d` per node runs a little larger.
//!  3. At θ→0 the exact pairwise antisymmetry is recovered ⇒ Σ mᵢaᵢ = 0.
//!
//! Plus structural gates on the Karras tree itself (2N−1 nodes, leaves a permutation
//! of 0..N, binary child structure, AABB containment) and determinism (the index
//! tie-break makes duplicate-Morton topology a deterministic function of the input).

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::{DirectSum, Lbvh, LbvhFlat};

const G: f64 = 1.0;
const EPS: f64 = 0.05;

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

/// θ→0 opens every internal node down to its single-body leaves, so the tree walk
/// *is* direct summation — the clean, topology-independent correctness gate. A
/// flattening or skip-pointer bug that dropped or double-counted a subtree blows this
/// far past f64 roundoff.
#[test]
fn lbvh_theta_to_zero_reproduces_direct_sum() {
    const N: usize = 120;
    let mut solver = Lbvh::new(G, EPS, 1e-6);
    for seed in 0..100u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);
        let got = accel(&mut solver, &s);
        let worst = worst_rel_err(&got, &exact);
        assert!(
            worst < 1e-9,
            "theta->0 must match the oracle to roundoff (seed {seed}): worst rel err {worst:e}"
        );
    }
}

/// At finite θ the monopole approximation drops the quadrupole, so the error is O(θ²):
/// bounded and growing with θ. Bounds carry ≥2× margin and are looser than the octree
/// `BarnesHut` gate (AABB longest-side cell size vs a cubic cell).
#[test]
fn lbvh_finite_theta_error_is_bounded_and_grows_with_theta() {
    const N: usize = 120;
    for seed in 0..100u64 {
        let s = cluster(seed, N);
        let exact = accel(&mut DirectSum::new(G, EPS), &s);

        let lo = accel(&mut Lbvh::new(G, EPS, 0.3), &s);
        let hi = accel(&mut Lbvh::new(G, EPS, 0.6), &s);

        let (rms_lo, worst_lo) = (rms_rel_err(&lo, &exact), worst_rel_err(&lo, &exact));
        let (rms_hi, worst_hi) = (rms_rel_err(&hi, &exact), worst_rel_err(&hi, &exact));

        assert!(rms_lo < 0.02, "theta=0.3 RMS err {rms_lo:e} (seed {seed})");
        assert!(rms_hi < 0.10, "theta=0.6 RMS err {rms_hi:e} (seed {seed})");
        assert!(
            worst_lo < 0.15,
            "theta=0.3 worst err {worst_lo:e} (seed {seed})"
        );
        assert!(
            worst_hi < 0.40,
            "theta=0.6 worst err {worst_hi:e} (seed {seed})"
        );
        assert!(
            rms_hi > rms_lo,
            "RMS error should grow with theta (seed {seed}): {rms_lo:e} -> {rms_hi:e}"
        );
    }
}

/// At θ→0 the tree is exact direct summation, so Σ mᵢaᵢ = 0 to the f64 floor (pairwise
/// antisymmetry recovered when every node opens to its leaves). A kernel that failed to
/// exclude the self term or mis-signed `dx` blows this up to O(1).
#[test]
fn lbvh_conserves_total_momentum_flux_at_theta_zero() {
    const N: usize = 200;
    let mut solver = Lbvh::new(G, EPS, 1e-6);
    for seed in 0..32u64 {
        let s = cluster(seed, N);
        let a = accel(&mut solver, &s);

        let mut net = DVec3::ZERO;
        let mut scale = 0.0;
        for (ai, &mi) in a.iter().zip(&s.mass) {
            net += *ai * mi;
            scale += ai.length() * mi;
        }
        let rel = net.length() / scale.max(1e-300);
        assert!(rel < 1e-9, "net internal force {rel:e} (seed {seed})");
    }
}

/// Karras structure: exactly `2N−1` nodes (N leaves + N−1 internal), the leaves are a
/// permutation of `0..N` (every particle in exactly one leaf), the layout is a strict
/// binary tree (left child at `k+1`, right child at `nodes[k+1].next`, right's subtree
/// ends at the parent's `next`), and every child AABB is contained in its parent's.
#[test]
fn lbvh_karras_structure_is_wellformed() {
    for &n in &[1usize, 2, 3, 5, 17, 128, 257] {
        for seed in 0..8u64 {
            let s = cluster(seed, n);
            let flat = LbvhFlat::build(&s.pos, &s.mass);
            let nodes = &flat.nodes;

            assert_eq!(nodes.len(), 2 * n - 1, "node count (n={n}, seed {seed})");

            // Leaves are a permutation of 0..N; internal count is N-1.
            let mut seen = vec![false; n];
            let (mut leaves, mut internal) = (0usize, 0usize);
            for nd in nodes {
                if nd.body_count > 0 {
                    assert_eq!(nd.body_count, 1, "LBVH leaves hold exactly one body");
                    let b = flat.leaf_bodies[nd.body_start as usize] as usize;
                    assert!(!seen[b], "body {b} in two leaves (n={n}, seed {seed})");
                    seen[b] = true;
                    leaves += 1;
                } else {
                    internal += 1;
                }
            }
            assert_eq!(leaves, n, "leaf count (n={n}, seed {seed})");
            assert_eq!(internal, n - 1, "internal count (n={n}, seed {seed})");
            assert!(seen.iter().all(|&b| b), "every particle is a leaf");

            // Skip pointers strictly increase and stay in range; root covers all.
            let end = nodes.len() as u32;
            for (k, nd) in nodes.iter().enumerate() {
                assert!(
                    nd.next > k as u32 && nd.next <= end,
                    "next out of order at {k} (n={n}, seed {seed})"
                );
            }
            assert_eq!(nodes[0].next, end, "root subtree must span the whole array");

            // Binary structure + AABB containment, checked from the flat form.
            let aabb = |nd: &galaxy_solvers::LbvhNode| {
                (nd.center - nd.half_extents, nd.center + nd.half_extents)
            };
            let contains = |(pmin, pmax): (DVec3, DVec3), (cmin, cmax): (DVec3, DVec3)| {
                let eps = 1e-9;
                cmin.cmpge(pmin - DVec3::splat(eps)).all()
                    && cmax.cmple(pmax + DVec3::splat(eps)).all()
            };
            for (k, nd) in nodes.iter().enumerate() {
                if nd.body_count > 0 {
                    continue; // leaf
                }
                let left = k + 1;
                let right = nodes[left].next as usize;
                assert!(
                    right < nd.next as usize,
                    "right child inside subtree at {k}"
                );
                assert_eq!(
                    nodes[right].next, nd.next,
                    "right child must end the parent subtree at {k}"
                );
                assert!(
                    contains(aabb(nd), aabb(&nodes[left])),
                    "left AABB ⊄ parent at {k}"
                );
                assert!(
                    contains(aabb(nd), aabb(&nodes[right])),
                    "right AABB ⊄ parent at {k}"
                );
            }

            // Root AABB contains every position.
            let (rmin, rmax) = aabb(&nodes[0]);
            let e = DVec3::splat(1e-9);
            for &p in &s.pos {
                assert!(
                    p.cmpge(rmin - e).all() && p.cmple(rmax + e).all(),
                    "root AABB misses a particle (n={n}, seed {seed})"
                );
            }
        }
    }
}

/// The build + walk are a deterministic function of the input, including when Morton
/// codes tie: coincident particles (identical positions) share a code, and the
/// `(code, index)` tie-break must still yield one fixed topology ⇒ bit-identical forces
/// run to run. A tie-break bug surfaces here as a mismatch.
#[test]
fn lbvh_is_deterministic_with_coincident_particles() {
    let mut pos: Vec<DVec3> = Vec::new();
    let mut mass: Vec<f64> = Vec::new();
    // A cluster plus a knot of exactly-coincident particles (degenerate Morton codes).
    let base = cluster(3, 60);
    pos.extend_from_slice(&base.pos);
    mass.extend_from_slice(&base.mass);
    for k in 0..8 {
        pos.push(DVec3::new(0.25, -0.1, 0.4));
        mass.push(0.5 + 0.05 * k as f64);
    }
    let s = State::from_phase_space(pos, mass.iter().map(|_| DVec3::ZERO).collect(), mass);

    let mut solver = Lbvh::new(G, EPS, 0.5);
    let a1 = accel(&mut solver, &s);
    let a2 = accel(&mut solver, &s);
    assert_eq!(a1, a2, "LBVH forces must be bit-deterministic run to run");
}

/// Degenerate sizes must not panic and must be physically trivial: an empty system
/// yields no accelerations; a lone particle feels no force (its only leaf holds just
/// itself, excluded as the self term).
#[test]
fn lbvh_handles_empty_and_single() {
    let mut solver = Lbvh::new(G, EPS, 0.5);

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
