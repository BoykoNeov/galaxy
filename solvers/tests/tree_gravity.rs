//! I-grav (milestones 10+11, merged) — the stale-tree active gravity walk.
//!
//! `hydro+gravity` mode subcycles gravity: build the Barnes-Hut tree ONCE per base
//! block, then on every fine tick walk only the ACTIVE subset against that cached
//! tree at the CURRENT (drifted) positions. Because the individual stepper drifts
//! ALL particles every fine tick, the near-field (opened-leaf) sources read from the
//! current `state.pos` are exact — the "predict inactive neighbours" of the plan is
//! achieved by drift-all, not a separate predictor. Only the far-field cell
//! multipoles (baked into the `FlatNode` COMs at build) are stale: a bounded,
//! *converging* approximation (the reason stale-tree works for long-range gravity
//! where a stale *grid* did not for short-range hydro, whose dense-knot h is tiny).
//!
//! GATE DESIGN:
//!   * ANCHOR — a FRESH cache (rebuilt at the current state) walked over ALL targets
//!     is BIT-IDENTICAL to `BarnesHut::accelerations`: `FlatTree::accel` reproduces
//!     `Octree::accel_node` bit-for-bit and the fill is a pure per-target map, so
//!     zero drift + all active ⇒ the walk IS the full solver.
//!   * SUBSET — on that fresh cache a subset walk equals the full accel at exactly
//!     the active indices (picks the right targets; writes only them).
//!   * CONVERGENCE — the load-bearing one: cache at p0, walk at p1 = p0 + v·δ (the
//!     stale far-COMs, current near-field), vs a fresh rebuild at p1. The error → 0
//!     as δ → 0 (exactly 0 at δ = 0) and shrinks with δ ⇒ far-COM staleness is a
//!     converging approximation, not a fixed bias.

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::{BarnesHut, TreeGravity};

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A star cloud (two offset clumps so the tree has real far/near structure) with a
/// mild shear velocity field (drift moves particles a non-trivial amount).
fn cloud(seed: u64, n: usize) -> State {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    for k in 0..n {
        let clump = if k % 2 == 0 {
            DVec3::new(-1.5, 0.0, 0.0)
        } else {
            DVec3::new(1.5, 0.3, 0.0)
        };
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * 1.2 + clump;
        vel.push(DVec3::new(-p.y, p.x, 0.0) * 0.4 + DVec3::new(0.2, 0.0, 0.0));
        pos.push(p);
    }
    State::from_phase_space(pos, vel, vec![1.0; n])
}

fn bh() -> BarnesHut {
    BarnesHut::new(1.0, 0.05, 0.5)
}

fn full_accel(state: &State) -> Vec<DVec3> {
    let n = state.len();
    let mut a = vec![DVec3::ZERO; n];
    bh().accelerations(state, &mut a);
    a
}

#[test]
fn fresh_cache_all_active_equals_barnes_hut_bit_identical() {
    let state = cloud(1, 400);
    let n = state.len();
    let a_full = full_accel(&state);

    let mut tg = TreeGravity::new(bh());
    tg.rebuild_gravity_cache(&state);
    let all: Vec<usize> = (0..n).collect();
    let mut a_walk = vec![DVec3::ZERO; n];
    tg.gravity_active_cached(&state, &all, &mut a_walk);

    assert_eq!(
        a_walk, a_full,
        "fresh-cache all-active walk must equal BarnesHut::accelerations bit-for-bit"
    );
}

#[test]
fn fresh_cache_subset_matches_full_at_active_indices() {
    let state = cloud(2, 400);
    let n = state.len();
    let a_full = full_accel(&state);

    let mut tg = TreeGravity::new(bh());
    tg.rebuild_gravity_cache(&state);
    let subset: Vec<usize> = (0..n).step_by(5).collect();
    // Sentinel the whole buffer so a mis-targeted write is caught.
    let sentinel = DVec3::new(f64::NAN, 0.0, 0.0);
    let mut a = vec![sentinel; n];
    tg.gravity_active_cached(&state, &subset, &mut a);
    for &i in &subset {
        assert_eq!(a[i], a_full[i], "subset walk differs from full at active {i}");
    }
}

#[test]
fn stale_walk_converges_to_the_rebuilt_reference_as_drift_shrinks() {
    let state0 = cloud(3, 500);
    let n = state0.len();
    let all: Vec<usize> = (0..n).collect();

    // Cache the tree at p0 (block start) ONCE; walk the drifted states without
    // rebuilding — the stale far-COMs, current near-field.
    let mut tg = TreeGravity::new(bh());
    tg.rebuild_gravity_cache(&state0);

    let err_at = |tg: &mut TreeGravity, delta: f64| -> f64 {
        let mut s = state0.clone();
        for (x, v) in s.pos.iter_mut().zip(&state0.vel) {
            *x += *v * delta;
        }
        let mut a_stale = vec![DVec3::ZERO; n];
        tg.gravity_active_cached(&s, &all, &mut a_stale); // cached tree (p0) + current pos (p1)
        let a_fresh = full_accel(&s); // rebuild-every-tick reference at p1
        a_stale
            .iter()
            .zip(&a_fresh)
            .map(|(x, y)| (*x - *y).length())
            .fold(0.0_f64, f64::max)
    };

    // Zero drift: the stale walk IS the fresh walk (same positions) — exactly 0.
    assert_eq!(err_at(&mut tg, 0.0), 0.0, "zero-drift stale walk must be exact");

    let e_big = err_at(&mut tg, 0.02);
    let e_small = err_at(&mut tg, 0.01);
    assert!(e_big > 0.0, "a real drift must produce a nonzero staleness error");
    assert!(
        e_small < e_big,
        "far-COM staleness must CONVERGE: err({:.3}) = {e_small:.3e} !< err({:.3}) = {e_big:.3e}",
        0.01,
        0.02
    );
}
