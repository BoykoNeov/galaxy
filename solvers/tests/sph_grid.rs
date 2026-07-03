//! Hash-grid neighbor search vs the O(N²) oracle (DESIGN.md M7a): the grid must
//! return EXACTLY the oracle's neighbor sets — same indices, same (ascending)
//! order — on uniform, clustered, wall-straddling, and coincident configurations.
//! This is the house discipline: the fast spatial structure is bit-gated against
//! brute force before anything downstream is allowed to consume it.

use galaxy_core::DVec3;
use galaxy_solvers::sph::{reference_neighbours, HashGrid};
use proptest::prelude::*;

/// Deterministic pseudo-random points via a small LCG (no external rand dep).
fn random_points(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * scale)
        .collect()
}

/// Grid vs oracle over every particle as query center, plus a few off-particle
/// centers. Exact equality: same indices, ascending.
fn assert_matches_oracle(pos: &[DVec3], cell: f64, r: f64) {
    let grid = HashGrid::build(pos, cell);
    for (i, &c) in pos.iter().enumerate() {
        let got = grid.neighbours_within(pos, c, r);
        let want = reference_neighbours(pos, c, r);
        assert_eq!(got, want, "neighbor set mismatch at particle {i} (r={r})");
    }
    for c in [
        DVec3::ZERO,
        DVec3::new(0.5 * cell, 0.5 * cell, 0.5 * cell),
        DVec3::new(-3.7, 2.2, 0.1),
    ] {
        assert_eq!(
            grid.neighbours_within(pos, c, r),
            reference_neighbours(pos, c, r),
            "neighbor set mismatch at off-particle center {c:?}"
        );
    }
}

#[test]
fn oracle_is_sorted_ascending_and_inclusive() {
    // Pin the oracle's own contract first (it defines the semantics): ascending
    // indices, boundary distance INCLUDED, the center particle included.
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0), // exactly at r = 1
        DVec3::new(0.0, 2.0, 0.0), // outside
        DVec3::new(0.5, 0.0, 0.0),
    ];
    let got = reference_neighbours(&pos, pos[0], 1.0);
    assert_eq!(
        got,
        vec![0, 1, 3],
        "≤ r inclusive, self included, ascending"
    );
}

#[test]
fn uniform_lattice_matches_oracle() {
    let mut pos = Vec::new();
    for x in 0..6 {
        for y in 0..6 {
            for z in 0..6 {
                pos.push(DVec3::new(x as f64, y as f64, z as f64));
            }
        }
    }
    assert_matches_oracle(&pos, 1.5, 1.25);
}

#[test]
fn random_cloud_matches_oracle_at_multiple_radii() {
    let pos = random_points(42, 500, 10.0);
    for r in [0.5, 1.0, 2.7] {
        // r < cell, r = cell, r > cell (multi-cell walk) all covered.
        assert_matches_oracle(&pos, 1.0, r);
    }
}

#[test]
fn clustered_cloud_matches_oracle() {
    // Two dense blobs + far outliers: exercises very uneven cell occupancy.
    let mut pos = random_points(7, 200, 0.5);
    pos.extend(
        random_points(8, 200, 0.5)
            .into_iter()
            .map(|p| p + DVec3::new(20.0, 0.0, 0.0)),
    );
    pos.push(DVec3::new(0.0, 500.0, 0.0));
    pos.push(DVec3::new(0.0, 0.0, -500.0));
    assert_matches_oracle(&pos, 0.4, 0.6);
}

#[test]
fn cell_wall_straddlers_match_oracle() {
    // Points exactly ON cell boundaries (integer multiples of the cell edge)
    // and queries whose radius lands exactly on a neighbor: the ≤ boundary and
    // the cell-assignment convention must both agree with brute force.
    let cell = 1.0;
    let mut pos = Vec::new();
    for x in 0..4 {
        for y in 0..4 {
            pos.push(DVec3::new(x as f64 * cell, y as f64 * cell, 0.0));
        }
    }
    pos.push(DVec3::new(0.5, 0.5, 0.0)); // interior straggler
    assert_matches_oracle(&pos, cell, 1.0); // r exactly = lattice spacing
    assert_matches_oracle(&pos, cell, 2.0_f64.sqrt()); // r exactly = diagonal
}

#[test]
fn coincident_particles_all_returned() {
    let mut pos = random_points(3, 50, 5.0);
    let knot = DVec3::new(1.0, -2.0, 0.5);
    let first = pos.len();
    for _ in 0..7 {
        pos.push(knot);
    }
    let grid = HashGrid::build(&pos, 1.0);
    let got = grid.neighbours_within(&pos, knot, 0.0);
    let want = reference_neighbours(&pos, knot, 0.0);
    assert_eq!(got, want);
    assert!(
        want.len() >= 7 && want.contains(&first),
        "all coincident copies must be found at r = 0"
    );
}

#[test]
fn empty_and_single_are_safe() {
    let empty: Vec<DVec3> = Vec::new();
    let grid = HashGrid::build(&empty, 1.0);
    assert_eq!(grid.neighbours_within(&empty, DVec3::ZERO, 5.0), vec![]);

    let one = vec![DVec3::new(0.5, 0.5, 0.5)];
    let grid = HashGrid::build(&one, 1.0);
    assert_eq!(grid.neighbours_within(&one, one[0], 0.1), vec![0]);
    assert_eq!(
        grid.neighbours_within(&one, DVec3::new(10.0, 0.0, 0.0), 0.1),
        vec![]
    );
}

proptest! {
    /// Any cloud, any radius: grid ≡ oracle, exactly.
    #[test]
    fn grid_equals_oracle(
        seed in 0u64..1000,
        n in 0usize..120,
        r in 0.0f64..3.0,
        cell in 0.3f64..2.0,
    ) {
        let pos = random_points(seed, n, 8.0);
        let grid = HashGrid::build(&pos, cell);
        for (i, &c) in pos.iter().enumerate() {
            prop_assert_eq!(
                grid.neighbours_within(&pos, c, r),
                reference_neighbours(&pos, c, r),
                "mismatch at particle {} (n={}, r={}, cell={})", i, n, r, cell
            );
        }
    }
}
