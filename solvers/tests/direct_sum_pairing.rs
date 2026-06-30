//! The DirectSum optimization (Newton's-third-law pairing) must preserve the
//! physics: its accelerations match an independent, unpaired O(N²) reference to
//! machine precision.
//!
//! Note on momentum: pairing's benefit is halving the per-pair force
//! evaluations (the expensive `sqrt`). It does NOT measurably improve momentum
//! drift for an acceleration-returning solver — each body's acceleration carries
//! the *other* body's mass with a different rounding order, and `v += a·dt`
//! round-trips through `a = F/m`. So this asserts equivalence, not a momentum
//! improvement.

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::DirectSum;

/// Deterministic pseudo-random cluster (LCG; no external rand dep).
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
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 2.0);
        vel.push(DVec3::ZERO);
        mass.push(0.1 + 0.9 * next());
    }
    State::from_phase_space(pos, vel, mass)
}

/// Independent unpaired reference: aᵢ = G Σ_{j≠i} mⱼ (xⱼ − xᵢ) / (r² + ε²)^{3/2}.
#[allow(clippy::needless_range_loop)]
fn reference_accelerations(s: &State, g: f64, softening: f64) -> Vec<DVec3> {
    let n = s.len();
    let eps2 = softening * softening;
    let mut acc = vec![DVec3::ZERO; n];
    for i in 0..n {
        let mut a = DVec3::ZERO;
        for j in 0..n {
            if i == j {
                continue;
            }
            let dx = s.pos[j] - s.pos[i];
            let r2 = dx.length_squared() + eps2;
            a += dx * (s.mass[j] / (r2 * r2.sqrt()));
        }
        acc[i] = a * g;
    }
    acc
}

#[test]
fn pairing_matches_unpaired_reference() {
    let s = cluster(0xBEEF, 200);
    let (g, eps) = (1.0, 0.05);
    let reference = reference_accelerations(&s, g, eps);

    let mut acc = vec![DVec3::ZERO; s.len()];
    DirectSum::new(g, eps).accelerations(&s, &mut acc);

    let scale = reference
        .iter()
        .map(|a| a.length())
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let max_rel = acc
        .iter()
        .zip(&reference)
        .map(|(a, r)| (*a - *r).length() / scale)
        .fold(0.0_f64, f64::max);
    assert!(
        max_rel < 1e-13,
        "pairing diverges from unpaired reference: {max_rel:e}"
    );
}
