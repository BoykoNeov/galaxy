//! Shared softened gravitational potential energy.
//!
//! Both solvers report the SAME exact O(N²) softened potential (BarnesHut does
//! not tree-accelerate the potential — it's a periodic diagnostic, not the
//! per-step path). Keeping the kernel in one place guarantees they stay
//! identical and lets both share the parallel reduction.
//!
//! The softened pair potential is `-G mᵢ mⱼ / sqrt(rᵢⱼ² + ε²)`, matching the
//! Plummer-softened force so energy conservation stays consistent (force = -∇U).

use galaxy_core::{DVec3, State};
use rayon::prelude::*;

/// Sum of the softened pair potential over particle `i` against all `j > i`.
/// Extracted so the serial and parallel drivers share one inner kernel.
#[inline]
fn row_potential(i: usize, pos: &[DVec3], mass: &[f64], g: f64, eps2: f64) -> f64 {
    let n = pos.len();
    let mut u = 0.0;
    for j in (i + 1)..n {
        let dx = pos[j] - pos[i];
        let r = (dx.length_squared() + eps2).sqrt();
        u -= g * mass[i] * mass[j] / r;
    }
    u
}

/// Serial reference: the exact nested-loop sum. Reproducible to the last bit and
/// used as the equivalence oracle for the parallel reduction.
pub fn potential_energy_serial(state: &State, g: f64, softening: f64) -> f64 {
    let eps2 = softening * softening;
    let n = state.len();
    let mut u = 0.0;
    for i in 0..n {
        u += row_potential(i, &state.pos, &state.mass, g, eps2);
    }
    u
}

/// Parallel reduction over the outer (per-`i`) rows. rayon splits the row range
/// and folds sub-sums, so the result reassociates the floating-point sum — equal
/// to `potential_energy_serial` only to a tight relative tolerance, NOT bit-for-
/// bit. Uses the global rayon pool.
///
/// NOTE: the fold shape depends on the thread count, so this value is not
/// bit-reproducible across machines / `RAYON_NUM_THREADS` (differs at ~1e-13
/// relative). That is fine — it feeds only energy *diagnostics*
/// (`core::diagnostics`), never the stepping path, so simulation trajectories
/// stay fully deterministic; and every consumer compares it with a relative
/// tolerance (validate uses ≥1e-9). Do NOT diff this number bit-exactly.
pub fn potential_energy_parallel(state: &State, g: f64, softening: f64) -> f64 {
    let eps2 = softening * softening;
    let n = state.len();
    (0..n)
        .into_par_iter()
        .map(|i| row_potential(i, &state.pos, &state.mass, g, eps2))
        .sum()
}
