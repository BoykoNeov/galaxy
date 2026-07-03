//! O(N²) reference oracles for the SPH toolkit, per the house discipline
//! (`reference_morton`/`reference_sort`/… precedent): brute-force, obviously
//! correct, fixed ascending iteration order. The fast paths are gated against
//! these bit-exact — same neighbor sets, same summation order, same bits.

use galaxy_core::DVec3;

/// Brute force: indices `j` (ascending) with `|pos[j] − center| ≤ r`.
pub fn reference_neighbours(pos: &[DVec3], center: DVec3, r: f64) -> Vec<usize> {
    let _ = (pos, center, r);
    todo!("M7a: O(N²) neighbor oracle")
}

/// Brute-force SPH density summation with per-particle ("gather") smoothing
/// lengths: `ρ_i = Σ_j m_j · W(|x_i − x_j|, h_i)`, summed over ascending `j`
/// (including `j = i`).
pub fn reference_density(pos: &[DVec3], mass: &[f64], h: &[f64]) -> Vec<f64> {
    let _ = (pos, mass, h);
    todo!("M7a: O(N²) density oracle")
}
