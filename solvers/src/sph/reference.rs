//! O(N²) reference oracles for the SPH toolkit, per the house discipline
//! (`reference_morton`/`reference_sort`/… precedent): brute-force, obviously
//! correct, fixed ascending iteration order. The fast paths are gated against
//! these bit-exact — same neighbor sets, same summation order, same bits.

use galaxy_core::DVec3;

use super::kernel::w;

/// Brute force: indices `j` (ascending) with `|pos[j] − center| ≤ r`.
pub fn reference_neighbours(pos: &[DVec3], center: DVec3, r: f64) -> Vec<usize> {
    let r2 = r * r;
    pos.iter()
        .enumerate()
        .filter(|(_, p)| (**p - center).length_squared() <= r2)
        .map(|(i, _)| i)
        .collect()
}

/// Brute-force SPH density summation with per-particle ("gather") smoothing
/// lengths: `ρ_i = Σ_j m_j · W(|x_i − x_j|, h_i)`, summed over ascending `j`
/// (including `j = i`). Out-of-support terms add an exact `0.0`, so the fast
/// path (which skips them) associates identically — the bit-exact gate relies
/// on this.
pub fn reference_density(pos: &[DVec3], mass: &[f64], h: &[f64]) -> Vec<f64> {
    (0..pos.len())
        .map(|i| {
            let mut rho = 0.0;
            for j in 0..pos.len() {
                rho += mass[j] * w((pos[i] - pos[j]).length(), h[i]);
            }
            rho
        })
        .collect()
}
