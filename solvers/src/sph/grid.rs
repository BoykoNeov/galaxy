//! Uniform hash grid for fixed-radius SPH neighbor queries.
//!
//! Sparse (HashMap-keyed) cells of edge `cell`: build is O(N), and only
//! occupied cells cost memory — a dense array would explode on clouds with far
//! outliers (a 1000-unit flier at cell 0.4 is a 10^10-cell box). Queries GATHER
//! per target over the `ceil(r/cell)`-cell neighborhood and return indices
//! sorted ascending, so any consumer summing over the returned list associates
//! floating-point work in a fixed order (the parallel ≡ serial bit-exactness
//! discipline). Map iteration order is never observed: buckets are keyed
//! lookups and results are sorted. Gated bit-exact against
//! [`super::reference::reference_neighbours`].

use std::collections::HashMap;

use galaxy_core::DVec3;

/// A uniform grid over the particles' positions.
///
/// `cell` is the bin edge length; queries with any radius work (the walk covers
/// `ceil(r/cell)` cells per axis), but `cell ≈` the typical query radius is the
/// efficient regime.
pub struct HashGrid {
    /// Bin edge length.
    cell: f64,
    /// Occupied cells: floor(coord/cell) per axis → particle indices, ascending.
    cells: HashMap<[i64; 3], Vec<u32>>,
}

impl HashGrid {
    /// Bin `pos` into cells of edge `cell` (> 0, finite — precondition; cell
    /// assignment is `floor(coord/cell)`, shared with the query walk so points
    /// exactly on a cell wall land consistently).
    pub fn build(pos: &[DVec3], cell: f64) -> Self {
        assert!(
            cell.is_finite() && cell > 0.0,
            "HashGrid cell must be positive and finite, got {cell}"
        );
        let mut cells: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
        for (i, p) in pos.iter().enumerate() {
            cells.entry(cell_of(*p, cell)).or_default().push(i as u32);
        }
        HashGrid { cell, cells }
    }

    /// Number of particles binned into the cell containing `p` — an O(1)
    /// local-occupancy probe (the adaptive-h bracket seed uses it to estimate
    /// the local spacing without a radius query).
    pub fn bin_len(&self, p: DVec3) -> usize {
        self.cells.get(&cell_of(p, self.cell)).map_or(0, Vec::len)
    }

    /// Indices `j` (ascending) of all particles with `|pos[j] − center| ≤ r`.
    /// `pos` must be the same slice the grid was built from. A particle exactly
    /// at distance `r` IS a neighbor (`≤`, matching the O(N²) oracle).
    pub fn neighbours_within(&self, pos: &[DVec3], center: DVec3, r: f64) -> Vec<usize> {
        if self.cells.is_empty() || r < 0.0 {
            return Vec::new();
        }
        let r2 = r * r;
        let lo = cell_of(center - DVec3::splat(r), self.cell);
        let hi = cell_of(center + DVec3::splat(r), self.cell);
        let mut out = Vec::new();
        for cx in lo[0]..=hi[0] {
            for cy in lo[1]..=hi[1] {
                for cz in lo[2]..=hi[2] {
                    if let Some(bucket) = self.cells.get(&[cx, cy, cz]) {
                        for &j in bucket {
                            if (pos[j as usize] - center).length_squared() <= r2 {
                                out.push(j as usize);
                            }
                        }
                    }
                }
            }
        }
        out.sort_unstable();
        out
    }
}

/// The cell containing `p`. `floor` (not truncation) so negative coordinates
/// bin correctly; coordinates are assumed within i64 range after division
/// (astronomically true for simulation units).
fn cell_of(p: DVec3, cell: f64) -> [i64; 3] {
    [
        (p.x / cell).floor() as i64,
        (p.y / cell).floor() as i64,
        (p.z / cell).floor() as i64,
    ]
}
