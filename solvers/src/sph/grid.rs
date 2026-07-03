//! Uniform hash grid for fixed-radius SPH neighbor queries.
//!
//! Counting-sort-by-cell layout: particles are binned into cubic cells of edge
//! `cell`, stored as a cell-start offset table plus one index array — compact,
//! rebuild-per-call cheap (O(N)), and rayon-friendly on the query side because
//! queries only read. Queries GATHER per target and return neighbor indices in
//! ascending order, so any consumer summing over the returned list associates
//! floating-point work in a fixed order (the parallel ≡ serial bit-exactness
//! discipline). Gated bit-exact against [`super::reference::reference_neighbours`].

use galaxy_core::DVec3;

/// A uniform grid over the particles' bounding box.
///
/// `cell` is the bin edge length; queries with any radius work (the walk covers
/// `ceil(r/cell)` cells per axis), but `cell ≈` the typical query radius is the
/// efficient regime.
pub struct HashGrid {
    /// Bin edge length.
    cell: f64,
    /// Minimum corner of the binned bounding box.
    origin: DVec3,
    /// Grid dimensions (cells per axis).
    dims: [usize; 3],
    /// CSR-style offsets: cell `c`'s particles are `indices[start[c]..start[c+1]]`.
    start: Vec<u32>,
    /// Particle indices, grouped by cell, ascending within each cell.
    indices: Vec<u32>,
}

impl HashGrid {
    /// Bin `pos` into cells of edge `cell` (> 0, finite).
    pub fn build(pos: &[DVec3], cell: f64) -> Self {
        let _ = (pos, cell);
        todo!("M7a: hash-grid build")
    }

    /// Indices `j` (ascending) of all particles with `|pos[j] − center| ≤ r`.
    /// `pos` must be the same slice the grid was built from. A particle exactly
    /// at distance `r` IS a neighbor (`≤`, matching the O(N²) oracle).
    pub fn neighbours_within(&self, pos: &[DVec3], center: DVec3, r: f64) -> Vec<usize> {
        let _ = (pos, center, r);
        todo!("M7a: hash-grid range query")
    }
}
