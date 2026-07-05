//! [`GpuNeighborGrid`]: GPU fixed-radius neighbor search for SPH (GPU-SPH G1).
//!
//! A Green-style **counting-sort spatial hash** over gas positions: cell coords
//! `floor(p/cell)` are hashed into a fixed-size table (NOT a dense array — a
//! dense linear-index grid explodes on far merger debris; NOT the CPU's sparse
//! `HashMap` either), particles are counting-sorted into their buckets, and a
//! query GATHERS per target over the `ceil(r/cell)`-cell neighborhood, testing
//! true distance at each candidate. It is the GPU analogue of the CPU
//! [`galaxy_solvers::sph::HashGrid`] and the first stage of the GPU-SPH port
//! (density → hydro force → CFL build on top of it).
//!
//! ## Grid-first, endpoint is the LBVH range query (D4)
//! The measured gas smoothing-length range (34×+ even in the undisturbed disk,
//! ~280× at pericenter) puts this firmly in the regime where a single-resolution
//! grid degenerates *at scale*, so the scale-forward endpoint is a max-h-augmented
//! LBVH range query reusing the Karras construction. This grid is the **grid-first
//! de-risk**: it brings up density/force/CFL against a known-correct, simple
//! neighbor structure before the novel, conservativeness-sensitive LBVH traversal
//! is also in the mix. It is kept isolated behind [`GpuNeighborGrid::query_all`] so
//! the grid↔LBVH swap is a module change, and it survives as a CPU-parity oracle /
//! small-N fallback afterward — not throwaway. See `kindled-resident-cascade.md`.
//!
//! ## Gate: the FILTERED pair set (swap-stable), not raw candidates
//! Correctness is gated as equality of the **filtered pair set** — pairs `(i,j)`
//! with `r_ij < SUPPORT·max(h_i,h_j)` (the true averaged-kernel coupling range) —
//! against `HashGrid`, NOT the raw candidate set. The raw candidate radius is a
//! *policy* (fork(a) global `SUPPORT·h_max` over-gather here; fork(b)/LBVH's
//! per-particle `SUPPORT·h_i` + prune later) that differs between structures while
//! the filtered set is invariant; gating the filtered set is what lets the LBVH
//! swap in without a false gate failure.

use bytemuck::{Pod, Zeroable};

use galaxy_core::DVec3;

use crate::GpuError;

/// Per-target neighbor candidate lists in CSR form: `flat[starts[i]..starts[i+1]]`
/// are the indices `j` (INCLUDING `j == i`, matching `HashGrid::neighbours_within`)
/// with `|pos[j] − pos[i]| ≤ radius`. Set-valued: consumers that need the exact
/// `HashGrid` ascending order should sort a copy (the G1 gate compares sets).
pub struct GpuNeighbours {
    /// CSR row offsets, length `n + 1`.
    starts: Vec<u32>,
    /// Concatenated candidate indices, length `starts[n]`.
    flat: Vec<u32>,
}

impl GpuNeighbours {
    /// Candidate indices for query point `i` (includes `i` itself).
    pub fn neighbours(&self, i: usize) -> &[u32] {
        let s = self.starts[i] as usize;
        let e = self.starts[i + 1] as usize;
        &self.flat[s..e]
    }

    /// Number of query points.
    pub fn len(&self) -> usize {
        self.starts.len().saturating_sub(1)
    }

    /// Whether there are no query points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// GPU Green-style spatial-hash neighbor search. Reusable wgpu compute context
/// built once ([`new`](Self::new)) and driven per [`query_all`](Self::query_all);
/// storage buffers grow lazily with N — the same bring-up idiom as
/// [`crate::GpuSorter`].
pub struct GpuNeighborGrid {
    // Filled by the green implementation: device/queue, the build (counting-sort)
    // and count/fill query pipelines, bind-group layouts, and lazily-grown storage.
    device: wgpu::Device,
    queue: wgpu::Queue,
}

/// Uniform for the build/query kernels: particle count, hash-table size, and the
/// cell edge / query radius (both f32 on the device, matching the f32-force story).
/// `dead_code` until the green implementation constructs it — the red commit ships
/// only the API surface.
#[repr(C)]
#[allow(dead_code)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    table_size: u32,
    cell: f32,
    radius: f32,
}

impl GpuNeighborGrid {
    /// Bring up a headless wgpu compute device and build the spatial-hash
    /// pipelines. Returns a typed [`GpuError`] (never panics) when no adapter is
    /// available, exactly like [`crate::GpuSorter::new`].
    pub fn new() -> Result<Self, GpuError> {
        todo!("GPU-SPH G1 green: bring up device + spatial-hash pipelines")
    }

    /// Build the spatial hash over `pos` with cell edge `cell`, then return, for
    /// every particle `i`, the candidate indices `j` with `|pos[j] − pos[i]| ≤
    /// radius` (including `i`). `cell` and `radius` are decoupled: `cell` sizes the
    /// buckets, `radius` sizes the `ceil(radius/cell)`-cell neighborhood walk — the
    /// wide-`h` regime is `radius ≫ cell`. `cell > 0`, finite.
    pub fn query_all(&mut self, pos: &[DVec3], cell: f64, radius: f64) -> GpuNeighbours {
        let _ = (&self.device, &self.queue, pos, cell, radius);
        todo!("GPU-SPH G1 green: counting-sort spatial hash + per-target gather")
    }
}
