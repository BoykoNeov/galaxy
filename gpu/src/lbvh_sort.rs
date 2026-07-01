//! [`GpuSorter`]: the GPU Morton sort — stage 2 of the GPU-resident LBVH build (DESIGN
//! M4d), the load-bearing risk of the whole port.
//!
//! Given the per-particle 30-bit Morton `codes` (the output of [`crate::GpuMortonBuilder`]
//! or the CPU [`galaxy_solvers::reference_morton`]) it produces, on the GPU (wgpu compute),
//! the permutation `order` that sorts the bodies by `(code, original index)` — the exact
//! ordering [`galaxy_solvers::reference_sort`] defines and the Karras tree-build stage
//! consumes.
//!
//! ## Pure integer — determinism is nearly free, correctness is the risk
//! Unlike the f32 Morton/direct-sum kernels, this stage touches **no floats**: `u32` codes
//! in, a `u32` permutation out. So its result is not merely deterministic on a given device
//! but **bit-for-bit equal to the f64 CPU reference** (the sort of a code array is the same
//! whether the arithmetic around it is f32 or f64). The real hazard is therefore not
//! nondeterminism but **scatter/scan correctness**, which the implementation makes
//! unarguable by a single-invocation stable counting sort (see the impl note).

use bytemuck::{Pod, Zeroable};

use crate::GpuError;

/// Result of the GPU Morton sort. `order` is the permutation of `0..n` sorting by
/// `(code, index)`; `sorted_codes[k] == input_codes[order[k]]` is the (non-decreasing)
/// key array, returned for free from the final key buffer so callers can check sortedness
/// without re-gathering.
pub struct GpuSort {
    /// The permutation of `0..n` that sorts bodies by `(code, original index)`.
    pub order: Vec<u32>,
    /// The codes in sorted order (`sorted_codes[k] == codes[order[k]]`), non-decreasing.
    pub sorted_codes: Vec<u32>,
}

/// Uniform block mirroring the WGSL `Params` (16-byte aligned): body count + current
/// 8-bit radix shift (`0`, `8`, `16`, `24`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n: u32,
    shift: u32,
    _pad: [u32; 2],
}

/// GPU Morton radix sort. Holds a reusable wgpu compute context built once and driven per
/// [`sort`](Self::sort) call; storage buffers grow lazily with N. Same bring-up idiom as
/// [`crate::GpuMortonBuilder`].
pub struct GpuSorter {
    _params: Params,
}

impl GpuSorter {
    /// Bring up a headless wgpu compute device and build the radix-sort pipeline. Returns a
    /// typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        todo!("GPU radix-sort bring-up (M4d impl commit)")
    }

    /// Sort the bodies by `(code, original index)` on the GPU, returning the permutation
    /// `order` and the sorted key array. `codes` may be empty (yields empty output).
    pub fn sort(&mut self, _codes: &[u32]) -> GpuSort {
        todo!("GPU radix sort (M4d impl commit)")
    }
}
