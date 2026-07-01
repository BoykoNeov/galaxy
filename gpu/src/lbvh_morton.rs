//! [`GpuMortonBuilder`]: the GPU Morton + bounding-box build stage — stage 1 of the
//! GPU-resident LBVH build (DESIGN M4c).
//!
//! Given particle positions it computes, on the GPU (wgpu compute, **f32**), the root
//! bounding box and the per-particle 30-bit Morton codes + their three quantized lanes.
//! It is the GPU port of the prologue of [`galaxy_solvers::LbvhFlat::build`] and is gated
//! directly against the CPU reference [`galaxy_solvers::reference_morton`].
//!
//! ## Two passes, f32 throughout
//! 1. **bbox reduction** — a single workgroup grid-strides all positions into per-lane
//!    private min/max, then a fixed-order shared-memory tree reduction folds to lane 0,
//!    which writes the bbox. min/max never round and are order-independent, so this is
//!    bit-exact and deterministic with no float atomics (which WGSL lacks anyway).
//! 2. **quantize** — one invocation per particle reconstructs the exact CPU bbox
//!    convention (pad/floor/scale) in f32, then floors + clamps each axis to `[0, 1023]`
//!    and interleaves the three lanes into a 30-bit code.
//!
//! ## Scope (state plainly)
//! This proves **quantization + the reduction pattern**. It does **not** prove the tree
//! matches the reference: f32 codes diverge from the f64 reference near cell boundaries
//! (a straddling particle floors into an adjacent 1024³ cell), so the eventual GPU tree
//! *topology* can differ — the expected analogue of the θ-straddle in [`crate::GpuTree`],
//! not a bug. The real correctness check is the later θ→0 physics gate on the deferred
//! `GpuLbvh`. This stage's gates are structural + tolerance + determinism only.

use galaxy_core::DVec3;

use crate::GpuError;

/// Result of the GPU Morton+bbox stage. `bbox_min`/`bbox_max` are the raw f32 reduction
/// output (for the reduction gate); `lanes`/`codes` are the quantized output feeding the
/// next stage (the GPU sort).
pub struct GpuMorton {
    /// Raw bounding-box low corner from the GPU reduction (f32, per axis).
    pub bbox_min: [f32; 3],
    /// Raw bounding-box high corner from the GPU reduction (f32, per axis).
    pub bbox_max: [f32; 3],
    /// Per-particle quantized lanes `[x, y, z]`, each in `[0, 1024)`.
    pub lanes: Vec<[u32; 3]>,
    /// Per-particle interleaved 30-bit Morton codes.
    pub codes: Vec<u32>,
}

/// GPU Morton + bounding-box build stage. Holds a reusable wgpu compute context built
/// once and driven per [`compute`](Self::compute) call; storage buffers grow lazily with
/// N. Same bring-up idiom as [`crate::GpuDirectSum`] (baseline storage-buffer compute, no
/// device features → no adapter narrowing).
pub struct GpuMortonBuilder {}

impl GpuMortonBuilder {
    /// Bring up a headless wgpu compute device and build the Morton+bbox pipelines.
    /// Returns a typed [`GpuError`] (never panics) when no adapter is available.
    pub fn new() -> Result<Self, GpuError> {
        todo!("GPU Morton+bbox bring-up (M4c implementation)")
    }

    /// Compute the bounding box + Morton codes for `pos` on the GPU. `pos` may be empty
    /// (yields empty `lanes`/`codes` and a degenerate bbox with no dispatch).
    pub fn compute(&mut self, pos: &[DVec3]) -> GpuMorton {
        let _ = pos;
        todo!("GPU Morton+bbox compute (M4c implementation)")
    }
}
