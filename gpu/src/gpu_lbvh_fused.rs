//! [`GpuLbvhFused`]: the single-device, GPU-resident **fuse** of the M4c–M4g LBVH pipeline
//! (DESIGN M4h) — the named scale refinement `GpuLbvh` (M4g) deferred.
//!
//! `GpuLbvh` (M4g) is the *reference-grade composition*: each build stage owns its own wgpu
//! device and the pointer tree / flat form round-trips through host memory between stages, so
//! a `GpuLbvh` holds several devices and pays ~5 CPU↔GPU sync points (readback + reupload) per
//! `accelerations` call. `GpuLbvhFused` runs the **whole pipeline on one device in one
//! submit**: `bodies` are uploaded once, every intermediate (Morton codes → sorted order →
//! gathered leaves → Karras pointer tree → DFS skip-pointer flat form) stays in GPU storage
//! buffers that flow directly from one compute pass to the next, and only the final `accel` is
//! read back. One upload + one readback; `N−1` fewer sync points.
//!
//! ## Same forces, same interface — a lossless refactor
//! It uses the **same f32 kernels** as the M4g chain (the complex traversal kernel is reused
//! verbatim; only the trivial `gather` and geometry-repack kernels are new), so on a given
//! device it reproduces the reference `GpuLbvh` forces (see the M4h "faithful refactor" gate).
//! The `(g, softening, theta)` semantics and the `ForceSolver` interface are unchanged.
//!
//! ## Scope: this fuses the *build pipeline*, not cross-step residency
//! M4h keeps particle state on the GPU across the **stages of one force evaluation**. Keeping
//! state GPU-resident across **integrator steps** (which would change the
//! `accelerations(&State)→acc` interface and touch the stepping loop) is a *separate* deferred
//! item — see DESIGN "Remaining M4+". This is a latency / architecture win (one submit, `N−1`
//! fewer sync points), the precondition for that residency, **not** a throughput speedup: the
//! single-invocation serial stages (sort, aggregate, flatten-structure) are unchanged and stay
//! the bottleneck; their parallel refinements remain deferred.

use galaxy_core::{DVec3, ForceSolver, State};

use crate::GpuError;

/// GPU Barnes-Hut force solver over a GPU-resident Morton Linear BVH, **fused onto a single
/// wgpu device** — the M4h refinement of [`crate::GpuLbvh`]. Same `(g, softening, theta)`
/// semantics; one upload + one readback per force evaluation.
pub struct GpuLbvhFused {
    _private: (),
}

impl GpuLbvhFused {
    /// Bring up the single fused compute device + all pipelines. Returns a typed [`GpuError`]
    /// (never panics) when no adapter is available.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        let _ = (g, softening, theta);
        todo!("M4h: fused single-device GPU LBVH")
    }
}

impl ForceSolver for GpuLbvhFused {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let _ = (state, acc);
        todo!("M4h: fused single-device GPU LBVH")
    }

    fn potential_energy(&self, state: &State) -> f64 {
        let _ = state;
        todo!("M4h: fused single-device GPU LBVH")
    }
}
