//! [`GpuResidentLeapfrog`]: GPU-**resident** leapfrog stepping (DESIGN M4i) — keeping particle
//! *state* on the GPU across integrator steps, the payoff M4h's single-device fuse unlocked.
//!
//! ## What residency buys, and what it costs
//! [`crate::GpuLbvhFused`] (M4h) fused the whole LBVH build+traverse onto one device in one
//! submit, but still **uploads state and reads back accel every `accelerations` call** — one
//! CPU↔GPU round-trip per force evaluation. `GpuResidentLeapfrog` closes that loop: `pos`, `vel`,
//! `mass`, and `acc` live in GPU storage buffers *across* steps, the kick/drift arithmetic runs
//! on the device, and nothing crosses the bus until an explicit [`snapshot`](Self::snapshot).
//! `bodies` (xyz=pos, w=mass) — already the traversal's input — doubles as the resident position
//! buffer, so the force pipeline reads state in place.
//!
//! This is *not* a throughput speedup (the M4h serial stages — sort, aggregate, flatten — are
//! unchanged and still dominate); it removes the per-step sync points, the point of residency.
//!
//! ## The precision cost is real and documented
//! The host-driven path ([`galaxy_core::LeapfrogKdk`] + a solver) keeps **authoritative
//! positions in f64** and re-narrows to f32 only to feed the GPU force kernel each step. The
//! resident path accumulates `pos += vel*dt` **in f32 across every step**, because WGSL has no
//! portable `f64`. Position updates below ~1e-6 of a coordinate's magnitude are lost, so energy
//! drifts more than the f64 leapfrog's clean bounded oscillation. This is acceptable for the
//! render money-shot (the renderer is f32 anyway) and mirrors the existing f32-force / f64-energy
//! inconsistency; **double-single (float-float) position accumulation is the deferred precision
//! follow-up.**
//!
//! ## Not a `ForceSolver`
//! The [`galaxy_core::ForceSolver`] interface is host-state-in / accel-out — fundamentally
//! incompatible with keeping state resident. So this is its own type with an
//! `upload → step* → snapshot` lifecycle, exactly as DESIGN "Remaining M4+" anticipated.

use galaxy_core::State;

use crate::GpuError;

/// GPU-resident kick-drift-kick leapfrog over the M4h fused LBVH force pipeline. State stays in
/// GPU buffers across [`step`](Self::step)s; only [`snapshot`](Self::snapshot) reads it back.
pub struct GpuResidentLeapfrog {
    #[allow(dead_code)]
    g: f64,
    #[allow(dead_code)]
    softening: f64,
    #[allow(dead_code)]
    theta: f64,
    #[allow(dead_code)]
    time: f64,
}

impl GpuResidentLeapfrog {
    /// Bring up the resident compute device + every pipeline (the M4h fused build/traverse plus
    /// the new kick/drift/reset kernels). Returns a typed [`GpuError`] on adapter/device failure.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        let _ = (g, softening, theta);
        todo!("M4i: bring up the resident pipeline")
    }

    /// Upload `state` (f64→f32 narrowed) into the resident GPU buffers, (re)allocating as `N`
    /// changes, and **prime** the acceleration (one force evaluation, no readback) so the first
    /// [`step`](Self::step)'s opening half-kick uses `a(x₀)`, not a stale value.
    pub fn upload(&mut self, state: &State) {
        let _ = state;
        todo!("M4i: upload + prime")
    }

    /// Advance one resident KDK step by `dt`: kick½ · drift · (reset+build+traverse into `acc`) ·
    /// kick½ — one submit, no readback. Requires a prior [`upload`](Self::upload).
    pub fn step(&mut self, dt: f64) {
        let _ = dt;
        todo!("M4i: one resident step")
    }

    /// Advance `steps` resident KDK steps of `dt`.
    pub fn step_many(&mut self, dt: f64, steps: u64) {
        let _ = (dt, steps);
        todo!("M4i: many resident steps")
    }

    /// Read the resident state back to the host as a fresh [`State`] (pos/vel widened f32→f64,
    /// mass/time host-tracked). The only device→host transfer.
    pub fn snapshot(&mut self) -> State {
        todo!("M4i: readback")
    }

    /// Simulation time after the steps taken so far.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Number of resident particles (0 before the first [`upload`](Self::upload)).
    pub fn len(&self) -> usize {
        todo!("M4i: len")
    }

    /// Whether no particles are resident.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
