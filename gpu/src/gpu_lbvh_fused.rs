//! [`GpuLbvhFused`]: the single-device, GPU-resident **fuse** of the M4c–M4g LBVH pipeline
//! (DESIGN M4h) — the named scale refinement [`crate::GpuLbvh`] (M4g) deferred.
//!
//! [`crate::GpuLbvh`] (M4g) is the *reference-grade composition*: each build stage owns its own
//! wgpu device and the pointer tree / flat form round-trips through host memory between stages,
//! so a `GpuLbvh` holds several devices and pays ~5 CPU↔GPU sync points (readback + reupload)
//! per `accelerations` call. `GpuLbvhFused` runs the **whole pipeline on one device in one
//! submit**: `bodies` are uploaded once, every intermediate — Morton codes → sorted order →
//! gathered leaves → Karras pointer tree → DFS skip-pointer flat form — stays in GPU storage
//! buffers that flow directly from one compute pass to the next (wgpu's automatic usage-tracked
//! barriers order the passes), and only the final `accel` is read back. One upload + one
//! readback — replacing the reference chain's ~5 readback/reupload round-trips (one per stage:
//! morton, sort, tree-build, flatten, traverse) with a single submit (≈4 fewer sync points).
//!
//! ## Shared pipeline: [`crate::fused_core::FusedCore`]
//! The whole build+traverse pipeline (device, pipelines, layouts, lazily-sized buffers, the pass
//! sequence) lives in [`FusedCore`], shared with [`crate::GpuResidentLeapfrog`] (M4i). Every
//! stage runs the **same f32 WGSL** as the M4g chain. `GpuLbvhFused` is a thin wrapper: it seeds
//! `bodies`/`idx_a`/`parent` by host `write_buffer`, encodes the shared build+traverse, and reads
//! back `accel` — one upload + one readback per force evaluation. On a given device this
//! reproduces the reference `GpuLbvh` forces bit-for-bit (the M4h faithful-refactor gate).
//! `(g, softening, theta)` semantics and the `ForceSolver` interface are unchanged.
//!
//! ## Scope: this fuses the *build pipeline*, not cross-step residency
//! M4h keeps particle state on the GPU across the **stages of one force evaluation**. Keeping
//! state GPU-resident across **integrator steps** is a *separate* item — landed as
//! [`crate::GpuResidentLeapfrog`] (M4i), which reuses [`FusedCore`]. This is a latency /
//! architecture win (one submit; ≈4 fewer CPU↔GPU sync points), the precondition for that
//! residency, **not** a throughput speedup: the single-invocation serial stages (sort, aggregate,
//! flatten-structure) are unchanged and stay the bottleneck; their parallel refinements remain
//! deferred.

use galaxy_core::{DVec3, ForceSolver, State};
use galaxy_solvers::NO_PARENT;

use crate::fused_core::FusedCore;
use crate::GpuError;

/// GPU Barnes-Hut force solver over a GPU-resident Morton Linear BVH, **fused onto a single
/// wgpu device** — the M4h refinement of [`crate::GpuLbvh`]. Same `(g, softening, theta)`
/// semantics; one upload + one readback per force evaluation. A thin host-upload/readback wrapper
/// over the shared [`FusedCore`] pipeline.
pub struct GpuLbvhFused {
    core: FusedCore,
}

impl GpuLbvhFused {
    /// Bring up the single fused compute device + every pipeline. Returns a typed [`GpuError`]
    /// (never panics) when no adapter is available.
    pub fn new(g: f64, softening: f64, theta: f64) -> Result<Self, GpuError> {
        Ok(GpuLbvhFused {
            core: FusedCore::new(g, softening, theta)?,
        })
    }
}

impl ForceSolver for GpuLbvhFused {
    fn accelerations(&mut self, state: &State, acc: &mut [DVec3]) {
        let n = state.len();
        assert_eq!(acc.len(), n, "acc length must match particle count");
        if n == 0 {
            return;
        }
        if n == 1 {
            // A lone particle feels no force (its only leaf holds just itself) — no dispatch.
            acc[0] = DVec3::ZERO;
            return;
        }

        let total = 2 * n - 1;
        self.core.ensure_capacity(n);
        let res = self.core.res.as_ref().expect("resources ensured above");

        // --- The only host→device uploads: bodies, the sort's index seed, and the parent
        // NO_PARENT init. `write_buffer` copies are scheduled before the submitted commands, so
        // they land before any compute pass. The f64→f32 narrowing here is the crate's owned
        // precision reduction; `bodies` (xyz=pos, w=mass) feeds morton, gather AND the traversal.
        let bodies: Vec<[f32; 4]> = (0..n)
            .map(|i| {
                let p = state.pos[i];
                [p.x as f32, p.y as f32, p.z as f32, state.mass[i] as f32]
            })
            .collect();
        self.core
            .queue
            .write_buffer(&res.bodies, 0, bytemuck::cast_slice(&bodies));
        let idx0: Vec<u32> = (0..n as u32).collect();
        self.core
            .queue
            .write_buffer(&res.idx_a, 0, bytemuck::cast_slice(&idx0));
        let parent_init = vec![NO_PARENT; total];
        self.core
            .queue
            .write_buffer(&res.parent, 0, bytemuck::cast_slice(&parent_init));

        self.core.write_uniforms(n);

        // --- One command encoder: the whole build + traverse (the shared pass sequence). ---
        let mut enc = self
            .core
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fused-lbvh-encoder"),
            });
        self.core.encode_build_traverse(&mut enc, n);

        let bytes = (n * std::mem::size_of::<[f32; 4]>()) as u64;
        enc.copy_buffer_to_buffer(&res.accel, 0, &res.readback, 0, bytes);
        self.core.queue.submit([enc.finish()]);

        // The single readback: map, block once, widen f32 accelerations back to f64. A map
        // failure is an exceptional GPU loss and panics rather than corrupt state.
        let slice = res.readback.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.core
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("gpu poll failed");
        rx.recv()
            .expect("map channel closed")
            .expect("gpu buffer map failed");

        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        for (i, a) in acc.iter_mut().enumerate() {
            let b = i * 4;
            *a = DVec3::new(floats[b] as f64, floats[b + 1] as f64, floats[b + 2] as f64);
        }
        drop(data);
        res.readback.unmap();
    }

    /// Softened potential energy, delegated to the shared CPU **f64** reduction — identical to
    /// `GpuLbvh`/`GpuTree`/`BarnesHut`. Same documented inconsistency: forces are f32 while this
    /// is f64, so an energy-drift diagnostic mixes a precision gap with integrator error; it is
    /// a periodic O(N²) diagnostic, not the per-step path.
    fn potential_energy(&self, state: &State) -> f64 {
        galaxy_solvers::potential::potential_energy_parallel(
            state,
            self.core.g,
            self.core.softening,
        )
    }
}
