//! One shared headless wgpu compute context for the whole crate.
//!
//! Every GPU compute struct here (`GpuDirectSum`, `GpuHydro`, the LBVH stages, …)
//! used to bring up its **own** `Instance` → `Adapter` → `Device` in its
//! constructor. Under the test suite that meant ~119 device creations, each a
//! ~0.3 s Vulkan bring-up, and — per the `lib.rs` teardown note — as many device
//! *destructions*, which is the churn the intermittent `STATUS_ACCESS_VIOLATION`
//! flake is blamed on.
//!
//! [`gpu_context`] instead creates the device **exactly once per process** and
//! hands every caller a clone. `wgpu::Device`/`wgpu::Queue` are `Arc`-backed, so
//! a clone is a cheap handle to the *same* underlying device — no field types or
//! public constructor signatures change, callers just stop paying the bring-up.
//!
//! Concurrency: `OnceLock::get_or_init` runs the creation closure exactly once
//! even when the parallel test harness calls in from many threads at once; later
//! (and concurrent-but-later) callers block until it is ready, then share it. The
//! device and queue are `Send + Sync` and safe to drive from multiple threads.

use crate::GpuError;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// A shared wgpu compute context: one device + queue reused by every GPU struct
/// in this crate. Cloning hands out another handle to the **same** device.
#[derive(Clone)]
pub(crate) struct GpuContext {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
}

/// How many times a wgpu device has actually been created this process. The
/// shared-context contract is that this stays at 1 no matter how many compute
/// structs get built — it is the direct test hook for "created once, shared".
static DEVICE_CREATIONS: AtomicUsize = AtomicUsize::new(0);

/// Number of real device creations so far this process (test hook — see the
/// module test and [`gpu_context`]). Test-only: the counter is *written* in
/// every build by [`create_async`], but only the test reads it back.
#[cfg(test)]
pub(crate) fn device_creations() -> usize {
    DEVICE_CREATIONS.load(Ordering::Relaxed)
}

/// The process-wide shared compute context. Created on first call and cached;
/// every later call returns a cheap clone of the same device/queue. Returns a
/// typed [`GpuError`] (never panics) when no adapter is available — cached, so a
/// headless box fails fast on every subsequent call too.
pub(crate) fn gpu_context() -> Result<GpuContext, GpuError> {
    static CTX: OnceLock<Result<GpuContext, GpuError>> = OnceLock::new();
    CTX.get_or_init(|| pollster::block_on(create_async()))
        .clone()
}

async fn create_async() -> Result<GpuContext, GpuError> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None, // headless
            force_fallback_adapter: false,
        })
        .await
        .map_err(|_| GpuError::NoAdapter)?;

    // Requests no device features and default limits — compute over storage
    // buffers is baseline, so this does not narrow adapter support, and the whole
    // crate deliberately stays on `Limits::default()` (8 storage buffers/stage)
    // for portability. Every former per-struct bring-up used this exact
    // descriptor, so one shared device satisfies all 12 consumers.
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("galaxy-gpu-shared-device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })
        .await
        .map_err(|e| GpuError::Device(e.to_string()))?;

    // Count only a *successful* creation — a headless box's failed adapter probe
    // must not tick this, and the `OnceLock` guarantees this runs at most once.
    DEVICE_CREATIONS.fetch_add(1, Ordering::Relaxed);
    Ok(GpuContext { device, queue })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract: no matter how many times the shared context is requested,
    /// the wgpu device is created exactly once and reused. Red against the
    /// `todo!()` stub (the first call panics); green once `create_async` builds
    /// and caches one device.
    #[test]
    fn context_is_created_once_and_shared() {
        let _a = gpu_context().expect("wgpu adapter required for shared-context test");
        let _b = gpu_context().expect("wgpu adapter required for shared-context test");
        assert_eq!(
            device_creations(),
            1,
            "the shared wgpu device must be created exactly once and reused"
        );
    }
}
