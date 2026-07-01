//! `galaxy-gpu`: GPU `ForceSolver` implementations (wgpu compute).
//!
//! [`GpuDirectSum`] is an exact O(N²) Plummer-softened direct summation run as a
//! wgpu **compute** kernel — the same algebra as the CPU [`galaxy_solvers::DirectSum`]
//! oracle, moved to the GPU for throughput at 10⁵–10⁶ particles. It is the *scale*
//! path, not a new approximation: with the same `(g, softening)` it computes the
//! same forces the oracle does, to f32 precision.
//!
//! ## Precision: f32 is forced by the toolchain, not chosen
//! wgpu/naga has no portable native f64 compute (`SHADER_FLOAT64` is rarely present
//! across backends), so the kernel runs in **f32**. The physics engine is f64, so
//! this is a genuine precision reduction — the honest lever is the **accumulation
//! strategy**, and float-float (`df64`) emulation of the `xᵢ − xⱼ` difference and the
//! accumulator is the named forward refinement for precision-critical runs. The
//! dominant f32 error is catastrophic cancellation in `xᵢ − xⱼ` (large coordinates,
//! close pairs) and small terms swallowed while summing N contributions into one f32
//! accumulator — both worst in the clustered, large-coordinate collision regime.
//!
//! ## Determinism: gather, not scatter
//! The kernel is a **gather**: one invocation per *target* `i` loops over all sources
//! `j`, accumulating in a private register and writing `acc[i]` exactly once. That is
//! bit-deterministic run-to-run **on a given device** (no float `atomicAdd`, whose
//! ordering is nondeterministic). Cross-device bit-equality is *not* guaranteed
//! (FMA/rounding differ), so the determinism gate is same-device.
//!
//! ## Scope
//! [`GpuDirectSum`] is O(N²) → realistically a few × 10⁶ particles, **not** 10⁷–10⁸.
//!
//! [`GpuTree`] is the Barnes-Hut O(N log N) step past that: the octree is built and
//! linearized on the CPU (reusing [`galaxy_solvers::FlatTree`]) and **traversed** on
//! the GPU by a stackless skip-pointer gather kernel — same f32/determinism story,
//! now with the tree approximation controlled by θ. It opens the 10⁷ band O(N²) cannot;
//! a GPU-resident build (Morton/LBVH) and TreePM/PM remain the deferred 10⁸ door.

pub mod gpu_direct_sum;
pub mod gpu_tree;

pub use gpu_direct_sum::GpuDirectSum;
pub use gpu_tree::GpuTree;

/// Errors bringing up or driving the GPU compute context. Returned rather than
/// panicking so callers can degrade to a CPU solver on a headless / GPU-less box.
#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    /// No wgpu adapter is available (e.g. a headless box with no GPU).
    #[error("no wgpu adapter available (headless GPU compute needs a GPU)")]
    NoAdapter,
    /// `request_device` failed.
    #[error("failed to create wgpu device: {0}")]
    Device(String),
    /// A GPU buffer could not be mapped for readback.
    #[error("failed to map GPU buffer for readback: {0}")]
    BufferMap(String),
}
