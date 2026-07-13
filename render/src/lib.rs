//! `galaxy-render`: the wgpu render stage — **frame-data → linear HDR**.
//!
//! Consumes Contract 3 frame-data (`galaxy-renderprep`) and additively blends each
//! particle as a Gaussian splat into an `Rgba32Float` accumulation buffer, then
//! reads it back as a linear HDR image and writes an OpenEXR file. Additive
//! blending is commutative → order-independent, no depth sort (DESIGN's recipe).
//!
//! Tonemapping/grading is deliberately **NOT** here — the renderer emits *linear*
//! HDR so `grade` can regrade thousands of frames without re-rendering. The
//! headless-wgpu feasibility was proven up front (see `bin/spike.rs`); this is the
//! productionized path built on that result.
//!
//! The GPU context ([`Renderer`]) is created **once** and reused across every frame
//! of a movie — adapter/device init is not paid per frame.

pub mod camera;
pub mod exr_io;
pub mod render;
pub mod rig;
pub mod volume;

pub use camera::{Camera, Projection};
pub use exr_io::{read_exr, write_exr};
pub use render::{HdrImage, RenderConfig, Renderer};
pub use rig::{ease_in_out, smooth_envelope, CameraPath, RigError};
pub use volume::{
    cluster_lights, temperature_color, GasFrame, GasLook, Light, ScatterLook, ShadowBake, TempColor,
};

/// Errors from the render stage.
#[derive(thiserror::Error, Debug)]
pub enum RenderError {
    /// No wgpu adapter is available (e.g. a headless box with no GPU / no software
    /// fallback). Returned rather than panicking so callers can degrade gracefully.
    #[error("no wgpu adapter available (headless render needs a GPU)")]
    NoAdapter,
    /// The adapter lacks a feature the renderer requires (notably `FLOAT32_BLENDABLE`
    /// for additive blending into the 32-bit-float accumulation buffer).
    #[error("adapter missing required feature: {0}")]
    MissingFeature(String),
    /// `request_device` failed.
    #[error("failed to create wgpu device: {0}")]
    Device(String),
    /// A GPU buffer could not be mapped for readback.
    #[error("failed to map GPU buffer for readback: {0}")]
    BufferMap(String),
    /// A [`render::RenderConfig`] parameter is outside its documented domain
    /// (e.g. a perspective splat-clamp window that is empty or non-finite).
    #[error("invalid render config: {0}")]
    Config(String),
    /// An OpenEXR read/write failed.
    #[error("OpenEXR error: {0}")]
    Exr(String),
    /// Underlying I/O failure.
    #[error("render I/O error: {0}")]
    Io(#[from] std::io::Error),
}
