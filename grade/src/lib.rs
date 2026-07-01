//! `galaxy-grade`: the tonemap/grade stage — **linear HDR EXR → 16-bit sRGB PNG**.
//!
//! This is the last, config-driven stage of the pipeline (DESIGN.md): it maps the
//! renderer's *unbounded linear* HDR down to a display-referred 16-bit PNG that
//! ffmpeg can mux into a movie. It is deliberately **decoupled from the renderer via
//! files** — it reads an EXR and writes a PNG, so 1000 frames can be regraded in
//! seconds without re-running physics or the GPU. It has no dependency on
//! `galaxy-render` (and so pulls in no wgpu).
//!
//! Grade = exposure → tone curve (ACES/Reinhard) → sRGB OETF → 16-bit quantize.

use std::path::Path;

/// The tone-mapping operator: how unbounded linear HDR is compressed to `[0, 1]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToneMap {
    /// Narkowicz's cheap ACES filmic approximation — the cinematic default.
    AcesApprox,
    /// Reinhard `x / (1 + x)` — simple, well-behaved, softer highlights.
    Reinhard,
}

/// Grading configuration. Config-driven so a whole frame sequence regrades from one
/// place without re-rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradeConfig {
    /// Linear exposure multiplier applied before the tone curve.
    pub exposure: f32,
    /// The tone-mapping operator.
    pub tonemap: ToneMap,
}

impl Default for GradeConfig {
    fn default() -> Self {
        GradeConfig {
            exposure: 1.0,
            tonemap: ToneMap::AcesApprox,
        }
    }
}

/// Errors from the grade stage.
#[derive(thiserror::Error, Debug)]
pub enum GradeError {
    /// Reading the input OpenEXR failed.
    #[error("OpenEXR read error: {0}")]
    Exr(String),
    /// Writing the output PNG failed.
    #[error("PNG write error: {0}")]
    Png(String),
    /// Underlying I/O failure.
    #[error("grade I/O error: {0}")]
    Io(#[from] std::io::Error),
}

/// Apply the tone curve `op` to a linear (already exposure-scaled) RGB triple,
/// producing display-referred values in `[0, 1]`.
pub fn tone_curve(c: [f32; 3], op: ToneMap) -> [f32; 3] {
    let _ = (c, op);
    todo!("ACES-approx / Reinhard per channel, clamped to [0,1]")
}

/// The sRGB opto-electronic transfer function (linear `[0,1]` → sRGB `[0,1]`).
pub fn linear_to_srgb(x: f32) -> f32 {
    let _ = x;
    todo!("piecewise sRGB OETF")
}

/// Grade one linear-HDR pixel to a 16-bit sRGB triple: exposure → tone curve →
/// sRGB encode → quantize to `[0, 65535]`.
pub fn tonemap(linear: [f32; 3], cfg: &GradeConfig) -> [u16; 3] {
    let _ = (linear, cfg);
    todo!("exposure, tone_curve, linear_to_srgb, quantize")
}

/// Grade a linear-HDR OpenEXR file into a 16-bit sRGB PNG under `cfg`.
pub fn grade_file<P: AsRef<Path>, Q: AsRef<Path>>(
    exr_path: P,
    png_path: Q,
    cfg: &GradeConfig,
) -> Result<(), GradeError> {
    let _ = (exr_path.as_ref(), png_path.as_ref(), cfg);
    todo!("read EXR RGB, tonemap each pixel, write 16-bit RGB PNG")
}
