//! OpenEXR read/write for [`HdrImage`]. The renderer writes **linear** HDR EXR
//! (32-bit float, lossless) — the hand-off to the `grade` stage. `read_exr` exists
//! mainly so the round-trip can be tested and so tools can re-ingest a frame.
//!
//! `exr` is pure Rust (no `unsafe`, no C linkage) — unlike HDF5 it is not a Windows
//! link landmine, so the EXR boundary is cheap to keep from day one.

use std::path::Path;

use crate::render::HdrImage;
use crate::RenderError;

/// Write a linear HDR image to an OpenEXR file (RGBA, 32-bit float).
pub fn write_exr<P: AsRef<Path>>(path: P, image: &HdrImage) -> Result<(), RenderError> {
    let _ = (path, image);
    todo!("exr::prelude::write_rgba_file from image.pixel(x, y)")
}

/// Read an OpenEXR file back into an [`HdrImage`] (RGBA, 32-bit float).
pub fn read_exr<P: AsRef<Path>>(path: P) -> Result<HdrImage, RenderError> {
    let _ = path;
    todo!("exr::prelude::read_first_rgba_layer_from_file")
}
