//! OpenEXR read/write for [`HdrImage`]. The renderer writes **linear** HDR EXR
//! (32-bit float, lossless) — the hand-off to the `grade` stage. `read_exr` exists
//! mainly so the round-trip can be tested and so tools can re-ingest a frame.
//!
//! `exr` is pure Rust (no `unsafe`, no C linkage) — unlike HDF5 it is not a Windows
//! link landmine, so the EXR boundary is cheap to keep from day one.

use std::path::Path;

// Import only the two entry points — NOT the prelude glob, which brings exr's
// 1-arg `Result` alias into scope and shadows `std::Result` in our signatures.
use exr::prelude::{read_first_rgba_layer_from_file, write_rgba_file};

use crate::render::HdrImage;
use crate::RenderError;

/// Write a linear HDR image to an OpenEXR file (RGBA, 32-bit float).
pub fn write_exr<P: AsRef<Path>>(path: P, image: &HdrImage) -> Result<(), RenderError> {
    write_rgba_file(
        path.as_ref(),
        image.width as usize,
        image.height as usize,
        |x, y| {
            let p = image.pixel(x as u32, y as u32);
            (p[0], p[1], p[2], p[3])
        },
    )
    .map_err(|e| RenderError::Exr(e.to_string()))
}

/// Read an OpenEXR file back into an [`HdrImage`] (RGBA, 32-bit float).
pub fn read_exr<P: AsRef<Path>>(path: P) -> Result<HdrImage, RenderError> {
    let image = read_first_rgba_layer_from_file(
        path.as_ref(),
        |resolution, _channels| HdrImage {
            width: resolution.width() as u32,
            height: resolution.height() as u32,
            pixels: vec![[0.0; 4]; resolution.width() * resolution.height()],
        },
        |img: &mut HdrImage, pos, (r, g, b, a): (f32, f32, f32, f32)| {
            let i = pos.y() * img.width as usize + pos.x();
            img.pixels[i] = [r, g, b, a];
        },
    )
    .map_err(|e| RenderError::Exr(e.to_string()))?;

    Ok(image.layer_data.channel_data.pixels)
}
