//! The wgpu splat renderer: [`Renderer`] holds the reusable GPU context + pipeline,
//! [`Renderer::render_frame`] rasterizes one frame of Contract 3 frame-data into a
//! linear HDR [`HdrImage`].

use galaxy_renderprep::FrameData;

use crate::camera::Camera;
use crate::RenderError;

/// Per-frame render settings. Camera lives separately (it changes per view, not per
/// frame); this is the raster target + splat shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Gaussian falloff constant `k`: a splat's intensity is `exp(-k · r²)` for `r`
    /// the normalized distance (0 at center, 1 at the quad edge). Larger = tighter.
    pub falloff: f32,
}

impl Default for RenderConfig {
    fn default() -> Self {
        RenderConfig {
            width: 1920,
            height: 1080,
            falloff: 6.0,
        }
    }
}

impl RenderConfig {
    /// The image aspect ratio (width / height), for camera framing.
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// A linear HDR image: `width × height` RGBA pixels, 32-bit float, row-major from
/// the top-left. Not tonemapped — this is what `grade` consumes.
#[derive(Clone, Debug, PartialEq)]
pub struct HdrImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// `width * height` RGBA pixels, row-major.
    pub pixels: Vec<[f32; 4]>,
}

impl HdrImage {
    /// The pixel at `(x, y)` (top-left origin).
    pub fn pixel(&self, x: u32, y: u32) -> [f32; 4] {
        self.pixels[(y * self.width + x) as usize]
    }

    /// Sum of each RGB channel over all pixels — the total accumulated flux, used by
    /// conservation/linearity invariants.
    pub fn total_flux(&self) -> [f64; 3] {
        let mut sum = [0.0f64; 3];
        for p in &self.pixels {
            sum[0] += p[0] as f64;
            sum[1] += p[1] as f64;
            sum[2] += p[2] as f64;
        }
        sum
    }
}

/// The reusable GPU rendering context: adapter/device/queue + the splat pipeline,
/// created once and driven for every frame of a movie.
pub struct Renderer {
    _todo: (),
}

impl Renderer {
    /// Bring up a headless wgpu device with the features the renderer needs and
    /// build the splat pipeline. Returns a typed [`RenderError`] (never panics) if
    /// no adapter or required feature is available.
    pub fn new() -> Result<Self, RenderError> {
        todo!("headless adapter + FLOAT32_BLENDABLE device + additive splat pipeline")
    }

    /// Render one frame: additively blend every particle in `frame` as a Gaussian
    /// splat, as seen by `camera`, into an `Rgba32Float` target of
    /// `cfg.width × cfg.height`, and read it back as a linear [`HdrImage`].
    pub fn render_frame(
        &self,
        frame: &FrameData,
        camera: &Camera,
        cfg: &RenderConfig,
    ) -> Result<HdrImage, RenderError> {
        let _ = (frame, camera, cfg);
        todo!("project splats, additive-blend, copy-to-buffer, read back")
    }
}
