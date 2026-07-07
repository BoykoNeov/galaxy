//! `galaxy-grade`: the tonemap/grade stage — **linear HDR EXR → 16-bit sRGB PNG**.
//!
//! This is the last, config-driven stage of the pipeline (DESIGN.md): it maps the
//! renderer's *unbounded linear* HDR down to a display-referred 16-bit PNG that
//! ffmpeg can mux into a movie. It is deliberately **decoupled from the renderer via
//! files** — it reads an EXR and writes a PNG, so 1000 frames can be regraded in
//! seconds without re-running physics or the GPU. It has no dependency on
//! `galaxy-render` (and so pulls in no wgpu).
//!
//! Grade = [bloom (linear, image-space)] → exposure → tone curve (ACES/Reinhard/
//! asinh) → sRGB OETF → 16-bit quantize.

mod bloom;
mod local;

pub use bloom::{bloom, BloomConfig};
pub use local::{apply_local_tonemap, local_gain, LocalToneConfig};

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

// Targeted import (NOT the prelude glob, which shadows `std::Result`).
use exr::prelude::read_first_rgba_layer_from_file;

/// The tone-mapping operator: how unbounded linear HDR is compressed to `[0, 1]`.
/// (`Eq` is not derived: `Asinh` carries an `f32` softening knob.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ToneMap {
    /// Narkowicz's cheap ACES filmic approximation — the cinematic default.
    AcesApprox,
    /// Reinhard `x / (1 + x)` — simple, well-behaved, softer highlights.
    Reinhard,
    /// Lupton-style asinh stretch `β·asinh(x/β)` (clamped to `[0, 1]`) — the
    /// astro-imaging curve: linear below the softening knob `β`, logarithmic
    /// above it, so faint tidal tails survive an exposure push without the
    /// additive cores blowing out. `beta` must be positive; it is floored at
    /// `f32::MIN_POSITIVE` so a degenerate `0.0` stays total (no NaN).
    Asinh {
        /// The softening knob β: the pivot between the linear and log regimes.
        beta: f32,
    },
}

/// Grading configuration. Config-driven so a whole frame sequence regrades from one
/// place without re-rendering.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GradeConfig {
    /// Linear exposure multiplier applied before the tone curve.
    pub exposure: f32,
    /// The tone-mapping operator.
    pub tonemap: ToneMap,
    /// Optional bloom, applied image-wide in linear space BEFORE exposure and the
    /// tone curve (`None` ⇒ off). An image-space op — `grade_file` runs it; the
    /// per-pixel [`tonemap`] cannot and does not.
    pub bloom: Option<BloomConfig>,
    /// Levels black point: the display-referred tone-curve output that maps to
    /// true black. Lifting it crushes background haze and separates faint stars
    /// from a scatter glow (contrast). `0.0` is neutral. See [`apply_levels`].
    pub black_point: f32,
    /// Levels white point: the tone-curve output that maps to display white
    /// (values above clip). `1.0` is neutral.
    pub white_point: f32,
    /// Levels midtone gamma (the Photoshop midpoint slider): `out = n^(1/gamma)`
    /// on the black/white-normalized signal. `> 1` brightens mids, `< 1` crushes
    /// them (haze suppression). `1.0` is neutral. Must be `> 0`.
    pub gamma: f32,
    /// Optional local (spatially-adaptive) tone compression, applied image-wide
    /// in linear space BEFORE the per-pixel exposure/tone-curve path (`None` ⇒
    /// off, bit-identical). Like [`bloom`] it is an image-space op — `grade_file`
    /// runs it; the per-pixel [`tonemap`] cannot and does not. See
    /// [`LocalToneConfig`]: it relieves the additive-splat "white-blob" by
    /// pulling exposure down where the surround is bright, so sub-cores inside
    /// the blob survive the tone curve.
    pub local: Option<LocalToneConfig>,
}

impl Default for GradeConfig {
    fn default() -> Self {
        GradeConfig {
            exposure: 1.0,
            tonemap: ToneMap::AcesApprox,
            bloom: None,
            black_point: 0.0,
            white_point: 1.0,
            gamma: 1.0,
            local: None,
        }
    }
}

impl GradeConfig {
    /// Validate the levels window and gamma. `black < white` (finite) and
    /// `gamma > 0` (finite); the neutral defaults `(0, 1, 1)` pass. Called by
    /// [`grade_file`] before a frame is touched.
    pub fn validate(&self) -> Result<(), GradeError> {
        let (b, w, g) = (self.black_point, self.white_point, self.gamma);
        if !b.is_finite() || !w.is_finite() || !g.is_finite() {
            return Err(GradeError::Config(format!(
                "levels must be finite: black={b}, white={w}, gamma={g}"
            )));
        }
        if b >= w {
            return Err(GradeError::Config(format!(
                "levels black_point ({b}) must be < white_point ({w})"
            )));
        }
        if g <= 0.0 {
            return Err(GradeError::Config(format!(
                "levels gamma ({g}) must be > 0"
            )));
        }
        if let Some(l) = &self.local {
            let (s, r, f) = (l.strength, l.radius, l.floor);
            if !s.is_finite() || !r.is_finite() || !f.is_finite() {
                return Err(GradeError::Config(format!(
                    "local knobs must be finite: strength={s}, radius={r}, floor={f}"
                )));
            }
            if s < 0.0 {
                // A negative strength would brighten bright regions — the operator
                // inverts and stops being a "never brightens" compressor.
                return Err(GradeError::Config(format!(
                    "local strength ({s}) must be >= 0"
                )));
            }
            if !(0.0..=1.0).contains(&f) {
                // floor > 1 would brighten; floor < 0 is meaningless.
                return Err(GradeError::Config(format!(
                    "local floor ({f}) must be in [0, 1]"
                )));
            }
        }
        Ok(())
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
    /// A malformed grading configuration (bad levels window or gamma).
    #[error("invalid grade config: {0}")]
    Config(String),
}

/// Apply the tone curve `op` to a linear (already exposure-scaled) RGB triple,
/// producing display-referred values in `[0, 1]`.
pub fn tone_curve(c: [f32; 3], op: ToneMap) -> [f32; 3] {
    c.map(|x| match op {
        ToneMap::AcesApprox => aces_approx(x),
        ToneMap::Reinhard => (x / (1.0 + x)).clamp(0.0, 1.0),
        ToneMap::Asinh { beta } => asinh_stretch(x, beta),
    })
}

/// Lupton-style asinh stretch `β·asinh(x/β)`, clamped to `[0, 1]`. Linear (unit
/// slope) for `x ≪ β`, logarithmic for `x ≫ β` — so as β grows the curve tends to
/// the identity, and for small β the highlights are held far below Reinhard's
/// asymptote. β is floored at `f32::MIN_POSITIVE`: at exactly `0.0` the raw
/// expression is `0·asinh(∞) = NaN`, and one NaN would poison the graded frame.
fn asinh_stretch(x: f32, beta: f32) -> f32 {
    let beta = beta.max(f32::MIN_POSITIVE);
    (beta * (x / beta).asinh()).clamp(0.0, 1.0)
}

/// Narkowicz (2015) ACES filmic approximation, clamped to `[0, 1]`.
fn aces_approx(x: f32) -> f32 {
    const A: f32 = 2.51;
    const B: f32 = 0.03;
    const C: f32 = 2.43;
    const D: f32 = 0.59;
    const E: f32 = 0.14;
    ((x * (A * x + B)) / (x * (C * x + D) + E)).clamp(0.0, 1.0)
}

/// The sRGB opto-electronic transfer function (linear `[0,1]` → sRGB `[0,1]`).
pub fn linear_to_srgb(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    if x <= 0.003_130_8 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

/// The levels curve on a display-referred value `x ∈ [0, 1]`:
/// `clamp((x − black)/(white − black), 0, 1) ^ (1/gamma)`. The neutral triple
/// `(0, 1, 1)` is the EXACT identity (special-cased so no `powf(1.0)` bit drift
/// leaks into the shipped neutral grade). Assumes a validated config
/// (`black < white`, `gamma > 0` — see [`GradeConfig::validate`]).
pub fn apply_levels(x: f32, black: f32, white: f32, gamma: f32) -> f32 {
    // Neutral is the EXACT identity — return `x` verbatim so no normalize/`powf`
    // bit drift leaks into the shipped neutral grade.
    if black == 0.0 && white == 1.0 && gamma == 1.0 {
        return x;
    }
    let denom = white - black;
    // A degenerate/inverted window (`white ≤ black`, rejected by `validate` but
    // reachable when `tonemap` is called directly) collapses to a hard threshold
    // at `white` — no `0/0` NaN leaking into the frame.
    if denom <= 0.0 {
        return if x >= white { 1.0 } else { 0.0 };
    }
    let n = ((x - black) / denom).clamp(0.0, 1.0);
    if gamma == 1.0 {
        n
    } else {
        n.powf(1.0 / gamma)
    }
}

/// Grade one linear-HDR pixel to a 16-bit sRGB triple: exposure → tone curve →
/// levels → sRGB encode → quantize to `[0, 65535]`.
pub fn tonemap(linear: [f32; 3], cfg: &GradeConfig) -> [u16; 3] {
    let exposed = linear.map(|c| c * cfg.exposure);
    let toned = tone_curve(exposed, cfg.tonemap);
    let mut out = [0u16; 3];
    for (o, &t) in out.iter_mut().zip(&toned) {
        // Levels run on the display-referred tone-curve output, before the OETF.
        let leveled = apply_levels(t, cfg.black_point, cfg.white_point, cfg.gamma);
        let s = linear_to_srgb(leveled);
        // Round-to-nearest into [0, 65535].
        out_quantize(o, s);
    }
    out
}

/// Quantize an sRGB value in `[0, 1]` to a 16-bit sample (round to nearest).
fn out_quantize(slot: &mut u16, srgb: f32) {
    *slot = (srgb.clamp(0.0, 1.0) * u16::MAX as f32 + 0.5) as u16;
}

/// Grade a linear-HDR OpenEXR file into a 16-bit sRGB PNG under `cfg`.
pub fn grade_file<P: AsRef<Path>, Q: AsRef<Path>>(
    exr_path: P,
    png_path: Q,
    cfg: &GradeConfig,
) -> Result<(), GradeError> {
    // Reject a degenerate levels window / gamma before touching the frame.
    cfg.validate()?;

    // Read the linear-HDR EXR into an RGB buffer (alpha dropped — grade is opaque).
    struct Rgb {
        w: usize,
        h: usize,
        px: Vec<[f32; 3]>,
    }
    let image = read_first_rgba_layer_from_file(
        exr_path.as_ref(),
        |resolution, _channels| Rgb {
            w: resolution.width(),
            h: resolution.height(),
            px: vec![[0.0; 3]; resolution.width() * resolution.height()],
        },
        |img: &mut Rgb, pos, (r, g, b, _a): (f32, f32, f32, f32)| {
            let i = pos.y() * img.w + pos.x();
            img.px[i] = [r, g, b];
        },
    )
    .map_err(|e| GradeError::Exr(e.to_string()))?;
    let mut rgb = image.layer_data.channel_data.pixels;

    // Bloom is an image-space op in LINEAR space — it must run over the whole
    // frame before the per-pixel exposure/tone-curve/quantize path.
    if let Some(bloom_cfg) = &cfg.bloom {
        rgb.px = bloom(&rgb.px, rgb.w, rgb.h, bloom_cfg);
    }

    // Local (spatially-adaptive) tone compression is likewise an image-space op:
    // it needs the surround, which the per-pixel path cannot see. It runs after
    // bloom (so it adapts to the final linear image) and before the per-pixel
    // exposure/tone-curve path — the gain is applied to the pre-exposure RGB and
    // the surround is computed on the exposed luminance (see `apply_local_tonemap`).
    if let Some(local_cfg) = &cfg.local {
        rgb.px = apply_local_tonemap(&rgb.px, rgb.w, rgb.h, cfg.exposure, local_cfg);
    }

    // Tonemap each pixel to a 16-bit sRGB triple, packed big-endian for PNG.
    let mut bytes = Vec::with_capacity(rgb.w * rgb.h * 6);
    for p in &rgb.px {
        for sample in tonemap(*p, cfg) {
            bytes.extend_from_slice(&sample.to_be_bytes());
        }
    }

    let writer = BufWriter::new(File::create(png_path)?);
    let mut encoder = png::Encoder::new(writer, rgb.w as u32, rgb.h as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut png_writer = encoder
        .write_header()
        .map_err(|e| GradeError::Png(e.to_string()))?;
    png_writer
        .write_image_data(&bytes)
        .map_err(|e| GradeError::Png(e.to_string()))?;
    png_writer
        .finish()
        .map_err(|e| GradeError::Png(e.to_string()))?;
    Ok(())
}
