//! Local (spatially-adaptive) tone compression — the fix for the "approach
//! white-blob": where many additive star splats pile into one region the global
//! tone curve saturates the whole core to a flat, featureless white. A global
//! curve *cannot* recover it — the pixels there are genuinely bright and every
//! one maps to the same near-1.0 output. The cure has to be spatial: pull down
//! exposure *where the surround is bright*, so the sub-cores inside the blob drop
//! back into the tone curve's responsive range and their internal structure
//! survives.
//!
//! Design (advisor-vetted, see the session notes):
//!
//! * **Spatial, never temporal.** The gain is a pure function of *this* frame's
//!   pixels and a fixed config — there is NO per-frame log-average "key". `grade`
//!   applies one [`crate::GradeConfig`] across a whole 1000-frame sequence
//!   independently; a content-derived key would make the effective exposure pump
//!   frame-to-frame and would actively fight the core-brightening we may want to
//!   *show* during a merger. `strength` is dimensionless against the **absolute**
//!   exposed luminance (a fixed reference of 1.0 folded into the knob), so
//!   equal input always yields equal gain.
//!
//! * **A scalar gain map, before the unchanged global curve.** We compute a
//!   per-pixel `g(x) = max(floor, 1 / (1 + strength·V(x)))`, where `V` is a
//!   large-σ Gaussian low-pass of the exposed luminance (the "surround"), and
//!   multiply the linear RGB by it. Because `g` is a single scalar per pixel the
//!   hue is preserved exactly (chroma ratios cancel), and because `g ≤ 1` it
//!   only ever *darkens* — never brightens, never inverts. Where the surround is
//!   dim, `V ≈ 0` so `g ≈ 1` and the pixel passes through untouched; the global
//!   [`crate::tonemap`] then handles it exactly as before. This keeps the
//!   per-pixel path global (all spatial logic lives in [`crate::grade_file`],
//!   the same split bloom uses) and makes `local: None` bit-identical.
//!
//! * **Halos are bounded, not eliminated.** A single-scale Gaussian surround
//!   produces a faint dark ring around bright regions (gradient reversal). The
//!   `floor` clamps the gain from below, bounding the darkening; the residual is
//!   a look call settled by an A/B, not a unit test. Bilateral / multi-scale is
//!   deferred until an A/B shows the single-scale ring is objectionable.

/// Local tone-compression configuration. `strength = 0` is a bit-exact no-op
/// (`g ≡ 1`), matching the neutral-knob convention of the rest of the grade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalToneConfig {
    /// Compression strength `k` in `g = 1 / (1 + k·V)`. `0` ⇒ identity gain.
    /// Larger `k` pulls bright surrounds down harder (more blob relief). Must be
    /// finite and `≥ 0` — a negative `k` would *brighten* bright regions and
    /// invert the operator.
    pub strength: f32,
    /// Gaussian σ (pixels) of the surround low-pass `V`. This is the critical
    /// knob: large enough to smooth the blob into a single surround value, small
    /// enough not to collapse to the whole-frame mean. Non-positive/non-finite
    /// degenerates to a δ kernel (`V = luminance`), like bloom's radius.
    pub radius: f32,
    /// Gain floor `g_min ∈ [0, 1]`: the hardest the operator may darken a pixel,
    /// which bounds the dark-halo ring. `1.0` pins `g ≡ 1` (no compression); a
    /// small value permits deep blob relief at the cost of a stronger ring.
    pub floor: f32,
}

/// The pointwise local gain for a given surround luminance `V`:
/// `g = max(floor, 1 / (1 + strength·V))`. Monotone *decreasing* in `V` (more
/// surround never raises the gain — the "never brightens" safety property) and
/// always in `[floor, 1]`. `strength = 0` returns exactly `1.0`.
pub fn local_gain(surround_luminance: f32, cfg: &LocalToneConfig) -> f32 {
    // Short-circuit the identity: `1/(1 + 0·V)` is already `1.0`, but the guard
    // also keeps a degenerate `V = +∞` (from `0·∞`) out of the arithmetic.
    if cfg.strength == 0.0 {
        return 1.0;
    }
    let g = 1.0 / (1.0 + cfg.strength * surround_luminance);
    g.max(cfg.floor)
}

/// Apply the local tone-compression gain map to a row-major linear-HDR image,
/// returning the gain-scaled linear RGB (still pre-exposure — the per-pixel
/// [`crate::tonemap`] re-applies `exposure` downstream). `V` is computed on the
/// **exposed** luminance (`exposure · Rec.709(rgb)`) so `strength` is calibrated
/// against where the tone curve actually saturates; the returned scale is applied
/// to the pre-exposure RGB (gain and exposure are both scalar multiplies and
/// commute). `strength = 0` returns a bit-exact copy of the input.
///
/// # Panics
/// If `pixels.len() != width * height` (programmer error, not a data path).
pub fn apply_local_tonemap(
    pixels: &[[f32; 3]],
    width: usize,
    height: usize,
    exposure: f32,
    cfg: &LocalToneConfig,
) -> Vec<[f32; 3]> {
    assert_eq!(
        pixels.len(),
        width * height,
        "apply_local_tonemap: pixel buffer does not match {width}x{height}"
    );
    // `strength = 0` (or an empty frame) is a bit-exact no-op.
    if pixels.is_empty() || cfg.strength == 0.0 {
        return pixels.to_vec();
    }

    // Surround = large-σ Gaussian low-pass of the EXPOSED luminance, so `strength`
    // is calibrated against where the tone curve actually saturates.
    let lum: Vec<f32> = pixels.iter().map(|p| exposure * luminance(*p)).collect();
    let kernel = crate::bloom::gaussian_kernel(cfg.radius);
    let surround = blur_scalar(&lum, width, height, &kernel);

    // Scale the (pre-exposure) linear RGB by the per-pixel scalar gain. A single
    // scalar per pixel preserves hue exactly and, with `g ≤ 1`, only darkens.
    pixels
        .iter()
        .zip(&surround)
        .map(|(p, &v)| {
            let g = local_gain(v, cfg);
            [p[0] * g, p[1] * g, p[2] * g]
        })
        .collect()
}

/// Rec.709 relative luminance of a linear RGB triple.
fn luminance(p: [f32; 3]) -> f32 {
    0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]
}

/// Separable Gaussian convolution of a scalar field with symmetric-reflected
/// borders (the same edge handling and kernel as [`crate::bloom`]). The kernel is
/// normalized, so a constant field blurs to a constant — the surround of a flat
/// region is that region's value.
fn blur_scalar(field: &[f32], w: usize, h: usize, kernel: &[f32]) -> Vec<f32> {
    use crate::bloom::reflect;
    let r = (kernel.len() / 2) as i64;
    // Rows.
    let mut rows = vec![0.0f32; w * h];
    for y in 0..h {
        let src = &field[y * w..(y + 1) * w];
        for (x, d) in rows[y * w..(y + 1) * w].iter_mut().enumerate() {
            for (j, &k) in kernel.iter().enumerate() {
                *d += k * src[reflect(x as i64 + j as i64 - r, w)];
            }
        }
    }
    // Columns.
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for (j, &k) in kernel.iter().enumerate() {
            let s = reflect(y as i64 + j as i64 - r, h);
            for x in 0..w {
                out[y * w + x] += k * rows[s * w + x];
            }
        }
    }
    out
}
