//! HDR bloom (M6b): the emissive-star-field halo, as a **pure CPU** linear-domain
//! image op — mip pyramid down, separable Gaussian per level, weighted up-add:
//! `out = img + strength · halo`.
//!
//! Placement (a deliberate deviation from DESIGN's render-stage recipe, argued in
//! `docs/plans/cinematic-toomre-bloom.md`): bloom runs at **grade time**, in linear
//! space *before* the tone curve, so the retained EXR stays the pristine pre-bloom
//! artifact and bloom strength/radius iterate seconds-cheap through `regrade`.
//! There is **no bright-pass threshold** — in the linear astro look every source
//! blooms in proportion to its flux, which keeps the operator linear (a gated
//! invariant).

/// Bloom configuration. `strength = 0` (or `levels = 0`) is a bit-exact no-op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomConfig {
    /// Halo mix: `out = img + strength · halo`. The halo carries exactly the
    /// image's flux, so this adds exactly `strength × total flux`.
    pub strength: f32,
    /// Mip-pyramid depth; the halo footprint doubles per level. Capped at the
    /// 1×1 mip floor for small images.
    pub levels: u32,
    /// Gaussian σ in pixels, applied at every pyramid level (so the same knob
    /// scales every halo octave together).
    pub radius: f32,
}

/// Apply bloom to a row-major linear-HDR image: `out = img + strength · halo`,
/// halo = mean over mip levels of (downsample^ℓ → Gaussian blur → upsample^ℓ).
/// Every stage is normalized per source pixel, so total flux is conserved:
/// `flux(out) = (1 + strength) · flux(img)`.
///
/// # Panics
/// If `pixels.len() != width * height` (programmer error, not a data path).
pub fn bloom(pixels: &[[f32; 3]], width: usize, height: usize, cfg: &BloomConfig) -> Vec<[f32; 3]> {
    assert_eq!(
        pixels.len(),
        width * height,
        "bloom: pixel buffer does not match {width}x{height}"
    );
    todo!("M6b: mip-pyramid HDR bloom under {cfg:?}")
}
