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
//!
//! Every stage (tent downsample, Gaussian blur, tent upsample) is written in
//! **scatter form, normalized per source pixel** — each input pixel distributes
//! exactly its own flux, with out-of-range taps deposited at the clamped border
//! index. That makes total flux conservation *exact by construction* (the gated
//! `flux(out) = (1 + strength)·flux(img)` invariant), where the usual
//! gather-with-clamp formulation loses flux at every edge. The tent (Burt–Adelson)
//! down/up kernels put the coarse taps ON the even fine pixels — no half-pixel
//! drift, so a centered impulse blooms into a dihedrally symmetric halo (the house
//! downsample gate) and 2^levels shifts are exactly equivariant.

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
    /// scales every halo octave together). Non-positive or non-finite values
    /// degenerate to no blur (a δ kernel) — total, never NaN.
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
    if pixels.is_empty() || cfg.strength == 0.0 || cfg.levels == 0 {
        return pixels.to_vec();
    }

    // Downsample chain: chain[ℓ-1] = reduce^ℓ(img), capped at the 1×1 mip floor
    // (further levels would be identical content — a cap, not an error).
    let mut chain: Vec<(Vec<[f32; 3]>, usize, usize)> = Vec::new();
    {
        let (mut src, mut w, mut h) = (pixels, width, height);
        for _ in 0..cfg.levels {
            if w == 1 && h == 1 {
                break;
            }
            chain.push(reduce(src, w, h));
            let last = chain.last().expect("just pushed");
            (src, w, h) = (&last.0, last.1, last.2);
        }
    }
    if chain.is_empty() {
        // 1×1 image: no coarser mip exists, so there is no halo to add.
        return pixels.to_vec();
    }

    // Collapse coarse → fine: acc_L = blur(chain[L]); acc_ℓ = blur(chain[ℓ]) +
    // expand(acc_{ℓ+1}). By linearity acc_1 expanded to full res is exactly
    // Σ_ℓ expand^ℓ(blur(reduce^ℓ(img))) — and the repeated tent expands smooth
    // each octave's blocks on the way up.
    let kernel = gaussian_kernel(cfg.radius);
    let (coarsest, cw, ch) = chain.last().expect("chain is non-empty");
    let mut acc = blur(coarsest, *cw, *ch, &kernel);
    let (mut aw, mut ah) = (*cw, *ch);
    for (img_l, lw, lh) in chain.iter().rev().skip(1) {
        let mut up = expand(&acc, aw, ah, *lw, *lh);
        for (u, b) in up.iter_mut().zip(blur(img_l, *lw, *lh, &kernel)) {
            for c in 0..3 {
                u[c] += b[c];
            }
        }
        (acc, aw, ah) = (up, *lw, *lh);
    }
    let halo = expand(&acc, aw, ah, width, height);

    // Mean over levels: each level's halo carries exactly the image's flux, so
    // the mix adds exactly strength·flux.
    let s = cfg.strength / chain.len() as f32;
    pixels
        .iter()
        .zip(&halo)
        .map(|(p, h)| [p[0] + s * h[0], p[1] + s * h[1], p[2] + s * h[2]])
        .collect()
}

/// Tent (Burt–Adelson) 2× downsample, separable, scatter-normalized. Coarse taps
/// sit on the EVEN fine pixels (coarse dim = ⌈n/2⌉): an even fine pixel sends its
/// whole value to its own tap; an odd one splits half/half between its two
/// neighbouring taps. Per-source weights sum to 1 ⇒ flux is conserved exactly
/// (coarse pixel *values* are ~4× — they hold the flux of a 2×2 patch).
fn reduce(img: &[[f32; 3]], w: usize, h: usize) -> (Vec<[f32; 3]>, usize, usize) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    // Rows: w → cw.
    let mut rows = vec![[0.0f32; 3]; cw * h];
    for y in 0..h {
        let (src, dst) = (&img[y * w..(y + 1) * w], &mut rows[y * cw..(y + 1) * cw]);
        for (x, v) in src.iter().enumerate() {
            if x % 2 == 0 {
                deposit(&mut dst[x / 2], v, 1.0);
            } else {
                deposit(&mut dst[(x - 1) / 2], v, 0.5);
                deposit(&mut dst[x.div_ceil(2).min(cw - 1)], v, 0.5);
            }
        }
    }
    // Columns: h → ch.
    let mut out = vec![[0.0f32; 3]; cw * ch];
    for y in 0..h {
        for x in 0..cw {
            let v = &rows[y * cw + x];
            if y % 2 == 0 {
                deposit(&mut out[(y / 2) * cw + x], v, 1.0);
            } else {
                deposit(&mut out[((y - 1) / 2) * cw + x], v, 0.5);
                deposit(&mut out[y.div_ceil(2).min(ch - 1) * cw + x], v, 0.5);
            }
        }
    }
    (out, cw, ch)
}

/// Tent 2× upsample to `(tw, th)`, the adjoint of [`reduce`]: each coarse tap X
/// distributes ½ to fine pixel 2X and ¼ to each fine neighbour 2X±1 (clamped to
/// the border, so per-source weights still sum to 1 ⇒ flux exact). Equivalent to
/// nearest-neighbour spreading followed by a [¼, ½, ¼] tent — i.e. linear
/// interpolation, which keeps an upsampled monotone halo monotone and blockless.
fn expand(img: &[[f32; 3]], cw: usize, ch: usize, tw: usize, th: usize) -> Vec<[f32; 3]> {
    // Rows: cw → tw.
    let mut rows = vec![[0.0f32; 3]; tw * ch];
    for y in 0..ch {
        let (src, dst) = (&img[y * cw..(y + 1) * cw], &mut rows[y * tw..(y + 1) * tw]);
        for (x, v) in src.iter().enumerate() {
            let base = 2 * x;
            deposit(&mut dst[base], v, 0.5);
            deposit(&mut dst[base.saturating_sub(1)], v, 0.25);
            deposit(&mut dst[(base + 1).min(tw - 1)], v, 0.25);
        }
    }
    // Columns: ch → th.
    let mut out = vec![[0.0f32; 3]; tw * th];
    for y in 0..ch {
        let base = 2 * y;
        for x in 0..tw {
            let v = &rows[y * tw + x];
            deposit(&mut out[base * tw + x], v, 0.5);
            deposit(&mut out[base.saturating_sub(1) * tw + x], v, 0.25);
            deposit(&mut out[(base + 1).min(th - 1) * tw + x], v, 0.25);
        }
    }
    out
}

/// Separable Gaussian blur, scatter form: each source pixel distributes its value
/// through the normalized kernel, out-of-range taps clamped to the border index —
/// per-source weights sum to 1 ⇒ flux conserved exactly (gather-with-clamp would
/// leak it at every edge).
fn blur(img: &[[f32; 3]], w: usize, h: usize, kernel: &[f32]) -> Vec<[f32; 3]> {
    let r = kernel.len() / 2;
    // Rows.
    let mut rows = vec![[0.0f32; 3]; w * h];
    for y in 0..h {
        let (src, dst) = (&img[y * w..(y + 1) * w], &mut rows[y * w..(y + 1) * w]);
        for (x, v) in src.iter().enumerate() {
            for (j, &k) in kernel.iter().enumerate() {
                let t = (x + j).saturating_sub(r).min(w - 1);
                deposit(&mut dst[t], v, k);
            }
        }
    }
    // Columns.
    let mut out = vec![[0.0f32; 3]; w * h];
    for y in 0..h {
        for x in 0..w {
            let v = &rows[y * w + x];
            for (j, &k) in kernel.iter().enumerate() {
                let t = (y + j).saturating_sub(r).min(h - 1);
                deposit(&mut out[t * w + x], v, k);
            }
        }
    }
    out
}

/// Normalized Gaussian kernel truncated at ⌈3σ⌉. Non-positive/non-finite σ (and
/// NaN, via `max`) degenerate to the δ kernel `[1.0]` — blur stays total.
fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let sigma = if sigma.is_finite() {
        sigma.max(0.0)
    } else {
        0.0
    };
    if sigma == 0.0 {
        return vec![1.0];
    }
    let r = (3.0 * sigma).ceil() as i64;
    let mut k: Vec<f32> = (-r..=r)
        .map(|o| (-((o * o) as f32) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = k.iter().sum();
    for v in &mut k {
        *v /= sum;
    }
    k
}

#[inline]
fn deposit(dst: &mut [f32; 3], v: &[f32; 3], weight: f32) {
    for c in 0..3 {
        dst[c] += weight * v[c];
    }
}
