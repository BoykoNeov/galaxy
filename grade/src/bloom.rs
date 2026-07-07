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
//! The two gated conservation laws are split between two mechanisms, because a
//! local, data-independent pipeline cannot deliver both exactly at the borders
//! (scatter normalization gives exact flux but piles the reflected halo flux into
//! a bright band along the frame edges — found on the first rendered A/B; gather
//! normalization gives exact constants but leaks flux at the same edges):
//!
//! * **Constants → constants (no border band):** every pyramid stage (tent
//!   downsample, Gaussian blur, tent upsample) is a *gather of convex
//!   combinations* over symmetric-reflected (edge-inclusive) neighbourhoods —
//!   mean-valued mips. A constant image passes through every stage unchanged, so
//!   no boundary handling can brighten or darken the frame edges.
//! * **Exact flux budget:** the summed halo is renormalized by one scalar,
//!   `strength · flux(img) / flux(halo)`, so the mix adds *exactly*
//!   `strength × total flux`. Both fluxes are summed in f64 over **sorted**
//!   channel values — permutation-invariant, so translated content yields the
//!   bit-identical renormalizer (the translation-equivariance gate).
//!
//! The tent kernels put the coarse taps ON the even fine pixels — no half-pixel
//! drift, so a centered impulse blooms into a dihedrally symmetric halo (the
//! house downsample gate) and 2^levels shifts are exactly equivariant.

/// Bloom configuration. `strength = 0` (or `levels = 0`) is a bit-exact no-op.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BloomConfig {
    /// Halo mix: `out = img + strength · halo`. The halo is renormalized to carry
    /// exactly the image's flux, so this adds exactly `strength × total flux`.
    pub strength: f32,
    /// Mip-pyramid depth; the halo footprint doubles per level. Capped at the
    /// 1×1 mip floor for small images.
    pub levels: u32,
    /// Gaussian σ in pixels, applied at every pyramid level (so the same knob
    /// scales every halo octave together). Non-positive or non-finite values
    /// degenerate to no blur (a δ kernel) — total, never NaN.
    pub radius: f32,
}

/// Apply bloom to a row-major linear-HDR image (nonnegative linear light):
/// `out = img + strength · halo`, halo = flux-renormalized mean over mip levels
/// of (downsample^ℓ → Gaussian blur → upsample^ℓ). Constants bloom to constants
/// (no border band) and `flux(out) = (1 + strength) · flux(img)` exactly.
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

    // The exact flux budget: scale the summed halo so the mix adds exactly
    // strength·flux(img) (the per-level mean is folded in — each level carries
    // ≈ the image's mean flux). A halo with no flux only arises from a fluxless
    // image (every stage averages nonnegative input); adding nothing is then
    // both correct and total.
    let halo_flux = sorted_flux(&halo);
    if !(halo_flux > 0.0 && halo_flux.is_finite()) {
        return pixels.to_vec();
    }
    let coeff = (cfg.strength as f64 * sorted_flux(pixels) / halo_flux) as f32;
    pixels
        .iter()
        .zip(&halo)
        .map(|(p, h)| {
            [
                p[0] + coeff * h[0],
                p[1] + coeff * h[1],
                p[2] + coeff * h[2],
            ]
        })
        .collect()
}

/// Tent (Burt–Adelson) 2× downsample to mean-valued mips: coarse tap X sits on
/// fine pixel 2X (no half-pixel drift) and gathers `¼·f(2X−1) + ½·f(2X) +
/// ¼·f(2X+1)` per dimension, with symmetric-reflected borders — a convex
/// combination, so constants reduce to constants exactly.
fn reduce(img: &[[f32; 3]], w: usize, h: usize) -> (Vec<[f32; 3]>, usize, usize) {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    // Rows: w → cw.
    let mut rows = vec![[0.0f32; 3]; cw * h];
    for y in 0..h {
        let src = &img[y * w..(y + 1) * w];
        for (x, d) in rows[y * cw..(y + 1) * cw].iter_mut().enumerate() {
            let c = 2 * x as i64;
            let (l, m, r) = (
                &src[reflect(c - 1, w)],
                &src[c as usize],
                &src[reflect(c + 1, w)],
            );
            for ch in 0..3 {
                d[ch] = 0.25 * l[ch] + 0.5 * m[ch] + 0.25 * r[ch];
            }
        }
    }
    // Columns: h → ch.
    let mut out = vec![[0.0f32; 3]; cw * ch];
    for y in 0..ch {
        let c = 2 * y as i64;
        let (t, m, b) = (reflect(c - 1, h), c as usize, reflect(c + 1, h));
        for x in 0..cw {
            let (tv, mv, bv) = (&rows[t * cw + x], &rows[m * cw + x], &rows[b * cw + x]);
            for ch in 0..3 {
                out[y * cw + x][ch] = 0.25 * tv[ch] + 0.5 * mv[ch] + 0.25 * bv[ch];
            }
        }
    }
    (out, cw, ch)
}

/// Tent 2× upsample to `(tw, th)`: linear interpolation on the tap grid. Fine
/// pixel 2X copies tap X; fine pixel 2X+1 averages taps X and X+1 (reflected at
/// the even-target edge) — convex combinations, so constants expand to constants
/// exactly, and a monotone halo stays monotone and blockless.
fn expand(img: &[[f32; 3]], cw: usize, ch: usize, tw: usize, th: usize) -> Vec<[f32; 3]> {
    // Rows: cw → tw.
    let mut rows = vec![[0.0f32; 3]; tw * ch];
    for y in 0..ch {
        let src = &img[y * cw..(y + 1) * cw];
        for (x, d) in rows[y * tw..(y + 1) * tw].iter_mut().enumerate() {
            if x % 2 == 0 {
                *d = src[x / 2];
            } else {
                let (l, r) = (&src[(x - 1) / 2], &src[reflect(x.div_ceil(2) as i64, cw)]);
                for ch in 0..3 {
                    d[ch] = 0.5 * (l[ch] + r[ch]);
                }
            }
        }
    }
    // Columns: ch → th.
    let mut out = vec![[0.0f32; 3]; tw * th];
    for y in 0..th {
        if y % 2 == 0 {
            out[y * tw..(y + 1) * tw].copy_from_slice(&rows[(y / 2) * tw..(y / 2 + 1) * tw]);
        } else {
            let (t, b) = ((y - 1) / 2, reflect(y.div_ceil(2) as i64, ch));
            for x in 0..tw {
                for ch in 0..3 {
                    out[y * tw + x][ch] = 0.5 * (rows[t * tw + x][ch] + rows[b * tw + x][ch]);
                }
            }
        }
    }
    out
}

/// Separable Gaussian convolution (gather) with symmetric-reflected borders. The
/// kernel is normalized, so every output is a convex combination — constants blur
/// to constants (to kernel-normalization fp), and no flux piles up at the edges.
fn blur(img: &[[f32; 3]], w: usize, h: usize, kernel: &[f32]) -> Vec<[f32; 3]> {
    let r = (kernel.len() / 2) as i64;
    // Rows.
    let mut rows = vec![[0.0f32; 3]; w * h];
    for y in 0..h {
        let src = &img[y * w..(y + 1) * w];
        for (x, d) in rows[y * w..(y + 1) * w].iter_mut().enumerate() {
            for (j, &k) in kernel.iter().enumerate() {
                let s = &src[reflect(x as i64 + j as i64 - r, w)];
                for ch in 0..3 {
                    d[ch] += k * s[ch];
                }
            }
        }
    }
    // Columns.
    let mut out = vec![[0.0f32; 3]; w * h];
    for y in 0..h {
        for (j, &k) in kernel.iter().enumerate() {
            let s = reflect(y as i64 + j as i64 - r, h);
            for x in 0..w {
                for ch in 0..3 {
                    out[y * w + x][ch] += k * rows[s * w + x][ch];
                }
            }
        }
    }
    out
}

/// Symmetric (edge-inclusive) reflection of index `i` into `[0, n)`:
/// `… 2 1 0 | 0 1 2 … n-1 | n-1 n-2 …` — total for any overrun (kernels wider
/// than the image keep bouncing).
pub(crate) fn reflect(i: i64, n: usize) -> usize {
    let period = 2 * n as i64;
    let m = i.rem_euclid(period);
    if m < n as i64 {
        m as usize
    } else {
        (period - 1 - m) as usize
    }
}

/// Total flux in f64 over **sorted** channel values: permutation-invariant, so
/// images with identical value multisets (e.g. translated halos) produce the
/// bit-identical sum — the flux renormalizer must not depend on WHERE the flux
/// sits, or translation equivariance would break in the last ulp.
fn sorted_flux(px: &[[f32; 3]]) -> f64 {
    let mut v: Vec<f32> = px.iter().flat_map(|p| p.iter().copied()).collect();
    v.sort_unstable_by(f32::total_cmp);
    v.iter().map(|&c| c as f64).sum()
}

/// Normalized Gaussian kernel truncated at ⌈3σ⌉. Non-positive/non-finite σ (and
/// NaN, via `max`) degenerate to the δ kernel `[1.0]` — blur stays total.
pub(crate) fn gaussian_kernel(sigma: f32) -> Vec<f32> {
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
