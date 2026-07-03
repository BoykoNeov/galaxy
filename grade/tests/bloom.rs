//! HDR bloom gates (DESIGN.md M6b). Bloom is a **pure CPU, linear-domain** image
//! op applied before the tone curve: `out = img + strength · halo`, where the halo
//! is a weighted mip-pyramid of Gaussian blurs (no bright-pass threshold — the
//! linear astro look blooms *everything* in proportion to its flux).
//!
//! Expectations are invariants of the operator definition, not read-back outputs:
//!   - `strength = 0` is a bit-exact no-op (so wiring bloom in changes nothing
//!     until it is asked for), as is `levels = 0` (an empty pyramid has no halo).
//!   - Bloom is LINEAR: `bloom(2·img) = 2·bloom(img)` **bit-exactly** — scaling by
//!     a power of two only shifts f32 exponents, so rounding commutes with it; any
//!     bright-pass threshold or nonlinearity would break this gate.
//!   - Every pyramid stage (downsample, blur, upsample) is normalized per *source*
//!     pixel, so total flux is conserved and the mix adds exactly `strength·flux`:
//!     `flux(out) = (1 + strength)·flux(img)` to fp tolerance.
//!   - A centered impulse on an odd-dimension image (the house downsample gate:
//!     odd dims put the center pixel exactly on the coarse-grid taps) produces a
//!     halo with full dihedral (mirror + transpose) symmetry that decays
//!     monotonically from the center.
//!   - Interior impulses shifted by the coarsest mip stride (2^levels — the only
//!     shifts that leave the pyramid grid alignment unchanged) shift the halo
//!     bit-exactly.
//!   - Deterministic and total (huge `levels` on a tiny image must cap, not hang).

use galaxy_grade::{bloom, BloomConfig};

/// Deterministic pseudo-random HDR-ish image (values in [0, 4)) via a bare LCG —
/// no rand dependency, same bits every run.
fn lcg_image(w: usize, h: usize, seed: u64) -> Vec<[f32; 3]> {
    let mut s = seed;
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) as f32 / (1u64 << 31) as f32) * 4.0
    };
    (0..w * h).map(|_| [next(), next(), next()]).collect()
}

/// A black image with one bright grey impulse at (x, y).
fn impulse(w: usize, h: usize, x: usize, y: usize, v: f32) -> Vec<[f32; 3]> {
    let mut px = vec![[0.0f32; 3]; w * h];
    px[y * w + x] = [v; 3];
    px
}

/// Total flux (sum over pixels and channels), accumulated in f64 so the test's
/// own summation error is far below the gate tolerance.
fn flux(px: &[[f32; 3]]) -> f64 {
    px.iter().flat_map(|p| p.iter()).map(|&c| c as f64).sum()
}

#[test]
fn strength_zero_is_a_bitexact_noop() {
    let img = lcg_image(37, 23, 0xB100_0001);
    let cfg = BloomConfig {
        strength: 0.0,
        levels: 4,
        radius: 2.0,
    };
    assert_eq!(bloom(&img, 37, 23, &cfg), img);
}

#[test]
fn levels_zero_is_a_bitexact_noop() {
    // An empty pyramid has no halo to add — documented degenerate, not an error.
    let img = lcg_image(16, 9, 0xB100_0002);
    let cfg = BloomConfig {
        strength: 0.8,
        levels: 0,
        radius: 2.0,
    };
    assert_eq!(bloom(&img, 16, 9, &cfg), img);
}

#[test]
fn bloom_is_linear_bitexact_under_power_of_two_scaling() {
    // No bright-pass threshold: bloom must be a linear operator. Gated at ×2,
    // where fp rounding commutes exactly (exponent shift only), so ANY
    // threshold/knee/nonlinearity fails this bit-exactly rather than within a
    // tolerance it could hide under.
    let img = lcg_image(32, 20, 0xB100_0003);
    let doubled: Vec<[f32; 3]> = img.iter().map(|p| p.map(|c| c * 2.0)).collect();
    let cfg = BloomConfig {
        strength: 0.7,
        levels: 3,
        radius: 1.5,
    };
    let a: Vec<[f32; 3]> = bloom(&img, 32, 20, &cfg)
        .iter()
        .map(|p| p.map(|c| c * 2.0))
        .collect();
    let b = bloom(&doubled, 32, 20, &cfg);
    assert_eq!(a, b, "bloom(2·img) must equal 2·bloom(img) bit-exactly");
}

#[test]
fn bloom_conserves_total_flux() {
    // Every stage is per-source-normalized (scatter form), so the halo carries
    // exactly the image's flux and the mix adds exactly strength·flux. The 1e-5
    // relative gate is pure f32 rounding headroom (~dozens of ops per value at
    // ~6e-8 each); a gather-with-clamp kernel or unnormalized weights loses flux
    // at the percent level and fails loudly.
    for (w, h) in [(33usize, 17usize), (32, 20)] {
        let img = lcg_image(w, h, 0xB100_0004);
        let f_in = flux(&img);
        for strength in [0.5f32, 2.0] {
            for levels in [1u32, 4] {
                let cfg = BloomConfig {
                    strength,
                    levels,
                    radius: 2.0,
                };
                let f_out = flux(&bloom(&img, w, h, &cfg));
                let expected = (1.0 + strength as f64) * f_in;
                let rel = (f_out - expected).abs() / expected;
                assert!(
                    rel < 1e-5,
                    "{w}x{h} strength={strength} levels={levels}: \
                     flux {f_out} vs (1+s)·flux {expected} (rel err {rel:.2e})"
                );
            }
        }
    }
}

/// Halo of a centered impulse: `(bloom(img) − img) / strength`, on a 129×129
/// image whose center (64, 64) sits exactly on every mip level's taps. Footprint
/// (3σ·2³ ≈ 48 px + tent spread) stays inside the 64 px margin, so no edge clamp
/// perturbs the symmetry being gated.
fn center_impulse_halo() -> (Vec<f32>, usize) {
    const N: usize = 129;
    let img = impulse(N, N, 64, 64, 8.0);
    let cfg = BloomConfig {
        strength: 1.0,
        levels: 3,
        radius: 2.0,
    };
    let out = bloom(&img, N, N, &cfg);
    // Channels are identical (grey impulse) — keep the red channel.
    let halo: Vec<f32> = out
        .iter()
        .zip(&img)
        .map(|(o, i)| (o[0] - i[0]) / cfg.strength)
        .collect();
    (halo, N)
}

#[test]
fn center_impulse_halo_is_dihedrally_symmetric() {
    // The house downsample gate: odd dimensions, center pixel on the coarse taps.
    // The halo must be symmetric under x-mirror, y-mirror, and transpose (the
    // dihedral group of the square — the full symmetry a separable pipeline can
    // have). Tolerance is relative to the halo peak: reflections reverse fp
    // accumulation order, so exact bits are not owed, but any half-pixel shift in
    // a downsample/upsample kernel displaces the halo by whole pixels and fails.
    let (halo, n) = center_impulse_halo();
    let peak = halo.iter().fold(0.0f32, |m, &v| m.max(v));
    assert!(peak > 0.0, "impulse must produce a nonzero halo");
    let tol = 1e-5 * peak;
    let at = |x: usize, y: usize| halo[y * n + x];
    for y in 0..n {
        for x in 0..n {
            let v = at(x, y);
            for (mx, my, name) in [
                (n - 1 - x, y, "x-mirror"),
                (x, n - 1 - y, "y-mirror"),
                (y, x, "transpose"),
            ] {
                let m = at(mx, my);
                assert!(
                    (v - m).abs() <= tol,
                    "{name} broken at ({x},{y}): {v} vs {m} (tol {tol:.3e})"
                );
            }
        }
    }
}

#[test]
fn center_impulse_halo_decays_monotonically_from_center() {
    // Along the central row and column the halo is a sum of same-center unimodal
    // bumps (one per level), so it must be non-increasing walking outward from
    // the center — and its peak IS the center. Slack is fp-only: a real ringing
    // or block artifact overshoots by far more than 1e-7·peak.
    let (halo, n) = center_impulse_halo();
    let c = n / 2;
    let peak = halo[c * n + c];
    let slack = 1e-7 * peak;
    let global_max = halo.iter().fold(0.0f32, |m, &v| m.max(v));
    assert!(
        peak >= global_max - slack,
        "halo peak {global_max} not at the impulse (center {peak})"
    );
    for d in 1..=c {
        for (cur, prev, axis) in [
            (halo[c * n + c + d], halo[c * n + c + d - 1], "+x"),
            (halo[c * n + c - d], halo[c * n + c - (d - 1)], "-x"),
            (halo[(c + d) * n + c], halo[(c + d - 1) * n + c], "+y"),
            (halo[(c - d) * n + c], halo[(c - (d - 1)) * n + c], "-y"),
        ] {
            assert!(
                cur <= prev + slack,
                "halo not monotone along {axis} at distance {d}: {prev} then {cur}"
            );
            assert!(cur >= 0.0, "halo negative along {axis} at {d}: {cur}");
        }
    }
}

#[test]
fn interior_impulse_translates_by_the_mip_stride_bitexactly() {
    // Shifting an interior impulse by the coarsest mip stride (2^levels = 4)
    // leaves every pyramid level's grid alignment unchanged, so the halo is the
    // same bits, shifted: identical fp operations on identically-aligned data
    // (zero pixels contribute exact +0.0 terms). Both impulses sit ≥ 36 px from
    // every edge — beyond the footprint — so no clamp fires.
    const N: usize = 128;
    let cfg = BloomConfig {
        strength: 1.0,
        levels: 2,
        radius: 2.0,
    };
    let a = bloom(&impulse(N, N, 60, 64, 8.0), N, N, &cfg);
    let b = bloom(&impulse(N, N, 64, 64, 8.0), N, N, &cfg);
    for dy in -30i64..=30 {
        for dx in -30i64..=30 {
            let (x, y) = ((60 + dx) as usize, (64 + dy) as usize);
            let (sx, sy) = ((64 + dx) as usize, (64 + dy) as usize);
            assert_eq!(
                a[y * N + x],
                b[sy * N + sx],
                "halo not translation-equivariant at offset ({dx},{dy})"
            );
        }
    }
}

#[test]
fn constant_image_blooms_to_constant_everywhere() {
    // The border gate (found on the first rendered A/B: a bright band along the
    // frame edges). Global flux conservation alone allows boundary handling to
    // PILE the reflected halo flux onto border rows; a constant image makes that
    // visible as out[border] > out[interior]. The doubly-stochastic stance: every
    // stage must conserve flux (per-source normalization, gated above) AND map
    // constants to constants (per-target normalization, gated here) — then
    // bloom(c) = (1+strength)·c at EVERY pixel, and there is no band. Dimensions
    // 31×22 hit both border parities (odd and even) through the mip chain.
    // Tolerance is fp-only (~100 ops/value at ~6e-8): the clamp pile-up this
    // guards against is a 10–50% effect.
    const W: usize = 31;
    const H: usize = 22;
    const C: f32 = 0.6;
    let img = vec![[C; 3]; W * H];
    let cfg = BloomConfig {
        strength: 0.8,
        levels: 4,
        radius: 2.0,
    };
    let expected = (1.0 + cfg.strength) * C;
    for (i, p) in bloom(&img, W, H, &cfg).iter().enumerate() {
        for &c in p {
            let rel = (c - expected).abs() / expected;
            assert!(
                rel < 1e-4,
                "pixel ({},{}): {c} vs constant {expected} (rel err {rel:.2e})",
                i % W,
                i / W
            );
        }
    }
}

#[test]
fn tiny_image_with_huge_levels_stays_total() {
    // The pyramid must cap at the 1×1 mip floor rather than loop or degenerate;
    // flux conservation must survive the cap.
    let img = lcg_image(4, 3, 0xB100_0005);
    let cfg = BloomConfig {
        strength: 1.0,
        levels: 16,
        radius: 3.0,
    };
    let out = bloom(&img, 4, 3, &cfg);
    assert!(out.iter().flat_map(|p| p.iter()).all(|c| c.is_finite()));
    let rel = (flux(&out) - 2.0 * flux(&img)).abs() / (2.0 * flux(&img));
    assert!(
        rel < 1e-5,
        "flux not conserved through the level cap: {rel:.2e}"
    );
}

#[test]
fn bloom_is_deterministic() {
    let img = lcg_image(31, 19, 0xB100_0006);
    let cfg = BloomConfig {
        strength: 1.3,
        levels: 3,
        radius: 1.7,
    };
    assert_eq!(bloom(&img, 31, 19, &cfg), bloom(&img, 31, 19, &cfg));
}
