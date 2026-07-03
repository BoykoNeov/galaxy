//! End-to-end grade of a linear-HDR EXR to a 16-bit sRGB PNG (DESIGN.md M3).
//! Writes a tiny known EXR, grades it, reads the PNG back, and checks each pixel
//! equals the per-pixel `tonemap()` — pinning the file pipeline, the sRGB/quantize,
//! and the 16-bit PNG byte order end to end.

use std::fs::File;
use std::io::BufReader;

use exr::prelude::write_rgba_file;
use galaxy_grade::{bloom, grade_file, tonemap, BloomConfig, GradeConfig, ToneMap};

fn scratch(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

/// Read a 16-bit RGB PNG back into `(width, height, Vec<[u16; 3]>)`.
fn read_png16(path: &std::path::Path) -> (u32, u32, Vec<[u16; 3]>) {
    let decoder = png::Decoder::new(BufReader::new(File::open(path).unwrap()));
    let mut reader = decoder.read_info().unwrap();
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgb);
    assert_eq!(info.bit_depth, png::BitDepth::Sixteen);
    // 16-bit samples are big-endian in the output buffer.
    let mut px = Vec::new();
    for chunk in buf[..info.buffer_size()].chunks_exact(6) {
        px.push([
            u16::from_be_bytes([chunk[0], chunk[1]]),
            u16::from_be_bytes([chunk[2], chunk[3]]),
            u16::from_be_bytes([chunk[4], chunk[5]]),
        ]);
    }
    (info.width, info.height, px)
}

#[test]
fn grades_exr_to_16bit_srgb_png() {
    // 2×1 linear-HDR image: a midtone grey and an over-range (HDR) red.
    let linear = [[0.5f32, 0.5, 0.5, 1.0], [4.0, 0.1, 0.0, 1.0]];
    let exr_path = scratch("galaxy_grade_in.exr");
    let png_path = scratch("galaxy_grade_out.png");

    write_rgba_file(&exr_path, 2, 1, |x, _y| {
        let p = linear[x];
        (p[0], p[1], p[2], p[3])
    })
    .unwrap();

    let cfg = GradeConfig::default();
    grade_file(&exr_path, &png_path, &cfg).unwrap();

    let (w, h, px) = read_png16(&png_path);
    let _ = std::fs::remove_file(&exr_path);
    let _ = std::fs::remove_file(&png_path);

    assert_eq!((w, h), (2, 1));
    for (i, lin) in linear.iter().enumerate() {
        let expected = tonemap([lin[0], lin[1], lin[2]], &cfg);
        for c in 0..3 {
            let d = (px[i][c] as i32 - expected[c] as i32).abs();
            assert!(
                d <= 1,
                "pixel {i} ch {c}: png {} vs tonemap {}",
                px[i][c],
                expected[c]
            );
        }
    }
}

#[test]
fn grade_file_applies_bloom_before_the_tone_curve() {
    // Bloom is an image-space linear-domain op, so it cannot live inside the
    // per-pixel `tonemap()` — grade_file must run it over the whole EXR image
    // FIRST, then tonemap each bloomed pixel. Oracle: `bloom()` (independently
    // gated in bloom.rs) composed with `tonemap()` (gated in tonemap.rs) — this
    // test pins the wiring: bloom actually applied, in linear space before the
    // curve, with the row-major pixel order shared between EXR and bloom.
    const W: usize = 9;
    const H: usize = 5;
    // Dim grey background with one hot pixel at the center — bloom must leak
    // flux into the neighbours, so a grade that ignores `cfg.bloom` mismatches.
    let mut linear = vec![[0.05f32, 0.05, 0.05]; W * H];
    linear[2 * W + 4] = [4.0, 2.0, 1.0];

    let exr_path = scratch("galaxy_grade_bloom_in.exr");
    let png_path = scratch("galaxy_grade_bloom_out.png");
    write_rgba_file(&exr_path, W, H, |x, y| {
        let p = linear[y * W + x];
        (p[0], p[1], p[2], 1.0f32)
    })
    .unwrap();

    let bloom_cfg = BloomConfig {
        strength: 1.5,
        levels: 2,
        radius: 1.0,
    };
    // Non-default exposure/curve so the composition order is exercised too.
    let cfg = GradeConfig {
        exposure: 2.0,
        tonemap: ToneMap::Reinhard,
        bloom: Some(bloom_cfg),
    };
    grade_file(&exr_path, &png_path, &cfg).unwrap();

    let (w, h, px) = read_png16(&png_path);
    let _ = std::fs::remove_file(&exr_path);
    let _ = std::fs::remove_file(&png_path);

    assert_eq!((w, h), (W as u32, H as u32));
    let bloomed = bloom(&linear, W, H, &bloom_cfg);
    for (i, lin) in bloomed.iter().enumerate() {
        let expected = tonemap(*lin, &cfg);
        for c in 0..3 {
            let d = (px[i][c] as i32 - expected[c] as i32).abs();
            assert!(
                d <= 1,
                "pixel {i} ch {c}: png {} vs bloom∘tonemap {}",
                px[i][c],
                expected[c]
            );
        }
    }
}
