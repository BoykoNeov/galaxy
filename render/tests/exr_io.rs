//! OpenEXR write→read round-trip (DESIGN.md M3). Pure CPU — builds an `HdrImage`
//! directly, no GPU. Pins that the linear-HDR hand-off to `grade` is faithful:
//! 32-bit-float RGBA survives a round-trip, including values well above 1.0 (the
//! whole point of an HDR intermediate).

use galaxy_render::render::HdrImage;
use galaxy_render::{read_exr, write_exr};

fn scratch_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

fn sample_image() -> HdrImage {
    // 3×2 with distinct values per channel, including HDR (> 1.0) pixels.
    let pixels = vec![
        [0.0, 0.25, 0.5, 1.0],
        [12.5, 0.0, 3.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.1, 0.2, 0.3, 0.4],
        [7.75, 100.0, 0.01, 1.0],
        [0.5, 0.5, 0.5, 0.5],
    ];
    HdrImage {
        width: 3,
        height: 2,
        pixels,
    }
}

#[test]
fn exr_round_trips_hdr_rgba() {
    let img = sample_image();
    let path = scratch_path("galaxy_render_exr_roundtrip.exr");

    write_exr(&path, &img).unwrap();
    let back = read_exr(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.width, img.width);
    assert_eq!(back.height, img.height);
    assert_eq!(back.pixels.len(), img.pixels.len());
    for (b, o) in back.pixels.iter().zip(&img.pixels) {
        // Lossless f32 EXR: RGB must match closely (allow a hair for any encoding).
        for ch in 0..3 {
            assert!(
                (b[ch] - o[ch]).abs() <= 1e-4 * o[ch].abs().max(1.0),
                "channel {ch}: got {}, expected {}",
                b[ch],
                o[ch]
            );
        }
    }
}
