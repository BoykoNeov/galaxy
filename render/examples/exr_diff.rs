//! Diagnostic: pixel-level diff of two linear EXRs (max |Δ|, max relative Δ,
//! count of differing pixels). Usage:
//! `cargo run -p galaxy-render --example exr_diff -- a.exr b.exr`

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [a, b] = &args[..] else {
        eprintln!("usage: exr_diff <a.exr> <b.exr>");
        std::process::exit(2);
    };
    let a = galaxy_render::read_exr(a).expect("read a");
    let b = galaxy_render::read_exr(b).expect("read b");
    assert_eq!((a.width, a.height), (b.width, b.height), "size mismatch");

    let (mut n_diff, mut max_abs, mut max_rel) = (0u64, 0.0f32, 0.0f32);
    let mut worst = (0u32, 0u32, 0usize, 0.0f32, 0.0f32);
    for y in 0..a.height {
        for x in 0..a.width {
            let (pa, pb) = (a.pixel(x, y), b.pixel(x, y));
            if pa != pb {
                n_diff += 1;
                for c in 0..4 {
                    let d = (pa[c] - pb[c]).abs();
                    let r = d / pa[c].abs().max(1e-30);
                    if d > max_abs {
                        max_abs = d;
                        worst = (x, y, c, pa[c], pb[c]);
                    }
                    max_rel = max_rel.max(r);
                }
            }
        }
    }
    println!(
        "{n_diff} differing pixels of {}; max abs {max_abs:e}, max rel {max_rel:e}",
        (a.width * a.height)
    );
    let (x, y, c, va, vb) = worst;
    println!("worst: ({x},{y}) ch{c}: {va} vs {vb}");
}
