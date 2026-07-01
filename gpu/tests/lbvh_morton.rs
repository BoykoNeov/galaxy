//! GPU Morton + bounding-box build stage (DESIGN M4c) validated against the CPU
//! reference [`galaxy_solvers::reference_morton`].
//!
//! The kernel runs in **f32** (wgpu/naga has no portable f64 compute — same constraint
//! as `GpuDirectSum`/`GpuTree`), so the codes are gated on **tolerance + determinism**,
//! not bit-equality vs the f64 reference. Because a 1-bit lane change jumps the code by
//! a large power of two, the tolerance is expressed in **lane** space (±1 per axis) in
//! the well-conditioned near-origin regime; a large-coordinate case is *characterized*,
//! not pinned. The reduction is gated separately (bit-exact over f32 positions) so a
//! failure localizes to either the reduction or the quantization.
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuMortonBuilder::new` returns
//! `NoAdapter` and the tests fail loudly (matches the M3/M4 GPU-invariants convention).

use galaxy_core::DVec3;
use galaxy_gpu::GpuMortonBuilder;
use galaxy_solvers::reference_morton;

fn builder() -> GpuMortonBuilder {
    GpuMortonBuilder::new().expect("wgpu adapter required for GPU Morton-stage tests")
}

/// Deterministic pseudo-random positions in a cube of half-width `r` centered at
/// `center` (same LCG as the other solver/GPU tests).
fn cloud(seed: u64, n: usize, r: f64, center: DVec3) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64) // in [0, 1)
    };
    (0..n)
        .map(|_| center + DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * (2.0 * r))
        .collect()
}

/// Local Morton interleave, mirroring the (unit-tested) private `morton3` in
/// `galaxy_solvers::lbvh` — used to check the GPU code is the interleave of its lanes.
fn expand10(v: u32) -> u32 {
    let mut x = v & 0x3ff;
    x = (x | (x << 16)) & 0x030000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}
fn morton3(l: [u32; 3]) -> u32 {
    expand10(l[0]) | (expand10(l[1]) << 1) | (expand10(l[2]) << 2)
}

// ---------------------------------------------------------------------------
// bbox reduction — isolated from f32-vs-f64 by reducing over the SAME f32 positions
// ---------------------------------------------------------------------------

/// The GPU bbox is a min/max reduction; min/max never round and are order-independent,
/// so over the same f32-narrowed positions the GPU result must equal a CPU reduction
/// **bit-for-bit**. Isolates the reduction logic from the quantization precision.
#[test]
fn gpu_bbox_reduction_matches_cpu_over_f32_positions() {
    let pos = cloud(0xB00, 4096, 3.0, DVec3::ZERO);
    let out = builder().compute(&pos);

    // CPU reduction over the identical f32-narrowed positions.
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    for &p in &pos {
        let f = [p.x as f32, p.y as f32, p.z as f32];
        for a in 0..3 {
            lo[a] = lo[a].min(f[a]);
            hi[a] = hi[a].max(f[a]);
        }
    }
    assert_eq!(
        out.bbox_min, lo,
        "GPU bbox min must bit-match the f32 CPU reduction"
    );
    assert_eq!(
        out.bbox_max, hi,
        "GPU bbox max must bit-match the f32 CPU reduction"
    );
}

/// The `.max(1e-12)` degenerate-axis floor path: collinear points (zero extent in two
/// axes) must still reduce to the exact per-axis min/max.
#[test]
fn gpu_bbox_reduction_handles_collinear_points() {
    let pos: Vec<DVec3> = (0..500)
        .map(|i| DVec3::new(i as f64 * 0.1, 2.0, -1.0))
        .collect();
    let out = builder().compute(&pos);
    assert_eq!(out.bbox_min, [0.0_f32, 2.0, -1.0]);
    assert_eq!(out.bbox_max, [49.9_f32, 2.0, -1.0]);
}

// ---------------------------------------------------------------------------
// per-lane agreement with the f64 reference (the core quantization gate)
// ---------------------------------------------------------------------------

/// In the well-conditioned near-origin regime the GPU (f32) quantized lanes agree with
/// the f64 reference to **±1 per axis**, with the vast majority exact. Disagreements are
/// boundary straddles (a particle whose pre-floor value sits within an f32 ulp of an
/// integer cell edge) — bounded, not a logic error.
#[test]
fn gpu_lanes_agree_with_reference_within_one() {
    for seed in [1u64, 7, 42, 1000] {
        let pos = cloud(seed, 8000, 3.0, DVec3::ZERO);
        let out = builder().compute(&pos);
        let refr = reference_morton(&pos);
        assert_eq!(out.lanes.len(), pos.len());

        let mut exact = 0usize;
        for (g, r) in out.lanes.iter().zip(&refr.lanes) {
            let mut all_eq = true;
            for a in 0..3 {
                let d = (g[a] as i64 - r[a] as i64).abs();
                assert!(
                    d <= 1,
                    "seed {seed}: lane {a} off by {d} (gpu {} vs ref {})",
                    g[a],
                    r[a]
                );
                all_eq &= g[a] == r[a];
            }
            if all_eq {
                exact += 1;
            }
        }
        // Boundary straddles are rare — the overwhelming majority must be exact.
        let frac = exact as f64 / pos.len() as f64;
        assert!(
            frac > 0.95,
            "seed {seed}: only {frac:.3} of lanes exact (expected >0.95)"
        );
    }
}

/// Large-coordinate regime: characterization, **not** a ±1 assertion. Far from the
/// origin the f32 `p − bmin` cancellation loses conditioning (the analogue of the
/// direct-sum "|x|≈5000 degrades to ~5e-3" honesty), so lanes may diverge from the f64
/// reference by more than one cell. Assert only the structural bound (lanes in range)
/// and record the observed max gap.
#[test]
fn gpu_lanes_large_coordinate_divergence_is_characterized() {
    let pos = cloud(3, 8000, 3.0, DVec3::splat(1.0e6));
    let out = builder().compute(&pos);
    let refr = reference_morton(&pos);

    let mut max_gap = 0i64;
    for (g, r) in out.lanes.iter().zip(&refr.lanes) {
        for a in 0..3 {
            assert!(g[a] < 1024, "lane out of range: {}", g[a]);
            max_gap = max_gap.max((g[a] as i64 - r[a] as i64).abs());
        }
    }
    // Characterization: at |x|≈1e6 with span≈6, an f32 ulp (~0.12) dwarfs a cell
    // (~6/1024 world units), so `p − bmin` cancellation coarsens the quantization and
    // lanes diverge from the f64 reference by many cells — the analogue of the
    // direct-sum "|x|≈5000 degrades to ~5e-3" honesty. Range still holds; not pinned to
    // ±1. (At |x|≈1e5 the gap is still only ~1 — the divergence is coordinate-driven.)
    eprintln!("large-coordinate max lane gap vs f64 reference: {max_gap}");
    assert!(
        max_gap > 1,
        "expected the large-coordinate regime to exceed the ±1 bound"
    );
}

// ---------------------------------------------------------------------------
// structural + determinism
// ---------------------------------------------------------------------------

/// Every code is a valid 30-bit interleave of its own in-range lanes.
#[test]
fn gpu_codes_are_interleave_of_in_range_lanes() {
    let pos = cloud(9, 5000, 3.0, DVec3::ZERO);
    let out = builder().compute(&pos);
    assert_eq!(out.codes.len(), out.lanes.len());
    for (c, l) in out.codes.iter().zip(&out.lanes) {
        for &lane in l {
            assert!(lane < 1024, "lane {lane} out of [0,1024)");
        }
        assert!(*c < (1u32 << 30), "code {c} exceeds 30 bits");
        assert_eq!(*c, morton3(*l), "code must be the interleave of its lanes");
    }
}

/// Same input ⇒ bit-identical lanes and codes on a given device (the hard determinism
/// claim; no float atomics, fixed reduction order).
#[test]
fn gpu_morton_is_bit_deterministic() {
    let pos = cloud(0xD00, 6000, 3.0, DVec3::ZERO);
    let mut b = builder();
    let a = b.compute(&pos);
    let c = b.compute(&pos);
    assert_eq!(a.codes, c.codes, "codes must be run-to-run deterministic");
    assert_eq!(a.lanes, c.lanes, "lanes must be run-to-run deterministic");
    assert_eq!(a.bbox_min, c.bbox_min);
    assert_eq!(a.bbox_max, c.bbox_max);
}

// ---------------------------------------------------------------------------
// edge cases
// ---------------------------------------------------------------------------

/// A single particle yields one in-range code and a degenerate bbox (min == max).
#[test]
fn gpu_single_particle() {
    let pos = vec![DVec3::new(2.0, -3.0, 4.0)];
    let out = builder().compute(&pos);
    assert_eq!(out.lanes.len(), 1);
    assert_eq!(out.codes.len(), 1);
    assert_eq!(out.bbox_min, [2.0_f32, -3.0, 4.0]);
    assert_eq!(out.bbox_max, [2.0_f32, -3.0, 4.0]);
    for a in 0..3 {
        assert!(out.lanes[0][a] < 1024);
    }
    assert_eq!(out.codes[0], morton3(out.lanes[0]));
}

/// Coincident particles land in the same cell ⇒ identical lanes and codes.
#[test]
fn gpu_coincident_particles_share_a_code() {
    let p = DVec3::new(1.0, 1.0, 1.0);
    let pos = vec![p; 64];
    let out = builder().compute(&pos);
    let first = out.codes[0];
    for c in &out.codes {
        assert_eq!(*c, first, "coincident particles must share a Morton code");
    }
    for l in &out.lanes {
        assert_eq!(*l, out.lanes[0], "coincident particles must share lanes");
    }
}

/// Empty input yields empty output and does not panic.
#[test]
fn gpu_empty_input() {
    let out = builder().compute(&[]);
    assert!(out.lanes.is_empty());
    assert!(out.codes.is_empty());
}
