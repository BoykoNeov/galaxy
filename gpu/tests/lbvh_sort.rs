//! GPU Morton sort (DESIGN M4d) validated against the CPU reference
//! [`galaxy_solvers::reference_sort`].
//!
//! This stage is **pure integer** (`u32` codes in, a `u32` permutation out), so unlike the
//! f32 Morton stage there is no precision caveat: the GPU permutation must equal the f64 CPU
//! reference **bit-for-bit**, not merely within a tolerance. The gate feeds CPU-computed
//! codes to the GPU sort and demands `order == reference_sort(codes)`, which is a *unique*
//! total order (the reference keys on the pair `(code, index)`, and `index` is unique), so a
//! correct stable radix reproduces the ascending-index tie-break exactly. Two pass-localizing
//! cases (codes differing only in the low byte / only in the high byte) let a failure point
//! at a specific radix pass rather than "the sort is wrong somewhere".
//!
//! GPU-gated: these need a wgpu adapter. Without one, `GpuSorter::new` returns `NoAdapter`
//! and the tests fail loudly (matches the M3/M4 GPU-invariants convention).

use galaxy_core::DVec3;
use galaxy_gpu::GpuSorter;
use galaxy_solvers::{reference_morton, reference_sort};

fn sorter() -> GpuSorter {
    GpuSorter::new().expect("wgpu adapter required for GPU sort-stage tests")
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

/// Deterministic pseudo-random `u32` codes in `[0, 2^30)` (the 30-bit Morton range), for
/// synthetic-code tests that don't go through positions.
fn random_codes(seed: u64, n: usize) -> Vec<u32> {
    let mut state = seed | 1;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32) & ((1u32 << 30) - 1)
        })
        .collect()
}

/// The core gate: the GPU sort of `codes` must equal the CPU reference bit-for-bit, be a
/// genuine permutation of `0..n`, and yield a non-decreasing sorted-key array consistent
/// with the permutation.
fn assert_matches_reference(codes: &[u32]) {
    let out = sorter().sort(codes);
    let expected = reference_sort(codes);

    assert_eq!(
        out.order, expected,
        "GPU order must bit-match reference_sort (pure integer — no tolerance)"
    );

    // order is a permutation of 0..n.
    let mut seen = out.order.clone();
    seen.sort_unstable();
    let identity: Vec<u32> = (0..codes.len() as u32).collect();
    assert_eq!(seen, identity, "order must be a permutation of 0..n");

    // sorted_codes is the gathered, non-decreasing key array.
    assert_eq!(out.sorted_codes.len(), codes.len());
    for (k, &i) in out.order.iter().enumerate() {
        assert_eq!(
            out.sorted_codes[k], codes[i as usize],
            "sorted_codes[k] must equal codes[order[k]]"
        );
        if k > 0 {
            assert!(
                out.sorted_codes[k - 1] <= out.sorted_codes[k],
                "sorted_codes must be non-decreasing"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// core: bit-exact agreement with the f64 CPU reference
// ---------------------------------------------------------------------------

/// Real Morton codes from position clouds sort bit-identically to the CPU reference.
#[test]
fn gpu_sort_matches_reference_on_morton_codes() {
    for seed in [1u64, 7, 42, 1000] {
        let pos = cloud(seed, 8000, 3.0, DVec3::ZERO);
        let codes = reference_morton(&pos).codes;
        assert_matches_reference(&codes);
    }
}

/// Uniformly random 30-bit codes (denser tail than clustered positions) also match.
#[test]
fn gpu_sort_matches_reference_on_random_codes() {
    for seed in [2u64, 99, 12345] {
        let codes = random_codes(seed, 10_000);
        assert_matches_reference(&codes);
    }
}

// ---------------------------------------------------------------------------
// tie-break stability: equal codes must order by ascending original index
// ---------------------------------------------------------------------------

/// Many duplicate codes (only 4 distinct values ⇒ heavy ties) must break ties by ascending
/// index, exactly as the reference does. Directly exercises stable scatter.
#[test]
fn gpu_sort_breaks_ties_by_ascending_index() {
    let codes: Vec<u32> = (0..5000u32).map(|i| i % 4).collect();
    assert_matches_reference(&codes);

    // Spot-check the invariant directly: equal codes appear in ascending index order.
    let out = sorter().sort(&codes);
    for k in 1..out.order.len() {
        let (a, b) = (out.order[k - 1], out.order[k]);
        if codes[a as usize] == codes[b as usize] {
            assert!(a < b, "ties must order by ascending index: {a} before {b}");
        }
    }
}

// ---------------------------------------------------------------------------
// pass localization: isolate a single 8-bit radix pass
// ---------------------------------------------------------------------------

/// Codes that differ ONLY in the low 8 bits — the whole ordering is decided by radix pass 1
/// (`shift = 0`). A failure here localizes to that pass.
#[test]
fn gpu_sort_localizes_to_low_byte_pass() {
    const HIGH: u32 = 0x00ab_cd00; // fixed above bit 8
    let codes: Vec<u32> = (0..4096u32).map(|i| HIGH | ((i.wrapping_mul(37)) & 0xff)).collect();
    assert_matches_reference(&codes);
}

/// Codes that differ ONLY in bits 24–29 (the top of a 30-bit code) — the ordering is decided
/// by radix pass 4 (`shift = 24`), with heavy ties in the lower passes. Localizes pass 4 and
/// the interaction of a late pass with the tie-break.
#[test]
fn gpu_sort_localizes_to_high_byte_pass() {
    // 6 significant bits (0..63) placed at bit 24; low 24 bits identical across all codes.
    let codes: Vec<u32> = (0..4096u32).map(|i| ((i % 64) << 24)).collect();
    assert_matches_reference(&codes);
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run)
// ---------------------------------------------------------------------------

/// Same codes ⇒ bit-identical order on a given device (integer histogram commutes, fixed-
/// order scatter — determinism is structural, not statistical).
#[test]
fn gpu_sort_is_bit_deterministic() {
    let pos = cloud(0x50, 6000, 3.0, DVec3::ZERO);
    let codes = reference_morton(&pos).codes;
    let mut s = sorter();
    let a = s.sort(&codes);
    let b = s.sort(&codes);
    assert_eq!(a.order, b.order, "order must be run-to-run deterministic");
    assert_eq!(a.sorted_codes, b.sorted_codes);
}

// ---------------------------------------------------------------------------
// structural / adversarial input orderings
// ---------------------------------------------------------------------------

/// Already-ascending distinct codes ⇒ identity permutation.
#[test]
fn gpu_sort_already_sorted_is_identity() {
    let codes: Vec<u32> = (0..3000u32).collect();
    let out = sorter().sort(&codes);
    let identity: Vec<u32> = (0..3000u32).collect();
    assert_eq!(out.order, identity, "sorted input ⇒ identity order");
    assert_eq!(out.sorted_codes, codes);
}

/// Strictly descending codes ⇒ fully reversed permutation.
#[test]
fn gpu_sort_reverse_sorted() {
    let codes: Vec<u32> = (0..3000u32).rev().collect();
    assert_matches_reference(&codes);
}

/// All-equal codes ⇒ identity permutation (stability preserves input order under total tie).
#[test]
fn gpu_sort_all_equal_is_identity() {
    let codes = vec![42u32; 2048];
    let out = sorter().sort(&codes);
    let identity: Vec<u32> = (0..2048u32).collect();
    assert_eq!(out.order, identity, "all-equal codes ⇒ identity order");
}

/// Large N exercises the serial scatter under load and all four radix passes (full 30-bit
/// range). Kept at 2^16 to stay well under the GPU driver watchdog.
#[test]
fn gpu_sort_large_n() {
    let codes = random_codes(0xBEEF, 65_536);
    assert_matches_reference(&codes);
}

// ---------------------------------------------------------------------------
// edge cases
// ---------------------------------------------------------------------------

/// A single code yields the trivial order and its own sorted key.
#[test]
fn gpu_sort_single() {
    let out = sorter().sort(&[7]);
    assert_eq!(out.order, vec![0]);
    assert_eq!(out.sorted_codes, vec![7]);
}

/// Empty input yields empty output and does not panic.
#[test]
fn gpu_sort_empty() {
    let out = sorter().sort(&[]);
    assert!(out.order.is_empty());
    assert!(out.sorted_codes.is_empty());
}
