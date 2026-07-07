//! DDA/hierarchical shadow-bake equivalence gates (plan: the named deferral of
//! umbral-lantern-lattice). [`ShadowBake::Dda`] is a pure acceleration of the
//! brute [`bake_shadows`]: it skips only samples it can prove add exactly
//! `κ·0·ds = 0` to τ, so it must produce a **bit-identical** [`ShadowVolumes`].
//!
//! The whole correctness story is one assertion — `bake_shadows(gas) ==
//! bake_shadows_dda(gas)` — because every value is `exp(−τ)` with τ finite ≥ 0
//! (no NaN, no signed zero), so `f32 ==` on the flat buffers is exact. These
//! gates drive that equivalence through the cases that break a naive skip:
//! the ±1-cell trilinear-stencil dilation (single occupied cell), the
//! two-grid mix (a sample is empty iff BOTH grids are), the fully-dense slab
//! (nothing to skip), and the all-empty frame (everything skipped).

use galaxy_render::volume::{
    bake_shadows, bake_shadows_with, GasFrame, GasLook, Light, ScatterLook, ShadowBake, SHADOW_RES,
};
use galaxy_renderprep::GasGrid;
use glam::Vec3;
use proptest::prelude::*;

// ---------- fixtures ----------

/// A zero-filled grid with the listed cells set to `rho` — the sparse gas the
/// DDA is built to skip around.
fn blob_grid(
    dims: [u32; 3],
    bounds_min: Vec3,
    bounds_max: Vec3,
    cells: &[(u32, u32, u32)],
    rho: f32,
) -> GasGrid {
    let mut g = GasGrid {
        dims,
        bounds_min,
        bounds_max,
        data: vec![0.0f32; (dims[0] * dims[1] * dims[2]) as usize],
    };
    for &(ix, iy, iz) in cells {
        let idx = g.index(ix, iy, iz);
        g.data[idx] = rho;
    }
    g
}

/// A fully-occupied uniform grid — the DDA has nothing to skip, yet must still
/// reproduce the brute bake bit-for-bit.
fn uniform_grid(rho: f32, bounds_min: Vec3, bounds_max: Vec3, dims: [u32; 3]) -> GasGrid {
    GasGrid {
        dims,
        bounds_min,
        bounds_max,
        data: vec![rho; (dims[0] * dims[1] * dims[2]) as usize],
    }
}

fn look(opacity: f32) -> GasLook {
    GasLook {
        color: [1.0, 1.0, 1.0],
        emissivity: 0.0,
        opacity,
        // The bake ignores the scatter block (it reads only lights + opacity +
        // density); a live scatter term keeps the fixtures honest anyway.
        scatter: Some(ScatterLook {
            strength: 1.0,
            anisotropy: 0.0,
            shadows: true,
            tint: [1.0; 3],
            softening: None,
        }),
    }
}

fn light(pos: Vec3) -> Light {
    Light {
        pos,
        radius: 0.0,
        rgb: [1.0, 1.0, 1.0],
    }
}

/// Fraction of voxels baked to exactly 1.0 (fully unshadowed) — used to assert a
/// fixture actually exercises BOTH the skipped (empty) and marched (occupied)
/// paths, so an equivalence pass is not vacuous.
fn frac_ones(sv: &galaxy_render::volume::ShadowVolumes) -> f64 {
    let ones = sv.data.iter().filter(|&&t| t == 1.0).count();
    ones as f64 / sv.data.len() as f64
}

// ---------- gates ----------

/// The default entry point is the brute reference; `_with(Brute)` is that same
/// path, and `_with(Dda)` must equal it.
#[test]
fn dda_default_and_brute_agree() {
    let g = uniform_grid(0.4, Vec3::splat(-1.0), Vec3::splat(1.0), [8, 8, 8]);
    let lights = [light(Vec3::new(0.0, 0.0, 3.0))];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(0.7),
    };
    let brute = bake_shadows(&gas);
    assert_eq!(
        brute,
        bake_shadows_with(&gas, ShadowBake::Brute),
        "bake_shadows must equal _with(Brute)"
    );
    assert_eq!(
        brute,
        bake_shadows_with(&gas, ShadowBake::Dda),
        "Dda must equal the brute reference bit-for-bit"
    );
}

/// Fully-dense slab: no empty cells, so the DDA skips nothing — still bit-exact.
#[test]
fn dda_matches_brute_on_a_dense_slab() {
    let g = uniform_grid(
        0.5,
        Vec3::new(-2.0, -2.0, -1.0),
        Vec3::new(2.0, 2.0, 0.0),
        [16, 16, 32],
    );
    let lights = [
        light(Vec3::new(0.0, 0.0, 10.0)),
        light(Vec3::new(1.3, -0.7, 4.0)),
    ];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(0.6),
    };
    assert_eq!(
        bake_shadows_with(&gas, ShadowBake::Dda),
        bake_shadows(&gas),
        "dense slab: DDA must match the brute bake"
    );
}

/// All-empty frame: every sample is skipped and every voxel bakes to exactly 1
/// under BOTH strategies (the degenerate skip-everything case).
#[test]
fn dda_matches_brute_on_an_empty_frame() {
    let g = blob_grid([12, 12, 12], Vec3::splat(-1.0), Vec3::splat(1.0), &[], 0.0);
    let lights = [
        light(Vec3::new(0.5, 0.5, 2.0)),
        light(Vec3::new(-2.0, 0.0, 0.0)),
    ];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(1.0),
    };
    let dda = bake_shadows_with(&gas, ShadowBake::Dda);
    assert_eq!(dda, bake_shadows(&gas), "empty frame: DDA must match brute");
    assert!(
        dda.data.iter().all(|&t| t == 1.0),
        "empty frame must bake to all ones"
    );
}

/// A single non-empty cell in an otherwise-empty grid — the pointed test for
/// the ±1-cell trilinear-stencil dilation the advisor flagged. The cell's
/// trilinear tail extends ONE cell in every direction, so chords through the
/// (zero-DATA) neighbour cells are genuinely shadowed. A DDA occupancy that is
/// NOT dilated marks those neighbours empty and skips their nonzero samples,
/// dropping τ below the brute value — the `assert_eq` catches it. The
/// mixed-fraction check proves the scene contains BOTH shadowed voxels (tail)
/// and fully-lit voxels (far from the cell), so the pass is not vacuous.
#[test]
fn dda_matches_brute_single_occupied_cell() {
    let dims = [24, 24, 24];
    let g = blob_grid(
        dims,
        Vec3::splat(-1.0),
        Vec3::splat(1.0),
        &[(12, 12, 12)],
        5.0,
    );
    // Lights on several sides so the single cell casts chords in every octant,
    // guaranteeing grazing chords through its zero-data neighbours.
    let lights = [
        light(Vec3::new(0.0, 0.0, 5.0)),
        light(Vec3::new(3.0, 2.0, -1.0)),
        light(Vec3::new(-2.5, -0.3, 0.4)),
    ];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(2.0),
    };
    let dda = bake_shadows_with(&gas, ShadowBake::Dda);
    let brute = bake_shadows(&gas);
    assert_eq!(
        dda, brute,
        "single occupied cell: DDA must match brute exactly"
    );
    let ones = frac_ones(&brute);
    assert!(
        ones > 0.5 && ones < 1.0,
        "fixture must produce both shadowed and lit voxels (ones fraction {ones})"
    );
}

/// A denser blob (a 3×3×3 lump plus a stray cell) with many lights inside and
/// outside the domain — the realistic sparse case the acceleration targets.
#[test]
fn dda_matches_brute_on_a_sparse_blob() {
    let dims = [20, 20, 20];
    let mut cells = Vec::new();
    for ix in 8..11 {
        for iy in 9..12 {
            for iz in 7..10 {
                cells.push((ix, iy, iz));
            }
        }
    }
    cells.push((2, 17, 3)); // a stray occupied cell far from the lump
    let g = blob_grid(dims, Vec3::splat(-1.5), Vec3::splat(1.5), &cells, 0.8);
    let lights = [
        light(Vec3::new(0.0, 0.0, 4.0)),
        light(Vec3::new(-3.0, 1.0, -0.5)), // outside the domain
        light(Vec3::new(0.2, -0.1, 0.3)),  // inside the domain
        light(Vec3::new(2.0, 2.0, 2.0)),
    ];
    let gas = GasFrame {
        grid0: &g,
        grid1: &g,
        mix: 0.0,
        lights: &lights,
        look: look(1.5),
    };
    let brute = bake_shadows(&gas);
    assert_eq!(
        bake_shadows_with(&gas, ShadowBake::Dda),
        brute,
        "sparse blob: DDA must match brute exactly"
    );
    let ones = frac_ones(&brute);
    assert!(
        ones > 0.3 && ones < 1.0,
        "sparse fixture degenerate (ones {ones})"
    );
}

/// Two DIFFERENT grids (distinct dims AND bounds) blended at a non-trivial mix:
/// a sample is empty iff BOTH grids are zero there, so the occupancy must union
/// the two grids' dilated supports over the union AABB. Exercises the fiddly
/// two-grid reconciliation.
#[test]
fn dda_matches_brute_on_a_two_grid_mix() {
    let g0 = blob_grid(
        [16, 12, 10],
        Vec3::new(-1.2, -1.0, -0.8),
        Vec3::new(1.0, 1.1, 0.9),
        &[(3, 4, 5), (4, 4, 5), (10, 8, 2)],
        0.9,
    );
    let g1 = blob_grid(
        [10, 18, 14],
        Vec3::new(-0.9, -1.1, -1.0),
        Vec3::new(1.2, 0.9, 1.1),
        &[(6, 9, 7), (6, 10, 7), (1, 1, 12)],
        0.6,
    );
    let lights = [
        light(Vec3::new(0.4, 0.3, 3.0)),
        light(Vec3::new(-2.0, -1.5, 0.2)),
        light(Vec3::new(0.1, 2.5, -0.4)),
    ];
    let gas = GasFrame {
        grid0: &g0,
        grid1: &g1,
        mix: 0.37,
        lights: &lights,
        look: look(1.8),
    };
    let brute = bake_shadows(&gas);
    assert_eq!(
        bake_shadows_with(&gas, ShadowBake::Dda),
        brute,
        "two-grid mix: DDA must match brute exactly"
    );
    let ones = frac_ones(&brute);
    assert!(
        ones > 0.3 && ones < 1.0,
        "two-grid fixture degenerate (ones {ones})"
    );
    // Endpoint mixes must also hold: at u = 0 grid1's support is irrelevant, at
    // u = 1 grid0's is — the occupancy's mix-aware skip must not drop the live
    // grid.
    for mix in [0.0f32, 1.0] {
        let gas = GasFrame { mix, ..gas };
        assert_eq!(
            bake_shadows_with(&gas, ShadowBake::Dda),
            bake_shadows(&gas),
            "two-grid mix = {mix}: DDA must match brute"
        );
    }
    let _ = SHADOW_RES;
}

// ---------- randomized invariant ----------

/// A small grid with a heavily-zero-biased density field and randomized dims /
/// bounds — the sparse geometry the DDA skip must survive across the board.
fn grid_strategy() -> impl Strategy<Value = GasGrid> {
    (3usize..=7, 3usize..=7, 3usize..=7).prop_flat_map(|(nx, ny, nz)| {
        let n = nx * ny * nz;
        (
            Just([nx as u32, ny as u32, nz as u32]),
            (-2.0f32..0.0, -2.0f32..0.0, -2.0f32..0.0),
            (1.0f32..3.0, 1.0f32..3.0, 1.0f32..3.0),
            // ~3:1 empty:occupied, so most cells are skippable but some are not.
            prop::collection::vec(prop_oneof![3 => Just(0.0f32), 1 => 0.1f32..1.0], n),
        )
            .prop_map(|(dims, bmin, ext, data)| {
                let bounds_min = Vec3::new(bmin.0, bmin.1, bmin.2);
                GasGrid {
                    dims,
                    bounds_min,
                    bounds_max: bounds_min + Vec3::new(ext.0, ext.1, ext.2),
                    data,
                }
            })
    })
}

proptest! {
    // The bake is O(SHADOW_RES³·lights) per case, so keep the case count modest;
    // small grids give short chords, keeping each case sub-second.
    #![proptest_config(ProptestConfig { cases: 24, ..ProptestConfig::default() })]

    /// The invariant that IS the feature: for arbitrary sparse two-grid frames,
    /// mixes, opacities, and light placements (inside and outside the domain),
    /// the DDA bake equals the brute reference to the last bit. This is the
    /// oracle the GPU mirror will inherit — fuzz it before trusting it.
    #[test]
    fn dda_equals_brute_randomized(
        g0 in grid_strategy(),
        g1 in grid_strategy(),
        mix in 0.0f32..=1.0,
        opacity in 0.1f32..3.0,
        lps in prop::collection::vec((-3.0f32..3.0, -3.0f32..3.0, -3.0f32..3.0), 1..=3),
    ) {
        let lights: Vec<Light> = lps.iter().map(|&(x, y, z)| light(Vec3::new(x, y, z))).collect();
        let gas = GasFrame {
            grid0: &g0,
            grid1: &g1,
            mix,
            lights: &lights,
            look: look(opacity),
        };
        prop_assert_eq!(
            bake_shadows_with(&gas, ShadowBake::Dda),
            bake_shadows(&gas)
        );
    }
}
