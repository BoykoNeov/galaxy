//! Adaptive octree light-clustering gates (tinted-octree-lanterns O1): the
//! deterministic greedy octree cut that replaces `cluster_lights`' fixed 8³
//! binning. These are CPU-only — the clusterer has no GPU mirror; the GPU
//! consumes its flat `Vec<Light>` output unchanged.
//!
//! The v1 binning-specific oracles (same-bin merge geometry, single global
//! softening radius) were retired from `scatter.rs` with the replacement; the
//! algorithm-independent contracts carry over here:
//!
//! - **Conservation + structure** (gate 1, proptest): RGB power conserved,
//!   count ≤ budget, lights inside the root cube, radii in `[0, ½·root diag]`.
//! - **Determinism** (gate 2): identical `Vec<Light>` across runs.
//! - **Degenerates** (gate 3): all-dark → empty; single/coincident → one light
//!   radius 0; a dark star moves neither centroid nor bounds.
//! - **Adaptivity** (gate 4): a compact cluster gets strictly more lights than
//!   any uniform 8³ split of the joint AABB would give it, at a far smaller
//!   softening radius than the v1 global radius.
//! - **Near-field accuracy** (gate 5): the clustered isotropic incident flux at
//!   a probe one unit from a compact cluster matches the exact per-star sum.
//! - **Budget-cap path** (gate 6): exercised at a *reachable* budget via
//!   [`cluster_lights_with`] (the shipped 512 is unreachable at `REFINE_TOL` —
//!   see the gate), plus the `≤ MAX_LIGHTS` safety contract on a large cloud.
//!
//! Off-path bit-identity (gate 7 in the plan) needs no test here: the surviving
//! `scatter.rs` / `volume.rs` suites never call the clusterer — that containment
//! is itself the proof the replacement is contained.

use galaxy_render::volume::{cluster_lights, cluster_lights_with, Light, MAX_LIGHTS};
use galaxy_renderprep::FrameData;
use glam::Vec3;
use proptest::prelude::*;

// ---------- helpers ----------

fn frame(pos: Vec<Vec3>, color: Vec<[f32; 3]>, brightness: Vec<f32>) -> FrameData {
    let n = pos.len();
    FrameData {
        pos,
        color,
        size: vec![0.1; n],
        brightness,
    }
}

/// Small deterministic LCG in `[0, 1)` — the octree cut must be reproducible, so
/// the fixtures below must be too (no `rand` thread entropy).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (1u64 << 31) as f32
    }
}

/// Scalar luminance of star `i` (Σ color·brightness) — the clustering weight and
/// the bounds gate; the f64 reference the octree folds match.
fn star_lum(f: &FrameData, i: usize) -> f64 {
    let c = f.color[i];
    (c[0] + c[1] + c[2]) as f64 * f.brightness[i] as f64
}

/// Total RGB power over the luminous stars, f64 — the conserved quantity.
fn frame_power(f: &FrameData) -> [f64; 3] {
    let mut p = [0.0f64; 3];
    for i in 0..f.len() {
        let b = f.brightness[i] as f64;
        for (acc, &c) in p.iter_mut().zip(f.color[i].iter()) {
            *acc += c as f64 * b;
        }
    }
    p
}

/// Total RGB power carried by the emitted lights, f64.
fn lights_power(lights: &[Light]) -> [f64; 3] {
    let mut p = [0.0f64; 3];
    for l in lights {
        for (acc, &c) in p.iter_mut().zip(l.rgb.iter()) {
            *acc += c as f64;
        }
    }
    p
}

/// The root cube of the luminous set (center, half) — center = luminous-AABB
/// center, half = ½·max extent (the Barnes-Hut/LBVH convention the cut uses).
/// Returns `None` for an all-dark frame.
fn root_cube(f: &FrameData) -> Option<(Vec3, f32)> {
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for i in 0..f.len() {
        if star_lum(f, i) > 0.0 {
            bmin = bmin.min(f.pos[i]);
            bmax = bmax.max(f.pos[i]);
            any = true;
        }
    }
    if !any {
        return None;
    }
    let center = 0.5 * (bmin + bmax);
    let half = 0.5 * (bmax - bmin).max_element().max(0.0);
    Some((center, half))
}

// ---------- gate 1: conservation + structure (proptest) ----------

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Random star sets (incl. dark stars) preserve total RGB power within the
    /// f32-cast floor, stay under budget, and every light sits inside the root
    /// cube with a radius in `[0, ½·root diagonal]`.
    #[test]
    fn conservation_and_structure(
        raw in prop::collection::vec(
            (
                (-100.0f32..100.0, -100.0f32..100.0, -100.0f32..100.0),
                (0.0f32..1.0, 0.0f32..1.0, 0.0f32..1.0),
                0.0f32..10.0,
            ),
            0..2000usize,
        )
    ) {
        let pos: Vec<Vec3> = raw.iter().map(|(p, _, _)| Vec3::new(p.0, p.1, p.2)).collect();
        let color: Vec<[f32; 3]> = raw.iter().map(|(_, c, _)| [c.0, c.1, c.2]).collect();
        let brightness: Vec<f32> = raw.iter().map(|(_, _, b)| *b).collect();
        let f = frame(pos, color, brightness);

        let lights = cluster_lights(&f);
        prop_assert!(lights.len() <= MAX_LIGHTS);

        // Power conservation: f64 folds on both sides, one f32 cast per leaf
        // channel at emission. Tolerance scales with the total (relative) plus a
        // small absolute floor for the all-dark case.
        let want = frame_power(&f);
        let got = lights_power(&lights);
        for ch in 0..3 {
            let tol = 1e-4 * want[ch] + 1e-3;
            prop_assert!(
                (got[ch] - want[ch]).abs() <= tol,
                "power channel {ch}: got {}, want {}, tol {}", got[ch], want[ch], tol
            );
        }

        if let Some((center, half)) = root_cube(&f) {
            let max_r = half * 3.0f32.sqrt(); // ½·root cube diagonal
            let eps = 1e-3 * half.max(1.0);
            for l in &lights {
                prop_assert!(l.radius >= 0.0);
                prop_assert!(l.radius <= max_r + eps, "radius {} > {}", l.radius, max_r);
                let d = (l.pos - center).abs();
                prop_assert!(
                    d.x <= half + eps && d.y <= half + eps && d.z <= half + eps,
                    "light {:?} outside root cube (center {:?}, half {half})", l.pos, center
                );
            }
        } else {
            // All-dark frame: no lights at all (gate 3 pins this directly too).
            prop_assert!(lights.is_empty());
        }
    }
}

// ---------- gate 2: determinism ----------

/// The greedy cut is a pure function of the frame — same input, bit-identical
/// `Vec<Light>` (the whole vec, order included).
#[test]
fn determinism() {
    let mut rng = Lcg(0xD37E834);
    let mut pos = Vec::new();
    let mut color = Vec::new();
    let mut b = Vec::new();
    for _ in 0..1500 {
        pos.push(Vec3::new(rng.next(), rng.next(), rng.next()) * 80.0 - Vec3::splat(40.0));
        color.push([rng.next(), rng.next(), rng.next()]);
        b.push(rng.next() * 8.0);
    }
    let f = frame(pos, color, b);
    let a = cluster_lights(&f);
    let b2 = cluster_lights(&f);
    assert_eq!(a, b2, "clustering is not deterministic");
    assert!(a.len() > 1, "fixture should refine to several lights");
}

// ---------- gate 3: degenerates ----------

/// All-dark → empty; single → one light at its position radius 0; N coincident
/// → one light radius 0; one bright + one dark → one light, dark star moved
/// neither centroid nor bounds.
#[test]
fn degenerates() {
    // all-dark (zero brightness)
    let dark = frame(
        vec![Vec3::new(1.0, 2.0, 3.0), Vec3::new(-4.0, 5.0, 6.0)],
        vec![[1.0, 1.0, 1.0]; 2],
        vec![0.0; 2],
    );
    assert!(cluster_lights(&dark).is_empty(), "all-dark → no lights");

    // all-dark (zero color)
    let black = frame(
        vec![Vec3::new(1.0, 2.0, 3.0)],
        vec![[0.0, 0.0, 0.0]],
        vec![5.0],
    );
    assert!(cluster_lights(&black).is_empty(), "zero-color → no lights");

    // single luminous star → one light at its position, radius 0
    let p = Vec3::new(7.0, -3.0, 2.0);
    let single = frame(vec![p], vec![[0.5, 0.25, 1.0]], vec![4.0]);
    let ls = cluster_lights(&single);
    assert_eq!(ls.len(), 1);
    assert_eq!(ls[0].pos, p);
    assert_eq!(ls[0].radius, 0.0);
    assert_eq!(ls[0].rgb, [2.0, 1.0, 4.0]); // color·brightness

    // N coincident stars → one light, radius 0 (spread 0 ⇒ metric 0 ⇒ no split)
    let coincident = frame(vec![p; 5], vec![[1.0, 1.0, 1.0]; 5], vec![2.0; 5]);
    let lc = cluster_lights(&coincident);
    assert_eq!(lc.len(), 1);
    assert_eq!(lc[0].pos, p);
    assert_eq!(lc[0].radius, 0.0);
    assert_eq!(lc[0].rgb, [10.0, 10.0, 10.0]); // 5·(1·2) per channel

    // one bright + one dark far away → one light at the bright star, the dark
    // star (lum 0) neither shifts the centroid nor stretches the bounds.
    let bright = Vec3::new(0.0, 0.0, 0.0);
    let mixed = frame(
        vec![bright, Vec3::new(1000.0, 1000.0, 1000.0)],
        vec![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        vec![3.0, 0.0],
    );
    let lm = cluster_lights(&mixed);
    assert_eq!(
        lm.len(),
        1,
        "dark star must not create a light or stretch bounds"
    );
    assert_eq!(lm[0].pos, bright, "dark star must not shift the centroid");
    assert_eq!(
        lm[0].radius, 0.0,
        "single luminous star ⇒ zero-side root cube"
    );
}

// ---------- gate 4: adaptivity (the point of the change) ----------

/// A bright compact cluster (64 stars in a 0.1-side cube at (2,2,2)) plus a
/// bright straggler at (11,11,11) and a dim sparse ±50 grid. Under a uniform 8³
/// split of the joint [−50,50]³ AABB the bin size is 12.5, so the whole region
/// `[0,12.5)³` — cluster AND straggler — collapses to a SINGLE light. The
/// octree instead resolves them: strictly more than one light lands in that
/// region, and every light near the compact cluster carries a softening radius
/// far below the v1 global radius (½·√3·(100/8) = 10.8253, a literal here — the
/// removed binning is not called).
#[test]
fn adaptivity_beats_uniform_binning() {
    let mut pos = Vec::new();
    let mut color = Vec::new();
    let mut b = Vec::new();
    // 64-star bright cluster in a 0.1 cube at (2,2,2)
    let base = Vec3::new(2.0, 2.0, 2.0);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let off = Vec3::new(i as f32, j as f32, k as f32) * (0.1 / 3.0) - Vec3::splat(0.05);
                pos.push(base + off);
                color.push([1.0, 1.0, 1.0]);
                b.push(100.0);
            }
        }
    }
    // bright straggler, same 8³ bin as the cluster
    pos.push(Vec3::new(11.0, 11.0, 11.0));
    color.push([1.0, 1.0, 1.0]);
    b.push(100.0);
    // dim sparse 3×3×3 grid at {−50, 0, 50}
    for x in [-50.0f32, 0.0, 50.0] {
        for y in [-50.0f32, 0.0, 50.0] {
            for z in [-50.0f32, 0.0, 50.0] {
                pos.push(Vec3::new(x, y, z));
                color.push([1.0, 1.0, 1.0]);
                b.push(1.0);
            }
        }
    }
    let f = frame(pos, color, b);
    let lights = cluster_lights(&f);

    // v1 8³ binning gives the [0,12.5)³ region exactly ONE light; the octree
    // gives strictly more (it separates the cluster from the straggler).
    let in_bin = lights
        .iter()
        .filter(|l| {
            (0.0..12.5).contains(&l.pos.x)
                && (0.0..12.5).contains(&l.pos.y)
                && (0.0..12.5).contains(&l.pos.z)
        })
        .count();
    assert!(
        in_bin >= 2,
        "octree must resolve the joint bin (got {in_bin}, v1 = 1)"
    );

    // v1 global softening radius = ½·√3·(bin side) = ½·√3·(100/8).
    let v1_global = 0.5 * 3.0f32.sqrt() * (100.0 / 8.0);
    let max_r_near = lights
        .iter()
        .filter(|l| (l.pos - base).length() < 3.0)
        .map(|l| l.radius)
        .fold(0.0f32, f32::max);
    assert!(
        max_r_near < v1_global,
        "cluster softening radius {max_r_near} not below the v1 global {v1_global}"
    );
}

// ---------- gate 5: near-field accuracy (hand-derived oracle) ----------

/// A standalone compact cluster (64 stars in a 0.1 cube at the origin, varied
/// brightness). One unit away the exact per-star isotropic incident flux
/// `Σ_i L_i/(4π d_i²)` and the clustered surrogate `Σ_k L_k/(4π(d_k²+r_k²))`
/// agree to a few percent — the cluster is tight so its lights sit near the
/// centroid and their softening radii are small. (Under v1 8³ binning of a
/// standalone cluster the single global radius is ½·√3·(0.1/8) ≈ 0.011, so v1
/// is accurate HERE too; gate 4 is where binning fails — this gate pins that
/// the octree's near-field value is physically correct, not merely different.)
#[test]
fn near_field_flux_matches_exact() {
    let mut pos = Vec::new();
    let mut color = Vec::new();
    let mut b = Vec::new();
    let mut rng = Lcg(0x51EED5);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let off = Vec3::new(i as f32, j as f32, k as f32) * (0.1 / 3.0) - Vec3::splat(0.05);
                pos.push(off);
                color.push([1.0, 1.0, 1.0]);
                b.push(0.5 + rng.next()); // [0.5, 1.5)
            }
        }
    }
    let f = frame(pos, color, b);
    let lights = cluster_lights(&f);

    let probe = Vec3::new(0.0, 0.0, 1.0);
    let four_pi = 4.0 * std::f32::consts::PI;

    let mut exact = 0.0f32;
    for i in 0..f.len() {
        let d2 = (probe - f.pos[i]).length_squared();
        let l = (f.color[i][0] + f.color[i][1] + f.color[i][2]) * f.brightness[i];
        exact += l / (four_pi * d2);
    }
    let mut clustered = 0.0f32;
    for l in &lights {
        let d2 = (probe - l.pos).length_squared();
        let lum = l.rgb[0] + l.rgb[1] + l.rgb[2];
        clustered += lum / (four_pi * (d2 + l.radius * l.radius));
    }

    let rel = (clustered - exact).abs() / exact;
    assert!(rel < 0.03, "near-field flux rel err {rel} exceeds 3%");
}

// ---------- gate 6: budget-cap path + safety contract ----------

/// A scale-free hierarchical LCG cloud. TWO facts, both load-bearing:
///
/// 1. **Cap arithmetic (at a reachable budget).** The greedy cap logic is
///    scale-free — all that matters is that the budget binds before natural
///    termination. We drive it at `budget = 16` under an EXPLICIT fine
///    `CAP_TOL = 1e-3`, deliberately NOT the shipped `REFINE_TOL` (a look const
///    now frozen at the coarser `1e-2`, under which this fixture terminates
///    BELOW 16 and the budget would never bind). At `1e-3` the cloud makes ~33
///    leaves unbudgeted, so 16 binds, and the greedy stop (a split adds 1–7
///    leaves, break when the next would breach) lands the count in
///    `[budget − 6, budget] = [10, 16]` — a property DERIVED from the
///    arithmetic, not fitted to output. Decoupling the tol from the ship look
///    const keeps this gate green across look retunes (that is `cluster_lights_with`'s
///    whole reason to exist).
/// 2. **Safety contract (realistic input).** On a large power-law-brightness
///    cloud the shipped `cluster_lights` (at the frozen `1e-2`) stays
///    `≤ MAX_LIGHTS`. The 512 budget is UNREACHABLE at any sane tol: each octree
///    level drops the metric ~32× (⅛ power × ¼ spread²), so `32² > 1/REFINE_TOL`
///    kills uniform refinement at ~64 leaves and heavy tails plateau in the low
///    hundreds — the cap is a GPU-buffer backstop, not the normal terminator. A
///    backstop that always passes with margin is still the correct thing to assert.
#[test]
fn budget_cap_path_and_safety() {
    // hierarchical fractal cloud: 8 top clusters × 8 sub × 64 stars = 4096
    let f6 = {
        let mut rng = Lcg(0xFAC7A1);
        let mut pos = Vec::new();
        let mut color = Vec::new();
        let mut b = Vec::new();
        for _ in 0..8 {
            let c0 = Vec3::new(rng.next(), rng.next(), rng.next()) * 100.0 - Vec3::splat(50.0);
            for _ in 0..8 {
                let c1 =
                    c0 + (Vec3::new(rng.next(), rng.next(), rng.next()) * 10.0 - Vec3::splat(5.0));
                for _ in 0..64 {
                    let c2 = c1
                        + (Vec3::new(rng.next(), rng.next(), rng.next()) * 1.0 - Vec3::splat(0.5));
                    pos.push(c2);
                    color.push([1.0, 1.0, 1.0]);
                    b.push(10.0);
                }
            }
        }
        frame(pos, color, b)
    };

    // Fine tol so the budget genuinely binds; independent of the shipped look
    // const REFINE_TOL (frozen at the coarser 1e-2, under which this fixture
    // terminates below 16). See the fn doc.
    const CAP_TOL: f64 = 1e-3;

    // Unbudgeted the cloud refines past 16, so budget = 16 genuinely binds.
    let natural = cluster_lights_with(&f6, CAP_TOL, MAX_LIGHTS).len();
    assert!(
        natural > 16,
        "fixture must exceed the test budget (got {natural})"
    );

    let budget = 16usize;
    let capped = cluster_lights_with(&f6, CAP_TOL, budget).len();
    assert!(
        capped <= budget && capped >= budget - 6,
        "budget-cap count {capped} outside the derived [{}, {budget}] slack window",
        budget - 6
    );

    // Safety contract: a large heavy-tailed cloud stays under the real budget.
    let big = {
        let mut rng = Lcg(0xB16);
        let mut pos = Vec::new();
        let mut color = Vec::new();
        let mut b = Vec::new();
        for _ in 0..20000 {
            pos.push(Vec3::new(rng.next(), rng.next(), rng.next()) * 100.0);
            color.push([1.0, 1.0, 1.0]);
            let u = rng.next();
            b.push(1.0 / (u + 0.001)); // heavy tail
        }
        frame(pos, color, b)
    };
    assert!(
        cluster_lights(&big).len() <= MAX_LIGHTS,
        "safety contract: shipped clustering exceeded MAX_LIGHTS"
    );
}
