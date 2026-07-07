//! Volumetric gas rendering (M7e, plan D9): the shared march rules and their
//! CPU reference implementations.
//!
//! The frame the renderer produces is
//!
//! ```text
//! L(pixel) = Σ_stars E·T(camera→star)  +  ∫ j(ρ)·T(camera→s) ds
//! ```
//!
//! — both terms ADDITIVE once each carries its own attenuation, so the
//! order-independent `Rgba32Float` additive target survives intact. This module
//! defines the exact numerical rules (ray generation, AABB clipping, step size,
//! quadrature, early exit) and implements them on the CPU; the WGSL shaders in
//! `render.rs` mirror them operation-for-operation, and the GPU gates in
//! `tests/volume.rs` hold the two within f32 tolerance. Density sampling
//! delegates to [`GasGrid::sample`] / [`sample_mix`] — the renderprep functions
//! documented as the M7e shader oracle.
//!
//! **March rule** (one definition, used by every path):
//! - Domain: the ray is clipped against the UNION AABB of both endpoint grids.
//!   Orthographic rays march the full chord (the splat path draws at all
//!   depths — the ortho camera sits at infinity, nothing is "behind" it);
//!   perspective rays start no earlier than the eye (`t ≥ 0`).
//! - Steps: `Δs_nominal =` [`step_size`] `= ½·min cell edge over both grids`;
//!   the chord `[t0, t1]` is divided into `n = ceil((t1−t0)/Δs_nominal)` equal
//!   steps (capped at [`MAX_STEPS`]) and ρ is sampled at step MIDPOINTS.
//! - Quadrature (gas pass): per step, emit THEN attenuate:
//!   `C += T·(emissivity·ρ)·Δs·color;  T *= exp(−κ·ρ·Δs)` — first-order in the
//!   emission/absorption coupling (relative error ≈ ½·κρΔs per step, gated
//!   against the analytic uniform slab), while the accumulated optical depth
//!   itself is midpoint-exact. Early exit when `T <` [`EXIT_TRANSMITTANCE`]
//!   (truncation error ≤ EXIT_TRANSMITTANCE·(emissivity/κ), gated).
//! - Star transmittance: the same clip + step rule over the star→camera
//!   segment, but pure optical depth `τ = Σ κ·ρ·Δs` summed then exponentiated
//!   once (`T = exp(−τ)`) — no emission, no early exit.
//!
//! Emission is `j(ρ) = emissivity·ρ` per unit length, tinted by `color`;
//! absorption is `κ·ρ` per unit length. Both are LOOK uniforms ([`GasLook`]),
//! not baked into the grid — the look iterates at re-render cost (plan D8).

use std::collections::BinaryHeap;

use glam::Vec3;

use galaxy_renderprep::{sample_mix, FrameData, GasGrid};

use crate::camera::{Camera, Projection};
use crate::render::HdrImage;

/// Early-exit threshold for the gas march: once transmittance falls below this,
/// everything behind contributes less than this fraction of the saturated
/// radiance `emissivity/κ` — the march stops. Shared verbatim by the WGSL
/// fragment shader (injected into the source) and the CPU mirror.
pub const EXIT_TRANSMITTANCE: f32 = 1e-4;

/// Hard cap on march steps per ray — a backstop against degenerate step/chord
/// ratios (never reached by sane grids: a 128³ diagonal at half-cell steps is
/// ~443 steps). Shared by shader and CPU mirror so both truncate identically.
pub const MAX_STEPS: u32 = 1 << 20;

/// Maximum point-light proxies emitted by [`cluster_lights`]: a quality
/// **constant** (like the gas grid resolution), not a look knob — the adaptive
/// octree cut spends at most this many lights approximating the stellar
/// distribution. `512` retires the old fixed `8³` worst case and keeps the
/// shadow-volume footprint `MAX_LIGHTS·SHADOW_RES³·4 B = 64 MiB` under wgpu's
/// default 128 MiB storage-binding limit.
pub const MAX_LIGHTS: usize = 512;

/// Relative error floor for the [`cluster_lights`] octree cut: refinement stops
/// when the worst leaf's metric (`P·spread²`) falls below `REFINE_TOL ×` the
/// root metric. This keeps the *typical* light count adaptive — a compact frame
/// clusters to few lights instead of greedily burning all [`MAX_LIGHTS`] every
/// frame. Frozen at `1e-2` from the gasrich QUICK A/B (coarse/1e-3/fine sweep).
///
/// This value ALSO sets scattered-core brightness: `scatter_softening` stayed
/// `None` (per-cluster radius), so a coarser cut (larger cells ⇒ larger softening
/// radii) dims and diffuses the scattered cores. `1e-2` was picked as the softer,
/// less blob-prone look — not a neutral quality knob. See the gasrich preset.
pub const REFINE_TOL: f64 = 1e-2;

/// Octree depth backstop for [`cluster_lights`]: two distinct positions
/// separate into different octants within ~mantissa-many levels, so this bound
/// only guards against float degeneracy — it is never the design terminator.
pub const MAX_DEPTH: u32 = 32;

/// Shadow-lattice resolution per axis for [`bake_shadows`]
/// (umbral-lantern-lattice): a quality **constant** like [`MAX_LIGHTS`], not a
/// look knob. Shadows are soft occlusion, so 32³ over the march domain is
/// honest resolution, and the worst-case GPU footprint
/// `MAX_LIGHTS·SHADOW_RES³·4 B = 64 MiB` stays under wgpu's default 128 MiB
/// storage-binding limit.
pub const SHADOW_RES: u32 = 32;

/// Per-light 3-D transmittance volumes (umbral-lantern-lattice): for each
/// clustered light `k`, `T_k = exp(−∫ κ·ρ_mix ds)` over the light → voxel
/// segment, tabulated at the centers of a [`SHADOW_RES`]³ lattice spanning the
/// union grid AABB (the march domain). Voxel centers sit at
/// `bounds_min + (i + 0.5)·cell` with `cell = (bounds_max − bounds_min) /
/// SHADOW_RES`; `data` is light-major, x-fastest within a volume (the grid
/// deposit order): `data[k·R³ + (iz·R + iy)·R + ix]`.
#[derive(Clone, Debug, PartialEq)]
pub struct ShadowVolumes {
    /// Lattice domain minimum — the union AABB of both endpoint grids.
    pub bounds_min: Vec3,
    /// Lattice domain maximum.
    pub bounds_max: Vec3,
    /// Number of light volumes (= the frame's light count).
    pub count: usize,
    /// `count · SHADOW_RES³` transmittances, light-major, x-fastest.
    pub data: Vec<f32>,
}

impl ShadowVolumes {
    /// Trilinear sample of light `k`'s volume at world point `p`, pure
    /// clamp-to-edge — [`GasGrid::sample`]'s cell-center arithmetic MINUS the
    /// zero-outside test. A transmittance has no natural zero outside the
    /// domain, and the march only samples inside the clipped chord; returning
    /// 0 (= fully shadowed) on an epsilon-outside float excursion would punch
    /// dark rims. Mirrored operation-for-operation by the WGSL
    /// `shadow_sample` (which runs the coordinate arithmetic in f32; the
    /// GPU ≡ CPU gates allow the difference).
    pub fn sample(&self, k: usize, p: Vec3) -> f32 {
        let r = SHADOW_RES;
        let cell = (self.bounds_max.as_dvec3() - self.bounds_min.as_dvec3()) / r as f64;
        let q = p.as_dvec3() - self.bounds_min.as_dvec3();
        // Continuous lattice coordinate: voxel center i sits at coordinate i.
        let cx = q.x / cell.x - 0.5;
        let cy = q.y / cell.y - 0.5;
        let cz = q.z / cell.z - 0.5;

        // Per-axis floor index + fraction, clamp-to-edge everywhere (GasGrid's
        // `axis` rule — the clamp extends past the bounds instead of zeroing).
        let axis = |c: f64| -> (u32, u32, f32) {
            let max = (r - 1) as f64;
            let c = c.clamp(0.0, max);
            let i0 = c.floor().min(max - 1.0).max(0.0) as u32;
            let i1 = (i0 + 1).min(r - 1);
            let t = (c - i0 as f64) as f32;
            (i0, i1, t)
        };
        let (x0, x1, tx) = axis(cx);
        let (y0, y1, ty) = axis(cy);
        let (z0, z1, tz) = axis(cz);

        // Two-product lerp: bit-exact at t = 0 and t = 1 (GasGrid's rule).
        let lerp = |a: f32, b: f32, t: f32| (1.0 - t) * a + t * b;
        let base = k * (r as usize).pow(3);
        let at = |ix: u32, iy: u32, iz: u32| self.data[base + ((iz * r + iy) * r + ix) as usize];

        let c00 = lerp(at(x0, y0, z0), at(x1, y0, z0), tx);
        let c10 = lerp(at(x0, y1, z0), at(x1, y1, z0), tx);
        let c01 = lerp(at(x0, y0, z1), at(x1, y0, z1), tx);
        let c11 = lerp(at(x0, y1, z1), at(x1, y1, z1), tx);
        let c0 = lerp(c00, c10, ty);
        let c1 = lerp(c01, c11, ty);
        lerp(c0, c1, tz)
    }
}

/// CPU reference for the shadow-bake compute prepass (umbral-lantern-lattice):
/// one [`SHADOW_RES`]³ transmittance volume per `gas.lights` entry, over the
/// union AABB of both endpoint grids. Per voxel: march FROM the light TOWARD
/// the voxel center (`t = 0` at the light), the segment clipped against the
/// union AABB and truncated at the voxel, the shared nominal step
/// ([`step_size`] — the density band-limit governs accuracy, not the shadow
/// lattice), τ summed then exponentiated once — [`star_transmittance`]'s exact
/// operation order, in f32 (the WGSL mirror discipline). An empty segment (no
/// gas between light and voxel, or a voxel coincident with the light) is
/// exactly `T = 1`. The shadow segment is geometric from the light's centroid:
/// the cluster softening radius applies only to the 1/d² intensity pole —
/// occlusion has no pole to kill.
pub fn bake_shadows(gas: &GasFrame) -> ShadowVolumes {
    let (bmin, bmax) = union_bounds(gas);
    let r = SHADOW_RES as usize;
    // Voxel centers in f32, `bmin + (i + ½)·cell` — the WGSL bake's exact
    // arithmetic, so a light placed on a center is `dist == 0` on both sides.
    let cell = (bmax - bmin) / SHADOW_RES as f32;
    let ds_nominal = step_size(gas.grid0, gas.grid1);
    let mut data = vec![0.0f32; gas.lights.len() * r * r * r];
    for (k, l) in gas.lights.iter().enumerate() {
        for iz in 0..r {
            for iy in 0..r {
                for ix in 0..r {
                    let vc = bmin
                        + (Vec3::new(ix as f32, iy as f32, iz as f32) + Vec3::splat(0.5)) * cell;
                    let idx = k * r * r * r + (iz * r + iy) * r + ix;
                    data[idx] = light_transmittance(gas, l.pos, vc, bmin, bmax, ds_nominal);
                }
            }
        }
    }
    ShadowVolumes {
        bounds_min: bmin,
        bounds_max: bmax,
        count: gas.lights.len(),
        data,
    }
}

/// Traversal strategy for the shadow bake (plan: DDA/hierarchical acceleration,
/// the named deferral of umbral-lantern-lattice). Both strategies produce a
/// **bit-identical** [`ShadowVolumes`]: [`ShadowBake::Dda`] is a pure speedup
/// that skips only samples it can prove add exactly `κ·0·ds = 0` to τ — the
/// density field is trilinear over deposited grids and returns exactly `0.0`
/// outside particle support, so adding those zeros is a float no-op and the two
/// bakes coincide to the last bit (the equivalence gate). [`ShadowBake::Brute`]
/// is the reference oracle: every sample of every chord evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ShadowBake {
    /// Evaluate every sample of the uniform chord lattice — the reference.
    #[default]
    Brute,
    /// Hierarchical empty-space skipping over a conservative occupancy grid;
    /// bit-identical to [`ShadowBake::Brute`], faster on sparse frames.
    Dda,
}

/// [`bake_shadows`] with a selectable traversal [`ShadowBake`]. The `Brute` arm
/// is [`bake_shadows`] verbatim (the reference); the `Dda` arm skips
/// provably-empty samples for the same result. Callers plumb the strategy from
/// the look/xtask; production keeps `Brute` as the default via [`bake_shadows`].
pub fn bake_shadows_with(gas: &GasFrame, bake: ShadowBake) -> ShadowVolumes {
    match bake {
        ShadowBake::Brute => bake_shadows(gas),
        ShadowBake::Dda => bake_shadows_dda(gas),
    }
}

/// Occupancy-lattice resolution per axis for [`bake_shadows_dda`]: a conservative
/// boolean grid over the union AABB whose active cells enclose everywhere the
/// density could be nonzero. A quality/perf **constant** (not a look knob): finer
/// tightens the skip but adds DDA steps; the hierarchical mip (a later, purely
/// perf refinement) restores big jumps over a fine base. Bit-exactness is
/// independent of this value — it only trades build cost against skip precision.
const SHADOW_OCC_RES: u32 = 64;

/// Coarsest level resolution per axis of the occupancy pyramid: the top of the
/// mip lets a shadow ray crossing a large empty region skip it in ONE coarse
/// step instead of walking every fine cell. Must divide [`SHADOW_OCC_RES`] by a
/// power of two (`64 → 32 → 16 → 8 → 4`, five levels). Purely a perf constant —
/// bit-exactness is level-count-independent (every level is a max-pool of the
/// base, so an inactive coarse cell still implies exactly-zero density).
const SHADOW_OCC_COARSEST: u32 = 4;

/// One resolution level of the [`Occupancy`] pyramid: a `res³` boolean lattice
/// over the union AABB, x-fastest.
struct OccLevel {
    res: u32,
    /// World-space cell edge lengths (`extent / res`).
    cell: Vec3,
    /// `1 / cell`, cached for the point → index map.
    inv_cell: Vec3,
    active: Vec<bool>,
}

impl OccLevel {
    /// Integer cell index of world point `p`, clamped to `[0, res)` (the march
    /// only samples inside the clipped chord, i.e. inside the AABB up to float
    /// slop at the faces).
    fn cell_of(&self, p: Vec3, bmin: Vec3) -> [usize; 3] {
        let c = ((p - bmin) * self.inv_cell).floor();
        let m = (self.res - 1) as usize;
        let ax = |v: f32| (v.max(0.0) as usize).min(m);
        [ax(c.x), ax(c.y), ax(c.z)]
    }

    fn is_active(&self, cell: [usize; 3]) -> bool {
        let r = self.res as usize;
        self.active[(cell[2] * r + cell[1]) * r + cell[0]]
    }

    /// Arc-length parameter (from `origin`, `dir` unit) at which the ray leaves
    /// `cell` — the nearest exit face. Axes with `|dir| < 1e-12` never cross (∞).
    fn cell_exit(&self, cell: [usize; 3], bmin: Vec3, origin: Vec3, dir: Vec3) -> f32 {
        let cmin = bmin + Vec3::new(cell[0] as f32, cell[1] as f32, cell[2] as f32) * self.cell;
        let cmax = cmin + self.cell;
        let mut s = f32::INFINITY;
        for a in 0..3 {
            if dir[a].abs() < 1e-12 {
                continue;
            }
            let boundary = if dir[a] > 0.0 { cmax[a] } else { cmin[a] };
            s = s.min((boundary - origin[a]) / dir[a]);
        }
        s
    }
}

/// A conservative HIERARCHICAL empty-space acceleration structure for the shadow
/// bake: a mip pyramid of boolean lattices over the union AABB (coarsest first,
/// finest = [`SHADOW_OCC_RES`]³ last), where an active cell encloses everywhere
/// [`density_at`] could be nonzero. The base is deliberately **over-inclusive**
/// — a cell may be active where the density is actually zero (harmless: the
/// march evaluates it and adds `κ·0·ds = 0`) — but never **under-inclusive**: a
/// sample with nonzero density always falls in an active base cell, and every
/// coarse level is a max-pool of the base, so an inactive coarse cell still
/// implies all its fine children (and hence the density everywhere within) are
/// zero. That is what makes the hierarchical DDA skip bit-exact.
struct Occupancy {
    /// Lattice minimum (the union AABB min — shared with the shadow lattice).
    bmin: Vec3,
    /// Pyramid levels, COARSEST first, finest (`SHADOW_OCC_RES`) last.
    levels: Vec<OccLevel>,
}

impl Occupancy {
    /// Build the base occupancy over `[bmin, bmax]` by splatting each nonzero
    /// density cell's *influence region* — the cell center ± one cell edge, the
    /// exact reach of the trilinear stencil (a point in geometric cell `i` reads
    /// centers `{i−1, i, i+1}`, so a nonzero center can raise the density
    /// anywhere within ±one cell) — into every occupancy cell it overlaps, then
    /// max-pool it up to [`SHADOW_OCC_COARSEST`]. The two endpoint grids are
    /// unioned, and a grid weighted exactly zero by the mix (`u = 0` drops
    /// grid1, `u = 1` drops grid0) is skipped since it cannot affect the density.
    /// The ±1-cell dilation is the whole correctness argument: it guarantees no
    /// nonzero sample is ever skipped.
    fn build(gas: &GasFrame, bmin: Vec3, bmax: Vec3) -> Occupancy {
        let r = SHADOW_OCC_RES as usize;
        let inv_cell = Vec3::ONE / ((bmax - bmin) / Vec3::splat(SHADOW_OCC_RES as f32));
        let mut base = vec![false; r * r * r];

        let mut mark_grid = |g: &GasGrid| {
            let gcell = g.cell_size().as_vec3();
            let gbmin = g.bounds_min;
            for iz in 0..g.dims[2] {
                for iy in 0..g.dims[1] {
                    for ix in 0..g.dims[0] {
                        if g.data[g.index(ix, iy, iz)] == 0.0 {
                            continue;
                        }
                        let center = gbmin
                            + (Vec3::new(ix as f32, iy as f32, iz as f32) + Vec3::splat(0.5))
                                * gcell;
                        // Influence AABB = center ± one grid cell (the stencil reach).
                        let amin = ((center - gcell - bmin) * inv_cell).floor();
                        let amax = ((center + gcell - bmin) * inv_cell).floor();
                        let clamp = |v: f32| (v.max(0.0) as usize).min(r - 1);
                        let (x0, x1) = (clamp(amin.x), clamp(amax.x));
                        let (y0, y1) = (clamp(amin.y), clamp(amax.y));
                        let (z0, z1) = (clamp(amin.z), clamp(amax.z));
                        for cz in z0..=z1 {
                            for cy in y0..=y1 {
                                for cx in x0..=x1 {
                                    base[(cz * r + cy) * r + cx] = true;
                                }
                            }
                        }
                    }
                }
            }
        };
        // Mix-aware: a grid weighted exactly 0 contributes exactly 0 density.
        if gas.mix != 1.0 {
            mark_grid(gas.grid0);
        }
        if gas.mix != 0.0 {
            mark_grid(gas.grid1);
        }

        let level = |res: u32, active: Vec<bool>| {
            let cell = (bmax - bmin) / Vec3::splat(res as f32);
            OccLevel {
                res,
                cell,
                inv_cell: Vec3::ONE / cell,
                active,
            }
        };
        // Fine → coarse: each coarser level is the 2³ OR-pool of the finer one.
        let mut levels = vec![level(SHADOW_OCC_RES, base)];
        while levels.last().unwrap().res > SHADOW_OCC_COARSEST {
            let finer = levels.last().unwrap();
            let fr = finer.res as usize;
            let cr = fr / 2;
            let mut coarse = vec![false; cr * cr * cr];
            for cz in 0..cr {
                for cy in 0..cr {
                    for cx in 0..cr {
                        let mut any = false;
                        'pool: for dz in 0..2 {
                            for dy in 0..2 {
                                for dx in 0..2 {
                                    let (fx, fy, fz) = (cx * 2 + dx, cy * 2 + dy, cz * 2 + dz);
                                    if finer.active[(fz * fr + fy) * fr + fx] {
                                        any = true;
                                        break 'pool;
                                    }
                                }
                            }
                        }
                        coarse[(cz * cr + cy) * cr + cx] = any;
                    }
                }
            }
            levels.push(level(cr as u32, coarse));
        }
        levels.reverse(); // coarsest first for the descent
        Occupancy { bmin, levels }
    }
}

/// The DDA/hierarchical accelerated bake: identical `ShadowVolumes` to
/// [`bake_shadows`], computed by skipping samples in provably-empty occupancy
/// cells. Bit-exact — the equivalence gate is `assert_eq!(bake_shadows(gas),
/// bake_shadows_dda(gas))`.
fn bake_shadows_dda(gas: &GasFrame) -> ShadowVolumes {
    let (bmin, bmax) = union_bounds(gas);
    let r = SHADOW_RES as usize;
    let cell = (bmax - bmin) / SHADOW_RES as f32;
    let ds_nominal = step_size(gas.grid0, gas.grid1);
    let occ = Occupancy::build(gas, bmin, bmax);
    let mut data = vec![0.0f32; gas.lights.len() * r * r * r];
    for (k, l) in gas.lights.iter().enumerate() {
        for iz in 0..r {
            for iy in 0..r {
                for ix in 0..r {
                    let vc = bmin
                        + (Vec3::new(ix as f32, iy as f32, iz as f32) + Vec3::splat(0.5)) * cell;
                    let idx = k * r * r * r + (iz * r + iy) * r + ix;
                    data[idx] =
                        light_transmittance_dda(gas, l.pos, vc, bmin, bmax, ds_nominal, &occ);
                }
            }
        }
    }
    ShadowVolumes {
        bounds_min: bmin,
        bounds_max: bmax,
        count: gas.lights.len(),
        data,
    }
}

/// [`light_transmittance`] with empty-space skipping over `occ`. It walks the
/// **identical** sample lattice (`s_i = t0 + (i+½)·ds` over the identical clipped
/// chord, τ summed in increasing `i`), but where a sample falls in an inactive
/// occupancy cell it DDA-jumps past the whole empty span instead of evaluating.
/// Every skipped sample is provably in an inactive (⇒ zero-density) cell, so the
/// skipped addends are exact zeros — the nonzero addends are the same values in
/// the same order, hence τ, and `exp(−τ)`, are bit-identical to the brute march.
fn light_transmittance_dda(
    gas: &GasFrame,
    light: Vec3,
    voxel: Vec3,
    bmin: Vec3,
    bmax: Vec3,
    ds_nominal: f32,
    occ: &Occupancy,
) -> f32 {
    let d = voxel - light;
    let dist = d.length();
    if dist == 0.0 {
        return 1.0;
    }
    let dir = d / dist;
    let Some((t0_raw, t1_raw)) = clip_aabb(light, dir, bmin, bmax) else {
        return 1.0;
    };
    let t0 = t0_raw.max(0.0);
    let t1 = t1_raw.min(dist);
    if t0 >= t1 {
        return 1.0;
    }
    let (n, ds) = steps(t0, t1, ds_nominal);
    let mut tau = 0.0_f32;
    let mut i: u32 = 0;
    while i < n {
        // Sample position — the exact brute expression, so an evaluated sample
        // is bit-identical to the reference's.
        let s = t0 + (i as f32 + 0.5) * ds;
        let p = light + dir * s;
        // Descend the pyramid coarsest → finest: the first inactive level jumps
        // across its (as-large-as-possible) empty cell; reaching the finest
        // level still active means a real occupied sample to evaluate.
        let mut jumped = false;
        for lvl in &occ.levels {
            let cell = lvl.cell_of(p, occ.bmin);
            if !lvl.is_active(cell) {
                // Inactive cell: every sample within it is a zero addend. Jump to
                // the first sample at/after the cell exit. The `−1` sample of
                // margin (re-classified next iteration) keeps a boundary sample —
                // which a float-ε over-estimate of the exit could otherwise
                // mis-skip — on the safe side; `max(i+1, …)` guarantees progress.
                let s_exit = lvl.cell_exit(cell, occ.bmin, light, dir);
                let jump = ((s_exit - t0) / ds - 0.5).ceil() as i64 - 1;
                i = ((i as i64 + 1).max(jump).min(n as i64)) as u32;
                jumped = true;
                break;
            }
        }
        if !jumped {
            tau += gas.look.opacity * density_at(gas, p) * ds;
            i += 1;
        }
    }
    (-tau).exp()
}

/// One shadow chord: `T = exp(−τ)` over the light → voxel-center segment,
/// clipped to the union AABB and truncated at the voxel — the bake's per-voxel
/// kernel ([`star_transmittance`]'s operation order: τ summed from the light
/// outward, one exponentiation, no early exit). An empty segment is exactly 1.
fn light_transmittance(
    gas: &GasFrame,
    light: Vec3,
    voxel: Vec3,
    bmin: Vec3,
    bmax: Vec3,
    ds_nominal: f32,
) -> f32 {
    let d = voxel - light;
    let dist = d.length();
    if dist == 0.0 {
        return 1.0; // light on the voxel center: zero path, unshadowed
    }
    let dir = d / dist;
    let Some((t0_raw, t1_raw)) = clip_aabb(light, dir, bmin, bmax) else {
        return 1.0;
    };
    // Only gas BETWEEN the light and the voxel occludes.
    let t0 = t0_raw.max(0.0);
    let t1 = t1_raw.min(dist);
    if t0 >= t1 {
        return 1.0;
    }
    let (n, ds) = steps(t0, t1, ds_nominal);
    let mut tau = 0.0_f32;
    for i in 0..n {
        let s = t0 + (i as f32 + 0.5) * ds;
        tau += gas.look.opacity * density_at(gas, light + dir * s) * ds;
    }
    (-tau).exp()
}

/// One point-light proxy for the single-scatter term: a cluster of stellar
/// splats collapsed to their emission-weighted centroid. `radius` softens the
/// inverse-square law (`d² + radius²`) — inside a cluster's own extent the
/// point approximation is invalid anyway, so the softening is honest and it
/// kills the 1/d² pole when a march sample lands on a light.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// Emission-weighted centroid of the cluster (world space).
    pub pos: Vec3,
    /// Softening radius: half the cluster bin-cell diagonal.
    pub radius: f32,
    /// Total RGB power of the cluster: Σ color·brightness over its stars
    /// (power is conserved exactly across the clustering).
    pub rgb: [f32; 3],
}

/// Cluster the frame's stellar splats into at most [`MAX_LIGHTS`] point lights
/// via a deterministic greedy octree cut over the *luminous* stars (total
/// emission > 0 — dark splats neither light gas nor stretch the bounds).
/// Refinement pops the leaf with the largest far-field flux-error surrogate
/// (`P·spread²`) and splits it by octant until every leaf falls below
/// [`REFINE_TOL`]×root or the budget is spent. Each leaf emits one light at its
/// luminance-weighted centroid with summed RGB power (exact in the f64 fold)
/// and a softening radius of half its own cell diagonal. All f64 accumulation,
/// deterministic index-order folds; the clustering has no GPU mirror (the GPU
/// consumes its flat output). An all-dark frame clusters to no lights.
///
/// This is the shipped entry point — the quality constants [`REFINE_TOL`] and
/// [`MAX_LIGHTS`] are baked in. It delegates to [`cluster_lights_with`], which
/// exposes both as parameters so a test can drive the budget-cap path at a
/// *reachable* budget (the shipped 512 is unreachable at `REFINE_TOL` — the
/// octree metric drops 32× per level, so natural distributions cap at ~64 leaves
/// and heavy-tailed ones at a few hundred; the cap is a GPU-buffer backstop,
/// never the normal terminator).
pub fn cluster_lights(frame: &FrameData) -> Vec<Light> {
    cluster_lights_with(frame, REFINE_TOL, MAX_LIGHTS)
}

/// The parameterized core of [`cluster_lights`]: identical greedy octree cut,
/// but with the error floor `tol` (relative to the root metric) and the light
/// `budget` (max leaf count) as arguments. Shipped clustering is
/// `cluster_lights_with(frame, REFINE_TOL, MAX_LIGHTS)`. Exposed so the gates
/// can exercise the budget-cap arithmetic at a budget the metric can actually
/// reach — the safety-critical path that protects the `MAX_LIGHTS·32³·4 B`
/// shadow buffer but is dead code under the shipped constants. Not a look knob;
/// production must call [`cluster_lights`].
pub fn cluster_lights_with(frame: &FrameData, tol: f64, budget: usize) -> Vec<Light> {
    // Total emissive power / scalar luminance of star `i` — the clustering
    // weight. Zero-power splats are invisible to the gas AND to the bounds (a
    // dark far-flung particle must not move the root cube).
    fn power(frame: &FrameData, i: usize) -> [f64; 3] {
        let c = frame.color[i];
        let b = frame.brightness[i] as f64;
        [c[0] as f64 * b, c[1] as f64 * b, c[2] as f64 * b]
    }
    fn lum(frame: &FrameData, i: usize) -> f64 {
        let p = power(frame, i);
        p[0] + p[1] + p[2]
    }

    // 1. Luminous set + bounds, in ascending star-index order (the fold
    //    discipline: every aggregate below folds members in this order).
    let mut bmin = Vec3::splat(f32::INFINITY);
    let mut bmax = Vec3::splat(f32::NEG_INFINITY);
    let mut root_members: Vec<usize> = Vec::new();
    for i in 0..frame.len() {
        if lum(frame, i) > 0.0 {
            bmin = bmin.min(frame.pos[i]);
            bmax = bmax.max(frame.pos[i]);
            root_members.push(i);
        }
    }
    if root_members.is_empty() {
        return Vec::new();
    }

    // Root cell = the luminous AABB expanded to a cube (center = AABB center,
    // side = max extent — the Barnes-Hut/LBVH root convention; octants stay
    // shape-regular). A coincident/single luminous set has side 0 ⇒ one light,
    // radius 0 (v1's degenerate contract).
    let ext = bmax - bmin;
    let root_center = 0.5 * (bmin + bmax);
    let root_half = 0.5 * ext.max_element().max(0.0);

    // 2. Node state: cubic cell + member indices + f64 aggregates (scalar
    //    luminance P, per-channel power, luminance-weighted centroid numerator,
    //    the member-position AABB whose half-diagonal is `spread`).
    struct Node {
        center: Vec3,
        half: f32,
        depth: u32,
        members: Vec<usize>,
        p: f64,
        rgb: [f64; 3],
        wpos: glam::DVec3,
        metric: f64,
        children: Vec<usize>,
    }
    fn make_node(
        arena: &mut Vec<Node>,
        frame: &FrameData,
        center: Vec3,
        half: f32,
        depth: u32,
        members: Vec<usize>,
    ) -> usize {
        let mut p = 0.0f64;
        let mut rgb = [0.0f64; 3];
        let mut wpos = glam::DVec3::ZERO;
        let mut amin = Vec3::splat(f32::INFINITY);
        let mut amax = Vec3::splat(f32::NEG_INFINITY);
        for &i in &members {
            let w = lum(frame, i);
            let pos = frame.pos[i];
            p += w;
            wpos += pos.as_dvec3() * w;
            for (acc, v) in rgb.iter_mut().zip(power(frame, i)) {
                *acc += v;
            }
            amin = amin.min(pos);
            amax = amax.max(pos);
        }
        // 3. Metric = P·spread² — the far-field flux-error surrogate. Using the
        //    member spread (not the cell diagonal) makes single-star and
        //    coincident nodes exactly metric 0: they terminate structurally.
        let spread = 0.5 * (amax - amin).length() as f64;
        let metric = p * spread * spread;
        let id = arena.len();
        arena.push(Node {
            center,
            half,
            depth,
            members,
            p,
            rgb,
            wpos,
            metric,
            children: Vec::new(),
        });
        id
    }

    let mut arena: Vec<Node> = Vec::new();
    let root = make_node(&mut arena, frame, root_center, root_half, 0, root_members);
    let root_metric = arena[root].metric;
    let threshold = tol * root_metric;

    // A node can be usefully split only if it has ≥ 2 members and is above the
    // depth backstop (nodes at MAX_DEPTH never enter the heap).
    let splittable = |n: &Node| n.depth < MAX_DEPTH && n.members.len() >= 2;

    // 4. Greedy refinement: a max-heap of splittable leaves ordered by
    //    (metric, id). Both keys fully deterministic ⇒ reproducible cut.
    struct HeapItem {
        metric: f64,
        id: usize,
    }
    impl PartialEq for HeapItem {
        fn eq(&self, o: &Self) -> bool {
            self.metric.total_cmp(&o.metric).is_eq() && self.id == o.id
        }
    }
    impl Eq for HeapItem {}
    impl Ord for HeapItem {
        fn cmp(&self, o: &Self) -> std::cmp::Ordering {
            self.metric.total_cmp(&o.metric).then(self.id.cmp(&o.id))
        }
    }
    impl PartialOrd for HeapItem {
        fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
            Some(self.cmp(o))
        }
    }

    // The tree always has ≥ 1 leaf (the root). A single/coincident luminous
    // set leaves the root unsplittable ⇒ it is the sole light.
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    let mut leaf_count = 1usize;
    if splittable(&arena[root]) {
        heap.push(HeapItem {
            metric: root_metric,
            id: root,
        });
    }

    while let Some(item) = heap.pop() {
        // Stop once the worst leaf falls below the error floor: the heap is
        // ordered by metric, so every remaining leaf is below it too.
        if item.metric <= threshold {
            break;
        }
        let (center, half, depth) = {
            let n = &arena[item.id];
            (n.center, n.half, n.depth)
        };
        // Partition members by octant, preserving ascending index order.
        let mut groups: [Vec<usize>; 8] = std::array::from_fn(|_| Vec::new());
        for &i in &arena[item.id].members {
            let p = frame.pos[i];
            let oct = usize::from(p.x >= center.x)
                | (usize::from(p.y >= center.y) << 1)
                | (usize::from(p.z >= center.z) << 2);
            groups[oct].push(i);
        }
        let nonempty = groups.iter().filter(|g| !g.is_empty()).count();
        // Budget: replacing this leaf (in hand, not counted) with `nonempty`
        // children. If that would exceed `budget`, keep it a leaf and stop
        // refinement (a split adds 1–7 leaves, so the greedy stop lands the
        // final count in `[budget − 6, budget]`).
        if leaf_count - 1 + nonempty > budget {
            break;
        }
        leaf_count = leaf_count - 1 + nonempty;
        let child_half = 0.5 * half;
        let mut child_ids: Vec<usize> = Vec::new();
        for (oct, g) in groups.into_iter().enumerate() {
            if g.is_empty() {
                continue;
            }
            let cx = center.x
                + if oct & 1 != 0 {
                    child_half
                } else {
                    -child_half
                };
            let cy = center.y
                + if oct & 2 != 0 {
                    child_half
                } else {
                    -child_half
                };
            let cz = center.z
                + if oct & 4 != 0 {
                    child_half
                } else {
                    -child_half
                };
            let cid = make_node(
                &mut arena,
                frame,
                Vec3::new(cx, cy, cz),
                child_half,
                depth + 1,
                g,
            );
            child_ids.push(cid);
            if splittable(&arena[cid]) {
                heap.push(HeapItem {
                    metric: arena[cid].metric,
                    id: cid,
                });
            }
        }
        arena[item.id].children = child_ids;
    }

    // 5. Emission: final leaves (childless nodes) in DFS pre-order by octant
    //    index — a canonical, reproducible order. Per leaf: pos = luminance-
    //    weighted centroid, rgb = Σ power (one f32 cast per channel), radius =
    //    ½ the leaf's OWN cell diagonal (v1's honesty rule, now per-light).
    let mut lights: Vec<Light> = Vec::new();
    let mut stack = vec![root];
    while let Some(nid) = stack.pop() {
        let n = &arena[nid];
        if n.children.is_empty() {
            lights.push(Light {
                pos: (n.wpos / n.p).as_vec3(),
                radius: n.half * 3.0f32.sqrt(),
                rgb: [n.rgb[0] as f32, n.rgb[1] as f32, n.rgb[2] as f32],
            });
        } else {
            for &c in n.children.iter().rev() {
                stack.push(c);
            }
        }
    }
    lights
}

/// The Henyey–Greenstein phase function `p(cosθ) = (1 − g²) / (4π · (1 + g² −
/// 2g·cosθ)^{3/2})`, normalized over the sphere (∫p dΩ = 1). `g = 0` is
/// isotropic (exactly 1/4π); `g → 1` forward-peaked. Callers keep |g| < 1
/// (the scenario layer validates); the denominator is then ≥ (1−|g|)² > 0.
pub fn hg_phase(cos_theta: f32, g: f32) -> f32 {
    // f64 internally: the CPU oracle is the reference the (f32) WGSL march is
    // gated against at 1e-3, and f64 keeps the g = 0 isotropic limit exactly
    // the correctly-rounded 1/4π.
    let g = g as f64;
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta as f64;
    ((1.0 - g2) / (4.0 * std::f64::consts::PI * denom * denom.sqrt())) as f32
}

/// The single-scatter look knobs, `Option`-gated on [`GasLook::scatter`]
/// (`None` = off = bit-compatible with the pre-scatter renderer).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScatterLook {
    /// Scattering coefficient σ_s per unit (ρ · path length) — the same units
    /// family as `opacity`, tuned by eyeball like κ/j. `0` disables the term.
    pub strength: f32,
    /// Henyey–Greenstein `g` ∈ (−1, 1): 0 isotropic, > 0 forward (backlit
    /// silver-lining), < 0 backward.
    pub anisotropy: f32,
    /// Per-light shadow volumes (umbral-lantern-lattice): when `true` the
    /// RENDERER bakes light→sample transmittances ([`bake_shadows`] is the CPU
    /// reference) and multiplies each light's incident term by `T_k`. `false`
    /// is the v1 unshadowed scatter — bit-compatible. NOTE: the CPU oracle
    /// [`march_gas`] keys on its `shadows` ARGUMENT, not this flag; the flag
    /// tells the renderer (and [`render_gas_cpu`]) whether to bake.
    pub shadows: bool,
    /// Chromatic scattering albedo (tinted-octree-lanterns): a per-channel
    /// multiplier on the SCATTERED radiance only — emission, absorption, the
    /// star splats, and the shadow bake are all untouched. Semantically a
    /// single-scatter albedo (dust reflects blue preferentially — the
    /// reflection-nebula look); it composes multiplicatively with each light's
    /// own RGB. `[1.0; 3]` is bit-identical to the pre-tint march (×1.0 is the
    /// exact identity, CPU and WGSL). Applied once per step OUTSIDE the
    /// per-light sum (it is constant across lights).
    pub tint: [f32; 3],
    /// Fixed scatter softening ε (galaxy-render controls pass): decouples the
    /// single-scatter 1/d² softening length from the light-cluster cell size.
    /// `None` = the v1 per-cluster radius softening (`d² + r_k²`), bit-identical
    /// to the pre-ε march and to the shipped path. `Some(ε)` replaces each
    /// `r_k` with one physical ε — floored at the gas voxel scale (sub-voxel ε
    /// is unresolvable spike noise and the worst temporal-flicker case) — so the
    /// INTEGRATED scattered energy is invariant to the octree [`REFINE_TOL`]
    /// (refinement stops being a hidden brightness knob). `strength` and ε then
    /// set the glow level/spread. Mirrored operation-for-operation by the WGSL
    /// march via a uniform slot; the floor is applied CPU-side before upload so
    /// both paths see the same ε.
    pub softening: Option<f32>,
}

/// Gas look uniforms (plan D8: the grid carries ρ only; everything visual lives
/// here and iterates at re-render cost).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasLook {
    /// Linear RGB tint of the gas emission.
    pub color: [f32; 3],
    /// Emissivity `j`: emitted radiance per unit (ρ · path length).
    pub emissivity: f32,
    /// Opacity `κ`: extinction per unit (ρ · path length). `0` disables
    /// absorption entirely (transmittance ≡ 1 — the emission-only mode, and the
    /// bit-compat limit the gas-off golden gate pins).
    pub opacity: f32,
    /// Single-scatter starlight (scattered-starlit-veil): `None` — or
    /// `strength = 0`, or an empty [`GasFrame::lights`] — is bit-compatible
    /// with the pre-scatter march. Default is UNSHADOWED single scatter; the
    /// optional [`ScatterLook::shadows`] adds the baked light→sample
    /// transmittance (umbral-lantern-lattice). The scattered radiance always
    /// rides the camera-path T.
    pub scatter: Option<ScatterLook>,
}

impl Default for GasLook {
    fn default() -> Self {
        GasLook {
            color: [1.0, 1.0, 1.0],
            emissivity: 1.0,
            opacity: 1.0,
            scatter: None,
        }
    }
}

/// One frame's gas input: the two snapshot-endpoint density grids and the
/// subframe mix `u` (M6c endpoint argument: grids are deposited ONLY at
/// snapshot endpoints; in-betweens blend the two, `ρ = (1−u)·ρ₀ + u·ρ₁`,
/// exactly [`sample_mix`]). A static frame passes the same grid twice with
/// `mix = 0`.
#[derive(Clone, Copy, Debug)]
pub struct GasFrame<'a> {
    /// Density grid at the earlier snapshot endpoint.
    pub grid0: &'a GasGrid,
    /// Density grid at the later snapshot endpoint.
    pub grid1: &'a GasGrid,
    /// Endpoint blend factor `u ∈ [0, 1]`: `0` = `grid0`, `1` = `grid1`.
    pub mix: f32,
    /// Emission/absorption look knobs.
    pub look: GasLook,
    /// Point-light proxies for the single-scatter term ([`cluster_lights`]
    /// output), camera-independent per-frame data. Empty when scattering is
    /// off — an empty list is bit-compatible with `scatter: None`.
    pub lights: &'a [Light],
}

/// The shared nominal step: half the smallest cell edge over BOTH endpoint
/// grids — fine enough that the grid's own band-limit (the deposition kernel)
/// dominates the quadrature error, coarse enough that a full-frame march stays
/// trivial GPU work.
pub fn step_size(grid0: &GasGrid, grid1: &GasGrid) -> f32 {
    let min_edge = |g: &GasGrid| {
        let c = g.cell_size();
        c.x.min(c.y).min(c.z)
    };
    (0.5 * min_edge(grid0).min(min_edge(grid1))) as f32
}

/// The march domain: the union AABB of both endpoint grids (each grid's own
/// sample function zeroes outside its own bounds, so the union over-covers
/// harmlessly and one clip serves both).
fn union_bounds(gas: &GasFrame) -> (Vec3, Vec3) {
    (
        gas.grid0.bounds_min.min(gas.grid1.bounds_min),
        gas.grid0.bounds_max.max(gas.grid1.bounds_max),
    )
}

/// Slab-clip the ray `origin + t·dir` against `[bmin, bmax]`: `Some((t0, t1))`
/// for a non-empty chord, `None` for a miss. Axes where `|dir| < 1e-12` are
/// resolved by an inside test instead of dividing (no 0·∞ NaNs). Mirrored
/// operation-for-operation by the WGSL `clip_aabb` (same ±1e30 sentinels).
fn clip_aabb(origin: Vec3, dir: Vec3, bmin: Vec3, bmax: Vec3) -> Option<(f32, f32)> {
    let mut t0 = -1e30_f32;
    let mut t1 = 1e30_f32;
    for a in 0..3 {
        if dir[a].abs() < 1e-12 {
            if origin[a] < bmin[a] || origin[a] > bmax[a] {
                return None;
            }
        } else {
            let ta = (bmin[a] - origin[a]) / dir[a];
            let tb = (bmax[a] - origin[a]) / dir[a];
            t0 = t0.max(ta.min(tb));
            t1 = t1.min(ta.max(tb));
        }
    }
    (t0 < t1).then_some((t0, t1))
}

/// The mixed density the march samples: exactly [`sample_mix`], the renderprep
/// CPU reference for the shader's two-texture blend.
fn density_at(gas: &GasFrame, p: Vec3) -> f32 {
    sample_mix(gas.grid0, gas.grid1, gas.mix, p)
}

/// Step count and effective step for a chord `[t0, t1]`: `n` equal steps of
/// the nominal size rounded up, capped at [`MAX_STEPS`].
fn steps(t0: f32, t1: f32, ds_nominal: f32) -> (u32, f32) {
    let n = (((t1 - t0) / ds_nominal).ceil() as u32).clamp(1, MAX_STEPS);
    (n, (t1 - t0) / n as f32)
}

/// The camera ray through the CENTER of pixel `(px, py)` of a `width × height`
/// image (top-left origin, matching [`HdrImage`]): returns `(origin, dir)`,
/// `dir` unit length.
///
/// Orthographic: origin on the target plane at the pixel's world position,
/// direction `forward` (all rays parallel). Perspective: origin at the eye
/// (`target − forward·distance`), direction through the pixel's point on the
/// target plane. NDC convention: pixel centers at `(px+½, py+½)`, `x` right,
/// `y` UP (row 0 is NDC y = +1) — exactly the splat vertex path's mapping, so
/// gas and stars agree per pixel. Pinned by hand oracles at corner pixels.
pub fn ray_for_pixel(camera: &Camera, width: u32, height: u32, px: u32, py: u32) -> (Vec3, Vec3) {
    let ndc_x = (px as f32 + 0.5) / (width as f32 / 2.0) - 1.0;
    let ndc_y = 1.0 - (py as f32 + 0.5) / (height as f32 / 2.0);
    let lateral =
        camera.right * (ndc_x * camera.half_extent.x) + camera.up * (ndc_y * camera.half_extent.y);
    match camera.projection {
        Projection::Orthographic => (camera.target + lateral, camera.forward),
        Projection::Perspective { distance, .. } => {
            let eye = camera.target - camera.forward * distance;
            (eye, (camera.target + lateral - eye).normalize())
        }
    }
}

/// CPU reference for the gas fragment march along one ray (module-doc march
/// rule verbatim): returns `(accumulated RGB radiance, final transmittance)`.
///
/// `t_min` clamps the chord start: perspective passes `0.0` (nothing behind
/// the eye), orthographic passes `f32::NEG_INFINITY` (the full chord). A ray
/// that misses both grids returns `([0,0,0], 1.0)`.
///
/// `shadows`: per-light shadow volumes (umbral-lantern-lattice) — `Some`
/// multiplies each light's incident term by the baked `T_k(p)`. The oracle
/// keys on THIS argument, not on [`ScatterLook::shadows`]: callers wanting the
/// shadowed march must pass `Some(&bake_shadows(gas))` themselves (as
/// [`render_gas_cpu`] does when the look asks) — passing the flag but `None`
/// marches unshadowed. It is an oracle API, not a safety rail.
pub fn march_gas(
    gas: &GasFrame,
    shadows: Option<&ShadowVolumes>,
    origin: Vec3,
    dir: Vec3,
    t_min: f32,
) -> ([f32; 3], f32) {
    let (bmin, bmax) = union_bounds(gas);
    let Some((t0_raw, t1)) = clip_aabb(origin, dir, bmin, bmax) else {
        return ([0.0; 3], 1.0);
    };
    let t0 = t0_raw.max(t_min);
    if t0 >= t1 {
        return ([0.0; 3], 1.0);
    }
    let (n, ds) = steps(t0, t1, step_size(gas.grid0, gas.grid1));

    // Single-scatter starlight is active only when the look asks for it AND
    // there are lights to scatter — either alone leaves the march bit-identical
    // to the pre-scatter path (the scatter term is a separate accumulation; the
    // emission/absorption arithmetic below is untouched).
    let scatter = gas
        .look
        .scatter
        .filter(|s| s.strength > 0.0 && !gas.lights.is_empty());

    // Fixed-ε scatter softening² (galaxy-render controls pass): `Some(ε)` floors
    // ε at the voxel scale (`2·step_size` = the smallest cell edge) and shares it
    // across all lights and samples, so the scattered brightness is invariant to
    // the light-cluster refinement tolerance. `None` ⇒ per-light `r_k²` computed
    // in the loop below, bit-identical to the shipped path.
    let scatter_soft2 = scatter.and_then(|s| s.softening).map(|e| {
        let e = e.max(2.0 * step_size(gas.grid0, gas.grid1));
        e * e
    });

    let mut t = 1.0_f32;
    let mut c = [0.0_f32; 3];
    for i in 0..n {
        let s = t0 + (i as f32 + 0.5) * ds;
        let p = origin + dir * s;
        let rho = density_at(gas, p);
        // Emit THEN attenuate (module-doc quadrature rule), the exact operation
        // order of the WGSL march.
        let e = t * gas.look.emissivity * rho * ds;
        c[0] += e * gas.look.color[0];
        c[1] += e * gas.look.color[1];
        c[2] += e * gas.look.color[2];
        if let Some(sl) = scatter {
            // Unshadowed single scatter: incident intensity L/(4π(d²+r²)) per
            // light, HG-phased between the light→sample propagation direction
            // and the sample→camera direction, then emitted like j — same T,
            // same Δs weight. Mirrored operation-for-operation by the WGSL
            // march (which computes the phase in f32; the gates allow 1e-3).
            let w_out = -dir;
            let mut inc = [0.0_f32; 3];
            for (k, l) in gas.lights.iter().enumerate() {
                let dv = p - l.pos;
                let d2_true = dv.length_squared();
                let d2 = d2_true + scatter_soft2.unwrap_or(l.radius * l.radius);
                if d2 <= 0.0 {
                    continue; // sample exactly on a zero-radius light
                }
                let mu = if d2_true > 0.0 {
                    dv.dot(w_out) / d2_true.sqrt()
                } else {
                    0.0
                };
                let mut f = hg_phase(mu, sl.anisotropy) / (4.0 * std::f32::consts::PI * d2);
                // Per-light shadowing (umbral-lantern-lattice): the baked
                // light→sample transmittance, trilinearly sampled. `None`
                // leaves `f` untouched — the v1 arithmetic, bit-identical.
                if let Some(sv) = shadows {
                    f *= sv.sample(k, p);
                }
                inc[0] += l.rgb[0] * f;
                inc[1] += l.rgb[1] * f;
                inc[2] += l.rgb[2] * f;
            }
            // Chromatic scattering albedo (tinted-octree-lanterns): a per-channel
            // multiplier on the scattered radiance, constant across lights so
            // applied once here OUTSIDE the per-light sum. The `* tint` is LAST
            // so a neutral `[1.0; 3]` reduces to `es * inc[ch]` bit-for-bit
            // (×1.0 is the exact identity; left-associative parse).
            let es = t * sl.strength * rho * ds;
            c[0] += es * inc[0] * sl.tint[0];
            c[1] += es * inc[1] * sl.tint[1];
            c[2] += es * inc[2] * sl.tint[2];
        }
        t *= (-(gas.look.opacity * rho * ds)).exp();
        if t < EXIT_TRANSMITTANCE {
            break;
        }
    }
    (c, t)
}

/// CPU reference for the transmittance compute prepass: `T = exp(−τ)` with
/// `τ = ∫ κ·ρ_mix ds` over the segment from `star` toward the camera (to the
/// eye for perspective, to the grid exit along `−forward` for orthographic),
/// clipped against the union grid AABB, same step rule as [`march_gas`].
/// A star with no gas in front returns exactly `1.0`.
pub fn star_transmittance(gas: &GasFrame, camera: &Camera, star: Vec3) -> f32 {
    let (dir, t_max) = match camera.projection {
        Projection::Orthographic => (-camera.forward, f32::INFINITY),
        Projection::Perspective { distance, .. } => {
            let eye = camera.target - camera.forward * distance;
            let d = eye - star;
            let dist = d.length();
            if dist == 0.0 {
                return 1.0; // star at the eye: zero path, unattenuated
            }
            (d / dist, dist)
        }
    };
    let (bmin, bmax) = union_bounds(gas);
    let Some((t0_raw, t1_raw)) = clip_aabb(star, dir, bmin, bmax) else {
        return 1.0;
    };
    // Only gas IN FRONT of the star (toward the camera, and no farther than
    // the eye) attenuates it.
    let t0 = t0_raw.max(0.0);
    let t1 = t1_raw.min(t_max);
    if t0 >= t1 {
        return 1.0;
    }
    let (n, ds) = steps(t0, t1, step_size(gas.grid0, gas.grid1));

    // Pure optical depth: sum τ, exponentiate once (no emission, no early
    // exit) — the exact operation order of the WGSL compute prepass.
    let mut tau = 0.0_f32;
    for i in 0..n {
        let s = t0 + (i as f32 + 0.5) * ds;
        tau += gas.look.opacity * density_at(gas, star + dir * s) * ds;
    }
    (-tau).exp()
}

/// Render the gas pass alone on the CPU: [`ray_for_pixel`] + [`march_gas`] per
/// pixel, `RGB = radiance`, `alpha = 1 − transmittance` (the per-pixel gas
/// opacity) — exactly what the GPU fullscreen pass additively blends into the
/// cleared target. This is the oracle image for the GPU ≡ CPU gates (small
/// resolutions only; it is a reference, not a fast path).
pub fn render_gas_cpu(gas: &GasFrame, camera: &Camera, width: u32, height: u32) -> HdrImage {
    let t_min = match camera.projection {
        Projection::Orthographic => f32::NEG_INFINITY, // the full chord
        Projection::Perspective { .. } => 0.0,         // nothing behind the eye
    };
    // Shadow volumes are baked once per image iff the look asks for them AND
    // the scatter term is active — exactly the renderer's on-device policy
    // (`GasUniforms.scat.w`), so the oracle and the GPU stay in lockstep.
    let shadows = gas
        .look
        .scatter
        .filter(|s| s.shadows && s.strength > 0.0 && !gas.lights.is_empty())
        .map(|_| bake_shadows(gas));
    let mut pixels = Vec::with_capacity((width * height) as usize);
    for py in 0..height {
        for px in 0..width {
            let (origin, dir) = ray_for_pixel(camera, width, height, px, py);
            let (c, t) = march_gas(gas, shadows.as_ref(), origin, dir, t_min);
            pixels.push([c[0], c[1], c[2], 1.0 - t]);
        }
    }
    HdrImage {
        width,
        height,
        pixels,
    }
}
