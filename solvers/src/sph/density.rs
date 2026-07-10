//! SPH density summation with adaptive smoothing lengths.
//!
//! `h_i` solves `N_i(h) = n_ngb` by bisection, where the kernel-weighted
//! neighbor count is the smooth (deterministically root-findable) analogue of
//! "particles within the support radius":
//!
//! ```text
//! N_i(h) = (4π/3) · (SUPPORT·h)³ · Σ_j W(|x_i − x_j|, h)
//! ```
//!
//! `N_i` is monotone nondecreasing in `h` (every summand is), rising from the
//! self-term floor `32/3` at `h → 0` to the plateau `(32/3)·n` once the whole
//! cloud is inside the support — so a root exists iff `n ≳ 4.5` for the default
//! target, and bisection is valid wherever it exists. Bisection runs to a fixed
//! relative tolerance on `h` from a deterministic bracket, so `h` is a pure
//! function of the positions — a warm-start (`h_init`) only seeds the bracket
//! and cannot move the converged value beyond that tolerance (gated).
//!
//! Systems with no root (under-populated, or a pathological coincident knot)
//! clamp deterministically to the bracket bounds: finite `h`, finite positive
//! `ρ`, no panic (gated).

use galaxy_core::DVec3;
use rayon::prelude::*;

use super::grid::HashGrid;
use super::kernel::{w, SUPPORT};

const PI: f64 = std::f64::consts::PI;

/// Adaptive-h configuration.
#[derive(Clone, Debug)]
pub struct DensityConfig {
    /// Target kernel-weighted neighbor count. Default 48; the cubic spline
    /// pairs above ~57 (pairing instability), so keep below that. Must exceed
    /// the self-term floor 32/3 ≈ 10.7 or no root exists for any cloud.
    pub n_ngb: f64,
    /// Bisection convergence: relative tolerance on `h`.
    pub h_tol_rel: f64,
}

impl Default for DensityConfig {
    fn default() -> Self {
        DensityConfig {
            n_ngb: 48.0,
            h_tol_rel: 1e-3,
        }
    }
}

/// Densities and the smoothing lengths they were computed with.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityResult {
    pub rho: Vec<f64>,
    pub h: Vec<f64>,
}

/// Grid-accelerated density with CALLER-SUPPLIED smoothing lengths (the fixed-h
/// special case the unit gates use). Gathers neighbors in ascending index, so
/// the sum associates exactly like [`super::reference::reference_density`]
/// (skipped far particles would contribute an exact `+0.0`) — gated bit-exact
/// against it.
pub fn density_fixed(pos: &[DVec3], mass: &[f64], h: &[f64]) -> Vec<f64> {
    if pos.is_empty() {
        return Vec::new();
    }
    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        h_max.is_finite() && h_max > 0.0,
        "density_fixed needs positive finite smoothing lengths"
    );
    let grid = HashGrid::build(pos, SUPPORT * h_max);
    (0..pos.len())
        .map(|i| {
            let ngb = grid.neighbours_within(pos, pos[i], SUPPORT * h[i]);
            let mut rho = 0.0;
            for &j in &ngb {
                rho += mass[j] * w((pos[i] - pos[j]).length(), h[i]);
            }
            rho
        })
        .collect()
}

/// Adaptive-h density, rayon over targets (each target's neighbor sum has a
/// fixed gather order, so the result is bit-identical to
/// [`density_adaptive_serial`] — gated). `h_init`, if given, seeds the
/// per-particle bisection bracket (warm start).
pub fn density_adaptive(
    pos: &[DVec3],
    mass: &[f64],
    cfg: &DensityConfig,
    h_init: Option<&[f64]>,
) -> DensityResult {
    density_impl(pos, mass, cfg, h_init, true)
}

/// Serial twin of [`density_adaptive`]: the same per-target computation without
/// the rayon dispatch, for the parallel ≡ serial bit-exactness gate.
pub fn density_adaptive_serial(
    pos: &[DVec3],
    mass: &[f64],
    cfg: &DensityConfig,
    h_init: Option<&[f64]>,
) -> DensityResult {
    density_impl(pos, mass, cfg, h_init, false)
}

/// Refresh `(rho, h)` for the `active` targets ONLY (I7 — the individual-timestep
/// efficiency path), writing each result in place at its own index. The grid and
/// per-particle bracket seeds are computed over ALL particles (density is a
/// per-target function of positions, independent of which targets are active), so
/// every active target's solve is **bit-identical** to what [`density_adaptive`]
/// would produce — the active pass is a strict subset of the full pass, not an
/// approximation. Inactive entries of `rho`/`h` are left untouched (the caller's
/// persistent scratch, refreshed once per base block when all rungs synchronize).
///
/// The per-target bracket hint is `h[i]` (its previous converged value, a warm
/// start); a non-finite or non-positive `h[i]` falls back to the occupancy seed,
/// so a zero-initialized `h` on the first call reproduces the `h_init = None` path
/// of [`density_adaptive`]. `rho`, `h`, `pos`, `mass` all have length `pos.len()`.
pub fn density_adaptive_active(
    pos: &[DVec3],
    mass: &[f64],
    cfg: &DensityConfig,
    active: &[usize],
    rho: &mut [f64],
    h: &mut [f64],
) {
    assert_eq!(rho.len(), pos.len(), "rho length must match particle count");
    assert_eq!(h.len(), pos.len(), "h length must match particle count");
    if pos.is_empty() {
        return;
    }
    let setup = DensitySetup::new(pos, mass, cfg);
    // Snapshot each active target's warm-start hint (its current `h`) in the
    // parallel map BEFORE any scatter, so no thread reads a half-written `h`.
    // `solve_one` is a pure function of positions + the per-target hint (no
    // cross-target coupling), so the scattered values are bit-identical to the
    // full pass at these indices regardless of iteration order.
    let updates: Vec<(usize, (f64, f64))> = active
        .par_iter()
        .map(|&i| (i, setup.solve_one(i, Some(h[i]))))
        .collect();
    for (i, (r, hh)) in updates {
        rho[i] = r;
        h[i] = hh;
    }
}

/// Shared setup for adaptive-h density: the query grid and per-particle bracket
/// seeds computed over ALL particles, plus the borrowed inputs. A per-target
/// solve ([`solve_one`](Self::solve_one)) restricted to any subset produces
/// **bit-identical** `(rho, h)` to the full pass — the grid and seeds do not
/// depend on which targets are solved, and the solve reads only positions + the
/// per-target hint. Both [`density_adaptive`] (over `0..n`) and
/// [`density_adaptive_active`] (over the active subset) run through it: one source
/// of truth for the density arithmetic, so the two paths cannot drift.
struct DensitySetup<'a> {
    pos: &'a [DVec3],
    mass: &'a [f64],
    cfg: &'a DensityConfig,
    seeds: Vec<f64>,
    h_cap: f64,
    grid: HashGrid,
}

impl<'a> DensitySetup<'a> {
    fn new(pos: &'a [DVec3], mass: &'a [f64], cfg: &'a DensityConfig) -> Self {
        let n = pos.len();
        assert!(n > 0, "DensitySetup requires a non-empty cloud");
        assert!(
            cfg.n_ngb > 32.0 / 3.0,
            "n_ngb must exceed the self-term floor 32/3, got {}",
            cfg.n_ngb
        );

        // Global spacing estimate: fallback seed and the cap scale.
        let (mut lo_c, mut hi_c) = (pos[0], pos[0]);
        for p in pos {
            lo_c = lo_c.min(*p);
            hi_c = hi_c.max(*p);
        }
        let extent = hi_c - lo_c;
        let diag = extent.length();
        let vol = extent.x * extent.y * extent.z;
        let s_est = if vol > 0.0 {
            (vol / n as f64).cbrt()
        } else if diag > 0.0 {
            diag / (n as f64).cbrt()
        } else {
            1.0 // fully degenerate cloud: any h is as meaningless as another
        };
        // Uniform cloud at spacing s hits the target when (4π/3)(2h)³/s³ = n_ngb.
        let to_seed = |spacing: f64| spacing * (3.0 * cfg.n_ngb / (32.0 * PI)).cbrt();
        let h_seed = to_seed(s_est);
        // Beyond ~the cloud diagonal the count plateaus at (32/3)n: nothing past
        // this cap can change, so rootless solves clamp here (finite, documented).
        let h_cap = (64.0 * h_seed).max(4.0 * diag);

        // Per-particle bracket seeds from LOCAL bin occupancy: a global-spacing
        // seed is wildly wrong for centrally-concentrated clouds (every galaxy),
        // where it brackets dense-center particles with ~10³–10⁴-candidate balls
        // and turns the solve quadratic in practice. Bins of edge 4·s_est hold
        // ~64 points of a uniform cloud; occupancy `c` rescales the local spacing
        // by c^(−1/3). The seed only positions the bracket — the converged h is a
        // pure function of positions to the bisection tolerance (gated), so this
        // is a performance choice, not a physics one.
        let occ_bin = 4.0 * s_est;
        let occupancy = HashGrid::build(pos, occ_bin);
        let seeds: Vec<f64> = pos
            .iter()
            .map(|&p| {
                let c = occupancy.bin_len(p).max(1) as f64; // own bin: O(1), ≥ 1
                to_seed(occ_bin / c.cbrt()).min(h_cap)
            })
            .collect();
        // The query grid follows the MEDIAN seed so dense-region bins stay small;
        // outskirt queries just walk more (mostly empty) bins.
        let mut sorted = seeds.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let cell = (SUPPORT * sorted[n / 2]).min(SUPPORT * h_seed).max(1e-12);
        let grid = HashGrid::build(pos, cell);

        DensitySetup {
            pos,
            mass,
            cfg,
            seeds,
            h_cap,
            grid,
        }
    }

    /// Kernel-weighted neighbour count for target `i` at smoothing length `h`
    /// over a candidate superset (gathered at ≥ 2h).
    #[inline]
    fn count(&self, i: usize, h: f64, cand: &[usize]) -> f64 {
        let mut sum = 0.0;
        for &j in cand {
            sum += w((self.pos[i] - self.pos[j]).length(), h);
        }
        (4.0 * PI / 3.0) * (SUPPORT * h).powi(3) * sum
    }

    /// Solve target `i`'s adaptive `(rho, h)`. `hint` is the bracket warm-start
    /// (its previous converged `h`); `None` or a non-finite/≤0 hint falls back to
    /// the occupancy seed. Reads only positions + `hint`, so a subset solve is
    /// bit-identical to the full solve.
    fn solve_one(&self, i: usize, hint: Option<f64>) -> (f64, f64) {
        let pos = self.pos;
        let cfg = self.cfg;
        let grid = &self.grid;
        let h_cap = self.h_cap;

        let seed = hint
            .filter(|&x| x.is_finite() && x > 0.0)
            .unwrap_or(self.seeds[i]);
        // A tight initial bracket keeps the candidate ball small (its volume
        // grows as the cube of the bracket top); the expand/shrink loops below
        // recover from any seed misestimate, re-querying as they go.
        let mut lo = (seed / 2.0).min(h_cap);
        let mut hi = (seed * 2.0).min(h_cap);
        let mut cand = grid.neighbours_within(pos, pos[i], SUPPORT * hi);

        // Expand up until the target is bracketed or the cap says "no root".
        while self.count(i, hi, &cand) < cfg.n_ngb && hi < h_cap {
            hi = (hi * 2.0).min(h_cap);
            cand = grid.neighbours_within(pos, pos[i], SUPPORT * hi);
        }
        // Expand down; bounded halvings terminate even for coincident knots
        // (where the count's h→0 limit can sit above the target).
        let mut shrinks = 0;
        while self.count(i, lo, &cand) > cfg.n_ngb && shrinks < 60 {
            lo /= 2.0;
            shrinks += 1;
        }

        let h = if self.count(i, hi, &cand) < cfg.n_ngb {
            hi // rootless: clamped at the cap (or the shrink floor)
        } else if self.count(i, lo, &cand) > cfg.n_ngb {
            lo
        } else {
            // Invariant: N(lo) ≤ n_ngb ≤ N(hi); N monotone ⇒ bisection.
            while hi - lo > cfg.h_tol_rel * hi {
                let mid = 0.5 * (lo + hi);
                if self.count(i, mid, &cand) < cfg.n_ngb {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            0.5 * (lo + hi)
        };

        let mut rho = 0.0;
        for &j in &cand {
            rho += self.mass[j] * w((pos[i] - pos[j]).length(), h);
        }
        (rho, h)
    }
}

fn density_impl(
    pos: &[DVec3],
    mass: &[f64],
    cfg: &DensityConfig,
    h_init: Option<&[f64]>,
    parallel: bool,
) -> DensityResult {
    let n = pos.len();
    if n == 0 {
        return DensityResult {
            rho: Vec::new(),
            h: Vec::new(),
        };
    }
    let setup = DensitySetup::new(pos, mass, cfg);
    let solve = |i: usize| setup.solve_one(i, h_init.map(|hh| hh[i]));
    let pairs: Vec<(f64, f64)> = if parallel {
        (0..n).into_par_iter().map(solve).collect()
    } else {
        (0..n).map(solve).collect()
    };
    let (rho, h) = pairs.into_iter().unzip();
    DensityResult { rho, h }
}
