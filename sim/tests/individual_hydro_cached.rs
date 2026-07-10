//! Hydro-only TREE CACHING: the driver builds the gravity tree ONCE per base block
//! (wrapping `BarnesHut` in `TreeGravity`) and walks it STALE on each fine tick,
//! instead of rebuilding a fresh octree every fine tick (the ×256/block cost the
//! `hydro+gravity` M-validate finding pinned as the dominant term). Stars stay on rung
//! 0 — this does NOT fold gravity into the rungs (that is `hydro+gravity`); the ONLY
//! change from fresh `hydro-only` is tree freshness.
//!
//! GATE DESIGN (advisor-vetted 2026-07-10):
//!   * FRESH PATH UNTOUCHED: `cache_gravity_tree = false` stays byte-identical to before
//!     — the layer below (I3/I4a bit-identity gates) must not be perturbed. Guaranteed
//!     structurally (the walk stays `g.accelerations`); the I3/I4a suites prove it.
//!   * FALLBACK BIT-IDENTITY: with a NON-caching gravity solver (bare `BarnesHut`),
//!     `cache_gravity_tree = true` is bit-for-bit identical to fresh — `rebuild_gravity_cache`
//!     is a no-op and `gravity_active_cached` DEFAULTS to the full fresh walk. So ALL of
//!     the cached↔fresh divergence comes from `TreeGravity`'s stale far-cell COMs and
//!     nothing else (the machinery itself introduces zero error).
//!   * CACHED DIFFERS + CONVERGES (the load-bearing gate): with `TreeGravity` at a FIXED
//!     courant, `D(c) = ‖cached(c) − fresh(c)‖` isolates tree freshness alone (identical
//!     rung structure, `dt_base`, integrator — only the walk differs). `D(coarse)` must
//!     sit well ABOVE roundoff (else the cache is silently rebuilt every tick and the
//!     feature is a no-op — the accidental-fresh bug a convergence-only gate would miss),
//!     and `D(fine) < D(coarse)` (the stale-COM error is O(courant) and vanishes). Stars
//!     stay on rung 0 (`max_rung` not clamped) — the run is a real multi-rung reference.
//!   * VALIDATION: `subcycle_gravity ⇒ cache_gravity_tree` (subcycling walks the cache).

use galaxy_core::{DVec3, ForceSolver, Species, State, StaticBackground};
use galaxy_io::Header;
use galaxy_sim::{
    run_individual, IndividualConfig, IndividualSummary, SimError, SnapshotSink, ThermalArm,
};
use galaxy_solvers::sph::{max_stable_dt, DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::{BarnesHut, TreeGravity};

struct NullSink;
impl SnapshotSink for NullSink {
    fn emit(&mut self, _h: &Header, _s: &State) -> Result<(), SimError> {
        Ok(())
    }
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn ball(rng: &mut impl FnMut() -> f64, n: usize, radius: f64) -> Vec<DVec3> {
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    pos
}

/// A dense gas core (strong, spatially-varying gravity well ⇒ the tree's far-cell COMs
/// evolve within a base block, so freezing them measurably perturbs the trajectory)
/// surrounded by orbiting stars. Same testbed shape as the `hydro+gravity` gates so the
/// cached↔fresh comparison sees a realistic star/gas force field.
fn core_and_stars(seed: u64) -> State {
    core_and_stars_sized(seed, 400, 300)
}

/// `core_and_stars` with explicit particle counts (the timing A/B scales this up so the
/// octree build is non-trivial — at a few hundred bodies the build is too cheap for the
/// build-once-vs-per-tick lever to register).
fn core_and_stars_sized(seed: u64, n_gas: usize, n_star: usize) -> State {
    let mut rng = lcg(seed);
    // Scale the core radius with N^(1/3) so density (⇒ smoothing length, CFL, rung depth)
    // stays comparable to the 400-gas correctness gates. Without this, packing thousands
    // of bodies into r=0.1 makes h→tiny, the CFL→tiny, and 2^r_max fine ticks explode.
    let radius = 0.1 * (n_gas as f64 / 400.0).cbrt();
    let gas = ball(&mut rng, n_gas, radius);
    let n_gas = gas.len();
    let mut pos = gas;
    let mut vel = vec![DVec3::ZERO; n_gas];
    for _ in 0..n_star {
        let r = 0.12 + rng() * 1.38;
        let dir = {
            let v = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5);
            v / v.length().max(1e-9)
        };
        let p = dir * r;
        pos.push(p);
        vel.push(DVec3::new(-p.y, p.x, 0.0) * 0.3);
    }
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0 / n as f64; n]);
    for (i, kind) in s.kind.iter_mut().enumerate() {
        *kind = if i < n_gas {
            Species::Gas
        } else {
            Species::Collisionless
        };
    }
    s
}

fn params() -> HydroParams {
    HydroParams {
        eos: Eos::Isothermal { c_s: 0.3 },
        ..HydroParams::default()
    }
}

fn bh() -> BarnesHut {
    BarnesHut::new(1.0, 0.05, 0.5)
}

/// `hydro-only` config; `cache` toggles the tree caching (fresh vs stale-once/block).
/// `subcycle_gravity` is always OFF (stars stay on rung 0 — no gravity rung folding).
fn cfg(courant: f64, cache: bool, output_dt: f64, n_outputs: u64) -> IndividualConfig {
    IndividualConfig {
        courant,
        dt_base_cap: f64::INFINITY, // non-binding ⇒ rung structure is courant-invariant
        r_max: 14,
        n_limit: 14, // == r_max ⇒ limiter non-binding (pure CFL rungs)
        cache_gravity_tree: cache,
        subcycle_gravity: false, // hydro-only: NO gravitational rung folding
        grav_eta: 1.0,
        eos: ThermalArm::Isothermal,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0x91A7,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    }
}

/// Run one config to completion, returning the final state + summary.
fn run_with<G: ForceSolver>(
    solver: &mut GravitySph<G>,
    config: &IndividualConfig,
) -> (State, IndividualSummary) {
    let mut state = core_and_stars(7);
    let bg = StaticBackground;
    let summary = run_individual(&mut state, solver, &bg, config, &mut NullSink)
        .expect("hydro-only cached run must complete");
    (state, summary)
}

/// The fresh baseline: bare `BarnesHut`, no caching.
fn run_fresh(courant: f64, output_dt: f64, n_outputs: u64) -> (State, IndividualSummary) {
    let mut solver = GravitySph::new(bh(), params(), DensityConfig::default());
    run_with(&mut solver, &cfg(courant, false, output_dt, n_outputs))
}

/// The cached path: `BarnesHut` wrapped in `TreeGravity`, tree built once/block.
fn run_cached(courant: f64, output_dt: f64, n_outputs: u64) -> (State, IndividualSummary) {
    let mut solver = GravitySph::new(TreeGravity::new(bh()), params(), DensityConfig::default())
        .with_gravity_cache(true);
    run_with(&mut solver, &cfg(courant, true, output_dt, n_outputs))
}

fn max_pos_diff(a: &State, b: &State) -> f64 {
    a.pos
        .iter()
        .zip(&b.pos)
        .map(|(p, q)| (*p - *q).length())
        .fold(0.0_f64, f64::max)
}

// --------------------------------------------------------------------------
// FALLBACK BIT-IDENTITY — the cached FLAG on a non-caching solver reproduces fresh
// EXACTLY, so the machinery adds zero error; all divergence is TreeGravity's stale COMs.
// --------------------------------------------------------------------------

#[test]
fn cached_flag_on_noncaching_solver_is_bit_identical_to_fresh() {
    let (output_dt, n_outputs) = (0.25, 1);
    let (fresh, _) = run_fresh(0.3, output_dt, n_outputs);

    // Bare BarnesHut with cache_gravity_tree = true: rebuild_gravity_cache is a no-op and
    // gravity_active_cached defaults to the full fresh walk ⇒ must equal fresh bit-for-bit.
    let mut solver =
        GravitySph::new(bh(), params(), DensityConfig::default()).with_gravity_cache(true);
    let (fallback, _) = run_with(&mut solver, &cfg(0.3, true, output_dt, n_outputs));

    for (i, (p, q)) in fresh.pos.iter().zip(&fallback.pos).enumerate() {
        assert_eq!(
            p, q,
            "cached flag on a non-caching solver must be bit-identical to fresh (pos[{i}])"
        );
    }
    for (i, (p, q)) in fresh.vel.iter().zip(&fallback.vel).enumerate() {
        assert_eq!(
            p, q,
            "cached flag on a non-caching solver must be bit-identical to fresh (vel[{i}])"
        );
    }
}

// --------------------------------------------------------------------------
// CACHED DIFFERS + CONVERGES — the load-bearing gate. Same courant ⇒ same rungs, so D
// isolates tree freshness. D(coarse) ≫ roundoff (not accidentally fresh) and shrinks.
// --------------------------------------------------------------------------

#[test]
fn cached_differs_from_fresh_and_converges_as_courant_falls() {
    let (output_dt, n_outputs) = (0.3, 2);
    let d = |courant: f64| -> (f64, u32) {
        let (fresh, _) = run_fresh(courant, output_dt, n_outputs);
        let (cached, cs) = run_cached(courant, output_dt, n_outputs);
        (max_pos_diff(&fresh, &cached), cs.max_rung)
    };
    let (d_coarse, rung_coarse) = d(0.4);
    let (d_fine, _) = d(0.1);
    // Measured: D(0.4) ≈ 7.6e-2, D(0.1) ≈ 2.0e-2 — the floor has ~5 orders of margin,
    // and the gap shrinks ~3.8× as courant drops 4× (the stale-COM error is O(courant)).

    // NON-VACUOUS: the stale tree genuinely moves the trajectory. If the cache were
    // (accidentally) rebuilt every fine tick, cached ≡ fresh and this would be ~roundoff.
    assert!(
        d_coarse > 1e-6,
        "cached must differ measurably from fresh — a ~roundoff gap means the tree is being \
         rebuilt every tick (the feature is a no-op): D(0.4) = {d_coarse:.3e}"
    );
    // CONVERGES: the frozen far-cell COMs are stale by ≤ one base step ⇒ O(courant) error.
    assert!(
        d_fine < d_coarse,
        "halving courant must shrink the cached↔fresh gap (stale-COM error is O(courant)): \
         D(0.1) = {d_fine:.3e} !< D(0.4) = {d_coarse:.3e}"
    );
    // The reference genuinely exercised multiple rungs (not clamped at r_max), so it is a
    // meaningful multi-rung run and stars were NOT folded onto fine gravity rungs.
    assert!(
        rung_coarse < 14,
        "reference run must not be clamped at r_max (max_rung = {rung_coarse})"
    );
}

// --------------------------------------------------------------------------
// VALIDATION — subcycling (folding gravity into rungs) requires the cached tree.
// --------------------------------------------------------------------------

#[test]
fn subcycle_without_cache_is_rejected() {
    let mut solver = GravitySph::new(TreeGravity::new(bh()), params(), DensityConfig::default())
        .with_gravity_cache(true);
    let mut state = core_and_stars(7);
    let bg = StaticBackground;
    let mut c = cfg(0.2, false, 0.25, 1);
    c.subcycle_gravity = true; // fold gravity into rungs …
    c.cache_gravity_tree = false; // … but no cache to walk — must be rejected
    let err = run_individual(&mut state, &mut solver, &bg, &c, &mut NullSink);
    assert!(
        matches!(err, Err(SimError::Config(_))),
        "subcycle_gravity without cache_gravity_tree must be a Config error, got {err:?}"
    );
}

// --------------------------------------------------------------------------
// TIMING A/B (ignored) — the whole POINT is speed; the correctness gates prove
// not-wrong, not faster. Run manually:
//   cargo test -p galaxy-sim --test individual_hydro_cached --release timing -- --ignored --nocapture
// Scales the testbed up so the octree build is non-trivial (the build-once-vs-×256
// lever is invisible at a few hundred bodies). Prints wall time; asserts nothing
// (machine-dependent). At QUICK/small N the win may be modest or absent (tree build is
// cheap, parallelism under-occupies — recall the G6 QUICK slowdown); the lever grows
// with N and rung depth. The correctness of the two runs is already gated above.
#[test]
#[ignore = "timing A/B — run manually with --release --ignored --nocapture"]
fn timing_fresh_vs_cached() {
    use std::time::Instant;
    // ~7000 particles like the M-validate QUICK run (density held constant via the
    // N^(1/3) radius scaling in core_and_stars_sized), a real rung spread from the core.
    let (n_gas, n_star) = (3000, 4000);
    let (output_dt, n_outputs) = (0.2, 2);
    // Bound r_max so the fine-tick count 2^r_max stays sane (the dense core would else
    // clamp to r_max=14 ⇒ 16384 ticks/block). This CAPS the win (deeper rungs ⇒ more
    // redundant fresh rebuilds ⇒ larger cached advantage), so the real shipping speedup
    // at r_max=10 is at least this.
    let conf = |cache: bool| IndividualConfig {
        r_max: 7,
        ..cfg(0.25, cache, output_dt, n_outputs)
    };

    let time_run = |cache: bool| -> (f64, IndividualSummary) {
        let mut state = core_and_stars_sized(7, n_gas, n_star);
        let bg = StaticBackground;
        let t0 = Instant::now();
        let summary = if cache {
            let mut solver =
                GravitySph::new(TreeGravity::new(bh()), params(), DensityConfig::default())
                    .with_gravity_cache(true);
            run_individual(&mut state, &mut solver, &bg, &conf(true), &mut NullSink).unwrap()
        } else {
            let mut solver = GravitySph::new(bh(), params(), DensityConfig::default());
            run_individual(&mut state, &mut solver, &bg, &conf(false), &mut NullSink).unwrap()
        };
        (t0.elapsed().as_secs_f64(), summary)
    };

    let (fresh, s_fresh) = time_run(false);
    let (cached, s_cached) = time_run(true);

    eprintln!(
        "n=({n_gas} gas, {n_star} star), {n_outputs}×{output_dt} outputs, courant 0.25, r_max 7\n\
         fresh   hydro-only : {fresh:.2}s (blocks {}, max_rung {})\n\
         cached  hydro-only : {cached:.2}s (blocks {}, max_rung {})\n\
         speedup            : {:.2}×",
        s_fresh.run.steps,
        s_fresh.max_rung,
        s_cached.run.steps,
        s_cached.max_rung,
        fresh / cached,
    );
}

// --------------------------------------------------------------------------
// MECHANISM CAUSAL TEST (ignored) — does the stale cached tree drive the gas
// core into a DEEPER min-dt / deeper rungs than fresh, on a controlled testbed?
// This is the causal counterpart to the full-res post-hoc finding (cached full-res
// hit a 6.4× deeper min-dt, sustained + bulk, → 5.6× slower). Same IC, ONLY tree
// freshness differs; r_max=14 (un-capped, matching the shipped full-res run) so the
// finest-rung flooding CAN manifest (the timing A/B above caps r_max=7, which
// structurally suppresses it). Records the min stable dt at every output for BOTH
// paths. A cached series that sinks below fresh isolates caching as the driver;
// a NULL (both comparable) does NOT exonerate caching — the quiescent synthetic
// core may simply lack the merger's supersonic infall. Run manually:
//   cargo test -p galaxy-sim --test individual_hydro_cached --release mechanism -- --ignored --nocapture
struct MinDtSink {
    params: HydroParams,
    cfg: DensityConfig,
    dts: Vec<f64>,
}
impl SnapshotSink for MinDtSink {
    fn emit(&mut self, _h: &Header, s: &State) -> Result<(), SimError> {
        // c_cfl = 1.0: the raw stable bound (same convention as the rung-spread tool),
        // identical for both arms so the fresh↔cached comparison is apples-to-apples.
        self.dts
            .push(max_stable_dt(s, &self.params, &self.cfg, 1.0));
        Ok(())
    }
}

#[test]
#[ignore = "mechanism causal test — run with --release --ignored --nocapture"]
fn mechanism_fresh_vs_cached_rmax14() {
    use std::time::Instant;
    // Enough gas for a real self-gravitating core (density held ~constant via the
    // N^(1/3) radius scaling); gas starts at rest ⇒ it collapses under self-gravity
    // vs c_s=0.3 pressure over the horizon. r_max=14 (from cfg) is NOT capped here.
    let (n_gas, n_star) = (3000, 1000);
    let (courant, output_dt, n_outputs) = (0.25, 0.2, 6); // horizon t = 1.2

    let run = |cache: bool| -> (f64, Vec<f64>, IndividualSummary) {
        let mut state = core_and_stars_sized(7, n_gas, n_star);
        let bg = StaticBackground;
        let mut sink = MinDtSink {
            params: params(),
            cfg: DensityConfig::default(),
            dts: Vec::new(),
        };
        let conf = cfg(courant, cache, output_dt, n_outputs); // r_max = 14 (un-capped)
        let t0 = Instant::now();
        let summary = if cache {
            let mut solver =
                GravitySph::new(TreeGravity::new(bh()), params(), DensityConfig::default())
                    .with_gravity_cache(true);
            run_individual(&mut state, &mut solver, &bg, &conf, &mut sink).unwrap()
        } else {
            let mut solver = GravitySph::new(bh(), params(), DensityConfig::default());
            run_individual(&mut state, &mut solver, &bg, &conf, &mut sink).unwrap()
        };
        (t0.elapsed().as_secs_f64(), sink.dts, summary)
    };

    let (t_fresh, dt_fresh, s_fresh) = run(false);
    let (t_cached, dt_cached, s_cached) = run(true);

    let series_min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let (min_fresh, min_cached) = (series_min(&dt_fresh), series_min(&dt_cached));

    eprintln!(
        "MECHANISM fresh↔cached, n=({n_gas} gas, {n_star} star), r_max 14, courant {courant}, \
         {n_outputs}×{output_dt} outputs (t={:.1})",
        n_outputs as f64 * output_dt
    );
    eprintln!("  output  min_stable_dt(fresh)   min_stable_dt(cached)   cached/fresh");
    for k in 0..dt_fresh.len().min(dt_cached.len()) {
        eprintln!(
            "  {k:>5}   {:>18.4e}   {:>20.4e}   {:>10.3}",
            dt_fresh[k],
            dt_cached[k],
            dt_cached[k] / dt_fresh[k],
        );
    }
    eprintln!(
        "  series-min dt : fresh {min_fresh:.4e}   cached {min_cached:.4e}   \
         cached is {:.2}× {} than fresh",
        (min_fresh / min_cached).max(min_cached / min_fresh),
        if min_cached < min_fresh {
            "DEEPER (smaller dt)"
        } else {
            "shallower"
        },
    );
    eprintln!(
        "  max_rung : fresh {}  cached {}   |   blocks : fresh {}  cached {}   |   \
         wall : fresh {t_fresh:.2}s  cached {t_cached:.2}s ({:.2}× fresh)",
        s_fresh.max_rung,
        s_cached.max_rung,
        s_fresh.run.steps,
        s_cached.run.steps,
        t_cached / t_fresh,
    );
}
