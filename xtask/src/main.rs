//! Movie orchestrator: builds a two-galaxy collision, steps it to snapshots, then
//! renderprep → render → grade → ffmpeg into a movie. Scenarios are **data**
//! (M6f): checked-in `scenario.toml` presets under `xtask/scenarios/` — the
//! originals (`disk`, `dm`, `cuspy`, gated to reproduce the pre-M6f hardcoded
//! constructors bit-for-bit) plus the Toomre encounter zoo (`retro`, `inclined`,
//! `bullseye`, `minor`) — or any user toml on the same schema (see `spec`).
//!
//! Usage: `cargo run -p galaxy-xtask --release [<preset>|<scenario.toml>] [out_dir]
//! [--color progenitor|initial-radius|dispersion] [--reuse-snapshots] [--gpu]`
//! (`--gpu` runs the gas-rich sim on the GPU-resident SPH stepper, G6.)
//!   * A bare first arg that is no preset name (and not a `.toml` path) is taken as
//!     `out_dir` with the `disk` scenario (back-compat with the original CLI).
//!   * `regrade <exr_dir> <png_dir> [--exposure E] [--tonemap aces|reinhard|asinh]
//!     [--beta B] [--bloom S] [--bloom-levels N] [--bloom-radius R]
//!     [--local S] [--local-radius R] [--local-floor F]
//!     [--black-point B] [--white-point W] [--gamma G]` re-grades
//!     retained linear EXRs into fresh PNGs (+ movie if ffmpeg is present) in seconds
//!     — no re-simulation, no re-render (the M6a look loop; bloom added in M6b).
//!   * `sph-demo <snapshot.snap> [--k N] [--n-ngb X]` runs the M7a SPH density
//!     estimator over a retained snapshot and prints the O(N) win (grid gather
//!     vs the O(N²) brute reference, swept over prefixes), the bit-exact match
//!     between them, and the Spearman agreement of the SPH field with the M6
//!     k-NN coloring — no sim, no GPU (the M7a density demo).
//!   * `gas-demo [out_dir] [--n N] [--res R] [--seed S] [--incline DEG]` (M7d)
//!     voxelizes a static synthetic sech² gas disk onto the renderprep density
//!     grid, round-trips frame-data v2 (gas block) through disk, and grades a
//!     3-panel column-density contact sheet — the dust-lane preview without a
//!     raymarcher; no sim, no GPU.
//!   * `volume-demo [out_dir] [--n N] [--stars N] [--res R] [--seed S]
//!     [--incline DEG] [--kappa K] [--emissivity J] [--exposure E]` (M7e)
//!     renders the volumetric composite on static synthetic data: raymarched
//!     gas (emission + absorption) over a star field with per-star
//!     transmittance, plus the attenuation-off A/B twin and a stars-only
//!     reference — no sim.
//!   * `temp-demo [out_dir] [--n N] [--stars N] [--res R] [--seed S]
//!     [--incline DEG] [--kappa K] [--emissivity J] [--exposure E]
//!     [--u-cold U] [--u-hot U] [--r-core R] [--u-lo U] [--u-hi U]
//!     [--cold R,G,B] [--hot R,G,B]` (H5-B) paints a synthetic hot-core /
//!     cold-outskirts internal-energy field on the volume-demo gas and renders
//!     it through the temperature colormap (ū = N/ρ → cold→hot lerp over a fixed
//!     `[u_lo, u_hi]` band), plus a flat-tint A/B control and a stars-only
//!     reference — the offline colormap look-dev before the adiabatic sim; no sim.
//!   * Set `GALAXY_MOVIE_QUICK=1` for a fast low-N, low-res preview (same physical
//!     time and dt, so the trajectory is faithful — only particle count, frame size,
//!     and frame cadence are reduced). Use it to sanity-check a scenario before a
//!     full-resolution render.
//!
//! Layout under `out_dir`: `snapshots/` `.snap`, `exr/` linear HDR, `frames/` PNGs,
//! `movie.mp4` (if ffmpeg is on PATH). The EXR layer is kept so the frames can be
//! regraded (different exposure/tonemap) without re-simulating or re-rendering.
//!
//! Motion (M6c): frames are Hermite-upsampled between snapshots — full renderprep
//! (incl. kNN density) on the snapshot cadence, 8 physically-informed in-betweens
//! per interval, 60 fps — so ~61 snapshots become a ~8 s continuous movie.
//!
//! Camera (M6d): each scenario picks a `Rig` — `Static` (the pre-M6d face-on
//! framing, bit-exact) or `OrbitTilt` (eased azimuth/tilt sweep with the zoom
//! breathing along a smoothed per-snapshot framing envelope).
//!
//! Coloring (M6e): `--color` picks what the colors *mean* — the progenitor palette
//! (default), a frozen initial-radius ramp (per-progenitor provenance gradient,
//! computed once from snapshot 0), or a per-frame σ_v ramp. Independently, every
//! scenario keys a hue shift toward blue-white on density *compression* vs each
//! particle's t=0 neighbourhood (the star-formation proxy — a visualization
//! stand-in, the sim is collisionless) and drives splat size off the same density
//! estimate (tight cores, soft diffuse splats). `--reuse-snapshots` re-preps and
//! re-renders retained snapshots without re-simulating (color modes iterate in
//! render time, not sim time).

use std::path::{Path, PathBuf};
use std::process::Command;

use galaxy_core::{DVec3, Species, State};
use galaxy_grade::{grade_file, BloomConfig, GradeConfig, ToneMap};
use galaxy_render::camera::DEFAULT_MARGIN;
use galaxy_render::{smooth_envelope, write_exr, Camera, CameraPath, RenderConfig, Renderer};
use galaxy_renderprep::{
    initial_radius_colors, knn_density, prepare, subframe, ColorMode, CompressionHue,
    DispersionColoring, FrameData, HermiteSpan, PrepConfig, RadialRamp,
};
use galaxy_solvers::sph::{
    density_adaptive, density_fixed, max_stable_dt, reference_density, DensityConfig, Eos,
    HashGrid, HydroParams, SUPPORT,
};
use galaxy_xtask::simulate::simulate_snapshots;
use galaxy_xtask::spec::{
    build_scenario, parse_scenario_toml, preset, Rig, Scenario, ScenarioSpec,
};
use galaxy_xtask::{
    accel_max_rel_vs_direct, framing_radius, parse_movie_args, parse_regrade_args, per_frame_radii,
    per_particle_grav_dt, rung_spread, ColorModeArg, ScenarioArg, AMDAHL_GASRICH_PERICENTER,
    DEFAULT_BLOOM_LEVELS, DEFAULT_BLOOM_RADIUS, DENSITY_K, THETA, W_HYDRO_DROP_FINEST,
};
use glam::Vec3;

// --- Shared render / grade look (all scenarios) --------------------------------
// The shared physics/look constants (G, kNN density tuning, splat-size clamps,
// frame sizes, subframe count) live in the lib (`galaxy_xtask`) so the M6f
// spec-driven builder shares them; the grade-side and mode-color knobs below are
// consumed only by this binary. Tuning provenance: DESIGN.md M3.6/M6a–M6e.
const FALLOFF: f32 = 6.0;
// M6e coloring. All kNN consumers reuse (DENSITY_K, scenario ε) so the O(N²)
// estimate runs once per snapshot no matter how many passes are on.
//   * Star-formation proxy (ON in every scenario): hue shift toward a young-
//     population blue-white, keyed on density compression ρ(t)/ρ(0) — only
//     tidally-compressed material lights up; undisturbed cores keep their color.
//     (A proxy: the sim is collisionless — see DESIGN M6e.)
//   * Size-by-density (ON): splat radius follows the local spacing (ρ_ref/ρ)^⅓,
//     clamped — tight cores, soft diffuse splats.
//   * σ_v ramp (--color dispersion): dynamically cold → warm, hot → blue-white
//     (the blackbody convention, matching the temperature-colored gas: red is
//     cold, blue is hot across the whole frame). The ramp is masked to the
//     scenario's luminous progenitors (`sf_progenitors`); the dark-matter halo
//     keeps its dim palette color — otherwise the ~5×-heavier halo particles,
//     which are dynamically HOT, take the bright hot end with no compensation
//     and swamp the frame center (the first disk-scene A/B blew out white). So
//     on a disk+halo scene the halo renders exactly as in progenitor mode and
//     only the disk stars carry the temperature ramp.
const SF_YOUNG: [f32; 3] = [0.7, 0.8, 1.0];
const SF_STRENGTH: f32 = 0.8;
const DISPERSION_COLD: [f32; 3] = [0.25, 0.4, 1.0];
const DISPERSION_HOT: [f32; 3] = [1.0, 0.5, 0.2];
const EXPOSURE: f32 = 1.0;
const TONEMAP: ToneMap = ToneMap::AcesApprox;
// Bloom (M6b), ON by default in all three scenarios. Strength tuned by A/B regrades
// of retained QUICK EXRs (0 / 0.3 / 0.45 / 0.6 / 1.2, cuspy under asinh exposure 4 +
// disk/dm under the ACES movie default): 0.3 is timid, 0.6 starts to haze the dense
// cuspy halo field, 1.2 washes out structure; 0.45 makes nuclei and knots glow while
// tails and halo dots stay resolved. Levels/radius are the documented CLI defaults.
const BLOOM_STRENGTH: f32 = 0.45;
const FPS: u32 = 60;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // `regrade` is a pure grade-stage pass over retained EXRs — no sim, no GPU.
    if args.first().map(String::as_str) == Some("regrade") {
        return regrade(&args[1..]);
    }

    // `sph-demo` is the M7a demo: run the grid-accelerated SPH density estimator
    // over a retained snapshot and prove the O(N) win against the O(N²) brute
    // reference — no sim, no GPU, no movie pipeline.
    if args.first().map(String::as_str) == Some("sph-demo") {
        return sph_demo(&args[1..]);
    }

    // `gas-demo` is the M7d demo: voxelize a static synthetic gas disk, round-
    // trip frame-data v2, and grade a column-density contact sheet — no sim.
    if args.first().map(String::as_str) == Some("gas-demo") {
        return gas_demo(&args[1..]);
    }

    // `volume-demo` is the M7e demo: the volumetric composite (raymarched gas +
    // per-star attenuation) on static synthetic data — no sim.
    if args.first().map(String::as_str) == Some("volume-demo") {
        return volume_demo(&args[1..]);
    }

    // `temp-demo` (H5-B) is the temperature-color look-dev demo: a synthetic
    // hot-core/cold-outskirts internal-energy field over the volume-demo scene,
    // rendered through the H1-H4 temperature colormap plus a flat-tint A/B twin —
    // proves the moment→ū→color path end-to-end and tunes the band offline; no sim.
    if args.first().map(String::as_str) == Some("temp-demo") {
        return temp_demo(&args[1..]);
    }

    // `rung-spread` is the I0 go/no-go measurement (docs/plans/laddered-ember-
    // cadence.md): histogram the per-instant per-particle gas CFL timesteps and
    // report the ideal-ceiling individual-timestep speedup — no sim, no render.
    if args.first().map(String::as_str) == Some("rung-spread") {
        return rung_spread_cmd(&args[1..]);
    }

    // `grav-rung-spread` is I0b (docs/plans/laddered-ember-cadence.md): the
    // GRAVITATIONAL analogue of rung-spread — histogram the per-instant per-particle
    // gravitational timesteps dt = eta·√(eps/|a|) and report the STAR-subset walk
    // factor lever (b)'s ~2.24× hangs on (I0 measured only gas CFL). No sim, no render.
    if args.first().map(String::as_str) == Some("grav-rung-spread") {
        return grav_rung_spread_cmd(&args[1..]);
    }

    let quick = std::env::var_os("GALAXY_MOVIE_QUICK").is_some();

    let movie = parse_movie_args(&args).map_err(|e| {
        format!(
            "{e}\nusage: [<preset>|<scenario.toml>] [out_dir] \
             [--color progenitor|initial-radius|dispersion] [--reuse-snapshots] [--gpu]"
        )
    })?;
    let spec: ScenarioSpec = match &movie.scenario {
        ScenarioArg::Preset(name) => {
            let text = preset(name).ok_or_else(|| format!("preset `{name}` missing"))?;
            parse_scenario_toml(text).map_err(|e| format!("preset `{name}`: {e}"))?
        }
        ScenarioArg::Path(path) => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            parse_scenario_toml(&text).map_err(|e| format!("{}: {e}", path.display()))?
        }
    };
    let out: PathBuf = movie.out_dir.clone().unwrap_or_else(|| {
        // The three original scenarios keep their pre-M6f default dirs (retained
        // snapshot/EXR workflows point at them); new ones derive from the name.
        std::env::temp_dir().join(match spec.name.as_str() {
            "dm" => "galaxy_dm_merger".to_string(),
            "cuspy" => "galaxy_cuspy_disk".to_string(),
            "disk" => "galaxy_movie".to_string(),
            other => format!("galaxy_{other}"),
        })
    });

    println!(
        "scenario = {} (color: {:?}){}",
        spec.name,
        movie.color,
        if quick { " (quick preview)" } else { "" }
    );
    println!("output → {}", out.display());

    let scenario = build_scenario(&spec, quick);
    println!("{}", scenario.info);
    run_movie(
        &scenario,
        &out,
        movie.color,
        movie.reuse_snapshots,
        movie.backend,
    )
}

/// The M6a look loop: re-grade a directory of retained linear-HDR EXRs into PNGs
/// under a new exposure/tone curve, then (optionally) ffmpeg them into a movie next
/// to the frames. Seconds instead of a re-render, because the EXR is the pristine
/// linear artifact. The movie step assumes the pipeline's `frame_%05d` stems; other
/// stems still regrade fine, ffmpeg just skips them with its usual message.
fn regrade(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: regrade <exr_dir> <png_dir> \
         [--exposure E] [--tonemap aces|reinhard|asinh] [--beta B] \
         [--bloom S] [--bloom-levels N] [--bloom-radius R] \
         [--local S] [--local-radius R] [--local-floor F] \
         [--black-point B] [--white-point W] [--gamma G]";
    let cfg = parse_regrade_args(args).map_err(|e| format!("regrade: {e}\n{USAGE}"))?;

    let mut exrs: Vec<PathBuf> = std::fs::read_dir(&cfg.exr_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "exr"))
        .collect();
    exrs.sort(); // frame_<i:05>.exr → lexicographic == frame order
    if exrs.is_empty() {
        return Err(format!("no .exr frames in {}", cfg.exr_dir.display()).into());
    }

    std::fs::create_dir_all(&cfg.png_dir)?;
    for exr in &exrs {
        // The extension filter above guarantees a stem exists.
        let Some(stem) = exr.file_stem() else {
            continue;
        };
        let png = cfg.png_dir.join(stem).with_extension("png");
        grade_file(exr, &png, &cfg.grade)?;
    }
    println!(
        "regraded {} frames ({:?}) → {}",
        exrs.len(),
        cfg.grade,
        cfg.png_dir.display()
    );
    encode_movie(&cfg.png_dir, &cfg.png_dir.join("movie.mp4"));
    Ok(())
}

/// The M7a demo (density side, per docs/plans/deep-orbiting-sunbeam.md): run the
/// SPH adaptive-h density estimator over a retained snapshot and prove the
/// grid-accelerated path is bit-identical to — but O(N) faster than — the O(N²)
/// brute reference. Three numbers carry the demo:
///
///   1. **The O(N) win.** With a fixed smoothing length (the one the adaptive
///      pass converged to), the grid gather [`density_fixed`] and the brute sum
///      [`reference_density`] compute the *same thing* two ways. Timing them
///      side by side isolates the data-structure speedup — no algorithm
///      difference to muddy the ratio.
///   2. **Provably correct, not just fast.** Those two paths gather in ascending
///      index and add exact `+0.0` for out-of-support terms, so they agree to
///      the bit — the max relative difference is the M7a bit-exact gate re-run
///      at snapshot scale, and it must print `0`.
///   3. **The SPH field vs the existing coloring.** The M6 star coloring keys on
///      an O(N²) k-NN density; the SPH field is a different estimator, so we
///      report the Spearman rank correlation between the two (log space) — how
///      closely a density *coloring* off the SPH field would track today's.
///
/// Usage: `sph-demo <snapshot.snap> [--k N] [--n-ngb X]`.
fn sph_demo(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    const USAGE: &str = "usage: sph-demo <snapshot.snap> [--k N] [--n-ngb X]";
    let mut path: Option<PathBuf> = None;
    let mut k = DENSITY_K;
    let mut n_ngb = DensityConfig::default().n_ngb;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--k" => {
                k = it
                    .next()
                    .ok_or("--k needs a value")?
                    .parse()
                    .map_err(|_| "--k must be a positive integer")?;
            }
            "--n-ngb" => {
                n_ngb = it
                    .next()
                    .ok_or("--n-ngb needs a value")?
                    .parse()
                    .map_err(|_| "--n-ngb must be a number")?;
            }
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument `{other}`\n{USAGE}").into()),
        }
    }
    let path = path.ok_or_else(|| format!("no snapshot given\n{USAGE}"))?;

    let (header, state) = galaxy_io::read_file(&path)?;
    let n = state.len();
    let n_gas = state
        .kind
        .iter()
        .filter(|k| matches!(k, galaxy_core::Species::Gas))
        .count();
    println!(
        "snapshot   {} (written by {})",
        path.display(),
        header.code_version
    );
    println!(
        "N          {n} particles ({n_gas} gas, {} collisionless)",
        n - n_gas
    );
    println!(
        "estimator  SPH cubic-spline, adaptive h @ N_ngb = {n_ngb} (mass density of the point set)"
    );
    if n < 2 {
        return Err("need at least two particles for a density demo".into());
    }

    // (1) Full adaptive-h SPH density via the grid path — the real per-snapshot
    //     cost renderprep will pay, including the per-particle h bisection.
    let cfg = DensityConfig {
        n_ngb,
        ..DensityConfig::default()
    };
    let t0 = Instant::now();
    let sph = density_adaptive(&state.pos, &state.mass, &cfg, None);
    let t_adaptive = t0.elapsed();

    // (2) Same fixed h, two data structures — isolates the O(N) win, exact match.
    //     Swept over prefixes of the snapshot so the *scaling* is visible: the
    //     brute column ~quadruples per doubling of N, the grid column ~doubles.
    let mut sweep: Vec<DensityTiming> = Vec::new();
    for frac in [4usize, 2, 1] {
        let m = n / frac;
        if m >= 2 {
            sweep.push(time_density(&state.pos[..m], &state.mass[..m], &cfg));
        }
    }
    let max_rel = sweep.last().expect("full set has ≥2 particles").max_rel;

    // (3) Agreement with the existing O(N²) k-NN coloring (a different estimator):
    //     rank correlation in log space — a density coloring off the SPH field
    //     would order particles this closely to today's.
    let t0 = Instant::now();
    let knn = knn_density(&state.pos, k, header.softening.max(1e-6));
    let t_knn = t0.elapsed();
    let rho_sph = &sph.rho;
    let rank_corr = spearman_log(rho_sph, &knn);

    let (lo, med, hi) = min_median_max(rho_sph);
    println!();
    println!("SPH density field   min {lo:.3e}  median {med:.3e}  max {hi:.3e}");
    println!();
    println!("--- the O(N) win (fixed h, identical output, different structure) ---");
    println!("        N     grid O(N)     brute O(N²)    speedup   max rel diff");
    for t in &sweep {
        println!(
            "  {:>7}   {:>8.1} ms    {:>9.1} ms    {:>5.1}×    {:.1e}",
            t.n,
            ms(t.t_grid),
            ms(t.t_brute),
            t.t_brute.as_secs_f64() / t.t_grid.as_secs_f64().max(f64::MIN_POSITIVE),
            t.max_rel,
        );
    }
    println!(
        "  (brute ~quadruples per doubling of N; grid ~doubles — the O(N) win. \
         max rel diff is the bit-exact gate: must be ~0.)"
    );
    println!();
    println!("--- full SPH pass and the k-NN coloring reference ---");
    println!(
        "  adaptive-h SPH density (grid, incl. h bisection)  {:>8.1} ms",
        ms(t_adaptive)
    );
    println!(
        "  k-NN density (k={k}, the M6 O(N²) color oracle)   {:>8.1} ms",
        ms(t_knn)
    );
    println!("  Spearman(log ρ_sph, log ρ_knn) = {rank_corr:.4}   (coloring agreement)");

    if max_rel > 1e-12 {
        return Err(format!(
            "grid and brute SPH density disagree (max rel {max_rel:.2e}) — the M7a \
             bit-exact invariant is broken"
        )
        .into());
    }
    Ok(())
}

/// The M7d demo (per the view-first amendment in docs/plans/deep-orbiting-
/// sunbeam.md): voxelize a STATIC synthetic gas disk — no gas dynamics exists
/// yet (M7b/M7c follow) — and show the dust-lane preview without a raymarcher:
///
///   1. Hand-roll an inclined sech²-thickness exponential gas disk
///      (`Species::Gas`, progenitor 4) from a deterministic seed.
///   2. `deposit_gas`: shared adaptive-h + kernel deposition onto the default
///      grid, timed; report the grid-mass capture (the conservation gate at
///      demo scale) and the chosen bounds.
///   3. Round-trip frame-data v2 with the gas block through disk, bit-exact.
///   4. Render the ρ-integral along each axis (column density) into a 3-panel
///      contact sheet, EXR → PNG via the existing grade path.
///
/// Usage: `gas-demo [out_dir] [--n N] [--res R] [--seed S] [--incline DEG]`.
fn gas_demo(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use galaxy_renderprep::{deposit_gas, frame, GasGridConfig};
    use std::time::Instant;

    const USAGE: &str = "usage: gas-demo [out_dir] [--n N] [--res R] [--seed S] [--incline DEG]";
    let mut out: Option<PathBuf> = None;
    let mut n: usize = 40_000;
    let mut res: u32 = 128;
    let mut seed: u64 = 42;
    let mut incline_deg: f64 = 60.0;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut num = |flag: &str| -> Result<String, String> {
            it.next().cloned().ok_or(format!("{flag} needs a value"))
        };
        match a.as_str() {
            "--n" => n = num("--n")?.parse().map_err(|_| "--n must be an integer")?,
            "--res" => {
                res = num("--res")?
                    .parse()
                    .map_err(|_| "--res must be an integer")?
            }
            "--seed" => {
                seed = num("--seed")?
                    .parse()
                    .map_err(|_| "--seed must be an integer")?
            }
            "--incline" => {
                incline_deg = num("--incline")?
                    .parse()
                    .map_err(|_| "--incline must be a number")?
            }
            other if out.is_none() => out = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument `{other}`\n{USAGE}").into()),
        }
    }
    let out = out.unwrap_or_else(|| std::env::temp_dir().join("galaxy_gas_demo"));
    std::fs::create_dir_all(&out)?;

    // 1. Static synthetic gas: inclined exponential disk with sech² thickness.
    let state = synthetic_gas_disk(n, seed, incline_deg.to_radians());
    let m_gas: f64 = state.mass.iter().sum();
    println!(
        "synthetic gas disk: {n} particles, incline {incline_deg}°, seed {seed} (total mass {m_gas})"
    );

    // 2. Voxelize (the real per-snapshot cost run_movie will pay in M7e).
    let cfg = GasGridConfig {
        dims: [res; 3],
        ..Default::default()
    };
    let t0 = Instant::now();
    let grid = deposit_gas(&state, &cfg).expect("all particles are gas");
    let t_dep = t0.elapsed();
    let cell = grid.cell_size();
    let vol = cell.x * cell.y * cell.z;
    let total: f64 = grid.data.iter().map(|&d| d as f64 * vol).sum();
    let peak = grid.data.iter().fold(0.0_f32, |a, &b| a.max(b));
    println!(
        "deposited {res}³ grid in {:.1} ms  (bounds {:.2}..{:.2}, cell {:.3})",
        ms(t_dep),
        grid.bounds_min.x,
        grid.bounds_max.x,
        cell.x,
    );
    println!(
        "grid mass ∫ρ dV = {total:.4} of {m_gas} deposited ({:.2}% captured); peak ρ {peak:.3}",
        100.0 * total / m_gas
    );

    // 3. Frame-data v2 round-trip on disk: the default prepare routes gas out
    //    of the splat list (empty stars), the grid rides the gas block.
    let prep_default = PrepConfig::default();
    let stars = prepare(&state, &prep_default);
    let header = galaxy_renderprep::FrameHeader::for_data(&stars, state.time);
    let frame_path = out.join("gas_frame.bin");
    frame::write_file(&frame_path, &header, &stars, Some(&grid))?;
    let (_, back_stars, back_gas) = frame::read_file(&frame_path)?;
    if back_stars != stars || back_gas.as_ref() != Some(&grid) {
        return Err("frame-data v2 round-trip is not bit-exact".into());
    }
    println!(
        "frame v2 round-trip OK: {} ({} splats + {res}³ gas block, {:.1} MB)",
        frame_path.display(),
        stars.len(),
        std::fs::metadata(&frame_path)?.len() as f64 / (1024.0 * 1024.0),
    );

    // 4. Contact sheet of axis-aligned column densities (∫ρ ds per pixel).
    let img = column_density_sheet(&grid);
    let exr = out.join("gas_slices.exr");
    let png = out.join("gas_slices.png");
    write_exr(&exr, &img)?;
    grade_file(
        &exr,
        &png,
        &GradeConfig {
            exposure: 1.0,
            tonemap: ToneMap::AcesApprox,
            bloom: None,
            ..GradeConfig::default()
        },
    )?;
    println!("contact sheet (∫ρ dz | ∫ρ dy | ∫ρ dx) → {}", png.display());
    Ok(())
}

/// A splitmix64 → uniform-[0,1) stream (the synthetic-demo PRNG).
fn splitmix_stream(seed: u64) -> impl FnMut() -> f64 {
    let mut s = seed;
    move || -> f64 {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z = z ^ (z >> 31);
        (z >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Positions of an inclined exponential disk (scale radius 1, truncated at 5)
/// with sech² vertical structure of scale height `z0`: radii by inverse-CDF
/// bisection of Σ ∝ e^(−R), z by the exact sech² inverse CDF z₀·atanh(2u−1),
/// the whole disk rotated about x by `incline`. Deterministic in `seed`; the
/// draw order (radius, azimuth, height) is fixed — the gas-demo look rides on
/// it. Shared by the M7d gas disk and the M7e star field.
fn synthetic_disk_positions(n: usize, seed: u64, incline: f64, z0: f64) -> Vec<galaxy_core::DVec3> {
    let mut rand = splitmix_stream(seed);
    let rmax = 5.0;
    // Exponential-disk enclosed-mass CDF, normalized to the truncation radius.
    let cdf = |x: f64| 1.0 - (1.0 + x) * (-x).exp();
    let norm = cdf(rmax);
    let (sin_i, cos_i) = incline.sin_cos();

    let mut pos = Vec::with_capacity(n);
    for _ in 0..n {
        // Invert the radial CDF by bisection (monotone, deterministic).
        let target = rand() * norm;
        let (mut lo, mut hi) = (0.0, rmax);
        while hi - lo > 1e-12 * rmax {
            let mid = 0.5 * (lo + hi);
            if cdf(mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let r = 0.5 * (lo + hi);
        let phi = rand() * std::f64::consts::TAU;
        let u = rand().clamp(1e-9, 1.0 - 1e-9);
        let z = z0 * (2.0 * u - 1.0).atanh();
        let (x, y) = (r * phi.cos(), r * phi.sin());
        // Incline about the x axis.
        pos.push(galaxy_core::DVec3::new(
            x,
            y * cos_i - z * sin_i,
            y * sin_i + z * cos_i,
        ));
    }
    pos
}

/// An inclined exponential gas disk with sech² vertical structure — STATIC
/// synthetic positions only (no dynamics: M7b/M7c land the physics).
/// Deterministic in `seed` (splitmix64, one stream).
fn synthetic_gas_disk(n: usize, seed: u64, incline: f64) -> State {
    let pos = synthetic_disk_positions(n, seed, incline, 0.12);
    State {
        vel: vec![galaxy_core::DVec3::ZERO; n],
        mass: vec![1.0 / n as f64; n],
        id: (0..n as u64).map(galaxy_core::ParticleId).collect(),
        progenitor: vec![galaxy_core::Progenitor(4); n], // gas1 tag (plan D1)
        kind: vec![galaxy_core::Species::Gas; n],
        u: vec![0.0; n],
        time: 0.0,
        a: 1.0,
        pos,
    }
}

/// A synthetic stellar scene sharing the gas disk's geometry (M7e demo): a
/// stellar disk 2.5× thicker than the gas (the lane cuts through the bright
/// body, the classic edge-on look) colored warm-core → blue-outskirts by
/// radius, plus a warm Plummer bulge for the dust lane to silhouette against.
/// Deterministic in `seed`; look values are demo-only (eyeballed, not gated).
fn synthetic_star_frame(n: usize, seed: u64, incline: f64) -> FrameData {
    let n_disk = n * 3 / 4;
    let n_bulge = n - n_disk;

    let mut pos: Vec<Vec3> = synthetic_disk_positions(n_disk, seed ^ 0x5354_4152, incline, 0.30)
        .into_iter()
        .map(|p| p.as_vec3())
        .collect();
    let mut color = Vec::with_capacity(n);
    let mut size = vec![0.02_f32; n_disk];
    let mut brightness = vec![1.0_f32; n_disk];
    for p in &pos {
        // Warm inner disk → blue outskirts (a young-population gradient).
        let t = (p.length() / 4.0).clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| (1.0 - t) * a + t * b;
        color.push([lerp(1.0, 0.62), lerp(0.88, 0.74), lerp(0.70, 1.0)]);
    }

    // Plummer bulge (a = 0.35, truncated at 5a): M(<r)/M = r³/(r²+a²)^{3/2},
    // inverted in closed form r = a/√(m^{-2/3} − 1).
    let mut rand = splitmix_stream(seed ^ 0x4255_4C47);
    let a = 0.35_f64;
    let m_trunc = 125.0 / 26.0_f64.powf(1.5); // enclosed mass fraction at 5a
    for _ in 0..n_bulge {
        let m = (rand() * m_trunc).clamp(1e-9, 1.0 - 1e-9);
        let r = a / (m.powf(-2.0 / 3.0) - 1.0).sqrt();
        let z = 2.0 * rand() - 1.0;
        let phi = rand() * std::f64::consts::TAU;
        let s = (1.0 - z * z).sqrt();
        pos.push(galaxy_core::DVec3::new(r * s * phi.cos(), r * s * phi.sin(), r * z).as_vec3());
        color.push([1.0, 0.86, 0.64]);
        size.push(0.022);
        brightness.push(1.3);
    }

    FrameData {
        pos,
        color,
        size,
        brightness,
    }
}

/// The M7e demo (per the view-first amendment in docs/plans/deep-orbiting-
/// sunbeam.md): the volumetric composite on STATIC synthetic data — no gas
/// dynamics exists yet (M7b/M7c follow), so the dust-lane geometry is the
/// inclined sech² disk from `gas-demo` over a synthetic star field:
///
///   1. Synthetic gas disk → `deposit_gas` (timed — the per-snapshot prep cost).
///   2. Synthetic stellar disk + bulge in the same geometry ([`synthetic_star_frame`]).
///   3. Three renders through the volumetric path (timed): the full composite
///      (gas emission + absorption + per-star attenuation), the same with
///      κ = 0 (attenuation OFF — the A/B pair for DESIGN), and stars-only.
///   4. EXR → PNG via the existing grade path (bloom applies to the composite
///      for free — the D9 selling point).
///
/// Usage: `volume-demo [out_dir] [--n N] [--stars N] [--res R] [--seed S]
///         [--incline DEG] [--kappa K] [--emissivity J] [--exposure E]`.
fn volume_demo(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use galaxy_render::{GasFrame, GasLook};
    use galaxy_renderprep::{deposit_gas, GasGridConfig};
    use std::time::Instant;

    const USAGE: &str = "usage: volume-demo [out_dir] [--n N] [--stars N] [--res R] [--seed S] \
                         [--incline DEG] [--kappa K] [--emissivity J] [--exposure E]";
    let mut out: Option<PathBuf> = None;
    let mut n: usize = 60_000;
    let mut n_stars: usize = 90_000;
    let mut res: u32 = 128;
    let mut seed: u64 = 42;
    let mut incline_deg: f64 = 78.0;
    let mut kappa: f32 = 6.0;
    let mut emissivity: f32 = 0.05;
    let mut exposure: f32 = 1.0;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut num = |flag: &str| -> Result<String, String> {
            it.next().cloned().ok_or(format!("{flag} needs a value"))
        };
        let bad = |flag: &str| format!("{flag} must be a number");
        match arg.as_str() {
            "--n" => n = num("--n")?.parse().map_err(|_| bad("--n"))?,
            "--stars" => n_stars = num("--stars")?.parse().map_err(|_| bad("--stars"))?,
            "--res" => res = num("--res")?.parse().map_err(|_| bad("--res"))?,
            "--seed" => seed = num("--seed")?.parse().map_err(|_| bad("--seed"))?,
            "--incline" => incline_deg = num("--incline")?.parse().map_err(|_| bad("--incline"))?,
            "--kappa" => kappa = num("--kappa")?.parse().map_err(|_| bad("--kappa"))?,
            "--emissivity" => {
                emissivity = num("--emissivity")?
                    .parse()
                    .map_err(|_| bad("--emissivity"))?
            }
            "--exposure" => exposure = num("--exposure")?.parse().map_err(|_| bad("--exposure"))?,
            other if out.is_none() => out = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument `{other}`\n{USAGE}").into()),
        }
    }
    let out = out.unwrap_or_else(|| std::env::temp_dir().join("galaxy_volume_demo"));
    std::fs::create_dir_all(&out)?;

    // 1. Gas: the M7d synthetic disk, voxelized.
    let state = synthetic_gas_disk(n, seed, incline_deg.to_radians());
    let t0 = Instant::now();
    let grid = deposit_gas(
        &state,
        &GasGridConfig {
            dims: [res; 3],
            ..Default::default()
        },
    )
    .expect("all particles are gas");
    println!(
        "gas: {n} particles, incline {incline_deg}° → {res}³ grid in {:.1} ms",
        ms(t0.elapsed())
    );

    // 2. Stars: disk + bulge in the same geometry.
    let stars = synthetic_star_frame(n_stars, seed, incline_deg.to_radians());
    println!("stars: {} splats (disk + bulge)", stars.len());

    // 3. Render the A/B(/reference) set.
    let rcfg = RenderConfig {
        width: 1920,
        height: 1080,
        falloff: FALLOFF,
        ..RenderConfig::default()
    };
    let (bmin, bmax) = stars.bounds();
    let camera = Camera::frame_bounds(
        bmin,
        bmax,
        Vec3::NEG_Z,
        Vec3::Y,
        DEFAULT_MARGIN,
        rcfg.aspect(),
    );
    let renderer = Renderer::new()?;
    let gcfg = GradeConfig {
        exposure,
        tonemap: TONEMAP,
        bloom: Some(BloomConfig {
            strength: BLOOM_STRENGTH,
            levels: DEFAULT_BLOOM_LEVELS,
            radius: DEFAULT_BLOOM_RADIUS,
        }),
        ..GradeConfig::default()
    };
    let gas_color = [0.55_f32, 0.62, 0.95];
    let emit = |name: &str, gas: Option<&GasFrame>| -> Result<(), Box<dyn std::error::Error>> {
        let t0 = Instant::now();
        let img = renderer.render_frame_with_gas(&stars, gas, &camera, &rcfg)?;
        let dt = t0.elapsed();
        let exr = out.join(format!("{name}.exr"));
        let png = out.join(format!("{name}.png"));
        write_exr(&exr, &img)?;
        grade_file(&exr, &png, &gcfg)?;
        let flux = img.total_flux();
        println!(
            "{name}: {:.1} ms render, flux [{:.0}, {:.0}, {:.0}] → {}",
            ms(dt),
            flux[0],
            flux[1],
            flux[2],
            png.display()
        );
        Ok(())
    };

    emit(
        "composite",
        Some(&GasFrame {
            grid0: &grid,
            grid1: &grid,
            temperature: None,
            mix: 0.0,
            lights: &[],
            look: GasLook {
                color: gas_color,
                emissivity,
                opacity: kappa,
                scatter: None,
            },
        }),
    )?;
    emit(
        "no_absorption",
        Some(&GasFrame {
            grid0: &grid,
            grid1: &grid,
            temperature: None,
            mix: 0.0,
            lights: &[],
            look: GasLook {
                color: gas_color,
                emissivity,
                opacity: 0.0,
                scatter: None,
            },
        }),
    )?;
    emit("stars_only", None)?;
    println!("A/B pair: composite.png (attenuation ON) vs no_absorption.png (κ = 0)");
    Ok(())
}

/// Temperature-color look-dev demo (H5 phase B). Takes the `volume-demo` scene
/// (synthetic sech² gas disk + star field) and paints a synthetic internal-energy
/// field on the gas — a hot Gaussian core fading to cold outskirts,
/// `u(r) = u_cold + (u_hot − u_cold)·exp(−(r/r_core)²)` — then renders it through
/// the H1-H4 temperature colormap (deposit the ρ + energy-moment pair, map ū = N/ρ
/// across a fixed `[u_lo, u_hi]` band to a cold→hot lerp). Emits three PNGs into
/// `out`: `temperature` (the colored gas), `flat` (the same geometry with a flat
/// tint — the A/B control proving the colormap is what changes the look), and
/// `stars_only` (reference). No sim: the field is synthetic, so the whole cold/hot/
/// band/color surface is tunable from the CLI at render cost (~seconds), which is
/// the point of doing this before wiring the adiabatic front-end (phase C).
///
/// Usage: `temp-demo [out_dir] [--n N] [--stars N] [--res R] [--seed S]
///         [--incline DEG] [--kappa K] [--emissivity J] [--exposure E]
///         [--u-cold U] [--u-hot U] [--r-core R] [--u-lo U] [--u-hi U]
///         [--cold R,G,B] [--hot R,G,B]`. The band defaults to `[u_cold, u_hot]`.
fn temp_demo(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use galaxy_render::{GasFrame, GasLook, TempColor};
    use galaxy_renderprep::{deposit_gas_with_temperature, GasGridConfig};
    use std::time::Instant;

    const USAGE: &str = "usage: temp-demo [out_dir] [--n N] [--stars N] [--res R] [--seed S] \
                         [--incline DEG] [--kappa K] [--emissivity J] [--exposure E] \
                         [--u-cold U] [--u-hot U] [--r-core R] [--u-lo U] [--u-hi U] \
                         [--cold R,G,B] [--hot R,G,B]";
    let mut out: Option<PathBuf> = None;
    let mut n: usize = 60_000;
    let mut n_stars: usize = 90_000;
    let mut res: u32 = 128;
    let mut seed: u64 = 42;
    let mut incline_deg: f64 = 78.0;
    let mut kappa: f32 = 6.0;
    let mut emissivity: f32 = 0.05;
    let mut exposure: f32 = 1.0;
    let mut u_cold: f64 = 0.02;
    let mut u_hot: f64 = 1.0;
    let mut r_core: f64 = 1.0;
    let mut u_lo: Option<f32> = None;
    let mut u_hi: Option<f32> = None;
    // Cold = deep blue (cool diffuse gas), hot = warm white-orange (shocked core).
    let mut cold: [f32; 3] = [0.25, 0.45, 1.0];
    let mut hot: [f32; 3] = [1.0, 0.72, 0.32];
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let mut val = |flag: &str| -> Result<String, String> {
            it.next().cloned().ok_or(format!("{flag} needs a value"))
        };
        let bad = |flag: &str| format!("{flag} must be a number");
        let rgb = |flag: &str, s: &str| -> Result<[f32; 3], String> {
            let v: Vec<f32> = s.split(',').filter_map(|c| c.trim().parse().ok()).collect();
            match v.as_slice() {
                [r, g, b] => Ok([*r, *g, *b]),
                _ => Err(format!(
                    "{flag} must be R,G,B (three comma-separated numbers)"
                )),
            }
        };
        match arg.as_str() {
            "--n" => n = val("--n")?.parse().map_err(|_| bad("--n"))?,
            "--stars" => n_stars = val("--stars")?.parse().map_err(|_| bad("--stars"))?,
            "--res" => res = val("--res")?.parse().map_err(|_| bad("--res"))?,
            "--seed" => seed = val("--seed")?.parse().map_err(|_| bad("--seed"))?,
            "--incline" => incline_deg = val("--incline")?.parse().map_err(|_| bad("--incline"))?,
            "--kappa" => kappa = val("--kappa")?.parse().map_err(|_| bad("--kappa"))?,
            "--emissivity" => {
                emissivity = val("--emissivity")?
                    .parse()
                    .map_err(|_| bad("--emissivity"))?
            }
            "--exposure" => exposure = val("--exposure")?.parse().map_err(|_| bad("--exposure"))?,
            "--u-cold" => u_cold = val("--u-cold")?.parse().map_err(|_| bad("--u-cold"))?,
            "--u-hot" => u_hot = val("--u-hot")?.parse().map_err(|_| bad("--u-hot"))?,
            "--r-core" => r_core = val("--r-core")?.parse().map_err(|_| bad("--r-core"))?,
            "--u-lo" => u_lo = Some(val("--u-lo")?.parse().map_err(|_| bad("--u-lo"))?),
            "--u-hi" => u_hi = Some(val("--u-hi")?.parse().map_err(|_| bad("--u-hi"))?),
            "--cold" => cold = rgb("--cold", &val("--cold")?)?,
            "--hot" => hot = rgb("--hot", &val("--hot")?)?,
            other if out.is_none() => out = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument `{other}`\n{USAGE}").into()),
        }
    }
    if r_core <= 0.0 {
        return Err("--r-core must be positive".into());
    }
    let out = out.unwrap_or_else(|| std::env::temp_dir().join("galaxy_temp_demo"));
    std::fs::create_dir_all(&out)?;
    let u_lo = u_lo.unwrap_or(u_cold as f32);
    let u_hi = u_hi.unwrap_or(u_hot as f32);

    // 1. Gas: the synthetic disk, then paint the synthetic hot-core energy field.
    let mut state = synthetic_gas_disk(n, seed, incline_deg.to_radians());
    for (u, &p) in state.u.iter_mut().zip(state.pos.iter()) {
        let r = p.length();
        *u = u_cold + (u_hot - u_cold) * (-(r / r_core).powi(2)).exp();
    }
    let t0 = Instant::now();
    let (rho, moment) = deposit_gas_with_temperature(
        &state,
        &GasGridConfig {
            dims: [res; 3],
            ..Default::default()
        },
    )
    .expect("all particles are gas");
    println!(
        "gas: {n} particles, incline {incline_deg}° → {res}³ ρ+moment grids in {:.1} ms \
         (u ∈ [{u_cold}, {u_hot}], r_core {r_core})",
        ms(t0.elapsed())
    );

    // 2. Stars: disk + bulge in the same geometry (context for the dust lane).
    let stars = synthetic_star_frame(n_stars, seed, incline_deg.to_radians());
    println!("stars: {} splats (disk + bulge)", stars.len());

    // 3. Render: temperature-colored, flat-tint control, stars-only reference.
    let rcfg = RenderConfig {
        width: 1920,
        height: 1080,
        falloff: FALLOFF,
        ..RenderConfig::default()
    };
    let (bmin, bmax) = stars.bounds();
    let camera = Camera::frame_bounds(
        bmin,
        bmax,
        Vec3::NEG_Z,
        Vec3::Y,
        DEFAULT_MARGIN,
        rcfg.aspect(),
    );
    let renderer = Renderer::new()?;
    let gcfg = GradeConfig {
        exposure,
        tonemap: TONEMAP,
        bloom: Some(BloomConfig {
            strength: BLOOM_STRENGTH,
            levels: DEFAULT_BLOOM_LEVELS,
            radius: DEFAULT_BLOOM_RADIUS,
        }),
        ..GradeConfig::default()
    };
    // Flat control uses the band midpoint color so the A/B isolates the *colormap*,
    // not overall brightness — same emissivity/κ, one tint vs the ū-driven ramp.
    let flat_color = [
        0.5 * (cold[0] + hot[0]),
        0.5 * (cold[1] + hot[1]),
        0.5 * (cold[2] + hot[2]),
    ];
    let look = |color: [f32; 3]| GasLook {
        color,
        emissivity,
        opacity: kappa,
        scatter: None,
    };
    let emit = |name: &str, gas: Option<&GasFrame>| -> Result<(), Box<dyn std::error::Error>> {
        let t0 = Instant::now();
        let img = renderer.render_frame_with_gas(&stars, gas, &camera, &rcfg)?;
        let dt = t0.elapsed();
        let exr = out.join(format!("{name}.exr"));
        let png = out.join(format!("{name}.png"));
        write_exr(&exr, &img)?;
        grade_file(&exr, &png, &gcfg)?;
        let flux = img.total_flux();
        println!(
            "{name}: {:.1} ms render, flux [{:.0}, {:.0}, {:.0}] → {}",
            ms(dt),
            flux[0],
            flux[1],
            flux[2],
            png.display()
        );
        Ok(())
    };

    emit(
        "temperature",
        Some(&GasFrame {
            grid0: &rho,
            grid1: &rho,
            temperature: Some(TempColor {
                moment0: &moment,
                moment1: &moment,
                cold,
                hot,
                u_lo,
                u_hi,
            }),
            mix: 0.0,
            lights: &[],
            look: look(flat_color),
        }),
    )?;
    emit(
        "flat",
        Some(&GasFrame {
            grid0: &rho,
            grid1: &rho,
            temperature: None,
            mix: 0.0,
            lights: &[],
            look: look(flat_color),
        }),
    )?;
    emit("stars_only", None)?;
    println!(
        "A/B pair: temperature.png (ū→color, band [{u_lo}, {u_hi}]) vs flat.png (single tint)"
    );
    Ok(())
}

/// One gas particle's CFL data (I0): its stable step `dt = c_cfl·h/v_sig` and the
/// two quantities that set it. `h`/`v_sig` are surfaced so the finest rung can be
/// eyeballed — a "win" driven by one particle with an artifact-tiny `h` is measuring
/// numerical noise, not a physical dense knot (the advisor's #7 sanity check).
#[derive(Clone, Copy)]
struct GasDt {
    dt: f64,
    h: f64,
    v_sig: f64,
}

/// A faithful copy of the **isothermal arm** of [`galaxy_solvers::sph::max_stable_dt`]
/// with the `min` fold removed — the per-particle `dt_i = c_cfl · h_i / v_sig,i` over
/// the gas subset, in gas (ascending-index) order. Byte-for-byte the same arithmetic
/// and gather order as the shipped scalar bound, so `min_i dt_i` equals
/// `max_stable_dt(...)` exactly (asserted at runtime by the caller — the I1 invariant
/// used here as a self-check).
///
/// Kept in the xtask, NOT the solver: I0 is a go/no-go measurement, so the shipped CFL
/// path stays textually untouched (the E-series verbatim pin) and I0 stays reversible
/// if the verdict is "stop". If it is "go", I1 lands the gated per-particle vector in
/// `cfl.rs` and this copy retires.
fn per_particle_gas_dt(state: &State, c_s: f64, cfg: &DensityConfig, c_cfl: f64) -> Vec<GasDt> {
    let gas: Vec<usize> = (0..state.len())
        .filter(|&i| state.kind[i] == Species::Gas)
        .collect();
    if gas.is_empty() {
        return Vec::new();
    }
    let gpos: Vec<DVec3> = gas.iter().map(|&i| state.pos[i]).collect();
    let gvel: Vec<DVec3> = gas.iter().map(|&i| state.vel[i]).collect();
    let gmass: Vec<f64> = gas.iter().map(|&i| state.mass[i]).collect();
    let dens = density_adaptive(&gpos, &gmass, cfg, None);
    let h = &dens.h;

    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    let grid = HashGrid::build(&gpos, SUPPORT * h_max);
    let two_cs = 2.0 * c_s;

    let mut out = Vec::with_capacity(gpos.len());
    for i in 0..gpos.len() {
        let ngb = grid.neighbours_within(&gpos, gpos[i], SUPPORT * h_max);
        let mut v_sig = two_cs;
        for &j in &ngb {
            if j == i {
                continue;
            }
            let r_ij = gpos[i] - gpos[j];
            let r = r_ij.length();
            if r == 0.0 || r >= SUPPORT * h[i].max(h[j]) {
                continue;
            }
            let w = (gvel[i] - gvel[j]).dot(r_ij) / r;
            if w < 0.0 {
                v_sig = v_sig.max(two_cs - 3.0 * w);
            }
        }
        out.push(GasDt {
            dt: c_cfl * h[i] / v_sig,
            h: h[i],
            v_sig,
        });
    }
    out
}

/// The I0 go/no-go measurement (docs/plans/laddered-ember-cadence.md). Loads a
/// `gasrich` snapshot (or a directory of them), computes the per-instant gas CFL
/// timestep of every gas particle, and histograms the power-of-two rung
/// distribution at the **pericenter** snapshot (the tightest global CFL bound,
/// where individual timesteps help most) and, for contrast, the first (early
/// diffuse) snapshot. Reports the ideal-ceiling individual-timestep speedup and
/// the go/no-go verdict against the plan's ≥3× / <2× thresholds.
///
/// The number this produces is the SPATIAL spread of `h_i/v_sig,i` across particles
/// at one instant — NOT the A5 headline 34× (that is the *temporal* range of the
/// global bound, already banked by global block-adaptive). See the plan's opening.
///
/// Usage: `rung-spread <snapshots_dir | snapshot.snap> [--c-s C] [--c-cfl C]`.
fn rung_spread_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: rung-spread <snapshots_dir | snapshot.snap> [--c-s C] [--c-cfl C]";
    let mut path: Option<PathBuf> = None;
    // gasrich isothermal sound speed; the pipeline's C_CFL cancels in the rung
    // ratios, so c_cfl only scales the printed dt — default 1.0 so the global-min-dt
    // curve reconciles directly with the A5 run log (min 3.42e-3 @ c_cfl=1).
    let mut c_s = 0.1_f64;
    let mut c_cfl = 1.0_f64;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut num = |flag: &str| -> Result<f64, String> {
            it.next()
                .ok_or(format!("{flag} needs a value"))?
                .parse()
                .map_err(|_| format!("{flag} must be a number"))
        };
        match a.as_str() {
            "--c-s" => c_s = num("--c-s")?,
            "--c-cfl" => c_cfl = num("--c-cfl")?,
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument `{other}`\n{USAGE}").into()),
        }
    }
    let path = path.ok_or_else(|| format!("no snapshot given\n{USAGE}"))?;
    if !(c_s.is_finite() && c_s > 0.0 && c_cfl.is_finite() && c_cfl > 0.0) {
        return Err(
            format!("--c-s and --c-cfl must be positive (got c_s={c_s}, c_cfl={c_cfl})").into(),
        );
    }

    // A directory ⇒ the full run (scan for the pericenter); a single file ⇒ just it.
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "snap"))
            .collect();
        v.sort(); // snapshot_<step:08>.snap → lexicographic == time order
        if v.is_empty() {
            return Err(format!("no .snap files under {}", path.display()).into());
        }
        v
    } else {
        vec![path.clone()]
    };

    let cfg = DensityConfig::default();
    println!(
        "I0 rung-spread — the go/no-go for individual (per-particle) timesteps\n\
         source     {}\n\
         params     isothermal c_s = {c_s}, c_cfl = {c_cfl}, N_ngb = {}\n\
         measuring  the per-instant SPATIAL spread of gas dt_i = c_cfl·h_i/v_sig,i\n\
         (NOT the A5 34× — that is the TEMPORAL range, already banked by global adaptive)\n",
        path.display(),
        cfg.n_ngb,
    );

    // Pass 1: the global-min-dt curve over every snapshot (the pericenter dip). Uses
    // the copy's own min (self-checked against the shipped scalar on the reported
    // snapshots below), so the whole curve rides one verified code path.
    struct Row {
        file: PathBuf,
        time: f64,
        scalar: f64,
        n_gas: usize,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(files.len());
    for f in &files {
        let (_hdr, state) = galaxy_io::read_file(f)?;
        let v = per_particle_gas_dt(&state, c_s, &cfg, c_cfl);
        if v.is_empty() {
            return Err(format!("{}: snapshot has no gas particles", f.display()).into());
        }
        let scalar = v.iter().map(|g| g.dt).fold(f64::INFINITY, f64::min);
        rows.push(Row {
            file: f.clone(),
            time: state.time,
            scalar,
            n_gas: v.len(),
        });
    }

    // Pericenter = tightest global bound; early-diffuse = first snapshot.
    let peri = rows
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.scalar.total_cmp(&b.scalar))
        .map(|(i, _)| i)
        .expect("rows non-empty");
    let loosest = rows.iter().map(|r| r.scalar).fold(0.0_f64, f64::max);

    if files.len() > 1 {
        println!("--- global min dt over the run (the pericenter dip) ---");
        println!("     snapshot                 t        min_i dt_i   N_gas");
        for (i, r) in rows.iter().enumerate() {
            let name = r.file.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            let mark = if i == peri { "  <- PERICENTER" } else { "" };
            println!(
                "  {name:<28} {:>6.2}   {:>10.4e}   {:>5}{mark}",
                r.time, r.scalar, r.n_gas
            );
        }
        println!(
            "  span: tightest {:.4e} (pericenter) .. loosest {:.4e}  (temporal range {:.1}×)\n",
            rows[peri].scalar,
            loosest,
            loosest / rows[peri].scalar,
        );
    }

    // Report the two snapshots. Pericenter carries the decision; the first snapshot
    // is within-run contrast (early, diffuse — before the encounter compresses the gas).
    let hydro = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };
    if files.len() > 1 {
        report_snapshot(
            "EARLY DIFFUSE (first snapshot — contrast only)",
            &rows[0].file,
            c_s,
            c_cfl,
            &cfg,
            &hydro,
            false,
        )?;
    }
    let peri_label = if files.len() > 1 {
        "PERICENTER (tightest global bound — the decision snapshot)"
    } else {
        "SNAPSHOT"
    };
    report_snapshot(peri_label, &rows[peri].file, c_s, c_cfl, &cfg, &hydro, true)?;

    println!(
        "\nNote: the speedup is an IDEAL CEILING — particle-updates per base block only.\n\
         It EXCLUDES the I7 grid-rebuild / neighbour-prediction overhead (I6's net number)\n\
         and is over GAS particles only (collisionless rows carry no hydro CFL, dt = +∞).\n\
         Formula: speedup = N_gas·2^r_max / Σ_i 2^r_i (the plan's printed 2^(r_max−r)\n\
         exponent is inverted; this matches its gloss ≈ N / effective short-rung count)."
    );
    Ok(())
}

/// Compute + print one snapshot's rung histogram and (if `decide`) the go/no-go
/// verdict. Asserts the copied per-particle vector's `min` equals the shipped
/// scalar `max_stable_dt` bit-for-bit — the correctness anchor that lets a parallel
/// copy stand in for the shipped bound without refactoring it.
fn report_snapshot(
    label: &str,
    file: &Path,
    c_s: f64,
    c_cfl: f64,
    cfg: &DensityConfig,
    hydro: &HydroParams,
    decide: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_hdr, state) = galaxy_io::read_file(file)?;
    let gas = per_particle_gas_dt(&state, c_s, cfg, c_cfl);
    let dts: Vec<f64> = gas.iter().map(|g| g.dt).collect();
    let vec_min = dts.iter().copied().fold(f64::INFINITY, f64::min);

    // The I1 invariant as a runtime self-check: the copy must reproduce the shipped
    // bound to the bit, else the whole measurement is measuring the wrong physics.
    let scalar = max_stable_dt(&state, hydro, cfg, c_cfl);
    if vec_min.to_bits() != scalar.to_bits() {
        return Err(format!(
            "I0 self-check FAILED on {}: copied vector min {vec_min:.17e} != shipped \
             max_stable_dt {scalar:.17e} — the CFL copy diverged from the solver",
            file.display()
        )
        .into());
    }

    let spread = rung_spread(&dts).ok_or("no finite gas dt in snapshot")?;
    let (dt_lo, dt_med, dt_hi) = min_median_max(&dts);
    let name = file.file_name().and_then(|s| s.to_str()).unwrap_or("?");

    let n_collisionless = state.len() - spread.n;
    println!("=== {label} ===");
    println!("  {name}  (t = {:.2})", state.time);
    println!(
        "  N_gas {} of {} total ({n_collisionless} collisionless — no hydro CFL, \
         dt = +∞, excluded from N)",
        spread.n,
        state.len(),
    );
    println!(
        "  dt: min {dt_lo:.4e}  median {dt_med:.4e}  max {dt_hi:.4e}  \
         (spatial dynamic range {:.1}×)",
        spread.dynamic_range(),
    );
    println!("  self-check: min_i dt_i == max_stable_dt = {scalar:.6e}  [bit-exact OK]");

    // The rung histogram: rung 0 is the coarsest (step dt_base), r_max the finest.
    let dt_base = spread.dt_base;
    let r_max = spread.r_max();
    let n = spread.n as f64;
    println!("  rung   step (dt_base/2^r)   count    frac   |");
    for (r, &cnt) in spread.counts.iter().enumerate() {
        let frac = cnt as f64 / n;
        let bar = "#".repeat((frac * 40.0).round() as usize);
        let tag = if r == r_max { " (finest)" } else { "" };
        println!(
            "  {r:>3}    {:>16.4e}   {cnt:>6}   {:>5.1}%  | {bar}{tag}",
            dt_base / 2f64.powi(r as i32),
            frac * 100.0,
        );
    }

    // Eyeball the finest rung (advisor #7): is it a physical knot (many particles)
    // or 1–2 outliers with an anomalous h? Compare their h to the overall gas h.
    let all_h: Vec<f64> = gas.iter().map(|g| g.h).collect();
    let (_, h_med, _) = min_median_max(&all_h);
    let finest: Vec<&GasDt> = gas
        .iter()
        .filter(|g| ((dt_base / g.dt).log2().ceil().max(0.0) as usize) == r_max)
        .collect();
    let finest_h: Vec<f64> = finest.iter().map(|g| g.h).collect();
    let finest_vsig: Vec<f64> = finest.iter().map(|g| g.v_sig).collect();
    let (fh_lo, _, fh_hi) = min_median_max(&finest_h);
    let (fv_lo, _, fv_hi) = min_median_max(&finest_vsig);
    println!(
        "  finest rung r={r_max}: {} particle(s) ({:.1}% of gas); h ∈ [{fh_lo:.3e}, {fh_hi:.3e}] \
         (overall median h {h_med:.3e}); v_sig ∈ [{fv_lo:.3e}, {fv_hi:.3e}]",
        finest.len(),
        spread.finest_fraction() * 100.0,
    );
    if finest.len() <= 2 && fh_hi < 0.25 * h_med {
        println!(
            "  ⚠ finest rung is 1–2 particles with h well below the median — likely a \
             numerical-h outlier, not a physical knot; discount its contribution."
        );
    }

    // Sensitivity to the resolved-tail depth (advisor): the speedup is invariant to
    // dt_base (the diffuse end) and governed by the finest rungs, so show how the
    // ceiling collapses if the tail is only resolved to rung c. A DIAGNOSTIC — a real
    // cap needs the I5 timestep limiter — not a second verdict.
    if r_max >= 1 {
        let caps: Vec<usize> = (r_max.saturating_sub(3)..=r_max)
            .filter(|&c| c >= 1)
            .collect();
        let cells: Vec<String> = caps
            .iter()
            .map(|&c| {
                let tag = if c == r_max { " (full tail)" } else { "" };
                format!("cap r={c} → {:.2}×{tag}", spread.speedup_at_cap(c))
            })
            .collect();
        println!(
            "  tail sensitivity (resolve fine rungs only to c):  {}",
            cells.join("   ")
        );
    }
    println!(
        "  IDEAL-CEILING SPEEDUP (individual vs global-adaptive): {:.2}×",
        spread.speedup
    );
    if decide {
        // Tail-fragility test: if dropping the single finest rung collapses the
        // ceiling below the STOP line, the "win" rides an under-resolved tail (a
        // small-number statistic at this resolution) and the number cannot settle
        // the go/no-go on its own — it needs the higher-resolution regime.
        let robust = spread.speedup_at_cap(r_max.saturating_sub(1));
        let fragile = spread.speedup >= 3.0 && robust < 2.0;
        let verdict = if fragile {
            format!(
                "INCONCLUSIVE — {:.2}× clears ≥3× but is TAIL-FRAGILE: resolving the \
                 finest rung one step coarser drops it to {robust:.2}× (<2×). The win \
                 rides a {}-particle ({:.1}%) tail — verify at full resolution before \
                 committing (do NOT start I1 on this number).",
                spread.speedup,
                spread.counts.last().copied().unwrap_or(0),
                spread.finest_fraction() * 100.0,
            )
        } else if spread.speedup >= 3.0 {
            format!(
                "GO — {:.2}× ideal ceiling and robust to the finest rung ({robust:.2}× \
                 without it); build individual timesteps.",
                spread.speedup
            )
        } else if spread.speedup < 2.0 {
            format!(
                "STOP — {:.2}× (<2× ideal ceiling). Global block-adaptive is enough; \
                 record the finding, the integrator rewrite is not worth it.",
                spread.speedup
            )
        } else {
            format!(
                "MARGINAL — {:.2}× (2–3× ideal ceiling); the I7 grid-rebuild/prediction \
                 overhead likely eats it. Distribution + user decide.",
                spread.speedup
            )
        };
        println!("  VERDICT: {verdict}");
    }
    println!();
    Ok(())
}

/// I0b (docs/plans/laddered-ember-cadence.md): the GRAVITATIONAL rung-spread — the
/// precondition for the individual-timestep plan's `hydro+gravity` mode. Where
/// `rung-spread` (I0) measured the *gas CFL* rung spread that lever (a) / hydro-only
/// exploits, this measures the **star gravitational-rung** spread that lever (b) —
/// subcycling the O(N·logN) gravity WALK on a stale tree — hangs on. It loads a
/// snapshot (or a run), computes every particle's gravitational step
/// `dt = eta·√(eps/|a|)` from the pipeline Barnes-Hut field, finds the pericenter
/// (the tightest STAR bound), histograms the star rungs there, and reprojects the
/// measured star drop-finest walk factor onto the plan's 2026-07-09 Amdahl split so
/// the ~2.24× `hydro+gravity` estimate rests on data, not a borrowed hydro factor.
///
/// The star (`Collisionless`) subset carries the verdict — under `hydro+gravity` the
/// gas walks on its *hydro* rung (it is active for hydro anyway), so only the stars
/// get a *gravitational* rung, and the star spread is the isolated lever-(b) number.
/// The all-N and gas one-liners are context only.
///
/// Usage: `grav-rung-spread <snapshots_dir | snapshot.snap> [--eps E] [--eta H] [--theta T]`.
fn grav_rung_spread_cmd(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str =
        "usage: grav-rung-spread <snapshots_dir | snapshot.snap> [--eps E] [--eta H] [--theta T]";
    let mut path: Option<PathBuf> = None;
    // eps = gasrich softening (feeds |a| itself, so it IS a knob on the force — but
    // the ε inside √(ε/|a|) cancels in the rung ratios). eta is purely cosmetic
    // (cancels entirely — scales every dt equally). theta = the pipeline opening
    // angle, so the rungs match what the real hydro+gravity walk would assign.
    let mut eps = 0.05_f64;
    let mut eta = 1.0_f64;
    let mut theta = THETA;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut num = |flag: &str| -> Result<f64, String> {
            it.next()
                .ok_or(format!("{flag} needs a value"))?
                .parse()
                .map_err(|_| format!("{flag} must be a number"))
        };
        match a.as_str() {
            "--eps" => eps = num("--eps")?,
            "--eta" => eta = num("--eta")?,
            "--theta" => theta = num("--theta")?,
            other if path.is_none() => path = Some(PathBuf::from(other)),
            other => return Err(format!("unexpected argument `{other}`\n{USAGE}").into()),
        }
    }
    let path = path.ok_or_else(|| format!("no snapshot given\n{USAGE}"))?;
    if !(eps.is_finite()
        && eps > 0.0
        && eta.is_finite()
        && eta > 0.0
        && theta.is_finite()
        && theta >= 0.0)
    {
        return Err(format!("--eps/--eta must be positive and --theta ≥ 0 (got eps={eps}, eta={eta}, theta={theta})").into());
    }

    // A directory ⇒ the full run (scan for the pericenter); a single file ⇒ just it.
    let files: Vec<PathBuf> = if path.is_dir() {
        let mut v: Vec<PathBuf> = std::fs::read_dir(&path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "snap"))
            .collect();
        v.sort(); // snapshot_<step:08>.snap → lexicographic == time order
        if v.is_empty() {
            return Err(format!("no .snap files under {}", path.display()).into());
        }
        v
    } else {
        vec![path.clone()]
    };

    println!(
        "I0b grav-rung-spread — the go/no-go PRECONDITION for the `hydro+gravity` mode\n\
         source     {}\n\
         params     Plummer softening ε = {eps}, η = {eta} (cancels), θ = {theta} (pipeline)\n\
         measuring  the per-instant SPATIAL spread of gravitational dt_i = η·√(ε/|a_i|)\n\
         verdict    the STAR subset (walk lever b); dt ∝ |a|^(−½) ⇒ spread NARROWER than hydro\n",
        path.display(),
    );

    // Pass 1: the STAR-subset min-dt curve over the run (the gravitational pericenter
    // = the tightest star bound = the max star |a|). Stars carry lever (b), so the
    // pericenter is chosen on their bound, not all-N.
    struct Row {
        file: PathBuf,
        time: f64,
        star_min_dt: f64,
        n_star: usize,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(files.len());
    for f in &files {
        let (_hdr, state) = galaxy_io::read_file(f)?;
        let g = per_particle_grav_dt(&state, eps, eta, theta);
        let star_min = g
            .iter()
            .filter(|gd| gd.kind == Species::Collisionless)
            .map(|gd| gd.dt)
            .fold(f64::INFINITY, f64::min);
        let n_star = g
            .iter()
            .filter(|gd| gd.kind == Species::Collisionless)
            .count();
        if n_star == 0 {
            return Err(format!(
                "{}: snapshot has no collisionless (star) particles",
                f.display()
            )
            .into());
        }
        rows.push(Row {
            file: f.clone(),
            time: state.time,
            star_min_dt: star_min,
            n_star,
        });
    }

    let peri = rows
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.star_min_dt.total_cmp(&b.star_min_dt))
        .map(|(i, _)| i)
        .expect("rows non-empty");
    let loosest = rows.iter().map(|r| r.star_min_dt).fold(0.0_f64, f64::max);

    if files.len() > 1 {
        println!("--- STAR min dt over the run (the gravitational pericenter dip) ---");
        println!("     snapshot                 t        min_i dt_i   N_star");
        for (i, r) in rows.iter().enumerate() {
            let name = r.file.file_name().and_then(|s| s.to_str()).unwrap_or("?");
            let mark = if i == peri { "  <- PERICENTER" } else { "" };
            println!(
                "  {name:<28} {:>6.2}   {:>10.4e}   {:>6}{mark}",
                r.time, r.star_min_dt, r.n_star
            );
        }
        println!(
            "  span: tightest {:.4e} (pericenter) .. loosest {:.4e}  (temporal range {:.1}×)\n",
            rows[peri].star_min_dt,
            loosest,
            loosest / rows[peri].star_min_dt,
        );
        // Contrast: the first (early diffuse) snapshot, verdict off.
        report_grav_snapshot(
            "EARLY DIFFUSE (first snapshot — contrast only)",
            &rows[0].file,
            eps,
            eta,
            theta,
            false,
        )?;
    }
    let peri_label = if files.len() > 1 {
        "PERICENTER (tightest star bound — the decision snapshot)"
    } else {
        "SNAPSHOT"
    };
    report_grav_snapshot(peri_label, &rows[peri].file, eps, eta, theta, true)?;

    Ok(())
}

/// Report one snapshot's gravitational rung spread. The STAR subset carries the
/// verdict (histogram + drop-finest walk factor + the Amdahl reprojection when
/// `decide`); all-N and gas are one-line context. Asserts nothing is dropped (the
/// inverted +∞ semantics vs the hydro tool: a zero-accel star is the coarsest rung,
/// not an exclusion) and cross-checks the θ Barnes-Hut field against exact direct sum.
fn report_grav_snapshot(
    label: &str,
    file: &Path,
    eps: f64,
    eta: f64,
    theta: f64,
    decide: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let (_hdr, state) = galaxy_io::read_file(file)?;
    let g = per_particle_grav_dt(&state, eps, eta, theta);
    let name = file.file_name().and_then(|s| s.to_str()).unwrap_or("?");

    let star_dts: Vec<f64> = g
        .iter()
        .filter(|gd| gd.kind == Species::Collisionless)
        .map(|gd| gd.dt)
        .collect();
    let gas_dts: Vec<f64> = g
        .iter()
        .filter(|gd| gd.kind == Species::Gas)
        .map(|gd| gd.dt)
        .collect();
    let all_dts: Vec<f64> = g.iter().map(|gd| gd.dt).collect();

    // Inverted +∞ semantics (advisor): unlike the hydro tool, a zero-accel particle
    // is the BEST-case coarsest rung, not a non-participant — so nothing must drop.
    let star_finite = star_dts
        .iter()
        .filter(|d| d.is_finite() && **d > 0.0)
        .count();
    if star_finite != star_dts.len() {
        return Err(format!(
            "{name}: {} of {} stars have |a| == 0 (dt = +∞) — a zero-accel star is a \
             coarse rung, not an exclusion; padding is inverted here. Investigate the \
             snapshot before trusting the walk factor.",
            star_dts.len() - star_finite,
            star_dts.len(),
        )
        .into());
    }

    let spread = rung_spread(&star_dts).ok_or("no finite star gravitational dt in snapshot")?;
    let (dt_lo, dt_med, dt_hi) = min_median_max(&star_dts);
    let r_max = spread.r_max();
    let drop_finest = spread.speedup_at_cap(r_max.saturating_sub(1));

    println!("=== {label} ===");
    println!("  {name}  (t = {:.2})", state.time);
    println!(
        "  N_star {} (Collisionless), N_gas {}, N_total {}",
        spread.n,
        gas_dts.len(),
        state.len(),
    );
    println!(
        "  STAR dt: min {dt_lo:.4e}  median {dt_med:.4e}  max {dt_hi:.4e}  \
         (spatial dynamic range {:.1}×)",
        spread.dynamic_range(),
    );

    // The star rung histogram — the decision distribution.
    let dt_base = spread.dt_base;
    let n = spread.n as f64;
    println!("  rung   step (dt_base/2^r)   count    frac   |");
    for (r, &cnt) in spread.counts.iter().enumerate() {
        let frac = cnt as f64 / n;
        let bar = "#".repeat((frac * 40.0).round() as usize);
        let tag = if r == r_max { " (finest)" } else { "" };
        println!(
            "  {r:>3}    {:>16.4e}   {cnt:>6}   {:>5.1}%  | {bar}{tag}",
            dt_base / 2f64.powi(r as i32),
            frac * 100.0,
        );
    }
    println!(
        "  finest rung r={r_max}: {} star(s) ({:.1}% of stars)",
        spread.counts.last().copied().unwrap_or(0),
        spread.finest_fraction() * 100.0,
    );

    // The θ cross-check: is the |a| field the rungs are built on trustworthy? (I0b's
    // stand-in for rung-spread's runtime self-check — there is no shipped grav bound.)
    let max_rel = accel_max_rel_vs_direct(&state, eps, theta);
    // A few-% |a| error is Barnes-Hut's DESIGNED θ=0.5 tolerance, not a defect — and
    // because dt ∝ |a|^(−½) it halves to ~sub-rung, so the rung assignment is robust
    // (run --theta 0 to confirm the distribution is θ-invariant). Only a gross error
    // (>10%) would actually move rungs.
    println!(
        "  θ cross-check: max |a_BH − a_exact| / |a_exact| = {max_rel:.3e}  \
         (θ={theta} vs exact direct sum — {})",
        if max_rel < 0.1 {
            "within BH tolerance; dt∝|a|^(−½) ⇒ rungs robust"
        } else {
            "LARGE — rungs may be θ-artefacts; tighten θ"
        },
    );

    // Context one-liners — all-N and gas drop-finest (NOT the verdict; gas walks on
    // its hydro rung under hydro+gravity, all-N mis-assigns that third).
    for (tag, dts) in [("all-N", &all_dts), ("gas", &gas_dts)] {
        if let Some(s) = rung_spread(dts) {
            let df = s.speedup_at_cap(s.r_max().saturating_sub(1));
            println!(
                "  [context] {tag:<5} full-tail {:.2}× / drop-finest {df:.2}×  (N={})",
                s.speedup, s.n,
            );
        }
    }

    println!(
        "  STAR WALK FACTOR (lever b): full-tail {:.2}× / DROP-FINEST {drop_finest:.2}×",
        spread.speedup,
    );

    if decide {
        // Reproject the measured star drop-finest onto the plan's Amdahl split. The
        // borrowed estimate used w_grav = W_HYDRO_DROP_FINEST (2.9×) → 2.24×; I0b
        // replaces it with the measured star factor.
        let b = AMDAHL_GASRICH_PERICENTER;
        let hydro_only = b.hydro_only_speedup(W_HYDRO_DROP_FINEST);
        let borrowed = b.hydro_plus_gravity_speedup(W_HYDRO_DROP_FINEST, W_HYDRO_DROP_FINEST);
        let measured = b.hydro_plus_gravity_speedup(W_HYDRO_DROP_FINEST, drop_finest);
        let delta_pct = (measured / hydro_only - 1.0) * 100.0;
        println!("\n  --- AMDAHL REPROJECTION (whole-sim, 2026-07-09 block split) ---");
        println!(
            "  hydro-only (lever a, ships regardless):        {hydro_only:.2}×  (clears the 30% bar)"
        );
        println!(
            "  hydro+gravity, plan's BORROWED w_grav=2.9×:     {borrowed:.2}×  (the ~2.24× estimate)"
        );
        println!(
            "  hydro+gravity, I0b MEASURED w_grav={drop_finest:.2}×:      {measured:.2}×  \
             (+{delta_pct:.0}% over hydro-only)"
        );

        // Verdict: does lever (b) add enough over hydro-only to justify the I-grav
        // design surface (gravity prediction + stale-tree gather + a gravitational-dt
        // floor)? hydro-only already clears the user's bar, so this is purely "is the
        // gravity scope expansion worth building".
        let verdict = if drop_finest < 1.3 {
            format!(
                "STARS BUNCH FINE — star drop-finest {drop_finest:.2}× is near 1×, so lever (b) \
                 adds only +{delta_pct:.0}% over hydro-only. Ship `hydro-only` and STOP; the \
                 I-grav design surface is not worth ~{measured:.2}× vs {hydro_only:.2}×."
            )
        } else if measured >= 2.0 {
            format!(
                "GO for `hydro+gravity` — measured star walk factor {drop_finest:.2}× lifts the \
                 whole-sim speedup to {measured:.2}× (+{delta_pct:.0}% over hydro-only's \
                 {hydro_only:.2}×); the gravity subcycling scope expansion pays. Build I-grav."
            )
        } else {
            format!(
                "MARGINAL — {measured:.2}× vs hydro-only {hydro_only:.2}× (+{delta_pct:.0}%). \
                 Lever (b) helps but the I-grav overhead (stale-tree gather + prediction) may \
                 eat the margin; a scope call, not a clear go."
            )
        };
        println!("  VERDICT: {verdict}");
        println!(
            "\n  Note: the walk factor is an IDEAL CEILING (walk-updates per base block only);\n\
             it EXCLUDES the I-grav stale-tree rebuild / gravity-prediction overhead. The\n\
             reprojection charges the WHOLE walk at the star factor — conservative, since the\n\
             ~⅓ gas share of the walk actually rides the (larger) hydro factor."
        );
    }
    println!();
    Ok(())
}

/// Integrate the grid along each axis into three column-density panels
/// (z | y | x views), normalized to a shared robust peak so the panels are
/// comparable, side by side in one linear HDR image (rows flipped so +vertical
/// is up, the render convention).
fn column_density_sheet(grid: &galaxy_renderprep::GasGrid) -> galaxy_render::HdrImage {
    let [nx, ny, nz] = grid.dims;
    let cell = grid.cell_size();
    let res = nx.max(ny).max(nz) as usize; // panels are dims-sized; cubic in practice
    let gap = 8usize;
    let width = 3 * res + 2 * gap;
    let mut panels = vec![vec![0.0f64; res * res]; 3];

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let rho = grid.data[grid.index(ix, iy, iz)] as f64;
                // ∫ρ dz → (x, y); ∫ρ dy → (x, z); ∫ρ dx → (y, z).
                panels[0][(iy as usize) * res + ix as usize] += rho * cell.z;
                panels[1][(iz as usize) * res + ix as usize] += rho * cell.y;
                panels[2][(iz as usize) * res + iy as usize] += rho * cell.x;
            }
        }
    }

    // Shared normalization: the 99.5th percentile of nonzero column densities,
    // so a lone peak cannot crush the lanes to black.
    let mut all: Vec<f64> = panels
        .iter()
        .flatten()
        .copied()
        .filter(|&v| v > 0.0)
        .collect();
    all.sort_by(|a, b| a.total_cmp(b));
    let scale = if all.is_empty() {
        1.0
    } else {
        1.0 / all[(all.len() - 1) * 995 / 1000].max(f64::MIN_POSITIVE)
    };

    let mut pixels = vec![[0.0f32, 0.0, 0.0, 1.0]; width * res];
    for (p, panel) in panels.iter().enumerate() {
        let x_off = p * (res + gap);
        for row in 0..res {
            for col in 0..res {
                let v = (panel[(res - 1 - row) * res + col] * scale) as f32;
                pixels[row * width + x_off + col] = [v, v, v, 1.0];
            }
        }
    }
    galaxy_render::HdrImage {
        width: width as u32,
        height: res as u32,
        pixels,
    }
}

fn ms(d: std::time::Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// One row of the O(N)-win table: grid vs brute density at a fixed h, on `n`
/// particles, and the (must-be-zero) relative difference between them.
#[derive(Clone, Copy)]
struct DensityTiming {
    n: usize,
    t_grid: std::time::Duration,
    t_brute: std::time::Duration,
    max_rel: f64,
}

/// Time the grid gather and the brute O(N²) sum at the *same* smoothing lengths
/// (the ones the adaptive pass converges to for this exact point set), so the
/// two compute the identical quantity and the timing ratio is a pure
/// data-structure comparison. Returns both wall-clocks and their max relative
/// difference (the bit-exact invariant, expected 0).
fn time_density(pos: &[glam::DVec3], mass: &[f64], cfg: &DensityConfig) -> DensityTiming {
    use std::time::Instant;
    let h = density_adaptive(pos, mass, cfg, None).h;
    let t0 = Instant::now();
    let rho_grid = density_fixed(pos, mass, &h);
    let t_grid = t0.elapsed();
    let t0 = Instant::now();
    let rho_brute = reference_density(pos, mass, &h);
    let t_brute = t0.elapsed();
    let max_rel = rho_grid
        .iter()
        .zip(&rho_brute)
        .map(|(&a, &b)| (a - b).abs() / b.abs().max(f64::MIN_POSITIVE))
        .fold(0.0_f64, f64::max);
    DensityTiming {
        n: pos.len(),
        t_grid,
        t_brute,
        max_rel,
    }
}

fn min_median_max(v: &[f64]) -> (f64, f64, f64) {
    let mut s: Vec<f64> = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    (s[0], s[s.len() / 2], s[s.len() - 1])
}

/// Spearman rank correlation of two positive fields (compared in log space, which
/// is monotone so it does not change ranks — it just makes ties from the `0.0`
/// k-NN sentinel explicit). Returns Pearson correlation of the fractional ranks;
/// `NaN`-safe via `total_cmp`.
fn spearman_log(a: &[f64], b: &[f64]) -> f64 {
    let ra = fractional_ranks(a);
    let rb = fractional_ranks(b);
    let n = ra.len() as f64;
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let mut cov = 0.0;
    let mut va = 0.0;
    let mut vb = 0.0;
    for (x, y) in ra.iter().zip(&rb) {
        cov += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    let denom = (va * vb).sqrt();
    if denom > 0.0 {
        cov / denom
    } else {
        0.0 // a constant field has no rank spread — correlation undefined → 0
    }
}

/// Average ("fractional") ranks, so tied values share the mean of the ranks they
/// span — the standard Spearman tie handling.
fn fractional_ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].total_cmp(&v[j]));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && v[idx[j]].total_cmp(&v[idx[i]]) == std::cmp::Ordering::Equal {
            j += 1;
        }
        // ranks i..j (0-based) are tied → share their average (1-based mean).
        let avg = ((i + j - 1) as f64) / 2.0 + 1.0;
        for &k in &idx[i..j] {
            ranks[k] = avg;
        }
        i = j;
    }
    ranks
}

/// The scenario-independent pipeline: simulate the IC to snapshots (or reuse
/// retained ones), renderprep every snapshot to frame-data under the requested
/// coloring mode, build the scenario's camera path (static framing or the M6d
/// orbit/tilt rig), then render + grade each frame and (optionally) ffmpeg them
/// into a movie.
fn run_movie(
    s: &Scenario,
    out: &Path,
    color: ColorModeArg,
    reuse_snapshots: bool,
    backend: galaxy_xtask::simulate::Backend,
) -> Result<(), Box<dyn std::error::Error>> {
    let snap_dir = out.join("snapshots");
    let exr_dir = out.join("exr");
    let frame_dir = out.join("frames");
    for d in [&snap_dir, &exr_dir, &frame_dir] {
        std::fs::create_dir_all(d)?;
    }

    // 1. Simulate → snapshots — unless the caller asked to reuse retained ones
    //    (M6e: coloring modes iterate in render time, not sim time).
    if !reuse_snapshots {
        // The gas-gated simulate step (M7c): Barnes-Hut for a gas-free scenario
        // (byte-identical to the pre-M7c pipeline), or GravitySph + CflGuard when
        // the scenario carries gas — with the fixed dt validated against the hydro
        // CFL bound at t=0 before the first snapshot.
        let t_sim = std::time::Instant::now();
        let summary = simulate_snapshots(s, &snap_dir, backend)?;
        println!(
            "simulated {} steps → {} snapshots (t_final = {:.2}) in {:.1} s",
            summary.steps,
            summary.snapshots_emitted,
            summary.final_time,
            t_sim.elapsed().as_secs_f64()
        );
    }

    // 2. Renderprep on the SNAPSHOT cadence: the full prepare (including the O(N²)
    //    kNN density pass) runs only on snapshot states; the Hermite subframes below
    //    lerp these endpoint attributes (M6c decision — density evolves on the
    //    snapshot timescale, so per-subframe kNN would cost minutes for no gain).
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(&snap_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "snap"))
        .collect();
    snaps.sort(); // snapshot_<step:08>.snap → lexicographic == step order
    if reuse_snapshots && snaps.is_empty() {
        return Err(format!(
            "--reuse-snapshots: no .snap files in {} (run the scenario once without it)",
            snap_dir.display()
        )
        .into());
    }
    let states: Vec<State> = snaps
        .iter()
        .map(|p| galaxy_io::read_file(p).map(|(_, st)| st))
        .collect::<Result<Vec<_>, _>>()?;
    if states.is_empty() {
        return Err(format!("no snapshots under {}", snap_dir.display()).into());
    }
    if reuse_snapshots {
        // Brightness/palette weighting and QUICK sizing are derived from the
        // scenario's own particle counts — retained snapshots from a different
        // N (a QUICK/full mix-up) would silently mis-weight every frame.
        let (found, expect) = (states[0].len(), s.state.len());
        if found != expect {
            return Err(format!(
                "--reuse-snapshots: snapshots hold {found} particles but the scenario \
                 builds {expect} — QUICK/full mismatch? (GALAXY_MOVIE_QUICK)"
            )
            .into());
        }
        println!("reusing {} retained snapshots (no re-sim)", states.len());
    }

    // The effective prep config (M6e): the scenario's base look + the requested
    // coloring mode + the star-formation compression proxy, both anchored to THIS
    // run's snapshot 0 (frozen ramp colors; reference densities ρ0).
    let t_prep = std::time::Instant::now();
    let prep = effective_prep(s, color, &states[0]);
    let frames: Vec<_> = states.iter().map(|st| prepare(st, &prep)).collect();
    println!(
        "prepared {} endpoint frames in {:.1} s",
        frames.len(),
        t_prep.elapsed().as_secs_f64()
    );

    // 2b. Gas voxelization (M7e, plan D8): one density grid per SNAPSHOT
    //     endpoint — `None` for gas-free states, i.e. every pre-M7c scenario,
    //     which keeps this a no-op today. Subframes bind BOTH endpoint grids
    //     and mix in-shader (the M6c endpoint argument; nothing gas-related is
    //     lerped on the CPU). Look values are placeholders until `[look.gas]`
    //     lands in M7f.
    let quick = std::env::var_os("GALAXY_MOVIE_QUICK").is_some();
    let gas_cfg = galaxy_renderprep::GasGridConfig {
        dims: [if quick { 64 } else { 128 }; 3],
        ..Default::default()
    };
    // Temperature colormap (incandescent-nebular-veil): when the scenario asks
    // for it, deposit the internal-energy moment N alongside ρ (one shared
    // h-solve), so the raymarcher can color by ū = N/ρ. Off ⇒ ρ only, moments
    // all `None` (the bit-compat flat-tint path).
    let temp_params = s.gas_look.as_ref().and_then(|gl| gl.temperature);
    let t_gas = std::time::Instant::now();
    let (gas_grids, gas_moments): (
        Vec<Option<galaxy_renderprep::GasGrid>>,
        Vec<Option<galaxy_renderprep::GasGrid>>,
    ) = states
        .iter()
        .map(|st| match temp_params {
            Some(_) => match galaxy_renderprep::deposit_gas_with_temperature(st, &gas_cfg) {
                Some((rho, mom)) => (Some(rho), Some(mom)),
                None => (None, None),
            },
            None => (galaxy_renderprep::deposit_gas(st, &gas_cfg), None),
        })
        .unzip();
    // The volumetric gas look is scenario data ([look.gas], M7f): `Some` iff the
    // scenario is gas-rich, else the neutral default (inert — a gas-free run has no
    // grids for it to touch).
    let gas_look = match &s.gas_look {
        Some(gl) => galaxy_render::GasLook {
            color: gl.color,
            emissivity: gl.emissivity,
            opacity: gl.opacity,
            // The single-scatter option (scattered-starlit-veil): `scattering
            // = 0` (or omitted) maps to `None` — the bit-compat off path.
            scatter: (gl.scattering > 0.0).then_some(galaxy_render::ScatterLook {
                strength: gl.scattering,
                anisotropy: gl.anisotropy,
                shadows: gl.shadows,
                tint: gl.scatter_tint,
                softening: gl.scatter_softening,
            }),
        },
        None => galaxy_render::GasLook::default(),
    };
    if gas_grids.iter().any(Option::is_some) {
        println!(
            "voxelized gas ({}³) for {} snapshots in {:.1} s",
            gas_cfg.dims[0],
            gas_grids.len(),
            t_gas.elapsed().as_secs_f64()
        );
    }

    // 3. The camera path (M6d). Static: one face-on framing over the whole run
    //    (centered on the zero-COM barycenter, sized to a robust percentile radius
    //    so a few escapers don't shrink the galaxies to dots) — bit-exact with the
    //    pre-M6d pipeline. OrbitTilt: eased azimuth/tilt sweep, with the zoom
    //    breathing along the smoothed per-snapshot envelope of the same percentile
    //    radius (3-D, since an orbiting camera has no preferred plane).
    let rcfg = RenderConfig {
        width: s.width,
        height: s.height,
        falloff: FALLOFF,
        // Screen-space splat cap (pinprick-starfield), per-scenario; absent =
        // INFINITY = off, bit-identical to the uncapped M6g render.
        max_splat_px: s.max_splat_px.unwrap_or(f32::INFINITY),
        // Per-light shadow bake strategy (DDA/hierarchical deferral); bit-identical
        // to the brute default, faster on sparse frames.
        shadow_bake: s.shadow_bake,
        ..RenderConfig::default()
    };
    let path = match s.rig {
        Rig::Static => {
            let radius = framing_radius(&frames, s.frame_percentile).max(1e-3);
            println!(
                "framing radius (p{:.0}) = {radius:.2}",
                s.frame_percentile * 100.0
            );
            CameraPath::fixed(Camera::face_on(
                Vec3::splat(-radius),
                Vec3::splat(radius),
                rcfg.aspect(),
            ))
        }
        Rig::OrbitTilt {
            azimuth_deg,
            tilt_deg,
            window,
        } => {
            let raw = per_frame_radii(&frames, s.frame_percentile);
            let envelope: Vec<f32> = smooth_envelope(&raw, window)
                .into_iter()
                .map(|r| r.max(1e-3))
                .collect();
            println!(
                "framing envelope (p{:.0}, ±{window} snapshots) = {:.2}..{:.2}",
                s.frame_percentile * 100.0,
                envelope.iter().copied().fold(f32::INFINITY, f32::min),
                envelope.iter().copied().fold(0.0f32, f32::max),
            );
            CameraPath::orbit_tilt(
                Vec3::ZERO,
                (azimuth_deg.0.to_radians(), azimuth_deg.1.to_radians()),
                (tilt_deg.0.to_radians(), tilt_deg.1.to_radians()),
                envelope,
                DEFAULT_MARGIN,
                rcfg.aspect(),
            )?
        }
        Rig::Dolly {
            direction_deg,
            distance_frac,
            fov_deg,
            near_frac,
        } => {
            // Anchor the scene-scale-free fractions to the FINAL snapshot's
            // framing radius: the dolly targets the end state (the remnant), so
            // "outside the scene" and "inside the remnant" mean that frame.
            let raw = per_frame_radii(&frames, s.frame_percentile);
            let anchor = raw.last().copied().unwrap_or(1.0).max(1e-3);
            println!(
                "dolly anchor radius (p{:.0}, final snapshot) = {anchor:.2}; \
                 eye {:.2} -> {:.2}, near {:.3}",
                s.frame_percentile * 100.0,
                distance_frac.0 * anchor,
                distance_frac.1 * anchor,
                near_frac * anchor,
            );
            CameraPath::dolly(
                Vec3::ZERO,
                direction_deg.0.to_radians(),
                direction_deg.1.to_radians(),
                (distance_frac.0 * anchor, distance_frac.1 * anchor),
                fov_deg.to_radians(),
                near_frac * anchor,
                rcfg.aspect(),
            )?
        }
    };
    let gcfg = GradeConfig {
        exposure: EXPOSURE,
        tonemap: TONEMAP,
        bloom: Some(BloomConfig {
            strength: BLOOM_STRENGTH,
            levels: DEFAULT_BLOOM_LEVELS,
            radius: DEFAULT_BLOOM_RADIUS,
        }),
        // Baked local tonemap ([look.local_tone]): the same spatial blob-relief the
        // `regrade --local` A/B settled on, now part of the movie grade instead of a
        // separate regrade pass. `None` (any scenario without the section) is
        // bit-identical to the pre-tonemap grade.
        local: s.local_tone,
        ..GradeConfig::default()
    };
    let renderer = Renderer::new()?;

    // 4. Hermite temporal upsampling (M6c): `subframes` in-betweens per snapshot
    //    interval, plus the final snapshot itself → (n-1)·subframes + 1 frames.
    let total = match states.len() {
        0 | 1 => states.len(),
        n => (n - 1) * s.subframes as usize + 1,
    };
    let emit = |i: usize,
                frame: &FrameData,
                gas: Option<galaxy_render::GasFrame>|
     -> Result<(), Box<dyn std::error::Error>> {
        // The movie's unit timeline: frame i of `total` (a single-frame movie
        // sits at u = 0, the path start).
        let u = i as f32 / total.saturating_sub(1).max(1) as f32;
        let img = renderer.render_frame_with_gas(frame, gas.as_ref(), &path.camera_at(u), &rcfg)?;
        if i == total / 2 {
            let flux = img.total_flux();
            let peak = img
                .pixels
                .iter()
                .flat_map(|p| &p[..3])
                .fold(0.0f32, |m, &c| m.max(c));
            println!("mid-frame sanity: total_flux {flux:?}, peak pixel {peak:.3}");
        }
        let exr = exr_dir.join(format!("frame_{i:05}.exr"));
        let png = frame_dir.join(format!("frame_{i:05}.png"));
        write_exr(&exr, &img)?;
        grade_file(&exr, &png, &gcfg)?;
        Ok(())
    };
    let t_render = std::time::Instant::now();
    let mut i = 0;
    // Per-frame scatter light counts (tinted-octree-lanterns AB): the octree
    // clusterer's output size K drives the per-sample light loop + shadow bake
    // (both O(K)); the A/B freezes REFINE_TOL against this distribution + the
    // wall-clock below. Empty on the no-scatter path.
    let mut light_counts: Vec<usize> = Vec::new();
    for w in 0..states.len().saturating_sub(1) {
        // The span validates the id/time gates once per snapshot pair (a silent
        // id mismatch would scramble the movie — fail loudly instead).
        let span = HermiteSpan::new(&states[w], &states[w + 1])?;
        for j in 0..s.subframes {
            let u = f64::from(j) / f64::from(s.subframes);
            let fd = subframe(&span, &frames[w], &frames[w + 1], u);
            // Scatter lights are per-frame data clustered from the SAME
            // interpolated splats the frame draws (camera-independent, so
            // prep-time camera decoupling — D9 — is untouched).
            let lights = match (gas_look.scatter, &gas_grids[w]) {
                (Some(_), Some(_)) => galaxy_render::cluster_lights(&fd),
                _ => Vec::new(),
            };
            if gas_look.scatter.is_some() {
                light_counts.push(lights.len());
            }
            // Gas rides as the two endpoint grids + the subframe mix u.
            let gas = match (&gas_grids[w], &gas_grids[w + 1]) {
                (Some(g0), Some(g1)) => Some(galaxy_render::GasFrame {
                    grid0: g0,
                    grid1: g1,
                    // Temperature colormap: the co-registered moment endpoints
                    // (present iff the scenario asked for it and both endpoints
                    // carry gas), else the flat-tint march.
                    temperature: match (temp_params, &gas_moments[w], &gas_moments[w + 1]) {
                        (Some(t), Some(m0), Some(m1)) => Some(galaxy_render::TempColor {
                            moment0: m0,
                            moment1: m1,
                            cold: t.cold,
                            hot: t.hot,
                            u_lo: t.u_lo,
                            u_hi: t.u_hi,
                        }),
                        _ => None,
                    },
                    mix: u as f32,
                    lights: &lights,
                    look: gas_look,
                }),
                _ => None,
            };
            emit(i, &fd, gas)?;
            i += 1;
        }
    }
    if let Some(last) = frames.last() {
        let lights = match (gas_look.scatter, gas_grids.last()) {
            (Some(_), Some(Some(_))) => galaxy_render::cluster_lights(last),
            _ => Vec::new(),
        };
        if gas_look.scatter.is_some() {
            light_counts.push(lights.len());
        }
        let gas = gas_grids
            .last()
            .and_then(Option::as_ref)
            .map(|g| galaxy_render::GasFrame {
                grid0: g,
                grid1: g,
                // Static last frame: one moment grid, mixed with itself (mix 0).
                temperature: match (temp_params, gas_moments.last().and_then(Option::as_ref)) {
                    (Some(t), Some(m)) => Some(galaxy_render::TempColor {
                        moment0: m,
                        moment1: m,
                        cold: t.cold,
                        hot: t.hot,
                        u_lo: t.u_lo,
                        u_hi: t.u_hi,
                    }),
                    _ => None,
                },
                mix: 0.0,
                lights: &lights,
                look: gas_look,
            });
        emit(i, last, gas)?;
        i += 1;
    }
    println!(
        "rendered + graded {i} frames → {} in {:.1} s",
        frame_dir.display(),
        t_render.elapsed().as_secs_f64()
    );
    if !light_counts.is_empty() {
        let mut sorted = light_counts.clone();
        sorted.sort_unstable();
        let n = sorted.len();
        let sum: usize = sorted.iter().sum();
        println!(
            "scatter light clusters (octree K): min {} / median {} / mean {:.1} / max {} over {n} frames",
            sorted[0],
            sorted[n / 2],
            sum as f64 / n as f64,
            sorted[n - 1],
        );
    }

    // 4. ffmpeg → movie (optional; leaves PNGs if ffmpeg is absent).
    encode_movie(&frame_dir, &out.join("movie.mp4"));
    Ok(())
}

/// The effective prep config for one movie run (M6e): the scenario's base look,
/// the `--color` mode mapped onto a concrete `ColorMode`, and the star-formation
/// compression proxy — the last two anchored to this run's own snapshot 0 (frozen
/// initial-radius colors; reference densities ρ0). Everything reuses
/// `(DENSITY_K, s.eps)`, so `prepare`'s shared cache runs ONE O(N²) pass per
/// snapshot however many features are on.
fn effective_prep(s: &Scenario, color: ColorModeArg, snap0: &State) -> PrepConfig {
    let mut prep = s.prep.clone();
    prep.color = match color {
        ColorModeArg::Progenitor => ColorMode::Progenitor,
        ColorModeArg::InitialRadius => ColorMode::Frozen(initial_radius_colors(
            snap0,
            &RadialRamp {
                ramps: s.ramp.clone(),
            },
        )),
        ColorModeArg::Dispersion => ColorMode::Dispersion(DispersionColoring {
            k: DENSITY_K,
            softening: s.eps,
            cold: DISPERSION_COLD,
            hot: DISPERSION_HOT,
            // Ramp only the luminous disks by σ_v; the dark-matter halo keeps its
            // dim palette color (its large mass would otherwise swamp the frame —
            // the same set the SF proxy treats as luminous).
            luminous: s
                .sf_progenitors
                .iter()
                .filter(|&&p| p < 64)
                .fold(0u64, |m, &p| m | (1u64 << p)),
        }),
    };
    // Star-formation proxy, masked to the scenario's luminous progenitors: a 0.0
    // reference density is the gated "no estimate" sentinel, so masked particles
    // keep their base color bit-exactly whatever their compression.
    let mut rho0 = knn_density(&snap0.pos, DENSITY_K, s.eps);
    for (r0, p) in rho0.iter_mut().zip(&snap0.progenitor) {
        if !s.sf_progenitors.contains(&p.0) {
            *r0 = 0.0;
        }
    }
    prep.compression = Some(CompressionHue {
        k: DENSITY_K,
        softening: s.eps,
        rho0,
        young: SF_YOUNG,
        strength: SF_STRENGTH,
    });
    prep
}

/// Invoke ffmpeg to mux the PNG sequence into an H.264 movie. ffmpeg is an external
/// tool, not a build dependency — if it is missing, print the command and leave the
/// frames on disk (they are the durable artifact).
fn encode_movie(frame_dir: &Path, movie: &Path) {
    let pattern = frame_dir.join("frame_%05d.png");
    let args = [
        "-y",
        "-framerate",
        &FPS.to_string(),
        "-i",
        &pattern.to_string_lossy(),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        &movie.to_string_lossy(),
    ]
    .map(|s| s.to_string());

    match Command::new("ffmpeg").args(&args).status() {
        Ok(s) if s.success() => println!("movie → {}", movie.display()),
        Ok(s) => eprintln!(
            "ffmpeg exited with {s}; PNG frames remain in {}",
            frame_dir.display()
        ),
        Err(_) => {
            eprintln!(
                "ffmpeg not found on PATH — PNG frames are in {}",
                frame_dir.display()
            );
            eprintln!("to encode manually:\n  ffmpeg {}", args.join(" "));
        }
    }
}
