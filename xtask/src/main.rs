//! Movie orchestrator: builds a two-galaxy collision, steps it to snapshots, then
//! renderprep → render → grade → ffmpeg into a movie. Scenarios are **data**
//! (M6f): checked-in `scenario.toml` presets under `xtask/scenarios/` — the
//! originals (`disk`, `dm`, `cuspy`, gated to reproduce the pre-M6f hardcoded
//! constructors bit-for-bit) plus the Toomre encounter zoo (`retro`, `inclined`,
//! `bullseye`, `minor`) — or any user toml on the same schema (see `spec`).
//!
//! Usage: `cargo run -p galaxy-xtask --release [<preset>|<scenario.toml>] [out_dir]
//! [--color progenitor|initial-radius|dispersion] [--reuse-snapshots]`
//!   * A bare first arg that is no preset name (and not a `.toml` path) is taken as
//!     `out_dir` with the `disk` scenario (back-compat with the original CLI).
//!   * `regrade <exr_dir> <png_dir> [--exposure E] [--tonemap aces|reinhard|asinh]
//!     [--beta B] [--bloom S] [--bloom-levels N] [--bloom-radius R]` re-grades
//!     retained linear EXRs into fresh PNGs (+ movie if ffmpeg is present) in seconds
//!     — no re-simulation, no re-render (the M6a look loop; bloom added in M6b).
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

use galaxy_core::{LeapfrogKdk, State, StaticBackground};
use galaxy_grade::{grade_file, BloomConfig, GradeConfig, ToneMap};
use galaxy_render::camera::DEFAULT_MARGIN;
use galaxy_render::{smooth_envelope, write_exr, Camera, CameraPath, RenderConfig, Renderer};
use galaxy_renderprep::{
    initial_radius_colors, knn_density, prepare, subframe, ColorMode, CompressionHue,
    DispersionColoring, FrameData, HermiteSpan, PrepConfig, RadialRamp,
};
use galaxy_sim::{run, DirectorySink, SimConfig};
use galaxy_xtask::spec::{
    build_scenario, parse_scenario_toml, preset, Rig, Scenario, ScenarioSpec,
};
use galaxy_xtask::{
    framing_radius, parse_movie_args, parse_regrade_args, per_frame_radii, ColorModeArg,
    ScenarioArg, DEFAULT_BLOOM_LEVELS, DEFAULT_BLOOM_RADIUS, DENSITY_K, G,
};
use glam::Vec3;

// --- Shared render / grade look (all scenarios) --------------------------------
// The shared physics/look constants (G, kNN density tuning, splat-size clamps,
// frame sizes, subframe count) live in the lib (`galaxy_xtask`) so the M6f
// spec-driven builder shares them; the grade-side and mode-color knobs below are
// consumed only by this binary. Tuning provenance: DESIGN.md M3.6/M6a–M6e.
const THETA: f64 = 0.5; // Barnes-Hut opening angle
const FALLOFF: f32 = 6.0;
// M6e coloring. All kNN consumers reuse (DENSITY_K, scenario ε) so the O(N²)
// estimate runs once per snapshot no matter how many passes are on.
//   * Star-formation proxy (ON in every scenario): hue shift toward a young-
//     population blue-white, keyed on density compression ρ(t)/ρ(0) — only
//     tidally-compressed material lights up; undisturbed cores keep their color.
//     (A proxy: the sim is collisionless — see DESIGN M6e.)
//   * Size-by-density (ON): splat radius follows the local spacing (ρ_ref/ρ)^⅓,
//     clamped — tight cores, soft diffuse splats.
//   * σ_v ramp (--color dispersion): dynamically cold → blue, hot → red-orange
//     (the astro convention: cold thin disks are young/blue, hot spheroids old/red).
//     Scoped to SINGLE-POPULATION subjects (the dm merger): the ramp replaces the
//     palette, and with it the 20× halo/disk brightness compensation — on the
//     disk scenarios the ~5×-heavier, ~2×-more-numerous halo particles swamp the
//     frame white at any ramp brightness that still shows the disks (the first
//     rendered A/B). A palette-luminance-weighted σ_v ramp is the named follow-up
//     if the mode is ever wanted on a disk+halo scene.
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

    let quick = std::env::var_os("GALAXY_MOVIE_QUICK").is_some();

    let movie = parse_movie_args(&args).map_err(|e| {
        format!(
            "{e}\nusage: [<preset>|<scenario.toml>] [out_dir] \
             [--color progenitor|initial-radius|dispersion] [--reuse-snapshots]"
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
    run_movie(&scenario, &out, movie.color, movie.reuse_snapshots)
}

/// The M6a look loop: re-grade a directory of retained linear-HDR EXRs into PNGs
/// under a new exposure/tone curve, then (optionally) ffmpeg them into a movie next
/// to the frames. Seconds instead of a re-render, because the EXR is the pristine
/// linear artifact. The movie step assumes the pipeline's `frame_%05d` stems; other
/// stems still regrade fine, ffmpeg just skips them with its usual message.
fn regrade(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    const USAGE: &str = "usage: regrade <exr_dir> <png_dir> \
         [--exposure E] [--tonemap aces|reinhard|asinh] [--beta B] \
         [--bloom S] [--bloom-levels N] [--bloom-radius R]";
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
        let mut state = s.state.clone();
        let mut solver = galaxy_solvers::BarnesHut::new(G, s.eps, THETA);
        let mut integ = LeapfrogKdk::new();
        let bg = StaticBackground;
        let cfg = SimConfig {
            dt: s.dt,
            n_steps: s.n_steps,
            snapshot_every: s.snapshot_every,
            softening: s.eps,
            rng_seed: s.seed,
            config_hash: 0,
            units: "nbody-G1".to_string(),
        };
        let mut sink = DirectorySink::new(&snap_dir)?;
        let summary = run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink)?;
        println!(
            "simulated {} steps → {} snapshots (t_final = {:.2})",
            summary.steps, summary.snapshots_emitted, summary.final_time
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
    let prep = effective_prep(s, color, &states[0]);
    let frames: Vec<_> = states.iter().map(|st| prepare(st, &prep)).collect();
    println!("prepared {} endpoint frames", frames.len());

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
        Rig::Dolly { .. } => todo!("M6g: dolly rig wiring"),
    };
    let gcfg = GradeConfig {
        exposure: EXPOSURE,
        tonemap: TONEMAP,
        bloom: Some(BloomConfig {
            strength: BLOOM_STRENGTH,
            levels: DEFAULT_BLOOM_LEVELS,
            radius: DEFAULT_BLOOM_RADIUS,
        }),
    };
    let renderer = Renderer::new()?;

    // 4. Hermite temporal upsampling (M6c): `subframes` in-betweens per snapshot
    //    interval, plus the final snapshot itself → (n-1)·subframes + 1 frames.
    let total = match states.len() {
        0 | 1 => states.len(),
        n => (n - 1) * s.subframes as usize + 1,
    };
    let emit = |i: usize, frame: &FrameData| -> Result<(), Box<dyn std::error::Error>> {
        // The movie's unit timeline: frame i of `total` (a single-frame movie
        // sits at u = 0, the path start).
        let u = i as f32 / total.saturating_sub(1).max(1) as f32;
        let img = renderer.render_frame(frame, &path.camera_at(u), &rcfg)?;
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
    let mut i = 0;
    for w in 0..states.len().saturating_sub(1) {
        // The span validates the id/time gates once per snapshot pair (a silent
        // id mismatch would scramble the movie — fail loudly instead).
        let span = HermiteSpan::new(&states[w], &states[w + 1])?;
        for j in 0..s.subframes {
            let u = f64::from(j) / f64::from(s.subframes);
            emit(i, &subframe(&span, &frames[w], &frames[w + 1], u))?;
            i += 1;
        }
    }
    if let Some(last) = frames.last() {
        emit(i, last)?;
        i += 1;
    }
    println!("rendered + graded {i} frames → {}", frame_dir.display());

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
