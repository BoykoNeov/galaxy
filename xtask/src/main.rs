//! Movie orchestrator: builds a two-galaxy collision, steps it to snapshots, then
//! renderprep → render → grade → ffmpeg into a movie. Two hardcoded scenarios (a
//! `scenario.toml` front-end is a later addition):
//!
//!   * `disk` (default) — a parabolic prograde encounter of two warm exponential-disk
//!     galaxies in live Plummer halos → thin curved **tidal tails** (the M3 demo).
//!   * `dm` — a 2:1 major **dark-matter merger** of two exponentially-truncated NFW
//!     halos (ρ∝r⁻¹ cusps) → a single triaxial remnant (the M5e payoff).
//!   * `cuspy` — the M5g payoff: a parabolic prograde encounter of two *cold* disks in
//!     live **cuspy** (exponentially-truncated NFW) halos → tidal tails on the
//!     realistic rising-to-flat rotation curve (the disk analogue of `dm`, and the
//!     cuspy analogue of `disk`).
//!
//! Usage: `cargo run -p galaxy-xtask --release [disk|dm|cuspy] [out_dir]`
//!   * A bare first arg that is none of `disk`/`dm`/`cuspy` is taken as `out_dir` with
//!     the `disk` scenario (back-compat with the original single-scenario CLI).
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

use std::path::{Path, PathBuf};
use std::process::Command;

use galaxy_core::{LeapfrogKdk, State, StaticBackground};
use galaxy_grade::{grade_file, BloomConfig, GradeConfig, ToneMap};
use galaxy_ic::{DiskCollision, ExponentialDisk, Nfw, NfwCollision, Plummer, TruncatedNfw};
use galaxy_render::camera::DEFAULT_MARGIN;
use galaxy_render::{smooth_envelope, write_exr, Camera, CameraPath, RenderConfig, Renderer};
use galaxy_renderprep::{prepare, subframe, DensityColoring, FrameData, HermiteSpan, PrepConfig};
use galaxy_sim::{run, DirectorySink, SimConfig};
use galaxy_xtask::{
    framing_radius, parse_regrade_args, per_frame_radii, DEFAULT_BLOOM_LEVELS, DEFAULT_BLOOM_RADIUS,
};
use glam::Vec3;

// --- Shared physics / look (both scenarios) ----------------------------------
const G: f64 = 1.0;
const THETA: f64 = 0.5; // Barnes-Hut opening angle
const FALLOFF: f32 = 6.0;
const PEAK_BRIGHTNESS: f32 = 0.3; // per-particle peak, so dense cores additively saturate

// Density-aware brightness boost (M3.6 estimator, switched ON and tuned in M6a).
// Tuned by A/B on rendered QUICK cuspy frames (strengths 0/1.5/3/6, DESIGN M6a):
// the mean reference ρ_ref is dominated by the dense inner disk, so the boost acts
// on nuclei and inner knots — strength 3 makes them read as bright cores, 1.5 is
// invisible (the boosted pixels were already tone-curve-saturated), 6 blows the
// nuclei into blobs. k = 32 (top of the documented 8–32 band) halves the estimator's
// shot noise vs 16 for negligible cost — temporal stability matters in a movie. The
// kNN distance floor reuses each scenario's force softening ε — the smallest
// separation the sim itself resolves.
const DENSITY_K: usize = 32;
const DENSITY_STRENGTH: f32 = 3.0;
const EXPOSURE: f32 = 1.0;
const TONEMAP: ToneMap = ToneMap::AcesApprox;
// Bloom (M6b), ON by default in all three scenarios. Strength tuned by A/B regrades
// of retained QUICK EXRs (0 / 0.3 / 0.45 / 0.6 / 1.2, cuspy under asinh exposure 4 +
// disk/dm under the ACES movie default): 0.3 is timid, 0.6 starts to haze the dense
// cuspy halo field, 1.2 washes out structure; 0.45 makes nuclei and knots glow while
// tails and halo dots stay resolved. Levels/radius are the documented CLI defaults.
const BLOOM_STRENGTH: f32 = 0.45;
// Hermite temporal upsampling (M6c): snapshots store full phase space, so cubic
// Hermite in-betweens are physically informed and cost no sim time. 8 subframes
// per snapshot at 60 fps turns the ~2 s / 30 fps flipbook into a ~8 s smooth
// movie (playback slows 4x per unit sim time; pericenter reads as continuous).
const SUBFRAMES: u32 = 8;
const FPS: u32 = 60;
const FRAME_W: u32 = 1280;
const FRAME_H: u32 = 720;
const QUICK_W: u32 = 640;
const QUICK_H: u32 = 360;

/// Everything a scenario hands the pipeline: the sampled IC plus the sim-timing,
/// softening, splat look, and framing that differ between the disk and DM movies.
/// The pipeline (`run_movie`) is single-sourced over this so both scenarios share
/// one sim→prep→render→grade→ffmpeg path.
struct Scenario {
    state: State,
    prep: PrepConfig,
    eps: f64,
    dt: f64,
    n_steps: u64,
    snapshot_every: u64,
    /// Hermite in-between frames per snapshot interval (M6c); 1 = no upsampling.
    subframes: u32,
    seed: u64,
    width: u32,
    height: u32,
    frame_percentile: f32,
    rig: Rig,
    info: String,
}

/// Per-scenario camera choreography (M6d). `Static` is the pre-M6d behavior:
/// one face-on framing over the whole run, bit-exact with the old pipeline.
enum Rig {
    Static,
    /// Eased azimuth/tilt sweep (degrees, start → end) with a breathing zoom:
    /// per-snapshot percentile radii smoothed by a ±`window`-snapshot envelope.
    OrbitTilt {
        azimuth_deg: (f32, f32),
        tilt_deg: (f32, f32),
        window: usize,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    // `regrade` is a pure grade-stage pass over retained EXRs — no sim, no GPU.
    if args.first().map(String::as_str) == Some("regrade") {
        return regrade(&args[1..]);
    }

    let quick = std::env::var_os("GALAXY_MOVIE_QUICK").is_some();

    // First positional selects the scenario; anything else is treated as the out dir
    // (back-compat with the original `... [out_dir]` CLI, which defaulted to disk).
    let (scenario_name, out_arg): (&str, Option<&str>) = match args.first().map(String::as_str) {
        Some("disk") => ("disk", args.get(1).map(String::as_str)),
        Some("dm") | Some("nfw") => ("dm", args.get(1).map(String::as_str)),
        Some("cuspy") | Some("disk-nfw") => ("cuspy", args.get(1).map(String::as_str)),
        Some(other) => ("disk", Some(other)),
        None => ("disk", None),
    };
    let out: PathBuf = out_arg.map(PathBuf::from).unwrap_or_else(|| {
        std::env::temp_dir().join(match scenario_name {
            "dm" => "galaxy_dm_merger",
            "cuspy" => "galaxy_cuspy_disk",
            _ => "galaxy_movie",
        })
    });

    println!(
        "scenario = {scenario_name}{}",
        if quick { " (quick preview)" } else { "" }
    );
    println!("output → {}", out.display());

    let scenario = match scenario_name {
        "dm" => dm_scenario(quick),
        "cuspy" => cuspy_scenario(quick),
        _ => disk_scenario(quick),
    };
    println!("{}", scenario.info);
    run_movie(&scenario, &out)
}

// --- Scenario: DM merger ------------------------------------------------------
// Two exponentially-truncated NFW halos (M5d) on a parabolic (Toomre) encounter — a
// 2:1 major dark-matter merger. Both are pure ρ∝r⁻¹ cusps with no disk, so the movie
// shows two cuspy blobs coalescing into one triaxial remnant — NOT thin tidal tails
// (those need cold disks). The passage is DEEP (peri=3 ≪ r_vir≈10) so the halos fully
// overlap at closest approach; dynamical friction in that overlap is what binds a
// marginally-bound (e=1) pair into a single remnant on the first passage.
const DM_HALO1_COLOR: [f32; 3] = [1.0, 0.55, 0.3]; // warm (primary)
const DM_HALO2_COLOR: [f32; 3] = [0.35, 0.6, 1.0]; // cool (secondary)
const DM_SPLAT_SIZE: f32 = 0.6; // world units — the NFW scene (~40u) is ~8× the disk scene
const DM_ECC: f64 = 1.0; // parabolic — the classic Toomre encounter
const DM_PERI: f64 = 3.0;
const DM_SEP: f64 = 40.0; // > r_vir1 + r_vir2 (=18) so the halos start on a clean approach
const DM_EPS: f64 = 0.05; // 0.05·r_s (r_s=1) — matches the NFW stability test's softening

fn dm_scenario(quick: bool) -> Scenario {
    // Primary: M_vir=1, r_s=1, c=10 ⇒ r_vir=10, exponential skirt r_d=3.
    let g1 = TruncatedNfw::new(Nfw::new(G, 1.0, 1.0, 10.0), 3.0);
    // Secondary: half the virial mass (2:1 major merger), r_s=0.8 ⇒ r_vir=8, r_d=2.4.
    let g2 = TruncatedNfw::new(Nfw::new(G, 0.5, 0.8, 10.0), 2.4);
    let collision = NfwCollision::new(g1, g2, DM_ECC, DM_PERI, DM_SEP);

    // Particle counts split 2:1 to match the mass ratio ⇒ EQUAL particle mass across
    // both halos (clean, uniform brightness weighting). Quick mode drops N ~6×.
    let (n1, n2) = if quick { (2000, 1000) } else { (12000, 6000) };
    let seed = 0x0DEA_D000;
    let state = collision.sample(n1, n2, seed);
    let particle_mass = g1.total_mass() / n1 as f64; // = g2.total_mass()/n2 by design

    // Timing: t_dyn≈1.2 (inner NFW scale). dt=0.02 ≈ 0.016·t_dyn resolves the deep
    // pericenter passage (a bit tighter than the stability test's 0.025·t_dyn). Total
    // T = n_steps·dt = 320 carries the run past first pericenter (t_peri≈104 for this
    // orbit, Barker's equation) through the second infall to full coalescence into a
    // single triaxial remnant — the halos are bound (dynamical friction robs the
    // orbital energy on the deep, fully-overlapping first passage).
    let dt = 0.02;
    let n_steps = 16_000;
    let snapshot_every = if quick { 400 } else { 200 }; // ~40 / ~80 snapshots

    let (width, height) = if quick {
        (QUICK_W, QUICK_H)
    } else {
        (FRAME_W, FRAME_H)
    };
    let info = format!(
        "IC: {} particles (halo1 {} + halo2 {}), particle mass {particle_mass:.3e}; \
         parabolic peri={DM_PERI} sep={DM_SEP} (r_vir 10+8), t_peri≈104, T={:.0}",
        n1 + n2,
        n1,
        n2,
        n_steps as f64 * dt,
    );

    Scenario {
        state,
        prep: PrepConfig {
            palette: vec![DM_HALO1_COLOR, DM_HALO2_COLOR],
            brightness_per_mass: PEAK_BRIGHTNESS / particle_mass as f32,
            size: DM_SPLAT_SIZE,
            density: Some(DensityColoring {
                k: DENSITY_K,
                softening: DM_EPS,
                strength: DENSITY_STRENGTH,
            }),
        },
        eps: DM_EPS,
        dt,
        n_steps,
        snapshot_every,
        subframes: SUBFRAMES,
        seed,
        width,
        height,
        // The diffuse skirt + a few post-merger escapers would blow up the AABB; a
        // slightly lower percentile than the disk movie crops them and keeps the
        // remnant filling the frame.
        frame_percentile: 0.97,
        // The remnant is TRIAXIAL — a half-turn orbit at a fixed ¾ tilt is what
        // shows it (a static face-on view reads as a round blob). Window ±6
        // snapshots ≈ the merger's dynamical time at this cadence.
        rig: Rig::OrbitTilt {
            azimuth_deg: (-90.0, 90.0),
            tilt_deg: (60.0, 60.0),
            window: 6,
        },
        info,
    }
}

// --- Scenario: disk collision (the original M3 movie) -------------------------
// A parabolic coplanar-PROGRADE encounter of two rotating warm exponential-disk
// galaxies (Toomre Q≈1.5), each a low-mass disk in a live Plummer halo → thin curved
// tidal tails. See the git history for the full physics rationale; this is the
// original hardcoded scenario, unchanged (same constants + deterministic pipeline ⇒
// same frames when not in quick mode), now behind the `disk` selector.
const HALO_M1: f64 = 1.0;
const HALO_A1: f64 = 1.0;
const DISK_M1: f64 = 0.15;
const DISK_RD1: f64 = 0.5;
const HALO_M2: f64 = 0.7;
const HALO_A2: f64 = 0.9;
const DISK_M2: f64 = 0.1;
const DISK_RD2: f64 = 0.45;
const DISK_HZ_FRAC: f64 = 0.1;
const DISK_RMAX_FRAC: f64 = 4.0;
const DISK_Q: f64 = 1.5;
const DISK_ECC: f64 = 1.0;
const DISK_PERI: f64 = 1.5;
const DISK_SEP: f64 = 8.0;
const DISK_EPS: f64 = 0.05;
const DISK_SPLAT_SIZE: f32 = 0.12;
const HALO1_COLOR: [f32; 3] = [0.05, 0.035, 0.025];
const DISK1_COLOR: [f32; 3] = [1.0, 0.5, 0.25]; // warm
const HALO2_COLOR: [f32; 3] = [0.025, 0.035, 0.05];
const DISK2_COLOR: [f32; 3] = [0.35, 0.6, 1.0]; // cool

fn disk_scenario(quick: bool) -> Scenario {
    let galaxy1 = ExponentialDisk::new(
        DISK_M1,
        DISK_RD1,
        DISK_HZ_FRAC * DISK_RD1,
        DISK_RMAX_FRAC * DISK_RD1,
        Plummer::new(G, HALO_M1, HALO_A1),
    )
    .with_toomre_q(DISK_Q);
    let galaxy2 = ExponentialDisk::new(
        DISK_M2,
        DISK_RD2,
        DISK_HZ_FRAC * DISK_RD2,
        DISK_RMAX_FRAC * DISK_RD2,
        Plummer::new(G, HALO_M2, HALO_A2),
    )
    .with_toomre_q(DISK_Q);
    let collision = DiskCollision::new(galaxy1, galaxy2, DISK_ECC, DISK_PERI, DISK_SEP);

    // Halos need enough particles for a smooth stabilizing potential; disks get many
    // for tail detail (disk flux is set by disk MASS, not count). Quick mode drops N.
    let (nh1, nd1, nh2, nd2) = if quick {
        (1500, 1500, 1000, 1000)
    } else {
        (5000, 5000, 3500, 3500)
    };
    let seed = 0x00C0_FFEE;
    let state = collision.sample(nh1, nd1, nh2, nd2, seed);
    let disk_particle_mass = DISK_M1 / nd1 as f64;

    let (width, height) = if quick {
        (QUICK_W, QUICK_H)
    } else {
        (FRAME_W, FRAME_H)
    };
    let info = format!(
        "IC: {} particles (halo {}+{}, disk {}+{}), disk particle mass {disk_particle_mass:.3e}",
        state.len(),
        nh1,
        nh2,
        nd1,
        nd2,
    );

    Scenario {
        state,
        prep: PrepConfig {
            palette: vec![HALO1_COLOR, DISK1_COLOR, HALO2_COLOR, DISK2_COLOR],
            brightness_per_mass: PEAK_BRIGHTNESS / disk_particle_mass as f32,
            size: DISK_SPLAT_SIZE,
            density: Some(DensityColoring {
                k: DENSITY_K,
                softening: DISK_EPS,
                strength: DENSITY_STRENGTH,
            }),
        },
        eps: DISK_EPS,
        dt: 0.02,
        n_steps: 1500,
        snapshot_every: 25, // → ~61 snapshots
        subframes: SUBFRAMES,
        seed,
        width,
        height,
        frame_percentile: 0.98,
        // The original M3 movie keeps its static face-on framing — the back-compat
        // exemplar (same constants + deterministic pipeline ⇒ same frames).
        rig: Rig::Static,
        info,
    }
}

// --- Scenario: cuspy-disk collision (the M5g payoff) --------------------------
// The disk analogue of the `dm` merger and the cuspy analogue of the `disk` movie:
// two rotating exponential disks, each embedded in a live *cuspy* exponentially-
// truncated NFW halo (ρ∝r⁻¹), on a parabolic coplanar-PROGRADE encounter → thin
// curved tidal tails riding on the realistic rising-to-flat rotation curve.
//
// The disks are COLD (no Toomre warmth). The warm knob the Plummer `disk` movie uses
// to survive several orbits is deliberately unavailable here: the warm dispersions
// read the halo density ρ(r), which diverges at an NFW cusp (DESIGN.md, M5f — warm-
// in-a-cusp is a scoped follow-up). The stabilization is therefore *resolution*: a
// cold cuspy disk over-rotates and flies apart if the live halo's inner N-body force
// falls below the analytic G·M(<r)/r² the disk is placed on, so the halos get many
// particles and a small softening (ε≈0.02·r_s), per the M5f cusp-resolution finding.
// This makes the scenario markedly heavier than the cored `disk` movie — QUICK mode
// keeps the halo N high enough to still resolve the cusp for a faithful preview.
const CUSPY_HALO_MVIR1: f64 = 1.0;
const CUSPY_HALO_RS1: f64 = 1.0;
const CUSPY_HALO_C1: f64 = 10.0; // ⇒ r_vir = 10
const CUSPY_HALO_RD1: f64 = 3.0; // exponential skirt scale
const CUSPY_DISK_M1: f64 = 0.12;
const CUSPY_DISK_RD1: f64 = 0.6;
const CUSPY_HALO_MVIR2: f64 = 0.7;
const CUSPY_HALO_RS2: f64 = 0.9;
const CUSPY_HALO_C2: f64 = 10.0; // ⇒ r_vir = 9
const CUSPY_HALO_RD2: f64 = 2.7;
const CUSPY_DISK_M2: f64 = 0.08;
const CUSPY_DISK_RD2: f64 = 0.5;
const CUSPY_DISK_HZ_FRAC: f64 = 0.1;
const CUSPY_DISK_RMAX_FRAC: f64 = 3.0;
const CUSPY_ECC: f64 = 1.0; // parabolic — the classic Toomre encounter
const CUSPY_PERI: f64 = 1.5;
const CUSPY_SEP: f64 = 8.0;
const CUSPY_EPS: f64 = 0.02; // 0.02·r_s — between the disk (0.05) and the M5f deep-cusp 0.01
const CUSPY_SPLAT_SIZE: f32 = 0.15;

fn cuspy_scenario(quick: bool) -> Scenario {
    let halo1 = TruncatedNfw::new(
        Nfw::new(G, CUSPY_HALO_MVIR1, CUSPY_HALO_RS1, CUSPY_HALO_C1),
        CUSPY_HALO_RD1,
    );
    let halo2 = TruncatedNfw::new(
        Nfw::new(G, CUSPY_HALO_MVIR2, CUSPY_HALO_RS2, CUSPY_HALO_C2),
        CUSPY_HALO_RD2,
    );
    // COLD disks — no `with_toomre_q` (warm dispersion diverges at the cusp).
    let galaxy1 = ExponentialDisk::new(
        CUSPY_DISK_M1,
        CUSPY_DISK_RD1,
        CUSPY_DISK_HZ_FRAC * CUSPY_DISK_RD1,
        CUSPY_DISK_RMAX_FRAC * CUSPY_DISK_RD1,
        halo1,
    );
    let galaxy2 = ExponentialDisk::new(
        CUSPY_DISK_M2,
        CUSPY_DISK_RD2,
        CUSPY_DISK_HZ_FRAC * CUSPY_DISK_RD2,
        CUSPY_DISK_RMAX_FRAC * CUSPY_DISK_RD2,
        halo2,
    );
    let collision = DiskCollision::new(galaxy1, galaxy2, CUSPY_ECC, CUSPY_PERI, CUSPY_SEP);

    // The cusp must be RESOLVED (M5f), so the halos are particle-heavy even in QUICK
    // mode (a low-N halo under-resolves the inner force and the cold disk blows out —
    // that would make a QUICK preview a false negative). Disks get many for tail detail.
    let (nh1, nd1, nh2, nd2) = if quick {
        (5000, 3000, 4000, 2000)
    } else {
        (10000, 5000, 8000, 4000)
    };
    let seed = 0x0CA5_D15C;
    let state = collision.sample(nh1, nd1, nh2, nd2, seed);
    let disk_particle_mass = CUSPY_DISK_M1 / nd1 as f64;

    let (width, height) = if quick {
        (QUICK_W, QUICK_H)
    } else {
        (FRAME_W, FRAME_H)
    };
    let info = format!(
        "IC: {} particles (cuspy halo {}+{}, cold disk {}+{}), disk particle mass \
         {disk_particle_mass:.3e}; parabolic peri={CUSPY_PERI} sep={CUSPY_SEP}, eps={CUSPY_EPS}",
        state.len(),
        nh1,
        nh2,
        nd1,
        nd2,
    );

    Scenario {
        state,
        prep: PrepConfig {
            palette: vec![HALO1_COLOR, DISK1_COLOR, HALO2_COLOR, DISK2_COLOR],
            brightness_per_mass: PEAK_BRIGHTNESS / disk_particle_mass as f32,
            size: CUSPY_SPLAT_SIZE,
            density: Some(DensityColoring {
                k: DENSITY_K,
                softening: CUSPY_EPS,
                strength: DENSITY_STRENGTH,
            }),
        },
        eps: CUSPY_EPS,
        dt: 0.02,
        n_steps: 1500,
        snapshot_every: 25, // → ~61 snapshots
        subframes: SUBFRAMES,
        seed,
        width,
        height,
        // The cuspy halo is far larger than the disk (r_vir=10 vs disk r_max≈1.8), so a
        // high percentile would frame on the diffuse halo and shrink the tails to dots.
        // A lower percentile crops the halo skirt and keeps the disk + tails filling the
        // frame (the dim halo still glows underneath).
        frame_percentile: 0.7,
        // The M6d choreography: start ¾-inclined (the 3-D structure face-on
        // flattens away), orbit slowly through first pericenter, and settle
        // toward face-on as the tails extend (tails read best face-on). The
        // zoom breathes via the ±8-snapshot envelope as the tails fling out.
        rig: Rig::OrbitTilt {
            azimuth_deg: (-90.0, 40.0),
            tilt_deg: (55.0, 25.0),
            window: 8,
        },
        info,
    }
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

/// The scenario-independent pipeline: simulate the IC to snapshots, renderprep every
/// snapshot to frame-data, build the scenario's camera path (static framing or the
/// M6d orbit/tilt rig), then render + grade each frame and (optionally) ffmpeg them
/// into a movie.
fn run_movie(s: &Scenario, out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let snap_dir = out.join("snapshots");
    let exr_dir = out.join("exr");
    let frame_dir = out.join("frames");
    for d in [&snap_dir, &exr_dir, &frame_dir] {
        std::fs::create_dir_all(d)?;
    }

    // 1. Simulate → snapshots.
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

    // 2. Renderprep on the SNAPSHOT cadence: the full prepare (including the O(N²)
    //    kNN density pass) runs only on snapshot states; the Hermite subframes below
    //    lerp these endpoint attributes (M6c decision — density evolves on the
    //    snapshot timescale, so per-subframe kNN would cost minutes for no gain).
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(&snap_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "snap"))
        .collect();
    snaps.sort(); // snapshot_<step:08>.snap → lexicographic == step order
    let states: Vec<State> = snaps
        .iter()
        .map(|p| galaxy_io::read_file(p).map(|(_, st)| st))
        .collect::<Result<Vec<_>, _>>()?;
    let frames: Vec<_> = states.iter().map(|st| prepare(st, &s.prep)).collect();
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
