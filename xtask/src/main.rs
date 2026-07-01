//! Movie orchestrator: builds a two-galaxy collision, steps it to snapshots, then
//! renderprep → render → grade → ffmpeg into a tidal-tail movie. The scenario is
//! hardcoded for the MVP (a `scenario.toml` front-end is a later addition).
//!
//! Usage: `cargo run -p galaxy-xtask --release [out_dir]`
//! Layout under `out_dir`: `snapshots/` `.snap`, `exr/` linear HDR, `frames/` PNGs,
//! `movie.mp4` (if ffmpeg is on PATH). The EXR layer is kept so the frames can be
//! regraded (different exposure/tonemap) without re-simulating or re-rendering.

use std::path::{Path, PathBuf};
use std::process::Command;

use galaxy_core::{LeapfrogKdk, StaticBackground};
use galaxy_grade::{grade_file, GradeConfig, ToneMap};
use galaxy_ic::{DiskCollision, ExponentialDisk, Plummer};
use galaxy_render::{write_exr, Camera, RenderConfig, Renderer};
use galaxy_renderprep::{prepare, PrepConfig};
use galaxy_sim::{run, DirectorySink, SimConfig};
use galaxy_xtask::framing_radius;
use glam::Vec3;

// --- Scenario: a parabolic (Toomre) coplanar-PROGRADE encounter of two rotating
//     exponential-disk galaxies — the IC that makes thin curved tidal tails (a
//     disk resonantly amplified in a prograde passage), which the earlier
//     two-Plummer movie physically could not. Each galaxy is a low-mass disk in a
//     live Plummer halo that carries most of the mass and stabilizes it. Both disks
//     default to prograde (spin +Z, co-rotating with the x–y orbit). The disks are
//     WARM (Toomre Q≈1.5): in-plane + vertical velocity dispersion balanced by the
//     asymmetric-drift rotation lag, so the disks resist the local fragmentation a
//     fully-cold (Q→0) disk suffers over the several orbits of the passage while
//     still amplifying the thin prograde tails.
const G: f64 = 1.0;
// Galaxy 1 (primary): halo + disk.
const HALO_M1: f64 = 1.0;
const HALO_A1: f64 = 1.0;
const DISK_M1: f64 = 0.15;
const DISK_RD1: f64 = 0.5;
// Galaxy 2 (secondary): a lighter disk galaxy.
const HALO_M2: f64 = 0.7;
const HALO_A2: f64 = 0.9;
const DISK_M2: f64 = 0.1;
const DISK_RD2: f64 = 0.45;
// Shared disk geometry (thin; truncated a few scale lengths out).
const DISK_HZ_FRAC: f64 = 0.1; // scale height = 0.1·Rd (thin disk)
const DISK_RMAX_FRAC: f64 = 4.0; // truncate at 4·Rd
const DISK_Q: f64 = 1.5; // Toomre warmth — Q>1 suppresses fragmentation, tails intact
const ECC: f64 = 1.0; // parabolic — the classic tidal-tail case
const PERI: f64 = 1.5;
const SEP: f64 = 8.0;
const EPS: f64 = 0.05;
const THETA: f64 = 0.5;
// Halos need enough particles for a smooth stabilizing potential; disks get many
// for tail detail (disk flux is set by disk MASS, not count, so extra disk
// particles buy resolution at no brightness cost).
const NH1: usize = 5000;
const ND1: usize = 5000;
const NH2: usize = 3500;
const ND2: usize = 3500;
const SEED: u64 = 0x00C0_FFEE;

// --- Time stepping ------------------------------------------------------------
const DT: f64 = 0.02;
const N_STEPS: u64 = 1500;
const SNAPSHOT_EVERY: u64 = 25; // → ~61 frames

// --- Look --------------------------------------------------------------------
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const FALLOFF: f32 = 6.0;
const SPLAT_SIZE: f32 = 0.12; // world units
const PEAK_BRIGHTNESS: f32 = 0.3; // per-DISK-particle peak, so dense cores saturate
const EXPOSURE: f32 = 1.0;
const FRAME_PERCENTILE: f32 = 0.98; // crop the top 2% escapers when framing
const FPS: u32 = 30;
// Four-species palette: the two DISKS carry full-magnitude two-tone hues (warm /
// cool) — they are the bright tails — while the two halos are dim tints, a soft
// background glow. Brightness scales with mass, and halo particles are ~10× more
// massive than disk particles, so the halo hues are pushed well below the disks'
// to keep the tails dominant. Order matches the progenitor tags: halo1=0, disk1=1,
// halo2=2, disk2=3.
const HALO1_COLOR: [f32; 3] = [0.05, 0.035, 0.025];
const DISK1_COLOR: [f32; 3] = [1.0, 0.5, 0.25]; // warm
const HALO2_COLOR: [f32; 3] = [0.025, 0.035, 0.05];
const DISK2_COLOR: [f32; 3] = [0.35, 0.6, 1.0]; // cool

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("galaxy_movie"));
    let snap_dir = out.join("snapshots");
    let exr_dir = out.join("exr");
    let frame_dir = out.join("frames");
    for d in [&snap_dir, &exr_dir, &frame_dir] {
        std::fs::create_dir_all(d)?;
    }
    println!("output → {}", out.display());

    // 1. Initial conditions: two rotating disk galaxies on a prograde encounter.
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
    // Default orientation is prograde/prograde (coplanar) — the cleanest-tail
    // passage. Set `collision.orient1/orient2` for retrograde or inclined disks.
    let collision = DiskCollision::new(galaxy1, galaxy2, ECC, PERI, SEP);
    let ic = collision.sample(NH1, ND1, NH2, ND2, SEED);
    // Brightness is tied to the DISK particle mass so a lone disk particle peaks
    // near PEAK_BRIGHTNESS and dense disk cores additively saturate to white; the
    // dim halo hues then keep the more-massive halo particles a faint background.
    let disk_particle_mass = DISK_M1 / ND1 as f64;
    println!(
        "IC: {} particles (halo {}+{}, disk {}+{}), disk particle mass {disk_particle_mass:.3e}",
        ic.len(),
        NH1,
        NH2,
        ND1,
        ND2,
    );

    // 2. Simulate → snapshots.
    let mut state = ic.clone();
    let mut solver = galaxy_solvers::BarnesHut::new(G, EPS, THETA);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = SimConfig {
        dt: DT,
        n_steps: N_STEPS,
        snapshot_every: SNAPSHOT_EVERY,
        softening: EPS,
        rng_seed: SEED,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    };
    let mut sink = DirectorySink::new(&snap_dir)?;
    let summary = run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink)?;
    println!(
        "simulated {} steps → {} snapshots (t_final = {:.2})",
        summary.steps, summary.snapshots_emitted, summary.final_time
    );

    // 3. Renderprep: snapshot → frame-data. Scale brightness so a lone particle
    //    peaks near PEAK_BRIGHTNESS and dense cores additively saturate to white.
    let prep = PrepConfig {
        palette: vec![HALO1_COLOR, DISK1_COLOR, HALO2_COLOR, DISK2_COLOR],
        brightness_per_mass: PEAK_BRIGHTNESS / disk_particle_mass as f32,
        size: SPLAT_SIZE,
    };
    let mut snaps: Vec<PathBuf> = std::fs::read_dir(&snap_dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "snap"))
        .collect();
    snaps.sort(); // snapshot_<step:08>.snap → lexicographic == step order
    let frames: Vec<_> = snaps
        .iter()
        .map(|p| galaxy_io::read_file(p).map(|(_, s)| prepare(&s, &prep)))
        .collect::<Result<Vec<_>, _>>()?;
    println!("prepared {} frames", frames.len());

    // 4. One stable camera over the whole run, then render + grade each frame.
    //    Centered on the origin (the zero-COM barycenter) and sized to a robust
    //    percentile radius so a few escapers don't shrink the galaxies to dots.
    let rcfg = RenderConfig {
        width: WIDTH,
        height: HEIGHT,
        falloff: FALLOFF,
    };
    let radius = framing_radius(&frames, FRAME_PERCENTILE).max(1e-3);
    println!(
        "framing radius (p{:.0}) = {radius:.2}",
        FRAME_PERCENTILE * 100.0
    );
    let camera = Camera::face_on(Vec3::splat(-radius), Vec3::splat(radius), rcfg.aspect());
    let gcfg = GradeConfig {
        exposure: EXPOSURE,
        tonemap: ToneMap::AcesApprox,
    };
    let renderer = Renderer::new()?;

    for (i, frame) in frames.iter().enumerate() {
        let img = renderer.render_frame(frame, &camera, &rcfg)?;
        if i == frames.len() / 2 {
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
    }
    println!(
        "rendered + graded {} frames → {}",
        frames.len(),
        frame_dir.display()
    );

    // 5. ffmpeg → movie (optional; leaves PNGs if ffmpeg is absent).
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
