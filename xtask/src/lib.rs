//! `galaxy-xtask`: the pipeline orchestrator (scenario → sim → renderprep → render
//! → grade → ffmpeg). The binary is the glue; this lib holds the pure, testable bits.

pub mod cfl_guard;
pub mod simulate;
pub mod spec;

use std::path::PathBuf;

use crate::simulate::Backend;
use galaxy_core::{DVec3, ForceSolver, Species, State};
use galaxy_grade::{BloomConfig, GradeConfig, LocalToneConfig, ToneMap};
use galaxy_renderprep::{FrameData, SigmaReference};
use galaxy_solvers::{BarnesHut, DirectSum};
use glam::Vec3;

// --- Shared physics / look constants (every scenario) --------------------------
// Scenario-independent pipeline constants; the per-scenario knobs live in the
// `scenario.toml` presets (see `spec`). Tuning provenance: DESIGN.md M3.6/M6a–M6e.

/// Gravitational constant of the N-body unit system (G = 1).
pub const G: f64 = 1.0;
/// Barnes-Hut opening angle θ for the movie pipeline's gravity solver.
pub const THETA: f64 = 0.5;
/// Per-particle peak brightness, so dense cores additively saturate.
pub const PEAK_BRIGHTNESS: f32 = 0.3;
/// kNN neighbour count for every density-driven feature (M3.6/M6a tuning).
pub const DENSITY_K: usize = 32;
/// Density brightness-boost strength (M6a tuning).
pub const DENSITY_STRENGTH: f32 = 3.0;
/// Size-by-density clamp: smallest splat as a fraction of the base size (M6e).
pub const SIZE_MIN_FRAC: f32 = 0.6;
/// Size-by-density clamp: largest splat as a fraction of the base size (M6e).
pub const SIZE_MAX_FRAC: f32 = 1.6;
/// Hermite in-between frames per snapshot interval (M6c).
pub const SUBFRAMES: u32 = 8;
/// Full-resolution frame size.
pub const FRAME_W: u32 = 1280;
/// Full-resolution frame size.
pub const FRAME_H: u32 = 720;
/// `GALAXY_MOVIE_QUICK=1` preview frame size.
pub const QUICK_W: u32 = 640;
/// `GALAXY_MOVIE_QUICK=1` preview frame size.
pub const QUICK_H: u32 = 360;

/// Default asinh softening knob β for `regrade --tonemap asinh` when `--beta` is not
/// given. A tuning constant, eyeballed against the rendered collision frames (M6a).
pub const DEFAULT_ASINH_BETA: f32 = 0.2;

/// Default bloom mip-pyramid depth for `regrade --bloom S` when `--bloom-levels` is
/// not given: 5 octaves span halo scales from ~2 px up to ~the frame at 720p (M6b).
pub const DEFAULT_BLOOM_LEVELS: u32 = 5;

/// Default per-level Gaussian σ (pixels) for `regrade --bloom S` when
/// `--bloom-radius` is not given (M6b).
pub const DEFAULT_BLOOM_RADIUS: f32 = 2.0;

/// Default surround σ (pixels) for `regrade --local S` when `--local-radius` is
/// not given: a broad low-pass that reads the additive-splat blob as one
/// surround without collapsing to the whole-frame mean. Starting value —
/// settled by the white-blob A/B.
pub const DEFAULT_LOCAL_RADIUS: f32 = 32.0;

/// Default gain floor for `regrade --local S` when `--local-floor` is not given:
/// caps the darkening at 5× (bounds the dark-halo ring). Starting value —
/// settled by the white-blob A/B.
pub const DEFAULT_LOCAL_FLOOR: f32 = 0.2;

/// Which M6e coloring mode a movie invocation asked for (`--color`). This is the
/// CLI-level selector; the scenario maps it onto a concrete `renderprep::ColorMode`
/// (frozen ramp colors need snapshot 0, which only the pipeline has).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum ColorModeArg {
    /// The flat progenitor palette — the pre-M6e look (default).
    #[default]
    Progenitor,
    /// Frozen initial-radius ramp, computed from snapshot 0.
    InitialRadius,
    /// Per-frame velocity-dispersion (σ_v) ramp.
    Dispersion,
}

/// Which σ_v → color endpoints the `--color dispersion` ramp uses
/// (`--dispersion-palette`). Independent of [`SigmaReference`] (the scale
/// population) — one dial picks the hues, the other picks the scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DispersionPalette {
    /// Blue tidal-tail arms (the default): dynamically COLD stars (low σ_v — the
    /// coherent streams and outer disk) read young→blue, HOT stars (the bulge)
    /// read old→red. The REVERSE of the gas blackbody scale, so stars and gas no
    /// longer share one convention. Aesthetic (loosely the age–σ relation).
    #[default]
    BlueArms,
    /// Blackbody direction, matching the temperature-colored gas so stars and gas
    /// share ONE scale: cold → red, hot → blue-white.
    Blackbody,
}

/// Which scenario a movie invocation selected: a checked-in preset by canonical
/// name, or a user-supplied `scenario.toml` path (M6f front-end).
#[derive(Clone, Debug, PartialEq)]
pub enum ScenarioArg {
    /// A named preset from [`spec::PRESETS`] (aliases resolved to canonical).
    Preset(String),
    /// A path to a custom `scenario.toml` (any first positional ending `.toml`).
    Path(PathBuf),
}

/// A parsed movie invocation: scenario selector, optional output dir, coloring
/// mode, and whether to reuse retained snapshots instead of re-simulating.
#[derive(Clone, Debug, PartialEq)]
pub struct MovieArgs {
    /// The selected scenario (preset name or custom toml path).
    pub scenario: ScenarioArg,
    /// Output directory; `None` means the scenario's default temp location.
    pub out_dir: Option<PathBuf>,
    /// The M6e coloring mode (default: progenitor palette).
    pub color: ColorModeArg,
    /// σ_v ramp endpoints for `--color dispersion` (`--dispersion-palette`,
    /// default [`DispersionPalette::BlueArms`]). Ignored by the other color modes.
    pub dispersion_palette: DispersionPalette,
    /// Which population fixes the σ_v scale for `--color dispersion`
    /// (`--dispersion-reference`, default [`SigmaReference::Full`]). Ignored by the
    /// other color modes.
    pub dispersion_reference: SigmaReference,
    /// Skip the simulation and read existing `snapshots/*.snap` under the out dir
    /// (errors downstream if none exist — reuse is an explicit promise).
    pub reuse_snapshots: bool,
    /// Which force backend runs the gas-rich simulate step (`--gpu` ⇒ the GPU-resident
    /// SPH stepper, G6). Ignored for a gas-free scenario (always CPU Barnes-Hut).
    pub backend: Backend,
}

/// Map movie CLI arguments (everything except a leading `regrade`) to a
/// [`MovieArgs`]: `[<preset>|<path/to/scenario.toml>] [out_dir]
/// [--color progenitor|initial-radius|dispersion] [--reuse-snapshots]`,
/// where `<preset>` is any [`spec::PRESETS`] name.
///
/// Back-compat rules preserved from the original positional CLI: `nfw` and
/// `disk-nfw` are aliases for `dm` / `cuspy`; a bare first positional that is no
/// scenario name (and not a `.toml` path) is taken as the out dir with the `disk`
/// scenario. Flags may come in any order. Errors (human-readable) on a third
/// positional, unknown flags, unknown color names, or `--color` without a value.
pub fn parse_movie_args(args: &[String]) -> Result<MovieArgs, String> {
    let mut positionals: Vec<&str> = Vec::new();
    let mut color = ColorModeArg::default();
    // NOTE (red): the two dispersion dials are not parsed yet — they take their
    // defaults, so `movie_parses_dispersion_dials` is red until the flags are wired.
    let dispersion_palette = DispersionPalette::default();
    let dispersion_reference = SigmaReference::default();
    let mut reuse_snapshots = false;
    let mut backend = Backend::default();

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--color" => {
                let name = it
                    .next()
                    .ok_or("--color needs a value: progenitor|initial-radius|dispersion")?;
                color = match name.as_str() {
                    "progenitor" => ColorModeArg::Progenitor,
                    "initial-radius" => ColorModeArg::InitialRadius,
                    "dispersion" => ColorModeArg::Dispersion,
                    other => {
                        return Err(format!(
                            "unknown color mode `{other}` (progenitor|initial-radius|dispersion)"
                        ))
                    }
                };
            }
            "--reuse-snapshots" => reuse_snapshots = true,
            "--gpu" => backend = Backend::Gpu,
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown flag `{flag}` (expected --color, --reuse-snapshots, --gpu)"
                ));
            }
            positional => positionals.push(positional),
        }
    }

    // First positional: a preset name/alias or a scenario.toml path, else the out
    // dir under the original single-scenario CLI (`xtask <out_dir>` defaulted to
    // the disk movie).
    let disk = || ScenarioArg::Preset("disk".to_string());
    let (scenario, out_dir) = match positionals.as_slice() {
        [] => (disk(), None),
        [one] => match resolve_scenario(one) {
            Some(s) => (s, None),
            None => (disk(), Some(*one)),
        },
        [scenario, out] => (
            resolve_scenario(scenario).ok_or_else(|| unknown_scenario(scenario))?,
            Some(*out),
        ),
        more => {
            return Err(format!(
                "at most two positionals [scenario] [out_dir], got {more:?}"
            ))
        }
    };

    Ok(MovieArgs {
        scenario,
        out_dir: out_dir.map(PathBuf::from),
        color,
        dispersion_palette,
        dispersion_reference,
        reuse_snapshots,
        backend,
    })
}

/// Resolve a scenario positional: a `.toml` path is a custom scenario, otherwise
/// an alias-canonicalized name is looked up in the preset registry. `None` means
/// "not a scenario" (the caller decides whether that makes it an out dir or an
/// error).
fn resolve_scenario(raw: &str) -> Option<ScenarioArg> {
    if raw.ends_with(".toml") {
        return Some(ScenarioArg::Path(PathBuf::from(raw)));
    }
    let canonical = match raw {
        "nfw" => "dm",
        "disk-nfw" => "cuspy",
        other => other,
    };
    spec::PRESETS
        .iter()
        .any(|(name, _)| *name == canonical)
        .then(|| ScenarioArg::Preset(canonical.to_string()))
}

fn unknown_scenario(raw: &str) -> String {
    let names: Vec<&str> = spec::PRESETS.iter().map(|(name, _)| *name).collect();
    format!(
        "unknown scenario `{raw}` (presets: {}; or a path to a scenario.toml)",
        names.join("|")
    )
}

/// A parsed `regrade` invocation: which EXR frames to read, where the PNGs go, and
/// the grade to apply.
#[derive(Clone, Debug, PartialEq)]
pub struct RegradeArgs {
    /// Directory holding the retained linear-HDR `.exr` frames.
    pub exr_dir: PathBuf,
    /// Output directory for the graded 16-bit `.png` frames.
    pub png_dir: PathBuf,
    /// The grade (exposure + tone curve) to apply to every frame.
    pub grade: GradeConfig,
}

/// Map `regrade` CLI arguments (everything after the `regrade` selector) to a
/// [`RegradeArgs`]: `<exr_dir> <png_dir> [--exposure E] [--tonemap
/// aces|reinhard|asinh] [--beta B] [--bloom S] [--bloom-levels N]
/// [--bloom-radius R] [--local S] [--local-radius R] [--local-floor F]`. Flags
/// may come in any order. Errors (as a human-readable message) on missing/extra
/// positionals, unknown flags or tonemap names, malformed or non-positive
/// numbers, `--beta` without `asinh`, and `--bloom-*`/`--local-*` sub-knobs
/// without their parent `--bloom`/`--local` (sub-knobs of a feature that is off
/// are a typo worth failing fast on, exactly like β).
pub fn parse_regrade_args(args: &[String]) -> Result<RegradeArgs, String> {
    let mut positionals: Vec<&str> = Vec::new();
    let mut exposure = 1.0f32;
    let mut tonemap_name: Option<&str> = None;
    let mut beta: Option<f32> = None;
    let mut bloom_strength: Option<f32> = None;
    let mut bloom_levels: Option<u32> = None;
    let mut bloom_radius: Option<f32> = None;
    let mut local_strength: Option<f32> = None;
    let mut local_radius: Option<f32> = None;
    let mut local_floor: Option<f32> = None;
    // Levels (galaxy-render controls): neutral until a flag moves them.
    let mut black_point = 0.0f32;
    let mut white_point = 1.0f32;
    let mut gamma = 1.0f32;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--exposure" => exposure = positive_flag_value("--exposure", it.next())?,
            "--beta" => beta = Some(positive_flag_value("--beta", it.next())?),
            // Levels window is any-sign finite (the window/gamma validity is
            // enforced together by GradeConfig::validate below); gamma is > 0.
            "--black-point" => black_point = finite_flag_value("--black-point", it.next())?,
            "--white-point" => white_point = finite_flag_value("--white-point", it.next())?,
            "--gamma" => gamma = positive_flag_value("--gamma", it.next())?,
            "--tonemap" => {
                tonemap_name = Some(
                    it.next()
                        .ok_or("--tonemap needs a value: aces|reinhard|asinh")?,
                )
            }
            "--bloom" => bloom_strength = Some(positive_flag_value("--bloom", it.next())?),
            "--bloom-levels" => {
                bloom_levels = Some(positive_int_flag_value("--bloom-levels", it.next())?)
            }
            "--bloom-radius" => {
                bloom_radius = Some(positive_flag_value("--bloom-radius", it.next())?)
            }
            "--local" => local_strength = Some(positive_flag_value("--local", it.next())?),
            "--local-radius" => {
                local_radius = Some(positive_flag_value("--local-radius", it.next())?)
            }
            // Floor is any-sign finite here; the [0, 1] bound is enforced together
            // with the rest of the local config by GradeConfig::validate below.
            "--local-floor" => local_floor = Some(finite_flag_value("--local-floor", it.next())?),
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown flag `{flag}` (expected --exposure, --tonemap, --beta, \
                     --bloom, --bloom-levels, --bloom-radius, --local, \
                     --local-radius, --local-floor, --black-point, \
                     --white-point, --gamma)"
                ));
            }
            positional => positionals.push(positional),
        }
    }

    let [exr_dir, png_dir] = positionals.as_slice() else {
        return Err(format!(
            "regrade needs exactly two positionals <exr_dir> <png_dir>, got {positionals:?}"
        ));
    };

    let tonemap = match tonemap_name.unwrap_or("aces") {
        "aces" => ToneMap::AcesApprox,
        "reinhard" => ToneMap::Reinhard,
        "asinh" => ToneMap::Asinh {
            beta: beta.take().unwrap_or(DEFAULT_ASINH_BETA),
        },
        other => return Err(format!("unknown tonemap `{other}` (aces|reinhard|asinh)")),
    };
    // A leftover --beta means the tonemap isn't asinh — β does not exist on the
    // other curves, so this is a mis-typed invocation, not a value to ignore.
    if beta.is_some() {
        return Err("--beta only applies to `--tonemap asinh`".to_string());
    }

    let bloom = match bloom_strength {
        Some(strength) => Some(BloomConfig {
            strength,
            levels: bloom_levels.take().unwrap_or(DEFAULT_BLOOM_LEVELS),
            radius: bloom_radius.take().unwrap_or(DEFAULT_BLOOM_RADIUS),
        }),
        None => {
            // Bloom sub-knobs without --bloom: same fail-fast stance as --beta.
            if bloom_levels.is_some() || bloom_radius.is_some() {
                return Err(
                    "--bloom-levels/--bloom-radius only apply together with --bloom".to_string(),
                );
            }
            None
        }
    };

    let local = match local_strength {
        Some(strength) => Some(LocalToneConfig {
            strength,
            radius: local_radius.take().unwrap_or(DEFAULT_LOCAL_RADIUS),
            floor: local_floor.take().unwrap_or(DEFAULT_LOCAL_FLOOR),
        }),
        None => {
            // Local sub-knobs without --local: same fail-fast stance as --beta.
            if local_radius.is_some() || local_floor.is_some() {
                return Err(
                    "--local-radius/--local-floor only apply together with --local".to_string(),
                );
            }
            None
        }
    };

    let grade = GradeConfig {
        exposure,
        tonemap,
        bloom,
        black_point,
        white_point,
        gamma,
        local,
    };
    // Fail fast on a degenerate levels window / gamma (black ≥ white, gamma ≤ 0,
    // non-finite) at parse time rather than deep in grade_file.
    grade.validate().map_err(|e| e.to_string())?;

    Ok(RegradeArgs {
        exr_dir: PathBuf::from(exr_dir),
        png_dir: PathBuf::from(png_dir),
        grade,
    })
}

/// Parse a flag's value as a strictly positive integer, with the flag name in
/// every failure message.
fn positive_int_flag_value(flag: &str, value: Option<&String>) -> Result<u32, String> {
    let raw = value.ok_or_else(|| format!("{flag} needs a value"))?;
    let parsed: u32 = raw
        .parse()
        .map_err(|_| format!("{flag} expects a positive integer, got `{raw}`"))?;
    if parsed == 0 {
        return Err(format!("{flag} must be positive, got `{raw}`"));
    }
    Ok(parsed)
}

/// Parse a flag's value as a strictly positive finite `f32`, with the flag name in
/// every failure message.
fn positive_flag_value(flag: &str, value: Option<&String>) -> Result<f32, String> {
    let raw = value.ok_or_else(|| format!("{flag} needs a value"))?;
    let parsed: f32 = raw
        .parse()
        .map_err(|_| format!("{flag} expects a number, got `{raw}`"))?;
    if !(parsed.is_finite() && parsed > 0.0) {
        return Err(format!("{flag} must be positive, got `{raw}`"));
    }
    Ok(parsed)
}

/// Parse a flag's value as a finite `f32` of any sign — the caller's own
/// validation constrains the range (used for the levels black/white points,
/// whose validity is the paired window `black < white`, not a per-value bound).
fn finite_flag_value(flag: &str, value: Option<&String>) -> Result<f32, String> {
    let raw = value.ok_or_else(|| format!("{flag} needs a value"))?;
    let parsed: f32 = raw
        .parse()
        .map_err(|_| format!("{flag} expects a number, got `{raw}`"))?;
    if !parsed.is_finite() {
        return Err(format!("{flag} must be finite, got `{raw}`"));
    }
    Ok(parsed)
}

/// The union of the axis-aligned bounding boxes of every frame — the scene extent
/// over the *whole* run. The renderer frames one camera from this so the view is
/// **stable across all frames** (per-frame auto-framing would make the galaxies
/// zoom/jitter as the tidal tails grow). Empty frames are ignored; an all-empty
/// input yields `(ZERO, ZERO)`.
pub fn union_bounds(frames: &[FrameData]) -> (Vec3, Vec3) {
    frames
        .iter()
        .filter(|f| !f.is_empty())
        .map(|f| f.bounds())
        .reduce(|(amin, amax), (bmin, bmax)| (amin.min(bmin), amax.max(bmax)))
        .unwrap_or((Vec3::ZERO, Vec3::ZERO))
}

/// The in-plane (x, y) radius enclosing `percentile` of all particles across every
/// frame — a **robust** scene extent for face-on framing. The union AABB is fragile:
/// a handful of far-escaping particles blow it up until the galaxies are dots and
/// off-center. Framing on the origin (the zero-COM barycenter) with a high-percentile
/// radius crops those few escapers while keeping the tidal tails. `percentile` is
/// clamped to `[0, 1]`; returns 0 when there are no particles.
pub fn framing_radius(frames: &[FrameData], percentile: f32) -> f32 {
    let mut radii: Vec<f32> = frames
        .iter()
        .flat_map(|f| f.pos.iter().map(|p| p.truncate().length()))
        .collect();
    if radii.is_empty() {
        return 0.0;
    }
    radii.sort_by(|a, b| a.total_cmp(b));
    let idx = (((radii.len() - 1) as f32) * percentile.clamp(0.0, 1.0)).round() as usize;
    radii[idx]
}

/// Per-frame `percentile` of the **3-D** radius about the origin (the zero-COM
/// barycenter) — the raw framing requirement the M6d envelope smooths, one entry
/// per frame. 3-D rather than in-plane because an orbiting/tilted camera must
/// enclose the scene from *any* view axis, and a sphere of radius `r` projects to
/// `r` in every orthographic view. Same percentile-index convention as
/// [`framing_radius`]; an empty frame contributes `0.0`.
pub fn per_frame_radii(frames: &[FrameData], percentile: f32) -> Vec<f32> {
    frames
        .iter()
        .map(|f| {
            let mut radii: Vec<f32> = f.pos.iter().map(|p| p.length()).collect();
            if radii.is_empty() {
                return 0.0;
            }
            radii.sort_by(|a, b| a.total_cmp(b));
            let idx = (((radii.len() - 1) as f32) * percentile.clamp(0.0, 1.0)).round() as usize;
            radii[idx]
        })
        .collect()
}

/// The per-instant power-of-two rung distribution of a set of gas CFL timesteps
/// — the I0 go/no-go measurement (docs/plans/laddered-ember-cadence.md). Each gas
/// particle is binned onto a rung `r = ⌈log2(dt_base / dt_i)⌉` below the coarsest
/// particle's step `dt_base = max_i dt_i`, so its individual step would be
/// `dt_base / 2^r` (≤ its own CFL requirement — the `⌈⌉` rounds toward the finer,
/// more-work side, so it *under*-states the win; the safe direction for a go/no-go).
///
/// [`RungSpread::speedup`] is the **ideal-ceiling** speedup of individual-timestep
/// SPH stepping over the global-adaptive path, which steps *every* gas particle at
/// the finest requirement. It counts particle-updates per base block only — it
/// EXCLUDES the I7 grid-rebuild / neighbour-prediction overhead (that is I6's net
/// number), and it is over GAS particles only (collisionless rows carry no hydro
/// CFL constraint — `dt = +∞`, the coarsest rung — and must not pad `N`).
#[derive(Clone, Debug, PartialEq)]
pub struct RungSpread {
    /// Gas particles binned — the speedup numerator `N` (NOT the whole-sim `N`).
    pub n: usize,
    /// Coarsest per-particle step `dt_base = max_i dt_i` (rung 0's step).
    pub dt_base: f64,
    /// Tightest per-particle step `min_i dt_i` (pins the finest rung `r_max`).
    pub dt_min: f64,
    /// Per-rung occupancy: `counts[r]` particles on rung `r` (step `dt_base/2^r`),
    /// for `r = 0..=r_max` where `r_max = counts.len() - 1`.
    pub counts: Vec<usize>,
    /// Ideal-ceiling speedup `N·2^r_max / Σ_i 2^r_i` (= `N / Σ_i 2^(r_i − r_max)`),
    /// individual vs global-adaptive. Excludes I7 overhead — an upper bound.
    pub speedup: f64,
}

impl RungSpread {
    /// The finest occupied rung `r_max` (steps `2^r_max` times per base block).
    pub fn r_max(&self) -> usize {
        self.counts.len().saturating_sub(1)
    }

    /// Fraction of gas particles on the finest rung — the minority that pins the
    /// global bound. A large fraction ⇒ most of the box needs the tiny step ⇒ a
    /// small win (the plan's "shocked region is a big fraction" stop case).
    pub fn finest_fraction(&self) -> f64 {
        match self.counts.last() {
            Some(&c) if self.n > 0 => c as f64 / self.n as f64,
            _ => 0.0,
        }
    }

    /// The raw per-instant spatial dynamic range `dt_base / dt_min` — the spread
    /// the rung scheme discretizes (the quantity individual timesteps exploit;
    /// distinct from A5's *temporal* 34× already banked by global adaptive).
    pub fn dynamic_range(&self) -> f64 {
        if self.dt_min > 0.0 {
            self.dt_base / self.dt_min
        } else {
            f64::INFINITY
        }
    }

    /// Ideal-ceiling speedup if the fine tail were resolved only down to rung `cap`
    /// — particles needing finer are clamped there (a real scheme needs the I5
    /// limiter to do this safely), and the global baseline steps everyone at `2^cap`.
    /// A **sensitivity diagnostic**, not a second verdict: it exposes how much of the
    /// win rides on the deepest (often smallest, least-resolved) rungs. `cap ≥ r_max`
    /// reproduces [`speedup`](Self::speedup); `cap = 0` gives 1× (no subdivision).
    pub fn speedup_at_cap(&self, cap: usize) -> f64 {
        let work: f64 = self
            .counts
            .iter()
            .enumerate()
            .map(|(r, &c)| c as f64 * (r.min(cap) as f64 - cap as f64).exp2())
            .sum();
        self.n as f64 / work
    }
}

/// Bin a slice of per-particle gas CFL timesteps into power-of-two rungs and
/// compute the ideal-ceiling individual-timestep speedup (see [`RungSpread`]).
/// Non-finite / non-positive entries are dropped (defensive — the caller passes
/// finite gas `dt`s). Returns `None` if nothing finite-positive remains.
pub fn rung_spread(dt: &[f64]) -> Option<RungSpread> {
    let finite: Vec<f64> = dt
        .iter()
        .copied()
        .filter(|d| d.is_finite() && *d > 0.0)
        .collect();
    if finite.is_empty() {
        return None;
    }
    let dt_base = finite.iter().copied().fold(f64::MIN_POSITIVE, f64::max);
    let dt_min = finite.iter().copied().fold(f64::INFINITY, f64::min);

    // r_i = ⌈log2(dt_base / dt_i)⌉, clamped ≥ 0 (dt_i ≤ dt_base ⇒ ratio ≥ 1 ⇒ r ≥ 0;
    // the clamp only guards fp noise at the coarsest particle where the ratio is 1).
    let rungs: Vec<usize> = finite
        .iter()
        .map(|&d| (dt_base / d).log2().ceil().max(0.0) as usize)
        .collect();
    let r_max = rungs.iter().copied().max().unwrap_or(0);

    let mut counts = vec![0usize; r_max + 1];
    for &r in &rungs {
        counts[r] += 1;
    }

    // speedup = N / Σ_i 2^(r_i − r_max): the finest rung contributes 1 per particle,
    // each coarser rung half as much. Formed this way (exponent ≤ 0) it needs no
    // large 2^r_max intermediate, so it cannot overflow for any r_max.
    let work: f64 = rungs
        .iter()
        .map(|&r| (r as f64 - r_max as f64).exp2())
        .sum();
    let speedup = finite.len() as f64 / work;

    Some(RungSpread {
        n: finite.len(),
        dt_base,
        dt_min,
        counts,
        speedup,
    })
}

// --- I0b: the GRAVITATIONAL rung-spread (precondition for `hydro+gravity`) ------
// docs/plans/laddered-ember-cadence.md, milestone I0b. I0/rung-spread measured the
// *gas CFL* rung spread (`h_i/v_sig,i`) — lever (a), the hydro-only win. Lever (b)
// — subcycling the gravity WALK on a stale tree — reduces the O(N·logN) walk to its
// active subset, and its payoff rests on a DIFFERENT distribution: the *star
// gravitational-rung* spread `dt_i = η·√(ε/|a_i|)`. That number is unmeasured; this
// is the tool that measures it, so the ~2.24× `hydro+gravity` claim rests on data,
// not a borrowed hydro factor. Like I0, it lives in the xtask (a go/no-go
// measurement), leaving the shipped force path textually untouched.

/// One particle's GRAVITATIONAL timestep datum (I0b): its stable step
/// `dt = η·√(ε/|a|)` (the standard Plummer-softened gravitational criterion), the
/// acceleration magnitude `|a|` that sets it, and its species.
///
/// Two facts about this `dt` govern how the histogram reads (both flagged by the
/// advisor):
/// - **η and the ε *prefactor* cancel in the rung ratios** — `dt_base/dt_i =
///   √(|a_i|/|a_min|)`, so the rung binning and speedup are invariant to `η` and to
///   the ε in `√(ε/·)`. `η` is purely cosmetic (scales every printed `dt` equally).
///   ε is NOT cosmetic where it enters `|a|` itself (BarnesHut softening), so it is a
///   real knob on the *force*, just not on the *spread*.
/// - **`dt ∝ |a|^(−½)` compresses the spread**: a given acceleration dynamic range
///   yields only ~half as many log₂ rungs as the same range in a `dt ∝ 1/v` (hydro)
///   criterion. So the gravitational spread tends to be NARROWER than hydro's — a
///   modest drop-finest here is the expected "stars bunch fine" regime, not a bug.
#[derive(Clone, Copy, Debug)]
pub struct GravDt {
    /// The gravitational stable step `η·√(ε/|a|)` (`+∞` when `|a| == 0`).
    pub dt: f64,
    /// Acceleration magnitude `|a_i|` (the quantity that actually sets the rung).
    pub a_mag: f64,
    /// Species — the star (`Collisionless`) subset carries the lever-(b) verdict;
    /// gas walks on its hydro rung under `hydro+gravity`, not on this one.
    pub kind: Species,
}

/// The pure gravitational-timestep map: `dt = η·√(ε/|a|)`, with `|a| == 0 ⇒ +∞`
/// (a particle feeling no net force needs no gravitational step — the coarsest,
/// best-case rung, NOT a non-participant; contrast the hydro `+∞` = collisionless).
pub fn grav_timestep(a_mag: f64, eps: f64, eta: f64) -> f64 {
    if a_mag > 0.0 {
        eta * (eps / a_mag).sqrt()
    } else {
        f64::INFINITY
    }
}

/// Per-particle gravitational timestep over EVERY particle in `state` (stars + gas),
/// in particle-index order, one [`GravDt`] each. The acceleration field is the
/// pipeline's own Barnes-Hut force (`G` = [`G`], softening `eps`, opening angle
/// `theta` — pass the pipeline [`THETA`] so the rungs match what the real sim would
/// assign), so this measures the rungs the actual `hydro+gravity` walk would use.
///
/// Returns one entry per particle (never drops any) so the caller can slice by
/// species and assert nothing is lost — the `+∞` here is a coarse-rung particle, not
/// an excluded one (the inverted semantics vs [`RungSpread`]/hydro). Non-hydro to the
/// core: this is gravity, which acts on all mass.
pub fn per_particle_grav_dt(state: &State, eps: f64, eta: f64, theta: f64) -> Vec<GravDt> {
    let mut acc = vec![DVec3::ZERO; state.len()];
    BarnesHut::new(G, eps, theta).accelerations(state, &mut acc);
    acc.iter()
        .zip(&state.kind)
        .map(|(a, &kind)| {
            let a_mag = a.length();
            GravDt {
                dt: grav_timestep(a_mag, eps, eta),
                a_mag,
                kind,
            }
        })
        .collect()
}

/// The θ cross-check (I0b's nearest analogue to rung-spread's runtime self-check
/// against the shipped scalar bound — there is no shipped gravitational bound, so
/// this validates the Barnes-Hut acceleration field the rungs are built from). Max
/// over particles of the relative difference `|a_BH − a_exact| / |a_exact|` between
/// the θ-approximated tree walk and the exact O(N²) Plummer direct sum at the SAME
/// softening. A small value confirms the θ=0.5 rungs are not tree-approximation
/// artefacts; it doubles as the tail's θ-sensitivity (cheap at ~7500 particles).
pub fn accel_max_rel_vs_direct(state: &State, eps: f64, theta: f64) -> f64 {
    let n = state.len();
    let mut a_bh = vec![DVec3::ZERO; n];
    let mut a_exact = vec![DVec3::ZERO; n];
    BarnesHut::new(G, eps, theta).accelerations(state, &mut a_bh);
    DirectSum::new(G, eps).accelerations(state, &mut a_exact);
    a_bh.iter()
        .zip(&a_exact)
        .map(|(bh, ex)| (*bh - *ex).length() / ex.length().max(f64::MIN_POSITIVE))
        .fold(0.0_f64, f64::max)
}

/// The per-force-block cost split that turns a measured rung factor into a
/// whole-sim speedup (the Amdahl reprojection). Costs in ms/block, measured
/// 2026-07-09 on the shipped-seed gasrich pericenter (`laddered-ember-cadence.md`,
/// "AMDAHL SPLIT"). They are structural (set by N, gas fraction, tree depth) but the
/// build:walk ratio specifically is clustering-sensitive — read the reprojection
/// with a ± and re-measure at a tighter pericenter if the scope call proceeds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AmdahlBlock {
    /// Gravity tree build (Barnes-Hut, O(N)) — the fixed floor NO lever can cut.
    pub build_ms: f64,
    /// Gravity walk (O(active·logN)) — reducible ONLY by lever (b) (stale-tree
    /// subcycle); this is the term I0b's star factor multiplies.
    pub walk_ms: f64,
    /// Density + hydro on the gas subset — reducible by lever (a) (the core rung
    /// win), factor = the gas hydro drop-finest ([`W_HYDRO_DROP_FINEST`]).
    pub hydro_ms: f64,
    /// Per-particle `dt` (CFL) — reducible by lever (a): free with hydro-only rungs
    /// (a rung IS the per-particle dt), fuses with the active-subset hydro solve.
    pub cfl_ms: f64,
}

/// The 2026-07-09 measured gasrich-pericenter block split (`build:walk = 0.68`).
pub const AMDAHL_GASRICH_PERICENTER: AmdahlBlock = AmdahlBlock {
    build_ms: 120.0,
    walk_ms: 176.0,
    hydro_ms: 347.0,
    cfl_ms: 134.0,
};

/// The robust lever-(a) hydro factor: the seed-sweep DROP-FINEST median (~2.9×), the
/// conservative gas-CFL rung speedup (I0 RESULT). Paired here with I0b's measured
/// star drop-finest so both Amdahl terms use the same robust (drop-the-finest-rung)
/// convention rather than the fragile full-tail ceiling.
pub const W_HYDRO_DROP_FINEST: f64 = 2.9;

impl AmdahlBlock {
    /// Total block cost with no rungs — the Amdahl denominator's `1×` baseline.
    pub fn total_ms(&self) -> f64 {
        self.build_ms + self.walk_ms + self.hydro_ms + self.cfl_ms
    }

    /// Whole-sim speedup of **`hydro-only`** rungs (lever a): gravity build+walk stay
    /// fixed (all-N walk, once per base block), the hydro+CFL terms reduce by
    /// `w_hydro`. With `w_hydro = W_HYDRO_DROP_FINEST` this reproduces the plan's
    /// ~1.68×.
    pub fn hydro_only_speedup(&self, w_hydro: f64) -> f64 {
        self.total_ms() / (self.build_ms + self.walk_ms + (self.hydro_ms + self.cfl_ms) / w_hydro)
    }

    /// Whole-sim speedup of **`hydro+gravity`** rungs (levers a+b): the O(N) build
    /// stays fixed, the walk reduces by `w_grav` (I0b's measured star drop-finest),
    /// the hydro+CFL terms by `w_hydro`. With `w_hydro = w_grav = W_HYDRO_DROP_FINEST`
    /// this reproduces the plan's ~2.24× (the borrowed-2.9× estimate I0b replaces).
    ///
    /// Conservative: it charges the WHOLE walk at the star factor, but under
    /// `hydro+gravity` the ~⅓ gas share of the walk actually rides the (typically
    /// larger) hydro factor, so the true speedup is ≥ this when `w_grav < w_hydro`.
    pub fn hydro_plus_gravity_speedup(&self, w_hydro: f64, w_grav: f64) -> f64 {
        self.total_ms()
            / (self.build_ms + self.walk_ms / w_grav + (self.hydro_ms + self.cfl_ms) / w_hydro)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_at(points: &[[f32; 3]]) -> FrameData {
        let pos: Vec<Vec3> = points.iter().map(|&[x, y, z]| Vec3::new(x, y, z)).collect();
        let n = pos.len();
        FrameData {
            pos,
            color: vec![[1.0; 3]; n],
            size: vec![1.0; n],
            brightness: vec![1.0; n],
        }
    }

    #[test]
    fn union_covers_every_frame() {
        let a = frame_at(&[[-2.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        let b = frame_at(&[[0.0, -3.0, 0.0], [3.0, 0.0, 2.0]]);
        let (min, max) = union_bounds(&[a, b]);
        assert_eq!(min, Vec3::new(-2.0, -3.0, 0.0));
        assert_eq!(max, Vec3::new(3.0, 1.0, 2.0));
    }

    #[test]
    fn union_ignores_empty_frames() {
        let empty = FrameData::default();
        let a = frame_at(&[[1.0, 2.0, 3.0]]);
        let (min, max) = union_bounds(&[empty, a]);
        assert_eq!(min, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(max, Vec3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn union_of_nothing_is_zero() {
        assert_eq!(union_bounds(&[]), (Vec3::ZERO, Vec3::ZERO));
    }

    #[test]
    fn framing_radius_is_the_in_plane_percentile() {
        // In-plane radii 1,2,3,4 (z ignored); one far escaper at 100.
        let f = frame_at(&[
            [1.0, 0.0, 9.0],
            [0.0, 2.0, -9.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [100.0, 0.0, 0.0],
        ]);
        // 100th percentile = the escaper; a high-but-not-max percentile ignores it.
        assert!((framing_radius(std::slice::from_ref(&f), 1.0) - 100.0).abs() < 1e-4);
        assert!((framing_radius(std::slice::from_ref(&f), 0.5) - 3.0).abs() < 1e-4);
    }

    #[test]
    fn framing_radius_of_nothing_is_zero() {
        assert_eq!(framing_radius(&[], 0.98), 0.0);
    }

    // --- per-frame 3-D framing radii (M6d) -------------------------------------

    #[test]
    fn per_frame_radii_takes_the_3d_percentile_of_each_frame() {
        // Frame a: 3-D radii {1, 5, 2} — the [0,3,4] point has in-plane radius 3
        // but 3-D radius 5, so percentile 1.0 → 5 pins the THREE-dimensional
        // metric. Frame b: a single pure-z point (in-plane radius 0) → 10.
        let a = frame_at(&[[1.0, 0.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 2.0]]);
        let b = frame_at(&[[0.0, 0.0, 10.0]]);
        let r = per_frame_radii(&[a.clone(), b], 1.0);
        assert_eq!(r.len(), 2);
        assert!((r[0] - 5.0).abs() < 1e-5, "{:?}", r);
        assert!((r[1] - 10.0).abs() < 1e-5, "{:?}", r);
        // Median convention matches framing_radius: sorted {1,2,5}, idx round(1.0) → 2.
        let m = per_frame_radii(std::slice::from_ref(&a), 0.5);
        assert!((m[0] - 2.0).abs() < 1e-5, "{:?}", m);
    }

    #[test]
    fn per_frame_radii_of_empty_frames_are_zero() {
        let r = per_frame_radii(&[FrameData::default(), frame_at(&[[3.0, 0.0, 0.0]])], 0.9);
        assert_eq!(r, vec![0.0, 3.0]);
        assert!(per_frame_radii(&[], 0.9).is_empty());
    }

    // --- rung spread / ideal-ceiling speedup (I0 go/no-go) ---------------------

    #[test]
    fn rung_spread_of_a_uniform_field_is_one_rung_no_win() {
        // Every particle needs the same dt ⇒ one rung, speedup exactly 1 (nothing
        // to gain — the global step already fits everyone).
        let s = rung_spread(&[0.01; 5]).unwrap();
        assert_eq!(s.n, 5);
        assert_eq!(s.counts, vec![5]);
        assert_eq!(s.r_max(), 0);
        assert_eq!(s.finest_fraction(), 1.0);
        assert!((s.speedup - 1.0).abs() < 1e-12, "{}", s.speedup);
        assert!((s.dynamic_range() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rung_spread_half_coarse_half_16x_finer_hits_the_hand_derived_ceiling() {
        // Half at dt_base, half needing 1/16 of it → rungs 0 and 4. Work per base
        // block = (n/2)·1 + (n/2)·16 = (n/2)·17 fine-steps vs the global N·16, so
        // speedup = 16N / ((N/2)·17) = 32/17 ≈ 1.882. Hand-derived, not self-checked.
        let mut dt = vec![1.0; 4];
        dt.extend([1.0 / 16.0; 4]);
        let s = rung_spread(&dt).unwrap();
        assert_eq!(s.n, 8);
        assert_eq!(s.counts, vec![4, 0, 0, 0, 4]);
        assert_eq!(s.r_max(), 4);
        assert!((s.finest_fraction() - 0.5).abs() < 1e-12);
        assert!((s.dynamic_range() - 16.0).abs() < 1e-9);
        assert!((s.speedup - 32.0 / 17.0).abs() < 1e-12, "{}", s.speedup);
    }

    #[test]
    fn rung_binning_rounds_toward_the_finer_rung() {
        // A 10× spread is not a power of two: ⌈log2 10⌉ = 4 (step dt_base/16 ≤
        // dt_i), NOT 3 — the conservative side that under-states the win.
        let s = rung_spread(&[1.0, 0.1]).unwrap();
        assert_eq!(s.counts.len(), 5); // rungs 0..=4
        assert_eq!(s.counts[0], 1);
        assert_eq!(s.counts[4], 1);
    }

    #[test]
    fn speedup_at_cap_brackets_the_full_tail() {
        // Same half/half field: capping at r_max reproduces the full speedup; capping
        // at rung 0 forbids all subdivision (1×); intermediate caps trade win for
        // stability, monotone in the cap.
        let mut dt = vec![1.0; 4];
        dt.extend([1.0 / 16.0; 4]);
        let s = rung_spread(&dt).unwrap();
        assert!((s.speedup_at_cap(s.r_max()) - s.speedup).abs() < 1e-12);
        assert!((s.speedup_at_cap(0) - 1.0).abs() < 1e-12);
        // cap 3: fine half clamped to rung 3 → work (4·2^-3 + 4·1) = 4.5 → 8/4.5.
        assert!(
            (s.speedup_at_cap(3) - 8.0 / 4.5).abs() < 1e-12,
            "{}",
            s.speedup_at_cap(3)
        );
        // Monotone non-decreasing in the cap (more resolved tail ⇒ ≥ win).
        assert!(s.speedup_at_cap(2) <= s.speedup_at_cap(3) + 1e-12);
        assert!(s.speedup_at_cap(3) <= s.speedup_at_cap(4) + 1e-12);
    }

    #[test]
    fn rung_spread_drops_infinite_and_empty_inputs() {
        // Collisionless +∞ rows must not survive to pad N (the verdict-flipping trap).
        let s = rung_spread(&[f64::INFINITY, 0.02, f64::INFINITY]).unwrap();
        assert_eq!(s.n, 1);
        assert_eq!(s.counts, vec![1]);
        assert!(rung_spread(&[]).is_none());
        assert!(rung_spread(&[f64::INFINITY, -1.0, 0.0]).is_none());
    }

    // --- I0b: gravitational rung spread (go/no-go precondition for hydro+gravity) --

    use galaxy_core::{DVec3, Species, State};

    #[test]
    fn grav_timestep_is_the_softened_criterion_and_infinite_at_zero_accel() {
        // dt = η·√(ε/|a|); a force-free particle (|a| = 0) sits on the coarsest rung
        // (dt = +∞ — the BEST case, not an exclusion; inverted vs the hydro +∞).
        let dt = grav_timestep(0.25, 0.5, 0.7);
        assert!((dt - 0.7 * (0.5f64 / 0.25).sqrt()).abs() < 1e-15, "{dt}");
        assert_eq!(grav_timestep(0.0, 0.5, 0.7), f64::INFINITY);
    }

    #[test]
    fn per_particle_grav_dt_matches_the_two_body_hand_derivation() {
        // Two equal unit masses at separation d along x. Each feels the other's
        // softened pull |a| = G·m·d / (d² + ε²)^{3/2} (G = 1), so
        // dt = η·√(ε/|a|). Hand-derived, independent of the function's own output —
        // and it exercises the full BarnesHut-wiring + dt-map end to end. (Barnes-Hut
        // on two bodies is exact for any θ: the sibling is a single-particle leaf.)
        let d = 2.0;
        let m = 1.0;
        let eps = 0.5;
        let eta = 0.7;
        let state = State::from_phase_space(
            vec![DVec3::ZERO, DVec3::new(d, 0.0, 0.0)],
            vec![DVec3::ZERO; 2],
            vec![m; 2],
        );
        let a_expected = G * m * d / (d * d + eps * eps).powf(1.5);
        let dt_expected = eta * (eps / a_expected).sqrt();

        let g = per_particle_grav_dt(&state, eps, eta, 0.5);
        assert_eq!(g.len(), 2, "one datum per particle, never dropped");
        for gd in &g {
            assert!((gd.a_mag - a_expected).abs() < 1e-12, "|a| {}", gd.a_mag);
            assert!((gd.dt - dt_expected).abs() < 1e-12, "dt {}", gd.dt);
            assert_eq!(gd.kind, Species::Collisionless);
        }
    }

    #[test]
    fn per_particle_grav_dt_preserves_species_and_returns_every_particle() {
        // The star subset (Collisionless) carries the lever-(b) verdict, so the
        // per-particle data must keep species and length so the caller can slice —
        // and a lone particle feels no force (|a| = 0 ⇒ dt = +∞), kept not dropped.
        let mut state = State::from_phase_space(
            vec![DVec3::new(3.0, 0.0, 0.0), DVec3::new(-3.0, 0.0, 0.0)],
            vec![DVec3::ZERO; 2],
            vec![1.0; 2],
        );
        state.kind = vec![Species::Collisionless, Species::Gas];
        let g = per_particle_grav_dt(&state, 0.05, 1.0, 0.5);
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].kind, Species::Collisionless);
        assert_eq!(g[1].kind, Species::Gas);

        // A single isolated particle: no force, coarsest rung, still returned.
        let solo = State::from_phase_space(vec![DVec3::ZERO], vec![DVec3::ZERO], vec![1.0]);
        let gs = per_particle_grav_dt(&solo, 0.05, 1.0, 0.5);
        assert_eq!(gs.len(), 1);
        assert_eq!(gs[0].a_mag, 0.0);
        assert_eq!(gs[0].dt, f64::INFINITY);
    }

    #[test]
    fn accel_cross_check_is_near_zero_for_a_well_separated_pair() {
        // θ can't approximate a two-body pair (each sees the other as a single leaf),
        // so the BH field equals the exact direct sum to roundoff — the self-check's
        // clean-field baseline.
        let state = State::from_phase_space(
            vec![DVec3::ZERO, DVec3::new(4.0, 0.0, 0.0)],
            vec![DVec3::ZERO; 2],
            vec![1.0; 2],
        );
        assert!(accel_max_rel_vs_direct(&state, 0.05, 0.5) < 1e-12);
    }

    #[test]
    fn amdahl_reprojects_the_plan_speedups() {
        let b = AMDAHL_GASRICH_PERICENTER;
        assert_eq!(b.total_ms(), 777.0);

        // hydro-only (lever a) at the robust drop-finest hydro factor → the plan's 1.68×.
        let hydro_only = b.hydro_only_speedup(W_HYDRO_DROP_FINEST);
        assert!((hydro_only - 1.68).abs() < 0.01, "hydro-only {hydro_only}");

        // hydro+gravity (levers a+b) with the plan's BORROWED 2.9× walk factor → 2.24×.
        let both = b.hydro_plus_gravity_speedup(W_HYDRO_DROP_FINEST, W_HYDRO_DROP_FINEST);
        assert!((both - 2.24).abs() < 0.01, "hydro+gravity {both}");
    }

    #[test]
    fn amdahl_gravity_subcycling_is_monotone_and_bracketed() {
        let b = AMDAHL_GASRICH_PERICENTER;
        let w_h = W_HYDRO_DROP_FINEST;
        // w_grav = 1 (walk not reduced) ≡ hydro-only exactly (lever b contributes nothing).
        assert!((b.hydro_plus_gravity_speedup(w_h, 1.0) - b.hydro_only_speedup(w_h)).abs() < 1e-12);
        // Strictly increasing in the walk factor: more star subcycling ⇒ more speedup.
        assert!(b.hydro_plus_gravity_speedup(w_h, 2.0) > b.hydro_plus_gravity_speedup(w_h, 1.5));
        // Upper bracket: a perfectly-subcycled walk (w_grav → ∞) removes the whole
        // walk term — the ceiling lever (b) can approach but never exceed.
        let ceil = b.total_ms() / (b.build_ms + (b.hydro_ms + b.cfl_ms) / w_h);
        assert!(b.hydro_plus_gravity_speedup(w_h, 1e12) <= ceil + 1e-9);
        assert!(b.hydro_plus_gravity_speedup(w_h, 1e12) > ceil - 1e-3);
    }

    // --- regrade arg parsing (M6a) -------------------------------------------

    use galaxy_grade::{LocalToneConfig, ToneMap};

    fn args(a: &[&str]) -> Vec<String> {
        a.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn regrade_defaults_to_unit_exposure_aces() {
        let r = parse_regrade_args(&args(&["in_exr", "out_png"])).unwrap();
        assert_eq!(r.exr_dir, PathBuf::from("in_exr"));
        assert_eq!(r.png_dir, PathBuf::from("out_png"));
        assert_eq!(
            r.grade,
            GradeConfig {
                exposure: 1.0,
                tonemap: ToneMap::AcesApprox,
                bloom: None,
                ..GradeConfig::default()
            }
        );
    }

    #[test]
    fn regrade_parses_levels_flags() {
        // Photoshop-style levels for contrast (galaxy-render controls): the
        // black/white window + midtone gamma reach GradeConfig from the CLI.
        let r = parse_regrade_args(&args(&[
            "e",
            "p",
            "--black-point",
            "0.05",
            "--white-point",
            "0.9",
            "--gamma",
            "1.4",
        ]))
        .unwrap();
        assert_eq!(r.grade.black_point, 0.05);
        assert_eq!(r.grade.white_point, 0.9);
        assert_eq!(r.grade.gamma, 1.4);
    }

    #[test]
    fn regrade_levels_default_to_neutral() {
        // Omitted levels flags leave the neutral (0, 1, 1) grade — the shipped
        // regrade stays bit-identical.
        let r = parse_regrade_args(&args(&["e", "p"])).unwrap();
        assert_eq!(r.grade.black_point, 0.0);
        assert_eq!(r.grade.white_point, 1.0);
        assert_eq!(r.grade.gamma, 1.0);
    }

    #[test]
    fn regrade_rejects_invalid_levels() {
        for bad in [
            // Inverted window (black ≥ white).
            &["e", "p", "--black-point", "0.8", "--white-point", "0.2"][..],
            // Non-positive gamma.
            &["e", "p", "--gamma", "0"][..],
            &["e", "p", "--gamma", "-1"][..],
            // Non-finite level.
            &["e", "p", "--black-point", "nan"][..],
        ] {
            assert!(
                parse_regrade_args(&args(bad)).is_err(),
                "should reject invalid levels: {bad:?}"
            );
        }
    }

    #[test]
    fn regrade_parses_full_asinh_invocation() {
        let r = parse_regrade_args(&args(&[
            "e",
            "p",
            "--exposure",
            "2.5",
            "--tonemap",
            "asinh",
            "--beta",
            "0.05",
        ]))
        .unwrap();
        assert_eq!(
            r.grade,
            GradeConfig {
                exposure: 2.5,
                tonemap: ToneMap::Asinh { beta: 0.05 },
                bloom: None,
                ..GradeConfig::default()
            }
        );
    }

    #[test]
    fn regrade_flags_are_order_independent() {
        // --beta before --tonemap must still land on the asinh curve.
        let a =
            parse_regrade_args(&args(&["e", "p", "--beta", "0.05", "--tonemap", "asinh"])).unwrap();
        let b =
            parse_regrade_args(&args(&["e", "p", "--tonemap", "asinh", "--beta", "0.05"])).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn regrade_selects_reinhard_by_name() {
        let r = parse_regrade_args(&args(&["e", "p", "--tonemap", "reinhard"])).unwrap();
        assert_eq!(r.grade.tonemap, ToneMap::Reinhard);
    }

    #[test]
    fn regrade_asinh_without_beta_uses_the_documented_default() {
        let r = parse_regrade_args(&args(&["e", "p", "--tonemap", "asinh"])).unwrap();
        assert_eq!(
            r.grade.tonemap,
            ToneMap::Asinh {
                beta: DEFAULT_ASINH_BETA
            }
        );
        const { assert!(DEFAULT_ASINH_BETA > 0.0) };
    }

    // --- regrade bloom flags (M6b) --------------------------------------------

    use galaxy_grade::BloomConfig;

    #[test]
    fn regrade_parses_full_bloom_invocation() {
        let r = parse_regrade_args(&args(&[
            "e",
            "p",
            "--bloom",
            "0.4",
            "--bloom-levels",
            "3",
            "--bloom-radius",
            "1.5",
        ]))
        .unwrap();
        assert_eq!(
            r.grade.bloom,
            Some(BloomConfig {
                strength: 0.4,
                levels: 3,
                radius: 1.5,
            })
        );
        // Bloom composes with the tone-curve flags untouched.
        assert_eq!(r.grade.exposure, 1.0);
        assert_eq!(r.grade.tonemap, ToneMap::AcesApprox);
    }

    #[test]
    fn regrade_bloom_defaults_levels_and_radius() {
        let r = parse_regrade_args(&args(&["e", "p", "--bloom", "0.4"])).unwrap();
        assert_eq!(
            r.grade.bloom,
            Some(BloomConfig {
                strength: 0.4,
                levels: DEFAULT_BLOOM_LEVELS,
                radius: DEFAULT_BLOOM_RADIUS,
            })
        );
        const { assert!(DEFAULT_BLOOM_LEVELS > 0) };
        const { assert!(DEFAULT_BLOOM_RADIUS > 0.0) };
    }

    #[test]
    fn regrade_bloom_flags_are_order_independent() {
        // Sub-flags before --bloom must still land on the same config.
        let a = parse_regrade_args(&args(&[
            "e",
            "p",
            "--bloom-radius",
            "1.5",
            "--bloom",
            "0.4",
        ]))
        .unwrap();
        let b = parse_regrade_args(&args(&[
            "e",
            "p",
            "--bloom",
            "0.4",
            "--bloom-radius",
            "1.5",
        ]))
        .unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn regrade_rejects_bad_bloom_invocations() {
        for (bad, why) in [
            (&["e", "p", "--bloom"][..], "flag missing its value"),
            (&["e", "p", "--bloom", "abc"][..], "non-numeric strength"),
            (
                &["e", "p", "--bloom", "0"][..],
                "non-positive strength (omit the flag for off)",
            ),
            (&["e", "p", "--bloom", "-0.5"][..], "negative strength"),
            (
                &["e", "p", "--bloom-levels", "3"][..],
                "levels without --bloom",
            ),
            (
                &["e", "p", "--bloom-radius", "2"][..],
                "radius without --bloom",
            ),
            (
                &["e", "p", "--bloom", "0.4", "--bloom-levels", "0"][..],
                "zero levels",
            ),
            (
                &["e", "p", "--bloom", "0.4", "--bloom-levels", "2.5"][..],
                "non-integer levels",
            ),
            (
                &["e", "p", "--bloom", "0.4", "--bloom-radius", "-1"][..],
                "non-positive radius",
            ),
        ] {
            assert!(
                parse_regrade_args(&args(bad)).is_err(),
                "should reject: {why} ({bad:?})"
            );
        }
    }

    // --- regrade local-tonemap flags (white-blob lever) -----------------------

    #[test]
    fn regrade_parses_full_local_invocation() {
        let r = parse_regrade_args(&args(&[
            "e",
            "p",
            "--local",
            "2.0",
            "--local-radius",
            "24",
            "--local-floor",
            "0.15",
        ]))
        .unwrap();
        assert_eq!(
            r.grade.local,
            Some(LocalToneConfig {
                strength: 2.0,
                radius: 24.0,
                floor: 0.15,
            })
        );
    }

    #[test]
    fn regrade_local_defaults_radius_and_floor() {
        let r = parse_regrade_args(&args(&["e", "p", "--local", "2.0"])).unwrap();
        assert_eq!(
            r.grade.local,
            Some(LocalToneConfig {
                strength: 2.0,
                radius: DEFAULT_LOCAL_RADIUS,
                floor: DEFAULT_LOCAL_FLOOR,
            })
        );
    }

    #[test]
    fn regrade_local_defaults_to_none() {
        // No --local ⇒ local stays off, and the regrade is bit-identical.
        let r = parse_regrade_args(&args(&["e", "p"])).unwrap();
        assert_eq!(r.grade.local, None);
    }

    #[test]
    fn regrade_local_flags_are_order_independent() {
        // Sub-flags before --local must still land on the same config.
        let a = parse_regrade_args(&args(&[
            "e",
            "p",
            "--local-floor",
            "0.1",
            "--local-radius",
            "16",
            "--local",
            "3.0",
        ]))
        .unwrap();
        let b = parse_regrade_args(&args(&[
            "e",
            "p",
            "--local",
            "3.0",
            "--local-radius",
            "16",
            "--local-floor",
            "0.1",
        ]))
        .unwrap();
        assert_eq!(a.grade.local, b.grade.local);
        assert_eq!(
            a.grade.local,
            Some(LocalToneConfig {
                strength: 3.0,
                radius: 16.0,
                floor: 0.1,
            })
        );
    }

    #[test]
    fn regrade_rejects_bad_local_invocations() {
        for (bad, why) in [
            (&["e", "p", "--local"][..], "flag missing its value"),
            (&["e", "p", "--local", "abc"][..], "non-numeric strength"),
            (
                &["e", "p", "--local", "0"][..],
                "zero strength (a no-op typo)",
            ),
            (&["e", "p", "--local", "-1"][..], "negative strength"),
            (
                &["e", "p", "--local-radius", "16"][..],
                "radius without --local",
            ),
            (
                &["e", "p", "--local-floor", "0.2"][..],
                "floor without --local",
            ),
            (
                &["e", "p", "--local", "2.0", "--local-floor", "1.5"][..],
                "floor above 1",
            ),
            (
                &["e", "p", "--local", "2.0", "--local-floor", "-0.1"][..],
                "floor below 0",
            ),
            (
                &["e", "p", "--local", "2.0", "--local-radius", "0"][..],
                "non-positive radius",
            ),
        ] {
            assert!(
                parse_regrade_args(&args(bad)).is_err(),
                "should reject: {why} ({bad:?})"
            );
        }
    }

    // --- movie arg parsing (M6e) -----------------------------------------------

    fn preset_arg(name: &str) -> ScenarioArg {
        ScenarioArg::Preset(name.to_string())
    }

    #[test]
    fn movie_defaults_to_disk_progenitor_fresh_sim() {
        let m = parse_movie_args(&args(&[])).unwrap();
        assert_eq!(
            m,
            MovieArgs {
                scenario: preset_arg("disk"),
                out_dir: None,
                color: ColorModeArg::Progenitor,
                dispersion_palette: DispersionPalette::BlueArms,
                dispersion_reference: SigmaReference::Full,
                reuse_snapshots: false,
                backend: Backend::Cpu,
            }
        );
    }

    #[test]
    fn movie_scenario_names_and_aliases_canonicalize() {
        for (raw, canonical) in [
            ("disk", "disk"),
            ("dm", "dm"),
            ("nfw", "dm"),
            ("cuspy", "cuspy"),
            ("disk-nfw", "cuspy"),
        ] {
            let m = parse_movie_args(&args(&[raw])).unwrap();
            assert_eq!(m.scenario, preset_arg(canonical), "{raw}");
            assert_eq!(m.out_dir, None);
        }
    }

    // --- scenario.toml front-end + zoo selectors (M6f) --------------------------

    #[test]
    fn movie_accepts_every_checked_in_preset_name() {
        // The CLI selector set IS the preset registry — a new preset toml must be
        // reachable without touching the parser.
        for (name, _) in spec::PRESETS {
            let m = parse_movie_args(&args(&[name])).unwrap();
            assert_eq!(m.scenario, preset_arg(name), "{name}");
            assert_eq!(m.out_dir, None);
        }
    }

    #[test]
    fn movie_first_positional_toml_path_is_a_custom_scenario() {
        let m = parse_movie_args(&args(&["zoo/mine.toml"])).unwrap();
        assert_eq!(
            m.scenario,
            ScenarioArg::Path(PathBuf::from("zoo/mine.toml"))
        );
        assert_eq!(m.out_dir, None);
    }

    #[test]
    fn movie_toml_path_composes_with_out_dir_and_flags() {
        let m = parse_movie_args(&args(&[
            "zoo/mine.toml",
            "out",
            "--color",
            "initial-radius",
            "--reuse-snapshots",
        ]))
        .unwrap();
        assert_eq!(
            m.scenario,
            ScenarioArg::Path(PathBuf::from("zoo/mine.toml"))
        );
        assert_eq!(m.out_dir, Some(PathBuf::from("out")));
        assert_eq!(m.color, ColorModeArg::InitialRadius);
        assert!(m.reuse_snapshots);
    }

    #[test]
    fn movie_second_positional_is_the_out_dir() {
        let m = parse_movie_args(&args(&["cuspy", "some/out"])).unwrap();
        assert_eq!(m.scenario, preset_arg("cuspy"));
        assert_eq!(m.out_dir, Some(PathBuf::from("some/out")));
    }

    #[test]
    fn movie_bare_first_positional_is_out_dir_with_disk_scenario() {
        // The original single-scenario CLI: `xtask <out_dir>` — must keep working
        // (a non-preset, non-`.toml` positional is an out dir).
        let m = parse_movie_args(&args(&["renders/mine"])).unwrap();
        assert_eq!(m.scenario, preset_arg("disk"));
        assert_eq!(m.out_dir, Some(PathBuf::from("renders/mine")));
    }

    #[test]
    fn movie_parses_color_modes() {
        for (name, mode) in [
            ("progenitor", ColorModeArg::Progenitor),
            ("initial-radius", ColorModeArg::InitialRadius),
            ("dispersion", ColorModeArg::Dispersion),
        ] {
            let m = parse_movie_args(&args(&["cuspy", "--color", name])).unwrap();
            assert_eq!(m.color, mode, "{name}");
        }
    }

    #[test]
    fn movie_parses_dispersion_dials() {
        // Both dials are independent and default to the "keep them" look (blue
        // tidal arms + full-population σ_ref); each is overridable on its own.
        let def = parse_movie_args(&args(&["cuspy", "--color", "dispersion"])).unwrap();
        assert_eq!(def.dispersion_palette, DispersionPalette::BlueArms);
        assert_eq!(def.dispersion_reference, SigmaReference::Full);

        for (name, pal) in [
            ("blue-arms", DispersionPalette::BlueArms),
            ("blackbody", DispersionPalette::Blackbody),
        ] {
            let m = parse_movie_args(&args(&["cuspy", "--dispersion-palette", name])).unwrap();
            assert_eq!(m.dispersion_palette, pal, "{name}");
        }
        for (name, refr) in [
            ("full", SigmaReference::Full),
            ("luminous", SigmaReference::Luminous),
        ] {
            let m = parse_movie_args(&args(&["cuspy", "--dispersion-reference", name])).unwrap();
            assert_eq!(m.dispersion_reference, refr, "{name}");
        }
    }

    #[test]
    fn movie_flags_are_order_independent_and_compose() {
        let a = parse_movie_args(&args(&[
            "--reuse-snapshots",
            "cuspy",
            "--color",
            "initial-radius",
            "out",
        ]))
        .unwrap();
        let b = parse_movie_args(&args(&[
            "cuspy",
            "out",
            "--color",
            "initial-radius",
            "--reuse-snapshots",
        ]))
        .unwrap();
        assert_eq!(a, b);
        assert!(a.reuse_snapshots);
        assert_eq!(a.color, ColorModeArg::InitialRadius);
        assert_eq!(a.out_dir, Some(PathBuf::from("out")));
    }

    #[test]
    fn movie_rejects_malformed_invocations() {
        for (bad, why) in [
            (&["disk", "out", "extra"][..], "third positional"),
            (&["--color"][..], "flag missing its value"),
            (&["--color", "rainbow"][..], "unknown color mode"),
            (&["--colour", "progenitor"][..], "unknown flag"),
            (&["--reuse"][..], "unknown flag (not the full name)"),
            (&["--dispersion-palette"][..], "palette flag missing its value"),
            (&["--dispersion-palette", "amber"][..], "unknown palette"),
            (&["--dispersion-reference"][..], "reference flag missing its value"),
            (&["--dispersion-reference", "halo"][..], "unknown reference"),
        ] {
            assert!(
                parse_movie_args(&args(bad)).is_err(),
                "should reject: {why} ({bad:?})"
            );
        }
    }

    #[test]
    fn regrade_rejects_malformed_invocations() {
        for (bad, why) in [
            (&["only_one"][..], "missing png_dir"),
            (&["a", "b", "c"][..], "extra positional"),
            (&["e", "p", "--frobnicate", "2.2"][..], "unknown flag"),
            (&["e", "p", "--exposure"][..], "flag missing its value"),
            (&["e", "p", "--exposure", "abc"][..], "non-numeric exposure"),
            (&["e", "p", "--exposure", "-1"][..], "non-positive exposure"),
            (&["e", "p", "--tonemap", "filmic"][..], "unknown tonemap"),
            (
                &["e", "p", "--tonemap", "asinh", "--beta", "0"][..],
                "non-positive beta",
            ),
            (
                &["e", "p", "--beta", "0.1"][..],
                "beta without the asinh tonemap",
            ),
            (
                &["e", "p", "--tonemap", "reinhard", "--beta", "0.1"][..],
                "beta with a non-asinh tonemap",
            ),
        ] {
            assert!(
                parse_regrade_args(&args(bad)).is_err(),
                "should reject: {why} ({bad:?})"
            );
        }
    }
}
