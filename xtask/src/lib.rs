//! `galaxy-xtask`: the pipeline orchestrator (scenario → sim → renderprep → render
//! → grade → ffmpeg). The binary is the glue; this lib holds the pure, testable bits.

pub mod spec;

use std::path::PathBuf;

use galaxy_grade::{BloomConfig, GradeConfig, ToneMap};
use galaxy_renderprep::FrameData;
use glam::Vec3;

// --- Shared physics / look constants (every scenario) --------------------------
// Scenario-independent pipeline constants; the per-scenario knobs live in the
// `scenario.toml` presets (see `spec`). Tuning provenance: DESIGN.md M3.6/M6a–M6e.

/// Gravitational constant of the N-body unit system (G = 1).
pub const G: f64 = 1.0;
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
    /// Skip the simulation and read existing `snapshots/*.snap` under the out dir
    /// (errors downstream if none exist — reuse is an explicit promise).
    pub reuse_snapshots: bool,
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
    let mut reuse_snapshots = false;

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
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown flag `{flag}` (expected --color, --reuse-snapshots)"
                ));
            }
            positional => positionals.push(positional),
        }
    }

    // First positional: a scenario name/alias, else the out dir under the original
    // single-scenario CLI (`xtask <out_dir>` defaulted to the disk movie).
    let (scenario, out_dir) = match positionals.as_slice() {
        [] => ("disk", None),
        [one] => match *one {
            "disk" | "dm" | "nfw" | "cuspy" | "disk-nfw" => (*one, None),
            other => ("disk", Some(other)),
        },
        [scenario, out] => (*scenario, Some(*out)),
        more => {
            return Err(format!(
                "at most two positionals [scenario] [out_dir], got {more:?}"
            ))
        }
    };
    let scenario = match scenario {
        "disk" => "disk",
        "dm" | "nfw" => "dm",
        "cuspy" | "disk-nfw" => "cuspy",
        other => return Err(format!("unknown scenario `{other}` (disk|dm|cuspy)")),
    };

    Ok(MovieArgs {
        scenario: ScenarioArg::Preset(scenario.to_string()),
        out_dir: out_dir.map(PathBuf::from),
        color,
        reuse_snapshots,
    })
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
/// [--bloom-radius R]`. Flags may come in any order. Errors (as a human-readable
/// message) on missing/extra positionals, unknown flags or tonemap names,
/// malformed or non-positive numbers, `--beta` without `asinh`, and
/// `--bloom-levels`/`--bloom-radius` without `--bloom` (sub-knobs of a feature
/// that is off are a typo worth failing fast on, exactly like β).
pub fn parse_regrade_args(args: &[String]) -> Result<RegradeArgs, String> {
    let mut positionals: Vec<&str> = Vec::new();
    let mut exposure = 1.0f32;
    let mut tonemap_name: Option<&str> = None;
    let mut beta: Option<f32> = None;
    let mut bloom_strength: Option<f32> = None;
    let mut bloom_levels: Option<u32> = None;
    let mut bloom_radius: Option<f32> = None;

    let mut it = args.iter();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--exposure" => exposure = positive_flag_value("--exposure", it.next())?,
            "--beta" => beta = Some(positive_flag_value("--beta", it.next())?),
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
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown flag `{flag}` (expected --exposure, --tonemap, --beta, \
                     --bloom, --bloom-levels, --bloom-radius)"
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

    Ok(RegradeArgs {
        exr_dir: PathBuf::from(exr_dir),
        png_dir: PathBuf::from(png_dir),
        grade: GradeConfig {
            exposure,
            tonemap,
            bloom,
        },
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

    // --- regrade arg parsing (M6a) -------------------------------------------

    use galaxy_grade::ToneMap;

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
            }
        );
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
                reuse_snapshots: false,
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
            (&["e", "p", "--gamma", "2.2"][..], "unknown flag"),
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
