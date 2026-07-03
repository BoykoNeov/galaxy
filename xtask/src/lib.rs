//! `galaxy-xtask`: the pipeline orchestrator (scenario → sim → renderprep → render
//! → grade → ffmpeg). The binary is the glue; this lib holds the pure, testable bits.

use std::path::PathBuf;

use galaxy_grade::{GradeConfig, ToneMap};
use galaxy_renderprep::FrameData;
use glam::Vec3;

/// Default asinh softening knob β for `regrade --tonemap asinh` when `--beta` is not
/// given. A tuning constant, eyeballed against the rendered collision frames (M6a).
pub const DEFAULT_ASINH_BETA: f32 = 0.2;

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
/// aces|reinhard|asinh] [--beta B]`. Flags may come in any order. Errors (as a
/// human-readable message) on missing/extra positionals, unknown flags or tonemap
/// names, malformed or non-positive numbers, and `--beta` without `asinh` (β only
/// exists on the asinh curve — anything else is a typo worth failing fast on).
pub fn parse_regrade_args(args: &[String]) -> Result<RegradeArgs, String> {
    let mut positionals: Vec<&str> = Vec::new();
    let mut exposure = 1.0f32;
    let mut tonemap_name: Option<&str> = None;
    let mut beta: Option<f32> = None;

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
            flag if flag.starts_with("--") => {
                return Err(format!(
                    "unknown flag `{flag}` (expected --exposure, --tonemap, --beta)"
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

    Ok(RegradeArgs {
        exr_dir: PathBuf::from(exr_dir),
        png_dir: PathBuf::from(png_dir),
        grade: GradeConfig { exposure, tonemap },
    })
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
