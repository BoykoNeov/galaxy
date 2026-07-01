//! `galaxy-xtask`: the pipeline orchestrator (scenario → sim → renderprep → render
//! → grade → ffmpeg). The binary is the glue; this lib holds the pure, testable bits.

use galaxy_renderprep::FrameData;
use glam::Vec3;

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
}
