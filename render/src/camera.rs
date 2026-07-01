//! Orthographic camera: projects world-space particle positions to normalized
//! device coordinates (NDC, `[-1, 1]²`) for the splat renderer.
//!
//! MVP is orthographic (no perspective foreshortening) — apt for an emissive star
//! field and trivial to reason about. The **view axis is a parameter**: it defaults
//! to the collision's orbital-plane normal (+Z, so the tidal tails render face-on),
//! but a caller can pick any axis for the deferred orbit views without a code change.

use glam::{Vec2, Vec3};

/// An orthographic view: a world-space box (centered at `target`, spanning
/// `±half_extent` along the screen `right`/`up` axes) mapped onto NDC `[-1, 1]²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    /// World point at the center of the view (projects to NDC origin).
    pub target: Vec3,
    /// Screen-right axis in world space (unit).
    pub right: Vec3,
    /// Screen-up axis in world space (unit).
    pub up: Vec3,
    /// View direction, into the screen (unit). `right × up`-consistent basis.
    pub forward: Vec3,
    /// Half-width / half-height of the view box in world units. The `x:y` ratio
    /// equals the target image aspect, so a world circle renders as a screen circle.
    pub half_extent: Vec2,
}

/// Default fractional margin left around the scene bounds when auto-framing.
pub const DEFAULT_MARGIN: f32 = 0.05;

impl Camera {
    /// Build an orthographic camera looking along `view_dir` with `up_hint` giving
    /// the vertical, framing a box of `half_extent` world units about `target`.
    /// The basis is orthonormalized: `right = view_dir × up_hint`, `up = right ×
    /// view_dir`.
    pub fn orthographic(target: Vec3, view_dir: Vec3, up_hint: Vec3, half_extent: Vec2) -> Self {
        let forward = view_dir.normalize();
        let right = forward.cross(up_hint).normalize();
        let up = right.cross(forward).normalize();
        Camera {
            target,
            right,
            up,
            forward,
            half_extent,
        }
    }

    /// Auto-frame the axis-aligned box `[min, max]` looking along `view_dir`: center
    /// the view on the box, size `half_extent` to enclose all 8 corners with a
    /// fractional `margin`, then widen the short axis so `half_extent.x:y == aspect`
    /// (image width:height) — no distortion, everything visible.
    pub fn frame_bounds(
        min: Vec3,
        max: Vec3,
        view_dir: Vec3,
        up_hint: Vec3,
        margin: f32,
        aspect: f32,
    ) -> Self {
        let target = (min + max) * 0.5;
        let forward = view_dir.normalize();
        let right = forward.cross(up_hint).normalize();
        let up = right.cross(forward).normalize();

        // Half-extent that encloses all 8 AABB corners along the screen axes.
        let mut er = 0.0f32;
        let mut eu = 0.0f32;
        for &x in &[min.x, max.x] {
            for &y in &[min.y, max.y] {
                for &z in &[min.z, max.z] {
                    let d = Vec3::new(x, y, z) - target;
                    er = er.max(d.dot(right).abs());
                    eu = eu.max(d.dot(up).abs());
                }
            }
        }
        // Margin, and a floor so a degenerate (flat) scene still yields a valid box.
        er = (er * (1.0 + margin)).max(1e-6);
        eu = (eu * (1.0 + margin)).max(1e-6);
        // Widen the short axis so the box matches the image aspect (no distortion).
        if er / eu < aspect {
            er = eu * aspect;
        } else {
            eu = er / aspect;
        }

        Self::orthographic(target, view_dir, up_hint, Vec2::new(er, eu))
    }

    /// Convenience: face-on view of `[min, max]` down the orbital normal (+Z), with
    /// +Y up and the default margin. This is the first-movie default.
    pub fn face_on(min: Vec3, max: Vec3, aspect: f32) -> Self {
        Self::frame_bounds(
            min,
            max,
            Vec3::new(0.0, 0.0, -1.0), // look toward -Z (camera on the +Z side)
            Vec3::Y,
            DEFAULT_MARGIN,
            aspect,
        )
    }

    /// Project a world position to NDC. Points inside the view box map to
    /// `[-1, 1]²`; the depth axis is dropped (orthographic, additive → no depth).
    pub fn project(&self, world: Vec3) -> Vec2 {
        let d = world - self.target;
        Vec2::new(
            d.dot(self.right) / self.half_extent.x,
            d.dot(self.up) / self.half_extent.y,
        )
    }

    /// NDC half-extent of a splat of the given world-space radius — the size to
    /// draw its quad. Isotropic in world space; the aspect-correct box keeps it
    /// isotropic on screen too.
    pub fn splat_ndc(&self, world_radius: f32) -> Vec2 {
        Vec2::new(
            world_radius / self.half_extent.x,
            world_radius / self.half_extent.y,
        )
    }
}
