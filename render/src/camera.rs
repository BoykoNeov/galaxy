//! Camera: projects world-space particle positions to normalized device
//! coordinates (NDC, `[-1, 1]²`) for the splat renderer.
//!
//! Two projections (M6g): **orthographic** (the default — apt for an emissive star
//! field, trivial to reason about, every pre-M6g movie) and **perspective** (an eye
//! at `distance` behind the target along `forward`, for inside-the-scene drama).
//! Both share the same framing parameterization: `half_extent` is the world-space
//! half-size of the view *at the target plane*, so at the target distance a
//! perspective camera frames exactly what the ortho camera frames — the rig can
//! swap projection without re-deriving its envelope. The **view axis is a
//! parameter**: it defaults to the collision's orbital-plane normal (+Z, so the
//! tidal tails render face-on), but a caller can pick any axis without a code
//! change.

use glam::{Vec2, Vec3};

/// Which projection maps view-space to NDC.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Projection {
    /// Parallel projection: depth is dropped entirely.
    Orthographic,
    /// Pinhole at `eye = target − forward·distance`. `half_extent` spans the view
    /// at the *target plane* (view depth = `distance`); lateral offsets and splat
    /// sizes shrink ∝ 1/depth. Particles with view depth ≤ `near` are culled.
    Perspective {
        /// Eye-to-target distance along `forward` (world units, > 0).
        distance: f32,
        /// Near-plane depth from the eye (world units, 0 < near < distance).
        /// Splats have no depth extent, so this culls whole quads — no clipping.
        near: f32,
    },
}

/// A view: a world-space window (centered at `target`, spanning `±half_extent`
/// along the screen `right`/`up` axes at the target plane) mapped onto NDC
/// `[-1, 1]²` by `projection`.
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
    /// Half-width / half-height of the view at the target plane, in world units.
    /// The `x:y` ratio equals the target image aspect, so a world circle renders
    /// as a screen circle.
    pub half_extent: Vec2,
    /// How view-space maps to NDC (orthographic or pinhole perspective).
    pub projection: Projection,
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
            projection: Projection::Orthographic,
        }
    }

    /// Build a perspective camera: eye at `distance` behind `target` along the
    /// view axis, looking at `target`, framing `half_extent` world units *at the
    /// target plane* (so it matches [`Camera::orthographic`] framing there).
    ///
    /// Panics (fail fast, programmer error) unless `distance` and `near` are
    /// finite with `0 < near < distance`.
    pub fn perspective(
        target: Vec3,
        view_dir: Vec3,
        up_hint: Vec3,
        half_extent: Vec2,
        distance: f32,
        near: f32,
    ) -> Self {
        let _ = (target, view_dir, up_hint, half_extent, distance, near);
        todo!("M6g: perspective camera constructor")
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

    /// Project a world position to NDC. Points inside the view window map to
    /// `[-1, 1]²`. Orthographic drops depth; perspective divides lateral view
    /// coordinates by view depth (scaled so the target plane matches ortho).
    /// Perspective projection of a point at or behind the eye plane is
    /// meaningless — callers cull by [`Camera::view_depth`] first (the renderer
    /// does; see the near-cull gates).
    pub fn project(&self, world: Vec3) -> Vec2 {
        let d = world - self.target;
        match self.projection {
            Projection::Orthographic => Vec2::new(
                d.dot(self.right) / self.half_extent.x,
                d.dot(self.up) / self.half_extent.y,
            ),
            Projection::Perspective { .. } => todo!("M6g: perspective projection"),
        }
    }

    /// Signed view depth of a world point: for perspective, the distance from
    /// the *eye* along `forward` (≤ 0 means at/behind the eye — cull); for
    /// orthographic, the distance past the target plane (unused by the splat
    /// path, defined for symmetry).
    pub fn view_depth(&self, world: Vec3) -> f32 {
        let _ = world;
        todo!("M6g: view depth")
    }

    /// NDC half-extent of a splat of world-space `radius` centered at `world` —
    /// the size to draw its quad. Orthographic: position-independent (equals
    /// [`Camera::splat_ndc`]). Perspective: shrinks ∝ 1/view-depth, matching
    /// [`Camera::splat_ndc`] exactly at the target plane. Peak intensity is NOT
    /// scaled by the renderer — surface brightness is distance-invariant, so
    /// integrated flux falls as 1/d² automatically (the physical law).
    pub fn splat_extent(&self, world: Vec3, radius: f32) -> Vec2 {
        let _ = (world, radius);
        todo!("M6g: position-dependent splat extent")
    }

    /// NDC half-extent of a splat of the given world-space radius *at the target
    /// plane*. Isotropic in world space; the aspect-correct window keeps it
    /// isotropic on screen too. For position-dependent perspective sizing use
    /// [`Camera::splat_extent`].
    pub fn splat_ndc(&self, world_radius: f32) -> Vec2 {
        Vec2::new(
            world_radius / self.half_extent.x,
            world_radius / self.half_extent.y,
        )
    }
}
