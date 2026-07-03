//! The cubic-spline (M4) SPH kernel, Monaghan (1992) convention: compact
//! support at `r = SUPPORT·h = 2h`, 3-D normalization `1/(π h³)` at the origin.
//!
//! ```text
//!             1      ⎧ 1 − (3/2)q² + (3/4)q³   0 ≤ q < 1
//! W(r, h) = ————— ·  ⎨ (1/4)(2 − q)³           1 ≤ q < 2      q = r/h
//!            π h³    ⎩ 0                        q ≥ 2
//! ```
//!
//! Value and first derivative are continuous at both knots (q = 1, 2), which is
//! what makes the pressure force well-behaved; the unit gates pin normalization,
//! the hand value at q = 0, compact support, and the analytic gradient.

use galaxy_core::DVec3;

/// Kernel support radius in units of `h`: `W(r, h) = 0` for `r ≥ SUPPORT * h`.
pub const SUPPORT: f64 = 2.0;

/// Kernel value `W(r, h)` (r ≥ 0, h > 0).
pub fn w(r: f64, h: f64) -> f64 {
    let _ = (r, h);
    todo!("M7a: cubic-spline kernel value")
}

/// Kernel gradient `∇_i W(|x_i − x_j|, h)` for the separation `r_ij = x_i − x_j`,
/// i.e. the gradient with respect to the first particle's position. Zero at
/// `r_ij = 0` (the kernel is smooth at the origin) and outside the support.
pub fn grad_w(r_ij: DVec3, h: f64) -> DVec3 {
    let _ = (r_ij, h);
    todo!("M7a: cubic-spline kernel gradient")
}
