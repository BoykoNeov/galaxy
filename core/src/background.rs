use crate::Background;

/// Non-expanding background: a(t) ≡ 1, H ≡ 0. With this background the
/// integrator reduces to standard Newtonian leapfrog. Cosmological expansion
/// attaches by swapping this for a Friedmann background in a later milestone.
#[derive(Clone, Copy, Debug, Default)]
pub struct StaticBackground;

impl Background for StaticBackground {
    fn scale_factor(&self, _t: f64) -> f64 {
        1.0
    }
    fn hubble(&self, _t: f64) -> f64 {
        0.0
    }
}
