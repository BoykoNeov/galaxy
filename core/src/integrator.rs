use crate::{Background, DVec3, ForceSolver, Integrator, State};

/// Kick-Drift-Kick leapfrog: symplectic, 2nd order, time-reversible, with
/// bounded (non-drifting) energy error. Caches accelerations between steps so
/// each step costs a single force evaluation after the first.
#[derive(Clone, Debug, Default)]
pub struct LeapfrogKdk {
    acc: Vec<DVec3>,
    primed: bool,
}

impl LeapfrogKdk {
    pub fn new() -> Self {
        Self {
            acc: Vec::new(),
            primed: false,
        }
    }
}

impl Integrator for LeapfrogKdk {
    fn step(
        &mut self,
        _state: &mut State,
        _solver: &mut dyn ForceSolver,
        _bg: &dyn Background,
        _dt: f64,
    ) {
        todo!("implement KDK leapfrog")
    }
}
