//! Conservation diagnostics. These are the headline validation signals: with a
//! symplectic integrator total energy should oscillate within a bound (not
//! drift), and momentum / angular momentum should be conserved to roundoff.

use crate::{DVec3, ForceSolver, State};

/// Total kinetic energy: 0.5 Σ mᵢ vᵢ².
pub fn kinetic_energy(_state: &State) -> f64 {
    todo!()
}

/// Total gravitational potential energy (delegates to the solver's softened kernel).
pub fn potential_energy(_state: &State, _solver: &dyn ForceSolver) -> f64 {
    todo!()
}

/// Total energy E = T + U.
pub fn total_energy(_state: &State, _solver: &dyn ForceSolver) -> f64 {
    todo!()
}

/// Total linear momentum: Σ mᵢ vᵢ.
pub fn total_momentum(_state: &State) -> DVec3 {
    todo!()
}

/// Total angular momentum about the origin: Σ mᵢ (rᵢ × vᵢ).
pub fn total_angular_momentum(_state: &State) -> DVec3 {
    todo!()
}

/// Center-of-mass position.
pub fn center_of_mass(_state: &State) -> DVec3 {
    todo!()
}
