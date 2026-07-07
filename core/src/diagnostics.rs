//! Conservation diagnostics. These are the headline validation signals: with a
//! symplectic integrator total energy should oscillate within a bound (not
//! drift), and momentum / angular momentum should be conserved to roundoff.

use crate::{DVec3, ForceSolver, State};

/// Total kinetic energy: 0.5 Σ mᵢ vᵢ².
pub fn kinetic_energy(state: &State) -> f64 {
    state
        .vel
        .iter()
        .zip(&state.mass)
        .map(|(v, m)| 0.5 * *m * v.length_squared())
        .sum()
}

/// Total gravitational potential energy (delegates to the solver's softened kernel).
pub fn potential_energy(state: &State, solver: &dyn ForceSolver) -> f64 {
    solver.potential_energy(state)
}

/// Total thermal (internal) energy: Σ mᵢ uᵢ, the adiabatic path's contribution
/// to the total energy. Zero on the isothermal path (`u ≡ 0`), where the EOS
/// fixes pressure from ρ alone and there is no thermal reservoir to conserve.
pub fn thermal_energy(state: &State) -> f64 {
    state.mass.iter().zip(&state.u).map(|(m, u)| *m * *u).sum()
}

/// Total energy E = T + U (gravitational) + U_thermal. On the isothermal path
/// `u ≡ 0`, so [`thermal_energy`] is identically zero and this is numerically
/// the pure-gravity total; on the adiabatic path it carries the gas internal
/// energy that the energy equation conserves.
pub fn total_energy(state: &State, solver: &dyn ForceSolver) -> f64 {
    kinetic_energy(state) + solver.potential_energy(state) + thermal_energy(state)
}

/// Total linear momentum: Σ mᵢ vᵢ.
pub fn total_momentum(state: &State) -> DVec3 {
    state
        .vel
        .iter()
        .zip(&state.mass)
        .map(|(v, m)| *v * *m)
        .fold(DVec3::ZERO, |acc, p| acc + p)
}

/// Total angular momentum about the origin: Σ mᵢ (rᵢ × vᵢ).
pub fn total_angular_momentum(state: &State) -> DVec3 {
    state
        .pos
        .iter()
        .zip(&state.vel)
        .zip(&state.mass)
        .map(|((r, v), m)| r.cross(*v) * *m)
        .fold(DVec3::ZERO, |acc, l| acc + l)
}

/// Center-of-mass position.
pub fn center_of_mass(state: &State) -> DVec3 {
    let mtot: f64 = state.mass.iter().sum();
    let weighted = state
        .pos
        .iter()
        .zip(&state.mass)
        .map(|(r, m)| *r * *m)
        .fold(DVec3::ZERO, |acc, w| acc + w);
    weighted / mtot
}
