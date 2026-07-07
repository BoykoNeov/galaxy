//! Diagnostics unit tests. Each compares against an independent hand-derived
//! expectation, not the function's own output.

use galaxy_core::{diagnostics, DVec3, State};

/// Two bodies: m=2 at the origin, m=3 at x=1, with simple velocities.
fn two_body() -> State {
    State::from_phase_space(
        vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
        vec![DVec3::new(0.0, 1.0, 0.0), DVec3::new(0.0, -1.0, 0.0)],
        vec![2.0, 3.0],
    )
}

#[test]
fn kinetic_energy_matches_hand_calc() {
    // 0.5 * (2*1^2 + 3*1^2) = 2.5
    let ke = diagnostics::kinetic_energy(&two_body());
    assert!((ke - 2.5).abs() < 1e-12, "ke = {ke}");
}

#[test]
fn total_momentum_matches_hand_calc() {
    // 2*(0,1,0) + 3*(0,-1,0) = (0,-1,0)
    let p = diagnostics::total_momentum(&two_body());
    assert!(
        (p - DVec3::new(0.0, -1.0, 0.0)).length() < 1e-12,
        "p = {p:?}"
    );
}

#[test]
fn angular_momentum_matches_hand_calc() {
    // body0 (at origin) contributes 0; body1: 3*((1,0,0) x (0,-1,0)) = (0,0,-3)
    let l = diagnostics::total_angular_momentum(&two_body());
    assert!(
        (l - DVec3::new(0.0, 0.0, -3.0)).length() < 1e-12,
        "l = {l:?}"
    );
}

#[test]
fn center_of_mass_matches_hand_calc() {
    // (2*0 + 3*1) / 5 = 0.6 in x
    let com = diagnostics::center_of_mass(&two_body());
    assert!(
        (com - DVec3::new(0.6, 0.0, 0.0)).length() < 1e-12,
        "com = {com:?}"
    );
}

#[test]
fn thermal_energy_matches_hand_calc() {
    // U_thermal = Σ mᵢ uᵢ = 2*1.5 + 3*4.0 = 3 + 12 = 15, weighted by mass
    // (u is SPECIFIC internal energy, per unit mass), independent hand value.
    let mut s = two_body();
    s.u = vec![1.5, 4.0];
    let ut = diagnostics::thermal_energy(&s);
    assert!((ut - 15.0).abs() < 1e-12, "u_thermal = {ut}");
}

#[test]
fn thermal_energy_is_zero_when_internal_energy_is_zero() {
    // The isothermal invariant: u ≡ 0 ⇒ no thermal reservoir.
    let s = two_body(); // from_phase_space sets u = 0
    let ut = diagnostics::thermal_energy(&s);
    assert!(ut.abs() < 1e-12, "isothermal u_thermal must be 0, got {ut}");
}
