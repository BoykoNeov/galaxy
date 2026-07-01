//! Validation of `Orientation`, the disk spin-orbit rotation. These are pure
//! geometry checks (no sampling): a rotation must be rigid, and each named
//! geometry must move the disk spin axis (+Z in the body frame) to the physically
//! correct direction relative to the orbital plane (x–y, normal +Z).
//!
//! Expectations are written independently of the implementation — the spin
//! directions come from the rotation definitions, not from calling the code and
//! trusting its output.

use galaxy_core::DVec3;
use galaxy_ic::Orientation;

/// The body-frame disk spin axis.
const SPIN: DVec3 = DVec3::Z;

fn angle_off_z(v: DVec3) -> f64 {
    (v.normalize().dot(DVec3::Z)).clamp(-1.0, 1.0).acos()
}

#[test]
fn prograde_is_the_identity() {
    let o = Orientation::prograde();
    for v in [DVec3::X, DVec3::Y, DVec3::Z, DVec3::new(1.0, -2.0, 3.0)] {
        assert!(
            (o.apply(v) - v).length() < 1e-15,
            "prograde must be identity, mapped {v:?} → {:?}",
            o.apply(v)
        );
    }
}

#[test]
fn retrograde_flips_the_spin_to_minus_z() {
    let o = Orientation::retrograde();
    let spin = o.apply(SPIN);
    assert!(
        (spin - (-DVec3::Z)).length() < 1e-12,
        "retrograde spin must be −Z, got {spin:?}"
    );
    // A π rotation about +x keeps the line of nodes (x) fixed and flips y with z.
    assert!(
        (o.apply(DVec3::X) - DVec3::X).length() < 1e-12,
        "node line +x must be fixed"
    );
}

#[test]
fn inclined_tilts_the_spin_by_the_requested_angle() {
    for &i in &[
        0.0_f64,
        0.3,
        1.0,
        std::f64::consts::FRAC_PI_2,
        2.5,
        std::f64::consts::PI,
    ] {
        let o = Orientation::inclined(i);
        let spin = o.apply(SPIN);
        assert!(
            (angle_off_z(spin) - i).abs() < 1e-12,
            "inclined({i}): spin tilt {} off +Z, expected {i}",
            angle_off_z(spin)
        );
        // Tilt is about the line of nodes (+x): the tilted spin has no x-component.
        assert!(
            spin.x.abs() < 1e-12,
            "inclined({i}) tilts about +x, spin.x={}",
            spin.x
        );
    }
    // inclined(0) == prograde, inclined(π) == retrograde.
    assert!((Orientation::inclined(0.0).apply(SPIN) - DVec3::Z).length() < 1e-12);
    assert!(
        (Orientation::inclined(std::f64::consts::PI).apply(SPIN) - (-DVec3::Z)).length() < 1e-12
    );
}

#[test]
fn from_angles_places_the_node_line_by_the_argument() {
    // Toomre parameters: tilt by `inclination` about a node line at azimuth
    // `argument`. The spin axis +Z must map to (sin i·sin ω, −sin i·cos ω, cos i),
    // a tilt of exactly `inclination` off +Z regardless of `argument`.
    let i = 0.7_f64;
    for &arg in &[0.0_f64, 0.9, 2.0, 4.5] {
        let o = Orientation::from_angles(i, arg);
        let spin = o.apply(SPIN);
        let want = DVec3::new(i.sin() * arg.sin(), -i.sin() * arg.cos(), i.cos());
        assert!(
            (spin - want).length() < 1e-12,
            "from_angles({i},{arg}): spin {spin:?} vs {want:?}"
        );
        assert!(
            (angle_off_z(spin) - i).abs() < 1e-12,
            "argument must not change the tilt off +Z"
        );
    }
    // Consistency with the named constructors.
    assert!(
        (Orientation::from_angles(0.0, 0.0).apply(DVec3::new(1.0, 2.0, 3.0))
            - DVec3::new(1.0, 2.0, 3.0))
        .length()
            < 1e-15,
        "from_angles(0,0) is the identity (prograde)"
    );
}

#[test]
fn orientation_is_rigid_length_and_dot_preserving() {
    let o = Orientation::from_angles(1.1, 2.3);
    let a = DVec3::new(1.0, -2.0, 0.5);
    let b = DVec3::new(-0.3, 0.7, 2.0);
    assert!(
        (o.apply(a).length() - a.length()).abs() < 1e-12,
        "rotation must preserve length"
    );
    assert!(
        (o.apply(a).dot(o.apply(b)) - a.dot(b)).abs() < 1e-12,
        "rotation must preserve dot products (rigid)"
    );
}
