//! The `Species` column (DESIGN.md M7a, plan D1): the physical particle-type
//! tag lives on `State` as one SoA column, defaults to `Collisionless`
//! everywhere gas isn't explicitly requested, and is covered by the SoA
//! consistency check.

use galaxy_core::{DVec3, Species, State};

#[test]
fn from_phase_space_defaults_to_collisionless() {
    let s = State::from_phase_space(
        vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
        vec![DVec3::ZERO; 2],
        vec![1.0, 2.0],
    );
    assert_eq!(s.kind, vec![Species::Collisionless; 2]);
    s.assert_consistent();
}

#[test]
#[should_panic(expected = "kind length mismatch")]
fn assert_consistent_covers_the_kind_column() {
    let mut s = State::from_phase_space(
        vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
        vec![DVec3::ZERO; 2],
        vec![1.0, 2.0],
    );
    s.kind.pop(); // a missed construction site shows up exactly like this
    s.assert_consistent();
}

#[test]
fn species_is_one_byte() {
    // The snapshot v2 column stores one u8 per particle; the enum must not grow.
    assert_eq!(std::mem::size_of::<Species>(), 1);
}
