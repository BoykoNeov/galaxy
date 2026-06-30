//! Snapshot round-trip and robustness (DESIGN.md M2 / Contract 1).
//!
//! The snapshot is the decoupling contract between the simulator and every
//! downstream consumer, so the tests pin two things:
//!   1. **Fidelity** — a write→read round-trip recovers the header exactly and the
//!      particle data exactly, with `mass` the single documented lossy field
//!      (stored f32) recovered to its f32-rounded value, not merely "close".
//!   2. **Robustness** — bad magic, an unknown version, and truncated/garbage
//!      input all produce a typed `Err`, never a panic or a silent wrong read.

use std::io::Cursor;

use galaxy_core::{DVec3, ParticleId, Progenitor, State};
use galaxy_io::snapshot::{self, SnapshotError, FORMAT_VERSION, MAGIC};
use galaxy_io::Header;

/// A small, fully-populated state with non-trivial values in every column:
/// distinct progenitors, non-contiguous-looking ids, and masses that are NOT
/// exactly representable in f32 (so the lossy round-trip is actually exercised).
fn sample_state() -> State {
    let pos = vec![
        DVec3::new(1.5, -2.25, 3.125),
        DVec3::new(-4.0, 5.5, -6.75),
        DVec3::new(0.1, 0.2, 0.3),
    ];
    let vel = vec![
        DVec3::new(-0.5, 0.25, -0.125),
        DVec3::new(7.0, -8.0, 9.0),
        DVec3::new(0.01, -0.02, 0.03),
    ];
    let mass = vec![0.1_f64, 0.3, 1.0 / 3.0];
    State {
        pos,
        vel,
        mass,
        id: vec![ParticleId(10), ParticleId(20), ParticleId(30)],
        progenitor: vec![Progenitor(0), Progenitor(1), Progenitor(0)],
        time: 12.5,
        a: 1.0,
    }
}

fn sample_header(state: &State) -> Header {
    Header::for_state(state, 7, 0.05, 0xABCD_1234, 0x9999_8888, "nbody-G1")
}

#[test]
fn round_trip_recovers_header_exactly() {
    let state = sample_state();
    let header = sample_header(&state);

    let mut buf = Vec::new();
    snapshot::to_writer(&mut buf, &header, &state).unwrap();
    let (back, _) = snapshot::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back, header, "header did not round-trip exactly");
}

#[test]
fn round_trip_recovers_particle_data() {
    let state = sample_state();
    let header = sample_header(&state);

    let mut buf = Vec::new();
    snapshot::to_writer(&mut buf, &header, &state).unwrap();
    let (back_header, back) = snapshot::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back.len(), state.len());
    // Positions and velocities are full f64 — bit-exact.
    assert_eq!(back.pos, state.pos, "positions not exact");
    assert_eq!(back.vel, state.vel, "velocities not exact");
    assert_eq!(back.id, state.id, "ids not exact");
    assert_eq!(back.progenitor, state.progenitor, "progenitors not exact");

    // Mass is the one lossy field (f32 storage): it must equal the f32-rounded
    // original exactly — the contract is "round-trips the stored f32", not "close".
    for (got, &orig) in back.mass.iter().zip(&state.mass) {
        assert_eq!(*got, orig as f32 as f64, "mass not the f32-rounded value");
    }

    // time / scale_factor come back via the header.
    assert_eq!(back.time, state.time);
    assert_eq!(back.a, state.a);
    assert_eq!(back_header.n_particles, state.len() as u64);
}

#[test]
fn header_for_state_takes_count_time_and_scale_from_state() {
    let state = sample_state();
    let h = Header::for_state(&state, 3, 0.02, 1, 2, "u");
    assert_eq!(h.n_particles, state.len() as u64);
    assert_eq!(h.time, state.time);
    assert_eq!(h.scale_factor, state.a);
    assert_eq!(h.step, 3);
    assert_eq!(h.softening, 0.02);
    assert!(
        !h.code_version.is_empty(),
        "code_version should be populated"
    );
}

#[test]
fn write_count_comes_from_state_not_header() {
    // Even if the header field disagrees, the on-disk count follows the data.
    let state = sample_state();
    let mut header = sample_header(&state);
    header.n_particles = 999; // deliberately wrong

    let mut buf = Vec::new();
    snapshot::to_writer(&mut buf, &header, &state).unwrap();
    let (back, st) = snapshot::from_reader(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(back.n_particles, state.len() as u64);
    assert_eq!(st.len(), state.len());
}

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = b"NOTGLXY!".to_vec();
    bytes.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    let err = snapshot::from_reader(&mut Cursor::new(&bytes)).unwrap_err();
    assert!(matches!(err, SnapshotError::BadMagic), "got {err:?}");
}

#[test]
fn unsupported_version_is_rejected() {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
    let err = snapshot::from_reader(&mut Cursor::new(&bytes)).unwrap_err();
    match err {
        SnapshotError::UnsupportedVersion { found, expected } => {
            assert_eq!(found, FORMAT_VERSION + 1);
            assert_eq!(expected, FORMAT_VERSION);
        }
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
}

#[test]
fn truncated_stream_is_an_error_not_a_panic() {
    let state = sample_state();
    let header = sample_header(&state);
    let mut buf = Vec::new();
    snapshot::to_writer(&mut buf, &header, &state).unwrap();

    // Cut the stream in the middle of the columns: header parses, data runs short.
    buf.truncate(buf.len() / 2);
    assert!(
        snapshot::from_reader(&mut Cursor::new(&buf)).is_err(),
        "a truncated snapshot must error, not panic or read garbage"
    );
}

#[test]
fn empty_and_garbage_streams_error() {
    assert!(snapshot::from_reader(&mut Cursor::new(Vec::new())).is_err());
    let garbage = vec![0xABu8; 5];
    assert!(snapshot::from_reader(&mut Cursor::new(garbage)).is_err());
}

#[test]
fn file_round_trip() {
    let state = sample_state();
    let header = sample_header(&state);

    let mut path = std::env::temp_dir();
    path.push(format!(
        "galaxy_io_file_round_trip_{}.snap",
        std::process::id()
    ));

    snapshot::write_file(&path, &header, &state).unwrap();
    let (back_header, back) = snapshot::read_file(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(back_header, header);
    assert_eq!(back.pos, state.pos);
    assert_eq!(back.id, state.id);
}

#[test]
fn empty_state_round_trips() {
    let state = State::from_phase_space(Vec::new(), Vec::new(), Vec::new());
    let header = Header::for_state(&state, 0, 0.01, 0, 0, "u");
    let mut buf = Vec::new();
    snapshot::to_writer(&mut buf, &header, &state).unwrap();
    let (back_h, back) = snapshot::from_reader(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(back.len(), 0);
    assert_eq!(back_h.n_particles, 0);
}
