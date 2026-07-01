//! Frame-data round-trip and robustness (DESIGN.md M3 / Contract 3).
//!
//! Frame-data is the decoupling boundary between renderprep and every consumer
//! (wgpu render, Blender), so the tests pin two things, mirroring the snapshot
//! contract tests:
//!   1. **Fidelity** — a write→read round-trip recovers the header and every column
//!      exactly. Unlike the snapshot there is no lossy field: the whole stage is
//!      already f32, so "exact" means bit-exact for all columns.
//!   2. **Robustness** — bad magic, an unknown version, and truncated input all
//!      produce a typed `Err`, never a panic or a silent wrong read.

use std::io::Cursor;

use galaxy_renderprep::frame::{self, FrameError, FRAME_VERSION, MAGIC};
use galaxy_renderprep::{FrameData, FrameHeader};
use glam::Vec3;

/// A small, fully-populated frame with distinct values in every column.
fn sample_frame() -> FrameData {
    FrameData {
        pos: vec![
            Vec3::new(1.5, -2.25, 3.125),
            Vec3::new(-4.0, 5.5, -6.75),
            Vec3::new(0.1, 0.2, 0.3),
        ],
        color: vec![[1.0, 0.4, 0.2], [0.2, 0.5, 1.0], [0.9, 0.9, 0.9]],
        size: vec![1.0, 2.5, 0.75],
        brightness: vec![10.0, 0.5, 3.25],
    }
}

#[test]
fn round_trip_recovers_header_exactly() {
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 12.5);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data).unwrap();
    let (back, _) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back, header, "header did not round-trip exactly");
}

#[test]
fn round_trip_recovers_columns_exactly() {
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 12.5);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data).unwrap();
    let (_, back) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

    // Everything is f32 — every column is bit-exact, no lossy field.
    assert_eq!(back, data, "frame columns did not round-trip exactly");
}

#[test]
fn header_for_data_takes_count_and_bounds_from_data() {
    let data = sample_frame();
    let h = FrameHeader::for_data(&data, 7.0);
    assert_eq!(h.n_particles, data.len() as u64);
    assert_eq!(h.time, 7.0);
    // Bounds must enclose every position.
    assert_eq!(h.bounds_min, Vec3::new(-4.0, -2.25, -6.75));
    assert_eq!(h.bounds_max, Vec3::new(1.5, 5.5, 3.125));
}

#[test]
fn write_count_comes_from_data_not_header() {
    // Even if the header field disagrees, the on-disk count follows the columns.
    let data = sample_frame();
    let mut header = FrameHeader::for_data(&data, 1.0);
    header.n_particles = 999; // deliberately wrong

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data).unwrap();
    let (back, back_data) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(back.n_particles, data.len() as u64);
    assert_eq!(back_data.len(), data.len());
}

#[test]
fn empty_frame_round_trips() {
    let data = FrameData::default();
    let header = FrameHeader::for_data(&data, 0.0);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data).unwrap();
    let (back_h, back) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back.len(), 0);
    assert_eq!(back_h.n_particles, 0);
}

#[test]
fn bounds_of_empty_frame_is_zero() {
    let data = FrameData::default();
    assert_eq!(data.bounds(), (Vec3::ZERO, Vec3::ZERO));
}

#[test]
fn bad_magic_is_rejected() {
    let mut bytes = b"NOTFRAME".to_vec();
    bytes.extend_from_slice(&FRAME_VERSION.to_le_bytes());
    let err = frame::from_reader(&mut Cursor::new(&bytes)).unwrap_err();
    assert!(matches!(err, FrameError::BadMagic), "got {err:?}");
}

#[test]
fn unsupported_version_is_rejected() {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&(FRAME_VERSION + 1).to_le_bytes());
    let err = frame::from_reader(&mut Cursor::new(&bytes)).unwrap_err();
    match err {
        FrameError::UnsupportedVersion { found, expected } => {
            assert_eq!(found, FRAME_VERSION + 1);
            assert_eq!(expected, FRAME_VERSION);
        }
        other => panic!("expected UnsupportedVersion, got {other:?}"),
    }
}

#[test]
fn truncated_stream_is_rejected() {
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 1.0);
    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data).unwrap();

    // Chop the columns in half: header parses, the column read hits EOF.
    buf.truncate(buf.len() - 20);
    let err = frame::from_reader(&mut Cursor::new(&buf)).unwrap_err();
    assert!(matches!(err, FrameError::Io(_)), "got {err:?}");
}
