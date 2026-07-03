//! Frame-data round-trip and robustness (DESIGN.md M3 / Contract 3; v2 in M7d).
//!
//! Frame-data is the decoupling boundary between renderprep and every consumer
//! (wgpu render, Blender), so the tests pin two things, mirroring the snapshot
//! contract tests:
//!   1. **Fidelity** — a write→read round-trip recovers the header and every column
//!      exactly. Unlike the snapshot there is no lossy field: the whole stage is
//!      already f32, so "exact" means bit-exact for all columns — and for the v2
//!      gas block (dims, bounds, every cell).
//!   2. **Robustness** — bad magic, an unknown version, unknown flag bits, and
//!      truncated input all produce a typed `Err`, never a panic or a silent
//!      wrong read.
//!   3. **Compatibility** — v1 streams (retained pre-gas frame files) still read,
//!      with no gas; a stars-only v2 stream reads back with exactly the v1
//!      semantics.

use std::io::Cursor;

use galaxy_renderprep::frame::{self, FrameError, FLAG_GAS, FRAME_VERSION, MAGIC};
use galaxy_renderprep::{FrameData, FrameHeader, GasGrid};
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

/// A small gas grid with distinct, non-round values.
fn sample_gas() -> GasGrid {
    GasGrid {
        dims: [3, 2, 2],
        bounds_min: Vec3::new(-1.25, -2.5, 0.5),
        bounds_max: Vec3::new(3.75, 1.5, 4.5),
        data: (0..12).map(|i| 0.125 * i as f32 + 0.01).collect(),
    }
}

#[test]
fn round_trip_recovers_header_exactly() {
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 12.5);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, None).unwrap();
    let (back, _, _) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back, header, "header did not round-trip exactly");
}

#[test]
fn round_trip_recovers_columns_exactly() {
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 12.5);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, None).unwrap();
    let (_, back, _) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

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
    frame::to_writer(&mut buf, &header, &data, None).unwrap();
    let (back, back_data, _) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();
    assert_eq!(back.n_particles, data.len() as u64);
    assert_eq!(back_data.len(), data.len());
}

#[test]
fn empty_frame_round_trips() {
    let data = FrameData::default();
    let header = FrameHeader::for_data(&data, 0.0);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, None).unwrap();
    let (back_h, back, back_gas) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back.len(), 0);
    assert_eq!(back_h.n_particles, 0);
    assert!(back_gas.is_none());
}

#[test]
fn bounds_of_empty_frame_is_zero() {
    let data = FrameData::default();
    assert_eq!(data.bounds(), (Vec3::ZERO, Vec3::ZERO));
}

// ---------- v2: the optional gas block ----------

#[test]
fn gas_block_round_trips_bit_exact() {
    let data = sample_frame();
    let gas = sample_gas();
    let header = FrameHeader::for_data(&data, 3.5);

    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, Some(&gas)).unwrap();
    let (back_h, back, back_gas) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();

    assert_eq!(back_h, header);
    assert_eq!(back, data, "star columns disturbed by the gas block");
    let back_gas = back_gas.expect("gas block was written but not read back");
    assert_eq!(back_gas, gas, "gas grid did not round-trip bit-exactly");
}

#[test]
fn stars_only_write_reads_with_no_gas() {
    // v2 without a gas block ≡ the v1 semantics: same header, same columns,
    // gas `None` — the "stars-only v2 ≡ v1" contract.
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 2.0);
    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, None).unwrap();
    let (back_h, back, back_gas) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();
    assert_eq!((back_h, back), (header, data));
    assert!(back_gas.is_none());
}

#[test]
fn gas_block_with_empty_stars_round_trips() {
    // A gas-only frame (no splats at all) is legal: the volumetric path does
    // not require any star to exist.
    let data = FrameData::default();
    let gas = sample_gas();
    let header = FrameHeader::for_data(&data, 0.0);
    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, Some(&gas)).unwrap();
    let (_, back, back_gas) = frame::from_reader(&mut Cursor::new(&buf)).unwrap();
    assert!(back.is_empty());
    assert_eq!(back_gas, Some(gas));
}

#[test]
fn malformed_gas_grid_is_rejected_on_write() {
    // data length disagreeing with dims is a typed error, not a bad stream.
    let data = sample_frame();
    let mut gas = sample_gas();
    gas.data.pop(); // now 11 values for 3×2×2 = 12 cells
    let header = FrameHeader::for_data(&data, 1.0);
    let mut buf = Vec::new();
    let err = frame::to_writer(&mut buf, &header, &data, Some(&gas)).unwrap_err();
    assert!(matches!(err, FrameError::Corrupt(_)), "got {err:?}");
}

// ---------- v1 compatibility ----------

/// Hand-serialize a v1 stream (magic, version 1, header, star columns — no
/// flags word, no gas block), byte-for-byte the pre-M7d writer's layout.
fn v1_bytes(header: &FrameHeader, data: &FrameData) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&MAGIC);
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&header.time.to_le_bytes());
    b.extend_from_slice(&(data.len() as u64).to_le_bytes());
    let (bmin, bmax) = data.bounds();
    for v in [bmin, bmax] {
        for c in v.to_array() {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    for p in &data.pos {
        for c in p.to_array() {
            b.extend_from_slice(&c.to_le_bytes());
        }
    }
    for c in &data.color {
        for x in c {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    for s in &data.size {
        b.extend_from_slice(&s.to_le_bytes());
    }
    for br in &data.brightness {
        b.extend_from_slice(&br.to_le_bytes());
    }
    b
}

#[test]
fn v1_stream_reads_with_no_gas() {
    // Retained pre-gas frame files must stay readable: exact columns, no gas.
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 9.25);
    let bytes = v1_bytes(&header, &data);
    let (back_h, back, back_gas) = frame::from_reader(&mut Cursor::new(&bytes)).unwrap();
    assert_eq!(back_h, header);
    assert_eq!(back, data);
    assert!(back_gas.is_none(), "a v1 stream cannot carry gas");
}

// ---------- robustness ----------

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
fn version_zero_is_rejected() {
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(&0u32.to_le_bytes());
    assert!(frame::from_reader(&mut Cursor::new(&bytes)).is_err());
}

#[test]
fn unknown_flag_bits_are_rejected() {
    // A flags word with bits this build does not understand means a frame
    // written by a future layout — reading on would misparse; fail loudly.
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 1.0);
    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, None).unwrap();
    // The flags word sits right after the fixed header: magic(8) + version(4) +
    // time(8) + n(8) + bounds(24) = offset 52.
    let unknown = (FLAG_GAS << 1) | FLAG_GAS;
    buf[52..56].copy_from_slice(&unknown.to_le_bytes());
    let err = frame::from_reader(&mut Cursor::new(&buf)).unwrap_err();
    assert!(matches!(err, FrameError::Corrupt(_)), "got {err:?}");
}

#[test]
fn truncated_stream_is_rejected() {
    let data = sample_frame();
    let header = FrameHeader::for_data(&data, 1.0);
    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, None).unwrap();

    // Chop the columns in half: header parses, the column read hits EOF.
    buf.truncate(buf.len() - 20);
    let err = frame::from_reader(&mut Cursor::new(&buf)).unwrap_err();
    assert!(matches!(err, FrameError::Io(_)), "got {err:?}");
}

#[test]
fn truncated_gas_block_is_rejected() {
    let data = sample_frame();
    let gas = sample_gas();
    let header = FrameHeader::for_data(&data, 1.0);
    let mut buf = Vec::new();
    frame::to_writer(&mut buf, &header, &data, Some(&gas)).unwrap();

    buf.truncate(buf.len() - 6); // inside the gas data
    let err = frame::from_reader(&mut Cursor::new(&buf)).unwrap_err();
    assert!(matches!(err, FrameError::Io(_)), "got {err:?}");
}
