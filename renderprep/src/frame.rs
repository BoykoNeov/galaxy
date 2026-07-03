//! Frame-data schema (DESIGN.md Contract 3) and its versioned little-endian
//! (de)serialization.
//!
//! On-disk layout (version 2):
//! ```text
//!   magic   : 8 bytes  = b"GLXYFRAM"
//!   version : u32       = FRAME_VERSION
//!   header  : time f64, n_particles u64,
//!             bounds_min (3×f32), bounds_max (3×f32)
//!   flags   : u32       (bit 0 = gas block present; other bits must be 0)
//!   columns : pos[n] (3×f32), color[n] (3×f32), size[n] (f32), brightness[n] (f32)
//!   gas     : (only if flag bit 0) dims (3×u32),
//!             gas_bounds_min (3×f32), gas_bounds_max (3×f32),
//!             data[dims.x·dims.y·dims.z] (f32, x-fastest)
//! ```
//! Columns are SoA (all of one field, then the next), like the snapshot format, so
//! a consumer can read only what it needs. Everything is f32 — unlike the snapshot,
//! there is no lossy field to call out; the whole stage is already an f32 projection
//! of the f64 physics state.
//!
//! Version history: v2 (M7d) inserted the `flags` word after the header and the
//! optional gas density grid after the star columns. The reader accepts v1
//! streams (no flags word, no gas block — retained pre-gas frame files) and
//! reports no gas for them; a stars-only v2 stream reads back with exactly the
//! v1 semantics. The writer always emits the current version.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use glam::Vec3;

use crate::gasgrid::GasGrid;

/// Magic bytes identifying a galaxy frame-data stream.
pub const MAGIC: [u8; 8] = *b"GLXYFRAM";
/// On-disk frame-data format version written by this build. Bumped when the
/// layout changes; older versions remain readable back to v1 (see module docs).
pub const FRAME_VERSION: u32 = 2;
/// Oldest on-disk frame-data version this build can still read.
pub const MIN_READ_VERSION: u32 = 1;
/// Flags-word bit: a gas density grid follows the star columns.
pub const FLAG_GAS: u32 = 1;

/// Errors from reading or writing frame-data.
#[derive(thiserror::Error, Debug)]
pub enum FrameError {
    /// Underlying I/O failure (a short read surfaces here as `UnexpectedEof`).
    #[error("frame-data I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The stream did not begin with the frame-data magic bytes.
    #[error("not galaxy frame-data (bad magic)")]
    BadMagic,
    /// The format version on disk is not one this build can read.
    #[error("unsupported frame-data version {found} (this build reads {expected})")]
    UnsupportedVersion { found: u32, expected: u32 },
    /// The stream is structurally invalid (e.g. an impossible length).
    #[error("corrupt frame-data: {0}")]
    Corrupt(String),
}

/// Frame-data header: run metadata plus the scene bounding box (so the renderer
/// can auto-frame a camera without re-reading every particle).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameHeader {
    /// Simulation time of the source snapshot (for overlays / bookkeeping).
    pub time: f64,
    /// Number of particles (authoritative on write: taken from the columns).
    pub n_particles: u64,
    /// Axis-aligned bounding box minimum over all particle positions.
    pub bounds_min: Vec3,
    /// Axis-aligned bounding box maximum over all particle positions.
    pub bounds_max: Vec3,
}

impl FrameHeader {
    /// Build a header for `data`, taking `n_particles` and `bounds` directly from
    /// it (authoritative, like the snapshot's count-from-state rule) and `time`
    /// from the caller, so the header cannot disagree with the columns.
    pub fn for_data(data: &FrameData, time: f64) -> Self {
        let (bounds_min, bounds_max) = data.bounds();
        FrameHeader {
            time,
            n_particles: data.len() as u64,
            bounds_min,
            bounds_max,
        }
    }
}

/// Per-particle visual attributes in Structure-of-Arrays layout (Contract 3).
///
/// All columns share one length. `pos` is camera-independent world space; the
/// camera lives in the renderer. `color` is linear emissive RGB; `brightness` is a
/// scalar multiplier applied at splat time; `size` is the splat radius.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FrameData {
    /// World-space particle positions (f32).
    pub pos: Vec<Vec3>,
    /// Linear emissive RGB per particle.
    pub color: Vec<[f32; 3]>,
    /// Splat radius per particle.
    pub size: Vec<f32>,
    /// Scalar brightness multiplier per particle.
    pub brightness: Vec<f32>,
}

impl FrameData {
    /// Number of particles.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// True if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// The axis-aligned bounding box `(min, max)` over all positions. Returns
    /// `(ZERO, ZERO)` for an empty frame (there is no meaningful box).
    pub fn bounds(&self) -> (Vec3, Vec3) {
        match self.pos.split_first() {
            None => (Vec3::ZERO, Vec3::ZERO),
            Some((&first, rest)) => rest
                .iter()
                .fold((first, first), |(mn, mx), &p| (mn.min(p), mx.max(p))),
        }
    }
}

/// Write frame-data to any sink. `n_particles` and `bounds` are taken from `data`
/// (authoritative); only `time` is taken from `header`. `gas` is the optional
/// voxelized gas density grid (frame-data v2); `None` writes a stars-only stream
/// whose read-back semantics are exactly v1's.
pub fn to_writer<W: Write>(
    writer: &mut W,
    header: &FrameHeader,
    data: &FrameData,
    gas: Option<&GasGrid>,
) -> Result<(), FrameError> {
    let _ = (writer, header, data, gas);
    todo!()
}

/// Read frame-data from any source, reconstructing
/// `(FrameHeader, FrameData, Option<GasGrid>)`. v1 streams read with no gas.
#[allow(clippy::type_complexity)]
pub fn from_reader<R: Read>(
    reader: &mut R,
) -> Result<(FrameHeader, FrameData, Option<GasGrid>), FrameError> {
    let _ = reader;
    todo!()
}

/// Convenience: write frame-data to a file (buffered).
pub fn write_file<P: AsRef<Path>>(
    path: P,
    header: &FrameHeader,
    data: &FrameData,
    gas: Option<&GasGrid>,
) -> Result<(), FrameError> {
    let mut writer = BufWriter::new(File::create(path)?);
    to_writer(&mut writer, header, data, gas)?;
    writer.flush()?; // surface flush errors instead of swallowing them on drop
    Ok(())
}

/// Convenience: read frame-data from a file (buffered).
#[allow(clippy::type_complexity)]
pub fn read_file<P: AsRef<Path>>(
    path: P,
) -> Result<(FrameHeader, FrameData, Option<GasGrid>), FrameError> {
    let mut reader = BufReader::new(File::open(path)?);
    from_reader(&mut reader)
}

// ---------- little-endian primitive (de)serialization ----------

fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f32<W: Write>(w: &mut W, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f64<W: Write>(w: &mut W, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_vec3<W: Write>(w: &mut W, v: Vec3) -> io::Result<()> {
    write_f32(w, v.x)?;
    write_f32(w, v.y)?;
    write_f32(w, v.z)
}

fn read_array<R: Read, const N: usize>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    Ok(u32::from_le_bytes(read_array(r)?))
}
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    Ok(u64::from_le_bytes(read_array(r)?))
}
fn read_f32<R: Read>(r: &mut R) -> io::Result<f32> {
    Ok(f32::from_le_bytes(read_array(r)?))
}
fn read_f64<R: Read>(r: &mut R) -> io::Result<f64> {
    Ok(f64::from_le_bytes(read_array(r)?))
}
fn read_vec3<R: Read>(r: &mut R) -> io::Result<Vec3> {
    Ok(Vec3::new(read_f32(r)?, read_f32(r)?, read_f32(r)?))
}
