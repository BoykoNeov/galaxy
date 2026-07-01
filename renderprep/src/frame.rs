//! Frame-data schema (DESIGN.md Contract 3) and its versioned little-endian
//! (de)serialization.
//!
//! On-disk layout (version 1):
//! ```text
//!   magic   : 8 bytes  = b"GLXYFRAM"
//!   version : u32       = FRAME_VERSION
//!   header  : time f64, n_particles u64,
//!             bounds_min (3×f32), bounds_max (3×f32)
//!   columns : pos[n] (3×f32), color[n] (3×f32), size[n] (f32), brightness[n] (f32)
//! ```
//! Columns are SoA (all of one field, then the next), like the snapshot format, so
//! a consumer can read only what it needs. Everything is f32 — unlike the snapshot,
//! there is no lossy field to call out; the whole stage is already an f32 projection
//! of the f64 physics state.

use std::path::Path;

use glam::Vec3;

/// Magic bytes identifying a galaxy frame-data stream.
pub const MAGIC: [u8; 8] = *b"GLXYFRAM";
/// On-disk frame-data format version. Bumped when the layout changes incompatibly.
pub const FRAME_VERSION: u32 = 1;

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
    /// `(ZERO, ZERO)` for an empty frame.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        todo!("compute AABB over pos")
    }
}

/// Write frame-data to any sink. `n_particles` and `bounds` are taken from `data`;
/// `time` is taken from `header`.
pub fn to_writer<W: std::io::Write>(
    _writer: &mut W,
    _header: &FrameHeader,
    _data: &FrameData,
) -> Result<(), FrameError> {
    todo!("serialize frame-data (versioned LE)")
}

/// Read frame-data from any source, reconstructing `(FrameHeader, FrameData)`.
pub fn from_reader<R: std::io::Read>(
    _reader: &mut R,
) -> Result<(FrameHeader, FrameData), FrameError> {
    todo!("deserialize frame-data (versioned LE)")
}

/// Convenience: write frame-data to a file (buffered).
pub fn write_file<P: AsRef<Path>>(
    _path: P,
    _header: &FrameHeader,
    _data: &FrameData,
) -> Result<(), FrameError> {
    todo!("buffered file write")
}

/// Convenience: read frame-data from a file (buffered).
pub fn read_file<P: AsRef<Path>>(_path: P) -> Result<(FrameHeader, FrameData), FrameError> {
    todo!("buffered file read")
}
