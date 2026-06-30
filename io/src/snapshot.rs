//! The Rust-native snapshot format: a versioned header followed by per-particle
//! Structure-of-Arrays columns, all little-endian.
//!
//! On-disk layout (version 1):
//! ```text
//!   magic   : 8 bytes  = b"GLXYSNAP"
//!   version : u32       = FORMAT_VERSION
//!   header  : time f64, step u64, scale_factor f64, softening f64,
//!             n_particles u64, rng_seed u64, config_hash u64,
//!             units String, code_version String         (Strings: u32 len + UTF-8)
//!   columns : pos[n] (3×f64), vel[n] (3×f64), mass[n] (f32),
//!             id[n] (u64), progenitor[n] (u16)
//! ```
//! Columns are stored SoA (all of one field, then the next) so a consumer can
//! read only the fields it needs. `n_particles` is authoritative on write — it is
//! always taken from the `State`, never from a caller-supplied header field.

use std::io::{Read, Write};
use std::path::Path;

use galaxy_core::State;

/// Magic bytes identifying a galaxy snapshot stream.
pub const MAGIC: [u8; 8] = *b"GLXYSNAP";
/// On-disk format version. Bumped when the layout changes incompatibly.
pub const FORMAT_VERSION: u32 = 1;

/// Errors from reading or writing a snapshot.
#[derive(thiserror::Error, Debug)]
pub enum SnapshotError {
    /// Underlying I/O failure (includes truncation: a short read surfaces here as
    /// `UnexpectedEof`).
    #[error("snapshot I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// The stream did not begin with the snapshot magic bytes.
    #[error("not a galaxy snapshot (bad magic)")]
    BadMagic,
    /// The format version on disk is not one this build can read.
    #[error("unsupported snapshot format version {found} (this build reads {expected})")]
    UnsupportedVersion { found: u32, expected: u32 },
    /// The stream is structurally invalid (e.g. an impossible length).
    #[error("corrupt snapshot: {0}")]
    Corrupt(String),
    /// A header string was not valid UTF-8.
    #[error("invalid UTF-8 in snapshot header: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
}

/// Snapshot header (DESIGN.md Contract 1). Carries the run metadata needed to
/// interpret the columns and to reproduce / validate the run.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    /// Simulation time of this snapshot.
    pub time: f64,
    /// Step index of this snapshot.
    pub step: u64,
    /// Cosmological scale factor `a` (1.0 for non-cosmological runs).
    pub scale_factor: f64,
    /// Gravitational softening length ε used by the force solver.
    pub softening: f64,
    /// Number of particles. Authoritative value is the `State` length on write.
    pub n_particles: u64,
    /// RNG seed that produced the run's initial conditions.
    pub rng_seed: u64,
    /// Hash of the scenario configuration (for reproducibility bookkeeping).
    pub config_hash: u64,
    /// Free-form units tag, e.g. `"nbody-G1"`.
    pub units: String,
    /// Code version that wrote the snapshot (e.g. the crate version).
    pub code_version: String,
}

impl Header {
    /// Build a header for `state`, taking `time`, `scale_factor`, and
    /// `n_particles` directly from it and `code_version` from this crate, so the
    /// header cannot disagree with the data it describes.
    pub fn for_state(
        state: &State,
        step: u64,
        softening: f64,
        rng_seed: u64,
        config_hash: u64,
        units: impl Into<String>,
    ) -> Self {
        let _ = (state, step, softening, rng_seed, config_hash, units);
        todo!()
    }
}

/// Write a snapshot to any sink. `n_particles` is taken from `state`; the rest of
/// the metadata is taken from `header`.
pub fn to_writer<W: Write>(
    writer: &mut W,
    header: &Header,
    state: &State,
) -> Result<(), SnapshotError> {
    let _ = (writer, header, state);
    todo!()
}

/// Read a snapshot from any source, reconstructing the `(Header, State)`. The
/// reconstructed `State` takes `time`/`a` from the header; `mass` comes back as
/// the f32-rounded value (the one lossy field).
pub fn from_reader<R: Read>(reader: &mut R) -> Result<(Header, State), SnapshotError> {
    let _ = reader;
    todo!()
}

/// Convenience: write a snapshot to a file (buffered).
pub fn write_file<P: AsRef<Path>>(
    path: P,
    header: &Header,
    state: &State,
) -> Result<(), SnapshotError> {
    let _ = (path.as_ref(), header, state);
    todo!()
}

/// Convenience: read a snapshot from a file (buffered).
pub fn read_file<P: AsRef<Path>>(path: P) -> Result<(Header, State), SnapshotError> {
    let _ = path.as_ref();
    todo!()
}
