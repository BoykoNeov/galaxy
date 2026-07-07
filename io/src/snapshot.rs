//! The Rust-native snapshot format: a versioned header followed by per-particle
//! Structure-of-Arrays columns, all little-endian.
//!
//! On-disk layout (version 3):
//! ```text
//!   magic   : 8 bytes  = b"GLXYSNAP"
//!   version : u32       = FORMAT_VERSION
//!   header  : time f64, step u64, scale_factor f64, softening f64,
//!             n_particles u64, rng_seed u64, config_hash u64,
//!             units String, code_version String         (Strings: u32 len + UTF-8)
//!   columns : pos[n] (3×f64), vel[n] (3×f64), mass[n] (f32),
//!             id[n] (u64), progenitor[n] (u16), kind[n] (u8), u[n] (f64)
//! ```
//! Columns are stored SoA (all of one field, then the next) so a consumer can
//! read only the fields it needs. `n_particles` is authoritative on write — it is
//! always taken from the `State`, never from a caller-supplied header field.
//!
//! The `u` column is stored **f64** (not f32 like `mass`): it is the evolved
//! thermodynamic variable of the adiabatic path and feeds the total-energy
//! conservation gate, so it cannot afford the f32 storage error `mass` accepts.
//!
//! Version history: v2 (M7a) appended the `kind` species column; v3 (energy
//! equation, Chain A) appended the per-particle internal-energy `u` column. The
//! reader accepts older streams: v1 (the retained pre-gas scenario zoo) defaults
//! every particle to `Species::Collisionless` (v1 predates gas), and v1/v2 both
//! default `u = 0.0` (they predate the energy equation, and `u = 0` is exactly
//! the inert isothermal value). The writer always emits the current version.

use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};

/// Upper bound on a header string length, to reject garbage before allocating.
const MAX_STR_LEN: usize = 1 << 16;

/// Magic bytes identifying a galaxy snapshot stream.
pub const MAGIC: [u8; 8] = *b"GLXYSNAP";
/// On-disk format version written by this build. Bumped when the layout
/// changes; older versions remain readable back to v1 (see the module docs).
pub const FORMAT_VERSION: u32 = 2;

/// Oldest on-disk format version this build can still read.
pub const MIN_READ_VERSION: u32 = 1;

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
    #[error("unsupported snapshot format version {found} (this build reads up to {expected})")]
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
        Header {
            time: state.time,
            step,
            scale_factor: state.a,
            softening,
            n_particles: state.len() as u64,
            rng_seed,
            config_hash,
            units: units.into(),
            code_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Write a snapshot to any sink. `n_particles` is taken from `state`; the rest of
/// the metadata is taken from `header`.
pub fn to_writer<W: Write>(
    writer: &mut W,
    header: &Header,
    state: &State,
) -> Result<(), SnapshotError> {
    let n = state.len();
    // Honor the no-panic-on-fallible-paths convention: a malformed State is a
    // typed error here rather than an index panic mid-write.
    if state.vel.len() != n
        || state.mass.len() != n
        || state.id.len() != n
        || state.progenitor.len() != n
        || state.kind.len() != n
        || state.u.len() != n
    {
        return Err(SnapshotError::Corrupt(
            "State SoA columns have mismatched lengths".to_string(),
        ));
    }

    writer.write_all(&MAGIC)?;
    write_u32(writer, FORMAT_VERSION)?;

    write_f64(writer, header.time)?;
    write_u64(writer, header.step)?;
    write_f64(writer, header.scale_factor)?;
    write_f64(writer, header.softening)?;
    write_u64(writer, n as u64)?; // authoritative count, taken from the data
    write_u64(writer, header.rng_seed)?;
    write_u64(writer, header.config_hash)?;
    write_str(writer, &header.units)?;
    write_str(writer, &header.code_version)?;

    // Columns, SoA: all of one field before the next.
    for &p in &state.pos {
        write_vec3(writer, p)?;
    }
    for &v in &state.vel {
        write_vec3(writer, v)?;
    }
    for &m in &state.mass {
        write_f32(writer, m as f32)?; // the one lossy field: f64 compute -> f32 storage
    }
    for id in &state.id {
        write_u64(writer, id.0)?;
    }
    for pr in &state.progenitor {
        write_u16(writer, pr.0)?;
    }
    for k in &state.kind {
        write_u8(writer, *k as u8)?;
    }
    // RED baseline (E1a): v3 will append the `u` column here.
    Ok(())
}

/// Read a snapshot from any source, reconstructing the `(Header, State)`. The
/// reconstructed `State` takes `time`/`a` from the header; `mass` comes back as
/// the f32-rounded value (the one lossy field).
pub fn from_reader<R: Read>(reader: &mut R) -> Result<(Header, State), SnapshotError> {
    let magic: [u8; 8] = read_array(reader)?;
    if magic != MAGIC {
        return Err(SnapshotError::BadMagic);
    }
    let version = read_u32(reader)?;
    if !(MIN_READ_VERSION..=FORMAT_VERSION).contains(&version) {
        return Err(SnapshotError::UnsupportedVersion {
            found: version,
            expected: FORMAT_VERSION,
        });
    }

    let time = read_f64(reader)?;
    let step = read_u64(reader)?;
    let scale_factor = read_f64(reader)?;
    let softening = read_f64(reader)?;
    let n_particles = read_u64(reader)?;
    let rng_seed = read_u64(reader)?;
    let config_hash = read_u64(reader)?;
    let units = read_str(reader)?;
    let code_version = read_str(reader)?;

    let n = usize::try_from(n_particles)
        .map_err(|_| SnapshotError::Corrupt(format!("n_particles {n_particles} too large")))?;
    // Capacity is only a hint — capped so a garbage count cannot trigger a huge
    // allocation; the read loops grow the vectors and error out on a short stream.
    let cap = n.min(1 << 20);

    let mut pos = Vec::with_capacity(cap);
    for _ in 0..n {
        pos.push(read_vec3(reader)?);
    }
    let mut vel = Vec::with_capacity(cap);
    for _ in 0..n {
        vel.push(read_vec3(reader)?);
    }
    let mut mass = Vec::with_capacity(cap);
    for _ in 0..n {
        mass.push(read_f32(reader)? as f64);
    }
    let mut id = Vec::with_capacity(cap);
    for _ in 0..n {
        id.push(ParticleId(read_u64(reader)?));
    }
    let mut progenitor = Vec::with_capacity(cap);
    for _ in 0..n {
        progenitor.push(Progenitor(read_u16(reader)?));
    }
    // The species column arrived in v2; v1 predates gas, so every v1 particle
    // is collisionless by construction.
    let kind = if version >= 2 {
        let mut kind = Vec::with_capacity(cap);
        for _ in 0..n {
            kind.push(match read_u8(reader)? {
                0 => Species::Collisionless,
                1 => Species::Gas,
                other => {
                    return Err(SnapshotError::Corrupt(format!(
                        "unknown species byte {other:#04x}"
                    )))
                }
            });
        }
        kind
    } else {
        vec![Species::Collisionless; n]
    };

    // RED baseline (E1a): v3 reads the `u` column here; for now every particle
    // defaults to `u = 0.0`, so a state with nonzero `u` fails to round-trip.
    let u = vec![0.0; n];

    let header = Header {
        time,
        step,
        scale_factor,
        softening,
        n_particles,
        rng_seed,
        config_hash,
        units,
        code_version,
    };
    let state = State {
        pos,
        vel,
        mass,
        id,
        progenitor,
        kind,
        u,
        time,
        a: scale_factor,
    };
    Ok((header, state))
}

/// Convenience: write a snapshot to a file (buffered).
pub fn write_file<P: AsRef<Path>>(
    path: P,
    header: &Header,
    state: &State,
) -> Result<(), SnapshotError> {
    let mut writer = BufWriter::new(File::create(path)?);
    to_writer(&mut writer, header, state)?;
    writer.flush()?; // surface flush errors instead of swallowing them on drop
    Ok(())
}

/// Convenience: read a snapshot from a file (buffered).
pub fn read_file<P: AsRef<Path>>(path: P) -> Result<(Header, State), SnapshotError> {
    let mut reader = BufReader::new(File::open(path)?);
    from_reader(&mut reader)
}

// ---------- little-endian primitive (de)serialization ----------

fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}
fn write_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
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
fn write_vec3<W: Write>(w: &mut W, v: DVec3) -> io::Result<()> {
    write_f64(w, v.x)?;
    write_f64(w, v.y)?;
    write_f64(w, v.z)
}
fn write_str<W: Write>(w: &mut W, s: &str) -> Result<(), SnapshotError> {
    if s.len() > MAX_STR_LEN {
        return Err(SnapshotError::Corrupt(format!(
            "header string of {} bytes exceeds the {MAX_STR_LEN}-byte limit",
            s.len()
        )));
    }
    write_u32(w, s.len() as u32)?;
    w.write_all(s.as_bytes())?;
    Ok(())
}

fn read_array<R: Read, const N: usize>(r: &mut R) -> io::Result<[u8; N]> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}
fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    Ok(read_array::<R, 1>(r)?[0])
}
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    Ok(u16::from_le_bytes(read_array(r)?))
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
fn read_vec3<R: Read>(r: &mut R) -> io::Result<DVec3> {
    Ok(DVec3::new(read_f64(r)?, read_f64(r)?, read_f64(r)?))
}
fn read_str<R: Read>(r: &mut R) -> Result<String, SnapshotError> {
    let len = read_u32(r)? as usize;
    if len > MAX_STR_LEN {
        return Err(SnapshotError::Corrupt(format!(
            "header string length {len} exceeds the {MAX_STR_LEN}-byte limit"
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8(buf)?)
}
