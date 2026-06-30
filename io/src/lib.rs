//! `galaxy-io`: snapshot read/write for the N-body engine.
//!
//! The Rust-native snapshot is the primary on-disk format (DESIGN.md Contract 1):
//! a versioned header plus per-particle Structure-of-Arrays columns, written
//! little-endian. The schema is the **decoupling contract** between the simulator
//! and every downstream consumer (renderprep, validation), so it is versioned and
//! changes deliberately. HDF5 export lives behind a separate `validation` feature
//! (a later milestone) to dodge the Windows HDF5 link landmine.
//!
//! Only the particle `mass` is stored lossily (f32, per the contract's mixed
//! f64-compute / f32-storage plan); positions and velocities are full f64 and
//! round-trip bit-exact.

pub mod snapshot;

pub use snapshot::{from_reader, read_file, to_writer, write_file, Header, SnapshotError};
