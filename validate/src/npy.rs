//! A minimal writer for the NumPy `.npy` v1.0 format — just enough to export
//! `f64` arrays so the REBOUND harness can `numpy.load` them with no extra deps.
//!
//! Layout: 6-byte magic `\x93NUMPY`, version `(1, 0)`, a little-endian `u16`
//! header length, then an ASCII Python-dict header (`descr`, `fortran_order`,
//! `shape`) padded with spaces and terminated by `\n` so that the data start
//! (10 + header length) is a multiple of 64. Data follows in C order, `<f8`.

use std::io::{self, Write};
use std::path::Path;

use glam::DVec3;

/// Write `data` as a C-order `(N, 3)` little-endian `f64` array.
pub fn write_vec3<W: Write>(w: &mut W, data: &[DVec3]) -> io::Result<()> {
    let _ = (w, data);
    todo!()
}

/// Write `data` as a `(N,)` little-endian `f64` array.
pub fn write_f64<W: Write>(w: &mut W, data: &[f64]) -> io::Result<()> {
    let _ = (w, data);
    todo!()
}

/// Convenience: write a `(N, 3)` array to a file.
pub fn write_vec3_file<P: AsRef<Path>>(path: P, data: &[DVec3]) -> io::Result<()> {
    let _ = (path.as_ref(), data);
    todo!()
}

/// Convenience: write a `(N,)` array to a file.
pub fn write_f64_file<P: AsRef<Path>>(path: P, data: &[f64]) -> io::Result<()> {
    let _ = (path.as_ref(), data);
    todo!()
}
