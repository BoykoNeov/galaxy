//! A minimal writer for the NumPy `.npy` v1.0 format — just enough to export
//! `f64` arrays so the REBOUND harness can `numpy.load` them with no extra deps.
//!
//! Layout: 6-byte magic `\x93NUMPY`, version `(1, 0)`, a little-endian `u16`
//! header length, then an ASCII Python-dict header (`descr`, `fortran_order`,
//! `shape`) padded with spaces and terminated by `\n` so that the data start
//! (10 + header length) is a multiple of 64. Data follows in C order, `<f8`.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use glam::DVec3;

const MAGIC: [u8; 6] = *b"\x93NUMPY";
/// Required alignment of the data section start (NumPy v1.0).
const ALIGN: usize = 64;
/// Bytes before the header string: magic(6) + version(2) + header-len(2).
const PREAMBLE: usize = 10;

/// Build the framed header (preamble + padded dict) for a `<f8` array of `shape`.
fn header(shape: &[usize]) -> Vec<u8> {
    let shape_str = match shape {
        [n] => format!("({n},)"),
        _ => {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("({})", dims.join(", "))
        }
    };
    let dict = format!("{{'descr': '<f8', 'fortran_order': False, 'shape': {shape_str}, }}");

    // Pad with spaces so PREAMBLE + (dict + pad + '\n') is a multiple of ALIGN,
    // with the newline as the final byte.
    let base = PREAMBLE + dict.len() + 1;
    let pad = (ALIGN - (base % ALIGN)) % ALIGN;
    let header_str_len = dict.len() + pad + 1;

    let mut out = Vec::with_capacity(PREAMBLE + header_str_len);
    out.extend_from_slice(&MAGIC);
    out.push(1); // major version
    out.push(0); // minor version
    out.extend_from_slice(&(header_str_len as u16).to_le_bytes());
    out.extend_from_slice(dict.as_bytes());
    out.extend(std::iter::repeat_n(b' ', pad));
    out.push(b'\n');
    out
}

/// Write `data` as a C-order `(N, 3)` little-endian `f64` array.
pub fn write_vec3<W: Write>(w: &mut W, data: &[DVec3]) -> io::Result<()> {
    w.write_all(&header(&[data.len(), 3]))?;
    for v in data {
        w.write_all(&v.x.to_le_bytes())?;
        w.write_all(&v.y.to_le_bytes())?;
        w.write_all(&v.z.to_le_bytes())?;
    }
    Ok(())
}

/// Write `data` as a `(N,)` little-endian `f64` array.
pub fn write_f64<W: Write>(w: &mut W, data: &[f64]) -> io::Result<()> {
    w.write_all(&header(&[data.len()]))?;
    for &x in data {
        w.write_all(&x.to_le_bytes())?;
    }
    Ok(())
}

/// Convenience: write a `(N, 3)` array to a file.
pub fn write_vec3_file<P: AsRef<Path>>(path: P, data: &[DVec3]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_vec3(&mut w, data)?;
    w.flush()
}

/// Convenience: write a `(N,)` array to a file.
pub fn write_f64_file<P: AsRef<Path>>(path: P, data: &[f64]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_f64(&mut w, data)?;
    w.flush()
}
