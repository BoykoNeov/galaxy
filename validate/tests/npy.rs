//! The `.npy` writer must emit spec-conformant NumPy v1.0 files: correct magic
//! and version, a 64-byte-aligned ASCII header advertising `<f8` / C-order / the
//! right shape, and C-order little-endian data. We re-parse the bytes in pure Rust
//! (no numpy) so the test is self-contained yet pins exactly what numpy will read.

use galaxy_validate::npy;
use glam::DVec3;

/// A parsed `.npy`: dtype descr, fortran_order flag, shape, and the raw f64 data.
struct Npy {
    descr: String,
    fortran_order: bool,
    shape: Vec<usize>,
    data: Vec<f64>,
}

fn parse_npy(bytes: &[u8]) -> Npy {
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "bad magic");
    assert_eq!(bytes[6], 1, "major version");
    assert_eq!(bytes[7], 0, "minor version");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let preamble = 10;
    assert_eq!(
        (preamble + header_len) % 64,
        0,
        "data start must be 64-byte aligned"
    );
    let header = std::str::from_utf8(&bytes[preamble..preamble + header_len]).unwrap();
    assert!(header.ends_with('\n'), "header must end with newline");

    // Pull the three fields out of the dict text (robust to spacing).
    let descr = extract(header, "'descr':")
        .trim_matches(|c| c == '\'' || c == ' ')
        .to_string();
    let fortran_order = extract(header, "'fortran_order':").contains("True");
    let shape = parse_shape(header);

    let raw = &bytes[preamble + header_len..];
    assert_eq!(raw.len() % 8, 0, "f8 data must be a multiple of 8 bytes");
    let data = raw
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();

    Npy {
        descr,
        fortran_order,
        shape,
        data,
    }
}

/// Text following `key` up to the next comma.
fn extract<'a>(header: &'a str, key: &str) -> &'a str {
    let start = header.find(key).expect("missing key") + key.len();
    let rest = &header[start..];
    let end = rest.find(',').unwrap_or(rest.len());
    rest[..end].trim()
}

fn parse_shape(header: &str) -> Vec<usize> {
    let open = header.find("'shape':").unwrap();
    let after = &header[open..];
    let lp = after.find('(').unwrap();
    let rp = after.find(')').unwrap();
    after[lp + 1..rp]
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().unwrap())
        .collect()
}

#[test]
fn vec3_array_is_well_formed_and_c_order() {
    let data = vec![
        DVec3::new(1.0, 2.0, 3.0),
        DVec3::new(-4.5, 5.25, -6.125),
        DVec3::new(0.1, 0.2, 0.3),
    ];
    let mut buf = Vec::new();
    npy::write_vec3(&mut buf, &data).unwrap();
    let p = parse_npy(&buf);

    assert_eq!(p.descr, "<f8");
    assert!(!p.fortran_order);
    assert_eq!(p.shape, vec![3, 3]);
    // C order: x0,y0,z0, x1,y1,z1, ...
    let expected: Vec<f64> = data.iter().flat_map(|v| [v.x, v.y, v.z]).collect();
    assert_eq!(p.data, expected);
}

#[test]
fn f64_array_is_well_formed() {
    let data = vec![0.1_f64, 1.0 / 3.0, -2.0, 1e300, 1e-300];
    let mut buf = Vec::new();
    npy::write_f64(&mut buf, &data).unwrap();
    let p = parse_npy(&buf);

    assert_eq!(p.descr, "<f8");
    assert!(!p.fortran_order);
    assert_eq!(p.shape, vec![5], "1-D shape should be (5,)");
    assert_eq!(p.data, data);
}

#[test]
fn empty_arrays_are_well_formed() {
    let mut buf = Vec::new();
    npy::write_vec3(&mut buf, &[]).unwrap();
    let p = parse_npy(&buf);
    assert_eq!(p.shape, vec![0, 3]);
    assert!(p.data.is_empty());

    let mut buf2 = Vec::new();
    npy::write_f64(&mut buf2, &[]).unwrap();
    let p2 = parse_npy(&buf2);
    assert_eq!(p2.shape, vec![0]);
    assert!(p2.data.is_empty());
}

#[test]
fn file_writers_round_trip() {
    let data = vec![DVec3::new(7.0, 8.0, 9.0)];
    let scalars = vec![42.0_f64, 43.0];

    let mut dir = std::env::temp_dir();
    dir.push(format!("galaxy_npy_test_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let vpath = dir.join("v.npy");
    let spath = dir.join("s.npy");

    npy::write_vec3_file(&vpath, &data).unwrap();
    npy::write_f64_file(&spath, &scalars).unwrap();

    let vp = parse_npy(&std::fs::read(&vpath).unwrap());
    let sp = parse_npy(&std::fs::read(&spath).unwrap());
    assert_eq!(vp.shape, vec![1, 3]);
    assert_eq!(vp.data, vec![7.0, 8.0, 9.0]);
    assert_eq!(sp.shape, vec![2]);
    assert_eq!(sp.data, scalars);

    let _ = std::fs::remove_dir_all(&dir);
}
