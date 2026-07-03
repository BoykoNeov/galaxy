//! Gas voxelization gates (DESIGN.md M7d, plan D8).
//!
//! The deposition is the SPH density estimate at cell centers, so its oracles
//! are the kernel itself and conservation: a single particle's grid IS the
//! sampled kernel (exact, not a tolerance), a lattice slab recovers the
//! analytic density in its interior (M7a's uniform-lattice bound), and the
//! grid integral recovers the deposited mass within a quadrature tolerance
//! justified by the cell-size / smoothing-length ratio. Determinism gates
//! mirror the solvers::sph discipline: parallel ≡ serial bit-exact.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};
use galaxy_renderprep::gasgrid::{
    deposit_fixed, deposit_fixed_serial, deposit_gas, sample_mix, GasGrid, GasGridConfig,
};
use galaxy_solvers::sph::{w, SUPPORT};
use glam::Vec3;

/// Deterministic splitmix64 → f64 in [0, 1) — the test-local PRNG convention.
fn splitmix(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z = z ^ (z >> 31);
    (z >> 11) as f64 / (1u64 << 53) as f64
}

/// A pseudo-random cloud of `n` points in a `[-r, r]³` box.
fn random_cloud(n: usize, r: f64, seed: u64) -> Vec<DVec3> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            DVec3::new(
                (2.0 * splitmix(&mut s) - 1.0) * r,
                (2.0 * splitmix(&mut s) - 1.0) * r,
                (2.0 * splitmix(&mut s) - 1.0) * r,
            )
        })
        .collect()
}

/// An all-gas `State` at the given positions with unit masses.
fn gas_state(pos: Vec<DVec3>) -> State {
    let n = pos.len();
    State {
        vel: vec![DVec3::ZERO; n],
        mass: vec![1.0; n],
        id: (0..n as u64).map(ParticleId).collect(),
        progenitor: vec![Progenitor(4); n],
        kind: vec![Species::Gas; n],
        time: 0.0,
        a: 1.0,
        pos,
    }
}

// ---------- grid geometry ----------

#[test]
fn index_is_x_fastest() {
    let g = GasGrid {
        dims: [4, 3, 2],
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::ONE,
        data: vec![0.0; 24],
    };
    assert_eq!(g.cell_count(), 24);
    assert_eq!(g.index(0, 0, 0), 0);
    assert_eq!(g.index(1, 0, 0), 1);
    assert_eq!(g.index(0, 1, 0), 4);
    assert_eq!(g.index(0, 0, 1), 12);
    assert_eq!(g.index(3, 2, 1), 23);
}

#[test]
fn cell_centers_are_half_cell_off_the_min_corner() {
    // Box [0,8]×[0,4]×[0,2] with dims (4,2,1): cell edges 2×2×2, so cell (0,0,0)
    // is centered at (1,1,1) and cell (3,1,0) at (7,3,1) — hand values.
    let g = GasGrid {
        dims: [4, 2, 1],
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::new(8.0, 4.0, 2.0),
        data: vec![0.0; 8],
    };
    assert_eq!(g.cell_size(), DVec3::new(2.0, 2.0, 2.0));
    assert_eq!(g.cell_center(0, 0, 0), DVec3::new(1.0, 1.0, 1.0));
    assert_eq!(g.cell_center(3, 1, 0), DVec3::new(7.0, 3.0, 1.0));
}

// ---------- deposition vs the kernel oracle ----------

#[test]
fn single_particle_at_a_cell_center_reproduces_the_sampled_kernel() {
    // One unit-mass particle exactly at a cell center: the deposited grid must
    // equal the kernel sampled at every cell center EXACTLY (same w(), same
    // distances, a one-term sum — no tolerance).
    let dims = [8, 8, 8];
    let (bmin, bmax) = (Vec3::splat(-2.0), Vec3::splat(2.0));
    // Cell edge 0.5 ⇒ cell (4,4,4) is centered at (0.25, 0.25, 0.25).
    let p = DVec3::new(0.25, 0.25, 0.25);
    let h = 0.8;
    let grid = deposit_fixed(&[p], &[1.0], &[h], dims, bmin, bmax);

    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let c = grid.cell_center(ix, iy, iz);
                let expect = w((c - p).length(), h) as f32;
                assert_eq!(
                    grid.data[grid.index(ix, iy, iz)],
                    expect,
                    "cell ({ix},{iy},{iz}) is not the sampled kernel"
                );
            }
        }
    }
    // The particle's own cell carries the kernel's origin value W(0) = 1/(πh³).
    let self_cell = grid.data[grid.index(4, 4, 4)];
    assert_eq!(self_cell, (1.0 / (std::f64::consts::PI * h * h * h)) as f32);
}

#[test]
fn mass_linearity_is_exact() {
    // Doubling every particle mass doubles every cell bit-exactly (×2 is exact
    // in IEEE and commutes with the fixed-order sum).
    let pos = random_cloud(200, 3.0, 11);
    let h = vec![0.9; 200];
    let m1 = vec![1.5; 200];
    let m2 = vec![3.0; 200];
    let dims = [16, 16, 16];
    let (bmin, bmax) = (Vec3::splat(-5.0), Vec3::splat(5.0));
    let g1 = deposit_fixed(&pos, &m1, &h, dims, bmin, bmax);
    let g2 = deposit_fixed(&pos, &m2, &h, dims, bmin, bmax);
    for (a, b) in g1.data.iter().zip(&g2.data) {
        assert_eq!(*b, 2.0 * *a);
    }
}

#[test]
fn total_grid_mass_recovers_deposited_mass() {
    // The grid integral Σ ρ_cell · V_cell midpoint-quadratures ∫ρ dV = M when
    // every kernel lies inside the bounds. Midpoint error per kernel is
    // O((Δ/h)²) for the C¹ cubic spline; at Δ = h/5 (below) that is well under
    // 1% — asserted at 1%, the measured headroom documented in the impl.
    let s = 1.0; // lattice spacing
    let n_side = 10usize;
    let mut pos = Vec::new();
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                pos.push(DVec3::new(ix as f64, iy as f64, iz as f64) * s);
            }
        }
    }
    let n = pos.len();
    let mass = vec![2.0; n];
    let h = vec![1.3 * s; n]; // M7a's lattice-quadrature regime
    let pad = SUPPORT * 1.3 * s; // full kernel support inside the box
    let lo = -pad;
    let hi = (n_side - 1) as f64 * s + pad;
    // Cell edge = (hi - lo)/dims ≈ 0.24 ≈ h/5.4.
    let dims = [64, 64, 64];
    let grid = deposit_fixed(
        &pos,
        &mass,
        &h,
        dims,
        Vec3::splat(lo as f32),
        Vec3::splat(hi as f32),
    );

    let cell = grid.cell_size();
    let vol = cell.x * cell.y * cell.z;
    let total: f64 = grid.data.iter().map(|&d| d as f64 * vol).sum();
    let expect = 2.0 * n as f64;
    let rel = (total - expect).abs() / expect;
    assert!(rel < 0.01, "grid mass {total} vs deposited {expect} (rel {rel})");
}

#[test]
fn uniform_slab_interior_is_flat_at_the_analytic_density() {
    // A uniform lattice has ρ = m/s³; the M7a uniform-lattice gate showed the
    // h = 1.3s kernel lattice-sum recovers it within 2%. Cell centers do not
    // coincide with particles, but the same lattice-quadrature argument bounds
    // interior cells identically — assert every interior cell within 2%.
    let s = 1.0;
    let n_side = 12usize;
    let mut pos = Vec::new();
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                pos.push(DVec3::new(ix as f64, iy as f64, iz as f64) * s);
            }
        }
    }
    let n = pos.len();
    let mass = vec![1.0; n];
    let h = vec![1.3 * s; n];
    let extent = (n_side - 1) as f64 * s;
    let dims = [24, 24, 24];
    let grid = deposit_fixed(
        &pos,
        &mass,
        &h,
        dims,
        Vec3::splat(0.0),
        Vec3::splat(extent as f32),
    );

    // Interior = cell centers ≥ 2h from the lattice boundary faces (outside
    // that ring the kernel support sticks out of the particle slab and the
    // density correctly rolls off).
    let margin = SUPPORT * 1.3 * s;
    let expect = 1.0 / (s * s * s);
    let mut checked = 0;
    for iz in 0..dims[2] {
        for iy in 0..dims[1] {
            for ix in 0..dims[0] {
                let c = grid.cell_center(ix, iy, iz);
                let interior = [c.x, c.y, c.z]
                    .iter()
                    .all(|&x| x >= margin && x <= extent - margin);
                if !interior {
                    continue;
                }
                checked += 1;
                let got = grid.data[grid.index(ix, iy, iz)] as f64;
                let rel = (got - expect).abs() / expect;
                assert!(rel < 0.02, "interior cell ({ix},{iy},{iz}): ρ {got} vs {expect}");
            }
        }
    }
    assert!(checked > 100, "slab too small: only {checked} interior cells checked");
}

// ---------- determinism ----------

#[test]
fn deposit_parallel_matches_serial_bit_exact() {
    // Varied smoothing lengths + a clustered cloud: the parallel gather must be
    // bit-identical to the serial one (fixed per-cell gather order).
    let mut pos = random_cloud(400, 2.0, 42);
    pos.extend(random_cloud(400, 0.3, 43)); // dense knot
    let n = pos.len();
    let mut seed = 7u64;
    let h: Vec<f64> = (0..n).map(|_| 0.2 + 0.6 * splitmix(&mut seed)).collect();
    let mass: Vec<f64> = (0..n).map(|_| 0.5 + splitmix(&mut seed)).collect();
    let dims = [20, 17, 23]; // deliberately unequal
    let (bmin, bmax) = (Vec3::splat(-3.0), Vec3::splat(3.0));

    let par = deposit_fixed(&pos, &mass, &h, dims, bmin, bmax);
    let ser = deposit_fixed_serial(&pos, &mass, &h, dims, bmin, bmax);
    assert_eq!(par, ser, "parallel and serial deposition disagree");
}

#[test]
fn deposit_gas_is_deterministic() {
    let state = gas_state(random_cloud(600, 4.0, 5));
    let cfg = GasGridConfig {
        dims: [32; 3],
        ..Default::default()
    };
    let a = deposit_gas(&state, &cfg).expect("gas state must produce a grid");
    let b = deposit_gas(&state, &cfg).expect("gas state must produce a grid");
    assert_eq!(a, b, "same state, same config, different grids");
}

// ---------- state-level wrapper: selection, bounds, edges ----------

#[test]
fn deposit_gas_returns_none_without_gas() {
    let mut state = gas_state(random_cloud(50, 1.0, 9));
    state.kind = vec![Species::Collisionless; state.len()];
    assert!(deposit_gas(&state, &GasGridConfig::default()).is_none());
}

#[test]
fn deposit_gas_ignores_collisionless_rows() {
    // A gas cloud near the origin plus a massive far-away star: the star must
    // affect neither the bounds (which would dilute the grid to nothing) nor
    // the deposited mass.
    let mut state = gas_state(random_cloud(500, 2.0, 21));
    state.pos.push(DVec3::new(1000.0, 0.0, 0.0));
    state.vel.push(DVec3::ZERO);
    state.mass.push(1e6);
    state.id.push(ParticleId(9999));
    state.progenitor.push(Progenitor(0));
    state.kind.push(Species::Collisionless);

    let cfg = GasGridConfig {
        dims: [32; 3],
        ..Default::default()
    };
    let grid = deposit_gas(&state, &cfg).expect("gas rows present");
    assert!(
        grid.bounds_max.x < 500.0,
        "bounds reached toward the collisionless star: {:?}",
        grid.bounds_max
    );
    let cell = grid.cell_size();
    let vol = cell.x * cell.y * cell.z;
    let total: f64 = grid.data.iter().map(|&d| d as f64 * vol).sum();
    // 500 unit masses of gas; the far star's 1e6 must not appear. The pad
    // covers the kernels of all particles inside the percentile radius; the
    // few outside it lose at most their own kernel mass.
    assert!(
        (total - 500.0).abs() / 500.0 < 0.05,
        "grid mass {total} vs gas mass 500"
    );
}

#[test]
fn deposit_gas_bounds_contain_the_percentile_radius() {
    // Documented convention: cube centered on the gas centroid, half-edge =
    // percentile radius + 2·h_max. Every particle within the percentile
    // distance of the centroid must therefore sit strictly inside the bounds.
    let state = gas_state(random_cloud(300, 5.0, 77));
    let cfg = GasGridConfig {
        dims: [16; 3],
        percentile: 0.9,
        ..Default::default()
    };
    let grid = deposit_gas(&state, &cfg).expect("gas rows present");

    let n = state.len() as f64;
    let centroid = state.pos.iter().fold(DVec3::ZERO, |a, &p| a + p) / n;
    let mut d: Vec<f64> = state.pos.iter().map(|p| (*p - centroid).length()).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    let idx = ((d.len() - 1) as f64 * 0.9).round() as usize;
    let r_p = d[idx];

    let inside = state
        .pos
        .iter()
        .filter(|p| (**p - centroid).length() <= r_p)
        .all(|p| {
            let q = p.as_vec3();
            q.cmpgt(grid.bounds_min).all() && q.cmplt(grid.bounds_max).all()
        });
    assert!(inside, "a percentile-radius particle fell outside the bounds");
}

#[test]
fn deposit_gas_handles_single_and_coincident_particles() {
    // Single particle: rootless adaptive-h clamps deterministically (M7a);
    // bounds degenerate to the 2·h_max pad cube. Everything stays finite and
    // the kernel mass is captured (support exactly touches the box faces, so
    // only quadrature error is lost).
    let single = gas_state(vec![DVec3::new(1.0, -2.0, 3.0)]);
    let cfg = GasGridConfig {
        dims: [32; 3],
        ..Default::default()
    };
    let grid = deposit_gas(&single, &cfg).expect("single gas particle");
    assert!(grid.data.iter().all(|d| d.is_finite()));
    let cell = grid.cell_size();
    let total: f64 = grid
        .data
        .iter()
        .map(|&d| d as f64 * cell.x * cell.y * cell.z)
        .sum();
    assert!(
        (total - 1.0).abs() < 0.05,
        "single-particle grid mass {total} vs 1"
    );

    // A coincident knot must not panic or produce non-finite cells.
    let knot = gas_state(vec![DVec3::splat(0.5); 4]);
    let grid = deposit_gas(&knot, &cfg).expect("coincident gas particles");
    assert!(grid.data.iter().all(|d| d.is_finite()));
}

// ---------- sampling: the CPU reference for the M7e shader mix ----------

#[test]
fn sample_returns_cell_values_at_centers_exactly() {
    let pos = random_cloud(100, 1.5, 3);
    let h = vec![0.5; 100];
    let mass = vec![1.0; 100];
    let dims = [9, 7, 5];
    let grid = deposit_fixed(&pos, &mass, &h, dims, Vec3::splat(-2.0), Vec3::splat(2.0));
    for &(ix, iy, iz) in &[(0u32, 0u32, 0u32), (4, 3, 2), (8, 6, 4), (1, 5, 3)] {
        let c = grid.cell_center(ix, iy, iz).as_vec3();
        assert_eq!(
            grid.sample(c),
            grid.data[grid.index(ix, iy, iz)],
            "sample at cell center ({ix},{iy},{iz}) is not the cell value"
        );
    }
}

#[test]
fn sample_is_zero_outside_bounds_and_lerps_between_centers() {
    let g = GasGrid {
        dims: [2, 1, 1],
        bounds_min: Vec3::ZERO,
        bounds_max: Vec3::new(2.0, 1.0, 1.0),
        data: vec![1.0, 3.0],
    };
    // Outside the box: exactly zero.
    assert_eq!(g.sample(Vec3::new(-0.1, 0.5, 0.5)), 0.0);
    assert_eq!(g.sample(Vec3::new(2.1, 0.5, 0.5)), 0.0);
    assert_eq!(g.sample(Vec3::new(1.0, 1.5, 0.5)), 0.0);
    // Midpoint between the two cell centers (0.5,·,·) and (1.5,·,·): the mean.
    let mid = g.sample(Vec3::new(1.0, 0.5, 0.5));
    assert!((mid - 2.0).abs() < 1e-6, "midpoint sample {mid} vs 2.0");
    // Within the outer half-cell ring: edge-clamped (GPU convention).
    assert_eq!(g.sample(Vec3::new(0.1, 0.5, 0.5)), 1.0);
    assert_eq!(g.sample(Vec3::new(1.9, 0.5, 0.5)), 3.0);
}

#[test]
fn sample_mix_endpoints_reproduce_the_grids_bit_exact() {
    // The two-product lerp degenerates to 1·a + 0·b at the endpoints — the CPU
    // reference the M7e shader-mix gate will compare against.
    let pos0 = random_cloud(150, 2.0, 100);
    let pos1 = random_cloud(150, 2.0, 200);
    let h = vec![0.6; 150];
    let mass = vec![1.0; 150];
    let g0 = deposit_fixed(&pos0, &mass, &h, [12; 3], Vec3::splat(-3.0), Vec3::splat(3.0));
    let g1 = deposit_fixed(&pos1, &mass, &h, [12; 3], Vec3::splat(-2.5), Vec3::splat(3.5));

    let mut seed = 55u64;
    for _ in 0..50 {
        let p = Vec3::new(
            (splitmix(&mut seed) * 8.0 - 4.0) as f32,
            (splitmix(&mut seed) * 8.0 - 4.0) as f32,
            (splitmix(&mut seed) * 8.0 - 4.0) as f32,
        );
        assert_eq!(sample_mix(&g0, &g1, 0.0, p), g0.sample(p));
        assert_eq!(sample_mix(&g0, &g1, 1.0, p), g1.sample(p));
    }
}
