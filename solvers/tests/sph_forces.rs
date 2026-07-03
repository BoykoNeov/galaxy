//! SPH hydro-force gates (DESIGN.md M7b): a two-particle hand oracle, exact
//! pairwise antisymmetry (Newton's 3rd law), linear+angular momentum to roundoff
//! on random clouds, ~zero net force on a uniform lattice, viscosity active only
//! on approach, and parallel ≡ serial bit-exactness. Expectations are derived
//! from the symmetric `P/ρ²` momentum equation with the kernel-average
//! symmetrization, never read back from the function under test.

use galaxy_core::{DVec3, ForceSolver, Species, State};
use galaxy_solvers::sph::{
    density_adaptive, hydro_accelerations, hydro_accelerations_serial, DensityConfig, GravitySph,
    HydroParams,
};
use galaxy_solvers::DirectSum;
use proptest::prelude::*;

const PI: f64 = std::f64::consts::PI;

fn random_points(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    (0..n)
        .map(|_| DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * scale)
        .collect()
}

/// x-component of `∇_i W(r_ij, h)` for the separation `r_ij = (−d, 0, 0)`,
/// hand-coded from the M4 spline — an oracle independent of `grad_w`.
/// `∇W = (dW/dr)·r̂`; with `r̂ = (−1,0,0)` the x-component is `−dW/dr`.
fn grad_x(d: f64, h: f64) -> f64 {
    let q = d / h;
    let norm = 1.0 / (PI * h * h * h);
    let dp = if q < 1.0 {
        -3.0 * q + 2.25 * q * q
    } else if q < 2.0 {
        let t = 2.0 - q;
        -0.75 * t * t
    } else {
        return 0.0;
    };
    // grad_w((−d,0,0), h).x = (−d)·(norm·dp/(h·d)) = −(norm·dp/h).
    -(norm * dp / h)
}

#[test]
fn two_particle_force_matches_the_hand_oracle() {
    // Two gas particles at rest on the x-axis, UNEQUAL h so the kernel average
    // W̄ = ½(W(h_0)+W(h_1)) is genuinely exercised (equal h would collapse it and
    // hide an averaging bug). At rest the viscosity is off, so
    //   a_0 = −m·(c_s²/ρ_0 + c_s²/ρ_1)·∇_0 W̄_01,   ∇_0 W̄_01.x = ½(g(h_0)+g(h_1)).
    let (h0, h1, d) = (1.0_f64, 1.4_f64, 0.8_f64); // q_0 = 0.8, q_1 = 0.571 (< 1)
    let m = 1.0_f64;
    let (rho0, rho1) = (2.0_f64, 3.0_f64);
    let params = HydroParams {
        sound_speed: 1.3,
        ..HydroParams::default()
    };
    let cs2 = params.sound_speed * params.sound_speed;

    let pos = vec![DVec3::ZERO, DVec3::new(d, 0.0, 0.0)];
    let vel = vec![DVec3::ZERO, DVec3::ZERO];
    let mass = vec![m, m];
    let rho = vec![rho0, rho1];
    let h = vec![h0, h1];
    let acc = hydro_accelerations(&pos, &vel, &mass, &rho, &h, &params);

    let grad_avg_x = 0.5 * (grad_x(d, h0) + grad_x(d, h1));
    let coeff = cs2 / rho0 + cs2 / rho1;
    let expect0_x = -m * coeff * grad_avg_x;

    // Sign sanity: two rest particles must push APART. Particle 1 is at +x, so
    // particle 0 must accelerate in −x.
    assert!(
        expect0_x < 0.0,
        "hand oracle sign wrong: expected repulsion"
    );
    let rel = (acc[0].x - expect0_x).abs() / expect0_x.abs();
    assert!(rel < 1e-12, "a_0.x = {} vs oracle {expect0_x}", acc[0].x);
    assert!(acc[0].y.abs() < 1e-14 && acc[0].z.abs() < 1e-14);
    // Antisymmetry (equal mass): a_1 == −a_0 exactly.
    assert_eq!(
        acc[1], -acc[0],
        "equal-mass pair must be exactly antisymmetric"
    );
}

#[test]
fn pairwise_force_is_exactly_antisymmetric() {
    // Newton's 3rd law: for equal masses the two accelerations are exact
    // negations (bit-exact), even with unequal h/ρ and nonzero velocity — the
    // grad-average is exactly negated and the coefficient is bit-identical.
    let params = HydroParams::default();
    let pos = vec![DVec3::new(0.1, -0.2, 0.3), DVec3::new(0.9, 0.4, -0.1)];
    let vel = vec![DVec3::new(0.2, 0.0, -0.1), DVec3::new(-0.3, 0.1, 0.05)];
    let mass = vec![0.75, 0.75];
    let rho = vec![1.7, 2.6];
    let h = vec![0.9, 1.25];
    let acc = hydro_accelerations(&pos, &vel, &mass, &rho, &h, &params);
    assert_eq!(
        acc[0], -acc[1],
        "equal-mass pair not bit-exactly antisymmetric"
    );
}

#[test]
fn uniform_lattice_interior_has_near_zero_net_force() {
    // Constant ρ and h with v = 0 ⇒ ∇P = 0: interior lattice particles see their
    // neighbors in ± pairs, whose grad-average contributions cancel to roundoff.
    // Edge particles (asymmetric neighborhoods) are excluded.
    let (nx, s) = (9usize, 1.0f64);
    let mut pos = Vec::new();
    for x in 0..nx {
        for y in 0..nx {
            for z in 0..nx {
                pos.push(DVec3::new(x as f64, y as f64, z as f64) * s);
            }
        }
    }
    let n = pos.len();
    let h = 1.3 * s;
    let params = HydroParams::default();
    let mass = vec![1.0; n];
    let rho = vec![1.0 / (s * s * s); n]; // uniform density m/s³
    let vel = vec![DVec3::ZERO; n];
    let acc = hydro_accelerations(&pos, &vel, &mass, &rho, &vec![h; n], &params);

    // Reference single-pair force scale: m·(2c_s²/ρ)·|grad_w| at one spacing.
    let ref_scale =
        mass[0] * (2.0 * params.sound_speed * params.sound_speed / rho[0]) * grad_x(s, h).abs();
    let margin = 2.0 * h; // support radius: interior = farther than 2h from every face
    let hi = (nx - 1) as f64 * s;
    let mut checked = 0;
    for (i, p) in pos.iter().enumerate() {
        let interior = [p.x, p.y, p.z]
            .iter()
            .all(|&c| c > margin && c < hi - margin);
        if interior {
            assert!(
                acc[i].length() < 1e-10 * ref_scale,
                "interior net force {} not ~0 (ref {ref_scale})",
                acc[i].length()
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "lattice too small: no interior particles");
}

#[test]
fn viscosity_activates_only_on_approach() {
    // Monaghan Π ≥ 0 only when v_ij·r_ij < 0. A receding pair must give the exact
    // same acceleration as the pressure-only (v = 0) case; an approaching pair
    // must differ, and in the direction that opposes approach (extra repulsion).
    let params = HydroParams::default();
    let pos = vec![DVec3::ZERO, DVec3::new(0.7, 0.0, 0.0)];
    let mass = vec![1.0, 1.0];
    let rho = vec![1.5, 1.5];
    let h = vec![0.9, 0.9];

    let rest = hydro_accelerations(&pos, &[DVec3::ZERO; 2], &mass, &rho, &h, &params);
    // Approaching: particle 0 moves +x (toward 1), particle 1 moves −x.
    let v_app = vec![DVec3::new(0.5, 0.0, 0.0), DVec3::new(-0.5, 0.0, 0.0)];
    let approach = hydro_accelerations(&pos, &v_app, &mass, &rho, &h, &params);
    // Receding: reverse the velocities.
    let v_rec = vec![DVec3::new(-0.5, 0.0, 0.0), DVec3::new(0.5, 0.0, 0.0)];
    let recede = hydro_accelerations(&pos, &v_rec, &mass, &rho, &h, &params);

    assert_eq!(recede, rest, "receding pair must have Π = 0 (≡ rest)");
    assert!(
        approach[0].x < rest[0].x,
        "approach must add repulsion: a0.x {} should be < rest {}",
        approach[0].x,
        rest[0].x
    );
    assert_eq!(
        approach[1], -approach[0],
        "viscous pair still antisymmetric"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    /// Linear + angular momentum conserved to roundoff on random gas clouds.
    /// A non-antisymmetric force (e.g. a one-sided kernel) would leave a residual
    /// O(1/N) of the total force magnitude — far above the 1e-9 roundoff gate.
    #[test]
    fn momentum_conserved_to_roundoff(seed in 1u64..5000) {
        let n = 60usize;
        let pos = random_points(seed, n, 6.0);
        let vel = random_points(seed ^ 0xABCD, n, 2.0);
        let mut lcg = seed | 1;
        let mut nf = || { lcg = lcg.wrapping_mul(6364136223846793005).wrapping_add(1); ((lcg >> 11) as f64)/((1u64<<53) as f64) };
        let mass: Vec<f64> = (0..n).map(|_| 0.5 + nf()).collect();
        let rho: Vec<f64> = (0..n).map(|_| 0.5 + nf()).collect();
        let h: Vec<f64> = (0..n).map(|_| 0.8 + 0.6 * nf()).collect();
        let params = HydroParams::default();
        let acc = hydro_accelerations(&pos, &vel, &mass, &rho, &h, &params);

        let mut p = DVec3::ZERO;      // Σ m_i a_i
        let mut l = DVec3::ZERO;      // Σ m_i x_i × a_i
        let mut fscale = 0.0;         // Σ |m_i a_i|
        let mut lscale = 0.0;         // Σ |m_i x_i × a_i|
        for i in 0..n {
            let f = acc[i] * mass[i];
            p += f;
            fscale += f.length();
            let tau = pos[i].cross(f);
            l += tau;
            lscale += tau.length();
        }
        prop_assert!(p.length() < 1e-9 * fscale, "linear momentum residual {} vs {fscale}", p.length());
        prop_assert!(l.length() < 1e-9 * lscale, "angular momentum residual {} vs {lscale}", l.length());
    }
}

#[test]
fn parallel_equals_serial_bit_exact() {
    let n = 500usize;
    let pos = random_points(9, n, 5.0);
    let vel = random_points(99, n, 1.5);
    let mass: Vec<f64> = (0..n).map(|i| 1.0 + (i % 4) as f64 * 0.3).collect();
    let rho: Vec<f64> = (0..n).map(|i| 0.7 + (i % 5) as f64 * 0.2).collect();
    let h: Vec<f64> = (0..n).map(|i| 0.9 + (i % 3) as f64 * 0.15).collect();
    let params = HydroParams::default();
    let par = hydro_accelerations(&pos, &vel, &mass, &rho, &h, &params);
    let ser = hydro_accelerations_serial(&pos, &vel, &mass, &rho, &h, &params);
    assert_eq!(par, ser, "rayon and serial hydro must be bit-identical");
}

#[test]
fn empty_input_yields_empty_output() {
    let params = HydroParams::default();
    assert!(hydro_accelerations(&[], &[], &[], &[], &[], &params).is_empty());
}

// --- GravitySph composite routing -----------------------------------------

/// Build a gas-only State from positions/velocities (equal mass).
fn gas_state(pos: Vec<DVec3>, vel: Vec<DVec3>, m: f64) -> State {
    let mut s = State::from_phase_space(pos.clone(), vel, vec![m; pos.len()]);
    for k in s.kind.iter_mut() {
        *k = Species::Gas;
    }
    s
}

#[test]
fn gravity_off_runs_the_identical_hydro_path() {
    // hydro_only ⇒ gas accelerations equal the standalone hydro path (adaptive
    // ρ/h then the pairwise force), bit-exact — the composite adds nothing to
    // the fluid math, it only skips the gravity term.
    let n = 200usize;
    let pos = random_points(7, n, 4.0);
    let vel = random_points(8, n, 1.0);
    let state = gas_state(pos.clone(), vel.clone(), 1.0);
    let params = HydroParams::default();
    let cfg = DensityConfig::default();

    let mut solver = GravitySph::<DirectSum>::hydro_only(params, cfg.clone());
    let mut acc = vec![DVec3::ZERO; n];
    solver.accelerations(&state, &mut acc);

    let dens = density_adaptive(&pos, &state.mass, &cfg, None);
    let expect = hydro_accelerations(&pos, &vel, &state.mass, &dens.rho, &dens.h, &params);
    assert_eq!(
        acc, expect,
        "gravity-off must equal the standalone hydro path"
    );
}

#[test]
fn mixed_species_routing_adds_hydro_to_gas_only() {
    // Collisionless particles feel only gravity; gas feels gravity + hydro.
    // Gravity acts on ALL particles (gas is mass to the gravity solver).
    let g = DirectSum::new(1.0, 0.05);
    let params = HydroParams::default();
    let cfg = DensityConfig::default();

    // 120 gas + 40 collisionless, interleaved species.
    let gas_pos = random_points(3, 120, 4.0);
    let gas_vel = random_points(4, 120, 1.0);
    let star_pos: Vec<DVec3> = random_points(5, 40, 6.0);
    let mut pos = gas_pos.clone();
    pos.extend_from_slice(&star_pos);
    let mut vel = gas_vel.clone();
    vel.extend(std::iter::repeat_n(DVec3::ZERO, 40));
    let mut state = State::from_phase_space(pos.clone(), vel.clone(), vec![1.0; 160]);
    for i in 0..120 {
        state.kind[i] = Species::Gas;
    }

    let mut solver = GravitySph::new(g, params, cfg.clone());
    let mut acc = vec![DVec3::ZERO; 160];
    solver.accelerations(&state, &mut acc);

    // Expected: bare gravity everywhere, hydro added on the gas subset.
    let mut grav = g;
    let mut grav_acc = vec![DVec3::ZERO; 160];
    grav.accelerations(&state, &mut grav_acc);
    let dens = density_adaptive(&gas_pos, &[1.0; 120], &cfg, None);
    let hydro = hydro_accelerations(&gas_pos, &gas_vel, &[1.0; 120], &dens.rho, &dens.h, &params);

    for i in 120..160 {
        assert_eq!(
            acc[i], grav_acc[i],
            "collisionless {i} must be gravity-only"
        );
    }
    for i in 0..120 {
        let expect = grav_acc[i] + hydro[i];
        assert!(
            (acc[i] - expect).length() < 1e-12 * expect.length().max(1e-30),
            "gas {i}: {} vs gravity+hydro {}",
            acc[i],
            expect
        );
    }
}

#[test]
fn potential_energy_delegates_to_gravity() {
    let g = DirectSum::new(1.0, 0.1);
    let pos = random_points(2, 50, 3.0);
    let state = gas_state(pos, vec![DVec3::ZERO; 50], 1.0);
    let params = HydroParams::default();
    let cfg = DensityConfig::default();

    let with_g = GravitySph::new(g, params, cfg.clone());
    assert_eq!(with_g.potential_energy(&state), g.potential_energy(&state));
    let no_g = GravitySph::<DirectSum>::hydro_only(params, cfg);
    assert_eq!(no_g.potential_energy(&state), 0.0);
}
