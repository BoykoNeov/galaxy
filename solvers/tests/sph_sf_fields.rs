//! `ForceSolver::sf_fields` gates (plan `natal-ember-forge.md`, S3): the SPH
//! fields the star-formation recipe consumes. A pure-gravity solver returns
//! all-zeros (no gas ⇒ no SF); `GravitySph` returns gas density bit-exact
//! against the O(N²) `reference_density` oracle and a velocity divergence whose
//! SIGN is correct on hand-built converging / diverging clouds, exactly zero on
//! a uniform-velocity flow, and `(0, 0)` on collisionless rows.

use galaxy_core::{DVec3, ForceSolver, ParticleId, Progenitor, Species, State};
use galaxy_solvers::sph::{
    density_adaptive, reference_density, DensityConfig, GravitySph, HydroParams,
};
use galaxy_solvers::DirectSum;

/// A cubic lattice of `nx³` gas particles at spacing `s`, unit mass, zero
/// velocity. Enough particles that the default `n_ngb = 48` solve has a root.
fn gas_lattice(nx: usize, s: f64) -> State {
    let mut pos = Vec::new();
    for x in 0..nx {
        for y in 0..nx {
            for z in 0..nx {
                pos.push(DVec3::new(x as f64, y as f64, z as f64) * s);
            }
        }
    }
    let n = pos.len();
    let vel = vec![DVec3::ZERO; n];
    let mass = vec![1.0; n];
    let mut state = State::from_phase_space(pos, vel, mass);
    state.kind = vec![Species::Gas; n];
    state
}

/// Centroid of the lattice, used to build radial (converging / diverging) fields.
fn centroid(state: &State) -> DVec3 {
    let s: DVec3 = state.pos.iter().copied().fold(DVec3::ZERO, |a, b| a + b);
    s / state.len() as f64
}

fn sph_solver() -> GravitySph<DirectSum> {
    GravitySph::hydro_only(HydroParams::default(), DensityConfig::default())
}

#[test]
fn default_solver_returns_zero_fields() {
    // A pure-gravity solver has no gas and no override ⇒ the trait default fires:
    // ρ = 0, ∇·v = 0 for every particle (inert for SF — a 0 row can't clear the
    // threshold). Mixed kinds present to prove the default ignores `kind`.
    let mut state = gas_lattice(4, 1.0);
    state.kind[0] = Species::Collisionless;
    state.kind[5] = Species::Collisionless;
    let solver = DirectSum::new(1.0, 0.05);
    let f = solver.sf_fields(&state);
    assert_eq!(f.rho, vec![0.0; state.len()], "pure gravity ⇒ ρ all zero");
    assert_eq!(
        f.div_v,
        vec![0.0; state.len()],
        "pure gravity ⇒ ∇·v all zero"
    );
}

#[test]
fn gravitysph_rho_matches_reference_density() {
    // ρ from sf_fields must be bit-exact to the brute-force oracle evaluated at
    // the SAME adaptive h (cold-start, h_init = None) — the load-bearing property
    // that lets sf_fields' ρ be reproduced by a fresh density solve.
    let state = gas_lattice(6, 1.0);
    let n = state.len();
    let f = sph_solver().sf_fields(&state);

    let cfg = DensityConfig::default();
    let dens = density_adaptive(&state.pos, &state.mass, &cfg, None);
    let oracle = reference_density(&state.pos, &state.mass, &dens.h);
    assert_eq!(
        f.rho, oracle,
        "sf_fields ρ must be bit-exact vs reference_density"
    );
    // All gas ⇒ every ρ strictly positive (self-term floor), no zeros leaked.
    assert!(f.rho.iter().all(|&r| r > 0.0), "gas ρ must be positive");
    assert_eq!(f.rho.len(), n);
    assert_eq!(f.div_v.len(), n);
}

#[test]
fn converging_cloud_has_negative_divergence() {
    // Uniformly-converging linear field v = −k·(x − x_c): the SPH gather makes
    // EVERY particle's ∇·v strictly negative (the sign is per-pair uniform for a
    // linear field, so edge particles are clean too).
    let mut state = gas_lattice(6, 1.0);
    let c = centroid(&state);
    let k = 0.3;
    for i in 0..state.len() {
        state.vel[i] = (state.pos[i] - c) * (-k);
    }
    let f = sph_solver().sf_fields(&state);
    assert!(
        f.div_v.iter().all(|&d| d < 0.0),
        "converging flow ⇒ ∇·v < 0 for every gas particle; got {:?}",
        f.div_v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
}

#[test]
fn diverging_cloud_has_positive_divergence() {
    // The mirror of the converging case: v = +k·(x − x_c) ⇒ every ∇·v > 0.
    let mut state = gas_lattice(6, 1.0);
    let c = centroid(&state);
    let k = 0.3;
    for i in 0..state.len() {
        state.vel[i] = (state.pos[i] - c) * k;
    }
    let f = sph_solver().sf_fields(&state);
    assert!(
        f.div_v.iter().all(|&d| d > 0.0),
        "diverging flow ⇒ ∇·v > 0 for every gas particle; got {:?}",
        f.div_v.iter().cloned().fold(f64::INFINITY, f64::min)
    );
}

#[test]
fn uniform_velocity_has_exactly_zero_divergence() {
    // Constant velocity field: v_j − v_i = 0 for every pair, so each term is a
    // hard 0.0 ⇒ ∇·v is EXACTLY 0.0 (bit-exact, not a tolerance). Catches formula
    // bugs that happen to preserve the sign on the radial tests.
    let mut state = gas_lattice(6, 1.0);
    for i in 0..state.len() {
        state.vel[i] = DVec3::new(1.5, -0.7, 0.3);
    }
    let f = sph_solver().sf_fields(&state);
    assert_eq!(
        f.div_v,
        vec![0.0; state.len()],
        "uniform velocity ⇒ ∇·v exactly zero"
    );
}

#[test]
fn collisionless_rows_carry_zero_fields() {
    // A mixed state: gas lattice + collisionless particles interleaved among the
    // gas. Collisionless rows carry (0, 0) EXACTLY; gas rows carry positive ρ.
    let mut state = gas_lattice(6, 1.0);
    let c = centroid(&state);
    for i in 0..state.len() {
        state.vel[i] = (state.pos[i] - c) * (-0.2);
    }
    // Append a handful of collisionless "stars" (position/velocity irrelevant —
    // they must not contribute to gas ρ and must read back (0, 0)).
    let n0 = state.len();
    for j in 0..5 {
        state.pos.push(c + DVec3::new(0.1 * j as f64, 0.0, 0.0));
        state.vel.push(DVec3::new(9.0, 9.0, 9.0));
        state.mass.push(1.0);
        state.id.push(ParticleId((n0 + j) as u64));
        state.progenitor.push(Progenitor(0));
        state.kind.push(Species::Collisionless);
        state.u.push(0.0);
        state.formation_time.push(State::PRIMORDIAL);
    }
    let f = sph_solver().sf_fields(&state);
    for j in 0..5 {
        let i = n0 + j;
        assert_eq!(f.rho[i], 0.0, "collisionless ρ must be 0 at row {i}");
        assert_eq!(f.div_v[i], 0.0, "collisionless ∇·v must be 0 at row {i}");
    }
    // Gas rows still populated (positive ρ, negative ∇·v from the converging field).
    assert!(f.rho[..n0].iter().all(|&r| r > 0.0), "gas ρ positive");
    assert!(f.div_v[..n0].iter().all(|&d| d < 0.0), "gas ∇·v negative");
}
