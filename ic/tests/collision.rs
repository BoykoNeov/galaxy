//! Validation of the two-galaxy collision IC (DESIGN.md M2).
//!
//! The deliverable has two independently-checkable halves, kept deliberately
//! separate (a Plummer sphere is not a point mass, so conflating them would
//! confound an exact orbital-setup check with the O((a/d)²) extended-mass
//! correction):
//!
//!  1. **Orbital setup** — the pure `relative_state` / `com_states` computation.
//!     Treated as a two-body Kepler problem between the two COMs, the relative
//!     orbit must reproduce the *requested* conic (eccentricity, semi-major axis,
//!     pericenter) when fed through an independent osculating-elements formula.
//!     This is exact, sampling-free, and is the analytic stand-in for the heavier
//!     REBOUND cross-check on the setup.
//!
//!  2. **Assembly** — the sampled `State`. The combined realization must be in the
//!     zero-COM/zero-momentum frame, carry the right masses, partition cleanly
//!     into two progenitors with contiguous unique ids, and — crucially — each
//!     galaxy must still be a valid Plummer sphere about its *own* COM (placement
//!     translates and boosts rigidly; it must not corrupt internal structure or
//!     leak the bulk velocity into the internal dispersion).

use std::collections::HashSet;

use galaxy_core::{diagnostics, DVec3, Progenitor, State};
use galaxy_ic::{Collision, Plummer};

/// Osculating (semi-major axis, eccentricity, eccentricity-vector) of a relative
/// two-body orbit with gravitational parameter `mu = G·Mtot`. Independent of the
/// code under test — this is the textbook closed form (same one `kepler.rs` uses).
fn elements(r_rel: DVec3, v_rel: DVec3, mu: f64) -> (f64, f64, DVec3) {
    let r = r_rel.length();
    let energy = 0.5 * v_rel.length_squared() - mu / r;
    let a = -mu / (2.0 * energy);
    let h = r_rel.cross(v_rel);
    let e_vec = v_rel.cross(h) / mu - r_rel / r;
    (a, e_vec.length(), e_vec)
}

/// Particles belonging to one progenitor, returned as a standalone single-galaxy
/// `State` (ids/time/a are irrelevant for the diagnostics that consume it).
fn extract_galaxy(s: &State, prog: Progenitor) -> State {
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    let mut mass = Vec::new();
    for i in 0..s.len() {
        if s.progenitor[i] == prog {
            pos.push(s.pos[i]);
            vel.push(s.vel[i]);
            mass.push(s.mass[i]);
        }
    }
    State::from_phase_space(pos, vel, mass)
}

fn median_radius_about_com(s: &State) -> f64 {
    let com = diagnostics::center_of_mass(s);
    let mut r: Vec<f64> = s.pos.iter().map(|p| (*p - com).length()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    r[r.len() / 2]
}

// ---------- 1. orbital setup (exact, sampling-free) ----------

#[test]
fn relative_orbit_recovers_requested_conic() {
    let g = 1.0;
    let galaxy1 = Plummer::new(g, 1.0, 1.0);
    let galaxy2 = Plummer::new(g, 0.5, 0.8);
    let mu = g * (galaxy1.total_mass + galaxy2.total_mass);

    // Bound (e<1), parabolic (e=1), and hyperbolic (e>1), each started well
    // outside pericenter on the incoming branch.
    for &(e, rp, r0) in &[
        (0.5_f64, 1.0_f64, 4.0_f64),
        (1.0, 1.0, 6.0),
        (1.5, 0.7, 5.0),
    ] {
        let c = Collision::new(galaxy1, galaxy2, e, rp, r0);
        let (r_rel, v_rel) = c.relative_state();

        // Starts at the requested separation, on the incoming branch.
        assert!(
            (r_rel.length() - r0).abs() < 1e-9 * r0,
            "e={e}: |r_rel|={} expected {r0}",
            r_rel.length()
        );
        assert!(
            r_rel.dot(v_rel) < 0.0,
            "e={e}: should be approaching (dr/dt<0), got r·v={}",
            r_rel.dot(v_rel)
        );

        let (a, ecc, _) = elements(r_rel, v_rel, mu);
        assert!(
            (ecc - e).abs() < 1e-9,
            "e={e}: recovered eccentricity {ecc}"
        );

        // Pericenter q = a(1−e) for a conic; recovered q must match the request.
        // (For the parabola a→∞ so test the orbital energy instead.)
        if (e - 1.0).abs() > 1e-9 {
            let q = a * (1.0 - e);
            assert!(
                (q - rp).abs() < 1e-7 * rp,
                "e={e}: pericenter {q} expected {rp}"
            );
        } else {
            let energy = 0.5 * v_rel.length_squared() - mu / r_rel.length();
            assert!(
                energy.abs() < 1e-9,
                "parabola should have ~zero energy, got {energy}"
            );
        }
    }
}

#[test]
fn eccentricity_vector_points_to_pericenter() {
    // Pericenter is placed along +x, so the Laplace–Runge–Lenz / eccentricity
    // vector (which points from focus to pericenter) must be ≈ +x.
    let g = 1.0;
    let g1 = Plummer::new(g, 1.3, 1.0);
    let g2 = Plummer::new(g, 0.7, 1.0);
    let mu = g * (g1.total_mass + g2.total_mass);
    let c = Collision::new(g1, g2, 0.6, 1.2, 5.0);
    let (r_rel, v_rel) = c.relative_state();
    let (_, _, e_vec) = elements(r_rel, v_rel, mu);
    let dir = e_vec.normalize();
    assert!(
        (dir - DVec3::X).length() < 1e-9,
        "eccentricity vector should point to +x pericenter, got {dir:?}"
    );
}

#[test]
fn com_states_split_into_zero_momentum_frame() {
    let g = 1.0;
    let (m1, m2) = (2.0, 0.5);
    let g1 = Plummer::new(g, m1, 1.0);
    let g2 = Plummer::new(g, m2, 1.0);
    let c = Collision::new(g1, g2, 0.8, 1.0, 5.0);

    let (r_rel, v_rel) = c.relative_state();
    let ((r1, v1), (r2, v2)) = c.com_states();

    // The COM split must reproduce the relative coordinates exactly...
    assert!(((r2 - r1) - r_rel).length() < 1e-12, "r2−r1 ≠ r_rel");
    assert!(((v2 - v1) - v_rel).length() < 1e-12, "v2−v1 ≠ v_rel");
    // ...and put the barycenter (and net momentum) at the origin.
    assert!((r1 * m1 + r2 * m2).length() < 1e-12, "COM not at origin");
    assert!(
        (v1 * m1 + v2 * m2).length() < 1e-12,
        "net momentum not zero"
    );
}

// ---------- 2. assembly of the sampled realization ----------

const N1: usize = 4000;
const N2: usize = 2000;
const SEED: u64 = 0xC011_DE;

fn sample_default() -> (Collision, State) {
    let g = 1.0;
    let g1 = Plummer::new(g, 1.0, 1.0);
    let g2 = Plummer::new(g, 0.5, 0.7);
    let c = Collision::new(g1, g2, 1.0, 1.5, 8.0);
    let s = c.sample(N1, N2, SEED);
    (c, s)
}

#[test]
fn combined_state_is_recentered_with_correct_mass() {
    let (c, s) = sample_default();
    assert_eq!(s.len(), N1 + N2, "particle count");
    s.assert_consistent();

    let mtot = c.galaxy1.total_mass + c.galaxy2.total_mass;
    let summed: f64 = s.mass.iter().sum();
    assert!(
        (summed - mtot).abs() < 1e-12 * mtot,
        "total mass {summed} ≠ {mtot}"
    );

    // Global zero-COM / zero-momentum frame.
    assert!(
        diagnostics::center_of_mass(&s).length() < 1e-9,
        "COM not zeroed"
    );
    assert!(
        diagnostics::total_momentum(&s).length() < 1e-9,
        "net momentum not zeroed"
    );
    assert_eq!(s.time, 0.0);
    assert_eq!(s.a, 1.0);
}

#[test]
fn progenitors_and_ids_partition_cleanly() {
    let (_, s) = sample_default();

    // First N1 particles are galaxy 1, the rest galaxy 2.
    assert!(
        (0..N1).all(|i| s.progenitor[i] == Progenitor(0)),
        "first block must be progenitor 0"
    );
    assert!(
        (N1..N1 + N2).all(|i| s.progenitor[i] == Progenitor(1)),
        "second block must be progenitor 1"
    );

    // Ids are contiguous 0..N and unique.
    let ids: HashSet<u64> = s.id.iter().map(|p| p.0).collect();
    assert_eq!(ids.len(), N1 + N2, "ids are not unique");
    assert_eq!(
        ids,
        (0..(N1 + N2) as u64).collect::<HashSet<_>>(),
        "ids are not the contiguous block 0..N"
    );
}

#[test]
fn each_galaxy_keeps_its_plummer_profile_and_internal_dispersion() {
    let (c, s) = sample_default();

    for (prog, model) in [(Progenitor(0), c.galaxy1), (Progenitor(1), c.galaxy2)] {
        let gal = extract_galaxy(&s, prog);

        // Internal structure survives the rigid placement: median radius about
        // the galaxy's own (displaced) COM matches the analytic half-mass radius.
        let median = median_radius_about_com(&gal);
        let rh = model.half_mass_radius();
        assert!(
            (median - rh).abs() < 0.05 * rh,
            "progenitor {prog:?}: median radius {median} vs r_h {rh}",
        );

        // The bulk boost must not leak into the internal velocity dispersion:
        // measured about the galaxy's mean velocity, the virial ratio is still ≈1.
        let vbar = gal.vel.iter().fold(DVec3::ZERO, |acc, v| acc + *v) / gal.len() as f64;
        let t_internal: f64 = gal
            .vel
            .iter()
            .zip(&gal.mass)
            .map(|(v, m)| 0.5 * *m * (*v - vbar).length_squared())
            .sum();
        let virial = 2.0 * t_internal / model.potential_energy().abs();
        assert!(
            (virial - 1.0).abs() < 0.06,
            "progenitor {prog:?}: internal 2T/|W| = {virial} (bulk velocity leaked in?)",
        );
    }
}

#[test]
fn galaxies_are_placed_at_their_com_orbital_states() {
    let (c, s) = sample_default();
    let ((r1, v1), (r2, v2)) = c.com_states();

    let gal1 = extract_galaxy(&s, Progenitor(0));
    let gal2 = extract_galaxy(&s, Progenitor(1));

    // Each galaxy's realized COM and bulk velocity match the requested orbital
    // placement (finite-N sampling noise is well under 1% of the scale radius).
    let tol1 = 0.02 * c.galaxy1.scale_radius;
    let tol2 = 0.02 * c.galaxy2.scale_radius;
    assert!(
        (diagnostics::center_of_mass(&gal1) - r1).length() < tol1,
        "galaxy 1 not centered at r1"
    );
    assert!(
        (diagnostics::center_of_mass(&gal2) - r2).length() < tol2,
        "galaxy 2 not centered at r2"
    );

    let vbar1 = gal1.vel.iter().fold(DVec3::ZERO, |a, v| a + *v) / gal1.len() as f64;
    let vbar2 = gal2.vel.iter().fold(DVec3::ZERO, |a, v| a + *v) / gal2.len() as f64;
    assert!((vbar1 - v1).length() < 0.02, "galaxy 1 bulk velocity ≠ v1");
    assert!((vbar2 - v2).length() < 0.02, "galaxy 2 bulk velocity ≠ v2");
}

#[test]
fn sample_is_deterministic_in_seed() {
    let g1 = Plummer::new(1.0, 1.0, 1.0);
    let g2 = Plummer::new(1.0, 0.5, 0.7);
    let c = Collision::new(g1, g2, 1.0, 1.5, 8.0);

    let a = c.sample(1000, 500, 7);
    let b = c.sample(1000, 500, 7);
    assert_eq!(a.pos, b.pos, "not deterministic (pos)");
    assert_eq!(a.vel, b.vel, "not deterministic (vel)");

    let d = c.sample(1000, 500, 8);
    assert!(d.pos != a.pos, "different seed gave identical draw");
}

#[test]
fn two_galaxies_are_drawn_independently() {
    // Same model and particle count for both galaxies, but the realizations must
    // differ — otherwise both progenitors share one draw (a seeding bug).
    let g1 = Plummer::new(1.0, 1.0, 1.0);
    let c = Collision::new(g1, g1, 1.0, 1.5, 8.0);
    let s = c.sample(1500, 1500, 99);
    let gal1 = extract_galaxy(&s, Progenitor(0));
    let gal2 = extract_galaxy(&s, Progenitor(1));
    // Compare internal (COM-subtracted) positions so the orbital offset doesn't
    // trivially "distinguish" two otherwise-identical draws.
    let com1 = diagnostics::center_of_mass(&gal1);
    let com2 = diagnostics::center_of_mass(&gal2);
    let internal1: Vec<DVec3> = gal1.pos.iter().map(|p| *p - com1).collect();
    let internal2: Vec<DVec3> = gal2.pos.iter().map(|p| *p - com2).collect();
    assert!(
        internal1 != internal2,
        "both galaxies share one realization"
    );
}
