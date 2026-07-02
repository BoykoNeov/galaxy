//! Validation of the two-galaxy **NFW–NFW collision** IC (`NfwCollision`) — the
//! demoable payoff of the M5 cuspy-halo ladder. It puts two exponentially-truncated
//! NFW halos ([`TruncatedNfw`], M5d, the self-consistent Springel & White model) on
//! a relative Kepler encounter. An NFW halo is spherical, isotropic and
//! non-rotating, so this is the direct analogue of the Plummer [`Collision`] (two
//! spheres, two progenitors, *no* spin-orbit orientation) — not the four-species
//! rotating [`DiskCollision`].
//!
//! Two independently-checkable halves, kept deliberately separate (an NFW halo is
//! not a point mass, so conflating them would confound an exact orbital-setup check
//! with the extended-mass correction that only the many-body evolution resolves):
//!
//!  1. **Orbital setup** — the pure `relative_state` / `com_states` computation,
//!     shared with `Collision`/`DiskCollision` via the `encounter` module. The
//!     relative orbit must recover the requested conic when the **combined** (full,
//!     skirt-inclusive) galaxy masses are used as the two-body masses. This is the
//!     load-bearing subtlety of the truncated model: `sample` places particles
//!     summing to `total_mass()` (virial + exponential skirt), so the orbit must be
//!     set for that mass, not `M_vir`.
//!  2. **Assembly** — the sampled `State`: count, total mass, the two-progenitor
//!     partition with contiguous unique ids, the global zero-COM/zero-momentum
//!     frame, and — crucially — each halo still a valid truncated-NFW realization
//!     about its *own* displaced COM (rigid placement must not corrupt internal
//!     structure or leak the bulk orbital velocity into the internal dispersion).
//!
//! Each halo's *dynamical* validity as an equilibrium is already gated by
//! `nfw_truncated_stability.rs`; a collision is meant to move, so there is no
//! evolve-and-stay-put run here (that would be over-build, not coverage).

use std::collections::HashSet;

use galaxy_core::{diagnostics, DVec3, Progenitor, State};
use galaxy_ic::{Nfw, NfwCollision, TruncatedNfw};

// ---------- helpers ----------

/// Osculating (semi-major axis, eccentricity, eccentricity-vector) of a relative
/// two-body orbit with gravitational parameter `mu = G·Mtot`. Independent of the
/// code under test — the textbook closed form (same one `kepler.rs` uses).
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

/// Half-mass radius of a truncated-NFW halo: the radius where M(<r) = M_tot/2,
/// found by bisecting the (monotone) `enclosed_mass`. Independent oracle — it uses
/// only the analytic/quadrature mass profile, never a realization.
fn half_mass_radius(t: &TruncatedNfw) -> f64 {
    let target = 0.5 * t.total_mass();
    // The skirt is exhausted well before r_vir + 60 r_d (matches the sampling test's
    // `r_far`); the half-mass radius is far inside this, so it brackets the root.
    let mut lo = 0.0;
    let mut hi = t.truncation_radius() + 60.0 * t.decay_length;
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if t.enclosed_mass(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Mean-subtracted internal ⟨|v|²⟩ of a (sub-)state, about its own mass-weighted
/// mean velocity — the internal kinetic temperature with any bulk drift removed.
fn internal_mean_v2(s: &State) -> f64 {
    let vbar = s.vel.iter().fold(DVec3::ZERO, |a, v| a + *v) / s.len() as f64;
    s.vel
        .iter()
        .map(|v| (*v - vbar).length_squared())
        .sum::<f64>()
        / s.len() as f64
}

/// Canonical asymmetric encounter: two *different* truncated-NFW halos (so the COM
/// split and orbital placement are genuinely asymmetric), started well outside both
/// virial radii on a parabolic (Toomre) approach. Both use the positivity-safe
/// r_d = 0.3 r_vir of the M5d sampling test.
fn fiducial() -> NfwCollision {
    let g1 = TruncatedNfw::new(Nfw::new(1.0, 1.0, 1.0, 10.0), 3.0); // r_vir = 10, r_d = 3
    let g2 = TruncatedNfw::new(Nfw::new(1.0, 0.6, 0.8, 10.0), 2.4); // r_vir = 8,  r_d = 2.4
                                                                    // Parabolic, pericenter 3, started at separation 30 (> r_vir1 + r_vir2 = 18, so
                                                                    // the two halos begin cleanly separated — a sensible fly-by demo).
    NfwCollision::new(g1, g2, 1.0, 3.0, 30.0)
}

const N1: usize = 6000;
const N2: usize = 4000;
const SEED: u64 = 0x0FF_5E77;

// ---------- 1. orbital setup (exact, sampling-free) ----------

#[test]
fn relative_orbit_recovers_requested_conic_from_combined_masses() {
    let g = 1.0;
    let g1 = TruncatedNfw::new(Nfw::new(g, 1.0, 1.0, 10.0), 3.0);
    let g2 = TruncatedNfw::new(Nfw::new(g, 0.6, 0.8, 10.0), 2.4);
    // The two-body mass is the FULL galaxy mass (virial + skirt), not just M_vir.
    let mu = g * (g1.total_mass() + g2.total_mass());

    // Bound (e<1), parabolic (e=1), hyperbolic (e>1), each started outside pericenter
    // on the incoming branch. For the bound case r0 must lie inside the apocenter
    // r_peri(1+e)/(1−e) = 3.0, so r0 = 2.5.
    for &(e, rp, r0) in &[
        (0.5_f64, 1.0_f64, 2.5_f64),
        (1.0, 1.0, 6.0),
        (1.5, 0.7, 5.0),
    ] {
        let c = NfwCollision::new(g1, g2, e, rp, r0);
        let (r_rel, v_rel) = c.relative_state();

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
    // Pericenter is placed along +x, so the eccentricity (Laplace–Runge–Lenz) vector,
    // which points from focus to pericenter, must be ≈ +x.
    let g = 1.0;
    let g1 = TruncatedNfw::new(Nfw::new(g, 1.3, 1.0, 10.0), 3.0);
    let g2 = TruncatedNfw::new(Nfw::new(g, 0.7, 1.0, 10.0), 3.0);
    let mu = g * (g1.total_mass() + g2.total_mass());
    // Bound orbit: r0 = 4.0 lies inside the apocenter r_peri(1+e)/(1−e) = 4.8.
    let c = NfwCollision::new(g1, g2, 0.6, 1.2, 4.0);
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
    let c = fiducial();
    let m1 = c.galaxy1.total_mass();
    let m2 = c.galaxy2.total_mass();

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

fn sample_default() -> (NfwCollision, State) {
    let c = fiducial();
    let s = c.sample(N1, N2, SEED);
    (c, s)
}

#[test]
fn combined_state_is_recentered_with_correct_mass() {
    let (c, s) = sample_default();
    assert_eq!(s.len(), N1 + N2, "particle count");
    s.assert_consistent();

    // Total mass is the sum of the FULL truncated masses (virial + skirt).
    let mtot = c.galaxy1.total_mass() + c.galaxy2.total_mass();
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
fn each_galaxy_keeps_its_profile_and_internal_kinematics() {
    let (c, s) = sample_default();

    for (prog, model, ref_seed) in [
        (Progenitor(0), c.galaxy1, 0xA1A1_u64),
        (Progenitor(1), c.galaxy2, 0xB2B2_u64),
    ] {
        let gal = extract_galaxy(&s, prog);

        // (a) Positions: the rigid placement preserves internal structure, so the
        // median radius about the galaxy's own (displaced) COM matches the analytic
        // half-mass radius (an independent oracle from inverting M(<r)).
        let median = median_radius_about_com(&gal);
        let rh = half_mass_radius(&model);
        assert!(
            (median - rh).abs() < 0.05 * rh,
            "progenitor {prog:?}: median radius {median} vs r_h {rh}"
        );

        // (b) Velocities: the bulk orbital boost must not leak into the internal
        // dispersion. The mean-subtracted internal ⟨v²⟩ must match that of an
        // independently-seeded isolated realization of the same model (statistical
        // agreement — a leaked bulk velocity would inflate it by ~|v_bulk|²).
        let internal_v2 = internal_mean_v2(&gal);
        let iso = model.sample(gal.len(), ref_seed);
        let iso_v2 = internal_mean_v2(&iso);
        assert!(
            (internal_v2 - iso_v2).abs() < 0.06 * iso_v2,
            "progenitor {prog:?}: internal ⟨v²⟩ {internal_v2} vs isolated {iso_v2} \
             (bulk velocity leaked in?)"
        );
    }
}

#[test]
fn galaxies_are_placed_at_their_com_orbital_states() {
    let (c, s) = sample_default();
    let ((r1, v1), (r2, v2)) = c.com_states();

    let gal1 = extract_galaxy(&s, Progenitor(0));
    let gal2 = extract_galaxy(&s, Progenitor(1));

    // Each galaxy is internally recentered by `TruncatedNfw::sample` *before*
    // placement, so its realized COM/bulk-velocity track the requested orbital state
    // to roundoff (not merely to finite-N sampling noise); the only residual is the
    // collision's global mass-weighted recenter, itself O(machine-ε).
    assert!(
        (diagnostics::center_of_mass(&gal1) - r1).length() < 1e-9,
        "galaxy 1 not centered at r1"
    );
    assert!(
        (diagnostics::center_of_mass(&gal2) - r2).length() < 1e-9,
        "galaxy 2 not centered at r2"
    );

    let vbar1 = gal1.vel.iter().fold(DVec3::ZERO, |a, v| a + *v) / gal1.len() as f64;
    let vbar2 = gal2.vel.iter().fold(DVec3::ZERO, |a, v| a + *v) / gal2.len() as f64;
    assert!((vbar1 - v1).length() < 1e-9, "galaxy 1 bulk velocity ≠ v1");
    assert!((vbar2 - v2).length() < 1e-9, "galaxy 2 bulk velocity ≠ v2");
}

#[test]
fn sample_is_deterministic_in_seed() {
    let c = fiducial();
    let a = c.sample(2000, 1500, 7);
    let b = c.sample(2000, 1500, 7);
    assert_eq!(a.pos, b.pos, "not deterministic (pos)");
    assert_eq!(a.vel, b.vel, "not deterministic (vel)");

    let d = c.sample(2000, 1500, 8);
    assert!(d.pos != a.pos, "different seed gave identical draw");
}

#[test]
fn two_galaxies_are_drawn_independently() {
    // Same model and particle count for both galaxies, but the realizations must
    // differ — otherwise both progenitors share one draw (a seeding bug, e.g. seeding
    // galaxy 2 with the raw `seed`).
    let g1 = TruncatedNfw::new(Nfw::new(1.0, 1.0, 1.0, 10.0), 3.0);
    let c = NfwCollision::new(g1, g1, 1.0, 3.0, 30.0);
    let s = c.sample(2500, 2500, 99);
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
