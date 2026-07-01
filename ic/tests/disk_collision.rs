//! Validation of the two-disk collision IC (`DiskCollision`) — the wiring that
//! puts two rotating disk galaxies on a Kepler encounter with per-disk spin-orbit
//! orientation. Three independently-checkable layers:
//!
//!  1. **Orbital setup** — shared with `Collision` via the `encounter` module, so
//!     the relative orbit must recover the requested conic when the *combined*
//!     galaxy masses (disk + halo) are used as the two-body masses.
//!  2. **Assembly** — count, total mass, the four-progenitor partition (two halos,
//!     two disks) with contiguous unique ids, and the global zero-COM/zero-momentum
//!     frame. Each galaxy sits at its requested COM orbital state.
//!  3. **Orientation** — the invariant that distinguishes prograde from retrograde
//!     from inclined is the DISK population's angular momentum: prograde disks spin
//!     +Z, retrograde flips a disk's L_z, and `inclined(i)` tilts a disk's L by `i`
//!     off +Z. Measured over the disk particles (the halo is a non-rotating sphere,
//!     whose finite-N shot noise is not the disk's spin).

use std::collections::HashSet;

use galaxy_core::{DVec3, Progenitor, State};
use galaxy_ic::{DiskCollision, ExponentialDisk, Orientation, Plummer};

// ---------- fiducial encounter ----------

/// Two submaximal cold disks with *different* total masses (so the COM split and
/// the orbital placement are asymmetric and therefore actually tested).
fn fiducial() -> DiskCollision {
    let g1 = ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, Plummer::new(1.0, 1.0, 1.0));
    let g2 = ExponentialDisk::new(0.06, 0.4, 0.04, 1.6, Plummer::new(1.0, 0.6, 0.8));
    // Parabolic Toomre encounter, started well outside pericenter.
    DiskCollision::new(g1, g2, 1.0, 1.5, 8.0)
}

const NH1: usize = 3000;
const ND1: usize = 3000;
const NH2: usize = 2000;
const ND2: usize = 2000;
const SEED: u64 = 0x0D15C_0111;

/// Osculating (semi-major axis, eccentricity) of a relative two-body orbit with
/// gravitational parameter `mu`. Independent textbook closed form.
fn elements(r_rel: DVec3, v_rel: DVec3, mu: f64) -> (f64, f64) {
    let r = r_rel.length();
    let energy = 0.5 * v_rel.length_squared() - mu / r;
    let a = -mu / (2.0 * energy);
    let h = r_rel.cross(v_rel);
    let e_vec = v_rel.cross(h) / mu - r_rel / r;
    (a, e_vec.length())
}

/// Indices of the particles carrying a given progenitor tag.
fn indices(s: &State, prog: Progenitor) -> Vec<usize> {
    (0..s.len()).filter(|&i| s.progenitor[i] == prog).collect()
}

/// Mass-weighted mean position and velocity over a set of particle indices.
fn mean_state(s: &State, idx: &[usize]) -> (DVec3, DVec3) {
    let mut mp = DVec3::ZERO;
    let mut mv = DVec3::ZERO;
    let mut m = 0.0;
    for &i in idx {
        mp += s.pos[i] * s.mass[i];
        mv += s.vel[i] * s.mass[i];
        m += s.mass[i];
    }
    (mp / m, mv / m)
}

/// Spin angular momentum of a sub-population about its OWN mean position/velocity,
/// so the galaxy's bulk orbital motion is removed and only the internal spin is
/// left. Returns the L vector.
fn spin_angular_momentum(s: &State, idx: &[usize]) -> DVec3 {
    let (rc, vc) = mean_state(s, idx);
    let mut l = DVec3::ZERO;
    for &i in idx {
        l += (s.pos[i] - rc).cross(s.vel[i] - vc) * s.mass[i];
    }
    l
}

// ---------- 1. orbital setup (shared conic) ----------

#[test]
fn relative_orbit_recovers_requested_conic_from_combined_masses() {
    let g = 1.0;
    let g1 = ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, Plummer::new(g, 1.0, 1.0));
    let g2 = ExponentialDisk::new(0.06, 0.4, 0.04, 1.6, Plummer::new(g, 0.6, 0.8));
    // The two-body mass is the FULL galaxy mass (disk + halo), not just the halo.
    let mu = g * (g1.total_mass() + g2.total_mass());

    for &(e, rp, r0) in &[(0.5_f64, 1.0_f64, 2.5_f64), (1.0, 1.0, 6.0), (1.5, 0.7, 5.0)] {
        let c = DiskCollision::new(g1, g2, e, rp, r0);
        let (r_rel, v_rel) = c.relative_state();
        assert!(
            (r_rel.length() - r0).abs() < 1e-9 * r0,
            "e={e}: |r_rel|={} expected {r0}",
            r_rel.length()
        );
        assert!(r_rel.dot(v_rel) < 0.0, "e={e}: should be approaching");
        let (a, ecc) = elements(r_rel, v_rel, mu);
        assert!((ecc - e).abs() < 1e-9, "e={e}: recovered eccentricity {ecc}");
        if (e - 1.0).abs() > 1e-9 {
            let q = a * (1.0 - e);
            assert!((q - rp).abs() < 1e-7 * rp, "e={e}: pericenter {q} expected {rp}");
        }
    }
}

#[test]
fn com_states_split_into_zero_momentum_frame() {
    let c = fiducial();
    let (r_rel, v_rel) = c.relative_state();
    let ((r1, v1), (r2, v2)) = c.com_states();
    let m1 = c.galaxy1.total_mass();
    let m2 = c.galaxy2.total_mass();
    assert!(((r2 - r1) - r_rel).length() < 1e-12, "r2−r1 ≠ r_rel");
    assert!(((v2 - v1) - v_rel).length() < 1e-12, "v2−v1 ≠ v_rel");
    assert!((r1 * m1 + r2 * m2).length() < 1e-12, "COM not at origin");
    assert!((v1 * m1 + v2 * m2).length() < 1e-12, "net momentum not zero");
}

// ---------- 2. assembly ----------

#[test]
fn combined_state_count_mass_and_frame() {
    let c = fiducial();
    let s = c.sample(NH1, ND1, NH2, ND2, SEED);
    s.assert_consistent();
    assert_eq!(s.len(), NH1 + ND1 + NH2 + ND2, "particle count");

    let mtot = c.galaxy1.total_mass() + c.galaxy2.total_mass();
    let summed: f64 = s.mass.iter().sum();
    assert!((summed - mtot).abs() < 1e-12 * mtot, "total mass {summed} ≠ {mtot}");

    // Global zero-COM / zero-momentum frame.
    let mut com = DVec3::ZERO;
    let mut mom = DVec3::ZERO;
    for i in 0..s.len() {
        com += s.pos[i] * s.mass[i];
        mom += s.vel[i] * s.mass[i];
    }
    assert!((com / mtot).length() < 1e-9, "COM not zeroed: {:?}", com / mtot);
    assert!(mom.length() < 1e-9, "net momentum not zeroed: {mom:?}");
    assert_eq!(s.time, 0.0);
    assert_eq!(s.a, 1.0);
}

#[test]
fn four_progenitors_partition_with_contiguous_ids() {
    let c = fiducial();
    let s = c.sample(NH1, ND1, NH2, ND2, SEED);

    // Galaxy 1 first (halo 0, disk 1), then galaxy 2 (halo 2, disk 3).
    let bounds = [0, NH1, NH1 + ND1, NH1 + ND1 + NH2, NH1 + ND1 + NH2 + ND2];
    for (block, &prog) in [Progenitor(0), Progenitor(1), Progenitor(2), Progenitor(3)]
        .iter()
        .enumerate()
    {
        assert!(
            (bounds[block]..bounds[block + 1]).all(|i| s.progenitor[i] == prog),
            "block {block} must be all {prog:?}"
        );
    }
    assert_eq!(indices(&s, Progenitor(0)).len(), NH1);
    assert_eq!(indices(&s, Progenitor(1)).len(), ND1);
    assert_eq!(indices(&s, Progenitor(2)).len(), NH2);
    assert_eq!(indices(&s, Progenitor(3)).len(), ND2);

    let ids: HashSet<u64> = s.id.iter().map(|p| p.0).collect();
    let n = NH1 + ND1 + NH2 + ND2;
    assert_eq!(ids.len(), n, "ids not unique");
    assert_eq!(ids, (0..n as u64).collect::<HashSet<_>>(), "ids not contiguous 0..N");
}

#[test]
fn galaxies_placed_at_their_com_orbital_states() {
    let c = fiducial();
    let s = c.sample(NH1, ND1, NH2, ND2, SEED);
    let ((r1, v1), (r2, v2)) = c.com_states();

    // Galaxy 1 = progenitors {0,1}; galaxy 2 = {2,3}.
    let g1: Vec<usize> = indices(&s, Progenitor(0))
        .into_iter()
        .chain(indices(&s, Progenitor(1)))
        .collect();
    let g2: Vec<usize> = indices(&s, Progenitor(2))
        .into_iter()
        .chain(indices(&s, Progenitor(3)))
        .collect();
    let (rc1, vc1) = mean_state(&s, &g1);
    let (rc2, vc2) = mean_state(&s, &g2);

    let tol = 0.03 * c.galaxy1.scale_length;
    assert!((rc1 - r1).length() < tol, "galaxy 1 COM {rc1:?} vs r1 {r1:?}");
    assert!((rc2 - r2).length() < tol, "galaxy 2 COM {rc2:?} vs r2 {r2:?}");
    assert!((vc1 - v1).length() < 0.03, "galaxy 1 bulk velocity ≠ v1");
    assert!((vc2 - v2).length() < 0.03, "galaxy 2 bulk velocity ≠ v2");
}

#[test]
fn sample_is_deterministic_in_seed() {
    let c = fiducial();
    let a = c.sample(1000, 1000, 800, 800, 7);
    let b = c.sample(1000, 1000, 800, 800, 7);
    assert_eq!(a.pos, b.pos, "not deterministic (pos)");
    assert_eq!(a.vel, b.vel, "not deterministic (vel)");
    let d = c.sample(1000, 1000, 800, 800, 8);
    assert!(d.pos != a.pos, "different seed gave identical draw");
}

// ---------- 3. orientation (the disk spin-orbit geometry) ----------

#[test]
fn default_encounter_is_coplanar_prograde_both_disks_spin_plus_z() {
    let c = fiducial();
    assert_eq!(c.orient1, Orientation::prograde(), "galaxy 1 defaults to prograde");
    assert_eq!(c.orient2, Orientation::prograde(), "galaxy 2 defaults to prograde");

    let s = c.sample(NH1, ND1, NH2, ND2, SEED);
    let l1 = spin_angular_momentum(&s, &indices(&s, Progenitor(1)));
    let l2 = spin_angular_momentum(&s, &indices(&s, Progenitor(3)));
    // Both disks co-rotate with the orbit (spin +Z), coherently axial.
    for (tag, l) in [("disk1", l1), ("disk2", l2)] {
        assert!(l.z > 0.0, "{tag} prograde spin must be +Z: L_z={}", l.z);
        assert!(
            l.z > 20.0 * l.x.abs().max(l.y.abs()),
            "{tag} spin not coherently axial: L={l:?}"
        );
    }
}

#[test]
fn retrograde_galaxy2_flips_only_its_disk_spin() {
    let mut c = fiducial();
    c.orient2 = Orientation::retrograde();
    let s = c.sample(NH1, ND1, NH2, ND2, SEED);

    let l1 = spin_angular_momentum(&s, &indices(&s, Progenitor(1)));
    let l2 = spin_angular_momentum(&s, &indices(&s, Progenitor(3)));
    assert!(l1.z > 0.0, "galaxy 1 still prograde: L_z={}", l1.z);
    assert!(l2.z < 0.0, "galaxy 2 retrograde must flip L_z: L_z={}", l2.z);
    // Retrograde is coplanar: the disk still lies in the orbital plane (spin axial).
    assert!(
        l2.z.abs() > 20.0 * l2.x.abs().max(l2.y.abs()),
        "retrograde disk should still be axial (in-plane): L={l2:?}"
    );
}

#[test]
fn inclined_galaxy1_tilts_its_disk_angular_momentum() {
    let incl = 0.6_f64; // radians
    let mut c = fiducial();
    c.orient1 = Orientation::inclined(incl);
    let s = c.sample(NH1, ND1, NH2, ND2, SEED);

    let l1 = spin_angular_momentum(&s, &indices(&s, Progenitor(1)));
    let tilt = (l1.normalize().dot(DVec3::Z)).clamp(-1.0, 1.0).acos();
    assert!(
        (tilt - incl).abs() < 0.05,
        "disk 1 L tilted {tilt} off +Z, expected {incl}"
    );
    // Galaxy 2 untouched — still prograde +Z.
    let l2 = spin_angular_momentum(&s, &indices(&s, Progenitor(3)));
    assert!(l2.z > 0.0 && l2.z > 20.0 * l2.x.abs().max(l2.y.abs()), "galaxy 2 must stay prograde");
}
