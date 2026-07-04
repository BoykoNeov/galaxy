//! Gas support for the two-disk collision (`DiskCollision::sample_gas`, M7c) — the
//! step that lets a disk-disk encounter carry isothermal SPH gas, the input the
//! gas-dynamical merger demo (and M7d/M7e) render through.
//!
//! A gas-rich encounter has up to SIX populations. They are tagged so the palette
//! colors the two gas disks apart from the four stellar populations, and so the
//! volumetric route keys on `kind` (D1):
//!   - galaxy 1: halo `Progenitor(0)`, disk `Progenitor(1)`, gas `Progenitor(4)`
//!   - galaxy 2: halo `Progenitor(2)`, disk `Progenitor(3)`, gas `Progenitor(5)`
//!
//! The gates below check the assembly (tags, counts, contiguous ids, the six-way
//! partition), the `kind ⟺ gas-progenitor` correspondence, the mixed gas-rich /
//! gas-free pairing, the gas-count-zero edge, the global zero-COM / zero-momentum
//! frame, and determinism. The stream-separation invariant that makes the gas
//! draws independent of the stellar mix-chain is the white-box unit gate in
//! `disk_collision.rs`; the single-galaxy gas physics (sech² layer, pressure-
//! corrected rotation, stellar bit-identity) is gated in `gas_disk_sampling.rs`.

use std::collections::HashSet;

use galaxy_core::{DVec3, Progenitor, Species, State};
use galaxy_ic::{DiskCollision, ExponentialDisk, Plummer};

// ---------- fiducial gas-rich encounter ----------

const NH1: usize = 2000;
const ND1: usize = 1500;
const NG1: usize = 1200;
const NH2: usize = 1500;
const ND2: usize = 1000;
const NG2: usize = 800;
const SEED: u64 = 0x0D15_C011;

/// Galaxy 1 model (mass 0.1) — gas-rich. `with_gas(0.5, 0.1)` reproduces the
/// stability-test disk's parameters (Q_gas ≥ 1, so `with_gas` does not reject it).
fn disk1_gas() -> ExponentialDisk {
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, Plummer::new(1.0, 1.0, 1.0)).with_gas(0.5, 0.1)
}

/// Galaxy 2 model (mass 0.06, different halo) — gas-rich, so the COM split and the
/// six-way partition are asymmetric and therefore actually tested.
fn disk2_gas() -> ExponentialDisk {
    ExponentialDisk::new(0.06, 0.4, 0.04, 1.6, Plummer::new(1.0, 0.6, 0.8)).with_gas(0.4, 0.1)
}

/// The gas-free counterpart of galaxy 2, same geometry (for the mixed pairing).
fn disk2_nogas() -> ExponentialDisk {
    ExponentialDisk::new(0.06, 0.4, 0.04, 1.6, Plummer::new(1.0, 0.6, 0.8))
}

/// A parabolic Toomre encounter of the two gas-rich disks, started outside pericenter.
fn gas_fiducial() -> DiskCollision {
    DiskCollision::new(disk1_gas(), disk2_gas(), 1.0, 1.5, 8.0)
}

/// Indices of the particles carrying a given progenitor tag.
fn indices(s: &State, prog: Progenitor) -> Vec<usize> {
    (0..s.len()).filter(|&i| s.progenitor[i] == prog).collect()
}

// ---------- assembly ----------

#[test]
fn six_populations_tagged_counted_with_contiguous_ids() {
    let s = gas_fiducial().sample_gas(NH1, ND1, NG1, NH2, ND2, NG2, SEED);

    let n = NH1 + ND1 + NG1 + NH2 + ND2 + NG2;
    assert_eq!(s.len(), n, "total particle count");

    // The six populations partition the realization with the documented tags.
    assert_eq!(
        indices(&s, Progenitor(0)).len(),
        NH1,
        "galaxy 1 halo = Progenitor(0)"
    );
    assert_eq!(
        indices(&s, Progenitor(1)).len(),
        ND1,
        "galaxy 1 disk = Progenitor(1)"
    );
    assert_eq!(
        indices(&s, Progenitor(4)).len(),
        NG1,
        "galaxy 1 gas = Progenitor(4)"
    );
    assert_eq!(
        indices(&s, Progenitor(2)).len(),
        NH2,
        "galaxy 2 halo = Progenitor(2)"
    );
    assert_eq!(
        indices(&s, Progenitor(3)).len(),
        ND2,
        "galaxy 2 disk = Progenitor(3)"
    );
    assert_eq!(
        indices(&s, Progenitor(5)).len(),
        NG2,
        "galaxy 2 gas = Progenitor(5)"
    );

    // Contiguous unique ids 0..n.
    let ids: HashSet<u64> = s.id.iter().map(|p| p.0).collect();
    assert_eq!(ids.len(), n, "ids unique");
    assert_eq!(
        ids,
        (0..n as u64).collect::<HashSet<_>>(),
        "ids contiguous 0..n"
    );
}

#[test]
fn gas_species_matches_the_two_gas_progenitors() {
    let s = gas_fiducial().sample_gas(NH1, ND1, NG1, NH2, ND2, NG2, SEED);

    for i in 0..s.len() {
        let is_gas_prog = matches!(s.progenitor[i], Progenitor(4) | Progenitor(5));
        let is_gas_kind = s.kind[i] == Species::Gas;
        assert_eq!(
            is_gas_kind, is_gas_prog,
            "particle {i}: kind={:?} but progenitor={:?} — gas species and gas \
             progenitors (4,5) must coincide",
            s.kind[i], s.progenitor[i]
        );
    }
    let n_gas = (0..s.len()).filter(|&i| s.kind[i] == Species::Gas).count();
    assert_eq!(n_gas, NG1 + NG2, "gas particle count");
}

// ---------- mixed and edge cases ----------

#[test]
fn gas_free_galaxy_contributes_only_halo_and_disk() {
    // Galaxy 1 gas-rich, galaxy 2 gas-free: only galaxy 1's gas exists.
    let c = DiskCollision::new(disk1_gas(), disk2_nogas(), 1.0, 1.5, 8.0);
    let s = c.sample_gas(NH1, ND1, NG1, NH2, ND2, NG2, SEED);

    // Galaxy 2 requested NG2 gas but is gas-free, so it contributes none.
    assert_eq!(
        s.len(),
        NH1 + ND1 + NG1 + NH2 + ND2,
        "gas-free galaxy 2 drops its gas"
    );
    assert_eq!(
        indices(&s, Progenitor(4)).len(),
        NG1,
        "galaxy 1 gas present"
    );
    assert!(indices(&s, Progenitor(5)).is_empty(), "galaxy 2 gas absent");
    let n_gas = (0..s.len()).filter(|&i| s.kind[i] == Species::Gas).count();
    assert_eq!(n_gas, NG1, "only galaxy 1 gas is present");
}

#[test]
fn zero_gas_counts_yield_a_purely_stellar_encounter() {
    let s = gas_fiducial().sample_gas(NH1, ND1, 0, NH2, ND2, 0, SEED);

    assert_eq!(s.len(), NH1 + ND1 + NH2 + ND2, "no gas particles");
    assert!(
        (0..s.len()).all(|i| s.kind[i] == Species::Collisionless),
        "n_gas = 0 ⇒ no Species::Gas particles"
    );
}

// ---------- global frame + determinism ----------

#[test]
fn whole_system_is_zero_com_and_zero_momentum() {
    let s = gas_fiducial().sample_gas(NH1, ND1, NG1, NH2, ND2, NG2, SEED);

    let mtot: f64 = s.mass.iter().sum();
    let com = (0..s.len()).fold(DVec3::ZERO, |a, i| a + s.pos[i] * s.mass[i]) / mtot;
    let mom = (0..s.len()).fold(DVec3::ZERO, |a, i| a + s.vel[i] * s.mass[i]);
    let scale = gas_fiducial().separation;
    assert!(com.length() < 1e-10 * scale, "COM not centered: {com:?}");
    assert!(mom.length() < 1e-10, "net momentum not zero: {mom:?}");
}

#[test]
fn sample_gas_is_deterministic_in_seed() {
    let c = gas_fiducial();
    let a = c.sample_gas(NH1, ND1, NG1, NH2, ND2, NG2, SEED);
    let b = c.sample_gas(NH1, ND1, NG1, NH2, ND2, NG2, SEED);
    assert_eq!(a.pos, b.pos, "positions deterministic");
    assert_eq!(a.vel, b.vel, "velocities deterministic");
    assert_eq!(a.mass, b.mass, "masses deterministic");
    assert_eq!(a.progenitor, b.progenitor, "tags deterministic");
    assert_eq!(a.kind, b.kind, "kinds deterministic");
}
