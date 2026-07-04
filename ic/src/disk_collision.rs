//! Two-galaxy collision of *rotating disk* galaxies — the IC that turns the
//! disk-tail physics (`ExponentialDisk`) into an actual encounter, and the last
//! wiring step before the two-disk tidal-tail movie.
//!
//! It is the disk analogue of [`crate::Collision`] (two Plummer spheres) and
//! reuses the same two-body Kepler placement (`encounter`), so the orbital setup
//! is guarded by the very same osculating-elements tests. The two additions over
//! the Plummer case are:
//!
//!  1. **Orientation.** Each disk carries an [`Orientation`] that sets its spin
//!     axis relative to the orbital plane (prograde / retrograde / inclined) — the
//!     Toomre spin-orbit geometry. The default (both `prograde`, spin +Z, orbit in
//!     x–y) is the coplanar prograde passage that makes the cleanest tails.
//!  2. **Four species.** A disk galaxy has two populations (halo, disk), so a
//!     disk-disk encounter has four. They are tagged so the renderer can color the
//!     two *disks* (the tails) apart from the two halos:
//!       - galaxy 1: halo `Progenitor(0)`, disk `Progenitor(1)`
//!       - galaxy 2: halo `Progenitor(2)`, disk `Progenitor(3)`
//!
//! The combined realization is delivered in the global zero-COM / zero-momentum
//! frame with contiguous unique ids, galaxy 1's particles first.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};

use crate::encounter;
use crate::{ExponentialDisk, Orientation, Plummer, SphericalHalo};

/// A two-body Kepler encounter between two rotating disk galaxies, each with its
/// own spin-orbit [`Orientation`].
///
/// Generic over the [`SphericalHalo`] `H` both disks are embedded in — a cored
/// [`Plummer`] (the default, so every existing `DiskCollision` mention still means
/// `DiskCollision<Plummer>` and compiles unchanged) or a cuspy
/// [`Nfw`](crate::Nfw)/[`Hernquist`](crate::Hernquist)/[`TruncatedNfw`](crate::TruncatedNfw).
/// Swapping `H` turns the coplanar tidal encounter into a cuspy-halo collision with a
/// realistic rising-to-flat rotation curve, with no change to the placement code — it
/// reads only `ExponentialDisk`'s generic surface (`g`, `total_mass`, `sample`), and
/// [`place_galaxy`] operates on the sampled `State`. Mirrors the swappable-`H` design
/// of [`ExponentialDisk`] itself (see `DESIGN.md`, M5f). Both galaxies share the same
/// `H` (an NFW–NFW or Plummer–Plummer pairing), exactly as [`crate::NfwCollision`]
/// pairs two `TruncatedNfw` halos.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiskCollision<H = Plummer> {
    /// The first galaxy — halo `Progenitor(0)`, disk `Progenitor(1)`.
    pub galaxy1: ExponentialDisk<H>,
    /// The second galaxy — halo `Progenitor(2)`, disk `Progenitor(3)`.
    pub galaxy2: ExponentialDisk<H>,
    /// Spin-orbit orientation of galaxy 1 (default `prograde`).
    pub orient1: Orientation,
    /// Spin-orbit orientation of galaxy 2 (default `prograde`).
    pub orient2: Orientation,
    /// Orbital eccentricity e (e=1 parabolic — the classic tidal case).
    pub eccentricity: f64,
    /// Pericenter separation of the relative orbit.
    pub pericenter: f64,
    /// Initial COM–COM separation (≥ pericenter; ≤ apocenter if bound).
    pub separation: f64,
}

impl<H: SphericalHalo> DiskCollision<H> {
    /// Construct a disk-disk encounter. Both galaxies must share the same `G`; the
    /// eccentricity and pericenter must be strictly positive and the initial
    /// separation must be at least the pericenter (and at most the apocenter for a
    /// bound orbit). Both disks start `prograde` (coplanar, spin +Z); set
    /// [`orient1`](Self::orient1) / [`orient2`](Self::orient2) for other geometries.
    pub fn new(
        galaxy1: ExponentialDisk<H>,
        galaxy2: ExponentialDisk<H>,
        eccentricity: f64,
        pericenter: f64,
        separation: f64,
    ) -> Self {
        assert_eq!(
            galaxy1.g, galaxy2.g,
            "both galaxies must share the same gravitational constant G"
        );
        encounter::validate_orbit(eccentricity, pericenter, separation);
        Self {
            galaxy1,
            galaxy2,
            orient1: Orientation::prograde(),
            orient2: Orientation::prograde(),
            eccentricity,
            pericenter,
            separation,
        }
    }

    /// The shared gravitational constant `G`.
    pub fn g(&self) -> f64 {
        self.galaxy1.g
    }

    /// Relative position/velocity `(r_rel, v_rel)` of the two COMs on the incoming
    /// branch of the Kepler orbit (see [`crate::Collision::relative_state`]).
    pub fn relative_state(&self) -> (DVec3, DVec3) {
        let mu = self.g() * (self.galaxy1.total_mass() + self.galaxy2.total_mass());
        encounter::relative_state(mu, self.eccentricity, self.pericenter, self.separation)
    }

    /// Per-galaxy COM `(position, velocity)` in the global zero-COM / zero-momentum
    /// frame: `((r1, v1), (r2, v2))`.
    pub fn com_states(&self) -> ((DVec3, DVec3), (DVec3, DVec3)) {
        let (r_rel, v_rel) = self.relative_state();
        encounter::com_states(
            self.galaxy1.total_mass(),
            self.galaxy2.total_mass(),
            r_rel,
            v_rel,
        )
    }

    /// Sample the full encounter: galaxy 1 gets `n_halo1` halo + `n_disk1` disk
    /// particles, galaxy 2 gets `n_halo2` + `n_disk2`. Each galaxy is sampled in
    /// its body frame, rotated by its [`Orientation`], then rigidly placed at its
    /// COM orbital state; the two are concatenated (galaxy 1 first), tagged into the
    /// four progenitors, given contiguous unique ids, and recentered to the global
    /// zero-COM / zero-momentum frame. Deterministic in `seed`.
    pub fn sample(
        &self,
        n_halo1: usize,
        n_disk1: usize,
        n_halo2: usize,
        n_disk2: usize,
        seed: u64,
    ) -> State {
        let ((r1, v1), (r2, v2)) = self.com_states();

        // Each galaxy is sampled in its own zero-COM/zero-momentum body frame from
        // well-separated PRNG streams. `ExponentialDisk::sample` internally consumes
        // THREE streams — `seed` (halo), `mix(seed)` (disk positions), and `mix²(seed)`
        // (disk velocity dispersion) — so galaxy 1 owns {seed, mix, mix²}. Galaxy 2
        // must start clear of all three, hence three mix steps: it owns
        // {mix³, mix⁴, mix⁵}, disjoint from galaxy 1 (SplitMix64's finalizer is a
        // bijection, so the six seeds are distinct). All three streams are reserved
        // whether or not the disks are warm, so warmth never shifts the seeding.
        let s1 = self.galaxy1.sample(n_halo1, n_disk1, seed);
        let s2 = self
            .galaxy2
            .sample(n_halo2, n_disk2, mix_seed(mix_seed(mix_seed(seed))));

        let n = n_halo1 + n_disk1 + n_halo2 + n_disk2;
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut progenitor = Vec::with_capacity(n);

        // Galaxy 1: rotate the body frame by its orientation, then rigidly place at
        // its COM orbital state. `ExponentialDisk::sample` already tags halo=0,
        // disk=1, so galaxy 1 needs no remap.
        place_galaxy(
            &s1,
            self.orient1,
            r1,
            v1,
            0,
            &mut pos,
            &mut vel,
            &mut mass,
            &mut progenitor,
        );
        // Galaxy 2: same, but shift its progenitor tags by +2 so its halo becomes
        // Progenitor(2) and its disk Progenitor(3) — four distinct species overall.
        place_galaxy(
            &s2,
            self.orient2,
            r2,
            v2,
            2,
            &mut pos,
            &mut vel,
            &mut mass,
            &mut progenitor,
        );

        // The per-galaxy COM split zeros the barycenter analytically, but rotation
        // roundoff and each galaxy's finite-N residual leave an O(machine-ε) offset;
        // a final mass-weighted recenter delivers the global zero-COM/zero-momentum
        // frame to roundoff.
        let mtot: f64 = mass.iter().sum();
        let mean_pos = pos
            .iter()
            .zip(&mass)
            .fold(DVec3::ZERO, |acc, (p, m)| acc + *p * *m)
            / mtot;
        let mean_vel = vel
            .iter()
            .zip(&mass)
            .fold(DVec3::ZERO, |acc, (v, m)| acc + *v * *m)
            / mtot;
        for p in &mut pos {
            *p -= mean_pos;
        }
        for v in &mut vel {
            *v -= mean_vel;
        }

        let id = (0..n as u64).map(ParticleId).collect();
        State {
            pos,
            vel,
            mass,
            id,
            progenitor,
            kind: vec![Species::Collisionless; n],
            time: 0.0,
            a: 1.0,
        }
    }

    /// Sample the encounter with an isothermal SPH gas layer in each galaxy that
    /// carries one (M7c): galaxy 1 gets `n_halo1` halo + `n_disk1` stellar + `n_gas1`
    /// gas particles, galaxy 2 gets `n_halo2` + `n_disk2` + `n_gas2`. A galaxy built
    /// *without* [`ExponentialDisk::with_gas`] ignores its gas count and contributes
    /// only halo + disk, so a mixed gas-rich / gas-free pairing is legal.
    ///
    /// Up to **six** populations, tagged so the palette colors them apart and the
    /// volumetric route keys on `kind`:
    ///   - galaxy 1: halo `Progenitor(0)`, disk `Progenitor(1)`, gas `Progenitor(4)`
    ///   - galaxy 2: halo `Progenitor(2)`, disk `Progenitor(3)`, gas `Progenitor(5)`
    ///
    /// The two galaxies draw from disjoint PRNG stream sets — galaxy 1 off `seed`,
    /// galaxy 2 off `mix³(seed)` — with each galaxy's gas stream sitting in a salted
    /// domain orthogonal to the stellar mix-chain (see [`crate::disk::gas_stream_seed`]),
    /// so galaxy 1's gas never collides with galaxy 2's halo seed. The stellar and halo
    /// particles stay bit-identical to the gas-free encounter at the same seed; the
    /// whole system is delivered in the global zero-COM / zero-momentum frame with
    /// contiguous unique ids. Deterministic in `seed`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample_gas(
        &self,
        n_halo1: usize,
        n_disk1: usize,
        n_gas1: usize,
        n_halo2: usize,
        n_disk2: usize,
        n_gas2: usize,
        seed: u64,
    ) -> State {
        let _ = (n_halo1, n_disk1, n_gas1, n_halo2, n_disk2, n_gas2, seed);
        todo!("DiskCollision::sample_gas — M7c gas encounter")
    }
}

/// Rotate a body-frame galaxy by `orient`, boost/translate it to its COM orbital
/// state `(r_com, v_com)`, remap its progenitor tags by `prog_shift`, and append
/// the result to the output buffers. A rotation is rigid, so it preserves the
/// galaxy's internal structure and its body-frame zero-COM/zero-momentum framing;
/// the placement is then a pure rigid-body move.
#[allow(clippy::too_many_arguments)]
fn place_galaxy(
    s: &State,
    orient: Orientation,
    r_com: DVec3,
    v_com: DVec3,
    prog_shift: u16,
    pos: &mut Vec<DVec3>,
    vel: &mut Vec<DVec3>,
    mass: &mut Vec<f64>,
    progenitor: &mut Vec<Progenitor>,
) {
    for i in 0..s.len() {
        pos.push(orient.apply(s.pos[i]) + r_com);
        vel.push(orient.apply(s.vel[i]) + v_com);
        mass.push(s.mass[i]);
        progenitor.push(Progenitor(s.progenitor[i].0 + prog_shift));
    }
}

/// One SplitMix64 step, deriving galaxy 2's seed from galaxy 1's so the two draw
/// from well-separated, independent PRNG streams. Mirrors `collision.rs`.
fn mix_seed(seed: u64) -> u64 {
    let z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use crate::disk::{gas_stream_seed, mix_seed};
    use std::collections::HashSet;

    /// A gas-rich `DiskCollision` spends EIGHT independent PRNG stream seeds — four
    /// per galaxy (halo, disk positions, disk velocities, gas) — and they must all be
    /// distinct, or two populations draw from the same underlying sequence.
    ///
    /// This is the white-box gate the black-box gas tests cannot see: galaxy 2's halo
    /// is seeded `mix³(seed)` whether or not the gas stream is salted, so "galaxy 2
    /// stellar ≡ gas-free, bit-exact" passes even with a colliding gas stream. The
    /// naive `gas_stream_seed = mix³` derivation makes galaxy 1's gas seed *equal*
    /// galaxy 2's halo seed (both `mix³(seed)`) — this assertion is what forbids it and
    /// forces the salted domain (D7).
    #[test]
    fn eight_stream_seeds_are_all_distinct() {
        const SEED: u64 = 0x0D15_C011;
        // Galaxy 1 owns {seed, mix, mix², gas(seed)}; galaxy 2 is spaced past all of
        // galaxy 1's stellar streams at mix³(seed) and owns {mix³, mix⁴, mix⁵, gas(mix³)}.
        let g2_base = mix_seed(mix_seed(mix_seed(SEED)));
        let seeds = [
            SEED,                                   // g1 halo
            mix_seed(SEED),                         // g1 disk positions
            mix_seed(mix_seed(SEED)),               // g1 disk velocities
            gas_stream_seed(SEED),                  // g1 gas
            g2_base,                                // g2 halo
            mix_seed(g2_base),                      // g2 disk positions
            mix_seed(mix_seed(g2_base)),            // g2 disk velocities
            gas_stream_seed(g2_base),               // g2 gas
        ];
        let distinct: HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            seeds.len(),
            "PRNG stream seeds collide (gas stream not orthogonal to the mix-chain): {seeds:0x?}"
        );
    }
}
