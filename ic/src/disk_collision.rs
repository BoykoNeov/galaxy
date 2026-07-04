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
        let mut kind = Vec::with_capacity(n);

        // Galaxy 1: rotate the body frame by its orientation, then rigidly place at
        // its COM orbital state. `ExponentialDisk::sample` already tags halo=0,
        // disk=1, so galaxy 1 (index 0) needs no remap.
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
            &mut kind,
        );
        // Galaxy 2 (index 1): its stellar tags shift by +2 so its halo becomes
        // Progenitor(2) and its disk Progenitor(3) — four distinct species overall.
        place_galaxy(
            &s2,
            self.orient2,
            r2,
            v2,
            1,
            &mut pos,
            &mut vel,
            &mut mass,
            &mut progenitor,
            &mut kind,
        );

        // The per-galaxy COM split zeros the barycenter analytically, but rotation
        // roundoff and each galaxy's finite-N residual leave an O(machine-ε) offset;
        // a final mass-weighted recenter delivers the global zero-COM/zero-momentum
        // frame to roundoff.
        recenter_zero_com(&mut pos, &mut vel, &mass);

        let id = (0..n as u64).map(ParticleId).collect();
        State {
            pos,
            vel,
            mass,
            id,
            progenitor,
            kind,
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
        let ((r1, v1), (r2, v2)) = self.com_states();

        // Each galaxy is sampled in its own body frame from disjoint stream sets.
        // Galaxy 1 owns {seed, mix, mix², gas(seed)}; galaxy 2 starts at mix³(seed),
        // clear of galaxy 1's THREE stellar streams — exactly the gas-free spacing
        // (`sample`). The gas streams sit in a salted domain orthogonal to the mix
        // chain (`disk::gas_stream_seed`), so adding a fourth (gas) stream per galaxy
        // never disturbs that spacing: galaxy 1's gas cannot collide with galaxy 2's
        // halo seed. A galaxy without a gas component ignores its gas count.
        let s1 = sample_one(&self.galaxy1, n_halo1, n_disk1, n_gas1, seed);
        let s2 = sample_one(
            &self.galaxy2,
            n_halo2,
            n_disk2,
            n_gas2,
            mix_seed(mix_seed(mix_seed(seed))),
        );

        let n = s1.len() + s2.len();
        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);
        let mut mass = Vec::with_capacity(n);
        let mut progenitor = Vec::with_capacity(n);
        let mut kind = Vec::with_capacity(n);

        // Galaxy 1 (index 0): halo 0, disk 1, gas 4 — no remap. Galaxy 2 (index 1):
        // halo→2, disk→3, gas→5. The gas remap is keyed on `kind`, not on a fragile
        // arithmetic on the tag value (D1).
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
            &mut kind,
        );
        place_galaxy(
            &s2,
            self.orient2,
            r2,
            v2,
            1,
            &mut pos,
            &mut vel,
            &mut mass,
            &mut progenitor,
            &mut kind,
        );

        // Each galaxy's `sample`/`sample_gas` is already zero-COM in its body frame,
        // and the COM split makes the placed barycenter vanish analytically; a final
        // mass-weighted recenter removes the O(machine-ε) rotation/finite-N residual.
        recenter_zero_com(&mut pos, &mut vel, &mass);

        let id = (0..n as u64).map(ParticleId).collect();
        State {
            pos,
            vel,
            mass,
            id,
            progenitor,
            kind,
            time: 0.0,
            a: 1.0,
        }
    }
}

/// Sample one galaxy for [`DiskCollision::sample_gas`]: with its gas layer if it
/// carries one ([`ExponentialDisk::with_gas`]), else halo + disk only (its gas count
/// is ignored). This is what makes a mixed gas-rich / gas-free pairing legal.
fn sample_one<H: SphericalHalo>(
    galaxy: &ExponentialDisk<H>,
    n_halo: usize,
    n_disk: usize,
    n_gas: usize,
    seed: u64,
) -> State {
    if galaxy.gas_params().is_some() {
        galaxy.sample_gas(n_halo, n_disk, n_gas, seed)
    } else {
        galaxy.sample(n_halo, n_disk, seed)
    }
}

/// Mass-weighted recenter of a placed realization to the global zero-COM /
/// zero-momentum frame, in place. Shared by [`DiskCollision::sample`] and
/// [`DiskCollision::sample_gas`].
fn recenter_zero_com(pos: &mut [DVec3], vel: &mut [DVec3], mass: &[f64]) {
    let mtot: f64 = mass.iter().sum();
    let mean_pos = pos
        .iter()
        .zip(mass)
        .fold(DVec3::ZERO, |acc, (p, m)| acc + *p * *m)
        / mtot;
    let mean_vel = vel
        .iter()
        .zip(mass)
        .fold(DVec3::ZERO, |acc, (v, m)| acc + *v * *m)
        / mtot;
    for p in pos.iter_mut() {
        *p -= mean_pos;
    }
    for v in vel.iter_mut() {
        *v -= mean_vel;
    }
}

/// Rotate a body-frame galaxy by `orient`, boost/translate it to its COM orbital
/// state `(r_com, v_com)`, remap its progenitor tags for galaxy `gal_index` (0 or 1),
/// and append the result — carrying each particle's `kind` — to the output buffers.
/// A rotation is rigid, so it preserves the galaxy's internal structure and its
/// body-frame zero-COM/zero-momentum framing; the placement is then a pure
/// rigid-body move.
///
/// Tag remap, keyed on `kind` so gas routing never depends on a fragile arithmetic
/// on the tag value (D1):
///   - stellar (body-frame halo 0 / disk 1): `tag + 2·gal_index` → galaxy 2 gets 2/3
///   - gas      (body-frame Progenitor(4)):   `4 + gal_index`     → gas1 = 4, gas2 = 5
#[allow(clippy::too_many_arguments)]
fn place_galaxy(
    s: &State,
    orient: Orientation,
    r_com: DVec3,
    v_com: DVec3,
    gal_index: u16,
    pos: &mut Vec<DVec3>,
    vel: &mut Vec<DVec3>,
    mass: &mut Vec<f64>,
    progenitor: &mut Vec<Progenitor>,
    kind: &mut Vec<Species>,
) {
    for i in 0..s.len() {
        pos.push(orient.apply(s.pos[i]) + r_com);
        vel.push(orient.apply(s.vel[i]) + v_com);
        mass.push(s.mass[i]);
        let tag = match s.kind[i] {
            Species::Gas => Progenitor(4 + gal_index),
            Species::Collisionless => Progenitor(s.progenitor[i].0 + 2 * gal_index),
        };
        progenitor.push(tag);
        kind.push(s.kind[i]);
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
            SEED,                        // g1 halo
            mix_seed(SEED),              // g1 disk positions
            mix_seed(mix_seed(SEED)),    // g1 disk velocities
            gas_stream_seed(SEED),       // g1 gas
            g2_base,                     // g2 halo
            mix_seed(g2_base),           // g2 disk positions
            mix_seed(mix_seed(g2_base)), // g2 disk velocities
            gas_stream_seed(g2_base),    // g2 gas
        ];
        let distinct: HashSet<u64> = seeds.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            seeds.len(),
            "PRNG stream seeds collide (gas stream not orthogonal to the mix-chain): {seeds:0x?}"
        );
    }
}
