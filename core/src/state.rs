use crate::DVec3;

/// Stable per-particle identity, preserved across the whole run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParticleId(pub u64);

/// Origin tag: which progenitor galaxy + species a particle belongs to.
/// This is what lets tidal tails be colored by their source galaxy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Progenitor(pub u16);

/// Physical species of a particle. Deliberately distinct from [`Progenitor`]
/// (pure identity: which galaxy + component): physics and render routing key
/// on `Species`, never on progenitor tags.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Species {
    /// Star / dark-matter particle: gravity only.
    Collisionless = 0,
    /// SPH gas particle: gravity + hydrodynamic forces (M7).
    Gas = 1,
}

/// Simulation state in Structure-of-Arrays layout (cache/SIMD friendly).
///
/// Positions and velocities are f64 (dynamic range + energy conservation).
/// `a` is the cosmological scale factor (1.0 for non-cosmological runs).
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub pos: Vec<DVec3>,
    pub vel: Vec<DVec3>,
    pub mass: Vec<f64>,
    pub id: Vec<ParticleId>,
    pub progenitor: Vec<Progenitor>,
    pub kind: Vec<Species>,
    /// Per-particle specific internal energy `u` (energy per unit mass), the
    /// EVOLVED thermodynamic variable of the adiabatic SPH path (energy
    /// equation, Chain A). Distinct in kind from the derived-never-stored h/ρ
    /// (D2): `u` has its own time derivative and is integrated forward, so it is
    /// a genuine state variable and MUST live here. `0.0` on collisionless rows
    /// (gravity-only particles carry no internal energy) and throughout the
    /// isothermal path, where the EOS fixes pressure from ρ alone and `u` is
    /// inert. Length `n`, like every other column.
    pub u: Vec<f64>,
    /// Per-particle time at which this particle FORMED as a star via star
    /// formation (gas → collisionless conversion; plan `natal-ember-forge.md`,
    /// Chain A step 4). The render colors young stars by their age
    /// `now − formation_time`. Every particle that did NOT form via SF in this
    /// run — every primordial star / DM / halo / disk particle AND every
    /// still-gas particle — carries the sentinel [`State::PRIMORDIAL`]
    /// (`−∞`), which means exactly "no SF formation time." `0.0` is a valid
    /// formation time (a star born at `t = 0`) so it cannot be the sentinel;
    /// `−∞` was chosen over `NaN` because `State` derives `PartialEq` and a `NaN`
    /// column would make `State == State` always false, poisoning every
    /// whole-`State` round-trip / byte-identity gate. Age-coloring needs no
    /// special case: `now − (−∞) = +∞`, whose tint ramp `exp(−age/τ)` is `0` (the
    /// base color) — exactly right, primordial stars ARE old. Length `n`.
    pub formation_time: Vec<f64>,
    pub time: f64,
    pub a: f64,
}

impl State {
    /// Sentinel `formation_time` for a particle that did NOT form via star
    /// formation in this run (every primordial star / DM / halo / disk particle,
    /// and every still-gas particle). See [`State::formation_time`] for why `−∞`
    /// rather than `NaN` (the derived `PartialEq` on `State`).
    pub const PRIMORDIAL: f64 = f64::NEG_INFINITY;

    /// Number of particles.
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    /// True if there are no particles.
    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    /// Build a non-cosmological state from phase-space arrays, assigning
    /// sequential ids and a single progenitor tag. Convenience for tests / ICs.
    pub fn from_phase_space(pos: Vec<DVec3>, vel: Vec<DVec3>, mass: Vec<f64>) -> Self {
        let n = pos.len();
        assert_eq!(vel.len(), n, "vel length must match pos");
        assert_eq!(mass.len(), n, "mass length must match pos");
        State {
            pos,
            vel,
            mass,
            id: (0..n as u64).map(ParticleId).collect(),
            progenitor: vec![Progenitor(0); n],
            kind: vec![Species::Collisionless; n],
            u: vec![0.0; n],
            formation_time: vec![State::PRIMORDIAL; n],
            time: 0.0,
            a: 1.0,
        }
    }

    /// Debug check that all SoA arrays agree in length.
    pub fn assert_consistent(&self) {
        let n = self.pos.len();
        assert_eq!(self.vel.len(), n, "vel length mismatch");
        assert_eq!(self.mass.len(), n, "mass length mismatch");
        assert_eq!(self.id.len(), n, "id length mismatch");
        assert_eq!(self.progenitor.len(), n, "progenitor length mismatch");
        assert_eq!(self.kind.len(), n, "kind length mismatch");
        assert_eq!(self.u.len(), n, "u length mismatch");
        assert_eq!(
            self.formation_time.len(),
            n,
            "formation_time length mismatch"
        );
    }
}
