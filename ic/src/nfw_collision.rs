//! Two-galaxy **NFW–NFW collision** initial conditions — the demoable payoff of
//! the M5 cuspy-halo ladder.

use galaxy_core::{DVec3, State};

use crate::TruncatedNfw;

/// A two-body Kepler encounter between two exponentially-truncated NFW halos.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NfwCollision {
    /// The first (primary) halo — tagged `Progenitor(0)`.
    pub galaxy1: TruncatedNfw,
    /// The second halo — tagged `Progenitor(1)`.
    pub galaxy2: TruncatedNfw,
    /// Orbital eccentricity e (e=1 parabolic — the classic Toomre encounter).
    pub eccentricity: f64,
    /// Pericenter separation of the relative orbit.
    pub pericenter: f64,
    /// Initial COM–COM separation (≥ pericenter; ≤ apocenter if bound).
    pub separation: f64,
}

impl NfwCollision {
    /// Construct an encounter. Both halos must share the same `G`; the eccentricity
    /// and pericenter must be strictly positive and the initial separation at least
    /// the pericenter (and at most the apocenter for a bound orbit).
    pub fn new(
        _galaxy1: TruncatedNfw,
        _galaxy2: TruncatedNfw,
        _eccentricity: f64,
        _pericenter: f64,
        _separation: f64,
    ) -> Self {
        todo!()
    }

    /// The shared gravitational constant `G`.
    pub fn g(&self) -> f64 {
        todo!()
    }

    /// Relative position/velocity `(r_rel, v_rel)` of the two COMs on the incoming
    /// branch of the Kepler orbit.
    pub fn relative_state(&self) -> (DVec3, DVec3) {
        todo!()
    }

    /// Per-halo COM `(position, velocity)` in the global zero-COM / zero-momentum
    /// frame: `((r1, v1), (r2, v2))`.
    pub fn com_states(&self) -> ((DVec3, DVec3), (DVec3, DVec3)) {
        todo!()
    }

    /// Sample the full collision: `n1` particles for halo 1 (`Progenitor(0)`) and
    /// `n2` for halo 2 (`Progenitor(1)`).
    pub fn sample(&self, _n1: usize, _n2: usize, _seed: u64) -> State {
        todo!()
    }
}
