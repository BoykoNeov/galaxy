//! `SphericalHalo`: the abstraction behind [`ExponentialDisk<H>`](crate::ExponentialDisk).
//!
//! A disk galaxy is a cold, low-mass stellar disk sitting *inside* a much more
//! massive spherical halo. Everything the disk sampler needs from that halo is a
//! handful of closed-form structural quantities — its `G`, its total (normalizing)
//! mass, its density ρ(r), its cumulative mass M(<r) — plus the ability to seed the
//! halo's own particles. This trait names exactly that surface, so the halo can be
//! swapped from a cored [`Plummer`] sphere to a cuspy [`Nfw`]/[`Hernquist`]/
//! [`TruncatedNfw`] without touching the disk code. The observable payoff: a cuspy
//! halo's enclosed mass rises steeply from the center, so the disk's rotation curve
//! *rises to a flat plateau* (the realistic CDM-galaxy shape) instead of turning
//! over the way it does in a Plummer halo.
//!
//! This mirrors the swappable-[`ForceSolver`](galaxy_core) pattern (see `DESIGN.md`):
//! a small trait keeps the door open to new models without rewriting the callers.
//!
//! Each implementation forwards to the model's own inherent methods (the source of
//! truth, exercised directly by the collision-IC tests) via a **type-qualified path**
//! (`Nfw::density(self, r)`, never `self.density(r)`) so the trait method can never
//! accidentally recurse into itself if an inherent method is later removed.

use galaxy_core::State;

use crate::{Hernquist, Nfw, Plummer, TruncatedNfw};

/// A spherically-symmetric mass model a disk can be embedded in.
///
/// The four structural accessors are closed forms (or the model's own quadrature);
/// [`sample`](SphericalHalo::sample) seeds the halo's particles in the
/// zero-COM/zero-momentum frame, tagged `Progenitor(0)` (halo species).
pub trait SphericalHalo {
    /// Gravitational constant `G`. A disk's own `g` is taken from this at construction.
    fn g(&self) -> f64;

    /// The total mass used to normalize particle sampling. For finite-mass models
    /// this is the analytic total; for the **untruncated** [`Nfw`] — whose total
    /// mass diverges — it is the virial mass `M_vir`, which is exactly what
    /// [`Nfw::sample`] realizes (particles drawn from the profile truncated at
    /// `r_vir`). So the trait stays self-consistent with the sampler.
    fn total_mass(&self) -> f64;

    /// Mass density ρ(r).
    fn density(&self, r: f64) -> f64;

    /// Cumulative mass M(<r) enclosed within radius `r`.
    fn enclosed_mass(&self, r: f64) -> f64;

    /// Draw `n` equal-mass particles in the zero-COM/zero-momentum frame, tagged
    /// `Progenitor(0)`. Deterministic in `seed`.
    fn sample(&self, n: usize, seed: u64) -> State;
}

impl SphericalHalo for Plummer {
    fn g(&self) -> f64 {
        self.g
    }
    fn total_mass(&self) -> f64 {
        self.total_mass
    }
    fn density(&self, r: f64) -> f64 {
        Plummer::density(self, r)
    }
    fn enclosed_mass(&self, r: f64) -> f64 {
        Plummer::enclosed_mass(self, r)
    }
    fn sample(&self, n: usize, seed: u64) -> State {
        Plummer::sample(self, n, seed)
    }
}

impl SphericalHalo for Hernquist {
    fn g(&self) -> f64 {
        self.g
    }
    fn total_mass(&self) -> f64 {
        self.total_mass
    }
    fn density(&self, r: f64) -> f64 {
        Hernquist::density(self, r)
    }
    fn enclosed_mass(&self, r: f64) -> f64 {
        Hernquist::enclosed_mass(self, r)
    }
    fn sample(&self, n: usize, seed: u64) -> State {
        Hernquist::sample(self, n, seed)
    }
}

impl SphericalHalo for Nfw {
    fn g(&self) -> f64 {
        self.g
    }
    /// The untruncated NFW total mass diverges; return `M_vir`, which is exactly
    /// what [`Nfw::sample`] realizes (profile truncated at `r_vir`).
    fn total_mass(&self) -> f64 {
        self.virial_mass
    }
    fn density(&self, r: f64) -> f64 {
        Nfw::density(self, r)
    }
    fn enclosed_mass(&self, r: f64) -> f64 {
        Nfw::enclosed_mass(self, r)
    }
    fn sample(&self, n: usize, seed: u64) -> State {
        Nfw::sample(self, n, seed)
    }
}

impl SphericalHalo for TruncatedNfw {
    fn g(&self) -> f64 {
        self.base.g
    }
    fn total_mass(&self) -> f64 {
        TruncatedNfw::total_mass(self)
    }
    fn density(&self, r: f64) -> f64 {
        TruncatedNfw::density(self, r)
    }
    fn enclosed_mass(&self, r: f64) -> f64 {
        TruncatedNfw::enclosed_mass(self, r)
    }
    fn sample(&self, n: usize, seed: u64) -> State {
        TruncatedNfw::sample(self, n, seed)
    }
}
