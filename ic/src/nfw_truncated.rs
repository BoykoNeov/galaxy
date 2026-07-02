//! The **exponentially-truncated NFW** halo (Springel & White 1999): the NFW
//! profile ([`crate::Nfw`]) inside the virial radius, smoothly continued beyond
//! it by an exponential cutoff so the total mass is **finite** and the density —
//! and its logarithmic slope — are **continuous** at r_vir. This removes the hard
//! r_vir edge of the plain [`crate::Nfw`] sampler (M5c) in favour of a physical
//! outer skirt, which is the whole point of the smooth truncation: a sharp cut
//! has no well-behaved equilibrium DF, a smooth one does.
//!
//! Profile (x = r/r_s, c = r_vir/r_s):
//! - r ≤ r_vir: ρ(r) = ρ_NFW(r) = ρ_s / (x (1+x)²)
//! - r > r_vir: ρ(r) = ρ_NFW(r_vir) · (r/r_vir)^ε · exp(−(r − r_vir)/r_d)
//!
//! The **decay length r_d is the free knob**; the exponent ε is fixed by
//! continuity of the logarithmic slope at r_vir. NFW's log-slope there is
//! −(1+3c)/(1+c) = −(r_s+3r_vir)/(r_s+r_vir); the truncated form's is ε − r_vir/r_d,
//! so
//!   ε = r_vir/r_d − (r_s + 3 r_vir)/(r_s + r_vir).
//! By construction ρ and dρ/dr are both continuous at r_vir.
//!
//! **Self-consistent equilibrium (Path A).** Unlike M5c — which sampled velocities
//! from the DF of the *untruncated* potential — this model builds its isotropic DF
//! by Eddington-inverting the **truncated** (ρ, Ψ) pair, so positions and
//! velocities share one potential. The truncated NFW potential has no closed form
//! (the outer skirt integral is incomplete-gamma-like), so Φ(r) is computed
//! semi-analytically: closed-form NFW pieces for the cusp plus a converging
//! quadrature for the exponential skirt. The reward is a genuinely self-consistent
//! IC — the M5c "outer halo re-virializes because the DF is untruncated" caveat is
//! gone. Method: Springel & White 1999; cf. Kazantzidis et al. 2004.

use galaxy_core::{DVec3, State};

use crate::eddington::SphericalModel;
use crate::Nfw;

/// An exponentially-truncated NFW halo: an [`Nfw`] base plus a decay length `r_d`
/// setting how fast the density falls beyond the base's virial radius.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TruncatedNfw {
    /// The underlying NFW halo; truncation begins at `base.virial_radius()`.
    pub base: Nfw,
    /// Exponential decay length `r_d` of the outer skirt (must be positive).
    pub decay_length: f64,
}

impl TruncatedNfw {
    /// Construct a truncated halo. `decay_length` must be strictly positive.
    pub fn new(base: Nfw, decay_length: f64) -> Self {
        assert!(decay_length > 0.0, "decay length must be positive");
        Self { base, decay_length }
    }

    /// The truncation radius r_t = r_vir, where the NFW profile joins the skirt.
    pub fn truncation_radius(&self) -> f64 {
        self.base.virial_radius()
    }

    /// The skirt exponent ε = r_vir/r_d − (r_s + 3 r_vir)/(r_s + r_vir), fixed by
    /// continuity of the logarithmic slope at r_vir.
    pub fn epsilon(&self) -> f64 {
        let _ = self;
        todo!("M5d: slope-continuity exponent")
    }

    /// Mass density ρ(r): NFW for r ≤ r_vir, exponential skirt beyond.
    pub fn density(&self, r: f64) -> f64 {
        let _ = r;
        todo!("M5d: piecewise truncated density")
    }

    /// Cumulative mass M(<r): NFW closed form inside r_vir, plus the numerically
    /// integrated skirt beyond. Finite for all r (unlike the untruncated NFW).
    pub fn enclosed_mass(&self, r: f64) -> f64 {
        let _ = r;
        todo!("M5d: truncated enclosed mass")
    }

    /// Total mass M(<∞) — finite because of the exponential cutoff.
    pub fn total_mass(&self) -> f64 {
        todo!("M5d: finite total mass")
    }

    /// Gravitational potential Φ(r) (numerical: closed-form NFW pieces + skirt
    /// quadrature). Ψ(r) = −Φ(r) feeds the Eddington inversion.
    pub fn potential(&self, r: f64) -> f64 {
        let _ = r;
        todo!("M5d: numerical truncated potential")
    }

    /// Structural dynamical time, inherited from the NFW base (inner scale).
    pub fn dynamical_time(&self) -> f64 {
        self.base.dynamical_time()
    }

    /// Draw `n` equal-mass particles: positions from the full truncated mass
    /// profile, velocities from the Eddington DF of the truncated (ρ, Ψ).
    /// Recentered to zero COM and zero net momentum.
    pub fn sample(&self, n: usize, seed: u64) -> State {
        let _ = (n, seed);
        todo!("M5d: sample the self-consistent truncated IC")
    }
}

impl SphericalModel for TruncatedNfw {
    fn density(&self, r: f64) -> f64 {
        TruncatedNfw::density(self, r)
    }
    fn relative_potential(&self, r: f64) -> f64 {
        -TruncatedNfw::potential(self, r)
    }
}
