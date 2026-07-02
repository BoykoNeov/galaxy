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
//! quadrature for the exponential skirt. Crucially, for r ≤ r_vir the potential is
//! *fully closed-form* (NFW mass + closed ∫ρ_NFW s ds + a constant skirt tail), so
//! the deep-energy region that dominates the DF carries no quadrature noise; only
//! the outer skirt is integrated, with a **fixed** panel count so the quadrature
//! error is smooth in r (its derivative — which Eddington takes by finite
//! difference — stays clean, keeping f(ℰ) positive rather than clamped).
//!
//! The reward is a genuinely self-consistent IC — the M5c "outer halo
//! re-virializes because the DF is untruncated" caveat is gone. Method: Springel &
//! White 1999; cf. Kazantzidis et al. 2004.

use galaxy_core::{DVec3, State};

use std::f64::consts::PI;

use crate::eddington::{EddingtonDf, SphericalModel};
use crate::Nfw;

/// Fixed Simpson panel count for the skirt integrals. Fixed (not r-dependent) so
/// the quadrature error is a smooth function of the integration limit — its
/// finite-difference derivative in the Eddington inversion is then noise-free.
const SKIRT_PANELS: usize = 1024;

/// The skirt is integrated/sampled out to r_vir + `SKIRT_CUTOFF`·r_d, beyond which
/// exp(−SKIRT_CUTOFF) makes the density utterly negligible (mass & potential
/// integrals have converged to far better than the test tolerances).
const SKIRT_CUTOFF: f64 = 40.0;

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
        let rs = self.base.scale_radius;
        let rvir = self.truncation_radius();
        rvir / self.decay_length - (rs + 3.0 * rvir) / (rs + rvir)
    }

    /// Radius past which the skirt is treated as empty (integration cutoff).
    fn outer_radius(&self) -> f64 {
        self.truncation_radius() + SKIRT_CUTOFF * self.decay_length
    }

    /// The skirt density ρ_NFW(r_vir)·(r/r_vir)^ε·exp(−(r−r_vir)/r_d) (valid r ≥ r_vir).
    fn skirt_density(&self, r: f64) -> f64 {
        let rvir = self.truncation_radius();
        let amp = self.base.density(rvir);
        amp * (r / rvir).powf(self.epsilon()) * (-(r - rvir) / self.decay_length).exp()
    }

    /// Mass density ρ(r): NFW for r ≤ r_vir, exponential skirt beyond.
    pub fn density(&self, r: f64) -> f64 {
        if r <= self.truncation_radius() {
            self.base.density(r)
        } else {
            self.skirt_density(r)
        }
    }

    /// Skirt shell mass ∫_a^b 4π s² ρ_skirt(s) ds (a, b ≥ r_vir).
    fn skirt_mass(&self, a: f64, b: f64) -> f64 {
        simpson(a, b, SKIRT_PANELS, |s| {
            4.0 * PI * s * s * self.skirt_density(s)
        })
    }

    /// The outer-potential integral ∫_a^b ρ_skirt(s) s ds (a, b ≥ r_vir).
    fn skirt_potential_integral(&self, a: f64, b: f64) -> f64 {
        simpson(a, b, SKIRT_PANELS, |s| s * self.skirt_density(s))
    }

    /// The closed-form NFW potential integral ∫_a^b ρ_NFW(s) s ds
    /// = ρ_s r_s² [1/(1+a/r_s) − 1/(1+b/r_s)].
    fn nfw_potential_integral(&self, a: f64, b: f64) -> f64 {
        let rs = self.base.scale_radius;
        let rho_s = self.base.characteristic_density();
        rho_s * rs * rs * (1.0 / (1.0 + a / rs) - 1.0 / (1.0 + b / rs))
    }

    /// Cumulative mass M(<r): NFW closed form inside r_vir, plus the numerically
    /// integrated skirt beyond. Finite for all r (unlike the untruncated NFW).
    pub fn enclosed_mass(&self, r: f64) -> f64 {
        let rvir = self.truncation_radius();
        if r <= rvir {
            self.base.enclosed_mass(r)
        } else {
            let upper = r.min(self.outer_radius());
            self.base.virial_mass + self.skirt_mass(rvir, upper)
        }
    }

    /// Total mass M(<∞) — finite because of the exponential cutoff.
    pub fn total_mass(&self) -> f64 {
        self.base.virial_mass + self.skirt_mass(self.truncation_radius(), self.outer_radius())
    }

    /// Gravitational potential Φ(r) = −[ G M(<r)/r + 4πG ∫_r^∞ ρ(s) s ds ]. For
    /// r ≤ r_vir the integral is closed-form (NFW piece + constant skirt tail); for
    /// r > r_vir only the skirt remains (fixed-panel quadrature). Ψ(r) = −Φ(r).
    pub fn potential(&self, r: f64) -> f64 {
        let g = self.base.g;
        let rvir = self.truncation_radius();
        let r_out = self.outer_radius();

        let outer = if r <= rvir {
            self.nfw_potential_integral(r, rvir) + self.skirt_potential_integral(rvir, r_out)
        } else if r < r_out {
            self.skirt_potential_integral(r, r_out)
        } else {
            0.0
        };

        let m_term = if r > 0.0 {
            g * self.enclosed_mass(r) / r
        } else {
            0.0 // M(<r)/r → 0 at the cusp; Ψ(0) = 4πG ∫₀^∞ ρ s ds is finite.
        };
        -(m_term + 4.0 * PI * g * outer)
    }

    /// Structural dynamical time, inherited from the NFW base (inner scale).
    pub fn dynamical_time(&self) -> f64 {
        self.base.dynamical_time()
    }

    /// Draw `n` equal-mass particles: positions from the full truncated mass
    /// profile, velocities from the Eddington DF of the truncated (ρ, Ψ).
    /// Recentered to zero COM and zero net momentum.
    pub fn sample(&self, n: usize, seed: u64) -> State {
        let mut rng = SplitMix64::new(seed);
        let rs = self.base.scale_radius;
        let total = self.total_mass();
        let m_each = total / n as f64;
        let r_out = self.outer_radius();

        // Self-consistent DF from the truncated (ρ, Ψ). Deterministic (no RNG),
        // so the whole sampler stays reproducible.
        let psi_max = -self.potential(0.0);
        let df = EddingtonDf::build(self, psi_max, 1e-3 * rs, 1e4 * rs);

        // Precompute cumulative-mass and Ψ tables on a linear radius grid so the
        // per-particle inner loop is table lookups, not fresh profile integrals.
        const NG: usize = 8192;
        let dr = r_out / (NG - 1) as f64;
        let mut m_tab = Vec::with_capacity(NG);
        let mut psi_tab = Vec::with_capacity(NG);
        for i in 0..NG {
            let r = i as f64 * dr;
            m_tab.push(self.enclosed_mass(r));
            psi_tab.push(-self.potential(r));
        }

        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);

        for _ in 0..n {
            // Position: invert the (monotone) mass table for target = u · M_tot.
            let target = rng.next_f64() * total;
            let j = m_tab.partition_point(|&m| m < target).clamp(1, NG - 1);
            let (m0, m1) = (m_tab[j - 1], m_tab[j]);
            let frac = if m1 > m0 {
                (target - m0) / (m1 - m0)
            } else {
                0.0
            };
            let r = (j - 1) as f64 * dr + frac * dr;
            pos.push(random_direction(&mut rng) * r);

            // Speed: Ψ(r) by linear interpolation of the table, then rejection-
            // sample v ∈ [0, v_esc] from p(v) ∝ v² f(Ψ − v²/2).
            let idx = ((r / dr).floor() as usize).min(NG - 2);
            let u = (r - idx as f64 * dr) / dr;
            let psi = psi_tab[idx] + u * (psi_tab[idx + 1] - psi_tab[idx]);
            let v_esc = (2.0 * psi).max(0.0).sqrt();
            let speed_pdf = |v: f64| -> f64 { v * v * df.f(psi - 0.5 * v * v) };
            const GRID: usize = 256;
            let mut ceil = 0.0_f64;
            for k in 1..GRID {
                ceil = ceil.max(speed_pdf(v_esc * k as f64 / GRID as f64));
            }
            ceil *= 1.5;

            let v = loop {
                let v = v_esc * rng.next_f64();
                let height = ceil * rng.next_f64();
                if height < speed_pdf(v) {
                    break v;
                }
            };
            vel.push(random_direction(&mut rng) * v);
        }

        // Recenter to zero COM and zero net momentum.
        let inv_n = 1.0 / n as f64;
        let mean_pos = pos.iter().fold(DVec3::ZERO, |acc, &p| acc + p) * inv_n;
        let mean_vel = vel.iter().fold(DVec3::ZERO, |acc, &v| acc + v) * inv_n;
        for p in &mut pos {
            *p -= mean_pos;
        }
        for v in &mut vel {
            *v -= mean_vel;
        }

        State::from_phase_space(pos, vel, vec![m_each; n])
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

/// Composite Simpson's rule over [a, b] with `n` panels (bumped to even).
fn simpson<F: Fn(f64) -> f64>(a: f64, b: f64, n: usize, f: F) -> f64 {
    if b <= a {
        return 0.0;
    }
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for k in 1..n {
        let w = if k % 2 == 1 { 4.0 } else { 2.0 };
        sum += w * f(a + k as f64 * h);
    }
    sum * h / 3.0
}

/// A unit vector drawn isotropically on the sphere (uniform in cosθ and φ).
fn random_direction(rng: &mut SplitMix64) -> DVec3 {
    use std::f64::consts::TAU;
    let cos_theta = 2.0 * rng.next_f64() - 1.0;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let phi = TAU * rng.next_f64();
    DVec3::new(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta)
}

/// SplitMix64: the project's tiny deterministic PRNG (see `plummer.rs`).
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
