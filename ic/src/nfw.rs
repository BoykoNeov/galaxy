//! The Navarro–Frenk–White (1996) profile: the near-universal density profile
//! of cold-dark-matter halos in cosmological simulations. Like Hernquist it has
//! a central ρ ∝ r⁻¹ cusp, but its envelope falls only as r⁻³, so the **total
//! mass diverges** (logarithmically) and the model must be **truncated**. There
//! is no closed-form isotropic distribution function, so velocities are sampled
//! from the numerically Eddington-inverted DF ([`crate::eddington`]) — the exact
//! equilibrium, not a local-Maxwellian approximation (Kazantzidis et al. 2004).
//!
//! Parameterized by the virial mass `M_vir`, scale radius `r_s`, and
//! concentration `c` (so the virial radius is r_vir = c·r_s and M(<r_vir)=M_vir).
//! With the characteristic mass M_s = 4π ρ_s r_s³ = M_vir / m(c),
//! m(c) = ln(1+c) − c/(1+c):
//! - density   ρ(r) = ρ_s / (x (1+x)²),                  x = r/r_s
//! - mass      M(<r) = M_s [ln(1+x) − x/(1+x)]           (untruncated)
//! - potential Φ(r)  = −G M_s ln(1+x) / r,  Φ(0) = −G M_s / r_s
//!
//! Positions are drawn by inverting the mass profile truncated at r_vir;
//! velocities use the DF of the *untruncated* potential (standard practice —
//! the truncation perturbs only the outermost shells).

use galaxy_core::{DVec3, State};

use std::f64::consts::PI;

use crate::eddington::EddingtonDf;

/// An NFW halo parameterized by G, virial mass, scale radius, and concentration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Nfw {
    /// Gravitational constant `G`.
    pub g: f64,
    /// Virial mass `M_vir` = M(<r_vir), the mass inside the virial radius.
    pub virial_mass: f64,
    /// Scale radius `r_s` (the cusp/envelope break, where the log-slope is −2).
    pub scale_radius: f64,
    /// Concentration `c` = r_vir / r_s.
    pub concentration: f64,
}

impl Nfw {
    /// Construct a model. All four parameters must be strictly positive.
    pub fn new(g: f64, virial_mass: f64, scale_radius: f64, concentration: f64) -> Self {
        assert!(g > 0.0, "G must be positive");
        assert!(virial_mass > 0.0, "virial mass must be positive");
        assert!(scale_radius > 0.0, "scale radius must be positive");
        assert!(concentration > 0.0, "concentration must be positive");
        Self {
            g,
            virial_mass,
            scale_radius,
            concentration,
        }
    }

    /// The dimensionless mass function m(c) = ln(1+c) − c/(1+c); M(<r) = M_s·m(x).
    pub fn mass_function(&self) -> f64 {
        let c = self.concentration;
        (1.0 + c).ln() - c / (1.0 + c)
    }

    /// Characteristic mass M_s = 4π ρ_s r_s³ = M_vir / m(c).
    pub fn characteristic_mass(&self) -> f64 {
        self.virial_mass / self.mass_function()
    }

    /// Characteristic density ρ_s = M_s / (4π r_s³).
    pub fn characteristic_density(&self) -> f64 {
        self.characteristic_mass() / (4.0 * PI * self.scale_radius.powi(3))
    }

    /// Virial radius r_vir = c · r_s.
    pub fn virial_radius(&self) -> f64 {
        self.concentration * self.scale_radius
    }

    /// Mass density ρ(r) = ρ_s / (x (1+x)²), x = r/r_s.
    pub fn density(&self, r: f64) -> f64 {
        let x = r / self.scale_radius;
        self.characteristic_density() / (x * (1.0 + x) * (1.0 + x))
    }

    /// Cumulative mass M(<r) = M_s [ln(1+x) − x/(1+x)] (untruncated).
    pub fn enclosed_mass(&self, r: f64) -> f64 {
        let x = r / self.scale_radius;
        self.characteristic_mass() * ((1.0 + x).ln() - x / (1.0 + x))
    }

    /// Gravitational potential Φ(r) = −G M_s ln(1+x)/r, with Φ(0) = −G M_s/r_s.
    pub fn potential(&self, r: f64) -> f64 {
        let gms = self.g * self.characteristic_mass();
        if r == 0.0 {
            // ln(1+x)/x → 1 as x → 0, so Φ(0) = −G M_s / r_s.
            return -gms / self.scale_radius;
        }
        let x = r / self.scale_radius;
        -gms * (1.0 + x).ln() / r
    }

    /// Structural dynamical time t_dyn = √(r_s³ / (G M_s)) (the inner scale).
    pub fn dynamical_time(&self) -> f64 {
        let rs = self.scale_radius;
        (rs * rs * rs / (self.g * self.characteristic_mass())).sqrt()
    }

    /// Draw `n` equal-mass particles: positions from the mass profile truncated
    /// at r_vir, velocities from the numerically Eddington-inverted DF of the
    /// untruncated potential. Recentered to zero COM and zero net momentum.
    pub fn sample(&self, n: usize, seed: u64) -> State {
        let mut rng = SplitMix64::new(seed);
        let rs = self.scale_radius;
        let r_vir = self.virial_radius();
        let m_each = self.virial_mass / n as f64;

        // Build the isotropic DF of the UNTRUNCATED halo once (no randomness, so
        // sampling stays deterministic). The wide radius bracket drives Ψ → 0 for
        // the outer edge of the Eddington integral; r_min resolves the cusp.
        let psi_max = -self.potential(0.0); // Ψ(0) = G M_s / r_s
        let df = EddingtonDf::build(self, psi_max, 1e-3 * rs, 1e4 * rs);

        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);

        for _ in 0..n {
            // Position: invert the truncated CDF M(<r)/M_vir = X on r ∈ [0, r_vir]
            // by bisection (M(<r) is monotonic). X ∈ [0,1) ⇒ r ∈ [0, r_vir).
            let target = rng.next_f64() * self.virial_mass;
            let mut lo = 0.0;
            let mut hi = r_vir;
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if self.enclosed_mass(mid) < target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let r = 0.5 * (lo + hi);
            pos.push(random_direction(&mut rng) * r);

            // Speed: rejection-sample v ∈ [0, v_esc] from p(v) ∝ v² f(Ψ − v²/2),
            // with a per-radius rejection ceiling found by scanning the PDF.
            let psi = -self.potential(r);
            let v_esc = (2.0 * psi).sqrt();
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

        // Recenter to zero COM and zero net momentum. Truncation at r_vir keeps
        // ⟨r⟩ finite, so the COM is well-conditioned (unlike an untruncated tail).
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

impl crate::eddington::SphericalModel for Nfw {
    fn density(&self, r: f64) -> f64 {
        Nfw::density(self, r)
    }
    fn relative_potential(&self, r: f64) -> f64 {
        -Nfw::potential(self, r)
    }
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
