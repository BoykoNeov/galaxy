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
        todo!()
    }

    /// Characteristic mass M_s = 4π ρ_s r_s³ = M_vir / m(c).
    pub fn characteristic_mass(&self) -> f64 {
        todo!()
    }

    /// Characteristic density ρ_s = M_s / (4π r_s³).
    pub fn characteristic_density(&self) -> f64 {
        todo!()
    }

    /// Virial radius r_vir = c · r_s.
    pub fn virial_radius(&self) -> f64 {
        todo!()
    }

    /// Mass density ρ(r) = ρ_s / (x (1+x)²), x = r/r_s.
    pub fn density(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Cumulative mass M(<r) = M_s [ln(1+x) − x/(1+x)] (untruncated).
    pub fn enclosed_mass(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Gravitational potential Φ(r) = −G M_s ln(1+x)/r, with Φ(0) = −G M_s/r_s.
    pub fn potential(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Structural dynamical time t_dyn = √(r_s³ / (G M_s)) (the inner scale).
    pub fn dynamical_time(&self) -> f64 {
        todo!()
    }

    /// Draw `n` equal-mass particles: positions from the mass profile truncated
    /// at r_vir, velocities from the numerically Eddington-inverted DF of the
    /// untruncated potential. Recentered to zero COM and zero net momentum.
    pub fn sample(&self, _n: usize, _seed: u64) -> State {
        todo!()
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
#[allow(dead_code)]
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
