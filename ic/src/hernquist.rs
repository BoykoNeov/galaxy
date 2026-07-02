//! The Hernquist (1990) model: a spherically-symmetric, self-gravitating
//! equilibrium with a **central cusp** ρ ∝ r⁻¹ and an r⁻⁴ envelope. Unlike the
//! cored Plummer sphere it has finite total mass yet a divergent central
//! density, making it the analytic stand-in for cuspy (de Vaucouleurs / NFW-like)
//! halos and bulges — and, crucially for this engine, it has a **closed-form
//! isotropic distribution function** f(ℰ), so it samples exactly like Plummer.
//!
//! Profile (total mass `M`, scale radius `a`):
//! - density   ρ(r) = (M / 2π) · a / (r (r + a)³)
//! - mass      M(<r) = M · r² / (r + a)²
//! - potential Φ(r)  = −G M / (r + a)
//!
//! The half-mass radius is r_h = (1 + √2) a ≈ 2.414 a and the total potential
//! energy is W = −G M² / (6 a) (so virial T = −W/2 = G M² / 12 a).
//!
//! Sampling draws *exactly* from the isotropic DF (Hernquist 1990, eq. 17;
//! Binney & Tremaine §4.3, eq. 4.51): positions by inverting M(<r) in closed
//! form, speeds by rejection-sampling the marginal speed PDF v² f(Ψ(r) − v²/2).
//! Unlike Plummer, the substitution q = v/v_esc does NOT make that PDF
//! radius-independent — the envelope changes shape with r — so the rejection
//! ceiling is computed **per radius**, not with a single global constant.

use galaxy_core::{DVec3, State};

/// A Hernquist model parameterized by gravitational constant, total mass, and
/// scale radius. Choose units freely (tests use G = M = a = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hernquist {
    /// Gravitational constant `G`.
    pub g: f64,
    /// Total mass `M`.
    pub total_mass: f64,
    /// Hernquist scale radius `a` (sets the cusp/envelope break; r_h ≈ 2.414a).
    pub scale_radius: f64,
}

impl Hernquist {
    /// Construct a model. All three parameters must be strictly positive.
    pub fn new(g: f64, total_mass: f64, scale_radius: f64) -> Self {
        assert!(g > 0.0, "G must be positive");
        assert!(total_mass > 0.0, "total mass must be positive");
        assert!(scale_radius > 0.0, "scale radius must be positive");
        Self {
            g,
            total_mass,
            scale_radius,
        }
    }

    /// Mass density ρ(r) = (M / 2π) · a / (r (r + a)³).
    pub fn density(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Cumulative mass within radius `r`: M(<r) = M · r² / (r + a)².
    pub fn enclosed_mass(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Gravitational potential Φ(r) = −G M / (r + a).
    pub fn potential(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Isotropic distribution function f(ℰ) as a function of the **relative**
    /// (binding) energy ℰ = Ψ(r) − v²/2, where Ψ = −Φ ≥ 0. Returns 0 for ℰ ≤ 0.
    /// Closed form (Hernquist 1990 eq. 17 / B&T eq. 4.51).
    pub fn df(&self, _energy: f64) -> f64 {
        todo!()
    }

    /// Radius enclosing half the total mass, r_h = (1 + √2) a ≈ 2.41421 a.
    pub fn half_mass_radius(&self) -> f64 {
        todo!()
    }

    /// Total gravitational potential energy W = −G M² / (6 a).
    pub fn potential_energy(&self) -> f64 {
        todo!()
    }

    /// Virial-equilibrium total kinetic energy T = −W/2 = G M² / (12 a).
    pub fn kinetic_energy(&self) -> f64 {
        todo!()
    }

    /// Structural dynamical time t_dyn = √(a³ / G M).
    pub fn dynamical_time(&self) -> f64 {
        todo!()
    }

    /// Draw `n` equal-mass particles from the analytic distribution function,
    /// deterministically seeded by `seed`. The realization is recentered to
    /// zero center of mass and zero net momentum.
    pub fn sample(&self, _n: usize, _seed: u64) -> State {
        todo!()
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
