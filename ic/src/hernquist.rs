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

use std::f64::consts::{PI, SQRT_2, TAU};

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
    pub fn density(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        self.total_mass * a / (2.0 * PI * r * (r + a).powi(3))
    }

    /// Cumulative mass within radius `r`: M(<r) = M · r² / (r + a)².
    pub fn enclosed_mass(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        self.total_mass * r * r / ((r + a) * (r + a))
    }

    /// Gravitational potential Φ(r) = −G M / (r + a).
    pub fn potential(&self, r: f64) -> f64 {
        -self.g * self.total_mass / (r + self.scale_radius)
    }

    /// Isotropic distribution function f(ℰ) as a function of the **relative**
    /// (binding) energy ℰ = Ψ(r) − v²/2, where Ψ = −Φ ≥ 0. Returns 0 for ℰ ≤ 0.
    ///
    /// Closed form (Hernquist 1990 eq. 17; Binney & Tremaine §4.3 eq. 4.51), in
    /// physical units. With v_g² = G M / a and the dimensionless binding energy
    /// q = ℰ / v_g² = ℰ a / (G M) ∈ (0, 1]:
    ///
    ///   f(ℰ) = M / (8√2 π³ a³ v_g³)
    ///        · [3 arcsin√q + √(q(1−q))(1−2q)(8q²−8q−3)] / (1−q)^(5/2).
    pub fn df(&self, energy: f64) -> f64 {
        if energy <= 0.0 {
            return 0.0;
        }
        let a = self.scale_radius;
        let gm = self.g * self.total_mass;
        let vg2 = gm / a; // v_g² = GM/a = Ψ(0), the deepest binding energy
        let q = energy / vg2; // dimensionless binding energy ∈ (0, 1]
        if q >= 1.0 {
            // The (1−q)^(−5/2) pole sits at ℰ = Ψ(0), reached only at r = 0, v = 0
            // (measure zero). Guard it so the sampler never divides by zero.
            return 0.0;
        }
        let vg = vg2.sqrt();
        let pref = self.total_mass / (8.0 * SQRT_2 * PI.powi(3) * a.powi(3) * vg.powi(3));
        let bracket = 3.0 * q.sqrt().asin()
            + (q * (1.0 - q)).sqrt() * (1.0 - 2.0 * q) * (8.0 * q * q - 8.0 * q - 3.0);
        pref * bracket / (1.0 - q).powf(2.5)
    }

    /// Radius enclosing half the total mass, r_h = (1 + √2) a ≈ 2.41421 a.
    ///
    /// From M(<r)/M = (r/(r+a))² = 1/2 ⇒ r/(r+a) = 1/√2 ⇒ r = a/(√2 − 1) = (1+√2)a.
    pub fn half_mass_radius(&self) -> f64 {
        (1.0 + SQRT_2) * self.scale_radius
    }

    /// Total gravitational potential energy W = −G M² / (6 a).
    pub fn potential_energy(&self) -> f64 {
        -self.g * self.total_mass * self.total_mass / (6.0 * self.scale_radius)
    }

    /// Virial-equilibrium total kinetic energy T = −W/2 = G M² / (12 a).
    pub fn kinetic_energy(&self) -> f64 {
        -0.5 * self.potential_energy()
    }

    /// Structural dynamical time t_dyn = √(a³ / G M).
    pub fn dynamical_time(&self) -> f64 {
        let a = self.scale_radius;
        (a * a * a / (self.g * self.total_mass)).sqrt()
    }

    /// Draw `n` equal-mass particles from the analytic distribution function,
    /// deterministically seeded by `seed`. The realization is recentered to
    /// zero center of mass and zero net momentum.
    pub fn sample(&self, n: usize, seed: u64) -> State {
        let mut rng = SplitMix64::new(seed);
        let a = self.scale_radius;
        let gm = self.g * self.total_mass;
        let m_each = self.total_mass / n as f64;

        // Truncation. Hernquist has finite mass but an r⁻⁴ tail whose FIRST
        // moment ⟨r⟩ = ∫ r · 4πr²ρ dr diverges (∝ ∫ dr/r), so the finite-N center
        // of mass is dominated by the single farthest particle (X → 1 gives
        // r → ∞) — recentering to it would drag the cusp far off origin. Real
        // Hernquist N-body ICs always truncate the tail; sampling X uniform on
        // [0, X_MAX) draws the profile truncated at r_max EXACTLY, because X is
        // itself the enclosed-mass fraction M(<r)/M. r_max = 300a keeps 99.34% of
        // the mass — negligible for the profile at the scales that matter, yet it
        // bounds ⟨r⟩ so the COM (and thus recentering) is well-conditioned.
        const R_MAX_FRAC: f64 = 300.0; // r_max in units of the scale radius a
        let x_max = (R_MAX_FRAC / (R_MAX_FRAC + 1.0)).powi(2); // M(<r_max)/M

        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);

        for _ in 0..n {
            // Position: invert M(<r)/M = (r/(r+a))² = X ⇒ r = a√X/(1−√X),
            // with X drawn uniformly on [0, X_MAX) to truncate the tail at r_max.
            let x = x_max * rng.next_f64();
            let root_x = x.sqrt();
            let r = a * root_x / (1.0 - root_x);
            pos.push(random_direction(&mut rng) * r);

            // Speed: rejection-sample v ∈ [0, v_esc] from p(v) ∝ v² f(Ψ − v²/2).
            // Unlike Plummer the envelope shape depends on r (through Ψ), so the
            // rejection ceiling is found per radius by scanning the PDF on a grid.
            let psi = gm / (r + a); // relative potential Ψ(r)
            let v_esc = (2.0 * psi).sqrt();
            let speed_pdf = |v: f64| -> f64 { v * v * self.df(psi - 0.5 * v * v) };
            // Envelope max on a fine grid, padded for between-grid peaks. The
            // integrand vanishes at both ends (v² at 0, f(0)=0 at v_esc), so the
            // maximum is interior and a grid scan captures it.
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

        // Recenter to zero COM and zero net momentum (equal masses ⇒ the means
        // are the COM and per-particle momentum), matching the Plummer sampler.
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

/// A unit vector drawn isotropically on the sphere (uniform in cosθ and φ).
fn random_direction(rng: &mut SplitMix64) -> DVec3 {
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
