//! The Plummer (1911) sphere: a spherically-symmetric, self-gravitating
//! equilibrium with a finite-density core. Its profile, potential, and
//! distribution function all have closed forms, which makes it the cleanest
//! possible first galaxy and an exactly-checkable IC.
//!
//! Profile (total mass `M`, scale radius `a`):
//! - density   ρ(r) = (3M / 4π a³) · (1 + r²/a²)^(−5/2)
//! - mass      M(<r) = M · r³ / (r² + a²)^(3/2)
//! - potential Φ(r)  = −G M / √(r² + a²)
//!
//! Note Φ has the same form as the Plummer-*softened* point mass used by the
//! force solvers — the model and the softening kernel are the same function.
//!
//! Sampling uses the Aarseth–Hénon–Wielen (1974) closed-form recipe, which
//! draws *exactly* from the isotropic distribution function f(ℰ) ∝ ℰ^(7/2):
//! positions by inverting M(<r), speeds by rejection-sampling the marginal
//! speed PDF g(q) = q²(1−q²)^(7/2) (q = v/v_esc) that follows from that f(ℰ).

use galaxy_core::{DVec3, State};

use std::f64::consts::{PI, TAU};

/// A Plummer model parameterized by gravitational constant, total mass, and
/// scale radius. Choose units freely (tests use G = M = a = 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plummer {
    /// Gravitational constant `G`.
    pub g: f64,
    /// Total mass `M`.
    pub total_mass: f64,
    /// Plummer scale radius `a` (the core size; the half-mass radius is ≈1.305a).
    pub scale_radius: f64,
}

impl Plummer {
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

    /// Mass density ρ(r) = (3M / 4π a³)(1 + r²/a²)^(−5/2).
    pub fn density(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        let x2 = (r / a) * (r / a);
        3.0 * self.total_mass / (4.0 * PI * a * a * a) * (1.0 + x2).powf(-2.5)
    }

    /// Cumulative mass within radius `r`: M(<r) = M · r³/(r² + a²)^(3/2).
    pub fn enclosed_mass(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        self.total_mass * r * r * r / (r * r + a * a).powf(1.5)
    }

    /// Gravitational potential Φ(r) = −G M / √(r² + a²).
    pub fn potential(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        -self.g * self.total_mass / (r * r + a * a).sqrt()
    }

    /// Radius enclosing half the total mass, r_h ≈ 1.30477 a.
    ///
    /// From M(<r)/M = (r²/(r²+a²))^(3/2) = 1/2 ⇒ r = a·√(y/(1−y)), y = 2^(−2/3).
    pub fn half_mass_radius(&self) -> f64 {
        let y = 0.5_f64.powf(2.0 / 3.0);
        self.scale_radius * (y / (1.0 - y)).sqrt()
    }

    /// Total gravitational potential energy W = −(3π/32) G M² / a.
    pub fn potential_energy(&self) -> f64 {
        -3.0 * PI * self.g * self.total_mass * self.total_mass / (32.0 * self.scale_radius)
    }

    /// Virial-equilibrium total kinetic energy T = −W/2.
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
        let m_each = self.total_mass / n as f64;
        // v_esc(r) = √(2GM) · (r² + a²)^(−1/4): the local escape speed.
        let v_esc_coeff = (2.0 * self.g * self.total_mass).sqrt();

        let mut pos = Vec::with_capacity(n);
        let mut vel = Vec::with_capacity(n);

        for _ in 0..n {
            // Position: invert M(<r)/M = X ⇒ r = a·(X^(−2/3) − 1)^(−1/2).
            // X ∈ [0,1) ⇒ X^(−2/3) ∈ (1, ∞], so the radicand is always > 0.
            let x = rng.next_f64();
            let r = a / (x.powf(-2.0 / 3.0) - 1.0).sqrt();
            pos.push(random_direction(&mut rng) * r);

            // Speed: rejection-sample q = v/v_esc from g(q) = q²(1−q²)^(7/2).
            // max g ≈ 0.0922, so 0.1 is a valid (tight) rejection ceiling.
            let q = loop {
                let q = rng.next_f64();
                let height = 0.1 * rng.next_f64();
                if height < q * q * (1.0 - q * q).powf(3.5) {
                    break q;
                }
            };
            let v_esc = v_esc_coeff * (r * r + a * a).powf(-0.25);
            vel.push(random_direction(&mut rng) * (q * v_esc));
        }

        // Recenter: subtract the mean position and velocity. With equal masses
        // these are the COM and the per-particle net momentum, so the result
        // has zero COM and zero total momentum to roundoff.
        let inv_n = 1.0 / n as f64;
        let mean_pos = pos.iter().fold(DVec3::ZERO, |acc, &p| acc + p) * inv_n;
        let mean_vel = vel.iter().fold(DVec3::ZERO, |acc, &v| acc + v) * inv_n;
        for p in &mut pos {
            *p -= mean_pos;
        }
        for v in &mut vel {
            *v -= mean_vel;
        }

        // Sequential ids, single progenitor, t = 0, a = 1 (non-cosmological).
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

/// SplitMix64: a tiny, fast, deterministic PRNG. Avoids an external `rand`
/// dependency (matching the project's hand-rolled-LCG convention) while giving
/// well-distributed draws. `next_f64` returns a value strictly in [0, 1).
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

    /// Uniform in [0, 1): the top 53 bits scaled by 2^(−53), so the maximum is
    /// (2^53 − 1)/2^53 < 1 and the value never reaches 1.0.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}
