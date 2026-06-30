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

use galaxy_core::State;

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
    pub fn density(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Cumulative mass within radius `r`: M(<r) = M · r³/(r² + a²)^(3/2).
    pub fn enclosed_mass(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Gravitational potential Φ(r) = −G M / √(r² + a²).
    pub fn potential(&self, _r: f64) -> f64 {
        todo!()
    }

    /// Radius enclosing half the total mass, r_h ≈ 1.30477 a.
    pub fn half_mass_radius(&self) -> f64 {
        todo!()
    }

    /// Total gravitational potential energy W = −(3π/32) G M² / a.
    pub fn potential_energy(&self) -> f64 {
        todo!()
    }

    /// Virial-equilibrium total kinetic energy T = −W/2.
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
