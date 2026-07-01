//! Pure two-body Kepler placement, shared by every collision IC.
//!
//! A galaxy collision starts by putting the two galaxies' centers of mass on a
//! relative two-body orbit — a computation that depends only on the two total
//! masses, `G`, and the conic (eccentricity + pericenter + starting separation),
//! *not* on the internal galaxy model. Factoring it here means the one set of
//! orbital-mechanics tests (`Collision`'s osculating-elements checks) guards the
//! placement for **all** collision types (Plummer `Collision`, `DiskCollision`,
//! and any future model), instead of each type carrying its own copy of the conic
//! math.
//!
//! Convention (fixed once, here): pericenter lies along **+x**, the orbit is in
//! the **x–y plane**, so the orbital angular momentum points along **+Z**. A disk
//! whose spin is +Z is therefore *prograde* (co-rotating with the encounter).

use galaxy_core::DVec3;

/// Validate the orbital parameters shared by every encounter: strictly positive
/// eccentricity and pericenter, an initial separation at least the pericenter,
/// and — for a bound orbit — a separation no larger than the apocenter (there is
/// no point on a bound conic beyond apocenter). Panics with a descriptive message
/// on violation (IC construction is a programming-time contract, not a runtime
/// fallible path).
pub(crate) fn validate_orbit(eccentricity: f64, pericenter: f64, separation: f64) {
    assert!(eccentricity > 0.0, "eccentricity must be positive");
    assert!(pericenter > 0.0, "pericenter must be positive");
    assert!(
        separation >= pericenter,
        "initial separation ({separation}) must be >= pericenter ({pericenter})"
    );
    if eccentricity < 1.0 {
        let apocenter = pericenter * (1.0 + eccentricity) / (1.0 - eccentricity);
        assert!(
            separation <= apocenter * (1.0 + 1e-12),
            "initial separation ({separation}) exceeds apocenter ({apocenter}) for a bound orbit (e={eccentricity})"
        );
    }
}

/// Relative position and velocity `(r_rel, v_rel)` of the two COMs, with
/// `r_rel = r2 − r1` and `v_rel = v2 − v1`, on the *incoming* branch of the Kepler
/// orbit with gravitational parameter `mu = G·(m1 + m2)`, eccentricity `e`,
/// pericenter `r_peri`, at COM–COM separation `r0`. Pericenter along +x; orbit in
/// the x–y plane.
pub(crate) fn relative_state(mu: f64, e: f64, r_peri: f64, r0: f64) -> (DVec3, DVec3) {
    // Conic about the focus: r(ν) = p / (1 + e·cos ν), with semi-latus rectum
    // p = r_peri·(1 + e) and specific angular momentum h = √(μ·p).
    let p = r_peri * (1.0 + e);
    let h = (mu * p).sqrt();

    // True anomaly at the starting separation, on the *incoming* branch (ν<0
    // ⇒ the bodies are approaching). Clamp guards float drift at the apsides.
    let cos_nu = ((p / r0 - 1.0) / e).clamp(-1.0, 1.0);
    let nu = -cos_nu.acos();
    let (sin_nu, cos_nu) = (nu.sin(), nu.cos());

    // Polar velocity components for a Kepler orbit:
    //   v_r = (μ/h)·e·sin ν,   v_θ = (μ/h)·(1 + e·cos ν) = h/r.
    let mu_over_h = mu / h;
    let v_r = mu_over_h * e * sin_nu;
    let v_t = mu_over_h * (1.0 + e * cos_nu);

    // Pericenter along +x, orbit in the x–y plane: radial r̂ = (cos ν, sin ν, 0),
    // transverse t̂ = (−sin ν, cos ν, 0).
    let r_hat = DVec3::new(cos_nu, sin_nu, 0.0);
    let t_hat = DVec3::new(-sin_nu, cos_nu, 0.0);
    let r_rel = r_hat * r0;
    let v_rel = r_hat * v_r + t_hat * v_t;
    (r_rel, v_rel)
}

/// Split the relative coordinates about the barycenter into per-galaxy COM states
/// `((r1, v1), (r2, v2))` in the global zero-COM / zero-momentum frame. By
/// construction `m1·r1 + m2·r2 = 0`, `m1·v1 + m2·v2 = 0`, `r2 − r1 = r_rel`, and
/// `v2 − v1 = v_rel`.
pub(crate) fn com_states(
    m1: f64,
    m2: f64,
    r_rel: DVec3,
    v_rel: DVec3,
) -> ((DVec3, DVec3), (DVec3, DVec3)) {
    let mtot = m1 + m2;
    // r1 = −(m2/M)·r_rel, r2 = +(m1/M)·r_rel ⇒ m1·r1 + m2·r2 = 0 (same for v).
    let f1 = -m2 / mtot;
    let f2 = m1 / mtot;
    ((r_rel * f1, v_rel * f1), (r_rel * f2, v_rel * f2))
}
