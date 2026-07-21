//! Coloring modes v2 (DESIGN.md M6e): pure color maps that diversify what the
//! movie's colors *mean*, beyond the flat progenitor palette.
//!
//! Three maps live here, all pure functions gated by exact/invariant tests:
//!
//! * [`initial_radius_colors`] — the **frozen-at-t=0 initial-radius ramp**: each
//!   particle is colored once, from its radius about its progenitor's mass-weighted
//!   COM in the *initial* snapshot, and keeps that color for the whole movie (keyed
//!   by particle index). Tidal tails then carry a visible gradient showing where
//!   their material came from.
//! * [`dispersion_colors`] — a σ_v ramp: dynamically cold material toward one color,
//!   dynamically hot toward another (the deferred refinement named in `prepare`).
//! * [`compression_colors`] — the **star-formation proxy**: a hue shift from the
//!   base color toward a "young population" blue-white, keyed on density
//!   *compression* `ρ(t)/ρ(0)` rather than absolute ρ, so only tidally-compressed
//!   material (bridges, shocked overlap, ring waves) lights up while pre-existing
//!   dense cores keep their old-population color. Honest caveat (DESIGN M6e): the
//!   sim is collisionless — stellar density stands in for gas compression; this is
//!   a standard visualization proxy, not physics.
//!
//! All ramps use the two-product lerp `(1−t)·a + t·b`, which is bit-exact at both
//! endpoints — the same discipline as `interp::subframe` — so "exactly the base
//! color when the effect is off" is a hard guarantee, not a tolerance.

use galaxy_core::{DVec3, State};

/// Per-progenitor `(inner, outer)` color ramp for [`initial_radius_colors`].
/// A progenitor id out of range wraps modulo the ramp count; an empty ramp list
/// falls back to white — exactly the palette conventions of `prepare`.
#[derive(Clone, Debug, PartialEq)]
pub struct RadialRamp {
    /// `(inner, outer)` linear-RGB endpoints, indexed by `progenitor % len`.
    pub ramps: Vec<([f32; 3], [f32; 3])>,
}

/// Frozen-at-t=0 initial-radius colors, one per particle (keyed by index).
///
/// Per progenitor: the mass-weighted COM and the **half-mass radius** `r_half`
/// (the smallest particle radius enclosing ≥ half the progenitor's mass) are
/// computed from `state`; each particle maps through `t = r / (r + r_half)` —
/// monotone, bounded in `[0, 1)`, exactly `0` at the COM and exactly `½` at
/// `r = r_half`, so the *median-mass* particle sits at the ramp midpoint and the
/// normalization is per-progenitor scale-free. A degenerate progenitor
/// (`r_half = 0`: single particle, or all coincident) gets the inner color.
pub fn initial_radius_colors(state: &State, ramp: &RadialRamp) -> Vec<[f32; 3]> {
    let n = state.len();
    if ramp.ramps.is_empty() {
        return vec![WHITE; n];
    }

    // Group particle indices by progenitor (a small dense id space — u16).
    let mut groups: Vec<(u16, Vec<usize>)> = Vec::new();
    for i in 0..n {
        let tag = state.progenitor[i].0;
        match groups.iter_mut().find(|(t, _)| *t == tag) {
            Some((_, members)) => members.push(i),
            None => groups.push((tag, vec![i])),
        }
    }

    let mut colors = vec![WHITE; n];
    for (tag, members) in &groups {
        // Mass-weighted COM of this progenitor's particles.
        let (msum, weighted) = members.iter().fold((0.0, DVec3::ZERO), |(m, w), &i| {
            (m + state.mass[i], w + state.pos[i] * state.mass[i])
        });
        let com = if msum > 0.0 {
            weighted / msum
        } else {
            DVec3::ZERO // massless group: any center works, radii decide nothing
        };

        // Half-mass radius: the smallest particle radius enclosing ≥ half the
        // progenitor's mass (for equal masses, the median-mass particle's radius).
        let mut radii: Vec<(f64, f64)> = members
            .iter()
            .map(|&i| (state.pos[i].distance(com), state.mass[i]))
            .collect();
        radii.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut cum = 0.0;
        let mut r_half = 0.0;
        for &(r, m) in &radii {
            cum += m;
            if cum >= 0.5 * msum {
                r_half = r;
                break;
            }
        }

        let (inner, outer) = ramp.ramps[*tag as usize % ramp.ramps.len()];
        for &i in members {
            let r = state.pos[i].distance(com);
            // t = r/(r + r_half): 0 at the COM, exactly ½ at r = r_half, → 1 far
            // out. A degenerate progenitor (r_half = 0) pins the inner color —
            // guarding the 0/0 at its own COM.
            let t = if r_half > 0.0 {
                (r / (r + r_half)) as f32
            } else {
                0.0
            };
            colors[i] = lerp3(inner, outer, t);
        }
    }
    colors
}

/// White fallback for an empty ramp list — the palette convention.
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

/// Two-product lerp, bit-exact at both endpoints (`t = 0` ⇒ `a`, `t = 1` ⇒ `b`).
fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let s = 1.0 - t;
    [
        s * a[0] + t * b[0],
        s * a[1] + t * b[1],
        s * a[2] + t * b[2],
    ]
}

/// σ_v color ramp: `t = σ² / (σ² + σ_ref²)` from `cold` toward `hot`, with
/// `σ_ref` the mean over the *positive* dispersions. Monotone, bounded in
/// `[0, 1)`, exactly `cold` at `σ = 0` (including the degenerate-neighbourhood
/// sentinel) and exactly the midpoint mix at `σ = σ_ref`. All-zero dispersions
/// (no reference) yield all-`cold`.
///
/// The map is quadratic in σ — the specific kinetic energy of the random
/// motions — rather than linear: in a self-gravitating body σ varies only by a
/// factor ~2–3 across the visible radii, and the linear ratio compressed that
/// onto a mushy `t ≈ 0.4–0.6` band (every particle the same midpoint gray in
/// the first rendered A/B); squaring doubles the contrast around the reference
/// while preserving every endpoint/midpoint/monotonicity guarantee.
pub fn dispersion_colors(sigma: &[f64], cold: [f32; 3], hot: [f32; 3]) -> Vec<[f32; 3]> {
    // Reference dispersion: the mean over the positive values — the same
    // discipline as the density boost's ρ_ref (0.0 is the degenerate sentinel).
    let (sum, count) = sigma
        .iter()
        .filter(|&&s| s > 0.0)
        .fold((0.0, 0usize), |(a, c), &s| (a + s, c + 1));
    if count == 0 {
        return vec![cold; sigma.len()]; // no reference: everything is cold
    }
    let sigma_ref = sum / count as f64;
    let ref2 = sigma_ref * sigma_ref;
    sigma
        .iter()
        .map(|&s| {
            // t = σ²/(σ² + σ_ref²): exactly 0 at σ = 0 (bit-exact cold via the
            // two-product lerp), exactly ½ at σ = σ_ref, → 1 as σ → ∞.
            let t = if s > 0.0 {
                (s * s / (s * s + ref2)) as f32
            } else {
                0.0
            };
            lerp3(cold, hot, t)
        })
        .collect()
}

/// Age-triggered star-formation tint (natal-ember-forge, F6): shift each
/// recently-formed star toward `young`, fading exponentially back to its base
/// color over ~`tau` sim-time.
///
/// For particle `i` the age is `a_i = now − formation_time[i]` and the mix is
/// `t = clamp(strength, 0, 1) · exp(−a_i / tau)`: a just-formed star (`a = 0`)
/// sits at `strength` of the way to `young`; the tint decays by `1/e` every
/// `tau` of sim-time. A **primordial** star carries
/// `formation_time = State::PRIMORDIAL` (`−∞`), so `a = now − (−∞) = +∞`,
/// `exp(−∞) = +0.0`, `t = strength · 0.0 = +0.0`, and the two-product [`lerp3`]
/// returns the base color **bit-exactly** — no `is_infinite` branch, exactly
/// what the `−∞` sentinel was chosen for (F1). `strength = 0` is likewise the
/// bit-exact identity.
///
/// `tau > 0` is a caller contract (the fade timescale; the scenario layer
/// validates it finite-and-positive at S6). `now` is the snapshot time —
/// view-INDEPENDENT (D9-safe), passed in by [`crate::prepare`] as `state.time`.
/// Because `formation_time ≤ now` in the pipeline (a star cannot form after the
/// snapshot it appears in), `a ≥ 0` and `exp(−a/tau) ∈ (0, 1]`, so the
/// strength-clamp alone bounds `t` to `[0, strength]` — the tint never
/// overshoots the `[base, young]` segment (no separate `t`-clamp, matching
/// [`compression_colors`]).
///
/// Panics if `base.len() != formation_time.len()` — a caller contract violation
/// (they must describe the same particles), matching [`compression_colors`].
pub fn age_colors(
    base: &[[f32; 3]],
    formation_time: &[f64],
    now: f64,
    young: [f32; 3],
    strength: f32,
    tau: f64,
) -> Vec<[f32; 3]> {
    assert_eq!(
        base.len(),
        formation_time.len(),
        "formation_time is not for these particles"
    );
    let s = strength.clamp(0.0, 1.0);
    base.iter()
        .zip(formation_time)
        .map(|(&b, &tf)| {
            // age = now − formation_time: +∞ for a primordial star (tf = −∞), so
            // exp(−age/tau) = +0.0 and t = +0.0 → base bit-exactly through the
            // two-product lerp. No is_infinite branch (F1). For age ≥ 0 the
            // exponential is in (0, 1], so the strength-clamp alone bounds t.
            let age = now - tf;
            let t = (s as f64 * (-age / tau).exp()) as f32;
            lerp3(b, young, t)
        })
        .collect()
}

/// Star-formation-proxy hue shift (compression-triggered): lerp each base color
/// toward `young` by `t = clamp(strength, 0, 1) · (1 − ρ0/max(ρ, ρ0))`.
///
/// The trigger is the density **increment** over the same particle's t=0
/// neighbourhood: `ρ ≤ ρ0` (uncompressed or rarefied) keeps the base color
/// *bit-exactly*, `ρ → ∞` saturates at `strength` of the way to `young` —
/// bounded on the `[base, young]` segment and monotone in `ρ`. The `0.0`
/// density sentinel on either side means "no neighbourhood", not a real void,
/// and keeps the base color. `strength = 0` is the identity.
///
/// Panics if the column lengths disagree — a caller contract violation
/// (`rho`/`rho0` must be estimates for these same particles), not a data
/// condition.
pub fn compression_colors(
    base: &[[f32; 3]],
    rho: &[f64],
    rho0: &[f64],
    young: [f32; 3],
    strength: f32,
) -> Vec<[f32; 3]> {
    assert_eq!(base.len(), rho.len(), "rho is not for these particles");
    assert_eq!(base.len(), rho0.len(), "rho0 is not for these particles");
    let s = strength.clamp(0.0, 1.0);
    base.iter()
        .zip(rho.iter().zip(rho0))
        .map(|(&b, (&r, &r0))| {
            // Compression fraction 1 − ρ0/max(ρ, ρ0) — the density_boost form with
            // the particle's own t=0 density as the reference: exactly 0 (t = 0 ⇒
            // bit-exact base via the two-product lerp) for ρ ≤ ρ0, → 1 as ρ → ∞.
            // A 0.0 on either side is the degenerate-kNN sentinel: no estimate,
            // no shift.
            let t = if r > 0.0 && r0 > 0.0 {
                s * (1.0 - r0 / r.max(r0)) as f32
            } else {
                0.0
            };
            lerp3(b, young, t)
        })
        .collect()
}
