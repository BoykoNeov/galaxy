//! Coloring modes v2 (DESIGN.md M6e): the pure color maps — initial-radius ramp,
//! velocity-dispersion ramp, and the compression-triggered star-formation hue.
//!
//! Expectations are hand-derived, never read back from the function under test:
//! symmetric particle layouts make the COM and half-mass radius exact, so the ramp
//! parameter `t = r/(r + r_half)` is a closed form; σ_ref and ρ-ratios are chosen
//! so the mapped `t` is an exact binary fraction wherever exactness is asserted.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};
use galaxy_renderprep::{
    age_colors, compression_colors, dispersion_colors, initial_radius_colors, RadialRamp,
};

/// Build a state from positions + progenitors with the given masses (vel zero —
/// the radius ramp is a positions-only map).
fn state_of(pos: Vec<DVec3>, progenitor: Vec<u16>, mass: Vec<f64>) -> State {
    let n = pos.len();
    assert_eq!(progenitor.len(), n);
    assert_eq!(mass.len(), n);
    State {
        vel: vec![DVec3::ZERO; n],
        id: (0..n as u64).map(ParticleId).collect(),
        progenitor: progenitor.into_iter().map(Progenitor).collect(),
        kind: vec![Species::Collisionless; n],
        u: vec![0.0; n],
        formation_time: vec![State::PRIMORDIAL; n],
        time: 0.0,
        a: 1.0,
        pos,
        mass,
    }
}

/// Two-product lerp — the exact-endpoint form the ramps are specified over.
fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    ]
}

fn assert_rgb_close(got: [f32; 3], want: [f32; 3], tol: f32) {
    for c in 0..3 {
        assert!(
            (got[c] - want[c]).abs() <= tol,
            "channel {c}: got {got:?}, want {want:?}"
        );
    }
}

const INNER: [f32; 3] = [1.0, 0.4, 0.1];
const OUTER: [f32; 3] = [0.2, 0.6, 0.9];

/// Nine equal-mass particles of one progenitor on the x-axis at 0, ±1, ±2, ±3, ±4:
/// COM is exactly the origin (symmetry), radii are [0,1,1,2,2,3,3,4,4], total mass
/// 9 with half-mass 4.5, and the cumulative mass first reaches ≥ 4.5 at radius 2 —
/// so r_half = 2 exactly, and t = r/(r+2) is a closed form per particle.
fn ramp_state() -> State {
    let xs = [0.0, 1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
    let pos = xs.iter().map(|&x| DVec3::new(x, 0.0, 0.0)).collect();
    state_of(pos, vec![0; 9], vec![1.0; 9])
}

// --------------------------------------------------------------------------
// Initial-radius ramp: endpoints, midpoint, monotonicity, normalization
// --------------------------------------------------------------------------

#[test]
fn ramp_particle_at_com_gets_exactly_the_inner_color() {
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER)],
    };
    let colors = initial_radius_colors(&ramp_state(), &ramp);
    // Particle 0 sits exactly at the COM: t = 0 → bit-exact inner endpoint.
    assert_eq!(colors[0], INNER);
}

#[test]
fn ramp_half_mass_radius_is_the_exact_midpoint() {
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER)],
    };
    let colors = initial_radius_colors(&ramp_state(), &ramp);
    // Particles at r = r_half = 2 (indices 3, 4): t = 2/(2+2) = 1/2 exactly —
    // the median-mass particle sits at the ramp midpoint (the per-progenitor
    // normalization gate). Tolerance only for the lerp's own rounding.
    let midpoint = mix(INNER, OUTER, 0.5);
    assert_rgb_close(colors[3], midpoint, 1e-6);
    assert_rgb_close(colors[4], midpoint, 1e-6);
}

#[test]
fn ramp_is_monotone_in_radius_toward_the_outer_color() {
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER)],
    };
    let colors = initial_radius_colors(&ramp_state(), &ramp);
    // Radii 0 < 1 < 2 < 3 < 4 at indices 0, 1, 3, 5, 7: every channel must move
    // strictly monotonically from inner toward outer, and never overshoot it.
    let ordered = [colors[0], colors[1], colors[3], colors[5], colors[7]];
    for c in 0..3 {
        let toward_outer = OUTER[c] - INNER[c]; // sign of the ramp direction
        for w in ordered.windows(2) {
            let step = (w[1][c] - w[0][c]) * toward_outer.signum();
            assert!(step > 0.0, "channel {c} not strictly monotone: {ordered:?}");
        }
        for col in &ordered {
            let lo = INNER[c].min(OUTER[c]);
            let hi = INNER[c].max(OUTER[c]);
            assert!(col[c] >= lo && col[c] <= hi, "channel {c} out of segment");
        }
    }
}

#[test]
fn ramp_normalization_is_per_progenitor_scale_and_position_free() {
    // Progenitor 1 is progenitor 0's layout scaled ×5 and shifted to (100, 7, -3).
    // The half-mass normalization is per progenitor, so matching particles must
    // land on identical ramp parameters — identical colors, exactly.
    let base = ramp_state();
    let offset = DVec3::new(100.0, 7.0, -3.0);
    let mut pos = base.pos.clone();
    let mut progenitor = vec![0u16; 9];
    for p in &base.pos {
        pos.push(*p * 5.0 + offset);
        progenitor.push(1);
    }
    let state = state_of(pos, progenitor, vec![1.0; 18]);
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER), (INNER, OUTER)],
    };
    let colors = initial_radius_colors(&state, &ramp);
    for i in 0..9 {
        assert_eq!(
            colors[i],
            colors[i + 9],
            "particle {i}: scaled/shifted twin must get the same color"
        );
    }
}

#[test]
fn ramp_com_and_half_mass_radius_are_mass_weighted() {
    // Two particles of one progenitor: m=3 at x=0, m=1 at x=4. The mass-weighted
    // COM is x=1 (an unweighted mean would sit at x=2 and give both particles the
    // same radius, hence the same color). Radii about x=1 are 1 and 3; cumulative
    // mass reaches half (2 of 4) at the first radius, so r_half = 1:
    //   t0 = 1/(1+1) = 1/2,  t1 = 3/(3+1) = 3/4.
    let state = state_of(
        vec![DVec3::ZERO, DVec3::new(4.0, 0.0, 0.0)],
        vec![0, 0],
        vec![3.0, 1.0],
    );
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER)],
    };
    let colors = initial_radius_colors(&state, &ramp);
    assert_ne!(colors[0], colors[1], "COM must be mass-weighted");
    assert_rgb_close(colors[0], mix(INNER, OUTER, 0.5), 1e-6);
    assert_rgb_close(colors[1], mix(INNER, OUTER, 0.75), 1e-6);
}

#[test]
fn ramp_degenerate_progenitors_get_the_inner_color() {
    // Progenitor 1 is a single particle (r_half = 0), progenitor 2 is two
    // coincident particles (all radii 0, r_half = 0): both degenerate cases must
    // yield the inner color exactly, not NaN from 0/0.
    let mut pos = ramp_state().pos;
    pos.push(DVec3::new(50.0, 0.0, 0.0)); // lone particle, progenitor 1
    pos.push(DVec3::new(-9.0, 2.0, 1.0)); // coincident pair, progenitor 2
    pos.push(DVec3::new(-9.0, 2.0, 1.0));
    let progenitor = vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2];
    let n = pos.len();
    let state = state_of(pos, progenitor, vec![1.0; n]);
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER), (INNER, OUTER), (INNER, OUTER)],
    };
    let colors = initial_radius_colors(&state, &ramp);
    assert_eq!(colors[9], INNER);
    assert_eq!(colors[10], INNER);
    assert_eq!(colors[11], INNER);
}

#[test]
fn ramp_progenitor_wraps_modulo_and_empty_ramps_fall_back_to_white() {
    // Progenitor 3 with a 2-ramp list wraps to ramps[1] — same convention as the
    // palette. An empty ramp list yields white for everyone.
    let state = state_of(
        vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
        vec![3, 3],
        vec![1.0, 1.0],
    );
    let distinct = ([0.9, 0.1, 0.1], [0.1, 0.9, 0.1]);
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER), distinct],
    };
    let wrapped = initial_radius_colors(&state, &ramp);
    let direct = initial_radius_colors(
        &state_of(
            vec![DVec3::ZERO, DVec3::new(1.0, 0.0, 0.0)],
            vec![1, 1],
            vec![1.0, 1.0],
        ),
        &ramp,
    );
    assert_eq!(wrapped, direct);

    let white = initial_radius_colors(&state, &RadialRamp { ramps: vec![] });
    assert!(white.iter().all(|&c| c == [1.0, 1.0, 1.0]));
}

#[test]
fn ramp_is_deterministic() {
    let state = ramp_state();
    let ramp = RadialRamp {
        ramps: vec![(INNER, OUTER)],
    };
    assert_eq!(
        initial_radius_colors(&state, &ramp),
        initial_radius_colors(&state, &ramp)
    );
}

// --------------------------------------------------------------------------
// Velocity-dispersion ramp
// --------------------------------------------------------------------------

const COLD: [f32; 3] = [0.1, 0.2, 0.9];
const HOT: [f32; 3] = [0.9, 0.8, 0.1];

#[test]
fn dispersion_zero_sigma_is_exactly_cold() {
    // σ = 0 (dynamically cold, or the degenerate-neighbourhood sentinel) must be
    // the bit-exact cold endpoint.
    let colors = dispersion_colors(&[0.0, 2.0], COLD, HOT);
    assert_eq!(colors[0], COLD);
    assert_ne!(colors[1], COLD);
}

#[test]
fn dispersion_sigma_at_the_mean_is_the_exact_midpoint() {
    // σ = [1, 2, 3]: σ_ref = mean = 2, so t(2) = 2/(2+2) = 1/2 exactly.
    let colors = dispersion_colors(&[1.0, 2.0, 3.0], COLD, HOT);
    assert_rgb_close(colors[1], mix(COLD, HOT, 0.5), 1e-6);
}

#[test]
fn dispersion_is_monotone_and_bounded_on_the_segment() {
    let colors = dispersion_colors(&[0.0, 0.5, 2.0, 8.0, 1000.0], COLD, HOT);
    for c in 0..3 {
        let toward_hot = (HOT[c] - COLD[c]).signum();
        for w in colors.windows(2) {
            assert!(
                (w[1][c] - w[0][c]) * toward_hot >= 0.0,
                "channel {c} not monotone: {colors:?}"
            );
        }
        let (lo, hi) = (COLD[c].min(HOT[c]), COLD[c].max(HOT[c]));
        for col in &colors {
            assert!(col[c] >= lo && col[c] <= hi, "channel {c} off segment");
        }
    }
}

#[test]
fn dispersion_all_zero_sigmas_are_all_cold() {
    // No positive dispersion → no reference → everyone stays cold (identity-like
    // degenerate case, mirroring density_boost's all-zero rule).
    let colors = dispersion_colors(&[0.0, 0.0, 0.0], COLD, HOT);
    assert!(colors.iter().all(|&c| c == COLD));
}

#[test]
fn dispersion_colors_are_deterministic() {
    let sigma = [0.3, 1.7, 0.0, 4.2];
    assert_eq!(
        dispersion_colors(&sigma, COLD, HOT),
        dispersion_colors(&sigma, COLD, HOT)
    );
}

// --------------------------------------------------------------------------
// Compression-triggered star-formation hue
// --------------------------------------------------------------------------

const BASE: [f32; 3] = [0.8, 0.4, 0.2];
const YOUNG: [f32; 3] = [0.5, 0.7, 1.0];

#[test]
fn compression_uncompressed_and_rarefied_keep_base_bit_exactly() {
    // ρ = ρ0 (untouched) and ρ < ρ0 (tidally stretched) must both keep the base
    // color bit-for-bit — only *compression* triggers the young-population shift,
    // so undisturbed dense cores keep their old-population color.
    let base = [BASE, BASE, BASE];
    let out = compression_colors(&base, &[3.0, 1.5, 6.0], &[3.0, 3.0, 3.0], YOUNG, 1.0);
    assert_eq!(out[0], BASE);
    assert_eq!(out[1], BASE);
    assert_ne!(out[2], BASE);
}

#[test]
fn compression_strength_zero_is_the_identity() {
    let base = [BASE, [0.1, 0.9, 0.5]];
    let out = compression_colors(&base, &[100.0, 100.0], &[1.0, 1.0], YOUNG, 0.0);
    assert_eq!(out, base.to_vec());
}

#[test]
fn compression_density_sentinels_keep_base() {
    // A 0.0 density on either side means "no neighbourhood" (degenerate kNN), not
    // a real void — the shift must not fire on it.
    let base = [BASE, BASE];
    let out = compression_colors(&base, &[0.0, 5.0], &[2.0, 0.0], YOUNG, 1.0);
    assert_eq!(out[0], BASE);
    assert_eq!(out[1], BASE);
}

#[test]
fn compression_doubled_density_at_full_strength_is_the_exact_midpoint() {
    // ρ = 2ρ0, strength 1: t = 1 − ρ0/ρ = 1/2 → the exact midpoint mix.
    let out = compression_colors(&[BASE], &[4.0], &[2.0], YOUNG, 1.0);
    assert_rgb_close(out[0], mix(BASE, YOUNG, 0.5), 1e-6);
}

#[test]
fn compression_is_monotone_in_density_and_bounded_by_strength() {
    // Ratios 1, 2, 4, 8, 1000 at strength 0.6: the shift grows monotonically with
    // compression but saturates at 0.6 of the way to young — never past it.
    let strength = 0.6;
    let rho = [1.0, 2.0, 4.0, 8.0, 1000.0];
    let base = [BASE; 5];
    let out = compression_colors(&base, &rho, &[1.0; 5], YOUNG, strength);
    let cap = mix(BASE, YOUNG, strength);
    for c in 0..3 {
        let toward_young = (YOUNG[c] - BASE[c]).signum();
        for w in out.windows(2) {
            assert!(
                (w[1][c] - w[0][c]) * toward_young >= 0.0,
                "channel {c} not monotone: {out:?}"
            );
        }
        for col in &out {
            let (lo, hi) = (BASE[c].min(cap[c]), BASE[c].max(cap[c]));
            assert!(
                col[c] >= lo - 1e-6 && col[c] <= hi + 1e-6,
                "channel {c} exceeds the strength cap: {col:?} vs {cap:?}"
            );
        }
    }
}

#[test]
fn compression_strength_above_one_clamps_to_the_young_endpoint() {
    // strength 5 must behave as strength 1: the shift stays on the [base, young]
    // segment for any compression, however extreme.
    let out = compression_colors(&[BASE], &[1e12], &[1.0], YOUNG, 5.0);
    for c in 0..3 {
        let (lo, hi) = (BASE[c].min(YOUNG[c]), BASE[c].max(YOUNG[c]));
        assert!(out[0][c] >= lo && out[0][c] <= hi, "off segment: {out:?}");
    }
    assert_rgb_close(out[0], YOUNG, 1e-6); // ratio 1e12 ≈ saturated
}

#[test]
fn compression_colors_are_deterministic() {
    let base = [BASE, YOUNG, [0.3, 0.3, 0.3]];
    let rho = [5.0, 1.0, 3.0];
    let rho0 = [2.0, 2.0, 3.0];
    assert_eq!(
        compression_colors(&base, &rho, &rho0, YOUNG, 0.8),
        compression_colors(&base, &rho, &rho0, YOUNG, 0.8)
    );
}

// --------------------------------------------------------------------------
// Age-triggered star-formation tint (natal-ember-forge): fade toward `young`
// for recently-formed stars, keyed on the real `formation_time`.
// --------------------------------------------------------------------------

#[test]
fn age_strength_zero_is_the_identity() {
    // strength 0 ⇒ t = 0 for every particle ⇒ base color bit-for-bit, whatever
    // the ages — the same hard off-guarantee compression carries.
    let base = [BASE, [0.1, 0.9, 0.5]];
    let out = age_colors(&base, &[5.0, 4.5], 5.0, YOUNG, 0.0, 2.0);
    assert_eq!(out, base.to_vec());
}

#[test]
fn age_primordial_stars_keep_base_bit_exactly() {
    // A primordial star carries formation_time = −∞, so age = now − (−∞) = +∞,
    // exp(−∞) = +0.0, t = +0.0, and the two-product lerp returns base exactly —
    // NO is_infinite branch (the property the −∞ sentinel was chosen for). The
    // freshly-formed neighbour (age 0, strength 1) DOES shift, proving the map is
    // live, not a no-op.
    let base = [BASE, BASE];
    let out = age_colors(
        &base,
        &[State::PRIMORDIAL, 5.0],
        5.0,
        YOUNG,
        1.0,
        2.0,
    );
    assert_eq!(out[0], BASE, "primordial star must stay its base color exactly");
    assert_ne!(out[1], BASE, "a freshly-formed star must shift");
}

#[test]
fn age_freshly_formed_at_full_strength_is_the_young_endpoint() {
    // age = 0 (formation_time == now), strength 1: t = 1·exp(0) = 1 → the young
    // endpoint bit-exactly (two-product lerp at t = 1).
    let out = age_colors(&[BASE], &[5.0], 5.0, YOUNG, 1.0, 3.0);
    assert_eq!(out[0], YOUNG);
}

#[test]
fn age_one_efold_is_the_reference_mix() {
    // age = tau (one fade timescale), strength 1: t = exp(−1) ≈ 0.3679 — the
    // reference decay point, hand-derived, not read back from the function.
    let tau = 2.0;
    let out = age_colors(&[BASE], &[3.0], 3.0 + tau, YOUNG, 1.0, tau);
    let want = mix(BASE, YOUNG, (-1.0f64).exp() as f32);
    assert_rgb_close(out[0], want, 1e-6);
}

#[test]
fn age_is_monotone_in_age_and_bounded_by_strength() {
    // Ages 0 < 1 < 2 < 4 < 20 at strength 0.6, tau 2: the tint fades
    // monotonically from the strength cap back toward base, never past either.
    let strength = 0.6;
    let tau = 2.0;
    let now = 10.0;
    let ages = [0.0, 1.0, 2.0, 4.0, 20.0];
    let ft: Vec<f64> = ages.iter().map(|a| now - a).collect();
    let base = vec![BASE; ages.len()];
    let out = age_colors(&base, &ft, now, YOUNG, strength, tau);
    let cap = mix(BASE, YOUNG, strength); // t at age 0
    for c in 0..3 {
        let toward_young = (YOUNG[c] - BASE[c]).signum();
        // Older (later window entries) ⇒ closer to base ⇒ moves AWAY from young.
        for w in out.windows(2) {
            assert!(
                (w[1][c] - w[0][c]) * toward_young <= 0.0,
                "channel {c} not monotone in age: {out:?}"
            );
        }
        let (lo, hi) = (BASE[c].min(cap[c]), BASE[c].max(cap[c]));
        for col in &out {
            assert!(
                col[c] >= lo - 1e-6 && col[c] <= hi + 1e-6,
                "channel {c} exceeds the strength cap: {col:?} vs {cap:?}"
            );
        }
    }
}

#[test]
fn age_strength_above_one_clamps_to_the_young_endpoint() {
    // strength 5 must behave as strength 1: a freshly-formed star lands on the
    // young endpoint, not past it.
    let out = age_colors(&[BASE], &[7.0], 7.0, YOUNG, 5.0, 1.0);
    assert_eq!(out[0], YOUNG);
}

#[test]
fn age_colors_are_deterministic() {
    let base = [BASE, YOUNG, [0.3, 0.3, 0.3]];
    let ft = [State::PRIMORDIAL, 4.0, 4.7];
    assert_eq!(
        age_colors(&base, &ft, 5.0, YOUNG, 0.8, 1.5),
        age_colors(&base, &ft, 5.0, YOUNG, 0.8, 1.5)
    );
}
