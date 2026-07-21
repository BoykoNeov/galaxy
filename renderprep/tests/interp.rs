//! Hermite temporal-upsampling gates (DESIGN.md M6c).
//!
//! Cubic Hermite interpolation between adjacent snapshots is what turns the
//! ~61-snapshot flipbook into a smooth 60 fps movie, so the gates are about
//! *trusting the in-betweens*: endpoints reproduced bit-exact, exactness on the
//! polynomial class the method claims (cubics), C¹ continuity at the joins, and
//! a two-body Kepler oracle with a tolerance derived from the method's O(Δt⁴)
//! local error — never from the function's own output.

use galaxy_core::{DVec3, ParticleId, State};
use galaxy_renderprep::{
    prepare, subframe, FrameData, GasSplats, HermiteSpan, InterpError, PrepConfig,
};
use glam::Vec3;

/// Build a snapshot at `time` from explicit phase-space columns (unit masses,
/// sequential ids, single progenitor — identity is what the interp checks).
fn snap(time: f64, pos: Vec<DVec3>, vel: Vec<DVec3>) -> State {
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    s.time = time;
    s
}

/// Assert two f64 agree to an absolute tolerance, with context.
fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!((a - b).abs() <= tol, "{what}: {a} vs {b} (tol {tol})");
}

/// A pair of generic, non-degenerate snapshots (nothing symmetric, nothing zero).
fn generic_pair() -> (State, State) {
    let s0 = snap(
        0.7,
        vec![DVec3::new(1.3, -0.2, 2.1), DVec3::new(-3.4, 0.9, 0.4)],
        vec![DVec3::new(0.3, 1.1, -0.6), DVec3::new(-0.2, -0.8, 0.5)],
    );
    let s1 = snap(
        1.9,
        vec![DVec3::new(1.8, 0.9, 1.2), DVec3::new(-3.9, -0.3, 1.1)],
        vec![DVec3::new(0.5, 0.7, -1.2), DVec3::new(-0.4, -1.1, 0.9)],
    );
    (s0, s1)
}

// --------------------------------------------------------------------------
// Endpoint reproduction (bit-exact) and C¹ continuity at the joins
// --------------------------------------------------------------------------

#[test]
fn u0_and_u1_reproduce_the_endpoints_bit_exact() {
    let (s0, s1) = generic_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    let (p, v) = span.sample(0.0);
    assert_eq!(p, s0.pos, "u=0 positions must be s0.pos bit-for-bit");
    assert_eq!(v, s0.vel, "u=0 velocities must be s0.vel bit-for-bit");

    let (p, v) = span.sample(1.0);
    assert_eq!(p, s1.pos, "u=1 positions must be s1.pos bit-for-bit");
    assert_eq!(v, s1.vel, "u=1 velocities must be s1.vel bit-for-bit");
}

#[test]
fn c1_continuity_across_a_snapshot_boundary() {
    // Three snapshots of one particle on a generic curved path. The interpolant
    // on [s0,s1] and the one on [s1,s2] must agree at the join in BOTH position
    // and velocity (C¹) — bit-exact, because both reproduce s1's phase space.
    let s0 = snap(
        0.0,
        vec![DVec3::new(0.1, 0.2, 0.3)],
        vec![DVec3::new(1.0, -0.5, 0.2)],
    );
    let s1 = snap(
        0.8,
        vec![DVec3::new(0.9, -0.1, 0.5)],
        vec![DVec3::new(0.7, 0.4, -0.3)],
    );
    let s2 = snap(
        1.5,
        vec![DVec3::new(1.4, 0.3, 0.1)],
        vec![DVec3::new(0.2, 0.6, -0.8)],
    );

    let left = HermiteSpan::new(&s0, &s1).unwrap();
    let right = HermiteSpan::new(&s1, &s2).unwrap();
    assert_eq!(left.sample(1.0), right.sample(0.0));
}

// --------------------------------------------------------------------------
// Exactness on the method's polynomial class
// --------------------------------------------------------------------------

#[test]
fn constant_velocity_motion_is_exact() {
    // p(t) = p0 + v·t is degree 1 ≪ 3, so the cubic must reproduce it (and its
    // constant derivative) to rounding.
    let p0 = DVec3::new(2.0, -1.0, 0.5);
    let v = DVec3::new(-0.3, 0.8, 1.7);
    let (t0, t1) = (1.0, 3.5);
    let s0 = snap(t0, vec![p0 + v * t0], vec![v]);
    let s1 = snap(t1, vec![p0 + v * t1], vec![v]);
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    for &u in &[0.1, 0.25, 0.5, 0.75, 0.9] {
        let t = t0 + u * (t1 - t0);
        let (p, vel) = span.sample(u);
        let expect = p0 + v * t;
        for c in 0..3 {
            close(p[0][c], expect[c], 1e-13, "linear position");
            close(vel[0][c], v[c], 1e-13, "linear velocity");
        }
    }
}

#[test]
fn cubic_trajectories_are_reproduced_exactly() {
    // Cubic Hermite is exact on cubics: p(t) = a + b·t + c·t² + d·t³ sampled at
    // the two endpoints (with its analytic derivative) must be reproduced at
    // every interior u to rounding (~1e-12 relative — a handful of f64 ops).
    let a = DVec3::new(0.4, -1.2, 2.0);
    let b = DVec3::new(1.1, 0.3, -0.7);
    let c = DVec3::new(-0.6, 0.9, 0.2);
    let d = DVec3::new(0.25, -0.4, 0.8);
    let p = |t: f64| a + b * t + c * (t * t) + d * (t * t * t);
    let dp = |t: f64| b + c * (2.0 * t) + d * (3.0 * t * t);

    let (t0, t1) = (0.3, 1.7);
    let s0 = snap(t0, vec![p(t0)], vec![dp(t0)]);
    let s1 = snap(t1, vec![p(t1)], vec![dp(t1)]);
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    for i in 1..10 {
        let u = i as f64 / 10.0;
        let t = t0 + u * (t1 - t0);
        let (pos, vel) = span.sample(u);
        for comp in 0..3 {
            close(pos[0][comp], p(t)[comp], 1e-12, "cubic position");
            close(vel[0][comp], dp(t)[comp], 1e-12, "cubic velocity");
        }
    }
}

// --------------------------------------------------------------------------
// Kepler oracles: analytic two-body orbits, tolerance from the method's order
// --------------------------------------------------------------------------

#[test]
fn kepler_circular_orbit_oracle() {
    // Circular two-body orbit, GM = R = ω = 1: p(t) = (cos t, sin t, 0),
    // v(t) = (-sin t, cos t, 0). Snapshot spacing Δt = 0.5 (~8% of the period —
    // the movie's coarse cadence). Cubic Hermite local error is bounded by
    //   |p - H| ≤ max|p⁗| · Δt⁴ / 384       (standard interpolation bound)
    // and every component's 4th derivative is |cos/sin| ≤ 1, so
    //   tol_pos = 0.5⁴/384 ≈ 1.63e-4  (×1.05 fp slack).
    // The derivative error bound is |p' - H'| ≤ max|p⁗| · Δt³ / 72 (conservative
    // form of the standard h³ bound) ⇒ tol_vel = 0.5³/72 ≈ 1.74e-3.
    let dt = 0.5;
    let pos_at = |t: f64| DVec3::new(t.cos(), t.sin(), 0.0);
    let vel_at = |t: f64| DVec3::new(-t.sin(), t.cos(), 0.0);

    let t0 = 0.3; // generic phase — no axis alignment
    let s0 = snap(t0, vec![pos_at(t0)], vec![vel_at(t0)]);
    let s1 = snap(t0 + dt, vec![pos_at(t0 + dt)], vec![vel_at(t0 + dt)]);
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    let tol_pos = dt.powi(4) / 384.0 * 1.05;
    let tol_vel = dt.powi(3) / 72.0;
    for i in 1..16 {
        let u = i as f64 / 16.0;
        let t = t0 + u * dt;
        let (p, v) = span.sample(u);
        for c in 0..3 {
            close(p[0][c], pos_at(t)[c], tol_pos, "Kepler circular position");
            close(v[0][c], vel_at(t)[c], tol_vel, "Kepler circular velocity");
        }
    }
}

#[test]
fn kepler_eccentric_orbit_oracle() {
    // Eccentric Kepler orbit (a = GM = 1, e = 0.5, mean motion n = 1), solved
    // analytically in the TEST via Kepler's equation (Newton on E - e·sinE = M)
    // — fully independent of the code under test. The span straddles perihelion,
    // where speed and curvature peak (the regime a circular orbit never enters).
    //
    // Tolerance: same |p - H| ≤ max|p⁗|·Δt⁴/384 bound, with max|p⁗| taken from
    // the ANALYTIC trajectory by a 5-point central finite difference scanned
    // over the span (×1.5 slack for the fd estimate itself).
    let e = 0.5f64;
    let b = (1.0 - e * e).sqrt();
    let ecc_anomaly = |m: f64| {
        let mut big_e = m;
        for _ in 0..30 {
            big_e -= (big_e - e * big_e.sin() - m) / (1.0 - e * big_e.cos());
        }
        big_e
    };
    let pos_at = |t: f64| {
        let big_e = ecc_anomaly(t);
        DVec3::new(big_e.cos() - e, b * big_e.sin(), 0.0)
    };
    let vel_at = |t: f64| {
        let big_e = ecc_anomaly(t);
        let e_dot = 1.0 / (1.0 - e * big_e.cos());
        DVec3::new(-big_e.sin() * e_dot, b * big_e.cos() * e_dot, 0.0)
    };

    let (t0, dt) = (-0.25, 0.5); // straddles perihelion at t = 0
    let s0 = snap(t0, vec![pos_at(t0)], vec![vel_at(t0)]);
    let s1 = snap(t0 + dt, vec![pos_at(t0 + dt)], vec![vel_at(t0 + dt)]);
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    // max|p⁗| over the span from the analytic solution (5-point stencil).
    let delta = 1e-2;
    let mut m4 = 0.0f64;
    for i in 0..=50 {
        let t = t0 + dt * i as f64 / 50.0;
        let d4 = (pos_at(t - 2.0 * delta) - pos_at(t - delta) * 4.0 + pos_at(t) * 6.0
            - pos_at(t + delta) * 4.0
            + pos_at(t + 2.0 * delta))
            / delta.powi(4);
        m4 = m4.max(d4.abs().max_element());
    }
    let tol = m4 * dt.powi(4) / 384.0 * 1.5;

    for i in 1..16 {
        let u = i as f64 / 16.0;
        let t = t0 + u * dt;
        let (p, _) = span.sample(u);
        let err = (p[0] - pos_at(t)).abs().max_element();
        assert!(
            err <= tol,
            "Kepler eccentric position: err {err} > tol {tol}"
        );
    }
}

// --------------------------------------------------------------------------
// Defensive gates: identity and time must be consistent before interpolating
// --------------------------------------------------------------------------

#[test]
fn mismatched_ids_are_rejected() {
    // Same particles, permuted order in s1 — silently interpolating would
    // scramble every in-between frame. Must be an IdMismatch error, not a panic.
    let (s0, mut s1) = generic_pair();
    s1.pos.swap(0, 1);
    s1.vel.swap(0, 1);
    s1.id.swap(0, 1);
    assert!(matches!(
        HermiteSpan::new(&s0, &s1),
        Err(InterpError::IdMismatch { index: 0, .. })
    ));
}

#[test]
fn mismatched_lengths_are_rejected() {
    let (s0, mut s1) = generic_pair();
    s1.pos.pop();
    s1.vel.pop();
    s1.mass.pop();
    s1.id.pop();
    s1.progenitor.pop();
    assert!(matches!(
        HermiteSpan::new(&s0, &s1),
        Err(InterpError::LengthMismatch { n0: 2, n1: 1 })
    ));
}

#[test]
fn non_increasing_time_is_rejected() {
    let (s0, mut s1) = generic_pair();
    s1.time = s0.time; // Δt = 0 — the cubic's coefficients divide by Δt
    assert!(matches!(
        HermiteSpan::new(&s0, &s1),
        Err(InterpError::NonIncreasingTime { .. })
    ));
}

// --------------------------------------------------------------------------
// Subframe assembly: Hermite positions + lerped endpoint attributes
// --------------------------------------------------------------------------

#[test]
fn subframe_endpoints_reproduce_the_prepared_frames_bit_exact() {
    let (mut s0, mut s1) = generic_pair();
    // Distinct progenitors so color varies; prepare both endpoints fully.
    s0.progenitor[1].0 = 1;
    s1.progenitor[1].0 = 1;
    let cfg = PrepConfig::default();
    let f0 = prepare(&s0, &cfg);
    let f1 = prepare(&s1, &cfg);
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    assert_eq!(subframe(&span, &f0, &f1, 0.0), f0);
    assert_eq!(subframe(&span, &f0, &f1, 1.0), f1);
}

#[test]
fn subframe_attributes_lerp_linearly() {
    // Hand-built endpoint frames with different brightness/color/size: at u the
    // attributes must be the exact linear blend (1-u)·f0 + u·f1.
    let (s0, s1) = generic_pair();
    let f0 = FrameData {
        pos: s0.pos.iter().map(|p| p.as_vec3()).collect(),
        color: vec![[1.0, 0.0, 0.2], [0.0, 0.5, 1.0]],
        size: vec![0.1, 0.3],
        brightness: vec![2.0, 4.0],
    };
    let f1 = FrameData {
        pos: s1.pos.iter().map(|p| p.as_vec3()).collect(),
        color: vec![[0.0, 1.0, 0.6], [1.0, 0.5, 0.0]],
        size: vec![0.5, 0.1],
        brightness: vec![6.0, 1.0],
    };
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    let u = 0.25f64;
    let f = subframe(&span, &f0, &f1, u);
    let (w0, w1) = (0.75f32, 0.25f32);
    for i in 0..2 {
        for c in 0..3 {
            close(
                f.color[i][c] as f64,
                (w0 * f0.color[i][c] + w1 * f1.color[i][c]) as f64,
                1e-6,
                "color lerp",
            );
        }
        close(
            f.brightness[i] as f64,
            (w0 * f0.brightness[i] + w1 * f1.brightness[i]) as f64,
            1e-6,
            "brightness lerp",
        );
        close(
            f.size[i] as f64,
            (w0 * f0.size[i] + w1 * f1.size[i]) as f64,
            1e-6,
            "size lerp",
        );
    }
}

#[test]
fn subframe_positions_are_hermite_not_lerp() {
    // A curved trajectory where the cubic visibly departs from the chord: the
    // subframe's positions must match span.sample(u), not the endpoint lerp.
    let s0 = snap(
        0.0,
        vec![DVec3::new(1.0, 0.0, 0.0)],
        vec![DVec3::new(0.0, 2.0, 0.0)],
    );
    let s1 = snap(
        1.0,
        vec![DVec3::new(-1.0, 0.0, 0.0)],
        vec![DVec3::new(0.0, -2.0, 0.0)],
    );
    let cfg = PrepConfig::default();
    let f0 = prepare(&s0, &cfg);
    let f1 = prepare(&s1, &cfg);
    let span = HermiteSpan::new(&s0, &s1).unwrap();

    let u = 0.5;
    let f = subframe(&span, &f0, &f1, u);
    let (hermite_pos, _) = span.sample(u);
    assert_eq!(f.pos[0], hermite_pos[0].as_vec3());
    // And the cubic genuinely differs from the chord midpoint (0,0,0) here.
    assert!(
        (f.pos[0] - Vec3::ZERO).length() > 0.1,
        "test would not distinguish Hermite from lerp"
    );
}

#[test]
#[should_panic]
fn subframe_rejects_frames_of_the_wrong_length() {
    let (s0, s1) = generic_pair();
    let cfg = PrepConfig::default();
    let f0 = prepare(&s0, &cfg);
    let mut f1 = prepare(&s1, &cfg);
    f1.pos.pop();
    f1.color.pop();
    f1.size.pop();
    f1.brightness.pop();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    let _ = subframe(&span, &f0, &f1, 0.5);
}

// --------------------------------------------------------------------------
// Determinism
// --------------------------------------------------------------------------

#[test]
fn sampling_is_deterministic() {
    let (s0, s1) = generic_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    assert_eq!(span.sample(0.37), span.sample(0.37));
}

#[test]
fn ids_used_for_the_gate_are_the_particle_ids() {
    // Same length, same order, but one particle is a DIFFERENT particle (new id,
    // as after any future add/remove) — still a mismatch even though the arrays
    // line up positionally.
    let (s0, mut s1) = generic_pair();
    s1.id[1] = ParticleId(99);
    assert!(matches!(
        HermiteSpan::new(&s0, &s1),
        Err(InterpError::IdMismatch { index: 1, .. })
    ));
}

// --------------------------------------------------------------------------
// Species routing through subframes (M7d): prepared frames may hold only the
// collisionless rows (gas is routed to the volumetric grid). `subframe` must
// accept such filtered frames, pairing them with the span's stellar particles.
// --------------------------------------------------------------------------

/// A generic mixed-species snapshot pair: rows 0/2/3 collisionless, row 1 gas.
fn mixed_pair() -> (State, State) {
    use galaxy_core::Species;
    let (mut s0, mut s1) = (
        snap(
            0.4,
            vec![
                DVec3::new(1.3, -0.2, 2.1),
                DVec3::new(0.2, 0.6, -0.3),
                DVec3::new(-3.4, 0.9, 0.4),
                DVec3::new(2.0, 1.5, -1.0),
            ],
            vec![
                DVec3::new(0.3, 1.1, -0.6),
                DVec3::new(-0.1, 0.2, 0.4),
                DVec3::new(-0.2, -0.8, 0.5),
                DVec3::new(0.6, -0.4, 0.1),
            ],
        ),
        snap(
            1.6,
            vec![
                DVec3::new(1.8, 0.9, 1.2),
                DVec3::new(0.4, 0.3, 0.1),
                DVec3::new(-3.9, -0.3, 1.1),
                DVec3::new(2.5, 1.1, -0.6),
            ],
            vec![
                DVec3::new(0.5, 0.7, -1.2),
                DVec3::new(0.2, -0.3, 0.6),
                DVec3::new(-0.4, -1.1, 0.9),
                DVec3::new(0.3, -0.7, 0.5),
            ],
        ),
    );
    let kinds = vec![
        Species::Collisionless,
        Species::Gas,
        Species::Collisionless,
        Species::Collisionless,
    ];
    s0.kind = kinds.clone();
    s1.kind = kinds;
    (s0, s1)
}

#[test]
fn filtered_subframe_reproduces_the_prepared_endpoints_bit_exact() {
    let (s0, s1) = mixed_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    let cfg = PrepConfig::default(); // gas routed out
    let f0 = prepare(&s0, &cfg);
    let f1 = prepare(&s1, &cfg);
    assert_eq!(f0.len(), 3, "precondition: gas row filtered");

    let at0 = subframe(&span, &f0, &f1, 0.0);
    assert_eq!(at0, f0, "u=0 must reproduce the prepared f0 bit-exactly");
    let at1 = subframe(&span, &f0, &f1, 1.0);
    assert_eq!(at1, f1, "u=1 must reproduce the prepared f1 bit-exactly");
}

#[test]
fn filtered_subframe_positions_follow_the_stellar_hermite_rows() {
    // Mid-span, the filtered subframe's positions must be exactly the f32
    // projection of the Hermite sample at the STELLAR indices (0, 2, 3) —
    // pairing splat row k with hermite row k instead would scramble the gas
    // row into the stars.
    let (s0, s1) = mixed_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    let cfg = PrepConfig::default();
    let f0 = prepare(&s0, &cfg);
    let f1 = prepare(&s1, &cfg);

    let u = 0.375;
    let sub = subframe(&span, &f0, &f1, u);
    let (hpos, _) = span.sample(u);
    assert_eq!(sub.len(), 3);
    assert_eq!(sub.pos[0], hpos[0].as_vec3());
    assert_eq!(sub.pos[1], hpos[2].as_vec3());
    assert_eq!(sub.pos[2], hpos[3].as_vec3());
}

#[test]
fn full_length_frames_still_interpolate_all_rows() {
    // Debug mode (gas_as_splats) keeps gas in the splat list; subframe must
    // keep accepting full-length frames on a mixed-species span.
    let (s0, s1) = mixed_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    let cfg = PrepConfig {
        gas_splats: GasSplats::Visible,
        ..PrepConfig::default()
    };
    let f0 = prepare(&s0, &cfg);
    let f1 = prepare(&s1, &cfg);
    assert_eq!(f0.len(), 4);

    assert_eq!(subframe(&span, &f0, &f1, 0.0), f0);
    let u = 0.5;
    let sub = subframe(&span, &f0, &f1, u);
    let (hpos, _) = span.sample(u);
    for (i, hp) in hpos.iter().enumerate() {
        assert_eq!(sub.pos[i], hp.as_vec3());
    }
}

/// A snapshot pair straddling a star-formation event: particle 1 is `Gas` at
/// `s0` and `Collisionless` at `s1` (in-place conversion — same index, same id).
/// The star set GROWS across the span; only the fixed-length `GasSplats::Hidden`
/// prep keeps the two endpoint frames the same length for the subframe interp.
fn sf_pair() -> (State, State) {
    use galaxy_core::Species;
    let (mut s0, mut s1) = mixed_pair(); // particle 1 is Gas in both
    s0.kind[1] = Species::Gas;
    s1.kind[1] = Species::Collisionless; // it formed a star between the snapshots
    (s0, s1)
}

#[test]
fn subframe_over_a_star_formation_span_fades_the_newborn_in() {
    // The natal-ember-forge smooth-interp fix: SF grows the splat set between
    // snapshots, which the routed (n_star) frames cannot pair (5001 vs 5000).
    // `GasSplats::Hidden` prep keeps both endpoints at the full length N, so the
    // subframe unfiltered path applies. The just-formed star (row 1) fades in:
    // brightness 0 at s0 (still gas), real at s1 (now a star), monotone between.
    let (s0, s1) = sf_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    let cfg = PrepConfig {
        gas_splats: GasSplats::Hidden,
        ..PrepConfig::default()
    };
    let f0 = prepare(&s0, &cfg);
    let f1 = prepare(&s1, &cfg);
    // Fixed length N=4 at both ends — nothing filtered, no length mismatch.
    assert_eq!(f0.len(), 4);
    assert_eq!(f1.len(), 4);
    assert_eq!(f0.brightness[1], 0.0, "row 1 is gas at s0 → invisible splat");
    assert!(f1.brightness[1] > 0.0, "row 1 is a star at s1 → real splat");

    // Endpoint reproduction, and no panic on the growing-star span.
    assert_eq!(subframe(&span, &f0, &f1, 0.0), f0, "u=0 reproduces f0");
    assert_eq!(subframe(&span, &f0, &f1, 1.0), f1, "u=1 reproduces f1");

    // The newborn's splat brightness ramps 0 → real, monotone across the span.
    let b = |u: f64| subframe(&span, &f0, &f1, u).brightness[1];
    let (b0, b25, b50, b75, b1) = (b(0.0), b(0.25), b(0.5), b(0.75), b(1.0));
    assert_eq!(b0, 0.0);
    assert_eq!(b1, f1.brightness[1]);
    assert!(
        b0 < b25 && b25 < b50 && b50 < b75 && b75 < b1,
        "monotone fade-in: {b0} {b25} {b50} {b75} {b1}"
    );
}

#[test]
#[should_panic(expected = "prepared frame")]
fn subframe_rejects_frames_of_impossible_length() {
    // Neither the full state length (4) nor the stellar count (3): a caller
    // contract violation, and pairing rows would silently scramble the movie.
    let (s0, s1) = mixed_pair();
    let span = HermiteSpan::new(&s0, &s1).unwrap();
    let bogus = FrameData {
        pos: vec![Vec3::ZERO; 2],
        color: vec![[1.0; 3]; 2],
        size: vec![1.0; 2],
        brightness: vec![1.0; 2],
    };
    let _ = subframe(&span, &bogus, &bogus, 0.5);
}
