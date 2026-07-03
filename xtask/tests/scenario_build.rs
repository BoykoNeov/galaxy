//! Gates for the M6f spec-driven scenario builder: the `scenario.toml` front-end
//! must reproduce what the hardcoded constructors built (same params, same seed
//! ⇒ same movie), and the spec→IC plumbing (fractions, orientations, counts,
//! cadence) must be exactly the documented mapping.
//!
//! Sampling gates run on shrunken particle counts — the relations under test
//! (bit-identical states, rigid-rotation equivalence) hold at any N.

use galaxy_core::DVec3;
use galaxy_ic::{DiskCollision, ExponentialDisk, Plummer, TruncatedNfw};
use galaxy_xtask::spec::{
    build_scenario, parse_scenario_toml, preset, DiskCounts, ModelSpec, Rig, ScenarioSpec,
};
use galaxy_xtask::{
    DENSITY_K, DENSITY_STRENGTH, FRAME_H, FRAME_W, G, PEAK_BRIGHTNESS, QUICK_H, QUICK_W,
    SIZE_MAX_FRAC, SIZE_MIN_FRAC, SUBFRAMES,
};

fn parse_preset(name: &str) -> ScenarioSpec {
    parse_scenario_toml(preset(name).expect("preset exists"))
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// Shrink a disk-model spec's QUICK counts so sampling gates stay fast.
fn shrink_quick(spec: &mut ScenarioSpec, c: DiskCounts) {
    match &mut spec.model {
        ModelSpec::DiskPlummer { counts, .. } => counts.quick = c,
        ModelSpec::DiskNfw { counts, .. } => counts.quick = c,
        other => panic!("not a disk model: {other:?}"),
    }
}

const TINY: DiskCounts = DiskCounts {
    halo1: 300,
    disk1: 250,
    halo2: 200,
    disk2: 150,
};

// --- the front-end reproduces the hardcoded construction ----------------------

#[test]
fn disk_preset_build_matches_direct_ic_construction() {
    // Independent hand-derived expectation: the pre-M6f hardcoded `disk`
    // constructor, inlined here with its original constants.
    let galaxy1 = ExponentialDisk::new(0.15, 0.5, 0.1 * 0.5, 4.0 * 0.5, Plummer::new(G, 1.0, 1.0))
        .with_toomre_q(1.5);
    let galaxy2 =
        ExponentialDisk::new(0.1, 0.45, 0.1 * 0.45, 4.0 * 0.45, Plummer::new(G, 0.7, 0.9))
            .with_toomre_q(1.5);
    let collision = DiskCollision::new(galaxy1, galaxy2, 1.0, 1.5, 8.0);
    let expect = collision.sample(TINY.halo1, TINY.disk1, TINY.halo2, TINY.disk2, 0x00C0_FFEE);

    let mut spec = parse_preset("disk");
    shrink_quick(&mut spec, TINY);
    let s = build_scenario(&spec, true);

    // Bit-identical realization: same params, same seed ⇒ same movie.
    assert_eq!(s.state, expect);

    // The prep look is the documented mapping of the spec's look + the shared
    // pipeline constants (density features ON, keyed to the scenario's ε).
    assert_eq!(
        s.prep.palette,
        vec![
            [0.05, 0.035, 0.025],
            [1.0, 0.5, 0.25],
            [0.025, 0.035, 0.05],
            [0.35, 0.6, 1.0]
        ]
    );
    assert_eq!(s.prep.size, 0.12);
    let disk_particle_mass = 0.15 / TINY.disk1 as f64;
    assert_eq!(
        s.prep.brightness_per_mass,
        PEAK_BRIGHTNESS / disk_particle_mass as f32
    );
    let density = s.prep.density.expect("density boost is ON (M6a)");
    assert_eq!(density.k, DENSITY_K);
    assert_eq!(density.softening, 0.05);
    assert_eq!(density.strength, DENSITY_STRENGTH);
    let sized = s.prep.size_by_density.expect("size-by-density is ON (M6e)");
    assert_eq!(sized.k, DENSITY_K);
    assert_eq!(sized.softening, 0.05);
    assert_eq!(sized.min_frac, SIZE_MIN_FRAC);
    assert_eq!(sized.max_frac, SIZE_MAX_FRAC);

    // Sim/timing/framing knobs land verbatim.
    assert_eq!(s.eps, 0.05);
    assert_eq!(s.dt, 0.02);
    assert_eq!(s.n_steps, 1500);
    assert_eq!(s.snapshot_every, 25);
    assert_eq!(s.subframes, SUBFRAMES);
    assert_eq!(s.seed, 0x00C0_FFEE);
    assert_eq!(s.frame_percentile, 0.98);
    assert_eq!(s.rig, Rig::Static);
    assert_eq!(s.sf_progenitors, vec![1, 3]);
    assert_eq!(s.ramp.len(), 4);
    assert_eq!(s.ramp[1], ([1.0, 0.35, 0.1], [0.55, 0.75, 1.0]));
}

#[test]
fn build_is_deterministic() {
    let mut spec = parse_preset("cuspy");
    shrink_quick(&mut spec, TINY);
    let a = build_scenario(&spec, true);
    let b = build_scenario(&spec, true);
    assert_eq!(a.state, b.state);
}

// --- quick vs full -------------------------------------------------------------

#[test]
fn quick_build_honours_counts_cadence_and_frame_size() {
    // dm carries a QUICK-specific snapshot cadence — the one preset that
    // exercises `snapshot_every_quick`.
    let spec = parse_preset("dm");
    let s = build_scenario(&spec, true);
    assert_eq!(s.state.len(), 2000 + 1000);
    assert_eq!(s.snapshot_every, 400);
    assert_eq!((s.width, s.height), (QUICK_W, QUICK_H));

    // Equal particle mass across both halos (counts split 2:1 like the masses),
    // and the brightness weighting derived from it — the dm movie's look.
    let g1 = TruncatedNfw::new(galaxy_ic::Nfw::new(G, 1.0, 1.0, 10.0), 3.0);
    let particle_mass = g1.total_mass() / 2000.0;
    assert_eq!(
        s.prep.brightness_per_mass,
        PEAK_BRIGHTNESS / particle_mass as f32
    );
    assert_eq!(
        s.rig,
        Rig::OrbitTilt {
            azimuth_deg: (-90.0, 90.0),
            tilt_deg: (60.0, 60.0),
            window: 6,
        }
    );
}

#[test]
fn full_build_uses_full_counts_and_frame_size() {
    let spec = parse_preset("disk");
    let s = build_scenario(&spec, false);
    assert_eq!(s.state.len(), 5000 + 5000 + 3500 + 3500);
    assert_eq!(s.snapshot_every, 25); // no quick cadence override on disk
    assert_eq!((s.width, s.height), (FRAME_W, FRAME_H));
}

// --- the zoo's one new physics door: orientations through the front-end --------

#[test]
fn retro_build_is_cuspy_rigidly_rotated_pi_about_x() {
    // Retrograde = the same realization with each galaxy's body frame rotated by
    // π about the line of nodes (+x): relative to its own COM, every particle of
    // `retro` must be the matching `cuspy` particle with (dx, dy, dz) ↦
    // (dx, −dy, −dz) — positions AND velocities. This pins the whole
    // orientation plumbing (toml → Orientation → place_galaxy) with zero
    // tolerance for a sign convention slip.
    let mut cuspy = parse_preset("cuspy");
    let mut retro = parse_preset("retro");
    shrink_quick(&mut cuspy, TINY);
    shrink_quick(&mut retro, TINY);
    let sc = build_scenario(&cuspy, true).state;
    let sr = build_scenario(&retro, true).state;
    assert_eq!(sc.len(), sr.len());

    // Per-galaxy mass-weighted COMs must agree (the orbit is identical; a rigid
    // rotation of a zero-mean body frame cannot move its COM).
    for galaxy_progs in [[0u16, 1u16], [2, 3]] {
        let idx: Vec<usize> = (0..sc.len())
            .filter(|&i| galaxy_progs.contains(&sc.progenitor[i].0))
            .collect();
        let com = |pts: &[DVec3], st: &galaxy_core::State| -> DVec3 {
            let m: f64 = idx.iter().map(|&i| st.mass[i]).sum();
            idx.iter().map(|&i| pts[i] * st.mass[i]).sum::<DVec3>() / m
        };
        let (pc, vc) = (com(&sc.pos, &sc), com(&sc.vel, &sc));
        let (pr, vr) = (com(&sr.pos, &sr), com(&sr.vel, &sr));
        assert!((pc - pr).length() < 1e-9, "COM pos moved: {pc} vs {pr}");
        assert!((vc - vr).length() < 1e-9, "COM vel moved: {vc} vs {vr}");

        for &i in &idx {
            for (c, r, what) in [
                (sc.pos[i] - pc, sr.pos[i] - pr, "pos"),
                (sc.vel[i] - vc, sr.vel[i] - vr, "vel"),
            ] {
                let expect = DVec3::new(c.x, -c.y, -c.z);
                assert!(
                    (r - expect).length() < 1e-9,
                    "particle {i} {what}: {r} vs Rx(π)·{c}"
                );
            }
        }
    }
}
