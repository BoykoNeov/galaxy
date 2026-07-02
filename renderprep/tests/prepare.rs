//! Snapshot → frame-data mapping (DESIGN.md M3, MVP: progenitor color + mass
//! brightness, pure map, no spatial tree).
//!
//! Expectations are hand-derived from the config, not read back from the function
//! under test: given a palette and a brightness-per-mass, the color of a particle
//! is exactly `palette[progenitor]` and its brightness is exactly
//! `brightness_per_mass * mass`.

use galaxy_core::{DVec3, ParticleId, Progenitor, State};
use galaxy_renderprep::{prepare, DensityColoring, PrepConfig};

/// Two particles from progenitor 0, one from progenitor 1, with distinct masses.
fn sample_state() -> State {
    State {
        pos: vec![
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(0.0, 0.0, -3.0),
        ],
        vel: vec![DVec3::ZERO; 3],
        mass: vec![2.0, 4.0, 1.0],
        id: vec![ParticleId(0), ParticleId(1), ParticleId(2)],
        progenitor: vec![Progenitor(0), Progenitor(1), Progenitor(0)],
        time: 5.0,
        a: 1.0,
    }
}

fn sample_config() -> PrepConfig {
    PrepConfig {
        palette: vec![[1.0, 0.4, 0.2], [0.2, 0.5, 1.0]],
        brightness_per_mass: 3.0,
        size: 0.5,
        density: None,
    }
}

#[test]
fn preserves_particle_order_and_count() {
    let state = sample_state();
    let data = prepare(&state, &sample_config());
    assert_eq!(data.len(), state.len());
    // Position column is the f32 projection of the physics positions, in order.
    assert_eq!(data.pos[0].as_dvec3(), state.pos[0]);
    assert_eq!(data.pos[2].as_dvec3(), state.pos[2]);
}

#[test]
fn progenitor_selects_palette_color() {
    let state = sample_state();
    let cfg = sample_config();
    let data = prepare(&state, &cfg);
    // Particles 0 and 2 are progenitor 0 → palette[0]; particle 1 → palette[1].
    assert_eq!(data.color[0], cfg.palette[0]);
    assert_eq!(data.color[1], cfg.palette[1]);
    assert_eq!(data.color[2], cfg.palette[0]);
}

#[test]
fn brightness_scales_linearly_with_mass() {
    let state = sample_state();
    let cfg = sample_config();
    let data = prepare(&state, &cfg);
    // brightness = brightness_per_mass * mass
    assert_eq!(data.brightness[0], 3.0 * 2.0);
    assert_eq!(data.brightness[1], 3.0 * 4.0);
    assert_eq!(data.brightness[2], 3.0 * 1.0);
}

#[test]
fn size_is_the_configured_constant() {
    let state = sample_state();
    let cfg = sample_config();
    let data = prepare(&state, &cfg);
    assert!(data.size.iter().all(|&s| s == cfg.size));
}

#[test]
fn out_of_range_progenitor_wraps_modulo_palette() {
    let mut state = sample_state();
    // progenitor 2 with a 2-entry palette wraps to palette[0].
    state.progenitor[1] = Progenitor(2);
    let cfg = sample_config();
    let data = prepare(&state, &cfg);
    assert_eq!(data.color[1], cfg.palette[0]);
}

// ---------------------------------------------------------------------------
// Density-aware pass (DESIGN M3): opt-in k-NN brightness boost. `None` keeps the
// pure map bit-for-bit; `Some` brightens dense neighbourhoods and never dims.
// ---------------------------------------------------------------------------

/// Six particles clumped tightly at the origin plus three far-flung sparse ones —
/// all equal mass / progenitor 0, so only *local density* distinguishes them. The
/// clump is a regular octahedron (vertices ±a on each axis): every vertex has four
/// nearest neighbours at a√2, so all six share the *same* k=2 NN distance and hence
/// the same (identical) density — each provably above the sparse-diluted mean, so
/// all six are boosted, none sits on the underdense side by geometric accident.
fn clustered_state() -> State {
    let a = 0.03;
    let mut pos = vec![
        DVec3::new(a, 0.0, 0.0),
        DVec3::new(-a, 0.0, 0.0),
        DVec3::new(0.0, a, 0.0),
        DVec3::new(0.0, -a, 0.0),
        DVec3::new(0.0, 0.0, a),
        DVec3::new(0.0, 0.0, -a),
    ];
    pos.push(DVec3::new(100.0, 0.0, 0.0));
    pos.push(DVec3::new(0.0, 100.0, 0.0));
    pos.push(DVec3::new(0.0, 0.0, 100.0));
    let n = pos.len();
    State {
        vel: vec![DVec3::ZERO; n],
        mass: vec![1.0; n],
        id: (0..n as u64).map(ParticleId).collect(),
        progenitor: vec![Progenitor(0); n],
        time: 0.0,
        a: 1.0,
        pos,
    }
}

#[test]
fn density_strength_zero_matches_none() {
    // strength 0 is the identity: the whole density path collapses to the pure map.
    let state = clustered_state();
    let base = PrepConfig {
        density: None,
        ..Default::default()
    };
    let warm = PrepConfig {
        density: Some(DensityColoring {
            k: 2,
            softening: 1e-6,
            strength: 0.0,
        }),
        ..Default::default()
    };
    let a = prepare(&state, &base);
    let b = prepare(&state, &warm);
    assert_eq!(a.brightness, b.brightness);
}

#[test]
fn density_brightens_dense_cluster_never_dims_sparse() {
    let state = clustered_state();
    let cfg = PrepConfig {
        brightness_per_mass: 1.0, // base brightness = mass = 1.0
        density: Some(DensityColoring {
            k: 2,
            softening: 1e-6,
            strength: 1.0,
        }),
        ..Default::default()
    };
    let data = prepare(&state, &cfg);
    // The six clumped particles (indices 0..6) sit above the mean density → boosted
    // strictly past the base 1.0; the three sparse ones (6..9) are underdense → left
    // at exactly the base (the boost never dims).
    for b in &data.brightness[0..6] {
        assert!(*b > 1.0, "clumped particle should be brightened, got {b}");
        assert!(*b <= 2.0, "boost is bounded by 1 + strength = 2, got {b}");
    }
    for b in &data.brightness[6..9] {
        assert_eq!(*b, 1.0, "sparse particle must keep full (base) brightness");
    }
}

#[test]
fn density_preserves_order_and_count() {
    let state = clustered_state();
    let cfg = PrepConfig {
        density: Some(DensityColoring {
            k: 2,
            softening: 1e-6,
            strength: 0.8,
        }),
        ..Default::default()
    };
    let data = prepare(&state, &cfg);
    assert_eq!(data.len(), state.len());
    for i in 0..state.len() {
        assert_eq!(data.pos[i].as_dvec3(), state.pos[i]);
    }
}

#[test]
fn density_tiny_state_does_not_panic() {
    // Fewer than k+1 particles → no k-th neighbour → density 0 → boost 1 (base).
    let state = State {
        pos: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 0.0)],
        vel: vec![DVec3::ZERO; 2],
        mass: vec![1.0, 1.0],
        id: vec![ParticleId(0), ParticleId(1)],
        progenitor: vec![Progenitor(0); 2],
        time: 0.0,
        a: 1.0,
    };
    let cfg = PrepConfig {
        brightness_per_mass: 1.0,
        density: Some(DensityColoring {
            k: 8, // > N-1, so the estimate is degenerate
            softening: 1e-6,
            strength: 1.0,
        }),
        ..Default::default()
    };
    let data = prepare(&state, &cfg);
    assert_eq!(data.brightness, vec![1.0, 1.0]);
}

#[test]
fn empty_palette_falls_back_to_white() {
    let state = sample_state();
    let cfg = PrepConfig {
        palette: vec![],
        brightness_per_mass: 1.0,
        size: 1.0,
        density: None,
    };
    let data = prepare(&state, &cfg);
    assert!(data.color.iter().all(|&c| c == [1.0, 1.0, 1.0]));
}
