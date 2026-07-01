//! Snapshot → frame-data mapping (DESIGN.md M3, MVP: progenitor color + mass
//! brightness, pure map, no spatial tree).
//!
//! Expectations are hand-derived from the config, not read back from the function
//! under test: given a palette and a brightness-per-mass, the color of a particle
//! is exactly `palette[progenitor]` and its brightness is exactly
//! `brightness_per_mass * mass`.

use galaxy_core::{DVec3, ParticleId, Progenitor, State};
use galaxy_renderprep::{prepare, PrepConfig};

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

#[test]
fn empty_palette_falls_back_to_white() {
    let state = sample_state();
    let cfg = PrepConfig {
        palette: vec![],
        brightness_per_mass: 1.0,
        size: 1.0,
    };
    let data = prepare(&state, &cfg);
    assert!(data.color.iter().all(|&c| c == [1.0, 1.0, 1.0]));
}
