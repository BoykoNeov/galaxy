//! Snapshot → frame-data mapping (DESIGN.md M3, MVP: progenitor color + mass
//! brightness, pure map, no spatial tree).
//!
//! Expectations are hand-derived from the config, not read back from the function
//! under test: given a palette and a brightness-per-mass, the color of a particle
//! is exactly `palette[progenitor]` and its brightness is exactly
//! `brightness_per_mass * mass`.

use galaxy_core::{DVec3, ParticleId, Progenitor, Species, State};
use galaxy_renderprep::{
    knn_density, prepare, ColorMode, CompressionHue, DensityColoring, DispersionColoring,
    PrepConfig, SigmaReference, SizeByDensity,
};

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
        kind: vec![Species::Collisionless; 3],
        u: vec![0.0; 3],
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
        ..Default::default()
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
        kind: vec![Species::Collisionless; n],
        u: vec![0.0; n],
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
    // Order preserved: column i is the f32 projection of state particle i. Compare
    // against the projection (not the exact f64 — the clump coords aren't f32-exact).
    for i in 0..state.len() {
        assert_eq!(data.pos[i], state.pos[i].as_vec3());
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
        kind: vec![Species::Collisionless; 2],
        u: vec![0.0; 2],
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
        ..Default::default()
    };
    let data = prepare(&state, &cfg);
    assert!(data.color.iter().all(|&c| c == [1.0, 1.0, 1.0]));
}

// ---------------------------------------------------------------------------
// Coloring modes v2 (DESIGN M6e): ColorMode, size-by-density, compression hue.
// The default config must stay the pre-M6e pure map bit-for-bit (the tests above
// pin that); these gate the new opt-in paths through `prepare`.
// ---------------------------------------------------------------------------

#[test]
fn default_config_is_the_pre_m6e_map() {
    // The bit-compat contract: every new knob defaults OFF.
    let d = PrepConfig::default();
    assert_eq!(d.color, ColorMode::Progenitor);
    assert_eq!(d.size_by_density, None);
    assert_eq!(d.compression, None);
}

#[test]
fn frozen_colors_are_bit_stable_across_frames() {
    // The frozen-at-t0 property: whatever the particles do later (positions, time),
    // Frozen colors come back exactly as given, keyed by index — palette ignored.
    let frozen = vec![[0.9, 0.1, 0.3], [0.2, 0.8, 0.4], [0.5, 0.5, 0.5]];
    let cfg = PrepConfig {
        color: ColorMode::Frozen(frozen.clone()),
        ..sample_config()
    };
    let early = sample_state();
    let mut late = sample_state();
    for p in &mut late.pos {
        *p = *p * 7.0 + DVec3::new(3.0, -2.0, 11.0);
    }
    late.time = 99.0;
    assert_eq!(prepare(&early, &cfg).color, frozen);
    assert_eq!(prepare(&late, &cfg).color, frozen);
}

/// Two spatially separated triplets with k=2 neighbourhoods internal to each:
/// clump A (indices 0..3) is dynamically cold (identical velocities), clump B
/// (3..6) has velocity spread. One progenitor, equal masses.
fn two_temperature_state() -> State {
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(0.1, 0.0, 0.0),
        DVec3::new(0.0, 0.1, 0.0),
        DVec3::new(100.0, 0.0, 0.0),
        DVec3::new(100.1, 0.0, 0.0),
        DVec3::new(100.0, 0.1, 0.0),
    ];
    let vel = vec![
        DVec3::new(1.0, 2.0, 3.0),
        DVec3::new(1.0, 2.0, 3.0),
        DVec3::new(1.0, 2.0, 3.0),
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(6.0, 0.0, 0.0),
        DVec3::new(3.0, 3.0, 0.0),
    ];
    State {
        mass: vec![1.0; 6],
        id: (0..6).map(ParticleId).collect(),
        progenitor: vec![Progenitor(0); 6],
        kind: vec![Species::Collisionless; 6],
        u: vec![0.0; 6],
        time: 0.0,
        a: 1.0,
        pos,
        vel,
    }
}

#[test]
fn dispersion_mode_colors_cold_clump_cold_and_hot_clump_hotter() {
    let cold = [0.1, 0.2, 0.9];
    let hot = [0.9, 0.8, 0.1];
    let cfg = PrepConfig {
        color: ColorMode::Dispersion(DispersionColoring {
            k: 2,
            softening: 1e-9,
            cold,
            hot,
            luminous: u64::MAX, // single progenitor, all luminous — pre-mask behavior
            reference: SigmaReference::Full,
        }),
        ..sample_config()
    };
    let data = prepare(&two_temperature_state(), &cfg);
    for i in 0..3 {
        assert_eq!(data.color[i], cold, "cold clump must be exactly cold");
    }
    for i in 3..6 {
        assert_ne!(data.color[i], cold, "hot clump must move off the cold end");
        for c in 0..3 {
            let toward_hot = (hot[c] - cold[c]).signum();
            assert!(
                (data.color[i][c] - cold[c]) * toward_hot > 0.0,
                "particle {i} channel {c} must move toward hot"
            );
            let (lo, hi) = (cold[c].min(hot[c]), cold[c].max(hot[c]));
            assert!(data.color[i][c] >= lo && data.color[i][c] <= hi);
        }
    }
}

/// Two far-apart clumps with *identical* internal geometry and velocities — so
/// their σ_v is identical — but different progenitors: clump A = progenitor 1
/// (the luminous disk), clump B = progenitor 0 (the dark-matter halo). Any color
/// difference between them can only come from the luminous mask, not from σ.
fn mixed_progenitor_state() -> State {
    let clump = |c: DVec3| {
        vec![
            c + DVec3::new(0.5, 0.0, 0.0),
            c + DVec3::new(-0.5, 0.0, 0.0),
            c + DVec3::new(0.0, 0.5, 0.0),
        ]
    };
    let pos = [clump(DVec3::ZERO), clump(DVec3::new(100.0, 0.0, 0.0))].concat();
    // The same non-degenerate velocity spread in both clumps ⇒ both are "hot".
    let hot_vel = || {
        vec![
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
        ]
    };
    State {
        mass: vec![1.0; 6],
        id: (0..6).map(ParticleId).collect(),
        progenitor: vec![
            Progenitor(1),
            Progenitor(1),
            Progenitor(1),
            Progenitor(0),
            Progenitor(0),
            Progenitor(0),
        ],
        kind: vec![Species::Collisionless; 6],
        u: vec![0.0; 6],
        time: 0.0,
        a: 1.0,
        pos,
        vel: [hot_vel(), hot_vel()].concat(),
    }
}

#[test]
fn dispersion_mask_keeps_non_luminous_progenitors_on_the_palette() {
    // The luminous mask must OVERRIDE σ: the halo clump is dynamically hot, but
    // because progenitor 0 is not luminous it keeps its dim palette color exactly,
    // while the equally-hot luminous disk clump ramps toward hot.
    let cold = [0.1, 0.2, 0.9];
    let hot = [0.9, 0.8, 0.1];
    let halo = [0.05, 0.04, 0.03]; // palette[0] — the dim halo compensation color
    let cfg = PrepConfig {
        palette: vec![halo, [1.0, 0.5, 0.25]],
        color: ColorMode::Dispersion(DispersionColoring {
            k: 2,
            softening: 1e-9,
            cold,
            hot,
            luminous: 1u64 << 1, // only progenitor 1 (the disk) gets the ramp
            reference: SigmaReference::Full,
        }),
        ..Default::default()
    };
    let data = prepare(&mixed_progenitor_state(), &cfg);
    // Halo (progenitor 0, non-luminous): exactly palette[0] despite its hot σ.
    for i in 3..6 {
        assert_eq!(
            data.color[i], halo,
            "masked halo particle {i} must keep palette[0], not the σ_v ramp"
        );
    }
    // Disk (progenitor 1, luminous): σ_v ramp applied — off cold, toward hot.
    for i in 0..3 {
        assert_ne!(
            data.color[i], halo,
            "luminous particle {i} must not be halo"
        );
        assert_ne!(
            data.color[i], cold,
            "luminous hot particle {i} must leave cold"
        );
        for c in 0..3 {
            let toward_hot = (hot[c] - cold[c]).signum();
            assert!(
                (data.color[i][c] - cold[c]) * toward_hot > 0.0,
                "luminous particle {i} channel {c} must move toward hot"
            );
        }
    }
}

/// Two far-apart clumps: a MODERATE-σ luminous disk (progenitor 1) and a HOT,
/// high-σ non-luminous halo (progenitor 0). Identical geometry; the halo's
/// velocity spread is 3× the disk's, so the halo is dynamically hotter.
fn disk_cool_halo_hot_state() -> State {
    let clump = |c: DVec3| {
        vec![
            c + DVec3::new(0.5, 0.0, 0.0),
            c + DVec3::new(-0.5, 0.0, 0.0),
            c + DVec3::new(0.0, 0.5, 0.0),
        ]
    };
    let pos = [clump(DVec3::ZERO), clump(DVec3::new(100.0, 0.0, 0.0))].concat();
    let vel = |s: f64| {
        vec![
            DVec3::new(s, 0.0, 0.0),
            DVec3::new(-s, 0.0, 0.0),
            DVec3::new(0.0, s, 0.0),
        ]
    };
    State {
        mass: vec![1.0; 6],
        id: (0..6).map(ParticleId).collect(),
        progenitor: vec![
            Progenitor(1), // luminous disk, moderate σ
            Progenitor(1),
            Progenitor(1),
            Progenitor(0), // non-luminous halo, hot σ (3×)
            Progenitor(0),
            Progenitor(0),
        ],
        kind: vec![Species::Collisionless; 6],
        u: vec![0.0; 6],
        time: 0.0,
        a: 1.0,
        pos,
        vel: [vel(1.0), vel(3.0)].concat(),
    }
}

#[test]
fn dispersion_reference_excludes_non_luminous_progenitors() {
    // The σ_v temperature SCALE (σ_ref inside dispersion_colors) must be set by
    // the luminous population alone. A hot dark-matter halo left in the reference
    // inflates σ_ref and crushes the colder luminous disk toward cold — the
    // "too much red" the user saw. Excluding the halo (the luminous mask applied
    // to the reference too) lowers σ_ref and lets the disk read hotter/bluer.
    //
    // Same state, two masks: disk-only vs disk+halo. Because each disk particle's
    // own σ is identical across the two runs, the ONLY difference is σ_ref, so
    // every disk particle must come out strictly hotter when the hot halo is out
    // of the reference. On the pre-fix code σ_ref ignores the mask (always the
    // full population) ⇒ the two runs are identical ⇒ this fails.
    let cold = [0.1, 0.2, 0.9];
    let hot = [0.9, 0.8, 0.1];
    let state = disk_cool_halo_hot_state();
    let make = |lum: u64| PrepConfig {
        color: ColorMode::Dispersion(DispersionColoring {
            k: 2,
            softening: 1e-9,
            cold,
            hot,
            luminous: lum,
            reference: SigmaReference::Luminous, // scale from the luminous set alone
        }),
        ..sample_config()
    };
    let excluded = prepare(&state, &make(1u64 << 1)); // disk only in σ_ref
    let included = prepare(&state, &make((1u64 << 1) | (1u64 << 0))); // + hot halo
    for i in 0..3 {
        for c in 0..3 {
            let toward_hot = (hot[c] - cold[c]).signum();
            assert!(
                (excluded.color[i][c] - included.color[i][c]) * toward_hot > 0.0,
                "disk particle {i} channel {c}: excluding the hot halo from σ_ref \
                 must make it hotter (excluded {} vs included {})",
                excluded.color[i][c],
                included.color[i][c]
            );
        }
    }
}

#[test]
fn dispersion_reference_full_ignores_the_luminous_mask() {
    // The FULL reference toggle sets σ_ref over the WHOLE population regardless of
    // the luminous mask — the pre-recentering "masked" mapping the user wants back.
    // Same disk particles, two luminous masks: disk-only vs disk+halo. Under
    // `Luminous` these differ (the recenter test above); under `Full` σ_ref is the
    // full-population mean in BOTH runs, so the ramped disk particles must come out
    // bit-for-bit IDENTICAL — the mask changes only who is ramped, never the scale.
    let cold = [0.1, 0.2, 0.9];
    let hot = [0.9, 0.8, 0.1];
    let state = disk_cool_halo_hot_state();
    let make = |lum: u64| PrepConfig {
        color: ColorMode::Dispersion(DispersionColoring {
            k: 2,
            softening: 1e-9,
            cold,
            hot,
            luminous: lum,
            reference: SigmaReference::Full, // scale from the whole population
        }),
        ..sample_config()
    };
    let disk_only = prepare(&state, &make(1u64 << 1)); // halo not ramped, but in σ_ref
    let disk_and_halo = prepare(&state, &make((1u64 << 1) | (1u64 << 0)));
    for i in 0..3 {
        assert_eq!(
            disk_only.color[i], disk_and_halo.color[i],
            "disk particle {i}: with Full reference σ_ref is the full-population mean \
             either way, so the luminous mask must not shift the ramped disk color"
        );
    }
}

#[test]
fn size_by_density_shrinks_dense_splats_and_softens_sparse_ones() {
    // The clustered octahedron + 3 sparse escapers: dense splats must come out
    // strictly smaller than sparse ones, all inside the clamp band.
    let state = clustered_state();
    let base = 1.0;
    let (min_frac, max_frac) = (0.25, 4.0);
    let cfg = PrepConfig {
        size: base,
        size_by_density: Some(SizeByDensity {
            k: 2,
            softening: 1e-6,
            min_frac,
            max_frac,
        }),
        ..Default::default()
    };
    let data = prepare(&state, &cfg);
    for i in 0..6 {
        for j in 6..9 {
            assert!(
                data.size[i] < data.size[j],
                "dense splat {i} ({}) must be tighter than sparse {j} ({})",
                data.size[i],
                data.size[j]
            );
        }
    }
    for s in &data.size {
        assert!(*s >= min_frac * base && *s <= max_frac * base);
    }
}

#[test]
fn compression_hue_shifts_only_the_compressed_clump() {
    // Two tight quads far apart. At t1 clump A has contracted to half scale
    // (ρ ×8); clump B is untouched — its kNN geometry is internal, so its density
    // is bit-identical to t0 and its color must stay exactly the base. Clump A at
    // full strength shifts t = 1 − ρ0/ρ = 7/8 of the way to young.
    let quad = |center: DVec3, scale: f64| -> Vec<DVec3> {
        vec![
            center + DVec3::new(scale * 0.5, 0.0, 0.0),
            center + DVec3::new(-scale * 0.5, 0.0, 0.0),
            center + DVec3::new(0.0, scale * 0.5, 0.0),
            center + DVec3::new(0.0, -scale * 0.5, 0.0),
        ]
    };
    let b_center = DVec3::new(100.0, 0.0, 0.0);
    let pos0: Vec<DVec3> = [quad(DVec3::ZERO, 1.0), quad(b_center, 1.0)].concat();
    let pos1: Vec<DVec3> = [quad(DVec3::ZERO, 0.5), quad(b_center, 1.0)].concat();
    let state_at = |pos: Vec<DVec3>| State {
        vel: vec![DVec3::ZERO; 8],
        mass: vec![1.0; 8],
        id: (0..8).map(ParticleId).collect(),
        progenitor: vec![Progenitor(0); 8],
        kind: vec![Species::Collisionless; 8],
        u: vec![0.0; 8],
        time: 0.0,
        a: 1.0,
        pos,
    };
    let (k, softening) = (2, 1e-9);
    let rho0 = knn_density(&pos0, k, softening);
    let base = [0.8, 0.4, 0.2];
    let young = [0.5, 0.7, 1.0];
    let cfg = PrepConfig {
        palette: vec![base],
        compression: Some(CompressionHue {
            k,
            softening,
            rho0,
            young,
            strength: 1.0,
        }),
        ..Default::default()
    };
    let data = prepare(&state_at(pos1), &cfg);
    // Untouched clump B: base color bit-exact (ρ = ρ0 ⇒ no shift at all).
    for i in 4..8 {
        assert_eq!(data.color[i], base, "uncompressed particle {i} shifted");
    }
    // Compressed clump A: exactly 7/8 of the way to young (ρ/ρ0 = 8), lerp-rounding
    // tolerance only.
    let t = 1.0 - 1.0 / 8.0;
    for i in 0..4 {
        for c in 0..3 {
            let want = (1.0 - t) * base[c] + t * young[c];
            assert!(
                (data.color[i][c] - want).abs() < 1e-5,
                "particle {i} channel {c}: got {}, want {want}",
                data.color[i][c]
            );
        }
    }
    // And the t0 frame itself: ρ = ρ0 everywhere ⇒ every color exactly base.
    let at_t0 = prepare(&state_at(pos0), &cfg);
    assert!(at_t0.color.iter().all(|&c| c == base));
}

#[test]
fn full_featured_prepare_is_deterministic() {
    // Every M6e path on at once — two calls, identical frames.
    let state = two_temperature_state();
    let cfg = PrepConfig {
        color: ColorMode::Dispersion(DispersionColoring {
            k: 2,
            softening: 1e-9,
            cold: [0.1, 0.2, 0.9],
            hot: [0.9, 0.8, 0.1],
            luminous: u64::MAX,
            reference: SigmaReference::Full,
        }),
        density: Some(DensityColoring {
            k: 2,
            softening: 1e-9,
            strength: 2.0,
        }),
        size_by_density: Some(SizeByDensity {
            k: 2,
            softening: 1e-9,
            min_frac: 0.5,
            max_frac: 2.0,
        }),
        compression: Some(CompressionHue {
            k: 2,
            softening: 1e-9,
            rho0: knn_density(&two_temperature_state().pos, 2, 1e-9),
            young: [0.6, 0.8, 1.0],
            strength: 0.7,
        }),
        ..Default::default()
    };
    assert_eq!(prepare(&state, &cfg), prepare(&state, &cfg));
}

// ---------------------------------------------------------------------------
// Species routing (M7d): gas leaves the splat list by default — it renders
// through the volumetric grid, not the additive star path. `gas_as_splats`
// keeps it (debug mode). The filter runs AFTER all attribute math, so stellar
// rows are identical either way, and gas-free states never take the filter.
// ---------------------------------------------------------------------------

/// Three stars interleaved with two gas particles (progenitors 4/5 per plan D1).
fn mixed_species_state() -> State {
    State {
        pos: vec![
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.5, 0.5, 0.0),
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(-0.5, 0.5, 1.0),
            DVec3::new(0.0, 0.0, -3.0),
        ],
        vel: vec![DVec3::ZERO; 5],
        mass: vec![2.0, 1.0, 4.0, 3.0, 1.0],
        id: (0..5).map(ParticleId).collect(),
        progenitor: vec![
            Progenitor(0),
            Progenitor(4),
            Progenitor(1),
            Progenitor(5),
            Progenitor(0),
        ],
        kind: vec![
            Species::Collisionless,
            Species::Gas,
            Species::Collisionless,
            Species::Gas,
            Species::Collisionless,
        ],
        u: vec![0.0; 5],
        time: 5.0,
        a: 1.0,
    }
}

#[test]
fn gas_leaves_the_splat_list_by_default() {
    let state = mixed_species_state();
    let cfg = sample_config();
    assert!(!cfg.gas_as_splats, "routing out is the default");
    let data = prepare(&state, &cfg);

    // Only the three collisionless rows survive, in their original order.
    assert_eq!(data.len(), 3);
    assert_eq!(data.pos[0].as_dvec3(), state.pos[0]);
    assert_eq!(data.pos[1].as_dvec3(), state.pos[2]);
    assert_eq!(data.pos[2].as_dvec3(), state.pos[4]);
    assert_eq!(data.brightness, vec![3.0 * 2.0, 3.0 * 4.0, 3.0 * 1.0]);
    assert_eq!(data.color[0], cfg.palette[0]); // progenitor 0
    assert_eq!(data.color[1], cfg.palette[1]); // progenitor 1
    assert_eq!(data.color[2], cfg.palette[0]); // progenitor 0
}

#[test]
fn gas_as_splats_keeps_the_full_list() {
    let state = mixed_species_state();
    let cfg = PrepConfig {
        gas_as_splats: true,
        ..sample_config()
    };
    let data = prepare(&state, &cfg);
    assert_eq!(data.len(), 5);
    // Gas rows are ordinary splats in debug mode: palette by progenitor
    // (4 and 5 wrap modulo the 2-entry palette), brightness by mass.
    assert_eq!(data.color[1], cfg.palette[0]); // progenitor 4 % 2 = 0
    assert_eq!(data.color[3], cfg.palette[1]); // progenitor 5 % 2 = 1
    assert_eq!(data.brightness[1], 3.0 * 1.0);
    assert_eq!(data.brightness[3], 3.0 * 3.0);
}

#[test]
fn routing_is_a_pure_filter_of_the_debug_map() {
    // With the k-NN features ON (density boost, size-by-density), the stellar
    // rows of the routed output must equal the corresponding rows of the
    // full (gas_as_splats) map bit-exactly: the filter runs after all
    // attribute math, and the neighbourhood estimates see ALL particles
    // (gas is mass) in both cases.
    let state = mixed_species_state();
    let full_cfg = PrepConfig {
        density: Some(DensityColoring {
            k: 2,
            softening: 1e-6,
            strength: 1.0,
        }),
        size_by_density: Some(SizeByDensity {
            k: 2,
            softening: 1e-6,
            min_frac: 0.5,
            max_frac: 2.0,
        }),
        gas_as_splats: true,
        ..sample_config()
    };
    let routed_cfg = PrepConfig {
        gas_as_splats: false,
        ..full_cfg.clone()
    };
    let full = prepare(&state, &full_cfg);
    let routed = prepare(&state, &routed_cfg);

    let stellar = [0usize, 2, 4];
    assert_eq!(routed.len(), stellar.len());
    for (out, &src) in stellar.iter().enumerate() {
        assert_eq!(routed.pos[out], full.pos[src]);
        assert_eq!(routed.color[out], full.color[src]);
        assert_eq!(routed.size[out], full.size[src]);
        assert_eq!(routed.brightness[out], full.brightness[src]);
    }
}

#[test]
fn all_gas_state_prepares_to_an_empty_frame() {
    let mut state = mixed_species_state();
    state.kind = vec![Species::Gas; state.len()];
    let data = prepare(&state, &sample_config());
    assert!(data.is_empty());
    assert_eq!(data.color.len(), 0);
    assert_eq!(data.size.len(), 0);
    assert_eq!(data.brightness.len(), 0);
}

#[test]
fn frozen_colors_are_indexed_by_state_row_not_splat_row() {
    // ColorMode::Frozen carries one color per particle of the STATE (computed
    // from snapshot 0 before routing); the filter must pick the stellar
    // entries, not the first n_star ones.
    let state = mixed_species_state();
    let frozen: Vec<[f32; 3]> = (0..5).map(|i| [i as f32 * 0.1, 0.5, 0.9]).collect();
    let cfg = PrepConfig {
        color: ColorMode::Frozen(frozen.clone()),
        ..sample_config()
    };
    let data = prepare(&state, &cfg);
    assert_eq!(data.len(), 3);
    assert_eq!(data.color[0], frozen[0]);
    assert_eq!(data.color[1], frozen[2]);
    assert_eq!(data.color[2], frozen[4]);
}
