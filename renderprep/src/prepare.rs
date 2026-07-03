//! Snapshot → frame-data mapping. The base map is a **pure map** (no spatial tree):
//! color by progenitor, brightness by mass. On top of that, opt-in passes (all off
//! by default, so the base map stays a bit-for-bit pure map):
//!
//! * [`DensityColoring`] — brightens dense neighbourhoods via a k-NN estimate,
//!   never dims (M3.6, tuned on in M6a).
//! * [`ColorMode`] (M6e) — what the colors *mean*: the progenitor palette
//!   (default), frozen per-particle colors (the initial-radius ramp, computed once
//!   at t=0 by [`crate::coloring::initial_radius_colors`]), or a per-frame σ_v
//!   ramp over the k-NN neighbourhood.
//! * [`SizeByDensity`] (M6e) — density-driven splat sizes: tight cores, soft
//!   diffuse splats.
//! * [`CompressionHue`] (M6e) — the star-formation proxy: hue shift toward a
//!   young-population blue-white, triggered by density *compression* vs the same
//!   particle's t=0 neighbourhood.
//!
//! All k-NN consumers that agree on `(k, softening)` share **one** O(N²) pass per
//! `prepare` call (the movie default wires them all to the scenario's `ε`).

use galaxy_core::State;

use crate::coloring::{compression_colors, dispersion_colors};
use crate::density::{density_boost, density_sizes, knn_neighbourhood, velocity_dispersion};
use crate::frame::FrameData;

pub use crate::density::{DensityColoring, SizeByDensity};

/// What a particle's color *means* (M6e). The default is today's progenitor
/// palette, bit-compatible with the pre-M6e map.
#[derive(Clone, Debug, PartialEq)]
pub enum ColorMode {
    /// `palette[progenitor % len]` — the flat per-galaxy palette (the default).
    Progenitor,
    /// Precomputed per-particle colors, keyed by particle index — the frozen
    /// initial-radius ramp (computed once from snapshot 0, constant thereafter).
    /// Must hold exactly one color per particle of the prepared state.
    Frozen(Vec<[f32; 3]>),
    /// Per-frame velocity-dispersion ramp over the k-NN neighbourhood.
    Dispersion(DispersionColoring),
}

/// Parameters for [`ColorMode::Dispersion`]: σ_v over the k-NN neighbourhood set,
/// mapped through a `cold → hot` ramp (see [`crate::coloring::dispersion_colors`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DispersionColoring {
    /// Neighbour count (see [`DensityColoring::k`]).
    pub k: usize,
    /// kNN distance floor — irrelevant to σ_v itself, but carried so this consumer
    /// can share the single per-`prepare` kNN pass with the density-driven ones.
    pub softening: f64,
    /// Color of dynamically cold material (σ = 0).
    pub cold: [f32; 3],
    /// Color σ_v ramps toward (σ ≫ σ_ref).
    pub hot: [f32; 3],
}

/// The star-formation proxy (M6e): hue shift toward `young` keyed on density
/// compression `ρ(t)/ρ(0)` (see [`crate::coloring::compression_colors`]).
#[derive(Clone, Debug, PartialEq)]
pub struct CompressionHue {
    /// Neighbour count (see [`DensityColoring::k`]).
    pub k: usize,
    /// kNN distance floor (see [`DensityColoring::softening`]).
    pub softening: f64,
    /// Reference densities from snapshot 0 (same estimator, same `(k, softening)`),
    /// one per particle — computed once by the caller and reused for every frame.
    pub rho0: Vec<f64>,
    /// The "young population" blue-white the hue shifts toward.
    pub young: [f32; 3],
    /// Saturation: compressed material shifts up to `strength` of the way to
    /// `young` (clamped to `[0, 1]`); `0.0` is the identity.
    pub strength: f32,
}

/// Configuration for the snapshot → frame-data map.
#[derive(Clone, Debug, PartialEq)]
pub struct PrepConfig {
    /// Emissive RGB palette indexed by `progenitor`. A progenitor id out of range
    /// wraps modulo the palette length; an empty palette falls back to white.
    pub palette: Vec<[f32; 3]>,
    /// Brightness per unit mass (brightness = `brightness_per_mass * mass`).
    pub brightness_per_mass: f32,
    /// Splat radius assigned to every particle (constant unless `size_by_density`).
    pub size: f32,
    /// Optional density-aware brightness modulation. `None` (the default) keeps the
    /// pure progenitor/mass map bit-for-bit; `Some(..)` brightens dense regions via
    /// a k-NN density estimate (never dims — see [`DensityColoring`]).
    pub density: Option<DensityColoring>,
    /// What the colors mean (M6e). [`ColorMode::Progenitor`] (the default) is the
    /// pre-M6e palette map, bit-for-bit.
    pub color: ColorMode,
    /// Optional density-driven splat sizing (M6e). `None` (the default) keeps the
    /// constant `size`.
    pub size_by_density: Option<SizeByDensity>,
    /// Optional star-formation-proxy hue shift (M6e). `None` (the default) leaves
    /// the base colors untouched.
    pub compression: Option<CompressionHue>,
}

impl Default for PrepConfig {
    /// A sensible two-galaxy default: progenitor 0 warm, progenitor 1 cool.
    fn default() -> Self {
        PrepConfig {
            palette: vec![[1.0, 0.45, 0.2], [0.3, 0.55, 1.0]],
            brightness_per_mass: 1.0,
            size: 1.0,
            density: None,
            color: ColorMode::Progenitor,
            size_by_density: None,
            compression: None,
        }
    }
}

/// White fallback when the palette is empty.
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

/// Map a physics `State` to renderable frame-data under `config`. Pure and
/// order-preserving: particle `i` in the state becomes column entry `i`.
///
/// Panics if `config` carries per-particle data of the wrong length
/// ([`ColorMode::Frozen`] colors, [`CompressionHue::rho0`]) — a caller contract
/// violation (they must be computed from this run's own snapshot 0), not a data
/// condition. Same stance as `interp::subframe`.
pub fn prepare(state: &State, config: &PrepConfig) -> FrameData {
    let n = state.len();
    let mut pos = Vec::with_capacity(n);
    let mut brightness = Vec::with_capacity(n);

    for i in 0..n {
        pos.push(state.pos[i].as_vec3()); // f64 -> f32 projection for the GPU
        brightness.push(config.brightness_per_mass * state.mass[i] as f32);
    }

    // One kNN pass per distinct (k, softening) among the consumers below (M6e):
    // the movie default points them all at the same parameters, so the O(N²)
    // estimate runs once per frame no matter how many features are on.
    let knn = KnnCache::for_config(state, config);

    // Base colors: what the color *means* is the mode's business (M6e). The
    // Progenitor arm is the pre-M6e pure map, bit-for-bit.
    let mut color: Vec<[f32; 3]> = match &config.color {
        ColorMode::Progenitor => (0..n)
            .map(|i| palette_color(config, state.progenitor[i].0))
            .collect(),
        ColorMode::Frozen(colors) => {
            let _ = colors;
            todo!("M6e: frozen per-particle colors")
        }
        ColorMode::Dispersion(dc) => {
            let sigma = velocity_dispersion(&state.vel, knn.neighbours(dc.k, dc.softening));
            dispersion_colors(&sigma, dc.cold, dc.hot)
        }
    };

    // Star-formation-proxy hue shift (M6e): compression vs the t=0 neighbourhood.
    if let Some(ch) = &config.compression {
        assert_eq!(
            ch.rho0.len(),
            n,
            "compression rho0 is not from this run's snapshot 0"
        );
        color = compression_colors(
            &color,
            knn.density(ch.k, ch.softening),
            &ch.rho0,
            ch.young,
            ch.strength,
        );
    }

    // Optional density-aware pass: brighten dense neighbourhoods (never dim). Off by
    // default, so the base map above is delivered bit-for-bit when `density == None`.
    // [red-phase note: still on knn_density directly; moves onto the shared cache in
    // the M6e implementation, gated bit-identical.]
    if let Some(dc) = &config.density {
        let density = crate::density::knn_density(&state.pos, dc.k, dc.softening);
        let boost = density_boost(&density, dc.strength);
        for (b, &g) in brightness.iter_mut().zip(&boost) {
            *b *= g;
        }
    }

    // Splat sizes: the configured constant, or density-driven (M6e).
    let size = match &config.size_by_density {
        None => vec![config.size; n],
        Some(sd) => density_sizes(
            knn.density(sd.k, sd.softening),
            config.size,
            sd.min_frac,
            sd.max_frac,
        ),
    };

    FrameData {
        pos,
        color,
        brightness,
        size,
    }
}

/// The shared kNN passes for one `prepare` call: one
/// [`knn_neighbourhood`] evaluation per **distinct** `(k, softening)` requested by
/// the config's consumers (density boost, dispersion coloring, size-by-density,
/// compression hue). Keys use the softening's bit pattern so the dedup is exact.
struct KnnCache {
    passes: Vec<((usize, u64), (Vec<f64>, Vec<Vec<usize>>))>,
}

impl KnnCache {
    fn for_config(state: &State, config: &PrepConfig) -> Self {
        let mut keys: Vec<(usize, u64)> = Vec::new();
        let mut want = |k: usize, soft: f64| {
            let key = (k, soft.to_bits());
            if !keys.contains(&key) {
                keys.push(key);
            }
        };
        if let ColorMode::Dispersion(dc) = &config.color {
            want(dc.k, dc.softening);
        }
        if let Some(ch) = &config.compression {
            want(ch.k, ch.softening);
        }
        if let Some(sd) = &config.size_by_density {
            want(sd.k, sd.softening);
        }
        let passes = keys
            .into_iter()
            .map(|key| {
                (
                    key,
                    knn_neighbourhood(&state.pos, key.0, f64::from_bits(key.1)),
                )
            })
            .collect();
        KnnCache { passes }
    }

    fn pass(&self, k: usize, softening: f64) -> &(Vec<f64>, Vec<Vec<usize>>) {
        let key = (k, softening.to_bits());
        &self
            .passes
            .iter()
            .find(|(pk, _)| *pk == key)
            .expect("kNN pass requested but not collected — for_config is out of sync")
            .1
    }

    fn density(&self, k: usize, softening: f64) -> &[f64] {
        &self.pass(k, softening).0
    }

    fn neighbours(&self, k: usize, softening: f64) -> &[Vec<usize>] {
        &self.pass(k, softening).1
    }
}

/// Emissive color for a progenitor: `palette[progenitor % len]`, or white if the
/// palette is empty. Wrapping keeps the map total and deterministic for any tag.
fn palette_color(config: &PrepConfig, progenitor: u16) -> [f32; 3] {
    if config.palette.is_empty() {
        WHITE
    } else {
        config.palette[progenitor as usize % config.palette.len()]
    }
}
