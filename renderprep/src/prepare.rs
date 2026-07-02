//! Snapshot → frame-data mapping. The base map is a **pure map** (no spatial tree):
//! color by progenitor, brightness by mass. On top of that, an **optional**
//! density-aware pass ([`DensityColoring`]) brightens dense neighbourhoods (cores,
//! tidal bridges) via a k-NN estimate while leaving the diffuse tails at full
//! brightness — off by default, so the base map stays a bit-for-bit pure map.
//! Velocity-dispersion coloring (same k-NN neighbourhood, `σ_v` over the neighbour
//! set) is the next deferred refinement.

use galaxy_core::State;

use crate::density::{density_boost, knn_density};
use crate::frame::FrameData;

pub use crate::density::DensityColoring;

/// Configuration for the snapshot → frame-data map.
#[derive(Clone, Debug, PartialEq)]
pub struct PrepConfig {
    /// Emissive RGB palette indexed by `progenitor`. A progenitor id out of range
    /// wraps modulo the palette length; an empty palette falls back to white.
    pub palette: Vec<[f32; 3]>,
    /// Brightness per unit mass (brightness = `brightness_per_mass * mass`).
    pub brightness_per_mass: f32,
    /// Splat radius assigned to every particle (constant for the MVP).
    pub size: f32,
    /// Optional density-aware brightness modulation. `None` (the default) keeps the
    /// pure progenitor/mass map bit-for-bit; `Some(..)` brightens dense regions via
    /// a k-NN density estimate (never dims — see [`DensityColoring`]).
    pub density: Option<DensityColoring>,
}

impl Default for PrepConfig {
    /// A sensible two-galaxy default: progenitor 0 warm, progenitor 1 cool.
    fn default() -> Self {
        PrepConfig {
            palette: vec![[1.0, 0.45, 0.2], [0.3, 0.55, 1.0]],
            brightness_per_mass: 1.0,
            size: 1.0,
            density: None,
        }
    }
}

/// White fallback when the palette is empty.
const WHITE: [f32; 3] = [1.0, 1.0, 1.0];

/// Map a physics `State` to renderable frame-data under `config`. Pure and
/// order-preserving: particle `i` in the state becomes column entry `i`.
pub fn prepare(state: &State, config: &PrepConfig) -> FrameData {
    let n = state.len();
    let mut pos = Vec::with_capacity(n);
    let mut color = Vec::with_capacity(n);
    let mut brightness = Vec::with_capacity(n);

    for i in 0..n {
        pos.push(state.pos[i].as_vec3()); // f64 -> f32 projection for the GPU
        color.push(palette_color(config, state.progenitor[i].0));
        brightness.push(config.brightness_per_mass * state.mass[i] as f32);
    }

    // Optional density-aware pass: brighten dense neighbourhoods (never dim). Off by
    // default, so the base map above is delivered bit-for-bit when `density == None`.
    if let Some(dc) = &config.density {
        let density = knn_density(&state.pos, dc.k, dc.softening);
        let boost = density_boost(&density, dc.strength);
        for (b, &g) in brightness.iter_mut().zip(&boost) {
            *b *= g;
        }
    }

    FrameData {
        pos,
        color,
        brightness,
        size: vec![config.size; n],
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
