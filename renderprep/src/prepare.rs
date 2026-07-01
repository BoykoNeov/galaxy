//! Snapshot → frame-data mapping. MVP is a **pure map** (no spatial tree): color
//! by progenitor, brightness by mass. Local density / velocity-dispersion coloring
//! (which needs a spatial kNN tree) is a deferred later pass — progenitor coloring
//! is the iconic tidal-tail money shot and needs no neighbourhood queries.

use galaxy_core::State;

use crate::frame::FrameData;

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
}

impl Default for PrepConfig {
    /// A sensible two-galaxy default: progenitor 0 warm, progenitor 1 cool.
    fn default() -> Self {
        PrepConfig {
            palette: vec![[1.0, 0.45, 0.2], [0.3, 0.55, 1.0]],
            brightness_per_mass: 1.0,
            size: 1.0,
        }
    }
}

/// Map a physics `State` to renderable frame-data under `config`. Pure and
/// order-preserving: particle `i` in the state becomes column entry `i`.
pub fn prepare(_state: &State, _config: &PrepConfig) -> FrameData {
    todo!("progenitor->color, mass->brightness, constant size")
}
