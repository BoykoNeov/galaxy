//! `galaxy-renderprep`: the render-prep stage — snapshots → **frame-data**.
//!
//! Frame-data (DESIGN.md **Contract 3**) is the decoupling boundary of the
//! visualization pipeline: the wgpu renderer *and* Blender are both just consumers
//! of it. It carries per-particle **visual** attributes (position, emissive color,
//! splat size, brightness) with the physics stripped away, so a frame can be
//! re-rendered from any camera without touching the simulator.
//!
//! Positions are **f32** here (not the f64 of the physics snapshot): frame-data
//! feeds a GPU vertex layout, and the headless-wgpu spike pinned that to f32. This
//! is a deliberate, lossy projection out of the compute domain — the last place
//! full f64 is needed is the force/integrate loop, which is upstream.
//!
//! The schema mirrors `galaxy-io`'s versioned little-endian style and is the
//! decoupling contract, so it is versioned and changes deliberately.

pub mod density;
pub mod frame;
pub mod prepare;

pub use density::{density_boost, knn_density, DensityColoring};
pub use frame::{FrameData, FrameError, FrameHeader};
pub use prepare::{prepare, PrepConfig};
