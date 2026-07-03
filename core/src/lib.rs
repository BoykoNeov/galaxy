//! `galaxy-core`: pure types, traits, and diagnostics for the N-body engine.
//!
//! No I/O, no rendering. Contains the `State` (Structure-of-Arrays), the
//! `ForceSolver` / `Integrator` / `Background` traits, the leapfrog
//! integrator, and conservation diagnostics. See `DESIGN.md` for context.

pub mod background;
pub mod diagnostics;
pub mod integrator;
pub mod state;
pub mod traits;

pub use background::StaticBackground;
pub use glam::{DQuat, DVec3};
pub use integrator::LeapfrogKdk;
pub use state::{ParticleId, Progenitor, Species, State};
pub use traits::{Background, ForceSolver, Integrator};
