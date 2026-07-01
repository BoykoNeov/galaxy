//! `galaxy-ic`: initial-condition generators for the N-body engine.
//!
//! Pure — produces a `galaxy_core::State` and nothing else (no I/O, no solver
//! dependency). The first model is the **Plummer sphere**, sampled exactly from
//! its analytic distribution function f(ℰ) ∝ ℰ^(7/2). It is the genuine first
//! galaxy and the building block for two-galaxy collision setups. See
//! `DESIGN.md` (M1) for context.

pub mod collision;
pub mod disk;
mod encounter;
pub mod plummer;

pub use collision::Collision;
pub use disk::ExponentialDisk;
pub use plummer::Plummer;
