//! Isothermal SPH toolkit (DESIGN.md M7): cubic-spline kernel, uniform hash-grid
//! neighbor search, and adaptive-h density summation.
//!
//! This module is the single source of truth for SPH smoothing math: the force
//! path (M7b) and renderprep's grid deposition (M7d) both consume it, the same
//! single-source discipline as the `reference_*` LBVH stages. Every fast path
//! here is gated against the O(N²) oracles in [`reference`].
//!
//! The smoothing length `h` is a DERIVED quantity — a deterministic function of
//! positions (adaptive bisection on the kernel-weighted neighbor count), never
//! stored on `State` or in snapshots. See the M7 plan (D2) for why: the
//! `ForceSolver` trait takes `&State`, so a stored column could never be kept
//! fresh and would go stale for every downstream consumer.

pub mod cfl;
pub mod density;
pub mod forces;
pub mod gravity_sph;
pub mod grid;
pub mod kernel;
pub mod reference;

pub use cfl::{max_stable_dt, max_stable_dt_per_particle, validate_dt, CflViolation};
pub use density::{
    density_adaptive, density_adaptive_serial, density_fixed, DensityConfig, DensityResult,
};
pub use forces::{
    hydro_accel_and_dudt, hydro_accel_and_dudt_serial, hydro_accelerations,
    hydro_accelerations_serial, Eos, HydroParams,
};
pub use gravity_sph::GravitySph;
pub use grid::HashGrid;
pub use kernel::{grad_w, w, SUPPORT};
pub use reference::{reference_density, reference_neighbours};
