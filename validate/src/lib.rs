//! `galaxy-validate`: cross-check tooling for the N-body engine.
//!
//! The always-on conservation and orbital-setup checks live as ordinary tests in
//! `galaxy-sim` and `galaxy-ic`. This crate provides the bridge for the *heavier,
//! manually-run* cross-check against REBOUND (DESIGN.md M2): a minimal NumPy
//! `.npy` writer plus an `export_collision` example that dumps a collision IC and
//! our run's diagnostics in a form the committed Python harness
//! (`validate/rebound/cross_check.py`) can consume. `.npy` is used so the harness
//! needs only NumPy — no extra parser, and no HDF5 (the Windows link landmine).
//!
//! Nothing here runs REBOUND in CI; the harness is run where REBOUND is installed.

pub mod npy;
