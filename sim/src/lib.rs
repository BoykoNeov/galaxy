//! `galaxy-sim`: the headless stepping engine.
//!
//! Drives `IC → (solver + integrator) stepping loop → snapshots`. The engine is
//! deliberately thin: it owns the time loop and the snapshot cadence and nothing
//! else. Force accuracy lives in the `ForceSolver`, the time discretization in the
//! `Integrator`, and the on-disk format in `galaxy-io` — all injected, so the
//! engine never needs to change when any of them is swapped (the 10^8 / cosmology
//! door stays open).
//!
//! Snapshots are delivered through a [`SnapshotSink`], so the same loop can write
//! numbered files ([`DirectorySink`]) in production or capture states in memory in
//! a test. Checkpoint/restart is a later milestone, not on the M2 path.

use std::path::{Path, PathBuf};

use galaxy_core::{Background, ForceSolver, Integrator, State};
use galaxy_io::Header;

/// Errors from a simulation run.
#[derive(thiserror::Error, Debug)]
pub enum SimError {
    /// A snapshot could not be written.
    #[error("snapshot error: {0}")]
    Snapshot(#[from] galaxy_io::SnapshotError),
    /// The run configuration is invalid.
    #[error("invalid simulation config: {0}")]
    Config(String),
}

/// Configuration for a run. Physics (G, θ) lives in the injected solver; this is
/// the time loop, the snapshot cadence, and the metadata stamped into headers.
#[derive(Clone, Debug, PartialEq)]
pub struct SimConfig {
    /// Timestep.
    pub dt: f64,
    /// Number of steps to integrate.
    pub n_steps: u64,
    /// Emit a snapshot every `snapshot_every` steps (must be ≥ 1). The initial
    /// conditions (step 0) and the final step are always emitted.
    pub snapshot_every: u64,
    /// Softening length to record in snapshot headers (must match the solver's).
    pub softening: f64,
    /// RNG seed that produced the IC, recorded in headers.
    pub rng_seed: u64,
    /// Scenario config hash, recorded in headers.
    pub config_hash: u64,
    /// Units tag, recorded in headers.
    pub units: String,
}

/// Summary returned by [`run`].
#[derive(Clone, Debug, PartialEq)]
pub struct RunSummary {
    /// Number of steps integrated.
    pub steps: u64,
    /// Simulation time after the final step.
    pub final_time: f64,
    /// Total number of snapshots emitted (including the IC and the final step).
    pub snapshots_emitted: u64,
}

/// A consumer of snapshots produced during a run.
pub trait SnapshotSink {
    /// Consume one snapshot. Called for the IC, at the configured cadence, and for
    /// the final step.
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError>;
}

/// A [`SnapshotSink`] that writes each snapshot to a numbered file in a directory,
/// named `snapshot_<step>.snap`.
pub struct DirectorySink {
    dir: PathBuf,
    written: u64,
}

impl DirectorySink {
    /// Create (or reuse) the output directory.
    pub fn new(dir: impl AsRef<Path>) -> Result<Self, SimError> {
        let _ = dir.as_ref();
        todo!()
    }

    /// Number of files written so far.
    pub fn written(&self) -> u64 {
        self.written
    }
}

impl SnapshotSink for DirectorySink {
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError> {
        let _ = (header, state);
        todo!()
    }
}

/// Run the stepping loop: emit the IC, integrate `n_steps`, emitting a snapshot
/// every `snapshot_every` steps and always capturing the final step.
pub fn run(
    state: &mut State,
    solver: &mut dyn ForceSolver,
    integ: &mut dyn Integrator,
    bg: &dyn Background,
    config: &SimConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<RunSummary, SimError> {
    let _ = (state, solver, integ, bg, config, sink);
    todo!()
}
