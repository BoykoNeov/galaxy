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
        let dir = dir.as_ref().to_path_buf();
        std::fs::create_dir_all(&dir).map_err(|e| SimError::Snapshot(e.into()))?;
        Ok(Self { dir, written: 0 })
    }

    /// Number of files written so far.
    pub fn written(&self) -> u64 {
        self.written
    }
}

impl SnapshotSink for DirectorySink {
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError> {
        let path = self.dir.join(format!("snapshot_{:08}.snap", header.step));
        galaxy_io::write_file(path, header, state)?;
        self.written += 1;
        Ok(())
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
    if config.snapshot_every == 0 {
        return Err(SimError::Config("snapshot_every must be >= 1".to_string()));
    }
    if !config.dt.is_finite() || config.dt <= 0.0 {
        return Err(SimError::Config(format!(
            "dt must be a positive finite number, got {}",
            config.dt
        )));
    }

    let mut emitted = 0u64;

    // Step 0: the initial conditions.
    emit_snapshot(state, 0, config, sink)?;
    emitted += 1;
    let mut last_emitted_step = 0u64;

    for step in 1..=config.n_steps {
        integ.step(state, solver, bg, config.dt);
        if step % config.snapshot_every == 0 {
            emit_snapshot(state, step, config, sink)?;
            emitted += 1;
            last_emitted_step = step;
        }
    }

    // Always capture the final step, unless it already landed on the cadence.
    if last_emitted_step != config.n_steps {
        emit_snapshot(state, config.n_steps, config, sink)?;
        emitted += 1;
    }

    Ok(RunSummary {
        steps: config.n_steps,
        final_time: state.time,
        snapshots_emitted: emitted,
    })
}

/// Stamp a header for the current state and hand it to the sink.
fn emit_snapshot(
    state: &State,
    step: u64,
    config: &SimConfig,
    sink: &mut dyn SnapshotSink,
) -> Result<(), SimError> {
    let header = Header::for_state(
        state,
        step,
        config.softening,
        config.rng_seed,
        config.config_hash,
        config.units.as_str(),
    );
    sink.emit(&header, state)
}
