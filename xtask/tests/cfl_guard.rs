//! Gates for the CFL sink decorator (`CflGuard`, M7c, D6): the fail-loud guard
//! that a fixed global `dt` stays within the hydro CFL bound at snapshot cadence.
//!
//! The bound `dt ≤ C_cfl · min_i h_i / v_sig,i` is a positive finite number for any
//! gas state, so the gates pick `dt` values that are unambiguously inside/outside
//! it (a tiny dt is always stable; a huge dt never is) rather than deriving the
//! threshold from `validate_dt`'s own output. What is tested is the *decorator's*
//! contract: delegate to the inner sink on a stable dt, and short-circuit with a
//! `SimError::Config` (never touching the inner sink) on a violation.

use std::cell::Cell;
use std::rc::Rc;

use galaxy_core::State;
use galaxy_io::Header;
use galaxy_sim::{SimError, SnapshotSink};
use galaxy_solvers::sph::{DensityConfig, Eos, HydroParams};
use galaxy_xtask::cfl_guard::{CflGuard, C_CFL};

use galaxy_ic::{ExponentialDisk, Plummer};

/// A small gas realization with enough gas particles for the adaptive-h density
/// pass (n_ngb = 48) to have a well-defined bound.
fn gas_state() -> State {
    ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, Plummer::new(1.0, 1.0, 1.0))
        .with_gas(0.5, 0.1)
        .sample_gas(600, 400, 800, 0xC_F1)
}

fn hydro() -> HydroParams {
    HydroParams {
        eos: Eos::Isothermal { c_s: 0.1 },
        ..HydroParams::default()
    }
}

fn header_for(state: &State) -> Header {
    Header::for_state(state, 0, 0.05, 0, 0, "nbody-G1")
}

/// A sink that records how many states reached it — the probe for "did the guard
/// delegate?". The counter is shared (`Rc<Cell>`) so the test keeps a handle after
/// the guard takes ownership of the sink. Its own `emit` always succeeds.
struct CountingSink {
    emits: Rc<Cell<usize>>,
}

impl SnapshotSink for CountingSink {
    fn emit(&mut self, _header: &Header, _state: &State) -> Result<(), SimError> {
        self.emits.set(self.emits.get() + 1);
        Ok(())
    }
}

fn counting_sink() -> (CountingSink, Rc<Cell<usize>>) {
    let counter = Rc::new(Cell::new(0));
    (
        CountingSink {
            emits: counter.clone(),
        },
        counter,
    )
}

#[test]
fn stable_dt_delegates_to_the_inner_sink() {
    let state = gas_state();
    let header = header_for(&state);
    let (sink, count) = counting_sink();
    // A tiny dt is inside the CFL bound for any finite-density gas.
    let mut guard = CflGuard::new(sink, hydro(), DensityConfig::default(), 1e-6, C_CFL);

    // Emit twice; both must pass and both must reach the inner sink.
    guard.emit(&header, &state).expect("stable dt must pass");
    guard.emit(&header, &state).expect("stable dt must pass");
    assert_eq!(
        count.get(),
        2,
        "the inner sink must receive every delegated emit"
    );
}

#[test]
fn over_large_dt_fails_loud_without_touching_the_inner_sink() {
    let state = gas_state();
    let header = header_for(&state);
    let (sink, count) = counting_sink();
    // A huge dt exceeds the CFL bound of any gas state.
    let mut guard = CflGuard::new(sink, hydro(), DensityConfig::default(), 1e6, C_CFL);

    match guard.emit(&header, &state) {
        Err(SimError::Config(_)) => {}
        other => panic!("expected SimError::Config on CFL violation, got {other:?}"),
    }
    assert_eq!(
        count.get(),
        0,
        "a violation must short-circuit before the inner sink"
    );
}

/// A gas-free state has no hydro CFL constraint (`validate_dt` bound = +∞), so the
/// guard must delegate regardless of `dt` — a purely collisionless run must never
/// trip the sentinel.
#[test]
fn gas_free_state_always_delegates() {
    let state = ExponentialDisk::new(0.1, 0.5, 0.05, 2.0, Plummer::new(1.0, 1.0, 1.0))
        .sample(600, 400, 0xC_F1);
    let header = header_for(&state);
    let (sink, count) = counting_sink();
    let mut guard = CflGuard::new(sink, hydro(), DensityConfig::default(), 1e6, C_CFL);

    guard
        .emit(&header, &state)
        .expect("gas-free ⇒ no CFL constraint ⇒ always delegate");
    assert_eq!(count.get(), 1, "gas-free ⇒ always delegate");
}
