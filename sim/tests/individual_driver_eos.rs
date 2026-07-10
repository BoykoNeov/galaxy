//! I8 — the `run_individual` EOS-arm DISPATCH (plan: laddered-ember-cadence.md).
//!
//! `run_individual` is ONE driver that dispatches on `IndividualConfig.eos`
//! ([`ThermalArm`]) to the right physics stepper — `Isothermal` →
//! `individual::ActiveSetKdk` (the frozen I3/I4a/I4b byte-path, `u` untouched),
//! `Adiabatic` → `individual::ActiveSetKdkThermal` (evolves the internal energy `u`
//! and floors the active subset at `u_min`, E4b). This is the driver wiring the I5
//! stepper was left without — NOT a second `run_individual_thermal`.
//!
//! GATE INTENT (advisor-vetted, 2026-07-10): the load-bearing trap is a VACUOUS
//! dispatch test — `ActiveSetKdkThermal` kicks `u` by `du/dt`, but `du/dt` comes from
//! `solver.accel_and_dudt`, and an ISOTHERMAL solver fills `du/dt ≡ 0`. Running an
//! isothermal solver through `eos:Adiabatic` would leave `u` flat and prove nothing.
//! So every gate here uses a REAL ADIABATIC solver (`Eos::Adiabatic`, initial `u > 0`)
//! and drives the SAME solver + IC through BOTH arms:
//!   * `eos:Adiabatic`  ⇒ `u` MUST evolve (PdV work etc.) — the arm was reached.
//!   * `eos:Isothermal` ⇒ `u` MUST stay byte-identical to the input — `ActiveSetKdk`
//!     never touches `u`, so this is the isothermal-arm byte-identity guard.
//! The IC is multi-rung (a dense core + diffuse halo ⇒ an `h` gradient ⇒ a real rung
//! spread), so the active-set mechanic is genuinely exercised, not fixed-dt in
//! disguise. These gate DISPATCH only — stepper correctness is the six I5 gates in
//! `individual_thermal_stepper.rs`, not re-proven here.

use galaxy_core::{DVec3, Species, State, StaticBackground};
use galaxy_io::Header;
use galaxy_sim::{run_individual, IndividualConfig, SimError, SnapshotSink, ThermalArm};
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

const GAMMA: f64 = 1.4;

/// In-memory sink keeping a full f64 copy of every snapshot (positions AND `u`).
#[derive(Default)]
struct CollectingSink {
    snaps: Vec<(Header, State)>,
}
impl SnapshotSink for CollectingSink {
    fn emit(&mut self, header: &Header, state: &State) -> Result<(), SimError> {
        self.snaps.push((header.clone(), state.clone()));
        Ok(())
    }
}

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn ball(rng: &mut impl FnMut() -> f64, n: usize, radius: f64) -> Vec<DVec3> {
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    pos
}

/// A CENTRALLY-CONCENTRATED gas cloud (dense core inside a diffuse halo) with
/// POSITIVE, non-uniform internal energy `u` — the same steep-`h`-gradient testbed
/// the isothermal driver gates use (⇒ a genuine ≥3-rung spread), but adiabatic-ready:
/// `u > 0` gives a real per-particle sound speed for the CFL and for `du/dt`.
fn core_halo_gas(seed: u64) -> State {
    let mut rng = lcg(seed);
    let mut pos = ball(&mut rng, 500, 0.1); // dense core
    pos.extend(ball(&mut rng, 100, 1.0)); // diffuse halo
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vec![DVec3::ZERO; n], vec![1.0; n]);
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    // Warm, positive, non-uniform internal energy.
    s.u = (0..n).map(|i| 0.5 + 0.1 * (i % 7) as f64).collect();
    s
}

fn adiabatic_solver() -> GravitySph<BarnesHut> {
    let params = HydroParams {
        eos: Eos::Adiabatic { gamma: GAMMA },
        ..HydroParams::default()
    };
    GravitySph::<BarnesHut>::hydro_only(params, DensityConfig::default())
}

/// Multi-rung config on the given EOS arm; base-dt cap non-binding, limiter
/// non-binding (this is a DISPATCH gate, not a limiter gate).
fn cfg(eos: ThermalArm, output_dt: f64, n_outputs: u64) -> IndividualConfig {
    IndividualConfig {
        courant: 0.1,
        dt_base_cap: f64::INFINITY,
        r_max: 10,
        n_limit: 10, // == r_max ⇒ limiter non-binding
        eos,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0x0E05,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    }
}

/// Run the SAME adiabatic solver + IC on the given EOS arm, returning the final `u`
/// column and the run summary.
fn run_arm(
    eos: ThermalArm,
    output_dt: f64,
    n_outputs: u64,
) -> (Vec<f64>, galaxy_sim::IndividualSummary) {
    let mut s = core_halo_gas(0x0ADA);
    let mut solver = adiabatic_solver();
    let bg = StaticBackground;
    let mut sink = CollectingSink::default();
    let summary = run_individual(
        &mut s,
        &mut solver,
        &bg,
        &cfg(eos, output_dt, n_outputs),
        &mut sink,
    )
    .unwrap();
    (sink.snaps.last().unwrap().1.u.clone(), summary)
}

// --------------------------------------------------------------------------
// GATE 1 — the EOS arm dispatches to the right stepper.
// --------------------------------------------------------------------------

#[test]
fn adiabatic_arm_evolves_u_isothermal_arm_leaves_it_byte_identical() {
    let u_init = core_halo_gas(0x0ADA).u; // the shared initial internal energy

    // Isothermal arm: ActiveSetKdk never touches `u` ⇒ byte-identical to the input.
    let (u_iso, sum_iso) = run_arm(ThermalArm::Isothermal, 0.05, 1);
    assert!(
        sum_iso.distinct_rungs >= 3,
        "dispatch gate must be multi-rung (got {} rungs) — else active-set is fixed-dt in disguise",
        sum_iso.distinct_rungs
    );
    assert_eq!(
        u_iso, u_init,
        "isothermal arm must leave the internal energy `u` byte-identical to the input"
    );
    assert_eq!(
        sum_iso.u_floor_energy, 0.0,
        "isothermal arm reports no `u`-floor leak"
    );

    // Adiabatic arm: same solver + IC, but now `u` is evolved by `du/dt` (PdV work),
    // so it MUST move off the input. `u_min = 0` ⇒ floor inert (u stays positive).
    let (u_adia, sum_adia) = run_arm(ThermalArm::Adiabatic { u_min: 0.0 }, 0.05, 1);
    assert_eq!(
        sum_adia.distinct_rungs, sum_iso.distinct_rungs,
        "same IC/courant ⇒ same rung structure on both arms"
    );
    let max_du = u_adia
        .iter()
        .zip(&u_init)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_du > 1e-4,
        "adiabatic arm must EVOLVE `u` (max |Δu| = {max_du:e}) — else the thermal \
         stepper was not reached (dispatch failed / `du/dt` ignored)"
    );
    assert_eq!(
        sum_adia.u_floor_energy, 0.0,
        "u_min = 0 floor must stay inert on this positive-u run"
    );
}

// --------------------------------------------------------------------------
// GATE 2 — the adiabatic arm's `u`-floor leak is surfaced in the summary.
// --------------------------------------------------------------------------

#[test]
fn adiabatic_arm_reports_u_floor_leak_in_summary() {
    // A floor ABOVE every particle's initial `u` (≈ 0.5–1.1) is clamped up at the very
    // first kick ⇒ a strictly positive, reported leak — proving the summary field is
    // wired to the dispatched thermal stepper's accumulated `u_floor_energy`, not the
    // hardcoded 0.0 of the isothermal arm.
    let u_min = 5.0;
    let (u_final, summary) = run_arm(ThermalArm::Adiabatic { u_min }, 0.02, 1);

    assert!(
        summary.u_floor_energy > 0.0,
        "a floor above every initial `u` must inject a positive, reported leak (got {})",
        summary.u_floor_energy
    );
    for (i, &u) in u_final.iter().enumerate() {
        assert!(
            u >= u_min - 1e-9,
            "particle {i}: floored `u` = {u} fell below the floor {u_min} at the sync boundary"
        );
    }
}
