//! Star-formation driver-hook gates (plan `natal-ember-forge.md`, S4/F4): the
//! [`form_stars`] operator is applied once per snapshot interval at the
//! output-cadence synchronization site of ALL THREE stepping loops (`run` /
//! `run_adaptive` / `run_individual`), through the single shared
//! `apply_star_formation` helper.
//!
//! The gates:
//! - **SF off (`sf = None`) is byte-identical** — the SF-owned columns (`kind`,
//!   `formation_time`) are untouched on every emitted snapshot of all three loops
//!   (the helper early-returns, so nothing else can change either).
//! - **SF on converts at the FIRST non-IC snapshot, not at the IC** — on a decisive
//!   dense+converging testbed (`p = 1` exactly, via a huge efficiency), the IC
//!   snapshot is still all gas and the first output snapshot has every gas particle
//!   flipped to a star stamped with `formation_time == that snapshot's time`.
//! - **Same conversions across loops** (the trap): the three loops are different
//!   integrators, so their states diverge in the low bits — but the conversion draw
//!   is a pure function of `(id, epoch, seed)`, so on an aligned grid + a decisive
//!   testbed the converted ID SET (not the bit-identical state) is identical across
//!   `run` / `run_adaptive` / `run_individual`.

use std::collections::BTreeSet;

use galaxy_core::{
    DVec3, LeapfrogKdk, ParticleId, Progenitor, Species, State, StaticBackground,
};
use galaxy_io::Header;
use galaxy_sim::{
    run, run_adaptive, run_individual, AdaptiveConfig, IndividualConfig, SimConfig,
    SnapshotSink, StarFormationConfig, ThermalArm,
};
use galaxy_sim::SimError;
use galaxy_solvers::sph::{DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

/// In-memory sink keeping a full f64 copy of every emitted snapshot — SF touches
/// `kind` / `formation_time`, neither of which survives the lossy f32 on-disk form.
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

/// A dense gas ball with a strongly radially CONVERGING velocity field `v = −k·x`
/// (so `∇·v ≪ 0` decisively on every particle) — the "dense AND converging" half of
/// the SF criterion is satisfied with a comfortable margin, robust to the low-bit
/// trajectory divergence between the three loops.
fn converging_gas_blob(seed: u64, n: usize, radius: f64, k: f64) -> State {
    let mut rng = lcg(seed);
    let mut pos = Vec::with_capacity(n);
    while pos.len() < n {
        let p = DVec3::new(rng() - 0.5, rng() - 0.5, rng() - 0.5) * (2.0 * radius);
        if p.length() <= radius {
            pos.push(p);
        }
    }
    let vel: Vec<DVec3> = pos.iter().map(|&p| -k * p).collect();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for kind in s.kind.iter_mut() {
        *kind = Species::Gas;
    }
    s
}

/// The converging gas blob plus a handful of pre-existing collisionless "stars"
/// (already old — `formation_time = PRIMORDIAL`). They are never SF candidates
/// (`kind != Gas`), so they must be EXCLUDED from every converted set identically on
/// all three loops — the non-trivial half of the cross-loop gate.
fn gas_blob_with_stars(seed: u64, n_gas: usize, radius: f64, k: f64, n_star: usize) -> State {
    let mut s = converging_gas_blob(seed, n_gas, radius, k);
    let n0 = s.len();
    for j in 0..n_star {
        s.pos.push(DVec3::new(0.01 * j as f64, 0.0, 0.0));
        s.vel.push(DVec3::ZERO);
        s.mass.push(1.0);
        s.id.push(ParticleId((n0 + j) as u64));
        s.progenitor.push(Progenitor(0));
        s.kind.push(Species::Collisionless);
        s.u.push(0.0);
        s.formation_time.push(State::PRIMORDIAL);
    }
    s
}

fn hydro_solver(c_s: f64) -> GravitySph<BarnesHut> {
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };
    GravitySph::<BarnesHut>::hydro_only(params, DensityConfig::default())
}

/// A decisive SF recipe: `rho_thresh` a hair above zero (every gas particle in the
/// dense blob clears it) and a huge efficiency so `p = 1 − exp(−ε·dt/t_ff) = 1.0`
/// exactly for any positive `dt_elapsed` — every candidate converts deterministically,
/// no sampling, robust to the trajectory micro-divergence between loops.
fn decisive_sf(seed: u64) -> StarFormationConfig {
    StarFormationConfig {
        rho_thresh: 1e-6,
        efficiency: 1e9,
        seed,
    }
}

fn sim_cfg(dt: f64, n_steps: u64, snapshot_every: u64, sf: Option<StarFormationConfig>) -> SimConfig {
    SimConfig {
        dt,
        n_steps,
        snapshot_every,
        softening: 0.05,
        rng_seed: 0x5F04,
        config_hash: 0xC0FFEE,
        units: "nbody-G1".to_string(),
        sf,
    }
}

fn adaptive_cfg(output_dt: f64, n_outputs: u64, sf: Option<StarFormationConfig>) -> AdaptiveConfig {
    AdaptiveConfig {
        courant: 0.25,
        max_growth: 1.25,
        block_steps: 8,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0x5F04,
        config_hash: 0xC0FFEE,
        units: "nbody-G1".to_string(),
        sf,
    }
}

fn individual_cfg(
    output_dt: f64,
    n_outputs: u64,
    sf: Option<StarFormationConfig>,
) -> IndividualConfig {
    IndividualConfig {
        courant: 0.25,
        dt_base_cap: f64::INFINITY,
        r_max: 10,
        n_limit: 10, // == r_max ⇒ limiter non-binding (pure CFL rungs)
        cache_gravity_tree: false,
        subcycle_gravity: false,
        grav_eta: 1.0,
        eos: ThermalArm::Isothermal,
        output_dt,
        n_outputs,
        softening: 0.05,
        rng_seed: 0x5F04,
        config_hash: 0xC0FFEE,
        units: "nbody-G1".to_string(),
        sf,
    }
}

/// The set of particle ids that FORMED via SF this run: collisionless AND carrying a
/// finite `formation_time` (a pre-existing star keeps the `PRIMORDIAL = −∞` sentinel,
/// which `is_finite()` rejects).
fn converted_ids(s: &State) -> BTreeSet<u64> {
    (0..s.len())
        .filter(|&i| s.kind[i] == Species::Collisionless && s.formation_time[i].is_finite())
        .map(|i| s.id[i].0)
        .collect()
}

// ---------------------------------------------------------------------------
// Gate 1: SF off (`sf = None`) is byte-identical on all three loops.
// ---------------------------------------------------------------------------

/// Assert every emitted snapshot left the SF-owned columns untouched: all gas is
/// still gas, all `formation_time` is still `PRIMORDIAL`. Because the helper's None
/// arm is a pure early return, this is the observable face of the full byte-identity.
fn assert_sf_off(snaps: &[(Header, State)]) {
    assert!(!snaps.is_empty(), "expected at least the IC snapshot");
    for (h, st) in snaps {
        assert!(
            st.kind.iter().all(|&k| k == Species::Gas),
            "sf = None must not flip any kind (snapshot step {})",
            h.step
        );
        assert!(
            st.formation_time.iter().all(|&f| f == State::PRIMORDIAL),
            "sf = None must not stamp any formation_time (snapshot step {})",
            h.step
        );
    }
}

#[test]
fn sf_none_is_byte_identity_fixed_dt() {
    let mut state = converging_gas_blob(0xA1, 200, 1.0, 0.3);
    let mut solver = hydro_solver(0.05);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = sim_cfg(0.001, 10, 5, None);
    let mut sink = CollectingSink::default();
    run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    assert_sf_off(&sink.snaps);
}

#[test]
fn sf_none_is_byte_identity_adaptive() {
    let mut state = converging_gas_blob(0xA2, 200, 1.0, 0.3);
    let mut solver = hydro_solver(0.05);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    let cfg = adaptive_cfg(0.01, 2, None);
    let mut sink = CollectingSink::default();
    run_adaptive(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
    assert_sf_off(&sink.snaps);
}

#[test]
fn sf_none_is_byte_identity_individual() {
    let mut state = converging_gas_blob(0xA3, 200, 1.0, 0.3);
    let mut solver = hydro_solver(0.05);
    let bg = StaticBackground;
    let cfg = individual_cfg(0.01, 2, None);
    let mut sink = CollectingSink::default();
    run_individual(&mut state, &mut solver, &bg, &cfg, &mut sink).unwrap();
    assert_sf_off(&sink.snaps);
}

// ---------------------------------------------------------------------------
// Gate 2: SF on converts at the FIRST non-IC snapshot (epoch 1), NOT at the IC,
// stamping `formation_time == that snapshot's time`.
// ---------------------------------------------------------------------------

#[test]
fn sf_some_converts_at_first_snapshot_not_at_ic_fixed_dt() {
    let mut state = converging_gas_blob(0xB1, 200, 1.0, 0.3);
    let n = state.len();
    let mut solver = hydro_solver(0.05);
    let mut integ = LeapfrogKdk::new();
    let bg = StaticBackground;
    // One output interval: n_steps == snapshot_every ⇒ snapshots at step 0 (IC) and
    // step 5 (the first, and only, SF call at epoch 1).
    let cfg = sim_cfg(0.001, 5, 5, Some(decisive_sf(0x1234)));
    let mut sink = CollectingSink::default();
    run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();

    assert_eq!(sink.snaps.len(), 2, "IC + one output snapshot");

    // The IC snapshot: SF has not fired — every particle is still gas, primordial.
    let ic = &sink.snaps[0].1;
    assert!(
        ic.kind.iter().all(|&k| k == Species::Gas),
        "no SF at the IC — all gas"
    );
    assert!(
        ic.formation_time.iter().all(|&f| f == State::PRIMORDIAL),
        "no SF at the IC — all primordial"
    );

    // The first output snapshot: every (dense, converging, p = 1) gas particle has
    // formed a star, stamped with THIS snapshot's time.
    let (h1, snap1) = &sink.snaps[1];
    let t_form = snap1.time;
    assert!(
        snap1.kind.iter().all(|&k| k == Species::Collisionless),
        "every candidate converted at epoch 1 (p = 1)"
    );
    assert!(
        snap1.formation_time.iter().all(|&f| f == t_form),
        "formation_time must equal the snapshot time {t_form} (step {})",
        h1.step
    );
    // Mass & N conserved by the in-place flip.
    assert_eq!(snap1.len(), n, "N unchanged by conversion");
}

// ---------------------------------------------------------------------------
// Gate 3 (THE TRAP): same conversions across the three loops. Assert the converted
// ID SET (state-independent, keyed on (id, epoch, seed)), NOT a bit-identical state.
// ---------------------------------------------------------------------------

#[test]
fn cross_loop_same_conversion_set() {
    // Aligned grid: the fixed-dt path runs 5 steps @ dt = 0.002 (one interval of
    // 0.01); the adaptive/individual paths run one output interval of output_dt =
    // 0.01. So all three fire SF exactly once, at epoch 1, with dt_elapsed = 0.01.
    let dt = 0.002;
    let snapshot_every = 5u64;
    let output_dt = snapshot_every as f64 * dt; // 0.01
    let sf = decisive_sf(0xC0DE);

    // Decisive, non-trivial testbed: a dense converging gas blob (all p = 1 ⇒ all
    // convert) PLUS pre-existing collisionless stars (never candidates ⇒ never in the
    // converted set). The converted set must therefore equal the gas id set on every
    // loop, and must exclude the star ids on every loop.
    let ic = gas_blob_with_stars(0xB2, 200, 1.0, 0.3, 7);
    let gas_ids: BTreeSet<u64> = (0..ic.len())
        .filter(|&i| ic.kind[i] == Species::Gas)
        .map(|i| ic.id[i].0)
        .collect();
    let star_ids: BTreeSet<u64> = (0..ic.len())
        .filter(|&i| ic.kind[i] == Species::Collisionless)
        .map(|i| ic.id[i].0)
        .collect();
    assert_eq!(gas_ids.len(), 200);
    assert_eq!(star_ids.len(), 7);

    let bg = StaticBackground;

    // Fixed-dt.
    let fixed_set = {
        let mut state = ic.clone();
        let mut solver = hydro_solver(0.05);
        let mut integ = LeapfrogKdk::new();
        let cfg = sim_cfg(dt, snapshot_every, snapshot_every, Some(sf));
        let mut sink = CollectingSink::default();
        run(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
        converted_ids(&sink.snaps.last().unwrap().1)
    };

    // Block-adaptive.
    let adaptive_set = {
        let mut state = ic.clone();
        let mut solver = hydro_solver(0.05);
        let mut integ = LeapfrogKdk::new();
        let cfg = adaptive_cfg(output_dt, 1, Some(sf));
        let mut sink = CollectingSink::default();
        run_adaptive(&mut state, &mut solver, &mut integ, &bg, &cfg, &mut sink).unwrap();
        converted_ids(&sink.snaps.last().unwrap().1)
    };

    // Individual per-particle rungs.
    let individual_set = {
        let mut state = ic.clone();
        let mut solver = hydro_solver(0.05);
        let cfg = individual_cfg(output_dt, 1, Some(sf));
        let mut sink = CollectingSink::default();
        run_individual(&mut state, &mut solver, &bg, &cfg, &mut sink).unwrap();
        converted_ids(&sink.snaps.last().unwrap().1)
    };

    // The converted set is exactly the gas ids — on every loop, identically.
    assert_eq!(fixed_set, gas_ids, "fixed-dt converts exactly the gas set");
    assert_eq!(adaptive_set, gas_ids, "adaptive converts exactly the gas set");
    assert_eq!(
        individual_set, gas_ids,
        "individual converts exactly the gas set"
    );
    // And therefore the three loops agree with each other.
    assert_eq!(fixed_set, adaptive_set, "fixed-dt vs adaptive");
    assert_eq!(fixed_set, individual_set, "fixed-dt vs individual");

    // The pre-existing stars were never candidates on any loop.
    assert!(fixed_set.is_disjoint(&star_ids));
    assert!(adaptive_set.is_disjoint(&star_ids));
    assert!(individual_set.is_disjoint(&star_ids));
}
