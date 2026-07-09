//! I4b — the Saitoh–Makino timestep limiter, shock-wakeup gate (CENTRAL correctness,
//! not a dial; plan: laddered-ember-cadence.md I5).
//!
//! A gas particle on a coarse (slow) rung sitting in cold gas that is suddenly hit by
//! a shock from a fast-rung neighbour would, WITHOUT the limiter, step at its stale
//! coarse `dt` straight THROUGH the shock arrival and mis-integrate it — corrupting
//! exactly the shocked-merger gas physics this engine exists to model. The limiter
//! forces any particle more than `n_limit` rungs coarser than a force-coupled
//! neighbour to wake (refine) at the base boundary, early enough to capture the shock.
//!
//! WHY BLOCK-BOUNDARY LIMITING SUFFICES (advisor-vetted + measured, 2026-07-09): the
//! neighbour coupling range (≈ 2h) far exceeds the per-base-block signal travel
//! (≈ courant·h), so the limiter grades fineness outward and wakes a victim many base
//! blocks before the shock physically reaches it — no mid-substep wakeup needed. This
//! holds in the band Mach ∈ [2/courant, ~10/courant]; below it the plain CFL already
//! refines in time (the gate would be vacuous), above it block-boundary grading can't
//! keep up (mid-tick wakeup would be required). This testbed sits in the band:
//! `ratio = shock_speed·dt_base/h ≈ Mach·courant/2 ≈ 1.9` (printed + asserted).
//!
//! THE GATE HAS TEETH ONLY IF LIMITER-OFF MEASURABLY MISSES. Because the CFL signal
//! velocity `v_sig` already carries an approaching neighbour's `−3w` term, own-CFL
//! refines a DIRECT approacher on its own — the limiter's distinct contribution is the
//! extra lead time from MULTI-HOP graded propagation, observable only above the ratio
//! threshold. So this test runs all three arms (fine-courant oracle, limiter-OFF,
//! limiter-ON) and asserts BOTH that OFF misses the oracle (non-vacuous) AND that ON
//! recovers it (the limiter is load-bearing), keyed on captured energy (the plan's
//! wording) with an RMS-position corroborator.

use galaxy_core::{DVec3, ForceSolver, Species, State, StaticBackground};
use galaxy_io::Header;
use galaxy_sim::individual::base_dt;
use galaxy_sim::{run_individual, IndividualConfig, IndividualSummary, SimError, SnapshotSink};
use galaxy_solvers::sph::{density_adaptive, DensityConfig, Eos, GravitySph, HydroParams};
use galaxy_solvers::BarnesHut;

const CS: f64 = 1.0;
const MACH: f64 = 15.0;
const T_END: f64 = 0.1;
const R_MAX: u32 = 12;

/// In-memory sink; the final snapshot state is the run's end state.
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

fn solver() -> GravitySph<BarnesHut> {
    let params = HydroParams {
        eos: Eos::Isothermal { c_s: CS },
        ..HydroParams::default()
    };
    GravitySph::<BarnesHut>::hydro_only(params, DensityConfig::default())
}

/// A high-Mach directional RAM (advisor-designed): a dense fast stream driven into a
/// cold at-rest target slab. The cold slab's back stays undisturbed (coarse rung); the
/// front is shocked as the stream arrives. `is_target[i]` flags the cold slab — the
/// struck region whose energy the gate checks. Isothermal, pure gas, `v_p = Mach·c_s`.
fn ram_ic() -> (State, Vec<bool>) {
    const HT: f64 = 1.0; // transverse half-width
    const LX: f64 = 3.0; // target length in x
    let v_p = MACH * CS;
    let s_t = 0.5; // target spacing
    let s_p = 0.3; // projectile spacing (denser ⇒ a strong, sustained shock)
    let m = 1.0; // equal particle mass

    let axis = |s: f64| -> Vec<f64> {
        let n = (HT / s).floor() as i64;
        (-n..=n).map(|k| k as f64 * s).collect()
    };
    let mut pos = Vec::new();
    let mut vel = Vec::new();
    let mut is_target = Vec::new();

    // Target: cold, at rest, x = 0 .. LX.
    let yt = axis(s_t);
    let nx_t = (LX / s_t).floor() as i64;
    for ix in 0..=nx_t {
        let x = ix as f64 * s_t;
        for &y in &yt {
            for &z in &yt {
                pos.push(DVec3::new(x, y, z));
                vel.push(DVec3::ZERO);
                is_target.push(true);
            }
        }
    }
    // Projectile: dense, moving +x at v_p, just left of the target face.
    let yp = axis(s_p);
    let nx_p = (1.5 / s_p).floor() as i64;
    for ix in 1..=nx_p {
        let x = -(ix as f64) * s_p;
        for &y in &yp {
            for &z in &yp {
                pos.push(DVec3::new(x, y, z));
                vel.push(DVec3::new(v_p, 0.0, 0.0));
                is_target.push(false);
            }
        }
    }
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![m; n]);
    for k in s.kind.iter_mut() {
        *k = Species::Gas;
    }
    (s, is_target)
}

fn cfg(courant: f64, n_limit: u32) -> IndividualConfig {
    IndividualConfig {
        courant,
        dt_base_cap: f64::INFINITY, // non-binding ⇒ dt_base = courant·dt_coarsest
        r_max: R_MAX,
        n_limit,
        output_dt: T_END, // one output interval ⇒ run to exactly T_END
        n_outputs: 1,
        softening: 0.05,
        rng_seed: 0,
        config_hash: 0,
        units: "nbody-G1".to_string(),
    }
}

/// Run the ram IC to `T_END` and return the end state + rung diagnostics.
fn run_ram(ic: &State, courant: f64, n_limit: u32) -> (State, IndividualSummary) {
    let mut s = ic.clone();
    let mut solv = solver();
    let bg = StaticBackground;
    let mut sink = CollectingSink::default();
    let summary =
        run_individual(&mut s, &mut solv, &bg, &cfg(courant, n_limit), &mut sink).unwrap();
    (sink.snaps.last().unwrap().1.clone(), summary)
}

/// Kinetic energy of the struck target subset.
fn target_ke(s: &State, is_target: &[bool]) -> f64 {
    (0..s.len())
        .filter(|&i| is_target[i])
        .map(|i| 0.5 * s.mass[i] * s.vel[i].length_squared())
        .sum()
}

/// RMS per-particle position error over the target subset (index-aligned to `oracle`).
fn target_pos_rms(a: &State, oracle: &State, is_target: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0;
    // Parallel arrays (a.pos, oracle.pos, is_target) indexed by particle — an
    // enumerate over one is no clearer.
    #[allow(clippy::needless_range_loop)]
    for i in 0..a.len() {
        if is_target[i] {
            sum += (a.pos[i] - oracle.pos[i]).length_squared();
            n += 1;
        }
    }
    (sum / n.max(1) as f64).sqrt()
}

#[test]
fn shock_into_slow_rung_region_wakes_and_captures_energy() {
    let (ic, is_target) = ram_ic();

    // The teeth regime is set by ratio = shock_speed·dt_base/h ≈ Mach·courant/2. At
    // the coarse courant the shock crosses a coupling range within one base block, so
    // own-CFL is a block too late and only the limiter's graded wakeup saves the
    // victim. Assert the testbed actually sits above the threshold (else vacuous).
    let coarse_courant = 0.2;
    let dt_pp = solver().max_stable_dt_per_particle(&ic);
    let dt_base = base_dt(&dt_pp, coarse_courant, f64::INFINITY);
    let dens = density_adaptive(&ic.pos, &ic.mass, &DensityConfig::default(), None);
    let h_target = {
        let mut hs: Vec<f64> = (0..ic.len())
            .filter(|&i| is_target[i])
            .map(|i| dens.h[i])
            .collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        hs[hs.len() / 2]
    };
    let ratio = MACH * CS * dt_base / h_target;
    println!("ratio = shock_speed·dt_base/h = {ratio:.3} (Mach {MACH}, dt_base {dt_base:.4})");
    assert!(
        ratio > 1.2,
        "testbed must sit in the teeth band (ratio {ratio:.3} > 1.2) — else own-CFL \
         refines in time and the limiter is untested"
    );
    // (Above ~10/courant the argument inverts — block-boundary grading can't keep up
    // and mid-tick wakeup would be required; this testbed stays in band.)

    // Oracle: fine courant, limiter non-binding. At this courant ratio ≈ 0.4 < 1, so
    // own-CFL refines the shock in time and the limiter is irrelevant — a
    // limiter-independent fully-resolved reference (its convergence was verified
    // out-of-band: |courant 0.05 − 0.025| ≈ 1.4e-3, small vs the errors below).
    let (oracle, ora_sum) = run_ram(&ic, 0.05, R_MAX);
    assert!(
        ora_sum.max_rung < R_MAX,
        "oracle finest rung {} must stay < r_max — reference must not be clamped",
        ora_sum.max_rung
    );
    let ora_ke = target_ke(&oracle, &is_target);

    // Limiter OFF (n_limit = r_max ⇒ non-binding) and ON (n_limit = 1) at the coarse
    // courant. Both must genuinely be multi-rung (else active-subset ≡ fixed-dt).
    let (off, off_sum) = run_ram(&ic, coarse_courant, R_MAX);
    let (on, _on_sum) = run_ram(&ic, coarse_courant, 1);
    assert!(
        off_sum.distinct_rungs >= 3,
        "coarse run must be multi-rung (got {}) — else this is fixed-dt in disguise",
        off_sum.distinct_rungs
    );

    let off_ke_err = (target_ke(&off, &is_target) - ora_ke).abs();
    let on_ke_err = (target_ke(&on, &is_target) - ora_ke).abs();
    let off_rms = target_pos_rms(&off, &oracle, &is_target);
    let on_rms = target_pos_rms(&on, &oracle, &is_target);
    println!(
        "oracle target-KE {ora_ke:.2}  |  OFF KE-err {off_ke_err:.2} ({:.2}%) RMS {off_rms:.3e}  |  \
         ON KE-err {on_ke_err:.2} ({:.2}%) RMS {on_rms:.3e}  |  KE-teeth {:.1}x RMS-teeth {:.1}x",
        100.0 * off_ke_err / ora_ke,
        100.0 * on_ke_err / ora_ke,
        off_ke_err / on_ke_err.max(1e-30),
        off_rms / on_rms.max(1e-30),
    );

    // TEETH (the gate is non-vacuous): limiter-OFF measurably MISSES the oracle — the
    // slow-rung target particles step through the shock at the wrong dt.
    assert!(
        off_ke_err > 0.02 * ora_ke,
        "limiter-OFF must mis-capture the shock energy (err {off_ke_err:.2} > 2% of {ora_ke:.2}) \
         — else the testbed has no teeth"
    );
    assert!(
        off_rms > 3e-3,
        "limiter-OFF target RMS {off_rms:.3e} must be sizeable"
    );

    // CENTRAL correctness: limiter-ON WAKES the struck particles and captures the same
    // energy as the fully-fine reference — close in absolute terms AND dramatically
    // closer than OFF (the load-bearing "the limiter recovers it" signal).
    assert!(
        on_ke_err < 0.012 * ora_ke,
        "limiter-ON must capture the shock energy within 1.2% of the reference (err \
         {on_ke_err:.2} = {:.2}%)",
        100.0 * on_ke_err / ora_ke
    );
    assert!(
        on_ke_err * 3.0 < off_ke_err,
        "limiter-ON must be ≥3× closer in energy than OFF: ON {on_ke_err:.2} vs OFF {off_ke_err:.2}"
    );
    assert!(
        on_rms * 1.5 < off_rms,
        "limiter-ON target positions must track the reference far better than OFF: \
         ON RMS {on_rms:.3e} vs OFF RMS {off_rms:.3e}"
    );
}
