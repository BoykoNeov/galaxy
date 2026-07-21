//! Physical star formation: dense, converging SPH gas genuinely converts to
//! collisionless star particles that carry a **formation time** (plan
//! `natal-ember-forge.md`, Chain A step 4). The render later colors young stars
//! by age `now − formation_time` (renderprep, milestone S5).
//!
//! This module is the pure, loop-agnostic OPERATOR (S2): [`form_stars`] takes a
//! state plus the SPH fields it needs (`ρ`, `∇·v`) and applies the recipe
//! in-place. The stepping driver calls it once per snapshot interval at the
//! output-cadence synchronization site (S4); the SPH fields come from a
//! D2-clean transient solver accessor (`ForceSolver::sf_fields`, S3).
//!
//! ## The load-bearing choice: whole-particle IN-PLACE conversion
//! A forming star reuses the gas particle's slot — flip `kind: Gas →
//! Collisionless`, stamp `formation_time`, zero the now-inert `u`. No new
//! particle is spawned, so N and total mass are conserved EXACTLY for free, and
//! there is no snapshot / renderprep / GPU-buffer resize cascade. Conversion is
//! one-way and touches exactly three columns (`kind`, `formation_time`, `u`);
//! `pos` / `vel` / `mass` / `id` / `progenitor` are untouched (the formed star
//! keeps its gas provenance tag — young/old is `formation_time`, not a new tag).

use galaxy_core::{Species, State};

/// Tunable star-formation recipe (F2/F7). Mirrors the `[physics.star_formation]`
/// scenario section 1:1 (`rho_thresh`, `efficiency`, `seed`); absent ⇒ SF is
/// `None` ⇒ every existing byte-path is untouched.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarFormationConfig {
    /// Density threshold `ρ_thresh`: gas below this never forms stars (the
    /// "dense" half of the two-part SF criterion). Must be `> 0`.
    pub rho_thresh: f64,
    /// Dimensionless efficiency `ε` per free-fall time. `0.0` ⇒ no conversions
    /// ever (`p = 1 − exp(0) = 0`), the SF-off gate.
    pub efficiency: f64,
    /// Global seed for the deterministic conversion draw (F3). Same seed ⇒ same
    /// conversion set, independent of particle ordering or thread scheduling.
    pub seed: u64,
}

/// What one [`form_stars`] call did — for run diagnostics and the monotonicity
/// gate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FormationSummary {
    /// Number of gas particles converted to stars this call.
    pub n_formed: usize,
    /// Total mass converted this call (`Σ mass[i]` over the converted set).
    pub mass_formed: f64,
}

/// Apply the star-formation recipe in place (F2). Pure w.r.t. its inputs: given
/// the same `(state, rho, div_v, dt_elapsed, cfg, epoch)` it makes the same
/// conversions, independent of iteration order or threading.
///
/// For each particle `i`:
/// 1. **Candidate** iff `kind[i] == Gas && rho[i] >= cfg.rho_thresh &&
///    div_v[i] < 0` (dense AND converging — the two-part standard SF criterion).
/// 2. **Probability** `p_i = 1 − exp(−ε · dt_elapsed / t_ff(ρ_i))`, with the
///    local free-fall time `t_ff(ρ) = √(3π / (32 G ρ))`, `G = 1` (N-body units).
///    The `1 − exp` form saturates at 1 for large `dt_elapsed` (SF fires at
///    snapshot cadence, not per step).
/// 3. **Convert** iff a deterministic uniform draw keyed on
///    `(id[i], epoch, cfg.seed)` is `< p_i`: flip `kind[i] → Collisionless`, set
///    `formation_time[i] = state.time`, zero the now-inert `u[i]`.
///
/// `epoch` is the SF-call index (== snapshot index); successive calls draw
/// independent substreams for the same particle. `rho` / `div_v` are parallel
/// arrays indexed like the SoA columns (length `state.len()`).
pub fn form_stars(
    state: &mut State,
    rho: &[f64],
    div_v: &[f64],
    dt_elapsed: f64,
    cfg: &StarFormationConfig,
    epoch: u64,
) -> FormationSummary {
    let n = state.len();
    debug_assert_eq!(rho.len(), n, "rho must be a per-particle array");
    debug_assert_eq!(div_v.len(), n, "div_v must be a per-particle array");

    let mut n_formed = 0usize;
    let mut mass_formed = 0.0;
    for i in 0..n {
        // Two-part standard SF criterion: gas that is BOTH dense AND converging.
        if state.kind[i] != Species::Gas || rho[i] < cfg.rho_thresh || div_v[i] >= 0.0 {
            continue;
        }
        // p = 1 − exp(−ε·dt/t_ff). At ε=0 this is 0; at ρ→∞ (or dt→∞) it
        // saturates at 1 (draws are in [0,1), so p can never be exceeded).
        let p = 1.0 - (-cfg.efficiency * dt_elapsed / t_ff(rho[i])).exp();
        if draw_uniform(state.id[i].0, epoch, cfg.seed) < p {
            state.kind[i] = Species::Collisionless;
            state.formation_time[i] = state.time;
            state.u[i] = 0.0; // gravity-only rows carry no internal energy
            n_formed += 1;
            mass_formed += state.mass[i];
        }
    }
    FormationSummary {
        n_formed,
        mass_formed,
    }
}

/// Local free-fall time `t_ff(ρ) = √(3π / (32 G ρ))`, with `G = 1` in the
/// project's N-body unit system ([`galaxy_xtask::G`]). Thread `G` through only
/// if a non-unit-`G` run is ever added (there is none today, and `efficiency` —
/// tuned empirically in the F7 A/B — absorbs any constant rescaling of `t_ff`).
fn t_ff(rho: f64) -> f64 {
    (3.0 * std::f64::consts::PI / (32.0 * rho)).sqrt()
}

/// Deterministic uniform in `[0, 1)` keyed on `(id, epoch, seed)` (F3). A pure
/// function of the keys — NOT a shared RNG advanced in iteration order — so the
/// conversion set is identical under `rayon` and under the active-subset order
/// of `run_individual`. Two nested SplitMix64 finalizer steps (the project's
/// tiny deterministic mixer; see `ic::disk::SplitMix64`) fully diffuse each key
/// before it folds into the next, and the final `>>11 / 2^53` matches
/// `SplitMix64::next_f64` exactly, giving a value in `[0, 1)`.
fn draw_uniform(id: u64, epoch: u64, seed: u64) -> f64 {
    let z = splitmix64_finalize(seed ^ id);
    let z = splitmix64_finalize(z ^ epoch);
    ((z >> 11) as f64) / ((1u64 << 53) as f64)
}

/// One SplitMix64 finalizer (the avalanche stage of the project's PRNG). Full
/// bit diffusion of its input; identical constants to `ic::disk::SplitMix64`.
fn splitmix64_finalize(x: u64) -> u64 {
    let z = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
