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
    let _ = (state, rho, div_v, dt_elapsed, cfg, epoch);
    todo!("S2: implement the star-formation recipe")
}
