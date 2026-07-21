//! `scenario.toml` front-end (M6f): the declarative schema that turns the
//! hardcoded movie scenarios into data, plus the builder that turns a parsed
//! [`ScenarioSpec`] into the runtime [`Scenario`] bundle the pipeline consumes.
//!
//! The three original scenarios (`disk`, `dm`, `cuspy`) are checked-in presets
//! under `xtask/scenarios/`, gated to reproduce the pre-M6f hardcoded
//! constructors exactly (same params, same seed ⇒ same movie). The Toomre
//! encounter zoo (`retro`, `inclined`, `bullseye`, `minor`) rides the same
//! schema — mostly config, exactly as DESIGN promised.
//!
//! Schema stance: **physics is gated, aesthetics are data.** Parsing validates
//! everything the IC constructors would assert on (so a bad toml fails with a
//! readable message, not a panic deep in `galaxy-ic`), plus the cross-field
//! invariants the pipeline relies on (palette/ramp lengths matching the
//! progenitor count, sf progenitors in range). Unknown keys are rejected — a
//! typo'd knob must fail loudly, not silently do nothing.

use serde::Deserialize;

use galaxy_core::{Species, State};
use galaxy_grade::LocalToneConfig;
use galaxy_ic::{
    DiskCollision, ExponentialDisk, Nfw, NfwCollision, Orientation, Plummer, SphericalHalo,
    TruncatedNfw,
};
use galaxy_render::ShadowBake;
use galaxy_renderprep::{AgeColoring, ColorMode, DensityColoring, PrepConfig, SizeByDensity};
use galaxy_sim::StarFormationConfig;

use crate::{
    DEFAULT_LOCAL_FLOOR, DEFAULT_LOCAL_RADIUS, DENSITY_K, DENSITY_STRENGTH, FRAME_H, FRAME_W, G,
    PEAK_BRIGHTNESS, QUICK_H, QUICK_W, SIZE_MAX_FRAC, SIZE_MIN_FRAC, SUBFRAMES,
};

/// A parsed, validated scenario description — everything a movie needs that is
/// *data*: the IC (galaxy models + spin-orbit orientations + particle counts),
/// the orbit, the sim timing, the look, and the camera choreography.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScenarioSpec {
    /// Scenario name (for presets, gated to match the registry key).
    pub name: String,
    /// The IC sampling seed — same spec + same seed ⇒ bit-identical movie.
    pub seed: u64,
    /// Galaxy models, orientations, and particle counts.
    pub model: ModelSpec,
    /// The two-body Kepler encounter of the galaxy COMs.
    pub orbit: OrbitSpec,
    /// Integrator timing, snapshot cadence, and force softening.
    pub sim: SimSpec,
    /// Splat look: palette, ramps, SF mask, framing percentile.
    pub look: LookSpec,
    /// Camera choreography (M6d rig).
    pub rig: RigSpec,
    /// Gated physics knobs (`[physics]`) — opt-in, not aesthetic data. Currently
    /// just star formation (`[physics.star_formation]`, natal-ember-forge F7).
    /// Absent ⇒ no SF ⇒ every existing byte-path is untouched (the SF-off gate).
    #[serde(default)]
    pub physics: Option<PhysicsSpec>,
}

/// Gated physics knobs (`[physics]`, natal-ember-forge F7). Opt-in — absent ⇒ the
/// pre-SF pipeline, byte-for-byte. A container for the (currently single) gated
/// physics feature so more can be added under `[physics.*]` without touching the
/// top-level schema.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhysicsSpec {
    /// Star formation (`[physics.star_formation]`): dense converging SPH gas
    /// converts in place to collisionless star particles carrying a formation
    /// time. Absent ⇒ no SF. Meaningful only on a gas-rich scenario.
    #[serde(default)]
    pub star_formation: Option<StarFormationSpec>,
}

/// The star-formation recipe (`[physics.star_formation]`, natal-ember-forge F7):
/// the TOML spelling of [`galaxy_sim::StarFormationConfig`]. All three knobs are
/// required — load-bearing physics, no aesthetic default. Meaningful only on a
/// gas-rich scenario; declared on a gas-free model it is a dead knob and rejected
/// loud.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StarFormationSpec {
    /// Density threshold `ρ_thresh`: gas below this never forms stars (the "dense"
    /// half of the two-part criterion). Must be finite and `> 0`.
    pub rho_thresh: f64,
    /// Dimensionless efficiency `ε` per free-fall time. Must be finite and `> 0`
    /// (absence already means "off" — a declared `0` is a no-op, rejected loud).
    pub efficiency: f64,
    /// Global seed for the deterministic conversion draw. Same seed ⇒ same
    /// conversion set, independent of particle ordering or thread scheduling.
    pub seed: u64,
}

/// Which collision IC the scenario builds. The variants mirror the three IC
/// families the engine has: warm/cold exponential disks in cored Plummer halos,
/// disks in cuspy truncated-NFW halos, and the pure dark-matter NFW merger.
///
/// Deserialized via [`ModelTable`] rather than serde's internal tagging: tagged
/// enums cannot `deny_unknown_fields`, and a typo'd knob inside `[model]` must
/// fail loudly, not silently do nothing.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(try_from = "ModelTable")]
pub enum ModelSpec {
    /// Two exponential disks in live Plummer halos (`DiskCollision<Plummer>`),
    /// optionally with an isothermal SPH gas component (M7c). One `[model.gas]`
    /// table applies the **same** gas fraction and sound speed to *both* disks:
    /// the isothermal solver's `c_s` is a single global, so two gas populations
    /// physically cannot carry different sound speeds in one run.
    DiskPlummer {
        galaxy1: DiskGalaxySpec<PlummerSpec>,
        galaxy2: DiskGalaxySpec<PlummerSpec>,
        counts: Counts<DiskCounts>,
        /// The shared gas component, or `None` for a purely stellar encounter.
        gas: Option<GasSpec>,
    },
    /// Two exponential disks in live truncated-NFW halos
    /// (`DiskCollision<TruncatedNfw>`) — the cusp-resolution rule (M5f) applies.
    DiskNfw {
        galaxy1: DiskGalaxySpec<NfwSpec>,
        galaxy2: DiskGalaxySpec<NfwSpec>,
        counts: Counts<DiskCounts>,
    },
    /// Two truncated-NFW halos, no disks (`NfwCollision`) — spherical, so no
    /// spin-orbit orientation exists.
    NfwMerger {
        galaxy1: NfwSpec,
        galaxy2: NfwSpec,
        counts: Counts<MergerCounts>,
    },
}

/// One disk galaxy: the exponential-disk parameters, its halo, and its Toomre
/// spin-orbit orientation relative to the orbital plane (degrees; both default
/// to 0 = coplanar prograde).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskGalaxySpec<H> {
    /// Total disk mass (submaximal: ≪ the halo mass).
    pub disk_mass: f64,
    /// Radial exponential scale length Rd.
    pub scale_length: f64,
    /// Vertical sech² scale height as a *fraction of Rd*.
    pub hz_frac: f64,
    /// Truncation radius as a *fraction of Rd* (must exceed 1).
    pub rmax_frac: f64,
    /// Optional Toomre-Q warmth; omit for the fully-cold disk (required at an
    /// NFW cusp, where the warm dispersion diverges — DESIGN M5f).
    #[serde(default)]
    pub toomre_q: Option<f64>,
    /// The live spherical halo the disk is embedded in.
    pub halo: H,
    /// Tilt of the disk spin off the orbital angular momentum (+Z), in degrees,
    /// about the line of nodes. 0 = prograde, 180 = retrograde, 90 = spin in
    /// the orbital plane.
    #[serde(default)]
    pub inclination_deg: f64,
    /// Azimuth of the line of nodes in the orbital plane, in degrees.
    #[serde(default)]
    pub argument_deg: f64,
}

/// A cored Plummer halo (`Plummer::new(G, mass, scale)`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlummerSpec {
    pub mass: f64,
    pub scale: f64,
}

/// An exponentially-truncated NFW halo
/// (`TruncatedNfw::new(Nfw::new(G, mvir, rs, c), skirt)`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NfwSpec {
    pub mvir: f64,
    pub rs: f64,
    pub c: f64,
    /// Exponential skirt scale length r_d of the truncation.
    pub skirt: f64,
}

/// An isothermal SPH gas component shared by both disks of a `disk-plummer`
/// encounter (M7c). The `fraction` of each disk's mass is re-tagged as gas with
/// the given `sound_speed`; the total disk mass and rotation curve are unchanged
/// (gas traces the same exponential profile). The one `sound_speed` is threaded
/// to both the IC's pressure equilibrium *and* the force solver's `HydroParams`
/// — the isothermal EOS `P = c_s²ρ` uses a single global `c_s`.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GasSpec {
    /// Gas mass fraction f_gas = M_gas/M_disk of each disk, in (0, 1).
    pub fraction: f64,
    /// Isothermal sound speed c_s (also the solver's `HydroParams.sound_speed`).
    pub sound_speed: f64,
    /// Adiabatic index γ (incandescent-nebular-veil, H5-C). `None` = isothermal
    /// (`P = c_s²ρ`, the M7c default — byte-unchanged). `Some(γ > 1)` switches the
    /// gas to the ideal-gas adiabatic EOS `P = (γ−1)ρu`, evolving each gas
    /// particle's internal energy so shock heating (e.g. at pericenter) shows up
    /// as temperature. `sound_speed` then seeds the initial `u = c_s²/(γ−1)`, so
    /// the t=0 pressure `(γ−1)ρu` equals the isothermal equilibrium `c_s²ρ` the
    /// sampler bakes — the disk starts in true force balance (no spurious startup
    /// contraction/heating), only real shocks light up.
    #[serde(default)]
    pub gamma: Option<f64>,
    /// Positive internal-energy floor `u_min` for the adiabatic thermal integrator
    /// (`u ← max(u, u_min)` after each half-kick), guarding a rarefaction
    /// undershoot into negative `u`/NaN `c_s` on a strong merger. `None` = 0.0
    /// (inert). Meaningful only with `gamma`.
    #[serde(default)]
    pub u_floor: Option<f64>,
}

/// Full-resolution vs QUICK-preview particle counts.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts<T> {
    pub full: T,
    pub quick: T,
}

/// Particle counts for a two-disk-galaxy encounter (four stellar species, plus
/// up to two gas species when the model carries a `[model.gas]` table).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskCounts {
    pub halo1: usize,
    pub disk1: usize,
    pub halo2: usize,
    pub disk2: usize,
    /// Gas particles in galaxy 1 (ignored, defaults to 0, when the model is
    /// gas-free — a gas-free `disk-plummer` never reads these).
    #[serde(default)]
    pub gas1: usize,
    /// Gas particles in galaxy 2 (same gas-free default rule as `gas1`).
    #[serde(default)]
    pub gas2: usize,
}

impl DiskCounts {
    /// Total particle count (gas counts are 0 for a gas-free encounter).
    pub fn total(&self) -> usize {
        self.halo1 + self.disk1 + self.halo2 + self.disk2 + self.gas1 + self.gas2
    }
}

/// Particle counts for a halo-halo merger (two species).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MergerCounts {
    pub halo1: usize,
    pub halo2: usize,
}

impl MergerCounts {
    /// Total particle count.
    pub fn total(&self) -> usize {
        self.halo1 + self.halo2
    }
}

/// The relative two-body Kepler orbit of the galaxy COMs.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OrbitSpec {
    /// e > 0: e = 1 parabolic (the classic Toomre encounter), e < 1 bound.
    pub eccentricity: f64,
    /// Closest COM–COM approach (> 0).
    pub pericenter: f64,
    /// Initial COM–COM separation (≥ pericenter; ≤ apocenter when bound).
    pub separation: f64,
}

/// Integrator timing, snapshot cadence, and force softening.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SimSpec {
    pub dt: f64,
    pub n_steps: u64,
    pub snapshot_every: u64,
    /// Optional coarser cadence for QUICK previews (defaults to
    /// `snapshot_every`).
    #[serde(default)]
    pub snapshot_every_quick: Option<u64>,
    /// Plummer force softening ε (also the kNN distance floor in renderprep).
    pub eps: f64,
    /// Block-adaptive timestepping on the gas path (`[sim.adaptive]`, plan
    /// courant-quickening-cadence). Its **presence** enables it; all knobs default.
    /// When enabled and the scenario carries gas, the simulate step chooses `dt` per
    /// block from the hydro CFL bound instead of using the fixed `dt` — so a run whose
    /// fixed `dt` would trip the CFL sentinel instead completes, tracking the bound
    /// down automatically. Snapshots stay on the SAME time grid
    /// (`output_dt = snapshot_every · dt`), so `dt` now only sets the output cadence;
    /// the actual substep is CFL-derived. A gas-free scenario ignores it (no hydro
    /// constraint), keeping its fixed-dt byte-identity.
    #[serde(default)]
    pub adaptive: Option<AdaptiveSpec>,
    /// Individual (per-particle rung) timestepping on the gas path (`[sim.individual]`,
    /// plan laddered-ember-cadence). Its **presence** enables it at `mode = hydro-only`;
    /// all knobs default. Mutually exclusive with `[sim.adaptive]` (both size the gas
    /// dt). A gas-free scenario ignores it. When enabled on a gas-rich scenario, the
    /// simulate step routes the gas path through `run_individual` (CPU only).
    #[serde(default)]
    pub individual: Option<IndividualSpec>,
}

/// Block-adaptive timestep policy (`[sim.adaptive]`). All fields default, so an empty
/// `[sim.adaptive]` table enables it at the shipped defaults.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AdaptiveSpec {
    /// Courant number applied to the CFL limit (0, 1). Default 0.25.
    #[serde(default = "default_courant")]
    pub courant: f64,
    /// Per-block dt growth cap (≥ 1) — dt shrinks instantly, grows gradually. Default 1.25.
    #[serde(default = "default_max_growth")]
    pub max_growth: f64,
    /// Max steps held at one dt before re-querying the bound (≥ 1, ≤ GPU MAX_BATCH 64).
    /// Default 16.
    #[serde(default = "default_block_steps")]
    pub block_steps: u64,
}

fn default_courant() -> f64 {
    0.25
}
fn default_max_growth() -> f64 {
    1.25
}
fn default_block_steps() -> u64 {
    16
}

/// Which driver path the individual-timestep toggle selects (`[sim.individual].mode`,
/// plan laddered-ember-cadence). A LAYERED toggle: each mode is droppable to the one
/// below, so gravity subcycling can be turned off independently of hydro rungs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum IndividualMode {
    /// No subcycling — the global fixed-dt `run` path (the layer below `hydro-only`);
    /// lets a scenario keep the `[sim.individual]` table while disabling the toggle.
    FixedDt,
    /// Per-particle hydro (SPH-CFL) rungs — collisionless stars ride the coarsest rung
    /// (lever a). The default: presence enables the primary lever, as `[sim.adaptive]`
    /// enables adaptive at its defaults.
    #[default]
    HydroOnly,
    /// `hydro-only` PLUS gravity subcycling (lever b, I-grav): collisionless stars get
    /// finite gravitational rungs and the gravity WALK runs against a once-per-base-block
    /// cached (stale) tree. Built (M9–M12) and wired.
    ///
    /// ⚠ **FLOODS at full res — no perf benefit at today's N.** The FULL M-validate
    /// (2026-07-11) found the stale tree drives the merger core into a sustained
    /// finest-rung flood (min-dt below the fresh-hydro-only floor, CFL range ballooning
    /// past 100× toward the cached-flood 196×) → SLOWER than `hydro-only`, not faster.
    /// The QUICK 2.55× speedup was a QUICK-only artifact (QUICK never reaches the
    /// supersonic pericenter infall). The flooded trajectory is convergent-but-coarse
    /// (O(courant) stale-COM error) — slow-and-imprecise, not incorrect. Retained as an
    /// opt-in toggle for scaling/completeness; **prefer `hydro-only` for production runs.**
    /// See docs/plans/laddered-ember-cadence.md (M-validate FULL).
    #[serde(rename = "hydro+gravity")]
    HydroGravity,
}

/// Individual (per-particle power-of-two rung) timestep policy (`[sim.individual]`,
/// plan laddered-ember-cadence). All knobs default, so an empty `[sim.individual]`
/// table enables the toggle at `mode = hydro-only` with the shipped defaults. Mutually
/// exclusive with `[sim.adaptive]` (both size the gas dt — declaring both is rejected).
/// A gas-free scenario ignores it (no hydro CFL constraint), keeping its fixed-dt path.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndividualSpec {
    /// Which driver path (see [`IndividualMode`]). Default `hydro-only`.
    #[serde(default)]
    pub mode: IndividualMode,
    /// Courant number applied to each particle's CFL limit (0, 1]. Default 0.25.
    #[serde(default = "default_courant")]
    pub courant: f64,
    /// Maximum rung depth: the finest sub-step is `dt_base / 2^r_max`. A particle
    /// needing a finer step CLAMPS here (bounded under-resolution, no red gate — the
    /// tracked saturation hazard), so it caps the fine-tick count `2^r_max` per block
    /// against a runaway deep rung. Default 10 (`≤ 1024` fine ticks/block). Must be in
    /// [1, 60].
    #[serde(default = "default_r_max")]
    pub r_max: u32,
    /// Saitoh–Makino limiter depth: no gas particle may sit more than `n_limit` rungs
    /// coarser than a force-coupled neighbour (CORRECTNESS — a slow particle struck by
    /// a shock from a fast neighbour must wake before mis-integrating it). Default 1
    /// (the limiter is NOT optional on a showpiece). `n_limit >= r_max` disables it.
    #[serde(default = "default_n_limit")]
    pub n_limit: u32,
    /// Cap on the base (coarsest-rung) timestep `dt_base` (> 0, may be `inf`). Default
    /// `inf` (non-binding): `dt_base` is then sized by `courant·dt_coarsest` clamped to
    /// the output interval. A finite cap keeps the diffuse majority's coarse step
    /// bounded below a scenario ceiling.
    #[serde(default = "default_dt_base_cap")]
    pub dt_base_cap: f64,
}

fn default_r_max() -> u32 {
    10
}
fn default_n_limit() -> u32 {
    1
}
fn default_dt_base_cap() -> f64 {
    f64::INFINITY
}

/// Splat look and framing.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookSpec {
    /// Base splat radius in world units.
    pub splat_size: f32,
    /// Optional screen-space cap on the splat half-extent in PIXELS
    /// (pinprick-starfield): world-sized splats balloon when the breathing rig
    /// zooms into a compact early scene — the cap keeps stars point-like at any
    /// zoom, flux-conserving (clamping down concentrates emission). Absent =
    /// off = bit-identical to the uncapped M6g render. Pixel units are
    /// resolution-literal: FULL 1080p wants ~3× the QUICK 360p value.
    #[serde(default)]
    pub max_splat_px: Option<f32>,
    /// Percentile radius the camera frames on (0, 1].
    pub frame_percentile: f32,
    /// Per-progenitor base colors — length must equal the model's progenitor
    /// count (4 for disk models, 2 for the merger).
    pub palette: Vec<[f32; 3]>,
    /// Per-progenitor `--color initial-radius` ramps — same length rule.
    pub ramps: Vec<RampSpec>,
    /// Progenitors the star-formation compression proxy applies to.
    pub sf_progenitors: Vec<u16>,
    /// Volumetric gas look (`[look.gas]`, M7f). Present **iff** the model carries
    /// `[model.gas]`: a gas-free scenario must not declare it (a dead
    /// volumetric look → loud reject), and a gas-rich one that omits it renders
    /// with [`GasLookValues::default`].
    #[serde(default)]
    pub gas: Option<GasLookSpec>,
    /// Local (spatially-adaptive) tone compression (`[look.local_tone]`,
    /// render-more-controls): the fix for the additive-splat "white-blob" on the
    /// approach. A **grade** knob (whole-frame, linear-HDR) — NOT gas-specific and
    /// NOT gated on scattering — so it lives at the `[look]` level beside
    /// `[look.gas]`, and the movie pipeline bakes it into the grade's
    /// [`galaxy_grade::GradeConfig::local`]. Omitted = off = bit-identical to the
    /// pre-tonemap grade.
    #[serde(default)]
    pub local_tone: Option<LocalToneSpec>,
    /// Age-based star coloring (`[look.age]`, natal-ember-forge F6/F7): tints stars
    /// formed via `[physics.star_formation]` toward `young`, fading back to base
    /// over `tau` sim-time (`t = strength · exp(−age/tau)`). A `[look]` coloring
    /// knob (like the SF compression proxy), NOT gas-gated — inert when no star
    /// carries a formation time (every primordial row ⇒ base color, bit-exact), so
    /// it is meaningful with SF in this run *or* on reused SF snapshots. Omitted ⇒
    /// no age tint (bit-identical). `strength = 0` is a no-op, rejected loud.
    #[serde(default)]
    pub age: Option<AgeColoringSpec>,
}

/// The age-coloring look (`[look.age]`, natal-ember-forge F6): the TOML spelling of
/// [`galaxy_renderprep::AgeColoring`]. Tints newly-formed stars toward `young`,
/// fading over `tau`. `strength = 0` (a bit-exact no-op) and a non-positive `tau`
/// are rejected loud (absence already means "off").
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AgeColoringSpec {
    /// The "young population" color a just-formed star tints toward (linear RGB;
    /// components finite and `≥ 0`).
    pub young: [f32; 3],
    /// Saturation: a freshly-formed star shifts up to `strength` of the way to
    /// `young` (clamped to `[0, 1]` in the map). Must be finite and `> 0`.
    pub strength: f32,
    /// Fade timescale in sim-time — the tint decays as `exp(−age / tau)`. Must be
    /// finite and `> 0`.
    pub tau: f64,
}

/// The local tone-compression look (`[look.local_tone]`): the TOML spelling of
/// [`galaxy_grade::LocalToneConfig`]. Only `strength` is required; `radius`
/// (Gaussian surround σ, pixels) and `floor` (gain floor) default to the same
/// values the `regrade --local` CLI uses ([`crate::DEFAULT_LOCAL_RADIUS`] /
/// [`crate::DEFAULT_LOCAL_FLOOR`]), so `strength = k` alone reproduces
/// `regrade --local k`. A declared `strength = 0` is a bit-exact no-op and is
/// rejected loud (absence already means "off").
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalToneSpec {
    /// Compression strength `k` in `g = 1/(1 + k·V)`. Must be finite and `> 0`.
    pub strength: f32,
    /// Gaussian σ of the surround low-pass, in PIXELS. Resolution-literal (tuned
    /// on QUICK 360p; a FULL 1080p render wants ~3× this, like `max_splat_px`).
    /// Omitted ⇒ [`crate::DEFAULT_LOCAL_RADIUS`].
    #[serde(default)]
    pub radius: Option<f32>,
    /// Gain floor `g_min ∈ [0, 1]` — the hardest the operator may darken a pixel,
    /// bounding the dark-halo ring. Omitted ⇒ [`crate::DEFAULT_LOCAL_FLOOR`].
    #[serde(default)]
    pub floor: Option<f32>,
}

/// The volumetric gas look (`[look.gas]`, M7f): the emission/absorption knobs the
/// raymarcher applies to the density grid. Only the three aesthetic knobs are
/// data; the grid **resolution** is a perf/quality global (the 64³ QUICK / 128³
/// full constants, like the frame dimensions), not a per-scenario aesthetic.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GasLookSpec {
    /// Linear-RGB tint of the gas emission.
    pub color: [f32; 3],
    /// Emissivity `j`: emitted radiance per unit (ρ · path length).
    pub emissivity: f32,
    /// Opacity `κ`: extinction per unit (ρ · path length). `0` = emission-only.
    pub opacity: f32,
    /// Single-scatter starlight strength σ_s (per ρ · path length, the same
    /// units family as `opacity`). Omitted or `0` = the feature is OFF and the
    /// render is bit-identical to the pre-scatter pipeline — the M7e-look
    /// sufficiency judgement is this one knob.
    pub scattering: Option<f32>,
    /// Henyey–Greenstein anisotropy g ∈ (−1, 1): 0 isotropic, > 0 forward
    /// (backlit silver-lining). Requires a positive `scattering` — on its own
    /// it is a dead knob and is rejected loud.
    pub anisotropy: Option<f32>,
    /// Per-light shadow volumes (umbral-lantern-lattice): `true` bakes
    /// light→sample transmittances so gas in its own shadow stops glowing.
    /// Omitted = `false` = the v1 unshadowed scatter (bit-compatible).
    /// Requires a positive `scattering` — present without one it is a dead
    /// knob and is rejected loud (whatever its value).
    pub shadows: Option<bool>,
    /// Chromatic scattering albedo (tinted-octree-lanterns): a per-channel
    /// multiplier on the scattered radiance only (dust reflects blue — the
    /// reflection-nebula look). Omitted = `[1.0; 3]` = neutral = bit-identical
    /// to the untinted scatter. Requires a positive `scattering` — present
    /// without one it is a dead knob and is rejected loud; an all-zero tint
    /// with a positive `scattering` zeroes the term and is rejected loud (set
    /// `scattering = 0` instead). Every component must be finite and ≥ 0.
    pub scatter_tint: Option<[f32; 3]>,
    /// Fixed scatter softening length ε (galaxy-render controls): replaces each
    /// light cluster's own cell radius as the single-scatter `1/d²` softening,
    /// so the scattered brightness is invariant to the octree `REFINE_TOL`
    /// (the hidden clustering→brightness coupling is removed). The renderer
    /// floors it at the gas voxel scale. Omitted = the v1 per-cluster radius
    /// softening (bit-identical to the shipped path). Requires a positive
    /// `scattering` — present without one it is a dead knob and is rejected
    /// loud; must be finite and `> 0`.
    pub scatter_softening: Option<f32>,
    /// Per-light shadow-volume bake strategy (the named deferral of
    /// umbral-lantern-lattice): `"brute"` (the reference) marches every voxel
    /// chord; `"dda"` skips provably-empty spans via a hierarchical occupancy —
    /// a **bit-identical** result, faster on sparse frames. Omitted = `brute`.
    /// Requires `shadows = true` (it accelerates the shadow bake) — present
    /// without it is a dead knob and is rejected loud.
    #[serde(default)]
    pub shadow_bake: Option<ShadowBakeSpec>,
    /// Temperature-dependent gas color (incandescent-nebular-veil): `Some`
    /// colors the gas emission by local temperature `T ∝ u` (the evolved
    /// internal energy) through a fixed cold→hot colormap, instead of the flat
    /// `color`. Self-contained (no scattering dependency); on ISOTHERMAL gas it
    /// renders a correct flat tint (all `u` equal), so it is meaningful only on
    /// an adiabatic run but never rejected on an isothermal one. Omitted = the
    /// flat-`color` emission, bit-identical.
    #[serde(default)]
    pub temperature: Option<TemperatureSpec>,
}

/// The `[look.gas.temperature]` colormap (incandescent-nebular-veil): the gas
/// emission color is a cold→hot lerp of `ū = N/ρ` (the SPH mass-weighted
/// specific internal energy) over the fixed band `[u_lo, u_hi]`. The band is a
/// temporally-constant reference (a per-frame max would flicker), in the same
/// code units as the solver's `u`.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemperatureSpec {
    /// Linear-RGB emission color of the coldest gas (`ū ≤ u_lo`).
    pub cold: [f32; 3],
    /// Linear-RGB emission color of the hottest gas (`ū ≥ u_hi`).
    pub hot: [f32; 3],
    /// Lower edge of the colormap band (specific internal energy `u`).
    pub u_lo: f32,
    /// Upper edge of the colormap band; must exceed `u_lo`.
    pub u_hi: f32,
}

/// TOML spelling of [`galaxy_render::ShadowBake`] (`shadow_bake = "brute" |
/// "dda"`). A separate type so the render crate stays serde-free; converted at
/// the scenario boundary.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ShadowBakeSpec {
    Brute,
    Dda,
}

impl From<ShadowBakeSpec> for ShadowBake {
    fn from(s: ShadowBakeSpec) -> ShadowBake {
        match s {
            ShadowBakeSpec::Brute => ShadowBake::Brute,
            ShadowBakeSpec::Dda => ShadowBake::Dda,
        }
    }
}

/// One progenitor's initial-radius color ramp.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RampSpec {
    pub inner: [f32; 3],
    pub outer: [f32; 3],
}

/// Camera choreography (the M6d rig), as data. Deserialized via [`RigTable`]
/// (same strict-table rationale as [`ModelSpec`]).
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(try_from = "RigTable")]
pub enum RigSpec {
    /// One static face-on framing over the whole run (pre-M6d, bit-exact).
    Static,
    /// Eased azimuth/tilt sweep (degrees, start → end) with the breathing zoom
    /// smoothed over ±`window` snapshots.
    OrbitTilt {
        azimuth_deg: [f32; 2],
        tilt_deg: [f32; 2],
        window: usize,
    },
    /// Fixed-direction perspective dolly (M6g): the eye approaches from
    /// `direction_deg` = [azimuth, tilt] (degrees, the orbit-tilt spherical
    /// convention) with constant vertical `fov_deg`. Distances are *fractions
    /// of the final framing radius* (scene-scale-free): the eye eases from
    /// `distance_frac[0]` to `distance_frac[1]` × that radius; the near plane
    /// sits at `near_frac` × the same radius.
    Dolly {
        direction_deg: [f32; 2],
        distance_frac: [f32; 2],
        fov_deg: f32,
        near_frac: f32,
    },
}

/// The raw `[model]` table: a strict superset of every model kind's keys, so an
/// unknown key is rejected here and each kind then checks it got exactly the
/// keys it needs.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ModelTable {
    kind: String,
    galaxy1: toml::Value,
    galaxy2: toml::Value,
    counts: toml::Value,
    /// Optional isothermal gas component. v1 supports it on `disk-plummer` only;
    /// the other kinds reject it (the IC supports NFW-halo gas too, but the
    /// pipeline demo is a Plummer merger — kept minimal).
    #[serde(default)]
    gas: Option<toml::Value>,
}

impl TryFrom<ModelTable> for ModelSpec {
    type Error = String;

    fn try_from(t: ModelTable) -> Result<Self, String> {
        fn field<T: serde::de::DeserializeOwned>(v: toml::Value, what: &str) -> Result<T, String> {
            v.try_into().map_err(|e| format!("model {what}: {e}"))
        }
        // Gas is a `disk-plummer`-only knob in v1: reject it up front for the
        // other kinds so a stray `[model.gas]` fails loud, not silently ignored.
        let no_gas_for = |kind: &str| -> Result<(), String> {
            if t.gas.is_some() {
                Err(format!(
                    "model kind `{kind}` takes no `[model.gas]` (gas is disk-plummer-only in v1)"
                ))
            } else {
                Ok(())
            }
        };
        match t.kind.as_str() {
            "disk-plummer" => Ok(ModelSpec::DiskPlummer {
                galaxy1: field(t.galaxy1, "galaxy1")?,
                galaxy2: field(t.galaxy2, "galaxy2")?,
                counts: field(t.counts, "counts")?,
                gas: t.gas.map(|v| field(v, "gas")).transpose()?,
            }),
            "disk-nfw" => {
                no_gas_for("disk-nfw")?;
                Ok(ModelSpec::DiskNfw {
                    galaxy1: field(t.galaxy1, "galaxy1")?,
                    galaxy2: field(t.galaxy2, "galaxy2")?,
                    counts: field(t.counts, "counts")?,
                })
            }
            "nfw-merger" => {
                no_gas_for("nfw-merger")?;
                Ok(ModelSpec::NfwMerger {
                    galaxy1: field(t.galaxy1, "galaxy1")?,
                    galaxy2: field(t.galaxy2, "galaxy2")?,
                    counts: field(t.counts, "counts")?,
                })
            }
            other => Err(format!(
                "unknown model kind `{other}` (disk-plummer|disk-nfw|nfw-merger)"
            )),
        }
    }
}

/// The raw `[rig]` table (strict superset of every rig kind's keys, so an
/// unknown key is rejected by serde and each kind then checks it got exactly
/// the keys it needs — no silently ignored knobs).
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RigTable {
    kind: String,
    azimuth_deg: Option<[f32; 2]>,
    tilt_deg: Option<[f32; 2]>,
    window: Option<usize>,
    direction_deg: Option<[f32; 2]>,
    distance_frac: Option<[f32; 2]>,
    fov_deg: Option<f32>,
    near_frac: Option<f32>,
}

impl RigTable {
    fn orbit_knobs(&self) -> bool {
        self.azimuth_deg.is_some() || self.tilt_deg.is_some() || self.window.is_some()
    }
    fn dolly_knobs(&self) -> bool {
        self.direction_deg.is_some()
            || self.distance_frac.is_some()
            || self.fov_deg.is_some()
            || self.near_frac.is_some()
    }
}

impl TryFrom<RigTable> for RigSpec {
    type Error = String;

    fn try_from(t: RigTable) -> Result<Self, String> {
        match t.kind.as_str() {
            "static" => {
                if t.orbit_knobs() || t.dolly_knobs() {
                    return Err("rig kind `static` takes no orbit-tilt/dolly knobs".into());
                }
                Ok(RigSpec::Static)
            }
            "orbit-tilt" => {
                if t.dolly_knobs() {
                    return Err("rig kind `orbit-tilt` takes no dolly knobs".into());
                }
                Ok(RigSpec::OrbitTilt {
                    azimuth_deg: t.azimuth_deg.ok_or("rig orbit-tilt needs azimuth_deg")?,
                    tilt_deg: t.tilt_deg.ok_or("rig orbit-tilt needs tilt_deg")?,
                    window: t.window.ok_or("rig orbit-tilt needs window")?,
                })
            }
            "dolly" => {
                if t.orbit_knobs() {
                    return Err("rig kind `dolly` takes no orbit-tilt knobs".into());
                }
                Ok(RigSpec::Dolly {
                    direction_deg: t.direction_deg.ok_or("rig dolly needs direction_deg")?,
                    distance_frac: t.distance_frac.ok_or("rig dolly needs distance_frac")?,
                    fov_deg: t.fov_deg.ok_or("rig dolly needs fov_deg")?,
                    near_frac: t.near_frac.ok_or("rig dolly needs near_frac")?,
                })
            }
            other => Err(format!(
                "unknown rig kind `{other}` (static|orbit-tilt|dolly)"
            )),
        }
    }
}

/// The checked-in scenario presets, embedded at compile time so `cargo run -p
/// galaxy-xtask <name>` works from any working directory.
pub const PRESETS: &[(&str, &str)] = &[
    ("disk", include_str!("../scenarios/disk.toml")),
    ("dm", include_str!("../scenarios/dm.toml")),
    ("cuspy", include_str!("../scenarios/cuspy.toml")),
    ("retro", include_str!("../scenarios/retro.toml")),
    ("inclined", include_str!("../scenarios/inclined.toml")),
    ("bullseye", include_str!("../scenarios/bullseye.toml")),
    ("minor", include_str!("../scenarios/minor.toml")),
    ("dolly", include_str!("../scenarios/dolly.toml")),
    ("gasrich", include_str!("../scenarios/gasrich.toml")),
    (
        "gasrich-adiabatic",
        include_str!("../scenarios/gasrich-adiabatic.toml"),
    ),
];

/// Look up a checked-in preset's toml text by canonical name.
pub fn preset(name: &str) -> Option<&'static str> {
    PRESETS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, text)| *text)
}

/// Parse and validate a `scenario.toml`. All errors — toml syntax, unknown
/// keys/kinds, and every physics/look invariant the pipeline relies on — come
/// back as a single human-readable message.
pub fn parse_scenario_toml(text: &str) -> Result<ScenarioSpec, String> {
    let spec: ScenarioSpec = toml::from_str(text).map_err(|e| format!("scenario.toml: {e}"))?;
    validate(&spec)?;
    Ok(spec)
}

/// Every invariant the IC constructors would `assert!` on (turned into readable
/// errors — a toml is user input, not a programming-time contract), plus the
/// cross-field rules the pipeline relies on.
fn validate(s: &ScenarioSpec) -> Result<(), String> {
    // Model: positive scales/masses, sane fractions, per-galaxy orientation.
    let n_prog: u16 = match &s.model {
        ModelSpec::DiskPlummer {
            galaxy1,
            galaxy2,
            counts,
            gas,
        } => {
            for (g, which) in [(galaxy1, "galaxy1"), (galaxy2, "galaxy2")] {
                validate_disk(g, which)?;
                positive(g.halo.mass, &format!("{which} halo mass"))?;
                positive(g.halo.scale, &format!("{which} halo scale"))?;
            }
            validate_counts(&[
                ("halo1", counts.full.halo1, counts.quick.halo1),
                ("disk1", counts.full.disk1, counts.quick.disk1),
                ("halo2", counts.full.halo2, counts.quick.halo2),
                ("disk2", counts.full.disk2, counts.quick.disk2),
            ])?;
            // Gas (M7c): positive gas counts, plus the IC's admissibility rule
            // (fraction ∈ (0,1), c_s > 0, min Q_gas ≥ 1) surfaced as a readable
            // error instead of a panic in `with_gas`. Q_gas depends on each disk's
            // Σ/κ, so both galaxies are checked with the shared (f, c_s).
            if let Some(gas) = gas {
                validate_counts(&[
                    ("gas1", counts.full.gas1, counts.quick.gas1),
                    ("gas2", counts.full.gas2, counts.quick.gas2),
                ])?;
                for (g, which) in [(galaxy1, "galaxy1"), (galaxy2, "galaxy2")] {
                    let disk = disk_galaxy(g, Plummer::new(G, g.halo.mass, g.halo.scale));
                    disk.check_gas(gas.fraction, gas.sound_speed)
                        .map_err(|e| format!("{which} gas: {e}"))?;
                }
                // Adiabatic EOS (H5-C): γ must exceed 1 (the ideal-gas index; γ ≤ 1
                // gives a non-physical / non-positive `(γ−1)` pressure coefficient and
                // a degenerate `u = c_s²/(γ−1)`). `u_floor`, when set, must be a finite
                // `≥ 0` energy.
                if let Some(g) = gas.gamma {
                    if !(g.is_finite() && g > 1.0) {
                        return Err(format!(
                            "gas gamma must be a finite number > 1 (ideal-gas adiabatic \
                             index), got {g}"
                        ));
                    }
                }
                if let Some(uf) = gas.u_floor {
                    if !(uf.is_finite() && uf >= 0.0) {
                        return Err(format!(
                            "gas u_floor must be a finite number >= 0, got {uf}"
                        ));
                    }
                }
            }
            4 // gas is not a splat, so the palette stays 4 stellar progenitors
        }
        ModelSpec::DiskNfw {
            galaxy1,
            galaxy2,
            counts,
        } => {
            for (g, which) in [(galaxy1, "galaxy1"), (galaxy2, "galaxy2")] {
                validate_disk(g, which)?;
                validate_nfw(&g.halo, which)?;
            }
            validate_counts(&[
                ("halo1", counts.full.halo1, counts.quick.halo1),
                ("disk1", counts.full.disk1, counts.quick.disk1),
                ("halo2", counts.full.halo2, counts.quick.halo2),
                ("disk2", counts.full.disk2, counts.quick.disk2),
            ])?;
            4
        }
        ModelSpec::NfwMerger {
            galaxy1,
            galaxy2,
            counts,
        } => {
            validate_nfw(galaxy1, "galaxy1")?;
            validate_nfw(galaxy2, "galaxy2")?;
            validate_counts(&[
                ("halo1", counts.full.halo1, counts.quick.halo1),
                ("halo2", counts.full.halo2, counts.quick.halo2),
            ])?;
            2
        }
    };

    // Orbit: `encounter::validate_orbit`, as a Result.
    let OrbitSpec {
        eccentricity: e,
        pericenter: peri,
        separation: sep,
    } = s.orbit;
    positive(e, "orbit eccentricity")?;
    positive(peri, "orbit pericenter")?;
    if sep < peri {
        return Err(format!(
            "orbit separation ({sep}) must be >= the pericenter ({peri})"
        ));
    }
    if e < 1.0 {
        let apo = peri * (1.0 + e) / (1.0 - e);
        if sep > apo * (1.0 + 1e-12) {
            return Err(format!(
                "orbit separation ({sep}) exceeds the apocenter ({apo}) of the bound orbit (e={e})"
            ));
        }
    }

    // Sim timing.
    positive(s.sim.dt, "sim dt")?;
    positive(s.sim.eps, "sim eps")?;
    if s.sim.n_steps == 0 {
        return Err("sim n_steps must be positive".into());
    }
    if s.sim.snapshot_every == 0 || s.sim.snapshot_every_quick == Some(0) {
        return Err("sim snapshot cadence must be positive".into());
    }

    // Gas-driver toggles ([sim.adaptive] vs [sim.individual]): both size the gas dt, so
    // declaring both is an ambiguous, contradictory intent — reject rather than silently
    // pick one. (Independent of gas presence: a gas-free scenario ignores both, but the
    // contradiction is still a config error worth catching early.)
    if s.sim.adaptive.is_some() && s.sim.individual.is_some() {
        return Err(
            "sim declares both [sim.adaptive] and [sim.individual] — they both \
                    size the gas timestep; pick one"
                .into(),
        );
    }
    if let Some(ind) = &s.sim.individual {
        // `hydro+gravity` (I-grav / lever b) is now built — it subcycles gravity on a
        // cached stale tree, giving collisionless stars finite gravitational rungs.
        if !(ind.courant.is_finite() && ind.courant > 0.0 && ind.courant <= 1.0) {
            return Err(format!(
                "[sim.individual] courant must be in (0, 1], got {}",
                ind.courant
            ));
        }
        if !(1..=60).contains(&ind.r_max) {
            return Err(format!(
                "[sim.individual] r_max must be in [1, 60], got {}",
                ind.r_max
            ));
        }
        if ind.dt_base_cap.is_nan() || ind.dt_base_cap <= 0.0 {
            return Err(format!(
                "[sim.individual] dt_base_cap must be positive (may be inf), got {}",
                ind.dt_base_cap
            ));
        }
    }

    // Look: splat/framing knobs + palette/ramp lengths tied to the model.
    positive(f64::from(s.look.splat_size), "look splat_size")?;
    if !(s.look.frame_percentile > 0.0 && s.look.frame_percentile <= 1.0) {
        return Err(format!(
            "look frame_percentile must be in (0, 1], got {}",
            s.look.frame_percentile
        ));
    }
    // Screen-space splat cap (pinprick-starfield): declared ⇒ finite and
    // positive (0 px would cull every splat — a dead scene, not a look).
    if let Some(cap) = s.look.max_splat_px {
        if !(cap.is_finite() && cap > 0.0) {
            return Err(format!(
                "look max_splat_px must be positive and finite, got {cap}"
            ));
        }
    }
    if s.look.palette.len() != n_prog as usize {
        return Err(format!(
            "look palette has {} colors but the model has {n_prog} progenitors",
            s.look.palette.len()
        ));
    }
    if s.look.ramps.len() != n_prog as usize {
        return Err(format!(
            "look has {} ramps but the model has {n_prog} progenitors",
            s.look.ramps.len()
        ));
    }
    if let Some(p) = s.look.sf_progenitors.iter().find(|p| **p >= n_prog) {
        return Err(format!(
            "look sf_progenitors names progenitor {p} but the model has only {n_prog}"
        ));
    }

    // Gas look ([look.gas], M7f): a gas-only knob. Present iff the model carries
    // `[model.gas]` — a `[look.gas]` on a gas-free model renders nothing (a dead
    // knob), so it fails loud rather than being silently ignored. When present,
    // its rates must be finite and non-negative (opacity 0 = emission-only is OK).
    let model_has_gas = matches!(&s.model, ModelSpec::DiskPlummer { gas: Some(_), .. });
    match (&s.look.gas, model_has_gas) {
        (Some(_), false) => {
            return Err(
                "look.gas is set but the model has no gas — a dead volumetric \
                        look (remove [look.gas], or add [model.gas])"
                    .into(),
            );
        }
        (Some(gl), true) => {
            if !gl.color.iter().all(|c| c.is_finite() && *c >= 0.0) {
                return Err(format!(
                    "look.gas color components must be finite and non-negative, got {:?}",
                    gl.color
                ));
            }
            if !(gl.emissivity.is_finite() && gl.emissivity >= 0.0) {
                return Err(format!(
                    "look.gas emissivity must be finite and non-negative, got {}",
                    gl.emissivity
                ));
            }
            if !(gl.opacity.is_finite() && gl.opacity >= 0.0) {
                return Err(format!(
                    "look.gas opacity must be finite and non-negative, got {}",
                    gl.opacity
                ));
            }
            // Single-scatter knobs (scattered-starlit-veil): σ_s is a rate like
            // opacity; |g| < 1 keeps the HG denominator positive; anisotropy
            // without a positive scattering shapes nothing — a dead knob.
            if let Some(sc) = gl.scattering {
                if !(sc.is_finite() && sc >= 0.0) {
                    return Err(format!(
                        "look.gas scattering must be finite and non-negative, got {sc}"
                    ));
                }
            }
            if let Some(g) = gl.anisotropy {
                if !(g.is_finite() && g.abs() < 1.0) {
                    return Err(format!(
                        "look.gas anisotropy must be finite with |g| < 1 \
                         (Henyey–Greenstein), got {g}"
                    ));
                }
                if !gl.scattering.is_some_and(|sc| sc > 0.0) {
                    return Err(
                        "look.gas anisotropy without a positive scattering is a dead \
                         knob (add scattering > 0, or remove anisotropy)"
                            .into(),
                    );
                }
            }
            // Shadow volumes (umbral-lantern-lattice): the same discipline —
            // the knob PRESENT without a live scatter term shapes nothing.
            if gl.shadows.is_some() && !gl.scattering.is_some_and(|sc| sc > 0.0) {
                return Err("look.gas shadows without a positive scattering is a dead \
                     knob (add scattering > 0, or remove shadows)"
                    .into());
            }
            // Shadow-bake strategy (DDA/hierarchical deferral): a bit-identical
            // acceleration of the shadow bake, so it shapes nothing unless
            // shadows are actually baked — present without `shadows = true` it is
            // a dead knob (same discipline).
            if gl.shadow_bake.is_some() && gl.shadows != Some(true) {
                return Err(
                    "look.gas shadow_bake without shadows = true is a dead knob \
                     (add shadows = true, or remove shadow_bake)"
                        .into(),
                );
            }
            // Chromatic scattering albedo (tinted-octree-lanterns): a physical
            // albedo — every component finite and ≥ 0. Present without a live
            // scatter term it shapes nothing (the dead-knob discipline); an
            // all-zero tint zeroes the term, which is `scattering = 0` said
            // louder — reject rather than silently blanking the scatter.
            if let Some(tint) = gl.scatter_tint {
                if !tint.iter().all(|c| c.is_finite() && *c >= 0.0) {
                    return Err(format!(
                        "look.gas scatter_tint components must be finite and \
                         non-negative, got {tint:?}"
                    ));
                }
                if !gl.scattering.is_some_and(|sc| sc > 0.0) {
                    return Err("look.gas scatter_tint without a positive scattering is a \
                         dead knob (add scattering > 0, or remove scatter_tint)"
                        .into());
                }
                if tint.iter().all(|c| *c == 0.0) {
                    return Err("look.gas scatter_tint is all-zero, which zeroes the \
                         scattered term — set scattering = 0 instead"
                        .into());
                }
            }
            // Fixed scatter softening ε (galaxy-render controls): a softening
            // LENGTH — finite and strictly positive. Present without a live
            // scatter term it shapes nothing (the same dead-knob discipline).
            if let Some(eps) = gl.scatter_softening {
                if !(eps.is_finite() && eps > 0.0) {
                    return Err(format!(
                        "look.gas scatter_softening must be finite and positive, got {eps}"
                    ));
                }
                if !gl.scattering.is_some_and(|sc| sc > 0.0) {
                    return Err(
                        "look.gas scatter_softening without a positive scattering is a \
                         dead knob (add scattering > 0, or remove scatter_softening)"
                            .into(),
                    );
                }
            }
            // Temperature colormap (incandescent-nebular-veil): the cold/hot
            // endpoints are linear RGB (finite, ≥ 0) and the band must have a
            // direction (finite u_lo < u_hi). Self-contained — NOT gated on
            // scattering; on isothermal gas it renders a correct flat tint, so
            // it is never a dead knob to declare.
            if let Some(t) = gl.temperature {
                if !t
                    .cold
                    .iter()
                    .chain(&t.hot)
                    .all(|c| c.is_finite() && *c >= 0.0)
                {
                    return Err(format!(
                        "look.gas temperature cold/hot components must be finite and \
                         non-negative, got cold {:?} hot {:?}",
                        t.cold, t.hot
                    ));
                }
                if !(t.u_lo.is_finite() && t.u_hi.is_finite() && t.u_lo < t.u_hi) {
                    return Err(format!(
                        "look.gas temperature needs finite u_lo < u_hi, got u_lo {} u_hi {}",
                        t.u_lo, t.u_hi
                    ));
                }
            }
        }
        (None, _) => {}
    }

    // Local tone compression ([look.local_tone], render-more-controls): a GRADE
    // knob (whole-frame, applied to the linear HDR before the tone curve), NOT a
    // gas knob — so it is NOT gated on scattering/gas the way [look.gas] is. Mirror
    // `GradeConfig::validate` at parse time so a bad knob fails with a scenario
    // message, not deep in the grade stage: strength finite & > 0 (a declared 0 is
    // a bit-exact no-op — reject it, absence already means "off"), radius finite &
    // > 0, floor finite & in [0, 1].
    if let Some(lt) = &s.look.local_tone {
        if !(lt.strength.is_finite() && lt.strength > 0.0) {
            return Err(format!(
                "look.local_tone strength must be finite and > 0 (absence means off), got {}",
                lt.strength
            ));
        }
        if let Some(r) = lt.radius {
            if !(r.is_finite() && r > 0.0) {
                return Err(format!(
                    "look.local_tone radius must be finite and > 0 (pixels), got {r}"
                ));
            }
        }
        if let Some(f) = lt.floor {
            if !(f.is_finite() && (0.0..=1.0).contains(&f)) {
                return Err(format!(
                    "look.local_tone floor must be finite and in [0, 1], got {f}"
                ));
            }
        }
    }

    // Age coloring ([look.age], natal-ember-forge F6/F7): a `[look]` coloring knob
    // (like the SF compression proxy), NOT gas-gated — it is inert when no star has a
    // formation time and meaningful on reused SF snapshots, so it is never a dead knob
    // to declare. Validate the values only: `young` a linear-RGB color (finite, ≥ 0),
    // `strength` finite & > 0 (a declared 0 is a bit-exact no-op — absence means off),
    // `tau` finite & > 0 (the fade timescale).
    if let Some(a) = &s.look.age {
        if !a.young.iter().all(|c| c.is_finite() && *c >= 0.0) {
            return Err(format!(
                "look.age young components must be finite and non-negative, got {:?}",
                a.young
            ));
        }
        if !(a.strength.is_finite() && a.strength > 0.0) {
            return Err(format!(
                "look.age strength must be finite and > 0 (absence means off), got {}",
                a.strength
            ));
        }
        if !(a.tau.is_finite() && a.tau > 0.0) {
            return Err(format!(
                "look.age tau must be finite and > 0 (the fade timescale), got {}",
                a.tau
            ));
        }
    }

    // Star formation ([physics.star_formation], natal-ember-forge F7): a gated physics
    // knob. It needs SPH gas — a pure-gravity solver's `sf_fields` returns zeros, so
    // nothing could ever convert; declared on a gas-free model it is a dead knob and
    // rejected loud (mirroring the [look.gas] gas-presence gate). When present, both
    // recipe knobs must be positive (`efficiency = 0` is a no-op — absence means off).
    if let Some(sf) = s.physics.and_then(|p| p.star_formation) {
        if !model_has_gas {
            return Err(
                "physics.star_formation is set but the model has no gas — a pure-gravity \
                 solver forms no stars (add [model.gas], or remove [physics.star_formation])"
                    .into(),
            );
        }
        if !(sf.rho_thresh.is_finite() && sf.rho_thresh > 0.0) {
            return Err(format!(
                "physics.star_formation rho_thresh must be finite and > 0, got {}",
                sf.rho_thresh
            ));
        }
        if !(sf.efficiency.is_finite() && sf.efficiency > 0.0) {
            return Err(format!(
                "physics.star_formation efficiency must be finite and > 0 (absence means \
                 off), got {}",
                sf.efficiency
            ));
        }
    }

    // Rig.
    match &s.rig {
        RigSpec::Static => {}
        RigSpec::OrbitTilt {
            azimuth_deg,
            tilt_deg,
            window,
        } => {
            if *window == 0 {
                return Err("rig window must be positive (snapshots of envelope smoothing)".into());
            }
            if !azimuth_deg.iter().chain(tilt_deg).all(|a| a.is_finite()) {
                return Err("rig angles must be finite".into());
            }
        }
        RigSpec::Dolly {
            direction_deg,
            distance_frac,
            fov_deg,
            near_frac,
        } => {
            if !direction_deg.iter().all(|a| a.is_finite()) {
                return Err("rig dolly direction_deg must be finite".into());
            }
            if !distance_frac.iter().all(|d| d.is_finite() && *d > 0.0) {
                return Err("rig dolly distance_frac must be finite and positive".into());
            }
            if !(fov_deg.is_finite() && *fov_deg > 0.0 && *fov_deg < 180.0) {
                return Err(format!(
                    "rig dolly fov_deg must be in (0, 180), got {fov_deg}"
                ));
            }
            // The eased distance never drops below the closer endpoint, so this
            // keeps the near plane in front of the eye for the whole move.
            let closest = distance_frac[0].min(distance_frac[1]);
            if !(near_frac.is_finite() && *near_frac > 0.0 && *near_frac < closest) {
                return Err(format!(
                    "rig dolly near_frac must satisfy 0 < near_frac < min(distance_frac) = \
                     {closest}, got {near_frac}"
                ));
            }
        }
    }
    Ok(())
}

/// The halo-independent disk-galaxy invariants.
fn validate_disk<H>(g: &DiskGalaxySpec<H>, which: &str) -> Result<(), String> {
    positive(g.disk_mass, &format!("{which} disk_mass"))?;
    positive(g.scale_length, &format!("{which} scale_length"))?;
    positive(g.hz_frac, &format!("{which} hz_frac"))?;
    if !(g.rmax_frac.is_finite() && g.rmax_frac > 1.0) {
        return Err(format!(
            "{which} rmax_frac must exceed 1 (truncation beyond the scale length), got {}",
            g.rmax_frac
        ));
    }
    if let Some(q) = g.toomre_q {
        positive(q, &format!("{which} toomre_q"))?;
    }
    if !(g.inclination_deg.is_finite() && g.argument_deg.is_finite()) {
        return Err(format!("{which} orientation angles must be finite"));
    }
    Ok(())
}

fn validate_nfw(h: &NfwSpec, which: &str) -> Result<(), String> {
    positive(h.mvir, &format!("{which} halo mvir"))?;
    positive(h.rs, &format!("{which} halo rs"))?;
    positive(h.skirt, &format!("{which} halo skirt"))?;
    if !(h.c.is_finite() && h.c > 1.0) {
        return Err(format!(
            "{which} halo concentration must exceed 1, got {}",
            h.c
        ));
    }
    Ok(())
}

fn validate_counts(counts: &[(&str, usize, usize)]) -> Result<(), String> {
    for (what, full, quick) in counts {
        if *full == 0 || *quick == 0 {
            return Err(format!("counts.{what} must be positive (full and quick)"));
        }
    }
    Ok(())
}

fn positive(v: f64, what: &str) -> Result<(), String> {
    if v.is_finite() && v > 0.0 {
        Ok(())
    } else {
        Err(format!("{what} must be positive and finite, got {v}"))
    }
}

// --- The runtime bundle -------------------------------------------------------

/// Everything a scenario hands the pipeline: the sampled IC plus the sim-timing,
/// softening, splat look, and framing knobs. Built from a [`ScenarioSpec`] by
/// [`build_scenario`]; the pipeline (`run_movie` in the binary) is single-sourced
/// over this.
#[derive(Clone)]
pub struct Scenario {
    pub state: State,
    pub prep: PrepConfig,
    pub eps: f64,
    pub dt: f64,
    pub n_steps: u64,
    pub snapshot_every: u64,
    /// Hermite in-between frames per snapshot interval (M6c); 1 = no upsampling.
    pub subframes: u32,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub frame_percentile: f32,
    /// Screen-space splat cap in pixels (pinprick-starfield), `None` = off —
    /// the runtime form of [`LookSpec::max_splat_px`], mapped verbatim into
    /// `RenderConfig.max_splat_px` by the movie pipeline.
    pub max_splat_px: Option<f32>,
    pub rig: Rig,
    /// Per-progenitor `(inner, outer)` ramp for `--color initial-radius` (M6e).
    pub ramp: Vec<([f32; 3], [f32; 3])>,
    /// Progenitors the star-formation compression proxy applies to (M6e).
    pub sf_progenitors: Vec<u16>,
    /// Isothermal gas sound speed c_s when the scenario is gas-rich (M7c), else
    /// `None`. The **single source** of c_s: the movie pipeline threads this same
    /// value into both the IC's pressure equilibrium (already baked into `state`)
    /// and the force solver's `HydroParams`, so the two cannot diverge.
    pub sound_speed: Option<f64>,
    /// Adiabatic index γ when the gas is adiabatic (`[model.gas].gamma`), else
    /// `None` (isothermal). `Some` implies [`sound_speed`](Self::sound_speed) is
    /// `Some` (adiabatic ⇒ gas-rich). Routes `simulate_snapshots` to the
    /// `LeapfrogKdkThermal` + `Eos::Adiabatic` block-adaptive path; the IC's gas
    /// `u` is seeded from `sound_speed` so the disk starts in pressure equilibrium.
    pub gamma: Option<f64>,
    /// Positive-`u` floor for the adiabatic thermal integrator (`[model.gas].u_floor`),
    /// `0.0` when unset or isothermal (inert).
    pub u_floor: f64,
    /// Volumetric gas look (M7f) when the scenario is gas-rich, else `None`.
    /// `Some` **iff** [`sound_speed`](Self::sound_speed) is `Some` (both are
    /// gas-only): the movie pipeline builds a `galaxy_render::GasLook` from it,
    /// falling back to [`GasLookValues::default`] when the model has gas but
    /// omits `[look.gas]`. Kept render-free (plain values, mirroring
    /// `sound_speed`) so the lib seam stays decoupled from the renderer.
    pub gas_look: Option<GasLookValues>,
    /// Per-light shadow-volume bake strategy, mapped verbatim into
    /// `RenderConfig.shadow_bake` (the runtime form of `[look.gas].shadow_bake`).
    /// `Brute` (default) or the bit-identical `Dda` acceleration. Inert unless
    /// the gas look bakes shadows.
    pub shadow_bake: ShadowBake,
    /// Local (spatially-adaptive) tone compression baked into the movie grade
    /// (`[look.local_tone]`, render-more-controls), `None` = off. The runtime form
    /// of [`LookSpec::local_tone`]: the movie pipeline drops it verbatim into
    /// `galaxy_grade::GradeConfig::local`, so the shipped movie carries the same
    /// local tonemap the `regrade --local` A/B settled on — no separate regrade
    /// pass. A grade type (not a render one), so it lives directly on the
    /// scenario, unlike the render-free `GasLookValues` mirror.
    pub local_tone: Option<LocalToneConfig>,
    /// Block-adaptive timestep policy (`[sim.adaptive]`, courant-quickening-cadence),
    /// `None` = fixed-dt. When `Some` and the scenario is gas-rich, `simulate_snapshots`
    /// routes the gas path through the adaptive driver; a gas-free scenario ignores it.
    pub adaptive: Option<AdaptiveSpec>,
    /// Individual-timestep policy (`[sim.individual]`, laddered-ember-cadence), `None` =
    /// off. When `Some` with `mode = hydro-only` and the scenario is gas-rich,
    /// `simulate_snapshots` routes the gas path through `run_individual` (CPU only); a
    /// gas-free scenario ignores it. Mutually exclusive with `adaptive` (rejected at
    /// parse if both are set).
    pub individual: Option<IndividualSpec>,
    /// Star-formation recipe (`[physics.star_formation]`, natal-ember-forge), `None`
    /// = off (byte-identical). The runtime form of [`StarFormationSpec`]: threaded
    /// into the CPU stepping config's `sf` field by `simulate_snapshots`, so dense
    /// converging gas converts to collisionless stars at each snapshot sync site.
    /// Only ever `Some` on a gas-rich scenario (rejected at parse on a gas-free
    /// model); the GPU-resident path does not apply the SF operator and rejects it
    /// loud rather than running silently SF-free.
    pub sf: Option<StarFormationConfig>,
    pub info: String,
}

/// The runtime form of [`GasLookSpec`] (render-free, mirroring how `sound_speed`
/// carries the gas c_s as a plain value). Its `Default` matches
/// `galaxy_render::GasLook::default` so a gas-rich scenario that omits `[look.gas]`
/// renders with the same neutral look the renderer would fall back to.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GasLookValues {
    pub color: [f32; 3],
    pub emissivity: f32,
    pub opacity: f32,
    /// Single-scatter strength σ_s; `0` = off (`GasLook::scatter = None`).
    pub scattering: f32,
    /// Henyey–Greenstein g; meaningful only with `scattering > 0`.
    pub anisotropy: f32,
    /// Per-light shadow volumes; meaningful only with `scattering > 0`.
    pub shadows: bool,
    /// Chromatic scattering albedo; `[1.0; 3]` = neutral. Meaningful only with
    /// `scattering > 0`.
    pub scatter_tint: [f32; 3],
    /// Fixed scatter softening ε; `None` = v1 per-cluster radius softening
    /// (bit-compat). Meaningful only with `scattering > 0`.
    pub scatter_softening: Option<f32>,
    /// Temperature colormap (incandescent-nebular-veil): `None` = flat-`color`
    /// emission. `Some` colors the gas by `ū = N/ρ` (`T ∝ u`). The renderer
    /// builds a `TempColor` from this plus the per-frame moment grids.
    pub temperature: Option<TemperatureSpec>,
}

impl Default for GasLookValues {
    fn default() -> Self {
        GasLookValues {
            color: [1.0, 1.0, 1.0],
            emissivity: 1.0,
            opacity: 1.0,
            scattering: 0.0,
            anisotropy: 0.0,
            shadows: false,
            scatter_tint: [1.0; 3],
            scatter_softening: None,
            temperature: None,
        }
    }
}

/// Per-scenario camera choreography (M6d), the runtime form of [`RigSpec`].
#[derive(Clone, Debug, PartialEq)]
pub enum Rig {
    Static,
    /// Eased azimuth/tilt sweep (degrees, start → end) with a breathing zoom.
    OrbitTilt {
        azimuth_deg: (f32, f32),
        tilt_deg: (f32, f32),
        window: usize,
    },
    /// Fixed-direction perspective dolly (M6g); the runtime form of
    /// [`RigSpec::Dolly`] (same fields, fractions still unresolved — the movie
    /// pipeline anchors them to the final framing radius it computes).
    Dolly {
        direction_deg: (f32, f32),
        distance_frac: (f32, f32),
        fov_deg: f32,
        near_frac: f32,
    },
}

/// Build the runtime [`Scenario`] from a validated spec: sample the IC
/// (deterministic in `spec.seed`), pick full/QUICK counts, cadence, and frame
/// size, and assemble the prep config (palette + the always-on M6a/M6e density
/// features, keyed to the scenario's ε). Same spec + same seed + same `quick`
/// ⇒ bit-identical `State`.
pub fn build_scenario(spec: &ScenarioSpec, quick: bool) -> Scenario {
    let orbit = &spec.orbit;
    // `unit_mass` is the mass whose particle carries PEAK_BRIGHTNESS: the disk
    // particle for disk models (disk flux is set by disk MASS, not count), the
    // (equal-by-design) halo particle for the merger.
    let (mut state, unit_mass, info, sound_speed) = match &spec.model {
        ModelSpec::DiskPlummer {
            galaxy1,
            galaxy2,
            counts,
            gas,
        } => {
            let c = if quick { counts.quick } else { counts.full };
            let h1 = Plummer::new(G, galaxy1.halo.mass, galaxy1.halo.scale);
            let h2 = Plummer::new(G, galaxy2.halo.mass, galaxy2.halo.scale);
            match gas {
                None => {
                    let (state, unit_mass) =
                        sample_disks(galaxy1, galaxy2, h1, h2, orbit, c, spec.seed);
                    let info = disk_info("halo", &state, &c, unit_mass, orbit, spec.sim.eps, None);
                    (state, unit_mass, info, None)
                }
                Some(gas) => {
                    let (state, unit_mass) =
                        sample_disks_gas(galaxy1, galaxy2, h1, h2, orbit, c, gas, spec.seed);
                    let info = disk_info(
                        "halo",
                        &state,
                        &c,
                        unit_mass,
                        orbit,
                        spec.sim.eps,
                        Some(gas),
                    );
                    // The ONE c_s: baked into `state`'s pressure equilibrium here,
                    // and handed to the force solver's HydroParams in `run_movie`.
                    (state, unit_mass, info, Some(gas.sound_speed))
                }
            }
        }
        ModelSpec::DiskNfw {
            galaxy1,
            galaxy2,
            counts,
        } => {
            let c = if quick { counts.quick } else { counts.full };
            let h1 = truncated_nfw(&galaxy1.halo);
            let h2 = truncated_nfw(&galaxy2.halo);
            let (state, unit_mass) = sample_disks(galaxy1, galaxy2, h1, h2, orbit, c, spec.seed);
            let info = disk_info(
                "cuspy halo",
                &state,
                &c,
                unit_mass,
                orbit,
                spec.sim.eps,
                None,
            );
            (state, unit_mass, info, None)
        }
        ModelSpec::NfwMerger {
            galaxy1,
            galaxy2,
            counts,
        } => {
            let c = if quick { counts.quick } else { counts.full };
            let g1 = truncated_nfw(galaxy1);
            let g2 = truncated_nfw(galaxy2);
            let collision = NfwCollision::new(
                g1,
                g2,
                orbit.eccentricity,
                orbit.pericenter,
                orbit.separation,
            );
            let state = collision.sample(c.halo1, c.halo2, spec.seed);
            let unit_mass = g1.total_mass() / c.halo1 as f64;
            let info = format!(
                "IC: {} particles (halo1 {} + halo2 {}), particle mass {unit_mass:.3e}; \
                 e={} peri={} sep={}, T={:.0}",
                state.len(),
                c.halo1,
                c.halo2,
                orbit.eccentricity,
                orbit.pericenter,
                orbit.separation,
                spec.sim.n_steps as f64 * spec.sim.dt,
            );
            (state, unit_mass, info, None)
        }
    };

    // Adiabatic gas EOS ([model.gas].gamma, H5-C): only a `disk-plummer` with a
    // `[model.gas]` table can be adiabatic; every other model is isothermal/gas-free.
    let (gamma, u_floor) = match &spec.model {
        ModelSpec::DiskPlummer { gas: Some(gas), .. } => (gas.gamma, gas.u_floor.unwrap_or(0.0)),
        _ => (None, 0.0),
    };
    // Seed each gas particle's internal energy so the t=0 pressure `(γ−1)ρu`
    // equals the isothermal equilibrium `c_s²ρ` the sampler baked ⇒ `u = c_s²/(γ−1)`.
    // The disk starts in true force balance (no spurious startup contraction/heating
    // that would manufacture fake temperature), so only real shocks light up. Non-gas
    // `u` stays 0; isothermal (`gamma` None) leaves all `u = 0` — byte-unchanged.
    if let (Some(c_s), Some(g)) = (sound_speed, gamma) {
        let u_init = c_s * c_s / (g - 1.0);
        for (u, k) in state.u.iter_mut().zip(&state.kind) {
            if *k == Species::Gas {
                *u = u_init;
            }
        }
    }

    let eps = spec.sim.eps;
    let (width, height) = if quick {
        (QUICK_W, QUICK_H)
    } else {
        (FRAME_W, FRAME_H)
    };
    let snapshot_every = if quick {
        spec.sim
            .snapshot_every_quick
            .unwrap_or(spec.sim.snapshot_every)
    } else {
        spec.sim.snapshot_every
    };

    Scenario {
        state,
        prep: PrepConfig {
            palette: spec.look.palette.clone(),
            brightness_per_mass: PEAK_BRIGHTNESS / unit_mass as f32,
            size: spec.look.splat_size,
            density: Some(DensityColoring {
                k: DENSITY_K,
                softening: eps,
                strength: DENSITY_STRENGTH,
            }),
            color: ColorMode::Progenitor, // --color may override in run_movie
            size_by_density: Some(SizeByDensity {
                k: DENSITY_K,
                softening: eps,
                min_frac: SIZE_MIN_FRAC,
                max_frac: SIZE_MAX_FRAC,
            }),
            compression: None, // filled by run_movie (rho0 needs snapshot 0)
            // Age coloring ([look.age], natal-ember-forge F6): a `[look]` coloring
            // knob, view-independent (reads formation_time + snapshot time, no kNN /
            // snap0), so unlike `compression` it needs no per-run anchor — it is baked
            // straight into the base prep here. `None` = no age tint (bit-identical).
            age: spec.look.age.map(|a| AgeColoring {
                young: a.young,
                strength: a.strength,
                tau: a.tau,
            }),
            gas_as_splats: false, // gas renders volumetrically (M7d), not as splats
        },
        eps,
        dt: spec.sim.dt,
        n_steps: spec.sim.n_steps,
        snapshot_every,
        subframes: SUBFRAMES,
        seed: spec.seed,
        width,
        height,
        frame_percentile: spec.look.frame_percentile,
        max_splat_px: spec.look.max_splat_px,
        rig: match &spec.rig {
            RigSpec::Static => Rig::Static,
            RigSpec::OrbitTilt {
                azimuth_deg,
                tilt_deg,
                window,
            } => Rig::OrbitTilt {
                azimuth_deg: (azimuth_deg[0], azimuth_deg[1]),
                tilt_deg: (tilt_deg[0], tilt_deg[1]),
                window: *window,
            },
            RigSpec::Dolly {
                direction_deg,
                distance_frac,
                fov_deg,
                near_frac,
            } => Rig::Dolly {
                direction_deg: (direction_deg[0], direction_deg[1]),
                distance_frac: (distance_frac[0], distance_frac[1]),
                fov_deg: *fov_deg,
                near_frac: *near_frac,
            },
        },
        ramp: spec.look.ramps.iter().map(|r| (r.inner, r.outer)).collect(),
        sf_progenitors: spec.look.sf_progenitors.clone(),
        sound_speed,
        // Adiabatic EOS (H5-C): `Some(γ)` iff `[model.gas].gamma` is set (⇒
        // gas-rich); routes the sim to the thermal + adaptive path. `u_floor`
        // is the thermal integrator's positive-`u` floor (0.0 = inert).
        gamma,
        u_floor,
        // Gas look (M7f): `Some` iff the scenario is gas-rich (tied to `sound_speed`,
        // the other gas-only field), taking the declared `[look.gas]` or the neutral
        // default the renderer falls back to when the model has gas but omits it.
        gas_look: sound_speed.map(|_| {
            spec.look
                .gas
                .map(|g| GasLookValues {
                    color: g.color,
                    emissivity: g.emissivity,
                    opacity: g.opacity,
                    scattering: g.scattering.unwrap_or(0.0),
                    anisotropy: g.anisotropy.unwrap_or(0.0),
                    shadows: g.shadows.unwrap_or(false),
                    scatter_tint: g.scatter_tint.unwrap_or([1.0; 3]),
                    scatter_softening: g.scatter_softening,
                    temperature: g.temperature,
                })
                .unwrap_or_default()
        }),
        // Shadow-bake strategy (DDA/hierarchical deferral): sourced from
        // `[look.gas].shadow_bake`, `Brute` by default. Bit-identical either way
        // — a perf choice routed to RenderConfig, not part of the GasLook mirror.
        shadow_bake: spec
            .look
            .gas
            .and_then(|g| g.shadow_bake)
            .map(Into::into)
            .unwrap_or_default(),
        // Local tonemap ([look.local_tone], render-more-controls): the runtime
        // `LocalToneConfig` the movie grade bakes, or `None` when omitted. Not
        // gas-gated (a whole-frame grade knob), so it resolves independently of
        // `sound_speed`. `radius`/`floor` default to the same constants the
        // `regrade --local` CLI uses, so `strength = k` reproduces `--local k`.
        local_tone: spec.look.local_tone.map(|lt| LocalToneConfig {
            strength: lt.strength,
            radius: lt.radius.unwrap_or(DEFAULT_LOCAL_RADIUS),
            floor: lt.floor.unwrap_or(DEFAULT_LOCAL_FLOOR),
        }),
        // Block-adaptive timestep policy ([sim.adaptive], courant-quickening-cadence):
        // carried verbatim; `simulate_snapshots` acts on it only when the scenario is
        // gas-rich (a gas-free run has no hydro CFL constraint and ignores it).
        adaptive: spec.sim.adaptive,
        // Individual-timestep policy ([sim.individual], laddered-ember-cadence): carried
        // verbatim; `simulate_snapshots` acts on it only when the scenario is gas-rich
        // and `mode = hydro-only`. Mutual exclusion with `adaptive` is enforced at parse.
        individual: spec.sim.individual,
        // Star formation ([physics.star_formation], natal-ember-forge F7): the runtime
        // recipe `simulate_snapshots` threads into the CPU stepping config's `sf` field.
        // `None` (absent, or a gas-free model — rejected at parse) ⇒ SF never runs ⇒
        // byte-identical. Validated at parse (rho_thresh/efficiency > 0, gas present).
        sf: spec
            .physics
            .and_then(|p| p.star_formation)
            .map(|sf| StarFormationConfig {
                rho_thresh: sf.rho_thresh,
                efficiency: sf.efficiency,
                seed: sf.seed,
            }),
        info,
    }
}

/// Instantiate one disk galaxy from its spec on the given halo (the fractions
/// scale with Rd; warmth applies only when the spec asks for it).
fn disk_galaxy<H: SphericalHalo, S>(g: &DiskGalaxySpec<S>, halo: H) -> ExponentialDisk<H> {
    let disk = ExponentialDisk::new(
        g.disk_mass,
        g.scale_length,
        g.hz_frac * g.scale_length,
        g.rmax_frac * g.scale_length,
        halo,
    );
    match g.toomre_q {
        Some(q) => disk.with_toomre_q(q),
        None => disk,
    }
}

/// Build the two-disk encounter with each galaxy's Toomre spin-orbit orientation
/// applied. Shared by the gas-free and gas-rich paths so their orbit/orientation
/// plumbing cannot drift; the two differ only in which sampler they then call.
fn oriented_collision<H: SphericalHalo, S>(
    d1: ExponentialDisk<H>,
    d2: ExponentialDisk<H>,
    g1: &DiskGalaxySpec<S>,
    g2: &DiskGalaxySpec<S>,
    orbit: &OrbitSpec,
) -> DiskCollision<H> {
    let mut collision = DiskCollision::new(
        d1,
        d2,
        orbit.eccentricity,
        orbit.pericenter,
        orbit.separation,
    );
    collision.orient1 = Orientation::from_angles(
        g1.inclination_deg.to_radians(),
        g1.argument_deg.to_radians(),
    );
    collision.orient2 = Orientation::from_angles(
        g2.inclination_deg.to_radians(),
        g2.argument_deg.to_radians(),
    );
    collision
}

/// Sample a two-disk-galaxy encounter with each galaxy's Toomre orientation
/// applied. Returns the realization and the disk-1 particle mass (the
/// brightness unit).
fn sample_disks<H: SphericalHalo, S>(
    g1: &DiskGalaxySpec<S>,
    g2: &DiskGalaxySpec<S>,
    halo1: H,
    halo2: H,
    orbit: &OrbitSpec,
    c: DiskCounts,
    seed: u64,
) -> (State, f64) {
    let collision = oriented_collision(
        disk_galaxy(g1, halo1),
        disk_galaxy(g2, halo2),
        g1,
        g2,
        orbit,
    );
    let state = collision.sample(c.halo1, c.disk1, c.halo2, c.disk2, seed);
    (state, g1.disk_mass / c.disk1 as f64)
}

/// Sample a gas-rich two-disk encounter: both disks carry the shared gas
/// component (same `f_gas` and `c_s`), giving the six-population
/// [`DiskCollision::sample_gas`] realization. The brightness unit is the
/// disk-1 **stellar** particle mass `(1 − f)·disk_mass/disk1` — gas splits the
/// disk mass and renders volumetrically, so it must not dilute the splat unit.
#[allow(clippy::too_many_arguments)]
fn sample_disks_gas<H: SphericalHalo, S>(
    g1: &DiskGalaxySpec<S>,
    g2: &DiskGalaxySpec<S>,
    halo1: H,
    halo2: H,
    orbit: &OrbitSpec,
    c: DiskCounts,
    gas: &GasSpec,
    seed: u64,
) -> (State, f64) {
    let d1 = disk_galaxy(g1, halo1).with_gas(gas.fraction, gas.sound_speed);
    let d2 = disk_galaxy(g2, halo2).with_gas(gas.fraction, gas.sound_speed);
    let collision = oriented_collision(d1, d2, g1, g2, orbit);
    let state = collision.sample_gas(c.halo1, c.disk1, c.gas1, c.halo2, c.disk2, c.gas2, seed);
    let stellar_particle_mass = (1.0 - gas.fraction) * g1.disk_mass / c.disk1 as f64;
    (state, stellar_particle_mass)
}

fn truncated_nfw(h: &NfwSpec) -> TruncatedNfw {
    TruncatedNfw::new(Nfw::new(G, h.mvir, h.rs, h.c), h.skirt)
}

fn disk_info(
    halo_word: &str,
    state: &State,
    c: &DiskCounts,
    disk_particle_mass: f64,
    orbit: &OrbitSpec,
    eps: f64,
    gas: Option<&GasSpec>,
) -> String {
    let gas_note = match gas {
        Some(g) => format!(
            ", gas {}+{} (f_gas={} c_s={})",
            c.gas1, c.gas2, g.fraction, g.sound_speed
        ),
        None => String::new(),
    };
    format!(
        "IC: {} particles ({halo_word} {}+{}, disk {}+{}{gas_note}), disk particle mass \
         {disk_particle_mass:.3e}; e={} peri={} sep={}, eps={eps}",
        state.len(),
        c.halo1,
        c.halo2,
        c.disk1,
        c.disk2,
        orbit.eccentricity,
        orbit.pericenter,
        orbit.separation,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_preset(name: &str) -> ScenarioSpec {
        let text = preset(name).unwrap_or_else(|| panic!("preset `{name}` missing"));
        parse_scenario_toml(text).unwrap_or_else(|e| panic!("preset `{name}`: {e}"))
    }

    // The shared disk-family look (disk / cuspy / the zoo reuse it).
    const HALO1: [f32; 3] = [0.05, 0.035, 0.025];
    const DISK1: [f32; 3] = [1.0, 0.5, 0.25];
    const HALO2: [f32; 3] = [0.025, 0.035, 0.05];
    const DISK2: [f32; 3] = [0.35, 0.6, 1.0];

    fn disk_family_ramps() -> Vec<RampSpec> {
        vec![
            RampSpec {
                inner: HALO1,
                outer: HALO1,
            },
            RampSpec {
                inner: [1.0, 0.35, 0.1],
                outer: [0.55, 0.75, 1.0],
            },
            RampSpec {
                inner: HALO2,
                outer: HALO2,
            },
            RampSpec {
                inner: [1.0, 0.3, 0.45],
                outer: [0.4, 0.9, 0.9],
            },
        ]
    }

    // --- the three pre-M6f scenarios must survive the front-end bit-for-bit ----
    //
    // These literals are independent copies of the constants the hardcoded
    // constructors carried (xtask/src/main.rs before M6f). If a preset toml
    // drifts, these fail — same params, same seed ⇒ same movie.

    #[test]
    fn disk_preset_reproduces_the_hardcoded_scenario() {
        let expect = ScenarioSpec {
            name: "disk".into(),
            seed: 0x00C0_FFEE,
            model: ModelSpec::DiskPlummer {
                galaxy1: DiskGalaxySpec {
                    disk_mass: 0.15,
                    scale_length: 0.5,
                    hz_frac: 0.1,
                    rmax_frac: 4.0,
                    toomre_q: Some(1.5),
                    halo: PlummerSpec {
                        mass: 1.0,
                        scale: 1.0,
                    },
                    inclination_deg: 0.0,
                    argument_deg: 0.0,
                },
                galaxy2: DiskGalaxySpec {
                    disk_mass: 0.1,
                    scale_length: 0.45,
                    hz_frac: 0.1,
                    rmax_frac: 4.0,
                    toomre_q: Some(1.5),
                    halo: PlummerSpec {
                        mass: 0.7,
                        scale: 0.9,
                    },
                    inclination_deg: 0.0,
                    argument_deg: 0.0,
                },
                counts: Counts {
                    full: DiskCounts {
                        halo1: 5000,
                        disk1: 5000,
                        halo2: 3500,
                        disk2: 3500,
                        gas1: 0,
                        gas2: 0,
                    },
                    quick: DiskCounts {
                        halo1: 1500,
                        disk1: 1500,
                        halo2: 1000,
                        disk2: 1000,
                        gas1: 0,
                        gas2: 0,
                    },
                },
                gas: None,
            },
            orbit: OrbitSpec {
                eccentricity: 1.0,
                pericenter: 1.5,
                separation: 8.0,
            },
            sim: SimSpec {
                dt: 0.02,
                n_steps: 1500,
                snapshot_every: 25,
                snapshot_every_quick: None,
                eps: 0.05,
                adaptive: None,
                individual: None,
            },
            look: LookSpec {
                splat_size: 0.12,
                max_splat_px: None,
                frame_percentile: 0.98,
                palette: vec![HALO1, DISK1, HALO2, DISK2],
                ramps: disk_family_ramps(),
                sf_progenitors: vec![1, 3],
                gas: None,
                local_tone: None,
                age: None,
            },
            rig: RigSpec::Static,
            physics: None,
        };
        assert_eq!(parse_preset("disk"), expect);
    }

    #[test]
    fn dm_preset_reproduces_the_hardcoded_scenario() {
        let expect = ScenarioSpec {
            name: "dm".into(),
            seed: 0x0DEA_D000,
            model: ModelSpec::NfwMerger {
                galaxy1: NfwSpec {
                    mvir: 1.0,
                    rs: 1.0,
                    c: 10.0,
                    skirt: 3.0,
                },
                galaxy2: NfwSpec {
                    mvir: 0.5,
                    rs: 0.8,
                    c: 10.0,
                    skirt: 2.4,
                },
                counts: Counts {
                    full: MergerCounts {
                        halo1: 12000,
                        halo2: 6000,
                    },
                    quick: MergerCounts {
                        halo1: 2000,
                        halo2: 1000,
                    },
                },
            },
            orbit: OrbitSpec {
                eccentricity: 1.0,
                pericenter: 3.0,
                separation: 40.0,
            },
            sim: SimSpec {
                dt: 0.02,
                n_steps: 16_000,
                snapshot_every: 200,
                snapshot_every_quick: Some(400),
                eps: 0.05,
                adaptive: None,
                individual: None,
            },
            look: LookSpec {
                splat_size: 0.6,
                max_splat_px: None,
                frame_percentile: 0.97,
                palette: vec![[1.0, 0.55, 0.3], [0.35, 0.6, 1.0]],
                ramps: vec![
                    RampSpec {
                        inner: [1.0, 0.55, 0.3],
                        outer: [1.0, 0.85, 0.65],
                    },
                    RampSpec {
                        inner: [0.35, 0.6, 1.0],
                        outer: [0.7, 0.85, 1.0],
                    },
                ],
                sf_progenitors: vec![0, 1],
                gas: None,
                local_tone: None,
                age: None,
            },
            rig: RigSpec::OrbitTilt {
                azimuth_deg: [-90.0, 90.0],
                tilt_deg: [60.0, 60.0],
                window: 6,
            },
            physics: None,
        };
        assert_eq!(parse_preset("dm"), expect);
    }

    #[test]
    fn cuspy_preset_reproduces_the_hardcoded_scenario() {
        let expect = ScenarioSpec {
            name: "cuspy".into(),
            seed: 0x0CA5_D15C,
            model: ModelSpec::DiskNfw {
                galaxy1: DiskGalaxySpec {
                    disk_mass: 0.12,
                    scale_length: 0.6,
                    hz_frac: 0.1,
                    rmax_frac: 3.0,
                    toomre_q: None, // COLD — warm-in-a-cusp diverges (M5f)
                    halo: NfwSpec {
                        mvir: 1.0,
                        rs: 1.0,
                        c: 10.0,
                        skirt: 3.0,
                    },
                    inclination_deg: 0.0,
                    argument_deg: 0.0,
                },
                galaxy2: DiskGalaxySpec {
                    disk_mass: 0.08,
                    scale_length: 0.5,
                    hz_frac: 0.1,
                    rmax_frac: 3.0,
                    toomre_q: None,
                    halo: NfwSpec {
                        mvir: 0.7,
                        rs: 0.9,
                        c: 10.0,
                        skirt: 2.7,
                    },
                    inclination_deg: 0.0,
                    argument_deg: 0.0,
                },
                counts: Counts {
                    full: DiskCounts {
                        halo1: 10000,
                        disk1: 5000,
                        halo2: 8000,
                        disk2: 4000,
                        gas1: 0,
                        gas2: 0,
                    },
                    quick: DiskCounts {
                        halo1: 5000,
                        disk1: 3000,
                        halo2: 4000,
                        disk2: 2000,
                        gas1: 0,
                        gas2: 0,
                    },
                },
            },
            orbit: OrbitSpec {
                eccentricity: 1.0,
                pericenter: 1.5,
                separation: 8.0,
            },
            sim: SimSpec {
                dt: 0.02,
                n_steps: 1500,
                snapshot_every: 25,
                snapshot_every_quick: None,
                eps: 0.02,
                adaptive: None,
                individual: None,
            },
            look: LookSpec {
                splat_size: 0.15,
                max_splat_px: None,
                frame_percentile: 0.7,
                palette: vec![HALO1, DISK1, HALO2, DISK2],
                ramps: disk_family_ramps(),
                sf_progenitors: vec![1, 3],
                gas: None,
                local_tone: None,
                age: None,
            },
            rig: RigSpec::OrbitTilt {
                azimuth_deg: [-90.0, 40.0],
                tilt_deg: [55.0, 25.0],
                window: 8,
            },
            physics: None,
        };
        assert_eq!(parse_preset("cuspy"), expect);
    }

    // --- the Toomre zoo: relational physics gates -------------------------------
    //
    // The zoo presets are tuning-fluid in look/rig (rule 2: aesthetics are
    // eyeballed, not gated), so these pin only the PHYSICS relations that make
    // each scenario what it is.

    #[test]
    fn retro_is_the_cuspy_twin_with_both_spins_flipped() {
        let cuspy = parse_preset("cuspy");
        let retro = parse_preset("retro");
        // Same realization: seed, orbit, timing, softening all identical.
        assert_eq!(retro.seed, cuspy.seed);
        assert_eq!(retro.orbit, cuspy.orbit);
        assert_eq!(retro.sim, cuspy.sim);
        // Same galaxies, spins flipped to 180° about the same node line.
        let mut expect = cuspy.model.clone();
        match &mut expect {
            ModelSpec::DiskNfw {
                galaxy1, galaxy2, ..
            } => {
                galaxy1.inclination_deg = 180.0;
                galaxy2.inclination_deg = 180.0;
            }
            other => panic!("cuspy must be a disk-nfw model, got {other:?}"),
        }
        assert_eq!(retro.model, expect);
    }

    #[test]
    fn inclined_is_the_cuspy_twin_with_galaxy1_tilted_45deg() {
        let cuspy = parse_preset("cuspy");
        let inclined = parse_preset("inclined");
        assert_eq!(inclined.seed, cuspy.seed);
        assert_eq!(inclined.orbit, cuspy.orbit);
        assert_eq!(inclined.sim, cuspy.sim);
        let mut expect = cuspy.model.clone();
        match &mut expect {
            ModelSpec::DiskNfw { galaxy1, .. } => galaxy1.inclination_deg = 45.0,
            other => panic!("cuspy must be a disk-nfw model, got {other:?}"),
        }
        assert_eq!(inclined.model, expect);
    }

    #[test]
    fn bullseye_punches_through_the_target_center_along_its_spin() {
        let b = parse_preset("bullseye");
        let ModelSpec::DiskPlummer {
            galaxy1, galaxy2, ..
        } = &b.model
        else {
            panic!(
                "bullseye is the warm Plummer-disk family, got {:?}",
                b.model
            );
        };
        // Target spin axis along +y: inclination 90° about the node line at 180°.
        // (spin = (sin i·sin ω, −sin i·cos ω, cos i) = (0, 1, 0) — the direction
        // the relative orbit crosses pericenter.)
        assert_eq!(galaxy1.inclination_deg, 90.0);
        assert_eq!(galaxy1.argument_deg, 180.0);
        // Near-central: the pericenter is well inside the target disk.
        assert!(
            b.orbit.pericenter <= 0.5 * galaxy1.scale_length,
            "peri {} vs Rd {}",
            b.orbit.pericenter,
            galaxy1.scale_length
        );
        // Parabolic single passage, compact intruder.
        assert_eq!(b.orbit.eccentricity, 1.0);
        assert!(galaxy2.halo.scale <= 0.5 * galaxy1.halo.scale);
    }

    #[test]
    fn minor_merger_is_one_to_ten_bound_and_starts_near_apocenter() {
        let m = parse_preset("minor");
        let ModelSpec::DiskPlummer {
            galaxy1, galaxy2, ..
        } = &m.model
        else {
            panic!("minor is the warm Plummer-disk family, got {:?}", m.model);
        };
        let m1 = galaxy1.disk_mass + galaxy1.halo.mass;
        let m2 = galaxy2.disk_mass + galaxy2.halo.mass;
        assert!(
            (m1 / m2 - 10.0).abs() < 0.5,
            "mass ratio {} is not ~1:10",
            m1 / m2
        );
        // Bound, so the satellite comes back for repeated stripping…
        let e = m.orbit.eccentricity;
        assert!(e < 1.0);
        // …and the run covers at least two pericenter passages: starting near
        // apocenter, passages happen every radial period T_r = 2π√(a³/μ).
        let apo = m.orbit.pericenter * (1.0 + e) / (1.0 - e);
        assert!(m.orbit.separation <= apo);
        assert!(m.orbit.separation >= 0.9 * apo, "start near turnaround");
        let a_sma = m.orbit.pericenter / (1.0 - e);
        let mu = m1 + m2; // G = 1
        let t_r = std::f64::consts::TAU * (a_sma.powi(3) / mu).sqrt();
        let t_total = m.sim.dt * m.sim.n_steps as f64;
        assert!(
            t_total >= 2.0 * t_r,
            "T={t_total} covers <2 radial periods (T_r={t_r})"
        );
    }

    #[test]
    fn dolly_is_the_cuspy_encounter_flown_through() {
        // The M6g demo: same physics realization as cuspy (the rig is the whole
        // point), approaching from outside the framed scene to inside the
        // remnant, with the near plane short of the closest approach.
        let cuspy = parse_preset("cuspy");
        let d = parse_preset("dolly");
        assert_eq!(d.seed, cuspy.seed);
        assert_eq!(d.orbit, cuspy.orbit);
        assert_eq!(d.sim, cuspy.sim);
        assert_eq!(d.model, cuspy.model);
        let RigSpec::Dolly {
            distance_frac,
            fov_deg,
            near_frac,
            ..
        } = d.rig
        else {
            panic!("dolly preset must carry a dolly rig, got {:?}", d.rig);
        };
        assert!(
            distance_frac[0] > 1.0,
            "must start outside the framed scene: {distance_frac:?}"
        );
        assert!(
            distance_frac[1] < 1.0,
            "must end inside the remnant: {distance_frac:?}"
        );
        assert!(near_frac > 0.0 && near_frac < distance_frac[1]);
        assert!(fov_deg > 0.0 && fov_deg < 180.0);
    }

    // --- the gasrich showpiece: gas-rich `disk` twin (M7f) ----------------------

    #[test]
    fn gasrich_is_the_disk_encounter_with_a_stable_volumetric_gas_layer() {
        let g = parse_preset("gasrich");
        let ModelSpec::DiskPlummer {
            galaxy1,
            galaxy2,
            counts,
            gas,
        } = &g.model
        else {
            panic!("gasrich must be a disk-plummer model, got {:?}", g.model);
        };
        let gas = gas.expect("gasrich carries a [model.gas] component");
        // Gas-rich by design (the whole point — dark dust lanes need column depth).
        assert!(
            (0.15..=0.35).contains(&gas.fraction),
            "f_gas {} is not gas-rich",
            gas.fraction
        );
        // Marginally STABLE: min Q_gas ≥ 1 for BOTH disks (else `with_gas` panics in
        // the pipeline). Checked through the same IC helper `validate` gates on.
        for (gxy, which) in [(galaxy1, "galaxy1"), (galaxy2, "galaxy2")] {
            let disk = disk_galaxy(gxy, Plummer::new(G, gxy.halo.mass, gxy.halo.scale));
            disk.check_gas(gas.fraction, gas.sound_speed)
                .unwrap_or_else(|e| panic!("gasrich {which} gas unstable: {e}"));
        }
        // Declares a volumetric look (the tuned showpiece knobs, not the default).
        assert!(g.look.gas.is_some(), "gasrich must declare [look.gas]");
        // QUICK gas counts are positive and modest so the demo stays runnable.
        assert!(
            counts.quick.gas1 > 0 && counts.quick.gas2 > 0,
            "gasrich QUICK must carry gas particles"
        );
        assert!(
            counts.quick.gas1 + counts.quick.gas2 <= 4000,
            "gasrich QUICK gas count should stay demo-runnable"
        );
    }

    #[test]
    fn gasrich_build_threads_the_gas_look_and_sound_speed() {
        // The runtime `Scenario` carries BOTH gas-only fields (`Some`) for gasrich.
        let s = build_scenario(&parse_preset("gasrich"), true);
        assert!(s.sound_speed.is_some(), "gasrich threads its c_s");
        let gl = s.gas_look.expect("gasrich threads its [look.gas]");
        // The declared look, not the neutral default.
        assert_ne!(gl, GasLookValues::default());
    }

    // --- temperature colormap ([look.gas.temperature], incandescent-nebular-veil) ---

    // Appended to a gas-rich preset ([look.gas] already present), a new sub-table.
    const TEMP_BLOCK: &str =
        "\n[look.gas.temperature]\ncold = [0.0, 0.1, 0.8]\nhot = [1.0, 0.3, 0.05]\nu_lo = 0.05\nu_hi = 0.5\n";

    #[test]
    fn look_gas_temperature_parses_and_threads() {
        let text = format!("{}{}", preset("gasrich").unwrap(), TEMP_BLOCK);
        let spec = parse_scenario_toml(&text).expect("valid temperature block");
        let s = build_scenario(&spec, true);
        let t = s
            .gas_look
            .expect("gasrich threads [look.gas]")
            .temperature
            .expect("temperature threaded into GasLookValues");
        assert_eq!(t.cold, [0.0, 0.1, 0.8]);
        assert_eq!(t.hot, [1.0, 0.3, 0.05]);
        assert_eq!(t.u_lo, 0.05);
        assert_eq!(t.u_hi, 0.5);
    }

    #[test]
    fn gasrich_without_temperature_threads_none() {
        let s = build_scenario(&parse_preset("gasrich"), true);
        assert!(
            s.gas_look.unwrap().temperature.is_none(),
            "no [look.gas.temperature] ⇒ None (flat-tint path)"
        );
    }

    #[test]
    fn temperature_validation_rejects_bad_bands_colors_and_typos() {
        let base = format!("{}{}", preset("gasrich").unwrap(), TEMP_BLOCK);
        for (bad, why) in [
            (
                base.replace("u_hi = 0.5", "u_hi = 0.05"),
                "degenerate band u_lo == u_hi",
            ),
            (base.replace("u_lo = 0.05", "u_lo = 0.9"), "u_lo > u_hi"),
            (
                base.replace("u_hi = 0.5", "u_hi = nan"),
                "non-finite band edge",
            ),
            (
                base.replace("hot = [1.0, 0.3, 0.05]", "hot = [-1.0, 0.3, 0.05]"),
                "negative color component",
            ),
            (
                base.replace("cold = [0.0, 0.1, 0.8]", "cold = [0.0, nan, 0.8]"),
                "non-finite color component",
            ),
            (format!("{base}bogus = 1.0\n"), "unknown temperature key"),
        ] {
            assert!(parse_scenario_toml(&bad).is_err(), "should reject: {why}");
        }
    }

    // --- adiabatic gas EOS ([model.gas].gamma, incandescent-nebular-veil H5-C) ---

    /// gasrich with an adiabatic EOS spliced into `[model.gas]` (γ = 5/3 monatomic),
    /// injected right after the existing `sound_speed` line so it lands in that table.
    fn adiabatic_gasrich_toml() -> String {
        preset("gasrich").unwrap().replace(
            "sound_speed = 0.1",
            "sound_speed = 0.1\ngamma = 1.6666666666666667",
        )
    }

    #[test]
    fn adiabatic_gamma_threads_and_seeds_gas_internal_energy() {
        let spec = parse_scenario_toml(&adiabatic_gasrich_toml()).expect("valid adiabatic gas");
        let s = build_scenario(&spec, true);
        let gamma = s.gamma.expect("adiabatic scenario carries γ");
        assert!((gamma - 5.0 / 3.0).abs() < 1e-12, "γ threads verbatim");
        let c_s = s.sound_speed.expect("gas-rich ⇒ sound_speed");
        // Each gas particle seeds `u` so the t=0 pressure equals the isothermal
        // equilibrium the sampler baked: `P=(γ−1)ρu = c_s²ρ ⇒ (γ−1)u = c_s²`. Gate
        // the PRESSURE invariant, not the rearranged formula — it discriminates
        // pressure-match (passes) from sound-speed-match (`c_s²/γ ≠ c_s²`, fails).
        let cs2 = c_s * c_s;
        let mut gas_seen = 0usize;
        for (&u, k) in s.state.u.iter().zip(&s.state.kind) {
            match k {
                Species::Gas => {
                    assert!(
                        ((gamma - 1.0) * u - cs2).abs() < 1e-12,
                        "gas u must satisfy (γ−1)u = c_s²: u={u}, (γ−1)u={}, c_s²={cs2}",
                        (gamma - 1.0) * u
                    );
                    gas_seen += 1;
                }
                Species::Collisionless => assert_eq!(u, 0.0, "non-gas u stays 0"),
            }
        }
        assert!(gas_seen > 0, "adiabatic gasrich must carry gas particles");
    }

    #[test]
    fn isothermal_gasrich_leaves_gamma_none_and_u_zero() {
        // No `[model.gas].gamma` ⇒ isothermal: γ None, all u = 0 (byte-unchanged path).
        let s = build_scenario(&parse_preset("gasrich"), true);
        assert!(s.gamma.is_none(), "isothermal gasrich carries no γ");
        assert!(
            s.state.u.iter().all(|&u| u == 0.0),
            "isothermal ⇒ u = 0 everywhere"
        );
    }

    #[test]
    fn adiabatic_validation_rejects_gamma_at_or_below_one() {
        for bad_gamma in ["1.0", "0.5", "-1.0"] {
            let text = preset("gasrich").unwrap().replace(
                "sound_speed = 0.1",
                &format!("sound_speed = 0.1\ngamma = {bad_gamma}"),
            );
            assert!(
                parse_scenario_toml(&text).is_err(),
                "γ = {bad_gamma} must be rejected (adiabatic index must exceed 1)"
            );
        }
    }

    #[test]
    fn gasrich_adiabatic_preset_is_adiabatic_with_blackbody_temperature() {
        // The shipped showpiece: adiabatic γ = 5/3, gas u seeded in pressure
        // equilibrium, and a blackbody [look.gas.temperature] colormap threaded.
        let s = build_scenario(&parse_preset("gasrich-adiabatic"), true);
        let gamma = s.gamma.expect("gasrich-adiabatic is adiabatic");
        assert!((gamma - 5.0 / 3.0).abs() < 1e-12);
        assert!(s.adaptive.is_some(), "adiabatic ⇒ must ship [sim.adaptive]");
        assert_eq!(s.u_floor, 1.0e-6, "the shipped positive-u floor");
        let c_s = s.sound_speed.expect("gas-rich");
        let cs2 = c_s * c_s;
        let gas_count = s
            .state
            .u
            .iter()
            .zip(&s.state.kind)
            .filter(|(_, k)| **k == Species::Gas)
            .inspect(|(&u, _)| {
                assert!(
                    ((gamma - 1.0) * u - cs2).abs() < 1e-12,
                    "gas u must start in pressure equilibrium (γ−1)u = c_s²"
                );
            })
            .count();
        assert!(gas_count > 0, "the showpiece carries gas");
        let t = s
            .gas_look
            .expect("gas-rich ⇒ [look.gas]")
            .temperature
            .expect("gasrich-adiabatic threads [look.gas.temperature]");
        assert_eq!(t.cold, [0.75, 0.13, 0.05], "blackbody cold = dark red");
        assert_eq!(t.hot, [0.75, 0.82, 1.0], "blackbody hot = blue-white");
        assert!(t.u_lo < t.u_hi, "a non-degenerate band");
    }

    // --- registry + validation --------------------------------------------------

    #[test]
    fn every_preset_parses_and_wears_its_registry_name() {
        for (name, text) in PRESETS {
            let spec = parse_scenario_toml(text).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(&spec.name, name, "preset `{name}` must name itself");
            assert_eq!(preset(name), Some(*text));
        }
        assert_eq!(preset("no-such-scenario"), None);
    }

    #[test]
    fn parse_rejects_broken_physics_and_typos() {
        let disk = preset("disk").unwrap();
        let cuspy = preset("cuspy").unwrap();
        for (bad, why) in [
            (
                disk.replace("separation = 8.0", "separation = 1.0"),
                "separation below pericenter",
            ),
            (
                disk.replace("eccentricity = 1.0", "eccentricity = 0.0"),
                "non-positive eccentricity",
            ),
            (disk.replace("dt = 0.02", "dt = 0.0"), "non-positive dt"),
            (
                disk.replace("n_steps = 1500", "n_steps = 0"),
                "zero steps",
            ),
            (
                disk.replace("snapshot_every = 25", "snapshot_every = 0"),
                "zero snapshot cadence",
            ),
            (disk.replace("eps = 0.05", "eps = -1.0"), "negative softening"),
            (
                disk.replace("rmax_frac = 4.0", "rmax_frac = 0.5"),
                "truncation inside the scale length",
            ),
            (
                disk.replace("toomre_q = 1.5", "toomre_q = -2.0"),
                "non-positive Toomre Q",
            ),
            (
                disk.replace("disk_mass = 0.15", "disk_mass = 0.0"),
                "non-positive disk mass",
            ),
            (
                disk.replace(
                    "halo = { mass = 1.0, scale = 1.0 }",
                    "halo = { mass = -1.0, scale = 1.0 }",
                ),
                "non-positive halo mass",
            ),
            (
                disk.replace(
                    "frame_percentile = 0.98",
                    "frame_percentile = 1.5",
                ),
                "percentile beyond 1",
            ),
            (
                disk.replace("splat_size = 0.12", "splat_size = 0.0"),
                "non-positive splat size",
            ),
            (
                disk.replace("sf_progenitors = [1, 3]", "sf_progenitors = [9]"),
                "sf progenitor out of range",
            ),
            (
                disk.replace(
                    "palette = [[0.05, 0.035, 0.025], [1.0, 0.5, 0.25], [0.025, 0.035, 0.05], [0.35, 0.6, 1.0]]",
                    "palette = [[0.05, 0.035, 0.025], [1.0, 0.5, 0.25]]",
                ),
                "palette length != progenitor count",
            ),
            (
                disk.replace("kind = \"disk-plummer\"", "kind = \"warp-drive\""),
                "unknown model kind",
            ),
            (
                disk.replace("halo1 = 5000", "halo1 = 0"),
                "zero particle count",
            ),
            (
                format!("{disk}\nunknown_knob = 1.0\n"),
                "unknown top-level key",
            ),
            (
                disk.replace("splat_size = 0.12", "splat_sise = 0.12"),
                "typo'd look key",
            ),
            (
                cuspy.replace(
                    "halo = { mvir = 1.0, rs = 1.0, c = 10.0, skirt = 3.0 }",
                    "halo = { mvir = 1.0, rs = 1.0, c = 0.5, skirt = 3.0 }",
                ),
                "NFW concentration below 1",
            ),
            (
                cuspy.replace("window = 8", "window = 0"),
                "zero rig smoothing window",
            ),
            ("name = \"x\"".to_string(), "missing sections"),
        ] {
            assert!(
                parse_scenario_toml(&bad).is_err(),
                "should reject: {why}"
            );
        }
    }

    #[test]
    fn parse_rejects_broken_dolly_rigs() {
        let dolly = preset("dolly").unwrap();
        let cuspy = preset("cuspy").unwrap();
        for (bad, why) in [
            (dolly.replace("fov_deg = 55.0", "fov_deg = 0.0"), "zero fov"),
            (
                dolly.replace("fov_deg = 55.0", "fov_deg = 180.0"),
                "fov at 180° (tan pole)",
            ),
            (
                dolly.replace("near_frac = 0.02", "near_frac = 0.9"),
                "near plane at/past the closest approach",
            ),
            (
                dolly.replace("distance_frac = [5.0, 0.8]", "distance_frac = [5.0, 0.0]"),
                "non-positive dolly distance",
            ),
            (dolly.replace("fov_deg = 55.0", ""), "dolly missing fov"),
            (
                dolly.replace("kind = \"dolly\"", "kind = \"static\""),
                "static with dolly knobs",
            ),
            (
                cuspy.replace("window = 8", "window = 8\nfov_deg = 55.0"),
                "orbit-tilt with a dolly knob",
            ),
            (
                dolly.replace(
                    "direction_deg = [-60.0, 55.0]",
                    "direction_deg = [-60.0, nan]",
                ),
                "non-finite dolly direction",
            ),
        ] {
            assert!(parse_scenario_toml(&bad).is_err(), "should reject: {why}");
        }
    }

    #[test]
    fn bound_orbit_separation_beyond_apocenter_is_rejected() {
        // minor is the bound preset: apo = 1.2·1.7/0.3 = 6.8.
        let minor = preset("minor").unwrap();
        let bad = minor.replace("separation = 6.5", "separation = 7.5");
        assert!(parse_scenario_toml(&bad).is_err());
    }

    // --- [look.gas] scatter_softening (galaxy-render controls pass) -------------

    /// A declared `scatter_softening` threads parse → build into the resolved gas
    /// look as `Some(ε)`; absent, it resolves to `None` (the v1 per-cluster radius
    /// softening the shipped gasrich uses).
    #[test]
    fn look_gas_scatter_softening_threads_to_values() {
        let gasrich = preset("gasrich").unwrap();
        // Absent by default: the shipped gasrich keeps the v1 radius softening.
        let s0 = build_scenario(&parse_scenario_toml(gasrich).unwrap(), true);
        assert_eq!(
            s0.gas_look.unwrap().scatter_softening,
            None,
            "absent scatter_softening must resolve to None (v1 radius softening)"
        );
        // Declared: threads through as Some(ε).
        let with = gasrich.replace("shadows = true", "shadows = true\nscatter_softening = 0.08");
        let s1 = build_scenario(&parse_scenario_toml(&with).unwrap(), true);
        assert_eq!(
            s1.gas_look.unwrap().scatter_softening,
            Some(0.08),
            "declared scatter_softening must thread through to the gas look"
        );
    }

    /// `scatter_softening` without a positive `scattering` shapes nothing — the
    /// same dead-knob discipline as anisotropy / shadows / scatter_tint.
    #[test]
    fn scatter_softening_without_scattering_is_a_dead_knob() {
        let gasrich = preset("gasrich").unwrap();
        // Isolate softening as the ONLY scatter knob over a dead (0) scattering,
        // so the rejection is attributable to softening, not aniso/shadows.
        let bad = gasrich
            .replace("scattering = 800.0", "scattering = 0.0")
            .replace("anisotropy = 0.5", "")
            .replace("shadows = true", "scatter_softening = 0.08");
        assert!(
            parse_scenario_toml(&bad).is_err(),
            "scatter_softening without a positive scattering must be rejected"
        );
    }

    /// `scatter_softening` is a length: finite and strictly positive.
    #[test]
    fn scatter_softening_must_be_positive_finite() {
        let gasrich = preset("gasrich").unwrap();
        for bad_val in ["-0.1", "0.0", "nan"] {
            let bad = gasrich.replace(
                "shadows = true",
                &format!("shadows = true\nscatter_softening = {bad_val}"),
            );
            assert!(
                parse_scenario_toml(&bad).is_err(),
                "scatter_softening = {bad_val} must be rejected"
            );
        }
    }

    // --- [look.gas] shadow_bake (DDA/hierarchical deferral) ---------------------

    /// A declared `shadow_bake` threads parse → build into `Scenario.shadow_bake`
    /// (routed to RenderConfig, bit-identical either way); absent, it resolves to
    /// the brute default. The shipped gasrich now declares `"dda"`, so it is the
    /// positive fixture; the brute default is checked by stripping the knob.
    #[test]
    fn look_gas_shadow_bake_threads_to_scenario() {
        let gasrich = preset("gasrich").unwrap();
        // The shipped gasrich declares "dda" — it threads through to the strategy.
        let s1 = build_scenario(&parse_scenario_toml(gasrich).unwrap(), true);
        assert_eq!(
            s1.shadow_bake,
            ShadowBake::Dda,
            "shipped gasrich must thread shadow_bake = \"dda\""
        );
        // Strip the knob: it resolves to the brute default (bit-identical output).
        let no_bake = gasrich.replace("shadow_bake = \"dda\"\n", "");
        let s0 = build_scenario(&parse_scenario_toml(&no_bake).unwrap(), true);
        assert_eq!(
            s0.shadow_bake,
            ShadowBake::Brute,
            "absent shadow_bake must resolve to Brute"
        );
    }

    /// `shadow_bake` without `shadows = true` accelerates a bake that never runs —
    /// the same dead-knob discipline as the other gas knobs.
    #[test]
    fn shadow_bake_without_shadows_is_a_dead_knob() {
        let gasrich = preset("gasrich").unwrap();
        // Strip the shipped knob first (avoid a duplicate key), then re-add it in
        // place of `shadows = true`: the bake strategy is declared, but no shadows.
        let bad = gasrich
            .replace("shadow_bake = \"dda\"\n", "")
            .replace("shadows = true", "shadow_bake = \"dda\"");
        assert!(
            parse_scenario_toml(&bad).is_err(),
            "shadow_bake without shadows = true must be rejected"
        );
    }

    // --- [look.local_tone] (render-more-controls: baked local tonemap) ----------

    /// A declared `[look.local_tone]` threads parse → build into
    /// `Scenario.local_tone` as `Some(LocalToneConfig)`, with `radius`/`floor`
    /// defaulting to the same values the `regrade --local` CLI uses — so the
    /// shipped gasrich bakes exactly the "s2" tonemap the A/B viewer settled on.
    #[test]
    fn look_local_tone_threads_to_scenario() {
        let gasrich = preset("gasrich").unwrap();
        let s = build_scenario(&parse_scenario_toml(gasrich).unwrap(), true);
        assert_eq!(
            s.local_tone,
            Some(LocalToneConfig {
                strength: 2.0,
                radius: crate::DEFAULT_LOCAL_RADIUS,
                floor: crate::DEFAULT_LOCAL_FLOOR,
            }),
            "shipped gasrich must bake the s2 local tonemap (strength 2.0 at CLI defaults)"
        );
    }

    /// A scenario without `[look.local_tone]` resolves to `None` — the movie grade
    /// stays bit-identical to the pre-tonemap pipeline (the neutral-off convention).
    #[test]
    fn absent_local_tone_resolves_to_none() {
        let disk = preset("disk").unwrap();
        let s = build_scenario(&parse_scenario_toml(disk).unwrap(), true);
        assert_eq!(
            s.local_tone, None,
            "a scenario without [look.local_tone] must bake no local tonemap"
        );
    }

    /// A declared `strength = 0` is a bit-exact no-op — reject it loud (absence
    /// already means "off"), matching the dead-knob discipline of the scatter knobs.
    #[test]
    fn local_tone_strength_zero_is_rejected() {
        let gasrich = preset("gasrich").unwrap();
        let bad = gasrich.replace("strength = 2.0", "strength = 0.0");
        assert!(
            parse_scenario_toml(&bad).is_err(),
            "look.local_tone strength = 0 (a no-op) must be rejected"
        );
    }

    /// `strength` must be finite and positive; `radius` finite and positive;
    /// `floor` finite and in [0, 1] — the window `GradeConfig::validate` enforces,
    /// caught at parse time with a scenario-attributable message.
    #[test]
    fn local_tone_knobs_are_validated() {
        let gasrich = preset("gasrich").unwrap();
        // strength: non-finite / negative.
        for bad_val in ["-1.0", "nan"] {
            let bad = gasrich.replace("strength = 2.0", &format!("strength = {bad_val}"));
            assert!(
                parse_scenario_toml(&bad).is_err(),
                "look.local_tone strength = {bad_val} must be rejected"
            );
        }
        // radius: finite and > 0.
        for bad_val in ["0.0", "-4.0", "nan"] {
            let bad = gasrich.replace(
                "strength = 2.0",
                &format!("strength = 2.0\nradius = {bad_val}"),
            );
            assert!(
                parse_scenario_toml(&bad).is_err(),
                "look.local_tone radius = {bad_val} must be rejected"
            );
        }
        // floor: finite and in [0, 1].
        for bad_val in ["-0.1", "1.5", "nan"] {
            let bad = gasrich.replace(
                "strength = 2.0",
                &format!("strength = 2.0\nfloor = {bad_val}"),
            );
            assert!(
                parse_scenario_toml(&bad).is_err(),
                "look.local_tone floor = {bad_val} must be rejected"
            );
        }
    }

    // --- [physics.star_formation] scenario wiring (S6, natal-ember-forge F7) -----

    /// A declared `[physics.star_formation]` on a gas-rich scenario threads
    /// rho_thresh/efficiency/seed 1:1 into `Scenario.sf` as the runtime
    /// `StarFormationConfig` the CPU stepping loops consume.
    #[test]
    fn physics_star_formation_threads_to_scenario() {
        let gasrich = preset("gasrich").unwrap();
        let toml = format!(
            "{gasrich}\n[physics.star_formation]\nrho_thresh = 0.25\nefficiency = 0.1\nseed = 42\n"
        );
        let s = build_scenario(&parse_scenario_toml(&toml).unwrap(), true);
        assert_eq!(
            s.sf,
            Some(StarFormationConfig {
                rho_thresh: 0.25,
                efficiency: 0.1,
                seed: 42,
            }),
            "[physics.star_formation] must thread into Scenario.sf verbatim"
        );
    }

    /// A scenario without `[physics.star_formation]` resolves to `sf = None` — the
    /// stepping loops never call the SF operator, so every byte-path is untouched.
    #[test]
    fn absent_physics_resolves_to_no_sf() {
        let gasrich = preset("gasrich").unwrap();
        let s = build_scenario(&parse_scenario_toml(gasrich).unwrap(), true);
        assert_eq!(
            s.sf, None,
            "a scenario without [physics.star_formation] must carry no SF recipe"
        );
    }

    /// Star formation needs SPH gas (a pure-gravity solver's `sf_fields` returns
    /// zeros ⇒ nothing ever converts). Declared on a gas-free model it is a dead
    /// knob and rejected loud, mirroring the `[look.gas]` gas-presence gate.
    #[test]
    fn star_formation_on_gas_free_is_rejected() {
        let disk = preset("disk").unwrap();
        let bad = format!(
            "{disk}\n[physics.star_formation]\nrho_thresh = 0.25\nefficiency = 0.1\nseed = 1\n"
        );
        let err = parse_scenario_toml(&bad)
            .expect_err("[physics.star_formation] on a gas-free model must reject");
        assert!(
            err.contains("gas"),
            "reject message should name the missing gas, got: {err}"
        );
    }

    /// `rho_thresh` must be finite and > 0; `efficiency` must be finite and > 0 (a
    /// declared 0 is a no-op — absence already means "off").
    #[test]
    fn star_formation_knobs_are_validated() {
        let gasrich = preset("gasrich").unwrap();
        let base = |rho: &str, eff: &str| {
            format!(
                "{gasrich}\n[physics.star_formation]\nrho_thresh = {rho}\nefficiency = {eff}\nseed = 1\n"
            )
        };
        for bad_val in ["0.0", "-1.0", "nan"] {
            assert!(
                parse_scenario_toml(&base(bad_val, "0.1")).is_err(),
                "star_formation rho_thresh = {bad_val} must be rejected"
            );
            assert!(
                parse_scenario_toml(&base("0.25", bad_val)).is_err(),
                "star_formation efficiency = {bad_val} must be rejected"
            );
        }
    }

    // --- [look.age] age coloring wiring (S6, natal-ember-forge F6/F7) ------------

    /// A declared `[look.age]` threads young/strength/tau into the prep config as
    /// the `AgeColoring` map (a `[look]` coloring knob, not gas-gated — so a
    /// gas-free scenario carries it too, inert until a star has a formation time).
    #[test]
    fn look_age_threads_to_prep() {
        let disk = preset("disk").unwrap();
        let toml =
            format!("{disk}\n[look.age]\nyoung = [0.6, 0.75, 1.0]\nstrength = 0.8\ntau = 4.0\n");
        let s = build_scenario(&parse_scenario_toml(&toml).unwrap(), true);
        assert_eq!(
            s.prep.age,
            Some(AgeColoring {
                young: [0.6, 0.75, 1.0],
                strength: 0.8,
                tau: 4.0,
            }),
            "[look.age] must thread into PrepConfig.age verbatim"
        );
    }

    /// A scenario without `[look.age]` resolves to `prep.age = None` — the base
    /// color map, byte-for-byte (the neutral-off convention).
    #[test]
    fn absent_age_resolves_to_none() {
        let disk = preset("disk").unwrap();
        let s = build_scenario(&parse_scenario_toml(disk).unwrap(), true);
        assert_eq!(
            s.prep.age, None,
            "a scenario without [look.age] must carry no age tint"
        );
    }

    /// `strength` must be finite and > 0 (a declared 0 is a bit-exact no-op); `tau`
    /// must be finite and > 0 (the fade timescale). Mirrors the `local_tone`
    /// dead-knob discipline.
    #[test]
    fn age_knobs_are_validated() {
        let disk = preset("disk").unwrap();
        let base = |strength: &str, tau: &str| {
            format!(
                "{disk}\n[look.age]\nyoung = [0.6, 0.75, 1.0]\nstrength = {strength}\ntau = {tau}\n"
            )
        };
        for bad_val in ["0.0", "-1.0", "nan"] {
            assert!(
                parse_scenario_toml(&base(bad_val, "4.0")).is_err(),
                "look.age strength = {bad_val} must be rejected"
            );
            assert!(
                parse_scenario_toml(&base("0.8", bad_val)).is_err(),
                "look.age tau = {bad_val} must be rejected"
            );
        }
    }
}
