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

use galaxy_core::State;
use galaxy_ic::{
    DiskCollision, ExponentialDisk, Nfw, NfwCollision, Orientation, Plummer, SphericalHalo,
    TruncatedNfw,
};
use galaxy_renderprep::{ColorMode, DensityColoring, PrepConfig, SizeByDensity};

use crate::{
    DENSITY_K, DENSITY_STRENGTH, FRAME_H, FRAME_W, G, PEAK_BRIGHTNESS, QUICK_H, QUICK_W,
    SIZE_MAX_FRAC, SIZE_MIN_FRAC, SUBFRAMES,
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
    /// Two exponential disks in live Plummer halos (`DiskCollision<Plummer>`).
    DiskPlummer {
        galaxy1: DiskGalaxySpec<PlummerSpec>,
        galaxy2: DiskGalaxySpec<PlummerSpec>,
        counts: Counts<DiskCounts>,
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

/// Full-resolution vs QUICK-preview particle counts.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Counts<T> {
    pub full: T,
    pub quick: T,
}

/// Particle counts for a two-disk-galaxy encounter (four species).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiskCounts {
    pub halo1: usize,
    pub disk1: usize,
    pub halo2: usize,
    pub disk2: usize,
}

impl DiskCounts {
    /// Total particle count.
    pub fn total(&self) -> usize {
        self.halo1 + self.disk1 + self.halo2 + self.disk2
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
}

/// Splat look and framing.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LookSpec {
    /// Base splat radius in world units.
    pub splat_size: f32,
    /// Percentile radius the camera frames on (0, 1].
    pub frame_percentile: f32,
    /// Per-progenitor base colors — length must equal the model's progenitor
    /// count (4 for disk models, 2 for the merger).
    pub palette: Vec<[f32; 3]>,
    /// Per-progenitor `--color initial-radius` ramps — same length rule.
    pub ramps: Vec<RampSpec>,
    /// Progenitors the star-formation compression proxy applies to.
    pub sf_progenitors: Vec<u16>,
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
}

impl TryFrom<ModelTable> for ModelSpec {
    type Error = String;

    fn try_from(t: ModelTable) -> Result<Self, String> {
        fn field<T: serde::de::DeserializeOwned>(v: toml::Value, what: &str) -> Result<T, String> {
            v.try_into().map_err(|e| format!("model {what}: {e}"))
        }
        match t.kind.as_str() {
            "disk-plummer" => Ok(ModelSpec::DiskPlummer {
                galaxy1: field(t.galaxy1, "galaxy1")?,
                galaxy2: field(t.galaxy2, "galaxy2")?,
                counts: field(t.counts, "counts")?,
            }),
            "disk-nfw" => Ok(ModelSpec::DiskNfw {
                galaxy1: field(t.galaxy1, "galaxy1")?,
                galaxy2: field(t.galaxy2, "galaxy2")?,
                counts: field(t.counts, "counts")?,
            }),
            "nfw-merger" => Ok(ModelSpec::NfwMerger {
                galaxy1: field(t.galaxy1, "galaxy1")?,
                galaxy2: field(t.galaxy2, "galaxy2")?,
                counts: field(t.counts, "counts")?,
            }),
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
            4
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

    // Look: splat/framing knobs + palette/ramp lengths tied to the model.
    positive(f64::from(s.look.splat_size), "look splat_size")?;
    if !(s.look.frame_percentile > 0.0 && s.look.frame_percentile <= 1.0) {
        return Err(format!(
            "look frame_percentile must be in (0, 1], got {}",
            s.look.frame_percentile
        ));
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
    pub rig: Rig,
    /// Per-progenitor `(inner, outer)` ramp for `--color initial-radius` (M6e).
    pub ramp: Vec<([f32; 3], [f32; 3])>,
    /// Progenitors the star-formation compression proxy applies to (M6e).
    pub sf_progenitors: Vec<u16>,
    pub info: String,
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
    let (state, unit_mass, info) = match &spec.model {
        ModelSpec::DiskPlummer {
            galaxy1,
            galaxy2,
            counts,
        } => {
            let c = if quick { counts.quick } else { counts.full };
            let h1 = Plummer::new(G, galaxy1.halo.mass, galaxy1.halo.scale);
            let h2 = Plummer::new(G, galaxy2.halo.mass, galaxy2.halo.scale);
            let (state, unit_mass) = sample_disks(galaxy1, galaxy2, h1, h2, orbit, c, spec.seed);
            let info = disk_info("halo", &state, &c, unit_mass, orbit, spec.sim.eps);
            (state, unit_mass, info)
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
            let info = disk_info("cuspy halo", &state, &c, unit_mass, orbit, spec.sim.eps);
            (state, unit_mass, info)
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
            (state, unit_mass, info)
        }
    };

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
    let mut collision = DiskCollision::new(
        disk_galaxy(g1, halo1),
        disk_galaxy(g2, halo2),
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
    let state = collision.sample(c.halo1, c.disk1, c.halo2, c.disk2, seed);
    (state, g1.disk_mass / c.disk1 as f64)
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
) -> String {
    format!(
        "IC: {} particles ({halo_word} {}+{}, disk {}+{}), disk particle mass \
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
                    },
                    quick: DiskCounts {
                        halo1: 1500,
                        disk1: 1500,
                        halo2: 1000,
                        disk2: 1000,
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
                eps: 0.05,
            },
            look: LookSpec {
                splat_size: 0.12,
                frame_percentile: 0.98,
                palette: vec![HALO1, DISK1, HALO2, DISK2],
                ramps: disk_family_ramps(),
                sf_progenitors: vec![1, 3],
            },
            rig: RigSpec::Static,
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
            },
            look: LookSpec {
                splat_size: 0.6,
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
            },
            rig: RigSpec::OrbitTilt {
                azimuth_deg: [-90.0, 90.0],
                tilt_deg: [60.0, 60.0],
                window: 6,
            },
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
                    },
                    quick: DiskCounts {
                        halo1: 5000,
                        disk1: 3000,
                        halo2: 4000,
                        disk2: 2000,
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
            },
            look: LookSpec {
                splat_size: 0.15,
                frame_percentile: 0.7,
                palette: vec![HALO1, DISK1, HALO2, DISK2],
                ramps: disk_family_ramps(),
                sf_progenitors: vec![1, 3],
            },
            rig: RigSpec::OrbitTilt {
                azimuth_deg: [-90.0, 40.0],
                tilt_deg: [55.0, 25.0],
                window: 8,
            },
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
}
