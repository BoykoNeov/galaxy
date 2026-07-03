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
use galaxy_renderprep::PrepConfig;

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
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
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

/// Camera choreography (the M6d rig), as data.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case")]
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
];

/// Look up a checked-in preset's toml text by canonical name.
pub fn preset(_name: &str) -> Option<&'static str> {
    todo!("M6f: preset registry lookup")
}

/// Parse and validate a `scenario.toml`. All errors — toml syntax, unknown
/// keys/kinds, and every physics/look invariant the pipeline relies on — come
/// back as a single human-readable message.
pub fn parse_scenario_toml(_text: &str) -> Result<ScenarioSpec, String> {
    todo!("M6f: toml parsing + validation")
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
}

/// Build the runtime [`Scenario`] from a validated spec: sample the IC
/// (deterministic in `spec.seed`), pick full/QUICK counts, cadence, and frame
/// size, and assemble the prep config (palette + the always-on M6a/M6e density
/// features, keyed to the scenario's ε). Same spec + same seed + same `quick`
/// ⇒ bit-identical `State`.
pub fn build_scenario(_spec: &ScenarioSpec, _quick: bool) -> Scenario {
    todo!("M6f: spec-driven scenario construction")
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
    fn bound_orbit_separation_beyond_apocenter_is_rejected() {
        // minor is the bound preset: apo = 1.2·1.7/0.3 = 6.8.
        let minor = preset("minor").unwrap();
        let bad = minor.replace("separation = 6.5", "separation = 7.5");
        assert!(parse_scenario_toml(&bad).is_err());
    }
}
