//! Gates for the `[model.gas]` front-end (M7c): a gas-rich `disk-plummer`
//! scenario must build the same six-population `DiskCollision::sample_gas` state
//! the IC produces directly, thread its single `sound_speed` onto the runtime
//! `Scenario`, and reject every gas mis-configuration the IC would panic on —
//! with a readable message, not a panic deep in `galaxy-ic`.
//!
//! Physics is gated: fraction ∈ (0,1), c_s > 0, a stable gas layer (min Q_gas ≥
//! 1), and positive gas counts. Aesthetics stay data — gas is *not* a splat
//! (it renders volumetrically), so the palette is unchanged (still 4 stellar
//! progenitors); gas color is the deferred `[look.gas]` uniform (M7f), never the
//! scenario palette.

use galaxy_ic::{DiskCollision, ExponentialDisk, Orientation, Plummer};
use galaxy_xtask::spec::{build_scenario, parse_scenario_toml, DiskCounts, ModelSpec};
use galaxy_xtask::{G, PEAK_BRIGHTNESS};

/// The QUICK counts the gas toml carries — small so the sampling gates stay fast.
const QC: DiskCounts = DiskCounts {
    halo1: 300,
    disk1: 250,
    halo2: 200,
    disk2: 150,
    gas1: 200,
    gas2: 150,
};

const SEED: u64 = 12345;
const FRACTION: f64 = 0.3;
const SOUND_SPEED: f64 = 0.1;

/// A valid gas-rich `disk-plummer` scenario: the `disk` preset galaxies, plus a
/// shared `[model.gas]` component and per-galaxy gas counts. c_s = 0.1, f = 0.3
/// keeps min Q_gas comfortably above 1 for these disks.
fn gas_toml() -> String {
    r#"
name = "gastest"
seed = 12345

[model]
kind = "disk-plummer"

[model.gas]
fraction = 0.3
sound_speed = 0.1

[model.galaxy1]
disk_mass = 0.15
scale_length = 0.5
hz_frac = 0.1
rmax_frac = 4.0
toomre_q = 1.5
halo = { mass = 1.0, scale = 1.0 }

[model.galaxy2]
disk_mass = 0.1
scale_length = 0.45
hz_frac = 0.1
rmax_frac = 4.0
toomre_q = 1.5
halo = { mass = 0.7, scale = 0.9 }

[model.counts.full]
halo1 = 800
disk1 = 600
halo2 = 600
disk2 = 500
gas1 = 400
gas2 = 300

[model.counts.quick]
halo1 = 300
disk1 = 250
halo2 = 200
disk2 = 150
gas1 = 200
gas2 = 150

[orbit]
eccentricity = 1.0
pericenter = 1.5
separation = 8.0

[sim]
dt = 0.01
n_steps = 100
snapshot_every = 10
eps = 0.05

[look]
splat_size = 0.12
frame_percentile = 0.98
palette = [[0.05, 0.035, 0.025], [1.0, 0.5, 0.25], [0.025, 0.035, 0.05], [0.35, 0.6, 1.0]]
sf_progenitors = [1, 3]

[[look.ramps]]
inner = [0.05, 0.035, 0.025]
outer = [0.05, 0.035, 0.025]

[[look.ramps]]
inner = [1.0, 0.35, 0.1]
outer = [0.55, 0.75, 1.0]

[[look.ramps]]
inner = [0.025, 0.035, 0.05]
outer = [0.025, 0.035, 0.05]

[[look.ramps]]
inner = [1.0, 0.3, 0.45]
outer = [0.4, 0.9, 0.9]

[rig]
kind = "static"
"#
    .to_string()
}

/// The hand-built expectation for `gas_toml`'s QUICK realization: both `disk`
/// galaxies warmed then given the shared gas component, in a coplanar-prograde
/// (zero-orientation) encounter, sampled with the gas counts. Independent of the
/// front-end — same params + same seed ⇒ same movie.
fn expected_gas_state() -> galaxy_core::State {
    let g1 = ExponentialDisk::new(0.15, 0.5, 0.1 * 0.5, 4.0 * 0.5, Plummer::new(G, 1.0, 1.0))
        .with_toomre_q(1.5)
        .with_gas(FRACTION, SOUND_SPEED);
    let g2 = ExponentialDisk::new(0.1, 0.45, 0.1 * 0.45, 4.0 * 0.45, Plummer::new(G, 0.7, 0.9))
        .with_toomre_q(1.5)
        .with_gas(FRACTION, SOUND_SPEED);
    let mut collision = DiskCollision::new(g1, g2, 1.0, 1.5, 8.0);
    // Zero orientations — coplanar prograde. Set explicitly so the expectation
    // matches `build_scenario`'s `Orientation::from_angles(0, 0)` bit-for-bit.
    collision.orient1 = Orientation::from_angles(0.0, 0.0);
    collision.orient2 = Orientation::from_angles(0.0, 0.0);
    collision.sample_gas(
        QC.halo1, QC.disk1, QC.gas1, QC.halo2, QC.disk2, QC.gas2, SEED,
    )
}

// --- the front-end reproduces the direct gas IC + carries the sound speed ------

#[test]
fn gas_scenario_builds_the_six_population_state() {
    let spec = parse_scenario_toml(&gas_toml()).expect("gas toml parses");
    // Sanity: the parsed model carries the shared gas component.
    match &spec.model {
        ModelSpec::DiskPlummer { gas, .. } => {
            let gas = gas.expect("gas component present");
            assert_eq!(gas.fraction, FRACTION);
            assert_eq!(gas.sound_speed, SOUND_SPEED);
        }
        other => panic!("gas toml must be disk-plummer, got {other:?}"),
    }

    let s = build_scenario(&spec, true);
    // Bit-identical to the direct `DiskCollision::sample_gas` realization.
    assert_eq!(s.state, expected_gas_state());
    // Six populations present: halos 0/2, disks 1/3, gas 4/5.
    let count = |p: u16| s.state.progenitor.iter().filter(|q| q.0 == p).count();
    assert_eq!(count(0), QC.halo1);
    assert_eq!(count(1), QC.disk1);
    assert_eq!(count(2), QC.halo2);
    assert_eq!(count(3), QC.disk2);
    assert_eq!(count(4), QC.gas1);
    assert_eq!(count(5), QC.gas2);
}

#[test]
fn gas_scenario_threads_the_single_sound_speed() {
    let s = build_scenario(&parse_scenario_toml(&gas_toml()).unwrap(), true);
    // The one c_s the pipeline hands both the IC and the force solver.
    assert_eq!(s.sound_speed, Some(SOUND_SPEED));
}

#[test]
fn gas_brightness_unit_is_the_stellar_disk_particle() {
    // Gas splits the disk mass, so the disk-1 SPLAT particle carries only the
    // stellar share (1 − f)·disk_mass/disk1 — gas renders volumetrically, not as
    // a splat, so it must not dilute the splat brightness unit.
    let s = build_scenario(&parse_scenario_toml(&gas_toml()).unwrap(), true);
    let stellar_particle_mass = (1.0 - FRACTION) * 0.15 / QC.disk1 as f64;
    assert_eq!(
        s.prep.brightness_per_mass,
        PEAK_BRIGHTNESS / stellar_particle_mass as f32
    );
}

#[test]
fn gas_free_build_reports_no_sound_speed() {
    // The single new runtime field must stay `None` on the gas-free path (the
    // byte-identity of the gas-free state itself is held by scenario_build.rs).
    let disk = parse_scenario_toml(galaxy_xtask::spec::preset("disk").unwrap()).unwrap();
    let s = build_scenario(&disk, true);
    assert_eq!(s.sound_speed, None);
}

// --- [look.gas] volumetric look threading (M7f) ---------------------------------

#[test]
fn gas_scenario_defaults_gas_look_when_look_gas_absent() {
    // `gas_toml` carries `[model.gas]` but NO `[look.gas]`: the runtime still gets
    // a gas look (`Some`), the neutral default the renderer would fall back to.
    let s = build_scenario(&parse_scenario_toml(&gas_toml()).unwrap(), true);
    assert_eq!(
        s.gas_look,
        Some(galaxy_xtask::spec::GasLookValues::default())
    );
}

#[test]
fn gas_look_values_default_matches_the_renderer_fallback() {
    // The whole point of the `GasLookValues::default` = `GasLook::default` promise
    // (a gas-rich scenario omitting `[look.gas]` renders with the neutral look the
    // renderer itself falls back to) rides on two hand-written `Default`s in
    // different crates. Nothing else gates them, so pin the equality field-by-field
    // — retuning one without the other would silently break the promise.
    let x = galaxy_xtask::spec::GasLookValues::default();
    let r = galaxy_render::GasLook::default();
    assert_eq!(x.color, r.color, "gas look default color must match");
    assert_eq!(
        x.emissivity, r.emissivity,
        "gas look default emissivity must match"
    );
    assert_eq!(x.opacity, r.opacity, "gas look default opacity must match");
    // The off ↔ off correspondence for the scatter option: the values' default
    // (scattering 0) must map to the renderer's default (scatter: None).
    assert_eq!(x.scattering, 0.0, "gas look default scattering must be off");
    assert_eq!(x.anisotropy, 0.0, "gas look default anisotropy must be 0");
    assert!(!x.shadows, "gas look default shadows must be off");
    assert!(
        r.scatter.is_none(),
        "renderer default must carry no scatter"
    );
}

#[test]
fn gas_scenario_carries_a_declared_look_gas() {
    // A `[look.gas]` table overrides the default and threads onto the `Scenario`.
    let toml = gas_toml().replace(
        "[rig]",
        "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = 12.0\n\n[rig]",
    );
    let s = build_scenario(&parse_scenario_toml(&toml).unwrap(), true);
    assert_eq!(
        s.gas_look,
        Some(galaxy_xtask::spec::GasLookValues {
            color: [0.5, 0.6, 0.95],
            emissivity: 0.25,
            opacity: 12.0,
            // Scatter knobs omitted ⇒ the option is OFF (bit-compat render).
            scattering: 0.0,
            anisotropy: 0.0,
            shadows: false,
        })
    );
}

// --- single-scatter starlight knobs (scattered-starlit-veil) ---------------------

#[test]
fn look_gas_scatter_knobs_thread_onto_the_scenario() {
    let toml = gas_toml().replace(
        "[rig]",
        "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = 12.0\n\
         scattering = 3.0\nanisotropy = 0.55\n\n[rig]",
    );
    let s = build_scenario(&parse_scenario_toml(&toml).unwrap(), true);
    assert_eq!(
        s.gas_look,
        Some(galaxy_xtask::spec::GasLookValues {
            color: [0.5, 0.6, 0.95],
            emissivity: 0.25,
            opacity: 12.0,
            scattering: 3.0,
            anisotropy: 0.55,
            shadows: false,
        })
    );
}

#[test]
fn look_gas_shadows_knob_threads_onto_the_scenario() {
    // umbral-lantern-lattice: the per-light shadow-volume option is one bool.
    let toml = gas_toml().replace(
        "[rig]",
        "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = 12.0\n\
         scattering = 3.0\nanisotropy = 0.55\nshadows = true\n\n[rig]",
    );
    let s = build_scenario(&parse_scenario_toml(&toml).unwrap(), true);
    assert_eq!(
        s.gas_look,
        Some(galaxy_xtask::spec::GasLookValues {
            color: [0.5, 0.6, 0.95],
            emissivity: 0.25,
            opacity: 12.0,
            scattering: 3.0,
            anisotropy: 0.55,
            shadows: true,
        })
    );
}

#[test]
fn look_gas_rejects_dead_shadows_knob() {
    // The anisotropy discipline (umbral-lantern-lattice): the `shadows` knob
    // PRESENT without a positive `scattering` shapes nothing — reject loud,
    // whatever its value.
    for (knobs, why) in [
        (
            "shadows = true",
            "shadows without scattering is a dead knob",
        ),
        (
            "scattering = 0.0\nshadows = true",
            "shadows with scattering = 0 is a dead knob",
        ),
        (
            "shadows = false",
            "a present-but-false shadows without scattering is still dead",
        ),
    ] {
        let toml = gas_toml().replace(
            "[rig]",
            &format!(
                "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = 12.0\n\
                 {knobs}\n\n[rig]"
            ),
        );
        assert!(parse_scenario_toml(&toml).is_err(), "should reject: {why}");
    }
}

#[test]
fn look_gas_rejects_bad_scatter_knobs() {
    for (knobs, why) in [
        ("scattering = -1.0", "negative scattering"),
        ("scattering = inf", "non-finite scattering"),
        ("scattering = 3.0\nanisotropy = 1.0", "|g| = 1 (HG pole)"),
        ("scattering = 3.0\nanisotropy = -1.5", "|g| > 1"),
        (
            "scattering = 3.0\nanisotropy = nan",
            "non-finite anisotropy",
        ),
        (
            "anisotropy = 0.5",
            "anisotropy without scattering is a dead knob",
        ),
        (
            "scattering = 0.0\nanisotropy = 0.5",
            "anisotropy with scattering = 0 is a dead knob",
        ),
    ] {
        let toml = gas_toml().replace(
            "[rig]",
            &format!(
                "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = 12.0\n\
                 {knobs}\n\n[rig]"
            ),
        );
        assert!(parse_scenario_toml(&toml).is_err(), "should reject: {why}");
    }
}

#[test]
fn gasrich_preset_carries_the_scatter_option_on() {
    // The showpiece preset ships with the option ENABLED so it can be judged;
    // disabling it (if the plain M7e look wins) is `scattering = 0` / knob
    // removal, which the off-is-off render gates hold bit-compatible.
    let s = build_scenario(
        &parse_scenario_toml(galaxy_xtask::spec::preset("gasrich").unwrap()).unwrap(),
        true,
    );
    let gl = s.gas_look.expect("gasrich threads its [look.gas]");
    assert!(
        gl.scattering > 0.0,
        "gasrich must ship with the scatter option on"
    );
    assert!(
        gl.anisotropy.is_finite() && gl.anisotropy.abs() < 1.0,
        "gasrich anisotropy {} must be a valid HG g",
        gl.anisotropy
    );
}

#[test]
fn gas_free_build_has_no_gas_look() {
    // Gas-only field: `None` on the gas-free path, exactly like `sound_speed`.
    let disk = parse_scenario_toml(galaxy_xtask::spec::preset("disk").unwrap()).unwrap();
    assert_eq!(build_scenario(&disk, true).gas_look, None);
}

#[test]
fn look_gas_on_a_gas_free_model_is_rejected() {
    // The `disk` preset is gas-free; a `[look.gas]` there renders nothing — a dead
    // knob, so it must fail loud, not silently do nothing.
    let disk = galaxy_xtask::spec::preset("disk").unwrap();
    let bad =
        format!("{disk}\n[look.gas]\ncolor = [1.0, 1.0, 1.0]\nemissivity = 1.0\nopacity = 1.0\n");
    assert!(
        parse_scenario_toml(&bad).is_err(),
        "look.gas on a gas-free model must be rejected"
    );
}

#[test]
fn look_gas_rejects_an_unknown_key() {
    // A typo'd volumetric knob must fail, not silently do nothing (deny_unknown_fields).
    let toml = gas_toml().replace(
        "[rig]",
        "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = 12.0\nkappa = 3.0\n\n[rig]",
    );
    assert!(
        parse_scenario_toml(&toml).is_err(),
        "unknown look.gas key must fail"
    );
}

#[test]
fn look_gas_rejects_negative_knobs() {
    // Emissivity/opacity are physical rates ≥ 0; a negative value is nonsense.
    let toml = gas_toml().replace(
        "[rig]",
        "[look.gas]\ncolor = [0.5, 0.6, 0.95]\nemissivity = 0.25\nopacity = -1.0\n\n[rig]",
    );
    assert!(
        parse_scenario_toml(&toml).is_err(),
        "negative opacity must fail"
    );
}

// --- physics is gated: readable rejects, not galaxy-ic panics -------------------

#[test]
fn parse_rejects_broken_gas_physics() {
    let base = gas_toml();
    for (bad, why) in [
        (
            base.replace("fraction = 0.3", "fraction = 1.5"),
            "f_gas > 1",
        ),
        (
            base.replace("fraction = 0.3", "fraction = 0.0"),
            "f_gas = 0",
        ),
        (
            base.replace("sound_speed = 0.1", "sound_speed = 0.0"),
            "non-positive sound speed",
        ),
        (
            // c_s ∝ Q_gas linearly ⇒ a tiny sound speed fragments the gas layer
            // (min Q_gas ≪ 1); the reject must name Q_gas.
            base.replace("sound_speed = 0.1", "sound_speed = 0.001"),
            "gravitationally unstable gas (min Q_gas < 1)",
        ),
        (base.replace("gas1 = 400", "gas1 = 0"), "zero gas count"),
    ] {
        assert!(parse_scenario_toml(&bad).is_err(), "should reject: {why}");
    }
}

#[test]
fn parse_rejects_unstable_gas_naming_q_gas() {
    let bad = gas_toml().replace("sound_speed = 0.1", "sound_speed = 0.001");
    let err = parse_scenario_toml(&bad).expect_err("unstable gas must be rejected");
    assert!(
        err.contains("Q_gas"),
        "instability message must name Q_gas, got: {err}"
    );
}

#[test]
fn parse_rejects_unknown_gas_key() {
    let bad = gas_toml().replace("sound_speed = 0.1", "sound_speeed = 0.1");
    assert!(
        parse_scenario_toml(&bad).is_err(),
        "typo'd gas key must fail"
    );
}

#[test]
fn parse_rejects_gas_on_non_disk_plummer_kinds() {
    // Gas is a disk-plummer-only knob in v1 (the IC supports NFW-halo gas, but
    // the pipeline keeps it minimal): a `[model.gas]` on any other kind fails.
    for kind in ["dm", "cuspy"] {
        let base = galaxy_xtask::spec::preset(kind).unwrap();
        let bad = format!("{base}\n[model.gas]\nfraction = 0.3\nsound_speed = 0.1\n");
        assert!(
            parse_scenario_toml(&bad).is_err(),
            "gas on `{kind}` must be rejected"
        );
    }
}
