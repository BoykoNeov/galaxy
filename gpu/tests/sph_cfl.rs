//! GPU-SPH G4 — the GPU CFL reduction ([`GpuCfl`]) validated against the CPU oracle
//! [`galaxy_solvers::sph::max_stable_dt`] and its per-target `dt_i = C_cfl·h_i/v_sig,i`.
//!
//! ## What is (and is not) gated (D1/D5/D6)
//! The GPU computes, per gas particle, the Gadget-style stable step
//! `dt_i = C_cfl · h_i / v_sig,i`, with `v_sig,i = max_j (2c_s − 3 w_ij)` over
//! APPROACHING neighbors (`w_ij = (v_i−v_j)·r̂_ij < 0`), floored at `2c_s`. The
//! milestone bound is `min_i dt_i` (D6's per-batch adaptive-dt substrate). `h` is an
//! INPUT (the [`GpuDensity`] pass ran first), bit-identical to what the oracle computes
//! internally — so this isolates the CFL reduction from the density root-find (G2).
//!
//! ## Why the SCALAR-min gate alone is too weak — gate the per-target VECTOR
//! `min_i dt_i` masks per-particle errors far worse than G3's RMS did: a per-target
//! radius bug (gathering `SUPPORT·h_i` instead of the global `SUPPORT·h_max`) only moves
//! the global min if the affected particle *is* the minimizer. So the sharp gate is
//! `gpu_cfl_matches_cpu_per_target` on the whole `dt_i` vector; the scalar-min gates
//! (`gpu_cfl_matches_oracle_*`) confirm the reduction and tie the whole thing to the
//! trusted `max_stable_dt` oracle on top.
//!
//! ## Two ways this differs from the G3 hydro force (CFL has no `grad_w` to save it)
//!   1. **The coupling cutoff is EXPLICIT.** G3 leaned on `grad_w` vanishing past
//!      `2·max(h_i,h_j)`; CFL has no kernel gradient, so the kernel must *reject*
//!      neighbors in `[SUPPORT·max(h_i,h_j), SUPPORT·h_max)` with a hard `r >=`
//!      test — else a far approacher wrongly raises `v_sig` and shrinks `dt` too much.
//!   2. **`w` divides by `r` (length), not `r²`.** The projected relative velocity is
//!      `(v_i−v_j)·r_ij / |r_ij|`; the sign of `w` decides approach/recede.
//!
//! ## The contract a `0` would violate (catastrophic)
//!   * No gas / empty ⇒ `max_stable_dt = +∞` (no hydro constraint), NOT `0` — a `0`
//!     would falsely report "every dt is too large."
//!   * A lone particle ⇒ FINITE `C_cfl·h/(2c_s)` (v_sig hits the floor), not `+∞`.
//!
//! GPU-gated: these need a wgpu adapter; without one `GpuCfl::new` returns `NoAdapter`
//! and the tests fail loudly (the M3/M4 GPU-invariants convention).

use galaxy_core::{DVec3, Species, State};
use galaxy_gpu::GpuCfl;
use galaxy_solvers::sph::{
    density_adaptive, max_stable_dt, DensityConfig, Eos, HashGrid, HydroParams, SUPPORT,
};

const C_CFL: f64 = 0.25; // ≠ 1 so a "forgot to multiply by C_cfl" bug can't hide (it
                         // factors out of the min); the analog of G3's varying mass.

fn cfl() -> GpuCfl {
    GpuCfl::new().expect("wgpu adapter required for GPU SPH CFL tests")
}

/// Deterministic pseudo-random `[0, 1)` stream (the same LCG as the other GPU tests).
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut state = seed | 1;
    move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Centrally-concentrated, origin-centered cloud (radius `R·u`, u uniform): a dense
/// core and diffuse tail so the adaptive `h` spans a wide range (the measured gas-disk
/// regime). Same generator as the G2/G3 gates.
fn concentrated_cloud(seed: u64, n: usize, radius: f64) -> Vec<DVec3> {
    let mut next = rng(seed);
    (0..n)
        .map(|_| {
            let r = radius * next();
            let z = 2.0 * next() - 1.0;
            let phi = std::f64::consts::TAU * next();
            let s = (1.0 - z * z).max(0.0).sqrt();
            DVec3::new(s * phi.cos(), s * phi.sin(), z) * r
        })
        .collect()
}

/// A MIXED velocity field: each component uniform in `[−scale, scale]`. Uncorrelated
/// with position, so neighbor pairs approach (`w < 0`) or recede (`w ≥ 0`) ~50/50 —
/// which is what drives `v_sig` above the `2c_s` floor for a material set of particles
/// (a static field would leave the whole max-over-approachers dead).
fn random_velocities(seed: u64, n: usize, scale: f64) -> Vec<DVec3> {
    let mut next = rng(seed);
    (0..n)
        .map(|_| {
            DVec3::new(
                scale * (2.0 * next() - 1.0),
                scale * (2.0 * next() - 1.0),
                scale * (2.0 * next() - 1.0),
            )
        })
        .collect()
}

fn gas_state(pos: Vec<DVec3>, vel: Vec<DVec3>) -> State {
    let n = pos.len();
    let mut s = State::from_phase_space(pos, vel, vec![1.0; n]);
    for k in s.kind.iter_mut() {
        *k = Species::Gas;
    }
    s
}

/// Per-target signal velocity `v_sig,i = max_j (2c_s − 3 w_ij)` over approaching
/// neighbors, floored at `2c_s` — an independent hand-transcription of the oracle's
/// inner loop ([`galaxy_solvers::sph::cfl::max_stable_dt`]), gathering at the GLOBAL
/// `SUPPORT·h_max` and rejecting each pair outside its own `SUPPORT·max(h_i,h_j)`.
fn per_target_v_sig(pos: &[DVec3], vel: &[DVec3], h: &[f64], cs: f64) -> Vec<f64> {
    let n = pos.len();
    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    let grid = HashGrid::build(pos, SUPPORT * h_max);
    let two_cs = 2.0 * cs;
    (0..n)
        .map(|i| {
            let ngb = grid.neighbours_within(pos, pos[i], SUPPORT * h_max);
            let mut v_sig = two_cs;
            for &j in &ngb {
                if j == i {
                    continue;
                }
                let r_ij = pos[i] - pos[j];
                let r = r_ij.length();
                if r == 0.0 || r >= SUPPORT * h[i].max(h[j]) {
                    continue; // outside the pair's force-coupling range ⇒ no drive
                }
                let w = (vel[i] - vel[j]).dot(r_ij) / r;
                if w < 0.0 {
                    v_sig = v_sig.max(two_cs - 3.0 * w);
                }
            }
            v_sig
        })
        .collect()
}

/// Per-target stable step `dt_i = C_cfl · h_i / v_sig,i` — the vector the GPU must match
/// element-wise (its min is `max_stable_dt`).
fn cpu_per_target_dt(pos: &[DVec3], vel: &[DVec3], h: &[f64], cs: f64, c_cfl: f64) -> Vec<f64> {
    let v_sig = per_target_v_sig(pos, vel, h, cs);
    (0..pos.len()).map(|i| c_cfl * h[i] / v_sig[i]).collect()
}

/// Worst per-element relative error. `dt_i` are strictly positive and bounded away from
/// zero (`v_sig ≥ 2c_s > 0`, `h > 0`), so a plain relative error per element is clean —
/// no RMS-accel normalization is needed (unlike the G3 force, which has near-nulls).
fn worst_rel(gpu: &[f64], cpu: &[f64]) -> f64 {
    gpu.iter()
        .zip(cpu)
        .map(|(g, c)| (g - c).abs() / c.abs())
        .fold(0.0_f64, f64::max)
}

/// Count ASYMMETRIC-coupling pairs that are ALSO APPROACHING:
/// `SUPPORT·min(h_i,h_j) ≤ r < SUPPORT·max(h_i,h_j)` and `w_ij < 0`. These are exactly
/// the pairs where the diffuse (large-h) particle's support reaches the compact one but
/// not vice-versa AND the approach drives `v_sig` — so they make the per-target radius
/// bug OBSERVABLE in the vector gate (a per-target-`h_i` gather would miss the drive on
/// the compact side). A field with none of these would leave the vector gate testing
/// only symmetric pairs, where the radius bug is invisible.
fn asymmetric_approaching_count(pos: &[DVec3], vel: &[DVec3], h: &[f64]) -> usize {
    let n = pos.len();
    let mut count = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            let r_ij = pos[i] - pos[j];
            let r = r_ij.length();
            let lo = SUPPORT * h[i].min(h[j]);
            let hi = SUPPORT * h[i].max(h[j]);
            if r >= lo && r < hi && (vel[i] - vel[j]).dot(r_ij) < 0.0 {
                count += 1;
            }
        }
    }
    count
}

// ---------------------------------------------------------------------------
// CPU-only precondition guard: the v_sig max-over-approachers is exercised (the
// analog of G3's "both sides of vr = 0" viscosity guard).
// ---------------------------------------------------------------------------

/// The vector/scalar gates only test the signal-velocity machinery if the field drives
/// `v_sig` above the `2c_s` floor for a MATERIAL fraction of particles — AND, crucially,
/// for the MINIMIZER (the particle that sets `max_stable_dt`). If the minimizer sat at
/// the floor, `max_stable_dt` would just be `C_cfl·h_min/(2c_s)` and the entire
/// approach-detection loop could be broken without moving the number. CPU-only, so green
/// from the red commit on — a standing guard on the test's velocity field.
#[test]
fn gpu_cfl_v_sig_above_floor_exercised() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 3000, 5.0);
    let vel = random_velocities(0x5EED, pos.len(), 1.0);
    let mass = vec![1.0; pos.len()];
    let cs = HydroParams::default().sound_speed();
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    let two_cs = 2.0 * cs;
    let v_sig = per_target_v_sig(&pos, &vel, &dens.h, cs);
    let above = v_sig.iter().filter(|&&v| v > two_cs).count();
    let frac = above as f64 / pos.len() as f64;
    assert!(
        frac > 0.2,
        "v_sig must exceed the 2c_s floor for a material fraction (got {frac:.2}) — else \
         the max-over-approachers is dead and the CFL gates test only the floor"
    );

    // The MINIMIZER must be driven above the floor, else max_stable_dt is set by the
    // static floor and a broken approach loop cannot be detected by the scalar gate.
    let dt = cpu_per_target_dt(&pos, &vel, &dens.h, cs, C_CFL);
    let argmin = (0..dt.len())
        .min_by(|&a, &b| dt[a].total_cmp(&dt[b]))
        .expect("non-empty");
    assert!(
        v_sig[argmin] > two_cs,
        "the minimizing particle {argmin} sits at the v_sig floor ({}) — the scalar-min \
         gate would not see a broken approach loop; pick a livelier field",
        v_sig[argmin]
    );
}

// ---------------------------------------------------------------------------
// SHARP gate: the per-target dt_i vector (catches a radius bug on ANY particle).
// ---------------------------------------------------------------------------

/// GPU per-target `dt_i` vs the CPU per-target oracle on the concentrated cloud with a
/// mixed velocity field. This is the sharp gate: unlike the scalar min, it fails when a
/// per-target-radius bug moves `dt_i` on ANY particle, not only the minimizer. The
/// precondition assert guarantees the field actually contains the asymmetric-coupling
/// approaching pairs that make that bug observable.
#[test]
fn gpu_cfl_matches_cpu_per_target() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 3000, 5.0);
    let vel = random_velocities(0x5EED, pos.len(), 1.0);
    let mass = vec![1.0; pos.len()];
    let params = HydroParams::default();
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    // Precondition: the cloud/field must contain asymmetric-coupling APPROACHING pairs,
    // else the per-target radius bug this gate targets is invisible (no pair where one
    // particle's support reaches the other but not vice-versa, with an active approach).
    let asym = asymmetric_approaching_count(&pos, &vel, &dens.h);
    assert!(
        asym > 100,
        "cloud/field must contain asymmetric-coupling APPROACHING pairs (got {asym}) so a \
         per-target-h_i gather bug is observable in the vector gate"
    );

    let cpu = cpu_per_target_dt(&pos, &vel, &dens.h, params.sound_speed(), C_CFL);
    let gpu = cfl().per_target_dt(&pos, &vel, &dens.h, &params, C_CFL);

    assert_eq!(gpu.len(), cpu.len());
    for (i, &d) in gpu.iter().enumerate() {
        assert!(
            d.is_finite() && d > 0.0,
            "dt_i must be finite positive at {i}: {d}"
        );
    }
    // dt_i is a single max + a single divide — NO accumulation — so the GPU-vs-oracle
    // agreement is tight f32: measured worst ≈ 1.0e-6 (one f32 divide plus a lone
    // near-boundary particle whose v_sig differs by a marginal-approacher ulp). The
    // bound sits ~10× above that — a REAL bug (wrong coupling cutoff, per-target radius,
    // or w/r² instead of w/r) is ≫ 1%, so this fails hard on any of them while keeping
    // cross-device f32 headroom. A loose MEASURED value here would be a smell (a
    // coupling/radius mismatch), not roundoff to absorb — hence the tight bound.
    let worst = worst_rel(&gpu, &cpu);
    assert!(
        worst < 1.0e-5,
        "GPU CFL per-target worst rel err {worst:.3e}"
    );
}

// ---------------------------------------------------------------------------
// scalar-min gates tied to the trusted max_stable_dt oracle.
// ---------------------------------------------------------------------------

/// GPU `max_stable_dt` (the reduction) vs the trusted oracle on the mixed-field cloud.
/// The oracle recomputes `h` internally from the gas positions; feeding the GPU the
/// `density_adaptive(pos,…)` `h` (all particles gas ⇒ same positions) keeps `h`
/// bit-identical, so any discrepancy is the reduction/CFL kernel, not the density path.
#[test]
fn gpu_cfl_matches_oracle_scalar() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xC0FFEE, 3000, 5.0);
    let vel = random_velocities(0x5EED, pos.len(), 1.0);
    let params = HydroParams::default();
    let state = gas_state(pos.clone(), vel.clone());
    let dens = density_adaptive(&pos, &state.mass, &cfg, None);

    let cpu = max_stable_dt(&state, &params, &cfg, C_CFL);
    let gpu = cfl().max_stable_dt(&pos, &vel, &dens.h, &params, C_CFL);
    assert!(
        gpu.is_finite() && gpu > 0.0,
        "bound must be finite positive: {gpu}"
    );
    let rel = (gpu - cpu).abs() / cpu;
    // Measured ≈ 2.2e-8 (essentially the one f32 divide of the minimizing particle);
    // bound at 1e-5 leaves room for the minimizer to shift across adapters.
    assert!(
        rel < 1.0e-5,
        "GPU max_stable_dt {gpu} vs oracle {cpu} (rel {rel:.3e})"
    );
}

/// The SHARP physical detector for the global-radius invariant, reusing the CPU oracle's
/// cross-support construction: a tight static clump (small h) plus one diffuse particle
/// a distance D away moving straight at the clump at V ≫ c_s (large h), with
/// `2·h_clump < D < 2·h_dist`. The clump's binding (min-h) particle only "sees" the
/// approacher through the neighbor's larger support, so the correct `v_sig = 2c_s + 3V`
/// and the bound tightens by `v_sig/(2c_s)` ≈ 150×. A GPU kernel that gathered only
/// `SUPPORT·h_i` per target would MISS the approacher and return the static floor —
/// blowing this gate wide open. So this ties the GPU to the oracle on exactly the
/// geometry the global-`SUPPORT·h_max` gather exists to handle.
#[test]
fn gpu_cfl_matches_oracle_cross_support() {
    let c_s = 1.0;
    let big_v = 100.0;
    let d = 5.0;

    let clump = concentrated_cloud(0x77, 200, 0.03); // tight ⇒ small h, all static
    let mut pos = clump.clone();
    pos.push(DVec3::new(0.0, 0.0, d)); // lone diffuse approacher
    let mut vel = vec![DVec3::ZERO; clump.len()];
    vel.push(DVec3::new(0.0, 0.0, -big_v)); // heading straight at the clump

    let cfg = DensityConfig::default();
    let params = HydroParams {
        eos: Eos::Isothermal { c_s },
        ..HydroParams::default()
    };
    let state = gas_state(pos.clone(), vel.clone());
    let dens = density_adaptive(&pos, &state.mass, &cfg, None);

    // Assert the geometry actually sits in the cross-support regime we mean to test.
    let h_dist = *dens.h.last().unwrap();
    let h_min = dens.h.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        2.0 * h_min < d,
        "clump 2·h_min = {} must NOT reach D = {d}",
        2.0 * h_min
    );
    assert!(
        2.0 * h_dist > d,
        "approacher 2·h_dist = {} must reach D = {d}",
        2.0 * h_dist
    );

    let cpu = max_stable_dt(&state, &params, &cfg, C_CFL);
    let gpu = cfl().max_stable_dt(&pos, &vel, &dens.h, &params, C_CFL);
    let rel = (gpu - cpu).abs() / cpu;
    // Measured ≈ 8.0e-8 (the GPU sees the cross-support approacher exactly as the oracle
    // does). A per-target-radius bug returns the static floor C_cfl·h_min/(2c_s), which
    // is v_sig/(2c_s) = (2c_s+3V)/(2c_s) ≈ 150× larger (rel ≈ 149) — so the 1e-4 bound
    // cleanly separates the correct answer from the bug by nine orders of magnitude.
    assert!(
        rel < 1.0e-4,
        "GPU max_stable_dt {gpu} vs oracle {cpu} (rel {rel:.3e}). A per-target-h_i gather \
         would return the static floor {} — {}× too large.",
        C_CFL * h_min / (2.0 * c_s),
        (2.0 * c_s + 3.0 * big_v) / (2.0 * c_s),
    );
}

// ---------------------------------------------------------------------------
// determinism (same-device, run-to-run) & the empty/single contract.
// ---------------------------------------------------------------------------

/// Same input ⇒ bit-identical per-target vector on a given device (each thread owns its
/// own output slot; the walk is a gather, no scatter race — same discipline as G1–G3).
#[test]
fn gpu_cfl_deterministic() {
    let cfg = DensityConfig::default();
    let pos = concentrated_cloud(0xD1CE, 1500, 4.0);
    let vel = random_velocities(0xFEED, pos.len(), 1.0);
    let mass = vec![1.0; pos.len()];
    let params = HydroParams::default();
    let dens = density_adaptive(&pos, &mass, &cfg, None);

    let mut g = cfl();
    let a = g.per_target_dt(&pos, &vel, &dens.h, &params, C_CFL);
    let b = g.per_target_dt(&pos, &vel, &dens.h, &params, C_CFL);
    assert_eq!(a, b, "per-target dt must be run-to-run identical");
}

/// Empty input ⇒ empty per-target vector, and — the catastrophic-`0` contract —
/// `max_stable_dt = +∞` (no gas ⇒ no hydro CFL constraint), NOT `0`.
#[test]
fn gpu_cfl_empty() {
    let mut g = cfl();
    let params = HydroParams::default();
    let v = g.per_target_dt(&[], &[], &[], &params, C_CFL);
    assert!(v.is_empty());
    assert_eq!(
        g.max_stable_dt(&[], &[], &[], &params, C_CFL),
        f64::INFINITY,
        "no gas ⇒ +∞ (a 0 would falsely say every dt is too large)"
    );
}

/// A lone particle has no neighbors, so `v_sig` hits the `2c_s` floor and its stable
/// step is the FINITE `C_cfl·h/(2c_s)` — the single-vs-empty distinction (a `+∞` here
/// would be as wrong as a `0` for the empty case).
#[test]
fn gpu_cfl_single_particle_floor() {
    let pos = [DVec3::new(0.5, -0.5, 2.0)];
    let vel = [DVec3::new(0.1, 0.2, -0.3)];
    let h = [0.3];
    let params = HydroParams {
        eos: Eos::Isothermal { c_s: 1.5 },
        ..HydroParams::default()
    };
    let mut g = cfl();
    let dt = g.per_target_dt(&pos, &vel, &h, &params, C_CFL);
    let expect = C_CFL * h[0] / (2.0 * params.sound_speed());
    assert_eq!(dt.len(), 1);
    assert!(
        (dt[0] - expect).abs() / expect < 1.0e-6,
        "lone particle dt {} must be the floor {expect}",
        dt[0]
    );
    let m = g.max_stable_dt(&pos, &vel, &h, &params, C_CFL);
    assert!(
        (m - expect).abs() / expect < 1.0e-6,
        "min {m} must equal the floor {expect}"
    );
}
