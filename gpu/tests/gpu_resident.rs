//! M4i/M4j/M4k gates for [`GpuResidentLeapfrog`]: GPU-resident leapfrog stepping with double-single
//! position accumulation and batched multi-step submits.
//!
//! ## The two load-bearing gates
//! 1. **Faithful / residency.** The *same* stepper type run two ways must agree: `resident`
//!    (upload → K steps → snapshot) vs `roundtrip` (loop: upload → 1 step → snapshot), so a stale
//!    intermediate, a missing cross-step barrier, or an un-re-primed acc shows up as divergence.
//!    M4i asserted this *bit-for-bit* (snapshot/upload was a lossless f32 identity); M4j's
//!    double-single positions retire that premise (the `lo` limb diverges at f64-eps scale through
//!    the tie non-uniqueness of the single-f64 channel), so it now bounds position drift far below
//!    an f32 ulp — see the gate for the mechanism and sizing.
//! 2. **Physics — f32 tolerance.** Vs the host-driven reference `LeapfrogKdk + GpuLbvhFused`,
//!    which holds the *force kernel identical* so the only variable is f32-GPU-KDK vs f64-host-KDK
//!    (the tightest discriminator). Momentum conservation + bounded energy ride on top.
//!
//! **M4k throughput gate** (gate 10) pins that `step_many` coalesces its steps into
//! `⌈K/MAX_BATCH⌉` submits rather than one-per-step; because batching only regroups encoders,
//! gates 1–9 re-validate the trajectory under it for free.
//!
//! GPU-gated: every test needs a wgpu adapter; without one the constructors return `NoAdapter`
//! and the tests fail loudly (they are not silently skipped).

use galaxy_core::{DVec3, ForceSolver, Integrator, LeapfrogKdk, State, StaticBackground};
use galaxy_gpu::{GpuLbvhFused, GpuResidentLeapfrog};

const G: f64 = 1.0;
const EPS: f64 = 0.05;
const THETA: f64 = 0.5;
const DT: f64 = 1e-3;

/// Deterministic pseudo-random cluster with small velocities (same LCG as the other GPU-LBVH
/// tests). Nonzero velocities give momentum/energy diagnostics a real scale to normalize by.
fn cluster(seed: u64, n: usize) -> State {
    let mut state = seed | 1;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut pos = Vec::with_capacity(n);
    let mut vel = Vec::with_capacity(n);
    let mut mass = Vec::with_capacity(n);
    for _ in 0..n {
        pos.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 3.0);
        vel.push(DVec3::new(next() - 0.5, next() - 0.5, next() - 0.5) * 0.1);
        mass.push(0.1 + 0.9 * next());
    }
    State::from_phase_space(pos, vel, mass)
}

/// f32-narrow a coordinate the way the GPU does (the identity a faithful snapshot preserves).
fn narrow(v: DVec3) -> DVec3 {
    DVec3::new(v.x as f32 as f64, v.y as f32 as f64, v.z as f32 as f64)
}

/// Max per-component absolute difference between two vector fields.
fn max_abs_diff(a: &[DVec3], b: &[DVec3]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(u, v)| {
            (u.x - v.x)
                .abs()
                .max((u.y - v.y).abs())
                .max((u.z - v.z).abs())
        })
        .fold(0.0_f64, f64::max)
}

/// Total (linear) momentum Σ mᵢ vᵢ.
fn momentum(s: &State) -> DVec3 {
    s.vel
        .iter()
        .zip(&s.mass)
        .fold(DVec3::ZERO, |acc, (v, &m)| acc + *v * m)
}

/// Total energy = Σ ½ mᵢ vᵢ² + softened potential (via the fused solver's f64 potential).
fn total_energy(s: &State, solver: &GpuLbvhFused) -> f64 {
    let ke: f64 = s
        .vel
        .iter()
        .zip(&s.mass)
        .map(|(v, &m)| 0.5 * m * v.length_squared())
        .sum();
    ke + solver.potential_energy(s)
}

fn resident() -> GpuResidentLeapfrog {
    GpuResidentLeapfrog::new(G, EPS, THETA).expect("wgpu adapter required for GPU resident tests")
}

// ---------------------------------------------------------------------------------------------
// Gate 1 — faithful / residency: matches a per-step host round-trip.
// ---------------------------------------------------------------------------------------------

/// Keeping state GPU-resident across K steps must reproduce the trajectory of round-tripping it
/// through the host between every step — the residency correctness gate (a stale buffer, missing
/// barrier, or un-re-primed acc perturbs `hi` by ≥1 f32-ulp ≈ 1e-7, freezes/garbles state, or
/// breaks determinism, all far above the tolerance below).
///
/// **Why not bit-for-bit any more (M4j).** M4i could assert *exact* equality because snapshot↔
/// upload was a lossless f32 identity. Double-single positions retire that premise: at a tie
/// (`|lo| = ½ulp(hi)`) the single-f64 snapshot channel can't preserve *which* `(hi, lo)` split
/// produced the value, so the resident path (carrying `lo` through K two-sums) and the round-trip
/// path (recombine→resplit each step) take different arithmetic routes in the **`lo` limb** and
/// diverge at f64-epsilon scale. Measured here: dp ≈ 1.8e-15, dv = 0. The tolerance is sized from
/// the mechanism — worst case ≈ `K·ulp(lo)` ≈ 20·7e-15 ≈ 1.4e-13 at this coordinate scale — with
/// ~7× headroom (project precedent) and still ~5 orders below an f32 ulp, so it discriminates
/// every real residency bug. Velocity round-trips exactly (`hi` never diverges when forces agree);
/// it is *theoretically* subject to the same tie effect, so a future N/K/seed change that nudges
/// it off exact equality is expected, not a bug.
#[test]
fn resident_matches_roundtrip_bit_for_bit() {
    const N: usize = 512;
    const K: u64 = 20;
    const TOL: f64 = 1e-12;
    let s0 = cluster(7, N);

    // Resident: one upload, K steps on device, one snapshot.
    let mut res = resident();
    res.upload(&s0);
    res.step_many(DT, K);
    let out_res = res.snapshot();

    // Round-trip: the SAME stepper, but re-upload + re-prime + snapshot every single step.
    let mut rt = resident();
    let mut s = s0.clone();
    for _ in 0..K {
        rt.upload(&s);
        rt.step(DT);
        s = rt.snapshot();
    }

    let dp = max_abs_diff(&out_res.pos, &s.pos);
    assert!(
        dp < TOL,
        "resident vs round-trip positions diverged by {dp:e} (> {TOL:e}) — residency bug"
    );
    assert_eq!(
        out_res.vel, s.vel,
        "resident vs round-trip velocities must be bit-for-bit"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 2 — physics: tracks the host-driven f64 leapfrog to an f32 tolerance.
// ---------------------------------------------------------------------------------------------

/// The resident stepper must track the host-driven reference `LeapfrogKdk + GpuLbvhFused` (same
/// f32 force kernel; only the KDK precision differs) within an f32-appropriate tolerance over a
/// short run. Tolerance is measurement-calibrated (project precedent), sized for f32 position
/// accumulation over K steps.
#[test]
fn resident_tracks_host_driven_leapfrog_within_f32_tol() {
    const N: usize = 512;
    const K: u64 = 30;
    // Measured f32-GPU-KDK vs f64-host-KDK over 30 steps: dp≈5e-7, dv≈1.5e-5 (velocity is the
    // larger, as the kick accumulates the f32 force each half-step). 1e-4 keeps ~7× headroom yet
    // still discriminates — a real KDK divergence would blow past it.
    const TOL: f64 = 1e-4;
    let s0 = cluster(3, N);

    // Reference: authoritative f64 KDK driving the fused f32 force solver.
    let mut fused = GpuLbvhFused::new(G, EPS, THETA).expect("wgpu adapter required");
    let mut leap = LeapfrogKdk::new();
    let bg = StaticBackground;
    let mut s_ref = s0.clone();
    for _ in 0..K {
        leap.step(&mut s_ref, &mut fused, &bg, DT);
    }

    // Resident.
    let mut res = resident();
    res.upload(&s0);
    res.step_many(DT, K);
    let s_res = res.snapshot();

    let dp = max_abs_diff(&s_ref.pos, &s_res.pos);
    let dv = max_abs_diff(&s_ref.vel, &s_res.vel);
    assert!(
        dp < TOL,
        "position drift vs host leapfrog {dp:e} exceeds {TOL:e}"
    );
    assert!(
        dv < TOL,
        "velocity drift vs host leapfrog {dv:e} exceeds {TOL:e}"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 3 — invariant: total momentum is conserved (gravity is internal).
// ---------------------------------------------------------------------------------------------

/// Internal gravitational forces conserve total linear momentum. Tested at **θ→0**: only there
/// are the tree forces exact direct sums, so Fᵢⱼ = −Fⱼᵢ to the f32 floor and Σ mᵢ vᵢ stays put.
/// (At finite θ the monopole acceptance breaks Newton's third law at O(θ²) — the same reason the
/// M4h momentum gate uses θ→0 — so a finite-θ momentum test would measure BH error, not a bug.)
#[test]
fn resident_conserves_total_momentum() {
    const N: usize = 512;
    const K: u64 = 40;
    let s0 = cluster(11, N);
    let p0 = momentum(&s0);

    let mut res = GpuResidentLeapfrog::new(G, EPS, 1e-6).expect("wgpu adapter required");
    res.upload(&s0);
    res.step_many(DT, K);
    let s = res.snapshot();
    let p1 = momentum(&s);

    let scale: f64 = s
        .vel
        .iter()
        .zip(&s.mass)
        .map(|(v, &m)| v.length() * m)
        .sum::<f64>()
        .max(1e-300);
    let rel = (p1 - p0).length() / scale;
    assert!(
        rel < 1e-4,
        "total momentum drifted by {rel:e} (should be ~0)"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 4 — invariant: energy stays bounded (leapfrog does not secularly heat).
// ---------------------------------------------------------------------------------------------

/// Leapfrog energy oscillates within a bound rather than drifting. The bound here is looser than
/// the f64 leapfrog's because f32 position accumulation adds noise — but energy must not run away.
#[test]
fn resident_energy_stays_bounded() {
    const N: usize = 256;
    const K: u64 = 200;
    let s0 = cluster(5, N);
    let probe = GpuLbvhFused::new(G, EPS, THETA).expect("wgpu adapter required");
    let e0 = total_energy(&s0, &probe);

    let mut res = resident();
    res.upload(&s0);
    res.step_many(DT, K);
    let s = res.snapshot();
    let e1 = total_energy(&s, &probe);

    let rel = ((e1 - e0) / e0.abs().max(1e-300)).abs();
    assert!(
        rel < 5e-2,
        "energy drifted by {rel:e} over {K} steps — not bounded"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 5 — upload/snapshot is a clean round-trip identity at zero steps.
// ---------------------------------------------------------------------------------------------

/// With no steps taken, snapshot must return the uploaded state with no hidden mutation. Because
/// positions are now carried as a **double-single** (hi+lo f32 pair) — upload splits the f64 input
/// into `hi + lo`, snapshot sums them back — the position round-trip recovers the input to ~f64
/// precision (a *tighter* identity than the old f32 narrowing, not a looser one). Velocity is
/// still a plain-f32 narrowing (DS is position-only), so it round-trips exactly to `narrow(v)`.
#[test]
fn zero_steps_round_trips_input() {
    let s0 = cluster(2, 300);
    let mut res = resident();
    res.upload(&s0);
    let out = res.snapshot();

    for (i, (&p, &v)) in s0.pos.iter().zip(&s0.vel).enumerate() {
        let dp = (out.pos[i] - p).length();
        assert!(
            dp < 1e-9,
            "pos[{i}] round-trip error {dp:e} exceeds f64-DS tol"
        );
        assert_eq!(out.vel[i], narrow(v), "vel[{i}] not a clean f32 narrowing");
    }
    assert_eq!(out.mass.len(), s0.mass.len());
    assert_eq!(res.time(), 0.0, "no steps => time unchanged");
}

// ---------------------------------------------------------------------------------------------
// Gate 6 — degenerate sizes: empty and single-particle are trivial, not panics.
// ---------------------------------------------------------------------------------------------

/// An empty system steps to nothing; a lone particle feels no force and drifts at constant
/// velocity (x = x₀ + v₀·dt·K), so it exercises drift-without-force.
#[test]
fn resident_handles_empty_and_single() {
    let empty = State::from_phase_space(vec![], vec![], vec![]);
    let mut r0 = resident();
    r0.upload(&empty);
    r0.step_many(DT, 5);
    let out = r0.snapshot();
    assert!(out.is_empty());

    let x0 = DVec3::new(1.0, -2.0, 0.5);
    let v0 = DVec3::new(0.05, 0.0, -0.02);
    let single = State::from_phase_space(vec![x0], vec![v0], vec![1.0]);
    const K: u64 = 10;
    let mut r1 = resident();
    r1.upload(&single);
    r1.step_many(DT, K);
    let s = r1.snapshot();
    let expect = narrow(x0) + narrow(v0) * (DT * K as f64);
    assert!(
        max_abs_diff(&s.pos, &[expect]) < 1e-5,
        "free particle should drift ballistically, got {:?}",
        s.pos[0]
    );
    assert!(
        max_abs_diff(&s.vel, &[narrow(v0)]) < 1e-6,
        "free particle velocity must not change"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 7 — same-device determinism: two independent runs agree bit-for-bit.
// ---------------------------------------------------------------------------------------------

/// Two independent resident sims on the same adapter, same input and steps, must produce
/// bit-identical snapshots (no order-dependent reduction leaks in the kick/drift/reset kernels).
#[test]
fn resident_stepping_is_deterministic() {
    const N: usize = 400;
    const K: u64 = 25;
    let s0 = cluster(9, N);

    let mut a = resident();
    a.upload(&s0);
    a.step_many(DT, K);
    let sa = a.snapshot();

    let mut b = resident();
    b.upload(&s0);
    b.step_many(DT, K);
    let sb = b.snapshot();

    assert_eq!(
        sa.pos, sb.pos,
        "resident stepping must be bit-deterministic (pos)"
    );
    assert_eq!(
        sa.vel, sb.vel,
        "resident stepping must be bit-deterministic (vel)"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 8 — time advances by exactly K·dt.
// ---------------------------------------------------------------------------------------------

/// Bookkeeping: time is host-tracked and advances by exactly `steps · dt`.
#[test]
fn resident_time_advances() {
    let s0 = cluster(1, 64);
    let mut res = resident();
    res.upload(&s0);
    res.step_many(DT, 17);
    assert!(
        (res.time() - 17.0 * DT).abs() < 1e-12,
        "time should be 17·dt, got {}",
        res.time()
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 9 — double-single position accumulation tracks the exact drift (M4j).
// ---------------------------------------------------------------------------------------------

/// The precision payoff: carrying `pos += vel·dt` as a **double-single** (hi+lo f32 pair, with a
/// compensated two-sum every step) tracks the exact real drift far tighter than a plain-f32
/// running sum — which loses the small per-step increment into a growing coordinate's ulp (DESIGN
/// M4i's "updates below ~1e-6 of a coordinate's magnitude are lost"). Isolated with a single
/// force-free particle: accel ≡ 0 (the n==1 branch clears it), velocity is constant, so this
/// measures the position accumulator *alone* — no tree, no f32-force noise, fully deterministic.
///
/// Large f32-exact coordinates (`x₀ = ±1024/512`) put the per-step increment near a single ulp,
/// so the f32 loss shows in only K=10⁴ steps (keeping the test's per-step submits cheap) instead
/// of the ~10⁶ a cluster-scale coordinate would need. Reference is the exact f64 sum
/// `x₀ + K·fl32(v·dt)` (the device increment, matched bit-for-bit). **Measured:** DS lands at
/// ~1e-8; a plain-f32 running sum drifts to ~3.5e-1 here — TOL 1e-5 sits deep in the gap,
/// discriminating both ways (green passes with ~1000× headroom; the old f32 path blows past it).
#[test]
fn resident_double_single_position_tracks_exact_drift() {
    // f32-exact coordinates ⇒ recovered x₀ == x₀ and the reference sum is unambiguous.
    let x0 = DVec3::new(1024.0, -1024.0, 512.0);
    let v0 = DVec3::new(1.0, -1.0, 0.5);
    const K: u64 = 10_000;
    const TOL: f64 = 1e-5;

    let single = State::from_phase_space(vec![x0], vec![v0], vec![1.0]);
    let mut res = resident();
    res.upload(&single);
    res.step_many(DT, K);
    let s = res.snapshot();

    // Per-step device increment: fl32(v·dt) in f32, exactly the shader's `vel·dt`.
    let d = DVec3::new(
        ((v0.x as f32) * (DT as f32)) as f64,
        ((v0.y as f32) * (DT as f32)) as f64,
        ((v0.z as f32) * (DT as f32)) as f64,
    );
    let expect = x0 + d * (K as f64);
    let err = (s.pos[0] - expect).length();
    assert!(
        err < TOL,
        "double-single drift error {err:e} exceeds {TOL:e} (a plain-f32 running sum would)"
    );
}

// ---------------------------------------------------------------------------------------------
// Gate 10 — throughput: step_many coalesces steps into ⌈K/MAX_BATCH⌉ submits (M4k).
// ---------------------------------------------------------------------------------------------

/// `step_many` must batch its steps into **one encoder/submit per `MAX_BATCH`-step chunk** —
/// dropping per-submit overhead (the named M4i throughput follow-up: M4i removed the per-step
/// *latency*, this removes the per-step *submit*) — rather than one submit per step. The submit
/// count is the *only* thing batching changes that is observable: the trajectory is bit-identical
/// (batching regroups encoders, it does not reorder the kick·drift·force·kick arithmetic), so
/// every other gate re-validates correctness under batching for free — this gate pins the count.
///
/// Read as a **before/after delta** so it is robust to the prime submit (`upload`) and the
/// snapshot submit surrounding the measured `step_many`. The cap is bounded to keep any single
/// submit under the OS GPU watchdog (see [`GpuResidentLeapfrog::MAX_BATCH`]).
#[test]
fn step_many_coalesces_into_bounded_submits() {
    const N: usize = 256;
    let s0 = cluster(13, N);
    let mut res = resident();
    res.upload(&s0);

    // K == MAX_BATCH: the whole run is a single submit.
    let before = res.submits();
    res.step_many(DT, GpuResidentLeapfrog::MAX_BATCH);
    let one_chunk = res.submits() - before;
    assert_eq!(
        one_chunk, 1,
        "MAX_BATCH steps must coalesce into ONE submit, got {one_chunk}"
    );

    // K > MAX_BATCH: chunked into ⌈K/MAX_BATCH⌉ submits (2·MAX_BATCH + 1 ⇒ 3 chunks).
    let k = 2 * GpuResidentLeapfrog::MAX_BATCH + 1;
    let before = res.submits();
    res.step_many(DT, k);
    let got = res.submits() - before;
    let want = k.div_ceil(GpuResidentLeapfrog::MAX_BATCH);
    assert_eq!(
        got,
        want,
        "expected ⌈{k}/{}⌉ = {want} submits, got {got}",
        GpuResidentLeapfrog::MAX_BATCH
    );
}
