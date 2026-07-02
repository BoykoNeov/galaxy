//! M4i gates for [`GpuResidentLeapfrog`]: GPU-resident leapfrog stepping.
//!
//! ## The two load-bearing gates
//! 1. **Faithful / residency — bit-for-bit.** The *same* stepper type run two ways must agree
//!    exactly: `resident` (upload → K steps → snapshot) vs `roundtrip` (loop: upload → 1 step →
//!    snapshot). Because f32→f64→f32 round-tripping through snapshot/upload is a lossless
//!    identity, the re-uploaded path feeds bit-identical f32 each step, so any divergence *is* a
//!    residency bug — a stale intermediate, a missing cross-step barrier, or an un-re-primed acc.
//!    This is M4h's faithful-refactor claim lifted to the step loop; it needs no new reference.
//! 2. **Physics — f32 tolerance.** Vs the host-driven reference `LeapfrogKdk + GpuLbvhFused`,
//!    which holds the *force kernel identical* so the only variable is f32-GPU-KDK vs f64-host-KDK
//!    (the tightest discriminator). Momentum conservation + bounded energy ride on top.
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
// Gate 1 — faithful / residency: bit-for-bit.
// ---------------------------------------------------------------------------------------------

/// Keeping state GPU-resident across K steps must produce **exactly** the same trajectory as
/// round-tripping it through the host between every step. Exact f64 equality: the round-trip is a
/// lossless f32 identity, so anything but equality is a residency bug (stale buffer / missing
/// barrier / un-re-primed acc).
#[test]
fn resident_matches_roundtrip_bit_for_bit() {
    const N: usize = 512;
    const K: u64 = 20;
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

    assert_eq!(
        out_res.pos, s.pos,
        "resident vs round-trip positions must be bit-for-bit"
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
    // Positions are O(1); f32 KDK vs f64 KDK over 30 steps stays well under this.
    const TOL: f64 = 2e-3;
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
// Gate 5 — upload/snapshot is a clean f32-narrowing identity at zero steps.
// ---------------------------------------------------------------------------------------------

/// With no steps taken, snapshot must return exactly the uploaded state narrowed to f32 — proving
/// upload and snapshot are inverse transfers with no hidden mutation.
#[test]
fn zero_steps_round_trips_f32_narrowed_input() {
    let s0 = cluster(2, 300);
    let mut res = resident();
    res.upload(&s0);
    let out = res.snapshot();

    for (i, (&p, &v)) in s0.pos.iter().zip(&s0.vel).enumerate() {
        assert_eq!(out.pos[i], narrow(p), "pos[{i}] not a clean f32 narrowing");
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
