# GPU-SPH — isothermal gas hydro on the GPU-resident stepper

Scoping doc, written 2026-07-04. Triggered by the full-res `gasrich` measurement
(`settling-cinder-vigil.md` → `M:\…\gasrich_fullres_measure\FINDINGS.md`): the
full-res sim is > 30 min under any completing dt (Finding B), which **triggers the
standing GPU-SPH gate**. The user's scale-forward stance (bigger models/data)
reinforces GPU-SPH as the primary investment over F1 (which is now moot — it
hand-optimizes the exact CPU loop GPU-SPH replaces).

**This is a scoping doc. Do NOT start implementing from it.** GPU-SPH is plausibly
the single biggest lift in the project — a full GPU-stepper extension spanning
neighbor search, an adaptive-h density root-find, the hydro force, and a CFL
reduction, each oracle-gated. It is broken into sub-milestones below.

---

## ⚠ What GPU-SPH buys — and what it does NOT

- **Buys:** the *speed/scale* win (Finding B) — moves the per-step O(N²)-ish hydro
  gather and the density/CFL work onto the GPU, where the gravity force already
  lives (`GpuResidentLeapfrog`). Also lays the **CFL-reduction substrate** that
  adaptive dt consumes.
- **Does NOT buy:** producibility of the full-res showpiece. **Finding A stands** —
  the full-res sim CFL-*aborts* at dt=0.005 because CFL bounds step *size*, not
  step *speed*. GPU-SPH makes the (unavoidably large) step count *tolerable*; it
  does not make the showpiece *runnable*. That needs **adaptive dt** (its own
  follow-up, groundwork laid here — see "Relationship to adaptive dt").

Do not let the plan (or the next session) imply GPU-SPH unblocks the showpiece.

---

## Architecture grounding (two lineages, verified 2026-07-04)

There are **two integrator lineages**, and GPU-SPH extends the second, not the first:

1. **CPU/host lineage** — `Integrator` (`LeapfrogKdk`) + `ForceSolver`
   (`core/src/traits.rs`). Authoritative positions in **f64**; the SPH solver is
   `GravitySph<G>` (`solvers/src/sph/gravity_sph.rs`): wrapped gravity `G`
   (e.g. `BarnesHut`) over ALL particles + hydro over the gas subset, recomputing
   ρ/h internally once per KDK step at post-drift positions. Bit-exact rayon≡serial,
   equal-mass pairwise antisymmetry (momentum to roundoff). **This is the GPU oracle.**

2. **GPU-resident lineage** — `GpuResidentLeapfrog` (`gpu/src/gpu_resident.rs`,
   M4i/M4k). NOT a `ForceSolver`: it owns its step loop. `pos`/`vel`/`mass`/`acc`
   live in GPU storage buffers *across* steps; kick/drift arithmetic runs in WGSL
   (`KICK_SHADER`/`DRIFT_SHADER`); the force pipeline is the fused Karras-LBVH walk
   (`FusedCore`, M4h). Forces are **f32**, positions carried **double-single**
   (`hi`+`lo` f32 pair, ~46-bit) so the small per-step drift isn't lost into the
   coordinate's ulp. `step` = one submit; `step_many` coalesces ≤ `MAX_BATCH` (64)
   steps into one encoder/submit — **at fixed dt across the batch** (the residency
   throughput win). `upload → step* → snapshot` lifecycle.

**So GPU-SPH bolts a hydro stage onto lineage 2**, over the gas subset, inside the
resident loop: **neighbor search → adaptive-h density (ρ,h) → hydro force → CFL
reduction**, added to `acc[gas]` before the kick. `xtask::simulate::simulate_snapshots`
gains a GPU branch alongside the existing `GravitySph` CPU branch.

---

## Key decisions (with rationale)

### D1 — f32 forces, TOLERANCE-gated against the CPU oracle (NOT bit-exact)
GPU forces are f32 (mirroring the gravity-GPU discipline: f32 force / f64 energy).
So GPU SPH cannot be bit-exact vs the f64 CPU path. Gate to a stated f32 tolerance,
exactly as the existing M4 gravity-GPU gates do — see D5 for what is (and is not) gateable.

### D2 — gather-per-target on the GPU (one thread per gas particle, no scatter)
Each gas-particle thread sums its *own* neighbor contributions into its own `acc`
slot. No scatter → no cross-thread accumulation race → deterministic per target.
Consequence: **momentum is conserved only to f32 roundoff**, not exactly (the CPU
path's *exact* antisymmetry comes from f64 + commutative coeff + negated grad, which
f32 breaks). Acceptable and consistent with f32-force philosophy; the gate is a
**bounded momentum-drift invariant** (D5), not exact conservation. NB: this means
the CPU F1 "scatter-by-plane" rewrite is **not** ported — the GPU eats the O(N²)
gather cost via parallelism, and gather-per-target is the determinism-safe shape.

### D3 — compensated sums need the DS XOR-barrier trap
Any error-free-transform / two-sum on the GPU (the density root-find or a
compensated force sum, if used) hits the f32-reassociation fold-to-zero trap.
Reuse the `DRIFT_SHADER` fix: XOR the bits with a runtime-uniform pinned to 0 to
block the fold; launder ALL intermediates. See [[gpu-double-single-reassociation]].
Verified on the Vulkan adapter only — gates prove it on the CI adapter, not universally.

### D4 — neighbor structure: GPU uniform hash grid (a real fork; reason stated)
**Decision: a GPU uniform hash grid** (cell-linked-list / counting-sort bucketing),
mirroring the CPU `sph::grid::HashGrid`.
- *For:* 1:1 with the oracle → the simplest possible gating; it is the SPH-standard
  GPU neighbor structure; adaptive-h gives a wide per-particle query radius that an
  exact fixed-radius grid handles naturally.
- *Against (the fork):* it is **net-new GPU code** with its own determinism gates,
  whereas the **Karras LBVH is already built and gated** (`gpu/src/gpu_lbvh*.rs`).
- *Why not reuse the LBVH:* it is a Barnes-Hut *approximation* walk (θ-criterion
  aggregate), not exact fixed-radius neighbor *enumeration*; forcing it into range
  queries fights its design. The gate-simplicity of a grid that matches the oracle
  cell-for-cell outweighs the "reuse proven code" pull. **Revisit if** the grid's
  determinism gates prove costlier than an LBVH range-query adapter.

### D5 — GATE DESIGN: no full-merger trajectory match (chaotic system)
A self-gravitating merger is chaotic (positive Lyapunov); an f32-vs-f64 force
difference e-folds over a fraction of a dynamical time, so GPU and CPU trajectories
diverge *macroscopically over thousands of steps by construction* — physics, not a
bug. A per-particle trajectory tolerance gate **fails on a correct implementation.**
Gate instead, mirroring the gravity-GPU discipline:
- **Per-/few-step tolerance gates, same input state:** GPU vs CPU `density`, `h`,
  `hydro_accelerations`, and `max_stable_dt` — each a deterministic function of
  positions → clean f32-tolerance gates. **This is where bugs are actually caught.**
- **Long-run invariants (not trajectory match):** bounded linear+angular momentum
  drift (isothermal ⇒ energy is NOT conserved, momentum is), no NaN/blowup, a sane
  (finite, positive, correctly-ranged) density field over a full run.
- **Showpiece + coarse statistics:** radial / mass / density profiles CPU-vs-GPU at
  a few snapshots — compare *distributions*, not particle IDs.

### D6 — batching × adaptive dt: BLOCK-ADAPTIVE dt (resolved, not open)
`step_many` batches 64 steps at fixed dt; per-step dt feedback would kill the
residency win. But the CFL bound evolves on the hydro/dynamical timescale (many
steps), so per-step feedback is unnecessary. **Recompute the CFL reduction once per
batch and set the next batch's dt** (× safety margin). Two cheap reconciliations:
(a) an on-GPU dispatch that writes `step_params` from the reduction — fully resident,
zero sync; or (b) a one-scalar readback every 64 steps — negligible vs ~10 LBVH
dispatches × 64. Batching survives. **Pericenter nuance:** the bound can dip below
the batch dt at closest approach → handle with a conservative margin and/or shorter
batches near pericenter and/or an on-GPU per-step CFL flag that aborts the batch. In
GPU-SPH scope: compute the reduction on-GPU and expose it; the "vary dt across
batches" policy is the adaptive-dt follow-up (shared substrate, below).

---

## Milestone breakdown (each TDD, oracle-gated per D5)

Sub-milestones, roughly in dependency order. Each lands red→green with its own gate.

- **G1 — GPU uniform hash grid** (D4): buckets gas positions at a query radius;
  gate = same neighbor sets as `sph::grid::HashGrid` on synthetic clouds
  (set equality, order-independent). Net-new GPU code; carries the determinism gates.
- **G2 — GPU adaptive-h density**: per-gas-particle bisection on the kernel-weighted
  count → (ρ, h). Gate = f32-tolerance vs `density_adaptive` on a
  centrally-concentrated cloud (wide h range). DS XOR-barrier if a compensated sum
  is used (D3).
- **G3 — GPU hydro force**: symmetric P/ρ² + Monaghan viscosity, coupling-range-gated
  (`r < 2·max(h_i,h_j)`), gather-per-target (D2). Gate = f32-tolerance vs
  `hydro_accelerations` on the same cloud + a bounded momentum-drift check.
- **G4 — GPU CFL reduction**: min over gas of `C·h/v_sig` (projected signal
  velocity). Gate = f32-tolerance vs `max_stable_dt`. Exposes the per-batch bound
  (D6) — the adaptive-dt substrate.
- **G5 — wire into `GpuResidentLeapfrog`**: add the hydro stage to the resident step
  (gravity over all + hydro on gas subset before the kick); block-adaptive dt plumbing
  (D6, compute+expose only). Gate = the long-run invariants + coarse-statistics
  profiles (D5), NOT a trajectory match.
- **G6 — `simulate_snapshots` GPU branch + re-run the QUICK gasrich merger**:
  GPU path selectable alongside the CPU `GravitySph` branch; gate = QUICK gasrich
  GPU-vs-CPU coarse statistics agree, wall-clock recorded. (Full-res still blocked on
  adaptive dt — do not expect a producible full-res showpiece here.)

Isothermal shock tube (`--release --ignored`) stays a gate throughout: the GPU hydro
path must match the analytic Riemann solution to the same tolerance the CPU path does.

---

## Risks & dependencies

- **Biggest lift in the project.** Four new GPU compute stages, each oracle-gated,
  plus resident-loop integration. Size expectations accordingly (multiple sessions).
- **Driver-dependence.** GPU float behavior is adapter-specific; gates prove
  correctness on the CI/Vulkan adapter, not universally (as with the DS work).
- **The oracle must stay honest.** The CPU SPH path (bit-exact, momentum-antisym) is
  the reference for every gate — do not "optimize" it in ways that change its numbers
  during this work (F1 is explicitly deferred/moot; leave the CPU force loop alone).
- **Neighbor-structure fork (D4)** is the one decision most likely to be revisited;
  keep G1 isolated so an LBVH-range-query swap stays possible.

---

## Relationship to adaptive dt (and the showpiece)

GPU-SPH and adaptive dt are **sequential-with-shared-substrate**, not independent:
GPU-SPH must compute the CFL reduction on-GPU anyway (G4/D6), and that same reduction
is exactly the input adaptive dt consumes. So "GPU-SPH first, then adaptive dt" (the
user's ordering) is coherent — GPU-SPH lays adaptive dt's groundwork. **Adaptive dt
is what finally makes the full-res showpiece producible** (Finding A); GPU-SPH makes
that producible run *fast enough to be practical* (Finding B). Both are wanted; this
doc scopes the first.

Related: `settling-cinder-vigil.md` (the measurement that triggered this, F1/F2/F4
disposition), `long-burning-beacon.md` (long-horizon ordering incl. the GPU-SPH gate),
`deep-orbiting-sunbeam.md` (M7 SPH per-series plan), DESIGN.md (SPH + CFL rationale),
[[gpu-double-single-reassociation]], [[m7b-sph-forces-decisions]].
