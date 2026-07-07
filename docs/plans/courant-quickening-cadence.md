# Adaptive dt — global block-adaptive timestepping on the SPH path

Scoping doc, written 2026-07-07. Triggered by the full-res `gasrich` measurement
(`settling-cinder-vigil.md` → `M:\…\gasrich_fullres_measure\FINDINGS.md`,
Finding A): the full-res showpiece is **unproducible at any verified fixed dt** —
the merger-wide-minimum CFL bound is unknown a priori and declines below 0.002
unpredictably, so a human cannot pick a safe fixed `dt`. GPU-SPH (now landed,
G1–G6) makes the step *count* tolerable but does **not** make the showpiece
runnable — CFL bounds step *size*, not step *speed*. Adaptive dt is the missing
lever. The GPU-SPH work deliberately laid its substrate: G4/G5c compute the CFL
reduction on-GPU (the no-readback scalar `min_stable_dt`), which is exactly
adaptive dt's per-block input.

**This is a scoping doc.** The design below is advisor-vetted (2026-07-07). The
real work is the **gate redesign** (the invariant gates collide with variable-dt
leapfrog head-on), not the dt-picking loop, which is small.

---

## ⚠ What adaptive dt buys — and what it does NOT

- **Buys (primary): producibility.** The run tracks the CFL bound down
  automatically and never aborts — no human guesses a merger-wide-safe fixed dt.
  This is Finding A's fix and the whole justification. It stands **independent of
  any speedup.**
- **Buys (secondary): a modest speedup** from the bound's dynamic range — the
  diffuse early/approach phase rides a larger dt than the pericenter knot. From
  the measurements this range is ~2–3× (QUICK t=0 bound ≈0.0034 vs dense-knot
  <0.002; full-res pre-pericenter 0.00475 vs knots <0.002). Modest — it sets the
  **block-growth-cap aggressiveness**, it is not the reason to build this.
- **Does NOT buy:** any change to the gas-free (collisionless) path, exact energy
  conservation, or time-reversibility (see D2 — variable dt forfeits both by
  construction).

Do not let a future session frame adaptive dt as a speed feature; it is a
correctness/usability feature that happens to also trim easy-step waste.

---

## Scope (v1)

- **Global** adaptive dt: one `dt` shared by all particles, recomputed at **block
  boundaries** (D1). NOT individual/per-particle timesteps — that is a much larger
  lift (per-particle time bins, active-particle lists, half-drift synchronization)
  and is explicitly deferred.
- **SPH (gas) path only**, on **both** lineages (CPU `GravitySph` via `sim::run`,
  GPU-resident via `simulate_gas_gpu`). The gas-free Barnes-Hut path is **literally
  untouched** — same fixed-dt `sim::run`, preserving its byte-identity gate. A
  gravitational (acceleration-based) timestep criterion for collisionless runs is a
  separate, later criterion (its own `max_stable_dt` contribution); v1's bound is
  hydro-CFL only.
- **Time-based snapshot cadence** replaces step-count cadence on the adaptive path
  (D3): variable dt means "every N steps" no longer maps to even movie spacing.

---

## Key decisions (with rationale)

### D1 — BLOCK-adaptive, and that is FORCED (not a style choice)
`GpuResidentLeapfrog::step_many` batches ≤`MAX_BATCH`(64) steps into one submit at
a **single** `dt` uniform — that batching IS the residency throughput win that
justifies GPU-SPH's existence. Per-step dt recomputation = one submit per step =
throws that win away. So the adaptation unit is the **block**: hold `dt` fixed
across a block of ≤`B` steps, recompute at the block boundary from
`min_stable_dt` (the no-readback scalar min G5c built — cheap to pull per boundary,
never the N-vector). The **CPU path adapts on the same block cadence** even though
it could adapt per step — NOT for a cross-path trajectory gate (D4 forbids
comparing trajectories across paths), but because it is the *same algorithm* and so
shares D2b's single contraction-staleness safety analysis. Block size `B` is a
tunable bounded by D2b (contraction staleness). Note: do NOT reprime the integrator
at a block boundary — the cached position-only acceleration carries across a dt
change correctly (velocity-Verlet), so re-priming just wastes a force eval.

### D2 — variable-dt leapfrog is NOT symplectic and NOT time-reversible
The dt-from-state dependence means forward and backward integration pick different
dt sequences → **time-reversibility is gone by construction**, and energy **drifts
secularly** instead of oscillating within a bound. This collides head-on with the
project's invariant gates:

- The existing leapfrog-reversibility test and "energy oscillates, does not drift"
  gate **will fail on the adaptive path** — correctly. They must NOT be run against
  it, and a future session must NOT "fix" them by loosening.
- **Keep the fixed-dt path's reversibility + energy-oscillation gates intact and
  untouched** (they gate the fixed-dt integrator, which is unchanged).
- **The adaptive path gets DIFFERENT gates** (see Gates below): (i) PRIMARY —
  **full-DURATION convergence to a fine-dt reference trajectory** on a non-chaotic
  testbed as courant → 0 (the coarse adaptive run must monotonically approach the
  fine one). Full-duration, not a short prefix: that is what catches the secular
  variable-dt error the (dropped) energy-drift gate was meant to catch — a fixed-dt
  symplectic run can't drift, a variable-dt one can, and convergence subsumes the
  energy curve as a trajectory functional. (ii) SECONDARY — **contraction staleness
  (D2b)**, the real second gate.

**NB — there is NO energy gate on the adaptive path (D4 constraint).** An isothermal
EOS is an implicit heat bath (DESIGN.md 1582–1583), so total energy legitimately
changes even with a perfect fixed-dt integrator — there is no flat baseline to
measure spurious *drift* against, and gas-free can't substitute (it returns `+∞`,
the adaptive loop needs a finite bound). Full-duration convergence replaces it.
Document this in the adaptive test module's header so the intent survives (a future
session must not add an energy-conservation gate here).

#### D2b — mid-block CFL staleness under CONTRACTION (the dual of frozen-h_max)
As gas contracts toward pericenter, `h↓` and `v_sig↑`, so the CFL bound **drops
within a block**. A dt picked at block start can exceed the tightened bound before
the block ends → an instability the boundary check never sees. Note the symmetry
with the already-documented frozen-h_max landmine: **frozen-h_max is safe under
contraction / breaks under expansion; held-dt is safe under expansion / dangerous
under contraction.** (The re-upload-per-interval cadence in `simulate_gas_gpu`
already handles the expansion side for the gather radius; this is the other side.)
Mitigations, all three:
1. **Safety factor** — pick `dt = SAFETY · bound` with `SAFETY < 1` (e.g. 0.5), so
   a within-block tightening has headroom before it crosses the actual bound.
2. **Per-block dt-growth cap** — `dt_new ≤ GROWTH · dt_old` (e.g. ≤1.25×) so dt
   never jumps up into a regime the just-passed block already disproved.
3. **Block-size bound** `B` — cap steps/block so a block's worth of contraction
   cannot cross the bound at the chosen `SAFETY`.
**Verification obligation:** show a block's contraction cannot cross the bound at
the chosen `B` and `SAFETY` — a stepped/contracting gate (analogous to the G5b
stepped-contraction staleness gate) that asserts the realized dt stayed ≤ the
end-of-block bound.

### D3 — snapshot cadence becomes TIME-based on the adaptive path
Fixed `snapshot_every` (steps) no longer yields even movie spacing under variable
dt. The adaptive loop is driven by `t_end` + `output_dt` (output time interval).
The loop sub-steps with adaptive dt and **clamps the final sub-step of each output
interval** so it lands exactly on the output time (KDK leaves pos+vel synchronized
only on completed steps → emit only on completed step/block boundaries, and make
the clamped step a completed step). This is a real semantic change to the movie
pipeline's simulate config; the gas-free path keeps step-count cadence unchanged.

### D4 — adaptive dt BREAKS the GPU-vs-CPU trajectory oracle → THREE decoupled gates
The bound is f32 on GPU, f64 on CPU → different dt each block → divergent
trajectories. The existing "GPU matches CPU to tolerance over a trajectory" gate
has nothing to stand on. Do NOT gate a full adaptive trajectory across paths.
Decouple:
- **(a) Bound-agreement gate** — CPU `max_stable_dt` vs GPU `min_stable_dt` agree
  to f32 tolerance at a **fixed** state (no trajectory). This is the only
  cross-path numeric gate.
- **(b) Fixed-dt trajectory oracle stays** — the existing G-series GPU-vs-CPU
  fixed-dt gates are unchanged and keep guarding the force/step machinery.
- **(c) Adaptivity correctness is per-path** — each path (CPU, GPU) independently
  converges to its own fixed-tiny-dt reference (D2 gate (i)). No cross-path
  trajectory comparison at adaptive dt.

### D5 — trait surface: `max_stable_dt(&State) -> f64`, default `+∞`
Add `fn max_stable_dt(&self, state: &State) -> f64 { f64::INFINITY }` to
`ForceSolver` (default = no constraint). `GravitySph` overrides it via the existing
`sph::max_stable_dt`; `BarnesHut` inherits `+∞` (gas-free untouched). **dt
selection lives in the loop**, not the integrator — `Integrator::step` still just
takes a `dt: f64`. This keeps the integrator/solver split intact and the gas-free
path byte-identical (its solver returns `+∞`, the loop's adaptive branch is never
taken because the driver only enables adaptation on the gas path).

### D6 — CflGuard is RETIRED on the adaptive path (kept fail-loud on the fixed-dt path)
Under adaptive dt the guard's premise (a human-chosen fixed dt might be too large)
is gone — dt is chosen *to satisfy* CFL. **Realized (A4):** the adaptive path does
NOT wrap `CflGuard` at all — its fixed-dt validation has no single `dt` to check
(dt varies per block), and stability is instead structural (`plan_block` picks
`dt ≤ courant·limit` by construction) + D2b-gated. So the guard is retired on the
adaptive branch, and kept in its original fail-loud role on the fixed-dt branch. (A
per-block runtime re-check `dt ≤ limit_end` would double the CFL cost; the D2b
safety is designed-in and test-gated, not re-asserted at runtime.)

### D7 — efficiency: do not compute density twice per block
The block-boundary CFL bound needs `h`; the block's first force eval also needs
`h`. Thread the boundary CFL's `h` into the block's first force evaluation if the
structure allows, rather than recomputing `density_adaptive` twice at the same
positions. (Deferred as an optimization if it complicates the first-cut; correct
first, then fuse — but flagged here so it is not forgotten.)

---

## Milestones (TDD: red test committed separately, then green)

**STATUS (2026-07-07): A1–A4 DONE, A5 deferred (ready-to-run).** All committed +
pushed. A1 `ForceSolver::max_stable_dt` (raw c_cfl=1 limit); A2 `sim::run_adaptive`
+ `plan_block` (8 gates); A3 `xtask::simulate::simulate_gas_gpu_adaptive` (per-path
convergence green); A4 `[sim.adaptive]` opt-in on `Scenario`, `simulate_snapshots`
routes the gas path through the adaptive driver, **gasrich preset flipped to
adaptive** (retires the dt=0.005-trips-CflGuard flag), gas-free byte-identity kept.
A5: the cheap real-preset gate is green; the full-res `--release --ignored` harness
(completes + fixed-abort contrast + prefix convergence + dynamic-range measurement)
is ready-to-run, the >30 min run deferred to a later session per the user's call.
NB: the gate set below is the ADVISOR-CORRECTED one in the Gates table + D2, NOT the
original bullet text (energy-drift dropped, momentum→tripwire).

Ordered CPU-first (the oracle and the simpler loop), GPU second (reuses the
substrate), driver/cadence last.

- **A1 — `ForceSolver::max_stable_dt` trait method (CPU).** Default `+∞`;
  `GravitySph` override delegates to `sph::max_stable_dt`; `BarnesHut` inherits
  `+∞`. Red: gas solver returns finite bound matching the free `max_stable_dt`,
  gas-free returns `+∞`. (D5)
- **A2 — global block-adaptive loop (CPU).** A new adaptive driver (not a change to
  `sim::run`'s fixed-dt signature) that: queries `max_stable_dt`, applies
  `SAFETY`/`GROWTH`/`B` (D2b), sub-steps to `output_dt` with the final-substep
  clamp (D3), emits on completed boundaries. Red: (i) convergence to a fixed-tiny-dt
  reference; (ii) bounded energy drift over a merger-timescale run; (iii)
  contraction-staleness gate — realized dt ≤ end-of-block bound. (D1, D2, D2b, D3)
- **A3 — GPU-resident block-adaptive branch.** `simulate_gas_gpu` recomputes dt at
  each block boundary from `min_stable_dt`, same `SAFETY`/`GROWTH`/`B`. Red:
  bound-agreement gate CPU vs GPU at fixed state (D4a); per-path convergence (D4c).
  Keep the existing fixed-dt G-series oracle green (D4b).
- **A4 — cadence + driver wiring.** Movie `Scenario`/`SimConfig` gains
  `output_dt`/`t_end` for the adaptive path; `simulate_snapshots` routes the gas
  path through the adaptive driver, gas-free unchanged. CflGuard demoted to
  safety-net assertion (D6). Red: even-time-spaced snapshots; gas-free byte
  identity preserved.
- **A5 — full-res producibility validation (the real "done").** Run the full-res
  `gasrich` showpiece end-to-end; it must **complete** (Finding A discharged) AND
  its coarse trajectory must **converge** to a fixed-fine-dt reference on a
  short prefix. Record the realized dt(t) curve and the achieved dynamic range
  (the "size the win" number, now measured not estimated). Calibrate `SAFETY`/
  `GROWTH`/`B` from this run.

"Done" = **the showpiece completes AND converges to the reference**, not "tests
green."

---

## Gates (summary — the load-bearing part)

| Gate | Path | What it asserts |
|---|---|---|
| Reversibility + energy-oscillation | fixed-dt | UNCHANGED, kept intact (D2) |
| **Full-duration convergence to fine-dt reference** | adaptive, per-path | PRIMARY: coarse run monotonically → fine as courant↓ (D2 i, D4c). Assert `err(c/2)<err(c)` + generous abs cap — NOT a numeric order factor (variable-dt leapfrog is between 1st and 2nd order). Testbed must MOVE the CFL bound (compress the blob) or it tests fixed-dt in disguise. |
| **Contraction staleness (D2b)** | adaptive | SECOND: realized dt ≤ end-of-block C=1 bound over a converging `v=−k·x` flow — the real instability guard |
| Momentum / L conservation | adaptive | TRIPWIRE only — conserved by construction for global adaptive (`Σmᵢaᵢ·dt/2=0`), won't catch a dt/clamp/growth bug; a cheap regression guard, not a correctness gate |
| Bound-agreement at fixed state | CPU vs GPU | `max_stable_dt` ≈ `min_stable_dt`, f32 tol (D4a) |
| Fixed-dt force/step oracle | CPU vs GPU | UNCHANGED (D4b) |
| Gas-free byte identity | gas-free | UNCHANGED (D5, A4) |
| ~~Energy drift~~ | ~~adaptive~~ | DROPPED — isothermal = heat bath, no flat baseline (D4); convergence subsumes it |

---

## Risks & dependencies

- **The gate redesign is the work, not the loop.** The dt-picking loop is small;
  the danger is a future session running the wrong gate against the adaptive path
  (D2). The test-module headers must state the intent.
- **Contraction staleness (D2b)** is the one numerical instability risk unique to
  block-holding; the safety factor + growth cap + block bound must be *shown*
  sufficient, not assumed.
- **Driver-dependence (GPU).** As with all GPU work, gates prove correctness on the
  CI/Vulkan adapter, not universally.
- **Cadence semantic change (D3)** touches the movie `Scenario`/`SimConfig`; keep
  the gas-free path's step-count cadence to avoid disturbing its byte-identity gate.

---

## Relationships

`kindled-resident-cascade.md` (GPU-SPH — laid this substrate; G4/G5c on-device CFL
min is adaptive dt's input; "Relationship to adaptive dt" section), `settling-
cinder-vigil.md` (the measurement that surfaced this — Finding A is what A5
discharges), `long-burning-beacon.md` (long-horizon ordering; "individual/adaptive
timesteps" named there as the CFL follow-up), DESIGN.md (leapfrog KDK symplecticity
rationale — D2 is where the adaptive path departs from it), [[m7b-sph-forces-
decisions]] (the CFL formulation `v_sig,i = max_j(2c_s − 3w_ij)`), [[gpu-sph-
series]] (frozen-h_max ↔ held-dt contraction/expansion symmetry, D2b).
