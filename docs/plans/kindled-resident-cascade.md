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
the CPU F1 "scatter-by-plane" rewrite is **not** ported — gather-per-target is the
determinism-safe shape. Whether the GPU *eats* the O(N²) via parallelism (global
over-gather) or *fixes* it (per-particle radius) is the upstream gather-radius fork
in D4; gather-per-target is the target shape under either.

### D3 — compensated sums need the DS XOR-barrier trap
Any error-free-transform / two-sum on the GPU (the density root-find or a
compensated force sum, if used) hits the f32-reassociation fold-to-zero trap.
Reuse the `DRIFT_SHADER` fix: XOR the bits with a runtime-uniform pinned to 0 to
block the fold; launder ALL intermediates. See [[gpu-double-single-reassociation]].
Verified on the Vulkan adapter only — gates prove it on the CI adapter, not universally.

### D4 — neighbor structure: the choice is DOWNSTREAM of the gather-radius policy
"GPU hash grid vs reuse the Karras LBVH" is the wrong axis to decide first. The
structure hangs off a prior decision D2 half-dodged: **do we keep the global-h_max
over-gather, or fix it to a per-particle radius?** State that upstream choice, then
the structure follows.

**The upstream fork (gather radius):**
- **(a) Keep the global-h_max over-gather** (D2 as written — every target enumerates
  all gas within a single global `SUPPORT·h_max`, then filters by
  `r < SUPPORT·max(h_i,h_j)`; parallelize the O(N²)). Simplest, oracle-identical
  pair sets, correct regardless of the h_j gather/scatter subtlety.
- **(b) Fix it to a per-particle radius** `SUPPORT·h_i` plus a **max-h-per-node/cell
  prune** to still capture distant large-h_j neighbors. This is the scale-forward
  move: at 10⁵–10⁶ gas particles the global over-gather is fatal no matter how many
  cores eat it.

**The structure follows from that fork — not the other way round:**
- **Under (a), both structures are O(N²) and ~equivalent.** Every query radius is
  ≈`SUPPORT·h_max`, a large fraction of the occupied gas domain: a grid at
  cell≈`SUPPORT·h_max` finds ~the whole pericenter knot in its 27-cell neighborhood,
  and a BVH range query at that radius *touches most of the tree* (no pruning). The
  LBVH's adaptivity buys nothing. So "simplest to build+gate" wins → **a GPU
  counting-sort spatial hash** (Green-style: hash cell coords into a fixed table,
  counting-sort particles — NOT a dense array, and NOT the CPU's sparse HashMap;
  hashing survives far debris without exploding).
- **Under (b), structure is decisive and the LBVH wins.** A single-resolution grid
  degenerates on the h dynamic range (one cell size cannot serve a dense knot *and* a
  sparse tail), whereas a **max-h-augmented BVH range query** handles variable
  per-particle radius naturally. Reuse here is the **Karras *construction*** (Morton →
  sort → build → flatten — the expensive, already-gated part), paired with a **new
  range-query traversal** (exact: test true distance at leaves). It is emphatically
  *not* the Barnes-Hut θ-walk — D4's earlier "LBVH can't do exact neighbors" argument
  strawmanned reuse by pointing at the approximation *traversal* rather than the
  *build*.

**Caveats that keep this honest (retired as load-bearing, kept as footnotes):**
- *Not "1:1 with the oracle":* the CPU `HashGrid` is a *sparse HashMap* grid; any GPU
  port (spatial hash or BVH) is only neighbor-set-*equal*, never structurally
  identical. Gate on equality of the **filtered pair set** (post `r < SUPPORT·max(h_i,
  h_j)`), which is invariant to the radius policy — not the raw candidate set.
- *Reuse is not free:* the existing LBVH is built over **all** particles for gravity,
  but hydro neighbors are **gas-only** → reuse means either a second Karras build over
  the gas subset each step, or traversing the all-particle tree and filtering stars at
  leaves (wasted descent). It reuses strictly more gated code than a grid
  (`GpuMorton`/`GpuSorter`/builder/flattener), but it is construction-*code* reuse,
  not a free existing tree.

**Decision + the number that settles it — MEASURED 2026-07-04.** The discriminator
was cheap and concrete: the gas smoothing-length dynamic range (h ~ ρ^(−1/3);
`density_adaptive` over the gas subset + the existing gasrich snapshots). Tool +
full table: `M:\…\measure_h_range\FINDINGS.md`. Result, robust p99.5/p0.5 (raw
h_max/h_min strips out to the bisection-clamp tails, so the *robust* ratio is the
honest one):
- **The undisturbed t=0 gasrich disk is already 34× — the clean anchor.** Zero
  escapees, no bracket-clamp pressure, an equilibrium disk: a centrally-concentrated
  gas disk *intrinsically* spans >30× in h (dense core vs diffuse edge). That alone
  is 3.4× past the ≤10× "grid stays fine" threshold, before any merger dynamics.
- Pericenter only widens it: QUICK climbs to ~280× robust (peak); full-res reads ~2×
  the QUICK raw ratio at matched early steps, so QUICK is a **lower bound** on full-res.

So the range is firmly in the **100×+ regime → a single-resolution uniform grid
degenerates** (one cell size cannot serve an h~0.02 knot and an h~10 diffuse-tail
particle; the large-h_j capture problem — dense `i` must find diffuse `j` out to
`SUPPORT·h_j`≈26 — is exactly what per-node max-h on a BVH solves and a grid cannot
cheaply). The same range also condemns (a) more than "O(N²)" implies: a dense-knot
target over-gathering at `SUPPORT·h_max` scans a ~(h_max/h_i)³ candidate volume — a
per-dense-particle constant of ~10⁶–10⁸, not a benign parallel-eaten O(N²).

**Resolved:**
- **Endpoint / structure = LBVH range query with max-h-augmented nodes**, reusing the
  Karras *construction* (not the θ-walk). This is the scale-forward target the
  measurement confirms. (A multi-level grid also survives the h-range, but it is
  net-new with no reuse edge, so the LBVH dominates it *in our situation* — built +
  gated construction already exists. Not relitigated.)
- **Gather radius = (b) per-particle `SUPPORT·h_i` + max-h prune** at scale; (a) is
  the throwaway-simple first-cut only.
- **G1 staging is UNCHANGED and is a separate call the number does NOT settle.** The
  h-range decides the *endpoint*, not whether G1 *starts* as a grid. Grid-first
  remains sound de-risking: bring up G2–G6 (density root-find, force, CFL) gated
  against a *known-correct* grid before also debugging the novel, conservativeness-
  sensitive max-h range traversal (f32 AABBs must never miss an in-radius neighbor).
  The grid is then a CPU-parity oracle + fallback, not throwaway. Going straight to
  LBVH at G1 (call it **G1′**) is a legitimate *de-risk-vs-avoid-throwaway* choice —
  make it explicitly, don't read it out of the h-range number.
- **Keep G1 isolated/swappable regardless** so grid↔LBVH stays a module change.

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

- **G1 — GPU neighbor structure** ✅ **DONE** (grid-first, commit `ad31013`;
  `gpu::sph_grid::GpuNeighborGrid`). Green-style counting-sort spatial hash on wgpu:
  single-invocation build (histogram → scan → stable scatter, GpuSorter discipline),
  per-target-parallel two-pass query (count → host exclusive-scan → fill, gather so
  deterministic). All 10 gates green (filtered pair set vs `HashGrid`, order-
  independent — radius-policy-invariant so it survives the (a)→(b)/LBVH swap).
  **Walk cap (advisor-vetted):** bucket edge = `max(cell, radius/4)` bounds the walk
  to ≤9³ cells — correctness-neutral (coarser bucket only enlarges buckets), and the
  thing that makes `cell ≪ radius` (wide-h) feasible on a uniform grid at all; the
  251-cell literal walk is infeasible and is exactly what the LBVH endpoint (D4) is
  for. Cell-match acceptance dedups far-debris hash collisions. Kept isolated behind
  `query_all` so grid↔LBVH is a module change. Endpoint remains the max-h LBVH range
  query per the measured 34×+ h-range; grid survives afterward as CPU-parity oracle /
  small-N fallback.
- **G2 — GPU adaptive-h density** ✅ **DONE** (commit `a5390eb`;
  `gpu::sph_density::GpuDensity`). Per-gas-particle bracket/bisection `N_i(h)=n_ngb`
  → (ρ, h) = `Σ m_j W`, all GPU-SIDE (each thread re-walks the spatial hash per trial
  `h` — no host CSR). Shares G1's hash/bin math via `GRID_HELPERS_WGSL` (one source of
  truth). 6 gates green, DECOUPLED per advisor: summation (`densities_at` vs
  `density_fixed`, worst 1e-6) split from the root-find (`densities` vs
  `density_adaptive`, worst h 9e-4 / ρ 1.3e-3). Key advisor traps handled: (1)
  clamp/rootless divergence is worst in wide-h → main-gate cloud ASSERTED fully rooted
  (`|N_i(h_i)−n_ngb|<0.5` ∀i; robust h-range ~48×), single-particle edge gated
  STRUCTURALLY not against the seed-dependent clamped h; (2) the walk is NOT
  walk-cappable (per-particle radius = D4) — centered walk + `MAX_SPAN` backstop so a
  non-rooted `h` blow-up is a bounded wrong answer, not a GPU hang. Global seed+cap
  host-side (unique root ⇒ `h` seed-independent, so CPU's occupancy seed is skipped).
  Plain f32 accumulation, NO DS barrier (ρ/N aren't error-free-transforms; D3 not
  triggered). Endpoint still LBVH (grid mirrors CPU cell at measured-regime scale).
- **G3 — GPU hydro force** ✅ **DONE** (commit `8789743`; `gpu::sph_hydro::GpuHydro`).
  Symmetric P/ρ² pressure + Monaghan viscosity against the kernel-average grad
  `½(∇W(h_i)+∇W(h_j))`, gather-per-target (D2), all f32 on the GPU. Reuses G1's
  `GRID_HELPERS_WGSL` (one source of truth). `ρ`/`h` are INPUTS (density ran first), so
  NO root-find / all-rooted subtlety — `h` is bit-identical to both paths. **Gather
  radius = global `SUPPORT·h_max`, never per-target** (the load-bearing invariant: a
  pair with `2h_i < r < 2h_j` gives force to BOTH i and j; per-target would break
  Newton's third law). Gates (measure-then-tighten): accuracy vs `hydro_accelerations`
  rms 2.9e-7 / worst 1.2e-5 → 1e-4 / 1e-3 (house `rms/worst_rel_err` metric); **momentum
  drift 2.1e-9** — the sharp antisymmetry detector: per-pair f32 antisymmetry is EXACT
  under equal mass (`grad_w(−r)=−grad_w(r)`, `coeff` commutative-equal), so drift is
  reduction roundoff only and an O(1e-2) drift = radius-leak/sign/asymmetric-coeff bug.
  Advisor-hardened: momentum gate asserts its cloud HAS asymmetric-coupling pairs
  (`SUPPORT·min ≤ r < SUPPORT·max`, measured ~52%) so it can't silently degrade;
  viscosity gate uses a MIXED velocity field + asserts the vr split (both branch sides
  live); accuracy gate uses VARYING mass (catches a `mass[i]`/`mass[j]` swap). Plain f32
  accumulation, NO DS barrier (D3 not triggered). 8-storage-buffer limit → mass/ρ/h
  packed into one `scalars` buffer (crate stays on `Limits::default()`).
- **G4 — GPU CFL reduction — ✅ DONE** (commit `8edae61`). `gpu::sph_cfl::GpuCfl`.
  Per-gas `dt_i = C·h_i/v_sig,i` (Gadget projected signal velocity `v_sig = max` over
  approaching neighbors of `2c_s−3w_ij`, floored at `2c_s`), gathered per target over the
  global `SUPPORT·h_max`, reused from G1's `GRID_HELPERS_WGSL`. `h` is an INPUT (density
  ran first). **Two load-bearing differences from the G3 force, both gated:** the coupling
  cutoff is EXPLICIT (`r >= SUPPORT·max(h_i,h_j) ⇒ skip` — no `grad_w` to vanish), and `w`
  divides by `r` (length) not `r²`. CFL needs neither mass nor ρ → eight storage buffers
  = pos/vel/h + four grid + dt_out (no packing). **SCOPE DIVERGENCE from line ~288 below
  ("compute the CFL reduction on-GPU"):** the O(N·ngb) signal-velocity work IS on GPU; the
  trivial O(N) `min_i dt_i` collapse is reduced HOST-side in G4 (f32 `min` is exact and
  order-independent, so no numerics live there). The GPU-resident NO-READBACK min is
  deferred to **G5**, where the resident stepper's dt-threading defines its interface —
  advisor-endorsed (building it now risks the wrong shape). Gate = f32-tolerance vs
  `max_stable_dt`: per-target VECTOR (sharp — the scalar min masks a per-target-radius bug
  unless the affected particle IS the minimizer) worst 1.0e-6 → 1e-5; scalar-min 2.2e-8 →
  1e-5; cross-support approacher 8.0e-8 → 1e-4 (a per-target-`h_i` gather returns the
  static floor ~150× too large). Guards: v_sig-above-floor incl. the minimizer,
  asymmetric-coupling approaching pairs exist (19463), `c_cfl ≠ 1`. Empty ⇒ `+∞` (NOT 0 —
  a 0 falsely says every dt is too large); single ⇒ finite floor `C·h/(2c_s)`.
- **G5 — wire into `GpuResidentLeapfrog`** (gravity over all + hydro on gas subset before
  the kick; block-adaptive dt plumbing D6, compute+expose only). Decomposed into three
  red→green landings (advisor-vetted): **G5a density → G5b hydro+scatter-add → G5c CFL
  no-readback min**. Gate = long-run invariants + coarse-statistics profiles (D5), NOT a
  trajectory match.
  - **G5a — resident gas density ✅ DONE** (commit `6d87995`; `Sph` in `gpu_resident`).
    Fully resident on `FusedCore`'s device: each force eval gathers the gas subset off
    `bodies` (a small gather kernel; `bodies.xyz` is the DS hi limb the gravity force also
    reads — D1), builds the gas grid, root-finds (ρ, h), left resident for G5b. Reuses the
    G2 density WGSL VERBATIM (`DENSITY_DECLS + GRID_HELPERS_WGSL + DENSITY_KERNELS`, now
    `pub(crate)`) → one source of truth with `GpuDensity`. Density seed params extracted to
    `sph_density::density_params` (shared). **Gate is vs the CPU oracle `density_adaptive`
    over the gas subset**, NOT the standalone GPU (which shares the WGSL and so couldn't
    catch a shared bug): measured h 7.5e-4 / ρ 9.1e-4 → 1e-3 / 1.3e-3, in line with G2.
    Interleaved gas map (non-trivial → catches an identity-map bug); prime-path inclusion
    (density runs in `upload`'s prime, so `a(x₀)` is complete); gas map rebuilt from
    `state.kind` each upload. **Two caveats carried to G5b (advisor):** (1) density seed
    params (esp. grid `cell`) are FIXED at upload from the initial gas bbox — benign for
    `h_seed` (seed-independent root, G2), but frozen `cell` is a residency artifact (merger
    contraction = safe stale-large direction; uniform expansion clips at MAX_SPAN). Validated
    only at/near the primed config. (2) **The G5a gate never steps** — so per-step density
    (gather off drifted `bodies`, frozen `cell` vs evolved positions) is assumed, not tested.
    **G5b's gate MUST step** (contracting-then-mixing cloud, bounded resident-vs-CPU drift);
    if drift shows, build the on-GPU gas-bbox reduction then (same machinery as G5c's CFL
    min). Shock tube NOT run this batch — defensible: G5a doesn't touch the hydro path, and
    the only shared change (`density_params` extraction) is confirmed behavior-preserving by
    the passing standalone `sph_density` gate. Run it at G5b when hydro is wired.
  - **G5b — resident hydro force + scatter-add ✅ DONE** (commit `40af380`; `Sph` in
    `gpu_resident`). Each force eval: gather gas pos/vel/mass off `bodies`/`vel` → density
    root-find (G5a) → pack `[mass,ρ,h]` → hydro force (G3 `sph_hydro` WGSL VERBATIM) →
    `gas_acc` → scatter-add into `accel`'s gas rows AFTER the gravity traverse (unique gas
    indices ⇒ race-free). Density (cell=SUPPORT·h_seed) and hydro (cell=SUPPORT·h_max) keep
    SEPARATE grids. **Grid sizing (advisor-vetted):** `h_max` is GPU-resident, so `upload`
    runs a density-only CALIBRATION submit, reads back `h`, and freezes the hydro gather
    radius = SUPPORT·h_max before the prime (one extra submit at upload only; per-step stays
    a single submit). Reuse: `sph_hydro` DECLS/KERNELS/Params/SUPPORT + `sph_density`
    Params.cell exposed `pub(crate)` (one source of truth — byte-identical to `GpuHydro`).
    New API: `snapshot_gas_accel` (pre-scatter hydro force) + `snapshot_accel` (full accel).
    **4 gates (measured, Vulkan):** accuracy `gas_acc` vs `hydro_accelerations` fed the GPU
    (ρ,h) — rms 1.6e-7 / worst 3.9e-6; momentum drift 1.6e-8 on a DEDICATED equal-mass
    non-uniform cloud (52% asym-coupling); scatter GPU-vs-GPU (gravity-only vs gas-mode —
    star rows exact, gas rows differ by exactly `gas_acc`); stepped contracting blob (21%
    contraction) ρ/h/gas_acc vs CPU at snapshot positions (h 8.7e-4 / ρ 1.4e-3 / rms 3.2e-7),
    viscosity off (pure pressure ⇒ position-deterministic). Shock tube via the transitive
    argument (option b): resident `gas_acc` ≈ CPU hydro (accuracy gate) ≈ analytic Riemann
    (the CPU shock tube), WGSL byte-identical — NOT re-run resident (no gravity-off mode).
    **⚠ Gates only exercise the frozen-grid SAFE direction — see the G6 precondition below.**
  - **G5c — CFL no-readback min + block-adaptive dt expose** (compute only, no dt policy).
- **G6 — `simulate_snapshots` GPU branch + re-run the QUICK gasrich merger**:
  GPU path selectable alongside the CPU `GravitySph` branch; gate = QUICK gasrich
  GPU-vs-CPU coarse statistics agree, wall-clock recorded. (Full-res still blocked on
  adaptive dt — do not expect a producible full-res showpiece here.)
  - **⚠ PRECONDITION (advisor) — the frozen-`h_max` expansion landmine.** G5b freezes the
    hydro gather radius = SUPPORT·h_max at upload. Contraction over-covers (safe, gated);
    **expansion under-covers → missing hydro pairs → Newton-3 breaks and gas forces silently
    weaken, with NO gate red.** The QUICK gasrich merger EXPANDS post-pericenter (tidal tails,
    diffusion), so a single-upload + thousands-of-steps run hits this. **Compounder:** one
    clamped (non-rooted) `h` in the IC — an escapee clamps to `h_cap`≈64·h_seed, which sets
    `h_max`, which freezes `cell`=SUPPORT·h_cap → the hydro grid collapses to ~1 bucket →
    O(N²) per target for the WHOLE run (standalone G3 recomputes per call and rides it out;
    the resident freezes it; real ICs have escapees). **Cheap mitigation already in code:**
    `upload` recalibrates `h_max` every call, so G6 should **re-upload each snapshot interval**
    (bounds staleness) rather than upload-once — deferring the on-GPU gas-bbox reduction (the
    "if drift shows" item, shared with G5c's CFL min) until a true single-upload long run needs
    it. Decide re-upload-per-interval EXPLICITLY in G6.
  - **API note (advisor):** `snapshot()` rebuilds `State` via `from_phase_space`, which
    defaults every species to `Collisionless` — it DROPS `kind`. G6's simulate branch must
    re-attach `Species` after each snapshot or the gas subset comes back empty (the exact
    failure the G5b stepped gate hit and now guards against by restoring `kind`).

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
