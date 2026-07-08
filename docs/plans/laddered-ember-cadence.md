# Individual (per-particle rung) timesteps on the SPH path

Scoping doc, written 2026-07-08. The deferred follow-up named in
`courant-quickening-cadence.md` (global block-adaptive, A1–A5 DONE) and in
`long-burning-beacon.md` Chain A step 5 ("INDIVIDUAL … block timesteps remain
the deferred follow-up"). Global adaptive gives ONE `dt` shared by all
particles, recomputed at block boundaries. This lifts that to a **per-particle**
timestep: each gas particle sits on its own power-of-two rung and is
re-integrated only when its rung is due, so the diffuse majority of the box
steps far less often than the shocked knot that pins the global bound.

**This is a scoping doc.** The design below is advisor-vetted (2026-07-08). The
first milestone is a **measurement that is a go/no-go** — see I0. Do not start
the integrator rewrite until the distribution number is in hand.

---

## ⚠ CRITICAL: the 34× is the WRONG number — measure the right one first

The A5 run's headline "CFL-bound dynamic range 34.2×" (min 3.42e-3, max
1.17e-1) is the **temporal** range: the value of the *global* (min-over-
particles) CFL bound at *different times* in the run. **Global block-adaptive
already captures every bit of that** — `sim::run_adaptive` sets `dt` from exactly
this global min at each block boundary (`plan_block`, `sim/src/lib.rs:242`). It
is banked. Individual timesteps do **not** re-bank it.

What individual timesteps buy is a **different** quantity: the **spatial spread
of `h_i / v_sig,i` across particles at a single instant** — at the moment the
global bound is pinned to its tightest knot, how many particles *actually* need
that tiny step versus could take 8–30× more. That distribution is exactly the
per-particle vector `solvers/src/sph/cfl.rs` computes and then `min`s away
(lines 103, 139). **It has never been measured.** It — not the 34× — governs
the speedup:

```text
expected wall-clock speedup ≈ N / (Σ_r active_count(r) · 2^(r_max − r))
                            ≈ N / (effective count of short-rung particles)
```

- Small dense knot sets the bound, rest of the box diffuse → **large win**.
- Shocked region is a big fraction of particles at pericenter → **<2×**, and
  the integrator rewrite is not worth it.

**I0 (below) is a cheap xtask that produces this number. Write it into this doc
before committing to the rewrite.** This is the tightest constraint; verify it
first.

---

## I0 RESULT (measured 2026-07-08) — INCONCLUSIVE at QUICK; FULL pending

The xtask exists: `galaxy-xtask rung-spread <snapshots_dir | .snap>` (isothermal
arm of `cfl.rs` copied verbatim, `min` removed; the copy's `min` is asserted
**bit-for-bit equal** to the shipped `max_stable_dt` on every reported snapshot —
the I1 invariant used as a runtime self-check, so the number rides a verified code
path without touching the shipped bound). It scans a run for the pericenter (the
tightest global bound), histograms the per-particle rungs there and at the early
diffuse snapshot, and reports the ideal-ceiling speedup `N_gas·2^r_max / Σ_i 2^r_i`
plus a tail-sensitivity sweep. Corrections folded in vs the sketch above:
- **`N` is GAS-ONLY.** Collisionless rows have `dt = +∞` (coarsest rung); padding
  `N` with them inflates the ratio ~3.4× and could flip the verdict. Speedup is of
  the **SPH/hydro stepping**, not whole-sim (gravity-over-all is untouched by rungs).
- **The plan's printed denominator `Σ_r n_r·2^(r_max−r)` has the exponent inverted**
  (gives speedup <1). Correct: `speedup = N_gas·2^r_max / Σ_i 2^r_i ≡ N_gas / Σ_i
  2^(r_i−r_max)`, matching the plan's own gloss "≈ N / effective short-rung count".
  `⌈log2⌉` binning under-states the win (safe side). The ceiling **excludes** I7
  overhead (that's I6's net number).
- Speedup is **invariant to `dt_base`** (the diffuse end); it is governed entirely
  by the fine tail `dt_min`.

**Only QUICK snapshots are retained** (`m7f_gasdemo`, and the QUICK `a5_movie`
render — both seed `0x00C0FFEE`, 2500 gas). The A5 **full-res** run's snapshots went
to a test tempdir and were **not** retained; only its log survives. So the number
below is QUICK — NOT the plan's decision regime.

QUICK pericenter (t=10, snapshot 20/60): **3.90× ideal ceiling**, N_gas=2500,
spatial dynamic range **537×** (vs the temporal 34× that global adaptive already
banks). **VERDICT: INCONCLUSIVE.** The 3.90× straddles the ≥3× GO line and is
**tail-fragile**: the finest rung r=10 is only **6 particles (0.2%)**; resolving one
rung coarser (cap r=9) drops the ceiling to **1.96× (<2× STOP)**. The 6 are
*physical* (v_sig ≈ 3.9, a real shock; h ≈ 0.035 vs median 0.089 — not an
artifact-`h` outlier), so the win is genuine per the definition — but it is a
small-number statistic at QUICK resolution.

**QUICK cannot settle the go/no-go, and the QUICK→FULL direction is two-sided:**
soft QUICK gas (large `h`) *narrows* the spread ⇒ FULL ≥ QUICK; but FULL has 2.8×
more gas and a better-resolved shock could put a *larger fraction* on the fine rungs
⇒ FULL < QUICK (the plan's own "<2×" stop case). These cancel — no extrapolation.

**Next (pending user, resource call):** (1) cheap QUICK **seed sweep** (the shipped
seed was chosen CFL-clean/mild — 3–4 other realizations bracket whether the
6-particle tail is this-seed noise); (2) **FULL-res regen** (~48 min adaptive sim,
or truncate `n_steps` to ~t=12 past pericenter) — the plan's primary regime, the
real number. The tool takes any snapshot dir, so FULL is zero rework. **Do NOT start
I1 until FULL clears ≥3× robustly (tail-fragility resolved).**

---

## What it buys — and what it does NOT

- **Buys (primary, IF I0 clears): wall-clock speedup** from the per-instant
  spatial spread of the CFL bound — the diffuse majority steps on coarse rungs
  while only the shocked minority steps fine. Magnitude set by I0, not assumed.
- **Buys (secondary): shock fidelity via the timestep limiter** — a slow-rung
  particle hit by a fast shock is forced awake (I5). This is a *correctness*
  gain over global adaptive's uniform coarse step in quiescent phases, not just
  speed.
- **Does NOT buy:** anything on the gas-free (collisionless) path, exact energy
  conservation, or time-reversibility (variable per-particle dt forfeits both,
  same as global adaptive — D2 of the parent plan carries over and worsens).
- **Does NOT buy** a GPU speedup — GPU is deferred (I-GPU); the resident path's
  single-`dt` batching IS its throughput win and per-particle rungs defeat it.

---

## Scope (v1)

- **CPU only.** SPH (gas) path. Both EOS arms (isothermal + adiabatic/thermal),
  because the thermal path (`LeapfrogKdkThermal` + `u`) is now the physics of
  interest post-E and its `du/dt` kick must be rung-aware too.
- **Third path, byte-untouched neighbours.** A new active-set integrator AND a
  new driver (`sim::run_individual`), added *beside* `run` (fixed-dt) and
  `run_adaptive` (global) — exactly how `run_adaptive` was added beside `run`.
  The fixed-dt and global-adaptive byte-paths are **literally untouched**, so
  their gates (fixed-dt reversibility/energy oscillation; global-adaptive
  convergence + D2b) stay intact and green.
- **Gas-free path untouched** — collisionless runs keep fixed-dt `run`; a
  gravitational per-particle criterion is a separate later item.
- GPU individual timesteps: **deferred**, rationale recorded (I-GPU).

---

## Key decisions (with rationale)

### I0 — MEASURE the per-instant rung distribution FIRST (go/no-go)
A tiny xtask: load a `gasrich` snapshot — **near pericenter, where the global
bound is tightest and individual timesteps help most** (and, for contrast, an
early diffuse snapshot) — run the existing per-particle CFL body with the `min`
**removed**, and histogram `dt_i = c_cfl · h_i / v_sig,i`. Reuses `cfl.rs`
almost verbatim (the per-`i` loop already exists; drop the `min_dt.min(...)`
fold and collect the vector). Report: the rung histogram, the fraction on the
tightest rung, and the projected speedup `N / Σ_r n_r · 2^(r_max−r)`.
**Decision rule:** projected speedup ≥ ~3× at pericenter → build. < ~2× → stop,
record the finding, global adaptive is enough. Hours, not days.

### I1 — per-particle CFL is a VECTOR, not the scalar min
`ForceSolver::max_stable_dt` returns `f64` (the min). Individual timesteps need
`h_i / v_sig,i` per gas particle. Add a per-particle variant — either
`max_stable_dt_per_particle(&State) -> Vec<f64>` on the trait (default: a
1-element or gas-length fill of `+∞` for non-hydro solvers) or a `sph`-level
free function the driver calls directly (mirroring how `sph::max_stable_dt` is
called). The scalar `max_stable_dt` stays as-is for the global path — the vector
is additive. Collisionless rows get `+∞` (never rung-limited by hydro).

### I2 — power-of-two rungs below a base dt, synchronized at the base boundary
Assign particle `i` to rung `r_i = clamp(⌈log2(dt_base / (courant·dt_i))⌉, 0,
r_max)`, so its step is `dt_base / 2^r_i`. All rungs synchronize at every
`dt_base` boundary (the standard KDK block-power-of-two scheme): pos AND vel are
consistent there, which is the only place snapshots may be emitted (mirrors the
global path's D3 emit-on-completed-boundary rule). `dt_base` itself is chosen
from the *coarsest* particle's CFL (or a scenario cap), and re-derived each base
block — so this composes with, not replaces, the global bound tracking.

### I3 — a NEW active-set integrator + NEW driver (leave the other two alone)
The existing integrators (`LeapfrogKdk`, `LeapfrogKdkThermal`) kick/drift the
WHOLE state with one `dt` (`core/src/integrator.rs`). Individual timesteps kick
only the *active* subset each fine tick and drift inactive particles by
prediction. That does not fit the `Integrator::step(dt)` interface, so it is a
**distinct type** with an active-mask-aware lifecycle, plus a `run_individual`
driver that owns the rung schedule. Do not branch the existing integrators.

### I4 — momentum is NOT conserved by construction here (a real fork, pick one)
The global-adaptive plan's momentum gate DOES NOT carry over. Global adaptive
kicks *all* particles with one dt, so `Σ mᵢ aᵢ = 0` exactly. Individual
timesteps kick only the active subset — the equal-and-opposite reaction on an
inactive neighbour is omitted this tick and only partially cancels at later,
differently-configured evaluations. So there is a **genuine, bounded momentum
error.** Fork:
- **(a) Gadget-style: kick active only.** Momentum drifts (bounded); cheap; the
  standard choice. **← v1 picks this.**
- **(b) Kick inactive neighbours too.** Momentum-preserving but re-touches
  inactive particles, partially defeating the savings I0 is measuring.

v1 = (a). The momentum gate becomes a **bounded-drift diagnostic** (measure it,
assert it stays under a documented bound over a merger-timescale run), NOT a
roundoff tripwire.

### I5 — the timestep limiter (Saitoh–Makino 2009) is CORRECTNESS, not a dial
The user's "if something's off, lower the timestep of members" maps onto this,
but frame it as **load-bearing**: a slow-rung particle sitting in cold gas that
is suddenly hit by a shock from a fast-rung neighbour will not "notice" until
its next scheduled wake-up — by then the shock has passed through a particle
integrated at the wrong (too-coarse) dt, corrupting exactly the shocked-merger-
gas physics that is this project's whole point (and now *adiabatic*, so the
mis-integrated `u` poisons temperature/pressure downstream). The limiter forces
any particle within `N_limit` rungs of a more-active neighbour to wake and
demote. Its correctness gate — drive a shock into a slow-rung region, assert the
struck particles wake and capture the same energy as a fully-fine reference — is
**central, alongside convergence.** `N_limit` (typically 1) is the only genuine
tuning dial; the mechanism is not optional.

### I6 — neighbour prediction: inactive neighbours must be drifted to current time
SPH force on an active target `i` gathers over neighbours `j`, and the viscosity
/ PdV terms depend on `v_ij` and `r_ij` **at the current sub-time** — but an
inactive `j` was last synchronized at an earlier base-sub-time. So inactive
neighbours must be **predicted** (drift-extrapolated: `x_j ≈ x_j^sync +
v_j·Δt`, and `v_j` predicted for the viscosity term) to the active tick's time
before the gather. This needs per-particle "last-sync pos/vel(/acc)" storage in
the integrator (NOT in `State` — the D2 "h/ρ/derived never stored in State"
discipline; predicted quantities are integrator-owned scratch, like the cached
`acc`). Decide the predictor order (drift-only vs drift+½a·Δt²) in I3; gate it
inside the convergence test.

### I7 — grid-rebuild cadence is the efficiency crux (do not rebuild every fine tick)
The naive loop rebuilds the O(N) `HashGrid` (`cfl.rs:65`, `forces.rs`) every
*fine* tick even when only a handful of particles are active — and if
grid-rebuild + neighbour-prediction cost is comparable to the gathered force
itself, the I0 savings **evaporate**. Decide the rebuild cadence in the plan:
rebuild at the coarsest (base) cadence and query the stale-but-dilated grid on
fine ticks (positions have moved < a fraction of `h`, so a small search-radius
dilation keeps neighbour lists complete — the same "safe over-gather" argument
as the frozen-`h_max` global-support gather), OR rebuild only when the active
fraction exceeds a threshold. This is the D7-analog "correct first, then fuse"
item, but it is more load-bearing here than in the global plan and must be
resolved, not deferred.

### I8 — thermal arm doubles the integrator surface
The active-set integrator must kick `u` (via `du/dt` from `accel_and_dudt`) and
apply the `u`-floor (E4b) **per active subset**, exactly as `LeapfrogKdkThermal`
does for the whole state. Isothermal (`LeapfrogKdk`, no `u`) is the simpler
first arm; the thermal arm lands second (I5-driver already in place). The
`u`-floor leak accounting must still be reported (bounded non-conservation).

### I-GPU — GPU individual timesteps DEFERRED (rationale recorded)
`GpuResidentLeapfrog::step_many` batches ≤`MAX_BATCH` steps into one submit at a
single `dt` uniform — that batching IS the residency throughput win. Per-particle
rungs mean per-particle active flags, predict kernels, and scatter-add on a
varying active set — research-grade on GPU and it *removes* the single-dt batch
win. v1 is CPU-only; the CPU path is the oracle a future GPU port would gate
against, exactly as the LBVH/G-series lineage did.

---

## Milestones (TDD: red test committed separately, then green)

- **I0 — measurement / go-no-go (xtask, NOT a red/green milestone). DONE (tool);
  INCONCLUSIVE at QUICK — see "I0 RESULT" above.** `galaxy-xtask rung-spread <dir>`
  histograms per-particle `h_i/v_sig,i` at pericenter + diffuse, min removed, with a
  bit-exact self-check vs `max_stable_dt`. QUICK = 3.90× but tail-fragile (STOP if
  the 6-particle 0.2% shock tail drops one rung); FULL-res (the decision regime) not
  retained → pending a regen. **Gate the rest of the plan on ≥ ~3× at pericenter that
  survives the tail-fragility test, measured at FULL res.**
- **I1 — per-particle CFL vector.** Red: the vector's `min` equals the existing
  scalar `max_stable_dt` bit-for-bit on a fixed state (the vector is a strict
  generalization); collisionless rows are `+∞`. (I1)
- **I2 — rung assignment (pure fn).** Red: power-of-two binning is monotone,
  clamped to `[0, r_max]`, and a uniform-CFL state maps every particle to the
  same rung. Unit-testable without stepping. (I2)
- **I3 — active-set KDK integrator + predictor (ISOTHERMAL first).** Red: (i)
  single-rung run reduces to global-adaptive **to tolerance** (NOT bit-identical
  — active-set ordering + prediction differ; decide the tolerance up front and
  do not overclaim); (ii) predictor keeps neighbour lists correct across a base
  block. (I3, I6, I7)
- **I4 — `sim::run_individual` driver + timestep limiter.** Red: (i)
  full-duration convergence to a fine-dt reference on a CFL-moving testbed
  (PRIMARY, per-path); (ii) **limiter shock-wakeup** — shock into a slow-rung
  region, assert wake-up + captured energy vs a fully-fine reference (CENTRAL);
  (iii) momentum bounded-drift diagnostic. (I4, I5)
- **I5 — thermal arm.** Red: adiabatic single-rung reduces to global-adaptive
  thermal to tolerance; `u`-floor leak reported; convergence holds with `du/dt`
  kicked per-rung. (I8)
- **I6 — full-res producibility + speedup validation (the real "done").** Run
  the full-res `gasrich` showpiece through `run_individual`; confirm it
  completes, converges to the reference, and delivers the I0-projected speedup
  (or explain the gap). "Done" = **completes AND converges AND the measured
  speedup justifies the path**, not "tests green."

---

## Gates (summary — the load-bearing part)

| Gate | Path | What it asserts |
|---|---|---|
| **Full-duration convergence to fine-dt reference** | individual, per-path | PRIMARY: coarse run → fine as courant↓ on a CFL-MOVING testbed (same discipline as global adaptive's convergence gate). |
| **Timestep-limiter shock-wakeup** | individual | CENTRAL correctness (I5): shock into a slow-rung region wakes the struck particles and captures the same energy as a fully-fine reference. |
| Reduces to global-adaptive at single rung | individual vs global | TOLERANCE (not bit-identical — active-set/prediction ordering; I3). |
| Rung synchronization | individual | pos+vel consistent at every `dt_base` boundary; snapshots emit only there (I2). |
| Momentum bounded-drift | individual | DIAGNOSTIC not tripwire (I4): drift stays under a documented bound (kick-active-only forfeits exact conservation). |
| Per-particle CFL vector `min` ≡ scalar | solver | I1: the vector is a strict generalization of the shipped scalar bound. |
| Fixed-dt reversibility + energy oscillation | fixed-dt | UNCHANGED, untouched. |
| Global-adaptive convergence + D2b | global | UNCHANGED, untouched. |
| Gas-free byte identity | gas-free | UNCHANGED. |
| ~~Energy conservation~~ | ~~individual~~ | NONE — isothermal heat bath + variable per-particle dt (D4/D2 carry over and worsen); convergence subsumes it. |

---

## Risks & dependencies

- **I0 can kill the plan.** If the pericenter spatial spread is narrow (<2×),
  the rewrite is not worth it — record the finding and stop. This is the whole
  reason I0 is first.
- **The efficiency trap (I7) is the second kill switch.** Grid rebuild +
  neighbour prediction per fine tick can eat the savings; the rebuild cadence
  must be resolved, and the I6 measurement must show the *net* win, not the
  ideal-integrator win.
- **The timestep limiter (I5) is not optional** — without it the adiabatic
  shocked-merger physics (this project's aim) is silently wrong, with no red
  gate unless the shock-wakeup test exists. Build the test with the mechanism.
- **Ordering non-determinism.** Active-set gathering must keep the fixed
  ascending-index gather order (the M7a parallel≡serial discipline) within each
  tick so the run is reproducible.
- **GPU deferral** means the showpiece speedup is CPU-only until a GPU port; the
  CPU path is deliberately the oracle for that later port.

---

## Relationships

`courant-quickening-cadence.md` (global block-adaptive — the parent; D1/D2/D2b/D3
carry over, D4-momentum does NOT — see I4), `smoldering-thermal-ledger.md` (the
E-series thermal path `run_individual` must keep rung-aware — I8),
`long-burning-beacon.md` (Chain A step 5 names this deferral),
`kindled-resident-cascade.md` (GPU-resident single-dt batching — why GPU is
deferred, I-GPU), [[adaptive-dt-series]] (the 34× is temporal, already banked —
the correction I0 acts on), [[m7b-sph-forces-decisions]] (the per-particle CFL
`v_sig,i` formulation I1 vectorizes), DESIGN.md (leapfrog KDK symplecticity — D2
is where per-particle dt departs from it further than global adaptive did).
