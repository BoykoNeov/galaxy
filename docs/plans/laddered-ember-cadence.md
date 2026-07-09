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

## I0 RESULT (measured 2026-07-08; seed sweep + Amdahl split 2026-07-09) — go/no-go pivots to a SCOPE call, not FULL regen

> **TL;DR (2026-07-09, revised):** the hydro rung ceiling (I0) is tail-fragile
> (drop-finest median ~2.9×), but the binding question is the Amdahl fraction — which
> path rungs actually accelerate. There are **TWO reducibility levers**, not one:
> **(a) per-active-particle dt recompute** (the CFL solve, 17 % of cost) comes *free*
> with hydro-only rungs — you cannot assign gas rungs without per-particle `dt`, and it
> reduces to the active subset and fuses with the hydro density solve; **(b) the gravity
> tree WALK** (build:walk = 0.68 ⇒ walk is the majority) reduces only if scope EXPANDS
> to subcycle gravity on a stale tree. So: **hydro-only rungs ≈ 1.68× (drop-finest) /
> ~2.0× (median) — clears the user's 30 % bar on lever (a) alone**; **+ gravity
> subcycling ≈ 2.24× / 3.06×**. The decision is a **scope call** (how far to push, not
> whether it pays), NOT a FULL regen. See the "AMDAHL SPLIT" subsection below.
>
> **Scope call RESOLVED (2026-07-09):** build BOTH levers as a **layered opt-in
> toggle** `[sim.individual].mode = fixed-dt | hydro-only | hydro+gravity`, each
> mode droppable to the one below (so gravity subcycling can be turned off later
> without losing the hydro-only win). ⚠ The ~2.24× (`hydro+gravity`) carries an
> **unmeasured** term — I0 measured *gas CFL* rungs; lever (b)'s walk factor is
> the *star gravitational-rung* spread, gated by a new precondition **I0b**
> (gravitational `rung-spread`) before any gravity-subcycling code. `hydro-only`
> (~1.68×) stands on I0's already-measured spread and clears the bar on its own.

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

### SEED SWEEP (2026-07-09) — the tail-fragility is STRUCTURAL, not seed-noise

Four fresh QUICK realizations (seeds `0x1234`, `0xDEADBEEF`, `0xCAFEBABE`,
`0x5EED`; distinct t=0 gas fingerprints ⇒ genuinely different ICs), each stepped
on the CPU adaptive path and passed through `rung-spread`. Each seed's
auto-selected tightest-bound snapshot (full-tail ideal ceiling → **drop-finest**
= cap one rung above the finest, the "is the tail load-bearing" test):

| seed | tightest | full | drop-finest | finest rung | count @ finest | haircut |
|---|---|---|---|---|---|---|
| `0x00C0FFEE` (shipped) | t=10 | 3.90× | **1.96×** | r=10 | 6 (0.2%) | −50% |
| `0x00C0FFEE` t=30 (same run) | t=30 | 3.44× | **1.84×** | r=12 | 98 (3.9%) | −46% |
| `0x1234` | t=30 | 2.91× | **1.88×** | r=13 | 383 (15.3%) | −35% |
| `0xDEADBEEF` | t=29.5 | 5.10× | 2.95× | r=16 | 132 (5.3%) | −42% |
| `0xCAFEBABE` | t=30 | 7.37× | 4.03× | r=15 | 58 (2.3%) | −45% |
| `0x5EED` | t=30 | 4.62× | 2.90× | r=16 | 219 (8.8%) | −37% |

**Three findings, in priority order:**

1. **The finest-rung haircut is 35–50% in EVERY realization.** ~40% of the ideal
   ceiling rides on the single finest rung *everywhere* — the sweep **validates**
   the plan's tail-fragility worry rather than dispelling it. It is a **structural**
   feature of the CFL distribution at a shocked pericenter, not a shipped-seed
   artifact.
2. **Tail *population* is NOT the robustness signal — magnitude is.** `0x1234` has
   the *most*-populated finest rung of any seed (383 particles, 15.3%) yet is the
   *most* fragile fresh seed (drop-finest **1.88×**, below 2×): it fails by fine-end
   **bunching** (rungs 11–13 hold 60.7% of gas ⇒ distribution not spread ⇒ low
   speedup), a different mechanism from the shipped seed's lonely 6-particle spike
   but the same bottom line. So "populated tail ⇒ robust" is **wrong**; the honest
   discriminator is the base-speedup magnitude, which is **realization-dependent**
   (full-tail 2.9–7.4×, median ~4.9×; drop-finest 1.9–4.0×, **median ~2.9×**).
3. **The shipped seed IS genuinely gentle** (confound-checked): its *own* t=30
   snapshot has min dt 1.5e-2 vs the fresh seeds' ~1e-3 (10× looser), and the tool
   picked its t=10 (min dt 3.4e-3) as tighter still — it was selected CFL-clean, so
   it is the mildest realization. But per finding (1) that gentleness does not make
   the *fragility* seed-specific.

**Verdict unchanged: INCONCLUSIVE — the sweep does NOT flip it to GO** (and does
not kill it). "Robustly ≥3×" is **not met at QUICK**: drop-finest median ~2.9× sits
below the 3× bar, and 3 of 6 measurements (incl. one fresh seed) fall below 2×. The
screen is **weakly encouraging on raw magnitude** (full-tail 2.9–7.4×) but
**confirms tail-fragility is structural**. Whether that universal ~40% tail actually
*pays* is exactly the I7 grid-rebuild/prediction-overhead question — and QUICK
cannot answer it.

**Next (pending user, resource call): FULL-res regen** (~48 min adaptive sim, or
truncate `n_steps` to ~t=12 past pericenter) — the plan's primary regime, the real
number. The tool takes any snapshot dir, so FULL is zero rework. The sweep sharpens
what FULL must report: **drop-finest is a co-headline, not a footnote**, paired with
a real **I7 overhead** number (grid-rebuild + neighbour-prediction cost vs the
gathered force) — because the ~40% tail's payoff hinges on that overhead. **Do NOT
start I1 until FULL clears ≥3× robustly (drop-finest, not just full-tail).**

### AMDAHL SPLIT (2026-07-09) — the rung ceiling is NOT the binding number; the gravity-cadence scope is

The seed sweep left "FULL regen" as the next step, but the user pushed back: a FULL
run at *today's* N is one more point on the weak (scaling) axis, and "even 30 % off a
2-day production bake is significant." That reframed the go/no-go, and the advisor
named the number both were circling: **the rung ceiling only accelerates the gas
(hydro) stepping — the whole-sim win is capped by the Amdahl fraction of that path,**
which comes from timing an *existing* snapshot (no regen). Measured on the shipped
pericenter (`a5_movie/snapshot_00000020`, N=7500 = 2500 gas + 5000 stars), per force
eval:

Measured at **one gentle snapshot** (the shipped seed's pericenter, already shown to be
the mildest realization in the seed sweep) — timings are structural (set by N, gas
fraction, tree depth) so more transferable than the rung spread, but the build:walk
ratio specifically is clustering-sensitive (walk rises at a tighter pericenter), so read
the split with a ± and re-measure if the scope call goes forward.

| term | cost/block | reducible under rungs? |
|---|---|---|
| gravity build (Barnes-Hut, O(N)) | 120 ms | **no** — fixed floor, rebuilt at most once/block |
| gravity walk (O(active·log N)) | 176 ms | **only via lever (b)** — stale-tree subcycle |
| density + hydro (gas subset) | 347 ms | **yes** — active-subset, the core rung win |
| CFL / per-particle `dt` | 134 ms | **yes via lever (a)** — see below |
| **total** | **777 ms** | |

(Per force eval: gravity 18.5 ms = build 7.5 : walk 11.0 = **0.68**; density+hydro
21.7 ms; ×16 steps/block. CFL is once/block.)

**The key correction (advisor, 2026-07-09): CFL is a reducibility LEVER, not a fixed
cost — and it comes FREE with hydro-only rungs.** You cannot put gas on individual rungs
without a per-particle `dt_i = c·h_i/v_sig,i`; a particle's rung IS that number. Under
rungs you recompute `dt_i` only when a particle *wakes* (active subset), and the density
+ v_sig it needs is the *same* solve the hydro force already does at that tick — so the
134 ms/block is the **non-rung** "compute all gas dt once per block" baseline, and under
rungs it reduces to the active subset and fuses with the force eval. Charging it as
"fixed" (my first pass) understated the hydro-only win. So there are **two levers:**

- **Lever (a) — per-active-particle dt recompute (CFL).** Free with hydro-only rungs;
  turns the 134 ms/block from fixed into active-subset-reducible.
- **Lever (b) — gravity WALK on a stale tree.** The O(N) build (120 ms/block) can't be
  cut, but the walk (176 ms/block, the majority since build:walk = 0.68) reduces to the
  active subset IF the tree is reused stale across the base block (2^r_max ≫ 1 fine
  ticks) + inactive neighbours are predicted — the I7 "safe over-gather on a stale
  spatial structure" argument applied to gravity. Rebuild-every-tick would be
  build × 2^r_max (catastrophic), so stale reuse is mandatory, and build:walk = 0.68 +
  a ≫1-tick block make it strongly favoured. **But this is a SCOPE EXPANSION** (gravity
  prediction + a gravitational-dt floor for the now-subcycled stars), beyond the plan's
  current "hydro-only rungs, gravity untouched."

**Whole-sim speedup (Amdahl, using the drop-finest rung 2.9× as the conservative/robust
factor, median 2.9× / ideal ∞ in parentheses):**

| scope | f_accel | drop-finest 2.9× | median-tail 4.9× | ideal |
|---|---|---|---|---|
| both fixed (my first-pass strawman) | 0.45 | 1.41× | 1.55× | 1.81× |
| **hydro-only rungs — lever (a) only** | **0.62** | **1.68×** | **1.97×** | 2.62× |
| **+ gravity subcycling — levers (a)+(b)** | **0.85** | **2.24×** | **3.06×** | 6.44× |

**So the conclusion flips vs my first pass: hydro-only rungs clear the user's 30 % bar
on lever (a) alone (~1.68× drop-finest, ~2.0× median) — they do NOT "lean STOP."** The
"hydro accelerates only ~54 %" framing was wrong: it silently parked the CFL solve in
the fixed bucket when it is inherently part of the rung machinery. **I3's "kick only the
active subset each fine tick"** already implies both the hydro reduction and (with stale
reuse) the gravity-walk reduction — the plan's own integrator is model-(a)+(b)-shaped.

**The go/no-go is therefore a SCOPE call — how FAR to push, not whether it pays:**
- **Hydro-only rungs (as scoped):** ~1.68× drop-finest / ~2.0× median before I7
  overhead. Clears 30 %. Simpler; still carries the variable-dt integration risk (breaks
  symplectic leapfrog — a permanent maintenance surface on an opt-in feature), so the
  honest bar is net-of-I7-overhead AND risk-discounted, but the headroom above 1.3× is
  now real, not marginal.
- **+ gravity subcycling (scope expansion):** ~2.24× / ~3.06×, but a bigger design
  (gravity prediction + stale-tree gather + a gravitational-dt floor for subcycled
  stars).

**Answering the user's two questions directly:** (1) *"how valid at 10–100× more?"* —
the accelerable fraction erodes at scale, but only **logarithmically**, not off a cliff:
at fixed gas fraction gravity ~ N log N and hydro ~ N_gas ~ fN, so gravity/hydro ~ log N
— a gentle slope. "More stars" inflates the O(N) build floor (over ALL N incl. stars),
which is the term lever (b) can't cut, so scale specifically favours *doing* the gravity
subcycling (lever b) rather than hydro-only — but hydro-only's lever (a) win survives the
log-N dilution comfortably at any realistic N. (2) *"30 % of a 2-day bake matters"* —
agreed, the bar is ~1.3×, and hydro-only rungs (~1.68×) clear it before the bigger
gravity design is even considered.

**FULL regen is now LOW value** for this decision: a same-N run resolves neither the
build:walk-at-scale trend nor the gravity-scope call, and the seed sweep already showed
the hydro ceiling is structurally tail-fragile. The next decision is a **scope call by
the user**, not a compute run. (Throwaway harness `xtask/examples/amdahl_split.rs`
measured this via `FlatTree::build` for the build floor + `GravitySph::accel_and_dudt`
minus `BarnesHut::accelerations` for the hydro/gravity split; deleted after — the
numbers are the deliverable, reconstructable, and if the plan proceeds it should be
promoted to a TDD'd `galaxy-xtask amdahl-split` subcommand beside `rung-spread`.)

---

## What it buys — and what it does NOT

- **Buys (primary, `hydro-only` mode): wall-clock speedup** from the per-instant
  spatial spread of the *gas CFL* bound — the diffuse majority steps on coarse
  rungs while only the shocked minority steps fine. Magnitude set by I0
  (~1.68× drop-finest whole-sim), not assumed. This lever (a) already clears the
  user's 30% bar without touching gravity.
- **Buys (`hydro+gravity` mode): a further speedup from subcycling gravity** on a
  stale tree — the O(N·logN) tree WALK reduces to the active subset (the O(N)
  *build* cannot; build:walk ≈ 0.68). Targets ~2.24× whole-sim, but the walk's
  effective factor rests on the *star gravitational-rung* spread, which is
  **unmeasured** (I0 measured gas CFL rungs only) — flagged, gated by I0b, and
  could run either way (see AMDAHL SPLIT + I-grav).
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
- **A LAYERED, opt-in toggle — three modes, each droppable to the one below.**
  The Amdahl split (below) found two independent reducibility levers, so the
  feature is a *sub-toggle*, not all-or-nothing — mirroring how `[sim.adaptive]`
  / `gasrich` opt in to global adaptive. `[sim.individual].mode`:
  - **`fixed-dt`** (default / OFF) — no rungs; the fixed-dt or global-adaptive
    path runs unchanged.
  - **`hydro-only`** — gas CFL rungs (lever **a**). Delivers ~1.68× drop-finest
    on lever (a) alone (clears the user's 30% bar); collisionless stars stay on
    the coarsest rung (hydro `dt = +∞`), gravity is walked over all-N once per
    base block as today. Carries the variable-dt integration risk (breaks
    symplectic leapfrog).
  - **`hydro+gravity`** — additionally subcycles gravity on a stale tree (lever
    **b**), giving currently-`dt=+∞` stars finite *gravitational* rungs. Targets
    ~2.24× (with an **unmeasured** walk-factor caveat — see I-grav / I0b).
  - **Dropping gravity subcycling later = flip `hydro+gravity` → `hydro-only`**,
    which falls back to 1.68×, *not* to fixed-dt. This is exactly the user's
    "toggleable if we decide later we don't want it": the fallback is graceful,
    one rung of the ladder, not the whole feature.
- **Collisionless-only (gas-free) runs** stay on fixed-dt `run` in *every* mode —
  the gravitational per-particle criterion added by `hydro+gravity` exists to
  subcycle STARS *within a gas run* (so the gravity walk reduces to an active
  subset), not to turn a pure-N-body run into an individual-timestep run.
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
**Decision rule (SUPERSEDED — see AMDAHL SPLIT):** the go/no-go is no longer
"≥3× at pericenter"; the Amdahl split reframed it as a *scope call* — how far up
the ladder to build. `hydro-only` (lever a) already clears the user's 30% bar at
~1.68×; `hydro+gravity` (lever b) targets ~2.24×. Hours, not days.

### I0b — MEASURE the star GRAVITATIONAL-rung spread (precondition for `hydro+gravity` ONLY)
I0/rung-spread measured **gas CFL rungs only** (`h_i / v_sig,i`). Lever (b)'s
~2.24× — the gravity-walk reduction — rests on the **star gravitational-rung**
distribution, a *different criterion* (`dt ~ η·√(ε / |a_i|)`, not `c·h/v_sig`)
over a *larger, different population* (5000 stars vs 2500 gas in the measured
snapshot). Borrowing gas's 2.9× drop-finest for the walk is an **unmeasured
extrapolation**, and it is genuinely two-sided: stars bunching fine near the
merger core weaken the factor; a broad spread with many coarse slow stars
strengthens it. (Direction note: the baseline `run_adaptive` already steps *all*
stars at the global min dt — wasteful for slow stars — so subcycling gravity
helps stars too; treating the walk as accelerable is defensible, just
un-quantified.) **I0b = a gravitational analogue of `rung-spread`**: histogram
`dt_i = η·√(ε/|a_i|)` over stars+gas at pericenter, drop-finest factor, projected
walk speedup. Run it **before any `hydro+gravity` (I-grav) code** — it firms up
the one number the ON path's payoff hangs on. It is NOT a precondition for
`hydro-only`, which stands on I0's already-measured gas spread. Measure it from
an existing snapshot; no regen. This is a distinct tool from `rung-spread`
(different criterion), so it is deliberately deferred to the point it pays,
per the advisor.

**I0b RESULT (2026-07-09) — MARGINAL, reopens the scope call (does NOT close it).**
`grav-rung-spread` (xtask) landed and ran on the retained gasrich QUICK run
(`m7f_gasdemo`, seed 0x00C0FFEE, ε=0.05, θ=0.5). At the star gravitational
pericenter (t=28, `snapshot_00005600`):

- **star drop-finest walk factor = 1.42×** (full-tail 2.84×). 56% of stars bunch
  on a single rung — exactly the `dt ∝ |a|^(−½)` compression the criterion
  predicts (spread NARROWER than the gas CFL spread, as flagged two-sidedly above:
  this run landed on the *bunch-fine* side).
- **Amdahl reprojection** (2026-07-09 block split: build 120 / walk 176 /
  hydro 347 / cfl 134 ms, total 777):
  - hydro-only (lever a, ships regardless): **1.68×**
  - hydro+gravity, MEASURED w_grav=1.42×: **1.90×** — only **+13%** over hydro-only.
  - hydro+gravity, full-tail w_grav=2.84×: **2.23×** — recovers the pre-registered
    **2.24×**, confirming that number was the *ideal ceiling* (borrowed 2.9×); the
    finest-rung penalty pulls the realistic drop-finest figure down to 1.90×.
- **Robustness:** a θ=0 exact (direct-sum) rerun gives an identical rung
  distribution (drop-finest 1.42×, cross-check 1.3e-14) ⇒ the bunching is
  **physical**, not a Barnes-Hut opening-angle artefact.
- **This is the best-moment figure** — measured at pericenter (widest spread);
  the whole-run average walk factor is ≤ this, so 1.90× is an upper read.

**Caveat (load-bearing, NOT harmless):** single seed, QUICK resolution. At FULL
(smaller ε, deeper resolved wells) peak |a| rises ⇒ the star gravitational spread
*widens* ⇒ subcycling gets MORE attractive, not less. The comparable hydro finding
needed a 4-seed sweep to call itself structural; I0b is one seed / one res. So the
verdict is **"the precondition came in below estimate (2.24× → 1.90×) — the
`hydro+gravity` ROI is marginal at QUICK and reopens the user's scope call,"** NOT
"don't build I-grav." Building `hydro-only` remains unconditionally worth it (I0's
1.68× stands on already-measured gas rungs). A FULL/seed-sweep confirmation of the
star spread is the natural gate before committing I-grav code, if the user wants
to pursue the gravity layer.

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

### I-grav — gravity subcycling (`hydro+gravity` mode ONLY; the lever-b design surface)
This is the whole cost of chasing ~2.24× over `hydro-only`'s ~1.68×, and it is
**gated OFF unless `[sim.individual].mode = "hydro+gravity"`**. Three coupled
pieces, none needed by `hydro-only`:

1. **A gravitational per-particle dt criterion for STARS.** Collisionless stars
   have hydro `dt = +∞` (coarsest rung) — under `hydro-only` they never subcycle,
   so the gravity walk stays all-N. To reduce the walk to an active subset, stars
   need a *finite* rung from a gravitational criterion `dt_i = η·√(ε/|a_i|)`
   (Plummer softening `ε`, `|a_i|` the gravitational accel). This is the item the
   old Scope parked as "a separate later item" — it is now **in scope, behind the
   toggle**. A floor keeps the coarsest slow stars from an unbounded rung.
2. **Stale-tree gravity gather (the efficiency crux, gravity edition of I7).**
   Rebuild the O(N) tree/grid ONCE per base block; on fine ticks, walk the
   *active subset* against the stale-but-dilated tree. The O(N) build is the
   fixed floor lever (b) cannot cut (and "more stars" inflates it — the log-N
   headwind); the walk is what reduces. Rebuild-every-fine-tick = build × 2^r_max
   = catastrophic, so stale reuse is mandatory. Same "safe over-gather" argument
   as the frozen-`h_max` hydro gather (I7) and the G-series LBVH endpoint.
3. **Gravity prediction of inactive neighbours (gravity edition of I6).** An
   active target's gravity walk gathers contributions from inactive stars/gas
   that were last synced earlier; those must be drift-predicted to the fine tick
   before the walk, exactly as SPH neighbours are (I6). Integrator-owned scratch,
   not `State`.

**Caveat carried on the ~2.24× — NOW MEASURED (I0b, 2026-07-09):** its walk factor
was the unmeasured star gravitational-rung spread. I0b landed it: the drop-finest
star walk factor is **1.42×**, reprojecting `hydro+gravity` to **1.90×** (only +13%
over hydro-only's 1.68×), not 2.24×. The 2.24× was the *ideal ceiling* (full-tail
2.84× ≈ borrowed 2.9×). Verdict MARGINAL at QUICK res / one seed; FULL plausibly
widens the spread (deeper wells). See the **I0b RESULT** block above — the payoff
reopens the user's scope call rather than clearing a bar.

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
  bit-exact self-check vs `max_stable_dt`. QUICK = 3.90× but tail-fragile; the
  2026-07-09 seed sweep (4 fresh seeds) confirms the ~40% finest-rung haircut is
  **structural** (drop-finest median ~2.9×, below 3×) not seed-noise. The 2026-07-09
  **Amdahl split** (revised) reframed the binding question as *which path rungs
  accelerate*: TWO reducibility levers — (a) per-active-particle dt recompute (the CFL
  solve, free with hydro-only rungs since a rung IS the per-particle dt) and (b) the
  gravity walk on a stale tree (needs scope expansion; build:walk=0.68). Whole-sim win:
  **hydro-only rungs ~1.68× drop-finest / ~2.0× median — clears the 30% bar on lever
  (a) alone**; +gravity subcycling ~2.24×/~3.06×. **The gate is no longer "≥3× at
  FULL"; it is a SCOPE call — how FAR to push (hydro-only vs +gravity), not whether it
  pays. Hydro-only already clears the bar.** FULL regen is low value for this decision
  (same-N, resolves neither the log-N scaling trend nor the scope call). See "AMDAHL
  SPLIT".
- **I0b — gravitational rung-spread (xtask; PRECONDITION for `hydro+gravity` ONLY,
  NOT for `hydro-only`). DONE 2026-07-09 — MARGINAL, see "I0b RESULT" above.**
  `galaxy-xtask grav-rung-spread <dir>` histograms the star gravitational criterion
  `dt_i = η·√(ε/|a_i|)` at pericenter + diffuse, drop-finest factor, θ cross-check vs
  direct sum, Amdahl reprojection. RESULT: star drop-finest walk factor **1.42×**
  (full-tail 2.84×) ⇒ `hydro+gravity` reprojects to **1.90×**, only +13% over
  hydro-only's 1.68× (NOT the 2.24× ceiling). θ=0 rerun identical ⇒ bunching is
  physical. Single seed / QUICK res — FULL plausibly widens the spread. The gravity
  layer's payoff is MARGINAL here; the scope call reopens rather than clears.
  (I0b, I-grav)
- **I1 — per-particle CFL vector. DONE 2026-07-09 (RED 5a90e40 / GREEN 3aa9cd3).**
  `sph::max_stable_dt_per_particle` + `ForceSolver::max_stable_dt_per_particle`
  (trait default `vec![+∞; len]`, `GravitySph` overrides at `c_cfl=1`). Vector is
  state-indexed (gas rows finite at their global index, collisionless `+∞`), a
  textually-verbatim parallel copy of the scalar's inner loop with the min-fold
  replaced by a store — the shipped scalar stays FROZEN. Gates: `min ≡ scalar`
  bit-for-bit (BOTH EOS arms), collisionless `+∞`, static-cloud FULL-vector
  closed-form pin, non-minimal `−3w` approacher pin (advisor teeth — `min ≡ scalar`
  only guards the minimal particle), GravitySph trait plumbing. (I1)
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
- **I6 — full-res producibility + speedup validation, `hydro-only` mode (the
  real "done" for lever a).** Run the full-res `gasrich` showpiece through
  `run_individual` in `hydro-only` mode; confirm it completes, converges to the
  reference, and delivers the ~1.68× hydro-only speedup (or explain the gap).
  "Done" = **completes AND converges AND the measured speedup justifies the
  path**, not "tests green." At this point the toggle ships at `hydro-only` and
  the 30% bar is cleared.
  - **I6→I-grav FULL star-spread gate (advisor, 2026-07-09, build phase).** I6
    emits FULL-res snapshots; re-running `grav-rung-spread` on them is a ~30s
    xtask (no regen), so **re-measure and REPORT the FULL star gravitational
    rung spread at the I6→I-grav boundary** before any I-grav commits. I0b's
    +13% (`hydro+gravity` 1.90×) was QUICK/one-seed; the physics predicts the
    spread WIDENS at FULL (deeper resolved wells ⇒ higher peak |a| ⇒ finer rungs
    for the fast stars, broader tail), so the FULL number is likely +30–40%, not
    +13%. Not an abort gate (the user chose to build the layer) — a **record**
    gate: land the real number before spending the I-grav design budget.
- **I-grav — gravity subcycling (`hydro+gravity` mode; the lever-b follow-on,
  ONLY after I0b clears).** Gated behind `[sim.individual].mode="hydro+gravity"`.
  Red: (i) star gravitational rung assignment + floor (pure fn, like I2); (ii)
  stale-tree gravity walk over the active subset converges to a
  rebuild-every-tick reference to tolerance (the I7 argument, gravity edition);
  (iii) gravity prediction of inactive neighbours keeps the walk correct across a
  base block (the I6 argument, gravity edition); (iv) full-res speedup validation
  vs `hydro-only` — deliver the I0b-measured walk factor toward ~2.24× or explain
  the gap. Sequenced AFTER I0b confirms the star gravitational-rung spread makes
  it worth building. (I0b, I-grav)

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
| **Stale-tree gravity walk convergence** | `hydro+gravity` only | I-grav: active-subset walk on a once-per-base-block tree → rebuild-every-tick reference to tolerance (the I7 over-gather argument, gravity edition). |
| Gravity neighbour prediction correct across a base block | `hydro+gravity` only | I-grav: drift-predicted inactive stars/gas keep the walk correct at fine ticks (the I6 argument, gravity edition). |
| Mode ladder falls back gracefully | toggle | `hydro+gravity`→`hydro-only`→`fixed-dt`: dropping a rung of the ladder never touches the paths below it (their gates stay green). |
| Fixed-dt reversibility + energy oscillation | fixed-dt | UNCHANGED, untouched. |
| Global-adaptive convergence + D2b | global | UNCHANGED, untouched. |
| Gas-free byte identity | gas-free | UNCHANGED. |
| ~~Energy conservation~~ | ~~individual~~ | NONE — isothermal heat bath + variable per-particle dt (D4/D2 carry over and worsen); convergence subsumes it. |

---

## Risks & dependencies

- **I0 is DONE and did NOT kill the plan** — it reframed the go/no-go into a
  scope call (AMDAHL SPLIT). `hydro-only` clears the 30% bar at ~1.68×; the
  original "<2× → stop" rule is superseded. The residual kill risk is now
  per-mode, not whole-plan.
- **The ~2.24× (`hydro+gravity`) rests on an UNMEASURED number.** The gravity-walk
  reduction assumes a favourable *star gravitational-rung* spread; I0 measured
  only gas CFL rungs. **I0b must land before I-grav code** — if the star spread
  is narrow (stars bunch fine near the core), lever (b) underdelivers and the
  right call is to ship `hydro-only` and stop. This is the gravity-path kill
  switch, and it is why gravity is a droppable rung of the toggle, not baked in.
- **The efficiency trap (I7) is the second kill switch.** Grid rebuild +
  neighbour prediction per fine tick can eat the savings; the rebuild cadence
  must be resolved, and the I6 measurement must show the *net* win, not the
  ideal-integrator win. The **gravity** stale-tree gather (I-grav) inherits this
  trap in a sharper form: the O(N) tree build is a fixed floor lever (b) cannot
  cut, and "more stars" inflates it (log-N headwind) — the net gravity win must
  be measured, not assumed.
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
