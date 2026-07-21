# Natal-ember forge — physical star formation (gas → star conversion)

Scoping doc, written 2026-07-21. Chain A step 4 of `long-burning-beacon.md`:
**star formation** — dense SPH gas genuinely converts to collisionless star
particles carrying a **formation time**, and the render colors new stars by
**age**. This is the physically-faithful endgame the `stellar-nursery-glow.md`
proxy (color stars by *ambient gas ρ*) was plumbing toward. It retires nothing
by force — it adds the real signal beside the proxies — but it is the milestone
that lets the merger's shocked lanes light up with genuinely-young stars.

Scope decisions locked with the user (2026-07-21):
- **Conversion + age-coloring only. NO feedback this milestone.** Feedback
  (energy injection into `u`) is the unstable, heavily-tunable half and is its
  own later session. Conversion + age-color is a complete, demoable,
  fully-gated milestone on its own.
- **Skip radiative cooling (Chain A step 3); build SF on the ISOTHERMAL path.**
  Isothermal gas is already an implicit heat bath — cooling's job is done for
  free, dense shocked lanes already form (that is literally the gasrich
  showpiece), and a density-threshold recipe converts them directly. A *visual*
  SF milestone needs dense-gas → star → age-color, not a calibrated cosmological
  SFR. Cooling stays a future step 3, unblocked by this.

Advisor-vetted (2026-07-21).

**STATUS — S1–S6 COMPLETE (2026-07-21).** Sim (S1–S4), age-color renderprep (S5),
and the `[physics.star_formation]` + `[look.age]` scenario wiring (S6) all shipped
and gated. The showpiece is a **separate `gasrich-sf` preset** (rho_thresh 0.5,
efficiency 0.5, chosen from a QUICK A/B — ~329 young stars over the natural merger;
age tint young=[0.7,0.8,1.0], strength 0.7, τ=6), leaving the flagship `gasrich`
smooth and SF-free (user call). GPU+SF is rejected loud; feedback and cooling stay
deferred.

**Open follow-up — smooth SF subframes.** S6 surfaced that SF is incompatible with
M6c Hermite subframe interpolation: in-place gas→star conversion GROWS the
species-routed splat set between snapshots, and `renderprep::subframe` requires a
fixed splat set per span (it pairs endpoint splat rows by index). The shipped fix
renders SF runs at **snapshot cadence** (`movie_frame_count` → one frame per
snapshot, direct endpoint emit — choppy, ~1 s at 60 fps for 61 frames). The proper
smooth fix is NOT id-matching (the particle keeps its index under in-place
conversion) but **full-N frames + a gas-skipping splat pass in the renderer** so the
subframe `!filtered` path runs on a constant-length frame — a real renderprep
milestone, unscheduled.

---

## The load-bearing choice: whole-particle IN-PLACE conversion

A forming star **reuses the gas particle's slot** — flip
`kind[i]: Gas → Collisionless`, stamp `formation_time[i]`, done. No new
particle is spawned. This is the choice that makes every gate cheap:

- **N and total mass are conserved EXACTLY, for free.** No mass/N
  reconciliation, no snapshot-count change, no renderprep / GPU-buffer resize
  cascade. Spawning new particles would touch all of that.
- Conversion is one-way (`Gas → Collisionless`, never back) and the formed star
  **keeps its progenitor tag** (4/5 — gas provenance survives; young/old is
  distinguished by `formation_time`, NOT by a new tag). Invent a "formed vs
  primordial" tag only if that specifically becomes a wanted color axis; it is
  not needed for age-coloring.
- The converted particle stops feeling hydro forces the instant its `kind`
  flips (the SPH gather keys on `Species::Gas`), which is physically right — a
  star is collisionless. Its `u` becomes inert (gravity-only rows carry no
  internal energy, the existing `u = 0.0` convention).

**Rejected:** spawning a new star particle per conversion (variable-N; a
snapshot-format, renderprep, and GPU-residency cascade for zero visual gain at
this scope) and sub-particle conversion / partial mass (fractional `Species` is
not representable and buys nothing for a visual).

---

## Key decisions (with rationale)

### F1 — a new `State.formation_time` column, added WITH conversion
Age-coloring is the whole payoff, and it needs the formation time per particle.
That is a genuine new state variable (like `u`: it has meaning only on some rows
and rides along on the rest), so it lives in `State` as an SoA column and bumps
the snapshot format version. **Add it in the conversion milestone, not a later
pass** — deferring it means two format-version bumps for one feature.
- **Sentinel `f64::NEG_INFINITY` = "no SF formation time"** (reconfirmed at F1
  over the tentative NaN) — every primordial star / DM / halo / disk particle
  and every still-gas particle carries `−∞` (they did not form via SF in this
  run). `0.0` is a *valid* formation time (a star born at `t = 0`), so it cannot
  be the sentinel. **NaN was rejected: `State` derives `PartialEq`, and a NaN
  column makes `State == State` always false** — poisoning every whole-`State`
  round-trip / byte-identity comparison across the codebase (the project's core
  gate discipline). `−∞` compares equal to itself and round-trips exactly.
  Age-coloring falls out with no `is_nan` branch: `age = now − (−∞) = +∞ ⇒
  exp(−age/τ) = 0 ⇒ base color`, which is exactly right — primordial stars ARE
  old. Exposed as `State::PRIMORDIAL` for concise literals.
- `from_phase_space` fills `vec![State::PRIMORDIAL; n]`; `assert_consistent`
  checks its length; `galaxy-io` round-trips it in a bumped schema version (v4;
  old v1–v3 snapshots read back `formation_time = PRIMORDIAL` — a clean default
  meaning exactly "primordial," so back-compat is semantically free).

### F2 — a pure, loop-agnostic SF operator
`star_formation::form_stars(state, rho, div_v, dt_elapsed, cfg, seed) →
FormationSummary` in a new `sim::star_formation` module (or `core` — decide at
F2; it mutates `State` in place and needs no I/O, so `core` is defensible, but
it is engine policy, so `sim` is the natural home). Pure w.r.t. its inputs:
given the same `(state, rho, div_v, dt_elapsed, cfg, seed)` it makes the same
conversions, order-independent. It:
1. Selects candidates: `kind[i] == Gas && rho[i] >= rho_thresh && div_v[i] < 0`
   (dense AND converging — the two-part standard SF criterion; the
   converging-flow gate `∇·v < 0` rejects gas that is expanding/shocked-through,
   not collapsing).
2. Draws a conversion probability `p_i = 1 − exp(−eff · dt_elapsed / t_ff(rho_i))`,
   `t_ff(ρ) = √(3π / (32 G ρ))` (local free-fall time), `eff` the dimensionless
   efficiency per free-fall time.
3. Converts iff a **deterministic** uniform draw `< p_i` (see F3): flips
   `kind[i] → Collisionless`, sets `formation_time[i] = state.time`.
- The `1 − exp(...)` form saturates correctly for a large `dt_elapsed` (fires at
  snapshot cadence, not per fine step — see F4), never exceeding probability 1.
- `FormationSummary { n_formed, mass_formed }` for run diagnostics / the
  monotonicity gate.

### F3 — determinism via a pure `(id, step, seed)` substream
The conversion draw is a pure function of `(particle id, formation step/epoch,
global seed)` via the project's **SplitMix64** discipline — NOT a shared RNG
advanced in iteration order. A shared, order-advanced RNG would break
reproducibility under `rayon` and under the active-subset iteration order of
`run_individual`. Same seed → same conversion set, independent of particle
ordering or thread scheduling. (The "epoch" is the SF-call index / snapshot
index, so successive calls draw independent streams for the same particle.)

### F4 — hook at the output-cadence SYNCHRONIZATION site
SF fires **once per snapshot interval, right before each snapshot emit**, with
`dt_elapsed` = sim-time since the previous SF call. That site is where `pos` and
`vel` are consistent across ALL THREE stepping loops (`run` / `run_adaptive` /
`run_individual` — the individual loop's own comment calls the output cadence
"the only place all rungs are synchronized"). One call site, identical semantics
on every path. Fixing SF to the snapshot cadence (coarse dt) is fine for a
visual — the `1 − exp` law handles the large dt — and it keeps SF off the hot
per-step / per-fine-tick path entirely.
- **Gate on fixed-dt `run` FIRST** (cleanest `dt_elapsed = snapshot_every · dt`),
  then wire the identical call into `run_adaptive` and `run_individual`.
- **SF is opt-in via `Option<StarFormationConfig>`.** `None` ⇒ the operator is
  never called ⇒ every existing byte-path is untouched (the byte-identity gate).

### F5 — the SF fields come from a D2-clean transient solver accessor
The recipe needs SPH gas density `ρ` and velocity divergence `∇·v` — both are
SPH gathers the solver already knows how to do. **D2 forbids STORING ρ in
`State`, not COMPUTING it**; a transient accessor that returns fresh `Vec`s is
D2-clean (exactly like `max_stable_dt_per_particle` / `coupled_pairs`). Add
`ForceSolver::sf_fields(&State) → SfFields { rho, div_v }` (both `Vec<f64>`,
length `n`; collisionless rows carry `ρ = 0`, `∇·v = 0`). Default returns zeros
(a pure-gravity solver has no gas → no SF); `GravitySph` overrides it, reusing
`density::density_adaptive` for ρ and adding the standard SPH divergence gather
`∇·v_i = (1/ρ_i) Σ_j m_j (v_j − v_i) · ∇_i W_ij` for `div_v` (new; the ρ half
already exists). `ρ = 0` rows can never pass the `ρ >= rho_thresh` gate, so the
zero-default is inert for SF, matching the trait's other "pure gravity is a
no-op" defaults.

### F6 — age-coloring in renderprep (swap the proxy signal)
`renderprep/src/coloring.rs` gets `age_colors(base, formation_time, now, young,
strength, τ)`: tint toward `young` (blue-white) for small age `a_i = now −
formation_time[i]`, ramp `t = strength · exp(−a_i / τ)` (fades a new star back
to base over ~`τ` sim-time), `formation_time == NaN → base exactly` (primordial
stays its color). Same two-product `lerp3` the other maps use, so **`strength =
0` (and every primordial/NaN row) is bit-identical to the no-tint render** — the
hard guarantee `compression_colors` / `dispersion_colors` carry. This is the
`stellar-nursery-glow` plumbing with the real signal: `now − formation_time`
instead of ambient-ρ. `now` is the snapshot time (in the frame data / header) —
view-INDEPENDENT (D9-safe), computed at prep time like every other map.

### F7 — `[physics.star_formation]` scenario knob + gasrich showpiece
A `[physics.star_formation]` scenario section (`rho_thresh`, `efficiency`,
`seed`) wired through xtask into the stepping config; absent ⇒ `None` ⇒
byte-identical. `[look]` gets the `age_colors` knob (young color, strength, τ).
gasrich A/B: render with SF off vs on at a couple of `(rho_thresh, efficiency)`
values, pick the ship values from the A/B (the `scatter` / `splat` /
`local-tone` A/B discipline), retain the A/B dir under `M:\claud_projects\temp`.
Feedback is explicitly OUT of this knob this milestone.

---

## Milestones (TDD: red test committed separately, then green)

- **S1 — `State.formation_time` column + snapshot format bump.** Red:
  `from_phase_space` NaN-fills; `assert_consistent` checks length; `galaxy-io`
  round-trips it under a bumped version; an old-version snapshot reads back
  `formation_time = NaN` (back-compat = "primordial"). (F1)
- **S2 — `form_stars` pure operator + recipe.** Red: SF-off (`efficiency = 0`)
  and empty-gas make no conversions; mass/N invariant across a conversion;
  determinism (same seed → same set, order-independent — shuffle the input,
  same conversions); one-way monotonicity (star count non-decreasing, no star
  ever reverts to gas); threshold selection (below `rho_thresh` never converts;
  `∇·v ≥ 0` never converts); and the statistical gate — a uniform-ρ,
  uniformly-converging box converts a fraction matching the analytic
  `p = 1 − exp(−eff·dt/t_ff)`, `t_ff = √(3π/32Gρ)`, to sampling tolerance.
  (F2, F3)
- **S3 — `ForceSolver::sf_fields` (ρ + ∇·v).** Red: default returns zeros;
  `GravitySph` ρ matches `reference_density`; `∇·v` matches a hand-built
  converging cloud (∇·v < 0) and a diverging cloud (∇·v > 0), sign-correct;
  collisionless rows carry `(0, 0)`. (F5)
- **S4 — driver hook (fixed-dt `run` first, then adaptive + individual).** Red:
  SF `None` ⇒ every snapshot byte-identical to the current pipeline (all three
  loops); SF `Some(cfg)` on a controlled seeded run converts the expected
  particles at the expected step, with `formation_time = state.time`; the
  identical call on `run_adaptive` / `run_individual` produces the same
  conversions given the same synced state. (F4)
- **S5 — `age_colors` renderprep map + `[look]` knob.** Red: `strength = 0`
  bit-identical to base; `formation_time == NaN → base` exactly; monotone +
  bounded in age; endpoint/reference exactness; knob-off frame-data byte
  identity. (F6)
- **S6 — `[physics.star_formation]` scenario wiring + gasrich A/B showpiece.**
  Red: knob absent ⇒ pipeline byte-identical; knob present ⇒ conversions happen
  and young stars tint. Then the QUICK/FULL A/B, pick + bake ship values.
  (F7)

---

## Gates (summary)

| Gate | What it asserts |
|---|---|
| Mass & N exact | whole-particle in-place flip conserves both, for free. |
| SF-off byte-identity | `Option = None` ⇒ every existing byte-path unchanged (all 3 loops + renderprep). |
| Determinism | same seed → same conversion set, order/thread-independent (SplitMix64 substream). |
| One-way monotonicity | star count non-decreasing; no star reverts to gas. |
| Threshold + converging-flow | below `rho_thresh` or `∇·v ≥ 0` ⇒ never converts. |
| Statistical calibration | uniform box converts at analytic `p = 1−exp(−eff·dt/t_ff)` to sampling tol. |
| `sf_fields` correctness | ρ vs `reference_density`; ∇·v sign-correct on hand-built converging/diverging clouds. |
| Age-color `strength = 0` | bit-identical to the no-tint render (two-product lerp). |
| View independence | age uses snapshot time only — no camera term at prep time (D9). |

---

## Risks / open items (named, not silently deferred)

- **GPU-resident path re-upload.** SF applied on the CPU at the snapshot sync
  point mutates `Species` + `formation_time`; the `--gpu` adaptive path reads
  back → (SF here) → re-uploads, and `reattach_columns` already warns "Species
  does not change without star formation." v1 gates SF on the **CPU** stepping
  paths (which is where gasrich full-res currently ships — `run_individual`
  hydro-only); making SF correct across the GPU-resident readback/re-upload
  boundary (flipped Species + formation_time must survive re-attach) is a
  tracked follow-up, not silently assumed working.
- **No feedback ⇒ runaway conversion is possible.** With no energy injected back
  into the gas, a dense clump can convert a large gas fraction quickly (nothing
  disperses it). For the visual this reads fine (stars light up where gas was
  dense); it is called out so the A/B `efficiency` is chosen to avoid a
  gas-annihilating runaway, and so feedback's absence is a conscious scope line,
  not an oversight.
- **Snapshot size.** `formation_time` adds 8 B/particle to every snapshot
  (~unchanged order of magnitude; ~62 → 70 B/particle). Acceptable; noted.
- **`∇·v` is a new SPH gather.** The ρ half is reused; the divergence half is
  new code (S3). If it proves fiddly, v1 can ship density-threshold-only and add
  converging-flow as a fast-follow — but the gate list assumes both, so build
  both unless a real obstacle appears.

---

## Relationships

`long-burning-beacon.md` (Chain A step 4 — this IS that milestone; cooling step 3
consciously leapfrogged), `stellar-nursery-glow.md` (the visual proxy this
supersedes with real physics — its `coloring.rs` plumbing is what F6 reuses),
`smoldering-thermal-ledger.md` (the energy-equation series `u` column F1
parallels; feedback, deferred here, injects into that `u`),
`laddered-ember-cadence.md` (the `run_individual` sync-site F4 hooks into),
`courant-quickening-cadence.md` (the `run_adaptive` sync-site F4 hooks into),
[[m7-sph-volumetrics-series]] / [[energy-equation-series]] (the SPH gas SF
operates on), [[temp-color-series]] (the other post-energy-equation visual;
orthogonal).
