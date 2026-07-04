# M7 review follow-ups — deferred perf/memory work

Written 2026-07-04, after the M7 series review. The review produced 7 findings.
Three landed immediately (see below); F7 was **rejected** (not worth widening
`ic`'s public API for a non-gating demo). This doc scopes the **three remaining
findings — F1, F2, F4 — for future separate sessions.** They are performance and
memory refactors, not correctness bugs; nothing here blocks anything, and none is
urgent at the current default (QUICK) scale.

## Already landed this batch (context, do NOT redo)

- **F3 (correctness):** `max_stable_dt` CFL now gathers at `2·h_max` and gates each
  pair on the true force-coupling range `r < 2·max(h_i,h_j)`, mirroring
  `hydro_accelerations`. Commits `8517dd9` [red] → `07639e5` [green], docs `8f0849c`.
- **F5:** dropped the redundant explicit `validate_dt` at t=0 in `simulate_snapshots`
  (the `CflGuard` already validates the t=0 IC before the first `.snap` write).
  Commit `299ac23`.
- **F6:** pinned `GasLookValues::default` == `galaxy_render::GasLook::default` with a
  cross-crate field-by-field test. Commit `122711c`.
- **F7 (REJECTED):** dedup the third hand-rolled SplitMix64 (`xtask/src/main.rs`
  `splitmix_stream`, also `ic/src/disk.rs`, Plummer). Rejected: it would widen `ic`'s
  public API for a non-gating demo whose determinism rides on the exact `[0,1)`
  mapping (`>>11 / 2^53`), needing a byte-identical pin test whose cost approaches the
  value. Leave as-is. If ever revisited, prefer a tiny shared internal util over
  exposing `ic::SplitMix64`.

---

## ⚠ Before doing ANY of F1/F2/F4: measure full-res first

All three findings bite specifically on the **full-res, edge-on `gasrich` showpiece
path** (user-gated), not on the QUICK default (64³, ~2500 gas, ~5 min). Full-res
wall-clock has **never been measured** (see the `GPU SPH gate` note in the
`m7-sph-volumetrics-series` memory and DESIGN.md).

**First action of the follow-up session: time one full-res `gasrich` render+sim**
(sim seconds AND render seconds separately). That single measurement decides:

- Whether **F1** is worth a milestone-sized rewrite, or whether the runtime already
  justifies pulling **GPU SPH** forward wholesale (which would make F1 moot — don't
  hand-optimize a CPU loop you're about to replace). This is the existing
  `GPU SPH gate`: measured full-res merger wall-clock > ~30 min ⇒ insert the GPU-SPH
  session; else it's an M8 opener.
- Whether **F2/F4** sit on the critical path (if voxelization + march dominate, the
  texture re-upload F2 targets may be a rounding error; if sim dominates, F2/F4 are
  premature vs F1/GPU-SPH).

Do not start code until this number exists.

---

## F1 — hydro force gather goes O(N²) under adaptive h  *(milestone-sized)*

**File:** `solvers/src/sph/forces.rs` (~line 96–119, `hydro_accelerations`).

**The finding.** The hydro force gathers neighbors at the **global** radius
`SUPPORT·h_max` per target. The comment there already concedes this: under a wide
adaptive-h range (a centrally-concentrated gas disk/merger) `h_max` is a
far-outskirts tail value, so every target's candidate ball covers most of the gas →
~O(N²) kernel work per step. This is the **per-step production path** (`GravitySph`
in `simulate_snapshots`, ~6000 steps per merger), so it dominates full-res sim
wall-clock. The deferral rationale in the comment ("the M7b shock tube has
near-uniform h") was overtaken by M7c wiring it into every step of every gas movie.

**Why the global radius is load-bearing (do NOT naively shrink it).** The averaged
kernel `W̄ = ½(W(h_i)+W(h_j))` is nonzero for `r < 2·max(h_i,h_j)`, so a pair with
`2·h_i < r < 2·h_j` contributes force to BOTH i and j. Querying only `2·h_i` gives
i's force to j but not j's to i — Newton's third law and momentum conservation
break. The radius is physics, not a "don't miss a neighbor" convenience.

**The fix (already named in the code comment; advisor-vetted template).** Port the
**M7d scatter-by-plane** pattern used in `renderprep` gas deposition: gather at
`2·h_i`, then SCATTER each pair's contribution to both i and j over the `h_j`-reach,
in **ascending particle index**. Dropped far terms are exact `+0.0`, so order and
bits are unchanged.

**The hard part / risk (highest in the review set).** This rewrites the core force
loop and MUST preserve, bit-for-bit:
- Equal-mass pair antisymmetry (`a_i` pair term is the exact negation of j's — coeff
  bit-identical by commutativity, `grad_avg` exactly negated) ⇒ linear + angular
  momentum conserved to roundoff.
- `rayon ≡ serial` determinism (fixed ascending gather/scatter order).

A scatter path with parallel accumulation is where determinism is easiest to lose —
the M7d deposition solved exactly this (scatter-by-plane in ascending index, not
gather-per-cell). Reuse that discipline; do not invent a new accumulation scheme.

**TDD gates.**
1. Bit-exact equality: new scatter accelerations == old gather accelerations on a
   centrally-concentrated cloud (wide h range) — same bits, faster.
2. Existing momentum/antisymmetry proptests stay green.
3. `rayon ≡ serial` bit-exact gate (mirror the existing density/forces gates).
4. Re-run the isothermal shock tube (`--release --ignored`) — L1(ρ), L1(u), star
   state must match the analytic Riemann solution as before.
5. A timing gate/probe showing the sub-quadratic scaling on an N-prefix sweep
   (mirror `xtask sph-demo`'s grid-vs-brute timing harness).

**Note on F1 vs F3 (do not conflate).** F3 (landed) changed only the CFL *sentinel*
to see the same coupling range. F1 changes the *force* computation. The physical
coupling range stays `2·max(h_i,h_j)` in both; F1 only reorganizes the compute. F3
does not depend on F1 and vice versa.

**Effort:** own milestone (red/green + shock-tube re-gate). **Payoff:** unmeasured —
gate it on the full-res timing above; may be superseded by GPU-SPH.

---

## F2 — endpoint gas textures re-uploaded every subframe  *(contained perf win)*

**File:** `render/src/render.rs` (~line 934, `render_frame_with_gas` /
`upload_grid`).

**The finding.** Both endpoint gas density grids are re-created as fresh 3D textures
and re-uploaded on **every** `render_frame_with_gas` call, though the same grid pair
serves all subframes of a movie window (only `mix` changes between subframes). At
128³, each `R32Float` grid is 8.4 MB; ~10 subframes/window × ~60 windows × 2 grids ≈
~10 GB of redundant texture creation + PCIe transfer per full-res movie.

**The fix.** Cache the two `TextureView`s in the **persistent `Renderer`**, keyed on
window (or grid identity). Re-upload only when the window advances (grid pair
changes); reuse across the window's subframes. Cuts gas uploads by the subframe
factor (~8–10×).

**Risk / cons.**
- Renderer-internal state only — no format/wire/determinism implications; the golden
  gates still pin correctness. Contained blast radius.
- A stale-key bug would silently render the wrong window's gas — the invalidation key
  must be correct (window index or a grid identity/generation counter). Add a test
  that advancing the window swaps the bound textures.
- Only helps the full-res movie path; noise on a single-frame render or QUICK.
- The "10 GB" is spread over a multi-minute render — verify (via the full-res timing)
  that texture transfer is actually on the critical path before spending effort; the
  CPU voxelization and the march likely dominate.

**Effort:** medium, self-contained. **Cleanest risk/reward of the three** — the pick
if you just want a safe banked win and the timing shows transfer matters.

---

## F4 — run_movie holds ALL snapshot gas grids in memory  *(memory, speculative)*

**File:** `xtask/src/main.rs` (~line 1007, `run_movie`, the `gas_grids` collect).

**The finding.** `run_movie` eagerly voxelizes and holds **all** snapshot gas grids
for the entire render (`states.iter().map(deposit_gas).collect()`), though the render
loop only ever needs the current window's two endpoint grids. gasrich FULL: 61
snapshots × 8.4 MB (128³ f32) ≈ 0.5 GB held for the whole render, on top of all
states and all prepared frames. Scales linearly with `n_steps / snapshot_every`.

**The fix.** Sliding two-grid window: deposit grid `w+1` as each window begins, drop
`w-1`. Keeps peak gas memory at ~17 MB regardless of snapshot count, with no change
in output.

**Risk / cons.**
- It's **memory, not speed** — makes nothing faster; 0.5 GB is not a problem on the
  current machine today. Only matters if snapshot count or resolution grows well past
  current (e.g. 256³, or much longer runs) — i.e. **speculative until you actually
  hit the ceiling.**
- More invasive than F2: it interleaves voxelization with the render loop, muddying
  the current clean "voxelize all, then render" phase separation — and loses the
  single-batch timing diagnostic (`voxelized gas (…³) for N snapshots in X s`).
- Lowest urgency of the perf trio.

**Effort:** medium, touches render-loop control flow. **Do only when a real memory
ceiling is hit**, or fold in opportunistically if F1/F2 already reshape this loop.

---

## Suggested session ordering

1. **Measure** full-res `gasrich` sim + render wall-clock (blocks everything; also
   feeds the standing GPU-SPH gate).
2. If sim dominates and > ~30 min → **GPU-SPH session** (F1 likely moot). Else if a
   CPU win is still wanted → **F1** (its own milestone, full TDD + shock-tube re-gate).
3. If render dominates and texture transfer is on the critical path → **F2** (safe,
   contained).
4. **F4** only when a memory ceiling is actually hit, or fold into F2/F1 if the loop
   is already being reshaped.

Related: `deep-orbiting-sunbeam.md` (M7 per-series plan), `long-burning-beacon.md`
(long-horizon ordering, incl. the GPU-SPH gate), DESIGN.md (SPH + CFL rationale).
