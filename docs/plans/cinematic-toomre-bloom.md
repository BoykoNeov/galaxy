# M6 — "the beautiful": visual / cinematic upgrade series (roadmap)

A session-by-session plan to take the movies from *physics diagnostic* to
*cinematic*, while preserving the pipeline's decoupling contracts (Contract 3
frame-data; retained linear-HDR EXR) and the house TDD discipline. Each session
is one milestone, independently demoable by a regenerated movie.

## Context — what the pipeline does today (audit, 2026-07)

The architecture supports far more than the movies currently use:

- **Camera:** one static orthographic `Camera::face_on` per movie, framed once
  over the whole run (`xtask/src/main.rs:440`). The view axis is already a
  parameter (built for orbit views, `render/src/camera.rs:36`) — never used.
- **Splats:** constant size per scenario (`PrepConfig.size`), Gaussian falloff
  `k=6`, flat progenitor palette. `FrameData.size` is per-particle in the
  contract but filled with a constant (`renderprep/src/prepare.rs:75`).
- **Density boost:** `DensityColoring` (kNN, M3.6) is landed and invariant-gated
  but **all three scenarios set `density: None`** — the M3.6 scope note says it
  stays off "until the mapping is tuned against a rendered collision frame."
- **No bloom.** DESIGN's rendering recipe names it ("mip down/blur/up"); not
  implemented anywhere.
- **Grade:** exposure × {ACES, Reinhard} only; every scenario uses exposure 1.0
  + ACES. EXRs are retained precisely so regrading is cheap — never exploited.
- **Motion:** ~61 snapshots → ~61 frames at 30 fps = a ~2-second flipbook.
  Snapshots store full `vel`, which makes physically-informed interpolation
  free — unused.
- **Coloring:** progenitor palette only. `prepare.rs` names velocity-dispersion
  coloring as "the next deferred refinement."
- **Scenarios:** all three (`disk`, `dm`, `cuspy`) are the same geometry —
  coplanar prograde parabolic encounter. The Toomre & Toomre zoo (retrograde,
  inclined, ring/bullseye, minor merger) is untouched, and DESIGN already names
  the `scenario.toml` front-end as deferred.

DESIGN's deferred-visuals list (M3 entry): bloom, kNN density/velocity-dispersion
coloring, Blender consumer, multi-camera/orbit views, `scenario.toml` front-end.
This plan sequences all of those except the Blender consumer, plus temporal
upsampling and the scenario zoo.

## Ground rules (every session)

1. **TDD per CLAUDE.md**: red tests committed separately (`[red]`, workspace
   must still build), then implementation; never weaken a test to pass.
2. **Unit-gate the math, eyeball the aesthetics** (the M3.6 precedent): pure
   functions (tone curves, kernels, interpolants, camera paths, color ramps)
   get exact/invariant gates; "the movie looks better" is an eyeballed QUICK
   render, and the chosen tuning constants get documented in DESIGN.
3. **Demo criterion**: every session ends with at least a regenerated
   `GALAXY_MOVIE_QUICK=1` cuspy movie (temp output under `M:\claud_projects\temp`),
   full-res re-render when the payoff warrants it.
4. **Contracts stay intact**: frame-data (Contract 3) remains the
   renderer-agnostic boundary; the EXR stays the pristine *linear* artifact so
   grade-time iteration stays seconds-cheap. Schema changes are versioned and
   deliberate.
5. **End-of-batch ritual**: DESIGN.md milestone entry + memory update + quality
   gate (`cargo test`, `clippy -D warnings`, `fmt --check`) + commit AND push.

## Session map

| Session | Milestone | One-liner | Effort | Depends on |
|---|---|---|---|---|
| 1 | **M6a** | Grade toolkit: asinh stretch, `regrade` subcommand, density boost ON | S | — |
| 2 | **M6b** | Bloom (CPU mip-chain, applied at grade time in linear space) | M | M6a (regrade loop) |
| 3 | **M6c** | Hermite temporal upsampling → smooth 60 fps movies | M | — |
| 4 | **M6d** | Camera rig: smoothed framing + orbit/tilt paths | M | best after M6c |
| 5 | **M6e** | Coloring modes v2: initial-radius ramp, σ_v, size-by-density | M | — |
| 6 | **M6f** | `scenario.toml` front-end + the Toomre encounter zoo | M–L | easier after M6a–e |
| 7 | **M6g** | Perspective camera + world-space vertex-shader projection | L | optional, last |

M6a→M6b are the look; M6c→M6d are the motion; M6e is the information content;
M6f multiplies subjects; M6g is the big-lift render upgrade (also the named
"10⁸ swap" in `render.rs`). M6c/M6e can be reordered freely; M6d wants M6c
first (an animated camera over 2 seconds of flipbook just amplifies the
choppiness).

---

## M6a — grade toolkit + the switched-off wins (Session 1, S)

**Goal:** exploit what's already built. Establish the fast iterate-on-look loop
(regrade EXRs in seconds) that every later session leans on.

Scope:
- **`ToneMap::Asinh`** in `grade`: the astro-imaging stretch (Lupton-style
  `asinh(x/β)` normalization, β the softening knob) — reveals faint tidal tails
  without blowing out the additive cores; ACES/Reinhard can't do both at once.
- **`regrade` xtask subcommand**: `xtask regrade <exr_dir> <png_dir>
  [--exposure E --tonemap aces|reinhard|asinh --beta B]` → PNGs + optional
  ffmpeg. The binary stays test-exempt I/O glue; any new pure helper (arg →
  `GradeConfig` mapping) is unit-tested in `xtask/lib.rs`.
- **Turn `DensityColoring` on** in all three scenarios: tune `(k, strength,
  softening)` against rendered QUICK frames (the exact follow-up the M3.6 scope
  note promised). Document chosen values in DESIGN.
- Optional if time: exposure sweep helper (render one frame at 3–5 exposures
  into a contact sheet).

Red-first gates: asinh hand values; monotonicity; `[0,1]` range; β→large
recovers ~linear at small x; highlight compression ordering asinh < Reinhard at
large x. (Density math is already gated; its *tuning* is eyeballed by rule 2.)

Demo: cuspy movie regraded with asinh + density boost — tails visibly brighter
without core clipping, zero re-simulation.

## M6b — bloom (Session 2, M)

**Goal:** the signature emissive-star-field look — cores halo out, bright knots
glow. DESIGN names the technique: mip down/blur/up.

**Placement decision (settle at session start, document in DESIGN):** DESIGN's
recipe puts bloom in the render stage *before* the EXR write. Recommendation:
deviate deliberately — keep the EXR **pre-bloom** (the pristine linear
artifact) and apply bloom **at grade time, in linear space, before the tone
curve**. Mathematically identical (both are linear-domain pre-tonemap), but
bloom-radius/strength iteration stays seconds-cheap via `regrade` instead of a
re-render. Bloom then lives in `grade` (which deliberately has no wgpu dep) as
a pure CPU function; a GPU port is a named perf follow-up only if CPU ever
hurts (it won't at 720p–1080p: separable Gaussians on a mip pyramid are cheap).

Scope:
- `bloom(pixels, w, h, &BloomConfig) -> pixels`: bright-mask-free HDR bloom
  (recommend **no threshold** for the linear astro look — flag as a decision),
  N-level mip pyramid, separable Gaussian per level, weighted up-add, final mix
  `out = img + strength · halo`.
- Wire into `grade_file`/`regrade` behind `BloomConfig { strength, levels,
  radius }`, `strength = 0` ⇒ bit-exact identity.

Red-first gates: strength=0 identity (bit-exact); linearity
(`bloom(a·img) = a·bloom(img)`); kernel flux preservation (normalized blur
conserves total flux to fp tolerance, so the mix adds exactly
`strength·flux`); impulse response → radially symmetric, monotone-decaying
halo; odd-dimension center-pixel symmetry (the house downsample gate);
interior translation equivariance; determinism.

Demo: before/after regrade of the retained cuspy EXRs; pick default strength
per scenario by eyeball; document.

## M6c — Hermite temporal upsampling (Session 3, M)

**Goal:** kill the flipbook. Snapshots store `pos` *and* `vel`, so cubic
Hermite interpolation between adjacent snapshots gives physically-informed
in-between frames at zero sim cost: ~61 snapshots → ~480+ frames, 60 fps,
longer runtime.

Scope:
- `renderprep::interp` (view-side concern — core stays pure physics):
  `hermite(s0, s1, dt, u)` → interpolated positions (and velocities, from the
  cubic's derivative — needed later for Doppler coloring in M6e).
- **Attribute strategy (decision, recommend):** run full `prepare` (including
  kNN density) only on the two endpoint snapshots; per subframe, take Hermite
  positions + linearly interpolated brightness/color/size. Re-running O(N²)
  kNN per subframe would be minutes-to-hours at N≈27k × ~500 frames; density
  evolves on the snapshot timescale anyway.
- Defensive gate at runtime: assert `id` arrays of s0/s1 are identical before
  interpolating (order-stability is expected from the in-place integrator, but
  assert it — a silent mismatch would scramble the movie).
- xtask: `subframes_per_snapshot` (≈8) per scenario, FPS 30→60.

Red-first gates: u=0 / u=1 reproduce endpoints bit-exact; constant-velocity
motion exact; cubic polynomial trajectories reproduced exactly (Hermite's
order); C¹ continuity across snapshot boundaries (velocity match at the
joins); **Kepler oracle** — two-body analytic orbit sampled coarsely, Hermite
in-betweens vs the analytic positions, tolerance justified by the O(Δt⁴) local
error of cubic Hermite at the chosen snapshot cadence.

Demo: cuspy movie at 60 fps — pericenter passage visibly continuous.

## M6d — camera rig: smoothed framing + orbit/tilt (Session 4, M)

**Goal:** replace the single static framing with a smooth, animated camera —
the difference between a diagram and a film. `Camera` already supports any
view axis; this session adds *time*.

Scope (in `render`, pure math; xtask wires per-scenario):
- **Smoothed framing envelope**: per-frame percentile radii → a temporally
  smoothed radius r(t) (e.g. moving-max envelope + Gaussian smoothing) so the
  zoom breathes with the action (tight at approach, widening as tails fling
  out) without per-frame jitter.
- **`CameraPath`**: time-parameterized `target` / view axis / half-extent.
  Procedural primitives first (no keyframe file format yet): slow azimuthal
  orbit θ(t) with ease-in-out; tilt from face-on toward a ¾ inclination;
  static path preserved as the back-compat default.
- Per-scenario choreography, e.g. cuspy: start ¾-inclined, slow orbit through
  first pericenter, gentle zoom-out as the tails extend.

Red-first gates: orthonormal camera basis at every sample; aspect preserved;
smoothness bound (|Δr|, |Δθ| per frame under explicit limits); ease endpoints
and zero end-derivatives; envelope ≥ the raw per-frame requirement it smooths
(never crops tighter than the percentile framing asked for); static path ≡
today's camera bit-exact; determinism.

Demo: cuspy at 60 fps with orbit + breathing zoom; an edge-on/¾ segment that
shows the 3-D structure face-on flattens away.

## M6e — coloring modes v2 (Session 5, M)

**Goal:** diversify what the colors *mean*. `prepare` grows a mode enum
(default = today's progenitor palette, bit-compatible).

Scope, in recommended order:
- **Initial-radius ramp** (the pure-visual winner): from snapshot 0, compute
  each particle's radius about its progenitor's COM, normalize per progenitor
  (e.g. by its half-mass radius), map through a per-progenitor color ramp
  (inner = warm/old → outer = blue/young, like real disks). Colors are frozen
  at t=0 and keyed by particle index for all later frames — tails then carry a
  visible gradient showing where the material came from.
- **Velocity-dispersion coloring** (the named-in-code deferred refinement):
  extend `knn_density` to also return neighbour **indices** (keep the existing
  d_k path bit-identical — gate it); σ_v = mean-subtracted dispersion over the
  neighbour set → color/brightness ramp. Same O(N²) reference-oracle stance as
  M3.6; tree acceleration stays the named follow-up.
- **Size-by-density** (small): feed the existing density estimate into
  per-particle `size` — tight cores, soft diffuse splats.
- **Star-formation proxy — density→hue** (the astrophysically-motivated one):
  we currently modulate only *brightness* by density (M3.6); this adds a *hue*
  shift toward blue-white in overdense regions. Physical rationale: compressed
  gas forms new stars, and young populations are dominated by short-lived blue
  OB stars — so starburst knots in bridges/tails and shocked overlap regions
  glow blue against the older, redder ambient population. Honest caveat (goes
  in DESIGN): the sim is collisionless — there is no gas — so local *stellar*
  density is a **proxy** for star formation, a standard visualization stand-in,
  not physics. Implementation rides the existing kNN estimate: hue lerp from
  the progenitor base color toward a "young population" blue-white, driven by
  the same mean-referenced, non-dimming-style bounded mapping as the M3.6
  brightness boost (dense → bluer; underdense → base color untouched, so the
  red/old tails stay progenitor-colored).
- **Doppler hue** (optional, may defer): hue shift by line-of-sight velocity.
  Caveat to settle honestly: it couples frame-data to a view axis, which
  Contract 3 deliberately avoids — if built, `prepare` takes a *declared* LOS
  and the coupling is documented (or vel joins FrameData in a schema v2).

Red-first gates: ramp monotonicity and endpoint colors; per-progenitor
normalization (median particle ≈ ramp midpoint); colors bit-stable across
frames (frozen-at-t0 property); knn index/d_k consistency (index i's distance
equals d_k; existing density outputs unchanged bit-for-bit); σ_v hand oracles
(two-population lattice); density-hue bounded on the [base color, young-blue]
segment, monotone in density, exactly base color for underdense particles and
at strength=0 (mirroring the M3.6 non-dimming discipline); mode=Progenitor ≡
today bit-exact; determinism.

Demo: one cuspy sim (no re-sim — snapshots retained) rendered in 2–3 modes
side by side.

## M6f — `scenario.toml` front-end + the Toomre zoo (Session 6, M–L)

**Goal:** diversify the *subject*. All current movies are coplanar prograde
parabolic encounters; the classic encounter zoo is mostly config once
scenarios are data. DESIGN already names the toml front-end as the next step
("so these hardcoded scenarios become data").

Scope:
- **`scenario.toml`** (xtask-only `serde`/`toml` dep): schema mirroring the
  `Scenario` struct + IC parameters; the three hardcoded scenarios become
  checked-in presets. Gate: parsing each preset reproduces the hardcoded
  constructor's `Scenario` (pure comparison in `xtask/lib.rs`) — same params,
  same seed ⇒ same movie.
- **The zoo** (each: QUICK eyeball → full render → documented expectation):
  - **Retrograde** twin of `cuspy` (flip one disk's spin): the control
    experiment — tails suppressed, the "why prograde matters" pair.
  - **Inclined** encounter (~45°): warped, genuinely 3-D tails (pairs with the
    M6d orbit camera).
  - **Bullseye / ring** (near-radial passage through the disk center along its
    spin axis — Cartwheel): expanding ring density wave. Verify the existing
    spin-orbit orientation surface covers an out-of-plane disk; if not, that
    extension is the one TDD-red IC change of the session.
  - **Minor merger** (1:10): satellite disruption → long-lived stream.
- Physics cautions carried over: cusp-resolution rule (M5f) applies to every
  cuspy variant — halo N and ε budgeted per scenario, QUICK kept honest.

Demo: the zoo contact sheet — four new movies + a DESIGN scenario table.

## M6g — perspective camera + world-space vertex projection (Session 7, L, optional)

**Goal:** the render-architecture upgrade `render.rs` already names ("the
world-space vertex-shader path is the 10⁸ swap"), plus perspective for
inside-the-scene drama.

Scope:
- Instance buffer carries **world-space** pos/size; view-projection matrix as
  a uniform; projection moves into the vertex shader (removes the per-frame
  CPU projection loop — required anyway before 10⁷–10⁸ particles).
- Perspective `Camera` variant; decisions to settle: brightness attenuation
  (physical 1/d² vs none — additive flux semantics change), splat size ∝ 1/d
  with near/far clamps, near-plane handling for particles behind the camera.
- Ortho stays default; every existing movie must be reproducible.

Red-first gates: GPU vertex-path ortho render ≡ CPU-projected path on the
existing test scenes (flux/pixel tolerance, same discipline as the landed
render gates); perspective projection vs hand-computed reference points;
behind-camera culling; degenerate-depth safety.

Demo: slow perspective dolly toward/through the merger remnant.

## Explicitly out of scope (this series)

- **Blender consumer** (deferred in DESIGN; frame-data contract already
  supports it — a hero-shot session for later).
- **Gas / SPH volumetrics** (Stage 5; ordered raymarch compositing — a
  different render model, explicitly not the splat path).
- **GPU bloom / kNN tree acceleration** — named perf follow-ups, gated on
  actual pain, oracles already in place per house style.
- **HDR video encode** (HDR10/PQ mp4) — the EXRs make it possible someday;
  not a current target.
