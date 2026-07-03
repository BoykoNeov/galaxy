# M7 — "the breathing": isothermal SPH gas + volumetric dust-lane render series

## Context

M6 completed the visual series for the collisionless engine. The one render
model DESIGN still owes is Stage 5: **gas**, composited by **volumetric
raymarching with absorption** (the ordered "over" operator — explicitly NOT
the additive splat path). Gas requires gas physics to exist at all, so this
series lands both halves, physics first:

- **v1 gas physics = isothermal SPH** (user-locked): fixed sound speed c_s,
  density summation, P = c_s²ρ pressure gradient, Monaghan artificial
  viscosity. NO energy equation (adiabatic/cooling deferred).
- **v1 compositing = full star attenuation** (user-locked): every star splat
  dimmed by exp(−τ) of the gas between it and the camera, gas itself
  raymarched emission+absorption — dark dust lanes silhouetted against the
  stellar cores, the iconic merger look.

Six sessions (M7a–M7f), each independently demoable, red-first per CLAUDE.md.
Scale target: CPU SPH + rayon at ~10⁴ (QUICK) to ~10⁵–2·10⁵ gas particles
(full). NOT the 10⁸ path; GPU SPH stays a named follow-up.

## Audit — what exists today (verified 2026-07)

- **No gas anywhere.** `State` (core/src/state.rs) = pos/vel (f64), mass
  (f64), id, progenitor (u16, saturated 0–3 by `DiskCollision`), time, a;
  derives `PartialEq` (xtask/tests/scenario_build.rs asserts `State`
  equality — NaN sentinels would silently break those gates). Struct-literal
  construction at ~9 sites.
- **One force hook.** `ForceSolver::accelerations(&mut self, &State, &mut
  [DVec3])` — takes `&State` (immutable), so a solver can never write
  per-particle quantities back. `LeapfrogKdk` calls it once per KDK step at
  post-drift positions with v_(n+1/2) velocities. Fixed global dt.
- **`sim::run` returns `Result<RunSummary, SimError>`**; `SnapshotSink::emit`
  returns `Result<(), SimError>` — an error-propagating hook at snapshot
  cadence already exists (the CFL sentinel rides it, no trait widening).
- **No neighbor range-query.** renderprep's `knn_density`/`knn_neighbourhood`
  are the O(N²) reference oracles; FlatTree/Lbvh have AABBs but no range API.
- **Snapshot v1 closed** (io/src/snapshot.rs, exact-match version check) —
  gas needs a deliberate v2. **Frame-data v1 stars-only** (GLXYFRAM,
  per-particle f32 SoA); DESIGN's Contract-3 sketch already anticipates "or
  density grid for gas". `run_movie` holds FrameData in memory (writes only
  EXR/PNG), so per-snapshot gas grids are RAM objects on the movie path.
- **Render (post-M6g)**: single additive splat pass, world-space instances,
  projection in the vertex shader, ortho golden-gated + perspective 1/d²
  gated, Rgba32Float + FLOAT32_BLENDABLE, no depth buffer. Camera/rig/
  grade/EXR reusable unchanged; no per-pixel ray generator yet.
- **ICs**: `ExponentialDisk<H>` (live halo + disk, warm via `with_toomre_q`);
  THREE PRNG streams per galaxy (seed, mix, mix²), `DiskCollision` galaxy 2
  at mix³ — the spacing invariant must survive a gas component.
- **renderprep depends only on galaxy-core** — sharing the SPH toolkit means
  one new dep edge (renderprep → galaxy-solvers).

## Ground rules (every session)

1. **TDD**: red tests committed separately (`[red]`, workspace compiles via
   `todo!()`), then implementation; tolerances justified by the method's
   order.
2. **Oracle discipline**: every fast path gated against an independent
   reference — O(N²) neighbor/density oracles, analytic solutions (isothermal
   Riemann, uniform-slab transmittance), retained golden gates for
   "gas-off ≡ old renderer".
3. **Unit-gate the math, eyeball the aesthetics** (M3.6/M6 precedent); chosen
   look knobs documented in DESIGN.
4. **Demo per session**; QUICK movies under `M:\claud_projects\temp`.
5. **Contracts versioned**: snapshot v2 and frame-data v2 are explicit bumps
   with v1 read-compat gates; EXR stays the pristine linear artifact (bloom
   stays at grade time, applies to the star+gas composite for free).
6. **Equilibrium-IC discipline**: every new IC must evolve-and-stay-put under
   the real solver stack (virial/moment checks alone insufficient).
7. **End-of-batch ritual**: DESIGN.md entry + memory + quality gate
   (`cargo test`, `clippy -D warnings`, `fmt --check`) + commit AND push.

## Design decisions (argued once, referenced per session)

**D1 — `State` gains ONE column: `kind: Vec<Species>`.**
`#[repr(u8)] enum Species { Collisionless, Gas }` in core. `progenitor`
stays a pure identity tag; gas gets NEW tags (DiskCollision: gas1=4, gas2=5)
so the LookSpec palette-length validation stays uniform and the debug
gas-as-splats mode has colors. Routing to the volumetric path keys on
`kind`, never on progenitor. ρ is never stored (recomputed each force call;
re-derived by grid deposition in renderprep).

**D2 — Smoothing length h is DERIVED, never stored.** The draft alternative
(h as a State/snapshot column) has a hard flaw: `accelerations` takes
`&State`, so the solver cannot write the adaptively-updated h back — the
stored column would go stale after step 1 and renderprep would deposit with
wrong h. Instead h is a deterministic function of positions: adaptive
bisection to a fixed tolerance on kernel-weighted neighbor count
(N_ngb ≈ 48; cubic-spline pairing guard: keep < ~57). The SPH solver
warm-starts its internal h across steps (bracket hint only — the converged
value is position-determined); renderprep recomputes h per snapshot via the
same shared routine (cheap with the hash grid at 2·10⁵). Pairwise forces
symmetrized via the kernel average W̄ = ½(W(h_i)+W(h_j)) → momentum
conservation exact. Grad-h (Springel–Hernquist) terms: named deferral.

**D3 — Snapshot v2 = one appended `kind[n] (u8)` column,** backward-readable:
reader accepts {1, 2}, v1 fills `Collisionless`; writer always emits v2;
garbage versions still rejected. Retained zoo snapshots stay re-preppable.

**D4 — SPH plugs in as a composite `ForceSolver`; the trait does not widen.**
`GravitySph<G: ForceSolver>` in solvers: gravity over ALL particles (gas is
just mass to BarnesHut; shared Plummer ε — softening/smoothing deliberately
decoupled in v1, documented) + hydro acceleration added to gas rows.
`&mut self` lets the SPH component recompute ρ/h internally — exactly once
per KDK step, at post-drift positions. Viscosity uses the velocities present
at that call (v_(n+1/2)) — the standard Gadget-style treatment; first-order
error in the viscous term only, invisible to momentum gates (pairwise force
stays antisymmetric). `potential_energy` delegates to the wrapped gravity
solver. **Total energy is NOT a gate for isothermal runs** (implicit heat
bath); the conservation gates are linear/angular momentum (exact to
roundoff) and the shock-tube oracle.

**D5 — The whole SPH toolkit lives in `solvers/src/sph/`:** cubic-spline M4
kernel (support 2h, analytic gradient), uniform hash grid (cell = 2·h_max,
counting-sort layout, gather per target in ascending neighbor index →
bit-exact parallel↔serial), adaptive-h density, O(N²) `reference_neighbours`
/ `reference_density` oracles. renderprep gains a `galaxy-solvers` dep (one
edge; single source of truth for kernel + neighbors + h, the `reference_*`
pattern). Tree range-query on FlatTree/Lbvh rejected for v1 — the hash grid
is O(N), simpler, and the SPH standard.

**D6 — Timestep: fixed global dt + fail-loud CFL sentinel, via the sink.**
No adaptive stepping in v1. `solvers::sph::validate_dt(state, params, dt)`
checks dt ≤ C_cfl · min_i h_i / v_sig,i (Monaghan signal velocity,
C_cfl ≈ 0.25). Called at t=0 before `run`, and at snapshot cadence by a
`CflGuard<S: SnapshotSink>` decorator whose `emit` returns
`Err(SimError::Config(...))`-style typed failure — the run dies loudly
instead of silently exploding, and neither `sim::run` nor any trait changes.
QUICK keeps the same dt (larger h at lower N makes the bound easier).

**D7 — Gas disk IC: isothermal vertical structure, pressure-corrected
rotation, salted seed domain.** `GasDisk` (ic/src/gas_disk.rs), composable
into `ExponentialDisk`/`DiskCollision`: exponential Σ_g with `gas_fraction`
splitting the disk mass (total rotation curve unchanged); vertical profile
sech²(z/z₀), z₀ = c_s²/(πGΣ_g) (documented as approximate in the
halo-dominated potential — the evolve-and-stay-put gate is the arbiter);
v_φ² = v_c² + c_s²·d ln(Σ_g/2z₀)/d ln R (closed form for the exponential,
clamped ≥ 0 near center like asymmetric drift); v_R = v_z = 0 (pressure is
the support). Toomre Q_gas = c_s·κ/(π·G·Σ_g) — note **π**, the gas value the
warm-disk comment already flags — computed at IC time, **fail-loud if < 1**
(no fragmentation physics exists; default c_s targets Q_gas ≈ 1.5–2). PRNG:
gas draws from a salted domain, base = splitmix(seed ^ GAS_SALT), so the
existing three-streams-per-galaxy spacing (galaxy 2 at mix³) is untouched —
gated by "stellar particles of a gas-enabled IC ≡ the gas-free IC bit-exact
at the same seed".

**D8 — Renderprep: single-channel ρ voxel grid, deposited at snapshot
endpoints, blended per subframe in the shader.** Kernel-weighted deposition
ρ(x_cell) = Σ_j m_j·W(|x_cell − x_j|, h_j) (gather per cell from a
cell-binned index → deterministic under rayon; NOT CIC — the kernel is the
correct band-limit and reuses D5). Default 128³ (QUICK 64³); cubic bounds
from a percentile radius of the gas population padded by 2·h_max
(camera-independent). Emission/absorption are NOT baked: the payload carries
ρ only; gas color, emissivity, and opacity κ are renderer uniforms — the
look iterates at re-render cost, not re-prep. Frame-data v2: GLXYFRAM
version → 2, flags word + optional gas block (dims, bounds, f32 data); v1
readable; stars-only v2 ≡ v1 semantics. Subframes: deposit ONLY at snapshot
endpoints (the M6c endpoint argument verbatim) and bind BOTH endpoint grids
as 3D textures with a mix factor u.

**D9 — Render: two additive passes + a per-star transmittance compute
prepass.** The frame is L(pixel) = Σ_stars E·T(cam→star) + ∫ j(ρ)·T(cam→s)ds
— both terms ADDITIVE once each carries its own attenuation, so the
order-independent Rgba32Float additive target survives intact and grade-time
bloom applies to the composite for free. Concretely:
  1. **Transmittance prepass** (compute): one thread per star marches the
     mixed ρ texture from star to camera (fixed step rule → deterministic),
     writes T = exp(−∫κρ ds) to a storage buffer. ~2·10⁵ stars × ~128 steps
     is trivial GPU work, and it is camera-correct per subframe.
  2. **Star pass**: the existing splat pipeline, emissive × T[instance].
     No gas / κ=0 ⇒ T ≡ 1.0 and ×1.0 is bit-exact in IEEE f32 ⇒ the landed
     M6g golden gates must pass unchanged.
  3. **Gas pass**: fullscreen triangle, per-pixel ray from the camera
     uniforms (ortho parallel / perspective eye-through-pixel), ray/AABB clip
     against grid bounds, front-to-back march (step ≈ half a voxel):
     C += T·j(ρ)·Δs, T *= exp(−κρΔs), early-exit at T < 1e-4; result
     additively blended into the same target.
  CPU per-star τ at prep time is REJECTED on a hard conflict: subframe
  cameras are generated at render time (Hermite u, rig u), so any
  camera-dependent quantity computed at prep time is stale for every
  subframe and would re-couple Contract 3 to a view axis. A camera-space
  transmittance volume is more machinery for no accuracy gain at this star
  count.

**D10 — Front-end**: `[model.gas]` (fraction, sound_speed, counts) lands
with the IC session; `[look.gas]` (color, emissivity, opacity, grid
resolution) with the finale; one new preset (`gasrich`, derived from an
existing merger geometry); QUICK reduces n_gas + grid res, keeps dt.

## Session map

| Session | Milestone | One-liner | Effort | Depends on |
|---|---|---|---|---|
| 1 | **M7a** | Gas plumbing (Species/snapshot v2) + kernel + hash-grid neighbors + adaptive-h density | M | — |
| 2 | **M7b** | SPH forces (pressure + Monaghan viscosity), `GravitySph`, CFL sentinel, isothermal shock tube | L | M7a |
| 3 | **M7c** | Isothermal gas-disk IC + evolve-and-stay-put gate + first gas-dynamical merger | M | M7b |
| 4 | **M7d** | Renderprep voxelization + frame-data v2 (optional gas block) | M | M7a (best demoed on M7c data) |
| 5 | **M7e** | Volumetric raymarch + full star attenuation (the money session) | L | M7d |
| 6 | **M7f** | scenario.toml gas knobs, `gasrich` preset, tuning, full-res showpiece | S–M | M7c + M7e |

Physics lands first (M7a→b→c), then the view side (M7d→e), then the
front-end (M7f).

### Amendment 2026-07-03 — view-first reorder (user-decided)

Remaining sessions run **M7d → M7e → M7b → M7c → M7f**. The dependency
table already permits it: M7d needs only M7a (landed), and every M7d/M7e
gate is analytic or synthetic (single-particle kernel exactness, uniform
slab, two-star ordering, gas-off ≡ M6g golden, GPU ≡ CPU mirror) — none
needs gas dynamics. Demo consequences:

- M7d/M7e demo on **static synthetic gas** (hand-rolled sech² disk
  positions, or a retained snapshot re-tagged `Species::Gas`) — the
  inclined dust-lane look without force code. The owed M7a demo
  (density side-by-side + timing) folds into the M7d session.
- The money demo (dynamically shocked merger dust lanes) moves to
  **M7c**, which renders through the already-landed volumetric path.
- κ/emissivity knobs tuned on static gas get re-tuned on real merger
  gas — absorbed by M7f's existing tuning pass.

**GPU SPH gate** (still not an M7 session): at M7c, measure the full-res
merger wall-clock. Painful (> ~30 min) → insert a GPU SPH session gated
against the CPU stack right after; otherwise it stays the M8-era opener.
GPU SPH can never precede M7b — the CPU forces are its oracle.
Long-horizon ordering beyond M7: `docs/plans/long-burning-beacon.md`.

### Amendment 2026-07-03 — M7e landed; sampling is manual-trilinear always

M7e landed (D9 implemented; M7a/M7d/M7e done, next M7b). One argued
deviation from D9's sketch: the gas textures are sampled by **manual
trilinear (8 `textureLoad`s) unconditionally**, not FLOAT32_FILTERABLE with
a manual fallback. Hardware samplers interpolate with ~8-bit fixed-point
subtexel weights, which would have made the "GPU march ≡ CPU reference"
gate tolerance hardware-dependent; the manual path replicates
`GasGrid::sample` in exact f32 arithmetic on every adapter, needs no
optional feature, and costs nothing measurable (1080p composite over 128³
+ 90k-star prepass ≈ 16 ms on the dev GPU). The FLOAT32_FILTERABLE fast
path is a named deferral if march time ever becomes the bottleneck. The
feature-detection spike D9 asked for is kept as the `gpu_features` example
(dev RTX 5090/Vulkan: both FLOAT32_BLENDABLE and FLOAT32_FILTERABLE
present). run_movie's endpoint-grid wiring (owed by the M7d entry) landed
with M7e; gas look constants stay hardcoded until `[look.gas]` (M7f).

---

## M7a — gas plumbing + SPH kernel + neighbors + density (Session 1, M)

Scope:
- `Species` enum + `kind` column on `State` (D1): update `from_phase_space`,
  `assert_consistent`, every struct-literal site (all ICs + snapshot reader +
  test helpers); existing ICs fill `Collisionless`.
- Snapshot v2 with v1 read-compat (D3).
- `solvers/src/sph/{kernel,grid,density,reference}.rs` (D5, D2): cubic-spline
  value/gradient/support; `HashGrid::build` + radius queries;
  `reference_neighbours` + `reference_density` O(N²) oracles; adaptive-h
  density (bisection to fixed tolerance, warm-start bracket, rayon over
  targets, fixed gather order).

Red-first gates: kernel normalization ∫W dV = 1 (quadrature, tol by
quadrature order); W(0) = 1/(πh³) hand value; W(2h) = 0; gradient vs central
difference (O(Δ²) tol); grid neighbor sets ≡ O(N²) oracle bit-exact (sorted
index lists; uniform AND clustered clouds; cell-wall straddlers; coincident
particles); uniform-lattice density → analytic m/s³ within documented
kernel-discretization tolerance; scaling law ρ(λx, λh) = ρ(x,h)/λ³;
adaptive-h recovers N_ngb within bisection tolerance, deterministic,
warm-start-independent (cold ≡ warm at the fixed tolerance); parallel ≡
serial bit-exact; snapshot v1 fixture → defaulted kind, v2 round-trip
bit-exact, garbage version rejected, M6f build-vs-IC `State` equality gates
still green; empty/single/N≤k edges.

Demo: side-by-side density coloring (grid-accelerated SPH vs O(N²) kNN) on a
retained cuspy snapshot + timing printout showing the O(N) win.

Files: core/src/state.rs, core/src/lib.rs, io/src/snapshot.rs,
solvers/src/sph/* (new), solvers/src/lib.rs; touch-list: ic/src/*.rs
construction sites, sim/renderprep tests that literal-construct State.

## M7b — SPH forces + composite solver + shock tube (Session 2, L)

Scope:
- `solvers/src/sph/forces.rs`: pairwise hydro acceleration, P = c_s²ρ,
  symmetrized kernel average (D2), Monaghan viscosity (α=1, β=2; Balsara
  deferred), fixed gather order.
- `solvers/src/sph/gravity_sph.rs`: `GravitySph<G>` (D4) with a gravity-off
  mode for pure-hydro tests.
- CFL sentinel (D6): `validate_dt` + `CflGuard` sink decorator; wired in
  xtask at t=0 and snapshot cadence.
- Shock-tube harness (solvers/tests/sph_shock_tube.rs): 16:1:1 lattice slab,
  4:1 density jump, gravity off, free ends — measure the central region
  before end rarefactions arrive (no periodic-BC machinery). Oracle: exact
  isothermal Riemann solution (closed form up to one scalar bisection inside
  the test).

Red-first gates: two-particle force hand oracle; Newton's-3rd-law pairwise
antisymmetry exact; global linear+angular momentum to roundoff on random gas
clouds (proptest); uniform-lattice interior ~zero net pressure force (edge
particles excluded, tol = lattice truncation); viscosity activates only on
approach (receding pair ⇒ Π=0); shock-tube density/velocity profiles vs
analytic (L1 bound justified by resolution + ~2–3h shock smearing,
documented); CFL sentinel trips on a deliberately over-large dt; parallel ≡
serial bit-exact.

Demo: shock-tube profile overlay (validate/sph/plot_shock.py, sibling of the
REBOUND harness) + a QUICK "gas ball" bounce movie (debug splats, gravity
off) showing pressure doing work.

## M7c — gas-disk IC + equilibrium + first gas sim (Session 3, M)

Scope:
- `ic/src/gas_disk.rs` (D7); gas tagged `Species::Gas`, progenitors 4/5.
- `ExponentialDisk`/`DiskCollision` grow the gas option (gas_fraction, c_s,
  n_gas); minimal `[model.gas]` in xtask/src/spec.rs so the demo runs the
  normal pipeline; palette covers gas progenitors (debug gas-as-splats).

Red-first gates: Σ normalization + enclosed-mass self-consistency; z₀(R)
hand values; realization recovers the sech² scale height statistically; v_φ
matches the pressure-corrected curve at bin means; ≥0 clamp near center (no
NaN); Q_gas < 1 rejected loudly; stellar part bit-identical to the gas-free
IC at the same seed (stream-spacing invariant, D7); zero net momentum/COM;
**evolve-and-stay-put**: isolated gas-rich disk under `GravitySph(BarnesHut)`
holds half-mass radius, thickness, ⟨v_φ⟩ profile within a few percent over
1–2 orbits (tolerance argued from the z₀ approximation) — plus the
differential proving the pressure term is load-bearing: removing the v_φ
pressure correction must measurably ring/expand the disk; CFL green at IC.

Demo: QUICK isolated gas-rich disk movie (disk visibly holds) + a first
QUICK gas-rich merger sim retained as M7d/M7e input.

## M7d — renderprep voxelization + frame-data v2 (Session 4, M)

Scope (D8):
- renderprep gains the `galaxy-solvers` dep; `renderprep/src/gasgrid.rs`:
  `GasGrid { dims, bounds, data }` + kernel deposition (h recomputed per
  snapshot via the shared solvers::sph routine), percentile bounds + 2·h_max
  pad.
- frame.rs v2: flags word + optional gas block; v1 readable.
- prepare.rs routes by `kind`: gas leaves the splat list (gas-as-splats
  becomes explicit opt-in); stellar outputs bit-identical to v1 on gas-free
  states.
- xtask run_movie: deposit per snapshot endpoint, hold both grids in memory,
  pass (grid0, grid1, u) toward render.

Red-first gates: single particle at a cell center → grid ≡ sampled kernel
exactly; total grid mass ≈ M_gas within a tolerance justified by grid
Nyquist vs h; uniform slab → flat interior density; deposition deterministic
+ parallel ≡ serial bit-exact; bounds contain the chosen percentile; frame
v1 fixture reads; v2 with/without gas block round-trips bit-exact; gas-free
state ⇒ prepare output bit-identical to today; grid lerp at u=0/u=1
reproduces endpoints (CPU reference for the shader mix).

Demo: dust-lane preview without the raymarcher — contact sheet of
axis-aligned ρ-integral slices (EXR→PNG via the existing grade path) from a
mid-collision M7c snapshot.

## M7e — volumetric raymarch + full star attenuation (Session 5, L)

Scope (D9):
- `render/src/volume.rs`: R32Float 3D textures (×2 endpoints + mix uniform),
  transmittance compute prepass (per-star march → storage buffer),
  fullscreen gas pass (ray-gen ortho + perspective from existing camera
  uniforms, AABB clip, front-to-back march, early exit), gas look uniforms.
- render/src/render.rs: splat vertex shader reads T[instance]; pass
  orchestration (prepass → stars → gas, one additive Rgba32Float target);
  request FLOAT32_FILTERABLE when available with a manual-trilinear WGSL
  fallback (8 fetches), fail-loud reporting which path is active — verify
  feature detection FIRST, before building around sampling.
- CPU mirror of the march (same step rule, same early-exit) as the oracle
  for GPU gates, per the flatten/aggregate precedent.

Red-first gates: **gas-off ⇒ bit-compatible** — no gas block / κ=0 passes
the landed M6g golden (flux + probe pixels) unchanged, the load-bearing
regression gate; analytic uniform slab: T = exp(−κρL) per star behind the
slab and gas radiance = (j/κ)(1−e^(−κρL)) per pixel (closed forms; tolerance
from first-order optical-depth quadrature at the chosen step count,
documented); two-star depth ordering: front star unattenuated, back star
dimmed by the full slab, swapping the camera side swaps the roles; ray-gen
hand oracles at corner pixels, ortho and perspective; GPU march ≡ CPU
reference within f32 tolerance; early-exit vs full march within the exit
threshold; emission linearity (2× emissivity ⇒ 2× gas flux); mix u=0/u=1 ⇒
endpoint grids; same-device determinism.

Demo: the series' signature — QUICK gas-rich merger movie with dark dust
lanes over attenuated stellar cores, glowing bridge gas; an A/B pair
(attenuation on/off) for DESIGN.

## M7f — scenario knobs, `gasrich` preset, tuning + showpiece (Session 6, S–M)

Scope: `[look.gas]` (color, emissivity, κ, grid res) + completed
`[model.gas]` in spec.rs with deny_unknown_fields; the `gasrich` preset
(existing merger geometry, gas_fraction ≈ 0.2, c_s for Q_gas ≈ 1.5–2); QUICK
path (n_gas ~6–10k, 64³, same dt); knob tuning by eyeball, documented; perf
pass only if runtime actually hurts. End-of-series ritual: DESIGN.md M7
entries, memory update, full quality gate, commit + push.

Red-first gates: preset parses and `build_scenario` reproduces the
hand-built spec (M6f pattern); palette/ramp validation covers 6 progenitors;
QUICK/full mapping; unknown gas keys rejected.

Demo: the full-res `gasrich` merger movie — dust lanes, glowing bridge,
attenuated cores.

## Explicitly out of scope (this series)

- Energy equation (adiabatic EOS, cooling/heating, entropy formulation).
- Star formation / feedback (the M6e density→blue proxy remains the
  visualization stand-in).
- GPU SPH (CPU + rayon only; the GPU door stays oracle-first, deferred).
- Adaptive / per-particle timesteps (fixed dt + CFL sentinel; individual
  timesteps are the named follow-up if the sentinel forces a painful dt).
- Periodic boundary conditions (shock tube uses the padded-ends trick).
- Grad-h correction terms; h-coupled gravitational softening for gas.
- Balsara switch / time-dependent viscosity α.
- Blender gas consumer, HDR video encode (unchanged deferrals).

## Risk notes

- **FLOAT32_FILTERABLE** (linear-sampling R32Float 3D textures) is distinct
  from the target's FLOAT32_BLENDABLE; present on the dev RTX 5090/Vulkan
  but not universal — the manual-trilinear fallback keeps M7e portable;
  spike feature detection first.
- **3D texture memory**: 128³ R32Float = 8 MB; two endpoint grids resident
  is trivial; grids live one-per-snapshot in frame-data, never per subframe.
- **f32 grid / τ precision**: ρ spans orders of magnitude; the banding risk
  is in the look, not correctness — the analytic slab gate bounds it.
- **Raymarch cost**: ~2M rays × ≤ few hundred steps with early exit is
  comfortable; per-star prepass is O(N_star × steps), trivial. Named
  mitigation if it hurts: half-res gas + upsample (deferred).
- **Pairing instability**: N_ngb ≤ ~57 for the cubic spline; shock-tube
  lattice spacing chosen so h ≈ 1.2–1.3 spacings; the uniform-lattice
  zero-force gate is the canary.
- **CPU SPH runtime**: 2·10⁵ gas × ~50 neighbors × ~1500 steps ≈ 10¹⁰ pair
  interactions — tens of minutes with rayon at full res; QUICK stays
  seconds-to-minutes. Adaptive-h warm-start keeps the density pass cheap.
- **Isothermal fragility**: with no fragmentation physics a Q_gas < 1 disk
  clumps artificially — the fail-loud Q gate and the c_s default keep v1 in
  the regime the physics supports.
- **State-construction ripple** (M7a): ~9 struct-literal sites + test
  helpers; `assert_consistent` grows the new column so any miss fails loudly
  in tests.
