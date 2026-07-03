# M7 — "the breathing": isothermal SPH gas + volumetric dust-lane render series (roadmap)

A session-by-session plan to add **gas** to the engine — v1 physics is
**isothermal SPH** (fixed sound speed c_s, density summation, P = c_s²ρ
pressure gradient, Monaghan artificial viscosity; **no energy equation**) —
and a **volumetric raymarched gas render path with full starlight
attenuation**: every star splat is dimmed by the gas optical depth between it
and the camera, and the gas itself is composited by emission+absorption
raymarching. Dust lanes over stellar cores — the DESIGN Stage-5 item, and the
one render model DESIGN explicitly forbids reusing the splat path for.

Each session is one milestone (M7a–M7f), independently demoable, red-first
per CLAUDE.md. Physics correctness lands before render spectacle.

## Context — what exists today (audit, 2026-07)

- **No gas anywhere.** `State` (`core/src/state.rs`) is pos/vel (f64 DVec3),
  mass (f64), id, progenitor (u16 species tag, saturated 0–3 by
  `DiskCollision`), time, a. Constructed by struct literal at ~9 sites (every
  IC + the snapshot reader). No particle-kind column, no smoothing length.
- **One force hook.** `ForceSolver::accelerations(&mut self, &State, &mut
  [DVec3])` + `potential_energy` (`core/src/traits.rs`) is the only way force
  enters; `LeapfrogKdk` calls it once per step (KDK caches the closing-kick
  acc for the next opening kick), at the *post-drift* positions with
  velocities at v_(n+1/2). Fixed global dt from `SimConfig`; no CFL machinery.
- **No neighbor range-query.** `renderprep/src/density.rs` has the O(N²) kNN
  reference oracle (`knn_density` / `knn_neighbourhood`); `FlatTree` / `Lbvh`
  carry per-node AABBs but no range query is implemented. House pattern:
  every fast spatial structure is gated against an O(N²) oracle.
- **Snapshot v1 is closed.** `io/src/snapshot.rs` FORMAT_VERSION=1,
  exact-match version check, hardcoded columns — gas columns force a v2.
- **Frame-data v1 is stars-only.** `GLXYFRAM` v1 is per-particle f32 SoA
  (pos/color/size/brightness). DESIGN's Contract-3 sketch already anticipates
  "or density grid for gas" — a versioned payload extension, not a bolt-on.
  `run_movie` keeps FrameData **in memory** (writes only EXR/PNG), so a gas
  grid per snapshot is a RAM object on the movie path, not a per-subframe
  disk cost.
- **Render (post-M6g)** is a single additive splat pass: world-space
  instances (`GpuSplat` pos/radius/emissive), projection in the vertex
  shader, ortho (golden-gated) + perspective (1/d² gated), `Rgba32Float` +
  `FLOAT32_BLENDABLE`, no depth buffer. Camera/CameraPath rigs are reusable;
  no per-pixel ray generator exists yet.
- **ICs**: `ExponentialDisk<H>` (live halo + stellar disk, warm via
  `with_toomre_q`) with THREE PRNG streams per galaxy (seed, mix(seed),
  mix²(seed)); `DiskCollision` galaxy 2 starts at mix³. Inserting a 4th
  stream *between* them breaks the spacing invariant — plan around it.
- **Scale target for this series**: CPU SPH at ~10⁴ gas particles (QUICK) to
  ~10⁵–2·10⁵ (full). This is NOT the 10⁸ path; GPU SPH is out of scope.

## Ground rules (every session)

1. **TDD per CLAUDE.md**: red tests committed separately (`[red]`, workspace
   still compiles via `todo!()` stubs), then implementation; never weaken a
   test to pass. Every tolerance justified by the method's order.
2. **Oracle discipline**: every fast path is gated against an independent
   reference — O(N²) neighbor loops for the grid, analytic solutions
   (isothermal Riemann, uniform slab transmittance) for physics and
   raymarching, the retained golden gates for "gas off ≡ old renderer".
3. **Unit-gate the math, eyeball the aesthetics** (M3.6/M6 precedent): kernel
   normalization, shock profiles, transmittance integrals get exact/invariant
   gates; "the dust lane looks right" is an eyeballed QUICK render with the
   chosen knobs documented in DESIGN.
4. **Demo criterion**: every session ends with a rendered artifact — a
   `GALAXY_MOVIE_QUICK=1` movie where possible (temp output under
   `M:\claud_projects\temp`), a rendered validation image where the movie
   machinery doesn't exist yet.
5. **Contracts stay versioned and deliberate**: snapshot v2 and frame-data v2
   are explicit schema bumps with v1 read-compat gates; the EXR stays the
   pristine linear artifact (bloom stays at grade time and applies to the
   star+gas composite for free).
6. **Equilibrium-IC discipline**: virial ratio alone is insufficient — every
   new IC must *evolve and stay put* under the real solver stack.
7. **End-of-batch ritual**: DESIGN.md milestone entry + memory update +
   quality gate (`cargo test`, `clippy -D warnings`, `fmt --check`) + commit
   AND push.

## Design decisions (recommended; argued once here, referenced per session)

**D1 — Gas fields live ON `State` as two new columns.**
`kind: Vec<Species>` (`#[repr(u8)] enum Species { Collisionless, Gas }`,
newtype-spirit per house style) and `h: Vec<f64>` (smoothing length; `0.0`
for collisionless — **not NaN**: `State` derives `PartialEq` and the M6f
build-vs-direct-IC `State` equality gate would silently never match again
with NaN). ρ is NOT stored on `State` — it is recomputed by the SPH solver
each force call (cheap relative to the pair loop) and re-derived in
renderprep by grid deposition (the deposition Σ_j m_j·W(x−x_j, h_j) *is* the
SPH density field), so there is no stale-density hazard and no column to keep
coherent. A separate `GasState` block is rejected: every consumer
(integrator, solvers, snapshot, renderprep, diagnostics) iterates one
`State`; splitting it breaks all of them for zero gain at 2·10⁵.
`progenitor` stays a pure identity tag; gas particles get NEW tags
(DiskCollision: halo1=0, disk1=1, halo2=2, disk2=3, **gas1=4, gas2=5**) and
the palette covers all progenitors — the gas entries drive the *debug
gas-as-splats* mode (M7c) and the LookSpec palette-length validation stays
uniform. Routing to the volumetric path keys on `kind`, never on progenitor.

**D2 — Snapshot v2, backward-readable.** FORMAT_VERSION → 2; two appended
columns `kind[n] (u8)`, `h[n] (f64)` (f64 matches the pos/vel lossless
discipline; mass stays the one documented f32-lossy field). The reader
accepts {1, 2}: v1 fills `kind=Collisionless, h=0.0` — the retained zoo
snapshots stay re-preppable/regradeable. Writer always emits v2. This is a
deliberate relaxation of the exact-match version check; gated (v1 fixture →
defaulted columns; v2 round-trip bit-exact; garbage version still rejected).

**D3 — SPH plugs in as a composite `ForceSolver`; the trait does not widen.**
`GravitySph<G: ForceSolver>` in `solvers`: `accelerations` = gravity over ALL
particles (gas is just mass to BarnesHut; shared Plummer ε — softening and
smoothing deliberately decoupled in v1, documented) + hydro acceleration
added to gas rows only. Because `accelerations` takes `&mut self`, the SPH
component recomputes ρ/h internally per call — which is exactly once per KDK
step, at the post-drift positions, as SPH requires. Artificial viscosity uses
the velocities present at that call (v_(n+1/2)) — the standard leapfrog-SPH
treatment (Gadget-2 does the same velocity prediction); this is a first-order
error in the *viscous term only*, documented, and invisible to the momentum
gates because the pairwise force stays antisymmetric. `potential_energy`
delegates to the wrapped gravity solver. **Total energy is NOT a gate for gas
runs** — an isothermal EOS is an implicit infinite heat bath; the
conservation gates for gas are linear/angular momentum (exact by pairwise
antisymmetry, to roundoff) and the shock-tube oracle.

**D4 — Smoothing length: adaptive-by-neighbor-count, symmetrized forces, no
grad-h terms.** v1 is "classic SPH" (Monaghan 1992): per-particle h_i solved
by bisection (fixed iteration count → deterministic) so the kernel-weighted
neighbor count ≈ N_ngb (default ~48; the cubic spline pairs above ~60 — the
pairing-instability guard), pairwise forces symmetrized via the kernel
average W_ij = 0.5·(W(h_i)+W(h_j)) so momentum conservation stays *exact*.
The Springel–Hernquist grad-h correction is a named deferral. Fixed-global-h
is rejected for production (an exponential disk spans orders of magnitude in
density) but remains the trivial special case the unit gates use.

**D5 — Neighbor search: uniform hash grid, in `solvers/src/sph/`.** Cell size
= support radius of h_max (2·h_max); counting-sort-by-cell layout (cell-start
offsets + particle index array — compact, rayon-friendly). Queries GATHER per
target in ascending neighbor index (fixed summation order → bit-exact
parallel↔serial, the house discipline). Gated bit-exact (sorted neighbor
index lists) against a new O(N²) `reference_neighbours` oracle, same stance
as `reference_sort`/`reference_morton`. A FlatTree/Lbvh range query is
rejected for v1: the hash grid is O(N), simpler, and the SPH standard; tree
range-query stays a follow-up if the h dynamic range ever hurts.

**D6 — Timestep: fixed global dt + fail-loud CFL sentinel.** No adaptive
stepping in v1. `sim::run` gains a check at t=0 and at every snapshot
interval: dt ≤ C_cfl · min_i h_i / v_sig,i with v_sig = c_s + the standard
Monaghan viscous signal term (1.2·(α·c_s + β·μ_max) over neighbors),
C_cfl ≈ 0.25. Violation is a typed `SimError::CflViolation` — the run dies
loudly instead of silently exploding. QUICK keeps the same dt (larger h at
lower N makes the bound *easier*, so QUICK stays honest).

**D7 — Kernel: cubic spline M4, support radius 2h, hosted in `core`.**
W(r,h) with compact support at r=2h (Monaghan convention; normalization
1/(πh³) at r=0 — hand-value gate), analytic gradient. It lives in
`core/src/sph_kernel.rs` (pure math, no I/O — core-legal) because it is the
single source of truth for BOTH the solver's pair forces and renderprep's
grid deposition, exactly the `reference_*` single-source pattern.

**D8 — Gas disk IC: isothermal vertical structure, pressure-corrected
rotation, salted seed domain.** `GasDisk` (new, `ic/src/gas_disk.rs`,
composable into `ExponentialDisk`/`DiskCollision`): Σ_g(R) exponential with
`gas_fraction` splitting the disk mass (M_gas = f·M_disk, the stellar disk
gets 1−f — the total rotation curve is unchanged); vertical profile
sech²(z/z₀) with the isothermal self-gravitating scale height
z₀ = c_s²/(πG·Σ_g) — documented (like σ_z was) as mildly approximate in the
halo-dominated potential, with the *evolve-and-stay-put gate as the real
arbiter*; v_φ² = v_c² + c_s²·d ln(Σ_g/2z₀)/d ln R (closed form for the
exponential; clamped ≥ 0 at small R exactly like asymmetric drift);
v_R = v_z = 0 (pressure is the support — no random velocities). Toomre
Q_gas = c_s·κ/(**π**·G·Σ_g) (the π the warm-disk comment already flags) is
computed at IC time and **fail-loud** if < 1 (no fragmentation physics
exists; default c_s chosen for Q_gas ≈ 1.5–2). PRNG: gas draws from a SALTED
domain, base = splitmix(seed ^ GAS_SALT) — the existing
three-streams-per-galaxy spacing (galaxy 2 at mix³) is untouched, gated by
"stellar particles of a gas-enabled IC ≡ the gas-free IC bit-exact at the
same seed".

**D9 — Renderprep: single-channel ρ voxel grid, deposited at snapshot
endpoints, blended per subframe in the shader.** Kernel-weighted deposition
(ρ(x_cell) = Σ_j m_j·W(|x_cell − x_j|, h_j), gathered per cell from a
cell-binned particle index → deterministic under rayon), NOT CIC (the kernel
is the correct band-limit and reuses D7). Default 128³ (QUICK 64³), cubic
bounds from a percentile radius of the gas population (camera-independent,
same robust-percentile philosophy as framing), per snapshot.
Emission/absorption are NOT baked: the payload carries ρ only; emissivity,
gas color, and opacity κ are renderer uniforms, so the gas look iterates at
re-render (not re-prep) cost. Frame-data v2: `GLXYFRAM` version → 2, header
gains a flags word, an optional gas block (dims u32×3, bounds f32×6,
data f32[nx·ny·nz]); v1 readable (no gas), stars-only v2 ≡ v1 semantics.
Subframes: deposit ONLY at snapshot endpoints (the M6c endpoint-kNN argument
verbatim — density evolves on the snapshot timescale) and bind BOTH endpoint
grids as 3D textures with a mix factor u — the blend costs one extra texture
sample, zero CPU.

**D10 — Render: two additive passes + a per-star transmittance compute
prepass. CPU tau in renderprep is rejected.** The physically correct frame is
L(pixel) = Σ_stars E_star·T(camera→star) + ∫ j(ρ)·T(camera→s) ds, and both
terms are ADDITIVE once each carries its own attenuation — so the existing
order-independent `Rgba32Float` additive target survives intact, and bloom
(grade-time) applies to the composite for free. Concretely:
  1. **Transmittance prepass** (compute): one thread per star marches the ρ
     3D texture from star to camera (fixed step count → deterministic),
     writes T = exp(−∫κρ ds) to a storage buffer; the splat vertex shader
     multiplies emissive by T[instance]. This is option (b) done once per
     *instance* instead of 6× per vertex. ~2·10⁵ stars × ~128 steps is
     trivial GPU work per frame.
  2. **Star pass**: the existing splat pipeline, emissive × T. κ=0 or no gas
     block ⇒ T ≡ 1 ⇒ the landed golden gates must pass unchanged
     (bit-compatible).
  3. **Gas pass**: fullscreen triangle, per-pixel ray from the camera
     uniforms (ortho: parallel rays; perspective: eye-through-pixel), ray/AABB
     clip against the grid bounds, front-to-back march (step ≈ half a voxel):
     C += T·j(ρ)·Δs, T *= exp(−κρΔs), early-exit at T < 1e-4; the result is
     additively blended into the same target.
  Option (c) — CPU per-star τ at prep time — is rejected on a hard conflict:
  M6c/M6d generate subframe cameras at RENDER time (Hermite u, rig u), so any
  camera-dependent quantity computed at prep time is stale for every subframe
  and would re-couple frame-data to a view axis, which Contract 3 exists to
  avoid. Option (a) — a camera-space transmittance *volume* — is more
  machinery (a per-frame resample + directional sweep) for no accuracy gain
  at this star count.

**D11 — Front-end**: `[model.gas]` (fraction, sound_speed, counts) lands with
the IC session; `[look.gas]` (color, emissivity, opacity, grid resolution)
with the finale; one new preset (`gasrich`, derived from the cuspy/disk
merger geometry); QUICK reduces n_gas and grid res, keeps dt.

## Session map

| Session | Milestone | One-liner | Effort | Depends on |
|---|---|---|---|---|
| 1 | **M7a** | Gas plumbing (State/snapshot v2) + kernel + hash-grid neighbors + adaptive-h density | M | — |
| 2 | **M7b** | SPH forces (pressure + Monaghan viscosity), `GravitySph`, CFL sentinel, isothermal shock tube | L | M7a |
| 3 | **M7c** | Isothermal gas-disk IC + evolve-and-stay-put gate + first gas-dynamical sim | M | M7b |
| 4 | **M7d** | Renderprep voxelization + frame-data v2 (optional gas block) | M | M7a (kernel); demos best on M7c data |
| 5 | **M7e** | Volumetric raymarch + full star attenuation (the money session) | L | M7d |
| 6 | **M7f** | scenario.toml gas knobs, `gasrich` preset, tuning, full-res showpiece | S–M | M7c + M7e |

Physics lands first (M7a→M7b→M7c), then the view side (M7d→M7e), then the
front-end (M7f). M7d touches no solver code and can start any time after
M7a, but its demo is far better with real M7c collision data.

---

## M7a — gas plumbing + SPH kernel + neighbors + density (Session 1, M)

**Goal:** the foundations everything else stands on: `State` learns what a
gas particle is, snapshots carry it, and SPH density with adaptive h is
computed fast and gated against an O(N²) oracle.

Scope:
- `Species` enum + `kind`/`h` columns on `State` (D1): update
  `from_phase_space`, `assert_consistent`, and every struct-literal
  construction site (all ICs + snapshot reader + test helpers) — mechanical,
  non-local; all existing ICs fill `Collisionless`/`0.0`.
- Snapshot v2 with v1 read-compat (D2).
- `core/src/sph_kernel.rs` (D7): cubic-spline value, gradient, support.
- `solvers/src/sph/grid.rs` (D5): `HashGrid::build(pos, cell)` +
  `neighbours_within(i, r)`; `solvers/src/sph/reference.rs`:
  `reference_neighbours` (O(N²)) + `reference_density` — the oracles.
- `solvers/src/sph/density.rs`: density summation + adaptive-h bisection on
  kernel-weighted neighbor count (D4), rayon over targets, fixed gather
  order, warm-start from the previous h (deterministic, cheap — plan for the
  per-step cost up front).

Red-first gates: kernel normalization ∫W dV = 1 (numeric quadrature, tol by
quadrature order); W(0) = 1/(πh³) hand value; W(2h) = 0; gradient vs central
difference (O(Δ²) tol); grid neighbor sets ≡ O(N²) oracle bit-exact (sorted
index lists; uniform AND clustered clouds; particles straddling cell walls;
coincident particles); density on a uniform lattice → analytic ρ = m/s³
within the documented kernel-discretization tolerance; scaling law
ρ(λx, λh) = ρ(x, h)/λ³; adaptive-h recovers N_ngb within the bisection
tolerance, deterministic; parallel ≡ serial bit-exact; snapshot v1 fixture →
defaulted gas columns, v2 round-trip bit-exact, the M6f build-vs-IC `State`
equality gates still green; empty/single/N≤k edges.

Demo: a rendered side-by-side image — a retained cuspy snapshot colored by
grid-accelerated SPH density vs the O(N²) kNN coloring (throwaway xtask dev
subcommand), plus the timing printout showing the O(N) win at N=30k.

Files: `core/src/state.rs`, `core/src/lib.rs`, `core/src/sph_kernel.rs`
(new), `io/src/snapshot.rs`, `solvers/src/sph/{mod,grid,density,reference}.rs`
(new), `solvers/src/lib.rs`; touch-list: `ic/src/*.rs` construction sites and
`sim`/`renderprep` tests that literal-construct `State`.

## M7b — SPH forces + composite solver + shock tube (Session 2, L)

**Goal:** gas pushes back. Pressure-gradient + artificial-viscosity
accelerations behind the existing `ForceSolver` trait, with the series'
headline physics validation: the isothermal shock tube against its analytic
solution.

Scope:
- `solvers/src/sph/forces.rs`: pairwise hydro acceleration, P = c_s²ρ,
  symmetrized kernel average (D4), Monaghan viscosity (α=1, β=2; Balsara
  switch deferred), fixed gather order per target.
- `solvers/src/sph/gravity_sph.rs`: `GravitySph<G: ForceSolver>` (D3), with a
  gravity-off mode for pure-hydro tests.
- CFL sentinel in `sim::run` (D6): `SimError::CflViolation`, checked at t=0
  and per snapshot interval; the energy diagnostic scoped to collisionless
  runs (documented — isothermal is not conservative).
- Shock-tube harness (`solvers/tests/sph_shock_tube.rs`): long 16:1:1 lattice
  slab, 4:1 density jump, gravity off, free ends — measure the central region
  before the end rarefactions arrive (the standard no-boundary trick; no
  periodic-BC machinery is built). Oracle: the exact isothermal Riemann
  solution (closed-form residual, one scalar bisection inside the test —
  house-legal "analytic up to a root-find").

Red-first gates: two-particle force hand oracle (equal-h pair, exact P/ρ²
form); Newton's 3rd law pairwise (m_i·a_i = −m_j·a_j exact); global linear +
angular momentum conserved to roundoff on random gas clouds (proptest);
uniform-lattice interior particles feel ~zero net pressure force (edge
particles excluded, tolerance = lattice truncation); viscosity activates
only on approach (receding pair ⇒ Π = 0); viscous pair force antisymmetric;
isothermal shock tube density/velocity profiles vs analytic, L1 error bound
justified by resolution + kernel width (document the expected shock smearing
of ~2–3h); CFL sentinel trips on a deliberately over-large dt; parallel ≡
serial bit-exact; determinism.

Demo: shock-tube profile overlay (rendered PNG via a small
`validate/sph/plot_shock.py`, sibling of the REBOUND harness) + a QUICK
"gas ball" bounce movie (uniform sphere, gravity off, debug splats) showing
pressure doing work.

Files: `solvers/src/sph/{forces,gravity_sph}.rs` (new),
`solvers/src/sph/mod.rs`, `sim/src/lib.rs`,
`solvers/tests/sph_shock_tube.rs` (new), `validate/sph/plot_shock.py` (new,
manual like REBOUND).

## M7c — isothermal gas-disk IC + equilibrium + first gas sim (Session 3, M)

**Goal:** a gas disk that *stays put* — the house equilibrium-IC discipline
applied to pressure support — then the first gas-dynamical collision.

Scope:
- `ic/src/gas_disk.rs` (D8): exponential Σ_g, sech² isothermal vertical
  structure, pressure-corrected v_φ (clamped), Q_gas fail-loud, salted PRNG
  domain; gas particles tagged `Species::Gas`, progenitor 4/5, h seeded from
  the local interparticle spacing (the bisection refines it on step 1).
- `ExponentialDisk`/`DiskCollision` grow the gas option (gas_fraction, c_s,
  n_gas); minimal `[model.gas]` spec table in `xtask/src/spec.rs` (D11 first
  half) so the demo runs through the normal pipeline.
- Palette/LookSpec ripple: palette length covers gas progenitors; the gas
  entries drive the debug gas-as-splats look (routing by `kind` arrives in
  M7d).

Red-first gates: Σ normalization + enclosed-mass self-consistency (the disk
IC pattern); z₀(R) hand values; a realization's vertical profile recovers the
sech² scale height statistically; v_φ matches the pressure-corrected curve at
bin means; the ≥0 clamp near center (no NaN); Q_gas < 1 rejected loudly; the
stellar part bit-identical to the gas-free IC at the same seed (the
stream-spacing invariant, D8); zero net momentum / COM; **evolve-and-stay-
put**: an isolated gas-rich disk under `GravitySph(BarnesHut)` holds
half-mass radius, thickness, and the ⟨v_φ⟩ profile within a few percent over
1–2 orbits (tolerance argued from the z₀ approximation, like the warm-disk
16× differential — plus the differential that proves the pressure term is
load-bearing: removing the v_φ pressure correction must make the disk
measurably ring/expand); CFL green at IC.

Demo: QUICK isolated gas-rich disk movie (gas as debug splats in its own
palette color) — the disk visibly holds; plus a first QUICK gas-rich merger
sim retained as M7d/M7e input.

Files: `ic/src/gas_disk.rs` (new), `ic/src/disk.rs`,
`ic/src/disk_collision.rs`, `ic/src/lib.rs`, `xtask/src/spec.rs`,
`xtask/src/main.rs`, `sim/tests/` equilibrium gate.

## M7d — renderprep gas voxelization + frame-data v2 (Session 4, M)

**Goal:** the Contract-3 boundary learns about gas: SPH particles → a
single-channel ρ voxel grid, versioned into the frame-data payload, with the
M6c endpoint discipline extended to grids.

Scope (D9):
- `renderprep/src/gasgrid.rs`: `GasGrid { dims, bounds, data: Vec<f32> }` +
  kernel-weighted deposition (gather per cell from a cell-binned particle
  index — deterministic under rayon), percentile-based cubic bounds padded by
  2·h_max.
- `frame.rs` v2: flags word + optional gas block; v1 readable; stars-only v2
  round-trips; count/bounds authoritative-from-data preserved.
- `prepare.rs`: routes by `kind` — gas particles leave the splat list (the
  debug gas-as-splats look becomes an explicit opt-in), stellar outputs
  bit-identical to v1 `prepare` on gas-free states.
- xtask `run_movie`: deposit per snapshot endpoint, hold both grids for the
  span (in memory — audited above), pass (grid0, grid1, u) toward render.

Red-first gates: single particle at a cell center → grid ≡ the sampled
kernel (exact values); total grid mass ≈ M_gas within a tolerance justified
by grid Nyquist vs h (documented); uniform slab of particles → flat interior
density (analytic); deposition deterministic + parallel ≡ serial bit-exact;
bounds contain the chosen percentile of gas; frame v1 fixture reads; v2
with/without gas block round-trips bit-exact; gas-free state ⇒ `prepare`
output bit-identical to today; grid lerp at u=0 / u=1 reproduces endpoints
(the CPU reference for the shader mix).

Demo: a dust-lane *preview without the raymarcher* — a contact sheet of grid
slices (axis-aligned ρ integrals written as EXR→PNG through the existing
grade path) from a mid-collision M7c snapshot: the gas morphology made
visible one session early.

Files: `renderprep/src/gasgrid.rs` (new), `renderprep/src/frame.rs`,
`renderprep/src/prepare.rs`, `renderprep/src/lib.rs`, `xtask/src/main.rs`.

## M7e — volumetric raymarch + full star attenuation (Session 5, L)

**Goal:** the money session — dust lanes over stellar cores. Gas raymarched
emission+absorption; every star dimmed by exp(−τ) to the camera; the
additive-target architecture and every landed golden gate survive.

Scope (D10):
- `render/src/volume.rs`: R32Float 3D textures (×2 endpoints + mix uniform),
  the transmittance compute prepass (per-star march → storage buffer), the
  fullscreen gas pass (ray-gen ortho + perspective from the existing camera
  uniforms, AABB clip, front-to-back march, early exit), gas look uniforms
  (color, emissivity, κ).
- `render/src/render.rs`: the splat vertex shader reads T[instance]; pass
  orchestration (prepass → stars → gas, all into the one `Rgba32Float`
  additive target); `Renderer::new` requests `FLOAT32_FILTERABLE` when
  available with a manual-trilinear WGSL fallback (8 fetches) — fail-loud
  reporting which path is active.
- A CPU mirror of the march (pure Rust: same step rule, same early-exit) as
  the oracle for the GPU gates, per the flatten/aggregate precedent.

Red-first gates: **gas-off ⇒ bit-compatible** — no gas block / κ=0 must pass
the landed M6g golden (flux + probe pixels) unchanged, the load-bearing
regression gate; analytic uniform slab: T = exp(−κρL) per star behind the
slab (closed form, tolerance from the step count — first-order quadrature of
the optical depth, documented) and gas radiance = (j/κ)(1 − e^(−κρL)) per
pixel (closed form); two-star depth ordering: front star unattenuated, back
star dimmed by the full slab, swapping the camera side swaps the roles;
ray-gen hand oracles at corner pixels, ortho and perspective; GPU march ≡
CPU reference within f32 tolerance; early-exit vs full march within the exit
threshold; emission linearity (2× emissivity ⇒ 2× gas flux); mix factor
u=0 / u=1 ⇒ the endpoint grids; determinism (same device).

Demo: the series' signature — QUICK gas-rich merger movie with dark dust
lanes silhouetted against the attenuated stellar cores, gas glow in the
bridge; an A/B pair (attenuation on/off) for DESIGN.

Files: `render/src/volume.rs` (new), `render/src/render.rs`,
`render/src/camera.rs` (ray-gen helpers), `render/src/lib.rs`,
`xtask/src/main.rs`.

## M7f — scenario knobs, `gasrich` preset, tuning + showpiece (Session 6, S–M)

**Goal:** make it a product: gas is a scenario option, the look iterates
cheaply, and the full-res showpiece lands.

Scope:
- `[look.gas]` in `xtask/src/spec.rs` (color, emissivity, opacity κ, grid
  resolution) with the deny_unknown_fields discipline; `[model.gas]`
  completed; the `gasrich` preset (cuspy-derived geometry, gas_fraction
  ≈ 0.2, c_s for Q_gas ≈ 1.5–2); QUICK path (n_gas ~6–10k, 64³ grid,
  same dt).
- Knob tuning by eyeball (rule 3), chosen values documented in DESIGN; a
  perf pass only if the CPU SPH run time actually hurts (rayon coverage
  audit; the named GPU-SPH follow-up stays deferred).
- End-of-series ritual: DESIGN.md M7 entries, memory update, full quality
  gate, commit + push.

Red-first gates: the preset parses and `build_scenario` reproduces the
hand-built spec (the M6f preset gate pattern); palette/ramp validation
covers 6 species; QUICK/full spec mapping; unknown gas keys rejected.

Demo: the full-res `gasrich` merger movie — dust lanes, glowing bridge,
attenuated cores — plus the DESIGN scenario-table row.

Files: `xtask/src/spec.rs`, `xtask/src/main.rs`, the preset TOML,
`DESIGN.md`.

## Explicitly out of scope (this series)

- **Energy equation** — adiabatic EOS, radiative cooling/heating, entropy
  formulation: v1 is isothermal by locked decision.
- **Star formation / feedback** (the M6e density→blue proxy remains the
  visualization stand-in).
- **GPU SPH** (CPU + rayon at 10⁵–2·10⁵ only; the GPU door stays the
  documented follow-up, oracle-first as always).
- **Adaptive / per-particle timesteps** — fixed global dt + the CFL sentinel
  (D6); individual timesteps are the named follow-up if the sentinel forces
  a painfully small global dt.
- **Periodic boundary conditions** (the shock tube uses the padded-ends
  trick; cosmology's periodic solver is a different milestone).
- **Grad-h correction terms** (Springel–Hernquist fully-conservative SPH) and
  **h-coupled gravitational softening** for gas — documented v1
  approximations.
- **Balsara viscosity switch / time-dependent α** — plain Monaghan α,β first.
- **Blender gas consumer**, **HDR video encode** — unchanged deferrals.

## Risk notes

- **`FLOAT32_FILTERABLE`**: linear-sampling R32Float 3D textures needs this
  wgpu feature (distinct from the render target's FLOAT32_BLENDABLE). Present
  on the dev box (RTX 5090/Vulkan) but not universal — the manual-trilinear
  fallback in M7e keeps the pass portable; verify feature detection in the
  first M7e spike, before the pipeline is built around sampling.
- **3D texture limits / memory**: 128³–256³ R32Float (8–64 MB) is far below
  the 2048 per-dimension wgpu limit; two endpoint grids resident is fine. The
  real cost is frame-data v2 file size IF grids were ever persisted per
  subframe — the movie path holds them in memory per snapshot (audited), and
  the contract stores one grid per *snapshot* frame-data file, never per
  subframe.
- **f32 grid / τ precision**: ρ spans orders of magnitude; f32 is ample for
  emission, and τ accumulates as a front-to-back transmittance product with
  early exit — the banding risk is in the *look*, not correctness; the
  analytic slab gate bounds it.
- **Raymarch cost at 1080p**: ~2M rays × ≤ a few hundred steps with early
  exit is comfortable on the dev GPU; QUICK renders at reduced res. The
  per-star transmittance prepass is O(N_star × steps) — trivial. If either
  ever hurts, half-res gas + upsample is the named mitigation (deferred).
- **Pairing instability / h vs spacing**: N_ngb kept ≤ ~57 for the cubic
  spline; the shock-tube lattice spacing chosen so h ≈ 1.2–1.3 spacings; the
  uniform-lattice zero-force gate is the canary.
- **CPU SPH runtime**: 2·10⁵ gas × ~50 neighbors × ~1500 steps ≈ 10¹⁰ pair
  interactions — expect tens of minutes with rayon for the full run;
  acceptable for an offline pipeline, QUICK stays seconds-to-minutes. The
  adaptive-h bisection warm-starts from last step's h (M7a) so it does not
  multiply the per-step density cost.
- **Isothermal fragility**: with no fragmentation physics, a cold gas disk
  (Q_gas < 1) clumps artificially — the fail-loud Q gate and the c_s default
  exist precisely to keep v1 in the regime the physics supports.
- **State-construction ripple** (M7a): ~9 struct-literal sites + every test
  helper; mechanical but easy to miss one — `assert_consistent` grows the new
  columns so any miss fails loudly in tests, not silently at render time.
