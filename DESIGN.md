# Galaxy Collider — Design

A headless Rust N-body engine for galaxy collisions / tidal tails, with an
offline, decoupled visualization pipeline. Start simple (collisionless,
10^5–10^6 particles); keep the architecture open to gas (SPH) and full
cosmological expansion (10^8 particles, comoving integration).

## Locked decisions

- **Language:** Rust from the start. Data-oriented (SoA), `rayon` for CPU
  parallelism, `wgpu` for GPU (force kernels later; rendering now).
- **No realtime.** Offline batch: `scenario → compute physics → compute
  visuals → result`. This decouples the renderer from the simulator entirely
  (they communicate only through files).
- **Renderer:** **wgpu additive-glow is primary** (scales to 10^8, ms/frame
  iteration, fully headless/scriptable/reproducible, physically apt for
  emissive star fields). **Blender/Cycles is a *parallel consumer*** of the
  same render-prep output, used for occasional cinematic hero shots and
  (later) volumetric *gas* frames where Cycles is best-in-class.
  - Decision driver: the 10^8 growth target — Blender does not survive that as
    a per-frame batch renderer; wgpu does.
- **Scale:** first milestone 10^5–10^6 (CPU Barnes-Hut + rayon). Architecture
  stays open to 10^8 via (a) swappable `ForceSolver` (BH → PM/TreePM) and
  (b) GPU-instanced rendering from day one.
- **Snapshots:** Rust-native main format (bincode / npy / Parquet). HDF5
  emitted **only for validation runs** (behind a feature flag) — dodges the
  Windows HDF5 C-library link landmine. If HDF5 is ever needed in the hot
  path, use `hdf5-metno` and prove the Windows build on day 1.

## Rendering recipe ("the beautiful")

- Stars/dark matter: **additive-blended Gaussian splats** into an
  **`Rgba32Float`** HDR accumulation buffer (32F, not 16F — galaxy cores
  saturate/band in 16-bit). Additive blending is **order-independent
  (commutative)** → no depth sort.
- Post: bloom (mip down/blur/up) → write **linear HDR (EXR)**.
- **Tonemap/grade is a SEPARATE config-driven stage** (ACES/Reinhard) → 16-bit
  PNG → ffmpeg. Lets you regrade 1000 frames in seconds without re-running
  physics.
- Color by **progenitor** (which galaxy + species), stellar age, or local
  velocity dispersion. Progenitor tag + additive glow = the iconic tidal-tail
  money shot.
- Gas (Stage 5): **volumetric raymarching with absorption** — the "over"
  operator, which is **ordered / NOT commutative**. A different compositing
  model from stars (emission-only vs emission+absorption). Do NOT reuse the
  splat path for gas.

## Architecture: 3-stage offline pipeline

```
scenario.toml ─▶ [sim]        ─▶ snapshots/*   (pos, vel, mass, id, progenitor + header)
snapshots/*   ─▶ [renderprep] ─▶ frame-data    (per-particle color/size/brightness; or density grid for gas)
frame-data    ─▶ [render:wgpu]─▶ frames/*.exr  (linear HDR)
frames/*.exr  ─▶ [grade]      ─▶ frames/*.png  ─▶ ffmpeg ─▶ movie.mp4
                                  └▶ (optional) Blender consumes frame-data for hero shots
```

The **render-prep stage is the decoupling boundary** — wgpu and Blender are
both just consumers of `frame-data`. Commit to the boundary, not to one
renderer.

## Cargo workspace

```
galaxy/                     (cargo workspace)
├─ core/         types, State (SoA), snapshot schema, ForceSolver/Integrator/Background traits — pure, no I/O
├─ solvers/      DirectSum (oracle), BarnesHut (workhorse)   [later: ParticleMesh, TreePM]
├─ ic/           Plummer sphere + two-galaxy Kepler-encounter collision (DONE) [next: Hernquist/NFW halo, exp disk, bulge] [later: cosmological ICs]
├─ io/           snapshot read/write: Rust-native versioned binary (DONE) [HDF5 behind a `validation` feature: later]
├─ sim/          headless engine: solver+integrator+IC+stepping loop → snapshots (DONE) [checkpoint/restart: later]
├─ renderprep/   snapshots → frame-data; spatial-tree kNN for local density/dispersion
├─ render/       wgpu: frame-data → linear HDR EXR (bin)
├─ grade/        EXR → tonemap(ACES) → 16-bit PNG (small; may merge into render)
├─ validate/     conservation + orbital-setup (always-on tests in sim/ic) + .npy export & REBOUND IAS15 cross-check harness (DONE, manual)
└─ xtask/        orchestrator: scenario.toml → sim → renderprep → render → grade → ffmpeg
```

## Contract 1 — snapshot schema (SoA, f64 compute)

Per-particle:
- `pos: DVec3` (f64), `vel: DVec3` (f64), `mass: f32`, `id: u64`, `progenitor: u16`

Header:
- `time`, `step`, `scale_factor a` (=1.0 if non-cosmological), softening `ε`,
  units, `n_particles`, `rng_seed`, `code_version`, `config_hash`

Memory: ~62 B/particle → ~6 GB at 10^8 (tight but fits); <100 MB at 10^6.
Plan at scale: mixed precision (f64 compute, f32 storage). v0 = f64 everywhere
(correctness first).

## Contract 2 — core traits

```rust
pub trait ForceSolver {
    fn accelerations(&mut self, s: &State, acc: &mut [DVec3]); // softening lives here
    fn potential_energy(&self, s: &State) -> f64;              // conservation diagnostics
}

pub trait Integrator {
    fn step(&mut self, s: &mut State, solver: &mut dyn ForceSolver,
            bg: &dyn Background, dt: f64);
}

/// Cheap insurance for cosmology. Static => a≡1, H≡0 (vanilla Newtonian leapfrog).
/// Friedmann (later) => a(t), Hubble drag 2(ȧ/a) — the cosmology lift, isolated here.
pub trait Background {
    fn scale_factor(&self, t: f64) -> f64;
    fn hubble(&self, t: f64) -> f64;
}
```

Default integrator: **leapfrog KDK** (symplectic, 2nd-order, bounded energy
error). Softening: Plummer or GADGET-style cubic spline. Small-N validation
oracle: REBOUND IAS15 (compare conserved/statistical quantities, not exact
late-time positions — N-body is chaotic).

## v0 build order (each milestone independently demoable)

- **M0** ✅ — core + DirectSum + leapfrog KDK + 2-body Kepler test + energy diagnostic (Stages 0–1)
- **M1** ✅ — BarnesHut + single equilibrium galaxy IC + "galaxy stays in equilibrium" test (Stages 2–3).
  Plummer sphere holds equilibrium over ~12 t_dyn under both DirectSum and the
  BarnesHut workhorse (Barnes 1994 opening criterion). BH validated at scale by an
  ignored smoke-test: N=30k forces match the oracle to 1.7e-3 RMS and BH runs 2.6×
  faster than direct sum (serial). rayon parallelism deferred (DESIGN prose, not an
  M1 bullet) → since landed (see M2/perf note).
- **M2** ✅ — two-galaxy collision IC → snapshots; conservation + small-N REBOUND cross-check (Stage 3).
  Two Plummer galaxies placed on a relative Kepler encounter (parabolic = Toomre
  tidal-tail case); the orbital setup is verified against an independent
  osculating-elements formula, and each galaxy keeps its Plummer profile about its
  own COM. Snapshots use a hand-rolled versioned little-endian format (`galaxy-io`,
  Contract 1; f64 pos/vel bit-exact, f32 mass storage). The `galaxy-sim` engine
  steps a collision and emits snapshots; its always-on test confirms bounded energy
  oscillation with linear/angular momentum conserved to roundoff under DirectSum.
  The REBOUND IAS15 cross-check is a **provided, manually-run harness** (`.npy`
  export + `validate/rebound/cross_check.py`), not gated in `cargo test` (REBOUND is
  an external dep; HDF5 is a Windows link landmine — bridged via NumPy). Its physics
  formulas are cross-validated against the engine to roundoff; it has not been run
  against REBOUND in this environment.
  - **rayon parallelism (landed post-M2):** the Barnes-Hut force fill runs over
    independent targets with `par_iter_mut` — **bit-exact** to the serial reference
    (no per-target sum is reassociated) and guarded by an equivalence + determinism
    test. At N=30k, parallel BH is ~22× faster than serial O(N²) DirectSum — the
    2.6× algorithmic win (tree vs direct sum) times ~8.7× from the parallel fill. The
    O(N²) softened potential (the energy diagnostic, still O(N²) even under BH) is a
    rayon reduction, equal to serial within 1e-12 relative (reductions reassociate,
    so tolerance-tested, not bit-exact). DirectSum's *force* path stays serial by
    choice (small-N oracle; its Newton's-third-law pairing would need a 2×-flops
    row-form to parallelize). Both solvers share one softened-potential kernel.
  - **parallel `Octree::build` (landed):** the build was the next Amdahl ceiling
    once the fill was parallel. `BuildMode::ParallelExact` (toggle on `BarnesHut`;
    `new` defaults to it — bit-exact, so default-on changes speed only, and
    `Serial` stays available via `with_build_mode` for single-thread debugging)
    reproduces the serial tree **bit-for-bit**. It is NOT a tolerance trade. `build_cell` recurses the same `octant`/`child_center`
    split the serial insert uses, keeps bodies ascending per bucket, and folds the
    aggregate bottom-up in octant order, so topology and every per-node
    `(mass, com, delta)` match to the bit and the whole `accelerations` path stays
    bit-exact. Each cell builds into its own arena (no shared-tree mutation → the
    "concurrent insertion" hazard is sidestepped, not solved) and is spliced with a
    child-pointer offset remap; the root bbox is a parallel min/max reduction
    (associative + exact). Large cells fan their 8 children across rayon; dense
    regions subdivide more, so task count adapts to density. Guarded by a unit test
    (structural octant-order tree compare, arena-order-independent) + integration
    force-equivalence & determinism tests, all on uniform *and* clustered clouds.
    Measured build speedup (release, best-of-5): uniform 1.85×/2.84×/3.57× at
    100k/500k/1M; clustered 1.17×/1.92×/1.95×. Sub-linear — the serial arena-splice
    copy at each level (moving descendant nodes into the parent arena, ~O(N log N)
    serial) is the remaining bottleneck. A tolerance-only **Morton bottom-up** build
    (linear arena, no splice copy, reassociated COM sums) is **deferred** as a third
    `BuildMode`, gated on this benchmark: the ~2× clustered result shows real
    headroom below core count, so it stays a live option rather than closed. See
    `barnes_hut::build_tests::bench_build` (ignored) to re-measure.
- **M3** ✅ — renderprep + wgpu render + grade → first tidal-tail movie. The full
  offline pipeline is landed end to end: `galaxy-xtask` builds a parabolic
  two-Plummer (Toomre) encounter, steps it with BarnesHut+leapfrog to snapshots,
  then renderprep→render→grade every frame and (optionally) ffmpeg→movie. Verified
  on a 6500-particle run — 61 frames showing a clear two-tone (progenitor-colored)
  interacting pair with a bridge of stripped material and tidal scatter. The camera
  is **stable across the run**: centered on the origin (zero-COM barycenter) and
  sized to a robust percentile radius (`framing_radius`), because the naive union
  AABB framed the few escaping particles and shrank the galaxies to dots. Per-frame
  EXR is retained so the sequence regrades (exposure/tonemap) without re-simulating.
  Deferred (not on the first-movie path): bloom, kNN density/velocity-dispersion
  coloring, Blender consumer, multi-camera/orbit views, a `scenario.toml` front-end.
  - **headless-wgpu feasibility spike (landed):** before building M3 around wgpu,
    a throwaway probe (`render/src/bin/spike.rs`) confirmed the risky part works on
    this box: a **headless** adapter (no surface) comes up (RTX 5090 / Vulkan), the
    `FLOAT32_BLENDABLE` device feature is available, and additive-blended Gaussian
    splats accumulate into an `Rgba32Float` offscreen target past 1.0 (32F headroom,
    no clamp), copy-to-buffer + map readback returns the pixels. This pins Contract 3
    positions to **f32** (GPU vertex layout). Portability caveat: additive blend into
    32F *requires* `FLOAT32_BLENDABLE` — not universal; adapters lacking it would need
    an `Rgba16Float` target (DESIGN rejects 16F for core banding) or compute-shader
    accumulation. wgpu 29 API notes: `multiview`→`multiview_mask`,
    `experimental_features` on `DeviceDescriptor`, `PollType::Wait` is a struct variant.
  - **Contract 3 + renderprep (landed):** the frame-data schema (`galaxy-renderprep`,
    `frame.rs`) is the decoupling boundary both wgpu and Blender consume — a versioned
    little-endian format (magic `GLXYFRAM`, v1) mirroring the snapshot layout, SoA and
    **all-f32** (pos, RGB color, size, brightness) so there is no lossy field to call
    out; count + AABB bounds are authoritative from the data on write. The `prepare`
    stage is the MVP **pure map** (no spatial tree): progenitor indexes a color palette
    (wraps modulo; empty→white), brightness = `brightness_per_mass · mass`, constant
    splat size, f64→f32 position projection, order preserved. Local density /
    velocity-dispersion coloring (needs a kNN tree) stays deferred — progenitor color is
    the money shot. Round-trip + robustness + map tests are always-on.
  - **camera plane (verified, drives the renderer):** the collision IC places the
    Kepler orbit **in the x–y plane** (`ic/collision.rs`: pericenter along +x, `r_rel`
    and `v_rel` have z=0), so the orbital-plane normal is **+Z** and a face-on camera
    looking down Z shows the tidal tails face-on (not edge-on). The renderer's default
    view axis is +Z; the view axis is a `Camera` parameter so the deferred orbit views
    are a config change, not a code change.
  - **wgpu render stage (landed):** `galaxy-render` productionizes the spike. A
    `Renderer` holds a **reusable** headless GPU context (built once, driven per frame
    — no per-frame adapter/device init, so a 1000-frame movie pays setup once);
    `render_frame` CPU-projects each particle to NDC via the `Camera`, draws it as an
    instanced Gaussian quad additively blended into an `Rgba32Float` target
    (`FLOAT32_BLENDABLE`), and reads back a **linear** `HdrImage` (256-aligned padded
    row copy, un-padded on read). No tonemapping here — that is `grade`'s job, so the
    HDR intermediate stays regradeable. Output is linear EXR via `exr` (pure-Rust, not
    a landmine). Errors are typed (`NoAdapter`/`MissingFeature`/…), never panics. CPU
    projection is the MVP choice; the world-space vertex-shader path is the 10⁸ swap.
    Tested by invariants (GPU-gated, always-on): additive **commutativity**
    (order-independent within relative tol), **flux linearity** (2× brightness → 2×
    total flux), **32F headroom** (overlap exceeds 1.0, no clamp), **centered-splat
    symmetry** (odd dims). Plus CPU camera-math and EXR round-trip tests. A left-over
    `bin/spike.rs` remains as the feasibility artifact.
- **M4+** — GPU force kernel / PM / TreePM / gas (SPH) / cosmology (Friedmann Background + periodic solver + IC pipeline)

## Validation strategy

- Analytic: Kepler 2-body (period, eccentricity).
- Conservation: total energy (use the **softened** potential), linear &
  angular momentum, COM drift. Leapfrog energy should *oscillate*, not drift.
- Equilibrium: isolated galaxy model must stay in equilibrium (tests IC
  sampling + force accuracy).
- Standard problems: Toomre & Toomre tidal tails; cold-collapse virialization
  (2T/|W| → 1). Cosmology later: Zel'dovich pancake, Santa Barbara cluster.
- REBOUND cross-check via HDF5 export (validation runs only).
