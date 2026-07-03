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
├─ solvers/      DirectSum (oracle), BarnesHut (workhorse), FlatTree (stackless octree for GPU) (DONE)   [later: ParticleMesh, TreePM]
├─ gpu/          GpuDirectSum — O(N²) direct sum; GpuTree — O(N log N) Barnes-Hut (CPU build + GPU stackless traverse); GpuLbvh / GpuLbvhFused — end-to-end GPU-resident Morton LBVH (multi-device reference / single-device fuse) (all f32, wgpu compute) (DONE) [later: cross-step state residency / TreePM / PM]
├─ ic/           Plummer sphere, Hernquist + NFW cuspy halos (closed-form + numerical-Eddington DFs), exp-disk-in-halo (cold + warm Toomre-Q), two-galaxy Kepler collision (Plummer + disk-disk w/ spin-orbit orientation + NFW–NFW) (DONE) [later: cosmological ICs]
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
- **M3** ✅ — renderprep + wgpu render + grade → first **collision** movie (the full
  offline visualization pipeline). `galaxy-xtask` builds a parabolic two-Plummer
  encounter, steps it with BarnesHut+leapfrog to snapshots, then
  renderprep→render→grade every frame and (optionally) ffmpeg→movie. Verified on a
  6500-particle run — 61 frames showing a clear two-tone (progenitor-colored)
  interacting pair with a **bridge of stripped material and diffuse tidal debris**.
  The camera is **stable across the run**: centered on the origin (zero-COM
  barycenter) and sized to a robust percentile radius (`framing_radius`), because the
  naive union AABB framed the few escaping particles and shrank the galaxies to dots.
  Per-frame EXR is retained so the sequence regrades (exposure/tonemap) without
  re-simulating.
  - **Accuracy of the "tidal-tail" goal:** classic thin Toomre tails come from
    *rotating, dynamically cold disks* (coherent disk angular momentum, resonantly
    amplified in a prograde passage). The current IC samples two **isotropic,
    non-rotating** Plummer spheres (`ic/plummer.rs`: position *and* velocity drawn in
    random directions, net momentum subtracted), so an encounter physically yields
    stripping / bridges / plumes — **not** thin curved streams. This is an IC
    property, not a render limitation: the tail *visual* is unlocked by the
    rotating-disk IC (**landed, see M3.5**), not by more rendering work.
    (Note for that tuning pass: the p98 `framing_radius` crop trims the outermost
    stripped material — loosen it when the goal is maximal tidal extent.)
  - Deferred (not on the first-movie path): bloom, kNN density/velocity-dispersion
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
    splat size, f64→f32 position projection, order preserved. This base map is the
    money shot; an **optional** density-aware pass now layers on top (**M3.6** below).
    Round-trip + robustness + map tests are always-on.
  - **Density-aware coloring (landed, M3.6) — the deferred kNN pass on top of the base
    map:** `galaxy_renderprep::density` adds a local **number-density** estimate and a
    brightness modulation driven by it, off by default (`PrepConfig.density: None` is the
    base map, bit-for-bit). `knn_density(pos, k, softening)` is the k-th nearest-neighbour
    density ρ_i = k / ((4/3)π d_{k,i}³), self excluded, brute-force **O(N²)** — the
    *reference oracle*, exactly as `DirectSum`/`reference_sort` are for their fast paths; a
    grid/tree acceleration is the named follow-up, to be gated bit-for-bit against it.
    - **Softening guards the frame against NaN.** The k-th NN distance is floored at
      `softening` **before** cubing, so collision cores / coincident particles yield a
      finite large density instead of `+∞ → NaN` that would poison a whole frame. `N ≤ k`
      (or `k == 0`) has no k-th neighbour → a defined `0.0` sentinel, not a panic.
    - **The mapping is deliberately *non-dimming*.** `density_boost` is mean-referenced,
      `boost_i = 1 + strength·(1 − ρ_ref/max(ρ_i, ρ_ref))`, bounded in `[1, 1+strength]`,
      monotone, and exactly `1` for underdense particles. This is the load-bearing design
      call: the halo dominates the density field, so a naive "denser→brighter,
      sparser→dimmer" power law would **darken the diffuse tidal tails** — the very feature
      the render exists to show. The non-dimming boost brightens cores/bridges and leaves
      the tails at full brightness.
    - **Gates:** exact hand oracles (two-particle pair distance; 1-D lattice k-th NN =
      ⌈k/2⌉·s; self-exclusion; coincident→finite via softening; inverse-cube scaling law;
      degenerate/empty/`k=0`→0; permutation-equivariance; determinism) for the estimator;
      hand values + non-dimming + bounded + monotone + identity-at-`strength=0`/all-zero
      for the boost; and end-to-end `prepare` gates (`strength=0` ≡ `density: None`; a
      dense octahedral clump brightened while sparse particles stay at base; order/count
      preserved; tiny `N≤k` state does not panic). All always-on (CPU, no GPU).
    - **Scope: estimator + mapping land and are invariant-gated; the *visual* payoff is
      not yet asserted.** Unit tests gate the density math, not "the movie shows tidal
      structure." `xtask` therefore leaves `density: None` until the mapping is tuned
      against a rendered collision frame (eyeballed), and **velocity-dispersion coloring**
      (σ_v over the same kNN neighbourhood — needs neighbour *indices*, not just d_k) stays
      the next deferred refinement.
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
- **M3.5** ✅ (IC only) — rotating exponential-disk IC (`ic/disk.rs`, `ExponentialDisk`),
  the IC that unlocks the thin Toomre tidal-tail visual M3 could not produce. A cold,
  low-mass exponential disk (surface density Σ₀e^(−R/Rd), truncated; sech² vertical
  layer) of `Progenitor(1)` particles is embedded in a **live Plummer halo/bulge**
  (`Progenitor(0)`) that carries most of the mass. The disk is placed on **cold
  near-circular orbits** with spin along **+Z**: v_φ(R) = v_c(R) from the *combined*
  enclosed mass (spherical Plummer + cylindrical disk), so the rotation curve is an
  **elementary closed form** — the exponential disk's Bessel-function potential is
  sidestepped, keeping the IC exactly checkable.
  - **Model choice: "cold kinematic"** (over warm-self-consistent / test-particle).
    The disk is **submaximal** (fiducial 10% of halo mass): the smooth halo dominates
    the rotation and dilutes the cold disk's self-gravity, which is the stabilization
    mechanism — a *maximal* cold disk has Toomre Q ≪ 1 and is **not** an equilibrium,
    so the Plummer "stays in equilibrium" gate is deliberately **not** reused.
  - **Gates:** solver-free analytic self-consistency (Σ₀ normalization, enclosed-mass
    ↔ density-derivative, v_c vs independent combined-enclosed-mass); a realization's
    radial CDF and ⟨v_φ⟩(R) on the analytic v_c; **coherent +Z spin** with zero net
    momentum/COM — the invariant that distinguishes a disk from isotropic Plummer;
    and a loose one-inner-orbit BarnesHut gate (energy + L_z conservation, bounded
    half-mass radius and thickness). Spin coherence is measured over the disk
    population (halo shot-noise ≠ disk spin); v_c compared at each bin's mean radius.
  - **Known caveats (documented, for the warm-disk follow-up):** the sech² layer has
    no vertical velocity support (v_z=0) → it is a geometric profile that settles, not
    a vertical equilibrium; the disk is fully cold (Q→0) and could fragment over the
    several orbits of a *collision* — a small in-plane dispersion is the knob to add.
  - **Collision wiring + orientation (landed):** `DiskCollision` is the disk analogue
    of `Collision` — two `ExponentialDisk` galaxies on the *same* two-body Kepler
    encounter, now factored into a shared `encounter` module so the one set of
    osculating-elements tests guards the placement for **both** collision types (the
    Plummer `Collision` delegates to it too; pure refactor, existing tests unchanged).
    Each galaxy carries an `Orientation` — the spin-orbit geometry — whose public API
    is the two Toomre angles (inclination + argument of the node), a `DQuat` under the
    hood: `prograde` (identity, spin +Z, co-rotating), `retrograde` (spin −Z),
    `inclined(i)` (tilt i off +Z about the line of nodes). A rotation is rigid, so it
    never disturbs a galaxy's internal structure or its zero-COM/zero-momentum framing.
    Four species are tagged (halo1=0, disk1=1, halo2=2, disk2=3) so the renderer colors
    the two *disks* (the tails) apart from the two halos. Gates: the shared conic
    recovery from the **combined** (disk+halo) masses; assembly (count, mass, four-way
    partition, contiguous ids, zero-COM/zero-momentum, each galaxy at its COM orbital
    state); and the orientation discriminators measured on the **disk** population's
    angular momentum — prograde disks spin +Z, `retrograde` flips a disk's L_z,
    `inclined(i)` tilts a disk's L by i off +Z, the other galaxy untouched (halo L is
    non-rotating shot noise, so it is *not* asserted). Galaxy 2 is seeded two SplitMix64
    steps clear of galaxy 1 so their internal halo/disk sub-streams never overlap.
  - **Two-disk tidal-tail movie (landed, verified):** `galaxy-xtask` now builds a
    coplanar-prograde parabolic encounter of two disk galaxies (17k particles) and runs
    the full sim→renderprep→render→grade pipeline. The rendered sequence shows two
    **thin, curved, two-tone tidal tails plus a connecting bridge** at pericenter — the
    genuine Toomre & Toomre structure the isotropic-Plummer movie could not make. The
    prograde disk resonance is the mechanism; a four-species palette (bright disks / dim
    halos, brightness tied to the disk particle mass) keeps the tails dominant.
    - **Confirmed cold-disk caveat (warm disk is the knob):** late in the passage a
      fully-cold (Q→0, v_z=0) disk diffuses and clumps — the predicted behavior, **not**
      a wiring bug. The physical knob is a small in-plane velocity dispersion, landed as
      the **warm-disk** milestone below. (How strongly the cold run visibly clumps is
      scenario-dependent: for the submaximal, halo-dominated fiducial the halo largely
      stabilizes even the cold disk, so the effect is mild there — see the movie note.)
      Do not chase this as an IC-assembly bug.
  - **Warm disk (landed):** opt-in velocity dispersion on `ExponentialDisk` via
    `with_toomre_q(q)` — the milestone that lets the disk survive the several orbits of a
    collision without the cold disk's local fragmentation, while keeping the thin
    prograde tails. The default `new(...)` disk stays fully cold (bit-identical), so
    every cold gate is untouched; `DiskCollision` passes warmth through with zero
    structural change (a warm disk is still just an `ExponentialDisk`). The kinematics
    are closed-form and exactly checkable, keeping the IC's ethos:
    - **σ_R from Toomre Q:** σ_R(R) = Q · 3.36 · G Σ(R) / κ(R), with the epicyclic
      frequency in closed form κ² = Ω² + G M'(R)/R², M'(R) = 4πR²ρ_halo(R) + 2πR Σ(R)
      (exact from the halo density + disk surface density — no numerical derivative).
    - **σ_φ** = σ_R · κ/(2Ω) (epicyclic ratio); **σ_z** = √(π G Σ hz) (self-gravitating
      sech² sheet — documented to mildly *under*-support since the halo-dominated disk
      gets extra vertical pull; the combined-potential σ_z is the forward refinement).
    - **Asymmetric drift** (mandatory, else the disk is over-supported and expands):
      v_c² − v̄_φ² = σ_R²·[κ²/(4Ω²) − 1 − d ln(ν σ_R²)/d ln R] (Binney & Tremaine
      eq. 4.228, midplane/aligned; cross-checked against the RAVE-paper bracket with
      η=0). v̄_φ² is **clamped ≥ 0** so R→0 (v_c→0, finite bracket) yields no NaN. The
      density-gradient term splits as 3·d lnΣ/d lnR (exact −R/Rd) − 2·d lnκ/d lnR; only
      d lnκ/d lnR uses a central difference of the closed-form κ, confining numerical
      differentiation to the small (few-percent) correction.
    - **Sampling:** v_R, v_z ∼ N(0,σ), v_φ = v̄_φ + N(0,σ_φ), drawn from a **separate
      third PRNG sub-stream** (mix²(seed)) via Box–Muller, so a warm and a cold disk
      with the same seed share every particle **position** — warmth perturbs only
      velocities. `DiskCollision` now reserves three streams per galaxy (galaxy 2 at
      mix³(seed)); its structural/orientation gates are seed-agnostic and stay green.
    - **Gates:** analytic self-consistency (κ vs the definitional κ² = R dΩ²/dR + 4Ω²
      via a different code path; σ_R *recovers* the input Q; the σ_φ/σ_R ratio; σ_z; the
      drift sign + O(σ_R²/v_c²) magnitude + ≥0 clamp); statistical realization recovery
      of Q, σ_z, and the ⟨v_φ⟩ lag; and the **dynamical acceptance** that proves the
      warmth did something — the warm disk holds equilibrium over an orbit, and (the
      differential that isolates the drift) removing *only* the drift from a Q=3
      realization makes the over-supported disk expand 3.0% in mean radius over two
      orbits vs 0.19% for the drifted disk (a 16× gap; the drift is load-bearing).
    - **Movie (verified):** `galaxy-xtask` now warms both disks (Q≈1.5); the two-tone
      thin tails + bridge survive intact. A same-seed cold (`Q=None`) baseline was
      rendered for comparison: the warm run's cores stay marginally more distinct and
      its tails/debris are modestly smoother, while the cold run merges to a rounder
      blob with grainier debris — but at this submaximal, halo-dominated, 1500-step
      scenario the difference is **subtle**, not the dramatic discrete-clumping the
      fully-cold *isolated*-disk caveat anticipated (the halo already stabilizes the
      cold disk here; a maximal or longer-integrated disk would separate more). The
      rigorous, non-visual evidence that the warmth is load-bearing is the exact
      kinematic gates + the dynamical drift differential (16× expansion), not the movie.
- **M4** — GPU force kernel / PM / TreePM / gas (SPH) / cosmology (Friedmann Background + periodic solver + IC pipeline)
  - **GPU direct-sum solver (landed):** `galaxy-gpu`'s `GpuDirectSum` is an exact
    O(N²) Plummer-softened direct summation run as a **wgpu compute** kernel — the same
    algebra as the CPU `DirectSum` oracle, moved to the GPU for throughput. It drops in
    behind the `ForceSolver` trait (the "swappable solvers" door), reusing a device/
    queue/pipeline built once with storage buffers grown lazily. This validates the
    GPU-**compute** infrastructure (the render stage was GPU-**graphics**) and is the
    first step of the 10⁸ scaling path.
    - **f32 is forced by the toolchain, not a design choice.** wgpu/naga has no
      portable native f64 compute (`SHADER_FLOAT64` is rarely present across backends),
      so the kernel runs in **f32** while the engine is f64. The honest lever is the
      **accumulation strategy**, and **float-float (`df64`) emulation** of the `xᵢ − xⱼ`
      difference and the accumulator is the named forward refinement for
      precision-critical runs. The dominant f32 error is *not* a uniform ~1e-6: it is
      catastrophic cancellation in `xᵢ − xⱼ` (large coordinates, close pairs) plus small
      terms swallowed while summing N contributions into one f32 accumulator — **worst**
      in the clustered, large-coordinate collision regime the GPU path is for. The gates
      pin this analytically: unit-box forces match the f64 oracle to < 3e-4 RMS, while a
      rigid offset to |x| ≈ 5000 degrades to ~5e-3 RMS (worst-pair ≈ √2·D·ε_f32/softening
      ≈ 1.7e-2) — a documented, coordinate-scale-driven precision floor, the analogue of
      "BH error grows with θ". Keep collision coordinates near the (zero-COM) origin.
    - **Gather, not scatter (determinism).** One invocation per *target* `i` loops over
      all sources `j`, accumulating in a private register and writing `accel[i]` exactly
      once — no float `atomicAdd` (whose ordering is nondeterministic). The fixed loop
      order makes it **bit-deterministic on a given device** (cross-device equality is
      *not* claimed: FMA/rounding differ), matching the parallel-BH "per-target acc never
      reassociated" discipline. Sources stream through a 256-wide workgroup tile
      (GPU-Gems N-body pattern); the self term (`dx=0`) and padded lanes (`mass=0`)
      contribute zero with no per-iteration branch. Requests **no** device features
      (baseline storage-buffer compute), so it does not narrow adapter support the way
      the renderer's `FLOAT32_BLENDABLE` does.
    - **Scope honesty.** O(N²) → realistically a few × 10⁶ particles, **not** 10⁷–10⁸.
      The 10⁸ door is a GPU *tree* / TreePM / PM solver, not brute force; this is
      infrastructure validation + an exact fast solver in the 10⁵–10⁶ band + a stepping
      stone. `potential_energy` delegates to the CPU **f64** reduction for the MVP — a
      documented inconsistency (the integrator then applies **f32** forces while energy
      is measured from an **f64** potential, so a drift diagnostic mixes a precision gap
      with integrator error; the potential is a periodic diagnostic, not the per-step
      path). The `accelerations(&State)→acc` interface also forces an upload+readback each
      step; negligible while O(N²) compute dominates, but it becomes the bottleneck for a
      future GPU-tree where state must stay GPU-resident.
    - **Gates:** equivalence vs the f64 `DirectSum` oracle at analytically-derived f32
      tolerances (unit-box + large-coordinate cancellation); same-device bit-determinism;
      Newton's-third-law momentum-flux (net internal force at the f32 floor); empty/single
      edge cases. GPU-gated (need a wgpu adapter), fail-loud like the M3 render invariants.
  - **GPU Barnes-Hut tree solver (landed, M4a):** `galaxy-gpu`'s `GpuTree` is the
    O(N log N) step past the O(N²) direct sum — the first genuine **GPU tree**. It is
    **CPU-build + GPU-traverse**: the octree is built and linearized on the CPU
    (reusing the tested build) and *walked* on the GPU by a stackless compute kernel.
    Same `ForceSolver` drop-in, same f32/determinism story as `GpuDirectSum`, now with
    the tree approximation controlled by θ (identical Barnes 1994 opening criterion as
    the CPU `BarnesHut`).
    - **Stackless skip-pointer traversal (the GPU-shaped representation).** A GPU has
      no recursion stack, so the recursive octree is linearized into
      `galaxy_solvers::FlatTree`: nodes in **DFS pre-order** (so a node's first child
      is always the next entry) each carrying a **skip pointer** `next` = the index one
      past that node's whole subtree. The per-target kernel walks with a single index:
      open a node → advance to `node+1`; accept a monopole / finish a leaf / skip an
      empty node → jump to `next`. Because a correct flatten makes the index **strictly
      increase every step**, the walk provably terminates in ≤ `n_nodes` steps — no
      stack, and no `next ≤ node` cycle that could hang the device (TDR). Leaves carry a
      `body_start`/`body_count` range into a concatenated leaf-index array and are
      resolved by exact direct sum (self term excluded); `body_count > 0` *is* the leaf
      test (every leaf holds ≥1 body, no internal node holds bodies).
    - **Gather + fixed order → determinism; reassociated vs the CPU (documented).** One
      invocation per target writes `acc[i]` once from a private accumulator in a fixed
      skip-pointer order — bit-deterministic **on a given device** (no float
      `atomicAdd`), matching the `GpuDirectSum` discipline. It is **not** bit-identical
      to the CPU `BarnesHut`: the stackless walk keeps one running accumulator over the
      DFS scan while the recursion folds each subtree separately then combines — a
      different but equally valid summation order (the exact analogue of
      `potential_energy_parallel`'s "reductions reassociate → tolerance-tested"). The
      f64 flat walk is pinned to the recursive `accel_node` at **reassociation
      precision** (observed worst gap ~1.6e-14 vs a 1e-11 bound) with the flatten
      topology pinned **exactly** (reachable node count + leaf bodies are a permutation
      of `0..n`). In f32 the opening *decision* also differs near threshold, flipping a
      few nodes → a discrete O(θ²) swing for those targets (why the GPU-vs-CPU-BH gate
      bounds RMS only).
    - **Scope honesty.** A genuine GPU *traversal* (the part that dominates at scale),
      but the **build stays on the CPU** (already rayon-parallel) and the state is
      re-uploaded each `accelerations` call — a **GPU-resident build** (Morton/LBVH) is
      the next deferred step (its CPU f64 reference / oracle landed as **M4b** below;
      the GPU port of that build is what remains), and the CPU build becomes the Amdahl
      ceiling well before 10⁸. Realistically opens the **10⁷ band** that O(N²)
      `GpuDirectSum` cannot. The
      f32 precision floor (large-coordinate `xᵢ−xⱼ` cancellation) is unchanged from the
      direct-sum kernel — the tree geometry narrows f64→f32 harmlessly (O(1e-6)); the
      dominant error is still the accumulation/cancellation, worst in the clustered,
      large-coordinate collision regime.
    - **Gates:** θ→0 reproduces the f64 `DirectSum` oracle to f32 (full open = direct
      sum, the *clean* traversal-isolation gate, no opening straddle); finite-θ error
      bounded and grows with θ; GPU-tree tracks the CPU `BarnesHut` at the same θ (RMS
      coarse guard); same-device bit-determinism; momentum-flux at the f32 floor at
      θ→0; empty/single edge cases. Plus the solvers-side f64 flatten test (bit-exact
      topology + reassociation-precision forces). GPU-gated, fail-loud.
  - **CPU LBVH reference (landed, M4b) — the oracle for the GPU-resident build:**
    `galaxy_solvers::Lbvh` is a Barnes-Hut monopole `ForceSolver` built on a **Morton-code
    Linear BVH** (Karras 2012 binary radix tree) instead of the octree. It is pure **CPU
    f64** and adds no GPU code — its purpose is to be the algorithmic + numerical reference
    the deferred GPU-resident build ports to, exactly as `FlatTree`'s CPU f64 walk is the
    oracle for `GpuTree` (one level up: the *build*, not just the traverse). The deliverable
    is the GPU-shaped build *pipeline* run in f64: bounding box → 30-bit Morton codes → sort
    by `(code, index)` → Karras binary radix tree → bottom-up aggregation → DFS skip-pointer
    flatten, then a stackless BVH walk.
    - **Why a binary radix tree, not the octree.** "LBVH" *is* the Karras binary tree:
      exactly `N` single-body leaves (Morton-sorted) + `N−1` internal nodes = `2N−1` total,
      each internal node with two children. Opening reuses the Barnes (1994) form, but the
      cell size is the node's AABB **longest side** `s = max(2·half_extents)` (a binary node
      may be non-cubic), not the octree cube's `2·half`. Because a binary node ≠ an octree
      cell, the `GpuTree` "vs CPU `BarnesHut` at the same θ" gate does **not** transfer to
      this path and is dropped, not fudged — the surviving pins are the topology-independent
      ones (θ→0, momentum flux) plus finite-θ bounded/grows.
    - **Determinism is designed in, for the future GPU sort.** Morton ties (same 1024³ cell,
      or exactly coincident particles) are broken by original index in the sort *and* by
      Karras's `δ` extending into the sorted position when codes are equal — so the tree
      topology, and therefore the forces, are a deterministic function of the input (a
      coincident-particle determinism gate pins it). That is the same tie-break a future GPU
      radix sort must implement; the bottom-up aggregation folds each node from its two
      children in fixed `(left, right)` order — the CPU analogue of the Karras atomic-*flag*
      combine (deterministic result, **no** float `atomicAdd`).
    - **Scope honesty.** No GPU code yet — this is the reference, not the GPU build. The
      build recurses (fine for the oracle; a scale build is iterative). 30-bit Morton (`u32`,
      1024³) is the first landing; **63-bit** (2× `u32` sort passes on the GPU) is the
      documented resolution refinement for the dense-core / large-coordinate regime.
      Coincident particles get distinct single-body leaves (index tie-break), not the
      octree's bucket-at-the-coincidence-floor.
    - **Gates:** θ→0 reproduces the f64 `DirectSum` oracle to roundoff (< 1e-9 worst rel
      err — the clean, **topology-independent** correctness gate); finite-θ RMS bounded and
      grows with θ (O(θ²), looser bounds than the octree gate for the longest-side `s`);
      momentum flux Σmᵢaᵢ=0 at θ→0; Karras structure (2N−1 nodes, leaves a permutation of
      `0..N`, strict binary child layout + AABB containment, all checked from the flat
      skip-pointer form); coincident-particle determinism; empty/single; and the Morton
      primitives (bit-spread, interleave, axis monotonicity). Always-on (no GPU adapter).
  - **GPU Morton + bounding-box kernel (landed, M4c) — first stage of the GPU-resident
    build:** `galaxy_gpu::GpuMortonBuilder` ports the *prologue* of `LbvhFlat::build`
    (bounding box → 30-bit Morton codes) to a two-pass wgpu **compute** stage (f32), gated
    directly against the CPU reference `galaxy_solvers::reference_morton` (extracted as the
    single source of truth for the pad/floor/scale convention). It is the smallest, lowest-
    risk slice — the analogue of how M4b was sliced off M4a — and wires into no solver yet
    (there is no `GpuLbvh`).
    - **Two passes, f32.** Pass 1 (`reduce`) folds the bbox in a **single workgroup**
      (grid-stride → shared-memory tree reduction): min/max never round and are order-
      independent, so the result is **bit-exact and deterministic with no float atomics**
      (which WGSL lacks — this is *why* the single-workgroup shape is chosen over a
      cross-workgroup atomic-min/max, whose monotone-bitcast trick is a rabbit hole for a
      reference stage). Pass 2 (`quantize`) reconstructs the **exact** CPU bbox convention
      in f32 (`center`, `half = max(0.5·ext, 1e-12)·(1+1e-9)`, `scale = 1024/size`) and
      floors+clamps each axis to `[0, 1023]`, then interleaves. The `(1+1e-9)` pad folds to
      `1.0` in f32; harmless — the `min(1023)` clamp catches the top-edge particle instead
      of the pad's nudge (a ≤1-lane effect the tolerance gate absorbs).
    - **No bit-equality vs f64; gate on lanes + determinism.** The GPU has no portable f64
      compute (same constraint as `GpuDirectSum`/`GpuTree`), so codes run in f32 and cannot
      bit-match the f64 reference near cell boundaries. Because a 1-bit lane change jumps the
      code by a large power of two, the tolerance is expressed in **lane** space: `|gpu_lane
      − ref_lane| ≤ 1` per axis in the well-conditioned near-origin regime (>95% exact). A
      large-coordinate case is **characterized, not pinned** — at `|x|≈1e6` the f32 `p−bmin`
      cancellation coarsens quantization to a max lane gap of ~6 vs the reference (the
      analogue of the direct-sum "|x|≈5000 → 5e-3" honesty).
    - **Scope honesty (stated plainly).** This proves **quantization + the reduction
      pattern**. It deliberately does **not** prove the tree matches the reference: f32
      boundary straddles mean the eventual GPU tree *topology* can differ from the CPU tree
      — the expected analogue of the `GpuTree` θ-straddle, **not** a bug. The real
      correctness check is the later θ→0 physics gate on the deferred `GpuLbvh`.
    - **Gates:** bbox reduction bit-exact vs a CPU reduction over the **same f32-narrowed**
      positions (isolates the reduction from precision; incl. the `1e-12` collinear floor);
      per-lane ±1 reference agreement near the origin; large-coordinate divergence
      characterized; codes are valid 30-bit interleaves of in-range lanes; same-device
      bit-determinism; single/coincident/empty edge cases. GPU-gated, fail-loud.
  - **GPU Morton sort (landed, M4d) — second stage of the GPU-resident build, the
    load-bearing risk:** `galaxy_gpu::GpuSorter` ports the LBVH build's sort step — `codes →
    order` by `(code, original index)` — to a wgpu **compute** LSD radix sort, gated directly
    against the CPU reference `galaxy_solvers::reference_sort` (extracted as the single source
    of truth for the tie-break, exactly as `reference_morton` was for M4c).
    - **Pure integer ⇒ the gate is bit-exact, not tolerance.** Unlike every prior GPU stage
      (f32, gated on tolerance + same-device determinism), the sort touches **no floats**:
      `u32` codes in, a `u32` permutation out. So the GPU result must equal the f64 CPU
      reference **bit-for-bit** — `order == reference_sort(codes)`, a *unique* total order
      because the reference keys on the pair `(code, index)` and `index` is unique. The real
      hazard is therefore not nondeterminism (an integer histogram commutes; a fixed-order
      scatter is deterministic by construction) but **scatter/scan correctness**.
    - **Single-invocation stable counting sort (correctness made unarguable).** `NUM_PASSES=4`
      passes of an 8-bit digit, one dispatch per pass, host-side ping-pong between two
      `(key, payload)` buffer pairs (even pass count ⇒ result back in buffer A). Each pass runs
      in a **single invocation** (`@workgroup_size(1)`): 256-bucket histogram → exclusive scan
      → **stable serial scatter** in ascending source order. With the payload seeded to `0..n`
      the stable scatter breaks code ties by ascending original index — exactly reproducing
      `reference_sort`. No atomics, no cross-invocation ordering: the single invocation buys
      *unarguable correctness*, not determinism (which is free here).
    - **Scope honesty.** This is a **reference-grade** sort, not the scale sort: one thread
      doing all the work is O(passes·N) serial. The named performance refinement (deferred,
      alongside GPU-resident state) is a **parallel stable scatter** — per-tile local ranks +
      a scanned global offset, the standard multi-workgroup radix — which reintroduces the
      scatter ordering this landing deliberately avoids; 63-bit codes stay a two-word (2× `u32`)
      pass. Land the simple correct thing, name the fast one.
    - **Gates:** `order == reference_sort` bit-exact on Morton-code clouds and uniform random
      30-bit codes; `order` a permutation of `0..n` with a non-decreasing gathered key array;
      tie-break stability (heavy-duplicate codes order by ascending index); two pass-localizing
      cases (differ only in the low byte → pass 1; only in bits 24–29 → pass 4); same-device
      bit-determinism; adversarial orderings (sorted / reversed / all-equal); large N (2¹⁶);
      empty/single edges. GPU-gated, fail-loud.
  - **GPU Karras tree-build + atomic-flag aggregation (landed, M4e) — third stage of the
    GPU-resident build:** `galaxy_gpu::GpuLbvhBuilder` ports the Karras binary-radix-tree
    build (`karras_internal`) and the bottom-up fold (`flatten`) to two wgpu **compute**
    passes, gated directly against the CPU references `galaxy_solvers::reference_karras`
    (topology) and `reference_aggregate` (fold) — extracted as the single sources of truth,
    as `reference_morton`/`reference_sort` were for M4c/M4d. It emits the raw **pointer
    tree** — per node: parent, two children (unified index: leaves `[0,N)`, internal
    `[N,2N-1)`), and aggregated AABB `min`/`max` + com + mass — and wires into no solver yet
    (there is no `GpuLbvh`).
    - **Half integer, half f32 ⇒ two gates.** The Karras **topology** is a pure-integer
      function of the sorted codes (δ = `clz(code_a ^ code_b)`, with a `32 + clz(a^b)`
      position tie-extension for equal codes), so the GPU `(left, right, parent)` must equal
      the reference **bit-for-bit** — the load-bearing gate, like the M4d sort. *This does
      not contradict the M4c f32-divergence note:* that divergence lives upstream in the
      Morton **codes**, not in this pure-integer step; fed the bit-exact `sorted_codes`, the
      topology is exact. The **aggregation** runs in f32 — AABB `min`/`max` folds never
      round and are order-independent (**bit-exact** vs an f32 CPU fold over the same
      narrowed leaves), while `com`/`mass` are f32-lossy → tolerance.
    - **The δ search is signed `i32`.** `delta` returns **−1** for out-of-range probes; a
      `u32` port would treat −1 as `0xFFFFFFFF` and win every range-boundary comparison (the
      load-bearing correctness trap). The **all-equal-codes** gate — every node on the
      position tie-break — is what surfaces it.
    - **Parallel topology, single-invocation aggregation.** `build_tree` is one invocation
      per internal node (race-free: each writes only its own children + its two children's
      parent slot — no atomics). `aggregate` is a **single invocation**: the parallel Karras
      atomic-flag walk needs a device-scope memory fence to publish a sibling's non-atomic
      AABB writes across workgroups, which WGSL 1.0 lacks (`storageBarrier` is
      workgroup-only) — so, exactly as the M4d sort collapsed to one invocation for
      *unarguable* correctness, the fold runs serially (the counter is still the Karras
      visit-**flag**: a node folds when its *second* child arrives, from its stored
      left/right in fixed order → order-independent, no float `atomicAdd`). The parallel
      atomic-flag build (with device fences) is the named scale refinement.
    - **Scope: raw pointer tree, flatten deferred.** It does **not** emit the DFS
      skip-pointer `LbvhFlat` form; deriving `center`/`half`/`delta` + the `next` skip
      pointer (a subtree-size prefix-sum / Euler-tour) is the **M4f** stage below, so the
      deferred `GpuLbvh` traverses the same form the CPU `LbvhFlat::accel` walk uses.
    - **Gates:** topology bit-exact vs `reference_karras` (Morton clouds, all-equal codes,
      heavy duplicates, monotone chain, large N 2¹⁶) + structural (2N−1 nodes, parent
      back-pointers, one parent per non-root node, `NO_PARENT` root); AABB min/max bit-exact
      + com/mass f32-tolerance vs `reference_aggregate` (incl. a monotone-chain deep-cascade
      case + coincident leaves), child AABB ⊆ parent, root bounds all; same-device
      bit-determinism (topology **and** aggregation); N=0/1/2 edges. GPU-gated, fail-loud.
  - **GPU DFS skip-pointer flatten (landed, M4f) — fourth stage of the GPU-resident
    build:** `galaxy_gpu::GpuLbvhFlattener` linearizes the M4e Karras pointer tree into the
    DFS pre-order skip-pointer `LbvhFlat` form on the GPU (wgpu compute), gated against the
    CPU reference `galaxy_solvers::reference_flatten` (extracted as the single source of
    truth, completing the `reference_morton`/`reference_sort`/`reference_karras`/
    `reference_aggregate`/`reference_flatten` set; `LbvhFlat::build` now drives that staged
    chain, so the CPU path is a stage-for-stage mirror of the GPU pipeline). This is the form
    the deferred `GpuLbvh` traversal walks with a single index — the exact shape the CPU
    `LbvhFlat::accel` walk uses.
    - **Two kernels, split like M4e's build/aggregate.** `flatten_structure` is a **single
      invocation**: the flatten is inherently serial (a DFS emission order) and WGSL 1.0 has
      no device-scope fence for a parallel Euler-tour, so — exactly as the M4e aggregation
      collapsed to one invocation — it runs the DESIGN's **subtree-size prefix** in two serial
      passes with *no recursion and no fixed-size stack*: (A) a bottom-up visit-flag climb
      (the M4e aggregate shape) computes `size[u]`; (B) a top-down pre-order walk assigns each
      node its DFS slot `d` (the emit counter) and writes `next = d + size[u]`, using an
      explicit stack **in a storage buffer** of capacity `2N−1` (a workgroup-local array
      overflows on the depth-`N−1` monotone chain — the guarded trap). `leaf_bodies[body_start]
      = order[u]` maps each sorted leaf back to its **original** particle index (the space the
      traversal excludes the self term in). `flatten_geometry` is genuinely parallel (one
      invocation per DFS slot): it derives `center`/`half`/`com`/`mass`/`delta` from the M4e
      aggregate at the slot's unified index — race-free, no atomics.
    - **Structure integer ⇒ bit-exact; geometry f32 ⇒ tolerance.** The DFS layout (`next` /
      `body_start` / `body_count` / `leaf_bodies`) is a pure-integer function of the fixed
      topology → **bit-exact** vs `reference_flatten` (the load-bearing gate — a dropped/
      double-counted subtree or a skip-pointer off-by-one shows up here). Geometry is f32:
      `center`/`half` = `(min±max)/2` (min/max exact under widening, the halving sum rounds),
      `com`/`mass` are f32-lossy folds, `delta = |com − center|` an f32-lossy sqrt → tolerance
      vs the f64 reference over the same narrowed leaves.
    - **Gates:** structure bit-exact vs `reference_flatten` (seeded clouds, monotone chain,
      all-equal codes, coincident, large N) + geometry f32-tolerance; topology-free invariants
      (a full-open skip-pointer walk — the θ→0 traversal's structural core — visits each body
      exactly once; skip pointers strictly increase; `N`/`N−1` leaf/internal counts; root spans
      the tree); same-device bit-determinism; N=0/1/2 edges. GPU-gated, fail-loud. Note WGSL
      reserves `meta`, so the slot-metadata buffer is `slot_meta`.
    - **Scope: re-uploads the pointer tree; traversal deferred.** `GpuLbvhFlattener` composes
      `GpuLbvhBuilder` and re-uploads the M4e tree to its own device (the M4d/M4e readback
      pattern). The **parallel Euler-tour flatten** and a **GPU-resident fuse** (no readback
      between build/aggregate/flatten) are the named scale refinements. Its consumer is the
      **M4g** `GpuLbvh` traversal below.
  - **GpuLbvh f32 traversal — end-to-end GPU-resident LBVH (landed, M4g) — fifth and final
    stage of the GPU port of the M4b LBVH build:** `galaxy_gpu::GpuLbvh` is the first solver to
    run the **whole f32 pipeline end-to-end** — `GpuMorton` (f32 codes) → `GpuSort` → gather →
    `GpuLbvhBuilder` → `GpuLbvhFlattener` → an f32 stackless **traversal** kernel over the M4f
    flat form. Same `(g, softening, theta)` semantics as `galaxy_solvers::Lbvh`; the traversal
    mirrors `LbvhFlat::accel` exactly but over the binary node's **per-axis `half_extents`**
    (cell size `s = 2·max(half)`, not `GpuTree`'s scalar-`half` cube — the easy pattern-match
    trap).
    - **θ→0 is where the end-to-end f32 topology straddle is finally checked.** Every earlier
      stage was gated against its CPU reference in isolation; this is the first gate on the
      whole f32 chain. The subtlety: f32 Morton can quantize a coordinate into a different cell
      than f64 (the M4c divergence), so the GPU tree's *topology* may differ from CPU `Lbvh`'s —
      but θ→0 opens every node to its leaves, so the walk *is* direct summation **regardless of
      topology**. It is therefore *insensitive* to the straddle, yet still catches any dropped/
      double-counted subtree or bad skip pointer. So θ→0 does not *assert* the topology matches;
      it shows the f32 pipeline runs end-to-end and *still* yields exact forces despite a
      possibly-different topology. (Using CPU Morton here would make the gate green but vacuous
      w.r.t. its purpose — the straddle only exists when the codes are computed in f32.)
    - **Gates:** θ→0 vs the f64 `DirectSum` oracle (RMS < 3e-4, worst < 5e-2 — the same f32
      floor as `GpuDirectSum`/`GpuTree`); finite-θ bounded + grows with θ, bounds **set from
      measurement** (max over 32 seeds ≈ 6.8e-3 at θ=0.3, 3.3e-2 at θ=0.6 — looser than
      `GpuTree`'s same-θ gate because `GpuLbvh` builds its *own* f32 tree, so whole cells differ,
      not just opening flips); tracks CPU `Lbvh` at same θ (coarse RMS); momentum-flux at θ→0;
      same-device bit-determinism; empty/single edges. Plus a **straddle-made-provable** gate:
      θ→0 alone is topology-insensitive, so it passes whether or not f32/f64 Morton ever
      actually diverged — and at the N=128 test scale they never do. So a dedicated test finds
      (at N=20000, where a particle lands within an f32 ulp of a cell boundary) seeds whose GPU
      f32 topology genuinely differs from the f64 `reference_karras` tree, asserts ≥1 such seed
      exists, then shows GpuLbvh at θ→0 *still* matches `DirectSum` on exactly those seeds — the
      straddle is thereby exercised **and** survived, not merely present in the code path.
      GPU-gated, fail-loud.
    - **Scope: reference-grade composition; single-device fuse deferred → landed as M4h.** Each
      stage owns its own wgpu device and the pointer tree / flat form round-trips through host
      memory between stages (the M4d/M4e/M4f readback pattern), so a `GpuLbvh` holds several
      devices and re-uploads between stages. The **single-device fuse** (no host round-trips
      between stages, one submit) is the named scale refinement — **landed as M4h below**;
      keeping state GPU-resident across *integrator steps* remains deferred (Remaining M4+).
  - **GpuLbvhFused — the single-device fuse of the whole M4c–M4g pipeline (landed, M4h):**
    `galaxy_gpu::GpuLbvhFused` runs the entire LBVH build+traverse on **one wgpu device in one
    submit**, where the M4g `GpuLbvh` is the *reference-grade composition* holding five separate
    devices and round-tripping the pointer tree / flat form through host memory between stages
    (~5 CPU↔GPU sync points per force eval). The fuse uploads `bodies` once, keeps every
    intermediate (f32 Morton codes → sorted order → gathered leaves → Karras pointer tree → DFS
    skip-pointer flat form) in GPU storage buffers that flow pass-to-pass, and reads back only
    the final `accel`: **one upload + one readback** — replacing the reference chain's ~5
    readback/reupload round-trips (one per stage) with a single submit (≈4 fewer sync points).
    - **Reuse over rewrite; only two trivial kernels are new.** Every stage runs the **same f32
      WGSL** as the M4g chain — the `reduce`/`quantize`/`radix_pass`/`build_tree`/`aggregate`/
      `flatten_structure`/traversal kernels are shared verbatim (their `SHADER` consts made
      `pub(crate)`, one source of truth). The only new code is a `gather`
      (`sorted_leaf[k] = bodies[order[k]]` — the host's between-stage gather in the reference)
      and a geometry kernel that writes `center`/`half`/`delta`/`com`/`mass` straight into the
      **traversal's** buffer packing (deleting the reference's host repack). The complex
      traversal kernel — the one most likely to hide a bug — is byte-for-byte the M4g kernel.
    - **Cross-pass dependencies ride wgpu's automatic barriers.** Each stage is its own compute
      pass in one encoder; the read-after-write hazards (quantize→sort→gather→build→aggregate→
      flatten→traverse) are ordered by wgpu's usage tracking — the same in-encoder multi-pass
      dependency the M4c/M4e/M4f stages already relied on, now spanning the whole pipeline.
      Buffer aliases fall out naturally: morton writes codes into the sort's key buffer A (the
      four *even* radix passes land the result back in A), and `slot_meta`
      (`[next, body_start, body_count, unified]`) doubles as the traversal's `node_meta`
      (w=unified ignored). Only the host-touched buffers (`bodies`/`idx_a`/`parent` uploaded,
      `counter` cleared, `accel`+`readback`) are stored; the intermediates are retained by the
      bind groups that reference them (grow rebuilds the set, dropping the old — no leak).
    - **Faithful-refactor gate: bit-for-bit, not a tolerance.** Because the fuse runs identical
      f32 kernels on identical inputs on a **given device** (no float `atomicAdd`, no
      order-dependent reduction), `GpuLbvhFused` reproduces the reference `GpuLbvh` forces
      **exactly** — measured `max |Δ| == 0` across θ∈{1e-6, 0.5} × N∈{128, 256, 2000} × 8 seeds
      (both the θ→0 leaf-direct-sum path and the finite-θ monopole-acceptance path, into the
      straddle regime). That is a *stronger* statement than the same-device determinism gates (a
      solver vs itself): two different device/pipeline setups on the same adapter agree to the
      bit. Same-device only; cross-device (FMA/rounding) equality is not claimed.
    - **Scope: this fuses the *build pipeline*, not cross-step residency — a latency win, not a
      speedup.** M4h keeps state GPU-resident across the **stages of one force evaluation**;
      keeping it resident across **integrator steps** (which changes the
      `accelerations(&State)→acc` interface and touches the stepping loop) stays deferred (see
      Remaining M4+). It removes ~4 CPU↔GPU sync points — the *precondition* for that residency — but
      is **not** a throughput speedup: the single-invocation serial stages (sort, aggregate,
      flatten-structure) are unchanged and stay the bottleneck; their parallel refinements remain
      deferred.
    - **Gates:** the full M4g physics suite replicated for `GpuLbvhFused` (θ→0 vs the f64
      `DirectSum` oracle to the f32 floor; finite-θ bounded + grows; tracks CPU `Lbvh` at same θ;
      momentum flux at θ→0; same-device bit-determinism; empty/single; and the real
      topology-straddle survival at N=20000) **plus** the bit-for-bit faithful-refactor gate vs
      the reference `GpuLbvh`. GPU-gated, fail-loud.
  - **GpuResidentLeapfrog — cross-step state residency (landed, M4i):** `galaxy_gpu::GpuResidentLeapfrog`
    is the payoff M4h unlocked: it keeps `pos`/`vel`/`mass`/`acc` in GPU storage buffers **across
    integrator steps**, runs the leapfrog KDK kick/drift on the device, and reads nothing back
    until an explicit `snapshot`. M4h still uploaded state and read back accel *every*
    `accelerations` call (one CPU↔GPU round-trip per force eval); M4i removes that per-step
    round-trip entirely. Lifecycle is `upload → step* → snapshot`, **not** `ForceSolver` — the
    `accelerations(&State)→acc` interface is host-state-in / accel-out, fundamentally incompatible
    with residency, so this is its own type (exactly as this Remaining-M4+ item anticipated).
    - **Reuse over rewrite: the shared `FusedCore`.** The whole M4c–M4g build+traverse pipeline
      (device, pipelines, layouts, lazily-sized buffers, the pass sequence) was factored out of
      `GpuLbvhFused` into a `pub(crate) FusedCore` that **both** the fused solver and the resident
      stepper drive — same f32 WGSL, one source of truth. The refactor is gated: the full M4h
      suite (incl. the bit-for-bit `GpuLbvhFused`-vs-`GpuLbvh` gate) still passes unchanged. The
      only *new* code is a `vel` buffer, three trivial kernels (`kick`: `v+=a·½dt`; `drift`:
      `x+=v·dt`; `reset`: re-seed `idx_a` iota + `parent`=`NO_PARENT` on the GPU each force eval,
      the on-device equivalent of the fused solver's per-call host writes), and a no-readback step
      encoder. `bodies` (xyz=pos, w=mass) — already the traversal input — **doubles as the resident
      position buffer**, so drift mutates the state the force pipeline reads in place.
    - **The precision cost is real and documented.** The host path keeps *authoritative* positions
      in **f64** and re-narrows each step; the resident path accumulates `pos += vel·dt` in **f32
      across every step** (no portable `f64` in WGSL). So energy drifts more than the f64 leapfrog's
      clean bounded oscillation — acceptable for the f32 render money-shot, and mirroring the
      existing f32-force / f64-energy note. **Double-single (float-float) position accumulation is
      the deferred precision follow-up.** This is a **latency/architecture win** (per-step sync
      points removed), **not** a throughput speedup: the serial M4h stages (sort, aggregate,
      flatten) are unchanged, and each `step` is still one submit — **batching K steps into one
      encoder** (dropping per-submit overhead too) is the named follow-up, distinct from residency
      (landed as M4k below).
    - **Gates (the two load-bearing ones + invariants).** (1) *Faithful/residency — bit-for-bit:*
      the same stepper run resident (`upload → K steps → snapshot`) vs round-tripped (per step:
      `upload → step → snapshot`) agrees **exactly**, since f32↔f64 snapshot/upload is a lossless
      identity — this is M4h's faithful-refactor claim lifted to the step loop, and any divergence
      *is* a residency bug (stale buffer / missing barrier / stale acc). (2) *Physics — f32 tol:*
      tracks the host-driven `LeapfrogKdk + GpuLbvhFused` reference (which holds the force kernel
      identical, so only f32-GPU-KDK vs f64-host-KDK varies). Plus momentum conservation at θ→0
      (where the tree forces are exact antisymmetric direct sums), bounded energy over a long run,
      zero-step f32-narrowing identity, empty/single (a lone particle drifts ballistically),
      same-device determinism, and time bookkeeping. GPU-gated, fail-loud.
  - **Double-single position accumulation (landed, M4j):** the M4i precision follow-up. WGSL has no
    portable `f64`, so resident positions are now carried as a **double-single** (`hi + lo`, an
    unevaluated f32 pair ≈ 46-bit mantissa): the drift kernel accumulates `pos += vel*dt` with a
    compensated Knuth two-sum + quick-two-sum renormalize, so the small per-step increment is no
    longer swallowed by the growing coordinate's f32 ulp. `hi` is `bodies.xyz` — the force pipeline
    still reads only that f32, so build/traverse (and their gates) are untouched and the *force*
    stays f32; `lo` is a resident-only buffer. `upload` splits the f64 input into `hi + lo` and
    `snapshot` sums them back, so the accumulated precision reaches the host. Velocity stays plain
    f32 (DS is position-only, matching this item's scope).
    - **The reassociation trap (and the fix).** The two error-free transforms rely on IEEE
      non-associativity, which consumer-GPU f32 compilers break by default: `(hi+d)−hi → d` and
      `(s+e)−s → e` fold the compensation to *exactly zero* (measured — without a barrier the DS
      result was bit-identical to a plain-f32 running sum). A plain `bitcast<f32>(bitcast<u32>(x))`
      launder was **not** enough (naga folds it). The working barrier XORs the bits with a **runtime
      uniform pinned to 0**: the compiler can't prove it's identity, so it can't fold the round-trip
      — forcing the true IEEE-rounded intermediate. **Verified on the Vulkan CI adapter; DX12 was
      not exercisable on the test machine (it fell back to Vulkan). GPU emulated-double is
      driver-dependent** — the gate proves it on the CI adapter, not universally (this mirrors the
      existing f32-force / f64-energy caveat; production large-world systems origin-rebase for
      exactly this reason).
    - **Gate cost: M4i's exact faithful gate relaxes to a tolerance.** M4i asserted resident-vs-
      round-trip *bit-for-bit* because snapshot↔upload was a lossless f32 identity. DS retires that
      premise: at a tie (`|lo| = ½ulp(hi)`) the single-f64 snapshot channel can't preserve which
      `(hi, lo)` split produced the value, so the resident path (carrying `lo` across steps) and the
      round-trip path (recombine→resplit each step) diverge in the **`lo` limb** at f64-epsilon scale
      (measured dp ≈ 1.8e-15, dv = 0). The gate now bounds position drift `< 1e-12` (sized ≈ `K·
      ulp(lo)` with headroom, ~5 orders below an f32 ulp, so every real residency bug still trips it)
      and keeps velocity exact. **New gate:** a force-free single particle drifts K=10⁴ steps and the
      DS accumulator tracks the exact f64 sum `x₀ + K·fl32(v·dt)` within 1e-5 (a plain-f32 running
      sum drifts to ~3.5e-1). GPU-gated, fail-loud.
  - **Batched multi-step submits (landed, M4k):** the remaining M4i throughput follow-up. M4i
    removed the per-step *latency* (round-trips) but left each `step` its own submit; `step_many`
    now coalesces up to `MAX_BATCH` steps into a **single encoder/submit** (`⌈steps/MAX_BATCH⌉`
    submits total), dropping the per-submit overhead. `step` stays the one-submit minimum-latency
    path. Batching only regroups encoders — wgpu's usage tracking inserts the same read-after-write
    barriers *between* steps (drift→force on `bodies`, close-kick→next drift on `vel`) that it
    already inserts within a step — so the trajectory is **bit-identical** to stepping one at a
    time (verified: 150 batched steps == 150 per-step, bit-for-bit); the two half-kicks are kept
    **unfused** across the step boundary (`kick½(a)·kick½(a)` ≠ f32-fused `kick(a·dt)`). The cap is
    a TDR/watchdog guard, **not** a throughput target: even 64 collapses the K=10⁴ drift gate from
    10⁴ submits to ~157. It is a fixed step count, hence **N-blind** — per-step GPU cost scales with
    N, so *in principle* a large-N sim could approach the watchdog at a cap safe for the small-N
    gates; whether that actually bites was measured in **M4l** (it does not, to ≥1M). **Gate:**
    `step_many` issues exactly `⌈K/MAX_BATCH⌉` submits (a before/after `submits()` delta); the
    pre-existing nine gates re-validate the trajectory under batching for free. GPU-gated, fail-loud.
  - **`MAX_BATCH` timing measurement — fixed cap is enough (M4l):** M4k left the per-N question
    open. Rather than guess a cost model, the `bench_step_cost` timing bench *measured* per-step
    resident-KDK wall-clock across N (RTX 5090 / Vulkan). The resident step turns out to be
    **overhead-bound** — its ~10 serial LBVH dispatches (reduce/quantize/sort/build/aggregate/
    flatten/traverse + the KDK kernels) dominate, not N-scaling compute — so per-step cost stays
    essentially flat (~0.1–0.4 ms, noise-dominated, no monotone N-trend) while N grows 2048×
    (512→1 048 576). A full 64-step submit is ≤ ~25 ms even at 1M particles, **≥20× under** a
    conservative 500 ms watchdog budget (¼ of the ~2 s TDR).
    So the fixed `MAX_BATCH = 64` is measured-safe to **≥1M**, and per-N *adaptive* sizing was
    **dropped as unnecessary**: at any N_ref honest to this data the cap never leaves 64 in the
    practical range, and an `n·log n` guard beyond it would over-shrink the batch ~10× against the
    measured (wildly sub-`n·log n`) growth — a disproven model, not a safeguard. Deferred until a
    real **10⁷–10⁸ crossover measurement** can set a knee from data. Caveat: the bench IC is a
    diffuse cluster; a concentrated distribution stresses traversal more, but the ≥10× headroom
    absorbs a 3–5× worse case. Deliverable is the ignored bench itself (the evidence), not new
    always-on gates — the existing M4k submit gate is unchanged.
  - **Remaining M4+:** the still-untouched **PM / TreePM / gas (SPH) / cosmology** (Friedmann
    Background + periodic solver + IC pipeline). M4i/M4j/M4k keep the *leapfrog + LBVH* path
    resident and throughput-tuned; extending residency to those solvers/integrators is future work.

- **M5** — **cuspy halo ICs (Hernquist + NFW)**, the analytic stand-ins for CDM halos/bulges
  that the cored Plummer sphere cannot represent (both have a central ρ ∝ r⁻¹ **cusp**). Built
  as a three-rung TDD **validation ladder** — each rung's oracle is the rung below, so NFW ends
  up honest despite having no closed-form answer to check against. Lives in `ic/` (pure).
  - **Hernquist analytic model (landed, M5a):** `ic::Hernquist`, the finite-mass twin of Plummer
    with a **closed-form isotropic DF** f(ℰ) (Hernquist 1990 eq. 17 / Binney & Tremaine §4.3
    eq. 4.51, in physical units — the explicit M and G factors matter). Mirrors the Plummer
    sampler (invert M(<r) for positions, rejection-sample speeds) with two profile-forced
    departures: (i) the q=v/v_esc substitution does NOT make the speed PDF radius-independent
    for Hernquist, so the rejection ceiling is found **per radius** by a grid scan; (ii) the
    r⁻⁴ tail has a **divergent first moment** ⟨r⟩, so the finite-N COM is dominated by the
    farthest particle and recentering drags the cusp off origin — fixed by **truncating at
    r_max = 300a** (sampling X uniform on [0, M(<r_max)/M) is exact; keeps 99.34 % of the mass).
    The DF's normalization is pinned with no external constant by the **density-recovery
    integral** ρ(r) = 4π ∫₀^Ψ f(ℰ)√(2(Ψ−ℰ)) dℰ; a **stability run** (sample → evolve ~12 t_dyn
    under DirectSum + BarnesHut, r_h stays put) is the only check of the velocity *shape* (a
    mis-scaled DF passes a single-snapshot virial check). Structure judged at r_h ≫ ε since the
    cusp is unresolved by Plummer softening.
  - **Numerical Eddington-inversion DF (landed, M5b):** `ic::eddington::EddingtonDf`, the reusable
    isotropic-DF builder (B&T eq. 4.46b) that recovers f(ℰ) from ρ and Ψ **alone** — the tool
    that gives NFW (no closed-form DF) an honest equilibrium instead of a local-Maxwellian
    approximation. Numerics chosen for stability: tabulate dρ/dΨ on a log-spaced radius grid
    (analytic-quality central differences, no per-point root inversion); kill the 1/√(ℰ−Ψ)
    endpoint singularity with Ψ = ℰ − u² → the smooth I(ℰ) = 2∫₀^√ℰ (dρ/dΨ)(ℰ−u²) du (Simpson);
    f(ℰ) = (1/√8 π²) dI/dℰ by central difference, clamped ≥ 0 and interpolated. **Validated with
    no oracle of its own** against the two models whose f(ℰ) IS closed-form: it reproduces
    `Hernquist::df` (itself validated in M5a, independent of Eddington) to < 3 % — pinning shape
    AND normalization — and Plummer's f(ℰ) ∝ ℰ^(7/2) shape to < 3 %.
  - **NFW truncated halo (landed, M5c):** `ic::Nfw`, the near-universal CDM halo (ρ ∝ r⁻¹ cusp,
    r⁻³ envelope ⇒ **divergent total mass**, **no closed-form DF**) — the payoff of M5b.
    Parameterized by (M_vir, r_s, concentration c): r_vir = c·r_s, M_s = M_vir/[ln(1+c)−c/(1+c)].
    Positions bisect the mass profile **truncated at r_vir** (truncation keeps ⟨r⟩ finite so the
    COM is well-conditioned); velocities come from the M5b Eddington DF of the **untruncated**
    potential (standard practice — truncation perturbs only the outermost shells). The DF-shape
    validation is delegated to M5b + an **Eddington-recovers-NFW-density** integral (the M5b
    machinery exercised on a model with no analytic DF); the stability run then confirms the
    *assembled* truncated IC holds together (r_h stable under DirectSum + BarnesHut), with
    tolerances budgeting the untruncated-DF edge re-virialization. Method + truncation follow
    Kazantzidis et al. 2004.
  - **Exponentially-truncated NFW — self-consistent (landed, M5d):** `ic::TruncatedNfw`, the
    Springel & White (1999) smooth truncation of the M5c halo: NFW inside r_vir, continued beyond
    it by ρ(r) = ρ_NFW(r_vir)·(r/r_vir)^ε·exp(−(r−r_vir)/r_d). The **decay length r_d is the free
    knob**; the exponent ε is *fixed by continuity of the logarithmic slope* at r_vir —
    ε = r_vir/r_d − (r_s+3r_vir)/(r_s+r_vir) — so both ρ and dρ/dr are continuous there (gated) and
    the total mass is now **finite** (a modest skirt beyond M_vir). This is the **self-consistent
    (Path A)** upgrade over M5c: velocities come from the Eddington DF of the **truncated** (ρ, Ψ)
    pair, not the untruncated potential, so positions and velocities share one potential and the
    M5c "outer halo re-virializes because the DF is untruncated" caveat is **gone** (the stability
    run holds the 90% Lagrangian radius to a *tighter* band than M5c, 0.18 vs 0.25). Why smooth-cut
    at all: a hard edge has no well-behaved equilibrium DF; a smooth one does.
    - **The numerical potential is the new machinery.** The truncated potential has no closed form
      (the outer skirt integral is incomplete-gamma-like), so Ψ(r) = G M(<r)/r + 4πG ∫_r^∞ ρ s ds is
      semi-analytic: **closed-form for r ≤ r_vir** (NFW mass + the closed ∫ρ_NFW s ds + a *constant*
      skirt tail whose r-derivative vanishes), numerical only in the outer skirt. The skirt
      quadrature uses a **fixed** Simpson panel count (not r-dependent) so its error is *smooth in
      the integration limit* — critical, because Eddington takes dρ/dΨ by finite difference and a
      node-stepping (non-smooth) error would be noise-amplified into a **negative, silently-clamped
      DF**. A dedicated gate asserts f(ℰ) > 0 strictly *without* hitting the ≥0 clamp (a known
      failure mode: too-sharp an r_d drives f negative; r_d = 0.3 r_vir is positivity-safe).
    - **Counterintuitive but correct:** the truncated central potential is *shallower* than the
      untruncated NFW (a gated Path-A fingerprint) — Ψ(0) = 4πG∫₀^∞ ρ s ds is a *convergent*
      integral even for NFW, and the exp skirt carries **less** ∫ρ s ds than the r⁻³ tail it
      replaces (equal slope at r_vir, then steepening as −1/r_d). The skirt still *adds* mass versus
      the M5c hard cut; the two facts weight radius differently (∫ρ s² vs ∫ρ s).
    - **Validation ladder (all closed-form / hand-integral oracles):** ε = slope-continuity value;
      ρ and log-slope continuous at r_vir; skirt matches its closed form; dM/dr = 4πr²ρ across both
      regions; finite total mass; **Eddington recovers the truncated density** (the M5b machinery on
      the numerical truncated potential); DF strictly positive (no clamp); realized truncated
      mass-CDF; realized **Jeans** dispersion (self-consistent ⇒ tight 6%); recentered / equal-mass
      / deterministic; plus the evolve-and-stay-put stability run. Method: Springel & White 1999.
  - **NFW–NFW collision IC (landed, M5e) — the demoable payoff:** `ic::NfwCollision` puts two
    exponentially-truncated NFW halos ([`TruncatedNfw`], M5d) on a relative two-body Kepler
    encounter. An NFW halo is spherical, isotropic and non-rotating, so this is the direct
    analogue of the Plummer [`Collision`] (two spheres, two progenitors, **no** spin-orbit
    orientation and **no** multi-species split — unlike the rotating [`DiskCollision`]). It
    delegates the orbital placement to the shared `encounter` module, so the one set of
    osculating-elements tests already guards this conic too. Two progenitors (halo1=0, halo2=1),
    contiguous ids, global zero-COM/zero-momentum frame, galaxy 1's particles first.
    - **The load-bearing detail: orbit on the FULL mass, not M_vir.** `TruncatedNfw::sample`
      places particles summing to `total_mass()` (virial + exponential skirt), so the two-body
      `mu` and the COM split use `total_mass()` — the mass actually present — not the "canonical"
      `M_vir`. Setting the orbit for `M_vir` would leave the realized velocities wrong for the
      intended conic (the final recenter would hide the momentum inconsistency but not fix it).
    - **Why the M5d truncated halo, not the hard-cut M5c `Nfw`.** A collision is exactly the
      regime the M5c caveat bites: M5c samples velocities from the *untruncated* DF, so its outer
      halo re-virializes — and the outer halo is the material tidally stripped into the bridges/
      debris the demo exists to show. M5d's self-consistent (ρ, Ψ) makes those outskirts a genuine
      equilibrium before the encounter perturbs them.
    - **Gates (all t=0 — a collision is meant to move):** the shared conic recovered from the
      **combined** `total_mass()` (bound/parabolic/hyperbolic; eccentricity-vector → +x pericenter;
      COM split into the zero-momentum frame); assembly (count, total mass = sum of full masses,
      two-progenitor partition, contiguous unique ids, global zero-COM/zero-momentum frame); each
      halo keeping its truncated-NFW profile (median radius about its own displaced COM vs the
      half-mass radius from inverting `enclosed_mass`) and internal dispersion (mean-subtracted
      ⟨v²⟩ vs an isolated realization — the bulk boost must not leak in); **exact** rigid placement
      (each halo is internally recentered by `sample` *before* placement, so its realized COM/bulk
      velocity track the requested orbital state to roundoff, 1e-9 — not merely to sampling noise);
      one-mix-step seeding independence; determinism. Each halo's *dynamical* equilibrium is already
      gated by `nfw_truncated_stability.rs`, so no evolve-and-stay-put run here. Method: Toomre &
      Toomre 1972 (parabolic encounter); Kazantzidis et al. 2004 (halo–halo mergers).
  - **DM-merger movie (landed) — the M5e visualization payoff.** `NfwCollision` is now
    wired into `galaxy-xtask` as the `dm` scenario (`cargo run -p galaxy-xtask --release dm`),
    the dark-matter analogue of the M3 disk-collision movie. Two exponentially-truncated NFW
    halos (M_vir 1.0 + 0.5, a **2:1 major merger**; r_vir 10 + 8) start on a **parabolic**
    (Toomre) encounter at COM separation 40 (> r_vir₁+r_vir₂ so they begin on a clean
    approach) with pericenter 3 (≪ r_vir, a **deep, fully-overlapping** passage). Particle
    counts split 2:1 (12k + 6k) to give **equal particle mass** across both halos, so a
    two-tone (warm-primary / cool-secondary) palette weights brightness uniformly and the
    ρ∝r⁻¹ cusps additively saturate their cores to white. Verified over the full arc (dt=0.02
    ≈ 0.016·t_dyn, T=320 ≈ 3·t_peri; t_peri≈104 by Barker's equation): two separated cusps →
    deep pericenter → **bound** recession (post-peri separation ≈15 ≪ the initial 40 — a
    point-mass parabolic orbit would return to 40; the deficit is orbital energy lost to
    **dynamical friction** in the overlap) → second infall → coalescence into a **single
    triaxial remnant** with mixed warm/cool populations and a diffuse stripped-debris halo.
    This is the correct DM major-merger endpoint — two cuspy blobs merging, **not** thin
    tidal tails (those need cold *disks*, cf. the `disk` scenario). Structural refactor: the
    xtask pipeline (sim→prep→render→grade→ffmpeg) is now single-sourced over a `Scenario`
    struct with `disk`/`dm` selected by the first CLI arg (the original disk movie is
    unchanged — same constants + deterministic pipeline — now behind `disk` and the default);
    `GALAXY_MOVIE_QUICK=1` gives a low-N,
    low-res, same-physics preview for scenario iteration. The orchestrator stays a
    **test-exempt I/O glue** binary: its pure helpers (`framing_radius`, `union_bounds`) are
    unit-tested in `xtask/lib.rs` and every physics invariant (total mass, two progenitors,
    zero-COM, exact rigid placement, internal equilibrium) is already gated in
    `ic/tests/nfw_collision.rs` + `nfw_truncated_stability.rs`, so re-testing here is redundant.

- **Cuspy-halo disk galaxy (landed, M5f) — the halo abstracted behind a trait.**
  The M3.5 `ExponentialDisk` was hard-wired to a `Plummer` halo. `SphericalHalo`
  (`ic/halo.rs`) names the exact surface the disk reads from its halo — `g()`,
  `total_mass()`, `density(r)`, `enclosed_mass(r)`, `sample(n, seed)` — so
  `ExponentialDisk<H = Plummer>` is now generic over it and a cold disk can sit in a
  cuspy [`Hernquist`]/[`Nfw`]/[`TruncatedNfw`] halo, not only the cored Plummer. The
  **default type parameter** keeps every existing `ExponentialDisk` mention meaning
  `<Plummer>` (zero blast radius; `DiskCollision`, xtask, and all M3.5 tests compile
  unchanged); each impl **forwards to the model's inherent methods** via a
  type-qualified path (`Nfw::density(self, r)`) so a trait method can never recurse.
  Mirrors the swappable-`ForceSolver` pattern. One deliberate non-uniformity: the
  untruncated NFW's total mass **diverges**, so `Nfw::total_mass()` returns `M_vir`,
  exactly what `Nfw::sample` realizes (truncated at r_vir).
  - **The payoff — a realistic rotation curve.** A cuspy halo's M(<r) rises steeply
    from the center, so the disk's rotation curve **rises to a flat plateau** (the CDM
    -galaxy shape) instead of turning over as it does in a Plummer core.
  - **Scope: the COLD disk** (no Toomre warmth this round). A cold disk on circular
    orbits is an equilibrium in *any* spherical potential — the honest first
    increment. The warm path leans on ρ(r), which **diverges** at a cusp, so warm-in
    -a-cusp is a deliberate follow-up.
  - **Gates.** Analytic self-consistency: v_c(R) matches √(G·[M_halo(<R)+M_disk(<R)]/R)
    against **independently hand-derived** cuspy enclosed masses — Hernquist
    M(<r)=M r²/(r+a)² and the NFW closed form — the discriminating check that the
    right (cuspy) mass profile feeds the disk. Plus a realization's tags/order, spin,
    zero-COM, and ⟨v_φ⟩(R) on the analytic v_c. The **NFW variant proves the trait
    genuinely abstracts multiple cuspy halos**, not just Hernquist.
  - **Finding — a cusp must be RESOLVED for the live-halo stability gate.** Unlike the
    cored Plummer disk (holds at N_halo=1000, ε=0.05·r_s), a cold disk in a *live*
    N-body cusp needs a **smaller softening fraction and more halo particles**
    (fiducial N_halo=6000, ε=0.01·r_s): at low resolution the N-body inward force in
    the inner cusp falls **several-fold below** the analytic G·M(<r)/r² the disk is
    placed on, so the disk over-rotates and flies apart (r_half drifts ~80% in one
    orbit). This is a resolution/softening artifact of the live halo, **not** an IC
    defect — the sampling is exact (analytic gates), the disk just reads a smooth
    force the under-resolved cusp doesn't deliver ("judge structure outside ε"). At
    resolution the disk holds: the gate checks the half-mass radius **and the 90%
    Lagrangian radius** (the latter catches a minority of inner, least-resolved
    particles blowing out that a median would mask) — both hold to <10%, E and L_z
    conserved. RMS thickness is a *loose* sanity bound only: the cold v_z=0 sheet is a
    geometric layer, not a vertical equilibrium, so it settles/phase-mixes vertically,
    more so in the steeper cuspy field.
- **Cuspy-disk collision (landed, M5g) — tidal tails on a cusp, the demoable payoff.**
  M5f abstracted the disk's halo behind `SphericalHalo`; M5g abstracts the *collision*
  the same way. `DiskCollision<H = Plummer>` (`ic/disk_collision.rs`) is now generic
  over the halo type, so two rotating disks can encounter each other inside live cuspy
  [`TruncatedNfw`] halos, not only cored Plummer. The **default type parameter** keeps
  every existing `DiskCollision` mention meaning `<Plummer>` (the M3 disk suite and the
  xtask `disk` scenario compile unchanged); the placement code is pure forwarding to
  `ExponentialDisk<H>`'s already-tested generic surface, and `place_galaxy` operates on
  the sampled `State`, so nothing in the assembly is halo-specific. This is the disk
  analogue of the M5e `NfwCollision` (which is pure blobs) and the cuspy analogue of the
  M3 `disk` movie.
  - **No red-first — deliberately.** Widening a type parameter is a *mechanical,
    behavior-preserving* change: there is no honest failing state (a `todo!()` inserted
    only to manufacture a red would be the very fake-impl anti-pattern the TDD rule
    forbids — confirmed with the reviewer). The safety net is instead that the existing
    Plummer `disk_collision.rs` suite **stays green**, proving the widening preserves
    behavior. The new `disk_collision_cuspy.rs` gates only what the generalization newly
    touches — the orbit recovering its conic from the *cuspy* combined masses, four-
    progenitor assembly / zero-COM from cuspy-sampled halos, the disks' surviving +Z
    spin. It does **not** re-derive the ⟨v_φ⟩(R) rotation curve: `sample` = (cuspy
    `ExponentialDisk::sample`, gated in `disk_in_cuspy_halo.rs`) ∘ (rigid placement,
    gated in `disk_collision.rs`), so re-testing it is the redundancy this doc calls out
    for the M5e case.
  - **The payoff scenario (`xtask cuspy`).** Two cold disks in truncated-NFW halos on a
    parabolic prograde passage. A QUICK preview shows the classic Toomre sequence: two
    coherent disks approach → an **S-shaped tidal bridge-and-tails** at first pericenter
    → the cores stay intact and bright as the pair separates. Genuine tidal features,
    not the DM-merger's blobs.
  - **Two cusp-forced choices in the scenario.** (1) The disks are **COLD**. The Plummer
    `disk` movie runs *warm* (Toomre Q≈1.5) to survive several orbits, but that knob
    reads ρ(r) — which **diverges at the cusp** — so warm-in-a-cusp stays the scoped
    follow-up; a cold disk on circular orbits is an equilibrium in any spherical
    potential, the honest increment. (2) The stabilizer is therefore **resolution, not
    warmth** (the M5f finding carries straight over): the halos are particle-heavy
    (nh 10000+8000 full, 5000+4000 quick — kept high *even in QUICK* so the low-N preview
    isn't a false negative for a cusp that only holds when resolved) with ε=0.02·r_s
    (between the disk movie's 0.05 and M5f's deep-cusp 0.01). Framed at **p70**, not
    p98, so the far-larger halo skirt (r_vir=10 vs disk r_max≈1.8) is cropped and the
    disk + tails fill the frame while the dim halo glows underneath.
  - [next: the warm cuspy disk once ρ(r)'s cusp divergence is handled (unlocks Q≈1.5
    survival over *several* orbits, i.e. a full merger rather than a single flyby); a
    `scenario.toml` front-end so these hardcoded scenarios become data.]
- **Grade toolkit: asinh stretch + `regrade` loop + density boost ON (landed, M6a) —
  the switched-off wins.** First session of the M6 visual series: exploit what was
  already built (retained linear EXRs, the invariant-gated M3.6 density estimator)
  and establish the seconds-cheap look-iteration loop every later session leans on.
  - **`ToneMap::Asinh { beta }`** (`grade`): the Lupton-style astro stretch
    `f(x; β) = β·asinh(x/β)`, clamped to `[0, 1]`. Linear (unit slope) below the
    softening knob β, logarithmic above — so exposure can be pushed hard enough to
    reveal the faint tidal tails while the log regime holds the additive cores far
    below where ACES/Reinhard flat-line (at x=100: Reinhard ≈0.990, asinh(β=0.1)
    ≈0.760 — that ordering is a gate). Other gates: hand values
    (asinh(1)=ln(1+√2)), monotone + `[0,1]` over a geometric HDR sweep, β→large
    recovers the identity at small x (`asinh(u)=u−u³/6+…`), and β floored at
    `f32::MIN_POSITIVE` so a degenerate `β=0` stays total (`0·asinh(∞)` would NaN a
    whole frame). `ToneMap` drops `Eq` for the f32 payload (`PartialEq` stays).
  - **`xtask regrade <exr_dir> <png_dir> [--exposure E --tonemap
    aces|reinhard|asinh --beta B]`**: grades every retained `.exr` into same-stem
    PNGs and (ffmpeg permitting) muxes `png_dir/movie.mp4` — self-contained in the
    target dir, never clobbers the original render's frames. ~61 frames regrade in
    seconds; this is the loop that makes grade-time iteration (and M6b's bloom
    tuning) cheap. The arg→`GradeConfig` mapping (`parse_regrade_args`) is pure and
    unit-gated (defaults, order-independent flags, fail-fast on unknown
    flags/tonemaps, non-positive numbers, `--beta` without asinh); the binary stays
    test-exempt I/O glue. Movie defaults are unchanged (ACES, exposure 1) — asinh
    is the regrade-time look; for the cuspy QUICK preview, `--exposure 4 --tonemap
    asinh` (β=0.2, the `DEFAULT_ASINH_BETA`) is the documented starting point:
    tails and bridge clearly lifted, cores intact, while exposure 8 / β=0.1 washes
    the halo shot-noise dots into the foreground.
  - **Density boost ON in all three scenarios** — `DensityColoring { k: 32,
    softening: ε (per scenario), strength: 3.0 }` — the exact follow-up the M3.6
    scope note promised, tuned by A/B against rendered QUICK cuspy frames
    (strengths 0/1.5/3/6 on bit-identical snapshots; a retained pre-M6a run gave
    the boost-off control). **Tuning finding:** the mean reference ρ_ref is
    dominated by the dense inner disk (cuspy frame 35: ρ_ref≈2.8e3 ≈ the disk's
    median, only ~3–4% of *halo* particles exceed it), so the boost acts on nuclei
    and inner-disk knots and leaves bridge/tails at base brightness (they sit below
    ρ_ref — the non-dimming design working as intended; tail reveal is the asinh
    grade's job, not the boost's). Consequences: strength 1.5 is *invisible* (its
    boosted pixels were already tone-curve-saturated — graded A/B frames come out
    statistically identical), 3.0 makes the nuclei read as bright compact cores
    instead of grainy patches, 6 blows them into structureless blobs. k=32 (top of
    the documented 8–32 band) halves the estimator's shot noise vs k=16 for
    negligible cost — temporal stability matters in a movie; the kNN floor reuses
    each scenario's force softening ε, the smallest separation the sim resolves.
  - [next: bloom at grade time in linear space (M6b) — the placement decision is
    already argued in the plan doc.]
- **HDR bloom at grade time (landed, M6b) — the emissive-star-field halo.**
  `bloom(pixels, w, h, &BloomConfig { strength, levels, radius })` in `grade`: mip
  pyramid down, separable Gaussian per level, up-add, `out = img + strength·halo`.
  Pure CPU, linear-domain, image-space — wired into `grade_file` behind
  `GradeConfig.bloom: Option<BloomConfig>` and into `regrade` as `--bloom S
  [--bloom-levels N] [--bloom-radius R]` (defaults 5 / 2.0).
  - **Placement (deliberate deviation from this doc's render-stage recipe):** bloom
    runs at *grade time*, in linear space *before* the tone curve — mathematically
    identical to pre-EXR bloom (both are linear-domain pre-tonemap), but the EXR
    stays the pristine **pre-bloom** artifact, so strength/radius iterate in
    seconds through the M6a regrade loop instead of a re-render. GPU bloom stays a
    named perf follow-up gated on actual pain (CPU at 720p is instant).
  - **No bright-pass threshold** (decision): in the linear astro look every source
    blooms in proportion to its flux — which keeps the operator LINEAR, gated
    bit-exactly via `bloom(2·img) = 2·bloom(img)` (×2 commutes with f32 rounding,
    so no threshold/knee can hide inside a tolerance).
  - **The border finding (caught by the first rendered A/B, then gated):** a
    flux-exact *scatter* pipeline (per-source-normalized kernels, out-of-range taps
    clamped to the border) piles the reflected halo flux into a **bright band along
    the frame edges** — a constant-image corner pixel came out 4.7×. A local,
    data-independent pipeline cannot be exactly flux-conserving AND map constants
    to constants at borders simultaneously, so the two laws are split: every
    pyramid stage (tent reduce / Gaussian blur / tent expand, all with
    symmetric-reflected borders) is a **gather of convex combinations** —
    mean-valued mips, constants pass through unchanged, no band possible — and the
    **flux budget is enforced by one explicit scalar**, `strength·flux(img)/
    flux(halo)`, so the mix adds exactly `strength × total flux`. The renormalizer
    sums f64 over **sorted** channel values (permutation-invariant), keeping
    translation equivariance bit-exact. Tent taps sit ON even fine pixels (no
    half-pixel drift).
  - Gates (`grade/tests/bloom.rs`): strength/levels-0 bit-exact no-ops; bit-exact
    ×2 linearity; flux `(1+strength)·flux(img)` at 1e-5; **constants bloom to
    constants at every pixel** (the border gate, 1e-4); odd-dimension
    center-impulse dihedral symmetry + monotone radial decay; bit-exact 2^levels
    translation equivariance; 1×1 mip-floor level cap; determinism; `grade_file`
    applies bloom image-wide before the per-pixel tone curve (wiring gate).
  - **Movie default: ON in all three scenarios, strength 0.45** (levels 5, radius
    2.0), tuned by A/B regrades of retained QUICK EXRs (0/0.3/0.45/0.6/1.2; cuspy
    under asinh exposure 4, disk/dm under the ACES default): 0.3 timid, 0.6 hazes
    the dense cuspy halo field, 1.2 washes structure; 0.45 makes nuclei/knots glow
    while tails and halo dots stay resolved.
  - [next: Hermite temporal upsampling to 60 fps (M6c).]
- **Hermite temporal upsampling to 60 fps (landed, M6c) — kill the flipbook.**
  Snapshots store full phase space (`pos` *and* `vel`), so cubic Hermite between
  adjacent snapshots gives physically-informed in-betweens at zero sim cost.
  `renderprep::interp` (view-side, not core): `HermiteSpan::new(s0, s1)` validates
  once (equal lengths, **identical `ParticleId` streams** — the defensive gate
  against a silent reorder scrambling the movie — strictly increasing finite
  time), then `sample(u)` returns positions *and* velocities (the cubic's
  analytic derivative — C¹ at joins, and the M6e Doppler input).
  - **Endpoint bit-exactness by construction:** the Hermite basis forms hit
    endpoint values exactly (`h00(0)=1`, others 0, etc.), and the velocity keeps
    the `v0`/`v1` terms unscaled (no `Δt·v/Δt` round-trip), so `u=0`/`u=1`
    reproduce the snapshots bit-for-bit — gated, not assumed.
  - **Attribute strategy (decision):** full `prepare` (including the O(N²) kNN
    density pass) runs only on the ~61 *snapshot* states; `subframe(span, f0,
    f1, u)` takes Hermite positions + a two-product lerp `(1-u)·a + u·b` of the
    prepared endpoint color/brightness/size (that form, not `a + u·(b-a)`, is
    what makes both endpoints bit-exact). Density evolves on the snapshot
    timescale; per-subframe kNN would multiply prep cost ~8× for no visible gain.
  - Gates (`renderprep/tests/interp.rs`): endpoint bit-exactness; C¹ at the
    joins; exactness on linear and cubic trajectories; **two Kepler oracles**
    solved analytically in the test — circular (tolerance = the cubic-Hermite
    local error bound `max|p⁗|·Δt⁴/384`, velocity `Δt³/72`) and eccentric e=0.5
    straddling perihelion (same bound, `max|p⁗|` from a 5-point finite
    difference of the *analytic* orbit); id/length/time rejection; subframe
    endpoint reproduction, exact attribute lerp, Hermite-not-chord positions;
    determinism.
  - xtask: 8 subframes per snapshot interval, FPS 30→60 — ~61 snapshots →
    ~481 frames, the ~2 s flipbook becomes a ~8 s continuous movie (playback
    4× slower per unit sim time; pericenter reads as continuous motion).
  - [next: camera rig — smoothed framing + orbit/tilt paths (M6d).]
- **Camera rig (landed, M6d) — the difference between a diagram and a film.**
  `render::rig` (pure CPU math; xtask wires a path per scenario): `ease_in_out`
  is the quintic smootherstep `6u⁵−15u⁴+10u³` (zero first derivative at both
  ends — the camera starts and stops at rest; peak slope 15/8 is what the
  smoothness gates budget against), evaluated in **f64** because the f32
  quintic wobbles ~5e-7 near u=1 — enough to break frame-to-frame monotonicity.
  `smooth_envelope(raw, window)` turns per-frame framing radii into a breathing
  zoom: moving max over ±window then a truncated Gaussian (σ = window/3, kernel
  radius = window, edges clamped).
  - **The envelope can never crop tighter than the raw requirement — exactly.**
    Because the kernel never reaches past the moving-max half-window, every
    output is a convex combination of maxima that each cover frame i, so
    `out[i] ≥ raw[i]` holds in exact arithmetic (a final elementwise max pins
    it in f32; gated with strict ≥ under proptest). The moving max also looks
    *ahead*: the camera widens before a transient needs the room.
  - **`CameraPath`:** `Fixed` (returns the given camera at every u — the
    pre-M6d pipeline bit-exactly, gated) or `OrbitTilt` — eased azimuth θ /
    polar tilt φ sweep about a target, radius sampled from the envelope track
    **linear in raw u, not eased time** (the envelope is already time-aligned
    to the action; easing applies to angles only).
  - **Pole-safe basis:** camera along r̂(θ,φ), view −r̂, screen-up −φ̂ — exactly
    orthogonal to the view axis for *every* (θ,φ), so there is no degenerate
    up-hint at face-on (orthonormality is a proptest over the whole domain);
    θ=−90°, φ=0 reproduces the historical +Y-up face-on orientation.
  - **Per-frame radii are the 3-D percentile** (xtask `per_frame_radii`), not
    the in-plane radius the static framing uses: an orbiting/tilted camera has
    no preferred plane, and a sphere of radius r projects to r in every
    orthographic view — the same percentile knob (`frame_percentile`) feeds it.
  - Gates (`render/tests/rig.rs`): ease endpoint/midpoint exactness, vanishing
    end-derivatives, monotonicity, symmetry; envelope ≥ raw (strict, proptest),
    hand-computed kernel values, step-response jumps ≤ H·max-weight,
    determinism; fixed-path bit-exactness; orbit/tilt orthonormal basis + aspect
    everywhere (proptest), endpoint angles, linear-in-u radius sampling,
    per-frame |Δangle| ≤ (|Δθ|+|Δφ|)·(15/8)/(n−1) and |Δr| smoothness budgets,
    parameter validation, determinism.
  - **Choreography (eyeballed per rule 2, QUICK renders):** `cuspy` starts
    ¾-inclined and settles toward face-on (tilt 55°→25°) through a slow 130°
    orbit (azimuth −90°→40°), ±8-snapshot envelope window — the inclined
    opening shows the disks as ellipses, the tails read face-on by the time
    they extend; `dm` half-turn orbit (−90°→90°) at fixed 60° tilt, ±6 window —
    a static view reads the triaxial remnant as a round blob, the orbit is
    what shows it; `disk` keeps the static face-on framing (back-compat
    exemplar). Envelope breathes ~5.6→8.5 world units on the QUICK cuspy run.
  - [next: coloring modes v2 — initial-radius ramp, σ_v, density→hue (M6e).]
- **Coloring modes v2 (landed, M6e) — diversify what the colors *mean*.**
  `prepare` grows `ColorMode {Progenitor | Frozen(colors) | Dispersion}` plus two
  independent opt-in passes — `SizeByDensity` and `CompressionHue` — all default
  OFF, so the pre-M6e map stays bit-for-bit (gated). Every kNN consumer carries
  `(k, softening)` and `prepare` dedups them through a `KnnCache`: the movie
  config points them all at `(DENSITY_K, ε)`, so the O(N²) estimate still runs
  **once** per snapshot however many passes are on.
  - **Initial-radius ramp** (`coloring::initial_radius_colors`): per progenitor,
    mass-weighted COM + half-mass radius from snapshot 0; `t = r/(r + r_half)` —
    exactly 0 at the COM, exactly ½ at `r_half` (the median-mass particle sits at
    the ramp midpoint; normalization is per-progenitor scale-free) — through an
    `(inner, outer)` ramp. The colors ride `ColorMode::Frozen` keyed by particle
    index, so tails carry a *provenance* gradient for the whole movie. xtask
    ramps: disks amber→blue-white vs rose→cyan (inner old/warm → outer
    young/blue, like real disks); halos constant at their dim palette color.
  - **σ_v coloring**: `knn_neighbourhood` extends the density estimator's
    partial-select to carry the k nearest *indices* (left partition + pivot;
    densities gated bit-identical to `knn_density`); `velocity_dispersion` is the
    rms deviation from the neighbourhood mean over `{self} ∪ neighbours`;
    `dispersion_colors` maps `t = σ²/(σ² + σ_ref²)` cold→hot. **Quadratic on
    purpose** (first A/B finding): σ spans only ~2–3× across a self-gravitating
    body, and the linear ratio painted everything midpoint-gray; squaring
    (specific kinetic energy) doubles the contrast under the same
    endpoint/midpoint/monotone gates. **Scoped to single-population subjects**
    (second finding): the ramp replaces the palette and with it the 20×
    halo/disk brightness compensation — on disk scenarios the ~5×-heavier,
    2×-more-numerous halo particles swamp the frame at any ramp brightness that
    still shows the disks. `--color dispersion` is the dm merger's mode (blue
    σ-cold envelope around the σ-hot remnant core); a palette-luminance-weighted
    σ ramp is the named follow-up if ever wanted on a disk+halo scene.
  - **Star-formation proxy** (`compression_colors`): hue lerp from the base
    color toward young-blue-white `[0.7, 0.8, 1.0]`, `t = strength·(1 −
    ρ0/max(ρ, ρ0))` — keyed on **compression** vs the same particle's t=0
    neighbourhood (the plan's recommended variant: absolute ρ would falsely blue
    the undisturbed dense cores; only tidally-compressed material lights up, and
    ρ ≤ ρ0 keeps the base bit-exactly). Honest caveat: the sim is collisionless —
    stellar density stands in for gas compression; a standard visualization
    proxy, not physics. ON by default in every scenario (strength 0.8), **masked
    to luminous progenitors** via the gated `ρ0 = 0` sentinel (third finding: the
    first A/B washed the frame white once the halos overlapped — ratio ≥ 2
    compression lifted 9k deliberately-dim halo particles ~15× in flux; also the
    physics — the disk-scenario halos are dark-matter stand-ins with no gas). The
    dm merger masks nothing: its halos are the luminous subject and the tint is
    a shocked-overlap diagnostic.
  - **Size-by-density** (`density_sizes`): `base·clamp((ρ_ref/ρ)^⅓, 0.6, 1.6)` —
    the SPH-style spacing law; mean-positive-reference and sentinel discipline
    exactly as `density_boost`. ON by default; at these clamps it reads as
    tight cores / soft halo dots without changing the scene's character.
  - **xtask**: `--color progenitor|initial-radius|dispersion` and
    `--reuse-snapshots` (re-prep + re-render retained snapshots, no re-sim — the
    prep-stage analogue of the M6a regrade loop; errors if snapshots are missing
    or the particle count mismatches the scenario, the QUICK/full trap).
    `parse_movie_args` is unit-gated in `xtask/lib.rs`.
  - **Doppler hue deferred** (per plan): it couples frame-data to a view axis,
    which Contract 3 deliberately avoids; `HermiteSpan::sample` already returns
    the velocities it would need.
  - Gates (`renderprep/tests/{coloring,density,prepare}.rs`, `xtask` lib): ramp
    COM/midpoint exactness, monotonicity, per-progenitor scale/position freedom,
    mass-weighted COM, degenerate progenitors; kNN index/d_k consistency and
    bit-identical densities; σ_v pair/lattice/two-population oracles, bulk-
    velocity invariance, empty-neighbourhood sentinel; size hand values, clamps,
    monotone, sentinel, inverted-band panic; compression exact-base guarantees
    (ρ≤ρ0 / strength 0 / sentinels), monotone bounded-by-strength, exact ½ at
    ρ=2ρ0; frozen-colors bit-stability across frames; default-config bit-compat;
    full-featured determinism; movie-arg parsing.
  - Demo: retained M6d QUICK snapshots re-rendered in seconds-to-minutes per
    mode — cuspy in progenitor (+SF proxy +sizes) and initial-radius, dm in
    dispersion (`M:\claud_projects\temp\m6e\`).
  - [next: `scenario.toml` front-end + the Toomre encounter zoo (M6f).]
- **`scenario.toml` front-end + the Toomre encounter zoo (landed, M6f) —
  scenarios become data.** `xtask::spec` grows the declarative schema
  (`ScenarioSpec`: `[model]` galaxies/orientations/counts, `[orbit]`, `[sim]`,
  `[look]`, `[rig]`), a compile-time-embedded preset registry
  (`xtask/scenarios/*.toml`), and `build_scenario(spec, quick)` — main.rs is
  now pure glue. serde/toml is an **xtask-only dependency by design**: the
  engine crates stay serde-free.
  - **Schema stance: physics is gated, aesthetics are data.** Validation turns
    every IC-constructor `assert!` into a readable error (a toml is user input,
    not a programming-time contract) plus the cross-field rules (palette/ramp
    lengths = the model's progenitor count, SF mask in range, bound-orbit
    separation ≤ apocenter). Unknown keys fail loudly *everywhere*: serde's
    tagged enums can't `deny_unknown_fields`, so `[model]`/`[rig]` deserialize
    through strict intermediate tables (`try_from`) — the one non-obvious serde
    decision of the session.
  - **Back-compat is the headline gate:** the three original scenarios are
    checked-in presets reproduced *field-for-field* against independent literal
    copies of the old hardcoded constants, and the disk preset's *build* is
    gated `State`-equal against direct inline IC construction — same params,
    same seed ⇒ same movie. CLI selector set == the preset registry (a new toml
    is reachable without touching the parser); `nfw`/`disk-nfw` aliases and the
    bare-out-dir positional keep working; any first positional ending `.toml`
    is a custom scenario.
  - **The orientation door was already open:** `Orientation::from_angles`
    covers the whole zoo — no IC change was needed. The plumbing is pinned by
    the sharpest gate of the session: building `retro` must equal building
    `cuspy` **rigidly rotated π about x** per galaxy (positions *and*
    velocities, 1e-9), which no sign-convention slip survives.
  - **The zoo** (all QUICK-rendered, `M:\claud_projects\temp\m6f\`; presets
    carry the expectations as comments):
    - `retro` — cuspy's controlled twin (same seed/orbit/sim, spins flipped
      180°): tidal tails *suppressed* — at the frames where prograde cuspy
      shows bridge + two tails, retro shows two intact round disks with mild
      heating. The "why prograde matters" A/B pair, cut-together-able because
      the rig choreography is identical.
    - `inclined` — galaxy 1 tilted 45°, galaxy 2 prograde: the tilted disk
      throws a genuinely 3-D warped, S-twisted tail while its partner's stays
      flat — the contrast in one frame; rig sweeps tilt 65°→20° through it.
    - `bullseye` — the Cartwheel geometry: target disk spin i=90°/ω=180°
      (spin axis = (0,1,0), exactly the pericenter passage direction), compact
      intruder on a parabolic near-central punch (peri = 0.3 Rd, slightly
      off-center along the node line). Expanding ring density wave confirmed —
      annular rim, dim center, displaced nucleus — dissolving by t≈18.
      Tuning findings (first QUICK A/B): the escaping intruder must be a
      *minority particle population* (~12%; equal flux via heavier particles)
      or any framing percentile above its share chases it off the ring;
      the compact scene (disk radius 2) wants p80 + splat 0.08 (p70/0.12
      mushed out); T=20 — the ring is dead past t≈18, cadence 16 keeps ~63
      snapshots.
    - `minor` — 1:10 satellite (0.115 vs 1.15) on a bound e=0.7, 40°-inclined
      orbit starting near apocenter, T=120 ≈ 2.7 radial periods (gated):
      blue stream arcs over the primary at first pericenter, satellite fully
      disrupted into a diffuse wrap by the third. Tuning finding: p97 framing
      chased the few % of far-slung escapers (envelope max 46 → galaxy became
      a dot); the stream lives inside the satellite's ~7-unit apocenter — p90.
  - **Physics cautions carried:** retro/inclined inherit cuspy's cusp-resolution
    budget (M5f: halo N high + ε=0.02 even in QUICK); bullseye/minor ride the
    warm (Q=1.5) Plummer family, so no cusp constraint.
  - Gates (`xtask` lib `spec::tests` + `tests/scenario_build.rs`): the three
    preset-reproduction equalities; retro/inclined pinned as cuspy twins on
    (seed, orbit, sim, model-with-flipped-spins) with look/rig left free (rule
    2: aesthetics are eyeballed, not gated); bullseye spin-axis/pericenter
    geometry; minor mass-ratio/bound/apocenter-start/≥2-radial-periods; a
    17-case reject table (broken physics, typo'd keys at every table depth,
    zero counts/cadence, percentile > 1, c ≤ 1, sub-pericenter separation,
    beyond-apocenter bound start); build determinism; QUICK vs full counts,
    cadence override, frame size; brightness unit = disk-1 (resp. equal-mass
    halo) particle mass; movie-arg parsing over the registry + `.toml` paths.
  - Demo: the four-movie QUICK zoo contact sheet under
    `M:\claud_projects\temp\m6f\` (retro/inclined at first pass; bullseye/minor
    after one documented tuning iteration each); full-res renders of all four
    from the same presets.
  - [next: perspective camera + world-space vertex-shader projection — the 10⁸
    swap (M6g, optional).]
- **Perspective camera + world-space vertex projection (landed, M6g) — the 10⁸
  swap.** Instances now carry **world-space** pos/radius (camera-independent);
  the camera basis + projection parameters ride a uniform and projection runs
  in the vertex shader — the per-frame CPU projection loop is gone, which is
  the render-architecture prerequisite for 10⁷–10⁸ particles. Plus the
  perspective `Camera` variant for inside-the-scene drama.
  - **Ortho is bit-compatible in formula and golden-gated in pixels:** the
    shader's ortho branch is the exact arithmetic of the retired CPU path, and
    `tests/vertex_path.rs` pins total flux + the six brightest probe pixels
    captured from the pre-swap renderer at 1e-3 relative — every pre-M6g movie
    reproduces.
  - **The three deferred decisions, settled by physics rather than tuning:**
    - *Brightness attenuation is the real optics law, for free.* Peak surface
      intensity stays fixed; screen size shrinks ∝ 1/depth; integrated flux
      therefore falls as 1/d² automatically (surface brightness of a resolved
      source is distance-invariant — no tuned attenuation factor, no semantics
      fork with the additive ortho path). Gated: two identical particles at
      depths d and 2d must show a 4.00× flux ratio.
    - *Sub-pixel clamp = the point-source regime.* Below `min_splat_px` the
      quad is drawn at the clamp size with emission dimmed by (true/clamped)²,
      so flux keeps the 1/d² law while distant stars stop shimmering as
      sub-pixel quads (gated: the ratio law survives clamping). The fill-rate
      max clamp *saturates* — no brightness boost on close fly-bys. Clamp
      window validated CPU-side (`RenderError::Config`): WGSL `clamp()` with
      an empty interval is UB.
    - *Near plane culls, never clips.* Splats have no depth extent, so a
      view-depth ≤ near degenerates the whole quad in the vertex shader — the
      1/z pole is never evaluated (gated: zero flux + all-finite pixels for
      particles at the eye plane, inside near, and exactly at near).
  - **Shared framing parameterization:** `Camera::perspective` takes
    `half_extent` *at the target plane* (eye at `distance` behind the target),
    so the pinhole reproduces the ortho projection exactly there (gated) and
    the rig can swap projection without re-deriving its envelope.
  - **Dolly rig (`CameraPath::dolly`) + `dolly` preset:** fixed spherical
    direction (the M6d basis, gated equal to a parked orbit-tilt), quintic-
    eased eye distance, **constant vertical fov** — a camera move, not a zoom
    (`half_extent.y = d·tan(fov_y/2)` at every u, gated). Scenario knobs are
    scene-scale-free: `distance_frac`/`near_frac` are fractions of the final
    snapshot's framing radius (the dolly targets the remnant), anchored in
    main.rs. The `dolly` preset is the cuspy realization bit-for-bit, flown
    from 5× the framed radius down to 0.8× — into the remnant's halo field —
    at 55° tilt. (End distance was 0.35× at landing; retuned by A/B re-render
    of the retained QUICK snapshots at 0.35/0.5/0.65/0.8 — 0.35 is great in
    motion but the parked closing stills are a wall of out-of-focus splats;
    0.8 keeps both nuclei and the star field resolved with the soft
    foreground still selling the depth.)
  - Notes for future GPU work: `target` is a WGSL reserved keyword; the shared
    uniform needs `VERTEX_FRAGMENT` visibility; one unreproduced subpixel-gate
    failure during a concurrent clippy build (8 clean re-runs — suspected GPU
    contention flake, watch).
  - Demo: the QUICK dolly movie under `M:\claud_projects\temp\m6g-dolly-quick\`.
- **M6 (complete) — "the beautiful": visual/cinematic series.** Asinh grade +
  regrade loop + density boost ON (M6a) → bloom (M6b) → Hermite 60 fps
  upsampling (M6c) → animated camera rig (M6d) → coloring modes v2 incl. the
  density→blue star-formation proxy (M6e) → `scenario.toml` front-end + the
  Toomre encounter zoo (M6f) → perspective/vertex-path render, the 10⁸ swap +
  dolly rig (M6g) — all landed above. Session-by-session plan with gates and
  decisions: `docs/plans/cinematic-toomre-bloom.md`.
- **M7 (in progress) — "the breathing": isothermal SPH gas + volumetric
  dust-lane render series.** v1 physics is isothermal SPH (fixed c_s, no
  energy equation); render is raymarched emission+absorption with **full star
  attenuation** (per-star exp(−τ) to the camera — dust lanes over the cores).
  Session plan with the argued design decisions (D1–D10):
  `docs/plans/deep-orbiting-sunbeam.md`. CPU + rayon at 10⁴–2·10⁵ gas
  particles; GPU SPH stays a named follow-up.
- **Gas plumbing + SPH kernel/neighbors/density (landed, M7a).** The
  foundations everything else in M7 stands on.
  - **`Species` column (D1):** `State` gains ONE column,
    `kind: Vec<Species>` (`#[repr(u8)] Collisionless | Gas`). `progenitor`
    stays a pure identity tag (gas will take new tags 4/5 in `DiskCollision`);
    physics/render routing keys on `kind`, never on progenitor. The default is
    `Collisionless` everywhere (`from_phase_space`, every IC, the snapshot
    reader for v1 streams) — chosen over a NaN-style sentinel because `State`
    derives `PartialEq` and the M6f build-vs-IC equality gates would silently
    never match again with NaN anywhere in the struct.
  - **Snapshot v2 (D3):** one appended `kind[n] (u8)` column;
    `FORMAT_VERSION = 2` with `MIN_READ_VERSION = 1` — the reader accepts
    {1, 2}, defaulting v1 particles to `Collisionless` (v1 predates gas, and
    the retained scenario-zoo snapshots stay re-preppable), the writer always
    emits v2. Unknown species bytes are `Corrupt`, not a transmute; the v1
    layout is pinned by an independent byte-level fixture writer in the tests.
  - **Smoothing length h is DERIVED, never stored (D2).** `ForceSolver` takes
    `&State`, so a stored h column could never be kept fresh — it would go
    stale after step 1 and renderprep would deposit with wrong h. Instead h is
    a deterministic function of positions: bisection to a fixed relative
    tolerance (1e-3) on the **kernel-weighted neighbor count**
    N(h) = (4π/3)(2h)³·Σ_j W(|x_ij|, h), which is monotone nondecreasing in h,
    rises from the self-term floor 32/3 (h→0) to the plateau (32/3)·n (whole
    cloud inside support) — so a root exists iff n ≳ 4.5 at the default target
    N_ngb = 48 (kept < ~57, the cubic-spline pairing-instability guard).
    Rootless solves (under-populated, coincident knots) clamp
    deterministically: finite h, finite positive ρ, never a panic. A
    warm-start only seeds the bisection bracket and cannot move the converged
    value beyond the tolerance (gated cold ≡ warm).
  - **SPH toolkit in `solvers::sph` (D5):** M4 cubic-spline kernel (compact
    support 2h, 1/(πh³) normalization, analytic gradient); **sparse hash-grid**
    neighbor search — HashMap-keyed cells, NOT a dense array (a far outlier at
    small cell size is a 10¹⁰-cell box), queries return ascending indices so
    consumers sum in a fixed order (parallel ≡ serial bit-exact); O(N²)
    `reference_neighbours`/`reference_density` oracles per the `reference_*`
    house pattern. renderprep will consume this same module for deposition
    (M7d) — single source of truth for all smoothing math.
  - **Bit-exactness trick:** the fast density path skips out-of-support
    particles the oracle includes; those terms are an exact `+0.0` and rho
    stays ≥ +0.0, so both sums associate identically — the gate is exact f64
    equality, not a tolerance.
  - Gates: kernel normalization by piecewise-aligned Simpson (1e-9, error
    O(n⁻⁴) argued), hand value W(0) = 1/(πh³), support, gradient vs central
    differences (O(δ²) argued), h/λ³ scaling; grid ≡ oracle bit-exact on
    lattice/random/clustered/wall-straddler/coincident clouds + a proptest
    sweep; fixed-h density bit-exact vs oracle; uniform-lattice ρ = m/s³ to 2%
    (lattice quadrature at h = 1.25 s, wrong normalization misses by ≥ 25%);
    adaptive-h recovers N_ngb ± 1 (bisection tol maps to ~0.07); scaling law;
    determinism + parallel ≡ serial bit-exact; warm-start invariance;
    under-populated clamp; snapshot v1 fixture/v2 round-trip/garbage rejection.
  - Demo: deferred to the next session (density-coloring side-by-side on a
    retained cuspy snapshot + O(N) timing vs the O(N²) kNN path).
  - [next: SPH forces (pressure + Monaghan viscosity), `GravitySph` composite,
    CFL sentinel, isothermal shock tube vs the analytic Riemann solution
    (M7b).]

## Validation strategy

- Analytic: Kepler 2-body (period, eccentricity).
- Conservation: total energy (use the **softened** potential), linear &
  angular momentum, COM drift. Leapfrog energy should *oscillate*, not drift.
- Equilibrium: isolated galaxy model must stay in equilibrium (tests IC
  sampling + force accuracy).
- Standard problems: Toomre & Toomre tidal tails; cold-collapse virialization
  (2T/|W| → 1). Cosmology later: Zel'dovich pancake, Santa Barbara cluster.
- REBOUND cross-check via HDF5 export (validation runs only).
