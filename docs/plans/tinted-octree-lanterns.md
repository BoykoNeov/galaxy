# Tinted octree lanterns — adaptive light clustering + scatter tint

The two remaining named deferrals of the scatter series
(`scattered-starlit-veil` v1, inherited through `umbral-lantern-lattice` v2):

1. **Octree light clustering** — replace `cluster_lights`' fixed 8³ uniform
   binning with an adaptive octree cut, so the light budget concentrates
   where the emission is instead of being spent uniformly over the whole
   luminous AABB.
2. **Scatter tint** — a per-channel look knob multiplying the scattered
   radiance (a chromatic scattering albedo: the reflection-nebula "dust
   scatters blue" knob). v1 scattered light carries only the lights' own RGB.

The two are independent; they share this plan because they close out the
scatter series together and share one A/B session.

## Why (what v1 binning gets wrong)

`cluster_lights` bins over the AABB of *all* luminous stars. During a merger
approach the AABB spans both galaxies plus the bridge, so each disk is
covered by only a handful of 1/8-extent bins: near-field illumination around
the cores is blobby, and the single global softening radius (half the
whole-AABB bin diagonal) smears the brightest, closest lights the most —
exactly where accuracy matters. Meanwhile most of the 512-bin budget sits
empty over the void between the disks. An adaptive cut spends the same
worst-case budget where the power is, and each light's softening radius
shrinks with its own cell.

## Part 1 — octree light clustering (replacement, not a knob)

**Replacement decision (flag it, then own it):** clustering quality is
constant-domain, not look-domain — `LIGHT_BINS` was documented as "a quality
constant, not a look knob," and the deferral language ("fixed-bin is v1")
frames the octree as v2 of the same function. So `cluster_lights` is
re-implemented in place, same signature; no new knob, no parallel v1 path
kept. Consequences, stated plainly:

- The scatter-ON look of shipped presets changes (precedent: κ retune, splat
  cap). The QUICK A/B below is the judge, and the new retained render becomes
  the pixel-identical control going forward.
- `scatter = 0` / `scatter: None` stays bit-identical end-to-end — the
  clusterer is never called on the off path. The off-run cross-version pin
  (byte-identical to the retained `shadow_ab/off`) still holds and the A/B
  re-verifies it.
- The GPU is untouched: it consumes a flat `Vec<Light>` of any count ≤ the
  budget; the shadow bake already dispatches `(R³/64, K)` with K from the
  frame. The octree is a CPU-only change behind an unchanged data contract.

### Algorithm (deterministic greedy cut, CPU, f64 accumulation)

Constants (quality constants like the grid res — NOT knobs):

- `MAX_LIGHTS: usize = 512` — retires `LIGHT_BINS` (= the old 8³ worst case);
  keeps the shadow-volume worst case `512·32³·4 B = 64 MiB` under wgpu's
  128 MiB storage-binding default, unchanged.
- `REFINE_TOL: f64 = 1e-3` — relative error floor: refinement stops when the
  worst node's metric falls below `REFINE_TOL ×` the root metric. This is
  what keeps the *typical* light count adaptive (a compact frame clusters to
  few lights) instead of greedily burning the full 512 every frame. Initial
  value; tuned once from the A/B's measured counts + wall-clock, then frozen
  and documented (the splat-cap precedent).
- `MAX_DEPTH: u32 = 32` — float-degeneracy backstop only (two distinct
  positions separate into different octants within ~mantissa-many levels;
  this bound is never the design terminator).

Build:

1. **Luminous set + root cube.** Same `power`/`lum` weights as v1
   (Σ color·brightness, f64; zero-power splats invisible to lights AND to the
   bounds). Root cell = the luminous AABB expanded to a cube (center = AABB
   center, side = max extent — the Barnes-Hut/LBVH root convention; octants
   stay shape-regular). All-dark frame → no lights. All-coincident → one
   light, radius 0 (degenerate cube side 0), exactly v1's degenerate contract.
2. **Node state.** Cubic cell + member star indices + f64 aggregates: scalar
   luminance `P`, per-channel power, luminance-weighted centroid, and the
   member-position AABB (whose half-diagonal is the node's `spread`).
   Aggregates fold in ascending star-index order (the deterministic fold
   discipline).
3. **Metric.** `metric = P · spread²` — the standard far-field surrogate for
   the flux error of collapsing a cluster to its centroid (relative error
   ~ (spread/d)², weighted by power). Using the *member spread*, not the cell
   diagonal, makes single-star and coincident nodes exactly metric 0 — they
   terminate structurally, no special cases.
4. **Greedy refinement.** Max-heap of splittable leaves ordered by
   `(f64::total_cmp on metric, node id)` — fully deterministic. Loop: pop the
   worst leaf; stop if its metric ≤ `REFINE_TOL × root_metric`; partition its
   members by octant (empty octants dropped; a split whose members all land
   in one octant is legal — leaf count unchanged, cell tightens, chain length
   bounded by the straddle depth); if the split would push the leaf count
   past `MAX_LIGHTS`, keep the node a leaf and stop refinement (slack ≤ 7,
   documented); otherwise replace the leaf with its children and push the
   splittable ones. Nodes at `MAX_DEPTH` never enter the heap.
5. **Emission.** Final leaves in DFS pre-order by octant index (a canonical,
   reproducible order). Per leaf: `pos` = luminance-weighted centroid,
   `rgb` = Σ power (conservation exact in the f64 fold, one cast per channel
   at emission — v1's rule), `radius` = ½ · the leaf's OWN cell diagonal —
   v1's honesty rule verbatim ("inside the cell the point proxy is invalid"),
   now per-light: refined cores get tight radii, which is the visible win.

Cost: O(N · depth) index partitioning per frame — trivial next to the march.
The knock-on costs are the per-sample light loop and the shadow bake, both
linear in K; the A/B measures per-frame K (old occupancy vs new) and
wall-clock, and `REFINE_TOL` is the lever if K inflates.

### API

- `volume::MAX_LIGHTS: usize = 512`, `REFINE_TOL`, `MAX_DEPTH` (module
  constants; `LIGHT_BINS` deleted — the SHADOW_RES doc comment's memory
  arithmetic re-keys to `MAX_LIGHTS`).
- `cluster_lights(&FrameData) -> Vec<Light>` — signature, `Light` layout, and
  call sites (renderer, xtask movie path ×2) all unchanged.

### Gates (`render/tests/cluster.rs`, new; CPU-only — the GPU never sees the clusterer)

The v1 binning-specific assertions in `render/tests/scatter.rs` (same-bin
merge geometry, global radius) are RETIRED and replaced by these — a
deliberate, documented contract change landing in the red commit; the
algorithm-independent assertions (power conservation, dark-star drop,
distant-clusters-stay-separate, all-dark-empty) carry over into the new file.

1. **Conservation + structure (proptest):** random star sets (N ≤ 2000,
   random colors/brightness incl. zeros) → total RGB power conserved within
   f32-cast tolerance; count ≤ `MAX_LIGHTS`; every light inside the root
   cube; every radius ≥ 0 and ≤ ½·root diagonal.
2. **Determinism:** two runs on the same frame → `Vec<Light>` equal
   bit-for-bit (`PartialEq` on the whole vec).
3. **Degenerates:** all-dark → empty; single star → exactly one light at its
   position, radius 0; N coincident stars → one light, radius 0; one bright +
   one dark star → one light, dark star moved neither centroid nor bounds.
4. **Adaptivity (the point):** a bright compact cluster (e.g. 64 stars in a
   0.1-side cube) plus a dim sparse spread across a 100-unit box → the
   compact cluster's region receives strictly more lights than under any
   uniform 8³ split of the joint AABB would give it (assert ≥ a hand-chosen
   count), and the max radius among lights near the cluster is ≪ the v1
   global radius (assert against the hand-computed v1 value — a number in
   the test, not a call into removed code).
5. **Near-field accuracy (hand-derived oracle):** for the gate-4
   configuration, the unshadowed isotropic incident flux
   `Σ_k L_k/(4π(d_k²+r_k²))` at a probe point one unit from the compact
   cluster matches the exact per-star sum within a few percent (tolerance
   justified from the metric bound at `REFINE_TOL`); documented in a comment:
   the same probe under 8³ binning misses by a factor recorded from a
   hand calculation.
6. **Budget saturation:** a scale-free fractal-ish cloud (hierarchical LCG
   clusters) large enough to exhaust the budget → count ≤ 512 and ≥ 512·½
   (the greedy stop leaves slack ≤ 7; the loose lower bound just proves the
   budget is actually reachable).
7. **Off-path pins:** the existing scatter/shadow gates (hand-built light
   lists, `scatter: None` bit-identity) must stay green untouched — they
   never call the clusterer, which is itself the proof the replacement is
   contained.

## Part 2 — scatter tint

### Model

```text
c += T · σ_s · ρ · Δs · tint ⊙ Σ_k p_HG(cosθ_k, g) · T_k(s) · L_k / (4π·(d_k²+r_k²))
```

`tint` is a per-channel multiplier on the scattered radiance only — emission,
absorption, star splats, and the shadow bake are untouched. Semantically a
chromatic single-scattering albedo (dust reflects blue preferentially — the
reflection-nebula look); it composes multiplicatively with each light's own
RGB. `[1,1,1]` is bit-identical to v2 (multiplication by literal 1.0 is
exact in f32, CPU and WGSL — pinned bitwise). Applied once per step outside
the per-light sum (it is constant across lights).

### API

- `ScatterLook.tint: [f32; 3]` — default `[1.0; 3]`; mechanical literal
  updates at every `ScatterLook { .. }` construction site land in the red
  commit (workspace must build).
- CPU `march_gas`: `c[ch] += es * inc[ch] * tint[ch]` (order chosen so the
  neutral tint reduces to the current expression bit-for-bit).
- WGSL: `GasUniforms` gains one `vec4<f32>` (`tint.xyz`, w unused) in BOTH
  mirrors (WGSL struct + `#[repr(C)]` Rust struct — buffer grows 16 B; the
  star prepass and shadow bake share the struct and recompile untouched).
  March: `c += (t * g.scat.x * rho * ds) * inc * g.tint.xyz;`.
- xtask: `[look.gas]` gains optional `scatter_tint: [f32; 3]`
  (`GasLookSpec.scatter_tint: Option<[f32; 3]>`, `GasLookValues.scatter_tint:
  [f32; 3]` default `[1.0; 3]`, threaded into `ScatterLook` in `main.rs`).
  Validation (the dead-knob discipline): every component finite and ≥ 0;
  present without a positive `scattering` → loud reject; all-zero tint with
  `scattering > 0` → loud reject ("zeroes the term — set scattering = 0
  instead"). The render layer itself accepts any finite tint (oracle API).

### Gates

CPU (`render/tests/scatter.rs`):
1. **Neutral is neutral, bitwise:** `tint = [1,1,1]` march bit-identical to
   the pre-tint march (radiance AND transmittance), shadows on and off.
2. **Per-channel linearity:** emissivity 0, `tint = [2,1,1]` → red scattered
   radiance exactly 2× the neutral run, green/blue bit-identical, T
   bit-identical.
3. **Zero tint kills the term:** `tint = [0,0,0]` radiance equals the
   `scatter: None` march values (adding `+0.0` to non-negative accumulators
   is exact).
GPU (`render/tests/scatter.rs`):
4. **GPU ≡ CPU** with a non-trivial tint (e.g. `[0.5, 0.8, 1.6]`), g ≠ 0,
   shadows ON, ortho + perspective, at the volume.rs tolerances
   (1e-3 rel + 1e-5 abs).
5. **Off-path bit-identity:** neutral-tint GPU frame bit-identical to the
   pre-tint frame; `scatter: None` stays bit-identical to the no-scatter
   frame (the new uniform field must not disturb the off path).
xtask (`xtask/tests/scenario_gas.rs`):
6. Knob parse + default `[1,1,1]`; rejects: negative / non-finite component,
   tint without positive `scattering`, all-zero tint; the shipped gasrich
   value (whatever the A/B decides) is declared by the preset test.

## Milestones

- **O1 [red]:** octree gates (`render/tests/cluster.rs`) + `MAX_LIGHTS`/
  `REFINE_TOL`/`MAX_DEPTH` constants + `cluster_lights` body replaced by
  `todo!()`-backed helper stubs where needed; v1 binning-specific assertions
  retired from `scatter.rs` in the same commit (documented contract change).
  Workspace builds (red-commit rule), cluster gates fail.
- **O1 [green]:** the greedy octree cut; all cluster gates + the untouched
  scatter/shadow/volume suites green.
- **T1 [red]:** tint gates + `ScatterLook.tint` field + mechanical
  `tint: [1.0; 3]` literals + xtask spec/validation stubs. Workspace builds;
  the two bitwise off-path pins may pass by construction (the umbral
  precedent), the rest fail.
- **T1 [green]:** CPU march term + WGSL mirror + uniform plumbing + xtask
  knob; all gates green.
- **AB:** QUICK gasrich A/B, retained under
  `M:\claud_projects\temp\octree_tint_ab`:
  1. `off` — `scattering` removed: must stay byte-identical to the retained
     `shadow_ab/off` control (the cross-version off-path pin).
  2. `octree` — shipped knobs, new clustering, neutral tint: judge vs the
     retained `shadow_ab/on` control (the v1-clustering ship); record
     per-frame light counts (old occupancy vs new), total_flux, wall-clock;
     tune `REFINE_TOL` here if counts/perf demand, then freeze it.
  3. `tint` bracket — 2–3 tints on top of (2), e.g. neutral vs a blue-leaning
     `[0.6, 0.8, 1.3]` vs a stronger variant; pick by eyeball.
  Ship decision recorded here (the scatter/shadows precedent: the showpiece
  exercises new features when they earn it; neutral tint is the safe default
  if none convinces).
- **D:** DESIGN.md scatter note, `long-burning-beacon.md` deferral tick,
  memory, quality gate (`fmt --check`, `clippy -D warnings`, workspace
  tests), commit + push.

## Commit sequence

1. `docs(plan): tinted-octree-lanterns — octree light clustering + scatter tint`
2. `test(render): octree light-clustering gates — tinted-octree-lanterns [red]`
3. `feat(render): adaptive octree light clustering replaces fixed 8³ binning`
4. `test(render): scatter tint gates — tinted-octree-lanterns [red]`
5. `feat(render): scatter_tint knob — chromatic scattering albedo`
6. `feat(render): gasrich ships <A/B outcome>; docs` (+ roadmap/DESIGN/memory)

## Deferrals (named)

- Per-sample tree traversal / lightcuts-style per-sample cuts (the cut is
  frozen per frame — required anyway by the baked shadow volumes and D9's
  camera-independence; a per-sample cut would also need per-node shadow
  handling).
- Temporal cut stabilization (hysteresis across frames). The cut, like v1's
  bins, re-derives from each frame's luminous AABB; scattered radiance is an
  integral over many lights so discrete cut changes should stay sub-visible —
  the A/B movie is the judge, and hysteresis is the named fix if it shimmers.
- Per-channel κ / colored *shadows* (tint colors the scattered light, not the
  occlusion — unchanged from umbral-lantern-lattice).
- DDA / hierarchical shadow-bake acceleration (inherited).
