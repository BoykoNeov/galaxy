# Umbral lantern lattice — per-light shadow volumes

The named v2 deferral of `scattered-starlit-veil`: light→sample
self-shadowing for the single-scatter term. v1 lets every light reach every
march sample unattenuated; with gasrich's κ = 100 a disk midplane is many
optical depths thick, so unshadowed scatter lights gas that physically sits
in its own shadow. This plan bakes, per point light, a 3-D transmittance
volume (the "K×voxels compute prepass" the deferral named) and multiplies it
into that light's contribution — OPTIONAL and bit-compatible when off.

## Physical model (v2)

```text
j_scat(s) = σ_s·ρ(s) · Σ_k  p_HG(cosθ_k, g) · T_k(s) · L_k / (4π·(d_k² + r_k²))

T_k(p)   = exp( −∫ κ·ρ_mix ds  over the light_k → p segment,
                clipped to the union grid AABB )
```

- **Baked, not traced per sample**: `T_k` is precomputed at the centers of a
  `SHADOW_RES`³ voxel lattice spanning the union AABB (the march domain) —
  one volume per light — then sampled trilinearly inside the march. A
  per-sample secondary march would cost K × steps per sample; the bake costs
  K × R³ threads once per frame.
- **`SHADOW_RES` = 32** is a quality **constant** (like `LIGHT_BINS` and the
  gas grid res), not a knob. Worst-case memory `LIGHT_BINS³·32³·4 B = 64 MiB`
  stays under wgpu's default 128 MiB storage-binding limit; shadows are
  low-frequency (soft occlusion, not contact shadows), so 32³ over the
  domain is honest resolution.
- **Bake rule**: march FROM the light TOWARD the voxel center (t = 0 at the
  light), the segment clipped against the union AABB and truncated at the
  voxel (`t0 = max(clip.t0, 0)`, `t1 = min(clip.t1, dist)`), the shared
  nominal step (half the min cell edge over both endpoint grids — the
  density band-limit governs accuracy, not the shadow lattice), τ summed
  then exponentiated once — `star_transmittance`'s exact operation order.
  An empty segment (no gas between light and voxel, or a voxel coincident
  with the light) is exactly `T = 1`.
- **Sampling rule**: `GasGrid::sample`'s cell-center trilinear arithmetic
  MINUS the zero-outside test — pure clamp-to-edge. A transmittance has no
  natural zero outside the domain, and the march only samples inside the
  clipped chord; returning 0 (= fully shadowed) on an epsilon-outside float
  excursion would punch dark rims.
- **Softening**: the shadow segment is geometric from the light's centroid.
  The cluster softening radius applies only to the 1/d² intensity pole —
  occlusion has no pole to kill.
- **Achromatic**: κ is scalar, so `T_k` is one f32 per voxel (colored
  shadows would need per-channel κ — a different look feature).

Standing constraints hold: the bake is camera-independent per-frame data
(D9); nothing new is baked into the ρ grid (D8 — the shadow lattice is
derived at render time from grid + lights + κ); the scattered term still
rides the camera-path T inside the ordered march.

## API

- `volume::SHADOW_RES: u32 = 32`.
- `volume::ShadowVolumes { bounds_min, bounds_max, count, data: Vec<f32> }`
  — `count·R³` values, light-major, x-fastest within a volume (the grid
  deposit order); `sample(&self, k: usize, p: Vec3) -> f32` trilinear
  clamp-to-edge.
- `volume::bake_shadows(gas: &GasFrame) -> ShadowVolumes` — the CPU
  reference of the WGSL compute prepass (f32 accumulation, the mirror
  discipline), one volume per `gas.lights` entry.
- `ScatterLook.shadows: bool` — `false` = v1 unshadowed = bit-compatible.
- `march_gas` gains a second parameter `shadows: Option<&ShadowVolumes>`:
  `Some` multiplies each light's incident term by `T_k(p)`. The ORACLE keys
  on the argument; the RENDERER keys on the look flag (`render_gas_cpu`
  bakes iff scatter is active ∧ `look.shadows` ∧ lights non-empty). A direct
  caller passing the flag but `None` gets the unshadowed march — documented
  loudly at the parameter (it is an oracle API, not a safety rail).
- WGSL: `GasUniforms.scat.w` (currently unused) carries the shadow flag.
  The gas pipeline gains `@group(2) @binding(0) var<storage, read> shadow`
  (dummy 4-byte buffer when off — never read, `scat.w = 0`, off-path
  arithmetic untouched). A new shadow-bake compute pipeline uses
  [uniforms, gas group, shadow-write group]; the `PointLight` declaration
  moves into `WGSL_GAS_COMMON` (the star prepass already binds the full gas
  group; a declared-but-unused binding changes nothing) and the lights
  binding gains COMPUTE visibility. The K×R³ bake dispatch is 2-D
  (`(R³/64, K)`) to respect the 65535 per-dimension workgroup limit.
- xtask: `[look.gas]` gains optional `shadows` (bool). The knob PRESENT
  without a positive `scattering` is a dead knob → loud reject (the
  anisotropy discipline). Absent = `false`.

## Gates

CPU (`render/tests/shadows.rs`):
1. Bake analytic: uniform slab, hand-computed `T = exp(−κρ·chord)` at
   chosen voxel centers for an outside light (axial + oblique); a voxel
   with no gas toward the light, and a voxel coincident with the light,
   both exactly 1.
2. `ShadowVolumes::sample` oracle: exact voxel values at voxel centers,
   hand lerp between neighbors, clamp-to-edge outside the bounds
   (NOT zero).
3. Shadowed march analytic: backlit axial far-field slab — the camera-side
   and light-side exponentials compose to a constant integrand, closed form
   `C = σ_s·ρ·p(1)·I·L·e^{−τ}` (tolerance covers quadrature + 32³
   trilinear, ~2%).
4. Monotonicity: shadowed ≤ unshadowed per channel, strictly < where
   occluding gas exists; camera-path T bit-identical.
5. Off is off, bitwise: `shadows: false` (flag) and `shadows: None`
   (argument) each bit-identical to the v1 scatter march.
GPU (`render/tests/shadows.rs`):
6. GPU ≡ CPU per-pixel with shadows on — ortho + perspective, pattern
   grids, non-trivial mix, lights inside AND outside the domain — at the
   volume.rs tolerances (1e-3 rel + 1e-5 abs). Holds the on-device
   bake + trilinear + march against `bake_shadows` + `sample` + `march_gas`
   end-to-end.
7. GPU off-path bit-identity: `shadows: false` composite bit-identical to
   the v1 scatter frame; `scatter: None` stays bit-identical to the
   no-scatter frame (the new binding must not disturb the off path).
8. GPU linearity: 2× strength ⇒ exactly 2× flux WITH shadows on (`T_k` is
   σ_s-independent).
xtask (`tests/scenario_gas.rs`):
9. Knob parse + default false; `shadows = true` without `scattering`, and
   with `scattering = 0`, both reject; gasrich declares the shipped value.

## Milestones

- **U1 [red]**: gates above + API stubs (`todo!()` bodies) + mechanical
  call-site updates (`march_gas` second argument `None`, `ScatterLook`
  literals gain `shadows: false`). Workspace builds, gates fail.
  DONE (71e0be2) — 8 gates red; the two bitwise off-path pins pass by
  construction, as expected.
- **U1 [green]**: CPU bake/sample/march + WGSL mirrors + renderer plumbing;
  all gates green. DONE (0a5ad57) — all 10 shadow gates + full workspace.
- **U2**: xtask knob + QUICK A/B (gasrich, shadows off vs on at σ = 800).
  DONE — knob plumbing landed with U1. A/B result: the shadows-OFF run is
  byte-identical (all 481 PNGs, exr_diff-zero EXRs) to the retained
  `splat_cap_ab/cap3` render — the cross-version v1 regression the in-test
  gates cannot pin. Shadows ON: ~11% less mid-frame scattered flux
  (total_flux [8833, 7665, 8494] → [7843, 6824, 7590], peak 3.82 → 3.33),
  core haze slightly tightened, ambient glow + lit bridge intact — the
  expected self-shielding, subtle at QUICK. **gasrich ships `shadows =
  true`** (the scatter precedent: the showpiece exercises the option; off
  is one knob removal, gated bit-identical); σ = 800 kept — an 11% dip is
  well inside the 200/800 bracket step. Retained A/B:
  `M:\claud_projects\temp\shadow_ab` (off / on, each with movie.mp4).
- **U3**: docs (DESIGN.md scatter note, roadmap, this plan), memory,
  quality gate, commit + push.

## Perf note

Bake cost is K·R³ threads × ≤ ~450 steps (128³ diagonal at half-cell
steps): worst-case seconds/frame on FULL with all 512 bins occupied, far
lighter on QUICK — acceptable for an offline renderer, and it runs only
when the knob is on.

## Deferrals (named)

- DDA / hierarchical bake acceleration (the brute march is the reference).
- Hardware-filtered shadow lookups (manual trilinear keeps GPU ≡ CPU exact).
- Per-channel κ / colored shadows.
- Octree light clustering and scatter tint (inherited from v1).
