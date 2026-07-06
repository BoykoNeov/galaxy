# Scattered starlit veil — single-scatter starlight on gas

Phase-2 visual item 1 of `long-burning-beacon.md`: illuminate the gas volume
from the stellar distribution, as an **option** that can be disabled if the
plain M7e emission+absorption look is judged sufficient (`scattering = 0`
or omitting the knobs restores today's output bit-for-bit).

## Physical model (v1)

The gas march gains a scattered source term alongside emission:

```text
L(pixel) = Σ_stars E·T(cam→star)
         + ∫ [ j·ρ(s) + j_scat(s) ] · T(cam→s) ds

j_scat(s) = σ_s · ρ(s) · Σ_k  p_HG(cosθ_k, g) · L_k / (4π · (d_k² + r_k²))
```

- **σ_s** (`scattering`): scattering coefficient per unit (ρ·length), the
  same units family as κ — a look knob, tuned by eyeball like κ/j (D8:
  knobs live in the look, the grid stays ρ-only). `0` = feature off.
- **Lights**: the stellar splats clustered into ≤ `LIGHT_BINS³` point
  proxies by emission-weighted binning over the star AABB. Each light
  carries RGB power `L_k = Σ color·brightness` (power conserved exactly)
  and a softening radius `r_k` = half its bin-cell diagonal (inside that
  radius the point-proxy approximation is invalid anyway — softening there
  is honest, and it kills the 1/d² pole).
- **Phase**: Henyey–Greenstein, `g` = `anisotropy` ∈ (−1, 1);
  `cosθ = ω_in·ω_out` with `ω_in` the light→sample propagation direction
  and `ω_out` the sample→camera direction. `g = 0` is isotropic (1/4π);
  `g > 0` is forward scattering — the backlit silver-lining look.
- **Attenuation**: scattered radiance is emitted inside the march, so it is
  attenuated toward the camera by the same T as emission (order-independent
  additive target survives). Light→sample self-shadowing is **deferred**
  (v2: per-light shadow volumes — a K×voxels compute prepass); v1 is
  unshadowed single scatter, documented at the knob.

Why this respects the standing constraints:
- **D9 (no prep-time camera dependence)**: the light list is
  camera-independent per-frame data; the only view-dependent factor (the
  phase angle) is evaluated at render time inside the march.
- **D8 (ρ-only voxels)**: nothing new is baked into the grid; lights derive
  from the frame's splats at render time, knobs live in `GasLook`.
- **Ordered gas compositing**: the term is added inside the existing
  front-to-back march, not as a separate additive splat.

## API

- `volume::Light { pos: Vec3, radius: f32, rgb: [f32; 3] }`
- `volume::cluster_lights(&FrameData) -> Vec<Light>` — deterministic
  index-order fold, f64 accumulators, bins ordered by bin index;
  `LIGHT_BINS = 8` is a quality **constant** (like grid res), not a knob.
- `volume::hg_phase(cos_theta, g)` — public so gates can oracle it.
- `volume::ScatterLook { strength: f32, anisotropy: f32 }`;
  `GasLook.scatter: Option<ScatterLook>` (`None` = off = bit-compat).
- `GasFrame.lights: &[Light]` (empty when scatter is off).
- WGSL: lights storage buffer at gas-group binding 3 (fragment-only; the
  prepass shares the layout but never reads it), `GasUniforms` gains a
  `scat` vec4 (strength, g, light count). CPU `march_gas` remains the
  operation-for-operation oracle.
- xtask: `[look.gas]` gains optional `scattering` / `anisotropy`;
  `GasLookValues` carries them as plain values (default 0 = off).
  Validation: `scattering` finite ≥ 0, `anisotropy` finite with |g| < 1,
  and `anisotropy` without a positive `scattering` is a dead knob → loud
  reject (the Some-iff discipline).

## Gates

CPU (`render/tests/scatter.rs`):
1. HG phase: ∫p dΩ = 1 by quadrature for g ∈ {0, 0.4, −0.7}; g = 0 is
   exactly 1/4π.
2. `cluster_lights`: hand star set — same-bin stars merge to the
   emission-weighted centroid, distant stars stay separate lights, total
   RGB power conserved exactly, zero-brightness stars drop out.
3. Far-field slab analytic: one light at D ≫ slab, transverse geometry
   (cosθ = 0) → radiance = (σ_s·I·p/κ)(1−e^{−κρL}), hand-computed p for
   g = 0 and g ≠ 0.
4. Inverse-square: light at D vs 2D → exactly 4× scattered radiance
   (within quadrature tolerance).
5. Off-is-off: strength 0 with lights present, and lights empty with
   strength > 0, both bit-identical to `scatter: None`.
6. Linearity: 2× strength ⇒ exactly 2× scattered radiance (emissivity 0),
   T bit-identical.
7. Forward-scattering ordering: backlit light (g = 0.6) out-scatters the
   mirrored front-lit geometry; equal at g = 0.
GPU:
8. GPU ≡ CPU per-pixel (ortho + perspective, pattern grids, hand-built
   light list, g ≠ 0) at the volume.rs tolerances (1e-3 rel + 1e-5 abs).
9. Scatter-off GPU frame bit-identical to the no-scatter frame (the M6g /
   gas-off golden in volume.rs keeps pinning the fully-off path).
10. GPU 2× strength ⇒ exactly 2× flux.
xtask (`tests/scenario_gas.rs`):
11. Knob parse + defaults; validation rejects (negative σ_s, |g| ≥ 1,
    dead anisotropy); gasrich preset declares the option.

## Milestones

- **S1 [red]**: gates above + API stubs (`todo!()` bodies), mechanical
  `scatter: None` / `lights: &[]` literal updates so the workspace builds.
- **S1 [green]**: CPU reference + WGSL mirror + renderer plumbing.
- **S2**: xtask knob plumbing + movie path (`cluster_lights` per emitted
  frame), gasrich preset tuned by eyeball (QUICK A/B render, scatter
  on/off), retained demo under `M:\claud_projects\temp`.
- **S3**: docs (DESIGN.md note, roadmap tick), memory, quality gate, push.

## Deferrals (named)

- Light→sample shadow volumes (per-light 3D transmittance prepass).
- Octree / adaptive light clustering (fixed-bin is v1).
- A scatter tint knob (scattered light carries the light's own RGB in v1).
