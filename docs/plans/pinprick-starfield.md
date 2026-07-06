# Pinprick starfield — screen-space cap on splat size

Follow-up to the scattered-starlit-veil A/B: the user flagged that early
gasrich shots read *defocused* while late shots are sharp. Diagnosis (from the
retained `scatter_ab` frames — scatter-off shows it identically, so it is not
the scattering): splats are **world-space** sized (M6g, `splat_size` sim
units) while the orbit-tilt rig frames from the per-snapshot percentile
radius. Early on the scene is compact → the camera is zoomed in → each
0.12-unit splat covers ~a 12 px blob at 640×360 and the overlapping Gaussians
read as defocus; by the end the tidal tails have widened the framing
severalfold and the same splats shrink to ~2–3 px points. Resolution doesn't
change the blob-to-frame ratio; it is a zoom artifact, present since M6g and
most visible under gasrich's tight `frame_percentile = 0.92`.

## The fix: `max_splat_px`

A real camera's star PSF is fixed in *pixels*, not world units — point
sources stay point-like at any zoom. Add the missing half of the existing
pixel-clamp window:

- `RenderConfig.max_splat_px: f32` — maximum on-screen splat **half-extent
  in pixels** (the `min_splat_px` convention). Default `f32::INFINITY` = off.
- Applies to **both projections**. This is load-bearing: the orbit-tilt rig
  (what gasrich uses) is *orthographic*, and the ortho branch currently has
  no clamps at all — a perspective-only knob would never bite.
- **Flux-conserving**: clamping down boosts emission by `(true/clamped)²`,
  the exact mirror of the `min_splat_px` dimming. Integrated flux is
  invariant under the cap — the clamp window `[min_splat_px, max_splat_px]`
  only reshapes the PSF, never the photometry, so the M6g surface-brightness
  → 1/d² flux law survives verbatim. Zooming into a star concentrates its
  flux into a brighter point (real astrophoto behavior; the tonemap/bloom
  chain is built for HDR peaks).
- **`max_splat_ndc` is unchanged and stays outermost**: it is a fill-rate
  guard against the 1/z divergence near the eye and deliberately *saturates*
  (no boost). Order: PSF window first (flux-conserving), guard second
  (saturating).
- Off is REALLY off: the cap is a taken-branch only when it bites, encoded
  as 0-means-off in the uniform (`viewport.z`, previously padding — the
  uniform block is shared by the star/gas/prepass shaders, which ignore it).
  Cap disabled ⇒ bit-identical arithmetic in both branches; the M6g ortho
  golden keeps pinning the default path.

Validation (fail loudly): a finite `max_splat_px` must be `> 0`; under
perspective it must also be `≥ min_splat_px` (a crossed window would make
the WGSL clamp UB — same rationale as the existing `clamps_valid` check).

xtask: optional `[look] max_splat_px` (default absent = off), validated
finite `> 0`; plumbed into the movie `RenderConfig`. Pixel units are
resolution-literal (the `min_splat_px` precedent): QUICK 360p ↔ FULL 1080p
needs ~3× the value for the same relative look — documented at the knob.

## Gates

`render/tests/splat_cap.rs` (GPU-gated like `vertex_path.rs`):
1. **Off-is-off (ortho)**: the 40-splat golden scene with a huge finite cap
   (1e6 px, enabled-but-never-biting) renders bit-identical to the off
   render; default config keeps matching the existing M6g golden gate.
2. **Cap conserves flux (ortho)**: one well-resolved splat (~38 px
   half-extent at 256²), cap at ~10 px → total flux equal to the uncapped
   render within 2% (both footprints well-resolved; Gaussian truncation is
   scale-relative so it cancels — the tolerance matches the inverse-square
   gate).
3. **Cap concentrates, not dims (ortho)**: same pair — the capped peak
   pixel is brighter by ≈ (true/clamped)² (2% tol), and the lit footprint
   shrinks to the cap.
4. **Perspective flux law survives the cap**: the inverse-square pair with
   the near splat capped → flux ratio still 4 within 2% (the cap must not
   break the 1/d² law it exists to coexist with).
5. **Window composition (perspective)**: `min_splat_px = 8, max_splat_px
   = 12` with one sub-pixel splat (clamped UP, dimmed) and one oversized
   splat (clamped DOWN, boosted) in opposite halves — both flux-correct.
6. **Validation**: cap ≤ 0 / NaN errors under both projections; finite cap
   < `min_splat_px` errors under perspective.

`xtask/tests/scenario_gas.rs` (or sibling): knob parses; absent ⇒ off;
rejects non-finite/≤ 0; gasrich declares it (`> 0`, exact value stays
tunable).

## Milestones

- **P1 [red]**: gates + API surface (`RenderConfig.max_splat_px` defaulting
  to `INFINITY`, spec/runtime Option field) — no shader/validation logic, so
  the behavior gates fail; workspace still builds.
- **P1 [green]**: WGSL clamp window + uniform plumbing + validation + xtask
  wiring, all gates green.
- **P2**: gasrich A/B on the reused `scatter_ab` snapshots (no re-sim) —
  bracket the cap at 360p (early blobs are ~6–7 px half-extent, late points
  ~2–3 px; candidates {2, 3, 4} px), pick by eyeball on early-shot sharpness
  vs core blowout, ship in the preset with a scale comment (FULL ≈ 3× the
  QUICK value).
- **P3**: docs (DESIGN.md rendering-recipe note), memory, quality gate,
  commit, push.

## Deferrals (named)

- Angular-size spec (fraction of frame height) instead of literal pixels —
  would make QUICK/FULL tuning transfer automatic; take it up if the 3×
  rule-of-thumb annoys.
- Applying the cap to the other presets — gasrich only for now; the rest
  keep the M6g look until judged.
