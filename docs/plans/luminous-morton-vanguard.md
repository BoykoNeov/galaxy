# M4c — GPU Morton + bounding-box kernel (first stage of the GPU-resident LBVH build)

## Context

DESIGN's **M4b** landed `galaxy_solvers::Lbvh`: the **CPU f64** reference for the
GPU-resident Morton/LBVH build — the full build *pipeline* run as the oracle
(bounding box → 30-bit Morton codes → sort by `(code, index)` → Karras binary
radix tree → bottom-up aggregation → DFS skip-pointer flatten). DESIGN names the
remaining M4 work as **the GPU port of that build, staged, each stage gated vs the
CPU reference**:

> Morton+bbox kernel → **GPU sort** [the load-bearing risk] → Karras tree-build
> kernel + atomic-flag bottom-up aggregation → a `GpuLbvh` f32 binary-BVH traversal
> kernel.

This plan is **stage 1: the GPU Morton + bounding-box kernel** — the smallest,
lowest-risk slice, exactly how M4b was sliced off M4a. It is the GPU port of the
*prologue* of `LbvhFlat::build` (`solvers/src/lbvh.rs:214-232`): compute the root
bounding cube over `pos`, then quantize each position to a 30-bit Morton code.

**Outcome:** a new `galaxy_gpu::lbvh_morton` module exposing a small non-`ForceSolver`
build stage — given positions, it computes the bounding box **and** the per-particle
Morton codes on the GPU (wgpu compute, f32), returned to the CPU for gating. It wires
into no solver yet (there is no `GpuLbvh`); it is validated directly against the CPU
`Lbvh` reference's bbox + codes.

### Scope honesty (state this plainly, per advisor)

This stage proves **quantization + the GPU reduction pattern**. It deliberately does
**not** prove the tree matches the reference — GPU codes computed in **f32** will
diverge from the **f64** reference near cell boundaries (a straddling particle floors
into an adjacent 1024³ cell), so the eventual GPU tree *topology* can differ from the
CPU tree. **That is expected, the exact analogue of the θ-straddle in `GpuTree`**, not
a bug. The real correctness check is the later **θ→0 physics gate on `GpuLbvh`**; this
stage's gates are **structural + tolerance + determinism** only.

**Explicitly deferred (documented, not built here):** GPU sort (next stage, the
load-bearing deterministic `u32` radix), Karras tree-build kernel + atomic-flag
aggregation, `GpuLbvh` traversal, 63-bit Morton (2× `u32` sort passes).

## Decisions locked before tests (advisor)

1. **No bit-equality vs the f64 reference.** The GPU has no portable f64 compute
   (`SHADER_FLOAT64` rarely present — same constraint as `GpuDirectSum`/`GpuTree`), so
   codes run in **f32**. Gate on **tolerance + determinism**, not bit-match.
2. **Compare quantized *lanes*, not interleaved codes.** A 1-bit lane change makes the
   Morton code jump by a large power of two, so a tolerance is only meaningful in lane
   space. The kernel outputs the three quantized lanes `[u32;3]` (x,y,z) per particle
   **alongside** the interleaved code; the primary gate asserts `|gpu_lane − ref_lane|
   ≤ 1` per axis in the near-origin regime.
3. **bbox on the GPU** (keeps DESIGN's "Morton+bbox" intact), via a **single-workgroup
   grid-stride shared-memory reduction**. min/max never round and are order-independent,
   so a fixed-order tree reduce is **bit-exact and deterministic for free** — no float
   atomics needed.
   - **Landmine (flag, do not step on):** WGSL has **no f32 atomics** (only
     `atomic<i32>`/`atomic<u32>`). Do **not** use cross-workgroup atomic min/max — that
     forces the monotone-ordered-bitcast trick and is a rabbit hole for a "smallest
     step." Single-workgroup reduction sidesteps it (perf is irrelevant for a reference
     stage; if multi-workgroup is ever needed, use two dispatches: partials → final,
     still no float atomics).
4. **Expose the CPU reference as a `pub fn`** in `solvers/src/lbvh.rs`, returning the
   quantized **lanes** (not just codes) + bbox. `LbvhFlat::build` is refactored to call
   it (single source of truth). This is a **behavior-preserving extraction** — the
   existing `Lbvh` tests must stay green **bit-identical** (confirm before touching GPU
   code).

## Approach

### CPU-side refactor (single source of truth)

Extract the bbox + code prologue from `LbvhFlat::build` into a public reference API in
`solvers/src/lbvh.rs`. Shape (names illustrative):

```rust
/// The root bounding cube convention shared by the CPU build and the GPU-build gate.
pub struct MortonBounds { pub bmin: DVec3, pub size: f64, pub scale: f64 }

/// Reference (f64) Morton quantization: the exact prologue the GPU stage ports.
/// Returns the bounds, per-particle quantized lanes, and interleaved 30-bit codes.
pub struct MortonReference { pub bounds: MortonBounds, pub lanes: Vec<[u32;3]>, pub codes: Vec<u32> }
pub fn reference_morton(pos: &[DVec3]) -> MortonReference;
```

`LbvhFlat::build` then calls `reference_morton(pos)` for its `bmin`/`scale`/`codes`
(the `.sort_by_key` and everything downstream unchanged). Split `morton_code` so it
returns the lanes and the interleave separately (`morton3` is already separate, so this
is trivial). **Expose lanes** because the ±1 gate needs them.

**Exact-convention checklist** — the reference (and therefore the GPU port it gates)
must preserve every constant from the current `build`:
- `center = 0.5*(lo+hi)`, `half = (0.5*(hi-lo).max_element()).max(1e-12) * (1.0+1e-9)`
- `bmin = center - splat(half)`, `size = 2*half`, `scale = 1024 / size`
- quantizer `q(v) = (floor((p-bmin)·scale).max(0.0) as u32).min(1023)`
- 30-bit interleave via `expand10`/`morton3`

### GPU stage (`gpu/src/lbvh_morton.rs`)

Reuse the `GpuDirectSum` context idiom verbatim: headless adapter → device/queue,
`Features::empty()` (baseline storage-buffer compute, no adapter narrowing), lazily
grown storage buffers, `pollster::block_on` bringup returning typed `GpuError`
(`NoAdapter` on a GPU-less box → tests fail loud like the M3/M4 GPU gates).

**Two compute passes, f32 throughout, one command encoder:**

1. **bbox reduction** — a single workgroup (e.g. 256 lanes) grid-strides the whole
   `pos` array into per-lane private min/max, writes them to `var<workgroup>` arrays,
   then a fixed-order tree reduction (`for stride = WG/2; stride>0; stride>>=1`) folds
   to lane 0, which writes `bbox_min`/`bbox_max` (`vec4<f32>`) to a small storage
   buffer. Deterministic (min/max exact, fixed fold order). Padded lanes seed
   `+INF`/`−INF` so they never win.
2. **derive bounds + quantize** — a second pass (one invocation per particle, standard
   `WG=256` tiling) reads `bbox_min`/`bbox_max`, reconstructs `center/half/bmin/size/
   scale` with the **exact same constant sequence** as the CPU reference (in f32), then
   per particle computes the three lanes (`floor`, `max(0)`, `min(1023)`) and the 30-bit
   interleave, writing `lanes` (`vec4<u32>`, w unused) and `code` (`u32`) storage
   buffers.

   - The `half/center` derivation could also be done on the CPU between passes and
     pushed as a uniform, but doing it in-shader keeps the whole "bbox→bounds→code" path
     GPU-resident (the milestone's point) and is the code the sort stage will build on.

Read back `lanes` + `codes`, widen to the return type. Output struct:

```rust
pub struct GpuMorton { pub bounds: MortonBounds, pub lanes: Vec<[u32;3]>, pub codes: Vec<u32> }
```

`bounds` here is reconstructed from the read-back f32 bbox (for the bbox gate). N=0 is
handled on the CPU (empty result, no dispatch), matching the solver convention.

## Files

- **`solvers/src/lbvh.rs`** — extract `reference_morton` + `MortonBounds`/
  `MortonReference` (pub); `LbvhFlat::build` calls it. Behavior-preserving.
- **`solvers/src/lib.rs`** — re-export the new public reference types/fn.
- **`gpu/src/lbvh_morton.rs`** (new) — the GPU Morton+bbox stage + `GpuMorton` output.
- **`gpu/src/lib.rs`** — `pub mod lbvh_morton;` + re-export; extend the crate doc with
  the "proves quantization + reduction, not tree match" scope note.
- **`gpu/tests/lbvh_morton.rs`** (new) — GPU-gated gates (below), reusing the `cluster`
  LCG helper block copied in the other GPU test files.
- **`DESIGN.md`** — add an **M4c** bullet under M4 documenting this GPU Morton+bbox
  stage; update the "Remaining M4+" prose so the *next* stage is the GPU sort.

## TDD gates (red-first, committed separately as `test(...): ... [red]`)

All GPU-gated (need a wgpu adapter; fail loud on `NoAdapter`, per M3/M4 convention).
Split by error source so a failure localizes (advisor):

- **bbox reduction correctness** — the GPU bbox equals a CPU reduction over the **same
  f32-narrowed positions**, bit-exact (min/max never round, order-independent). Isolates
  the reduction logic from f32-vs-f64. Include a case that exercises the `.max(1e-12)`
  floor (collinear / coincident points → zero-extent axis).
- **per-lane reference agreement (±1)** — over seeded near-origin clusters (coords ~few
  units, the well-conditioned regime), assert `|gpu_lane − ref_lane| ≤ 1` for each of
  x,y,z per particle, **and** that the vast majority match exactly. This is the core
  "GPU quantization ≈ f64 reference" gate.
- **large-coordinate characterization (not an assertion)** — a rigidly-offset cluster
  (|x|≈5000) is *documented* to diverge more widely (the f32 pre-floor value loses
  conditioning), mirroring the direct-sum "unit-box 3e-4 / |x|≈5000 5e-3" honesty. Assert
  only the loose structural bound (still in range), and record the observed max lane gap
  in a comment — do **not** pin ±1 here.
- **structural** — every GPU code in `[0, 2³⁰)`; `lanes` each in `[0, 1024)`; the code
  equals `morton3` of its own lanes (interleave self-consistency, recomputed on CPU from
  the read-back lanes).
- **same-device determinism** — run the stage twice on identical input ⇒ **bit-identical**
  codes **and** lanes (the hard claim; matches `GpuDirectSum`/`GpuTree`).
- **edge cases** — N=1 (one well-defined code, bbox is the `1e-12`-floored degenerate
  cube); coincident particles (identical lanes ⇒ identical codes); N=0 (empty result, no
  dispatch).

**CPU-side (solvers) refactor gate:** the existing `solvers/tests/lbvh.rs` + the
in-module `morton_tests` must **stay green bit-identical** after the extraction — this
is the behavior-preserving check; run it before writing any GPU code.

*Optional sharpening (non-blocking, add if cheap):* a **CPU f32 reference** with the
identical op sequence gives a tight algorithm-isolation gate (exact modulo GPU FMA
reorder), separating "kernel logic right" from "f32-vs-f64 precision." The ±1-vs-f64
gate alone is sufficient to land the milestone.

## Verification

1. `cargo test -p galaxy-solvers` — confirm the refactor keeps every existing `lbvh`
   test green **before** touching GPU code (behavior-preserving).
2. `cargo test -p galaxy-gpu` — confirm the new GPU gates **fail** first (red commit),
   then pass after the kernel lands (green commit). Never edit a test to pass.
3. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
4. `cargo test` (workspace) — nothing else regresses.

## Commit sequence

1. `refactor(solvers): expose reference Morton bbox+lanes+codes (single source of truth)`
   — behavior-preserving extraction; existing tests stay green (no `[red]`).
2. `test(gpu): red gates for GPU Morton+bbox build stage [red]` (+ API stubs with
   `todo!()` bodies so the tests compile and fail).
3. `feat(gpu): GPU Morton + bounding-box kernel (first LBVH-build stage)`.
4. `docs(design): land GPU Morton+bbox stage (M4c); next remaining stage is the GPU sort`.

Then push (per memory: push to origin after each batch).
