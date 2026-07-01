# M4e — GPU Karras tree-build kernel + atomic-flag bottom-up aggregation (third stage of the GPU-resident LBVH build)

## Context

DESIGN's **M4b** landed `galaxy_solvers::Lbvh`: the **CPU f64** reference for the
GPU-resident Morton/LBVH build — the full build *pipeline* run as the oracle
(bounding box → 30-bit Morton codes → sort by `(code, index)` → **Karras binary
radix tree → bottom-up aggregation** → DFS skip-pointer flatten → stackless walk).
DESIGN names the remaining M4 work as **the GPU port of that build, staged, each
stage gated vs the CPU reference**:

> Morton+bbox kernel (landed M4c) → GPU sort (landed M4d, the load-bearing risk) →
> **Karras tree-build kernel + atomic-flag bottom-up aggregation** → a `GpuLbvh` f32
> binary-BVH traversal kernel.

This plan is **stage 3: the Karras tree-build kernel + atomic-flag bottom-up
aggregation**. It is the GPU port of `karras_internal` (`solvers/src/lbvh.rs:188-237`)
and the bottom-up fold inside `flatten` (`solvers/src/lbvh.rs:449-468`). Its input is
the M4d GPU sort's `sorted_codes` (plus the leaf payload: per-leaf position + mass,
pre-gathered into sorted order). Its output is the **raw pointer-based binary radix
tree**: for each internal node its two children + parent, and for every node the
bottom-up-aggregated `{aabb_min, aabb_max, com, mass}`.

**Outcome:** a new `galaxy_gpu::lbvh_tree` module exposing a small non-`ForceSolver`
build stage `GpuLbvhBuilder` → `GpuLbvhTree`. It wires into no solver yet (there is
no `GpuLbvh`); it is validated directly against two newly-extracted CPU reference
functions — `reference_karras` (topology) and `reference_aggregate` (fold).

### The two-part gate — the key framing (state it plainly)

This stage splits cleanly into a **pure-integer** part and an **f32** part, and each
gets the *tightest gate its arithmetic allows* — exactly mirroring how M4d (integer
sort) was bit-exact where M4c (f32 quantize) was tolerance:

1. **Karras topology is pure integer ⇒ bit-exact gate.** The tree is built from
   `δ = clz(code_a ^ code_b)` over the sorted `u32` codes (with the `32 + clz(a^b)`
   tie-extension for equal codes, `lbvh.rs:153-165`). No floats enter. So given the
   same `sorted_codes`, the GPU topology (`left`, `right`, `parent`) must equal
   `reference_karras` **bit-for-bit** — the load-bearing gate, analogous to
   `order == reference_sort` in M4d.
2. **Aggregation is f32 ⇒ split gate.** The AABB `min`/`max` folds never round and are
   order-independent, so they are **bit-exact vs a CPU f32 fold** over the same
   f32-narrowed positions (same discipline as the M4c bbox reduction). Only `com`
   (mass-weighted division) and any derived `center`/`half`/`delta` are genuinely
   f32-lossy → **tolerance-gated** vs the f64 `reference_aggregate`. `mass` is an f32
   sum folded in fixed `(left, right)` order → deterministic, tolerance vs f64.

**This does not contradict the M4c f32-divergence scope note.** M4c warns the eventual
*end-to-end* GPU tree topology can differ from the CPU tree — but that divergence lives
**upstream, in the f32 Morton codes**, not in the Karras step. Gating *this* kernel in
isolation feeds it reference-sorted codes (== the GPU sort's bit-exact `sorted_codes`
per M4d), so the topology is bit-for-bit. The end-to-end f32 straddle stays deferred to
the later **θ→0 physics gate on `GpuLbvh`**.

### Scope honesty

**Explicitly deferred (documented, not built here):**
- **The DFS skip-pointer flatten.** M4e emits the *raw pointer tree* (parent + tagged
  children + raw `min`/`max`/`com`/`mass`), **not** the `LbvhFlat` DFS pre-order
  skip-pointer form. Deriving `center`/`half_extents`/`delta` and the `next` skip
  pointer is a separate concern. **Named next step:** a flatten stage that lowers the
  pointer tree to the CPU `LbvhFlat` layout (a subtree-size prefix-sum / Euler-tour
  parallel primitive), so `GpuLbvh` can traverse the **same DFS skip-pointer form the
  CPU `LbvhFlat::accel` walk uses** — reusing the M4b structural gate and the CPU walk
  as its oracle. (Escape/"rope" pointers are the local-computation alternative if the
  prefix-sum flatten proves heavy; the raw pointer tree M4e emits suffices for either.)
- **`GpuLbvh` traversal**, GPU-resident state, 63-bit Morton, a parallel (multi-tile)
  aggregation. This stage's aggregation is reference-grade (one thread per leaf walking
  up), not the scale build.

## Decisions locked before tests (advisor)

1. **The WGSL Karras search is signed `i32`, not `u32`** — the #1 silent-corruption
   risk. `karras_internal` deliberately probes out-of-range indices and `delta` returns
   **−1** for them (`lbvh.rs:155`). Under `u32`, −1 is `0xFFFFFFFF` and compares
   **greater** than any valid δ, breaking (a) the direction pick `δ(i,i+1) > δ(i,i-1)`,
   (b) the `while δ(...) > delta_min` range search, and (c) `gamma = i + s·dir +
   dir.min(0)` (`0xFFFFFFFF.min(0) == 0` under `u32`, must be −1). δ's range is
   `[−1, 63]` (the `32 + clz` tie extension), which fits `i32` — make **δ and the search
   indices `i32`**. A `u32` port can pass near-origin tests and still be wrong at range
   boundaries; the **all-equal-codes** gate is what surfaces it (every node falls onto
   the `32 + clz(a^b)` position tie-break). Also assert-in-comment that WGSL
   `countLeadingZeros(0u) == 32u` matches Rust `0u32.leading_zeros()`.
2. **Zero the atomic visit-counter buffer every build.** The "second child to arrive
   folds the parent" logic needs each internal node's counter starting at 0. Buffers are
   reused/grown via the `ensure_capacity` idiom, so a stale nonzero counter makes a
   parent fold on the *first* arrival with one child's AABB missing. Clear the counter
   region (a `clear_buffer` or a zero `write_buffer`) before the aggregation dispatch.
3. **The atomic is an integer flag/counter, never a float `atomicAdd`.** The visit
   counter (per internal node) only gates *when* a node folds; the fold itself reads the
   node's **stored `left`/`right` child indices** and combines them in fixed
   `(left, right)` order — so the result is independent of which child thread won the
   atomic. Determinism is structural (the M3/M4 "gather, not scatter" discipline), not
   luck. Same-device bit-determinism is gated on **aggregation as well as topology**.
4. **Gate the AABB in `min`/`max` space, not `center`/`half`.** `min`/`max` folds are
   exact and order-independent → bit-exact vs a CPU f32 fold. `center = (min+max)/2`,
   `half = (max−min)/2`, `delta = |com−center|` introduce f32 rounding — they are
   **derived, tolerance-only**, and belong to the deferred flatten stage. So the stage
   output stores raw `min`/`max`/`com`/`mass`.
5. **Share the per-node search + the fold between reference and the existing path** —
   don't duplicate the Karras tie-break. Extract `karras_node(codes, i) -> (left,
   right)` (the body of the current per-node closure) and `fold_agg(left, right) ->
   Agg` (the fixed-order fold at `lbvh.rs:449-468`, incl. the massless-midpoint
   fallback). `karras_internal` and `flatten` call them → their output is **bit-identical**.
   `reference_karras`/`reference_aggregate` build on the same helpers. A second Karras
   that disagrees on the tie-break is the drift risk this avoids.
6. **Pass pre-gathered sorted pos/mass** (leaf `k` → `pos[k]`, `mass[k]`), not
   `order` + unsorted with a GPU-side gather. Cleanest boundary for a reference-grade
   stage; the GPU-side gather is a named perf deferral.

## Approach

### CPU-side refactor (single source of truth)

In `solvers/src/lbvh.rs`, behavior-preserving:

```rust
/// Canonical unified node index space for the GPU-shaped Karras tree:
/// leaves occupy `[0, N)` (sorted order), internal nodes `[N, 2N-1)` (internal
/// node `i` at `N+i`; the root is internal 0 ⇒ unified index `N`). A unified
/// index `u` is a leaf iff `u < N`. Parent of the root is `NO_PARENT` (u32::MAX).
pub const NO_PARENT: u32 = u32::MAX;

pub struct KarrasTree {
    pub n: usize,
    /// len `N-1`; unified child indices of internal node `i`.
    pub left: Vec<u32>,
    pub right: Vec<u32>,
    /// len `2N-1`; parent unified index per node (`NO_PARENT` for the root).
    pub parent: Vec<u32>,
}
pub fn reference_karras(sorted_codes: &[u32]) -> KarrasTree;

/// f64 bottom-up aggregate per unified node (len `2N-1` each), folded in the same
/// fixed (left,right) order and massless-midpoint fallback as `flatten`.
pub struct KarrasAgg {
    pub aabb_min: Vec<DVec3>,
    pub aabb_max: Vec<DVec3>,
    pub com: Vec<DVec3>,
    pub mass: Vec<f64>,
}
pub fn reference_aggregate(tree: &KarrasTree, sorted_pos: &[DVec3], sorted_mass: &[f64]) -> KarrasAgg;
```

- Extract `karras_node(codes, i)` from the closure in `karras_internal`; `karras_internal`
  becomes `(0..n-1).map(|i| karras_node(codes, i)).collect()` (returns the existing
  `Internal{left,right: ChildRef}` — unchanged). `reference_karras` also calls
  `karras_node`, converts each `ChildRef{leaf,idx}` to a unified index (leaf → `idx`,
  internal → `N + idx`), and records `parent[child] = N + i`.
- Extract `fold_agg(la, ra) -> Agg` from the fold at `lbvh.rs:449-468`; `flatten`'s inline
  fold becomes a call to it. `reference_aggregate` walks the tree bottom-up (children
  before parents — the unified layout has internal `i`'s children at strictly smaller
  internal indices in Karras order is *not* guaranteed, so aggregate by a post-order /
  visited-count pass, or recurse from the root) using `fold_agg`, seeding leaves from
  `sorted_pos`/`sorted_mass`.
- Re-export the new pub items from `solvers/src/lib.rs`.

**Hard gate:** `cargo test -p galaxy-solvers` stays green **bit-identical** after the
extraction, *before* any GPU code is written.

### GPU stage (`gpu/src/lbvh_tree.rs`)

Reuse the `GpuMortonBuilder`/`GpuSorter` context idiom verbatim (headless adapter →
device/queue, `Features::empty()`, lazily grown storage buffers, `pollster::block_on`
bringup returning typed `GpuError`, N=0/1 handled CPU-side with no dispatch).

**Two compute passes, one command encoder:**

1. **`build` (Karras topology)** — one invocation per internal node (`WG=256` tiling,
   dispatch `(N-1).div_ceil(256)`). Reads `sorted_codes` (storage `array<u32>`), computes
   `δ`/direction/range/split with **signed `i32`** throughout (per decision 1), writes
   `left[i]`, `right[i]` (unified indices) and, for each of its two children,
   `parent[child] = N + i`. The root's parent is pre-seeded `NO_PARENT`.
2. **`aggregate` (atomic-flag bottom-up)** — one invocation per **leaf** (dispatch
   `N.div_ceil(256)`). Leaf `k` seeds `aabb_min[k]=aabb_max[k]=com[k]=pos[k]`,
   `mass[k]=mass_k`, then walks up via `parent`: at each internal node it
   `atomicAdd(counter[node-N], 1u)`; the **first** child to arrive (old value 0) returns;
   the **second** (old value 1 — both children final) folds the parent from its stored
   `left`/`right` in fixed order (mass sum, mass-weighted com with massless-midpoint
   fallback, `min`/`max` union), then continues to *its* parent until it folds the root
   and stops. `counter` (storage `array<atomic<u32>>`, len `N-1`) is **zeroed before the
   pass** (decision 2).

`aabb_min`/`aabb_max`/`com` are `vec4<f32>` storage (w unused, 16-byte aligned like the
Morton stage's `lanes`); `mass` is `array<f32>`. Read back topology + aggregates, widen
to the return type.

```rust
pub struct GpuLbvhTree {
    pub n: usize,
    pub left: Vec<u32>,    // len N-1 (unified child indices)
    pub right: Vec<u32>,   // len N-1
    pub parent: Vec<u32>,  // len 2N-1 (NO_PARENT for root)
    pub aabb_min: Vec<[f32; 3]>, // len 2N-1
    pub aabb_max: Vec<[f32; 3]>, // len 2N-1
    pub com: Vec<[f32; 3]>,      // len 2N-1
    pub mass: Vec<f32>,          // len 2N-1
}
```

N=1 short-circuits on the CPU: one leaf, `parent=[NO_PARENT]`, its own position as the
AABB/com, its mass. N=0 → empty result.

## Files

- **`solvers/src/lbvh.rs`** — extract `karras_node` + `fold_agg`; add `reference_karras`
  + `reference_aggregate` + `KarrasTree`/`KarrasAgg`/`NO_PARENT` (pub). Behavior-preserving.
- **`solvers/src/lib.rs`** — re-export the new public reference types/fns.
- **`gpu/src/lbvh_tree.rs`** (new) — `GpuLbvhBuilder` stage + `GpuLbvhTree` output.
- **`gpu/src/lib.rs`** — `pub mod lbvh_tree;` + re-export; extend the crate doc with the
  "topology bit-exact / aggregation f32; raw pointer tree, flatten deferred" scope note.
- **`gpu/tests/lbvh_tree.rs`** (new) — GPU-gated gates (below), reusing the `cloud` LCG
  helper block from the other GPU test files.
- **`DESIGN.md`** — add an **M4e** bullet under M4; update the "Remaining M4+" prose so
  the *next* stage is the flatten → `GpuLbvh` traversal.

## TDD gates (red-first, committed separately as `test(...): ... [red]`)

All GPU-gated (need a wgpu adapter; fail loud on `NoAdapter`, per M3/M4 convention).

**Topology (bit-exact, integer — the load-bearing gate):**
- GPU `(left, right, parent)` **== `reference_karras`** bit-for-bit over: seeded Morton-code
  clouds; **all-equal codes** (forces every node onto the `32 + clz(a^b)` position tie-break
  — the case that surfaces an `i32`/`u32` signedness bug or a `countLeadingZeros(0)`
  mismatch); heavy-duplicate codes; a monotone chain (`0,1,2,…` — degenerate right-leaning
  tree); sorted and reversed adversarial orderings; large N (2¹⁶).
- Structural self-consistency (recomputed on CPU from the read-back tree): `2N-1` nodes;
  the `N` leaves are exactly the unified indices `[0, N)`; every child's `parent` points
  back to the node that lists it; the root (unified `N`) has `parent == NO_PARENT`.

**Aggregation:**
- **AABB bit-exact:** GPU `aabb_min`/`aabb_max` equal a CPU **f32** fold over the same
  f32-narrowed sorted positions (min/max never round, order-independent). Include a
  degenerate case (coincident / collinear leaves → zero-extent boxes).
- **com/mass tolerance:** GPU `com`/`mass` match f64 `reference_aggregate` within an f32
  tolerance justified by the fold depth; every child's AABB ⊆ its parent's AABB; the
  root AABB contains all input positions.
- **same-device determinism:** run the stage twice on identical input ⇒ **bit-identical**
  topology **and** aggregation (proves the fold is order-independent under the atomic
  race, not lucky).

**Edges:** N=1 (single leaf, `NO_PARENT`, AABB = its position); N=2 (one internal, two
leaves); coincident particles (distinct leaves, identical positions); N=0 (empty, no
dispatch).

**CPU-side (solvers) refactor gate:** the existing `solvers/tests/lbvh.rs` + in-module
tests must **stay green bit-identical** after the `karras_node`/`fold_agg` extraction —
run before writing any GPU code.

## Verification

1. `cargo test -p galaxy-solvers` — confirm the refactor keeps every existing `lbvh` test
   green bit-identical **before** touching GPU code (behavior-preserving).
2. `cargo test -p galaxy-gpu` — confirm the new GPU gates **fail** first (red commit),
   then pass after the kernel lands (green commit). Never edit a test to pass.
3. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
4. `cargo test` (workspace) — nothing else regresses.

## Commit sequence

1. `refactor(solvers): expose reference_karras + reference_aggregate (single source of truth)`
   — behavior-preserving extraction; existing tests stay green (no `[red]`).
2. `test(gpu): red gates for the GPU Karras tree-build + aggregation stage [red]` (+ API
   stubs with `todo!()` bodies so the tests compile and fail).
3. `feat(gpu): GPU Karras tree-build kernel + atomic-flag bottom-up aggregation (M4e)`.
4. `docs(design): land GPU Karras tree-build + aggregation (M4e); next stage is the flatten → GpuLbvh traversal`.

Then push (per memory: push to origin after each batch).
