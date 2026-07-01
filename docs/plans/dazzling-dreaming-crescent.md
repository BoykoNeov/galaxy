# M4b — CPU LBVH reference solver (first step of GPU-resident Morton/LBVH build)

## Context

DESIGN's M4a landed `GpuTree`: a GPU Barnes-Hut solver that is **CPU-build +
GPU-traverse**. The octree is still built and linearized on the CPU
(`galaxy_solvers::FlatTree`) and only *walked* on the GPU. DESIGN names the next
step explicitly:

> a **GPU-resident build (Morton/LBVH)** is the next deferred step, and the CPU
> build becomes the Amdahl ceiling well before 10⁸.

A full GPU-resident LBVH is a multi-kernel subproject: bounding box → Morton
codes → **sort** → Karras binary radix tree → bottom-up aggregation → a new
binary-BVH traversal kernel, plus GPU-sort determinism work. The user asked for
**a smaller step now, the rest in future**.

The smaller step is the piece that de-risks all of the above and is the oracle
every future GPU stage gates against: **a CPU f64 LBVH reference solver**. This
is the exact relationship `FlatTree` (CPU f64 walk) already has to `GpuTree` —
we mirror it one level up, for the *build* rather than only the *traverse*.

**Outcome of this step:** a new `galaxy_solvers::Lbvh` `ForceSolver` — a
Barnes-Hut force solver built on a Morton-code Linear BVH (Karras 2012 binary
radix tree) instead of the octree — validated against the `DirectSum` oracle.
It introduces no GPU code. It is the algorithmic + numerical reference the
future GPU Morton/sort/tree/aggregate kernels and `GpuLbvh` traversal port to,
stage by stage.

**Explicitly deferred to future steps (documented, not built here):**
- Morton-code + bbox GPU kernel (gated vs this CPU stage)
- GPU sort (hand-rolled deterministic u32 radix **or** `wgpu_sort` dep — decided
  then, with a same-device determinism gate; the sort is the load-bearing risk)
- Karras tree-build GPU kernel + bottom-up aggregation via atomic-*flag* (not
  float `atomicAdd` → deterministic result)
- `GpuLbvh` solver with the f32 binary-BVH traversal kernel
- 63-bit Morton (2× u32 sort passes) as a resolution refinement over 30-bit
- (Separate, larger, NOT this milestone) GPU-resident particle *state* across
  steps — DESIGN already treats the per-step upload/readback as its own future
  bottleneck; it fights the `accelerations(&State)->acc` interface.

## Approach

New module `solvers/src/lbvh.rs`, exported from `solvers/src/lib.rs`. Pure f64,
no I/O, `ForceSolver` drop-in — same `(g, softening, theta)` semantics and same
Plummer-softened kernel as `BarnesHut`, so it is directly comparable.

### Pipeline (all CPU f64, deterministic)

1. **Bounding box** over `pos` — reuse the exact convention in
   `barnes_hut::Octree::build_serial` (cube center, `half` with the `*(1.0+1e-9)`
   pad and `.max(1e-12)` floor) so quantization never lands a particle on the
   upper edge.
2. **Morton codes** — quantize each position to `[0, 1024)` per axis (**30-bit**,
   `u32`), bit-interleave (`expand10`/`part1by2`) into one code. 63-bit deferred.
3. **Sort** by key `(morton, original_index)` — stable, ties broken by original
   index. This deterministic tie-break is what a future GPU sort must also do
   (advisor); on CPU it is `sort_by_key` on the pair.
4. **Karras binary radix tree** (Karras 2012, *Maximizing Parallelism in the
   Construction of BVHs*): `N` leaves (sorted order) + `N-1` internal nodes.
   Parent/child ranges from the longest-common-prefix `δ(i,j)`; **equal Morton
   codes fall back to comparing the sorted indices** (the standard Karras
   duplicate-key handling — also makes topology deterministic). Built directly
   as flat index arrays (per internal node: `left`, `right`; per node: `parent`).
5. **Bottom-up aggregation** — fold each internal node from its two children in
   fixed `(left, right)` order into `(mass, com, aabb_min, aabb_max)`; leaf =
   its single body (`LEAF_CAP` analogue = 1, matching the octree). Fixed child
   order → deterministic sums. Derive per node: `center = 0.5*(min+max)`,
   `s = (max-min).max_element()` (longest AABB side), `delta = |com - center|`
   (Barnes 1994 correction).
6. **Linearize to a skip-pointer DFS array** (`LbvhFlat`, analogous to
   `FlatTree`): DFS pre-order, each node carrying `next` = one-past-subtree, so
   the future GPU kernel is a direct mirror of the existing stackless walk. A
   node's first child is `self+1`; `next > self` always → the walk strictly
   increases and provably terminates.
7. **Stackless BVH traversal** (f64): open a node → `self+1`; accept a monopole
   when `!inside && θ·(d − delta) ≥ s` (same Barnes form as `FlatTree::accel`);
   leaf → direct term excluding self; empty (`mass ≤ 0`) → `next`. Returns accel
   needing `× g`, matching the crate convention.

Softened potential: delegate to the shared
`galaxy_solvers::potential::potential_energy_parallel` (same as `BarnesHut`).

## Files

- **`solvers/src/lbvh.rs`** (new) — `Lbvh` solver + internal Morton/Karras/flatten.
- **`solvers/src/lib.rs`** — add `pub mod lbvh;` and re-export `Lbvh` (+ any public
  `LbvhFlat`/node type, mirroring the `FlatTree`/`FlatNode` re-exports).
- **`solvers/tests/lbvh.rs`** (new) — integration gates (below).
- **`DESIGN.md`** — add an **M4b** bullet under M4 documenting this CPU LBVH
  reference as the oracle, and list the GPU-port stages above as the remaining
  deferred M4 work (update the M4a "next deferred step" prose to point here).

## TDD gates (red-first, committed separately as `test(...): ... [red]`)

Physics gates are **topology-independent** where noted, so they hold for *any*
correct BVH over all particles — that is what makes θ→0 the clean correctness
gate. All are CPU, always-on (no GPU adapter needed), f64 tolerances (not f32).

- **θ→0 = DirectSum** — full open ⇒ exact direct summation; RMS + worst-case rel
  err to f64 roundoff scale (≈1e-12–1e-10), across seeded clustered clouds. A
  dropped/double-counted subtree blows this up. *(topology-independent)*
- **finite-θ bounded & grows** — RMS error at θ=0.3 < θ=0.6, both bounded
  (O(θ²) truncation), matching the `BarnesHut` accuracy-trade gate shape.
- **momentum flux** — Σ mᵢaᵢ = 0 at θ→0 to f64 floor (Newton's third law;
  catches self-term or `dx`-sign bugs). *(topology-independent)*
- **structural** — exactly `2N-1` nodes; every original particle appears in
  exactly one leaf (reachable-set is a permutation of `0..N`); every internal
  node has 2 children; child AABB ⊆ parent AABB; root AABB contains all
  positions; flat `next` strictly increases along the walk (termination).
- **Morton correctness** — `expand10`/interleave round-trips on hand values;
  the code is monotone along a hand-checked axis-ordered set.
- **determinism** — same input ⇒ bit-identical forces (asserts the index
  tie-break makes duplicate-Morton topology deterministic).
- **empty / single** — N=0 yields no accel; a lone particle feels zero force
  (its only leaf holds just itself, excluded as self).

Reuse the existing test scaffolding: the LCG `cluster(seed, n)` generator and the
`rms_rel_err` / `worst_rel_err` / `rms_accel` helpers already in
`solvers/tests/barnes_hut.rs` and `gpu/tests/gpu_tree.rs` (copy the helper block,
as those test files already do).

## Verification

1. `cargo test -p galaxy-solvers` — confirm the red gates **fail** first (commit
   red), then pass after implementing (commit green). Never edit a test to pass.
2. `cargo clippy --all-targets -- -D warnings` and `cargo fmt --check`.
3. `cargo test` (workspace) — nothing else regresses.
4. Sanity cross-check: at a common θ (e.g. 0.5), `Lbvh` and `BarnesHut` forces
   agree to O(θ²) RMS on the same cluster (both are monopole Barnes-Hut, different
   tree shapes) — a loose informal check, not a bit gate (binary node ≠ octree
   cell, so no same-topology assertion; this is why the GPU "vs CPU BH same-θ"
   gate is dropped for the LBVH path).

## Commit sequence

1. `test(solvers): red gates for CPU LBVH reference solver [red]` (+ API stubs
   with `todo!()` bodies so tests compile and fail).
2. `feat(solvers): CPU Morton/Karras LBVH reference ForceSolver (Lbvh)`.
3. `docs(design): land CPU LBVH reference (M4b); stage remaining GPU-resident build`.

Then push (per memory: push to origin after each batch).
