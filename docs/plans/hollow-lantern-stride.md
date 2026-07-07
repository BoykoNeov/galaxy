# Hollow Lantern Stride — DDA/hierarchical shadow bake

The named deferral of `umbral-lantern-lattice`: accelerate the per-light
shadow-volume bake without changing a single output bit. `ShadowBake::Dda`
strides past the empty space between the light and the gas instead of marching
every step of it — the brute bake stays the reference oracle.

## The bit-exact insight

The density the bake integrates is [`sample_mix`] over two deposited grids:
trilinear inside particle support, **exactly `0.0`** outside it. A shadow
chord's optical depth is `τ = Σ κ·ρ(sᵢ)·ds` summed left-to-right over the fixed
sample lattice `sᵢ = t0 + (i+½)·ds`. Where `ρ(sᵢ) = 0`, the addend is
`κ·0·ds = 0.0` — and adding `0.0` to a finite float is a **no-op**. So a bake
that evaluates the *same* nonzero samples in the *same* order and merely skips
the zero ones produces a bit-identical `τ`, hence a bit-identical `exp(−τ)`.

Every `ShadowVolumes` value is `exp(−τ)` with `τ` finite ≥ 0 — no NaN, no
signed zero — so `f32 ==` is exact and the whole correctness story collapses to
one gate: `assert_eq!(bake_shadows(gas), bake_shadows_dda(gas))`.

## The occupancy pyramid

A conservative acceleration structure: a boolean lattice over the union AABB
where a cell is active wherever the density *could* be nonzero. Built by
splatting each nonzero density cell's **influence region** — its center ± one
cell edge, the exact reach of the trilinear stencil (a point in geometric cell
`i` reads centers `{i−1, i, i+1}`) — into every occupancy cell it overlaps, the
two endpoint grids unioned (mix-aware: a grid weighted exactly 0 is dropped).

- **The ±1 dilation is the whole correctness argument.** It guarantees an
  inactive cell truly has all eight stencil texels zero everywhere inside it, so
  a skipped sample is provably zero. Over-inclusion (a cell active where the
  density is actually zero) is harmless — the march evaluates it, gets 0, adds 0.
  Under-inclusion would drop a nonzero sample → darkened rims, not a crash: the
  pointed single-occupied-cell gate targets exactly this off-by-one.
- **Hierarchical.** Max-pool the base (`SHADOW_OCC_RES = 64`) up to
  `SHADOW_OCC_COARSEST = 4` (64 → 32 → 16 → 8 → 4). The march descends
  coarsest → finest per sample: the first inactive level jumps across its
  as-large-as-possible empty cell; reaching the finest level active is a real
  occupied sample. An inactive coarse cell is still an OR-pool of the base, so
  it still implies exactly-zero density — the mip is **pure perf, no new gate**.
- **The DDA jump.** On an inactive cell, jump to the first sample at/after the
  cell exit (Amanatides–Woo nearest face), with one sample of float-ε margin
  (the boundary sample is re-classified, not skipped) and a `+1` progress guard.
  Skipped samples are provably inside the inactive cell.

## GPU mirror

The GPU consumes the **CPU-packed** pyramid (`pack_shadow_occupancy`,
coarsest-first u32s) rather than rebuilding occupancy in WGSL — one occupancy
implementation to keep correct. It stays conservative against the GPU's own
density because the uploaded 3-D texture holds the identical `GasGrid` data:
an inactive cell ⇒ all eight `sample_one` texels zero ⇒ exactly `0.0` ⇒ no-op.
`cs_shadow_bake_dda` mirrors `cs_shadow_bake` with **byte-identical** `n`/`ds`/`s`
and τ accumulation, so evaluated samples match bit-for-bit — and skipping is
only removal of `+0.0` from a left-to-right fold, so there is **no
reassociation exposure** (unlike the M4j two-sum trap).

## Milestones

- **D1 [red]** (390886a): `ShadowBake{Brute,Dda}` + `bake_shadows_with`, Dda
  arm `todo!()`; equivalence gates (dilation, two-grid mix, dense, empty).
- **D2 [green]** (6956eec): CPU DDA bake, single-level occupancy; 6 gates +
  a 24-case randomized `brute==dda` invariant green.
- **D3** (c6fdf90): hierarchical mip — pure perf, gates unchanged. Measured
  (128³, 64 lights): 3% fill 2.1×→8.3×, 15% fill 1.8×→3.0×.
- **D4 [red]** (28f7a7f) / **D5 [green]** (58c1bed): GPU WGSL mirror.
  `RenderConfig.shadow_bake`, occupancy upload, `shadow_dda_pipeline`. Gates:
  GPU-Dda ≡ GPU-brute **bit-identical** pixels (the exact gate held — no driver
  FMA divergence between the two modules) + GPU-Dda ≡ CPU brute reference within
  the `1e-3 rel + 1e-5 abs` GPU tolerance.
- **D6** (f8eb899, this): `[look.gas].shadow_bake = "brute"|"dda"` knob
  (dead-knob gated: requires `shadows = true`) → `RenderConfig`; docs, memory.

## Perf

Measured net speedup (bit-identical throughout): 2× at gasrich-QUICK scale
(64³ thin disk, 8–20 lights, CPU bake incl. occupancy build), 2–8× at 128³
by sparsity. The empty-space walk was the bottleneck for the sparse frames the
galaxy actually produces (compact gas + large empty margin + shadow rays from
distant clustered lights); the mip removes it. Default stays `brute` (the
reference); `dda` is opt-in, zero visual risk.

## Deferrals (named)

- Building the occupancy pyramid on-device (CPU build + ~1.2 MB upload per frame
  is negligible against a multi-second FULL bake; the CPU build reuses the one
  tested implementation).
- Bit-packing the occupancy (one u32/cell is ~1.2 MB, well under limits).
- Tuning `SHADOW_OCC_RES` / `SHADOW_OCC_COARSEST` (bit-exactness is independent
  of both; they only trade build cost against skip precision).
