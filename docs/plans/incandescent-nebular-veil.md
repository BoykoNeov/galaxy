# Temperature-dependent gas color — `incandescent-nebular-veil`

Phase-2 visual session, unblocked by the energy equation (`smoldering-thermal-ledger`,
E1–E4). Gas is currently rendered with a single flat tint (`GasLook.color`); this
session drives the gas color from the **local temperature** (`T ∝ u`, the evolved
per-particle internal energy) so shocked/compressed merger gas glows hotter than
the cold outskirts.

Advisor-vetted 2026-07-13. Two scope calls made by the user:
1. **New adiabatic preset** (`gasrich-adiabatic`) for the showpiece — the existing
   isothermal `gasrich` stays untouched.
2. **Frame-data v3 (serialized) is DEFERRED** — the movie/showpiece path passes
   `GasGrid` in memory (`xtask/main.rs` never calls `frame::to_writer`/`from_reader`;
   confirmed only `renderprep` tests exercise the on-disk format). The serialized
   temperature channel is a separate later milestone for the Blender/decoupled
   consumer.

---

## The one design decision — a temperature voxel channel

Temperature color needs a **per-voxel temperature field**, which is a genuine
physics output (like ρ), not a look choice. So the D8 "voxel payload stays ρ-only"
rule is renegotiated — but cleanly:

> **D8 (renegotiated):** the voxel payload carries physical scalar *fields* (ρ, and
> now the internal-energy moment `N = Σ mⱼ uⱼ W`); the *mapping* of those fields to
> color / emission / opacity stays render-side uniforms. A look change (colormap,
> reference temperature, ramp endpoints) still iterates at re-render cost, never
> re-prep. The temperature field is a physics output, not a look knob.

**Deposit the moment `N = Σ mⱼ uⱼ W`, not a pre-divided `ū`** (advisor). The
smoothed intensive temperature is `ū = N / ρ`, computed *in the shader* against the
existing ρ grid. Why the moment and not `ū`:
- One new channel, reuses the ρ grid (not two channels).
- Correct under the subframe `mix`: mix `N₀,N₁` and mix `ρ₀,ρ₁` independently, then
  divide — mass-weighting stays consistent across the endpoint blend.
- The `0/0` in empty cells is harmless: emission is `∝ ρ`, so a garbage color there
  is multiplied by ~0. Guard with `ū = N / max(ρ, ε)` to avoid NaN.
- Bit-exact `parallel ≡ serial` holds for `N` by the identical `+0.0`
  scatter-by-plane argument as ρ (the terms a global gather would add beyond a
  particle's support are exact `+0.0`s).

**Colormap normalization uses a FIXED reference** (advisor). A per-frame data-derived
max would make the color *flicker* frame-to-frame as peak temperature drifts. The
ramp domain `[u_lo, u_hi]` (or a soft `u_ref`) is a **temporally-constant render
uniform**. MVP: two-stop `cold → hot` linear lerp with clamped
`t = (ū − u_lo) / (u_hi − u_lo)` — mirrors `coloring::dispersion_colors` exactly.

**MVP is tint-only** (advisor): drive only `color` from the ramp; keep emissivity
`j` and opacity `κ` uniform. Hotter-gas-brighter (`j = j(T)`) is a later refinement.

---

## Scope

**IN (this session):**
- `renderprep/gasgrid.rs`: `N = Σ mⱼuⱼW` voxel channel, deposited from `state.u`.
- `render/volume.rs`: CPU raymarch samples `N`, computes `ū = N/max(ρ,ε)`, maps to
  color via a fixed-reference cold→hot ramp; feature-off ≡ current render bit-identical.
- `render/render.rs`: WGSL parity — `temp0/temp1` 3D textures + colormap uniforms,
  op-for-op mirror; GPU ≡ CPU gate.
- `xtask/spec.rs`: `[look.gas]` temperature knobs + validation + wiring.
- New `gasrich-adiabatic` preset (Eos::Adiabatic) as the integration showpiece.

**DEFERRED (named, not built here):**
- Frame-data **v3 serialization** (temperature channel on disk) — for the
  Blender/decoupled consumer; the movie path passes `GasGrid` in memory.
- `j = j(T)` emissivity/opacity coupling to temperature.
- **GPU adiabatic sim** (already deferred upstream by the energy-eq series) — so a
  full-res adiabatic showpiece is a **CPU adaptive** run (~1hr-class, like A5).
- Blackbody / multi-stop physical color curves (MVP is a two-stop ramp).

---

## Milestones (TDD: red tests committed separately, then green)

### H1 — Temperature voxel channel (`renderprep/gasgrid.rs`)
`GasGrid` grows a second channel (`umoment: Vec<f32>`, same dims/layout as `data`);
`deposit_gas`/`deposit_impl` accumulate `N += mⱼ·uⱼ·W(d, hⱼ)` in lockstep with the ρ
accumulation, reading `state.u`. `GasGrid::sample` gains a moment-sampler (or a
generic per-channel sample).
- **Gates:** single gas particle at a cell center ⇒ `N = m·u·W` exactly;
  `parallel ≡ serial` bit-exact for `N` (same `+0.0` argument as ρ); mass/u
  linearity; a uniform-`u` field recovers `ū = N/ρ = u` (flat) — the isothermal
  sanity; empty cells `N = 0`; collisionless rows ignored (filtered before deposit).

### H2 — Colormap + CPU raymarch (`render/volume.rs`)
`GasLook` gains `temp_color: Option<TempColorLook>` where
`TempColorLook { cold: [f32;3], hot: [f32;3], u_lo: f32, u_hi: f32 }`. `None` = the
current flat `look.color` path, **bit-identical**. `march_gas` samples `N` (and `ρ`),
computes `ū = N/max(ρ,ε)`, `t = clamp((ū−u_lo)/(u_hi−u_lo), 0, 1)`,
`color = lerp3(cold, hot, t)` — replacing `gas.look.color`. Subframe: mix `N`, mix
`ρ`, then divide.
- **Gates:** `temp_color = None` ≡ current render bit-identical; synthetic hot-core
  grid → hand-oracle color at known cells; flat `ū` ⇒ flat color = single-tint
  equivalent; ramp endpoints `t=0→cold`, `t=1→hot` exact via `lerp3`; empty-cell
  guard yields no NaN.

### H3 — GPU parity (`render/render.rs` WGSL)
Add `temp0`/`temp1` `texture_3d<f32>` bindings + colormap fields to `GasUniforms`;
WGSL mirrors `march_gas` op-for-op (sample `N`, divide by existing `rho`, map, lerp).
Feature-off path takes `look.color` so it stays bit-identical to the existing golden.
- **Gates:** GPU ≡ CPU both projections (1e-3 rel + 1e-5 abs, the existing volume
  tolerance) on a synthetic hot-core grid + mix 0.37; `temp_color` off ≡ existing GPU
  render bit-identical.

### H4 — `[look.gas]` knobs + wiring (`xtask/spec.rs`, `main.rs`)
`GasLookSpec` gains a `[look.gas.temperature]` sub-table (cold/hot RGB, `u_lo`,
`u_hi`) or `temp_color` fields; `#[serde(deny_unknown_fields)]` preserved. Validation:
finite, non-negative, `u_lo < u_hi`. `GasLookValues` + `Default` (temp off). Threaded
into the `GasLook` built at `main.rs:~1773`.
- **Gates:** parse/validate (unknown-field reject; `u_lo < u_hi` enforced; finite);
  default spec ⇒ `temp_color = None`; an on-spec ⇒ correct `GasLook`.

### H5 — `gasrich-adiabatic` showpiece preset + integration
New preset: `Eos::Adiabatic { gamma: 1.4 }`, initial `u` picked so `c_s0 =
√(γ(γ−1)u)` sits in the warm/CFL-stable band (comparable to gasrich's `c_s=0.1`), on
the **adaptive-dt** path (A-series). `[look.gas].temperature` on with a cold-blue →
hot-white/red default ramp. QUICK run must complete CFL-clean and show *non-flat* gas
color (shocked pericenter gas hotter than outskirts).
- **Gates:** preset validates + QUICK completes CFL-clean (adaptive); QUICK render
  produces temperature-varying gas color (A/B smoke against a flat-tint control — not
  a golden). FULL ~1hr showpiece is a user-gated `--release --ignored` run.

---

## Standing-constraint deltas (update `long-burning-beacon.md` on completion)
- **D8 renegotiated** to "physical scalar fields (ρ, N); mapping stays render
  uniforms" (text above). Look changes still re-render, never re-prep.
- **Frame-data stays v2** on disk; the serialized temperature channel is deferred.
- **D2 unviolated:** `N`/`ū` are re-derived per snapshot at prep time, never stored
  in `State` (h/ρ/u_moment all derived-never-stored on the render side).
- Temperature color is **EOS-conditional in practice**: meaningful only on adiabatic
  runs; on isothermal gas it renders a (correct) flat tint.

## Process
Standard project discipline: tests before impl; `todo!()` API surface so tests
compile (red commit must `cargo build --workspace`); red committed separately
(`test(...): … [red]`); `./gate.ps1` before each commit; commit + push at end of
batch (origin: github.com/BoykoNeov/galaxy).
