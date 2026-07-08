# Stellar-nursery glow — color stars by ambient gas density

Scoping doc, written 2026-07-08. A visual session: tint star splats toward a
young-population blue-white where they are embedded in dense SPH gas, so the
merger's dense lanes and shocked bridges — where stars really form — light up
with new-star color. Value-per-effort visual on the Phase-2 track
(`long-burning-beacon.md`).

> **SEQUENCING: `laddered-ember-cadence.md` (individual timesteps) executes
> FIRST.** This plan is queued after it (with a tuning pass on that work if the
> per-particle-rung run shows something off). Nothing here depends on the
> timestep work — the two are orthogonal (one is the SPH stepping loop, one is
> renderprep-time coloring) — but the agreed order is: perf first, this second.

---

## What it is — and the honesty it fixes

`renderprep/src/coloring.rs` already has `compression_colors`, an M6e
"star-formation proxy" that tints toward young-blue by density. But read its
caveat (`coloring.rs:17-19`): *"the sim is collisionless — stellar density
stands in for gas compression."* At M6e there was **no gas**, so it faked it
with stellar density.

M7 gave the project **real SPH gas**. This map replaces the fake: sample the
*actual gas density field* at each star's position and tint by it. Denser
ambient gas → bluer/younger tint. It reads as "stars glowing where the gas is
dense enough to breed them," and it is honest about what it shows — a better-
grounded proxy than the M6e stellar-density stand-in.

**It is still a proxy, not physics** — see "Relationship to star formation."

---

## Key decisions (with rationale)

### G1 — sample gas ρ at star positions by TRILINEAR from the existing voxel grid
The gas density field is already computed and voxelized every snapshot for the
volumetric raymarch (`renderprep/src/gasgrid.rs`, `GasGrid`: ρ at cell centers,
x-fastest, world-space bounds — trilinear-samplable as-is). Sampling a star's
ambient ρ = one trilinear fetch from that grid at the star's world position.
- **Chosen:** trilinear from `GasGrid`. Cheap; lives at renderprep time next to
  the other `coloring.rs` maps; render-res-coarse but that is *fine for a color
  proxy* (we are picking a hue, not resolving a shock).
- **Rejected for v1:** per-star SPH gather over gas neighbours — accurate but
  expensive and a second neighbour-search path, unjustified for a tint.
- Stars outside the grid bounds (halo, tidal tails) sample ρ = 0 → base color,
  exactly (the degenerate-sentinel convention the existing maps already use).

### G2 — a new pure coloring fn parallel to `compression_colors`, bit-exact off
Add `gas_density_colors(base, gas_rho, young, strength)` beside the existing
maps, using the same two-product lerp (`lerp3`, `coloring.rs:111`) that is
bit-exact at both endpoints. So `strength = 0` (and any star at ρ = 0) is the
identity — **`strength = 0` gated bit-identical to the no-tint render**, the
same hard guarantee `compression_colors`/`initial_radius_colors` carry, not a
tolerance. The ramp is `t = strength · ρ/(ρ + ρ_ref)` (monotone, bounded in
`[0, strength]`, ½ at the reference), with `ρ_ref` the mean over positive
ambient densities — the `dispersion_colors`/`density_boost` reference discipline
(`coloring.rs:132-143`), so the map is scale-free per snapshot.

### G3 — view-independent, so it obeys the standing D9 rule
Ambient gas density is a scalar field, no camera dependence — computed at
renderprep time like every other `coloring.rs` map, never re-coupling the frame
data to a view axis (`long-burning-beacon.md` "no prep-time camera dependence").

### G4 — honest caveat in the doc comment, like the map it replaces
An *old* star drifting through a dense cloud would tint "young," which is not
real. In practice it reads fine (dense gas lives in arms/shocked bridges where
young blue stars belong; old stars mostly are not in dense gas). Label it
honestly in the fn doc, exactly as `compression_colors` labels its proxy nature.

---

## Relationship to star formation (why this is a stepping-stone)

The *physically faithful* version of "denser gas breeds new stars" is Chain A
step 4 (star formation + feedback): gas genuinely converts to star particles
carrying a **formation time**, and you color by true stellar **age**. That is
the endgame. This map is the visual stepping-stone: it delivers ~80% of the
"young stars in the dense lanes" look now, cheaply and bit-exactly gated, with
the render path already built. When SF lands, swap the proxy signal (ambient
gas ρ) for the real one (stellar age since conversion) — the coloring plumbing
is unchanged. Retires nothing yet (the M6e proxy stays as an alt mode); it adds
a better-grounded option.

---

## Milestones (TDD: red test committed separately, then green)

- **N1 — `gas_density_colors` pure fn.** Red: `strength = 0` bit-identical to
  base; ρ = 0 → base exactly; monotone + bounded in ρ; endpoint/reference
  exactness (the `coloring.rs` invariant-test discipline). (G2, G4)
- **N2 — trilinear `GasGrid` sampler.** Red: exact at a cell center, linear
  between two centers, ρ = 0 outside bounds; bit-exact at grid nodes. (G1)
- **N3 — wire into `prepare` + a `[look]` scenario knob.** Red: knob off ⇒
  frame-data byte-identical to the current pipeline; on ⇒ the tint appears.
  (G1, G3)
- **N4 — QUICK A/B showpiece.** Render `gasrich` with the map off vs on at a
  couple of `strength` values; pick the ship value from the A/B (the
  `scatter`/`splat`/`local-tone` A/B discipline). Retain the A/B dir under
  `M:\claud_projects\temp`.

---

## Gates (summary)

| Gate | What it asserts |
|---|---|
| `strength = 0` bit-identical | the map off is the pre-map render, exactly (two-product lerp). |
| Monotone + bounded in ρ | tint never overshoots `[base, young]`; denser ⇒ (weakly) younger. |
| Trilinear exactness | exact at cell centers/nodes, ρ = 0 outside bounds. |
| Knob-off frame-data byte identity | the scenario knob off ⇒ current pipeline unchanged. |
| View independence | no camera term at prep time (D9). |

---

## Relationships

`laddered-ember-cadence.md` (**executes FIRST** — see the sequencing note),
`long-burning-beacon.md` (Phase-2 visual track; Chain A step 4 is the physical
endgame this proxies), DESIGN.md M6e (the `compression_colors` proxy this
supersedes visually), [[render-more-controls]] / [[tinted-octree-lanterns]] (the
A/B + ship-value discipline N4 follows), [[m7-sph-volumetrics-series]] (the
`GasGrid` density field G1 samples).
