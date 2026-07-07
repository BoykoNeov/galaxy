# Long-horizon roadmap — M7 and beyond

Decided 2026-07-03. This is the strategic ordering document: it records what
comes after each milestone, which tracks are independent, and which future
options conflict with decisions already landed. Per-session detail stays in
the per-series plan docs (M7: `deep-orbiting-sunbeam.md`); architecture
rationale stays in DESIGN.md. This doc only orders the work.

Governing sequence (user-locked):

1. **Finish M7, view-first reorder** — visuals sooner; GPU SPH pulled in only
   when a measured runtime justifies it.
2. **Visual improvements track** — the compatible/orthogonal render work.
3. **Physics track** — energy equation chain, scale path, cosmology.

---

## Phase 1 — M7 completion, reordered view-first

M7a is landed. The remaining sessions run in this order:

| Order | Session | One-liner | Depends on |
|---|---|---|---|
| 1 | **M7d** | Renderprep voxelization + frame-data v2 | M7a (landed) |
| 2 | **M7e** | Volumetric raymarch + full star attenuation | M7d |
| 3 | **M7b** | SPH forces + `GravitySph` + CFL sentinel + shock tube | M7a |
| 4 | **M7c** | Gas-disk IC + evolve-and-stay-put + first gas merger | M7b |
| 5 | **M7f** | scenario knobs, `gasrich` preset, tuning, showpiece | M7c + M7e |

**Why the reorder is safe:** every M7d/M7e correctness gate is analytic or
synthetic — single-particle kernel exactness, uniform-slab transmittance,
two-star depth ordering, gas-off ≡ M6g golden, GPU march ≡ CPU mirror. None
needs gas *dynamics* to exist. The plan's own dependency table already said
M7d depends only on M7a.

**Demo adjustments under the reorder:**
- M7d/M7e demo on **static synthetic gas** — a hand-rolled sech² disk
  distribution (positions only) or a retained stellar snapshot re-tagged
  `Species::Gas`, voxelized and rendered as an inclined dust-lane disk over
  the stellar splats. ~80% of the look with zero SPH force code.
- The "money demo" (dynamically shocked merger dust lanes) moves to **M7c**,
  which by then renders through the already-landed volumetric path.
- Accepted cost: κ/emissivity knobs eyeballed on static gas get re-tuned on
  real merger gas — M7f budgets that tuning pass anyway.
- The owed **M7a demo** (density-coloring side-by-side + timing) slots
  naturally into the M7d session, which touches the same density machinery.

### GPU SPH — decision gate, not a scheduled session

GPU SPH does **not** run before M7b under any reordering: the project's
oracle-first GPU discipline (the LBVH lineage) requires a validated CPU
reference to gate against, and CPU SPH forces don't exist until M7b. M7a's
fixed gather order / parallel≡serial bit-exactness were designed so the GPU
port slots in later without rework — deferral accrues no interest.

**The gate:** at M7c, measure the full-res wall-clock (2·10⁵ gas, ~1500
steps) for a complete merger sim.
- **> ~30 min and actually painful** → insert the GPU SPH session right after
  M7c (before or after M7f, whichever hurts less), gated bit-for-bit /
  tolerance-gated against the CPU stack under the evolve-and-stay-put and
  shock-tube gates.
- **Otherwise** → GPU SPH is the M8-era opener on the physics track, on the
  standing "perf pass only if runtime actually hurts" posture.

## Phase 2 — visual improvements (after M7f)

Ordered by value-per-effort; all compatible with the landed M7 contracts.

1. **Starlight scattering on gas** — DONE (plan `scattered-starlit-veil`):
   unshadowed single scatter from ≤ 8³ clustered stellar point lights,
   HG phase, `[look.gas] scattering`/`anisotropy` knobs, gasrich ships it
   ON (σ=800, g=0.5 from the QUICK A/B bracket).
   `scattering = 0` is gated bit-identical to the pre-scatter render,
   so the "judge the plain M7e look" decision stays open as a one-knob
   toggle. Per-light shadow volumes — DONE (plan `umbral-lantern-lattice`):
   baked 32³ light→sample transmittance per light (K×voxels compute
   prepass), `[look.gas] shadows` bool, gasrich ships it ON; off is gated
   bit-identical to unshadowed v1. Octree clustering + scatter tint — DONE
   (plan `tinted-octree-lanterns`): adaptive octree cut replaces the 8³
   binning (`REFINE_TOL` frozen at `1e-2`, K ≈ 18–24 lights/frame), and
   `[look.gas] scatter_tint` tints the scattered radiance (gasrich ships
   `[0.6, 0.8, 1.3]`); neutral tint is bit-identical. Remaining named
   deferral: DDA/hierarchical bake.
2. **Camera / cinematics** — more rigs, orbit paths, presets. Orthogonal to
   everything; cheap filler between larger sessions.
3. **HDR video encode** — long-standing deferral; EXR chain is ready for it.
4. **Blender gas consumer** — Cycles volumetrics reading the frame-data v2
   gas block. DESIGN always named Cycles best-in-class here; this is the
   contract's second consumer and a good format-stability test.
5. **Half-res gas march + upsample** — perf-gated only; do not build ahead
   of measured raymarch pain.
6. **Temperature-dependent gas color** — *blocked on the energy equation*
   (Phase 3); listed here so nobody attempts it against isothermal gas,
   where it is meaningless.

Standing rejection that shapes all of these: **no camera-dependent quantity
at prep time** (the D9/Doppler-hue rule). Any visual feature must compute
view-dependent terms at render time or it re-couples Contract 3 to a view
axis.

## Phase 3 — physics track

Two independent chains; interleave with Phase 2 as appetite dictates.

**Chain A — gas physics depth** (strictly ordered):
1. **GPU SPH** (if the Phase-1 gate didn't already pull it in).
2. **Energy equation** — adiabatic EOS, entropy formulation
   (Springel–Hernquist). Re-enables total energy as a conservation gate
   (D4 disabled it for isothermal). Unblocks temperature-dependent color.
3. **Radiative cooling/heating.**
4. **Star formation + feedback** — gas→collisionless conversion via the
   `Species` column; gas progenitor tags 4/5 were reserved for this.
   Retires the M6e density→blue proxy and lifts the Q_gas ≥ 1 fail-loud
   gate (fragmentation becomes physical).
5. Accuracy riders, any time after M7b: grad-h terms, Balsara switch,
   individual/adaptive timesteps (the named follow-up if the CFL sentinel
   forces a painful global dt).

**Chain B — scale + cosmology** (independent of Chain A):
1. **TreePM / ParticleMesh solvers** behind `ForceSolver` —
   `GravitySph<G>` is generic over G, so gas rides along untouched.
2. **Cross-step GPU state residency** for the tree solvers; the 10⁷–10⁸
   crossover measurement sets the BH knee from data.
3. **Comoving integration** — Friedmann a(t), Hubble drag (the isolated
   integrator lift DESIGN sketched).
4. **Periodic boundary conditions** — required by PM; the one place Chain B
   collides with current infrastructure (hash grid, Barnes-Hut, shock-tube
   harness all assume open boundaries). Budget it as real work, not a flag.
5. **Cosmological ICs** — Zel'dovich pancake gate, then Santa Barbara
   cluster (the eventual Chain A + Chain B meet-point).

Infrastructure deferrals, slot anywhere: checkpoint/restart in `sim`, HDF5
behind the `validation` feature.

## Standing constraints (do not violate; renegotiate explicitly)

- **h/ρ never stored in `State`** — `accelerations(&State)` is immutable;
  stored per-particle solver outputs go stale (D2's hard argument).
- **Voxel payload stays ρ-only** — emission/opacity are render uniforms so
  the look iterates at re-render cost (D8).
- **No prep-time camera dependence** — subframe cameras exist only at render
  time (D9; same rule that deferred Doppler hue).
- **Gas compositing is ordered, never additive-splat** — absorption does not
  commute (the CLAUDE.md gotcha).
- **Total energy is not a gate for isothermal runs** — momentum + shock-tube
  oracle until the energy equation lands (D4).
- **10⁸ gas ≠ this SPH** — CPU SPH is explicitly not the 10⁸ path; scale
  comes from GPU SPH or a different hydro method.

## The one strategic fork

**SPH vs. moving-mesh / AMR hydro.** Everything above is additive; this is
the only genuine either/or. The particle-only SoA `State`, snapshot format,
and kernel-deposition renderprep all assume Lagrangian particles — switching
hydro method later means rewriting the gas half. SPH's known weaknesses
(contact discontinuities, subsonic turbulence) are acceptable for merger
dust-lane visuals, which is the project's aim; the fork is recorded so the
choice is conscious, and it is **decided: SPH** for this project's scope.
