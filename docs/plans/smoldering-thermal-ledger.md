# Smoldering thermal ledger — the energy equation (adiabatic SPH)

Per-series plan for **Phase 3 · Chain A · item 2** of `long-burning-beacon.md`:
replace the isothermal EOS with an **adiabatic** one driven by an evolved
per-particle internal energy `u`, re-enabling total-energy conservation as a
gate and unblocking temperature-dependent gas color.

Decided 2026-07-07. Advisor-vetted. Governs the E1–E4 session arc below.

---

## The one decision that overrode the roadmap: `u`, not entropy `A`

`long-burning-beacon.md` named the "entropy formulation (Springel–Hernquist)".
We evolve **internal energy `u`** instead. Rationale (user-confirmed —
"computationally lighter variant, keep the option to approach the other path
later"):

- **Downstream is energy-native.** The very next Chain-A milestones — radiative
  cooling (`du/dt = −Λ(T,ρ)/ρ`), SF feedback (thermal `Δu` injection), and
  temperature-color (`T ∝ u`) — all operate on `u`/`T`. Entropy `A` would force
  an `A↔u` round-trip every substep the moment any source term lands.
- **Entropy's headline guarantee is gated behind deferred work.** Simultaneous
  energy+entropy conservation in the SH scheme needs the grad-h `f_i` terms,
  which are an "accuracy rider" deferred to later in Chain A. So picking `A`
  now would pay its complexity without collecting its benefit.
- **Lighter.** One evolved scalar in the *same* neighbor loop as the force; no
  conversions; temperature falls out directly.

**Reversibility is preserved by design.** The EOS becomes an enum on
`HydroParams` (`Isothermal { c_s } | Adiabatic { gamma }`). Entropy `A` can be
added later as a third variant (`Entropy { gamma }`) without disturbing either
existing mode — the "other path" stays open exactly as requested. The `u`
formulation's only genuine cost is a **positive-`u` floor** (energy-formulation
wart: `u` can go negative under aggressive PdV/viscous heating), a documented,
bounded, clamped non-conservation.

## Why `u` in `State` is NOT a D2 violation

The standing constraint "h/ρ never stored" (D2) is about *derived,
stale-prone* solver outputs: `accelerations(&State)` is immutable, so a stored
derived column silently desyncs from positions. `u` is different in kind: it is
**evolved** — a genuine state variable like `pos`/`vel`, integrated forward by
its own derivative. It *must* be stored; there is nowhere else for it to live.
This is the first evolved gas-only `State` column and the reason for the
snapshot bump. h/ρ remain derived-never-stored, untouched.

## What the EOS touches — three sites, not one

Per-particle sound speed `c_s,i = sqrt(γ(γ−1) u_i)` and pressure
`P_i = (γ−1) ρ_i u_i` replace the constant `c_s`/`P = c_s²ρ` in **three**
places:

1. **`forces.rs`** — `term_i = P_i/ρ_i²`; viscosity signal speed uses
   `½(c_s,i + c_s,j)` (was constant `c_s`).
2. **`cfl.rs`** — `v_sig,i = max_j (c_s,i + c_s,j − 3 w_ij)`, floored at
   `c_s,i + c_s,j`-scale (generalizes the current `2·c_s`).
3. **`diagnostics.rs`** — `U_thermal = Σ m_i u_i`;
   `total_energy = T + U_grav + U_thermal`.

Isothermal stays a **live mode**: adiabatic is purely additive, so every
existing gate (isothermal Riemann shock tube, gasrich showpiece, all GPU
gates) stays green untouched. **gasrich is NOT force-flipped.** Adiabatic gets
its **own new γ=1.4 Sod oracle**.

## Integrating `du/dt` in the KDK leapfrog

`du/dt = (P_i/ρ_i²) Σ_j m_j v_ij·∇_i W̄_ij + (viscous heating)` shares the hydro
neighbor loop → computed in **one fused pass** with the acceleration.

Trait surface (least-invasive, preserves gas-free byte-identity):

```rust
// default delegates to accelerations + zero dudt; pure-gravity / GPU
// implementers are unchanged, so their existing byte-identity holds.
fn accel_and_dudt(&mut self, state: &State, acc: &mut [DVec3], dudt: &mut [f64]) {
    self.accelerations(state, acc);
    dudt.fill(0.0);
}
```

`GravitySph` overrides it (single fused pass). A **thermal-aware integrator
branch** (or a distinct `LeapfrogKdkThermal`) calls `accel_and_dudt` and kicks
`u` alongside `v`; the existing `LeapfrogKdk` on pure gravity keeps calling
`accelerations` untouched → existing energy-oscillation / byte-identity gates
hold, same discipline the adaptive-dt series used.

**Ordering (verify by test, not assertion):** pressure now depends on `u`, so
the force evaluation must see a `u` consistent with `x`. `u` is half-kicked
around the drift, interleaved with the `v` half-kicks; `du/dt` is evaluated at
the same `v_{n+1/2}` the force uses — a first-order timing term, exactly the
tolerance already accepted for the viscosity term (the note in
`gravity_sph.rs`). No higher-order `u` scheme.

## The energy gate tolerance (why it is NOT 1e-6)

Without the deferred grad-h `f_i` terms, the discretized total energy has an
`O(∇h)` error. A naive "conserved to 1e-6" gate *will* fail for the right
reason. Per CLAUDE.md ("tolerances justified by the method's order"), the gate
is **oscillation-bounded at the ~%-level**, justified in-doc by deferred
grad-h. It checks *bounded oscillation, not drift* — the leapfrog property.
The isothermal path keeps NO energy gate (heat bath); the gate turns on for
**adiabatic only**. This partially renegotiates D4 (see below).

## GPU: deferred, explicitly

Oracle-first, exactly as G1–G6. The CPU adiabatic path must exist and be gated
(E1–E4) before any on-device `u` evolution. This milestone touches **no GPU
code**. A later gated GPU-adiabatic sub-milestone mirrors the G-series.

---

## Session arc — four TDD slices, each a hard gate

### E1 — EOS enum + `u` column + per-particle pressure/`c_s` + `U_thermal` (`du/dt ≡ 0`, frozen)
No integrator change yet — the most contained slice. Split into two commits:

**E1a — `u` column + snapshot v3 + `U_thermal` — DONE (2026-07-08).**
- `State.u: Vec<f64>` column: into `assert_consistent`, `from_phase_space`
  (all `0.0`); threaded through every `State` literal / IC constructor;
  non-gas rows inert `0.0`.
- Snapshot **v3**: append `u[n]` (f64) after `kind`; reader gates on version and
  defaults `u=0.0` for v1/v2 (exact same version-tolerant pattern v2 used for
  `kind`). Retained pre-energy-equation snapshot zoo still reads.
- `diagnostics::thermal_energy = Σ mᵢuᵢ`, folded into `total_energy` (inert on
  the isothermal path where `u≡0` → every existing energy gate numerically
  unchanged).
- Gates green: hand-derived `U_thermal`, isothermal `u≡0` invariant,
  `FORMAT_VERSION==3`, bit-exact `u` round-trip, v1/v2 forward-compat reads.

**E1b — EOS enum + per-particle pressure/`c_s` — DONE (2026-07-08).** Red
(`1f4dc53`) + green (`57161d0`), gate green (fmt/clippy/full test suite).
Design locked advisor-vetted below; implemented as designed with no
deviations. E2 (PdV work + thermal integrator) next.

*API shape:*
- `Eos::Isothermal { c_s } | Eos::Adiabatic { gamma }`, **replacing** the
  `HydroParams.sound_speed` field. Default `Isothermal { c_s: 1.0 }`.
  `#[derive(Clone, Copy, Debug)]` on `Eos` — **load-bearing**: `HydroParams` is
  `Copy` and ~15 sites rely on it (e.g. `potential_energy_delegates_to_gravity`
  passes `params` by value twice); a non-`Copy` `Eos` silently drops `HydroParams`'s
  `Copy` and breaks those sites with unrelated-looking move errors. `Eos` is
  all-f64 so `Copy` is free.
- Add `HydroParams::sound_speed(&self) -> f64` accessor returning the isothermal
  `c_s` (isothermal-only consumers: GPU src, `cfl.rs`).
- Thread `u: &[f64]` into `hydro_accelerations`/`_serial` **after `h`, before
  `params`** (unavoidable — adiabatic `P=(γ−1)ρu` needs it). `GravitySph`
  gathers gas-subset `u` from `State.u` and passes it (all-`0.0` on isothermal →
  ignored). Ripple is mechanical: every call site adds `&u`; GPU **tests** add
  `&u` but no shader/logic change ("touches no GPU code" holds).
- Per particle: Isothermal → `P_i=c_s²ρ_i`, `c_s,i=c_s`; Adiabatic →
  `P_i=(γ−1)ρ_i u_i`, `c_s,i=√(γ(γ−1)u_i)`. Viscosity `c̄=½(c_s,i+c_s,j)`.

*Scope guards (advisor):*
- **Do NOT unify the isothermal inner loop.** Keep `term_i = cs2/rho[i]` and the
  visc `c̄ = params.sound_speed` expressions **verbatim** inside `match eos {
  Isothermal => … }` so byte-identity is *structural* (zero reassociation risk).
  (A unified `term[]`/`cs[]` path is IEEE-safe too — `½(c_s+c_s)==c_s`, exact —
  but the un-unified branch removes the one place byte-identity could slip.)
- **`cfl.rs`: enum-adapt only** — read `c_s` via the accessor for the isothermal
  `two_cs`, behavior identical. Per-particle adiabatic `v_sig` is **E4**, not here.
- **`du/dt` stays frozen (≡0).** No integrator, no thermal kick — that is E2.
- **Gas-`u` setter: minimal.** All three gates build `State`/arrays directly
  (`State.u` is `pub`); do NOT wire scenario/IC `u` now.

*Gates (write byte-identity FIRST):*
- **Isothermal byte-identity** to pre-E1b: frozen-literal regression test —
  capture the exact f64 bits of the current (pre-change) isothermal force on a
  fixed compact cloud, embed as `f64::from_bits` literals, `assert_eq!`. (The
  existing hand-oracle test is `rel<1e-12`, which is NOT byte-identity.)
- **Adiabatic `P=(γ−1)ρu`** correct on a two-particle hand case (`rel<1e-12`).
- **Static adiabatic blob stays put**: uniform-ρ/uniform-`u` lattice → `term_i`
  uniform → ∇P=0 → interior net accel ≈ 0 (adiabatic twin of
  `uniform_lattice_interior_has_near_zero_net_force`). Single force eval, no
  integration.

*TDD structure:* red commit keeps the **isothermal branch fully working** and
`todo!()`s **only the adiabatic branch** → isothermal + byte-identity tests
pass, adiabatic tests panic (red). Green = implement the adiabatic branch.
Workspace must `cargo build` at red ([[red-commit-must-compile-workspace]]).

### E2 — `du/dt` PdV work + thermal integrator (no shocks)

Split into two sub-milestones (mirrors the E1a/E1b data-layer /
physics-layer split) so each lands as its own red/green TDD cycle:

#### E2a — fused `accel_and_dudt` pass (plumbing; no integrator, no multi-step gate) — DONE (2026-07-08)
Red (`370d196`) + green. Advisor-vetted (see below): "`du/dt` using `term_i`
alone is the *exact* energy-conserving partner of the symmetric momentum
force — not an approximation" (worked the conservation: the residual `dE/dt`
sum is pairwise-antisymmetric under i↔j and vanishes exactly, mod viscosity +
time-integration error). Implemented as designed, no deviations:
- `ForceSolver::accel_and_dudt` default method (`core/src/traits.rs`); default
  impl calls `accelerations` and zero-fills `dudt` (isothermal/gravity-only
  solvers get `du/dt≡0` for free).
- Refactored `hydro_impl` → `hydro_accel_and_dudt_impl`
  (`solvers/src/sph/forces.rs`) to compute accel AND `du_i/dt` in the same
  neighbor loop: `du_i/dt = term_i · Σ_j m_j (v_ij·grad_avg)` — **PdV term
  only**, using `term_i` alone (not `term_i+term_j+visc`); viscous heating is
  E3. `hydro_accelerations`/`_serial` are now thin wrappers over the fused
  function dropping `dudt`, so accel output stays byte-identical (structural,
  not incidental — same pattern as the E1b isothermal-verbatim guard; the
  `isothermal_regression_pins_pre_e1b_bits` frozen-bits gate confirms it).
- New public `hydro_accel_and_dudt` / `hydro_accel_and_dudt_serial`.
- `GravitySph::accel_and_dudt` override: fused hydro over the gas subset,
  `dudt[i]=0` for non-gas rows. `GravitySph::accelerations` itself is
  untouched (separate code path, not routed through the fused function) —
  lower risk than sharing, at the cost of some duplication.
- **Gates (all green):** hand-computed 2-particle `dudt` oracle, isothermal
  AND adiabatic (`dudt_matches_the_hand_oracle_{isothermal,adiabatic}`);
  parallel≡serial bit-exact for `dudt`; fused-path accel bit-identical to
  `hydro_accelerations`; existing isothermal byte-identity regression
  untouched; `GravitySph` routing (gas-only fused-hydro match, non-gas
  `dudt≡0`, and `accel_and_dudt`'s accel bit-identical to `accelerations()`);
  default-impl delegation gate on a toy `ForceSolver` (`core/tests/traits.rs`).

#### E2b — thermal integrator + adiabatic-compression gate (the physics validation) — DONE (2026-07-08)
Red `b7aa609` + green `3ee4fc7`, full gate green (341.7s). `LeapfrogKdkThermal`
kicks `u` alongside `v` at both half-kicks via `accel_and_dudt` (2nd half at
post-drift positions with `v_{n+1/2}`). Homologous-lattice gate
(`solvers/tests/sph_adiabatic_compression.rs`), gravity+viscosity OFF: interior
particles track `u∝s^{-3(γ-1)}`, `ρ∝s⁻³` to L1(u)≤1.3e-3 / L1(ρ)≤4.4e-3;
energy oscillation max 8.3e-4 (bounded, not drift). Tolerances U_TOL 5e-3 /
RHO_TOL 1e-2 / E_TOL 4e-3 calibrated few× above those observed floors. Fast 5³
smoke test in the normal gate; 11³/67-step convergence run is `#[ignore]`
(manual `--release --ignored`). Original advisor deltas (kept for the record):

Advisor-vetted (2026-07-08) before implementation:
- **Gravity MUST be OFF in the compression test** — the plan below states
  viscosity-off but was silent on gravity, and that's the one real gap. If
  `GravitySph` runs gravity, interior particles feel central attraction, the
  ballistic `s(t)=1-kt` homology breaks, and the analytic reference stops
  being code-independent (it'd depend on the N-body force). Use the same
  gravity-off/pure-hydro path (`GravitySph::hydro_only`) the isothermal shock
  tube uses, and say so explicitly in the test.
- **Size the interior exclusion margin by sound-crossing distance, not a
  fixed multiple of `h`** — this resolves the plan's open sizing question
  below. The contaminating signal is the rarefaction wave launched from the
  lattice boundary, propagating inward at `c_s`. Checked interior particles
  need to sit `≥ max(2·h_max, c_s·t_end)` from the boundary, using the
  **peak** `c_s` over the run (since `u`, and so `c_s`, grows under
  compression) — neither "initial `h`" nor "shrunk/final `h`" as originally
  posed; it's the wave-crossing length.
- The `u`-kick timing/ordering (interleaved with the `v` half-kicks, see
  below) is the highest-risk implementation detail, and the total-energy
  oscillation gate below is what actually validates it: a mismatch between
  the force's and `du/dt`'s kernel-averaged `∇W̄`, or a sign/ordering bug,
  shows up as energy drift, not as a failed accel bit-check. Verify the
  scheme empirically (tune the gate tolerance above the O(dt) first-order
  timing floor, tight enough to catch a real bug) — do not assert it.
- Affirmed independently, not shortcuts: `du/dt` using `term_i` alone (E2a) is
  the exact energy-conserving partner of the symmetric force (see E2a note
  above); a separate `LeapfrogKdkThermal` over branching `LeapfrogKdk` is
  right (byte-identity discipline, matches the adaptive-dt series); the
  self-similar reference below (`s(t)=1-kt`, `ρ∝s⁻³`, `u∝s^{-3(γ-1)}`,
  reducing to `PVᵞ=const`) is correct and genuinely code-independent.

Design:
- `LeapfrogKdkThermal` (`core/src/integrator.rs` + `lib.rs` export): mirrors
  `LeapfrogKdk` (KDK, caches `acc`/`dudt` between steps), calls
  `accel_and_dudt`, kicks `state.u` alongside `state.vel` at both half-kicks.
  `LeapfrogKdk` itself stays untouched.
- Homologous-lattice adiabatic-compression test: uniform-ρ/uniform-`u`
  lattice, imposed velocity field `v_i = -k·(pos_i − center)`, **gravity off**
  (see above) and **viscosity off** (`alpha=0, beta=0` — isolates pure PdV
  physics; E2's `du/dt` has no term to absorb viscous dissipation, so leaving
  viscosity on would leak energy the gate can't attribute). Self-similar
  scaling gives a closed-form, code-independent reference: separations scale
  by `s(t)=1-kt`, `ρ(t)=ρ0/s(t)³`, `u(t)=u0·s(t)^(-3(γ-1))` (integrated first
  law; reduces to the standard adiabat `u∝ρ^(γ-1)`, i.e. `PV^γ=const`). Check
  interior particles' `u(t)`/`ρ(t)` track this over several steps. Lattice
  size/step count/compression amount: enough steps to see the adiabat, not so
  much boundary/sound-crossing effects contaminate the checked interior
  particles — margin sized per the sound-crossing rule above.
- Total-energy oscillation-bounded (~%-level, not drift) gate over the run,
  reusing `diagnostics::total_energy` and the `physics.rs` `max_e_err`
  pattern.
- Momentum tripwire (exact pairwise antisymmetry) confirmed unaffected by the
  thermal integrator.
- Dt is fixed/manual for this gate, not CFL-driven —
  `HydroParams::sound_speed()` panics on `Adiabatic` (per-particle CFL is
  E4).

### E3 — viscous/shock heating + adiabatic Sod shock tube

Split into **E3a** (viscous heating term + unit gates) / **E3b** (γ=1.4 Sod
oracle + shock-tube dynamical gate), mirroring the E1/E2 data-layer /
physics-layer rhythm. Advisor-vetted 2026-07-08 (deltas folded in below).

#### E3a — viscous heating term in `du/dt` (code + unit gates)
- Add the Monaghan viscous-heating partner to the fused `du/dt` in `forces.rs`,
  **both** EOS branches:
  `du_i/dt = Σ_j m_j (term_i + ½·Π_ij)(v_ij·∇_i W̄_ij)`.
  The `½` and `+` sign are **load-bearing**: `d(KE)/dt|visc + d(U)/dt|visc = 0`
  **pairwise** (Π_ij symmetric, ∇W̄ antisymmetric → the ½ cancels them
  term-by-term, mod time integration — the same exact-cancellation structure as
  the E2a PdV proof). Provably `≥ 0` (`vr<0 ⇒ Π>0`, `v_ij·∇W̄>0` since
  `dW/dr<0`) → this is the **entropy source**.
- `Π_ij` is the SAME viscosity already computed for the accel `coeff` — reuse it
  (isothermal `c̄=c_s`, adiabatic `c̄=½(c_s,i+c_s,j)`). **Accel path UNTOUCHED**
  → no byte-identity risk; `isothermal_regression_pins_pre_e1b_bits` pins accel
  only. Isothermal `dudt` is dead output (`LeapfrogKdk` drops it) but heating
  goes in both branches for symmetry.
- **The existing E2a `dudt` oracles are APPROACHING** (`dudt_matches_the_hand_
  oracle_{isothermal,adiabatic}`: `v_ij·r_ij = −0.4 < 0`, default α/β on), so
  their PdV-only expectation is now wrong. Update both to the full
  `term_i + ½Π_ij` form in the **red** commit (deliberate spec change,
  advisor-endorsed). The heating addend is identical for `dudt[0]`/`dudt[1]`
  (`Π_ij` and `v_ij·∇W̄` are both swap-invariant), so `term_i` still
  distinguishes them.
- **Gates:** updated 2-particle oracles (iso + adiabatic, `rel<1e-12`); a new
  approaching-vs-receding heating gate (receding ⇒ Π=0 ⇒ `dudt` unchanged;
  approaching ⇒ `dudt` strictly larger, heating `≥0` verified); parallel≡serial
  still bit-exact; accel byte-identity regression untouched.

#### E3b — γ=1.4 Sod oracle + shock-tube dynamical gate
- **Exact adiabatic Riemann solver** (Toro): Newton/bisection on the star
  pressure `p*` from `f_L(p)+f_R(p)+(v_R−v_L)=0` (rarefaction/shock branch per
  side), then sample the 5 regions (L | rarefaction fan | contact | post-shock |
  R) at `ξ=x/t`. **Spot-check vs canonical Sod** (`ρ_L=1,P_L=1 / ρ_R=0.125,
  P_R=0.1`, γ=1.4, `v=0`): `p*≈0.30313`, `v*≈0.92745`, `ρ*_L≈0.42632` (post-
  rarefaction), `ρ*_R≈0.26557` (post-shock) — hand-value gate BEFORE trusting
  any profile failure (isothermal-oracle discipline, lines 104–117). Verify the
  four numbers against a published Sod table, not advisor recall alone.
- **Sod IC:** two glued equal-mass lattice blocks (ρ jump via spacing, as the
  isothermal tube), per-particle `u` set for the pressure jump:
  `u_i = P/((γ−1)ρ)` ⇒ `u_L=2.5`, `u_R=2.0`. Gravity OFF (`hydro_only`),
  **viscosity ON** (defaults) — heating now closes the budget.
- **dt FIXED/unguarded** (no adiabatic CFL until E4 — `sound_speed()` panics on
  Adiabatic, so an unstable dt yields garbage, not a guard trip): pick a
  conservative Courant from `c_L=√(γP_L/ρ_L)≈1.18` and the finer (left)
  spacing; mirror the isothermal `dt≈0.02` start, validate stability
  empirically.
- **Gates** (dynamical `#[ignore]` run + fast smoke twin, like E2b):
  - ρ, v **and P** profiles match the exact solution to a method-order L1 (P via
    `u` is the new independent physics — validating it is the point). A narrow
    contact-blip band (8:1 density ⇒ 2:1 spacing h-mismatch, worse than the
    isothermal 1.59:1) may be excluded from the tight L1 while shock + rarefaction
    stay in-window.
  - **Total energy oscillation-bounded (viscosity ON)** — the SHARP validator of
    the E3a ½Π heating term (a wrong factor/sign drifts energy, not the profile).
  - **Entropy 2nd law:** total `Σ mᵢsᵢ` (`sᵢ=(γ−1)uᵢ/ρᵢ^{γ−1}`) monotonically
    **non-decreasing** over the run (the clean statement the isothermal path
    could never make; viscous heating `≥0 ⇒ ↑`, rarefaction isentropic ⇒ ~0).
    Secondary spot check `s*_R > s_R` (~5.5% jump, contact-blip-muddied → kept
    secondary).
  - Momentum tripwire unaffected.
  - Tolerances calibrated from a `--release --ignored` run (E2b discipline).

### E4 — per-particle CFL into the adaptive-dt path + negative-`u` floor
- Wire per-particle `c_s,i` into `cfl.rs` / `max_stable_dt`; adaptive-dt path
  consumes it.
- Positive-`u` floor `u ← max(u, u_min)` with a logged, bounded
  non-conservation accounting.
- **Gates:** adaptive-dt adiabatic run holds the convergence + contraction-
  staleness gates (as the isothermal adaptive path does); floor engages only
  under genuine over-cooling and its energy leak is bounded/reported.

**Out of arc (named, not built here):** temperature-dependent gas color is a
*separate later Phase-2 visual session* — now unblocked, but it needs a
frame-data payload decision (voxel stays ρ-only today) and is not in this arc.
grad-h `f_i` terms and the Balsara switch remain deferred accuracy riders.

---

## Standing-constraint deltas (update `long-burning-beacon.md` on completion)

- **D4 partially renegotiated:** total energy becomes a conservation gate on
  the **adiabatic** path (oscillation-bounded, grad-h-justified tolerance);
  isothermal runs remain a heat bath with NO energy gate. Update the D4 line.
- **D2 unviolated:** `u` is evolved, not derived; h/ρ stay derived-never-stored.
- **Voxel payload stays ρ-only** — untouched; temperature-color's frame-data
  bump is a separate later milestone.
- **Snapshot bump is sim-format only** (`galaxy-io` v3); frame-data v2 is not
  touched.
- **Negative-`u` floor** is a known, documented energy-formulation wart with a
  bounded, reported leak.

## TDD / process

Standard project discipline: tests before impl; API surface with `todo!()`
bodies; confirm red; commit `[red]` tests separately (workspace must still
`cargo build`); implement without touching tests; `./gate.ps1` each slice;
end-of-batch ritual (memory + docs, gate, commit AND push) per session.
