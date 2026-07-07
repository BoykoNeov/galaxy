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
No integrator change yet — the most contained slice.
- `HydroParams::Eos` enum (`Isothermal { c_s } | Adiabatic { gamma }`), threaded
  through `forces.rs`/`cfl.rs`.
- `State.u: Vec<f64>` column: into `assert_consistent`, `from_phase_space`
  (all `0.0`), a gas-`u` setter; non-gas rows inert `0.0`.
- Snapshot **v3**: append `u[n]` (f64) after `kind`; reader defaults `u=0.0`
  for v1/v2 (exact same version-tolerant pattern v2 used for `kind`).
- `diagnostics::thermal_energy` + `total_energy` includes it.
- **Gates:** isothermal path **byte-identical** to pre-E1 (EOS enum default =
  Isothermal, no behavior change); adiabatic `P = (γ−1)ρu` correct on a hand
  case; a static adiabatic blob (`du/dt` frozen) stays put; snapshot v3
  round-trips and reads v2 with `u=0`.

### E2 — `du/dt` PdV work + thermal integrator (no shocks)
- `accel_and_dudt` fused pass; PdV term only (viscous heating in E3).
- Thermal integrator branch kicking `u`.
- **Gates:** smooth adiabatic compression of a gas ball heats per `P V^γ =
  const` (independent hand-derived expectation, not the code's own output);
  total energy oscillation-bounded (~%-level) over the run; momentum tripwire
  unchanged (force still exactly antisymmetric).

### E3 — viscous/shock heating + adiabatic Sod shock tube
- Add the viscous heating term to `du/dt`.
- New **γ=1.4 Sod oracle** (classic Riemann; the isothermal oracle cannot
  validate it).
- **Gates:** Sod matches the analytic Riemann solution (density/velocity/
  pressure profiles within method-order tolerance); energy conserved across the
  shock; entropy **increases** through the shock (Rankine–Hugoniot — the
  second-law check the isothermal path could never make).

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
