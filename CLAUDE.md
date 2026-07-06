# Galaxy Collider — Claude working notes

Headless Rust N-body engine for galaxy collisions, with an offline,
file-decoupled visualization pipeline. Full architecture & decisions live in
DESIGN.md — read it, don't duplicate it here.

## Stack
- Rust workspace (edition 2021). `glam` (DVec3 math), `rayon` (CPU parallel, later),
  `wgpu` (GPU render, later), `proptest` (property tests).
- Compute in f64. `State` is Structure-of-Arrays.

## Commands
- Build:  `cargo build`
- Test:   `cargo test`   (single crate: `cargo test -p galaxy-core`)
- Lint:   `cargo clippy --all-targets -- -D warnings`
- Format: `cargo fmt`  /  check: `cargo fmt --check`
- Gate:   `./gate.ps1` — the full quality gate (fmt → clippy → test), ordered
  cheapest-fail-first so a format/lint slip costs seconds, not the whole test
  run. `-SkipTests` runs just fmt + clippy. Tests link `[profile.dev]
  opt-level = 2` (Cargo.toml): the compute-bound N-body/SPH proptests run ~9×
  faster optimized (debug-assertions + overflow-checks stay on; no fast-math,
  so invariants hold). Trade-off: editing `core` recompiles the tree at opt-2
  (~2× the debug compile) — worth it for the ~2.6× faster gate run.

## TDD workflow (REQUIRED)
1. Write tests BEFORE implementation.
2. Define only the API surface (signatures with `todo!()` bodies) so tests compile.
3. Run `cargo test` and CONFIRM TESTS FAIL. Never write a fake/mock impl that
   passes trivially — stubs must be `todo!()`.
4. Commit the failing tests SEPARATELY from the implementation (mark `test(...): ... [red]`).
5. Implement until tests pass. Do NOT modify tests to make them pass.

## Testing standards
- Prefer invariants over example outputs: energy / momentum / angular-momentum
  conservation, leapfrog time-reversibility, orbit closure — use `proptest`.
- Parameterize inputs; compare against independent hand-derived expectations,
  not the function's own output.
- Tolerances justified by the method's order; document why. Leapfrog energy should
  OSCILLATE within a bound, not drift.
- Physics tests that combine solver+integrator live in `*/tests/` (integration).

## Rust conventions
- Newtypes for identity (`ParticleId`, `Progenitor`) — don't pass bare u64/u16.
- Errors via `Result` + `thiserror` in library code; fail fast with context.
  No silent `.unwrap()`/`.expect()` on fallible lib paths (tests may unwrap).
- Keep `core` pure (no I/O, no rendering). Solvers implement `ForceSolver` and are
  swappable without touching callers (this is the 10^8 / cosmology door).

## Git etiquette
- Conventional Commits (`feat:`, `test:`, `chore:`, `docs:`, `fix:`).
- Every commit compiles and passes tests — EXCEPT the deliberate red test commit.
- Commit tests separately; tie commits to milestones (M0, M1, …).
- NOTE: this harness mandates a `Co-Authored-By: Claude …` trailer, which overrides
  the upstream best-practice of omitting AI references.

## Gotchas (what to get right)
- Use the SOFTENED potential in energy diagnostics so it matches the softened force.
- Additive star splatting is order-independent; gas raymarching (absorption) is NOT —
  don't reuse the splat path for gas.
- HDF5 on Windows is a linker landmine — keep it to validation runs only.
- Accumulate the HDR render buffer in 32-bit float (cores saturate in 16-bit).
