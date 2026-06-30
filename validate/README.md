# galaxy-validate

Cross-check tooling for the N-body engine.

Validation in this project comes in two tiers:

1. **Always-on (`cargo test`)** — the non-negotiable checks, run on every build:
   - *Conservation*: a small two-galaxy collision under the exact `DirectSum`
     oracle conserves total energy (bounded symplectic oscillation), linear and
     angular momentum (to roundoff), and keeps the barycenter fixed
     (`galaxy-sim` tests).
   - *Orbital setup*: the collision's relative two-body orbit reproduces the
     requested conic (eccentricity, pericenter, parabolic zero-energy) via an
     independent osculating-elements formula (`galaxy-ic` tests).
   - *Analytic Kepler* and *equilibrium* (`galaxy-solvers`, `galaxy-ic` tests).

   These do **not** claim an independent-integrator cross-check — they verify the
   setup and the conservation laws. The independent integrator is REBOUND, below.

2. **Manual REBOUND cross-check (this crate)** — the heavier check that an
   INDEPENDENT high-order integrator (REBOUND IAS15) on the **same** initial
   conditions describes the same physical system and reaches a statistically
   consistent state through the approach and first encounter. It is **not** part
   of `cargo test` (REBOUND is a C-extension Python package; on Windows the HDF5
   path is a link landmine, so we bridge with NumPy `.npy`, not HDF5).

## Running the REBOUND cross-check

```sh
# 1. Export a collision IC + our run's diagnostics as .npy (release: DirectSum is O(N^2))
cargo run -p galaxy-validate --release --example export_collision -- /path/to/out

# 2. Cross-check against REBOUND (needs `pip install rebound numpy`)
python validate/rebound/cross_check.py /path/to/out
```

The export writes (all `<f8` C-order `.npy`):

```
out/ic/{pos,vel,mass,progenitor,acc}.npy   # IC phase space + our DirectSum IC forces
out/ours/{time,energy,rh0,rh1}.npy         # our run's diagnostics vs time
out/ours/{final_pos,final_vel}.npy         # our final state
out/params.json                            # G, softening, dt, counts, IC energy
```

## What the harness checks (and its limits)

- **t=0 forces** — REBOUND's accelerations (`update_acceleration`, Plummer
  softening `sim.softening`, `gravity="basic"`) must match our exported
  `DirectSum` accelerations to ~roundoff. This is the spine: it proves *same G,
  masses, softening kernel, and positions* before a single step — no chaos, no
  tolerance guessing.
- **t=0 energy** — softened total energy computed across the language boundary
  agrees with our exported value.
- **Energy conservation** — REBOUND IAS15 to ~machine epsilon; our leapfrog a
  bounded oscillation (~1e-4 here).
- **Per-progenitor half-mass radius vs time** — tight (5%) while the galaxies are
  still distinct clumps (approach / early encounter); a gross-divergence guard
  only afterward.

It does **NOT** validate late-time chaotic evolution — N-body is chaotic, so
trajectories (and eventually even coarse statistics) diverge between any two
integrators. That is physics, not a bug.

During development the harness's `.npy` physics formulas (softened energy,
softened accelerations, half-mass radius) were checked against the Rust engine and
agreed to roundoff — an ad-hoc check, not part of `cargo test`. So a harness
failure most likely points at REBOUND configuration or the engine, not the bridge.
If `sim.update_acceleration()` is missing in your REBOUND version, the t=0 force
check is skipped and the remaining checks still run.
