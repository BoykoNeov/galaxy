#!/usr/bin/env python
"""Cross-check the galaxy-collider engine against REBOUND (IAS15).

This is the heavier, *manually-run* half of the M2 validation. The always-on
conservation and orbital-setup checks live in the Rust test suite; this harness
adds an INDEPENDENT high-order integrator (REBOUND IAS15) on the SAME initial
conditions and confirms the two codes describe the same physical system and reach
a statistically consistent state through the approach and first encounter.

It does NOT validate late-time chaotic evolution — N-body is chaotic, so positions
(and eventually even coarse statistics) diverge between any two integrators. The
tight assertions are restricted to t=0 and the pre/early-encounter phase.

Usage (where REBOUND + NumPy are installed):
    1. cargo run -p galaxy-validate --release --example export_collision -- <dir>
    2. python validate/rebound/cross_check.py <dir>

REBOUND softening is Plummer-form (F = -G m1 m2 dr / (r^2 + b^2)^{3/2}, with
b = sim.softening), identical to our DirectSum kernel — see the REBOUND docs.
sim.energy() is UNSOFTENED, so we compute the softened energy ourselves with the
same formula the Rust side uses.
"""
import json
import sys

import numpy as np

try:
    import rebound
except ImportError:
    sys.exit(
        "REBOUND is not installed. `pip install rebound` and re-run.\n"
        "This harness is intentionally NOT part of `cargo test`."
    )

# --- tolerances ---------------------------------------------------------------
ACC_REL_TOL = 1e-10      # t=0 force agreement (same G, masses, softening, positions)
ENERGY_REL_TOL = 1e-9    # t=0 softened-energy self-consistency across the boundary
REBOUND_DRIFT_TOL = 1e-6 # IAS15 should conserve (its own) energy to ~machine eps
OUR_DRIFT_TOL = 2e-3     # our leapfrog: a bounded oscillation, not drift
APPROACH_FRAC = 0.4      # first 40% of the run = pre/early-encounter (tight tol)
APPROACH_RH_TOL = 0.05   # half-mass radii agree to 5% while progenitors are distinct
LATE_RH_TOL = 0.5        # gross-divergence guard only; chaos is expected after pericenter


def softened_energy(pos, vel, mass, g, eps):
    ke = 0.5 * np.sum(mass * np.sum(vel * vel, axis=1))
    pe = 0.0
    for i in range(len(mass)):
        dx = pos[i + 1:] - pos[i]
        r = np.sqrt(np.sum(dx * dx, axis=1) + eps * eps)
        pe -= g * mass[i] * np.sum(mass[i + 1:] / r)
    return ke + pe


def half_mass_radius(pos, mass, sel):
    p, m = pos[sel], mass[sel]
    com = np.sum(p * m[:, None], axis=0) / np.sum(m)
    r = np.sort(np.sqrt(np.sum((p - com) ** 2, axis=1)))
    return r[len(r) // 2]


def build_rebound(pos, vel, mass, g, eps):
    sim = rebound.Simulation()
    sim.G = g
    sim.softening = eps          # Plummer-form, matches our DirectSum kernel
    sim.gravity = "basic"        # O(N^2) direct gravity (exact, like DirectSum)
    sim.integrator = "ias15"     # high-order adaptive: the independent oracle
    for m, p, v in zip(mass, pos, vel):
        sim.add(m=float(m), x=p[0], y=p[1], z=p[2], vx=v[0], vy=v[1], vz=v[2])
    # Our IC is already in the zero-COM/zero-momentum frame, so we do NOT call
    # move_to_com() — keeping positions identical for the exact t=0 force check.
    return sim


def rebound_state(sim):
    n = sim.N
    pos = np.array([[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z] for i in range(n)])
    vel = np.array([[sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz] for i in range(n)])
    return pos, vel


def rebound_accel(sim):
    sim.update_acceleration()  # fills particles[i].a{x,y,z} for the current state
    return np.array([[sim.particles[i].ax, sim.particles[i].ay, sim.particles[i].az]
                     for i in range(sim.N)])


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python cross_check.py <export-dir>")
    d = sys.argv[1].rstrip("/\\")
    params = json.load(open(d + "/params.json"))
    g, eps = params["g"], params["softening"]

    pos = np.load(d + "/ic/pos.npy")
    vel = np.load(d + "/ic/vel.npy")
    mass = np.load(d + "/ic/mass.npy")
    acc_ref = np.load(d + "/ic/acc.npy")          # our DirectSum IC accelerations
    prog = np.load(d + "/ic/progenitor.npy").astype(int)

    t_series = np.load(d + "/ours/time.npy")
    our_energy = np.load(d + "/ours/energy.npy")
    our_rh0 = np.load(d + "/ours/rh0.npy")
    our_rh1 = np.load(d + "/ours/rh1.npy")

    failures = []

    def check(name, ok, detail):
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}: {detail}")
        if not ok:
            failures.append(name)

    sim = build_rebound(pos, vel, mass, g, eps)

    print("== t=0: same physical system ==")
    # SPINE: REBOUND's force evaluation must match our exact softened DirectSum
    # forces. If G, masses, softening kernel, or positions differ, this fails
    # immediately — no integration, no chaos, no tolerance guessing.
    # update_acceleration() is the gravity-eval trigger; if it's absent in this
    # REBOUND version we lose only this one check — the energy self-consistency,
    # conservation parity, and half-mass-radius checks below still cross-check.
    try:
        acc_reb = rebound_accel(sim)
        arms = np.sqrt(np.mean(np.sum(acc_ref ** 2, axis=1)))
        acc_rel = np.max(np.linalg.norm(acc_reb - acc_ref, axis=1)) / arms
        check("t=0 accelerations match DirectSum", acc_rel < ACC_REL_TOL, f"rel err {acc_rel:.3e}")
    except (AttributeError, TypeError) as e:
        print(f"  [SKIP] t=0 accelerations: sim.update_acceleration() unavailable "
              f"({e}); relying on the energy/conservation/half-mass checks below")

    e0 = softened_energy(pos, vel, mass, g, eps)
    e_rel = abs(e0 - params["ic_softened_energy"]) / abs(params["ic_softened_energy"])
    check("t=0 softened energy self-consistent", e_rel < ENERGY_REL_TOL, f"rel err {e_rel:.3e}")

    print("== integrate and compare ==")
    reb_energy = np.empty_like(t_series)
    reb_rh0 = np.empty_like(t_series)
    reb_rh1 = np.empty_like(t_series)
    for k, t in enumerate(t_series):
        sim.integrate(float(t))
        rp, rv = rebound_state(sim)
        reb_energy[k] = softened_energy(rp, rv, mass, g, eps)
        reb_rh0[k] = half_mass_radius(rp, mass, prog == 0)
        reb_rh1[k] = half_mass_radius(rp, mass, prog == 1)

    reb_drift = np.max(np.abs((reb_energy - reb_energy[0]) / reb_energy[0]))
    our_drift = np.max(np.abs((our_energy - our_energy[0]) / our_energy[0]))
    check("REBOUND conserves energy", reb_drift < REBOUND_DRIFT_TOL, f"max drift {reb_drift:.3e}")
    check("our engine conserves energy", our_drift < OUR_DRIFT_TOL, f"max drift {our_drift:.3e}")

    # Statistical agreement: per-progenitor half-mass radius vs time. Tight while
    # the galaxies are still distinct clumps (approach / early encounter); only a
    # gross-divergence guard afterwards, where phase mixing + chaos take over.
    print("== half-mass radii (ours vs REBOUND) ==")
    print(f"  {'t':>7} {'rh0_our':>9} {'rh0_reb':>9} {'rh1_our':>9} {'rh1_reb':>9}")
    n_approach = max(1, int(APPROACH_FRAC * len(t_series)))
    worst_approach = 0.0
    worst_late = 0.0
    for k, t in enumerate(t_series):
        d0 = abs(reb_rh0[k] - our_rh0[k]) / our_rh0[k]
        d1 = abs(reb_rh1[k] - our_rh1[k]) / our_rh1[k]
        if k < n_approach:
            worst_approach = max(worst_approach, d0, d1)
        else:
            worst_late = max(worst_late, d0, d1)
        if k % 5 == 0 or k == len(t_series) - 1:
            print(f"  {t:7.2f} {our_rh0[k]:9.4f} {reb_rh0[k]:9.4f} {our_rh1[k]:9.4f} {reb_rh1[k]:9.4f}")

    check("approach-phase half-mass radii agree", worst_approach < APPROACH_RH_TOL,
          f"worst rel diff {worst_approach:.3e} over first {n_approach} snapshots")
    check("no gross late-time divergence", worst_late < LATE_RH_TOL,
          f"worst rel diff {worst_late:.3e} (chaos expected; guard only)")

    print()
    if failures:
        sys.exit(f"CROSS-CHECK FAILED: {', '.join(failures)}")
    print("CROSS-CHECK PASSED (setup + early encounter consistent with REBOUND IAS15)")


if __name__ == "__main__":
    main()
