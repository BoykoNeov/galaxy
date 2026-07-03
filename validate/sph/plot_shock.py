#!/usr/bin/env python
"""Overlay the isothermal SPH shock-tube profile on the exact Riemann solution.

The M7b demo (sibling of validate/rebound/cross_check.py). The Rust side runs the
same IC/solver as the `sph_shock_tube` gate and dumps a JSON profile; this script
draws the SPH particle density and x-velocity against the closed-form isothermal
Riemann solution (left rarefaction + right shock).

Usage:
    cargo run -p galaxy-validate --release --example export_shock_tube -- prof.json
    python validate/sph/plot_shock.py prof.json [out.png]
"""
import json
import sys

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib/numpy required: `pip install matplotlib numpy`")


def riemann(xi, cs, rho_l, rho_r, rho_s, u_s, s_shock):
    """Analytic self-similar isothermal Riemann state (rho, u) at xi = x/t."""
    head, tail = -cs, u_s - cs
    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    left = xi <= head
    fan = (xi > head) & (xi < tail)
    star = (xi >= tail) & (xi < s_shock)
    right = xi >= s_shock
    rho[left], u[left] = rho_l, 0.0
    rho[fan] = rho_l * np.exp(-1.0 - xi[fan] / cs)
    u[fan] = xi[fan] + cs
    rho[star], u[star] = rho_s, u_s
    rho[right], u[right] = rho_r, 0.0
    return rho, u


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "shock_profile.json"
    out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".png"
    d = json.load(open(src))

    x = np.array(d["x"])
    t = d["t"]
    xi_line = np.linspace(x.min() / t, x.max() / t, 800)
    rho_a, u_a = riemann(
        xi_line, d["cs"], d["rho_l"], d["rho_r"], d["rho_star"], d["u_star"], d["s_shock"]
    )
    x_line = xi_line * t

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for ax, sph, ana, label in (
        (ax1, d["rho"], rho_a, r"density $\rho$"),
        (ax2, d["u"], u_a, r"velocity $u_x$"),
    ):
        ax.plot(x, sph, "o", ms=2.5, alpha=0.35, color="#1f77b4", label="SPH particles")
        ax.plot(x_line, ana, "-", lw=2, color="#d62728", label="exact isothermal Riemann")
        ax.axvline(d["s_shock"] * t, ls=":", color="gray", lw=1)
        ax.set_ylabel(label)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.2)
    ax2.set_xlabel("x")
    ax1.set_title(
        f"Isothermal SPH shock tube  (t={t:.2f}, "
        rf"$\rho_*$={d['rho_star']:.3f}, S={d['s_shock']:.3f})"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
