//! SPH velocity divergence `∇·v` — the "converging flow" half of the
//! star-formation criterion (plan `natal-ember-forge.md`, F5). Standard SPH
//! gather estimate:
//!
//! ```text
//! ∇·v_i = (1/ρ_i) Σ_{j≠i} m_j (v_j − v_i) · ∇_i W(r_ij, h_i)
//! ```
//!
//! A per-target `h_i` gather (NOT the force path's kernel-average `W̄` over the
//! global `h_max`): `∇·v` is a diagnostic scalar with no antisymmetry /
//! conservation requirement, so the simple gather matching the density estimate
//! is exactly right — the `W̄` machinery exists only to protect Newton's third
//! law in the *force*, which is irrelevant here. Like the density path, each
//! target gathers neighbours in ascending index (fixed order), so the rayon
//! result is order-independent (parallel ≡ serial) for free.
//!
//! Sign: the kernel gradient points down-gradient (`∇_i W = (neg scalar)·r_ij`),
//! so a converging field `v = −k(x−x_c)` gives every term `< 0` ⇒ `∇·v < 0`, and
//! a diverging field flips it — sign-correct for every particle regardless of
//! edge position (a linear velocity field makes the sign per-pair uniform).

use galaxy_core::DVec3;
use rayon::prelude::*;

use super::grid::HashGrid;
use super::kernel::{grad_w, SUPPORT};

/// Per-particle SPH velocity divergence over a gas subset (all slices length
/// `pos.len()`; `rho`/`h` are the adaptive-h density solve's output for the same
/// subset). `rho_i > 0` for every gas particle (the density sum includes the
/// self-term `m_i·W(0,h_i) > 0`), so the `1/ρ_i` denominator never divides by
/// zero — a particle with no neighbours gets an empty `j≠i` sum ⇒ `∇·v = 0`.
pub fn velocity_divergence(
    pos: &[DVec3],
    vel: &[DVec3],
    mass: &[f64],
    rho: &[f64],
    h: &[f64],
) -> Vec<f64> {
    if pos.is_empty() {
        return Vec::new();
    }
    let h_max = h.iter().fold(0.0_f64, |a, &b| a.max(b));
    assert!(
        h_max.is_finite() && h_max > 0.0,
        "velocity_divergence needs positive finite smoothing lengths"
    );
    // Grid at the GLOBAL support so a per-target query at SUPPORT·h_i (≤ h_max) is
    // always valid — the same superset-gather pattern as `density_fixed`. Each
    // target's neighbour sum runs in ascending index (fixed order), so the rayon
    // collect is bit-identical to a serial one.
    let grid = HashGrid::build(pos, SUPPORT * h_max);
    (0..pos.len())
        .into_par_iter()
        .map(|i| {
            let ngb = grid.neighbours_within(pos, pos[i], SUPPORT * h[i]);
            let mut sum = 0.0;
            for &j in &ngb {
                if j == i {
                    continue;
                }
                // ∇_i W(r_ij, h_i), r_ij = x_i − x_j; zero past SUPPORT·h_i (so the
                // wider grid gather only adds exact 0.0 terms — order-independent).
                let gw = grad_w(pos[i] - pos[j], h[i]);
                sum += mass[j] * (vel[j] - vel[i]).dot(gw);
            }
            // ρ_i > 0 (self-term floor), so this never divides by zero; an empty
            // j≠i sum leaves 0/ρ_i = 0.
            sum / rho[i]
        })
        .collect()
}
