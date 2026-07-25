#!/usr/bin/env python

"""
Sweep `variational_spec_cond`'s `max_alpha` (the per-iteration line-search
step-size cap) and track, at fixed iteration checkpoints, the spectral
width (M_pq) and shape error (via `angle_matched_shape_error`) achieved.

This is a regression check as much as a diagnostic: `hwM_pq(alpha)` along
the descent direction can have an extremely narrow true minimum surrounded
by a much worse, nearly flat plateau (see the `xatol` fix in
`variational_spec_cond`'s line search). Before that fix, some `max_alpha`
values could get stuck on the plateau and diverge; after it, the
trajectories for every `max_alpha` here should coincide, tracing out one
shared shape-error-vs-spectral-width tradeoff curve.

Uses the same dofs as spline_surface_plot.py.
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.objectives.shape_errors import (
    angle_matched_shape_error,
    build_angle_matched_reference,
)

matplotlib.use("qtagg")

SPLINE_KWARGS = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 4,
    "nfp": 2,
    "M": 12,
    "N": 12,
    "p_u": 3,
    "p_v": 3,
    "cs_equispaced": True,
    "rays_equispaced": False,
    "cs_global_angle_free": False,
    "axis_angles_fixed": True,
    "cs_basis": "polar",
    "nurbs": False,
    "knot_parametrization": "chord",
}

NEW_X = np.array(
    [
        0.3016216,
        0.20890161,
        0.44766483,
        0.49065546,
        0.31258963,
        2.72104182,
        0.13272592,
        0.01054766,
        0.55550307,
        0.34290225,
        0.24441989,
        0.54088171,
        1.42225604,
        2.49345601,
        3.26726428,
        3.90173828,
        5.75958653,
        0.03943073,
        0.17513655,
        0.61373303,
        0.17959677,
        0.26507675,
        0.72124269,
        1.57072331,
        2.02542396,
        3.16716533,
        4.71228105,
        5.49176038,
        0.04404639,
        0.59041262,
        0.48036493,
        0.16049688,
        1.39121464,
        1.62414016,
        1.68508211,
        2.00728408,
        -0.28335058,
    ]
)

MAX_ALPHAS = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
CHUNK = 20
N_CHUNKS = 50  # checkpoints up to niters = CHUNK * N_CHUNKS = 400


def build_reference_surface():
    surf = SurfaceBSpline(default_r=0.01, **SPLINE_KWARGS)
    surf.axis.fix("r_axis_0")
    surf.set_dofs_from_vec(NEW_X)
    return surf.to_RZFourier(collocation="arclength", spec_cond=None)


def hwM_pq(rz_surf, p=4, q=1):
    """Mirrors variational_spec_cond's internal hwM_pq, computed externally
    (that closure isn't exposed) so trajectories can be checkpointed."""
    m_arr = np.arange(0, rz_surf.mpol + 1)
    rbc = rz_surf.rc
    zbs = rz_surf.zs
    num = np.einsum("m,mn->mn", m_arr ** (p + q), rbc**2 + zbs**2)
    denom = np.einsum("m,mn->mn", m_arr**p, rbc**2 + zbs**2)
    return np.sum(num) / np.sum(denom)


def run_sweep(rz_before):
    reference = build_angle_matched_reference(rz_before, nu=128, nv=128)

    results = {
        ma: {"niters": [], "M": [], "mean_err": [], "max_err": [], "t": []}
        for ma in MAX_ALPHAS
    }
    for ma in MAX_ALPHAS:
        current = rz_before
        t_cum = 0.0
        for c in range(1, N_CHUNKS + 1):
            t0 = time.perf_counter()
            current = current.variational_spec_cond(
                niters=CHUNK,
                max_alpha=ma,
                ftol=0,
                Mtol=0,
                verbose=False,
            )
            t_cum += time.perf_counter() - t0

            err = angle_matched_shape_error(current, reference)
            results[ma]["niters"].append(c * CHUNK)
            results[ma]["M"].append(hwM_pq(current))
            results[ma]["mean_err"].append(np.mean(err))
            results[ma]["max_err"].append(np.max(err))
            results[ma]["t"].append(t_cum)
        print(
            f"max_alpha={ma:.0e}  final M={results[ma]['M'][-1]:.4f}  "
            f"mean_err={results[ma]['mean_err'][-1]:.4e}  t={t_cum:.2f}s"
        )
    return results


def plot_sweep(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(results)))

    for (ma, r), color in zip(results.items(), colors):
        axes[0].plot(
            r["niters"],
            r["M"],
            "o-",
            ms=3,
            color=color,
            label=f"max_alpha={ma:.0e}",
        )
        axes[1].semilogy(
            r["niters"],
            r["mean_err"],
            "o-",
            ms=3,
            color=color,
            label=f"max_alpha={ma:.0e}",
        )
        axes[2].plot(
            r["M"],
            r["mean_err"],
            "o-",
            ms=3,
            color=color,
            label=f"max_alpha={ma:.0e}",
        )

    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("M_pq (spectral width)")
    axes[0].set_title("Convergence of spectral width")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel("mean shape error")
    axes[1].set_title("Shape error growth")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel("M_pq (spectral width)")
    axes[2].set_ylabel("mean shape error")
    axes[2].set_title(
        "Shape error vs spectral width achieved\n(the actual tradeoff)"
    )
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    axes[2].invert_xaxis()

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    rz_before = build_reference_surface()
    print(f"initial M_pq = {hwM_pq(rz_before):.4f}\n")

    results = run_sweep(rz_before)
    plot_sweep(results)
    plt.show()
