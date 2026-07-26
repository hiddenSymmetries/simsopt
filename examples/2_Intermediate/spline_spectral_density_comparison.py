#!/usr/bin/env python

"""
Compare the RZFourier spectral power (vs poloidal mode number) obtained from
a SurfaceBSpline via the "exact", "uniform", and "arclength" collocation
methods in `to_RZFourier`, before and after `variational_spec_cond`. Also
records the wall-clock time to obtain each result, for M=N in {3, 6, 12, 24}.

Uses the same dofs as spline_surface_plot.py.
"""

import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo.surfacespline import SurfaceBSpline

matplotlib.use("qtagg")

BASE_SPLINE_KWARGS = {
    "axis_points": 3,
    "points_per_cs": 6,
    "n_cs": 4,
    "nfp": 2,
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

M_VALUES = [3, 6, 12, 24]
COLLOCATIONS = ["exact", "uniform", "arclength"]

SPEC_COND_OPTIONS = {
    "plot": False,
    "ftol": 1e-4,
    "Mtol": 1.1,
    "shapetol": 0.01,
    "niters": 400,
    "verbose": False,
    "cutoff": 1e-6,
}


def build_surface(M):
    kwargs = dict(BASE_SPLINE_KWARGS, M=M, N=M)
    surf = SurfaceBSpline(default_r=0.01, **kwargs)
    surf.axis.fix("r_axis_0")
    surf.set_dofs_from_vec(NEW_X)
    return surf


def spectral_power(rz_surf):
    """Spectral power per poloidal mode m: sum over n of (rc**2 + zs**2)."""
    return np.sum(rz_surf.rc**2 + rz_surf.zs**2, axis=1)


def run():
    results = {}
    for M in M_VALUES:
        for collocation in COLLOCATIONS:
            surf = build_surface(M)

            t0 = time.perf_counter()
            rz_before = surf.to_RZFourier(collocation=collocation, spec_cond=None)
            t_ft = time.perf_counter() - t0

            t0 = time.perf_counter()
            rz_after = rz_before.variational_spec_cond(**SPEC_COND_OPTIONS)
            t_cond = time.perf_counter() - t0

            results[(M, collocation)] = {
                "before": spectral_power(rz_before),
                "after": spectral_power(rz_after),
                "t_ft": t_ft,
                "t_after": t_ft + t_cond,
            }
            print(
                f"M=N={M:>3}  collocation={collocation:<10}  "
                f"t_ft={t_ft:6.3f}s  t_after={t_ft + t_cond:6.3f}s"
            )
    return results


def plot_spectral_power(results):
    fig, axes = plt.subplots(1, len(M_VALUES), figsize=(4.5 * len(M_VALUES), 4.5))
    colors = {"exact": "C0", "uniform": "C1", "arclength": "C2"}
    for ax, M in zip(axes, M_VALUES):
        for collocation in COLLOCATIONS:
            r = results[(M, collocation)]
            m = np.arange(len(r["before"]))
            color = colors[collocation]
            ax.semilogy(
                m, r["before"], ":", marker="o", ms=3, color=color,
                label=f"{collocation}, before condensation",
            )
            ax.semilogy(
                m, r["after"], "-", marker="o", ms=3, color=color,
                label=f"{collocation}, after condensation",
            )
        ax.set_title(f"M=N={M}")
        ax.set_xlabel("poloidal mode number m")
        ax.set_xlim(0, M)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel(r"spectral power  $P_m = \sum_n (R_{mn}^2 + Z_{mn}^2)$")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Spectral power vs poloidal mode number")
    fig.tight_layout()
    return fig


def plot_wall_clock(results):
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {"exact": "C0", "uniform": "C1", "arclength": "C2"}
    for collocation in COLLOCATIONS:
        t_ft = [results[(M, collocation)]["t_ft"] for M in M_VALUES]
        t_after = [results[(M, collocation)]["t_after"] for M in M_VALUES]
        ax.plot(
            M_VALUES, t_ft, "o-", color=colors[collocation],
            label=f"{collocation}, before condensation",
        )
        ax.plot(
            M_VALUES, t_after, "s--", color=colors[collocation],
            label=f"{collocation}, after condensation",
        )
    ax.set_xlabel("M = N")
    ax.set_ylabel("wall-clock time [s]")
    ax.set_yscale("log")
    ax.set_xticks(M_VALUES)
    ax.legend(fontsize=8)
    ax.set_title("Wall-clock time to obtain each result")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    results = run()
    plot_spectral_power(results)
    plot_wall_clock(results)
    plt.show()
