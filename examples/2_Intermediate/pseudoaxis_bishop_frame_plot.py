#!/usr/bin/env python
"""
Build a PseudoAxis with randomly perturbed dofs and plot it together with
its Bishop (rotation-minimizing) frame -- see PseudoAxis.bishop_frame's
docstring for the construction (discrete double-reflection method,
corrected to close exactly across the nfp field periods).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from simsopt.geo.plotting import fix_matplotlib_3d
from simsopt.geo.surfacespline import PseudoAxis

matplotlib.use("qtagg")


def random_pseudoaxis(
    n_ctrl_pts=3,
    p=3,
    nfp=2,
    stellsym=True,
    quadpoints=200,
    perturbation_scale=0.5,
    seed=None,
    max_attempts=50,
):
    """
    Build a PseudoAxis initialized to a circle, then perturb its free
    dofs by independent Gaussian noise (perturbation_scale is the
    standard deviation, in the dofs' own units -- r/z control points are
    length-like, zeta control points are radians).

    At large perturbation_scale (especially with few control points), a
    draw can push the axis's projection onto the XY-plane into winding
    backward in toroidal angle somewhere -- gamma_impl's Newton solve
    then either fails outright or (worse) silently converges to the
    wrong branch, since the "point at toroidal angle phi" isn't uniquely
    defined there anymore. Rather than plot whatever comes out, each draw
    is checked with PseudoAxis.is_toroidally_monotonic() and re-drawn
    (same seed's rng, next value) if it fails, up to max_attempts.
    """
    if seed is None:
        seed = np.random.SeedSequence().entropy
    print(f"random_pseudoaxis seed: {seed}")
    rng = np.random.default_rng(seed)

    axis = PseudoAxis(
        n_ctrl_pts=n_ctrl_pts,
        p=p,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints=quadpoints,
    )
    x_default = axis.x.copy()
    for attempt in range(max_attempts):
        axis.x = x_default + perturbation_scale * rng.standard_normal(
            len(x_default)
        )
        if axis.is_toroidally_monotonic():
            return axis
        print(f"  attempt {attempt}: non-monotonic axis, redrawing")

    raise RuntimeError(
        f"random_pseudoaxis: no toroidally-monotonic axis found in "
        f"{max_attempts} attempts at perturbation_scale={perturbation_scale} "
        "-- try lowering it."
    )


def plot_bishop_frame(axis, n_frames=24, frame_length=None, ax=None):
    """
    Plot the axis curve (via Curve.plot()) together with its Bishop frame
    (T, N, B), sampled at n_frames points evenly spaced around the closed
    curve, as colored quivers. The phi=0 point (start of the propagation)
    is marked explicitly, with the R-Z half-plane there drawn as a
    translucent patch, so it's visually unambiguous whether N (green)
    starts out lying in it -- with 3*n_frames arrows on screen, picking
    out "the phi=0 one" by eye alone is not reliable.
    """
    ax = axis.plot(ax=ax, show=False, color="k", lw=1.5)

    t_frames = np.linspace(1e-6, 1 - 1e-6, n_frames, endpoint=False)
    data = np.zeros((n_frames, 3))
    axis.gamma_impl(data, t_frames)
    gamma_frames = data
    T, N, B = axis.bishop_frame(t_frames)

    span = axis.gamma().max(axis=0) - axis.gamma().min(axis=0)
    if frame_length is None:
        frame_length = 0.12 * np.max(span)

    for vec, color in ((T, "tab:red"), (N, "tab:green"), (B, "tab:blue")):
        ax.quiver(
            gamma_frames[:, 0],
            gamma_frames[:, 1],
            gamma_frames[:, 2],
            vec[:, 0],
            vec[:, 1],
            vec[:, 2],
            length=frame_length,
            color=color,
            normalize=True,
        )

    # mark the phi=0 point and draw the R-Z half-plane there for a direct
    # visual check of where N starts
    ax.scatter(
        *gamma_frames[0], color="magenta", s=200, marker="*",
        edgecolor="k", linewidth=0.8, zorder=10,
    )
    x0, y0, z0 = gamma_frames[0]
    r_hat = np.array([x0, y0, 0.0])
    r_hat /= np.linalg.norm(r_hat)
    plane_size = 0.8 * np.max(span)
    r_grid, z_grid = np.meshgrid(
        np.linspace(-plane_size, plane_size, 2),
        np.linspace(-plane_size, plane_size, 2),
    )
    plane_x = x0 + r_grid * r_hat[0]
    plane_y = y0 + r_grid * r_hat[1]
    plane_z = z0 + z_grid
    ax.plot_surface(
        plane_x, plane_y, plane_z,
        color="gold", alpha=0.35, edgecolor="darkorange", linewidth=0.5,
        zorder=1,
    )

    proxies = [
        Line2D([0], [0], color=c, lw=2)
        for c in ("tab:red", "tab:green", "tab:blue")
    ]
    ax.legend(
        proxies + [Line2D([0], [0], color="k", marker="*", lw=0)],
        [
            "T (tangent)",
            "N (Bishop normal)",
            "B (binormal)",
            "phi=0 (R-Z plane shown)",
        ],
    )
    ax.set_title(
        f"PseudoAxis + Bishop frame (nfp={axis.nfp}, stellsym={axis.stellsym})"
    )
    fix_matplotlib_3d(ax)
    return ax


if __name__ == "__main__":
    axis = random_pseudoaxis()
    plot_bishop_frame(axis)
    plt.show()
