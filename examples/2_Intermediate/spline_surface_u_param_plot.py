#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.util.spline_helpers import b_p, print_dofs_nicely

matplotlib.use("qtagg")


def plot_colored_by_u(
    surf,
    ax=None,
    nu=64,
    nv=64,
    cmap="viridis",
):
    """
    Plot the spline surface, coloring each face by the spline's own u
    parameter (the poloidal NURBS parameter, not a physical angle -- see
    gamma_lin's docstring). Otherwise mirrors SurfaceBSpline.plot()'s
    surface-generation block.
    """
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    u = np.linspace(0, 2 * np.pi, nu, endpoint=True)
    v = np.linspace(0, 2 * np.pi, nv, endpoint=True)
    v_grid, u_grid = np.meshgrid(v, u)

    x_surf, y_surf, z_surf = surf.surf_callable(
        u_grid.flatten(), v_grid.flatten()
    )
    x_surf = x_surf.reshape(nu, nv)
    y_surf = y_surf.reshape(nu, nv)
    z_surf = z_surf.reshape(nu, nv)

    # facecolors are per-face, not per-vertex -- use each face's corner u value
    u_face = u_grid[:-1, :-1]
    norm = plt.Normalize(vmin=0, vmax=2 * np.pi)
    facecolors = plt.get_cmap(cmap)(norm(u_face))

    ax.plot_surface(
        x_surf,
        y_surf,
        z_surf,
        facecolors=facecolors,
        shade=False,
        rcount=nu,
        ccount=nv,
    )
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("Spline surface colored by u (spline poloidal parameter)")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, shrink=0.6, label="u")

    return ax


def plot_basis_functions(surf, n_eval=1000):
    """
    Plot the u- and v-direction B-spline basis functions using the
    surface's own current knot vectors (knots_u, knots_v from
    _control_net_and_knots), one line per basis function.
    """
    _, _, knots_u, knots_v = surf._control_net_and_knots()
    p_u, p_v = surf.p_u, surf.p_v

    fig, (ax_u, ax_v) = plt.subplots(2, 1, figsize=(8, 7))

    for ax, knots, p, label in (
        (ax_u, knots_u, p_u, "u"),
        (ax_v, knots_v, p_v, "v"),
    ):
        x = np.linspace(knots[p], knots[-p - 1], n_eval, endpoint=False)
        basis = b_p(knots, p, x)
        for i in range(basis.shape[1]):
            ax.plot(x, basis[:, i], lw=1)
        for k in knots:
            if knots[p] <= k <= knots[-p - 1]:
                ax.axvline(k, color="k", lw=0.5, alpha=0.3)
        ax.set_xlim(knots[p], knots[-p - 1])
        ax.set_xlabel(label)
        ax.set_ylabel(f"$B_{{i,{p}}}({label})$")
        ax.set_title(
            f"{label}-direction basis functions (knot_parametrization={surf.knot_parametrization!r})"
        )

    fig.tight_layout()
    return fig, (ax_u, ax_v)


if __name__ == "__main__":
    spline_kwargs = {
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

    spline_surf = SurfaceBSpline(
        default_r=0.01,
        **spline_kwargs,
    )
    spline_surf.axis.fix("r_axis_0")
    new_x = np.array(
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
    spline_surf.set_dofs_from_vec(new_x)

    print_dofs_nicely(spline_surf)

    spline_surf.plot()
    plt.show()

    plot_colored_by_u(spline_surf)
    plt.show()

    plot_basis_functions(spline_surf)
    plt.show()
