"""
Heatmap of exact shape error (max and RMS) over a full square (M, N)
sweep -- M and N each independently from 4 up to 32 in steps of 2 --
extending the (M, N) pattern originally swept by
shape_error_vs_mode_number.py (since removed, which only covered the
N <= M triangle) to also cover N > M.

Unlike that removed script, this recomputes SurfaceBSpline.ft()'s
discrete quadrature projection from scratch at every (M, N) pair, each
with its own Nyquist-adequate (nu, nv) collocation grid: nu = 2*M+1
poloidally, nv = 2*N*nfp+2 (rounded up from 2*N*nfp+1 to stay even)
toroidally -- see shape_error_vs_quadrature_resolution.py for why the
nfp factor on nv matters (ft()'s cos/sin argument is n*(nfp*zeta), and
zeta spans the full torus in the "exact" collocation).

Errors are normalized by the average minor radius, from a separate,
fixed M = N = 32 Fourier surface (SurfaceBSpline doesn't implement
gammadash1, so minor_radius() isn't available directly on the spline)
to give a dimensionless shape error.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.pyplot import rcParams

from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.objectives.shape_errors import (
    build_exact_shape_reference,
    exact_shape_error,
)


def build_spline_surf(MPOL, NTOR):
    """
    Identical SurfaceBSpline (spline_kwargs + dofs) to the one built in
    spectral_condensation_solve/vmec_solves_different_theta.py -- the
    ground-truth boundary used throughout this study -- except for the
    declared (M, N) Fourier resolution, which only affects ft()'s default
    truncation/grid, not the spline's own geometry (dofs, axis, cross
    sections are unaffected by M, N).
    """
    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 6,
        "nfp": 3,
        "M": MPOL,
        "N": NTOR,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
        "knot_parametrization": "uniform",
    }
    spline_surf = SurfaceBSpline(default_r=0.3, **spline_kwargs)
    new_x = np.array(
        [
            5.3896343573543060e-02,
            5.0883230048103856e-01,
            2.2489121027379422e-01,
            1.5929768581422210e00,
            1.0304048534365813e-01,
            4.3412658709604501e-01,
            1.9194536849729077e-01,
            5.3647438961678784e-01,
            1.7794171743465446e00,
            3.6038044630128279e00,
            4.9259924741397159e00,
            2.5036189767220018e-01,
            3.5802891724900809e-01,
            9.6212580154516142e-02,
            5.2064914206501489e-01,
            2.0474558259649123e00,
            3.8964954930212143e00,
            5.3228745694944024e00,
            3.2185070391451231e-01,
            2.5146308466749123e-01,
            2.1954123698502295e-01,
            4.5420692619007164e-01,
            2.2214282741552713e00,
            3.0009540172725946e00,
            5.4975172012762998e00,
            3.6991589429711308e-01,
            1.6285912057838300e-01,
            3.6446964917388275e-01,
            3.2242251631177343e-01,
            1.7920172479908414e00,
            3.0210349920179866e00,
            5.4977786325813431e00,
            3.8958534284832108e-01,
            1.9630111359305188e-01,
            4.0847103311261590e-01,
            1.0489177365984679e00,
            2.0043666242346596e00,
            1.3580534688161376e00,
            5.0279307449500388e-01,
            -7.0064725504961656e-01,
        ]
    )
    spline_surf.x = new_x
    return spline_surf


def compute_reference_minor_radius(M_MAX):
    """
    Average minor radius used to normalize shape error into a
    dimensionless quantity. SurfaceBSpline doesn't implement gammadash1,
    so `minor_radius()` (which needs it, via the cross-sectional-area
    formula) isn't available directly on the spline surface -- as a
    stand-in, this uses a fixed, very high-resolution (M = N = M_MAX)
    exact Fourier transform of the ground-truth spline, at least as fine
    as any (M, N) pair tested in this script.
    """
    norm_spline_surf = build_spline_surf(M_MAX, M_MAX)
    norm_rz_surf = norm_spline_surf.to_RZFourier(
        collocation="exact",
        nu=2 * M_MAX + 1,
        nv=2 * M_MAX * norm_spline_surf.nfp + 2,
        spec_cond=None,
    )
    return norm_rz_surf.minor_radius()


if __name__ == "__main__":
    M_MAX = 32

    # the spline's own declared (M, N) has zero effect on its actual
    # geometry (see build_spline_surf's docstring) -- so the ground-truth
    # reference point grid is built once, from any one instance, and
    # reused for every (M, N) pair below
    reference_surf = build_spline_surf(M_MAX, M_MAX)
    nfp = reference_surf.nfp

    minor_radius = compute_reference_minor_radius(M_MAX)
    print(
        f"minor radius (from M=N={M_MAX} Fourier surface): "
        f"{minor_radius:.6e}"
    )

    n_cross_sections = 8
    n_theta_ref = 200
    phi_1d = np.linspace(0, 2 * np.pi, n_cross_sections, endpoint=False)
    reference = build_exact_shape_reference(
        reference_surf, phi_1d, ntheta=n_theta_ref
    )

    # full square (M, N) sweep: M and N each independently from 4 to 32
    # in steps of 2
    M_MIN = 4
    M_values = list(range(M_MIN, M_MAX + 1, 2))
    N_values = list(range(M_MIN, M_MAX + 1, 2))
    mn_pairs = [(M, N) for M in M_values for N in N_values]

    max_error = np.empty((len(M_values), len(N_values)))
    rms_error = np.empty((len(M_values), len(N_values)))

    for M, N in mn_pairs:
        spline_surf = build_spline_surf(M, N)
        # fresh "exact" (uncondensed) Fourier transform at exactly this
        # (M, N) -- not a truncation of a higher-resolution one -- with
        # its own Nyquist-adequate (nu, nv) collocation grid
        rz_surf = spline_surf.to_RZFourier(
            collocation="exact",
            nu=2 * M + 1,
            # ft() requires nv to be even -- +2 (not +1) keeps it above
            # the 2*N*nfp+1 Nyquist threshold while staying even.
            nv=2 * N * nfp + 2,
            spec_cond=None,
        )
        errors = exact_shape_error(rz_surf, reference) / minor_radius
        i = M_values.index(M)
        j = N_values.index(N)
        max_error[i, j] = np.max(errors)
        rms_error[i, j] = np.sqrt(np.mean(errors**2))
        print(
            f"(M={M:2d}, N={N:2d}): max error = {max_error[i, j]:.3e}, "
            f"rms error = {rms_error[i, j]:.3e}"
        )

    ########################
    # plotting

    rcParams.update(
        {
            "font.size": 8,
            "text.usetex": True,
            "font.weight": "book",
            "font.family": "Futura PT",
            "lines.linewidth": 1,
            "axes.linewidth": 0.5,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
        }
    )
    paper_width = 174 / 25.4  # full paper width, in inches

    fig, axes = plt.subplots(
        1, 2, figsize=(paper_width, paper_width * 0.5), dpi=600
    )
    fig.subplots_adjust(wspace=0.6)

    for ax, data, title in zip(
        axes,
        (max_error, rms_error),
        ("Max normalized error", "RMS normalized error"),
    ):
        norm = LogNorm(vmin=data.min(), vmax=data.max())
        im = ax.imshow(
            data, origin="lower", aspect="equal", cmap="viridis", norm=norm
        )
        ax.set_xticks(range(len(N_values)))
        ax.set_xticklabels(N_values, rotation=90)
        ax.set_yticks(range(len(M_values)))
        ax.set_yticklabels(M_values)
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"$M$")
        ax.set_title(title)
        fig.colorbar(
            im, ax=ax, label="Normalized shape error",
            fraction=0.046, pad=0.04,
        )

    plt.savefig(
        dpi=600,
        fname="shape_error_heatmap_MN.pdf",
        bbox_inches="tight",
    )
    plt.show()
