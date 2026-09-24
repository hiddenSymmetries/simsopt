"""
Same convergence study as shape_error_vs_mode_number.py -- exact
point-to-curve shape error between the ground-truth spline surface and
its "exact" (uncondensed) Fourier transform, as a function of the
Fourier mode number M = N -- but *without* truncation: instead of
computing one high-resolution transform and slicing its coefficient
matrix down to each lower mode number, this recomputes the transform
from scratch at each M = N, each with its own Nyquist-correct (nu, nv)
collocation grid (nu = 2*M+1 poloidally, nv = 2*N*nfp+2 toroidally,
rounded up to stay even -- see shape_error_vs_mode_number.py and
shape_error_vs_quadrature_resolution.py for why the nfp factor on nv
matters).

Since ft() computes a discrete quadrature projection (not a continuous
integral), truncation and fresh recomputation are only *exactly*
equivalent in the limit of infinite quadrature resolution; comparing this
script's curve against shape_error_vs_mode_number.py's checks how close
that equivalence is in practice, at these Nyquist-adequate grids.

Errors are normalized by the average minor radius (see
compute_reference_minor_radius) to give a dimensionless shape error.
"""

import matplotlib.pyplot as plt
import numpy as np
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
    ground-truth boundary used throughout that study -- except for the
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


def compute_reference_minor_radius():
    """
    Average minor radius used to normalize shape error into a
    dimensionless quantity. SurfaceBSpline doesn't implement gammadash1,
    so `minor_radius()` (which needs it, via the cross-sectional-area
    formula) isn't available directly on the spline surface -- as a
    stand-in, this uses a fixed, very high-resolution (M = N = 32) exact
    Fourier transform of the ground-truth spline, at least as fine as any
    mode number tested in this script.
    """
    norm_spline_surf = build_spline_surf(32, 32)
    norm_rz_surf = norm_spline_surf.to_RZFourier(
        collocation="exact",
        # nu=2 * 32 + 1,
        # nv=2 * 32 * norm_spline_surf.nfp + 2,
        spec_cond=None,
    )
    return norm_rz_surf.minor_radius()


if __name__ == "__main__":
    mode_numbers = np.arange(3, 33)  # M = N = 3, 4, ..., 32

    # the spline's own declared (M, N) has zero effect on its actual
    # geometry (see build_spline_surf's docstring) -- so the ground-truth
    # reference point grid is built once, from any one instance, and
    # reused for every M = N below
    reference_surf = build_spline_surf(mode_numbers[-1], mode_numbers[-1])
    n_cross_sections = 8
    n_theta_ref = 200
    phi_1d = np.linspace(0, 2 * np.pi, n_cross_sections, endpoint=False)
    reference = build_exact_shape_reference(
        reference_surf, phi_1d, ntheta=n_theta_ref
    )

    minor_radius = compute_reference_minor_radius()
    print(f"minor radius (from M=N=32 Fourier surface): {minor_radius:.6e}")

    max_error = np.empty_like(mode_numbers, dtype=float)
    rms_error = np.empty_like(mode_numbers, dtype=float)

    for i, m in enumerate(mode_numbers):
        spline_surf = build_spline_surf(m, m)
        # fresh "exact" (uncondensed) Fourier transform at exactly this
        # (M, N) = (m, m) -- not a truncation of a higher-resolution one
        rz_surf = spline_surf.to_RZFourier(
            collocation="exact",
            nu=2 * m + 1,
            # ft() requires nv to be even -- +2 (not +1) keeps it above
            # the 2*N*nfp+1 Nyquist threshold while staying even.
            nv=2 * m * spline_surf.nfp + 2,
            spec_cond=None,
        )
        errors = exact_shape_error(rz_surf, reference) / minor_radius
        max_error[i] = np.max(errors)
        rms_error[i] = np.sqrt(np.mean(errors**2))
        print(
            f"M=N={m:2d}: max error = {max_error[i]:.3e}, "
            f"rms error = {rms_error[i]:.3e}"
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
    paper_width = 174 / 25.4 / 2  # half-width figure, in inches

    fig, ax = plt.subplots(figsize=(paper_width, paper_width * 0.8), dpi=600)
    ax.semilogy(mode_numbers, max_error, "o-", markersize=3, label="Max error")
    ax.semilogy(mode_numbers, rms_error, "s-", markersize=3, label="RMS error")
    ax.set_xlabel("Fourier mode number $M = N$")
    ax.set_ylabel("Normalized shape error")
    ax.legend(frameon=False)

    plt.savefig(
        dpi=600,
        fname="shape_error_vs_mode_number_recomputed.pdf",
        bbox_inches="tight",
    )
    plt.show()
