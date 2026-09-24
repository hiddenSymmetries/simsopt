"""
How the exact shape error between the ground-truth spline surface and its
"exact" (uncondensed) Fourier transform depends on the (nu, nv) collocation
grid used by SurfaceBSpline.ft() to *compute* that transform -- at FIXED
Fourier mode resolution (M = N = 12), unlike
shape_error_vs_mode_number.py (which fixes the collocation grid and varies
the truncation mode number instead).

ft() projects onto the (M, N) Fourier basis via a discrete quadrature sum
over an (nu, nv) grid in (theta, zeta), not a continuous integral. That
sum only reproduces the true Fourier coefficients if the sampled signal
(R, Z from the spline's gamma_impl) has no content above the grid's
Nyquist limit -- but a NURBS/B-spline surface has broadband content, so
too coarse a grid aliases high-frequency content back into every
coefficient, including low-order ones. The relevant Nyquist thresholds
are nu >= 2*M + 1 in the poloidal direction, and -- because the toroidal
collocation angle spans the *full* torus while ft()'s cos/sin argument is
n*(nfp*zeta) -- nv >= 2*N*nfp + 1 in the toroidal direction (the physical
frequency being sampled there is n*nfp, not n). This script sweeps nu and
nv independently over a 2D grid, at fixed M = N = 12, to show that
dependence directly, including the asymmetry between the two thresholds.

Errors are normalized by the average minor radius (see
compute_reference_minor_radius) to give a dimensionless shape error.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.pyplot import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable

from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.objectives.shape_errors import (
    build_exact_shape_reference,
    exact_shape_error,
)

MPOL=12
NTOR=8

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


def compute_reference_minor_radius():
    """
    Average minor radius used to normalize shape error into a
    dimensionless quantity. SurfaceBSpline doesn't implement gammadash1,
    so `minor_radius()` (which needs it, via the cross-sectional-area
    formula) isn't available directly on the spline surface -- as a
    stand-in, this uses a fixed, very high-resolution (M = N = 32) exact
    Fourier transform of the ground-truth spline, well beyond the M = N =
    12 resolution used for the (nu, nv) sweep below.
    """
    norm_spline_surf = build_spline_surf(32, 32)
    norm_rz_surf = norm_spline_surf.to_RZFourier(
        collocation="exact",
        nu=2 * 32 + 1,
        nv=2 * 32 * norm_spline_surf.nfp + 2,
        spec_cond=None,
    )
    return norm_rz_surf.minor_radius()


if __name__ == "__main__":
    spline_surf = build_spline_surf(MPOL, NTOR)
    M, N, nfp = spline_surf.M, spline_surf.N, spline_surf.nfp

    # Nyquist thresholds: nu resolves poloidal frequency up to M; nv
    # resolves toroidal frequency up to N*nfp (the cos/sin argument is
    # n*(nfp*zeta), and zeta spans the full torus in ft()'s "exact"
    # collocation).
    nu_nyquist = 2 * M + 1
    nv_nyquist = 2 * N * nfp + 1

    # fixed reference point grid on the ground-truth spline surface,
    # built once and reused for every (nu, nv) pair below
    n_cross_sections = 8
    n_theta_ref = 200
    phi_1d = np.linspace(0, 2 * np.pi, n_cross_sections, endpoint=False)
    reference = build_exact_shape_reference(
        spline_surf, phi_1d, ntheta=n_theta_ref
    )

    minor_radius = compute_reference_minor_radius()
    print(f"minor radius (from M=N=32 Fourier surface): {minor_radius:.6e}")

    # Densely (log-spaced) sampled so the heatmap below is a fine-grained
    # grid rather than a handful of blown-up blocks -- each cell is one
    # actual (nu, nv) sample, so a coarse sweep looks "blocky" no matter
    # how it's rendered.
    nu_values = np.unique(np.round(np.geomspace(6, 150, 24)).astype(int))
    nu_values = np.sort(np.unique(np.append(nu_values, nu_nyquist)))

    # ft() asserts nv must be even -- round every sample (and the
    # toroidal Nyquist threshold) up to the nearest even value.
    nv_nyquist_even = nv_nyquist + (nv_nyquist % 2)
    nv_values = np.round(np.geomspace(6, 220, 24)).astype(int)
    nv_values = nv_values + (nv_values % 2)
    nv_values = np.sort(np.unique(np.append(nv_values, nv_nyquist_even)))

    max_error = np.empty((len(nv_values), len(nu_values)))
    rms_error = np.empty((len(nv_values), len(nu_values)))

    for j, nv in enumerate(nv_values):
        for i, nu in enumerate(nu_values):
            rz_surf = spline_surf.to_RZFourier(
                collocation="exact",
                nu=int(nu),
                nv=int(nv),
                spec_cond=None,
            )
            errors = exact_shape_error(rz_surf, reference) / minor_radius
            max_error[j, i] = np.max(errors)
            rms_error[j, i] = np.sqrt(np.mean(errors**2))
        print(f"nv={nv:4d} done")

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

    # each heatmap cell is one categorical (index-spaced) sample, but the
    # *value* range swept in nv is larger than in nu (nv_values spans up
    # to ~220, nu_values only up to ~150) -- reflect that in the plotted
    # aspect ratio (height/width per cell) rather than forcing a square,
    # so the panels come out as tall rectangles.
    aspect_ratio = (nv_values.max() - nv_values.min()) / (
        nu_values.max() - nu_values.min()
    )

    fig, (ax_max, ax_rms) = plt.subplots(
        1, 2, figsize=(paper_width * 0.65, paper_width * 0.6), dpi=600
    )
    fig.subplots_adjust(wspace=0.3)

    norm = LogNorm(
        vmin=min(max_error.min(), rms_error.min()),
        vmax=max(max_error.max(), rms_error.max()),
    )

    for ax, data, title in zip(
        (ax_max, ax_rms),
        (max_error, rms_error),
        ("Max normalized error", "RMS normalized error"),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            aspect=aspect_ratio,
            norm=norm,
            cmap="viridis",
        )
        ax.set_xticks(range(len(nu_values))[::4])
        ax.set_xticklabels(nu_values[::4], rotation=90)
        ax.set_yticks(range(len(nv_values))[::4])
        ax.set_yticklabels(nv_values[::4])
        ax.set_xlabel(r"$n_u$")
        ax.set_ylabel(r"$n_v$")
        ax.set_title(title)

        # Nyquist thresholds: nu >= 2M+1 (poloidal), nv >= 2*N*nfp+1
        # (toroidal) -- drawn at the matching tick's categorical index,
        # since both threshold values are included exactly in
        # nu_values/nv_values above.
        ax.axvline(
            np.where(nu_values == nu_nyquist)[0][0],
            color="w",
            linestyle="--",
            linewidth=0.7,
        )
        ax.axhline(
            np.where(nv_values == nv_nyquist_even)[0][0],
            color="w",
            linestyle="--",
            linewidth=0.7,
        )

    # Carving a colorbar axis out of ax_rms via a divider (rather than
    # fig.colorbar(im, ax=..., ...)) makes the colorbar's height match
    # ax_rms's actual (aspect-locked) box height -- but doing that to
    # ax_rms alone would also shrink its available width relative to
    # ax_max (which keeps its full column), making the two panels
    # different sizes. So an equal-sized *invisible* spacer is carved
    # from ax_max too, purely to keep both panels' available width (and
    # hence their final aspect-locked box size) the same.
    spacer = make_axes_locatable(ax_max).append_axes(
        "right", size="5%", pad=0.15
    )
    spacer.set_axis_off()
    cax = make_axes_locatable(ax_rms).append_axes(
        "right", size="5%", pad=0.15
    )
    fig.colorbar(im, cax=cax, label="Normalized shape error")

    plt.savefig(
        dpi=600,
        fname="shape_error_vs_quadrature_resolution.pdf",
        bbox_inches="tight",
    )
    #plt.show()
