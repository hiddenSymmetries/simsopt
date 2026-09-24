#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition
from simsopt.util.spline_helpers import print_dofs_nicely

matplotlib.use("qtagg")
mpi = MpiPartition()
mpi.write()


def boundary_poincare_plot(
    ax,
    rbc,
    zbs,
    phi,
    N,
    M,
    nfp,
    plotting_kwargs,
    scatter=False,
    ntheta=200,
):
    xn = np.arange(-N, N + 1, 1)
    xm = np.arange(0, M + 1, 1)
    theta = np.linspace(0, 2 * np.pi, num=ntheta)

    R = np.zeros((ntheta, 1))
    Z = np.zeros((ntheta, 1))

    for i in range(rbc.shape[0]):
        for j in range(rbc.shape[1]):
            if rbc[i, j] != 0 or zbs[i, j] != 0:
                angle = xm[j] * theta - xn[i] * phi * nfp
                R = R + rbc[i, j] * np.cos(angle)  # /(np.abs(i) + np.abs(j))
                Z = Z + zbs[i, j] * np.sin(angle)  # /(np.abs(i) + np.abs(j))
    if not scatter:
        ax.plot(R.flatten(), Z.flatten(), **plotting_kwargs)
    else:
        ax.scatter(R.flatten(), Z.flatten(), **plotting_kwargs)


def write_doflist_maxlist_minlist(spline_kwargs):

    template_surf = SurfaceBSpline(**spline_kwargs)
    template_surf.axis.fix("r_axis_0")

    doflist = template_surf.dof_names

    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    return doflist, ub, lb


if __name__ == "__main__":
    # target_surf.plot()

    # spline_kwargs = {
    #     "axis_points": 3,
    #     "points_per_cs": 6,
    #     "n_cs": 4,
    #     "nfp": 2,
    #     "M": 12,
    #     "N": 12,
    #     "p_u": 3,
    #     "p_v": 3,
    #     "cs_equispaced": True,
    #     "rays_equispaced": False,
    #     "cs_global_angle_free": False,
    #     "axis_angles_fixed": True,
    #     "cs_basis": "polar",
    #     "nurbs": False,
    #     "knot_parametrization": "chord",
    # }

    spline_kwargs = {
        "axis_points": 3,
        "points_per_cs": 4,
        "n_cs": 5,
        "nfp": 2,
        "M": 5,
        "N": 4,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": False,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": False,
        "cs_basis": "polar",
        "nurbs": False,
        "use_bishop_frame": True,
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    spline_surf = SurfaceBSpline(
        default_r=0.3,
        **spline_kwargs,
    )
    spline_surf.axis.fix("r_axis_0")
    # new_x = np.array(
    #     [
    #         0.3016216,
    #         0.20890161,
    #         0.44766483,
    #         0.49065546,
    #         0.31258963,
    #         2.72104182,
    #         0.13272592,
    #         0.01054766,
    #         0.55550307,
    #         0.34290225,
    #         0.24441989,
    #         0.54088171,
    #         1.42225604,
    #         2.49345601,
    #         3.26726428,
    #         3.90173828,
    #         5.75958653,
    #         0.03943073,
    #         0.17513655,
    #         0.61373303,
    #         0.17959677,
    #         0.26507675,
    #         0.72124269,
    #         1.57072331,
    #         2.02542396,
    #         3.16716533,
    #         4.71228105,
    #         5.49176038,
    #         0.04404639,
    #         0.59041262,
    #         0.48036493,
    #         0.16049688,
    #         1.39121464,
    #         1.62414016,
    #         1.68508211,
    #         2.00728408,
    #         -0.28335058,
    #     ]
    # )
    new_x = np.array(
        [
            3.49151814e-05,
            4.08858863e-01,
            1.56187918e-01,
            1.35325832e00,
            3.03304348e-02,
            3.29309487e-01,
            1.48040786e-01,
            4.07487352e-01,
            1.12945811e00,
            3.45660993e00,
            4.70892314e00,
            1.15714960e-01,
            2.23743388e-01,
            1.71422619e-01,
            3.38760347e-01,
            7.89176573e-01,
            3.72275569e00,
            4.44881220e00,
            1.68678313e-01,
            1.41421904e-01,
            2.44246896e-01,
            2.07282078e-01,
            7.96088440e-01,
            3.44958011e00,
            4.29717483e00,
            1.91544620e-01,
            1.15674640e-01,
            2.65035209e-01,
            1.61074550e00,
            7.37716609e-01,
            5.62288940e-01,
            2.69764752e-01,
            7.31087514e-01,
            3.67273919e-01,
            7.79087610e-01,
            1.20291494e00,
        ]
    )
    print(len(new_x))
    # spline_surf.set_dofs_from_vec(new_x)
    spline_surf.x = new_x
    print_dofs_nicely(spline_surf)

    spline_surf.plot()
    plt.show()

    rz_surf = spline_surf.to_RZFourier(
        # nu=64,
        # nv=64,
        # nv_interp=128,
        # nu_interp=128,
        # collocation="arclength",
        plot=True,
        # spec_cond=None,
        # spec_cond_options={
        #     "plot": False,
        #     "ftol": 1e-4,
        #     "Mtol": 1.1,
        #     "shapetol": None,
        #     "niters": 2000,
        #     "verbose": True,
        #     "cutoff": 1e-5,
        # },
    )

    # condensed, data = rz_surf.condense_spectrum(method='trf', Fourier_continuation=True)
    # plot_spectral_condensation(rz_surf, condensed, data)

    plt.show()

    vmec = Vmec.vmec_from_surf(
        nfp=rz_surf.nfp, surf=rz_surf, mpi=mpi, ns=13, M=12, N=12, ftol=1e-7
    )
    vmec.run()
    print(f"Volume: {vmec.volume()}")
