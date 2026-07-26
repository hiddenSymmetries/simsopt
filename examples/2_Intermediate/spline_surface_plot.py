#!/usr/bin/env python

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition

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
        "points_per_cs": 6,
        "n_cs": 4,
        "nfp": 2,
        "M": 9,
        "N": 4,
        "p_u": 3,
        "p_v": 3,
        "cs_equispaced": True,
        "rays_equispaced": False,
        "cs_global_angle_free": False,
        "axis_angles_fixed": True,
        "cs_basis": "polar",
        "nurbs": False,
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    spline_surf = SurfaceBSpline(
        default_r=0.01,
        **spline_kwargs,
    )
    # spline_surf.axis.fix("r_axis_0")
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
            2.0000000000000001e-01,
            9.9597390897503890e-01,
            7.2354447285001011e-01,
            1.8373680489429420e-01,
            7.4080008889747889e-02,
            1.7277373578408293e00,
            5.6004016192311468e-01,
            5.6561991144567270e-01,
            9.3835239823039318e-02,
            7.2862642717010773e-01,
            9.4236825976400629e-01,
            4.3392422557339869e-02,
            6.7242520755879132e-01,
            2.4060791218275561e00,
            3.2071623862510825e00,
            4.1815655401253533e00,
            4.7615560684814104e00,
            6.0452073638196380e-01,
            7.5036420449916888e-02,
            2.3349181975491382e-01,
            9.5785571063713104e-01,
            4.6084243568794853e-01,
            2.8924641255692163e-01,
            1.0321246418536854e00,
            2.4722677398833128e00,
            2.7018030544678071e00,
            4.7120679980911344e00,
            5.4712259566603905e00,
            3.6596217549786203e-01,
            6.5196101409231599e-01,
            3.1293304513700815e-01,
            5.6265042163640422e-01,
            1.0003151991732455e00,
            2.2827030067980010e00,
            4.6092815759096667e-01,
            1.3160135525818673e00,
            2.2932334941128407e00,
            -8.0360753830415255e-01,
        ]
    )
    print(len(new_x))
    # spline_surf.set_dofs_from_vec(new_x)
    spline_surf.x = new_x
    # print_dofs_nicely(spline_surf)

    spline_surf.plot()
    plt.show()

    rz_surf = spline_surf.to_RZFourier(
        nu=64,
        nv=64,
        nv_interp=128,
        nu_interp=128,
        collocation="arclength",
        plot=True,
        spec_cond=None,
        spec_cond_options={
            "plot": False,
            "ftol": 1e-4,
            "Mtol": 1.1,
            "shapetol": None,
            "niters": 2000,
            "verbose": True,
            "cutoff": 1e-5,
        },
    )

    # condensed, data = rz_surf.condense_spectrum(method='trf', Fourier_continuation=True)
    # plot_spectral_condensation(rz_surf, condensed, data)

    plt.show()

    vmec = Vmec.vmec_from_surf(
        nfp=rz_surf.nfp, surf=rz_surf, mpi=mpi, ns=13, M=12, N=12, ftol=1e-7
    )
    vmec.run()
    print(f"Volume: {vmec.volume()}")
