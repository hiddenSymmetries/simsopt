from simsopt.geo.surfacespline import SurfaceBSpline
import numpy as np
from simsopt.mhd import Vmec
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve
from simsopt import make_optimizable
import matplotlib.pyplot as plt
import time
import matplotlib
matplotlib.use('QtAgg')

import numpy as np 
from simsopt._core.finite_difference import FiniteDifference, MPIFiniteDifference
from simsopt._core import Optimizable
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.util.mpi import MpiPartition, proc0_print


def print_dofs_nicely(surf):
    dofs_lb_ub = list(zip(surf.x, surf.lower_bounds, surf.upper_bounds))
    dofs_dict = dict(zip(surf.dof_names, dofs_lb_ub))
    print("{:<30} {:<20} {:<20} {:<20}".format('dof','value','lower bound', 'upper bound'))
    for k, v in dofs_dict.items():
        val, lb, ub = v
        print("{:<30} {:<20} {:<20} {:<20}".format(k, val, lb, ub))

   
    
if __name__ == "__main__":

    dofs = np.array(
        [ 4.79206134e-01,  1.71881223e-01,  3.20635390e-01,  1.99000498e+00,
        4.03564462e-01,  3.53192851e-01,  3.43185836e-01,  1.54871144e-01,
        2.59455795e-02,  2.35596967e+00,  2.99686851e+00,  5.39410080e+00,
        1.99338737e-01,  4.77019391e-01,  3.51276052e-01,  2.79138429e-01,
       -1.68598284e-01,  2.32130343e+00,  2.92408397e+00,  5.49772295e+00,
        1.92640939e-03,  5.44511407e-01,  2.88717763e-01,  4.70683902e-01,
       -2.71822099e-01,  1.96275122e+00,  3.10365083e+00,  5.33613628e+00,
        7.15609865e-03,  6.08880965e-01,  2.34785101e-01,  1.20149277e+00,
        9.97700863e-01,  1.59999593e+00, -3.20033234e-01]
    )
    mpi = MpiPartition(1)
    mpi.write()

    spline_kwargs = {
        'axis_points':3,
        'points_per_cs':4,
        'n_cs':5,
        'nfp':2,
        'M':9,
        'N':4,
        'p_u':3,
        'p_v':3,
        'cs_equispaced': True,
        'rays_equispaced': False,
        'cs_global_angle_free':False,
        'axis_angles_fixed':True,
        'cs_basis':'polar',
        'nurbs': False,
        }

    surf = SurfaceBSpline(
            **spline_kwargs
    )

    print(f'ndofs: {surf.dof_size}')
    print(len(dofs))
    surf.set_dofs_from_vec(np.array(dofs))
    print_dofs_nicely(surf)

    surf.plot()
    plt.show()

    rz_surf = surf.to_RZFourier(
        nu=64,
        nv=64,
        nv_interp=128,
        nu_interp=128,
        collocation='arclength',
        plot=True,
        spec_cond=True,
        spec_cond_options={
        'plot':True,
        'ftol':1e-4,
        'Mtol':1.1,
        'shapetol':None,
        'niters':5000,
        'verbose':True,
        'cutoff':1e-5
        }
    )
    plt.show()
    vmec = Vmec.vmec_from_surf(
        mpi=mpi,
        nfp=surf.nfp,
        surf=rz_surf,
        ns=50, 
        ftol=1e-8,
        verbose=True
    )
    vmec.run()

    #qa
    # ar_penalty = ar_target(vmec, 6)
    # iota_penalty = np.sqrt(10)*iota_target(vmec, 0.42)
    # qs_penalty = alan_QuasisymmetryRatioResidual(vmec, np.linspace(0.02, 1, 20), 1, 0)

    # qs_s1 = np.sqrt(5)*alan_QuasisymmetryRatioResidual(vmec, np.arange(1.0, 1.1, 0.1), 1, 0)
    # iota_edge_penalty = np.sqrt(10)*(vmec.iota_edge() - 0.42)
    # iota_axis_penalty = np.sqrt(10)*(vmec.iota_axis () - 0.42)

    # residuals = np.array([ar_penalty] + [iota_edge_penalty] + [iota_axis_penalty])
    # residuals = np.concatenate((residuals, qs_penalty, qs_s1))
    # res = 0.5*np.sum(residuals**2)
    # print(vmec.mean_iota())
    # print(vmec.aspect())
    # print(np.array([res]))

    # vmec = vmec_from_surf(
    #     surf.nfp,
    #     rz_surf,
    #     ns=13
    # )
    # vmec.run()

    # #qh
    # ar_penalty = ar_target(vmec, 8)
    # # iota_penalty = np.sqrt(10)*iota_target(vmec, 0.42)
    # qs_penalty = alan_QuasisymmetryRatioResidual(vmec, np.linspace(0.02, 1, 20), 1, -1)

    # qs_s1 = np.sqrt(10)*alan_QuasisymmetryRatioResidual(vmec, np.arange(1.0, 1.1, 0.1), 1, -1)
    # iota_edge_penalty = np.sqrt(10)*(vmec.iota_edge() + 1.24)
    # iota_axis_penalty = np.sqrt(10)*(vmec.iota_axis () + 1.24)

    # residuals = np.array([ar_penalty] + [iota_edge_penalty] + [iota_axis_penalty])
    # residuals = np.concatenate((residuals, qs_penalty, qs_s1))

    # print(f'QS: {np.sum(qs_penalty**2)}')
    # print(f'iota: {vmec.mean_iota()}')
    # print(f'AR: {vmec.aspect()}')
    # print(f'residuals: {0.5*np.sum(residuals**2)}')

    # print(f'residuals: {np.sum(residuals**2)}')
    # return residuals


