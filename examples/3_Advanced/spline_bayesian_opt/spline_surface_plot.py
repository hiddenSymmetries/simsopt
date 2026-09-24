#!/usr/bin/env python

import numpy as np 
from simsopt.geo import surfacespline
from simsopt.mhd import Vmec
from simsopt.objectives.polygonal_shape_errors import pointwise_minimum_poly_distance, jaccard_index
from simsopt.util.spline_helpers import print_dofs_nicely
from simsopt.util.mpi import MpiPartition, proc0_print
from simsopt.solve.mpi import least_squares_mpi_solve
from simsopt.objectives.least_squares import LeastSquaresProblem
import matplotlib.pyplot as plt
from simsopt._core import Optimizable
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.geo import SurfaceRZFourier
from mpi4py import MPI
from matplotlib.gridspec import GridSpec
import matplotlib
import fnmatch

matplotlib.use('qtagg')
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
    xn = np.arange(-N,N+1,1)
    xm = np.arange(0,M+1,1)
    theta = np.linspace(0,2*np.pi,num=ntheta)

    R = np.zeros((ntheta,1))
    Z = np.zeros((ntheta,1))

    for i in range(rbc.shape[0]):
        for j in range(rbc.shape[1]):
            if rbc[i,j] !=0 or zbs[i,j] != 0:
                angle = xm[j]*theta - xn[i]*phi*nfp
                R = R + rbc[i,j]*np.cos(angle)#/(np.abs(i) + np.abs(j))
                Z = Z + zbs[i,j]*np.sin(angle)#/(np.abs(i) + np.abs(j))
    if not scatter:
        ax.plot(R.flatten(), Z.flatten(), **plotting_kwargs)
    else:
        ax.scatter(R.flatten(), Z.flatten(), **plotting_kwargs)

def write_doflist_maxlist_minlist(
        spline_kwargs
    ):

    template_surf = SurfaceBSpline(
        **spline_kwargs
    )
    template_surf.axis.fix('r_axis_0')

    doflist = template_surf.dof_names

    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    return doflist, ub, lb

if __name__ == "__main__":
    # target_surf.plot()

    spline_kwargs =  {
        'axis_points':3,
        'points_per_cs':4,
        'n_cs':6,
        'nfp':2,
        'M':9,
        'N':4,
        'p_u':3,
        'p_v':3,
        'cs_equispaced':True,
        'rays_equispaced':False,
        'cs_global_angle_free':False,
        'axis_angles_fixed':True,
        'cs_basis':'polar',
        'nurbs': False,
    }

    dof_list, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    spline_surf = SurfaceBSpline(
        default_r = 0.01,
        **spline_kwargs,
    )
    spline_surf.axis.fix('r_axis_0')
    np.random.seed(0)
    # new_x = np.array([ 0.5526,  0.1896,  0.5566,  1.9130,  0.4934,  0.1615,  0.4846,  0.1972,
    #     -0.0270,  2.0036,  2.9170,  4.5571,  0.4708,  0.3047,  0.4955,  0.2498,
    #     -0.1260,  2.2485,  2.8514,  4.8790,  0.4962,  0.4291,  0.3253,  0.3026,
    #     -0.6083,  2.1424,  2.8093,  5.2548,  0.2731,  0.5776,  0.2534,  0.6046,
    #     -0.7372,  1.6139,  3.5759,  5.4406,  0.0470,  0.6393,  0.2018,  1.1784,
    #     1.4337,  1.9447, -0.2466])
    # spline_surf.set_dofs_from_vec(new_x)

    print_dofs_nicely(spline_surf)

    spline_surf.plot()
    plt.show()

    rz_surf = spline_surf.to_RZFourier(
        nu=64,
        nv=64,
        nv_interp=128,
        nu_interp=128,
        collocation='arclength',
        plot=False,
        spec_cond_options={
            'plot':False,
            'ftol':1e-4,
            'Mtol':1.1,
            'shapetol':None,
            'niters':2000,
            'verbose':False,
            'cutoff':1e-5
        }
    )

    vmec = Vmec.vmec_from_surf(
        nfp=rz_surf.nfp,
        surf=rz_surf,
        mpi=mpi,
        ns=13,
        M=12, 
        N=12,
        ftol=1e-7
    )
    vmec.run()
