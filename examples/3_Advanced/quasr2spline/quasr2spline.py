#!/usr/bin/env python

import numpy as np 
from simsopt.geo import surfacespline
from simsopt.mhd import Vmec
from simsopt.objectives.polygonal_shape_errors import pointwise_minimum_poly_distance
from simsopt.util.mpi import MpiPartition, proc0_print
from simsopt.solve.mpi import least_squares_mpi_solve
from simsopt.objectives.least_squares import LeastSquaresProblem
import matplotlib.pyplot as plt
from simsopt._core import Optimizable
from simsopt.geo.surfacespline import SurfaceBSpline
from simsopt.geo import SurfaceRZFourier
from mpi4py import MPI
import argparse
import fnmatch

mpi = MpiPartition()
mpi.write()

def write_doflist_maxlist_minlist(
        spline_kwargs
    ):

    template_surf = SurfaceBSpline(
        **spline_kwargs
    )

    doflist = template_surf.dof_names
    # lb = template_surf.lower_bounds
    # ub = template_surf.upper_bounds
    
    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    cs_theta_indices = [fnmatch.fnmatch(dof, 'CrossSectionFixedZeta*r*') for dof in template_surf.dof_names]
    # r_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*r_axis*') for dof in template_surf.dof_names]
    # z_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*z_axis*') for dof in template_surf.dof_names]
    
    lb[cs_theta_indices] = 0
    ub[cs_theta_indices] = 2*np.pi

    #cs_theta_indices = [fnmatch.fnmatch(dof, 'CrossSectionFixedZeta0*r*') for dof in template_surf.dof_names] + 
    

    #lb[r_axis_indices] = 0.7
    #ub[r_axis_indices] = 1.23

    #lb[z_axis_indices] = -0.6
    #ub[z_axis_indices] = 0.6

    #lb_dict = dict(zip(doflist, lb))
    #ub_dict = dict(zip(doflist, ub))

    #print(f'lb_dict: {lb_dict}')
    #print(f'ub_dict: {ub_dict}')

    return doflist, ub, lb

class PASOpt(Optimizable):
    def __init__(
            self,
            surf,
            spline_kwargs,
            target_surf: SurfaceRZFourier,
            nu,
            nv            
        ):
        self.surf = surf
        self.spline_kwargs = spline_kwargs
        self.nu = nu
        self.nv = nv
        self.target_surf = target_surf
        Optimizable.__init__(self, depends_on=[surf])

    def J_se(self):
        new_surf = SurfaceBSpline(
            **self.spline_kwargs
        )
        new_surf.set_dofs_from_vec(self.x)
        rzsurf = new_surf.to_RZFourier(
            collocation='arclength',
            plot=False,
            spec_cond=True,
            spec_cond_options={
            'plot':False,
            'ftol':1e-4,
            'Mtol':1.1,
            'shapetol':None,
            'niters':5000,
            'verbose':False,
            'cutoff':1e-6
            }
        )
        shape_error = pointwise_minimum_poly_distance(
                rzsurf,
                target_surf,
                nu = 64,
                nv = 64     
            )
        return shape_error #np.append(shape_error, vol_error)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, help = 'string of vmec input file')
    input_args = vars(parser.parse_args())
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    input_vmec = Vmec(input_args['fname'])
    print(f'Solving from file {input_vmec}')

    target_surf = input_vmec.boundary

    # target_surf.plot()

    spline_kwargs =  {
        'axis_points': 3,
        'points_per_cs': 6,
        'n_cs': 5,
        'nfp': 3,
        'M': 9,
        'N': 4,
        'p_u': 3,
        'p_v': 3, 
        'cs_equispaced': True,
        'rays_equispaced': False,
        'cs_global_angle_free': False,
        'axis_angles_fixed': True,
        'cs_basis': 'polar',
        'nurbs':False
    }

    _, ub, lb = write_doflist_maxlist_minlist(spline_kwargs)

    spline_surf = SurfaceBSpline(
        default_r = 0.05,
        **spline_kwargs,
    )

    myopt = PASOpt(
        surf=spline_surf,
        spline_kwargs=spline_kwargs,
        target_surf=target_surf,
        nu=32,
        nv=32     
    )

    myopt.upper_bounds = lb
    myopt.lower_bounds = ub

    prob = LeastSquaresProblem.from_tuples([
        (myopt.J_se, 0, 1)
    ])
    # prob.bounds = (spline_surf.lower_bounds, spline_surf.upper_bounds)
    proc0_print(f'prob.bounds: {prob.bounds}')
    proc0_print(f'initial: {repr(spline_surf.x)}, ndofs = {len(spline_surf.x)}')

    default_optimizer_args = {
        'abs_step':1e-4,
        'rel_step':1e-8,
        'diff_method':'centered',
    }

    least_squares_mpi_solve(prob, mpi=mpi, grad=True, **default_optimizer_args)

    if rank == 0:
        print(repr(myopt.x))
        fig, ax, = plt.subplots(subplot_kw={"projection": "3d"})
        target_surf.plot(ax=ax)
        new_surf = SurfaceBSpline(
            **myopt.spline_kwargs
        )
        new_surf.set_dofs_from_vec(myopt.x)
        new_surf.plot(ax=ax)
        plt.show()
    


   