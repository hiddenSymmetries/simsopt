from simsopt.geo import SurfaceBSpline
import numpy as np
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition
import fnmatch
import matplotlib.pyplot as plt

mpi = MpiPartition()
mpi.write()

def write_doflist_maxlist_minlist(
        spline_kwargs
    ):

    template_surf = SurfaceBSpline(
        **spline_kwargs
    )
    template_surf.axis.fix('r_axis_0')
    # known_optimum = np.array([ 0.5023717 ,  0.19458653,  0.50599969,  1.73906393,  0.46845626,
    #     0.16181296,  0.47729433,  0.18940503, -0.02458537,  2.06682836,
    #     3.00162033,  5.04692768,  0.52305608,  0.27774597,  0.45964695,
    #     0.25508091, -0.12494219,  2.2341014 ,  2.83891472,  4.98002593,
    #     0.463996  ,  0.39733259,  0.29569942,  0.28507657, -0.55297321,
    #     2.12714243,  2.89056155,  5.37017002,  0.30340924,  0.55441033,
    #     0.23038154,  0.55127744, -0.77153937,  1.74519114,  3.25082506,
    #     5.27762822,  0.05223678,  0.58122454,  0.18346383,  1.1875926 ,
    #     1.30337977,  1.78981399, -0.27279951])

    # ub = np.maximum(known_optimum * 1.1, known_optimum * 0.9)
    # lb = np.minimum(known_optimum * 1.1, known_optimum * 0.9)
    # doflist = None
    doflist = template_surf.dof_names
    # lb = template_surf.lower_bounds
    # ub = template_surf.upper_bounds
    
    lb = np.copy(template_surf.lower_bounds)
    ub = np.copy(template_surf.upper_bounds)

    # cs_r_indices = [fnmatch.fnmatch(dof, 'CrossSectionFixedZeta*r*') for dof in template_surf.dof_names]
    # r_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*r_axis*') for dof in template_surf.dof_names]
    # z_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*z_axis*') for dof in template_surf.dof_names]
    
    # lb[cs_r_indices] = 0.1
    # ub[cs_r_indices] = 0.6

    # lb[r_axis_indices] = 0.7
    # ub[r_axis_indices] = 1.2

    # lb[z_axis_indices] = -0.5
    # ub[z_axis_indices] = 0.5

    # lb_dict = dict(zip(doflist, lb))
    # ub_dict = dict(zip(doflist, ub))

    # # print(f'lb_dict: {lb_dict}')
    # # print(f'ub_dict: {ub_dict}')

    return doflist, ub, lb

def to_unit_cube(x, lb, ub):
    """Project to [0, 1]^d from hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1 #and x.ndim == 2
    print(x.shape)
    xx = (x - lb) / (ub - lb)
    return xx

def from_unit_cube(x, lb, ub):
    """Project from [0, 1]^d to hypercube with bounds lb and ub"""
    assert np.all(lb < ub) and lb.ndim == 1 and ub.ndim == 1, f'lb: {lb}, ub: {ub}' #and x.ndim == 2
    xx = x * (ub - lb) + lb
    return xx

def is_valid_vmec(vmec):
    try:
        vmec.need_to_run_code = True
        vmec.run()
    except:
        return False
    return True
