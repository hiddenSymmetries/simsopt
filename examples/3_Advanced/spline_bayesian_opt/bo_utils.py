from simsopt.geo import SurfaceBSpline
import numpy as np
from simsopt.mhd import Vmec
from simsopt.util.mpi import MpiPartition
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

    cs_r_indices = [fnmatch.fnmatch(dof, 'CrossSectionFixedZeta*r*') for dof in template_surf.dof_names]
    r_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*r_axis*') for dof in template_surf.dof_names]
    z_axis_indices = [fnmatch.fnmatch(dof, 'PseudoAxis*z_axis*') for dof in template_surf.dof_names]
    
    lb[cs_r_indices] = 0.1
    ub[cs_r_indices] = 0.6

    lb[r_axis_indices] = 0.7
    ub[r_axis_indices] = 1.2

    lb[z_axis_indices] = -0.5
    ub[z_axis_indices] = 0.5

    lb_dict = dict(zip(doflist, lb))
    ub_dict = dict(zip(doflist, ub))

    # print(f'lb_dict: {lb_dict}')
    # print(f'ub_dict: {ub_dict}')

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

def vmec_from_dofs_dict(
        new_dof_dict,
        spline_kwargs
    ):
    '''
    Generate VMEC input file given a dictionary with boundary modes
    '''
    surf = SurfaceBSpline(
        **spline_kwargs
    )
    #print(surf.dof_names)
    # x = [new_dof_dict[key] for key in surf.dof_names]
    # print(new_dof_dict.values())
    x = np.fromiter(new_dof_dict.values(), dtype=np.float64)
    surf.x = np.array(x)
    # runtime params
    vmec = Vmec.vmec_from_surf(surf.nfp, surf)
    return vmec

def is_valid_vmec(vmec):
    try:
        vmec.need_to_run_code = True
        vmec.run()
    except:
        return False
    return True
