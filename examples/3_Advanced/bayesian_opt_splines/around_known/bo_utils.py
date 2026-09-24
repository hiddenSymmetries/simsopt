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

    known_optimum = np.array([ 0.3016216 ,  0.20890161,  0.44766483,  0.49065546,  0.31258963,
        2.72104182,  0.13272592,  0.01054766,  0.55550307,  0.34290225,
        0.24441989,  0.54088171,  1.42225604,  2.49345601,  3.26726428,
        3.90173828,  5.75958653,  0.03943073,  0.17513655,  0.61373303,
        0.17959677,  0.26507675,  0.72124269,  1.57072331,  2.02542396,
        3.16716533,  4.71228105,  5.49176038,  0.04404639,  0.59041262,
        0.48036493,  0.16049688,  1.39121464,  1.62414016,  1.68508211,
        2.00728408, -0.28335058])

    ub = np.maximum(known_optimum * 1.01, known_optimum * 0.99)
    lb = np.minimum(known_optimum * 1.01, known_optimum * 0.99)
    doflist = None

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
