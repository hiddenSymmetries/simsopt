#!/usr/bin/env python
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve
import os
import numpy as np

"""
Optimize for quasi-helical symmetry (M=1, N=1) at a given radius.
"""

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
mpi = MpiPartition(25)

vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start'), mpi=mpi)

weight_function2 = lambda s: np.exp(-s**2/0.1**2)
print(vmec.iota_weighted(weight_function2))
print(vmec.iota_axis())
weight_function1 = lambda s: np.exp(-(s-1)**2/0.1**2)
print(vmec.iota_weighted(weight_function1))
print(vmec.iota_edge())
target_function = lambda s: 0.68
print(vmec.iota_target(target_function))
target_function = lambda s: 0.68*s
print(vmec.iota_target(target_function))
print(vmec.well_weighted(weight_function1,weight_function2))
vmec.d_iota_target(target_function,1e-2)
