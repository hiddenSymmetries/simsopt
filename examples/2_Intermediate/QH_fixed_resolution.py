#!/usr/bin/env python

from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve
import os

"""
Optimize for quasi-helical symmetry (M=1, N=1) at a given radius.
"""

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
mpi = MpiPartition(25)

vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start'), mpi=mpi)

# Define parameter space:
surf = vmec.boundary
surf.all_fixed()
max_mode = 2
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.set_fixed("rc(0,0)")  # Major radius

# Configure quasisymmetry objective:
qs = Quasisymmetry(Boozer(vmec),
                   0.5,  # Radius to target
                   1, 1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem([(vmec.aspect, 7, 1),
                            (qs, 0, 1)],
                           rel_step=1e-3, abs_step=1e-5)

# To keep this example fast, we stop after the first function
# evaluation. For a "real" optimization, remove the max_nfev
# parameter.
least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=1)
