#!/usr/bin/env python

import os
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

"""
Optimize for quasi-helical symmetry (M=1, N=1) at a given radius.
"""

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
proc0_print("Running 2_Intermediate/QH_fixed_resolution_boozer.py")
proc0_print("====================================================")

mpi = MpiPartition(25)

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
max_mode = 2
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

# Configure quasisymmetry objective:
qs = Quasisymmetry(Boozer(vmec),
                   0.5,  # Radius to target
                   1, 1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
                                        (qs.J, 0, 1)])

# Make sure all procs run vmec:
qs_initial = qs.J()

proc0_print(f"Quasisymmetry objective before optimization: {qs_initial}")

# To keep this example fast, we stop after the first function
# evaluation. For a "real" optimization, remove the max_nfev
# parameter.
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-3, abs_step=1e-5, max_nfev=1)

# Make sure all procs run vmec:
qs_final = qs.J()

proc0_print(f"Final aspect ratio is {vmec.aspect()}")
proc0_print(f"Quasisymmetry objective after optimization: {qs_final}")

proc0_print("End of 2_Intermediate/QH_fixed_resolution_boozer.py")
proc0_print("===================================================")
