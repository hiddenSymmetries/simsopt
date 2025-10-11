#!/usr/bin/env python

import os
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print

"""
Optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume.
Run this example with mpirun -n 2 python QH_fixed_resolution.py
"""

max_nfev = 10  # Maximum number of function evaluations
max_mode = 1  # Maximum poloidal and toroidal mode numbers to vary

# This problem has 8 degrees of freedom, so we can use 8 + 1 = 9
# concurrent function evaluations for 1-sided finite difference
# gradients.
proc0_print("Running 2_Intermediate/QH_fixed_resolution.py")
proc0_print("=============================================")

mpi = MpiPartition(2)

# For forming filenames for vmec, pathlib sometimes does not work, so use os.path.join instead.
filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

proc0_print('Parameter space:', surf.dof_names)

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
                                        (qs.residuals, 0, 1)])

# Make sure all procs participate in computing the objective:
prob.objective()

proc0_print("Quasisymmetry objective before optimization:", qs.total())
proc0_print("Total objective before optimization:", prob.objective())

# To keep this example fast, we stop after the first max_nfev function
# evaluation. For a "real" optimization, increase this value.
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=max_nfev)

# Make sure all procs participate in computing the objective:
prob.objective()

proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("Quasisymmetry objective after optimization:", qs.total())
proc0_print("Total objective after optimization:", prob.objective())

vmec.write_input("input.QH_fixed_resolution_final")

proc0_print("End of 2_Intermediate/QH_fixed_resolution.py")
proc0_print("============================================")
