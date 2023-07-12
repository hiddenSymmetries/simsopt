#!/usr/bin/env python

import os

import numpy as np

from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import ConstrainedProblem
from simsopt.solve.mpi import constrained_mpi_solve

"""
Optimize a VMEC equilibrium for quasi-helical symmetry (M=1, N=-1)
throughout the volume.

Solve as a constrained opt problem
min QH symmetry error
s.t. 
  aspect ratio <= 8
  -1.05 <= iota <= -1

Run with e.g.
  mpiexec -n 48 constrained_optimization.py

(Any number of processors will work.)
"""

mpi = MpiPartition()
mpi.write()

if mpi.proc0_world:
    print("Running 2_Intermediate/constrained_optimization.py")
    print("==================================================")


vmec_input = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
vmec = Vmec(vmec_input, mpi=mpi, verbose=False)
surf = vmec.boundary

# Configure quasisymmetry objective:
qs = QuasisymmetryRatioResidual(vmec,
                                np.arange(0, 1.01, 0.1),  # Radii to target
                                helicity_m=1, helicity_n=-1)  # (M, N) you want in |B|
# nonlinear constraints
tuples_nlc = [(vmec.aspect, -np.inf, 8), (vmec.mean_iota, -1.05, -1.0)]

# define problem
prob = ConstrainedProblem(qs.total, tuples_nlc=tuples_nlc)

vmec.run()
if mpi.proc0_world:
    print("Initial Quasisymmetry:", qs.total())
    print("Initial aspect ratio:", vmec.aspect())
    print("Initial rotational transform:", vmec.mean_iota())


# Fourier modes of the boundary with m <= max_mode and |n| <= max_mode
# will be varied in the optimization. A larger range of modes are
# included in the VMEC calculations.
for step in range(3):
    max_mode = step + 1

    # VMEC's mpol & ntor will be 3, 4, 5:
    vmec.indata.mpol = 3 + step
    vmec.indata.ntor = vmec.indata.mpol

    if mpi.proc0_world:
        print("Beginning optimization with max_mode =", max_mode, \
              ", vmec mpol=ntor=", vmec.indata.mpol, \
              ". Previous vmec iteration = ", vmec.iter)

    # Define parameter space:
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, 
                     nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")  # Major radius

    # put bound constraints on the variables
    n_dofs = len(surf.x)
    surf.upper_bounds = 10*np.ones(n_dofs)
    surf.lower_bounds = -5*np.ones(n_dofs)
    surf.set_upper_bound("rc(1,0)", 1.0)

    # solver options
    options = {'disp': True, 'ftol': 1e-7, 'maxiter': 1}
    # solve the problem
    constrained_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, options=options)
    xopt = prob.x

    # Preserve the output file from the last iteration, so it is not
    # deleted when vmec runs again:
    vmec.files_to_delete = []

    # evaluate the solution
    surf.x = xopt
    vmec.run()
    if mpi.proc0_world:
        print("")
        print(f"Completed optimization with max_mode ={max_mode}. ")
        print(f"Final vmec iteration = {vmec.iter}")
        print("Quasisymmetry:", qs.total())
        print("aspect ratio:", vmec.aspect())
        print("rotational transform:", vmec.mean_iota())


if mpi.proc0_world:
    print("")
    print("End of 2_Intermediate/constrained_optimization.py")
    print("=================================================")

