#!/usr/bin/env python

from simsopt.util.mpi import MpiPartition, log
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt import LeastSquaresProblem
from simsopt.solve.mpi import least_squares_mpi_solve
import os

"""
This script solve the problem in
https://github.com/landreman/stellopt_scenarios/tree/master/7DOF_varyAxisAndElongation_targetIotaAndQuasisymmetry
See that website for a detailed description of the problem.
"""

# This next line turns on detailed logging. It can be commented out if
# you do not want such verbose output.
log()

mpi = MpiPartition()
vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.stellopt_scenarios_7dof'), mpi)

# We will optimize in the space of Garabedian coefficients:
surf = vmec.boundary.to_Garabedian()
vmec.boundary = surf

# Define parameter space:
surf.all_fixed()
surf.fixed_range(mmin=0, mmax=2, nmin=-1, nmax=1, fixed=False)
surf.set_fixed("Delta(1,0)")  # toroidally-averaged major radius
surf.set_fixed("Delta(0,0)")  # toroidally-averaged minor radius

# Define objective function:
boozer = Boozer(vmec, mpol=32, ntor=16)
qs = Quasisymmetry(boozer,
                   1.0,  # Radius to target
                   1, 0,  # (M, N) you want in |B|
                   normalization="symmetric",
                   weight="stellopt")

# Objective function is 100 * (iota - (-0.41))^2 + 1 * (qs - 0)^2
prob = LeastSquaresProblem([(vmec.iota_axis, -0.41, 100),
                            (qs, 0, 1)])

residuals = prob.f()
vals = prob.dofs.f()
if mpi.proc0_world:
    print("Initial values before shifting and scaling:  ", vals[:10])
    print("Initial residuals after shifting and scaling:", residuals[:10])
    print("size of residuals:", len(residuals))
    print("Initial objective function:", prob.objective())
    print("Parameter space:")
    for name in prob.dofs.names:
        print(name)
    print("Initial state vector:", prob.x)
    print("Initial iota on axis:", vmec.iota_axis())
#exit(0)

least_squares_mpi_solve(prob, mpi, grad=True)

residuals = prob.f()
vals = prob.dofs.f()
if mpi.proc0_world:
    print("Final values before shifting and scaling:  ", vals[:10])
    print("Final residuals after shifting and scaling:", residuals[:10])
    print("Final objective function:", prob.objective())
    print("Final state vector:", prob.x)
    print("Final iota on axis:", vmec.iota_axis())
