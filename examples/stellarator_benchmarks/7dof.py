#!/usr/bin/env python

import os
from simsopt.util import MpiPartition, log
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.geo import SurfaceGarabedian

"""
This script solve the problem in
https://github.com/landreman/stellopt_scenarios/tree/master/7DOF_varyAxisAndElongation_targetIotaAndQuasisymmetry
See that website for a detailed description of the problem.
"""
print("Running 7dof.py")
print("===============")
# This next line turns on detailed logging. It can be commented out if
# you do not want such verbose output.
log()

mpi = MpiPartition()
vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.stellopt_scenarios_7dof'), mpi)

# We will optimize in the space of Garabedian coefficients:
surf = SurfaceGarabedian.from_RZFourier(vmec.boundary)
vmec.boundary = surf

# Define parameter space:
surf.fix_all()
surf.fix_range(mmin=0, mmax=2, nmin=-1, nmax=1, fixed=False)
surf.fix("Delta(1,0)")  # toroidally-averaged major radius
surf.fix("Delta(0,0)")  # toroidally-averaged minor radius

# Define objective function:
boozer = Boozer(vmec, mpol=32, ntor=16)
qs = Quasisymmetry(boozer,
                   1.0,  # Radius to target
                   1, 0,  # (M, N) you want in |B|
                   normalization="symmetric",
                   weight="stellopt")

# Objective function is 100 * (iota - (-0.41))^2 + 1 * (qs - 0)^2
prob = LeastSquaresProblem.from_tuples([(vmec.iota_axis, -0.41, 100),
                                        (qs.J, 0, 1)])
objective = prob.objective()
if mpi.proc0_world:
    print("Initial objective function:", objective)
    #print("Parameter space:")
    #print(prob.dof_names)
    print("Initial state vector:", prob.x)
    print("Initial iota on axis:", vmec.iota_axis())
#exit(0)

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
#ci = True

# Only do 1 iteration if we are running the tests in Gitub
# Actions. Leave max_nfev as None for a real optimization.
max_nfev = 5 if ci else None
least_squares_mpi_solve(prob, mpi, grad=True, max_nfev=max_nfev)

objective = prob.objective()
if mpi.proc0_world:
    print("Final objective function:", objective)
    print("Final state vector:", prob.x)
    print("Final iota on axis:", vmec.iota_axis())

print("End of 7dof.py")
print("===============")
