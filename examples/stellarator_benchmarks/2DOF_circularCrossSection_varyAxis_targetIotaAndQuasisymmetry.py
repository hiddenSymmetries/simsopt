#!/usr/bin/env python

from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt import LeastSquaresProblem
from simsopt import least_squares_serial_solve
import os

"""
This script solve the problem in
https://github.com/landreman/stellopt_scenarios/tree/master/2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry
See that website for a detailed description of the problem and plots
of the objective function landscape.
"""

vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.2DOF_circularCrossSection_varyAxis_targetIotaAndQuasisymmetry'))

# Define parameter space:
vmec.boundary.all_fixed()
vmec.boundary.set_fixed("rc(0,1)", False)
vmec.boundary.set_fixed("zs(0,1)", False)

# Define objective function:
boozer = Boozer(vmec, mpol=32, ntor=16)
qs = Quasisymmetry(boozer,
                   1.0,  # Radius to target
                   1, 0,  # (M, N) you want in |B|
                   normalization="symmetric",
                   weight="stellopt_ornl")

# Objective function is 100 * (iota - (-0.41))^2 + 1 * (qs - 0)^2
prob = LeastSquaresProblem([(vmec.iota_axis, -0.41, 100),
                            (qs, 0, 1)])

least_squares_serial_solve(prob)

print("Final values before shifting and scaling:", prob.dofs.f())
print("Final residuals:", prob.f())
print("Final state vector:", prob.x)
print("Final iota on axis:", vmec.iota_axis())
