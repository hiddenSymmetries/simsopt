#!/usr/bin/env python3

import os
from mpi4py import MPI
import numpy as np

from simsopt.mhd import Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.util import MpiPartition, log
from simsopt.solve import least_squares_mpi_solve
from simsopt.geo import SurfaceGarabedian

"""
This script implements the "1DOF_circularCrossSection_varyAxis_targetIota"
example from
https://github.com/landreman/stellopt_scenarios

This example demonstrates optimizing a surface shape using the
Garabedian representation instead of VMEC's RBC/ZBS representation.
This optimization problem has one independent variable, the Garabedian
Delta_{m=1, n=-1} coefficient, representing the helical excursion of
the magnetic axis. The objective function is (iota - iota_target)^2,
where iota is measured on the magnetic axis.

Details of the optimum and a plot of the objective function landscape
can be found here:
https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyAxis_targetIota
"""
print("Running 1DOF_circularCrossSection_varyAxis_targetIota.py")
print("========================================================")

# Print detailed logging info. This line could be commented out if desired.
log()

# In the next line, we can adjust how many groups the pool of MPI
# processes is split into.
mpi = MpiPartition(ngroups=1)
mpi.write()

# Start with a default surface, which is axisymmetric with major
# radius 1 and minor radius 0.1.
equil = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.1DOF_Garabedian'), mpi=mpi)

# We will optimize in the space of Garabedian coefficients rather than
# RBC/ZBS coefficients. To do this, we convert the boundary to the
# Garabedian representation:
surf = SurfaceGarabedian.from_RZFourier(equil.boundary)
equil.boundary = surf

# VMEC parameters are all fixed by default, while surface parameters
# are all non-fixed by default.  You can choose which parameters are
# optimized by setting their 'fixed' attributes.
surf.fix_all()
surf.unfix('Delta(1,-1)')

# Each function we want in the objective function is then equipped
# with a shift and weight, to become a term in a least-squares
# objective function.  A list of terms are combined to form a
# nonlinear-least-squares problem.
desired_iota = -0.41
prob = LeastSquaresProblem.from_tuples([(equil.iota_axis, desired_iota, 1)])

# Solve the minimization problem. We can choose whether to use a
# derivative-free or derivative-based algorithm.
least_squares_mpi_solve(prob, mpi, grad=False)

# Make sure all procs call VMEC:
objective = prob.objective()
if mpi.proc0_world:
    print("At the optimum,")
    print(" Delta(m=1,n=-1) = ", surf.get_Delta(1, -1))
    print(" iota on axis = ", equil.iota_axis())
    print(" objective function = ", objective)

assert np.abs(surf.get_Delta(1, -1) - 0.08575) < 1.0e-4
assert np.abs(equil.iota_axis() - desired_iota) < 1.0e-5
assert prob.objective() < 1.0e-15
print("End of 1DOF_circularCrossSection_varyAxis_targetIota.py")
print("========================================================")
