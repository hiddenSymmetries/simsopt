#!/usr/bin/env python3

import numpy as np

from simsopt.mhd import Gvec
from simsopt.objectives import LeastSquaresProblem
from simsopt.util import MpiPartition, log, proc0_print
from simsopt.solve import least_squares_mpi_solve
from simsopt.geo import SurfaceRZFourier, SurfaceGarabedian

"""
This script implements the "1DOF_circularCrossSection_varyAxis_targetIota"
example from
https://github.com/landreman/stellopt_scenarios

This example demonstrates optimizing a surface shape using the
Garabedian representation instead of GVEC's RBC/ZBS representation.
This optimization problem has one independent variable, the Garabedian
Delta_{m=1, n=-1} coefficient, representing the helical excursion of
the magnetic axis. The objective function is (iota - iota_target)^2,
where iota is measured on the magnetic axis.

Details of the optimum and a plot of the objective function landscape
can be found here:
https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyAxis_targetIota
"""
proc0_print("Running 1DOF_circularCrossSection_varyAxis_targetIota_gvec.py")
proc0_print("=============================================================")

# Print detailed logging info. This line could be commented out if desired.
log()

# GVEC is parallelized with OpenMP, so we want a worker group for each
# MPI process (the default). By setting OMP_NUM_THREADS we can control
# how many threads each GVEC run uses.
mpi = MpiPartition()
mpi.write()

# Create the boundary object & set the initial boundary shape.
# Set boundary modes like inputs/1DOF_Garabedian.sp & inputs/input.1DOF_Garabedian
surf = SurfaceRZFourier(nfp=5, mpol=4, ntor=4, stellsym=True)
surf.set_rc(0, 0, 1.0)  # m, n, value - in VMEC it is (n, m)!
surf.set_rc(0, 1, 0.1)
surf.set_zs(0, 1, 0.1)
surf.set_rc(1, 0, 0.01)
surf.set_zs(1, 0, 0.01)

# We will optimize in the space of Garabedian coefficients rather than
# RBC/ZBS coefficients. To do this, we convert the boundary to the
# Garabedian representation:
surf = SurfaceGarabedian.from_RZFourier(surf)

# Create the GVEC optimizable
equil = Gvec(
    boundary=surf,
    current=0.0,
    mpi=mpi,
    delete_intermediates=True,
    parameters=dict(
        totalIter=10000,
        minimize_tol=1e-6,
    )
)
equil.logger.setLevel("WARNING")

# GVEC parameters (phiedge) are all fixed by default, while surface parameters
# are all non-fixed by default.  You can choose which parameters are
# optimized by setting their 'fixed' attributes.
surf.fix_all()
surf.unfix('Delta(1,-1)')

# Each function we want in the objective function is then equipped
# with a shift and weight, to become a term in a least-squares
# objective function.  A list of terms are combined to form a
# nonlinear-least-squares problem.
desired_iota = 0.41  # flipped sign compared to VMEC
prob = LeastSquaresProblem.from_tuples([(equil.iota_axis, desired_iota, 1)])

objective = prob.objective()
proc0_print("Initial state,")
proc0_print(" Delta(m=1,n=-1) = ", surf.get_Delta(1, -1))
proc0_print(" iota on axis = ", equil.iota_axis())
proc0_print(" objective function = ", objective)

# Solve the minimization problem. We can choose whether to use a
# derivative-free or derivative-based algorithm.
least_squares_mpi_solve(prob, mpi, grad=True)

if mpi.proc0_world:
    objective = prob.objective()
    print("At the optimum,")
    print(" Delta(m=1,n=-1) = ", surf.get_Delta(1, -1))
    print(" desired iota = ", desired_iota)
    print(" iota on axis = ", equil.iota_axis())
    print(" objective function = ", objective)

    assert np.abs(surf.get_Delta(1, -1) - 0.08575) < 1.0e-3
    assert np.abs(equil.iota_axis() - desired_iota) < 1.0e-5
    assert prob.objective() < 1.0e-15
    print("End of 1DOF_circularCrossSection_varyAxis_targetIota_gvec.py")
    print("============================================================")
