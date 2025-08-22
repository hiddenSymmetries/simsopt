#!/usr/bin/env python3

import numpy as np

from simsopt.mhd import Gvec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, log, proc0_print
from simsopt.geo import SurfaceRZFourier

"""
This script implements the "2DOF_vmecOnly_targetIotaAndVolume" example from
https://github.com/landreman/stellopt_scenarios

This optimization problem has two independent variables, representing
the helical shape of the magnetic axis. The problem also has two
objectives: the plasma volume and the rotational transform on the
magnetic axis.

The resolution in this example (i.e. ns, mpol, and ntor) is somewhat
lower than in the stellopt_scenarios version of the example, just so
this example runs fast.

Details of the optimum and a plot of the objective function landscape
can be found here:
https://github.com/landreman/stellopt_scenarios/tree/master/2DOF_vmecOnly_targetIotaAndVolume
"""
proc0_print("Running 2DOF_gvecOnly_targetIotaAndVolume.py")
proc0_print("============================================")
# This next line turns on detailed logging. It can be commented out if
# you do not want such verbose output.
log()

# GVEC is parallelized with OpenMP, so we want a worker group for each
# MPI process (the default). By setting OMP_NUM_THREADS we can control
# how many threads each GVEC run uses.
mpi = MpiPartition()
mpi.write()

# Create the boundary object & set the initial boundary shape.
# Set boundary modes like inputs/input.2DOF_vmecOnly_targetIotaAndVolume
surf = SurfaceRZFourier(nfp=5, mpol=4, ntor=4, stellsym=True)
surf.set_rc(0, 0, 1.0)
surf.set_rc(0, 1, 0.0) # helicity of the axis
surf.set_zs(0, 1, 0.0)
surf.set_rc(1, 0, 0.1) # axisymmetric circular or elliptical cross-section
surf.set_zs(1, 0, 0.1)
surf.set_rc(1, 1, 0.05) # rotating elongation
surf.set_zs(1, 1, -0.05)

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
surf.unfix('rc(1,1)')
surf.unfix('zs(1,1)')

# Each Target is then equipped with a shift and weight, to become a
# term in a least-squares objective function.  A list of terms are
# combined to form a nonlinear-least-squares problem.
desired_volume = 0.15
volume_weight = 1
term1 = (equil.volume, desired_volume, volume_weight)

desired_iota = -0.41  # flipped sign compared to VMEC
iota_weight = 1
term2 = (equil.iota_axis, desired_iota, iota_weight)

prob = LeastSquaresProblem.from_tuples([term1, term2])
proc0_print(f"Length of x is {len(prob.x)}")

# Solve the minimization problem:
least_squares_mpi_solve(prob, mpi, grad=True)

objective = prob.objective()
proc0_print("At the optimum,")
proc0_print(" rc(m=1,n=1) = ", surf.get_rc(1, 1))
proc0_print(" zs(m=1,n=1) = ", surf.get_zs(1, 1))
proc0_print(" volume, desired              = ", desired_volume)
proc0_print(" volume, according to GVEC    = ", equil.volume())
proc0_print(" volume, according to Surface = ", surf.volume())
proc0_print(" iota, desired = ", desired_iota)
proc0_print(" iota on axis  = ", equil.iota_axis())
proc0_print(" objective function = ", objective)

if mpi.proc0_world:
    assert np.abs(surf.get_rc(1, 1) - 0.0313066948) < 1.0e-3
    assert np.abs(surf.get_zs(1, 1) - (-0.031232391)) < 1.0e-3
    assert np.abs(equil.volume() - 0.178091) < 1.0e-3
    assert np.abs(surf.volume() - 0.178091) < 1.0e-3
    assert np.abs(equil.iota_axis() - 0.4114567) < 5.0e-3
    assert prob.objective() < 1.0e-2
proc0_print("End of 2DOF_gvecOnly_targetIotaAndVolume.py")
proc0_print("============================================")
