#!/usr/bin/env python3

import os
import numpy as np

from simsopt.mhd import Vmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.util import MpiPartition, log
from simsopt.solve import least_squares_mpi_solve

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
print("Running 2DOF_vmecOnly_targetIotaAndVolume.py")
print("============================================")
# This next line turns on detailed logging. It can be commented out if
# you do not want such verbose output.
log()

# In the next line, we can adjust how many groups the pool of MPI
# processes is split into.
mpi = MpiPartition(ngroups=3)
mpi.write()

# Initialize VMEC from an input file:
filename = os.path.join(os.path.dirname(__file__), 'inputs',
                        'input.2DOF_vmecOnly_targetIotaAndVolume')
equil = Vmec(filename, mpi)
surf = equil.boundary

# VMEC parameters are all fixed by default, while surface parameters
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

desired_iota = 0.41
iota_weight = 1
term2 = (equil.iota_axis, desired_iota, iota_weight)

prob = LeastSquaresProblem.from_tuples([term1, term2])
print(f"Length of x is {len(prob.x)}")

# Solve the minimization problem:
least_squares_mpi_solve(prob, mpi, grad=True)

objective = prob.objective()
if mpi.proc0_world:
    print("At the optimum,")
    print(" rc(m=1,n=1) = ", surf.get_rc(1, 1))
    print(" zs(m=1,n=1) = ", surf.get_zs(1, 1))
    print(" volume, according to VMEC    = ", equil.volume())
    print(" volume, according to Surface = ", surf.volume())
    print(" iota on axis = ", equil.iota_axis())
    print(" objective function = ", objective)

    assert np.abs(surf.get_rc(1, 1) - 0.0313066948) < 1.0e-3
    assert np.abs(surf.get_zs(1, 1) - (-0.031232391)) < 1.0e-3
    assert np.abs(equil.volume() - 0.178091) < 1.0e-3
    assert np.abs(surf.volume() - 0.178091) < 1.0e-3
    assert np.abs(equil.iota_axis() - 0.4114567) < 1.0e-4
    assert prob.objective() < 1.0e-2
print("End of 2DOF_vmecOnly_targetIotaAndVolume.py")
print("============================================")
