#!/usr/bin/env python3

import os
import logging
import numpy as np

from simsopt.mhd import Spec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve

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

print("Running 2DOF_specOnly_targetIotaAndVolume.py")
print("============================================")
# This next line turns on detailed logging. It can be commented out if
# you do not want such verbose output.
logging.basicConfig(level=logging.INFO)

# Initialize SPEC from an input file
equil = Spec(os.path.join(os.path.dirname(__file__), 'inputs', '2DOF_targetIotaAndVolume.sp'))

# If the xspec executable is not in PATH, the path to the executable
# should be specified as in the following line:
# equil = Spec('2DOF_targetIotaAndVolume.sp', exe='/Users/mattland/SPEC/xspec')

surf = equil.boundary

# VMEC parameters are all fixed by default, while surface parameters are all non-fixed by default.
# You can choose which parameters are optimized by setting their 'fixed' attributes.
surf.fix_all()
surf.unfix('rc(1,1)')
surf.unfix('zs(1,1)')

# Each Target is then equipped with a shift and weight, to become a
# term in a least-squares objective function.  A list of terms are
# combined to form a nonlinear-least-squares problem.
desired_volume = 0.15
volume_weight = 1
term1 = (equil.volume, desired_volume, volume_weight)

desired_iota = -0.41
iota_weight = 1
term2 = (equil.iota, desired_iota, iota_weight)

prob = LeastSquaresProblem.from_tuples([term1, term2])

# Solve the minimization problem:
least_squares_serial_solve(prob, grad=True)

print("At the optimum,")
print(" rc(m=1,n=1) = ", surf.get_rc(1, 1))
print(" zs(m=1,n=1) = ", surf.get_zs(1, 1))
print(" volume, according to VMEC    = ", equil.volume())
print(" volume, according to Surface = ", surf.volume())
print(" iota on axis = ", equil.iota())
print(" objective function = ", prob.objective())

# The tests here are based on values from the VMEC version in
# https://github.com/landreman/stellopt_scenarios/tree/master/2DOF_vmecOnly_targetIotaAndVolume
# Due to this and the fact that we don't yet have iota on axis from SPEC, the tolerances are wide.
assert np.abs(surf.get_rc(1, 1) - 0.0313066948) < 0.001
assert np.abs(surf.get_zs(1, 1) - (-0.031232391)) < 0.001
assert np.abs(equil.volume() - 0.178091) < 0.001
assert np.abs(surf.volume() - 0.178091) < 0.001
assert np.abs(equil.iota() - (-0.4114567)) < 0.001
assert (prob.objective() - 7.912501330E-04) < 0.2e-4
print("End of 2DOF_specOnly_targetIotaAndVolume.py")
print("============================================")
