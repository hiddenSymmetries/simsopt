#!/usr/bin/env python3

import logging
import numpy as np

from simsopt.mhd import Spec
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_serial_solve

"""
This script implements the "1DOF_circularCrossSection_varyR0_targetVolume"
example from
https://github.com/landreman/stellopt_scenarios

This optimization problem has one independent variable, representing
the mean major radius. The problem also has one objective: the plasma
volume. There is not actually any need to run an equilibrium code like
VMEC since the objective function can be computed directly from the
boundary shape. But this problem is a fast way to test the
optimization infrastructure with VMEC.

Details of the optimum and a plot of the objective function landscape
can be found here:
https://github.com/landreman/stellopt_scenarios/tree/master/1DOF_circularCrossSection_varyR0_targetVolume

This example uses the serial solver instead of the MPI solver, since
the MPI case is covered by the example
stellopt_scenarios_1DOF_circularCrossSection_varyAxis_targetIota_spec.
"""
print("Running 1DOF_circularCrossSection_varyR0_targetVolume_spec.py")
print("=============================================================")
# Print detailed logging info. This could be commented out if desired.
logging.basicConfig(level=logging.DEBUG)

# Create a Spec object:
equil = Spec()

# Start with a default surface, which is axisymmetric with major
# radius 1 and minor radius 0.1.
equil.boundary = SurfaceRZFourier(nfp=5, mpol=1, ntor=1)
surf = equil.boundary

# Set the initial boundary shape. Here is one syntax:
surf.set('rc(0,0)', 1.0)
# Here is another syntax:
surf.set_rc(0, 1, 0.1)
surf.set_zs(0, 1, 0.1)

surf.set_rc(1, 0, 0.1)
surf.set_zs(1, 0, 0.1)

print('rc:', surf.rc)
print('zs:', surf.zs)

# Surface parameters are all non-fixed by default.  You can choose
# which parameters are optimized by setting their 'fixed' attributes.
surf.fix_all()
surf.unfix('rc(0,0)')

# Each Target is then equipped with a shift and weight, to become a
# term in a least-squares objective function.  A list of terms are
# combined to form a nonlinear-least-squares problem.
desired_volume = 0.15
prob = LeastSquaresProblem.from_tuples([(equil.volume, desired_volume, 1)])

# Solve the minimization problem. We can choose whether to use a
# derivative-free or derivative-based algorithm.
least_squares_serial_solve(prob, grad=True)

print("At the optimum,")
print(" rc(m=0,n=0) = ", surf.get_rc(0, 0))
print(" volume, according to SPEC    = ", equil.volume())
print(" volume, according to Surface = ", surf.volume())
print(" objective function = ", prob.objective())

assert np.abs(surf.get_rc(0, 0) - 0.7599088773175) < 1.0e-5
assert np.abs(equil.volume() - 0.15) < 1.0e-6
assert np.abs(surf.volume() - 0.15) < 1.0e-6
assert prob.objective() < 1.0e-15
print("End of 1DOF_circularCrossSection_varyR0_targetVolume_spec.py")
print("============================================================")
