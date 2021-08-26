#!/usr/bin/env python

import os
import numpy as np
import matplotlib.pyplot as plt
from simsopt.mhd.vmec import Vmec, IotaTargetMetric
from simsopt.objectives.least_squares import LeastSquaresProblem
from simsopt.solve.serial import least_squares_serial_solve

"""
Here, we perform an optimization begining with a 3 field period rotating ellipse
boundary to obtain iota = 0.381966. The derivatives are obtained with an adjoint
method. This is based on a published result in
Paul, Landreman, and Antonsen, Journal of Plasma Physics (2021). The number of
modes in the optimization space is slowly increased from |m|,|n| <= 2 to 5. At
the end, the initial and final profiles are plotted.
"""

# target profile of rotational transform, here just a constant:
target_function = lambda s: 0.381966
adjoint_epsilon = 1.e-1  # perturbation amplitude for adjoint solve

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.rotating_ellipse')
vmec = Vmec(filename, ntheta=100, nphi=100)

vmec.run()
iotas_init = vmec.wout.iotas

obj = IotaTargetMetric(vmec, target_function, adjoint_epsilon)
prob = LeastSquaresProblem([(obj, 0, 1)])

surf = vmec.boundary
surf.all_fixed(True)
# Slowly increase range of modes in optimization space
for max_mode in range(3, 6):
    print(max_mode)
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)

    least_squares_serial_solve(prob, grad=True, ftol=1e-12, gtol=1e-12, xtol=1e-12)

    # Preserve the output file from the last iteration, so it is not
    # deleted when vmec runs again:
    vmec.files_to_delete = []

vmec.run()
iotas_final = vmec.wout.iotas

# Plot result
plt.figure()
plt.plot(vmec.s_half_grid, iotas_init[1:], color='green')
plt.plot(vmec.s_half_grid, iotas_final[1:], color='red')
plt.axhline(target_function(0), color='blue')
plt.legend(['Initial', 'Final', 'Target'])
plt.xlabel(r'$s$')
plt.ylabel(r'$\iota$')
plt.show()
