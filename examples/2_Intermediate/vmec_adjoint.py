#!/usr/bin/env python

import os
import numpy as np
from simsopt.mhd import Vmec
from simsopt.mhd import IotaTargetMetric
from scipy.optimize import minimize
from simsopt._core.util import ObjectiveFailure

"""
Here, we perform an optimization begining with a 3 field period rotating ellipse
boundary to obtain iota = 0.381966. The derivatives are obtained with an adjoint
method. This is based on a published result in
Paul, Landreman, and Antonsen, Journal of Plasma Physics (2021). The number of
modes in the optimization space is slowly increased from |m|,|n| <= 2 to 5. At
the end, the initial and final profiles are plotted.
"""

# check whether we're in CI, in that case we make the run a bit cheaper
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']

# target profile of rotational transform, here just a constant:
target_function = lambda s: 0.381966
adjoint_epsilon = 1.e-1  # perturbation amplitude for adjoint solve

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.rotating_ellipse')
vmec = Vmec(filename, ntheta=100, nphi=100)

# Lower resolution and function evaluations if in CI
if ci:
    vmec.indata.mpol = 6
    vmec.indata.ntor = 6
    vmec.indata.ns_array = np.zeros_like(vmec.indata.ns_array)
    vmec.indata.ns_array[0] = 9
    vmec.indata.ns_array[0] = 49
    maxres = 4
    maxfun = 1
else:
    maxres = 6
    maxfun = 15000
    import matplotlib.pyplot as plt

vmec.run()
iotas_init = vmec.wout.iotas

obj = IotaTargetMetric(vmec, target_function, adjoint_epsilon)

surf = vmec.boundary
surf.fix_all()
# Slowly increase range of modes in optimization space
for max_mode in range(3, maxres):
    surf.fixed_range(mmin=0, mmax=max_mode,
                     nmin=-max_mode, nmax=max_mode, fixed=False)

    # Define objective function and derivative that handle ObjectiveFailure
    def J(dofs):
        dofs_prev = obj.x
        try:
            obj.x = dofs
            return obj.J()  # , obj.dJ()[obj.dofs_free_status]
        except ObjectiveFailure:
            obj.x = dofs_prev
            return 2*obj.J()  # , 2*obj.dJ()[obj.dofs_free_status]
    res = minimize(
        fun=J, x0=obj.x, jac=False, method='L-BFGS-B',
        options={'maxfun': maxfun, 'ftol': 1e-8, 'gtol': 1e-8})
    print(f"max_mode={max_mode:d}  res={res['fun']:.3f}, "
          f"jac={np.linalg.norm(res['jac']):.3f}")

    # Preserve the output file from the last iteration, so it is not
    # deleted when vmec runs again:
    vmec.files_to_delete = []

vmec.run()
iotas_final = vmec.wout.iotas

if not ci:
    # Plot result
    plt.figure()
    plt.plot(vmec.s_half_grid, iotas_init[1:], color='green')
    plt.plot(vmec.s_half_grid, iotas_final[1:], color='red')
    plt.axhline(target_function(0), color='blue')
    plt.legend(['Initial', 'Final', 'Target'])
    plt.xlabel(r'$s$')
    plt.ylabel(r'$\iota$')
    plt.show()
