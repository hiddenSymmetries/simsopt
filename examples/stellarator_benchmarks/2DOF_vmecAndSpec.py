#!/usr/bin/env python3

import os
import logging
import numpy as np

from simsopt.util import MpiPartition, log
from simsopt.mhd import Vmec, Spec
from simsopt.objectives import LeastSquaresProblem
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
print("Running 2DOF_vmecAndSpec.py")
print("=========================")
# This next line turns on detailed logging. It can be commented out if
# you do not want such verbose output.
log(logging.INFO)

mpi = MpiPartition()

# Initialize VMEC from an input file:
vmec = Vmec(os.path.join(os.path.dirname(__file__), 'inputs', 'input.2DOF_vmecOnly_targetIotaAndVolume'), mpi=mpi)
surf = vmec.boundary

# Initialize SPEC from an input file:
spec = Spec(os.path.join(os.path.dirname(__file__), 'inputs', '2DOF_targetIotaAndVolume.sp'), mpi=mpi)

# Set the SPEC boundary to be the same object as the VMEC boundary!
spec.boundary = surf

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
term1 = (spec.volume, desired_volume, volume_weight)

desired_iota = 0.41
iota_weight = 1
term2 = (vmec.iota_axis, desired_iota, iota_weight)

prob = LeastSquaresProblem.from_tuples([term1, term2])

# Solve the minimization problem:
least_squares_mpi_solve(prob, mpi=mpi, grad=True)

# Evaluate quantities on all processes, in case communication is
# required:
final_objective = prob.objective()
vmec_volume = vmec.volume()
spec_volume = spec.volume()
surf_volume = surf.volume()
vmec_iota = vmec.iota_axis()
spec_iota = spec.iota()

if mpi.proc0_world:
    logging.info("At the optimum,")
    logging.info(f" objective function = {final_objective}")
    logging.info(f" rc(m=1,n=1) = {surf.get_rc(1, 1)}")
    logging.info(f" zs(m=1,n=1) = {surf.get_zs(1, 1)}")
    logging.info(f" volume, according to VMEC    = {vmec_volume}")
    logging.info(f" volume, according to SPEC    = {spec_volume}")
    logging.info(f" volume, according to Surface = {surf_volume}")
    logging.info(f" iota on axis, from VMEC       = {vmec_iota}")
    logging.info(f" iota at mid-radius, from SPEC = {spec_iota}")

    assert np.abs(surf.get_rc(1, 1) - 0.0313066948) < 1.0e-3
    assert np.abs(surf.get_zs(1, 1) - (-0.031232391)) < 1.0e-3
    assert np.abs(spec_volume - 0.178091) < 1.0e-3
    assert np.abs(vmec_volume - 0.178091) < 1.0e-3
    assert np.abs(surf_volume - 0.178091) < 1.0e-3
    assert np.abs(vmec_iota - 0.4114567) < 1.0e-4
    assert final_objective < 1.0e-2
print("End of  2DOF_vmecAndSpec.py")
print("=========================")
