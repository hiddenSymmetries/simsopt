#!/usr/bin/env python

import os
import numpy as np

from simsopt.util import log, MpiPartition
from simsopt.mhd import Vmec, Spec, Boozer, Quasisymmetry
from simsopt.mhd.spec import Residue
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve

"""
In this example, we simultaneously optimize for quasisymmetry and
the elimination of magnetic islands, with both VMEC and SPEC called in
the objective function.

Below, the argument max_nfev=1 in least_squares_mpi_solve causes the
optimization to stop after only a single iteration, so this example
does not take too long to run. For a real optimization, that argument
should be removed.
"""

log()
mpi = MpiPartition()
mpi.write()

vmec_filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA_iota0.4_withIslands')
vmec = Vmec(vmec_filename, mpi=mpi)
surf = vmec.boundary

spec_filename = os.path.join(os.path.dirname(__file__), 'inputs', 'nfp2_QA_iota0.4_withIslands.sp')
spec = Spec(spec_filename, mpi=mpi)

# This next line is where the boundary surface objects of VMEC and
# SPEC are linked:
spec.boundary = surf

# Define parameter space:
surf.fix_all()
surf.fixed_range(mmin=0, mmax=3,
                 nmin=-3, nmax=3, fixed=False)
surf.fix("rc(0,0)")  # Major radius

# Configure quasisymmetry objective:
qs = Quasisymmetry(Boozer(vmec),
                   0.5,  # Radius to target
                   1, 0)  # (M, N) you want in |B|

# iota = p / q
p = -2
q = 5
residue1 = Residue(spec, p, q)
residue2 = Residue(spec, p, q, theta=np.pi)

if mpi.group == 0:
    r1 = residue1.J()
    r2 = residue2.J()
if mpi.proc0_world:
    print("Initial residues:", r1, r2)
#exit(0)

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 6, 1),
                                        (vmec.iota_axis, 0.385, 1),
                                        (vmec.iota_edge, 0.415, 1),
                                        (qs.J, 0, 1),
                                        (residue1.J, 0, 2),
                                        (residue2.J, 0, 2)])

# Check whether we're in the CI. If so, just do a single function
# evaluation rather than a real optimization.
ci = "CI" in os.environ and os.environ['CI'].lower() in ['1', 'true']
if ci:
    obj = prob.objective()
else:
    # Remove the max_nfev=1 in the next line to do a serious optimization:
    least_squares_mpi_solve(prob, mpi=mpi, grad=True, max_nfev=1)

if mpi.group == 0:
    r1 = residue1.J()
    r2 = residue2.J()
if mpi.proc0_world:
    print("Final residues:", r1, r2)

print("Good bye")
