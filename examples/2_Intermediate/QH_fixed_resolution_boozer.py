#!/usr/bin/env python

import os
import re
from simsopt.mhd import Vmec, Boozer, Quasisymmetry
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
import numpy as np

"""
Optimize for quasi-helical symmetry (M=1, N=1) at a given radius.
"""

max_mode = 1
max_nfev = 10

# This problem has 24 degrees of freedom, so we can use 24 + 1 = 25
# concurrent function evaluations for 1-sided finite difference
# gradients.
proc0_print("Running 2_Intermediate/QH_fixed_resolution_boozer.py")
proc0_print("====================================================")

# Keep all ranks in a single VMEC worker group. The Boozer transform is
# performed only on the group leader, while the remaining ranks still
# participate in the VMEC solve.
mpi = MpiPartition()

filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp4_QH_warm_start')
vmec = Vmec(filename, mpi=mpi, verbose=False)
vmec.indata.mpol = max_mode + 2
vmec.indata.ntor = vmec.indata.mpol

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,
                 nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius

def create_simsopt_x_scale(dof_names, alpha=1.6, min_scale=1e-6):
    x_scale = np.ones(len(dof_names))
    pattern = r'[rz][cs]\((\d+),(-?\d+)\)'
    modes = []
    mode_indices = []
    for i, name in enumerate(dof_names):
        m = re.search(pattern, name)
        modes.append([int(m.group(1)), int(m.group(2))])
        mode_indices.append(i)
    modes = np.array(modes)
    mode_level = np.max(np.abs(modes), axis=1)
    x_scale = np.exp(-alpha * mode_level) / np.exp(-alpha)
    mode_scales = np.maximum(x_scale, min_scale)
    for idx, scale in zip(mode_indices, mode_scales): x_scale[idx] = scale
    return x_scale

# Configure quasisymmetry objective:
qs = Quasisymmetry(Boozer(vmec, mpol=16, ntor=16, use_wout_file=True),
                   0.5,  # Radius to target
                   1, 1)  # (M, N) you want in |B|

# Define objective function
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7, 1),
                                        (qs.J, 0, 1)])

x_scale = create_simsopt_x_scale(prob.dof_names, alpha=1.2, min_scale=1e-9)

# Make sure all procs run vmec:
qs_initial = np.sum(qs.J()**2)

proc0_print(f"Quasisymmetry objective (sum of squares) before optimization: {qs_initial}")

# To keep this example fast, we stop after the first function
# evaluation. For a "real" optimization, remove the max_nfev
# parameter.
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, max_nfev=max_nfev, gtol=1e-7, x_scale=x_scale)

# Make sure all procs run vmec:
qs_final = np.sum(qs.J()**2)

proc0_print(f"Final aspect ratio is {vmec.aspect()}")
proc0_print(f"Quasisymmetry objective (sum of squares) after optimization: {qs_final}")

proc0_print("End of 2_Intermediate/QH_fixed_resolution_boozer.py")
proc0_print("===================================================")
