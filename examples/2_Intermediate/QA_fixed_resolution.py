#!/usr/bin/env python
import os
import re
import numpy as np
from simsopt.mhd import Vmec, QuasisymmetryRatioResidual
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
max_nfev = 10  # Maximum number of function evaluations
max_mode = 1  # Maximum poloidal and toroidal mode numbers to vary
iota_target = 0.44  # Target mean iota
aspect_target = 2.0  # Target aspect ratio
proc0_print("Running 2_Intermediate/QA_fixed_resolution.py")
proc0_print("=============================================")
mpi = MpiPartition()
filename = os.path.join(os.path.dirname(__file__), 'inputs', 'input.nfp2_QA')
vmec = Vmec(filename, mpi=mpi, verbose=False)
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode,  nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")
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
qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
prob = LeastSquaresProblem.from_tuples([(vmec.aspect, aspect_target, 1), (qs.residuals, 0, 1), (vmec.mean_iota, iota_target, 1)])
x_scale = create_simsopt_x_scale(prob.dof_names, alpha=1.2, min_scale=1e-9)
prob.objective()
proc0_print("Quasisymmetry objective before optimization:", qs.total())
proc0_print("Total objective before optimization:", prob.objective())
least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, max_nfev=max_nfev, gtol=1e-7, x_scale=x_scale)
prob.objective()
proc0_print("Final aspect ratio:", vmec.aspect())
proc0_print("Quasisymmetry objective after optimization:", qs.total())
proc0_print("Total objective after optimization:", prob.objective())
vmec.write_input("input.QA_fixed_resolution_final")