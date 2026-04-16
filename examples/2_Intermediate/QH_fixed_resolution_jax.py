#!/usr/bin/env python

import os

import numpy as np

from simsopt.mhd import VmecJax
from simsopt.solve import build_vmec_objective_stage, least_squares_jax_solve
from simsopt.util import proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume.
Run this example with python QH_fixed_resolution_jax.py
"""

max_nfev = 10  # Maximum number of function evaluations
max_mode = 1  # Maximum poloidal and toroidal mode numbers to vary

proc0_print("Running 2_Intermediate/QH_fixed_resolution_jax.py")
proc0_print("=================================================")

# For forming filenames for VMEC, pathlib sometimes does not work, so use os.path.join instead.
filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
vmec = VmecJax(filename, verbose=False)
vmec.use_residual_autodiff_defaults(
    outer_method="scipy",
    residual_adjoint_mode="chunked",
    stateless_evaluations=False,
    optimization_profile="qh",
)
vmec.set_solver_options(residual_derivative_backend="discrete_adjoint")

# Define objective function and parameter space:
objective_tuples = [("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)]
stage = build_vmec_objective_stage(
    vmec,
    max_mode=max_mode,
    objective_tuples=objective_tuples,
    surfaces=np.arange(0, 1.01, 0.1),
    helicity_m=1,
    helicity_n=-1,
    x_scale_alpha=1.2,
    x_scale_min=1e-9,
)
surf = stage.extras["surf"]
qs = stage.extras["qs"]
residuals_from_state = stage.extras["residuals_from_state"]

proc0_print("Parameter space:", stage.free_names)

state = vmec.solve_state_for_objective(stage.x0)
residual = np.asarray(residuals_from_state(state), dtype=float)

proc0_print("Quasisymmetry objective before optimization:", float(np.asarray(qs.total_from_state(state))))
proc0_print("Total objective before optimization:", float(np.dot(residual, residual)))

# Unlike the classic example, the Jacobian here comes from vmec_jax through
# JAX autodiff / discrete adjoints instead of finite differences.
result = least_squares_jax_solve(
    stage.residuals,
    stage.x0,
    method="scipy",
    jac="jax",
    max_nfev=max_nfev,
    gtol=1e-7,
    x_scale=stage.x_scale,
    jit=True,
    verbose=1,
)

surf.set_free_params(result["x"])
state = vmec.solve_state_for_objective(result["x"])
residual = np.asarray(residuals_from_state(state), dtype=float)

proc0_print("Final aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(state))))
proc0_print("Quasisymmetry objective after optimization:", float(np.asarray(qs.total_from_state(state))))
proc0_print("Total objective after optimization:", float(np.dot(residual, residual)))

proc0_print("End of 2_Intermediate/QH_fixed_resolution_jax.py")
proc0_print("===============================================")
