#!/usr/bin/env python

import os
import re
import time

import numpy as np
import jax.numpy as jnp

from simsopt.mhd import VmecJax, QuasisymmetryRatioResidualJax
from simsopt.solve import least_squares_jax_solve
from simsopt.util import proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume, using the VMEC-only quasisymmetry diagnostic.
"""

MAX_MODE = 3
MAX_NFEV = 20
PRINT_TIMINGS = True

# Native JAX fixed-boundary solve. Change to "lbfgs" for the other fast native
# path, or to "vmec2000" for the slower parity-friendly reference solve that
# more closely tracks the original VMEC2000 equilibrium history.
VMEC_SOLVER = "gd"
VMEC_MAX_ITER = 500
VMEC_GRAD_TOL = 1e-3

# The VMEC-only QS objective is stiffer than the Boozer-based variant, so use
# a JAX-native L-BFGS outer loop by default. Change to "gradient_descent" if
# you want the simplest fully native fallback path instead.
OUTER_METHOD = "lbfgs"
OUTER_STEP_SIZE = 1e-6


def create_simsopt_x_scale(dof_names, alpha=1.6, min_scale=1e-6):
    """Create the mode-weighted parameter scaling used by the SIMSOPT examples."""
    x_scale = np.ones(len(dof_names))
    pattern = r"[rz][cs]\((\d+),(-?\d+)\)"
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
    for idx, scale in zip(mode_indices, mode_scales):
        x_scale[idx] = scale
    return x_scale


proc0_print("Running 2_Intermediate/QH_fixed_resolution_jax.py")
proc0_print("==================================================")

t_start = time.perf_counter()
filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
vmec = VmecJax(filename, verbose=False)
vmec.set_solver_options(
    solver=VMEC_SOLVER,
    warm_start_iters=0,
    max_iter=VMEC_MAX_ITER,
    grad_tol=VMEC_GRAD_TOL,
)
vmec.indata.mpol = MAX_MODE + 2
vmec.indata.ntor = vmec.indata.mpol

surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=MAX_MODE, nmin=-MAX_MODE, nmax=MAX_MODE, fixed=False)
surf.fix("rc(0,0)")

proc0_print("Parameter space:", surf.dof_names)

qs = QuasisymmetryRatioResidualJax(
    vmec,
    np.arange(0, 1.01, 0.1),
    helicity_m=1,
    helicity_n=-1,
)


def residuals(x_free):
    state = vmec._solve_state(x_free)
    aspect = vmec.aspect_equilibrium_from_state_jax(state)
    qs_res = qs.residuals_from_state(state)
    return jnp.concatenate([jnp.array([aspect - 7.0]), qs_res])


x0 = surf.get_free_params()
free_names = [name for name, free in zip(surf.dof_names, surf.free) if free]
x_scale = create_simsopt_x_scale(free_names, alpha=1.2, min_scale=1e-9)

if PRINT_TIMINGS:
    proc0_print(f"Startup setup time: {time.perf_counter() - t_start:.3f}s")

proc0_print("Evaluating initial quasisymmetry objective...")
t_qs0 = time.perf_counter()
state0 = vmec._solve_state(jnp.asarray(x0))
qs_initial = float(np.asarray(qs.total_from_state(state0)))
if PRINT_TIMINGS:
    proc0_print(f"Initial QS evaluation time: {time.perf_counter() - t_qs0:.3f}s")

proc0_print(f"Quasisymmetry objective before optimization: {qs_initial}")
proc0_print(f"Total objective before optimization: {float(np.sum(np.asarray(residuals(jnp.asarray(x0))) ** 2))}")

t_solve = time.perf_counter()
result = least_squares_jax_solve(
    residuals,
    x0,
    method=OUTER_METHOD,
    max_nfev=MAX_NFEV,
    gtol=1e-7,
    x_scale=x_scale,
    step_size=OUTER_STEP_SIZE if OUTER_METHOD == "gradient_descent" else 1e-2,
    jit=False,
)
if PRINT_TIMINGS:
    proc0_print(f"Outer solve time: {time.perf_counter() - t_solve:.3f}s")

x_opt = result["x"]
surf.set_free_params(x_opt)
state_opt = vmec._solve_state(jnp.asarray(x_opt))
qs_final = float(np.asarray(qs.total_from_state(state_opt)))
aspect_final = float(np.asarray(vmec.aspect_equilibrium_from_state_jax(state_opt)))
total_final = float(np.sum(np.asarray(qs.residuals_from_state(state_opt)) ** 2) + (aspect_final - 7.0) ** 2)

proc0_print(f"Final aspect ratio: {aspect_final}")
proc0_print(f"Quasisymmetry objective after optimization: {qs_final}")
proc0_print(f"Total objective after optimization: {total_final}")

proc0_print("End of 2_Intermediate/QH_fixed_resolution_jax.py")
proc0_print("================================================")
