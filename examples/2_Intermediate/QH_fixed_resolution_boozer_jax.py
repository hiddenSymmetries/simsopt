#!/usr/bin/env python

import argparse
import os
import re
import time

import numpy as np

from simsopt.mhd import VmecJax, BoozerJax, QuasisymmetryJax
from simsopt.solve import least_squares_jax_solve
from simsopt.util import proc0_print

import jax
import jax.numpy as jnp

"""
Optimize for quasi-helical symmetry (M=1, N=1) at a given radius using JAX.
"""

max_mode = 1
max_nfev = 10

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true", help="Reduce VMEC iterations for quick verification.")
parser.add_argument("--no-jit", action="store_true", help="Disable JIT (default is to JIT everything).")
parser.add_argument("--warm-start-iters", type=int, default=0, help="Optional explicit VMEC warm-start LBFGS iterations.")
parser.add_argument("--vmec-max-iter", type=int, default=None, help="Override VMEC iteration budget.")
parser.add_argument("--vmec-grad-tol", type=float, default=None, help="Override VMEC residual tolerance.")
parser.add_argument(
    "--solver",
    default="residual",
    choices=["residual", "initial", "gd", "lbfgs", "vmec2000"],
    help="VMEC-JAX solver mode. 'residual' matches the VMEC residual iteration.",
)
parser.add_argument(
    "--method",
    default="scipy",
    choices=["scipy", "lbfgs", "gradient_descent"],
    help="Outer least-squares solver.",
)
parser.add_argument(
    "--aspect-mode",
    default="boundary",
    choices=["boundary", "equilibrium"],
    help="Use the fast boundary aspect or the slower equilibrium aspect for the target.",
)
parser.add_argument(
    "--jacobian",
    default="jax",
    choices=["jax", "2-point", "3-point"],
    help="Jacobian mode for scipy least-squares.",
)
parser.add_argument("--timings", action="store_true", help="Print startup timings.")
args = parser.parse_args()
jacobian_mode = args.jacobian
if args.no_jit and jacobian_mode == "jax":
    jacobian_mode = "2-point"

proc0_print("Running 2_Intermediate/QH_fixed_resolution_boozer_jax.py")
proc0_print("========================================================")

t_start = time.perf_counter()
filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
vmec = VmecJax(filename, verbose=False)
top_level_jit = not args.no_jit
if args.aspect_mode == "equilibrium" or jacobian_mode != "jax":
    top_level_jit = False
vmec_opts = {"solver": args.solver, "warm_start_iters": int(args.warm_start_iters)}
if args.fast:
    vmec_opts["max_iter"] = 500
    vmec_opts["grad_tol"] = 1e-3
if args.vmec_max_iter is not None:
    vmec_opts["max_iter"] = int(args.vmec_max_iter)
if args.vmec_grad_tol is not None:
    vmec_opts["grad_tol"] = float(args.vmec_grad_tol)
vmec.set_solver_options(**vmec_opts)
vmec.indata.mpol = max_mode + 2
vmec.indata.ntor = vmec.indata.mpol

# Define parameter space:
surf = vmec.boundary
surf.fix_all()
surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
surf.fix("rc(0,0)")  # Major radius


def create_simsopt_x_scale(dof_names, alpha=1.6, min_scale=1e-6):
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


# Configure quasisymmetry objective:
qs = QuasisymmetryJax(BoozerJax(vmec, mpol=16, ntor=16, jit=True),
                      0.5,  # Radius to target
                      1, 1)  # (M, N) you want in |B|


def residuals(x_free):
    if args.aspect_mode == "equilibrium":
        aspect = jnp.asarray(vmec.aspect_equilibrium(x_free))
    else:
        aspect = vmec.aspect_jax(x_free)
    qs_res = qs.J(x_free)
    return jnp.concatenate([jnp.array([aspect - 7.0]), qs_res])


x0 = surf.get_free_params()
free_names = [name for name, free in zip(surf.dof_names, surf.free) if free]
x_scale = create_simsopt_x_scale(free_names, alpha=1.2, min_scale=1e-9)
if args.timings:
    proc0_print(f"Startup setup time: {time.perf_counter() - t_start:.3f}s")

residuals_jit = residuals
if top_level_jit:
    qs.prepare(jnp.asarray(x0))
    try:
        residuals_jit = jax.jit(residuals)
        _ = residuals_jit(jnp.asarray(x0)).block_until_ready()
    except Exception as exc:
        raise RuntimeError(
            f"JIT failed ({type(exc).__name__}: {exc}). "
            "Use --no-jit to run without compilation."
        ) from exc

proc0_print("Evaluating initial quasisymmetry objective...")
t_qs0 = time.perf_counter()
qs_initial = np.sum(np.asarray(qs.J(jnp.asarray(x0))) ** 2)
proc0_print(f"Quasisymmetry objective (sum of squares) before optimization: {qs_initial}")
if args.timings:
    proc0_print(f"Initial QS evaluation time: {time.perf_counter() - t_qs0:.3f}s")

# To keep this example fast, we stop after the first function evaluation.
t_solve = time.perf_counter()
result = least_squares_jax_solve(
    lambda x: residuals_jit(x),
    x0,
    method=args.method,
    max_nfev=max_nfev,
    gtol=1e-7,
    x_scale=x_scale,
    step_size=1e-2,
    jit=top_level_jit and jacobian_mode == "jax" and args.aspect_mode != "equilibrium",
    jac=jacobian_mode if args.method == "scipy" else None,
)
if args.timings:
    proc0_print(f"Outer solve time: {time.perf_counter() - t_solve:.3f}s")

x_opt = result["x"]
surf.set_free_params(x_opt)

qs_final = np.sum(np.asarray(qs.J(jnp.asarray(x_opt))) ** 2)

final_aspect = vmec.aspect_equilibrium(x_opt) if args.aspect_mode == "equilibrium" else vmec.aspect(x_opt)
proc0_print(f"Final aspect ratio is {final_aspect}")
proc0_print(f"Quasisymmetry objective (sum of squares) after optimization: {qs_final}")

proc0_print("End of 2_Intermediate/QH_fixed_resolution_boozer_jax.py")
proc0_print("========================================================")
