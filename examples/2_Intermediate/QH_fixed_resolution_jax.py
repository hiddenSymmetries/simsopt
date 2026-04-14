#!/usr/bin/env python

import argparse
import os
import time

import numpy as np

from simsopt.mhd import VmecJax
from simsopt.solve import build_vmec_objective_stage, least_squares_jax_solve
from simsopt.util import proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume using autodiff through vmec_jax.

Run this example with:
  python QH_fixed_resolution_jax.py
"""

DEFAULT_MAX_NFEV = 10
DEFAULT_MAX_MODE = 1
DEFAULT_METHOD = "gauss_newton"
DEFAULT_STEP_SIZE = 5e-5
DEFAULT_JIT = True
DEFAULT_ADJOINT_MODE = "chunked"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    parser.add_argument("--max-mode", type=int, default=DEFAULT_MAX_MODE)
    parser.add_argument(
        "--method",
        choices=[
            "gradient_descent",
            "scipy",
            "gauss_newton",
            "truncated_gauss_newton",
            "trust_region",
            "levenberg_marquardt",
            "lbfgs",
        ],
        default=DEFAULT_METHOD,
    )
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--adjoint-mode", choices=["lineax", "auto", "chunked", "dense"], default=DEFAULT_ADJOINT_MODE)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=DEFAULT_JIT)
    parser.add_argument("--stateless-evaluations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--vmec-max-iter", type=int, default=None)
    parser.add_argument("--vmec-grad-tol", type=float, default=None)
    parser.add_argument("--implicit-cg-max-iter", type=int, default=None)
    parser.add_argument("--implicit-cg-tol", type=float, default=None)
    return parser.parse_args()


def objective_value(stage, x):
    residual = np.asarray(stage.residuals(x), dtype=float)
    return float(np.dot(residual, residual))


def main():
    args = parse_args()
    max_mode = int(args.max_mode)
    max_nfev = int(args.max_nfev)
    method = str(args.method).strip().lower()

    proc0_print("Running 2_Intermediate/QH_fixed_resolution_jax.py")
    proc0_print("==================================================")

    filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
    vmec = VmecJax(filename, verbose=False)
    vmec.indata.mpol = max_mode + 2
    vmec.indata.ntor = max_mode + 2
    vmec.use_residual_autodiff_defaults(
        outer_method=method,
        residual_adjoint_mode=str(args.adjoint_mode).strip().lower(),
        stateless_evaluations=bool(args.stateless_evaluations),
        optimization_profile="qh",
    )
    vmec.set_solver_options(
        max_iter=args.vmec_max_iter,
        grad_tol=args.vmec_grad_tol,
        implicit_cg_max_iter=args.implicit_cg_max_iter,
        implicit_cg_tol=args.implicit_cg_tol,
    )

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")

    stage = build_vmec_objective_stage(
        vmec,
        max_mode=max_mode,
        objective_tuples=[("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)],
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    qs = stage.extras["qs"]

    proc0_print("Parameter space:", stage.free_names)
    initial_state = vmec._solve_state(stage.x0)

    proc0_print(
        "Solver settings:",
        {
            "method": method,
            "step_size": float(args.step_size),
            "vmec_max_iter": int(vmec._max_iter),
            "vmec_grad_tol": float(vmec._grad_tol),
            "implicit_cg_max_iter": int(vmec._implicit_cg_max_iter),
            "implicit_cg_tol": float(vmec._implicit_cg_tol),
            "implicit_damping": float(vmec._implicit_damping),
            "jit": bool(args.jit),
            "adjoint_mode": str(vmec._residual_adjoint_mode),
            "tangent_mode": str(vmec._residual_tangent_mode),
            "stateless_evaluations": bool(vmec._stateless_evaluations),
        },
    )

    proc0_print("Quasisymmetry objective before optimization:", float(np.asarray(qs.total_from_state(initial_state))))
    proc0_print("Initial aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(initial_state))))
    proc0_print("Total objective before optimization:", objective_value(stage, stage.x0))

    solve_start = time.perf_counter()
    result = least_squares_jax_solve(
        stage.residuals,
        stage.x0,
        method=method,
        max_nfev=max_nfev,
        gtol=1e-7,
        step_size=float(args.step_size),
        x_scale=stage.x_scale,
        jit=bool(args.jit),
        verbose=1,
        jac="jax" if method == "scipy" else None,
        profile=bool(args.profile),
    )
    solve_elapsed = time.perf_counter() - solve_start

    surf.set_free_params(result["x"])
    state_opt = vmec._solve_state(result["x"])

    proc0_print("Final aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(state_opt))))
    proc0_print("Quasisymmetry objective after optimization:", float(np.asarray(qs.total_from_state(state_opt))))
    proc0_print("Total objective after optimization:", objective_value(stage, result["x"]))
    proc0_print("Result summary:", {key: result.get(key) for key in ("success", "status", "nfev")})
    if args.profile:
        proc0_print("Solver profile:", result.get("profile"))
    if args.timings:
        proc0_print("Solve wall time [s]:", solve_elapsed)

    proc0_print("End of 2_Intermediate/QH_fixed_resolution_jax.py")
    proc0_print("================================================")


if __name__ == "__main__":
    main()
