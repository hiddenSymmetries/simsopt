#!/usr/bin/env python

import argparse
import os
import time

import numpy as np

from simsopt.mhd import VmecJax
from simsopt.solve import build_vmec_objective_stage, least_squares_jax_solve
from simsopt.util import proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-axisymmetry (M=1, N=0)
throughout the volume, using autodiff and the VMEC-only quasisymmetry
diagnostic.

Run this example with:
  python QA_fixed_resolution_jax.py
"""

DEFAULT_MAX_NFEV = 10
DEFAULT_MAX_MODE = 1
DEFAULT_METHOD = "scipy"
DEFAULT_JAC = "jax"
DEFAULT_RESIDUAL_DERIVATIVE_BACKEND = "discrete_adjoint"
DEFAULT_JIT = True
DEFAULT_STEP_SIZE = 1.0
DEFAULT_ADJOINT_MODE = "lineax"
DEFAULT_ASPECT_TARGET = 2.0
DEFAULT_IOTA_TARGET = 0.41


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
    parser.add_argument("--jac", choices=["jax", "reverse", "2-point", "3-point"], default=DEFAULT_JAC)
    parser.add_argument(
        "--residual-derivative-backend",
        choices=["implicit", "discrete_adjoint"],
        default=DEFAULT_RESIDUAL_DERIVATIVE_BACKEND,
    )
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=DEFAULT_JIT)
    parser.add_argument("--stateless-evaluations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--aspect-target", type=float, default=DEFAULT_ASPECT_TARGET)
    parser.add_argument("--iota-target", type=float, default=DEFAULT_IOTA_TARGET)
    parser.add_argument("--vmec-max-iter", type=int, default=None)
    parser.add_argument("--vmec-grad-tol", type=float, default=None)
    parser.add_argument("--implicit-cg-max-iter", type=int, default=None)
    parser.add_argument("--implicit-cg-tol", type=float, default=None)
    return parser.parse_args()


def objective_value_from_state(vmec, qs, state, *, aspect_target, iota_target):
    aspect_residual = np.asarray([vmec.aspect_equilibrium_from_state_jax(state) - float(aspect_target)], dtype=float)
    iota_residual = np.asarray([vmec.mean_iota_from_state_jax(state) - float(iota_target)], dtype=float)
    qs_residual = np.asarray(qs.residuals_from_state(state), dtype=float)
    residual = np.concatenate((aspect_residual, iota_residual, qs_residual))
    return float(np.dot(residual, residual))


def solve_state_for_report(vmec, x):
    if str(getattr(vmec, "_residual_derivative_backend", "")) == "discrete_adjoint":
        step_size = (
            float(vmec._step_size_override)
            if getattr(vmec, "_step_size_override", None) is not None
            else float(vmec._indata_raw.get_float("DELT", 1.0))
        )
        return vmec._solve_state_residual_forward(x, step_size=step_size)
    return vmec._solve_state(x)


def main():
    args = parse_args()
    max_mode = int(args.max_mode)
    max_nfev = int(args.max_nfev)
    method = str(args.method).strip().lower()
    jac_mode = str(args.jac).strip().lower()
    aspect_target = float(args.aspect_target)
    iota_target = float(args.iota_target)

    proc0_print("Running 2_Intermediate/QA_fixed_resolution_jax.py")
    proc0_print("=================================================")

    filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp2_QA")
    vmec = VmecJax(filename, verbose=False)
    vmec.indata.mpol = max_mode + 2
    vmec.indata.ntor = max_mode + 2
    vmec.use_residual_autodiff_defaults(
        outer_method=method,
        residual_adjoint_mode=str(args.adjoint_mode).strip().lower(),
        stateless_evaluations=bool(args.stateless_evaluations),
    )
    vmec.set_solver_options(
        max_iter=args.vmec_max_iter,
        grad_tol=args.vmec_grad_tol,
        implicit_cg_max_iter=args.implicit_cg_max_iter,
        implicit_cg_tol=args.implicit_cg_tol,
        residual_derivative_backend=str(args.residual_derivative_backend).strip().lower(),
    )

    stage = build_vmec_objective_stage(
        vmec,
        max_mode=max_mode,
        objective_tuples=[
            ("aspect", aspect_target, 1.0),
            ("mean_iota", iota_target, 1.0),
            ("qs", 0.0, 1.0),
        ],
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=0,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    surf = stage.extras["surf"]
    qs = stage.extras["qs"]
    solve_jit = bool(args.jit)
    if (
        method == "scipy"
        and jac_mode in ("reverse", "rev", "jacrev")
        and str(vmec._residual_derivative_backend) == "discrete_adjoint"
    ):
        solve_jit = False

    initial_state = solve_state_for_report(vmec, stage.x0)

    proc0_print("Parameter space:", stage.free_names)
    proc0_print(
        "Objective tuples:",
        [("aspect", aspect_target, 1.0), ("mean_iota", iota_target, 1.0), ("qs", 0.0, 1.0)],
    )
    proc0_print(
        "Reference setup:",
        {
            "reference": "QA",
            "input": "input.nfp2_QA",
            "helicity": (1, 0),
            "max_mode": max_mode,
            "vmec_mpol": max_mode + 2,
            "vmec_ntor": max_mode + 2,
        },
    )
    proc0_print(
        "Solver settings:",
        {
            "method": method,
            "jac": jac_mode,
            "residual_derivative_backend": str(vmec._residual_derivative_backend),
            "vmec_max_iter": int(vmec._max_iter),
            "vmec_grad_tol": float(vmec._grad_tol),
            "implicit_cg_max_iter": int(vmec._implicit_cg_max_iter),
            "implicit_cg_tol": float(vmec._implicit_cg_tol),
            "jit": solve_jit,
            "adjoint_mode": str(vmec._residual_adjoint_mode),
            "stateless_evaluations": bool(vmec._stateless_evaluations),
        },
    )

    proc0_print("Quasisymmetry objective before optimization:", float(np.asarray(qs.total_from_state(initial_state))))
    proc0_print("Initial aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(initial_state))))
    proc0_print("Initial mean iota:", float(np.asarray(vmec.mean_iota_from_state_jax(initial_state))))
    proc0_print(
        "Total objective before optimization:",
        objective_value_from_state(vmec, qs, initial_state, aspect_target=aspect_target, iota_target=iota_target),
    )

    solve_start = time.perf_counter()
    result = least_squares_jax_solve(
        stage.residuals,
        stage.x0,
        method=method,
        max_nfev=max_nfev,
        gtol=1e-7,
        step_size=float(args.step_size),
        x_scale=stage.x_scale,
        jit=solve_jit,
        verbose=1,
        jac=jac_mode if method == "scipy" else None,
        profile=bool(args.profile),
    )
    solve_elapsed = time.perf_counter() - solve_start

    surf.set_free_params(result["x"])
    state_opt = solve_state_for_report(vmec, result["x"])

    proc0_print("Final aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(state_opt))))
    proc0_print("Mean iota after optimization:", float(np.asarray(vmec.mean_iota_from_state_jax(state_opt))))
    proc0_print("Quasisymmetry objective after optimization:", float(np.asarray(qs.total_from_state(state_opt))))
    proc0_print(
        "Total objective after optimization:",
        objective_value_from_state(vmec, qs, state_opt, aspect_target=aspect_target, iota_target=iota_target),
    )
    proc0_print("Result summary:", {key: result.get(key) for key in ("success", "status", "nfev")})
    if args.profile:
        proc0_print("Solver profile:", result.get("profile"))
    if args.timings:
        proc0_print("Solve wall time [s]:", solve_elapsed)

    proc0_print("End of 2_Intermediate/QA_fixed_resolution_jax.py")
    proc0_print("===============================================")


if __name__ == "__main__":
    main()
