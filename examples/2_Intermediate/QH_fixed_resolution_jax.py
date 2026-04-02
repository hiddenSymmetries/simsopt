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
throughout the volume, using autodiff and the VMEC-only quasisymmetry
diagnostic.

Run this example with:
  python QH_fixed_resolution_jax.py
"""

DEFAULT_MAX_NFEV = 10
DEFAULT_MAX_MODE = 1
DEFAULT_STATELESS_EVALUATIONS = False
DEFAULT_WALL_CLOCK_BUDGET = 0.0

REFERENCE_CONFIGS = {
    "qh": {
        "label": "QH",
        "input": "input.nfp4_QH_warm_start",
        "helicity_m": 1,
        "helicity_n": -1,
        "aspect_target": 7.0,
        "mean_iota_target": None,
        "default_method": "gradient_descent",
        "default_step_size": 5e-5,
        "default_adjoint_mode": "chunked",
        "default_jit": False,
    },
    "qa": {
        "label": "QA",
        "input": "input.nfp2_QA",
        "helicity_m": 1,
        "helicity_n": 0,
        "aspect_target": 2.0,
        "mean_iota_target": 0.44,
        "default_method": "truncated_gauss_newton",
        "default_step_size": 1.0,
        "default_adjoint_mode": "lineax",
        "default_jit": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    parser.add_argument("--max-mode", type=int, default=DEFAULT_MAX_MODE)
    parser.add_argument("--reference", choices=sorted(REFERENCE_CONFIGS), default="qh")
    parser.add_argument("--iota-target", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=[
            "auto",
            "gradient_descent",
            "gauss_newton",
            "truncated_gauss_newton",
            "trust_region",
            "levenberg_marquardt",
            "scipy",
            "lbfgs",
        ],
        default="auto",
    )
    parser.add_argument("--step-size", type=float, default=None)
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--adjoint-mode", choices=["lineax", "auto", "chunked", "dense"], default=None)
    parser.add_argument("--stateless-evaluations", action="store_true")
    parser.add_argument("--wall-clock-budget", type=float, default=DEFAULT_WALL_CLOCK_BUDGET)
    return parser.parse_args()


def objective_tuples_for_reference(reference_key, *, iota_target_override=None):
    ref = REFERENCE_CONFIGS[str(reference_key).strip().lower()]
    objective_tuples = [
        ("aspect", float(ref["aspect_target"]), 1.0),
        ("qs", 0.0, 1.0),
    ]
    mean_iota_target = ref["mean_iota_target"]
    if iota_target_override is not None:
        mean_iota_target = float(iota_target_override)
    if mean_iota_target is not None:
        objective_tuples.append(("mean_iota", float(mean_iota_target), 1.0))
    return objective_tuples


def build_stage(vmec, *, max_mode, reference_key, objective_tuples):
    ref = REFERENCE_CONFIGS[str(reference_key).strip().lower()]
    return build_vmec_objective_stage(
        vmec,
        max_mode=int(max_mode),
        objective_tuples=objective_tuples,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=int(ref["helicity_m"]),
        helicity_n=int(ref["helicity_n"]),
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )


def objective_value(stage, x_free):
    residual = np.asarray(stage.residuals(x_free), dtype=float)
    return float(np.dot(residual, residual))


def main():
    args = parse_args()
    reference_key = str(args.reference).strip().lower()
    ref = REFERENCE_CONFIGS[reference_key]
    max_nfev = int(args.max_nfev)
    max_mode = int(args.max_mode)
    outer_method = ref["default_method"] if str(args.method).strip().lower() == "auto" else str(args.method).strip().lower()
    outer_step_size = float(ref["default_step_size"] if args.step_size is None else args.step_size)
    adjoint_mode = ref["default_adjoint_mode"] if args.adjoint_mode is None else str(args.adjoint_mode).strip().lower()
    use_jit = bool(ref["default_jit"]) if args.jit is None else bool(args.jit)
    objective_tuples = objective_tuples_for_reference(reference_key, iota_target_override=args.iota_target)

    proc0_print("Running 2_Intermediate/QH_fixed_resolution_jax.py")
    proc0_print("==================================================")

    filename = os.path.join(os.path.dirname(__file__), "inputs", ref["input"])
    vmec = VmecJax(filename, verbose=False)
    vmec.use_residual_autodiff_defaults(
        outer_method=outer_method,
        residual_adjoint_mode=adjoint_mode,
        stateless_evaluations=args.stateless_evaluations,
    )

    stage = build_stage(
        vmec,
        max_mode=max_mode,
        reference_key=reference_key,
        objective_tuples=objective_tuples,
    )
    surf = stage.extras["surf"]
    qs = stage.extras["qs"]
    initial_state = vmec._solve_state(stage.x0)

    proc0_print("Parameter space:", stage.free_names)
    proc0_print("Objectives:", objective_tuples)
    proc0_print(
        "Reference setup:",
        {
            "reference": ref["label"],
            "input": ref["input"],
            "helicity": (int(ref["helicity_m"]), int(ref["helicity_n"])),
            "max_mode": max_mode,
            "vmec_mpol": max_mode + 2,
            "vmec_ntor": max_mode + 2,
        },
    )
    proc0_print(
        "Solver settings:",
        {
            "method": outer_method,
            "step_size": outer_step_size,
            "vmec_max_iter": int(vmec._max_iter),
            "vmec_grad_tol": float(vmec._grad_tol),
            "jit": use_jit,
            "adjoint_mode": adjoint_mode,
            "stateless_evaluations": bool(args.stateless_evaluations),
            "wall_clock_budget_s": float(args.wall_clock_budget),
        },
    )

    if qs is not None:
        proc0_print("Quasisymmetry objective before optimization:", float(np.asarray(qs.total_from_state(initial_state))))
    proc0_print("Initial aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(initial_state))))
    proc0_print("Initial mean iota:", float(np.asarray(vmec.mean_iota_from_state_jax(initial_state))))
    proc0_print("Total objective before optimization:", objective_value(stage, stage.x0))

    solve_start = time.perf_counter()
    result = least_squares_jax_solve(
        stage.residuals,
        stage.x0,
        method=outer_method,
        max_nfev=max_nfev,
        gtol=1e-7,
        step_size=outer_step_size,
        x_scale=stage.x_scale,
        jit=use_jit,
        verbose=1,
        jac="jax" if outer_method == "scipy" else None,
        profile=bool(args.profile),
        wall_clock_limit_s=float(args.wall_clock_budget) if float(args.wall_clock_budget) > 0.0 else None,
    )
    solve_elapsed = time.perf_counter() - solve_start

    surf.set_free_params(result["x"])
    state_opt = vmec._solve_state(result["x"])

    proc0_print("Final aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(state_opt))))
    if qs is not None:
        proc0_print("Quasisymmetry objective after optimization:", float(np.asarray(qs.total_from_state(state_opt))))
    if any(str(name).strip().lower() in ("mean_iota", "iota") for name, _, _ in objective_tuples):
        proc0_print("Mean iota after optimization:", float(np.asarray(vmec.mean_iota_from_state_jax(state_opt))))
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
