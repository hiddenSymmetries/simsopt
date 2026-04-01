#!/usr/bin/env python

import argparse
import os
import time

import numpy as np

from simsopt.mhd import VmecJax
from simsopt.solve import build_vmec_objective_stage, run_continuation_jax_solve
from simsopt.util import proc0_print

"""
Optimize a VMEC-JAX equilibrium for quasi-helical symmetry (M=1, N=1)
throughout the volume, using autodiff and the VMEC-only quasisymmetry diagnostic.
"""

DEFAULT_MAX_NFEV = 10
DEFAULT_MAX_MODE = 1
DEFAULT_OUTER_METHOD = "auto"
DEFAULT_OUTER_STEP_SIZE = None
DEFAULT_JIT = True

# Keep the top-level controls close to QH_fixed_resolution.py.
DEFAULT_USE_RESOLUTION_CONTINUATION = True
DEFAULT_ADJOINT_MODE = "lineax"
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
        "default_step_size": 5e-7,
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
    },
}


def configure_vmec_surface(vmec, max_mode):
    vmec.indata.mpol = int(max_mode) + 2
    vmec.indata.ntor = vmec.indata.mpol

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=int(max_mode), nmin=-int(max_mode), nmax=int(max_mode), fixed=False)
    surf.fix("rc(0,0)")
    return surf


def resolve_reference_setup(vmec, *, reference_name, max_mode, iota_target_override=None):
    reference_key = str(reference_name).strip().lower()
    ref = dict(REFERENCE_CONFIGS[reference_key])
    surf = configure_vmec_surface(vmec, max_mode)
    x0 = np.asarray(surf.get_free_params(), dtype=float)
    state0 = vmec._solve_state(x0)

    mean_iota_target = ref["mean_iota_target"]
    if iota_target_override is not None:
        mean_iota_target = float(iota_target_override)
    objective_tuples = [
        ("aspect", float(ref["aspect_target"]), 1.0),
        ("qs", 0.0, 1.0),
    ]
    if reference_key != "qh" and mean_iota_target is not None:
        objective_tuples.append(("mean_iota", float(mean_iota_target), 1.0))
    return {
        "reference_name": str(reference_name).strip().lower(),
        "label": ref["label"],
        "input": ref["input"],
        "helicity_m": int(ref["helicity_m"]),
        "helicity_n": int(ref["helicity_n"]),
        "objective_tuples": objective_tuples,
        "state0": state0,
        "surf": surf,
    }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=DEFAULT_MAX_NFEV)
    parser.add_argument("--max-mode", type=int, default=DEFAULT_MAX_MODE)
    parser.add_argument("--reference", choices=sorted(REFERENCE_CONFIGS), default="qh")
    parser.add_argument("--iota-target", type=float, default=None)
    parser.add_argument(
        "--method",
        choices=["auto", "gradient_descent", "gauss_newton", "truncated_gauss_newton", "trust_region", "levenberg_marquardt", "scipy", "lbfgs"],
        default=DEFAULT_OUTER_METHOD,
    )
    parser.add_argument("--step-size", type=float, default=DEFAULT_OUTER_STEP_SIZE)
    parser.add_argument("--no-resolution-continuation", action="store_true")
    parser.add_argument("--timings", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=DEFAULT_JIT)
    parser.add_argument("--adjoint-mode", choices=["lineax", "auto"], default=DEFAULT_ADJOINT_MODE)
    parser.add_argument("--stateless-evaluations", action="store_true")
    parser.add_argument("--wall-clock-budget", type=float, default=DEFAULT_WALL_CLOCK_BUDGET)
    return parser.parse_args()


def main():
    args = parse_args()
    max_nfev = int(args.max_nfev)
    max_mode = int(args.max_mode)
    use_resolution_continuation = DEFAULT_USE_RESOLUTION_CONTINUATION and not args.no_resolution_continuation
    reference_name = str(args.reference).strip().lower()
    reference = REFERENCE_CONFIGS[reference_name]
    outer_method = reference["default_method"] if str(args.method).strip().lower() == "auto" else str(args.method).strip().lower()
    outer_step_size = float(reference["default_step_size"] if args.step_size is None else args.step_size)

    proc0_print("Running 2_Intermediate/QH_fixed_resolution_jax.py")
    proc0_print("==================================================")

    filename = os.path.join(os.path.dirname(__file__), "inputs", reference["input"])
    vmec = VmecJax(filename, verbose=False)
    vmec.use_residual_autodiff_defaults(
        outer_method=outer_method,
        residual_adjoint_mode=args.adjoint_mode,
        stateless_evaluations=args.stateless_evaluations,
    )

    setup = resolve_reference_setup(
        vmec,
        reference_name=reference_name,
        max_mode=max_mode,
        iota_target_override=args.iota_target,
    )
    objective_tuples = setup["objective_tuples"]
    initial_state = setup["state0"]

    if use_resolution_continuation and max_mode > 1:
        continuation_modes = list(range(1, max_mode + 1))
    else:
        continuation_modes = [max_mode]

    surfaces = np.arange(0, 1.01, 0.1)

    def stage_builder(stage_max_mode):
        return build_vmec_objective_stage(
            vmec,
            max_mode=stage_max_mode,
            objective_tuples=objective_tuples,
            surfaces=surfaces,
            helicity_m=setup["helicity_m"],
            helicity_n=setup["helicity_n"],
            x_scale_alpha=1.2,
            x_scale_min=1e-9,
        )

    stage = stage_builder(max_mode)
    surf = stage.extras["surf"]
    qs = stage.extras["qs"]

    proc0_print("Parameter space:", stage.free_names)
    proc0_print("Objectives:", objective_tuples)
    proc0_print(
        "Reference setup:",
        {
            "reference": setup["label"],
            "input": reference["input"],
            "helicity": (setup["helicity_m"], setup["helicity_n"]),
            "max_mode": int(max_mode),
            "vmec_mpol": int(max_mode) + 2,
            "vmec_ntor": int(max_mode) + 2,
        },
    )
    proc0_print("Continuation modes:", continuation_modes)
    proc0_print(
        "Solver settings:",
        {
            "method": outer_method,
            "step_size": outer_step_size,
            "vmec_max_iter": int(vmec._max_iter),
            "vmec_grad_tol": float(vmec._grad_tol),
            "jit": bool(args.jit),
            "adjoint_mode": args.adjoint_mode,
            "stateless_evaluations": bool(args.stateless_evaluations),
            "wall_clock_budget_s": float(args.wall_clock_budget),
        },
    )

    if qs is not None:
        proc0_print("Quasisymmetry objective before optimization:", float(np.asarray(qs.total_from_state(initial_state))))
    proc0_print("Initial aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(initial_state))))
    proc0_print("Initial mean iota:", float(np.asarray(vmec.mean_iota_from_state_jax(initial_state))))
    proc0_print("Total objective before optimization:", float(np.sum(np.asarray(stage.residuals(stage.x0)) ** 2)))

    solve_start = time.perf_counter()
    solve_result = run_continuation_jax_solve(
        stage_builder,
        continuation_modes,
        method=outer_method,
        max_nfev=max_nfev,
        gtol=1e-7,
        step_size=outer_step_size,
        jit=args.jit,
        verbose=1,
        jac="jax" if outer_method == "scipy" else None,
        profile=args.profile,
        wall_clock_limit_s=args.wall_clock_budget if float(args.wall_clock_budget) > 0.0 else None,
    )
    solve_elapsed = time.perf_counter() - solve_start

    surf.set_free_params(solve_result["result"]["x"])
    state_opt = vmec._solve_state(solve_result["result"]["x"])

    proc0_print("Final aspect ratio:", float(np.asarray(vmec.aspect_equilibrium_from_state_jax(state_opt))))
    if qs is not None:
        proc0_print("Quasisymmetry objective after optimization:", float(np.asarray(qs.total_from_state(state_opt))))
    if any(str(name).strip().lower() in ("mean_iota", "iota") for name, _, _ in objective_tuples):
        proc0_print("Mean iota after optimization:", float(np.asarray(vmec.mean_iota_from_state_jax(state_opt))))
    proc0_print("Total objective after optimization:", float(np.sum(np.asarray(stage.residuals(solve_result["result"]["x"])) ** 2)))
    proc0_print("Result summary:", {key: solve_result["result"].get(key) for key in ("success", "status", "nfev")})
    if args.profile:
        proc0_print("Solver profile:", solve_result["result"].get("profile"))
    if args.timings:
        proc0_print("Solve wall time [s]:", solve_elapsed)

    proc0_print("End of 2_Intermediate/QH_fixed_resolution_jax.py")
    proc0_print("================================================")


if __name__ == "__main__":
    main()
