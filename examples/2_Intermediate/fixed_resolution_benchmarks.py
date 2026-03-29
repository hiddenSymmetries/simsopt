#!/usr/bin/env python

import argparse
import os
import signal
import time

import numpy as np

from simsopt.mhd import QuasisymmetryRatioResidual, Vmec, VmecJax
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import build_vmec_objective_stage, run_continuation_jax_solve
from simsopt.util import MpiPartition, proc0_print


class BenchmarkTimeout(Exception):
    pass


class stage_timeout:
    def __init__(self, seconds):
        self.seconds = int(seconds)
        self.previous_handler = None

    def __enter__(self):
        if self.seconds <= 0:
            return self

        def _handle_timeout(signum, frame):
            raise BenchmarkTimeout(f"timed out after {self.seconds} seconds")

        self.previous_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self.previous_handler)
        return False


def mode_weighted_x_scale(dof_names, alpha=1.2, min_scale=1e-9):
    import re

    x_scale = np.ones(len(dof_names))
    pattern = r"[rz][cs]\((\d+),(-?\d+)\)"
    for index, name in enumerate(dof_names):
        match = re.search(pattern, name)
        if match is None:
            continue
        mode_level = max(abs(int(match.group(1))), abs(int(match.group(2))))
        scale = np.exp(-alpha * mode_level) / np.exp(-alpha)
        x_scale[index] = max(scale, min_scale)
    return x_scale


def summarize_case(name, payload):
    proc0_print(f"[{name}] {payload}")


def run_classic_qh(max_mode, max_nfev):
    mpi = MpiPartition()
    filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
    vmec = Vmec(filename, mpi=mpi, verbose=False)
    vmec.indata.mpol = max_mode + 2
    vmec.indata.ntor = vmec.indata.mpol
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7.0, 1.0), (qs.residuals, 0.0, 1.0)])

    start = time.perf_counter()
    total_before = prob.objective()
    qs_before = qs.total()
    from simsopt.solve import least_squares_mpi_solve

    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=max_nfev)
    elapsed = time.perf_counter() - start
    total_after = prob.objective()
    qs_after = qs.total()
    return {
        "problem": "classic_qh",
        "max_mode": int(max_mode),
        "max_nfev": int(max_nfev),
        "elapsed_s": elapsed,
        "qs_before": float(qs_before),
        "qs_after": float(qs_after),
        "total_before": float(total_before),
        "total_after": float(total_after),
    }


def run_classic_qa(max_mode, max_nfev):
    mpi = MpiPartition()
    filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp2_QA")
    vmec = Vmec(filename, mpi=mpi, verbose=False)
    vmec.indata.mpol = max_mode + 2
    vmec.indata.ntor = vmec.indata.mpol
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=max_mode, nmin=-max_mode, nmax=max_mode, fixed=False)
    surf.fix("rc(0,0)")
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=0)
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 2.0, 1.0), (qs.residuals, 0.0, 1.0), (vmec.mean_iota, 0.44, 1.0)])
    x_scale = mode_weighted_x_scale(prob.dof_names)

    start = time.perf_counter()
    total_before = prob.objective()
    qs_before = qs.total()
    from simsopt.solve import least_squares_mpi_solve

    least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-7, max_nfev=max_nfev, gtol=1e-7, x_scale=x_scale)
    elapsed = time.perf_counter() - start
    total_after = prob.objective()
    qs_after = qs.total()
    return {
        "problem": "classic_qa",
        "max_mode": int(max_mode),
        "max_nfev": int(max_nfev),
        "elapsed_s": elapsed,
        "qs_before": float(qs_before),
        "qs_after": float(qs_after),
        "total_before": float(total_before),
        "total_after": float(total_after),
    }


def run_jax_qh(max_mode, max_nfev, method, step_size, continuation, *, adjoint_mode="lineax", stateless_evaluations=False, profile=False, jit=False, wall_clock_budget_s=None):
    filename = os.path.join(os.path.dirname(__file__), "inputs", "input.nfp4_QH_warm_start")
    vmec = VmecJax(filename, verbose=False)
    vmec.use_residual_autodiff_defaults(
        outer_method=method,
        residual_adjoint_mode=adjoint_mode,
        stateless_evaluations=stateless_evaluations,
    )
    objective_tuples = [("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)]
    surfaces = np.arange(0, 1.01, 0.1)

    def stage_builder(stage_max_mode):
        return build_vmec_objective_stage(
            vmec,
            max_mode=stage_max_mode,
            objective_tuples=objective_tuples,
            surfaces=surfaces,
            helicity_m=1,
            helicity_n=-1,
            x_scale_alpha=1.2,
            x_scale_min=1e-9,
        )

    continuation_modes = list(range(1, max_mode + 1)) if continuation and max_mode > 1 else [max_mode]
    initial_stage = stage_builder(max_mode)
    initial_state = vmec._solve_state(initial_stage.x0)
    qs_before = float(np.asarray(initial_stage.extras["qs"].total_from_state(initial_state)))
    total_before = float(np.sum(np.asarray(initial_stage.residuals(initial_stage.x0)) ** 2))
    start = time.perf_counter()
    result = run_continuation_jax_solve(
        stage_builder,
        continuation_modes,
        method=method,
        max_nfev=max_nfev,
        gtol=1e-7,
        step_size=step_size,
        jit=jit,
        verbose=0,
        jac="jax" if method == "scipy" else None,
        profile=profile,
        wall_clock_limit_s=wall_clock_budget_s,
    )
    elapsed = time.perf_counter() - start
    x = result["result"]["x"]
    final_stage = stage_builder(max_mode)
    final_state = vmec._solve_state(x)
    qs_after = float(np.asarray(final_stage.extras["qs"].total_from_state(final_state)))
    total_after = float(np.sum(np.asarray(final_stage.residuals(x)) ** 2))
    return {
        "problem": "jax_qh",
        "max_mode": int(max_mode),
        "continuation_modes": continuation_modes,
        "method": method,
        "adjoint_mode": adjoint_mode,
        "stateless_evaluations": bool(stateless_evaluations),
        "step_size": float(step_size),
        "jit": bool(jit),
        "max_nfev": int(max_nfev),
        "elapsed_s": elapsed,
        "qs_before": qs_before,
        "qs_after": qs_after,
        "total_before": total_before,
        "total_after": total_after,
        "nfev": result["result"].get("nfev"),
        "success": result["result"].get("success"),
        "status": result["result"].get("status"),
        "wall_clock_budget_s": None if wall_clock_budget_s is None else float(wall_clock_budget_s),
        "wall_clock_exhausted": result["result"].get("wall_clock_exhausted"),
        "profile": result["result"].get("profile"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["classic_qh", "classic_qa", "jax_qh", "all"], default="all")
    parser.add_argument("--max-mode", type=int, default=2)
    parser.add_argument("--max-nfev", type=int, default=4)
    parser.add_argument(
        "--method",
        choices=["gradient_descent", "gauss_newton", "truncated_gauss_newton", "trust_region", "levenberg_marquardt", "scipy", "lbfgs"],
        default="gradient_descent",
    )
    parser.add_argument("--step-size", type=float, default=5e-7)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-continuation", action="store_true")
    parser.add_argument("--adjoint-mode", choices=["lineax", "auto"], default="lineax")
    parser.add_argument("--stateless-evaluations", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--jit", action="store_true")
    parser.add_argument("--wall-clock-budget", type=float, default=0.0)
    args = parser.parse_args()

    cases = []
    if args.problem in ("classic_qh", "all"):
        cases.append(("classic_qh", lambda: run_classic_qh(args.max_mode, args.max_nfev)))
    if args.problem in ("classic_qa", "all"):
        cases.append(("classic_qa", lambda: run_classic_qa(args.max_mode, args.max_nfev)))
    if args.problem in ("jax_qh", "all"):
        cases.append(
            (
                "jax_qh",
                lambda: run_jax_qh(
                    args.max_mode,
                    args.max_nfev,
                    args.method,
                    args.step_size,
                    not args.no_continuation,
                    adjoint_mode=args.adjoint_mode,
                    stateless_evaluations=args.stateless_evaluations,
                    profile=args.profile,
                    jit=args.jit,
                    wall_clock_budget_s=args.wall_clock_budget if float(args.wall_clock_budget) > 0.0 else None,
                ),
            )
        )

    for name, runner in cases:
        try:
            with stage_timeout(args.timeout):
                summarize_case(name, runner())
        except BenchmarkTimeout as exc:
            summarize_case(name, {"problem": name, "timed_out": True, "message": str(exc)})


if __name__ == "__main__":
    main()