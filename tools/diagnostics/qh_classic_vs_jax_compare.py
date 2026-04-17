#!/usr/bin/env python

import argparse
import gc
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from scipy.optimize import least_squares

from simsopt._core.finite_difference import FiniteDifference
from simsopt.mhd import QuasisymmetryRatioResidual, Vmec, VmecJax
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import build_vmec_objective_stage


DEFAULT_OUTPUT_DIR = Path("/Users/rogeriojorge/local/tests/qh_compare_outputs")
WALL_CLOCK_LIMIT_S = 300.0
MAX_NFEV = 20


class WallClockStop(RuntimeError):
    pass


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(value)
    return int(value) * 1024


def current_rss_bytes():
    return int(psutil.Process(os.getpid()).memory_info().rss)


def qh_input_filename():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "examples",
        "2_Intermediate",
        "inputs",
        "input.nfp4_QH_warm_start",
    )


def check_wall_clock(start_time, wall_clock_limit_s):
    elapsed = time.perf_counter() - start_time
    if elapsed > float(wall_clock_limit_s):
        raise WallClockStop(f"timed out after {wall_clock_limit_s:.1f} s")
    return elapsed


def make_classic_context(max_mode):
    filename = qh_input_filename()
    vmec = Vmec(filename, verbose=False)
    vmec.indata.mpol = int(max_mode) + 2
    vmec.indata.ntor = vmec.indata.mpol
    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=int(max_mode), nmin=-int(max_mode), nmax=int(max_mode), fixed=False)
    surf.fix("rc(0,0)")
    qs = QuasisymmetryRatioResidual(vmec, np.arange(0, 1.01, 0.1), helicity_m=1, helicity_n=-1)
    prob = LeastSquaresProblem.from_tuples([(vmec.aspect, 7.0, 1.0), (qs.residuals, 0.0, 1.0)])
    x0 = np.asarray(surf.x, dtype=float)
    return {
        "solver": "classic",
        "vmec": vmec,
        "surf": surf,
        "qs": qs,
        "prob": prob,
        "x0": x0,
        "x_scale": 1.0,
    }


def make_jax_context(max_mode):
    filename = qh_input_filename()
    vmec = VmecJax(filename, verbose=False)
    vmec.use_residual_autodiff_defaults(
        outer_method="scipy",
        residual_adjoint_mode="chunked",
        stateless_evaluations=False,
        optimization_profile="qh",
    )
    vmec.set_solver_options(residual_derivative_backend="discrete_adjoint")
    objective_tuples = [("aspect", 7.0, 1.0), ("qs", 0.0, 1.0)]
    stage = build_vmec_objective_stage(
        vmec,
        max_mode=int(max_mode),
        objective_tuples=objective_tuples,
        surfaces=np.arange(0, 1.01, 0.1),
        helicity_m=1,
        helicity_n=-1,
        x_scale_alpha=1.2,
        x_scale_min=1e-9,
    )
    return {
        "solver": "jax",
        "vmec": vmec,
        "stage": stage,
        "qs": stage.extras["qs"],
        "residuals_from_state": stage.extras["residuals_from_state"],
        "x0": np.asarray(stage.x0, dtype=float),
        "x_scale": np.asarray(stage.x_scale, dtype=float),
    }


def classic_metrics(context, x):
    context["prob"].x = np.asarray(x, dtype=float)
    total = float(context["prob"].objective())
    qs_objective = float(context["qs"].total())
    aspect_value = float(context["vmec"].aspect())
    aspect_term = float((aspect_value - 7.0) ** 2)
    return {
        "total_objective": total,
        "qs_objective": qs_objective,
        "aspect_term": aspect_term,
        "aspect_value": aspect_value,
    }


def jax_metrics(context, x):
    x = np.asarray(x, dtype=float)
    state_payload = getattr(context["stage"].residuals, "scipy_state_payload", None)
    if callable(state_payload):
        state, _payload, residual = state_payload(x)
        residual = np.asarray(residual, dtype=float)
    else:
        state = context["vmec"].solve_state_for_objective(x)
        residual = np.asarray(context["residuals_from_state"](state), dtype=float)
    qs_objective = float(np.asarray(context["qs"].total_from_state(state)))
    aspect_value = float(np.asarray(context["vmec"].aspect_equilibrium_from_state_jax(state)))
    aspect_term = float((aspect_value - 7.0) ** 2)
    total = float(np.dot(residual, residual))
    return {
        "total_objective": total,
        "qs_objective": qs_objective,
        "aspect_term": aspect_term,
        "aspect_value": aspect_value,
    }


def make_iteration_logger(context, solver_name, start_time, wall_clock_limit_s):
    rows = []
    counters = {
        "residual_calls": 0,
        "jacobian_calls": 0,
    }
    last_logged_x = None
    last_eval_x = None

    def log_snapshot(label, x):
        nonlocal last_logged_x
        x = np.asarray(x, dtype=float)
        snapshot = {
            "label": str(label),
            "iteration": len(rows),
            "nfev_observed": int(counters["residual_calls"]),
            "njev_observed": int(counters["jacobian_calls"]),
            "elapsed_s": float(time.perf_counter() - start_time),
            "rss_bytes": current_rss_bytes(),
            "peak_rss_bytes": peak_rss_bytes(),
            "x": x.tolist(),
        }
        rows.append(snapshot)
        last_logged_x = x.copy()
        return snapshot

    def residual_wrapper(fun):
        def wrapped(x):
            nonlocal last_eval_x
            counters["residual_calls"] += 1
            last_eval_x = np.asarray(x, dtype=float).copy()
            check_wall_clock(start_time, wall_clock_limit_s)
            return fun(x)

        return wrapped

    def jacobian_wrapper(fun):
        def wrapped(x):
            nonlocal last_eval_x
            counters["jacobian_calls"] += 1
            last_eval_x = np.asarray(x, dtype=float).copy()
            check_wall_clock(start_time, wall_clock_limit_s)
            return fun(x)

        return wrapped

    def callback(xk, *args, **kwargs):
        check_wall_clock(start_time, wall_clock_limit_s)
        log_snapshot("callback", xk)
        check_wall_clock(start_time, wall_clock_limit_s)

    def finalize(final_x, label):
        if final_x is None:
            final_x = last_eval_x if last_eval_x is not None else last_logged_x
        if final_x is None:
            return
        final_x = np.asarray(final_x, dtype=float)
        if last_logged_x is None or np.linalg.norm(final_x - last_logged_x) > 0.0:
            log_snapshot(label, final_x)

    return rows, counters, log_snapshot, residual_wrapper, jacobian_wrapper, callback, finalize


def enrich_iteration_metrics(rows, context, solver_name):
    enriched = []
    for row in rows:
        x = np.asarray(row["x"], dtype=float)
        if solver_name == "classic":
            metrics = classic_metrics(context, x)
        else:
            metrics = jax_metrics(context, x)
        merged = dict(row)
        merged.update(metrics)
        enriched.append(merged)
    return enriched


def run_case(case_name, max_mode, max_nfev, wall_clock_limit_s, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{case_name}_mode{max_mode}_summary.json"
    iterations_path = output_dir / f"{case_name}_mode{max_mode}_iterations.jsonl"

    if case_name == "classic":
        context = make_classic_context(max_mode)
    elif case_name == "jax":
        context = make_jax_context(max_mode)
    else:
        raise ValueError(f"Unsupported case: {case_name}")

    start_time = time.perf_counter()
    rows, counters, log_snapshot, residual_wrapper, jacobian_wrapper, callback, finalize = make_iteration_logger(
        context,
        case_name,
        start_time,
        wall_clock_limit_s,
    )
    log_snapshot("initial", context["x0"])

    result = None
    timed_out = False
    error_text = None
    metrics_error_text = None

    try:
        if case_name == "classic":
            residual_fun = residual_wrapper(lambda x: np.asarray(context["prob"].residuals(x), dtype=float))
            fd = FiniteDifference(
                context["prob"].residuals,
                x0=context["x0"],
                abs_step=1.0e-8,
                rel_step=1.0e-5,
                diff_method="forward",
            )
            jac_fun = jacobian_wrapper(lambda x: np.asarray(fd.jac(np.asarray(x, dtype=float)), dtype=float))
            result = least_squares(
                residual_fun,
                context["x0"],
                jac=jac_fun,
                x_scale=context["x_scale"],
                max_nfev=int(max_nfev),
                ftol=1.0e-8,
                xtol=1.0e-8,
                gtol=1.0e-8,
                verbose=2,
                callback=callback,
            )
        else:
            residual_fun = residual_wrapper(
                lambda x: np.asarray(context["stage"].residuals.scipy_residuals(np.asarray(x, dtype=float)), dtype=float)
            )
            jac_fun = jacobian_wrapper(
                lambda x: np.asarray(context["stage"].residuals.scipy_jacobian(np.asarray(x, dtype=float)), dtype=float)
            )
            result = least_squares(
                residual_fun,
                context["x0"],
                jac=jac_fun,
                x_scale=context["x_scale"],
                max_nfev=int(max_nfev),
                ftol=1.0e-7,
                xtol=1.0e-7,
                gtol=1.0e-7,
                verbose=2,
                callback=callback,
            )
    except WallClockStop as exc:
        timed_out = True
        error_text = str(exc)
    except Exception:
        error_text = traceback.format_exc()

    final_x = None
    if result is not None:
        final_x = np.asarray(result.x, dtype=float)
    finalize(final_x, "final")
    try:
        rows = enrich_iteration_metrics(rows, context, case_name)
    except Exception:
        metrics_error_text = traceback.format_exc()

    summary = {
        "case": case_name,
        "max_mode": int(max_mode),
        "max_nfev": int(max_nfev),
        "wall_clock_limit_s": float(wall_clock_limit_s),
        "timed_out": bool(timed_out),
        "error": error_text,
        "elapsed_s": float(time.perf_counter() - start_time),
        "peak_rss_bytes": peak_rss_bytes(),
        "residual_calls": int(counters["residual_calls"]),
        "jacobian_calls": int(counters["jacobian_calls"]),
        "n_iterations_logged": len(rows),
        "metrics_error": metrics_error_text,
    }
    if result is not None:
        summary.update(
            {
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "nfev": int(result.nfev),
                "njev": None if result.njev is None else int(result.njev),
                "optimality": float(result.optimality),
            }
        )
    if rows:
        summary["initial"] = rows[0]
        summary["final"] = rows[-1]

    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    with iterations_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    return summary_path, iterations_path


def run_case_subprocess(case_name, max_mode, max_nfev, wall_clock_limit_s, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = output_dir / f"{case_name}_mode{max_mode}.stdout.txt"
    stderr_path = output_dir / f"{case_name}_mode{max_mode}.stderr.txt"
    cmd = [
        sys.executable,
        __file__,
        "--run-case",
        case_name,
        "--max-mode",
        str(max_mode),
        "--max-nfev",
        str(max_nfev),
        "--wall-clock-limit",
        str(wall_clock_limit_s),
        "--output-dir",
        str(output_dir),
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=float(wall_clock_limit_s) + 120.0,
    )
    stdout_path.write_text(proc.stdout)
    stderr_path.write_text(proc.stderr)
    summary_path = output_dir / f"{case_name}_mode{max_mode}_summary.json"
    iterations_path = output_dir / f"{case_name}_mode{max_mode}_iterations.jsonl"
    return {
        "returncode": int(proc.returncode),
        "summary_path": summary_path,
        "iterations_path": iterations_path,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
    }


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_summary_markdown(results, output_dir):
    output_dir = Path(output_dir)
    summary_md = output_dir / "summary.md"
    lines = [
        "# QH Classic vs JAX comparison",
        "",
        "- Stable comparison set generated from direct per-case runs.",
        "- Classic runs used a 300 s wall-clock cap.",
        "- JAX runs used shorter caps to stay inside the exact-path stable window.",
        f"- Max nfev requested per case: {MAX_NFEV}",
        "",
        "| case | mode | wall cap (s) | timed out | success | elapsed (s) | peak RSS (GB) | final total | final qs | final aspect |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        summary = item["summary"]
        final_row = summary.get("final", {})
        lines.append(
            "| {case} | {mode} | {wall_cap:.0f} | {timed_out} | {success} | {elapsed:.2f} | {peak:.2f} | {total:.6f} | {qs:.6f} | {aspect:.6f} |".format(
                case=summary["case"],
                mode=summary["max_mode"],
                wall_cap=float(summary.get("wall_clock_limit_s", float("nan"))),
                timed_out=summary.get("timed_out", False),
                success=summary.get("success", False),
                elapsed=float(summary.get("elapsed_s", 0.0)),
                peak=float(summary.get("peak_rss_bytes", 0)) / (1024.0 ** 3),
                total=float(final_row.get("total_objective", float("nan"))),
                qs=float(final_row.get("qs_objective", float("nan"))),
                aspect=float(final_row.get("aspect_value", float("nan"))),
            )
        )
    summary_md.write_text("\n".join(lines) + "\n")
    return summary_md


def make_mode_plot(mode, series_by_case, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"classic": "#1f77b4", "jax": "#d62728"}
    labels = {"classic": "Classic VMEC2000", "jax": "VMEC-JAX discrete adjoint"}
    for case_name, rows in series_by_case.items():
        if not rows:
            continue
        color = colors[case_name]
        label = labels[case_name]
        iteration = [int(row["iteration"]) for row in rows]
        elapsed = [float(row["elapsed_s"]) for row in rows]
        rss_gb = [float(row["rss_bytes"]) / (1024.0 ** 3) for row in rows]
        total = [float(row["total_objective"]) for row in rows if "total_objective" in row]
        qs_objective = [float(row["qs_objective"]) for row in rows if "qs_objective" in row]
        aspect_term = [float(row["aspect_term"]) for row in rows if "aspect_term" in row]
        aspect_value = [float(row["aspect_value"]) for row in rows if "aspect_value" in row]
        objective_iteration = [int(row["iteration"]) for row in rows if "total_objective" in row]
        aspect_iteration = [int(row["iteration"]) for row in rows if "aspect_value" in row]

        axes[0, 0].plot(iteration, elapsed, marker="o", color=color, label=label)
        axes[0, 1].plot(iteration, rss_gb, marker="o", color=color, label=label)
        if total:
            axes[1, 0].plot(objective_iteration, total, marker="o", color=color, label=f"{label} total")
            axes[1, 0].plot(objective_iteration, qs_objective, marker="s", linestyle="--", color=color, alpha=0.8, label=f"{label} qs")
            axes[1, 0].plot(objective_iteration, aspect_term, marker="^", linestyle=":", color=color, alpha=0.8, label=f"{label} aspect term")
        if aspect_value:
            axes[1, 1].plot(aspect_iteration, aspect_value, marker="o", color=color, label=label)

    axes[0, 0].set_title(f"Mode {mode}: runtime by iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Elapsed time (s)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title(f"Mode {mode}: memory by iteration")
    axes[0, 1].set_xlabel("Iteration")
    axes[0, 1].set_ylabel("RSS (GB)")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title(f"Mode {mode}: objective terms by iteration")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Objective")
    axes[1, 0].set_yscale("log")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title(f"Mode {mode}: aspect value by iteration")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].set_ylabel("Aspect")
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.legend(fontsize=8)

    output_path = Path(output_dir) / f"qh_compare_mode{mode}_iterations.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_parent(output_dir, max_nfev, wall_clock_limit_s):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for max_mode in (1, 2):
        for case_name in ("classic", "jax"):
            gc.collect()
            child = run_case_subprocess(case_name, max_mode, max_nfev, wall_clock_limit_s, output_dir)
            if child["summary_path"].exists():
                summary = load_json(child["summary_path"])
            else:
                summary = {
                    "case": case_name,
                    "max_mode": max_mode,
                    "max_nfev": max_nfev,
                    "wall_clock_limit_s": wall_clock_limit_s,
                    "timed_out": False,
                    "success": False,
                    "status": -999,
                    "message": f"missing summary, child returncode={child['returncode']}",
                    "error": str(child["stderr_path"]),
                    "elapsed_s": float("nan"),
                    "peak_rss_bytes": 0,
                }
            iterations = load_jsonl(child["iterations_path"]) if child["iterations_path"].exists() else []
            results.append(
                {
                    "case": case_name,
                    "mode": max_mode,
                    "summary": summary,
                    "iterations": iterations,
                    "stdout_path": str(child["stdout_path"]),
                    "stderr_path": str(child["stderr_path"]),
                }
            )

    plot_paths = []
    for max_mode in (1, 2):
        series = {item["case"]: item["iterations"] for item in results if item["mode"] == max_mode}
        plot_paths.append(make_mode_plot(max_mode, series, output_dir))

    summary_md = write_summary_markdown(results, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "output_dir": str(output_dir),
        "summary_markdown": str(summary_md),
        "plots": [str(path) for path in plot_paths],
        "results": results,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-case", choices=["classic", "jax"], default=None)
    parser.add_argument("--max-mode", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--wall-clock-limit", type=float, default=WALL_CLOCK_LIMIT_S)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if args.run_case is not None:
        run_case(args.run_case, args.max_mode, args.max_nfev, args.wall_clock_limit, args.output_dir)
        return

    manifest_path = run_parent(args.output_dir, args.max_nfev, args.wall_clock_limit)
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
