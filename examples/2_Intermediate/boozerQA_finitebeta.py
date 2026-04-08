#!/usr/bin/env python3

import csv
import os
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from simsopt.configs import get_data
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, BoozerSurface, CurveLength, FiniteBetaBoozerSurface, SurfaceCurrentFieldProvider, Volume, curves_to_vtk
from simsopt.geo.surfaceobjectives import MU0, surface_field_nonquasisymmetric_ratio

r"""
This example is the finite-beta counterpart to boozerQA.py.

The script supports three finite-beta inner solves:

- a direct no-VC self-consistent closure based on a surface-current Biot-Savart provider,
- a single-surface virtual-casing validation closure,
- prescribed or VMEC-backed interface data.

The script can also wrap some of these inner solves inside reduced QA loops and
run pressure-jump continuation scans from the vacuum limit to the target
finite-beta case.
"""


default_out_dir = Path(__file__).resolve().parent / "output"
OUT_DIR = os.environ.get("SIMSOPT_FINITE_BETA_OUT_DIR", str(default_out_dir))
if not OUT_DIR.endswith(os.sep):
    OUT_DIR = OUT_DIR + os.sep
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA_finitebeta.py")
print("==============================================")

legacy_synthetic_mode = os.environ.get("SIMSOPT_FINITE_BETA_SYNTHETIC")
default_mode = "single-surface-vc" if legacy_synthetic_mode is None else ("prescribed" if legacy_synthetic_mode == "1" else "vmec")
requested_mode = os.environ.get("SIMSOPT_FINITE_BETA_MODE", default_mode).strip().lower()
if requested_mode in ("single-surface-vc", "vc", "vc-validation"):
    mode = "single-surface-vc"
elif requested_mode == "self-consistent":
    mode = "self-consistent"
else:
    mode = requested_mode

nphi_default = 8
ntheta_default = 8
nphi = int(os.environ.get("SIMSOPT_FINITE_BETA_NPHI", str(nphi_default)))
ntheta = int(os.environ.get("SIMSOPT_FINITE_BETA_NTHETA", str(ntheta_default)))
ls_max_nfev = int(os.environ.get("SIMSOPT_FINITE_BETA_MAX_NFEV", "30"))
qa_maxiter_default = 3
qa_maxiter = int(os.environ.get("SIMSOPT_FINITE_BETA_QA_MAXITER", str(qa_maxiter_default)))
inner_verbose = os.environ.get("SIMSOPT_FINITE_BETA_INNER_VERBOSE", "0").strip().lower() in ("1", "true", "yes", "on")
write_closure_diagnostics = os.environ.get("SIMSOPT_FINITE_BETA_WRITE_CLOSURE_DIAGNOSTICS", "1").strip().lower() in ("1", "true", "yes", "on")
closure_vc_digits = int(os.environ.get("SIMSOPT_FINITE_BETA_COMPARE_VC_DIGITS", os.environ.get("SIMSOPT_FINITE_BETA_VC_DIGITS", "6")))


def stellsym_exact_grid(requested_nphi, requested_ntheta, min_ntor=3, min_mpol=3):
    ntor = max(min_ntor, (int(requested_nphi) - 1) // 2)
    mpol = max(min_mpol, (int(requested_ntheta) - 1) // 2)
    quadpoints_phi = np.linspace(0, 1 / nfp, 2 * ntor + 1, endpoint=False)
    quadpoints_theta = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)
    return mpol, ntor, quadpoints_phi, quadpoints_theta


def print_blocks(title, result):
    print(title)
    for name, values in result["blocks"].items():
        print(f"{name:>13s} residual norm = {np.linalg.norm(values):.6e}")


def print_solver_summary(title, result):
    print(
        f"{title}: success={result['success']}, lsq_nfev={result['iter']}, "
        f"iota={result['iota']:.6e}, G={result['G']:.6e}, I={result['I']:.6e}, "
        f"||r||={result['residual_norm']:.6e}"
    )


def block_norms(block_result):
    return {name: np.linalg.norm(values) for name, values in block_result["blocks"].items()}


def clone_surface(surface):
    return surface.__class__(
        mpol=surface.mpol,
        ntor=surface.ntor,
        stellsym=surface.stellsym,
        nfp=surface.nfp,
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        dofs=surface.dofs,
    )


def make_progress_reporter():
    state = {
        "eval_count": 0,
        "best_J": np.inf,
        "last_print_time": time.time(),
        "history": [],
    }

    def report(J, J_nonqs, J_iota, J_mr, result, solved_blocks):
        state["eval_count"] += 1
        improved = J < state["best_J"] * (1 - 1e-3)
        if improved:
            state["best_J"] = J

        norms = block_norms(solved_blocks)

        state["history"].append([
            state["eval_count"],
            J,
            J_nonqs,
            J_iota,
            J_mr,
            result["residual_norm"],
            result["surface"].major_radius(),
            result["iota"],
            result["iter"],
            state["best_J"],
            norms.get("boozer", 0.0),
            norms.get("normal", 0.0),
            norms.get("pressure", 0.0),
            norms.get("jump", 0.0),
            norms.get("sheet_current", 0.0),
        ])

        now = time.time()
        should_print = (
            state["eval_count"] <= 2
            or improved
            or state["eval_count"] % 50 == 0
            or (now - state["last_print_time"]) >= 20.0
        )
        if not should_print:
            return

        state["last_print_time"] = now
        marker = "best" if improved else "checkpoint"
        print(
            f"QA {marker} #{state['eval_count']}: J={J:.6e}, best={state['best_J']:.6e}, "
            f"nonQS={J_nonqs:.6e}, Jiota={J_iota:.6e}, Jmr={J_mr:.6e}, "
            f"iota={result['iota']:.6e}, mr={result['surface'].major_radius():.6e}, "
            f"lsq_nfev={result['iter']}, ||r||={result['residual_norm']:.6e}, "
            f"boozer={norms.get('boozer', 0.0):.3e}, normal={norms.get('normal', 0.0):.3e}, "
            f"pressure={norms.get('pressure', 0.0):.3e}, jump={norms.get('jump', 0.0):.3e}"
        )

    report.state = state
    return report


def write_residual_vtk(surface, result):
    extra_data = {
        "B_out_normal": result["blocks"]["normal"][:, :, None],
        "pressure_balance": result["blocks"]["pressure"][:, :, None],
        "jump_x": result["blocks"]["jump"][:, :, [0]],
        "jump_y": result["blocks"]["jump"][:, :, [1]],
        "jump_z": result["blocks"]["jump"][:, :, [2]],
    }
    surface.to_vtk(OUT_DIR + "boozerQA_finitebeta_boundary", extra_data=extra_data)


def write_single_surface_vc_vtk(surface, result, prefix="boozerQA_finitebeta_boundary"):
    blocks = result["blocks"]
    B_total = result["B_total"]
    B_external = result["B_external"]
    B_coils = result["B_coils"]
    coil_match_norm = np.linalg.norm(blocks["coil_match"], axis=2)
    sheet_current_norm = np.linalg.norm(blocks["sheet_current"], axis=2)
    extra_data = {
        "B_total_x": B_total[:, :, [0]],
        "B_total_y": B_total[:, :, [1]],
        "B_total_z": B_total[:, :, [2]],
        "B_external_x": B_external[:, :, [0]],
        "B_external_y": B_external[:, :, [1]],
        "B_external_z": B_external[:, :, [2]],
        "B_coils_x": B_coils[:, :, [0]],
        "B_coils_y": B_coils[:, :, [1]],
        "B_coils_z": B_coils[:, :, [2]],
        "modB_total": np.linalg.norm(B_total, axis=2)[:, :, None],
        "modB_external": np.linalg.norm(B_external, axis=2)[:, :, None],
        "modB_coils": np.linalg.norm(B_coils, axis=2)[:, :, None],
        "coil_match_x": blocks["coil_match"][:, :, [0]],
        "coil_match_y": blocks["coil_match"][:, :, [1]],
        "coil_match_z": blocks["coil_match"][:, :, [2]],
        "coil_match_norm": coil_match_norm[:, :, None],
        "normal_residual": blocks["normal"][:, :, None],
        "pressure_balance": blocks["pressure"][:, :, None],
        "sheet_current_x": blocks["sheet_current"][:, :, [0]],
        "sheet_current_y": blocks["sheet_current"][:, :, [1]],
        "sheet_current_z": blocks["sheet_current"][:, :, [2]],
        "sheet_current_norm": sheet_current_norm[:, :, None],
    }
    surface.to_vtk(OUT_DIR + prefix, extra_data=extra_data)


def write_surface_vtk(name, surface):
    surface.to_vtk(OUT_DIR + name)


def write_curves_vtk(name, curves):
    if curves:
        curves_to_vtk(curves, OUT_DIR + name)


def write_single_surface_maps(name, result):
    if plt is None:
        return

    fields = [
        (np.linalg.norm(result["B_total"], axis=2), "|B_total|"),
        (np.linalg.norm(result["B_external"], axis=2), "|B_external|"),
        (np.linalg.norm(result["B_coils"], axis=2), "|B_coils|"),
        (np.linalg.norm(result["blocks"]["coil_match"], axis=2), "|B_external - B_coils|"),
        (result["blocks"]["normal"], "B_external . n"),
        (result["blocks"]["pressure"], "pressure balance"),
        (np.linalg.norm(result["blocks"]["sheet_current"], axis=2), "|sheet current|"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()
    for ax, (data, title) in zip(axes, fields):
        im = ax.imshow(data.T, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("phi index")
        ax.set_ylabel("theta index")
        fig.colorbar(im, ax=ax, shrink=0.85)
    for ax in axes[len(fields):]:
        ax.axis("off")

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved surface field map plot to {png_path}")


def surface_qs_error_map(surface, field, quasi_poloidal=False):
    modB = np.linalg.norm(field, axis=2)
    dS = np.linalg.norm(surface.normal(), axis=2)
    axis = 1 if quasi_poloidal else 0
    B_qs = np.mean(modB * dS, axis=axis) / np.maximum(np.mean(dS, axis=axis), 1e-30)
    if axis == 0:
        B_qs = B_qs[None, :]
    else:
        B_qs = B_qs[:, None]
    scale = max(float(np.sqrt(np.mean(B_qs**2))), 1e-30)
    return (modB - B_qs) / scale


def surface_magnetic_pressure_map(result):
    modB_external_sq = np.sum(result["B_external"]**2, axis=2)
    modB_total_sq = np.sum(result["B_total"]**2, axis=2)
    return (modB_external_sq - modB_total_sq) / (2.0 * MU0)


def area_weighted_surface_average(surface, values):
    weights = np.linalg.norm(surface.normal(), axis=2)
    return float(np.sum(values * weights) / max(float(np.sum(weights)), 1e-30))


def mean_surface_magnetic_pressure(surface, field):
    modB_sq = np.sum(np.asarray(field)**2, axis=2)
    return area_weighted_surface_average(surface, modB_sq / (2.0 * MU0))


def mean_surface_pressure_jump(surface, B_in, B_out):
    pressure_map = (
        np.sum(np.asarray(B_out)**2, axis=2) - np.sum(np.asarray(B_in)**2, axis=2)
    ) / (2.0 * MU0)
    return area_weighted_surface_average(surface, pressure_map)


def _parse_optional_plasma_beta():
    beta_raw = os.environ.get("SIMSOPT_FINITE_BETA_PLASMA_BETA")
    legacy_beta_raw = os.environ.get("SIMSOPT_FINITE_BETA_BETA")
    if beta_raw is not None and legacy_beta_raw is not None:
        raise ValueError("Set only one of SIMSOPT_FINITE_BETA_PLASMA_BETA or SIMSOPT_FINITE_BETA_BETA.")

    raw_value = beta_raw if beta_raw is not None else legacy_beta_raw
    if raw_value is None:
        return None

    text = raw_value.strip()
    if text.endswith("%"):
        return 1e-2 * float(text[:-1])
    return float(text)


def resolve_pressure_jump_input(surface, field):
    pressure_jump_raw = os.environ.get("SIMSOPT_FINITE_BETA_PRESSURE_JUMP")
    plasma_beta = _parse_optional_plasma_beta()
    if pressure_jump_raw is not None and plasma_beta is not None:
        raise ValueError("Set either SIMSOPT_FINITE_BETA_PRESSURE_JUMP or SIMSOPT_FINITE_BETA_PLASMA_BETA, not both.")

    reference_magnetic_pressure = mean_surface_magnetic_pressure(surface, field)
    if plasma_beta is not None:
        pressure_jump = plasma_beta * reference_magnetic_pressure
    elif pressure_jump_raw is not None:
        pressure_jump = float(pressure_jump_raw)
        plasma_beta = pressure_jump / max(reference_magnetic_pressure, 1e-30)
    else:
        pressure_jump = 0.0
        plasma_beta = 0.0

    return float(pressure_jump), float(plasma_beta), float(reference_magnetic_pressure)


def retarget_pressure_jump_to_beta(finite_beta, iota, G, I, lambda_current, target_plasma_beta):
    reference_result = finite_beta.self_consistent_single_surface_residual(
        iota=iota,
        G=G,
        I=I,
        lambda_current=lambda_current,
        pressure_jump=0.0,
    )
    reference_magnetic_pressure = mean_surface_magnetic_pressure(finite_beta.surface, reference_result["B_total"])
    pressure_jump = float(target_plasma_beta * reference_magnetic_pressure)
    finite_beta.pressure_jump = pressure_jump
    return pressure_jump, float(reference_magnetic_pressure), reference_result


def achieved_plasma_beta(surface, field, pressure_jump):
    return float(pressure_jump / max(mean_surface_magnetic_pressure(surface, field), 1e-30))


def summarize_single_surface_state(surface, result, pressure_jump, target_plasma_beta=None, reference_magnetic_pressure=None):
    blocks = result["blocks"]
    mean_total_magnetic_pressure = mean_surface_magnetic_pressure(surface, result["B_total"])
    return {
        "pressure_jump": float(pressure_jump),
        "plasma_beta": float(pressure_jump / max(mean_total_magnetic_pressure, 1e-30)),
        "target_plasma_beta": float(target_plasma_beta if target_plasma_beta is not None else np.nan),
        "reference_magnetic_pressure": float(reference_magnetic_pressure if reference_magnetic_pressure is not None else np.nan),
        "mean_total_magnetic_pressure": float(mean_total_magnetic_pressure),
        "iota": float(result.get("iota", np.nan)),
        "G": float(result.get("G", np.nan)),
        "I": float(result.get("I", np.nan)),
        "lambda_current": float(result.get("lambda_current", np.nan)),
        "major_radius": float(surface.major_radius()),
        "nonqs": float(surface_field_nonquasisymmetric_ratio(surface, result["B_total"])),
        "raw_residual_norm": float(result.get("raw_residual_norm", np.linalg.norm(result["residual"]))),
        "weighted_residual_norm": float(result.get("residual_norm", np.linalg.norm(result["residual"]))),
        "coil_match_norm": float(np.linalg.norm(blocks["coil_match"])),
        "normal_norm": float(np.linalg.norm(blocks["normal"])),
        "pressure_norm": float(np.linalg.norm(blocks["pressure"])),
        "sheet_current_norm": float(np.linalg.norm(blocks["sheet_current"])),
        "mean_magnetic_pressure": float(np.mean(surface_magnetic_pressure_map(result))),
        "std_magnetic_pressure": float(np.std(surface_magnetic_pressure_map(result))),
    }


def summarize_frozen_closure_diagnostics(diagnostics):
    direct_blocks = diagnostics["direct"]["vc_style"]["blocks"]
    direct_residual_blocks = diagnostics["direct"]["residual_blocks"]
    vc_blocks = diagnostics["virtual_casing"]["blocks"]
    differences = diagnostics["differences"]
    return {
        "pressure_jump": float(diagnostics["pressure_jump"]),
        "lambda_current": float(diagnostics["lambda_current"]),
        "direct_boozer_norm": float(np.linalg.norm(direct_residual_blocks["boozer"])),
        "direct_jump_norm": float(np.linalg.norm(direct_residual_blocks["jump"])),
        "direct_coil_match_norm": float(np.linalg.norm(direct_blocks["coil_match"])),
        "vc_coil_match_norm": float(np.linalg.norm(vc_blocks["coil_match"])),
        "direct_normal_norm": float(np.linalg.norm(direct_blocks["normal"])),
        "vc_normal_norm": float(np.linalg.norm(vc_blocks["normal"])),
        "direct_pressure_norm": float(np.linalg.norm(direct_blocks["pressure"])),
        "vc_pressure_norm": float(np.linalg.norm(vc_blocks["pressure"])),
        "direct_sheet_current_norm": float(np.linalg.norm(direct_blocks["sheet_current"])),
        "vc_sheet_current_norm": float(np.linalg.norm(vc_blocks["sheet_current"])),
        "B_external_rel_norm": float(differences["B_external_rel_norm"]),
        "coil_match_rel_norm": float(differences["coil_match_rel_norm"]),
        "normal_rel_norm": float(differences["normal_rel_norm"]),
        "pressure_rel_norm": float(differences["pressure_rel_norm"]),
        "sheet_current_rel_norm": float(differences["sheet_current_rel_norm"]),
    }


def write_frozen_closure_outputs(name, diagnostics):
    summary = summarize_frozen_closure_diagnostics(diagnostics)
    csv_path = Path(OUT_DIR) / f"{name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, f"{value:.16e}"])
    print(f"Saved frozen-state closure comparison CSV to {csv_path}")

    if plt is None:
        return summary

    direct = diagnostics["direct"]
    vc_result = diagnostics["virtual_casing"]
    diffs = diagnostics["differences"]
    direct_blocks = direct["vc_style"]["blocks"]
    vc_blocks = vc_result["blocks"]

    panels = [
        (
            np.linalg.norm(direct["B_external"], axis=2),
            np.linalg.norm(vc_result["B_external"], axis=2),
            np.linalg.norm(diffs["B_external"], axis=2),
            "|B_external|",
        ),
        (
            np.linalg.norm(direct_blocks["coil_match"], axis=2),
            np.linalg.norm(vc_blocks["coil_match"], axis=2),
            np.linalg.norm(diffs["coil_match"], axis=2),
            "|B_external - B_coils|",
        ),
        (
            direct_blocks["normal"],
            vc_blocks["normal"],
            diffs["normal"],
            "B_external . n",
        ),
        (
            direct_blocks["pressure"],
            vc_blocks["pressure"],
            diffs["pressure"],
            "pressure residual",
        ),
    ]

    fig, axes = plt.subplots(len(panels), 3, figsize=(13, 3.2 * len(panels)), constrained_layout=True)
    if len(panels) == 1:
        axes = np.asarray([axes])
    for row, (direct_data, vc_data, diff_data, title) in enumerate(panels):
        data_min = min(float(np.min(direct_data)), float(np.min(vc_data)))
        data_max = max(float(np.max(direct_data)), float(np.max(vc_data)))
        for col, (data, header) in enumerate(((direct_data, "Direct closure"), (vc_data, "Full VC"))):
            ax = axes[row, col]
            im = ax.imshow(data.T, origin="lower", aspect="auto", vmin=data_min, vmax=data_max)
            ax.set_title(f"{header} {title}")
            ax.set_xlabel("phi index")
            ax.set_ylabel("theta index")
            fig.colorbar(im, ax=ax, shrink=0.82)

        diff_ax = axes[row, 2]
        diff_im = diff_ax.imshow(diff_data.T, origin="lower", aspect="auto")
        diff_ax.set_title(f"Direct - VC {title}")
        diff_ax.set_xlabel("phi index")
        diff_ax.set_ylabel("theta index")
        fig.colorbar(diff_im, ax=diff_ax, shrink=0.82)

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved frozen-state closure comparison plot to {png_path}")
    return summary


def capture_curve_geometries(curves):
    geometries = []
    for curve in curves:
        if hasattr(curve, "gamma"):
            geometries.append(np.asarray(curve.gamma()).copy())
    return geometries


def _set_equal_3d_limits(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.55 * np.max(maxs - mins)
    radius = max(float(radius), 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def write_geometry_comparison(name, initial_curves, initial_surface_gamma, final_curves, final_surface_gamma):
    if plt is None:
        return

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    axes = [fig.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
    panels = [
        (axes[0], initial_curves, initial_surface_gamma, "Initial coils and surface"),
        (axes[1], final_curves, final_surface_gamma, "Final coils and surface"),
    ]

    all_points = []
    for ax, curves, surface_gamma, title in panels:
        for curve_xyz in curves:
            ax.plot(curve_xyz[:, 0], curve_xyz[:, 1], curve_xyz[:, 2], color="tab:blue", linewidth=1.0, alpha=0.85)
            all_points.append(curve_xyz)
        stride_phi = max(1, surface_gamma.shape[0] // 16)
        stride_theta = max(1, surface_gamma.shape[1] // 16)
        ax.plot_wireframe(
            surface_gamma[::stride_phi, ::stride_theta, 0],
            surface_gamma[::stride_phi, ::stride_theta, 1],
            surface_gamma[::stride_phi, ::stride_theta, 2],
            color="tab:orange",
            linewidth=0.6,
            alpha=0.9,
        )
        all_points.append(surface_gamma.reshape((-1, 3)))
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    stacked = np.vstack(all_points)
    for ax in axes:
        _set_equal_3d_limits(ax, stacked)
        ax.view_init(elev=24, azim=45)

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved geometry comparison plot to {png_path}")


def write_state_comparison(name, initial_surface, initial_result, final_surface, final_result):
    if plt is None:
        return

    panels = [
        (np.linalg.norm(initial_result["B_total"], axis=2), np.linalg.norm(final_result["B_total"], axis=2), "|B_total|"),
        (surface_qs_error_map(initial_surface, initial_result["B_total"]), surface_qs_error_map(final_surface, final_result["B_total"]), "local QS error"),
        (surface_magnetic_pressure_map(initial_result), surface_magnetic_pressure_map(final_result), "magnetic pressure jump"),
        (np.linalg.norm(initial_result["blocks"]["coil_match"], axis=2), np.linalg.norm(final_result["blocks"]["coil_match"], axis=2), "|B_external - B_coils|"),
        (initial_result["blocks"]["pressure"], final_result["blocks"]["pressure"], "pressure-balance residual"),
        (np.linalg.norm(initial_result["blocks"]["sheet_current"], axis=2), np.linalg.norm(final_result["blocks"]["sheet_current"], axis=2), "|sheet current|"),
    ]

    fig, axes = plt.subplots(len(panels), 4, figsize=(15, 3 * len(panels)), constrained_layout=True)
    for row, (initial_data, final_data, title) in enumerate(panels):
        data_min = min(float(np.min(initial_data)), float(np.min(final_data)))
        data_max = max(float(np.max(initial_data)), float(np.max(final_data)))
        for col, (data, header) in enumerate(((initial_data, "Initial"), (final_data, "Final"))):
            ax = axes[row, 2 * col]
            im = ax.imshow(data.T, origin="lower", aspect="auto", vmin=data_min, vmax=data_max)
            ax.set_title(f"{header} {title}")
            ax.set_xlabel("phi index")
            ax.set_ylabel("theta index")
            fig.colorbar(im, ax=ax, shrink=0.8)

            diff_ax = axes[row, 2 * col + 1]
            if col == 0:
                diff_ax.axis("off")
            else:
                diff = final_data - initial_data
                diff_im = diff_ax.imshow(diff.T, origin="lower", aspect="auto")
                diff_ax.set_title(f"Final - Initial {title}")
                diff_ax.set_xlabel("phi index")
                diff_ax.set_ylabel("theta index")
                fig.colorbar(diff_im, ax=diff_ax, shrink=0.8)

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved state comparison plot to {png_path}")


def write_objective_comparison(name, initial_summary, final_summary):
    metrics = [
        ("plasma_beta", "plasma beta"),
        ("nonqs", "nonQS"),
        ("weighted_residual_norm", "weighted residual"),
        ("raw_residual_norm", "raw residual"),
        ("coil_match_norm", "coil match"),
        ("normal_norm", "normal"),
        ("pressure_norm", "pressure"),
        ("sheet_current_norm", "sheet current"),
        ("major_radius", "major radius"),
    ]

    rows = [[name, initial_summary[name], final_summary[name]] for name, _ in metrics]
    csv_path = Path(OUT_DIR) / f"{name}.csv"
    np.savetxt(csv_path, np.asarray([[row[1], row[2]] for row in rows], dtype=float), delimiter=",", header="initial,final", comments="")
    print(f"Saved objective comparison CSV to {csv_path}")

    if plt is None:
        return

    labels = [label for _, label in metrics]
    initial_values = np.asarray([initial_summary[key] for key, _ in metrics], dtype=float)
    final_values = np.asarray([final_summary[key] for key, _ in metrics], dtype=float)
    x = np.arange(len(metrics))

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), constrained_layout=True)
    width = 0.36
    axes[0].bar(x - width / 2, np.maximum(initial_values, 1e-30), width=width, label="initial")
    axes[0].bar(x + width / 2, np.maximum(final_values, 1e-30), width=width, label="final")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("metric value")
    axes[0].set_title("Initial vs final finite-beta objective components")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    ratios = final_values / np.maximum(initial_values, 1e-30)
    axes[1].bar(x, ratios)
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("final / initial")
    axes[1].set_title("Relative change in objective components")
    axes[1].grid(True, alpha=0.3)

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved objective comparison plot to {png_path}")


def write_inner_continuation_history(name, continuation_history):
    if len(continuation_history) == 0:
        return

    rows = np.asarray([
        [
            row["pressure_jump"],
            row.get("nfev", row.get("iter", np.nan)),
            row.get("cost", row.get("weighted_residual_norm", np.nan)),
            row.get("weighted_residual_norm", np.nan),
            row.get("raw_residual_norm", np.nan),
            row.get("coil_match_norm", np.nan),
            row.get("normal_norm", np.nan),
            row.get("pressure_norm", np.nan),
        ]
        for row in continuation_history
    ], dtype=float)
    csv_path = Path(OUT_DIR) / f"{name}.csv"
    np.savetxt(
        csv_path,
        rows,
        delimiter=",",
        header="pressure_jump,nfev,cost,weighted_residual_norm,raw_residual_norm,coil_match_norm,normal_norm,pressure_norm",
        comments="",
    )
    print(f"Saved inner continuation CSV to {csv_path}")

    if plt is None:
        return

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(rows[:, 0], rows[:, 1], marker="o", linewidth=2, label="nfev")
    axes[0].plot(rows[:, 0], rows[:, 2], marker="o", linewidth=2, label="cost")
    axes[0].set_ylabel("work / cost")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(rows[:, 0], np.maximum(rows[:, 3], 1e-30), marker="o", linewidth=2, label="weighted residual")
    axes[1].semilogy(rows[:, 0], np.maximum(rows[:, 4], 1e-30), marker="o", linewidth=2, label="raw residual")
    axes[1].semilogy(rows[:, 0], np.maximum(rows[:, 5], 1e-30), marker="o", linewidth=2, label="coil match")
    axes[1].semilogy(rows[:, 0], np.maximum(rows[:, 7], 1e-30), marker="o", linewidth=2, label="pressure")
    axes[1].set_xlabel("pressure jump stage")
    axes[1].set_ylabel("residual norms")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(ncol=2)

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved inner continuation plot to {png_path}")


def write_single_surface_outer_history(history):
    if len(history) == 0:
        return

    history_array = np.asarray(history, dtype=float)
    csv_path = Path(OUT_DIR) / "boozerQA_finitebeta_vc_outer_history.csv"
    header = (
        "eval,J,J_nonqs,J_res,J_iota,J_mr,J_beta,J_length,J_coil_reg,raw_residual_norm,weighted_residual_norm,"
        "iota,major_radius,target_plasma_beta,achieved_plasma_beta,reference_magnetic_pressure,pressure_jump,"
        "coil_match_norm,normal_norm,pressure_norm,sheet_current_norm"
    )
    np.savetxt(csv_path, history_array, delimiter=",", header=header, comments="")
    print(f"Saved outer QA history CSV to {csv_path}")

    if plt is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(8, 14), constrained_layout=True)
    axes[0].semilogy(history_array[:, 0], history_array[:, 1], linewidth=2, label="J")
    axes[0].semilogy(history_array[:, 0], history_array[:, 2], linewidth=2, label="nonQS")
    axes[0].semilogy(history_array[:, 0], history_array[:, 3], linewidth=2, label="residual penalty")
    axes[0].semilogy(history_array[:, 0], np.maximum(history_array[:, 6], 1e-30), linewidth=2, label="beta penalty")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("objective terms")

    axes[1].semilogy(history_array[:, 0], np.maximum(history_array[:, 9], 1e-30), linewidth=2, label="raw residual")
    axes[1].semilogy(history_array[:, 0], np.maximum(history_array[:, 10], 1e-30), linewidth=2, label="weighted residual")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel("inner residual")

    axes[2].plot(history_array[:, 0], history_array[:, 11], linewidth=2, label="iota")
    axes[2].plot(history_array[:, 0], history_array[:, 12], linewidth=2, label="major radius")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylabel("state")

    axes[3].plot(history_array[:, 0], history_array[:, 13], linewidth=2, linestyle="--", label="target beta")
    axes[3].plot(history_array[:, 0], history_array[:, 14], linewidth=2, label="achieved beta")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].set_xlabel("outer evaluation")
    axes[3].set_ylabel("beta")

    png_path = Path(OUT_DIR) / "boozerQA_finitebeta_vc_outer_history.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved outer QA history plot to {png_path}")


def write_pressure_scan_outputs(rows):
    if len(rows) == 0:
        return

    data = np.asarray(rows, dtype=float)
    csv_path = Path(OUT_DIR) / "boozerQA_finitebeta_pressure_scan.csv"
    header = (
        "pressure_jump,target_plasma_beta,iota,I,lambda_current,nonqs,major_radius,raw_residual_norm,weighted_residual_norm,"
        "coil_match_norm,normal_norm,pressure_norm,sheet_current_norm,vacuum_iota_diff,vacuum_rel_field_diff"
    )
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"Saved pressure scan CSV to {csv_path}")

    if plt is None:
        return

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)
    axes[0].plot(data[:, 0], data[:, 2], marker="o", linewidth=2, label="iota")
    axes[0].plot(data[:, 0], data[:, 3], marker="o", linewidth=2, label="I")
    axes[0].plot(data[:, 0], data[:, 4], marker="o", linewidth=2, label="lambda")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("state")

    axes[1].semilogy(data[:, 0], np.maximum(data[:, 7], 1e-30), marker="o", linewidth=2, label="raw residual")
    axes[1].semilogy(data[:, 0], np.maximum(data[:, 8], 1e-30), marker="o", linewidth=2, label="weighted residual")
    axes[1].semilogy(data[:, 0], np.maximum(data[:, 9], 1e-30), marker="o", linewidth=2, label="coil match")
    axes[1].semilogy(data[:, 0], np.maximum(data[:, 11], 1e-30), marker="o", linewidth=2, label="pressure")
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel("residuals")

    axes[2].plot(data[:, 0], data[:, 5], marker="o", linewidth=2, label="nonQS")
    axes[2].semilogy(data[:, 0], np.maximum(data[:, 13], 1e-30), marker="o", linewidth=2, label="|iota-iota_vac|", color="tab:red")
    axes[2].semilogy(data[:, 0], np.maximum(data[:, 14], 1e-30), marker="o", linewidth=2, label="rel |B-B_vac|", color="tab:green")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_xlabel("pressure jump")
    axes[2].set_ylabel("QA and vacuum-limit metrics")

    png_path = Path(OUT_DIR) / "boozerQA_finitebeta_pressure_scan.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved pressure scan plot to {png_path}")


def write_history_outputs(history, initial_residual_norm):
    if len(history) == 0:
        return

    history_array = np.asarray(history, dtype=float)
    csv_path = Path(OUT_DIR) / "boozerQA_finitebeta_history.csv"
    header = (
        "eval,J,J_nonqs,J_iota,J_mr,residual_norm,major_radius,iota,lsq_nfev,best_J,"
        "boozer_norm,normal_norm,pressure_norm,jump_norm,sheet_current_norm"
    )
    np.savetxt(csv_path, history_array, delimiter=",", header=header, comments="")
    print(f"Saved optimization history CSV to {csv_path}")

    if plt is None:
        print("matplotlib is not available, so no PNG history plot was written.")
        return

    rel_residual = history_array[:, 5] / max(initial_residual_norm, 1e-30)
    rel_objective = history_array[:, 1] / max(history_array[0, 1], 1e-30)
    rel_nonqs = history_array[:, 2] / max(history_array[0, 2], 1e-30)
    rel_boozer = history_array[:, 10] / max(history_array[0, 10], 1e-30)
    rel_normal = history_array[:, 11] / max(history_array[0, 11], 1e-30)
    rel_pressure = history_array[:, 12] / max(history_array[0, 12], 1e-30)
    rel_jump = history_array[:, 13] / max(history_array[0, 13], 1e-30)
    rel_sheet_current = history_array[:, 14] / max(history_array[0, 14], 1e-30)

    fig, axes = plt.subplots(3, 1, figsize=(8, 11), constrained_layout=True)
    axes[0].semilogy(history_array[:, 0], rel_objective, linewidth=2, label="relative J")
    axes[0].semilogy(history_array[:, 0], rel_nonqs, linewidth=2, label="relative nonQS")
    axes[0].set_ylabel("relative objective")
    axes[0].set_title("Finite-beta QA optimization history")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].semilogy(history_array[:, 0], rel_residual, linewidth=2, label="relative surface residual")
    axes[1].set_ylabel("relative residual")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].semilogy(history_array[:, 0], rel_boozer, linewidth=2, label="boozer")
    axes[2].semilogy(history_array[:, 0], rel_normal, linewidth=2, label="normal")
    axes[2].semilogy(history_array[:, 0], rel_pressure, linewidth=2, label="pressure")
    axes[2].semilogy(history_array[:, 0], rel_jump, linewidth=2, label="jump")
    axes[2].semilogy(history_array[:, 0], rel_sheet_current, linewidth=2, label="sheet current")
    axes[2].set_xlabel("outer objective evaluation")
    axes[2].set_ylabel("relative block norm")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(ncol=2)

    png_path = Path(OUT_DIR) / "boozerQA_finitebeta_history.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved optimization history plot to {png_path}")


def compute_biotsavart_field(surface, biotsavart):
    x = surface.gamma().reshape((-1, 3))
    biotsavart.set_points(x)
    biotsavart.compute(0)
    return biotsavart.B().reshape(surface.gamma().shape)


def resolve_target_pressure_jump_from_coils(surface, biotsavart, target_plasma_beta):
    B_coils = compute_biotsavart_field(surface, biotsavart)
    reference_magnetic_pressure = mean_surface_magnetic_pressure(surface, B_coils)
    pressure_jump = float(target_plasma_beta * reference_magnetic_pressure)
    return pressure_jump, float(reference_magnetic_pressure), B_coils


def run_direct_self_consistent_continuation(
    finite_beta,
    field_provider,
    iota,
    G,
    I,
    current_potential,
    pressure_jump,
    optimize_surface,
    continuation_steps,
):
    continuation_steps = max(1, int(continuation_steps))
    if continuation_steps == 1 or abs(pressure_jump) == 0.0:
        pressure_schedule = np.asarray([float(pressure_jump)])
    else:
        pressure_schedule = np.linspace(0.0, float(pressure_jump), continuation_steps)

    result = None
    current_iota = float(iota)
    current_G = float(G)
    current_I = float(I)
    current_potential_local = np.asarray(current_potential).copy()
    stage_history = []
    for step_pressure in pressure_schedule:
        finite_beta.pressure_jump = float(step_pressure)
        result = finite_beta.run_code(
            iota=current_iota,
            G=current_G,
            I=current_I,
            current_potential=current_potential_local,
            field_provider=field_provider,
            optimize_G=False,
            optimize_surface=optimize_surface,
        )
        current_iota = float(result['iota'])
        current_G = float(result['G'])
        current_I = float(result['I'])
        current_potential_local = np.asarray(result['current_potential']).copy()
        stage_history.append({
            'pressure_jump': float(step_pressure),
            'success': bool(result['success']),
            'nfev': int(result['iter']),
            'weighted_residual_norm': float(result['residual_norm']),
            'iota': float(result['iota']),
            'G': float(result['G']),
            'I': float(result['I']),
        })

    result = dict(result)
    result['continuation_history'] = stage_history
    finite_beta.pressure_jump = float(pressure_jump)
    return result


def write_scalar_summary(name, summary):
    csv_path = Path(OUT_DIR) / f"{name}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            if isinstance(value, (float, np.floating)):
                writer.writerow([key, f"{float(value):.16e}"])
            else:
                writer.writerow([key, value])
    print(f"Saved scalar summary CSV to {csv_path}")


def write_direct_outer_history(history):
    if len(history) == 0:
        return

    history_array = np.asarray(history, dtype=float)
    csv_path = Path(OUT_DIR) / "boozerQA_finitebeta_direct_outer_history.csv"
    header = (
        "eval,J,J_nonqs,J_res,J_iota,J_mr,J_beta,J_surface_reg,weighted_residual_norm,iota,major_radius,"
        "target_plasma_beta,achieved_plasma_beta,reference_magnetic_pressure,pressure_jump,"
        "boozer_norm,normal_norm,pressure_norm,jump_norm,sheet_current_norm"
    )
    np.savetxt(csv_path, history_array, delimiter=",", header=header, comments="")
    print(f"Saved direct outer QA history CSV to {csv_path}")

    if plt is None:
        return

    fig, axes = plt.subplots(4, 1, figsize=(8, 14), constrained_layout=True)
    axes[0].semilogy(history_array[:, 0], np.maximum(history_array[:, 1], 1e-30), linewidth=2, label="J")
    axes[0].semilogy(history_array[:, 0], np.maximum(history_array[:, 2], 1e-30), linewidth=2, label="nonQS")
    axes[0].semilogy(history_array[:, 0], np.maximum(history_array[:, 3], 1e-30), linewidth=2, label="residual penalty")
    axes[0].semilogy(history_array[:, 0], np.maximum(history_array[:, 6], 1e-30), linewidth=2, label="beta penalty")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylabel("objective terms")

    axes[1].semilogy(history_array[:, 0], np.maximum(history_array[:, 8], 1e-30), linewidth=2, label="weighted residual")
    axes[1].semilogy(history_array[:, 0], np.maximum(history_array[:, 15], 1e-30), linewidth=2, label="boozer")
    axes[1].semilogy(history_array[:, 0], np.maximum(history_array[:, 17], 1e-30), linewidth=2, label="pressure")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylabel("residual norms")

    axes[2].plot(history_array[:, 0], history_array[:, 9], linewidth=2, label="iota")
    axes[2].plot(history_array[:, 0], history_array[:, 10], linewidth=2, label="major radius")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_ylabel("state")

    axes[3].plot(history_array[:, 0], history_array[:, 11], linewidth=2, linestyle="--", label="target beta")
    axes[3].plot(history_array[:, 0], history_array[:, 12], linewidth=2, label="achieved beta")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()
    axes[3].set_xlabel("outer evaluation")
    axes[3].set_ylabel("beta")

    png_path = Path(OUT_DIR) / "boozerQA_finitebeta_direct_outer_history.png"
    fig.savefig(png_path, dpi=160)
    plt.close(fig)
    print(f"Saved direct outer QA history plot to {png_path}")


def collect_vmec_beta_summary(vmec):
    beta_summary = {}
    for name in dir(vmec.wout):
        if "beta" not in name.lower():
            continue
        value = getattr(vmec.wout, name)
        if np.isscalar(value):
            beta_summary[name] = float(value)
    if len(beta_summary) == 0:
        for name in ("wb", "wp"):
            if hasattr(vmec.wout, name):
                value = getattr(vmec.wout, name)
                if np.isscalar(value):
                    beta_summary[name] = float(value)
    return beta_summary


def write_vmec_benchmark_outputs(name, direct_field, vmec_field, summary):
    write_scalar_summary(name + "_summary", summary)
    if plt is None:
        return

    direct_modB = np.linalg.norm(direct_field, axis=2)
    vmec_modB = np.linalg.norm(vmec_field, axis=2)
    modB_diff = vmec_modB - direct_modB
    rel_diff = modB_diff / np.maximum(direct_modB, 1e-30)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    datasets = [
        (direct_modB, "Direct |B|"),
        (vmec_modB, "VMEC |B|"),
        (modB_diff, "VMEC - direct |B|"),
        (rel_diff, "(VMEC - direct) / direct |B|"),
    ]
    for ax, (data, title) in zip(axes.flatten(), datasets):
        im = ax.imshow(data.T, origin="lower", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("phi index")
        ax.set_ylabel("theta index")
        fig.colorbar(im, ax=ax, shrink=0.82)

    png_path = Path(OUT_DIR) / f"{name}.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved VMEC benchmark plot to {png_path}")


def apply_field_alignment(field, phi_shift=0, theta_shift=0, flip_phi=False, flip_theta=False):
    aligned = np.asarray(field)
    if flip_phi:
        aligned = np.flip(aligned, axis=0)
    if flip_theta:
        aligned = np.flip(aligned, axis=1)
    aligned = np.roll(aligned, int(phi_shift), axis=0)
    aligned = np.roll(aligned, int(theta_shift), axis=1)
    return aligned


def find_best_field_alignment(reference_field, candidate_field):
    reference = np.asarray(reference_field)
    candidate = np.asarray(candidate_field)
    best = None
    for flip_phi in (False, True):
        for flip_theta in (False, True):
            transformed = apply_field_alignment(candidate, flip_phi=flip_phi, flip_theta=flip_theta)
            for phi_shift in range(reference.shape[0]):
                shifted_phi = np.roll(transformed, phi_shift, axis=0)
                for theta_shift in range(reference.shape[1]):
                    aligned = np.roll(shifted_phi, theta_shift, axis=1)
                    rel_diff = float(np.linalg.norm(aligned - reference) / max(np.linalg.norm(reference), 1e-30))
                    if best is None or rel_diff < best["rel_diff"]:
                        best = {
                            "aligned_field": aligned.copy(),
                            "rel_diff": rel_diff,
                            "phi_shift": int(phi_shift),
                            "theta_shift": int(theta_shift),
                            "flip_phi": bool(flip_phi),
                            "flip_theta": bool(flip_theta),
                        }
    return best


def export_and_benchmark_vmec_from_surface(
    surface,
    direct_field,
    direct_iota,
    direct_I,
    target_plasma_beta,
    reference_magnetic_pressure,
    pressure_jump,
    nphi_benchmark,
    ntheta_benchmark,
):
    from simsopt.mhd.profiles import ProfilePolynomial
    from simsopt.mhd.vmec import Vmec
    from simsopt.mhd.vmec_diagnostics import B_cartesian

    boundary_rz = surface.to_RZFourier()
    vmec_input_name = os.environ.get(
        "SIMSOPT_FINITE_BETA_VMEC_INPUT_NAME",
        "input.boozerQA_finitebeta_direct_beta3",
    )
    vmec_input_path = Path(OUT_DIR) / vmec_input_name

    constraint_mode = os.environ.get(
        "SIMSOPT_FINITE_BETA_VMEC_CONSTRAINT_MODE",
        "iota",
    ).strip().lower()
    if constraint_mode not in ("current", "iota", "none"):
        raise ValueError("SIMSOPT_FINITE_BETA_VMEC_CONSTRAINT_MODE must be one of: current, iota, none")
    direct_toroidal_current = float((2.0 * np.pi / MU0) * float(direct_I))

    def configure_vmec_equilibrium(vmec_obj, axis_pressure_value, phiedge_value, direct_iota_value):
        vmec_obj.boundary = boundary_rz
        vmec_obj.indata.lfreeb = False
        vmec_obj.indata.mpol = int(boundary_rz.mpol)
        vmec_obj.indata.ntor = int(boundary_rz.ntor)
        vmec_obj.indata.phiedge = float(phiedge_value)
        vmec_obj.indata.delt = float(os.environ.get("SIMSOPT_FINITE_BETA_VMEC_DELT", str(vmec_obj.indata.delt)))
        vmec_obj.indata.niter = int(os.environ.get("SIMSOPT_FINITE_BETA_VMEC_NITER", str(vmec_obj.indata.niter)))
        vmec_obj.indata.nstep = int(os.environ.get("SIMSOPT_FINITE_BETA_VMEC_NSTEP", str(vmec_obj.indata.nstep)))
        vmec_obj.indata.ns_array[:] = 0
        vmec_obj.indata.niter_array[:] = 0
        vmec_obj.indata.ftol_array[:] = -1.0
        ns_schedule = [int(value.strip()) for value in os.environ.get("SIMSOPT_FINITE_BETA_VMEC_NS_ARRAY", "13,25,49").split(",") if value.strip()]
        niter_schedule = [int(value.strip()) for value in os.environ.get("SIMSOPT_FINITE_BETA_VMEC_NITER_ARRAY", "400,1200,4000").split(",") if value.strip()]
        ftol_schedule = [float(value.strip()) for value in os.environ.get("SIMSOPT_FINITE_BETA_VMEC_FTOL_ARRAY", "1e-8,1e-10,1e-12").split(",") if value.strip()]
        stage_count = min(len(ns_schedule), len(niter_schedule), len(ftol_schedule), vmec_obj.indata.ns_array.size)
        for idx in range(stage_count):
            vmec_obj.indata.ns_array[idx] = ns_schedule[idx]
            vmec_obj.indata.niter_array[idx] = niter_schedule[idx]
            vmec_obj.indata.ftol_array[idx] = ftol_schedule[idx]
        vmec_obj.pressure_profile = ProfilePolynomial([float(axis_pressure_value), -float(axis_pressure_value)])
        vmec_obj.current_profile = None
        vmec_obj.iota_profile = None
        if constraint_mode == "iota":
            vmec_obj.indata.ncurr = 0
            vmec_obj.indata.piota_type = "power_series"
            vmec_obj.iota_profile = ProfilePolynomial([float(direct_iota_value)])
        elif constraint_mode == "current":
            vmec_obj.indata.ncurr = 1
            vmec_obj.indata.pcurr_type = "power_series"
            vmec_obj.current_profile = ProfilePolynomial([float(direct_toroidal_current)])

    def build_boundary_surface(vmec_obj):
        boundary = SurfaceRZFourier.from_nphi_ntheta(
            mpol=vmec_obj.wout.mpol,
            ntor=vmec_obj.wout.ntor,
            nfp=vmec_obj.wout.nfp,
            stellsym=not bool(vmec_obj.wout.lasym),
            nphi=nphi_benchmark,
            ntheta=ntheta_benchmark,
            range="field period",
        )
        boundary.x = vmec_obj.boundary.x
        return boundary

    calibration_vmec = Vmec(None, verbose=False, nphi=max(nphi_benchmark, 32), ntheta=max(ntheta_benchmark, 32), range_surface="field period")
    configure_vmec_equilibrium(calibration_vmec, axis_pressure_value=0.0, phiedge_value=1.0, direct_iota_value=direct_iota)
    calibration_vmec.run()
    Bx_cal, By_cal, Bz_cal = B_cartesian(calibration_vmec, nphi=nphi_benchmark, ntheta=ntheta_benchmark, range="field period")
    calibration_field = np.stack((Bx_cal, By_cal, Bz_cal), axis=2)
    calibration_boundary = build_boundary_surface(calibration_vmec)
    calibration_magnetic_pressure = mean_surface_magnetic_pressure(calibration_boundary, calibration_field)
    phiedge_scale = float(np.sqrt(reference_magnetic_pressure / max(calibration_magnetic_pressure, 1e-30)))

    axis_pressure = float(max(2.0 * target_plasma_beta * reference_magnetic_pressure, 0.0))
    vmec = Vmec(None, verbose=False, nphi=max(nphi_benchmark, 32), ntheta=max(ntheta_benchmark, 32), range_surface="field period")
    configure_vmec_equilibrium(vmec, axis_pressure_value=axis_pressure, phiedge_value=phiedge_scale, direct_iota_value=direct_iota)
    vmec.write_input(str(vmec_input_path))
    print(f"Wrote VMEC input to {vmec_input_path}")

    vmec_run = Vmec(str(vmec_input_path), verbose=False, nphi=max(nphi_benchmark, 32), ntheta=max(ntheta_benchmark, 32), range_surface="field period")
    vmec_run.run()
    Bx, By, Bz = B_cartesian(vmec_run, nphi=nphi_benchmark, ntheta=ntheta_benchmark, range="field period")
    vmec_field = np.stack((Bx, By, Bz), axis=2)
    best_alignment = find_best_field_alignment(direct_field, vmec_field)
    aligned_vmec_field = best_alignment["aligned_field"]
    boundary_vmec = build_boundary_surface(vmec_run)
    beta_summary = collect_vmec_beta_summary(vmec_run)
    benchmark_summary = {
        'target_plasma_beta': float(target_plasma_beta),
        'direct_pressure_jump': float(pressure_jump),
        'direct_iota': float(direct_iota),
        'direct_I': float(direct_I),
        'direct_toroidal_current': float(direct_toroidal_current),
        'vmec_constraint_mode': constraint_mode,
        'vmec_axis_pressure': float(axis_pressure),
        'vmec_phiedge': float(phiedge_scale),
        'vmec_reference_magnetic_pressure': float(mean_surface_magnetic_pressure(boundary_vmec, vmec_field)),
        'direct_nonqs': float(surface_field_nonquasisymmetric_ratio(surface, direct_field)),
        'vmec_nonqs': float(surface_field_nonquasisymmetric_ratio(boundary_vmec, vmec_field)),
        'direct_vs_vmec_rel_field_diff': float(np.linalg.norm(vmec_field - direct_field) / max(np.linalg.norm(direct_field), 1e-30)),
        'direct_vs_vmec_rel_field_diff_aligned': float(best_alignment['rel_diff']),
        'vmec_alignment_phi_shift': int(best_alignment['phi_shift']),
        'vmec_alignment_theta_shift': int(best_alignment['theta_shift']),
        'vmec_alignment_flip_phi': int(best_alignment['flip_phi']),
        'vmec_alignment_flip_theta': int(best_alignment['flip_theta']),
        'vmec_iota_axis': float(vmec_run.iota_axis()),
        'vmec_iota_edge': float(vmec_run.iota_edge()),
        'vmec_mean_iota': float(vmec_run.mean_iota()),
        'vmec_toroidal_current': float(vmec_run.wout.ctor),
        'vmec_toroidal_current_rel_error': float((float(vmec_run.wout.ctor) - direct_toroidal_current) / max(abs(direct_toroidal_current), 1e-30)),
        'vmec_volume': float(vmec_run.volume()),
    }
    benchmark_summary.update(beta_summary)
    write_vmec_benchmark_outputs("boozerQA_finitebeta_direct_vmec_benchmark", direct_field, aligned_vmec_field, benchmark_summary)
    return benchmark_summary, vmec_input_path


if mode == "self-consistent":
    print("Using direct no-VC self-consistent finite-beta case")
    print(
        f"Mode={mode}, grid: nphi={nphi}, ntheta={ntheta}, "
        f"ls_max_nfev={ls_max_nfev}, qa_maxiter={qa_maxiter}"
    )

    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    all_curves = [coil.curve for coil in bs.coils]
    for current in base_currents:
        current.fix_all()
    current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
    mpol, ntor, phis, thetas = stellsym_exact_grid(nphi, ntheta)
    print(
        f"Using Boozer-exact seed grid nphi={len(phis)}, ntheta={len(thetas)}, "
        f"mpol={mpol}, ntor={ntor}"
    )
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)
    optimize_surface = os.environ.get("SIMSOPT_FINITE_BETA_OPTIMIZE_SURFACE", "1").strip().lower() in ("1", "true", "yes", "on")
    continuation_steps = int(os.environ.get("SIMSOPT_FINITE_BETA_SELFCONSISTENT_CONTINUATION_STEPS", "4"))
    do_outer_qa = os.environ.get("SIMSOPT_FINITE_BETA_OUTER_QA", "0").strip().lower() in ("1", "true", "yes", "on")
    qa_dof_count = int(os.environ.get("SIMSOPT_FINITE_BETA_QA_DOF_COUNT", "12"))
    direct_qa_continuation_steps = int(os.environ.get("SIMSOPT_FINITE_BETA_SELFCONSISTENT_QA_CONTINUATION_STEPS", "2"))
    residual_penalty_weight = float(os.environ.get("SIMSOPT_FINITE_BETA_QA_RESIDUAL_WEIGHT", "1.0"))
    beta_penalty_weight = float(os.environ.get("SIMSOPT_FINITE_BETA_QA_BETA_WEIGHT", "10.0"))
    surface_reg_weight = float(os.environ.get("SIMSOPT_FINITE_BETA_QA_SURFACE_REG_WEIGHT", "1e-2"))
    export_vmec = os.environ.get("SIMSOPT_FINITE_BETA_EXPORT_VMEC", "0").strip().lower() in ("1", "true", "yes", "on")
    field_provider = SurfaceCurrentFieldProvider(
        offset_distance=float(os.environ["SIMSOPT_FINITE_BETA_SELFCONSISTENT_OFFSET"]) if "SIMSOPT_FINITE_BETA_SELFCONSISTENT_OFFSET" in os.environ else None,
        offset_scale=float(os.environ.get("SIMSOPT_FINITE_BETA_SELFCONSISTENT_OFFSET_SCALE", "0.25")),
        min_offset=float(os.environ.get("SIMSOPT_FINITE_BETA_SELFCONSISTENT_MIN_OFFSET", "1e-5")),
        chunk_size=int(os.environ.get("SIMSOPT_FINITE_BETA_SELFCONSISTENT_CHUNK_SIZE", "0")) or None,
    )
    vol = Volume(surface)
    vol_target = vol.J()

    seed_surface = BoozerSurface(bs, surface, vol, vol_target)
    seed = seed_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=20, iota=-0.35, G=G0)
    surface.x = seed_surface.surface.x.copy()
    print(f"Vacuum seed: success={seed['success']}, iota={seed['iota']:.6e}, G={seed['G']:.6e}")

    B_vacuum = compute_biotsavart_field(surface, bs)
    pressure_jump, plasma_beta, reference_magnetic_pressure = resolve_pressure_jump_input(surface, B_vacuum)
    continuation_steps = max(1, continuation_steps if abs(pressure_jump) > 0.0 else 1)
    print(
        f"Target finite-beta input: pressure_jump={pressure_jump:.6e}, plasma_beta={plasma_beta:.6%}, "
        f"reference <B^2/(2 mu0)>={reference_magnetic_pressure:.6e}"
    )

    finite_beta = FiniteBetaBoozerSurface(
        bs,
        surface,
        vol,
        vol_target,
        pressure_jump=0.0,
        options={
            'verbose': inner_verbose,
            'ls_max_nfev': ls_max_nfev,
        },
    )

    initial_potential = np.zeros(surface.gamma().shape[:2])
    initial = finite_beta.residual_blocks(
        iota=seed['iota'],
        G=seed['G'],
        I=0.0,
        current_potential=initial_potential,
        field_provider=field_provider,
    )
    print_blocks("Initial direct self-consistent residual norms:", initial)
    write_curves_vtk("curves_init", all_curves)
    write_surface_vtk("surf_init", surface)
    write_residual_vtk(surface, initial)
    if write_closure_diagnostics:
        initial_closure = finite_beta.frozen_state_closure_diagnostics(
            iota=seed['iota'],
            G=seed['G'],
            I=0.0,
            current_potential=initial_potential,
            field_provider=field_provider,
            pressure_jump=0.0,
            vc_digits=closure_vc_digits,
        )
        initial_closure_summary = write_frozen_closure_outputs(
            "boozerQA_finitebeta_direct_vs_vc_initial",
            initial_closure,
        )
        print(
            f"Frozen initial direct-vs-VC: rel ||Bext_direct-Bext_vc||="
            f"{initial_closure_summary['B_external_rel_norm']:.6e}, "
            f"direct jump={initial_closure_summary['direct_jump_norm']:.6e}, "
            f"vc pressure={initial_closure_summary['vc_pressure_norm']:.6e}"
        )
    print(f"Saved initial reference curves to {Path(OUT_DIR) / 'curves_init.vtu'}")
    print(f"Saved initial surface to {Path(OUT_DIR) / 'surf_init.vts'}")

    if continuation_steps == 1:
        continuation_pressures = np.asarray([pressure_jump], dtype=float)
    else:
        continuation_pressures = np.linspace(0.0, pressure_jump, continuation_steps)
    result = None
    current_iota = seed['iota']
    current_G = seed['G']
    current_I = 0.0
    current_potential = initial_potential
    for step_index, step_pressure in enumerate(continuation_pressures, start=1):
        result = run_direct_self_consistent_continuation(
            finite_beta,
            field_provider,
            iota=current_iota,
            G=current_G,
            I=current_I,
            current_potential=current_potential,
            pressure_jump=step_pressure,
            optimize_surface=optimize_surface,
            continuation_steps=1,
        )
        current_iota = result['iota']
        current_G = result['G']
        current_I = result['I']
        current_potential = result['current_potential']
        print(
            f"Continuation step {step_index}/{len(continuation_pressures)}: pressure_jump={step_pressure:.6e}, "
            f"success={result['success']}, iota={result['iota']:.6e}, I={result['I']:.6e}, ||r||={result['residual_norm']:.6e}"
        )

    if do_outer_qa and qa_maxiter > 0:
        surface_dofs0 = surface.x.copy()
        qa_dof_count = min(max(1, qa_dof_count), surface_dofs0.size)
        subset_idx = np.arange(qa_dof_count)
        outer_history = []
        iota_ref = float(result['iota'])
        mr_ref = float(surface.major_radius())

        def outer_fun(subset_dofs):
            full_dofs = surface_dofs0.copy()
            full_dofs[subset_idx] = subset_dofs
            surface.x = full_dofs

            target_pressure_jump, current_reference_magnetic_pressure, _ = resolve_target_pressure_jump_from_coils(
                surface,
                bs,
                plasma_beta,
            )
            solved_result = run_direct_self_consistent_continuation(
                finite_beta,
                field_provider,
                iota=finite_beta.res['iota'],
                G=finite_beta.res['G'],
                I=finite_beta.res['I'],
                current_potential=finite_beta.res['current_potential'],
                pressure_jump=target_pressure_jump,
                optimize_surface=False,
                continuation_steps=direct_qa_continuation_steps,
            )
            solved_blocks = finite_beta.residual_blocks(
                iota=solved_result['iota'],
                G=solved_result['G'],
                I=solved_result['I'],
                current_potential=solved_result['current_potential'],
                field_provider=field_provider,
                pressure_jump=target_pressure_jump,
            )
            B_in_local, _ = finite_beta.resolve_field_components(
                field_provider=field_provider,
                iota=solved_result['iota'],
                G=solved_result['G'],
                I=solved_result['I'],
                current_potential=solved_result['current_potential'],
            )
            J_nonqs = surface_field_nonquasisymmetric_ratio(surface, B_in_local)
            J_res = 0.5 * residual_penalty_weight * solved_result['residual_norm']**2
            J_iota = 0.5 * (solved_result['iota'] - iota_ref)**2
            J_mr = 0.5 * (surface.major_radius() - mr_ref)**2
            actual_beta = achieved_plasma_beta(surface, B_in_local, target_pressure_jump)
            J_beta = 0.5 * beta_penalty_weight * (actual_beta - plasma_beta)**2
            J_surface_reg = 0.5 * surface_reg_weight * np.sum((subset_dofs - surface_dofs0[subset_idx])**2)
            J = J_nonqs + J_res + J_iota + J_mr + J_beta + J_surface_reg

            outer_history.append([
                len(outer_history) + 1,
                J,
                J_nonqs,
                J_res,
                J_iota,
                J_mr,
                J_beta,
                J_surface_reg,
                solved_result['residual_norm'],
                solved_result['iota'],
                surface.major_radius(),
                plasma_beta,
                actual_beta,
                current_reference_magnetic_pressure,
                target_pressure_jump,
                np.linalg.norm(solved_blocks['blocks']['boozer']),
                np.linalg.norm(solved_blocks['blocks']['normal']),
                np.linalg.norm(solved_blocks['blocks']['pressure']),
                np.linalg.norm(solved_blocks['blocks']['jump']),
                np.linalg.norm(solved_blocks['blocks']['sheet_current']),
            ])
            print(
                f"Direct-QA eval #{len(outer_history)}: J={J:.6e}, nonQS={J_nonqs:.6e}, beta={actual_beta:.6%}, "
                f"res={solved_result['residual_norm']:.6e}, iota={solved_result['iota']:.6e}, "
                f"mr={surface.major_radius():.6e}, pressure_jump={target_pressure_jump:.6e}"
            )
            return J

        print(
            f"Running reduced direct outer QA optimization on {qa_dof_count} surface dofs "
            f"with maxiter={qa_maxiter}"
        )
        opt_result = minimize(
            outer_fun,
            surface_dofs0[subset_idx],
            method="L-BFGS-B",
            options={"maxiter": qa_maxiter, "maxfun": max(40, qa_maxiter * (qa_dof_count + 1))},
        )
        print(
            f"Outer direct QA optimization: success={opt_result.success}, status={opt_result.status}, "
            f"nit={opt_result.nit}, nfev={opt_result.nfev}, final_J={opt_result.fun:.6e}"
        )

        surface_dofs_opt = surface_dofs0.copy()
        surface_dofs_opt[subset_idx] = opt_result.x
        surface.x = surface_dofs_opt
        pressure_jump, reference_magnetic_pressure, _ = resolve_target_pressure_jump_from_coils(surface, bs, plasma_beta)
        result = run_direct_self_consistent_continuation(
            finite_beta,
            field_provider,
            iota=finite_beta.res['iota'],
            G=finite_beta.res['G'],
            I=finite_beta.res['I'],
            current_potential=finite_beta.res['current_potential'],
            pressure_jump=pressure_jump,
            optimize_surface=False,
            continuation_steps=max(continuation_steps, direct_qa_continuation_steps),
        )
        write_direct_outer_history(outer_history)

    solved = finite_beta.residual_blocks(
        iota=result['iota'],
        G=result['G'],
        I=result['I'],
        current_potential=result['current_potential'],
        field_provider=field_provider,
    )
    B_in, B_out = finite_beta.resolve_field_components(
        field_provider=field_provider,
        iota=result['iota'],
        G=result['G'],
        I=result['I'],
        current_potential=result['current_potential'],
    )
    print_blocks("Solved direct self-consistent residual norms:", solved)
    print_solver_summary("Direct self-consistent solve", result)
    print(
        f"Final nonQS ratio={surface_field_nonquasisymmetric_ratio(surface, B_in):.6e}, "
        f"achieved plasma_beta={achieved_plasma_beta(surface, B_in, finite_beta.pressure_jump):.6%}, "
        f"||B_out-B_in||={np.linalg.norm(B_out - B_in):.6e}"
    )
    write_inner_continuation_history("boozerQA_finitebeta_direct_inner_continuation", result.get("continuation_history", []))
    if write_closure_diagnostics:
        final_closure = finite_beta.frozen_state_closure_diagnostics(
            iota=result['iota'],
            G=result['G'],
            I=result['I'],
            current_potential=result['current_potential'],
            field_provider=field_provider,
            pressure_jump=finite_beta.pressure_jump,
            vc_digits=closure_vc_digits,
        )
        final_closure_summary = write_frozen_closure_outputs(
            "boozerQA_finitebeta_direct_vs_vc_final",
            final_closure,
        )
        print(
            f"Frozen final direct-vs-VC: rel ||Bext_direct-Bext_vc||="
            f"{final_closure_summary['B_external_rel_norm']:.6e}, "
            f"direct jump={final_closure_summary['direct_jump_norm']:.6e}, "
            f"direct pressure={final_closure_summary['direct_pressure_norm']:.6e}, "
            f"vc pressure={final_closure_summary['vc_pressure_norm']:.6e}"
        )
    write_curves_vtk("curves_opt", all_curves)
    write_surface_vtk("surf_opt", surface)
    write_residual_vtk(surface, solved)
    if export_vmec:
        benchmark_summary, vmec_input_path = export_and_benchmark_vmec_from_surface(
            surface,
            B_in,
            direct_iota=result['iota'],
            direct_I=result['I'],
            target_plasma_beta=plasma_beta,
            reference_magnetic_pressure=reference_magnetic_pressure,
            pressure_jump=finite_beta.pressure_jump,
            nphi_benchmark=surface.quadpoints_phi.size,
            ntheta_benchmark=surface.quadpoints_theta.size,
        )
        print(
            f"VMEC benchmark: input={vmec_input_path.name}, raw rel field diff={benchmark_summary['direct_vs_vmec_rel_field_diff']:.6e}, "
            f"aligned rel field diff={benchmark_summary['direct_vs_vmec_rel_field_diff_aligned']:.6e}, "
            f"VMEC nonQS={benchmark_summary['vmec_nonqs']:.6e}, iota_edge={benchmark_summary['vmec_iota_edge']:.6e}, "
            f"toroidal current={benchmark_summary['vmec_toroidal_current']:.6e} A"
        )
        beta_keys = [key for key in benchmark_summary if 'beta' in key.lower() and key not in ('target_plasma_beta',)]
        if len(beta_keys) > 0:
            print("VMEC beta summary: " + ", ".join(f"{key}={benchmark_summary[key]:.6e}" for key in beta_keys))
    print(f"Saved final reference curves to {Path(OUT_DIR) / 'curves_opt.vtu'}")
    print(f"Saved final surface to {Path(OUT_DIR) / 'surf_opt.vts'}")
    print("Direct no-VC self-consistent finite-beta solve complete.")

elif mode == "single-surface-vc":
    print("Using single-surface virtual-casing finite-beta validation case")
    print(
        f"Mode={mode}, grid: nphi={nphi}, ntheta={ntheta}, "
        f"ls_max_nfev={ls_max_nfev}, qa_maxiter={qa_maxiter}"
    )
    print("This branch is a VC-closed validation surrogate, not a pure no-VC self-consistent finite-beta solve.")

    base_curves, base_currents, ma, nfp, bs = get_data("ncsx")
    all_curves = [coil.curve for coil in bs.coils]
    for current in base_currents:
        current.fix_all()
    current_sum = nfp * sum(abs(c.get_value()) for c in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))
    mpol, ntor, phis, thetas = stellsym_exact_grid(nphi, ntheta)
    print(
        f"Using Boozer-exact seed grid nphi={len(phis)}, ntheta={len(thetas)}, "
        f"mpol={mpol}, ntor={ntor}"
    )
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)
    optimize_surface = os.environ.get("SIMSOPT_FINITE_BETA_OPTIMIZE_SURFACE", "1").strip().lower() in ("1", "true", "yes", "on")
    do_outer_qa = os.environ.get("SIMSOPT_FINITE_BETA_OUTER_QA", "0").strip().lower() in ("1", "true", "yes", "on")
    qa_dof_count = int(os.environ.get("SIMSOPT_FINITE_BETA_QA_DOF_COUNT", "3"))
    vc_digits = int(os.environ.get("SIMSOPT_FINITE_BETA_VC_DIGITS", "4"))
    residual_penalty_weight = float(os.environ.get("SIMSOPT_FINITE_BETA_QA_RESIDUAL_WEIGHT", "1.0"))
    beta_penalty_weight = float(os.environ.get("SIMSOPT_FINITE_BETA_QA_BETA_WEIGHT", "1.0"))
    coil_reg_weight = float(os.environ.get("SIMSOPT_FINITE_BETA_QA_COIL_REG_WEIGHT", "1e-3"))
    vol = Volume(surface)
    vol_target = vol.J()

    seed_surface = BoozerSurface(bs, surface, vol, vol_target)
    seed = seed_surface.solve_residual_equation_exactly_newton(tol=1e-12, maxiter=20, iota=-0.35, G=G0)
    surface.x = seed_surface.surface.x.copy()
    print(f"Vacuum seed: success={seed['success']}, iota={seed['iota']:.6e}, G={seed['G']:.6e}")

    vacuum_finite_beta = FiniteBetaBoozerSurface(
        bs,
        surface,
        vol,
        vol_target,
        pressure_jump=0.0,
        options={
            'verbose': inner_verbose,
            'ls_max_nfev': ls_max_nfev,
            'vc_digits': vc_digits,
            'vc_pressure_continuation_steps': 1,
            'vc_surface_hybrid_jacobian': True,
            'vc_surface_fd_rel_step': float(os.environ.get("SIMSOPT_FINITE_BETA_VC_SURFACE_FD_STEP", "1e-7")),
        },
    )

    vacuum_reference = vacuum_finite_beta.self_consistent_single_surface_residual(iota=seed['iota'], G=seed['G'], I=0.0)
    pressure_jump, plasma_beta, reference_magnetic_pressure = resolve_pressure_jump_input(surface, vacuum_reference['B_total'])
    pressure_scan_count = int(os.environ.get("SIMSOPT_FINITE_BETA_PRESSURE_SCAN_COUNT", "5" if abs(pressure_jump) > 0.0 else "1"))
    pressure_scan_max = float(os.environ.get("SIMSOPT_FINITE_BETA_PRESSURE_SCAN_MAX", str(pressure_jump)))
    vc_continuation_steps = int(os.environ.get("SIMSOPT_FINITE_BETA_VC_CONTINUATION_STEPS", "4" if abs(pressure_jump) > 0.0 else "1"))
    print(
        f"Target finite-beta input: pressure_jump={pressure_jump:.6e}, plasma_beta={plasma_beta:.6%}, "
        f"reference <B^2/(2 mu0)>={reference_magnetic_pressure:.6e}"
    )

    finite_beta = FiniteBetaBoozerSurface(
        bs,
        surface,
        vol,
        vol_target,
        pressure_jump=pressure_jump,
        options={
            'verbose': inner_verbose,
            'ls_max_nfev': ls_max_nfev,
            'vc_digits': vc_digits,
            'vc_pressure_continuation_steps': vc_continuation_steps,
            'vc_surface_hybrid_jacobian': True,
            'vc_surface_fd_rel_step': float(os.environ.get("SIMSOPT_FINITE_BETA_VC_SURFACE_FD_STEP", "1e-7")),
        },
    )

    initial = finite_beta.self_consistent_single_surface_residual(iota=seed['iota'], G=seed['G'], I=0.0)
    initial_curve_geometries = capture_curve_geometries(all_curves)
    initial_surface_dofs = surface.x.copy()
    initial_surface_gamma = surface.gamma().copy()
    print_blocks("Initial single-surface VC residual norms:", {'blocks': initial['blocks']})
    write_curves_vtk("curves_init", all_curves)
    write_surface_vtk("surf_init", surface)
    write_single_surface_vc_vtk(surface, initial, prefix="boozerQA_finitebeta_boundary_init")
    write_single_surface_maps("boozerQA_finitebeta_initial_maps", initial)
    print(f"Saved initial reference curves to {Path(OUT_DIR) / 'curves_init.vtu'}")
    print(f"Saved initial surface to {Path(OUT_DIR) / 'surf_init.vts'}")

    result = finite_beta.run_code_single_surface_vc(
        iota=seed['iota'],
        G=seed['G'],
        I=0.0,
        optimize_iota=True,
        optimize_lambda_current=True,
        optimize_surface=optimize_surface,
    )

    solved = finite_beta.self_consistent_single_surface_residual(
        iota=result['iota'],
        G=result['G'],
        I=result['I'],
        lambda_current=result['lambda_current'],
    )
    print_blocks("Solved single-surface VC residual norms:", {'blocks': solved['blocks']})
    print(
        f"Single-surface VC solve: success={result['success']}, lsq_nfev={result['iter']}, "
        f"iota={result['iota']:.6e}, G={result['G']:.6e}, I={result['I']:.6e}, "
        f"lambda={result['lambda_current']:.6e}, ||r||={result['residual_norm']:.6e}, "
        f"raw ||r||={result['raw_residual_norm']:.6e}"
    )
    print(
        f"Final nonQS ratio={surface_field_nonquasisymmetric_ratio(surface, result['B_total']):.6e}, "
        f"achieved plasma_beta={achieved_plasma_beta(surface, result['B_total'], finite_beta.pressure_jump):.6%}"
    )

    if do_outer_qa and qa_maxiter > 0 and bs.x.size > 0:
        curve_lengths = [CurveLength(c) for c in base_curves]
        length_target = float(sum(cl.J() for cl in curve_lengths))
        bs_dofs0 = bs.x.copy()
        qa_dof_count = min(qa_dof_count, bs_dofs0.size)
        subset_idx = np.arange(qa_dof_count)
        outer_history = []
        iota_ref = result['iota']
        mr_ref = surface.major_radius()

        def outer_fun(subset_dofs):
            previous_surface = finite_beta.surface.x.copy()
            previous_res = finite_beta.res.copy() if finite_beta.res is not None else None
            previous_pressure_jump = finite_beta.pressure_jump

            full_dofs = bs_dofs0.copy()
            full_dofs[subset_idx] = subset_dofs
            bs.x = full_dofs

            try:
                target_pressure_jump, current_reference_magnetic_pressure, _ = retarget_pressure_jump_to_beta(
                    finite_beta,
                    iota=finite_beta.res['iota'],
                    G=finite_beta._default_G(),
                    I=finite_beta.res['I'],
                    lambda_current=finite_beta.res['lambda_current'],
                    target_plasma_beta=plasma_beta,
                )
                solved_result = finite_beta.run_code_single_surface_vc(
                    iota=finite_beta.res['iota'],
                    G=finite_beta._default_G(),
                    I=finite_beta.res['I'],
                    optimize_iota=True,
                    optimize_lambda_current=True,
                    optimize_surface=optimize_surface,
                )
                solved_blocks = finite_beta.self_consistent_single_surface_residual(
                    iota=solved_result['iota'],
                    G=solved_result['G'],
                    I=solved_result['I'],
                    lambda_current=solved_result['lambda_current'],
                )
                J_nonqs = surface_field_nonquasisymmetric_ratio(surface, solved_result['B_total'])
                J_res = 0.5 * residual_penalty_weight * solved_result['residual_norm']**2
                J_iota = 0.5 * (solved_result['iota'] - iota_ref)**2
                J_mr = 0.5 * (surface.major_radius() - mr_ref)**2
                actual_beta = achieved_plasma_beta(surface, solved_result['B_total'], finite_beta.pressure_jump)
                J_beta = 0.5 * beta_penalty_weight * (actual_beta - plasma_beta)**2
                total_length = sum(cl.J() for cl in curve_lengths)
                J_length = 0.5 * max(total_length - length_target, 0.0)**2
                J_coil_reg = 0.5 * coil_reg_weight * np.sum((subset_dofs - bs_dofs0[subset_idx])**2)
                J = J_nonqs + J_res + J_iota + J_mr + J_beta + J_length + J_coil_reg
            except Exception:
                if previous_res is not None:
                    finite_beta.res = previous_res
                finite_beta.surface.x = previous_surface
                finite_beta.pressure_jump = previous_pressure_jump
                J = 1e6
                solved_result = previous_res if previous_res is not None else result
                solved_blocks = solved
                J_nonqs = 1e6
                J_res = 1e6
                J_iota = 0.0
                J_mr = 0.0
                J_beta = 0.0
                J_length = 0.0
                J_coil_reg = 0.0
                target_pressure_jump = previous_pressure_jump
                current_reference_magnetic_pressure = np.nan
                actual_beta = np.nan

            outer_history.append([
                len(outer_history) + 1,
                J,
                J_nonqs,
                J_res,
                J_iota,
                J_mr,
                J_beta,
                J_length,
                J_coil_reg,
                solved_result['raw_residual_norm'],
                solved_result['residual_norm'],
                solved_result['iota'],
                surface.major_radius(),
                plasma_beta,
                actual_beta,
                current_reference_magnetic_pressure,
                target_pressure_jump,
                np.linalg.norm(solved_blocks['blocks']['coil_match']),
                np.linalg.norm(solved_blocks['blocks']['normal']),
                np.linalg.norm(solved_blocks['blocks']['pressure']),
                np.linalg.norm(solved_blocks['blocks']['sheet_current']),
            ])

            print(
                f"VC-QA eval #{len(outer_history)}: J={J:.6e}, nonQS={J_nonqs:.6e}, beta={actual_beta:.6%}, "
                f"res={solved_result['residual_norm']:.6e}, raw_res={solved_result['raw_residual_norm']:.6e}, "
                f"iota={solved_result['iota']:.6e}, mr={surface.major_radius():.6e}, "
                f"pressure_jump={target_pressure_jump:.6e}"
            )
            return J

        print(
            f"Running reduced outer coil QA optimization on {qa_dof_count} coil dofs "
            f"with maxiter={qa_maxiter}"
        )
        opt_result = minimize(
            outer_fun,
            bs_dofs0[subset_idx],
            method="L-BFGS-B",
            options={"maxiter": qa_maxiter, "maxfun": max(20, qa_maxiter * (qa_dof_count + 1))},
        )
        print(
            f"Outer VC-QA optimization: success={opt_result.success}, status={opt_result.status}, "
            f"nit={opt_result.nit}, nfev={opt_result.nfev}, final_J={opt_result.fun:.6e}"
        )

        bs_dofs_opt = bs_dofs0.copy()
        bs_dofs_opt[subset_idx] = opt_result.x
        bs.x = bs_dofs_opt
        pressure_jump, reference_magnetic_pressure, _ = retarget_pressure_jump_to_beta(
            finite_beta,
            iota=finite_beta.res['iota'],
            G=finite_beta._default_G(),
            I=finite_beta.res['I'],
            lambda_current=finite_beta.res['lambda_current'],
            target_plasma_beta=plasma_beta,
        )
        result = finite_beta.run_code_single_surface_vc(
            iota=finite_beta.res['iota'],
            G=finite_beta._default_G(),
            I=finite_beta.res['I'],
            optimize_iota=True,
            optimize_lambda_current=True,
            optimize_surface=optimize_surface,
        )
        solved = finite_beta.self_consistent_single_surface_residual(
            iota=result['iota'],
            G=result['G'],
            I=result['I'],
            lambda_current=result['lambda_current'],
        )
        print_blocks("Post-QA single-surface VC residual norms:", {'blocks': solved['blocks']})
        print(
            f"Post-QA nonQS ratio={surface_field_nonquasisymmetric_ratio(surface, result['B_total']):.6e}, "
            f"achieved plasma_beta={achieved_plasma_beta(surface, result['B_total'], finite_beta.pressure_jump):.6%}"
        )
        write_single_surface_outer_history(outer_history)

    scan_rows = []
    if pressure_scan_count > 1 and pressure_scan_max >= 0.0:
        print(
            f"Running pressure continuation scan with {pressure_scan_count} points from 0 to {pressure_scan_max:.6e}"
        )
        original_pressure_jump = finite_beta.pressure_jump
        scan_pressures = np.linspace(0.0, pressure_scan_max, pressure_scan_count)
        vacuum_result = None
        for idx, scan_pressure in enumerate(scan_pressures):
            finite_beta.pressure_jump = float(scan_pressure)
            scan_result = finite_beta.run_code_single_surface_vc(
                iota=finite_beta.res['iota'],
                G=finite_beta._default_G(),
                I=finite_beta.res['I'],
                optimize_iota=True,
                optimize_lambda_current=True,
                optimize_surface=optimize_surface,
            )
            scan_blocks = finite_beta.self_consistent_single_surface_residual(
                iota=scan_result['iota'],
                G=scan_result['G'],
                I=scan_result['I'],
                lambda_current=scan_result['lambda_current'],
                pressure_jump=scan_pressure,
            )
            if vacuum_result is None:
                vacuum_result = {
                    'iota': scan_result['iota'],
                    'B_total': scan_result['B_total'].copy(),
                }
            rel_field_diff = np.linalg.norm(scan_result['B_total'] - vacuum_result['B_total']) / max(
                np.linalg.norm(vacuum_result['B_total']),
                1e-30,
            )
            scan_rows.append([
                scan_pressure,
                scan_pressure / max(reference_magnetic_pressure, 1e-30),
                scan_result['iota'],
                scan_result['I'],
                scan_result['lambda_current'],
                surface_field_nonquasisymmetric_ratio(surface, scan_result['B_total']),
                surface.major_radius(),
                scan_result['raw_residual_norm'],
                scan_result['residual_norm'],
                np.linalg.norm(scan_blocks['blocks']['coil_match']),
                np.linalg.norm(scan_blocks['blocks']['normal']),
                np.linalg.norm(scan_blocks['blocks']['pressure']),
                np.linalg.norm(scan_blocks['blocks']['sheet_current']),
                abs(scan_result['iota'] - vacuum_result['iota']),
                rel_field_diff,
            ])
            prefix = f"boozerQA_finitebeta_scan_{idx:03d}"
            write_single_surface_vc_vtk(surface, scan_blocks, prefix=prefix)
            if idx in (0, len(scan_pressures) - 1):
                write_single_surface_maps(prefix + "_maps", scan_blocks)
            print(
                f"Pressure scan point {idx + 1}/{len(scan_pressures)}: pressure_jump={scan_pressure:.6e}, "
                f"plasma_beta={scan_pressure / max(reference_magnetic_pressure, 1e-30):.6%}, "
                f"iota={scan_result['iota']:.6e}, raw_res={scan_result['raw_residual_norm']:.6e}, "
                f"nonQS={surface_field_nonquasisymmetric_ratio(surface, scan_result['B_total']):.6e}"
            )

        finite_beta.pressure_jump = original_pressure_jump
        if abs(original_pressure_jump - pressure_scan_max) > 1e-15:
            result = finite_beta.run_code_single_surface_vc(
                iota=finite_beta.res['iota'],
                G=finite_beta._default_G(),
                I=finite_beta.res['I'],
                optimize_iota=True,
                optimize_lambda_current=True,
                optimize_surface=optimize_surface,
            )
            solved = finite_beta.self_consistent_single_surface_residual(
                iota=result['iota'],
                G=result['G'],
                I=result['I'],
                lambda_current=result['lambda_current'],
            )
        write_pressure_scan_outputs(scan_rows)
        if len(scan_rows) > 0:
            print(
                f"Vacuum-limit check: at pressure_jump=0, iota={scan_rows[0][2]:.6e}; "
                f"at pressure_jump={scan_rows[-1][0]:.6e}, |iota-iota_vac|={scan_rows[-1][13]:.6e}, "
                f"rel |B-B_vac|={scan_rows[-1][14]:.6e}"
            )

    final_curve_geometries = capture_curve_geometries(all_curves)
    final_surface_gamma = surface.gamma().copy()
    final_reference_magnetic_pressure = reference_magnetic_pressure
    if np.isfinite(plasma_beta):
        pressure_jump, final_reference_magnetic_pressure, _ = retarget_pressure_jump_to_beta(
            finite_beta,
            iota=result['iota'],
            G=result['G'],
            I=result['I'],
            lambda_current=result['lambda_current'],
            target_plasma_beta=plasma_beta,
        )
    initial_surface = clone_surface(surface)
    initial_surface.x = initial_surface_dofs.copy()
    initial_summary = summarize_single_surface_state(
        initial_surface,
        initial,
        pressure_jump,
        target_plasma_beta=plasma_beta,
        reference_magnetic_pressure=reference_magnetic_pressure,
    )
    initial_summary["iota"] = float(seed['iota'])
    initial_summary["G"] = float(seed['G'])
    initial_summary["I"] = 0.0
    initial_summary["lambda_current"] = float(seed['G'])
    final_summary = summarize_single_surface_state(
        surface,
        solved,
        finite_beta.pressure_jump,
        target_plasma_beta=plasma_beta,
        reference_magnetic_pressure=final_reference_magnetic_pressure,
    )
    final_summary["iota"] = float(result['iota'])
    final_summary["G"] = float(result['G'])
    final_summary["I"] = float(result['I'])
    final_summary["lambda_current"] = float(result['lambda_current'])
    write_geometry_comparison(
        "boozerQA_finitebeta_geometry_comparison",
        initial_curve_geometries,
        initial_surface_gamma,
        final_curve_geometries,
        final_surface_gamma,
    )
    write_state_comparison(
        "boozerQA_finitebeta_state_comparison",
        initial_surface,
        initial,
        surface,
        solved,
    )
    write_objective_comparison("boozerQA_finitebeta_objective_comparison", initial_summary, final_summary)
    write_inner_continuation_history("boozerQA_finitebeta_inner_continuation", result.get("continuation_history", []))

    write_curves_vtk("curves_opt", all_curves)
    write_surface_vtk("surf_opt", surface)
    write_single_surface_vc_vtk(surface, solved)
    write_single_surface_maps("boozerQA_finitebeta_final_maps", solved)
    print(f"Saved final reference curves to {Path(OUT_DIR) / 'curves_opt.vtu'}")
    print(f"Saved final surface to {Path(OUT_DIR) / 'surf_opt.vts'}")
    print("Single-surface virtual-casing finite-beta solve complete.")

elif mode in ("prescribed", "prescribed-fixed", "fixed"):
    print("Using prescribed-field manufactured finite-beta test case")
    print(
        f"Mode={mode}, grid: nphi={nphi}, ntheta={ntheta}, "
        f"ls_max_nfev={ls_max_nfev}, qa_maxiter={qa_maxiter}"
    )
    print(f"Inner least-squares tables: {'on' if inner_verbose else 'off'}")

    base_curves, _, ma, nfp, bs = get_data("ncsx")
    all_curves = [coil.curve for coil in bs.coils]
    phis = np.linspace(0, 1 / nfp, nphi, endpoint=False)
    thetas = np.linspace(0, 1, ntheta, endpoint=False)
    surface = SurfaceXYZTensorFourier(
        mpol=3,
        ntor=3,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(ma, 0.1, flip_theta=True)

    iota_target = -0.31
    G_target = 1.4
    vol = Volume(surface)
    vol_target = vol.J()

    def B_in(surface):
        tang = surface.gammadash1() + iota_target * surface.gammadash2()
        tang_norm_sq = np.sum(tang**2, axis=2)
        return (G_target / tang_norm_sq)[:, :, None] * tang

    def B_out(surface):
        return B_in(surface)

    B_in0 = B_in(surface)
    B_out0 = B_out(surface)
    pressure_jump, plasma_beta, reference_magnetic_pressure = resolve_pressure_jump_input(surface, B_in0 + B_out0)

    print(
        f"Target interface data: iota_target={iota_target:.6e}, "
        f"G_target={G_target:.6e}, pressure_jump={pressure_jump:.6e}, plasma_beta={plasma_beta:.6%}, "
        f"reference <B^2/(2 mu0)>={reference_magnetic_pressure:.6e}"
    )

    boozer_surface = FiniteBetaBoozerSurface(
        None,
        surface,
        vol,
        vol_target,
        pressure_jump=pressure_jump,
        options={'verbose': inner_verbose, 'ls_max_nfev': ls_max_nfev},
    )

    initial = boozer_surface.residual_blocks(iota=-0.1, G=G_target, I=0.0, B_in=B_in0, B_out=B_out0)
    print_blocks("Initial residual norms:", initial)
    write_curves_vtk("curves_init", all_curves)
    write_surface_vtk("surf_init", surface)
    print(f"Saved initial reference curves to {Path(OUT_DIR) / 'curves_init.vtu'}")
    print(f"Saved initial surface to {Path(OUT_DIR) / 'surf_init.vts'}")

    if mode in ("prescribed-fixed", "fixed"):
        res = boozer_surface.run_code(iota=-0.1, G=G_target, I=0.0, B_in=B_in0, B_out=B_out0, optimize_G=False)
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in0,
            B_out=B_out0,
        )
        print_blocks("Solved residual norms:", solved)
        print_solver_summary("Initial fixed-surface solve", res)

        surface.x = surface.x + 1e-3 * np.random.default_rng(7).standard_normal(surface.x.shape)
        res = boozer_surface.run_code(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in0,
            B_out=B_out0,
            optimize_G=False,
            optimize_surface=True,
        )
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in0,
            B_out=B_out0,
        )
        print_blocks("Analytic fixed-field surface solve residual norms:", solved)
        print(
            f"Fixed-field surface solve: success={res['success']}, "
            f"lsq_nfev={res['iter']}, iota={res['iota']:.6e}, "
            f"mr={surface.major_radius():.6e}, ||r||={res['residual_norm']:.6e}"
        )
        write_curves_vtk("curves_opt", all_curves)
        write_surface_vtk("surf_opt", surface)
        write_residual_vtk(surface, solved)
        print(f"Saved final reference curves to {Path(OUT_DIR) / 'curves_opt.vtu'}")
        print(f"Saved final surface to {Path(OUT_DIR) / 'surf_opt.vts'}")
        print("Prescribed fixed-field finite-beta solve complete.")

    else:
        res = boozer_surface.run_code(iota=-0.1, G=G_target, I=0.0, B_in=B_in, B_out=B_out, optimize_G=False)
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in,
            B_out=B_out,
        )
        print_blocks("Solved residual norms:", solved)
        print_solver_summary("Initial prescribed-field solve", res)
        print("The inner least-squares solve is warm-started, so later outer iterations often converge in one function evaluation.")

        iota0 = res["iota"]
        mr0 = surface.major_radius()
        report_progress = make_progress_reporter()

        def fun(dofs):
            surface.x = dofs
            res_local = boozer_surface.run_code(
                iota=boozer_surface.res["iota"],
                G=G_target,
                I=boozer_surface.res["I"],
                current_potential=boozer_surface.res["current_potential"],
                B_in=B_in,
                B_out=B_out,
                optimize_G=False,
            )
            solved_local = boozer_surface.residual_blocks(
                iota=res_local["iota"],
                G=res_local["G"],
                I=res_local["I"],
                current_potential=res_local["current_potential"],
                B_in=B_in,
                B_out=B_out,
            )
            field = B_in(surface)
            J_nonqs = surface_field_nonquasisymmetric_ratio(surface, field)
            J_iota = 0.5 * (res_local["iota"] - iota0)**2
            J_mr = 0.5 * (surface.major_radius() - mr0)**2
            J = J_nonqs + J_iota + J_mr
            report_progress(J, J_nonqs, J_iota, J_mr, res_local, solved_local)
            return J

        print("Running prescribed-field finite-beta QA optimization")
        opt_result = minimize(fun, surface.x, method="L-BFGS-B", options={"maxiter": qa_maxiter})
        print(
            f"Outer QA optimization: success={opt_result.success}, status={opt_result.status}, "
            f"nit={opt_result.nit}, nfev={opt_result.nfev}, final_J={opt_result.fun:.6e}"
        )

        res = boozer_surface.run_code(
            iota=boozer_surface.res["iota"],
            G=G_target,
            I=boozer_surface.res["I"],
            current_potential=boozer_surface.res["current_potential"],
            B_in=B_in,
            B_out=B_out,
            optimize_G=False,
        )
        field = B_in(surface)
        solved = boozer_surface.residual_blocks(
            iota=res["iota"],
            G=res["G"],
            I=res["I"],
            current_potential=res["current_potential"],
            B_in=B_in,
            B_out=B_out,
        )
        print(
            f"Final QA ratio={surface_field_nonquasisymmetric_ratio(surface, field):.6e}, "
            f"iota={res['iota']:.6e}, mr={surface.major_radius():.6e}, "
            f"lsq_nfev={res['iter']}, ||r||={res['residual_norm']:.6e}"
        )
        write_curves_vtk("curves_opt", all_curves)
        write_surface_vtk("surf_opt", surface)
        write_residual_vtk(surface, solved)
        write_history_outputs(report_progress.state["history"], np.linalg.norm(initial["residual"]))
        print(f"Saved final reference curves to {Path(OUT_DIR) / 'curves_opt.vtu'}")
        print(f"Saved final surface to {Path(OUT_DIR) / 'surf_opt.vts'}")
        print("The history plot now includes individual Boozer, normal, pressure, jump, and sheet-current block norms.")
        print("This is the closest analog here to a DESC-style residual diagnostic, but it is still a single-surface interface residual, not a full volumetric force-balance residual.")
        print("Prescribed-field finite-beta QA solve complete.")

elif mode == "vmec":
    print("Using VMEC-backed self-consistent finite-beta case")
    from simsopt.mhd.virtual_casing import VirtualCasing
    from simsopt.mhd.vmec import Vmec

    test_dir = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
    default_vmec_input = test_dir / "input.W7-X_without_coil_ripple_beta0p05_d23p4_tm"
    vmec_filename = Path(os.environ.get("SIMSOPT_FINITE_BETA_VMEC", str(default_vmec_input))).resolve()
    if not vmec_filename.exists():
        raise FileNotFoundError(
            f"VMEC input or wout file not found: {vmec_filename}. Set SIMSOPT_FINITE_BETA_VMEC to a valid path."
        )

    print(f"VMEC file: {vmec_filename}")
    print(f"Grid: nphi={nphi}, ntheta={ntheta}, ls_max_nfev={ls_max_nfev}")

    vmec = Vmec(str(vmec_filename))
    vmec.run()

    vc_override = os.environ.get("SIMSOPT_FINITE_BETA_VCASING")
    if vc_override:
        vc_filename = Path(vc_override).resolve()
    else:
        vc_filename = vmec_filename.with_name(vmec_filename.name.replace("input.", "vcasing_") + ".nc")

    if vc_filename.exists():
        print(f"Loading saved virtual casing result from {vc_filename}")
        vc = VirtualCasing.load(str(vc_filename))
    else:
        print("Running virtual casing calculation")
        try:
            vc = VirtualCasing.from_vmec(
                vmec,
                src_nphi=nphi,
                src_ntheta=ntheta,
                trgt_nphi=nphi,
                trgt_ntheta=ntheta,
                filename=str(vc_filename),
            )
        except ModuleNotFoundError as err:
            if err.name == "virtual_casing":
                raise ModuleNotFoundError(
                    "The auxiliary VMEC-backed mode needs the optional virtual_casing package when no cached vcasing file is available. "
                    "Install virtual_casing or set SIMSOPT_FINITE_BETA_VCASING to an existing vcasing_*.nc file."
                ) from err
            raise

    vc_total_shape = np.asarray(vc.B_total).shape[:2]
    vc_external_shape = np.asarray(vc.B_external).shape[:2]
    if vc_total_shape != vc_external_shape:
        raise ValueError(
            "VMEC-backed mode requires virtual-casing total and external fields on the same grid. "
            f"Got B_total shape {vc_total_shape} and B_external shape {vc_external_shape}. "
            "Regenerate the vcasing file with matching src/trgt resolutions."
        )
    field_nphi, field_ntheta = vc_total_shape
    if (field_nphi, field_ntheta) != (nphi, ntheta):
        print(
            f"Using cached virtual-casing field grid nphi={field_nphi}, ntheta={field_ntheta} "
            f"instead of requested nphi={nphi}, ntheta={ntheta}"
        )

    boundary_rz = SurfaceRZFourier.from_nphi_ntheta(
        mpol=vmec.wout.mpol,
        ntor=vmec.wout.ntor,
        nfp=vmec.wout.nfp,
        stellsym=not bool(vmec.wout.lasym),
        nphi=field_nphi,
        ntheta=field_ntheta,
        range="half period",
    )
    boundary_rz.x = vmec.boundary.x

    surface = SurfaceXYZTensorFourier(
        mpol=vmec.wout.mpol,
        ntor=vmec.wout.ntor,
        nfp=vmec.wout.nfp,
        stellsym=not bool(vmec.wout.lasym),
        quadpoints_phi=boundary_rz.quadpoints_phi,
        quadpoints_theta=boundary_rz.quadpoints_theta,
    )
    surface.least_squares_fit(boundary_rz.gamma())
    vol = Volume(surface)
    vol_target = vol.J()

    boozer_surface = FiniteBetaBoozerSurface(
        None,
        surface,
        vol,
        vol_target,
        pressure_jump=0.0,
        options={'verbose': inner_verbose, 'ls_max_nfev': ls_max_nfev},
    )

    B_in = vc.B_total - vc.B_external
    B_out = vc.B_external
    B_total = B_in + B_out
    requested_pressure_jump_raw = os.environ.get("SIMSOPT_FINITE_BETA_PRESSURE_JUMP")
    requested_plasma_beta = _parse_optional_plasma_beta()
    if requested_pressure_jump_raw is None and requested_plasma_beta is None:
        reference_magnetic_pressure = mean_surface_magnetic_pressure(surface, B_total)
        pressure_jump = mean_surface_pressure_jump(surface, B_in, B_out)
        plasma_beta = pressure_jump / max(reference_magnetic_pressure, 1e-30)
        print(
            "Target finite-beta input inferred from VMEC interface fields: "
            f"pressure_jump={pressure_jump:.6e}, plasma_beta={plasma_beta:.6%}, "
            f"reference <B^2/(2 mu0)>={reference_magnetic_pressure:.6e}"
        )
    else:
        pressure_jump, plasma_beta, reference_magnetic_pressure = resolve_pressure_jump_input(surface, B_total)
        print(
            f"Target finite-beta input: pressure_jump={pressure_jump:.6e}, plasma_beta={plasma_beta:.6%}, "
            f"reference <B^2/(2 mu0)>={reference_magnetic_pressure:.6e}"
        )
    boozer_surface.pressure_jump = pressure_jump
    initial = boozer_surface.residual_blocks(iota=vmec.iota_edge(), G=0.0, I=0.0, B_in=B_in, B_out=B_out)
    print_blocks("Initial residual norms:", initial)
    write_surface_vtk("surf_init", surface)
    print(f"Saved initial surface to {Path(OUT_DIR) / 'surf_init.vts'}")
    print(f"Initial self-consistent nonQS ratio={surface_field_nonquasisymmetric_ratio(surface, B_total):.6e}")

    res = boozer_surface.run_code(iota=vmec.iota_edge(), G=0.0, I=0.0, B_in=B_in, B_out=B_out, optimize_G=False)
    solved = boozer_surface.residual_blocks(
        iota=res["iota"],
        G=res["G"],
        I=res["I"],
        current_potential=res["current_potential"],
        B_in=B_in,
        B_out=B_out,
    )
    print_blocks("Solved residual norms:", solved)
    print_solver_summary("VMEC-backed solve", res)
    print(f"Final self-consistent nonQS ratio={surface_field_nonquasisymmetric_ratio(surface, B_total):.6e}")
    write_surface_vtk("surf_opt", surface)
    write_residual_vtk(surface, solved)
    print(f"Saved final surface to {Path(OUT_DIR) / 'surf_opt.vts'}")
    if res["success"]:
        print("VMEC-backed self-consistent finite-beta solve converged.")
    else:
        print("VMEC-backed self-consistent finite-beta run finished without convergence; increase SIMSOPT_FINITE_BETA_MAX_NFEV for a tighter solve.")

else:
    raise ValueError(
        "SIMSOPT_FINITE_BETA_MODE must be one of 'self-consistent', 'single-surface-vc', 'vc-validation', 'prescribed', "
        "'prescribed-fixed', or 'vmec'."
    )

print("End of 2_Intermediate/boozerQA_finitebeta.py")
print("=============================================")