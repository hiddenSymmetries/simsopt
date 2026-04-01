# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""JAX-native least-squares solvers and fixed-resolution VMEC-JAX helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import time
from types import SimpleNamespace
from typing import Callable, Dict, Sequence

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - optional dependency
    jax = None
    jnp = None
    _import_error = exc
else:
    _import_error = None

try:
    import jaxopt  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    jaxopt = None

try:
    from scipy.optimize import least_squares as _scipy_least_squares  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _scipy_least_squares = None

__all__ = [
    "least_squares_jax_solve",
    "build_vmec_objective_stage",
    "run_continuation_jax_solve",
]


@dataclass
class VmecObjectiveStage:
    x0: np.ndarray
    x_scale: np.ndarray
    residuals: Callable[[jnp.ndarray], jnp.ndarray]
    free_names: list[str]
    extras: dict


def _require_jax():
    """Raise a helpful error when the optional JAX dependency is unavailable."""
    if jax is None or jnp is None:
        raise ImportError(
            "least_squares_jax_solve requires JAX. "
            "Install jax/jaxlib before using this solver."
        ) from _import_error


def _normalize_scale(x0, x_scale):
    """Return a nonzero parameter scaling vector in solver coordinates."""
    if x_scale is None:
        return jnp.ones_like(x0)
    scale = jnp.asarray(x_scale, dtype=x0.dtype)
    scale = jnp.where(scale == 0, jnp.ones_like(scale), scale)
    return scale


def _block_until_ready(value):
    try:
        return jax.block_until_ready(value)
    except Exception:
        return value


def _sync_numpy(value):
    value = _block_until_ready(value)
    return np.asarray(value)


def _safe_float(value) -> float:
    arr = _sync_numpy(value)
    if arr.shape == ():
        return float(arr)
    return float(np.asarray(arr).reshape(-1)[0])


def _make_profile(enabled: bool):
    if not enabled:
        return None
    return {
        "residual_calls": 0,
        "residual_wall_s": 0.0,
        "jacobian_calls": 0,
        "jacobian_wall_s": 0.0,
        "objective_calls": 0,
        "objective_wall_s": 0.0,
        "value_and_grad_calls": 0,
        "value_and_grad_wall_s": 0.0,
        "normal_matvec_calls": 0,
        "normal_matvec_wall_s": 0.0,
        "line_search_calls": 0,
        "line_search_wall_s": 0.0,
        "fallback_trial_calls": 0,
        "backtrack_trial_calls": 0,
        "cauchy_step_calls": 0,
    }


def _profiled_call(profile, count_key: str, wall_key: str, fn, *args):
    if profile is None:
        return fn(*args)
    start = time.perf_counter()
    result = fn(*args)
    _block_until_ready(result)
    profile[count_key] += 1
    profile[wall_key] += time.perf_counter() - start
    return result


def _deadline_from_limit(wall_clock_limit_s):
    if wall_clock_limit_s is None:
        return None
    wall_clock_limit_s = float(wall_clock_limit_s)
    if wall_clock_limit_s <= 0.0:
        return None
    return time.perf_counter() + wall_clock_limit_s


def _deadline_exhausted(deadline) -> bool:
    return deadline is not None and time.perf_counter() >= deadline


def _remaining_wall_clock(deadline):
    if deadline is None:
        return None
    return max(0.0, deadline - time.perf_counter())


def _mode_weighted_x_scale(dof_names: Sequence[str], alpha=1.2, min_scale=1e-9):
    x_scale = np.ones(len(dof_names), dtype=float)
    pattern = re.compile(r"[rz][cs]\((\d+),(-?\d+)\)")
    for index, name in enumerate(dof_names):
        match = pattern.search(str(name))
        if match is None:
            continue
        mode_level = max(abs(int(match.group(1))), abs(int(match.group(2))))
        scale = math.exp(-alpha * mode_level) / math.exp(-alpha)
        x_scale[index] = max(scale, min_scale)
    return x_scale


def _lift_free_params(previous_names, previous_x, next_names, next_x0):
    previous_map = {str(name): float(value) for name, value in zip(previous_names, np.asarray(previous_x, dtype=float))}
    lifted = np.asarray(next_x0, dtype=float).copy()
    for index, name in enumerate(next_names):
        if str(name) in previous_map:
            lifted[index] = previous_map[str(name)]
    return lifted


def _directional_fd_epsilon(y, direction):
    dtype = getattr(y, "dtype", jnp.float64)
    eps = jnp.sqrt(jnp.asarray(np.finfo(np.float64).eps, dtype=dtype))
    scale = jnp.maximum(1.0, jnp.linalg.norm(y))
    direction_norm = jnp.maximum(1.0, jnp.linalg.norm(direction))
    return eps * scale / direction_norm


def _build_gradient_preconditioner(gradient, x_scale, spectral_scale=1.0):
    grad = jnp.asarray(gradient)
    scale = jnp.asarray(x_scale, dtype=grad.dtype)
    scale = jnp.where(scale == 0.0, 1.0, scale)
    diag = 1.0 / jnp.maximum(jnp.abs(grad) / scale + 1e-6, 1e-6)
    return jnp.asarray(float(spectral_scale), dtype=grad.dtype) * diag


def _boundary_intersection_norm(step, direction, trust_radius):
    step_norm = float(jnp.linalg.norm(step))
    if step_norm >= trust_radius:
        return step, True
    direction_norm = float(jnp.linalg.norm(direction))
    if direction_norm == 0.0:
        return step, False
    remaining_sq = max(0.0, trust_radius * trust_radius - step_norm * step_norm)
    dot_sd = float(jnp.dot(step, direction))
    a = direction_norm * direction_norm
    b = 2.0 * dot_sd
    c = step_norm * step_norm - trust_radius * trust_radius
    disc = max(0.0, b * b - 4.0 * a * c)
    tau = (-b + math.sqrt(disc)) / (2.0 * a)
    tau = max(0.0, tau)
    return step + tau * direction, True


def _scaled_cauchy_step(gradient, normal_matvec, trust_radius, *, h_direction=None, profile=None):
    if profile is not None:
        profile["cauchy_step_calls"] += 1
    grad = jnp.asarray(gradient)
    grad_norm = float(jnp.linalg.norm(grad))
    if grad_norm == 0.0:
        return {
            "step": jnp.zeros_like(grad),
            "predicted_reduction": 0.0,
            "hit_boundary": False,
            "h_direction": jnp.zeros_like(grad),
        }
    direction = -grad
    h_direction = normal_matvec(direction) if h_direction is None else h_direction
    curvature = float(jnp.dot(direction, h_direction))
    if curvature <= 0.0:
        alpha = trust_radius / grad_norm
    else:
        alpha = min((grad_norm * grad_norm) / curvature, trust_radius / grad_norm)
    step = alpha * direction
    predicted_reduction = max(0.0, alpha * grad_norm * grad_norm - 0.5 * alpha * alpha * curvature)
    hit_boundary = abs(float(jnp.linalg.norm(step)) - trust_radius) <= 1e-12 or alpha == trust_radius / grad_norm
    return {
        "step": step,
        "predicted_reduction": predicted_reduction,
        "hit_boundary": hit_boundary,
        "h_direction": h_direction,
    }


def _steihaug_trust_region_cg(gradient, normal_matvec, trust_radius, *, preconditioner=None, max_iter=20, tol=1e-4):
    grad = jnp.asarray(gradient)
    step = jnp.zeros_like(grad)
    residual = -grad
    if preconditioner is None:
        z = residual
    else:
        z = residual * jnp.asarray(preconditioner)
    direction = z
    rz_old = float(jnp.dot(residual, z))
    if rz_old <= 0.0:
        return {
            "step": step,
            "predicted_reduction": 0.0,
            "hit_boundary": False,
            "iterations": 0,
            "first_h_direction": jnp.zeros_like(grad),
        }

    first_h_direction = None
    for iteration in range(int(max_iter)):
        h_direction = normal_matvec(direction)
        if first_h_direction is None:
            first_h_direction = h_direction
        curvature = float(jnp.dot(direction, h_direction))
        if curvature <= 0.0:
            boundary_step, _ = _boundary_intersection_norm(step, direction, trust_radius)
            predicted = float(jnp.dot(-grad, boundary_step) - 0.5 * jnp.dot(boundary_step, normal_matvec(boundary_step)))
            return {
                "step": boundary_step,
                "predicted_reduction": max(0.0, predicted),
                "hit_boundary": True,
                "iterations": iteration + 1,
                "first_h_direction": first_h_direction,
            }

        alpha = rz_old / curvature
        next_step = step + alpha * direction
        if float(jnp.linalg.norm(next_step)) >= trust_radius:
            boundary_step, _ = _boundary_intersection_norm(step, direction, trust_radius)
            predicted = float(jnp.dot(-grad, boundary_step) - 0.5 * jnp.dot(boundary_step, normal_matvec(boundary_step)))
            return {
                "step": boundary_step,
                "predicted_reduction": max(0.0, predicted),
                "hit_boundary": True,
                "iterations": iteration + 1,
                "first_h_direction": first_h_direction,
            }

        step = next_step
        residual = residual - alpha * h_direction
        if float(jnp.linalg.norm(residual)) <= tol * max(1.0, float(jnp.linalg.norm(grad))):
            predicted = float(jnp.dot(-grad, step) - 0.5 * jnp.dot(step, normal_matvec(step)))
            return {
                "step": step,
                "predicted_reduction": max(0.0, predicted),
                "hit_boundary": False,
                "iterations": iteration + 1,
                "first_h_direction": first_h_direction,
            }

        if preconditioner is None:
            z = residual
        else:
            z = residual * jnp.asarray(preconditioner)
        rz_new = float(jnp.dot(residual, z))
        if rz_old == 0.0:
            break
        beta = rz_new / rz_old
        direction = z + beta * direction
        rz_old = rz_new

    predicted = float(jnp.dot(-grad, step) - 0.5 * jnp.dot(step, normal_matvec(step)))
    if first_h_direction is None:
        first_h_direction = jnp.zeros_like(grad)
    return {
        "step": step,
        "predicted_reduction": max(0.0, predicted),
        "hit_boundary": False,
        "iterations": int(max_iter),
        "first_h_direction": first_h_direction,
    }


def build_vmec_objective_stage(
    vmec,
    *,
    max_mode: int,
    objective_tuples,
    surfaces,
    helicity_m: int,
    helicity_n: int,
    x_scale_alpha: float = 1.2,
    x_scale_min: float = 1e-9,
):
    from simsopt.mhd import QuasisymmetryRatioResidualJax

    vmec.indata.mpol = int(max_mode) + 2
    vmec.indata.ntor = vmec.indata.mpol

    surf = vmec.boundary
    surf.fix_all()
    surf.fixed_range(mmin=0, mmax=int(max_mode), nmin=-int(max_mode), nmax=int(max_mode), fixed=False)
    surf.fix("rc(0,0)")

    free_names = [name for name, free in zip(surf.dof_names, surf.free) if free]
    x0 = np.asarray(surf.get_free_params(), dtype=float)
    x_scale = _mode_weighted_x_scale(free_names, alpha=x_scale_alpha, min_scale=x_scale_min)

    qs = None
    if any(str(name).strip().lower() == "qs" for name, _, _ in objective_tuples):
        qs = QuasisymmetryRatioResidualJax(
            vmec,
            surfaces,
            helicity_m=int(helicity_m),
            helicity_n=int(helicity_n),
        )

    def residuals(x_free):
        x_free = jnp.asarray(x_free, dtype=jnp.float64)
        state = vmec._solve_state(x_free)
        blocks = []
        for name, target, weight in objective_tuples:
            key = str(name).strip().lower()
            weight_value = float(weight)
            if key == "aspect":
                value = vmec.aspect_equilibrium_from_state_jax(state)
                blocks.append(jnp.asarray([weight_value * (value - float(target))], dtype=jnp.float64))
            elif key in ("mean_iota", "iota"):
                value = vmec.mean_iota_from_state_jax(state)
                blocks.append(jnp.asarray([weight_value * (value - float(target))], dtype=jnp.float64))
            elif key == "qs":
                if abs(float(target)) > 0.0:
                    raise ValueError("The VMEC-JAX QS tuple currently expects target=0.0.")
                blocks.append(weight_value * jnp.asarray(qs.residuals_from_state(state), dtype=jnp.float64))
            else:
                raise ValueError(f"Unsupported VMEC-JAX objective tuple name: {name}")
        if not blocks:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.concatenate(blocks)

    return VmecObjectiveStage(
        x0=x0,
        x_scale=np.asarray(x_scale, dtype=float),
        residuals=residuals,
        free_names=list(free_names),
        extras={
            "surf": surf,
            "qs": qs,
            "objective_tuples": list(objective_tuples),
            "surfaces": np.asarray(surfaces, dtype=float),
            "max_mode": int(max_mode),
        },
    )


def least_squares_jax_solve(
    residual_fun: Callable[[jnp.ndarray], jnp.ndarray],
    x0,
    *,
    method: str = "scipy",
    max_nfev: int = 50,
    gtol: float = 1e-7,
    step_size: float = 1e-2,
    x_scale=None,
    jit: bool = True,
    verbose: int = 1,
    jac: str | Callable | None = None,
    profile: bool = False,
    wall_clock_limit_s: float | None = None,
) -> Dict[str, object]:
    """Solve a nonlinear least-squares problem with JAX derivatives."""
    _require_jax()

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    scale = _normalize_scale(x0, x_scale)
    y0 = x0 / scale
    method = str(method).strip().lower()
    profile_data = _make_profile(profile)
    deadline = _deadline_from_limit(wall_clock_limit_s)

    def residuals_y_raw(y):
        return residual_fun(y * scale)

    def objective_y_raw(y):
        r = residuals_y_raw(y)
        return 0.5 * jnp.sum(r * r)

    def residuals_y(y):
        return _profiled_call(profile_data, "residual_calls", "residual_wall_s", residuals_y_raw, y)

    def objective_y(y):
        return _profiled_call(profile_data, "objective_calls", "objective_wall_s", objective_y_raw, y)

    def residual_cost_eval(y):
        residual = residuals_y(y)
        return 0.5 * jnp.sum(residual * residual), residual

    jac_residual_y = jax.jacfwd(residuals_y_raw)
    if jit:
        jac_residual_y = jax.jit(jac_residual_y)

    def jacobian_y(y):
        return _profiled_call(profile_data, "jacobian_calls", "jacobian_wall_s", jac_residual_y, y)

    value_and_grad_y = jax.value_and_grad(objective_y_raw)
    if jit:
        value_and_grad_y = jax.jit(value_and_grad_y)

    def profiled_value_and_grad(y):
        return _profiled_call(profile_data, "value_and_grad_calls", "value_and_grad_wall_s", value_and_grad_y, y)

    def accept_by_backtracking(y, cost, grad, direction, *, initial_step=1.0, c1=1e-4, max_trials=12):
        step_norm_sq = float(jnp.dot(grad, direction))
        best_y = y
        best_cost = cost
        best_step_label = "none"
        for trial in range(int(max_trials)):
            if _deadline_exhausted(deadline):
                break
            alpha = float(initial_step) * (0.5 ** trial)
            trial_y = y + alpha * direction
            start = time.perf_counter()
            trial_cost = objective_y(trial_y)
            if profile_data is not None:
                profile_data["line_search_calls"] += 1
                profile_data["line_search_wall_s"] += time.perf_counter() - start
            if float(trial_cost) < float(best_cost):
                best_y = trial_y
                best_cost = trial_cost
                best_step_label = f"bt_{trial}" if trial else "full"
            if float(trial_cost) <= float(cost) + c1 * alpha * step_norm_sq:
                return best_y, best_cost, best_step_label, True
        return best_y, best_cost, best_step_label, float(best_cost) < float(cost)

    status = 0
    success = False
    wall_clock_exhausted = False
    nfev = 0
    grad = jnp.zeros_like(y0)
    y = y0
    cost = objective_y(y)
    step_description = None

    if method in ("scipy", "least_squares"):
        if _scipy_least_squares is None:
            if jaxopt is not None:
                method = "lbfgs"
            else:
                raise ImportError("scipy is required for method='scipy'.")
        else:
            jac_scipy = "jax" if jac is None else jac
            def residuals_numpy(y_np):
                return np.asarray(residuals_y_raw(jnp.asarray(y_np)))

            def jac_numpy(y_np):
                J = jac_residual_y(jnp.asarray(y_np))
                return np.asarray(J)

            verbose_scipy = 2 if verbose else 0
            res = _scipy_least_squares(
                residuals_numpy,
                np.asarray(y0),
                jac=jac_numpy if jac_scipy == "jax" else jac_scipy,
                max_nfev=int(max_nfev),
                xtol=float(gtol),
                ftol=float(gtol),
                gtol=float(gtol),
                verbose=verbose_scipy,
            )
            y = jnp.asarray(res.x, dtype=y0.dtype)
            if res.jac is not None:
                grad = jnp.asarray(res.jac.T @ res.fun, dtype=y0.dtype)
            cost = jnp.asarray(0.5 * np.sum(res.fun * res.fun), dtype=y0.dtype)
            nfev = int(res.nfev)
            status = 1 if bool(res.success) else 0
            success = status == 1
    elif method == "lbfgs":
        if jaxopt is None:
            method = "gradient_descent"
        else:
            solver = jaxopt.LBFGS(fun=objective_y_raw, maxiter=int(max_nfev), tol=float(gtol))
            result = solver.run(y0)
            y = result.params
            grad = result.state.grad
            cost = objective_y(y)
            nfev = int(result.state.iter_num)
            status = 1 if float(result.state.error) <= float(gtol) else 0
            success = status == 1

    elif method == "gradient_descent":
        y = y0
        cost = objective_y(y)
        for iteration in range(int(max_nfev)):
            if _deadline_exhausted(deadline):
                wall_clock_exhausted = True
                status = 2
                break
            cost, grad = profiled_value_and_grad(y)
            nfev = iteration + 1
            grad_norm = float(jnp.linalg.norm(grad))
            if verbose:
                print(f"iter {iteration:3d}  cost={float(cost):.6e}  grad_norm={grad_norm:.3e}")
            if grad_norm < float(gtol):
                status = 1
                success = True
                break
            direction = -grad
            y_trial, cost_trial, step_description, accepted = accept_by_backtracking(
                y,
                cost,
                grad,
                direction,
                initial_step=float(step_size),
            )
            if not accepted:
                break
            y = y_trial
            cost = cost_trial
    elif method == "gauss_newton":
        y = y0
        cost = objective_y(y)
        for iteration in range(int(max_nfev)):
            if _deadline_exhausted(deadline):
                wall_clock_exhausted = True
                status = 2
                break
            residual = residuals_y(y)
            J = jacobian_y(y)
            J_np = np.asarray(J)
            residual_np = np.asarray(residual)
            grad = jnp.asarray(J_np.T @ residual_np, dtype=y.dtype)
            nfev = iteration + 1
            grad_norm = float(jnp.linalg.norm(grad))
            if verbose:
                print(f"iter {iteration:3d}  cost={float(0.5 * np.dot(residual_np, residual_np)):.6e}  grad_norm={grad_norm:.3e}")
            if grad_norm < float(gtol):
                status = 1
                success = True
                cost = jnp.asarray(0.5 * np.dot(residual_np, residual_np), dtype=y.dtype)
                break
            step_np, *_ = np.linalg.lstsq(J_np, -residual_np, rcond=None)
            direction = jnp.asarray(step_np, dtype=y.dtype)
            cost_current = jnp.asarray(0.5 * np.dot(residual_np, residual_np), dtype=y.dtype)
            y_trial, cost_trial, step_description, accepted = accept_by_backtracking(y, cost_current, grad, direction)
            if not accepted:
                break
            y = y_trial
            cost = cost_trial
    elif method == "trust_region":
        y = y0
        cost = objective_y(y)
        trust_radius = max(float(step_size), 1.0)
        for iteration in range(int(max_nfev)):
            if _deadline_exhausted(deadline):
                wall_clock_exhausted = True
                status = 2
                break
            residual = residuals_y(y)
            J = jacobian_y(y)
            J_np = np.asarray(J)
            residual_np = np.asarray(residual)
            grad_np = J_np.T @ residual_np
            H_np = J_np.T @ J_np
            grad = jnp.asarray(grad_np, dtype=y.dtype)
            nfev = iteration + 1
            grad_norm = float(np.linalg.norm(grad_np))
            if grad_norm < float(gtol):
                status = 1
                success = True
                cost = jnp.asarray(0.5 * np.dot(residual_np, residual_np), dtype=y.dtype)
                break
            try:
                step_np = np.linalg.solve(H_np + 1e-10 * np.eye(H_np.shape[0]), -grad_np)
            except np.linalg.LinAlgError:
                step_np = -grad_np
            step_norm = float(np.linalg.norm(step_np))
            if step_norm > trust_radius and step_norm > 0.0:
                step_np = step_np * (trust_radius / step_norm)
            trial_y = y + jnp.asarray(step_np, dtype=y.dtype)
            trial_cost = objective_y(trial_y)
            predicted = max(1e-16, float(-grad_np @ step_np - 0.5 * step_np @ (H_np @ step_np)))
            actual = float(0.5 * np.dot(residual_np, residual_np) - float(trial_cost))
            ratio = actual / predicted
            if verbose:
                print(f"iter {iteration:3d}  cost={float(0.5 * np.dot(residual_np, residual_np)):.6e}  grad_norm={grad_norm:.3e}  tr_ratio={ratio:.3e}")
            if ratio > 1e-4 and float(trial_cost) < float(0.5 * np.dot(residual_np, residual_np)):
                y = trial_y
                cost = trial_cost
                step_description = "trust_region"
                if ratio > 0.75:
                    trust_radius *= 2.0
            else:
                trust_radius *= 0.5
                if trust_radius < 1e-12:
                    break
    elif method == "levenberg_marquardt":
        y = y0
        cost = objective_y(y)
        damping = max(float(step_size), 1e-6)
        for iteration in range(int(max_nfev)):
            if _deadline_exhausted(deadline):
                wall_clock_exhausted = True
                status = 2
                break
            residual = residuals_y(y)
            J = jacobian_y(y)
            J_np = np.asarray(J)
            residual_np = np.asarray(residual)
            grad_np = J_np.T @ residual_np
            H_np = J_np.T @ J_np
            grad = jnp.asarray(grad_np, dtype=y.dtype)
            nfev = iteration + 1
            grad_norm = float(np.linalg.norm(grad_np))
            cost_current = float(0.5 * np.dot(residual_np, residual_np))
            if verbose:
                print(f"iter {iteration:3d}  cost={cost_current:.6e}  grad_norm={grad_norm:.3e}  damping={damping:.3e}")
            if grad_norm < float(gtol):
                status = 1
                success = True
                cost = jnp.asarray(cost_current, dtype=y.dtype)
                break
            accepted = False
            for _ in range(8):
                if _deadline_exhausted(deadline):
                    wall_clock_exhausted = True
                    status = 2
                    break
                try:
                    step_np = np.linalg.solve(H_np + damping * np.eye(H_np.shape[0]), -grad_np)
                except np.linalg.LinAlgError:
                    step_np = -grad_np / max(damping, 1.0)
                trial_y = y + jnp.asarray(step_np, dtype=y.dtype)
                trial_cost = objective_y(trial_y)
                if float(trial_cost) < cost_current:
                    y = trial_y
                    cost = trial_cost
                    damping = max(damping * 0.3, 1e-8)
                    step_description = "lm"
                    accepted = True
                    break
                damping = min(damping * 10.0, 1e8)
            if status == 2 or not accepted:
                break
    elif method == "truncated_gauss_newton":
        y = y0
        cost = objective_y(y)
        trust_radius = max(float(step_size), 1.0)
        damping = 1e-6
        previous_y = None
        previous_grad = None
        spectral_scale = 1.0
        for iteration in range(int(max_nfev)):
            if _deadline_exhausted(deadline):
                wall_clock_exhausted = True
                status = 2
                break
            residual = residuals_y(y)
            _, vjp_fun = jax.vjp(residuals_y_raw, y)
            grad = vjp_fun(residual)[0]
            grad_norm = float(jnp.linalg.norm(grad))
            nfev = iteration + 1
            if previous_y is not None and previous_grad is not None:
                s = y - previous_y
                z = grad - previous_grad
                denom = float(jnp.dot(s, z))
                if denom > 0.0:
                    spectral_scale = float(np.clip(float(jnp.dot(s, s)) / denom, 1e-3, 1e3))
            M = _build_gradient_preconditioner(grad, scale, spectral_scale=spectral_scale)

            def normal_matvec(direction):
                direction = jnp.asarray(direction)
                epsilon = _directional_fd_epsilon(y, direction)
                start = time.perf_counter()
                residual_plus = residuals_y(y + epsilon * direction)
                residual_minus = residuals_y(y - epsilon * direction)
                jv = (residual_plus - residual_minus) / (2.0 * epsilon)
                value = vjp_fun(jv)[0] + damping * direction
                if profile_data is not None:
                    profile_data["normal_matvec_calls"] += 1
                    profile_data["normal_matvec_wall_s"] += time.perf_counter() - start
                return value

            if verbose:
                print(
                    f"iter {iteration:3d}  cost={float(0.5 * jnp.sum(residual * residual)):.6e}  "
                    f"grad_norm={grad_norm:.3e}  trust_radius={trust_radius:.3e}  spectral_M={spectral_scale:.3e}"
                )
            if grad_norm < float(gtol):
                status = 1
                success = True
                cost = 0.5 * jnp.sum(residual * residual)
                break

            cg_info = _steihaug_trust_region_cg(
                grad,
                normal_matvec,
                trust_radius,
                preconditioner=M,
                max_iter=min(20, max(5, int(grad.shape[0]))),
                tol=1e-3,
            )
            cauchy_info = _scaled_cauchy_step(
                grad,
                normal_matvec,
                trust_radius,
                h_direction=cg_info.get("first_h_direction"),
                profile=profile_data,
            )

            choose_cauchy = (
                cauchy_info["predicted_reduction"] > 0.0
                and cg_info["hit_boundary"]
                and cauchy_info["predicted_reduction"] >= 0.9 * max(cg_info["predicted_reduction"], 1e-16)
            )
            trial_info = cauchy_info if choose_cauchy else cg_info
            step_label = "cauchy" if choose_cauchy else "cg"
            trial_step = trial_info["step"]
            trial_predicted = max(float(trial_info["predicted_reduction"]), 1e-16)
            base_cost = float(0.5 * jnp.sum(residual * residual))

            trial_y = y + trial_step
            if _deadline_exhausted(deadline):
                wall_clock_exhausted = True
                status = 2
                break
            trial_cost, _ = residual_cost_eval(trial_y)
            actual = base_cost - float(trial_cost)
            ratio = actual / trial_predicted

            if float(trial_cost) >= base_cost:
                scales = (0.125, 0.25, 0.5) if bool(trial_info.get("hit_boundary")) else (0.5, 0.25, 0.125)
                accepted = False
                for scale_factor in scales:
                    if _deadline_exhausted(deadline):
                        wall_clock_exhausted = True
                        status = 2
                        break
                    if profile_data is not None:
                        profile_data["backtrack_trial_calls"] += 1
                    scaled_step = float(scale_factor) * trial_step
                    scaled_cost, _ = residual_cost_eval(y + scaled_step)
                    if float(scaled_cost) < base_cost:
                        trial_step = scaled_step
                        trial_cost = scaled_cost
                        actual = base_cost - float(trial_cost)
                        ratio = actual / trial_predicted
                        step_label += "_bt"
                        accepted = True
                        break
                if not accepted and status != 2:
                    trust_radius *= 0.5
                    if choose_cauchy:
                        step_description = "cauchy_rejected"
                    else:
                        step_description = "cg_rejected"
                    continue

            if ratio > 1e-4 and float(trial_cost) < base_cost:
                previous_y = y
                previous_grad = grad
                y = y + trial_step
                cost = trial_cost
                step_description = step_label
                if ratio > 0.75:
                    trust_radius = max(trust_radius, 2.0 * float(jnp.linalg.norm(trial_step)))
                elif ratio < 0.25:
                    trust_radius *= 0.5
            else:
                trust_radius *= 0.5
                step_description = f"{step_label}_rejected"
                if trust_radius < 1e-12:
                    break
    else:
        raise ValueError(
            "method must be one of 'scipy', 'lbfgs', 'gradient_descent', 'gauss_newton', "
            "'trust_region', 'levenberg_marquardt', or 'truncated_gauss_newton'"
        )

    if status == 0 and _deadline_exhausted(deadline):
        status = 2
        wall_clock_exhausted = True

    x_opt = np.asarray(y * scale)
    return {
        "x": x_opt,
        "status": int(status),
        "success": bool(success),
        "nfev": int(nfev),
        "cost": float(cost),
        "grad": np.asarray(grad),
        "profile": profile_data,
        "wall_clock_exhausted": bool(wall_clock_exhausted),
        "step": step_description,
    }


def run_continuation_jax_solve(
    stage_builder: Callable[[int], VmecObjectiveStage],
    continuation_modes,
    *,
    method: str = "gradient_descent",
    max_nfev: int = 50,
    gtol: float = 1e-7,
    step_size: float = 1e-2,
    jit: bool = True,
    verbose: int = 1,
    jac: str | Callable | None = None,
    profile: bool = False,
    wall_clock_limit_s: float | None = None,
):
    continuation_modes = [int(mode) for mode in continuation_modes]
    if not continuation_modes:
        raise ValueError("continuation_modes must contain at least one stage")

    deadline = _deadline_from_limit(wall_clock_limit_s)
    stage_results = []
    current_x = None
    current_names = None
    last_result = None
    last_mode = None

    for mode in continuation_modes:
        if _deadline_exhausted(deadline):
            break
        stage = stage_builder(mode)
        x0 = np.asarray(stage.x0, dtype=float)
        if current_x is not None and current_names is not None:
            x0 = _lift_free_params(current_names, current_x, stage.free_names, x0)
            stage.extras["surf"].set_free_params(x0)
        remaining = _remaining_wall_clock(deadline)
        result = least_squares_jax_solve(
            stage.residuals,
            x0,
            method=method,
            max_nfev=max_nfev,
            gtol=gtol,
            step_size=step_size,
            x_scale=stage.x_scale,
            jit=jit,
            verbose=verbose,
            jac=jac,
            profile=profile,
            wall_clock_limit_s=remaining,
        )
        stage_results.append({"mode": mode, "result": result})
        current_x = np.asarray(result["x"], dtype=float)
        current_names = list(stage.free_names)
        last_result = result
        last_mode = mode
        if result.get("status") == 2:
            break

    final_mode = continuation_modes[-1]
    if current_x is None:
        final_stage = stage_builder(final_mode)
        current_x = np.asarray(final_stage.x0, dtype=float)
        current_names = list(final_stage.free_names)
        last_result = {
            "x": current_x,
            "status": 2 if _deadline_exhausted(deadline) else 0,
            "success": False,
            "nfev": 0,
            "cost": float(np.sum(np.asarray(final_stage.residuals(current_x)) ** 2) / 2.0),
            "grad": np.zeros_like(current_x),
            "profile": None,
            "wall_clock_exhausted": bool(_deadline_exhausted(deadline)),
            "step": None,
        }
    elif last_mode != final_mode:
        final_stage = stage_builder(final_mode)
        current_x = _lift_free_params(current_names, current_x, final_stage.free_names, final_stage.x0)
        current_names = list(final_stage.free_names)
        last_result = dict(last_result)
        last_result["x"] = np.asarray(current_x, dtype=float)
        last_result["wall_clock_exhausted"] = True
        if last_result.get("status", 0) == 0:
            last_result["status"] = 2

    return {
        "stage_results": stage_results,
        "result": last_result,
    }
