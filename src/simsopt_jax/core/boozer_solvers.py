"""JAX Boozer surface solvers and coil adjoints.

Solvers use float64 on the problem's device and flag singular steps.
Coil adjoints include label and current-derived G dependence.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import lu_factor, lu_solve

from simsopt_jax.backend.jit import jit
from simsopt_jax.pytree import pytree_dataclass

from ._math_utils import as_jax_float64
from .boozer_problem import (
    BoozerProblem,
    _G_from_coil_currents,
    boozer_exact_residual,
    boozer_penalty_constraints,
    boozer_penalty_residual,
    boozer_surface_residual,
)
from .quasi_newton import QuasiNewtonResult, _select, minimize_bfgs, minimize_lbfgsb
from .specs import GroupedCoilSetSpec

# Native refines the penalty Newton step with a second solve below this gradient norm.
_REFINE_BELOW = 1e-9
# Native's norm before the first BoozerExact iteration.
_EXACT_INITIAL_NORM = 1e6


@pytree_dataclass(data=("x", "residual", "jacobian", "iterations", "norm", "singular"))
class BoozerExactNewtonResult:
    """Store a BoozerExact iterate, residual, Jacobian, and termination state.

    Args:
        x (jax.Array): Final shape (nsurface + 2,) vector [all surface DOFs in
            meters, dimensionless iota, G in tesla meters]; G is always present.
        residual (jax.Array): All unweighted Boozer residuals, shape (3 * npoints,), at
            x.
        jacobian (jax.Array): Masked exact-system Jacobian, shape (nb, nx), at x.
        iterations (jax.Array): Scalar shape () int32 completed step count.
        norm (jax.Array): Scalar shape () Euclidean norm, at the final iterate, of the
            unweighted, unscaled Boozer residual rows selected by residual_rows,
            followed by label minus targetlabel and, without stellarator symmetry,
            z(0, 0). These terms have mixed physical units. See
            :func:`boozer_exact_newton` for the norm when no step is allowed.
        singular (jax.Array): Scalar shape () bool indicating an exactly singular finite
            step matrix; x retains the iterate that encountered it.
    """

    x: jax.Array
    residual: jax.Array
    jacobian: jax.Array
    iterations: jax.Array
    norm: jax.Array
    singular: jax.Array


@pytree_dataclass(data=("x", "gradient", "hessian", "iterations", "norm", "singular"))
class BoozerPenaltyNewtonResult:
    """Store a penalty iterate, unshifted derivatives, and termination state.

    Args:
        x (jax.Array): Final shape (nx,) vector [all surface DOFs in meters,
            dimensionless iota, optional G in tesla meters]; G is present only when
            optimize_G is true, with nx = nsurface + 1 + int(optimize_G).
        gradient (jax.Array): Penalty gradient, shape (nx,), at x.
        hessian (jax.Array): Unshifted penalty Hessian, shape (nx, nx), at x.
        iterations (jax.Array): Scalar shape () int32 completed step count.
        norm (jax.Array): Scalar shape () Final gradient Euclidean norm; units follow
            the residual or gradient.
        singular (jax.Array): Scalar shape () bool indicating an exactly singular finite
            step matrix; x retains the iterate that encountered it.
    """

    x: jax.Array
    gradient: jax.Array
    hessian: jax.Array
    iterations: jax.Array
    norm: jax.Array
    singular: jax.Array


@pytree_dataclass(
    data=("x", "residual", "gradient", "normal_matrix", "iterations", "norm", "singular")
)
class BoozerGaussNewtonResult:
    """Store a Gauss-Newton iterate, normal equations, and termination state.

    Args:
        x (jax.Array): Final shape (nx,) vector [all surface DOFs in meters,
            dimensionless iota, optional G in tesla meters]; G is present only when
            optimize_G is true, with nx = nsurface + 1 + int(optimize_G).
        residual (jax.Array): Normalized penalty residuals, shape (3 * npoints + 2,), at
            x.
        gradient (jax.Array): J transpose r, shape (nx,), at x.
        normal_matrix (jax.Array): J transpose J, shape (nx, nx), at x.
        iterations (jax.Array): Scalar shape () int32 completed step count.
        norm (jax.Array): Scalar shape () Final gradient Euclidean norm; units follow
            the residual or gradient.
        singular (jax.Array): Scalar shape () bool indicating an exactly singular finite
            step matrix; x retains the iterate that encountered it.
    """

    x: jax.Array
    residual: jax.Array
    gradient: jax.Array
    normal_matrix: jax.Array
    iterations: jax.Array
    norm: jax.Array
    singular: jax.Array


def _lu(matrix: jax.Array):
    """Return LU factors and flag zero pivots of finite matrices; preserve NaNs."""
    lu, pivots = lu_factor(matrix)
    dropped_nan = jnp.any(jnp.isnan(matrix)) & ~jnp.any(jnp.isnan(lu))
    lu = jnp.where(dropped_nan, jnp.nan, lu)
    return (lu, pivots), jnp.all(jnp.isfinite(matrix)) & jnp.any(jnp.diagonal(lu) == 0)


def _restore_on_failure(final, initial, norm, initial_norm, singular):
    """Restore initial values after a worse or non-finite norm, unless singular."""
    restore = ~singular & ~(jnp.isfinite(norm) & (norm <= initial_norm))
    return _select(restore, initial, final)


def _advance(x, step, iterations, singular):
    """Advance the iterate and step count unless the step matrix is singular."""
    return jnp.where(singular, x, x - step), jnp.where(singular, iterations, iterations + 1)


def _continues(iterations, norm, singular, tol, maxiter):
    """Continue within iteration and norm limits, stopping for NaNs or singularity."""
    return (iterations < maxiter) & (norm > tol) & ~singular


@partial(jit, static_argnames=("G_from_currents",))
def boozer_exact_newton(
    problem: BoozerProblem,
    x: jax.Array,
    residual_rows: jax.Array,
    tol: jax.Array,
    maxiter: jax.Array,
    *,
    G_from_currents: bool = False,
) -> BoozerExactNewtonResult:
    """Solve the masked BoozerExact system by undamped Newton iteration.

    Success uses the final system norm, including at the step cap; nonpositive
    or NaN ``maxiter`` returns norm ``1e6`` without stepping. A worse or
    non-finite final norm restores the initial iterate and its system unless a
    step is singular; the returned norm remains the final one.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nsurface + 2,) initial [surface DOFs, iota, G], cast
            to float64; (nsurface + 1,) without G when G_from_currents is true.
        residual_rows (jax.Array): Shape (nrows,) integer indices of the flattened
            Boozer residual, in native mask order.
        tol (jax.Array): Scalar shape () stopping tolerance for the Euclidean system
            norm, combining unscaled masked residual and label/z constraint units.
        maxiter (jax.Array): Scalar shape () float64 iteration cap; may be infinite.
            Positive fractional caps allow steps while the integer count is below it.
        G_from_currents (bool): Append initial G computed from the coil currents, then
            optimize it.

    Returns:
        BoozerExactNewtonResult: Result object with the final (or restored starting)
            iterate, its residual and Jacobian, the final norm and the singular-step
            flag; input problem is unchanged.
    """
    x = as_jax_float64(x)
    if G_from_currents:
        x = jnp.concatenate((x, _G_from_coil_currents(problem.coils)[None]))

    def system(x):
        b, jacobian = boozer_exact_residual(problem, x, residual_rows, derivatives=1)
        return b, jacobian, jnp.linalg.norm(b)

    def proceed(state):
        _, _, _, norm, iterations, singular = state
        return (iterations < maxiter) & ~(norm <= tol) & ~singular

    def step(state):
        x, b, jacobian, _, iterations, _ = state
        factors, singular = _lu(jacobian)
        dx = lu_solve(factors, b)
        dx = dx + lu_solve(factors, b - jacobian @ dx)
        x, iterations = _advance(x, dx, iterations, singular)
        return (x, *system(x), iterations, singular)

    start = system(x)
    _, initial_jacobian, initial_norm = start
    final_x, _, final_jacobian, norm, iterations, singular = lax.while_loop(
        proceed, step, (x, *start, jnp.asarray(0, jnp.int32), jnp.asarray(False))
    )
    # Native's loop evaluates the system only once it runs (0 < maxiter).
    sentinel = jnp.asarray(_EXACT_INITIAL_NORM, x.dtype)
    ran = 0 < maxiter
    initial_norm, norm = (jnp.where(ran, value, sentinel) for value in (initial_norm, norm))
    x, jacobian = _restore_on_failure(
        (final_x, final_jacobian), (x, initial_jacobian), norm, initial_norm, singular
    )
    (residual,) = boozer_surface_residual(problem, x, optimize_G=True)
    return BoozerExactNewtonResult(
        x=x,
        residual=residual,
        jacobian=jacobian,
        iterations=iterations,
        norm=norm,
        singular=singular,
    )


@partial(jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_newton(
    problem: BoozerProblem,
    x: jax.Array,
    tol: jax.Array,
    maxiter: jax.Array,
    stab: jax.Array,
    divergence_factor: jax.Array | None = None,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> BoozerPenaltyNewtonResult:
    """Minimize the Boozer penalty by undamped Newton iteration.

    A larger or non-finite final gradient norm restores the initial iterate
    and its derivatives unless a step is singular; the norm remains final.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        tol (jax.Array): Scalar shape () Euclidean gradient-norm tolerance, including
            at the last permitted step.
        maxiter (jax.Array): Scalar shape () float64 iteration cap; may be infinite.
            Positive fractional caps allow steps while the integer count is below it.
        stab (jax.Array): Scalar shape () diagonal Hessian shift in native units,
            applied only to the step solve.
        divergence_factor (jax.Array | None): Scalar shape () dimensionless bound on
            the gradient norm relative to its starting value, beyond which the walk
            stops; None disables the check.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        BoozerPenaltyNewtonResult: Result object with the final (or restored starting)
            iterate and its unshifted derivatives, the final gradient norm and the
            singular-step flag; input problem is unchanged.
    """
    x = as_jax_float64(x)

    def derivatives(x):
        _, gradient, hessian = boozer_penalty_constraints(
            problem, x, derivatives=2, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        return gradient, hessian

    start = (x, *derivatives(x))
    initial_norm = jnp.linalg.norm(start[1])

    def proceed(state):
        _, _, _, iterations, norm, singular = state
        continues = _continues(iterations, norm, singular, tol, maxiter)
        if divergence_factor is None:
            return continues
        return continues & (norm <= divergence_factor * initial_norm)

    def step(state):
        x, gradient, hessian, iterations, norm, _ = state
        # Native's ``stab * np.identity(n)`` entry by entry: XLA turns a product
        # with the identity into a select, which drops IEEE's ``inf * 0 = NaN``.
        diagonal = jnp.eye(x.shape[0], dtype=bool)
        shifted = hessian + jnp.where(diagonal, stab, stab * 0)
        factors, singular = _lu(shifted)
        dx = lu_solve(factors, gradient)
        dx = lax.cond(
            norm < _REFINE_BELOW,
            lambda dx: dx + lu_solve(factors, gradient - shifted @ dx),
            lambda dx: dx,
            dx,
        )
        x, iterations = _advance(x, dx, iterations, singular)
        gradient, hessian = derivatives(x)
        return x, gradient, hessian, iterations, jnp.linalg.norm(gradient), singular

    initial = (*start, jnp.asarray(0, jnp.int32), initial_norm, jnp.asarray(False))
    *final, iterations, norm, singular = lax.while_loop(proceed, step, initial)
    x, gradient, hessian = _restore_on_failure(tuple(final), start, norm, initial_norm, singular)
    return BoozerPenaltyNewtonResult(x, gradient, hessian, iterations, norm, singular)


@partial(jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_gauss_newton(
    problem: BoozerProblem,
    x: jax.Array,
    tol: jax.Array,
    maxiter: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> BoozerGaussNewtonResult:
    """Minimize the Boozer penalty by damped Gauss-Newton iteration.

    A larger or non-finite final gradient norm restores the initial iterate
    and its normal equations unless a step is singular; the norm remains final.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        tol (jax.Array): Scalar shape () Euclidean penalty-gradient norm tolerance.
        maxiter (jax.Array): Scalar shape () float64 iteration cap; may be infinite.
            Positive fractional caps allow steps while the integer count is below it.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        BoozerGaussNewtonResult: Result object with the final (or restored starting)
            iterate, its residual and normal equations, the final gradient norm and the
            singular-step flag; input problem is unchanged.
    """
    x = as_jax_float64(x)

    def normal_equations(x):
        residual, jacobian = boozer_penalty_residual(
            problem, x, derivatives=1, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        return residual, jacobian.T @ residual, jacobian.T @ jacobian

    def proceed(state):
        _, _, _, _, iterations, norm, _, singular = state
        return _continues(iterations, norm, singular, tol, maxiter)

    def step(state):
        x, _, gradient, normal_matrix, iterations, _, damping, _ = state
        factors, singular = _lu(normal_matrix + damping * jnp.diag(jnp.diag(normal_matrix)))
        x, iterations = _advance(x, lu_solve(factors, gradient), iterations, singular)
        residual, gradient, normal_matrix = normal_equations(x)
        norm = jnp.linalg.norm(gradient)
        return x, residual, gradient, normal_matrix, iterations, norm, damping * (1 / 3), singular

    start = (x, *normal_equations(x))
    initial_norm = jnp.linalg.norm(start[2])
    initial = (
        *start,
        jnp.asarray(0, jnp.int32),
        initial_norm,
        jnp.asarray(1.0, x.dtype),
        jnp.asarray(False),
    )
    *final, iterations, norm, _, singular = lax.while_loop(proceed, step, initial)
    x, residual, gradient, normal_matrix = _restore_on_failure(tuple(final), start, norm, initial_norm, singular)
    return BoozerGaussNewtonResult(x, residual, gradient, normal_matrix, iterations, norm, singular)


def _penalty_value_and_grad(problem: BoozerProblem, optimize_G: bool, weight_inv_modB: bool):
    """Return a callable for the penalty value and gradient in x."""
    return partial(
        boozer_penalty_constraints,
        problem,
        derivatives=1,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )


@partial(jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_bfgs(
    problem: BoozerProblem,
    x: jax.Array,
    gtol: jax.Array,
    maxiter: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> QuasiNewtonResult:
    """Minimize the Boozer penalty with device BFGS.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        gtol (jax.Array): Scalar shape () bound on the gradient's infinity norm.
        maxiter (jax.Array): Scalar shape () float64 iteration cap, which may be
            infinite.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        QuasiNewtonResult: SciPy's result fields; ``message`` indexes
            :data:`~simsopt_jax.core.quasi_newton.BFGS_MESSAGES`.
    """
    x = as_jax_float64(x)
    return minimize_bfgs(_penalty_value_and_grad(problem, optimize_G, weight_inv_modB), x, gtol, maxiter)


@partial(jit, static_argnames=("maxcor", "optimize_G", "weight_inv_modB"))
def boozer_penalty_lbfgsb(
    problem: BoozerProblem,
    x: jax.Array,
    ftol: jax.Array,
    gtol: jax.Array,
    maxiter: jax.Array,
    *,
    maxcor: int,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> QuasiNewtonResult:
    """Minimize the Boozer penalty with device L-BFGS-B without bounds.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) full DOFs (m), iota (dimensionless), and optional
            G (T m), cast to float64; nx = nsurface + 1 + int(optimize_G).
        ftol (jax.Array): Scalar shape () relative reduction tolerance of the penalty.
        gtol (jax.Array): Scalar shape () bound on the gradient's infinity norm.
        maxiter (jax.Array): Scalar shape () float64 iteration cap, which may be
            infinite.
        maxcor (int): Corrections held.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        QuasiNewtonResult: SciPy's result fields; ``message`` indexes
            :data:`~simsopt_jax.core.quasi_newton.LBFGSB_MESSAGES`.
    """
    x = as_jax_float64(x)
    return minimize_lbfgsb(
        _penalty_value_and_grad(problem, optimize_G, weight_inv_modB), x, ftol, gtol, maxiter, maxcor=maxcor
    )


@jit
def boozer_exact_residual_coil_vjp(
    problem: BoozerProblem, x: jax.Array, residual_rows: jax.Array, cotangent: jax.Array
) -> GroupedCoilSetSpec:
    """Apply a cotangent to the BoozerExact system's coil derivative at fixed x.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nsurface + 2,) [surface DOFs, iota, G] held fixed.
        residual_rows (jax.Array): Shape (nrows,) integer indices of the flattened
            Boozer residual, in native mask order.
        cotangent (jax.Array): Shape (nb,) multiplier of the exact-system equations,
            including constraint rows.

    Returns:
        GroupedCoilSetSpec: Coil geometry/current cotangent with the same leaf
            shapes and metadata as problem.coils, including label dependence.
    """

    def system(coils):
        return boozer_exact_residual(replace(problem, coils=coils), x, residual_rows)

    _, pullback = jax.vjp(system, problem.coils)
    return pullback(cotangent)[0]


@partial(jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_coil_vjp(
    problem: BoozerProblem,
    x: jax.Array,
    cotangent: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> GroupedCoilSetSpec:
    """Apply a cotangent to the penalty gradient's coil derivative at fixed x.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) vector [all surface DOFs in meters, dimensionless
            iota, optional G in tesla meters]; nx = nsurface + 1 + int(optimize_G).
        cotangent (jax.Array): Shape (nx,) multiplier of the penalty gradient.
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        GroupedCoilSetSpec: Cotangent object of the directional penalty derivative, with
            the same leaf shapes and metadata as problem.coils; includes label and
            current-derived G dependence.
    """

    def penalty(coils, point):
        return boozer_penalty_constraints(
            replace(problem, coils=coils), point, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )

    def directional_derivative(coils):
        return jax.jvp(partial(penalty, coils), (x,), (cotangent,))[1]

    return jax.grad(directional_derivative)(problem.coils)


@partial(jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_residual_objective(
    problem: BoozerProblem,
    x: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> tuple[jax.Array, jax.Array, GroupedCoilSetSpec]:
    """Evaluate the Boozer residual objective and its partial x and coil gradients.

    The objective uses the normalized Boozer and weighted label residuals,
    omitting the z penalty; coil gradients include all explicit dependence.

    Args:
        problem (BoozerProblem): Surface, coil and label snapshot with native units and
            DOF ordering.
        x (jax.Array): Shape (nx,) vector [all surface DOFs in meters, dimensionless
            iota, optional G in tesla meters]; nx = nsurface + 1 + int(optimize_G).
        optimize_G (bool): Include G as the last decision variable; otherwise use mu0
            times the sum of absolute coil currents.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        tuple[jax.Array, jax.Array, GroupedCoilSetSpec]: Objective scalar shape (),
            partial gradient in x of shape (nx,), and coil cotangents with matching leaf
            shapes. Uses the normalized Boozer and label residuals, omitting the z
            penalty; units follow the native penalty.
    """
    x = as_jax_float64(x)

    def objective(coils, point):
        (residual,) = boozer_penalty_residual(
            replace(problem, coils=coils), point, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        rtil = residual[:-1]
        return 0.5 * jnp.sum(rtil * rtil)

    value, (dcoils, dx) = jax.value_and_grad(objective, argnums=(0, 1))(problem.coils, x)
    return value, dx, dcoils
