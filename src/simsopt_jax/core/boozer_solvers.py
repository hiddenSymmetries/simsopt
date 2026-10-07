"""Native ``BoozerSurface`` solver loops and coil adjoints, in JAX.

Each solver runs native's iteration on a :class:`BoozerProblem` as one jitted
program (a ``lax.while_loop`` on the problem's device):

- :func:`boozer_exact_newton` is ``solve_residual_equation_exactly_newton``
  (BoozerExact): undamped Newton on the masked residual system of
  :func:`boozer_exact_residual`, one LU solve plus one refinement step;
- :func:`boozer_penalty_newton` is ``minimize_boozer_penalty_constraints_newton``:
  Newton on :func:`boozer_penalty_constraints` with its analytic Hessian
  shifted by ``stab``, refined once the gradient norm is below ``1e-9``;
- :func:`boozer_penalty_gauss_newton` is the damped Gauss-Newton of
  ``minimize_boozer_penalty_constraints_ls(method='manual')`` on
  :func:`boozer_penalty_residual`.

The stopping tests are native's, including how a NaN norm ends them, and so
is the ``norm`` whose ``norm <= tol`` is native's ``success``. ``tol``,
``maxiter`` and ``stab`` are float64 operands (``maxiter`` may be ``inf``),
so new values reuse the program. Where native's ``np.linalg.solve`` raises
``LinAlgError`` (an exactly zero LU pivot of a finite matrix), the loop stops
with ``singular`` set and ``x`` at the iterate that met it; the caller
raises. NaN steps continue, as natively. There is no damping, divergence
guard or restore of the starting point.

The coil adjoints are ``res['vjp']`` as cotangents of ``problem.coils``:
:func:`boozer_exact_residual_coil_vjp` is native's;
:func:`boozer_penalty_coil_vjp` differentiates the whole penalty, where
native's ``boozer_surface_dlsqgrad_dcoils_vjp`` drops the coil dependence of
the label (a ``ToroidalFlux`` label's vector potential) and of ``G`` from
the currents, and so is not the derivative of the solved surface there.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import lu_factor, lu_solve

from simsopt_jax.pytree import pytree_dataclass

from .boozer_problem import (
    BoozerProblem,
    _G_from_coil_currents,
    boozer_exact_residual,
    boozer_penalty_constraints,
    boozer_penalty_residual,
    boozer_surface_residual,
)
from .specs import GroupedCoilSetSpec

__all__ = [
    "BoozerExactNewtonResult",
    "BoozerGaussNewtonResult",
    "BoozerPenaltyNewtonResult",
    "boozer_exact_newton",
    "boozer_exact_residual_coil_vjp",
    "boozer_penalty_coil_vjp",
    "boozer_penalty_gauss_newton",
    "boozer_penalty_newton",
]

# Native refines the penalty Newton step with a second solve below this gradient norm.
_REFINE_BELOW = 1e-9
# Native's norm before the first BoozerExact iteration.
_EXACT_INITIAL_NORM = 1e6


@pytree_dataclass(data=("x", "residual", "jacobian", "iterations", "norm", "singular"))
class BoozerExactNewtonResult:
    """``x = [surface DOFs, iota, G]``; ``residual`` the unweighted residual at
    every point and ``jacobian`` the BoozerExact system's Jacobian, both at
    ``x``; ``iterations`` native's ``iter``."""

    x: jax.Array
    residual: jax.Array
    jacobian: jax.Array
    iterations: jax.Array
    norm: jax.Array
    singular: jax.Array


@pytree_dataclass(data=("x", "gradient", "hessian", "iterations", "norm", "singular"))
class BoozerPenaltyNewtonResult:
    """The penalty's gradient and (unshifted) Hessian at the final ``x``."""

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
    """At the final ``x``: the residuals ``r``, ``J^T r`` and ``J^T J``."""

    x: jax.Array
    residual: jax.Array
    gradient: jax.Array
    normal_matrix: jax.Array
    iterations: jax.Array
    norm: jax.Array
    singular: jax.Array


def _lu(matrix: jax.Array):
    """LU factors of ``matrix`` and whether LAPACK, under native's
    ``np.linalg.solve``, would stop at an exactly zero pivot: only for a
    finite matrix (NaNs propagate there; cuSOLVER can leave zero pivots).

    LAPACK keeps every NaN of the matrix in its factors, and its solves carry
    them. cuSOLVER's ``getrf`` can return finite factors for a matrix with a
    NaN; those factors are replaced by NaN (so the solve is all NaN, which is
    LAPACK's result for most such matrices)."""
    lu, pivots = lu_factor(matrix)
    dropped_nan = jnp.any(jnp.isnan(matrix)) & ~jnp.any(jnp.isnan(lu))
    lu = jnp.where(dropped_nan, jnp.nan, lu)
    return (lu, pivots), jnp.all(jnp.isfinite(matrix)) & jnp.any(jnp.diagonal(lu) == 0)


def _advance(x, step, iterations, singular):
    """Native leaves ``x`` and its count as they were when the solve raises."""
    return jnp.where(singular, x, x - step), jnp.where(singular, iterations, iterations + 1)


def _continues(iterations, norm, singular, tol, maxiter):
    """The penalty solvers' loop test: as natively, a NaN ``norm`` stops it."""
    return (iterations < maxiter) & (norm > tol) & ~singular


@partial(jax.jit, static_argnames=("G_from_currents",))
def boozer_exact_newton(
    problem: BoozerProblem,
    x: jax.Array,
    residual_rows: jax.Array,
    tol: jax.Array,
    maxiter: jax.Array,
    *,
    G_from_currents: bool = False,
) -> BoozerExactNewtonResult:
    """Native ``solve_residual_equation_exactly_newton`` from ``x = [surface
    DOFs, iota, G]`` (``[surface DOFs, iota]`` with ``G_from_currents``, which
    starts from native's ``G`` of the coil currents), on the BoozerExact rows
    of :func:`simsopt_jax_adapters.geo.boozer_problem.boozer_exact_residual_rows`.

    As natively, ``norm`` at ``maxiter`` is the one checked before the last
    step (``1e6`` if none was taken), so such a solve never succeeds.
    """
    if G_from_currents:
        x = jnp.concatenate((x, _G_from_coil_currents(problem.coils)[None]))

    def system(x):
        b, jacobian = boozer_exact_residual(problem, x, residual_rows, derivatives=1)
        return b, jacobian, jnp.linalg.norm(b)

    def proceed(state):
        _, _, _, norm, iterations, _, singular = state
        return (iterations < maxiter) & ~(norm <= tol) & ~singular

    def step(state):
        x, b, jacobian, norm, iterations, _, _ = state
        factors, singular = _lu(jacobian)
        dx = lu_solve(factors, b)
        dx = dx + lu_solve(factors, b - jacobian @ dx)
        x, iterations = _advance(x, dx, iterations, singular)
        return (x, *system(x), iterations, norm, singular)

    initial = (
        x,
        *system(x),
        jnp.asarray(0, jnp.int32),
        jnp.asarray(_EXACT_INITIAL_NORM, x.dtype),
        jnp.asarray(False),
    )
    x, _, jacobian, norm, iterations, checked_norm, singular = lax.while_loop(proceed, step, initial)
    (residual,) = boozer_surface_residual(problem, x, optimize_G=True)
    return BoozerExactNewtonResult(
        x=x,
        residual=residual,
        jacobian=jacobian,
        iterations=iterations,
        norm=jnp.where(iterations < maxiter, norm, checked_norm),
        singular=singular,
    )


@partial(jax.jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_newton(
    problem: BoozerProblem,
    x: jax.Array,
    tol: jax.Array,
    maxiter: jax.Array,
    stab: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> BoozerPenaltyNewtonResult:
    """Native ``minimize_boozer_penalty_constraints_newton`` from ``x``, with
    the problem's ``constraint_weight``; ``norm`` is the final gradient norm."""

    def derivatives(x):
        _, gradient, hessian = boozer_penalty_constraints(
            problem, x, derivatives=2, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        return gradient, hessian

    def proceed(state):
        _, _, _, iterations, norm, singular = state
        return _continues(iterations, norm, singular, tol, maxiter)

    def step(state):
        x, gradient, hessian, iterations, norm, _ = state
        # Native's ``stab * np.identity(n)`` entry by entry: XLA turns a product
        # with the identity into a select, which drops IEEE's ``inf * 0 = NaN``.
        diagonal = jnp.eye(x.shape[0], dtype=bool)
        shifted = hessian + jnp.where(diagonal, stab, stab * 0)
        factors, singular = _lu(shifted)
        dx = lu_solve(factors, gradient)
        dx = jnp.where(norm < _REFINE_BELOW, dx + lu_solve(factors, gradient - shifted @ dx), dx)
        x, iterations = _advance(x, dx, iterations, singular)
        gradient, hessian = derivatives(x)
        return x, gradient, hessian, iterations, jnp.linalg.norm(gradient), singular

    gradient, hessian = derivatives(x)
    initial = (
        x,
        gradient,
        hessian,
        jnp.asarray(0, jnp.int32),
        jnp.linalg.norm(gradient),
        jnp.asarray(False),
    )
    return BoozerPenaltyNewtonResult(*lax.while_loop(proceed, step, initial))


@partial(jax.jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_gauss_newton(
    problem: BoozerProblem,
    x: jax.Array,
    tol: jax.Array,
    maxiter: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> BoozerGaussNewtonResult:
    """Native ``minimize_boozer_penalty_constraints_ls(method='manual')`` from
    ``x``: steps ``(J^T J + lam diag(J^T J))^{-1} J^T r`` with ``lam = 1, 1/3,
    1/9, ...``; ``norm`` is the final ``|J^T r|``."""

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

    residual, gradient, normal_matrix = normal_equations(x)
    initial = (
        x,
        residual,
        gradient,
        normal_matrix,
        jnp.asarray(0, jnp.int32),
        jnp.linalg.norm(gradient),
        jnp.asarray(1.0, x.dtype),
        jnp.asarray(False),
    )
    x, residual, gradient, normal_matrix, iterations, norm, _, singular = lax.while_loop(
        proceed, step, initial
    )
    return BoozerGaussNewtonResult(x, residual, gradient, normal_matrix, iterations, norm, singular)


@jax.jit
def boozer_exact_residual_coil_vjp(
    problem: BoozerProblem, x: jax.Array, residual_rows: jax.Array, cotangent: jax.Array
) -> GroupedCoilSetSpec:
    """``cotangent^T db/dcoils`` for the BoozerExact system ``b`` of
    :func:`boozer_exact_residual` at ``x = [surface DOFs, iota, G]``, ``G``
    held: native ``boozer_surface_dexactresidual_dcoils_dcurrents_vjp`` with
    ``lm = cotangent``, label row included."""

    def system(coils):
        return boozer_exact_residual(replace(problem, coils=coils), x, residual_rows)

    _, pullback = jax.vjp(system, problem.coils)
    return pullback(cotangent)[0]


@partial(jax.jit, static_argnames=("optimize_G", "weight_inv_modB"))
def boozer_penalty_coil_vjp(
    problem: BoozerProblem,
    x: jax.Array,
    cotangent: jax.Array,
    *,
    optimize_G: bool,
    weight_inv_modB: bool,
) -> GroupedCoilSetSpec:
    """``cotangent^T d(grad f)/dcoils`` for the penalty ``f`` of
    :func:`boozer_penalty_constraints` at ``x``: the coil term of the BoozerLS
    adjoint (``res['vjp']`` with ``lm = cotangent``).

    Every coil dependence of ``f`` is differentiated: the Boozer residual, a
    ``ToroidalFlux`` label and, when it is not a variable, ``G`` from the
    currents. Native ``boozer_surface_dlsqgrad_dcoils_vjp`` keeps only the
    first, so it agrees with this one for ``Volume``, ``Area`` and
    ``AspectRatio`` labels with ``G`` optimized or the currents fixed.
    """

    def penalty(coils, point):
        return boozer_penalty_constraints(
            replace(problem, coils=coils), point, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )

    def directional_derivative(coils):
        return jax.jvp(partial(penalty, coils), (x,), (cotangent,))[1]

    return jax.grad(directional_derivative)(problem.coils)
