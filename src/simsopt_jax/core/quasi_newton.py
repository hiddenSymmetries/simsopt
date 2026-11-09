# This module ports SciPy 1.18's BFGS (scipy/optimize/_optimize.py and
# _linesearch.py) and L-BFGS-B (scipy/optimize/__lbfgsb.c) to JAX. They are
# distributed under the following licenses, whose notices, conditions and
# disclaimers are retained here.
#
# SciPy (scipy/LICENSE.txt):
#
# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# scipy/optimize/_optimize.py, whose _minimize_bfgs minimize_bfgs ports:
#
# ******NOTICE***************
# optimize.py module by Travis E. Oliphant
#
# You may copy and use this module as you see fit with no
# guarantee implied provided you keep this notice in all copies.
# *****END NOTICE************
#
# SciPy's C translation of L-BFGS-B 3.0, with DCSRCH and DCSTEP
# (scipy/optimize/__lbfgsb.c):
#
# Copyright (C) 2024 SciPy developers
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# a. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# b. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# c. Names of the SciPy Developers may not be used to endorse or promote
#    products derived from this software without specific prior written
#    permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
#
# L-BFGS-B 3.0 (BSD-3) is by Ciyou Zhu, Richard Byrd, Jorge Nocedal and Jose
# Luis Morales (https://users.iems.northwestern.edu/~nocedal/lbfgsb.html).
# DCSRCH and DCSTEP are from MINPACK-1 (Jorge J. More' and David J. Thuente,
# Argonne National Laboratory, 1983) and MINPACK-2 (Brett M. Averick, Richard
# G. Carter and Jorge J. More', Argonne National Laboratory and University of
# Minnesota, 1993).

"""SciPy's BFGS and unbounded L-BFGS-B, each one compiled JAX loop.

:func:`minimize_bfgs` is ``scipy.optimize.minimize(fun, x0, jac=True,
method="BFGS")``: ``_minimize_bfgs`` with its ``_line_search_wolfe12``
(MINPACK-2's DCSRCH through ``line_search_wolfe1``, then
``line_search_wolfe2`` where that fails). :func:`minimize_lbfgsb` is
``method="L-BFGS-B"`` without bounds: SciPy's C translation of L-BFGS-B 3.0,
with its compact limited-memory matrices, its ``lnsrlb`` line search and its
stopping tests. ``value_and_grad(x)`` returns ``fun`` and its gradient.

Each solver is a ``lax.while_loop`` whose body evaluates ``value_and_grad``
once, at the point SciPy evaluates next, then runs SciPy's logic without
branching up to the next evaluation: the objective is traced once, and the
tolerances and caps are operands, so a caller that jits a solver with its
problem as an argument compiles it once.

Statuses, messages (``BFGS_MESSAGES[message]``, ``LBFGSB_MESSAGES[message]``),
iteration counts, ``nfev``/``njev`` (``ScalarFunction`` counts a point again
only when it differs from the last one), ``hess_inv`` and SciPy's checks of
its inputs without bounds are SciPy's. These claims are checked against
SciPy 1.18 only; other SciPy versions may differ in them.

Iterates agree with SciPy to round-off, not bitwise: XLA orders sums and may
fuse a multiply and an add into one rounding (it does on CPU and GPU), and BFGS
forms SciPy's update from outer products, in O(n^2) where SciPy's matrix
products cost O(n^3). A decision that hinges on the last bit (a memory reset
under underflow, or a trial point equal to an earlier one, which SciPy does not
count again) can go the other way; run op by op, the port decides as SciPy.
A long walk magnifies the round-off and can part from SciPy's (L-BFGS-B's
sooner): the counts differ, a converged walk ends at SciPy's solution only to
what ``gtol`` allows (L-BFGS-B's possibly by its other test), and a walk
stopped by ``maxiter`` can end elsewhere.

Known differences in counting: SciPy compares a point with its cached one by
array equality, so a point with a NaN coordinate never matches it and is
evaluated and counted again (at a NaN-containing ``x0``, a second time at the
start; not on the exits for invalid inputs, which request nothing after the
first evaluation). The port counts it once, so for such points ``nfev``,
``njev`` and, at an L-BFGS-B ``maxfun`` limit, the status and message can
differ from SciPy's.

Known difference in range: SciPy updates BFGS's inverse Hessian as
``A1 @ (H @ A2) + rho s s^T`` with ``A1 = I - rho s y^T`` and ``A2 = A1^T``
formed as matrices, two dense products costing O(n^3) per iteration; the port
applies ``A2`` and then ``A1`` as rank-one updates, ``H - (H y)(rho s)^T`` and
then minus ``(rho s)(y^T H A2)``, matrix-vector and outer products costing
O(n^2). Its intermediates ``H y`` and ``y^T H A2`` carry the scale of ``H``
times ``y``, which SciPy's never do (``rho`` multiplies ``s y^T`` before ``H``
does), so where that exceeds float64's range the port's update is NaN and
SciPy's can be finite: ``H = [[1e200]]``, ``s = [1]``, ``y = [1e200]`` gives
NaN in the port and ``[[1e-200]]`` in SciPy. SciPy's update overflows only
where an entry of ``s y^T``, ``H A2`` or the result does.

Not provided: bounds, callbacks and SciPy's other options
(``c1`` and ``c2`` keep SciPy's defaults, so its check of them cannot fail;
``norm``, ``eps``, ``hess_inv0``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular

from simsopt_jax.pytree import pytree_dataclass

from .dcsrch import (
    CONVERGENCE,
    ERROR,
    FG,
    DcsrchState,
    _py_min,
    _select,
    dcsrch_iterate,
    dcsrch_start,
)

__all__ = [
    "BFGS_MESSAGES",
    "LBFGSB_MESSAGES",
    "LbfgsbCorrections",
    "QuasiNewtonResult",
    "minimize_bfgs",
    "minimize_lbfgsb",
]

ValueAndGrad = Callable[[jax.Array], tuple[jax.Array, jax.Array]]

# SciPy's ``_status_message`` entries, indexed by BFGS's ``status``.
BFGS_MESSAGES = (
    "Optimization terminated successfully.",
    "Maximum number of iterations has been exceeded.",
    "Desired error not necessarily achieved due to precision loss.",
    "NaN result encountered.",
)

# SciPy's L-BFGS-B messages, ``status_messages[task] + ": " + task_messages[msg]``.
LBFGSB_MESSAGES = (
    "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL",
    "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
    "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
    "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT",
    "ABNORMAL: ",
    "ERROR: FACTR < 0",
    "ERROR: M <= 0",
    "ERROR: N <= 0",
)
(
    _CONV_GRAD,
    _CONV_F,
    _STOP_MAXITER,
    _STOP_MAXFUN,
    _ABNORMAL,
    _ERROR_FACTR,
    _ERROR_M,
    _ERROR_N,
) = range(len(LBFGSB_MESSAGES))


@pytree_dataclass(data=("s", "y", "count"))
class LbfgsbCorrections:
    """L-BFGS-B's ``hess_inv``: SciPy's ``LbfgsInvHessProduct(s[:count], y[:count])``.

    Args:
        s (jax.Array): Steps, shape (maxcor, n), in the slot order of SciPy's ring buffer.
        y (jax.Array): Gradient changes, shape (maxcor, n), in the same order.
        count (jax.Array): Scalar int32 number of pairs held (SciPy's ``n_corrs``).
    """

    s: jax.Array
    y: jax.Array
    count: jax.Array


@pytree_dataclass(data=("x", "fun", "jac", "iterations", "nfev", "njev", "status", "message", "hess_inv"))
class QuasiNewtonResult:
    """SciPy's ``OptimizeResult`` fields of a solve, as arrays.

    Args:
        x (jax.Array): Final iterate, shape (n,).
        fun (jax.Array): Scalar objective at ``x``.
        jac (jax.Array): Gradient at ``x``, shape (n,).
        iterations (jax.Array): Scalar int32 ``nit``.
        nfev (jax.Array): Scalar int32 objective evaluations as SciPy counts them.
        njev (jax.Array): Scalar int32 gradient evaluations as SciPy counts them.
        status (jax.Array): Scalar int32 SciPy ``status`` (0 is success).
        message (jax.Array): Scalar int32 index of SciPy's message in
            :data:`BFGS_MESSAGES` or :data:`LBFGSB_MESSAGES`.
        hess_inv (jax.Array | LbfgsbCorrections): SciPy's ``hess_inv``: BFGS's inverse
            Hessian, shape (n, n), or L-BFGS-B's correction pairs.
    """

    x: jax.Array
    fun: jax.Array
    jac: jax.Array
    iterations: jax.Array
    nfev: jax.Array
    njev: jax.Array
    status: jax.Array
    message: jax.Array
    hess_inv: jax.Array | LbfgsbCorrections


def _raises(result, *operands):
    """Whether NumPy, under ``errstate(divide='raise', over='raise',
    invalid='raise')``, raises on the operation giving ``result``: an infinity
    from finite operands (overflow or division by zero) or a NaN from operands
    without one (invalid)."""
    finite = jnp.all(jnp.stack([jnp.isfinite(operand) for operand in operands]))
    no_nan = ~jnp.any(jnp.stack([jnp.isnan(operand) for operand in operands]))
    return (jnp.isinf(result) & finite) | (jnp.isnan(result) & no_nan)


class _Checked:
    """Records which operations of SciPy's interpolants would raise."""

    def __init__(self):
        self.raised = []

    def __call__(self, result, *operands):
        self.raised.append(_raises(result, *operands))
        return result

    def value(self, xmin):
        """SciPy's result: ``xmin`` unless an operation raised or it is not finite."""
        return xmin, ~jnp.any(jnp.stack(self.raised)) & jnp.isfinite(xmin)


def _cubicmin(a, fa, fpa, b, fb, c, fc):
    """SciPy's ``_cubicmin``: the minimizer of the cubic through ``(a, fa)``,
    ``(b, fb)``, ``(c, fc)`` with slope ``fpa`` at ``a``, and whether SciPy
    returns it (not ``None``). ``np.dot`` checks no floating-point flags."""
    op = _Checked()
    C = fpa
    db = op(b - a, b, a)
    dc = op(c - a, c, a)
    product = op(db * dc, db, dc)
    squared = op(product**2, product)
    spread = op(db - dc, db, dc)
    denom = op(squared * spread, squared, spread)
    dc2, db2 = op(dc**2, dc), op(db**2, db)
    dc3, db3 = op(dc**3, dc), op(db**3, db)
    first = op(fb - fa, fb, fa)
    slope_b = op(C * db, C, db)
    rhs_b = op(first - slope_b, first, slope_b)
    second = op(fc - fa, fc, fa)
    slope_c = op(C * dc, C, dc)
    rhs_c = op(second - slope_c, second, slope_c)
    A = dc2 * rhs_b + (-db2) * rhs_c
    B = (-dc3) * rhs_b + db3 * rhs_c
    A = op(A / denom, A, denom)
    B = op(B / denom, B, denom)
    BB = op(B * B, B)
    A3 = op(3 * A, A)
    A3C = op(A3 * C, A3, C)
    radical = op(BB - A3C, BB, A3C)
    root = op(jnp.sqrt(radical), radical)
    numerator = op(-B + root, B, root)
    quotient = op(numerator / A3, numerator, A3)
    return op.value(op(a + quotient, a, quotient))


def _quadmin(a, fa, fpa, b, fb):
    """SciPy's ``_quadmin``: the minimizer of the quadratic through ``(a, fa)``
    and ``(b, fb)`` with slope ``fpa`` at ``a``, and whether SciPy returns it."""
    op = _Checked()
    D, C = fa, fpa
    db = op(b - a * 1.0, b, a)
    first = op(fb - D, fb, D)
    slope = op(C * db, C, db)
    numerator = op(first - slope, first, slope)
    db2 = op(db * db, db)
    B = op(numerator / db2, numerator, db2)
    B2 = op(2.0 * B, B)
    quotient = op(C / B2, C, B2)
    return op.value(op(a - quotient, a, quotient))


# --- BFGS -------------------------------------------------------------------------------

# ``_minimize_bfgs``'s line-search arguments: ``c1``, ``c2``, ``amin``, ``amax``;
# DCSRCH's ``xtol`` and evaluation cap in ``line_search_wolfe1``; the caps and
# interpolation checks of ``scalar_search_wolfe2`` and its ``_zoom``.
_C1 = 1e-4
_C2 = 0.9
_AMIN = 1e-100
_AMAX = 1e100
_WOLFE1_XTOL = 1e-14
_WOLFE1_MAXITER = 100
_WOLFE2_MAXITER = 10
_ZOOM_MAXITER = 10
_ZOOM_DELTA1 = 0.2
_ZOOM_DELTA2 = 0.1
# ``rhok`` where ``y^T s`` is exactly zero.
_RHOK_DIVIDE_BY_ZERO = 1000.0
# line_search_wolfe1's ``DCSRCH(phi, derphi, c1, c2, xtol, amin, amax)``.
_wolfe1_start = partial(dcsrch_start, ftol=_C1, stpmin=_AMIN, stpmax=_AMAX)
_wolfe1_iterate = partial(
    dcsrch_iterate, ftol=_C1, gtol=_C2, xtol=_WOLFE1_XTOL, stpmin=_AMIN, stpmax=_AMAX, lbfgsb=False
)

# What the pending evaluation is for: the start, a DCSRCH trial, a
# ``scalar_search_wolfe2`` step, a ``_zoom`` trial; or the solve ended.
_INIT, _WOLFE1, _WOLFE2, _ZOOM, _DONE = range(5)


@pytree_dataclass(
    data=(
        "phase",
        "x_trial",
        "x",
        "f",
        "g",
        "H",
        "k",
        "old_old_fval",
        "gnorm",
        "pk",
        "derphi0",
        "alpha",
        "line",
        "wolfe1_evals",
        "alpha0",
        "phi_a0",
        "derphi_a0",
        "wolfe2_iter",
        "a_lo",
        "a_hi",
        "phi_lo",
        "phi_hi",
        "derphi_lo",
        "a_rec",
        "phi_rec",
        "zoom_iter",
        "precision_lost",
        "nfev",
        "njev",
        "last_x",
        "grad_counted",
    )
)
class _BfgsState:
    phase: jax.Array
    x_trial: jax.Array  # the point to evaluate next: x + alpha pk
    x: jax.Array
    f: jax.Array  # SciPy's old_fval
    g: jax.Array  # gfk
    H: jax.Array
    k: jax.Array
    old_old_fval: jax.Array
    gnorm: jax.Array
    pk: jax.Array
    derphi0: jax.Array
    alpha: jax.Array  # the step of x_trial
    line: DcsrchState
    wolfe1_evals: jax.Array
    alpha0: jax.Array
    phi_a0: jax.Array
    derphi_a0: jax.Array
    wolfe2_iter: jax.Array
    a_lo: jax.Array
    a_hi: jax.Array
    phi_lo: jax.Array
    phi_hi: jax.Array
    derphi_lo: jax.Array
    a_rec: jax.Array
    phi_rec: jax.Array
    zoom_iter: jax.Array
    precision_lost: jax.Array
    nfev: jax.Array
    njev: jax.Array
    last_x: jax.Array  # the point of SciPy's cached value and gradient
    grad_counted: jax.Array  # the gradient at last_x has been requested


def _initial_step(phi0, old_phi0, derphi0):
    """Both line searches' first step: ``min(1, 1.01 * 2 (phi0 - old_phi0) / derphi0)``,
    1 where that is negative or ``derphi0`` is zero."""
    alpha1 = jnp.where(derphi0 != 0, _py_min(1.0, 1.01 * 2 * (phi0 - old_phi0) / derphi0), 1.0)
    return jnp.where(alpha1 < 0, 1.0, alpha1)


def _vecnorm(x):
    """SciPy's ``vecnorm(x)``: ``sum(|x|^2)^(1/2)``."""
    return jnp.sum(jnp.abs(x) ** 2) ** (1.0 / 2)


def _zoom_trial(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, a_rec, phi_rec, i):
    """``_zoom``'s next step: the cubic interpolant from the second iteration,
    else the quadratic, else bisection, each kept only inside its margin."""
    dalpha = a_hi - a_lo
    a = jnp.where(dalpha < 0, a_hi, a_lo)
    b = jnp.where(dalpha < 0, a_lo, a_hi)
    cchk = _ZOOM_DELTA1 * dalpha
    cubic, cubic_ok = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
    use_cubic = (i > 0) & cubic_ok & ~(cubic > b - cchk) & ~(cubic < a + cchk)
    qchk = _ZOOM_DELTA2 * dalpha
    quadratic, quadratic_ok = _quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
    use_quadratic = quadratic_ok & ~(quadratic > b - qchk) & ~(quadratic < a + qchk)
    return jnp.where(use_cubic, cubic, jnp.where(use_quadratic, quadratic, a_lo + 0.5 * dalpha))


def _bfgs_inverse_hessian_update(H: jax.Array, sk: jax.Array, yk: jax.Array) -> jax.Array:
    """SciPy's BFGS update of the inverse Hessian ``H`` by the step ``sk`` and gradient change ``yk``:
    ``A1 H A2 + rho s s^T``, with ``rho = 1 / (y^T s)`` (SciPy's 1000 where ``y^T s`` is exactly
    zero), ``A1 = I - rho s y^T`` and ``A2 = A1^T``.

    Args:
        H (jax.Array): Inverse Hessian, shape (n, n), in (x units)^2 per objective unit.
        sk (jax.Array): Step, shape (n,), in x units.
        yk (jax.Array): Gradient change over the step, shape (n,), in objective per x units.

    Returns:
        jax.Array: Updated inverse Hessian, shape (n, n), in H's units.
    """
    rhok_inv = jnp.dot(yk, sk)
    rhok = jnp.where(rhok_inv == 0.0, _RHOK_DIVIDE_BY_ZERO, 1.0 / rhok_inv)
    # SciPy's factors applied in its order as rank-one updates, H A2 = H -
    # (H y) (rho s)^T and then A1 (H A2): the expanded rank-two form can cancel
    # to an indefinite H. Each outer product takes rho s, as SciPy's (rho s) s^T
    # does, so none underflows for |s| below 1e-154; outer products keep the
    # update O(n^2). H y can overflow where SciPy's products do not (module
    # docstring, "Known difference in range").
    rho_s = rhok * sk
    HA2 = H - jnp.outer(H @ yk, rho_s)
    return HA2 - jnp.outer(rho_s, yk @ HA2) + jnp.outer(rho_s, sk)


def _bfgs_step(s: _BfgsState, f_t, g_t, gtol, maxiter, xrtol) -> _BfgsState:
    """SciPy's BFGS from the value ``f_t`` and gradient ``g_t`` at
    ``s.x_trial`` up to its next evaluation (or its end)."""
    x_t = s.x_trial
    moved = ~jnp.all(x_t == s.last_x)
    grad_counted = s.grad_counted & ~moved
    phi0, derphi0, alpha = s.f, s.derphi0, s.alpha
    dphi_t = jnp.dot(g_t, s.pk)
    curvature = jnp.abs(dphi_t) <= -_C2 * derphi0

    # The first evaluation, at x0.
    init = s.phase == _INIT
    gnorm0 = jnp.max(jnp.abs(g_t))

    # line_search_wolfe1: DCSRCH after phi and derphi at alpha. Its loop
    # stops after 100 evaluations and at a non-finite step.
    wolfe1 = s.phase == _WOLFE1
    wolfe1_evals = s.wolfe1_evals + 1
    line, wolfe1_step, task = _wolfe1_iterate(s.line, alpha, f_t, dphi_t)
    wolfe1_running = wolfe1 & (wolfe1_evals < _WOLFE1_MAXITER)
    wolfe1_accept = wolfe1_running & (task == CONVERGENCE)
    wolfe1_next = wolfe1_running & (task == FG) & jnp.isfinite(wolfe1_step)
    wolfe1_failed = wolfe1 & ~wolfe1_accept & ~wolfe1_next

    # scalar_search_wolfe2: its loop body ``i`` after phi(alpha1); after the
    # last body's evaluation (``for ... else``) the step is taken as it is.
    wolfe2 = s.phase == _WOLFE2
    i = s.wolfe2_iter
    wolfe2_last = wolfe2 & (i >= _WOLFE2_MAXITER)
    wolfe2_body = wolfe2 & ~wolfe2_last
    no_step = (alpha == 0) | (s.alpha0 > _AMAX)
    too_high = (f_t > phi0 + _C1 * alpha * derphi0) | ((f_t >= s.phi_a0) & (i > 0))
    wolfe2_failed = wolfe2_body & no_step
    zoom_below = wolfe2_body & ~no_step & too_high
    wolfe2_slope = wolfe2_body & ~no_step & ~too_high
    wolfe2_accept = wolfe2_last | (wolfe2_slope & curvature)
    zoom_above = wolfe2_slope & ~curvature & (dphi_t >= 0)
    wolfe2_next = wolfe2_slope & ~curvature & ~(dphi_t >= 0)

    # _zoom after phi(a_j): a high value replaces a_hi; otherwise derphi(a_j)
    # decides, and a_j replaces a_lo.
    zoom = s.phase == _ZOOM
    zoom_high = (f_t > phi0 + _C1 * alpha * derphi0) | (f_t >= s.phi_lo)
    zoom_slope = zoom & ~zoom_high
    zoom_accept = zoom_slope & curvature
    flip = dphi_t * (s.a_hi - s.a_lo) >= 0
    a_rec = jnp.where(zoom_high | flip, s.a_hi, s.a_lo)
    phi_rec = jnp.where(zoom_high | flip, s.phi_hi, s.phi_lo)
    a_hi = jnp.where(zoom_high, alpha, jnp.where(flip, s.a_lo, s.a_hi))
    phi_hi = jnp.where(zoom_high, f_t, jnp.where(flip, s.phi_lo, s.phi_hi))
    a_lo = jnp.where(zoom_high, s.a_lo, alpha)
    phi_lo = jnp.where(zoom_high, s.phi_lo, f_t)
    derphi_lo = jnp.where(zoom_high, s.derphi_lo, dphi_t)
    zoom_iter = s.zoom_iter + 1
    zoom_failed = zoom & ~zoom_accept & (zoom_iter > _ZOOM_MAXITER)
    zoom_next = zoom & ~zoom_accept & ~zoom_failed

    # A zoom started by scalar_search_wolfe2 brackets with (alpha0, alpha1).
    zoom_start = zoom_below | zoom_above
    a_lo = jnp.where(zoom_below, s.alpha0, jnp.where(zoom_above, alpha, a_lo))
    a_hi = jnp.where(zoom_below, alpha, jnp.where(zoom_above, s.alpha0, a_hi))
    phi_lo = jnp.where(zoom_below, s.phi_a0, jnp.where(zoom_above, f_t, phi_lo))
    phi_hi = jnp.where(zoom_below, f_t, jnp.where(zoom_above, s.phi_a0, phi_hi))
    derphi_lo = jnp.where(zoom_below, s.derphi_a0, jnp.where(zoom_above, dphi_t, derphi_lo))
    a_rec = jnp.where(zoom_start, 0.0, a_rec)
    phi_rec = jnp.where(zoom_start, phi0, phi_rec)
    zoom_iter = jnp.where(zoom_start, 0, zoom_iter)
    zoom_step = _zoom_trial(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, a_rec, phi_rec, zoom_iter)

    # derphi (or, after scalar_search_wolfe2's last body, BFGS's own
    # myfprime) asks for the gradient at x_t.
    grad_called = init | wolfe1 | wolfe2_last | wolfe2_slope | zoom_slope
    njev = s.njev + (grad_called & ~grad_counted)

    s = replace(
        s,
        nfev=s.nfev + moved,
        njev=njev,
        last_x=x_t,
        grad_counted=grad_counted | grad_called,
        line=line,
        wolfe1_evals=wolfe1_evals,
        alpha0=jnp.where(wolfe2_next, alpha, s.alpha0),
        phi_a0=jnp.where(wolfe2_next, f_t, s.phi_a0),
        derphi_a0=jnp.where(wolfe2_next, dphi_t, s.derphi_a0),
        wolfe2_iter=i + 1,
        a_lo=a_lo,
        a_hi=a_hi,
        phi_lo=phi_lo,
        phi_hi=phi_hi,
        derphi_lo=derphi_lo,
        a_rec=a_rec,
        phi_rec=phi_rec,
        zoom_iter=zoom_iter,
        alpha=jnp.where(
            wolfe1_next,
            wolfe1_step,
            jnp.where(wolfe2_next, _py_min(2 * alpha, _AMAX), jnp.where(zoom_start | zoom_next, zoom_step, alpha)),
        ),
        phase=jnp.where(zoom_start | zoom_next, _ZOOM, s.phase),
    )

    # An accepted step at x_t ends the iteration; SciPy then stops on the
    # gradient norm, a small step or a non-finite value, else updates H.
    accept = wolfe1_accept | wolfe2_accept | zoom_accept
    sk = alpha * s.pk
    yk = g_t - s.g
    k = s.k + 1
    gnorm = jnp.max(jnp.abs(g_t))
    converged = gnorm <= gtol
    small_step = alpha * _vecnorm(s.pk) <= xrtol * (xrtol + _vecnorm(x_t))
    non_finite = ~jnp.isfinite(f_t)
    stop = converged | small_step | non_finite
    H = _bfgs_inverse_hessian_update(s.H, sk, yk)
    iterate = accept & ~stop & (gnorm > gtol) & (k < maxiter)
    s = replace(
        s,
        x=jnp.where(accept, x_t, s.x),
        f=jnp.where(accept, f_t, s.f),
        g=jnp.where(accept, g_t, s.g),
        old_old_fval=jnp.where(accept, s.f, s.old_old_fval),
        k=jnp.where(accept, k, s.k),
        gnorm=jnp.where(accept, gnorm, jnp.where(init, gnorm0, s.gnorm)),
        H=jnp.where(accept & ~stop, H, s.H),
    )
    s = replace(
        s,
        x=jnp.where(init, x_t, s.x),
        f=jnp.where(init, f_t, s.f),
        g=jnp.where(init, g_t, s.g),
        old_old_fval=jnp.where(init, f_t + jnp.linalg.norm(g_t) / 2, s.old_old_fval),
    )

    # A new iteration: pk = -H gfk and DCSRCH's start, which falls back to
    # scalar_search_wolfe2 on an error, as does a failed DCSRCH.
    start = (init & (gnorm0 > gtol) & (s.k < maxiter)) | iterate
    pk = -(s.H @ s.g)
    derphi0 = jnp.dot(s.g, pk)
    alpha1 = _initial_step(s.f, s.old_old_fval, derphi0)
    line, start_task = _wolfe1_start(alpha1, s.f, derphi0)
    wolfe1_start = start & (start_task != ERROR)
    s = replace(
        s,
        pk=jnp.where(start, pk, s.pk),
        derphi0=jnp.where(start, derphi0, s.derphi0),
        line=_select(wolfe1_start, line, s.line),
        wolfe1_evals=jnp.where(wolfe1_start, 0, s.wolfe1_evals),
        alpha=jnp.where(wolfe1_start, alpha1, s.alpha),
        phase=jnp.where(wolfe1_start | wolfe1_next, _WOLFE1, s.phase),
    )
    wolfe2_start = wolfe1_failed | (start & (start_task == ERROR))
    alpha1 = _py_min(_initial_step(s.f, s.old_old_fval, s.derphi0), _AMAX)
    s = replace(
        s,
        alpha=jnp.where(wolfe2_start, alpha1, s.alpha),
        alpha0=jnp.where(wolfe2_start, 0.0, s.alpha0),
        phi_a0=jnp.where(wolfe2_start, s.f, s.phi_a0),
        derphi_a0=jnp.where(wolfe2_start, s.derphi0, s.derphi_a0),
        wolfe2_iter=jnp.where(wolfe2_start, 0, s.wolfe2_iter),
        phase=jnp.where(wolfe2_start | wolfe2_next, _WOLFE2, s.phase),
    )

    line_search_failed = wolfe2_failed | zoom_failed
    done = (init & ~start) | line_search_failed | (accept & ~iterate)
    return replace(
        s,
        phase=jnp.where(done, _DONE, s.phase),
        precision_lost=s.precision_lost | line_search_failed | (accept & ~converged & ~small_step & non_finite),
        x_trial=s.x + s.alpha * s.pk,
    )


def minimize_bfgs(value_and_grad: ValueAndGrad, x0: jax.Array, gtol, maxiter, xrtol=0.0) -> QuasiNewtonResult:
    """``scipy.optimize.minimize(value_and_grad, x0, jac=True, method="BFGS",
    options={"gtol": gtol, "maxiter": maxiter, "xrtol": xrtol})``.

    Without variables SciPy's infinity norm of the gradient raises after it
    evaluates the objective at ``x0``; so does this. A NaN coordinate in
    ``x0`` makes SciPy count its first evaluation twice, this once (see the
    module docstring).

    Args:
        value_and_grad (Callable): ``x -> (fun(x), grad fun(x))``, traceable.
        x0 (jax.Array): Starting point, shape (n,).
        gtol (jax.Array | float): Stop when the gradient's infinity norm is at most gtol.
        maxiter (jax.Array | float): Iteration cap, which may be infinite.
        xrtol (jax.Array | float): Relative step tolerance; SciPy's default 0.

    Returns:
        QuasiNewtonResult: SciPy's fields; ``message`` indexes :data:`BFGS_MESSAGES`
            and ``hess_inv`` is the inverse Hessian.
    """
    if x0.shape[0] == 0:
        # SciPy's ScalarFunction evaluates x0 before the norm of the empty gradient raises.
        value_and_grad(x0)
        raise ValueError("zero-size array to reduction operation maximum which has no identity")
    zero = jnp.zeros((), x0.dtype)
    count = jnp.asarray(0, jnp.int32)
    line, _ = _wolfe1_start(zero, zero, zero)
    state = _BfgsState(
        phase=jnp.asarray(_INIT, jnp.int32),
        x_trial=x0,
        x=x0,
        f=zero,
        g=jnp.zeros_like(x0),
        H=jnp.eye(x0.shape[0], dtype=x0.dtype),
        k=count,
        old_old_fval=zero,
        gnorm=zero,
        pk=jnp.zeros_like(x0),
        derphi0=zero,
        alpha=zero,
        line=line,
        wolfe1_evals=count,
        alpha0=zero,
        phi_a0=zero,
        derphi_a0=zero,
        wolfe2_iter=count,
        a_lo=zero,
        a_hi=zero,
        phi_lo=zero,
        phi_hi=zero,
        derphi_lo=zero,
        a_rec=zero,
        phi_rec=zero,
        zoom_iter=count,
        precision_lost=jnp.asarray(False),
        nfev=count,
        njev=count,
        last_x=jnp.full_like(x0, jnp.nan),
        grad_counted=jnp.asarray(False),
    )

    def body(s):
        return _bfgs_step(s, *value_and_grad(s.x_trial), gtol, maxiter, xrtol)

    s = lax.while_loop(lambda s: s.phase != _DONE, body, state)
    nan = jnp.isnan(s.gnorm) | jnp.isnan(s.f) | jnp.any(jnp.isnan(s.x))
    status = jnp.where(s.precision_lost, 2, jnp.where(s.k >= maxiter, 1, jnp.where(nan, 3, 0)))
    status = status.astype(jnp.int32)
    return QuasiNewtonResult(s.x, s.f, s.g, s.k, s.nfev, s.njev, status, status, s.H)


# --- L-BFGS-B without bounds --------------------------------------------------------

# lnsrlb's DCSRCH tolerances and its step bound without bounds on x; mainlb's
# machine epsilon; SciPy's ``factr = ftol / np.finfo(float).eps``.
_LNSRLB_FTOL = 1e-3
_LNSRLB_GTOL = 0.9
_LNSRLB_XTOL = 0.1
_STPMX = 1e10
_EPSMCH = 2.220446049250313e-16
_FLOAT_EPS = float(np.finfo(float).eps)
_lnsrlb_start = partial(dcsrch_start, ftol=_LNSRLB_FTOL, stpmin=0.0, stpmax=_STPMX)
_lnsrlb_iterate = partial(
    dcsrch_iterate, ftol=_LNSRLB_FTOL, gtol=_LNSRLB_GTOL, xtol=_LNSRLB_XTOL, stpmin=0.0, stpmax=_STPMX, lbfgsb=True
)

# Steps of formt's k-sum per loop trip. Rolled, every step pays a GPU launch and
# loop overhead; fully unrolled, the graph and its compile time grow with
# ``maxcor``. At BoozerLS's ``maxcor`` 200, 25 keeps nearly all of the unrolled
# warm speed and the rolled solver's compile time.
_FORMT_UNROLL = 25

# What the pending evaluation is for: the start or a line-search trial; or the end.
_LB_INIT, _LB_SEARCH, _LB_DONE = range(3)


@pytree_dataclass(
    data=(
        "ws",
        "wy",
        "sy",
        "ss",
        "yy",
        "syu",
        "theta",
        "col",
        "iupdat",
        "head",
    )
)
class _LbfgsbMemory:
    """The limited-memory matrices, corrections oldest first (row or index
    ``j < col``; the rest zero), unlike SciPy's ring buffer: ``ws``/``wy``
    hold ``s_j``/``y_j``; ``sy`` and ``ss`` are SciPy's (``s_i . y_j`` below
    the diagonal, ``s_i . s_j`` above, diagonals as matupd stores them);
    ``yy`` (``y_i . y_j``, lower) and ``syu`` (``s_i . y_j``, upper) are
    formk's ``wn1`` blocks for all variables free. ``head`` is the slot of
    SciPy's ring buffer that holds the oldest correction."""

    ws: jax.Array
    wy: jax.Array
    sy: jax.Array
    ss: jax.Array
    yy: jax.Array
    syu: jax.Array
    theta: jax.Array
    col: jax.Array
    iupdat: jax.Array
    head: jax.Array


@pytree_dataclass(
    data=(
        "phase",
        "x_trial",
        "x",
        "f",
        "g",
        "t",
        "r",
        "fold",
        "d",
        "z",
        "stp",
        "gdold",
        "dtd",
        "ifun",
        "line",
        "memory",
        "iterations",
        "sbgnrm",
        "nfev",
        "njev",
        "last_x",
        "message",
    )
)
class _LbfgsbState:
    phase: jax.Array
    x_trial: jax.Array
    x: jax.Array
    f: jax.Array
    g: jax.Array
    t: jax.Array  # x at the start of the line search
    r: jax.Array  # g at the start of the line search
    fold: jax.Array
    d: jax.Array
    z: jax.Array
    stp: jax.Array
    gdold: jax.Array
    dtd: jax.Array
    ifun: jax.Array
    line: DcsrchState
    memory: _LbfgsbMemory
    iterations: jax.Array
    sbgnrm: jax.Array
    nfev: jax.Array
    njev: jax.Array
    last_x: jax.Array
    message: jax.Array


def _empty_memory(x0: jax.Array, maxcor: int) -> _LbfgsbMemory:
    vectors = jnp.zeros((maxcor, x0.shape[0]), x0.dtype)
    matrix = jnp.zeros((maxcor, maxcor), x0.dtype)
    return _LbfgsbMemory(
        ws=vectors,
        wy=vectors,
        sy=matrix,
        ss=matrix,
        yy=matrix,
        syu=matrix,
        theta=jnp.ones((), x0.dtype),
        col=jnp.asarray(0, jnp.int32),
        iupdat=jnp.asarray(0, jnp.int32),
        head=jnp.asarray(0, jnp.int32),
    )


def _projgr(g):
    """projgr without bounds: the infinity norm, NaN if ``g`` has one."""
    return jnp.max(jnp.abs(g))


def _cholesky_failed(matrix):
    """``dpotrf``'s nonzero ``info``: JAX's factor of a matrix that is not
    positive definite is NaN."""
    factor = lax.linalg.cholesky(matrix, symmetrize_input=False)
    return factor, jnp.any(jnp.isnan(factor))


def _matupd(memory: _LbfgsbMemory, s, y, rr, dr, stp, dtd) -> _LbfgsbMemory:
    """matupd, with formk's new ``wn1`` row and column: the correction ``(s,
    y)`` is appended, dropping the oldest when ``maxcor`` are held."""
    maxcor = memory.ws.shape[0]
    full = memory.iupdat >= maxcor
    col = jnp.minimum(memory.iupdat + 1, maxcor)
    last = col - 1

    def shift(rows):
        return jnp.where(full, jnp.roll(rows, -1, axis=0), rows)

    def shift2(matrix):
        return jnp.where(full, jnp.roll(matrix, (-1, -1), axis=(0, 1)), matrix)

    ws = shift(memory.ws).at[last].set(s)
    wy = shift(memory.wy).at[last].set(y)
    index = jnp.arange(maxcor)
    older, held = index < last, index <= last
    return _LbfgsbMemory(
        ws=ws,
        wy=wy,
        sy=shift2(memory.sy).at[last].set(jnp.where(older, wy @ s, 0.0)).at[last, last].set(dr),
        ss=shift2(memory.ss)
        .at[:, last]
        .set(jnp.where(older, ws @ s, 0.0))
        .at[last, last]
        .set(jnp.where(stp == 1.0, dtd, stp * stp * dtd)),
        yy=shift2(memory.yy).at[last].set(jnp.where(held, wy @ y, 0.0)),
        syu=shift2(memory.syu).at[:, last].set(jnp.where(held, ws @ y, 0.0)),
        theta=rr / dr,
        col=col,
        iupdat=memory.iupdat + 1,
        head=jnp.where(full, (memory.head + 1) % maxcor, memory.head),
    )


def _held(memory: _LbfgsbMemory):
    held = jnp.arange(memory.ws.shape[0]) < memory.col
    return held, held[:, None] & held[None, :], jnp.eye(memory.ws.shape[0], dtype=memory.ws.dtype)


def _formt_failed(memory: _LbfgsbMemory):
    """formt's ``info``: ``T = theta S^T S + L D^-1 L^T`` is not positive
    definite (``L`` and ``D`` the strictly lower part and the diagonal of
    ``sy``). Each term ``sy_ik sy_jk / sy_kk``, k < min(i, j), is multiplied,
    divided and added in order of k, as in SciPy's C, so tiny terms underflow
    as there (whether T keeps its last pivot can turn on it). The sum runs up
    to ``_FORMT_UNROLL`` steps per loop trip. Each term's mask (k < min(i, j)
    among held rows) depends on ``col`` through ``held``: a mask from k alone
    is a constant wherever the loop is fully unrolled, and XLA hoists such
    constants out of the solver's loop as ``maxcor`` matrices of scratch.
    (Without bounds T is not used otherwise.)"""
    held, pairs, eye = _held(memory)
    D = jnp.where(held, jnp.diagonal(memory.sy), 1.0)
    L = jnp.where(pairs, jnp.tril(memory.sy, -1), 0.0)
    index = jnp.arange(memory.sy.shape[0])

    def add_term(k, ddum):
        rows = held & (index > k)
        return ddum + jnp.where(rows[:, None] & rows, jnp.outer(L[:, k], L[:, k]) / D[k], 0.0)

    ddum = lax.fori_loop(0, memory.sy.shape[0], add_term, jnp.zeros_like(memory.sy), unroll=_FORMT_UNROLL)
    SS = jnp.triu(memory.ss) + jnp.triu(memory.ss, 1).T
    T = jnp.where(pairs, ddum + memory.theta * SS, eye)
    return _cholesky_failed(T)[1]


def _subspace_point(x, g, memory: _LbfgsbMemory):
    """formk, cmprlb and subsm with every variable free: the minimizer ``z``
    of the limited-memory model, and formk's ``info``.

    formk factors ``K = [[Y^T Y / theta + D, R^T], [R, 0]]``-derived blocks as
    the upper triangle ``[[U11, W], [0, U22]]``, ``U11^T U11 = Y^T Y / theta +
    D``, ``W = U11^-T R^T`` (``R`` the upper triangle of ``S^T Y``) and
    ``U22^T U22 = W^T W``; subsm solves with it for the correction to ``-g``."""
    held, pairs, eye = _held(memory)
    theta = memory.theta
    D = jnp.where(held, jnp.diagonal(memory.sy), 0.0)
    YY = jnp.tril(memory.yy) + jnp.tril(memory.yy, -1).T
    L11, failed11 = _cholesky_failed(jnp.where(pairs, YY / theta + jnp.diag(D), eye))
    W = solve_triangular(L11, jnp.where(pairs, jnp.triu(memory.syu).T, 0.0), lower=True)
    L22, failed22 = _cholesky_failed(jnp.where(pairs, W.T @ W, eye))

    r = -g
    v1 = solve_triangular(L11, memory.wy @ r, lower=True)
    v2 = solve_triangular(L22, theta * (memory.ws @ r) - W.T @ v1, lower=True)
    v2 = solve_triangular(L22, v2, lower=True, trans="T")
    v1 = solve_triangular(L11, -v1 - W @ v2, lower=True, trans="T")
    d = (r + (v1 @ memory.wy) / theta + v2 @ memory.ws) * (1.0 / theta)
    return x + d, failed11 | failed22


def _search_direction(x, g, sbgnrm, memory: _LbfgsbMemory):
    """mainlb from label 222 to lnsrlb's descent test: the subspace minimizer
    when corrections are held, else (or after formk fails or its direction
    does not descend, which empties the memory) the generalized Cauchy point,
    ``x - g / theta`` without bounds. Returns ``z``, ``d = z - x``,
    ``g . d``, the memory and whether the Cauchy direction does not descend
    (mainlb's ``ABNORMAL``)."""
    z, formk_failed = _subspace_point(x, g, memory)
    d = z - x
    gd = jnp.dot(g, d)
    restart = (memory.col > 0) & (formk_failed | (gd >= 0))
    memory = _select(restart, _empty_memory(x, memory.ws.shape[0]), memory)

    cauchy = memory.col == 0
    f1 = -jnp.dot(g, g)
    f2 = -memory.theta * f1
    dtm = -f1 / f2
    dtm = jnp.where(dtm <= 0.0, 0.0, dtm)
    z_cauchy = jnp.where(sbgnrm <= 0.0, x, x + (0.0 + dtm) * -g)
    d_cauchy = z_cauchy - x
    gd_cauchy = jnp.dot(g, d_cauchy)
    z = jnp.where(cauchy, z_cauchy, z)
    d = jnp.where(cauchy, d_cauchy, d)
    gd = jnp.where(cauchy, gd_cauchy, gd)
    return z, d, gd, memory, cauchy & (gd >= 0)


def _lbfgsb_step(s: _LbfgsbState, f_t, g_t, ftol, gtol, maxiter, maxfun, maxls) -> _LbfgsbState:
    """SciPy's L-BFGS-B (setulb and its Python driver) from the value
    ``f_t`` and gradient ``g_t`` at ``s.x_trial`` up to its next evaluation."""
    x_t = s.x_trial
    moved = ~jnp.all(x_t == s.last_x)
    nfev, njev = s.nfev + moved, s.njev + moved
    init = s.phase == _LB_INIT
    searching = s.phase == _LB_SEARCH
    # errclb at the start: a negative factr stops SciPy before it iterates,
    # with f and g as it allocated them (zero).
    factr = ftol / _FLOAT_EPS
    factr_error = init & (factr < 0)
    started = init & ~factr_error

    # lnsrlb after the trial: DCSRCH, then another trial (at most maxls), or
    # the new iterate.
    gd = jnp.dot(g_t, s.d)
    line, step, task = _lnsrlb_iterate(s.line, s.stp, f_t, gd)
    new_x = searching & (task != FG)
    exhausted = searching & (task == FG) & (s.ifun >= maxls)
    trial = searching & (task == FG) & ~exhausted

    # At a new iterate the driver stops at maxiter or maxfun, then mainlb
    # tests the projected gradient and the relative reduction of f.
    iterations = s.iterations + new_x
    sbgnrm = jnp.where(init | new_x, _projgr(g_t), s.sbgnrm)
    stop_maxiter = new_x & (iterations >= maxiter)
    stop_maxfun = new_x & ~stop_maxiter & (nfev > maxfun)
    tested = started | (new_x & ~stop_maxiter & ~stop_maxfun)
    conv_grad = tested & (sbgnrm <= gtol)
    reduction_scale = jnp.fmax(jnp.fmax(jnp.abs(s.fold), jnp.abs(f_t)), 1.0)
    conv_f = tested & ~init & ~conv_grad & ((s.fold - f_t) <= factr * _EPSMCH * reduction_scale)
    update = tested & ~init & ~conv_grad & ~conv_f

    # The BFGS update, skipped without enough curvature; formt failing
    # empties the memory.
    y = g_t - s.r
    unit_step = s.stp == 1.0
    dr = jnp.where(unit_step, gd - s.gdold, (gd - s.gdold) * s.stp)
    ddum = jnp.where(unit_step, -s.gdold, -s.gdold * s.stp)
    correction = jnp.where(unit_step, s.d, s.stp * s.d)
    updated = _matupd(s.memory, correction, y, jnp.linalg.norm(y) ** 2, dr, s.stp, s.dtd)
    do_update = update & ~(dr <= _EPSMCH * ddum)
    memory = _select(do_update, updated, s.memory)
    memory = _select(do_update & _formt_failed(memory), _empty_memory(x_t, memory.ws.shape[0]), memory)

    # A line search that reaches maxls returns to its start: ABNORMAL without
    # corrections, else a restart with an empty memory.
    abnormal = exhausted & (s.memory.col == 0)
    restart = exhausted & (s.memory.col > 0)
    memory = _select(restart, _empty_memory(x_t, memory.ws.shape[0]), memory)
    x = jnp.where(exhausted, s.t, x_t)
    g = jnp.where(exhausted, s.r, jnp.where(factr_error, 0.0, g_t))
    f = jnp.where(exhausted, s.fold, jnp.where(factr_error, 0.0, f_t))

    # A new line search from x along d = z - x: lnsrlb's setup and first trial.
    search = (started & ~conv_grad) | update | restart
    z, d, gd_new, memory_new, not_descent = _search_direction(x, g, sbgnrm, memory)
    memory = _select(search, memory_new, memory)
    dnorm = jnp.linalg.norm(d)
    stp = jnp.where(iterations == 0, jnp.fmin(1.0 / dnorm, _STPMX), 1.0)
    line_new, _ = _lnsrlb_start(stp, f, gd_new)
    abnormal = abnormal | (search & not_descent)
    search = search & ~not_descent

    stp = jnp.where(search, stp, jnp.where(trial, step, s.stp))
    d = jnp.where(search, d, s.d)
    z = jnp.where(search, z, s.z)
    t = jnp.where(search, x, s.t)
    message = jnp.select(
        (factr_error, conv_grad, conv_f, stop_maxiter, stop_maxfun),
        (_ERROR_FACTR, _CONV_GRAD, _CONV_F, _STOP_MAXITER, _STOP_MAXFUN),
        _ABNORMAL,
    ).astype(jnp.int32)
    done = factr_error | conv_grad | conv_f | stop_maxiter | stop_maxfun | abnormal
    return _LbfgsbState(
        phase=jnp.where(done, _LB_DONE, _LB_SEARCH).astype(jnp.int32),
        x_trial=jnp.where(stp == 1.0, z, stp * d + t),
        x=x,
        f=f,
        g=g,
        t=t,
        r=jnp.where(search, g, s.r),
        fold=jnp.where(search, f, s.fold),
        d=d,
        z=z,
        stp=stp,
        gdold=jnp.where(search, gd_new, s.gdold),
        dtd=jnp.where(search, dnorm**2, s.dtd),
        ifun=jnp.where(search, 1, s.ifun + 1),
        line=_select(search, line_new, line),
        memory=memory,
        iterations=iterations,
        sbgnrm=sbgnrm,
        nfev=nfev,
        njev=njev,
        last_x=x_t,
        message=jnp.where(done, message, s.message),
    )


def _lbfgsb_input_error(x0: jax.Array, factr, maxcor: int, maxiter, maxfun) -> QuasiNewtonResult:
    """SciPy's result without variables or corrections: errclb stops it
    before it iterates, after its one evaluation at ``x0``, with ``fun`` and
    ``jac`` zero; a negative ``factr`` overrides the message, as in errclb.
    ``maxcor < 0`` gives C's ``M <= 0``, where SciPy then raises from NumPy."""
    count, one = jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32)
    message = jnp.where(factr < 0, _ERROR_FACTR, _ERROR_M if maxcor <= 0 else _ERROR_N).astype(jnp.int32)
    status = jnp.where((1 > maxfun) | (0 >= maxiter), 1, 2).astype(jnp.int32)
    pairs = jnp.zeros((max(maxcor, 0), x0.shape[0]), x0.dtype)
    return QuasiNewtonResult(
        x0, jnp.zeros((), x0.dtype), jnp.zeros_like(x0), count, one, one, status, message,
        LbfgsbCorrections(pairs, pairs, count),
    )


def minimize_lbfgsb(
    value_and_grad: ValueAndGrad,
    x0: jax.Array,
    ftol,
    gtol,
    maxiter,
    *,
    maxcor: int,
    maxfun=15000,
    maxls: int = 20,
) -> QuasiNewtonResult:
    """``scipy.optimize.minimize(value_and_grad, x0, jac=True,
    method="L-BFGS-B", options={...})`` without bounds.

    As in SciPy, the objective is evaluated at ``x0`` first; then ``maxls <=
    0`` raises, and a negative ``ftol``, ``maxcor <= 0`` or no variables end
    the solve before it iterates, with the C code's ``ERROR`` message. (For
    ``maxcor < 0`` SciPy instead raises a NumPy ``ValueError`` from a reshape
    of its work array, after the C code's verdict.) A NaN coordinate in ``x0``
    makes SciPy count its first evaluation twice, this once, and so ``maxfun``
    can stop the two at different points (see the module docstring).

    Args:
        value_and_grad (Callable): ``x -> (fun(x), grad fun(x))``, traceable.
        x0 (jax.Array): Starting point, shape (n,).
        ftol (jax.Array | float): Stop when ``(f_k - f_k+1) / max(|f_k|, |f_k+1|, 1)``
            is at most ftol (SciPy's ``factr * epsmch``).
        gtol (jax.Array | float): Stop when the gradient's infinity norm is at most gtol.
        maxiter (jax.Array | float): Iteration cap, which may be infinite.
        maxcor (int): Corrections held (static: it sets the memory's shape).
        maxfun (jax.Array | float): Evaluation cap, checked after each iteration.
        maxls (int): Trials per line search (static).

    Returns:
        QuasiNewtonResult: SciPy's fields; ``message`` indexes :data:`LBFGSB_MESSAGES`
            and ``hess_inv`` holds the correction pairs.
    """
    if not maxls > 0 or x0.shape[0] == 0 or maxcor <= 0:
        # SciPy's ScalarFunction evaluates x0 before the maxls check and C's checks.
        value_and_grad(x0)
        if not maxls > 0:
            raise ValueError("maxls must be positive.")
        return _lbfgsb_input_error(x0, ftol / _FLOAT_EPS, maxcor, maxiter, maxfun)
    zero = jnp.zeros((), x0.dtype)
    count = jnp.asarray(0, jnp.int32)
    line, _ = _lnsrlb_start(zero, zero, zero)
    state = _LbfgsbState(
        phase=jnp.asarray(_LB_INIT, jnp.int32),
        x_trial=x0,
        x=x0,
        f=zero,
        g=jnp.zeros_like(x0),
        t=x0,
        r=jnp.zeros_like(x0),
        fold=zero,
        d=jnp.zeros_like(x0),
        z=x0,
        stp=zero,
        gdold=zero,
        dtd=zero,
        ifun=count,
        line=line,
        memory=_empty_memory(x0, maxcor),
        iterations=count,
        sbgnrm=zero,
        nfev=count,
        njev=count,
        last_x=jnp.full_like(x0, jnp.nan),
        message=count,
    )

    def body(s):
        return _lbfgsb_step(s, *value_and_grad(s.x_trial), ftol, gtol, maxiter, maxfun, maxls)

    s = lax.while_loop(lambda s: s.phase != _LB_DONE, body, state)
    converged = (s.message == _CONV_GRAD) | (s.message == _CONV_F)
    status = jnp.where(converged, 0, jnp.where((s.nfev > maxfun) | (s.iterations >= maxiter), 1, 2))
    memory = s.memory
    # SciPy's ring buffer holds correction j in slot (head + j) mod maxcor.
    corrections = LbfgsbCorrections(
        jnp.roll(memory.ws, memory.head, axis=0), jnp.roll(memory.wy, memory.head, axis=0), memory.col
    )
    return QuasiNewtonResult(
        s.x, s.f, s.g, s.iterations, s.nfev, s.njev, status.astype(jnp.int32), s.message, corrections
    )
