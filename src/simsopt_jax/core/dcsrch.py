# This module ports the DCSRCH line search of SciPy 1.18
# (scipy/optimize/_dcsrch.py and scipy/optimize/__lbfgsb.c) to JAX. They are
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

"""MINPACK-2's DCSRCH line search (More and Thuente), as SciPy 1.18 runs it, in JAX.

SciPy carries two translations of DCSRCH and its DCSTEP: the Python one of
``scipy/optimize/_dcsrch.py``, behind ``line_search_wolfe1`` (BFGS's first
line search), and the C one inside ``scipy/optimize/__lbfgsb.c`` (L-BFGS-B's
``lnsrlb``). They decide alike on finite values and differ only at a tie in
DCSTEP's first case and where a NaN or an infinite derivative reaches a sign,
``min``, ``max`` or a clip: Python's builtins and ``np.clip`` against C's
``fmin``/``fmax``. ``lbfgsb`` selects the C translation.

:func:`dcsrch_start` is the call with ``task = 'START'`` and
:func:`dcsrch_iterate` a later call with ``f`` and ``g`` at ``stp``. Both are
pure functions of :class:`DcsrchState`, without branches, for use inside a
compiled loop; ``ftol``, ``gtol``, ``xtol``, ``stpmin`` and ``stpmax`` are
the caller's SciPy constants.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

from simsopt_jax.pytree import pytree_dataclass

__all__ = [
    "CONVERGENCE",
    "ERROR",
    "FG",
    "WARNING",
    "DcsrchState",
    "dcsrch_iterate",
    "dcsrch_start",
]

# ``task``: evaluate ``f`` and ``g`` at the returned ``stp``, or the search ended.
FG = 0
CONVERGENCE = 1
WARNING = 2
ERROR = 3

_P5 = 0.5
_P66 = 0.66
_XTRAPL = 1.1
_XTRAPU = 4.0


@pytree_dataclass(
    data=(
        "brackt",
        "stage",
        "ginit",
        "gtest",
        "gx",
        "gy",
        "finit",
        "fx",
        "fy",
        "stx",
        "sty",
        "stmin",
        "stmax",
        "width",
        "width1",
    )
)
class DcsrchState:
    """DCSRCH's saved variables (SciPy's ``isave``/``dsave``) between calls.

    Args:
        brackt (jax.Array): Scalar bool, a minimizer is bracketed by ``stx`` and ``sty``.
        stage (jax.Array): Scalar int32, 1 until a step meets the sufficient decrease
            test with a nonnegative derivative, then 2.
        ginit (jax.Array): Scalar derivative at ``stp = 0``.
        gtest (jax.Array): Scalar ``ftol * ginit``.
        gx (jax.Array): Scalar derivative at ``stx``.
        gy (jax.Array): Scalar derivative at ``sty``.
        finit (jax.Array): Scalar function value at ``stp = 0``.
        fx (jax.Array): Scalar function value at ``stx``.
        fy (jax.Array): Scalar function value at ``sty``.
        stx (jax.Array): Scalar step with the least function value so far.
        sty (jax.Array): Scalar other endpoint of the interval of uncertainty.
        stmin (jax.Array): Scalar lower end of the interval for the next step.
        stmax (jax.Array): Scalar upper end of the interval for the next step.
        width (jax.Array): Scalar width of the interval of uncertainty.
        width1 (jax.Array): Scalar width before the last bisection test.
    """

    brackt: jax.Array
    stage: jax.Array
    ginit: jax.Array
    gtest: jax.Array
    gx: jax.Array
    gy: jax.Array
    finit: jax.Array
    fx: jax.Array
    fy: jax.Array
    stx: jax.Array
    sty: jax.Array
    stmin: jax.Array
    stmax: jax.Array
    width: jax.Array
    width1: jax.Array


def _select(predicate, on_true, on_false):
    """``jnp.where(predicate, ...)`` leaf by leaf over two pytrees of one structure."""
    return jax.tree.map(lambda a, b: jnp.where(predicate, a, b), on_true, on_false)


def _py_min(a, b):
    """Python's ``min(a, b)``: ``b`` only if ``b < a`` (so a NaN ``b`` gives ``a``)."""
    return jnp.where(b < a, b, a)


def _py_max(a, b):
    """Python's ``max(a, b)``: ``b`` only if ``b > a``."""
    return jnp.where(b > a, b, a)


def dcsrch_start(stp, f, g, *, ftol: float, stpmin: float, stpmax: float):
    """DCSRCH with ``task = 'START'``: ``f`` and ``g`` are the value and the
    derivative at ``stp = 0`` and ``stp`` is the first trial step.

    Args:
        stp (jax.Array): Scalar first trial step.
        f (jax.Array): Scalar function value at zero step.
        g (jax.Array): Scalar directional derivative at zero step.
        ftol (float): Sufficient decrease tolerance.
        stpmin (float): Lower bound of the step.
        stpmax (float): Upper bound of the step.

    Returns:
        tuple[DcsrchState, jax.Array]: The saved state and ``task``: ``FG`` (evaluate
            at ``stp``) or ``ERROR`` (``stp`` outside its bounds or ``g >= 0``).
    """
    task = jnp.where((stp < stpmin) | (stp > stpmax) | (g >= 0), ERROR, FG)
    zero = jnp.zeros_like(f)
    width = jnp.full_like(f, stpmax - stpmin)
    state = DcsrchState(
        brackt=jnp.asarray(False),
        stage=jnp.asarray(1, jnp.int32),
        ginit=g,
        gtest=ftol * g,
        gx=g,
        gy=g,
        finit=f,
        fx=f,
        fy=f,
        stx=zero,
        sty=zero,
        stmin=zero,
        stmax=stp + _XTRAPU * stp,
        width=width,
        width1=width / _P5,
    )
    return state, task


def _scale(theta, da, db, lbfgsb: bool):
    """DCSTEP's ``s``, the largest of ``|theta|``, ``|da|`` and ``|db|``."""
    if lbfgsb:
        return jnp.fmax(jnp.abs(theta), jnp.fmax(jnp.abs(da), jnp.abs(db)))
    return _py_max(_py_max(jnp.abs(theta), jnp.abs(da)), jnp.abs(db))


def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax, lbfgsb: bool):
    """DCSTEP: the safeguarded trial step and the updated interval, each of
    its four cases computed and the one that applies selected."""
    minimum, maximum = (jnp.fmin, jnp.fmax) if lbfgsb else (_py_min, _py_max)
    sgnd = dp * (dx / jnp.abs(dx)) if lbfgsb else jnp.sign(dp) * jnp.sign(dx)

    # First case: a higher function value; the minimum is bracketed.
    theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp
    s = _scale(theta, dx, dp, lbfgsb)
    gamma = s * jnp.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
    gamma = jnp.where(stp < stx, -gamma, gamma)
    r = ((gamma - dx) + theta) / (((gamma - dx) + gamma) + dp)
    stpc = stx + r * (stp - stx)
    stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx)
    distance_c, distance_q = jnp.abs(stpc - stx), jnp.abs(stpq - stx)
    cubic_closer = distance_c < distance_q if lbfgsb else distance_c <= distance_q
    stpf1 = jnp.where(cubic_closer, stpc, stpc + (stpq - stpc) / 2.0)

    # Second case: a lower value and derivatives of opposite sign; bracketed.
    gamma = s * jnp.sqrt((theta / s) ** 2 - (dx / s) * (dp / s))
    gamma = jnp.where(stp > stx, -gamma, gamma)
    r = ((gamma - dp) + theta) / (((gamma - dp) + gamma) + dx)
    stpc = stp + r * (stx - stp)
    stpq = stp + (dp / (dp - dx)) * (stx - stp)
    stpf2 = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)

    # Third case: a lower value, same-sign derivatives, decreasing in magnitude.
    radicand = (theta / s) ** 2 - (dx / s) * (dp / s)
    gamma = s * jnp.sqrt(jnp.where(radicand > 0, radicand, 0.0))
    gamma = jnp.where(stp > stx, -gamma, gamma)
    r = ((gamma - dp) + theta) / ((gamma + (dx - dp)) + gamma)
    stpc = jnp.where((r < 0.0) & (gamma != 0.0), stp + r * (stx - stp), jnp.where(stp > stx, stpmax, stpmin))
    stpq = stp + (dp / (dp - dx)) * (stx - stp)
    toward_sty = stp + 0.66 * (sty - stp)
    bracketed = jnp.where(jnp.abs(stpc - stp) < jnp.abs(stpq - stp), stpc, stpq)
    bracketed = jnp.where(stp > stx, minimum(toward_sty, bracketed), maximum(toward_sty, bracketed))
    extrapolated = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
    if lbfgsb:
        extrapolated = jnp.fmax(stpmin, jnp.fmin(stpmax, extrapolated))
    else:
        extrapolated = jnp.clip(extrapolated, stpmin, stpmax)
    stpf3 = jnp.where(brackt, bracketed, extrapolated)

    # Fourth case: a lower value, same-sign derivatives not decreasing in magnitude.
    theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp
    s = _scale(theta, dy, dp, lbfgsb)
    gamma = s * jnp.sqrt((theta / s) ** 2 - (dy / s) * (dp / s))
    gamma = jnp.where(stp > sty, -gamma, gamma)
    r = ((gamma - dp) + theta) / (((gamma - dp) + gamma) + dy)
    stpf4 = jnp.where(brackt, stp + r * (sty - stp), jnp.where(stp > stx, stpmax, stpmin))

    higher = fp > fx
    opposite = ~higher & (sgnd < 0.0)
    decreasing = ~higher & ~opposite & (jnp.abs(dp) < jnp.abs(dx))
    stpf = jnp.where(higher, stpf1, jnp.where(opposite, stpf2, jnp.where(decreasing, stpf3, stpf4)))

    # The interval: a higher value replaces ``sty``; otherwise ``stp`` becomes
    # ``stx``, and ``stx`` becomes ``sty`` when the derivatives change sign.
    stx_to_sty = ~higher & (sgnd < 0.0)
    sty = jnp.where(higher, stp, jnp.where(stx_to_sty, stx, sty))
    fy = jnp.where(higher, fp, jnp.where(stx_to_sty, fx, fy))
    dy = jnp.where(higher, dp, jnp.where(stx_to_sty, dx, dy))
    stx = jnp.where(higher, stx, stp)
    fx = jnp.where(higher, fx, fp)
    dx = jnp.where(higher, dx, dp)
    return stx, fx, dx, sty, fy, dy, stpf, brackt | higher | opposite


def dcsrch_iterate(
    state: DcsrchState,
    stp,
    f,
    g,
    *,
    ftol: float,
    gtol: float,
    xtol: float,
    stpmin: float,
    stpmax: float,
    lbfgsb: bool,
):
    """DCSRCH after an evaluation: ``f`` and ``g`` at ``stp``.

    Args:
        state (DcsrchState): Saved state of the previous call.
        stp (jax.Array): Scalar step just evaluated.
        f (jax.Array): Scalar function value at ``stp``.
        g (jax.Array): Scalar directional derivative at ``stp``.
        ftol (float): Sufficient decrease tolerance.
        gtol (float): Curvature tolerance.
        xtol (float): Relative interval-width tolerance.
        stpmin (float): Lower bound of the step.
        stpmax (float): Upper bound of the step.
        lbfgsb (bool): Use the C translation of SciPy's L-BFGS-B instead of the
            Python one of ``line_search_wolfe1``.

    Returns:
        tuple[DcsrchState, jax.Array, jax.Array]: The saved state, the step (the
            next trial if ``task`` is ``FG``, else ``stp``) and ``task``:
            ``FG``, ``CONVERGENCE`` or ``WARNING``.
    """
    ftest = state.finit + stp * state.gtest
    stage = jnp.where((state.stage == 1) & (f <= ftest) & (g >= 0), 2, state.stage)
    brackt = state.brackt
    warning = (
        (brackt & ((stp <= state.stmin) | (stp >= state.stmax)))
        | (brackt & (state.stmax - state.stmin <= xtol * state.stmax))
        | ((stp == stpmax) & (f <= ftest) & (g <= state.gtest))
        | ((stp == stpmin) & ((f > ftest) | (g >= state.gtest)))
    )
    converged = (f <= ftest) & (jnp.abs(g) <= gtol * (-state.ginit))
    task = jnp.where(converged, CONVERGENCE, jnp.where(warning, WARNING, FG))

    # In the first stage the step comes from the modified function
    # ``psi(stp) = f(stp) - f(0) - ftol stp f'(0)`` (More and Thuente).
    modified = (stage == 1) & (f <= state.fx) & (f > ftest)
    gtest = jnp.where(modified, state.gtest, 0.0)
    stx, fx, gx, sty, fy, gy, new_stp, brackt = _dcstep(
        state.stx,
        jnp.where(modified, state.fx - state.stx * state.gtest, state.fx),
        jnp.where(modified, state.gx - state.gtest, state.gx),
        state.sty,
        jnp.where(modified, state.fy - state.sty * state.gtest, state.fy),
        jnp.where(modified, state.gy - state.gtest, state.gy),
        stp,
        jnp.where(modified, f - stp * state.gtest, f),
        jnp.where(modified, g - state.gtest, g),
        brackt,
        state.stmin,
        state.stmax,
        lbfgsb,
    )
    fx = jnp.where(modified, fx + stx * gtest, fx)
    fy = jnp.where(modified, fy + sty * gtest, fy)
    gx = jnp.where(modified, gx + gtest, gx)
    gy = jnp.where(modified, gy + gtest, gy)

    # Bisect when the interval of uncertainty does not shrink enough.
    gap = jnp.abs(sty - stx)
    new_stp = jnp.where(brackt & (gap >= _P66 * state.width1), stx + _P5 * (sty - stx), new_stp)
    width1 = jnp.where(brackt, state.width, state.width1)
    width = jnp.where(brackt, gap, state.width)
    minimum, maximum = (jnp.fmin, jnp.fmax) if lbfgsb else (_py_min, _py_max)
    stmin = jnp.where(brackt, minimum(stx, sty), new_stp + _XTRAPL * (new_stp - stx))
    stmax = jnp.where(brackt, maximum(stx, sty), new_stp + _XTRAPU * (new_stp - stx))
    if lbfgsb:
        new_stp = jnp.fmin(jnp.fmax(new_stp, stpmin), stpmax)
    else:
        new_stp = jnp.clip(new_stp, stpmin, stpmax)
    stalled = brackt & ((new_stp <= stmin) | (new_stp >= stmax) | (stmax - stmin <= xtol * stmax))
    new_stp = jnp.where(stalled, stx, new_stp)

    searching = task == FG
    updated = replace(
        state,
        brackt=brackt,
        stage=stage,
        gx=gx,
        gy=gy,
        fx=fx,
        fy=fy,
        stx=stx,
        sty=sty,
        stmin=stmin,
        stmax=stmax,
        width=width,
        width1=width1,
    )
    # A finished search keeps its saved variables but the stage, as SciPy's do.
    finished = replace(state, stage=stage)
    state = _select(searching, updated, finished)
    return state, jnp.where(searching, new_stp, stp), task
