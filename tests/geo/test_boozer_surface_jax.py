"""JAX device solvers against native ``BoozerSurface`` and SciPy."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import boozer_solvers
    from simsopt_jax.core.quasi_newton import BFGS_MESSAGES, LBFGSB_MESSAGES, _bfgs_inverse_hessian_update, _cubicmin
    from simsopt_jax.core.quasi_newton import _quadmin, _zoom_trial, minimize_bfgs, minimize_lbfgsb

from unittest_jax_support import assert_matches_native, host_array
from unittest_jax_support import parity_default_device, parity_rng

from collections.abc import Callable
from functools import partial
import inspect
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult, minimize
from scipy.optimize import _linesearch as scipy_linesearch


# The device quasi-Newton solvers against SciPy (TestQuasiNewtonJax): each solver runs beside
# scipy.optimize.minimize(jac=True), SciPy evaluating the same JAX objective on the CPU, on every lane.

# Records are compared up to k = 10, numbers up to the k listed (see test_early_iterates_match_scipy).
_EARLY_ITERATIONS = 10
_EARLY_NUMERIC_ITERATIONS = {("L-BFGS-B", "quadratic"): 9}
_EARLY_RTOL = 1e-12
# gtol = 1e-6 leaves two points with unit smallest curvature this far apart (measured worst 3.1e-7).
_FINAL_RTOL = 1e-5
_FAILURE_RTOL = 1e-12
_TINY_RTOL = 1e-14
# Forty or more updates, each rounded differently (worst 1.1e-9, BFGS on the Rosenbrock function).
_CONVERGED_HESS_INV_RTOL = 1e-7
# Worst early gaps (JAX 0.12.0.1, SciPy 1.18.1): iterates 4.3e-13; fun and jac, which magnify them, 1.2e-12.
_EARLY_FUN_JAC_RTOL = 10 * _EARLY_RTOL
# Near zero at a converged solution, so compared absolutely (worst 3.3e-24 and 4.5e-14).
_CONVERGED_FUN_ATOL = 1e-14
_CONVERGED_JAC_ATOL = 1e-12
_LBFGSB_PGTOL = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
_LBFGSB_FTOL = "CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH"
# Converged walks that round-off decides, each with the messages besides SciPy's
# that round-off also gives SciPy itself; their counts are not compared.
_ROUND_OFF_WALKS = {
    # 180 to 200 iterations; round-off also picks the convergence test, for SciPy too
    # (SciPy 1.18 from x0 moved by an ulp, on a NumPy objective: 2 FTOL stops in 40 draws).
    ("L-BFGS-B", "quadratic"): frozenset({_LBFGSB_PGTOL, _LBFGSB_FTOL}),
    # Its count changes with how XLA compiles the loop (JAX 0.12.0.1: 114 CPU, 96 GPU; SciPy 109).
    ("L-BFGS-B", "ill-conditioned"): frozenset(),
}
_GTOL = 1e-6
_LANES = ("cpu", "gpu")


def _quadratic(x, params):
    matrix, center = params
    r = x - center
    return 0.5 * r @ (matrix @ r)


def _rosenbrock(x, params):
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2)


def _ill_conditioned(x, params):
    return 0.5 * jnp.sum(params * (x - 1.0) ** 2) + 0.25 * jnp.sum(x**4)


def _contradicting(x, params):
    return 0.5 * jnp.sum(x**2)


def _nan_wall(x, params):
    return jnp.where(x[0] > 1.5, jnp.nan, jnp.sum((x - 2.0) ** 2))


def _inf_wall(x, params):
    valley = jnp.sum((x - 0.5) ** 2) + 3.0 * jnp.sum((x - 0.5) ** 4)
    return jnp.where(x[0] > 0.6, jnp.inf, valley)


def _linear(x, params):
    return jnp.sum(x)


def _steep(x, params):
    return 1e110 * (0.5 * jnp.sum(jnp.arange(1.0, x.shape[0] + 1) * x**2) + jnp.sum(x**4))


def _unavailable(x, params):
    raise RuntimeError("objective unavailable")


def _tiny_scale(x, params):
    u = x * 1e100
    return jnp.sum(1e-200 * (u**2 / 2 + 0.1 * jnp.sin(u)))


def _badly_scaled(x, params):
    return 5e17 * jnp.sum(x**2)


_OBJECTIVES = {
    "quadratic": _quadratic, "rosenbrock": _rosenbrock, "ill-conditioned": _ill_conditioned,
    "contradicting": _contradicting, "nan-wall": _nan_wall, "inf-wall": _inf_wall, "linear": _linear, "steep": _steep,
    "tiny-scale": _tiny_scale, "badly-scaled": _badly_scaled, "unavailable": _unavailable,
}


def _value_and_grad(name, x, params):
    """The objective's value and gradient; ``contradicting`` returns the gradient of its negative."""
    value, grad = jax.value_and_grad(_OBJECTIVES[name])(x, params)
    return (value, -grad) if name == "contradicting" else (value, grad)


def _problems():
    """``name -> (x0, params)`` for the comparison problems."""
    rng = parity_rng(11)
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    matrix = (q * np.logspace(0, 3, 20)) @ q.T
    return {
        "quadratic": (rng.uniform(-1, 1, 20), (0.5 * (matrix + matrix.T), rng.uniform(-1, 1, 20))),
        "rosenbrock": (rng.uniform(-2, 2, 6), np.zeros(0)),
        "ill-conditioned": (rng.uniform(-3, 3, 8), np.logspace(0, 6, 8)),
    }


if JAX_IMPORT_ERROR is None:
    # The compiled solvers exist only where the JAX runtime does, so discovery
    # can import this module and skip it without JAX.

    @partial(jit, static_argnames=("name",))
    def _device_bfgs(x0, params, gtol, maxiter, name):
        return minimize_bfgs(partial(_value_and_grad, name, params=params), x0, gtol, maxiter)

    @partial(jit, static_argnames=("name", "maxcor", "maxls"))
    def _device_lbfgsb(x0, params, ftol, gtol, maxiter, maxls, name, maxcor):
        return minimize_lbfgsb(
            partial(_value_and_grad, name, params=params), x0, ftol, gtol, maxiter, maxcor=maxcor, maxls=maxls
        )


def _scipy(name, x0, params, method, options) -> OptimizeResult:
    """SciPy on the same JAX objective, evaluated on the CPU."""
    with parity_default_device("cpu"):
        params = jax.tree.map(jnp.asarray, params)

        def fun(x):
            value, grad = _value_and_grad(name, jnp.asarray(x), params)
            return host_array(value), host_array(grad)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return minimize(fun, x0, jac=True, method=method, options=options)


def _device(name, x0, params, method, options, lane):
    """The device solver on ``lane`` with SciPy's options; returns the host result and SciPy's message for it."""
    with parity_default_device(lane):
        x0, params = jnp.asarray(x0), jax.tree.map(jnp.asarray, params)
        if method == "BFGS":
            result = _device_bfgs(x0, params, options["gtol"], float(options["maxiter"]), name)
            messages = BFGS_MESSAGES
        else:
            result = _device_lbfgsb(x0, params, options["ftol"], options["gtol"], float(options["maxiter"]),
                                    options.get("maxls", 20), name, options["maxcor"])
            messages = LBFGSB_MESSAGES
        result = jax.tree.map(host_array, result)
    return result, messages[int(result.message)]


def _options(method, maxiter, **extra):
    if method == "BFGS":
        return {"gtol": _GTOL, "maxiter": maxiter, **extra}
    return {"gtol": _GTOL, "ftol": 1e-14, "maxiter": maxiter, "maxcor": 3, **extra}


# The SciPy version these comparisons (records, messages and the private
# _linesearch helpers) hold for, the one the port is tested against. Another SciPy
# is not their reference. Where JAX is unavailable, JaxTestCase's reason stands.
_SCIPY_REFERENCE = (1, 18)
_SCIPY_VERSION = tuple(int(part) for part in scipy.__version__.split(".")[:2])


@unittest.skipUnless(
    JAX_IMPORT_ERROR is not None or _SCIPY_VERSION == _SCIPY_REFERENCE,
    f"compares with SciPy {'.'.join(map(str, _SCIPY_REFERENCE))}, not {scipy.__version__}",
)
class TestQuasiNewtonJax(JaxTestCase):
    def _assert_scipy_record(self, expected: OptimizeResult, result, message: str, name: str):
        """Status, message, nit, nfev and njev are SciPy's."""
        self.assertEqual(
            (int(result.status), message, int(result.iterations), int(result.nfev), int(result.njev)),
            (expected.status, expected.message, expected.nit, expected.nfev, expected.njev),
            f"{name}: (status, message, nit, nfev, njev) differ from SciPy's",
        )

    def _assert_scipy_hess_inv(self, expected: OptimizeResult, result, name: str, rtol: float):
        """``hess_inv`` is SciPy's: BFGS's matrix, or L-BFGS-B's correction pairs in SciPy's (ring-buffer) order."""
        if isinstance(expected.hess_inv, LbfgsInvHessProduct):
            corrections = result.hess_inv
            count = int(corrections.count)
            self.assertEqual(count, expected.hess_inv.n_corrs, f"{name}: number of corrections")
            assert_matches_native(corrections.s[:count], expected.hess_inv.sk, f"{name} hess_inv sk", rtol)
            assert_matches_native(corrections.y[:count], expected.hess_inv.yk, f"{name} hess_inv yk", rtol)
        else:
            assert_matches_native(result.hess_inv, expected.hess_inv, f"{name} hess_inv", rtol)

    def _assert_ends_as_scipy(self, cases, params=np.zeros(0)):
        """Each ``case: (name, x0, method, options, status)`` ends with SciPy's status, record and ``x``, every lane."""
        for case, (name, x0, method, options, status) in cases.items():
            expected = _scipy(name, x0, params, method, options)
            for lane in _LANES:
                with self.subTest(lane=lane, case=case), self.case():
                    self.assertEqual(expected.status, status, f"{case}: SciPy's status changed")
                    result, message = _device(name, x0, params, method, options, lane)
                    self._assert_scipy_record(expected, result, message, case)
                    assert_matches_native(result.x, expected.x, f"{case} x", _FAILURE_RTOL)

    def test_early_iterates_match_scipy(self):
        """Capped at k = 1, ..., 10 iterations, both solvers end with SciPy's record and, where defined, its iterate."""
        # SciPy's own runs from x0 moved by one ulp per entry differ from its run from x0 by at most 8.6e-14 (yk) and
        # 2.6e-12 (jac) after nine L-BFGS-B iterations on the quadratic, but by 2.0e-12 and 1.1e-11 after ten: ten
        # iterations there magnify round-off about 1e4-fold, so a run that rounds differently, as compiled XLA does,
        # can miss the bounds with no error in its algorithm (JAX 0.12.0.1 GPU lane: yk 9.2e-13, jac 5.2e-12). The
        # numbers are compared up to nine there (_EARLY_NUMERIC_ITERATIONS), the record up to ten.
        for method in ("BFGS", "L-BFGS-B"):
            for name, (x0, params) in _problems().items():
                expected = [
                    _scipy(name, x0, params, method, _options(method, k)) for k in range(1, _EARLY_ITERATIONS + 1)
                ]
                for lane in _LANES:
                    with self.subTest(lane=lane, method=method, name=name), self.case():
                        self._case_early_iterates_match_scipy(lane, method, name, expected)

    def _case_early_iterates_match_scipy(self, lane, method, name, scipy_results):
        x0, params = _problems()[name]
        numeric_iterations = _EARLY_NUMERIC_ITERATIONS.get((method, name), _EARLY_ITERATIONS)
        for k, expected in enumerate(scipy_results, start=1):
            options = _options(method, k)
            self.assertEqual(expected.status, 1, f"{name}: SciPy converged before k={k}")
            result, message = _device(name, x0, params, method, options, lane)
            self._assert_scipy_record(expected, result, message, f"{method} {name} k={k}")
            if k > numeric_iterations:
                continue
            assert_matches_native(result.x, expected.x, f"{method} {name} x at k={k}", _EARLY_RTOL)
            rtol = _EARLY_FUN_JAC_RTOL
            assert_matches_native(result.fun, expected.fun, f"{method} {name} fun at k={k}", rtol, abs(expected.fun))
            assert_matches_native(result.jac, expected.jac, f"{method} {name} jac at k={k}", rtol)
            self._assert_scipy_hess_inv(expected, result, f"{method} {name} k={k}", _EARLY_RTOL)

    def test_converged_solves_match_scipy(self):
        """Run to convergence, both solvers end as SciPy does; ``_ROUND_OFF_WALKS`` end at its solution to ``gtol``."""
        # Status, message, counts, x, fun, jac (to round-off of their scale at x0) and hess_inv; the round-off walks
        # keep SciPy's status and its message or one that round-off gives SciPy from x0 moved by an ulp.
        for method in ("BFGS", "L-BFGS-B"):
            options = _options(method, 5000) if method == "BFGS" else _options(method, 5000, maxcor=10)
            for name, (x0, params) in _problems().items():
                expected = _scipy(name, x0, params, method, options)
                for lane in _LANES:
                    with self.subTest(lane=lane, method=method, name=name), self.case():
                        self._case_converged_solves_match_scipy(lane, method, name, options, expected)

    def _case_converged_solves_match_scipy(self, lane, method, name, options, expected):
        x0, params = _problems()[name]
        self.assertEqual(expected.status, 0, f"{name}: SciPy did not converge")
        result, message = _device(name, x0, params, method, options, lane)
        if (method, name) in _ROUND_OFF_WALKS:  # Round-off decides these counts; see _ROUND_OFF_WALKS.
            self.assertEqual(int(result.status), expected.status, f"{method} {name}: status differs from SciPy's")
            self.assertIn(message, {expected.message} | _ROUND_OFF_WALKS[(method, name)],
                          f"{method} {name}: neither SciPy's message nor one its round-off gives")
            assert_matches_native(result.x, expected.x, f"{method} {name} solution", _FINAL_RTOL)
            return
        self._assert_scipy_record(expected, result, message, f"{method} {name}")
        assert_matches_native(result.x, expected.x, f"{method} {name} solution", _EARLY_RTOL)
        assert_matches_native(result.fun, expected.fun, f"{method} {name} fun", _CONVERGED_FUN_ATOL, 1.0)
        assert_matches_native(result.jac, expected.jac, f"{method} {name} jac", _CONVERGED_JAC_ATOL, 1.0)
        self._assert_scipy_hess_inv(expected, result, f"{method} {name}", _CONVERGED_HESS_INV_RTOL)

    def test_failed_line_searches_end_as_in_scipy(self):
        """A contradicting gradient: BFGS loses precision (status 2), L-BFGS-B is ABNORMAL."""
        self._assert_ends_as_scipy({
            "BFGS": ("contradicting", np.ones(4), "BFGS", _options("BFGS", 100), 2),
            "L-BFGS-B": ("contradicting", np.ones(4), "L-BFGS-B", _options("L-BFGS-B", 100), 2),
            # One trial per line search: a search that needs a second restarts
            # from an empty memory, and fails without corrections.
            "L-BFGS-B restart": (
                "rosenbrock", _problems()["rosenbrock"][0], "L-BFGS-B", _options("L-BFGS-B", 100, maxls=1), 2
            ),
        })

    def test_bfgs_line_search_fallbacks_take_scipys_paths(self):
        """DCSRCH failing (a NaN step past an infinite wall, below ``amin``) or ``wolfe2`` exhausted: SciPy's."""
        self._assert_ends_as_scipy({
            # Converges after zooming in scalar_search_wolfe2.
            "infinite wall": ("inf-wall", np.array([0.0, 0.2]), "BFGS", _options("BFGS", 100), 0),
            # Each iteration ends with the tenth doubling, its gradient requested by BFGS.
            "unbounded": ("linear", np.array([0.5, -1.0]), "BFGS", _options("BFGS", 5), 1),
            # One iteration: the inverse Hessian is still the identity.
            "steep": ("steep", np.array([0.3, -0.2, 0.25]), "BFGS", _options("BFGS", 1), 1),
        })

    def test_bfgs_update_keeps_badly_scaled_inverse_hessian(self):
        """On f = 5e17 x^2 SciPy's factored update keeps H = 1 / (2 * 5e17), and so does the port."""
        # The first update's rho s s^T (1e-18) lies below the round-off of terms the size of H = 1. The port keeps H
        # after that first update (with SciPy's record) and to convergence (with SciPy's status); an expanded rank-two
        # update leaves H wrong after the first (on the CPU negative, ending with status 2; on the GPU 200 times too
        # large, which the next update repairs). The count to convergence is not compared: the second step lands
        # within round-off of x = 0, exactly on it on the GPU, which then stops one update before SciPy. H still is:
        # in one variable each update sets H to s / y = 1 / (2 * 5e17), to round-off, whatever H was.
        x0 = np.ones(1)
        for maxiter in (1, 100):
            options = _options("BFGS", maxiter)
            expected = _scipy("badly-scaled", x0, np.zeros(0), "BFGS", options)
            for lane in _LANES:
                with self.subTest(lane=lane, maxiter=maxiter), self.case():
                    self.assertEqual(expected.status, 1 if maxiter == 1 else 0, "SciPy's status changed")
                    result, message = _device("badly-scaled", x0, np.zeros(0), "BFGS", options, lane)
                    if maxiter == 1:
                        self._assert_scipy_record(expected, result, message, "badly scaled, one update")
                    else:
                        self.assertEqual((int(result.status), message), (expected.status, expected.message))
                    self._assert_scipy_hess_inv(expected, result, f"badly scaled, maxiter={maxiter}", _FAILURE_RTOL)

    def test_bfgs_update_is_scipys_where_s_s_underflows(self):
        """Where ``s s^T`` and ``(H y) s^T`` underflow (s ~ 1e-165) the update is SciPy's, eager and compiled."""
        # y ~ 1e27 s, so y^T s ~ 1e-303, SciPy's s y^T and the updated H ~ 1e-27 are normal; forming s s^T or
        # (H y) s^T before scaling by rho loses rank-one terms (errors of 4 and 11 times the largest entry). Iterates
        # reach such steps only along round-off-driven walks (BFGS on (a x)^2 / 2 after about a dozen iterations), so
        # the update is compared directly with SciPy's expression in _minimize_bfgs, as for a step of unit size.
        updates = {"eager": _bfgs_inverse_hessian_update, "compiled": jit(_bfgs_inverse_hessian_update)}
        curvature = np.array([1.0e27, 2.5e27])
        for case, scale in (("underflowing step", 1e-165), ("unit step", 1.0)):
            sk = scale * np.array([2.0, -1.3])
            yk = curvature * sk + scale * np.array([0.3e26, -0.1e26])
            hess_inv = np.array([[0.8e-27, 0.1e-27], [0.1e-27, 0.4e-27]])
            if scale < 1.0:
                self.assertTrue(np.all(np.outer(sk, sk) == 0.0) and np.dot(yk, sk) > 1e-307, case)
            rhok = 1.0 / np.dot(yk, sk)
            identity = np.eye(2)
            A1 = identity - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
            A2 = identity - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
            expected = np.dot(A1, np.dot(hess_inv, A2)) + (rhok * sk[:, np.newaxis] * sk[np.newaxis, :])
            for lane in _LANES:
                for path, update in updates.items():
                    with self.subTest(case=case, lane=lane, path=path), self.case(), parity_default_device(lane):
                        updated = host_array(update(jnp.asarray(hess_inv), jnp.asarray(sk), jnp.asarray(yk)))
                        assert_matches_native(updated, expected, f"{case} H, {path}", _FAILURE_RTOL)

    def test_zoom_interpolants_match_scipy(self):
        """Cubic and quadratic minimizers and ``_zoom``'s first two trials are SciPy's (``None`` where SciPy raises)."""
        # SciPy's raise on coincident points, a negative radical, overflow and infinities; the trials are bisection or
        # quadratic, then cubic.
        rng = parity_rng(12)
        draws = [tuple(rng.uniform(-2.0, 2.0, 7)) for _ in range(200)]
        a, fa, fpa, b, fb, c, fc = draws[0]
        edges = [
            (a, fa, fpa, a, fb, c, fc), (a, fa, fpa, b, fb, a, fc), (a, fa, fpa, b, fb, b, fc),
            (0.0, 0.0, -1.0, 1.0, 10.0, 2.0, -50.0), (a, fa, fpa, b * 1e200, fb, c * 1e200, fc),
            (a, fa, fpa, b, np.inf, c, fc), (a, fa, fpa, b, fb, c, np.nan),
        ]
        for case, (a, fa, fpa, b, fb, c, fc) in enumerate(draws + edges):
            for name, scipy_min, jax_min, args in (
                ("cubic", scipy_linesearch._cubicmin, _cubicmin, (a, fa, fpa, b, fb, c, fc)),
                ("quadratic", scipy_linesearch._quadmin, _quadmin, (a, fa, fpa, b, fb)),
            ):
                expected = scipy_min(*(np.float64(value) for value in args))
                value, valid = (host_array(out) for out in jax_min(*(jnp.float64(value) for value in args)))
                self.assertEqual(bool(valid), expected is not None, f"{name} case {case}: SciPy gives {expected}")
                if expected is not None:
                    np.testing.assert_allclose(value, expected, rtol=1e-12, err_msg=f"{name} case {case}")

        class Captured(Exception):
            pass

        brackets = [(rng.uniform(0.0, 1.0), rng.uniform(1.0, 3.0)) for _ in range(100)]
        for case, (a_lo, a_hi, phi_lo, phi_hi, derphi_lo) in enumerate(
            # Half the brackets are reversed (a_lo > a_hi), as _zoom's callers pass them.
            (*(bracket if case % 2 else bracket[::-1]), -rng.uniform(), rng.uniform(), -rng.uniform())
            for case, bracket in enumerate(brackets)
        ):
            phi0, derphi0, high = 0.0, -1.0, phi_lo + 1.0
            trials = []

            def phi(alpha):
                # A high value: the trial replaces a_hi, and a_hi becomes a_rec.
                trials.append(alpha)
                if len(trials) == 2:
                    raise Captured
                return high

            with self.assertRaises(Captured):
                scipy_linesearch._zoom(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi, None, phi0, derphi0, 1e-4, 0.9, None)
            first = _zoom_trial(a_lo, a_hi, phi_lo, phi_hi, derphi_lo, 0.0, phi0, 0)
            second = _zoom_trial(a_lo, first, phi_lo, high, derphi_lo, a_hi, phi_hi, 1)
            np.testing.assert_allclose([host_array(first), host_array(second)], trials, rtol=1e-12,
                                       err_msg=f"zoom trials, case {case}")

    def test_lbfgsb_tiny_corrections_match_scipy_op_by_op(self):
        """Corrections near 1e-100 empty the memory as in SciPy: iterates, records and pairs held, op by op (CPU)."""
        # They underflow the products of formt's L D^-1 L^T (sy_ik sy_jk / sy_kk, multiplied before dividing as in
        # SciPy's C), so T loses its last pivot and the memory empties. For k = 1, ..., 8 the iterates (to 1e-14),
        # counts, statuses, messages and pairs held (whose drops are the resets) are SciPy's. Op by op: compiled, XLA
        # fuses multiply-adds into one rounding, an ulp from SciPy, and DCSTEP's cubic, whose terms cancel ten orders of
        # magnitude here, magnifies that to 1e-5; the reset turns on a pivot of rounding residue, which only the same
        # LAPACK Cholesky rounds alike.
        x0 = np.array([2e-100])
        pairs, scipy_pairs = [], []
        for k in range(1, 9):
            options = {"gtol": 0.0, "ftol": 0.0, "maxiter": k, "maxcor": 3}
            with self.subTest(k=k), self.case(), jax.disable_jit():
                expected = _scipy("tiny-scale", x0, np.zeros(0), "L-BFGS-B", options)
                result, message = _device("tiny-scale", x0, np.zeros(0), "L-BFGS-B", options, "cpu")
                self._assert_scipy_record(expected, result, message, f"tiny-scale k={k}")
                assert_matches_native(result.x, expected.x, f"tiny-scale x at k={k}", _TINY_RTOL)
                pairs.append(int(result.hess_inv.count))
                scipy_pairs.append(expected.hess_inv.n_corrs)
        self.assertEqual(pairs, scipy_pairs, "pairs held after k = 1, ..., 8 differ from SciPy's")
        resets = sum(after < before for before, after in zip(scipy_pairs, scipy_pairs[1:]))
        self.assertEqual(resets, 2, f"SciPy's pairs held {scipy_pairs} no longer show its two resets")

    def test_invalid_inputs_end_as_in_scipy(self):
        """SciPy's input checks without bounds: stops before iterating or ``ValueError``; the objective goes first."""
        # L-BFGS-B stops on a negative ftol (factr), no corrections or no variables, with SciPy's status, message,
        # counters, zero fun and jac and empty hess_inv; maxls <= 0 raises SciPy's ValueError, and so does BFGS without
        # variables. Each exit first evaluates the objective at x0, as SciPy does, so an objective that raises raises.
        x0, params = _problems()["rosenbrock"]
        stopped = {
            "ftol < 0": (x0, _options("L-BFGS-B", 100, ftol=-1e-8)),
            "maxcor = 0": (x0, _options("L-BFGS-B", 100, maxcor=0)),
            "no variables": (np.zeros(0), _options("L-BFGS-B", 100)),
        }
        for case, (start, options) in stopped.items():
            expected = _scipy("rosenbrock", start, params, "L-BFGS-B", options)
            for lane in _LANES:
                with self.subTest(lane=lane, case=case), self.case():
                    self.assertEqual(expected.status, 2, f"{case}: SciPy did not stop")
                    result, message = _device("rosenbrock", start, params, "L-BFGS-B", options, lane)
                    self._assert_scipy_record(expected, result, message, case)
                    for field in ("x", "fun", "jac"):
                        np.testing.assert_array_equal(getattr(result, field), expected[field],
                                                      err_msg=f"{case}: {field} differs from SciPy's")
                    self._assert_scipy_hess_inv(expected, result, case, 0.0)
        raised = {
            "L-BFGS-B maxls = 0": ("L-BFGS-B", x0, _options("L-BFGS-B", 100, maxls=0)),
            "BFGS without variables": ("BFGS", np.zeros(0), _options("BFGS", 100)),
        }
        for case, row in raised.items():
            self._assert_raises_like_scipy(ValueError, case, "rosenbrock", *row)
        # Every exit evaluates the objective at x0 first, so its errors come first.
        exits = {case: ("L-BFGS-B", start, options) for case, (start, options) in stopped.items()} | raised
        for case, row in exits.items():
            self._assert_raises_like_scipy(RuntimeError, f"{case}, objective raising", "unavailable", *row)

    def _assert_raises_like_scipy(self, error, case, name, method, start, options, *, params=np.zeros(0)):
        """SciPy and the device solver (on the CPU) raise ``error`` with the same message."""
        with self.subTest(case=case), self.case():
            with self.assertRaises(error) as scipy_error:
                _scipy(name, start, params, method, options)
            with self.assertRaises(error) as device_error:
                _device(name, start, params, method, options, "cpu")
            self.assertEqual(str(device_error.exception), str(scipy_error.exception), case)

    def test_nan_and_maxiter_zero_end_as_in_scipy(self):
        """A NaN wall gives BFGS status 3; maxiter=0 stops BFGS at x0 and L-BFGS-B after one step."""
        x0, params = _problems()["rosenbrock"]
        self._assert_ends_as_scipy({
            "BFGS nan": ("nan-wall", np.zeros(3), "BFGS", _options("BFGS", 100), 3),
            "BFGS maxiter=0": ("rosenbrock", x0, "BFGS", _options("BFGS", 0), 1),
            "L-BFGS-B maxiter=0": ("rosenbrock", x0, "L-BFGS-B", _options("L-BFGS-B", 0), 1),
        }, params)


# The compiled solver loops on synthetic systems (TestBoozerSolverLoopsJax). No Boozer problem gives an exactly
# singular matrix after a regular first step, so each loop is driven by a two-unknown system substituted for its
# formulation: the identity at the start, losing its second pivot once the first unknown reaches zero, which the
# first step does exactly. Native's np.linalg.solve raises there; the loop must stop, flag singular and keep the
# iterate it reached (not restore its start although its norm grew, since native raises before that). A second
# system, x with the constant matrix 0.05 I, multiplies the norm at every step (as native's
# test_penalty_newton_divergence_factor): the loop runs to maxiter, reports the final norm and restores its start.

_START = np.array([1.0, 0.0])
_TOL = 1e-12
_MAXITER = 10.0


def _matrix(x: jax.Array) -> jax.Array:
    """``diag(1, x[0])``: the identity at the start, singular once ``x[0] = 0``."""
    return jnp.diag(jnp.stack((jnp.ones_like(x[0]), x[0])))


def _growth(x: jax.Array) -> jax.Array:
    """``1 + x[1]^2``: 1 at the start, 2 at the first exact and Newton iterate, so those walks end at a larger norm."""
    return 1 + x[1] ** 2


def _exact_system(problem, x, residual_rows, *, derivatives=0):
    return _growth(x) * jnp.ones(2, x.dtype), _matrix(x)


def _exact_final_residual(problem, x, *, optimize_G, weight_inv_modB=False):
    return (x,)


def _penalty_derivatives(problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True):
    return jnp.zeros((), x.dtype), _growth(x) * jnp.ones(2, x.dtype), _matrix(x)


def _penalty_residuals(problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True):
    # J^T r = [2, x[0]] at the start: the damped first step (lam = 1) is [1, 1/2];
    # at the iterate it reaches, |J^T r| = 2.5 > |[2, 1]|.
    return _growth(x) * jnp.array([2.0, 1.0], x.dtype), _matrix(x)


# Per loop: the formulation(s) substituted, the iterate after the first step and
# the matrix native's np.linalg.solve factorises at the second step.
_CASES = {
    "exact": (
        {"boozer_exact_residual": _exact_system, "boozer_surface_residual": _exact_final_residual},
        np.array([0.0, -1.0]), np.diag([1.0, 0.0]),
    ),
    "penalty-newton": (
        {"boozer_penalty_constraints": _penalty_derivatives}, np.array([0.0, -1.0]), np.diag([1.0, 0.0])
    ),
    "gauss-newton": (
        {"boozer_penalty_residual": _penalty_residuals}, np.array([0.0, -0.5]), np.diag([1.0 + 1.0 / 3.0, 0.0])
    ),
}


_GROWTH = 0.05


def _growing_exact_system(problem, x, residual_rows, *, derivatives=0):
    # Refined Newton steps x - x / 0.05 = -19 x.
    return x, _GROWTH * jnp.eye(2, dtype=x.dtype)


def _growing_penalty_derivatives(problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True):
    # Newton steps x - x / 0.05 = -19 x.
    return 0.5 * x @ x, x, _GROWTH * jnp.eye(2, dtype=x.dtype)


def _growing_penalty_residuals(problem, x, *, derivatives=0, optimize_G=False, weight_inv_modB=True):
    # J^T r = 0.05 x and J^T J = 0.0025 I: the damped first step (lam = 1) gives -9 x.
    return x, _GROWTH * jnp.eye(2, dtype=x.dtype)


# Per loop: the formulation substituted, the norm after two steps (BoozerExact
# and penalty Newton: x = 361 x0; Gauss-Newton: steps 10 x and 15 x give x = 126 x0, where
# |J^T r| = 0.05 * 126) and the values at the start that the restored result carries.
_GROWING_CASES = {
    "exact": (
        {"boozer_exact_residual": _growing_exact_system, "boozer_surface_residual": _exact_final_residual},
        361.0, {"residual": _START, "jacobian": _GROWTH * np.eye(2)},
    ),
    "penalty-newton": (
        {"boozer_penalty_constraints": _growing_penalty_derivatives},
        361.0, {"gradient": _START, "hessian": _GROWTH * np.eye(2)},
    ),
    "gauss-newton": (
        {"boozer_penalty_residual": _growing_penalty_residuals},
        _GROWTH * 126.0, {"residual": _START, "gradient": _GROWTH * _START, "normal_matrix": _GROWTH**2 * np.eye(2)},
    ),
}


def _compiled_afresh(solver, static_argnames: tuple[str, ...]):
    """``solver``'s body compiled anew, so it traces the substituted formulation."""
    body: Callable[..., object] = inspect.unwrap(solver)
    return jit(body, static_argnames=static_argnames)


def _substitute(patches, formulations):
    """Patch each ``boozer_solvers`` formulation for the rest of the case."""
    for attribute, formulation in formulations.items():
        patches.enter_context(mock.patch.object(boozer_solvers, attribute, formulation))


def _run(name: str, x: jax.Array, maxiter: float = _MAXITER, divergence_factor=None):
    """The loop ``name`` compiled afresh (so it traces the substituted formulation); ``divergence_factor`` goes to
    the penalty Newton."""
    tol, cap = jnp.asarray(_TOL), jnp.asarray(maxiter)
    if name == "exact":
        loop = _compiled_afresh(boozer_solvers.boozer_exact_newton, ("G_from_currents",))
        return loop(None, x, jnp.arange(2, dtype=jnp.int32), tol, cap)
    options = {"optimize_G": True, "weight_inv_modB": False}
    if name == "penalty-newton":
        loop = _compiled_afresh(boozer_solvers.boozer_penalty_newton, tuple(options))
        factor = None if divergence_factor is None else jnp.asarray(divergence_factor)
        return loop(None, x, tol, cap, jnp.asarray(0.0), factor, **options)
    loop = _compiled_afresh(boozer_solvers.boozer_penalty_gauss_newton, tuple(options))
    return loop(None, x, tol, cap, **options)


class TestBoozerSolverLoopsJax(JaxTestCase):
    def test_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(self):
        """Every device solver stops on a singular second step and retains its last completed iterate."""
        for case_id_0, parity_lane in zip(("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True):
            for name in _CASES:
                with self.subTest(case_id_0=case_id_0, parity_lane=parity_lane, name=name), self.case() as patches:
                    self._case_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(parity_lane, name, patches)

    def _case_loop_stops_at_a_singular_second_step_and_keeps_its_iterate(self, parity_lane, name, patches):
        formulations, reached, second_matrix = _CASES[name]
        with self.assertRaisesRegex(np.linalg.LinAlgError, "Singular matrix"):
            np.linalg.solve(second_matrix, np.ones(2))
        _substitute(patches, formulations)
        with parity_default_device(parity_lane):
            result = _run(name, jnp.asarray(_START))
            singular, iterations, x = (host_array(value) for value in (result.singular, result.iterations, result.x))
        self.assertTrue(bool(singular), f"{name}: the singular second step was not flagged")
        self.assertTrue(int(iterations) == 1,
                        f"{name}: {int(iterations)} steps counted; native raises during the second")
        np.testing.assert_array_equal(x, reached, err_msg=f"{name}: the loop did not keep the iterate it reached")

    def test_growing_walk_runs_to_maxiter_and_restores_its_start(self):
        """A walk whose norm grows at every step ends at ``maxiter`` with its final norm and restores its start."""
        for case_id_0, parity_lane in zip(("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True):
            for name in _GROWING_CASES:
                with self.subTest(case_id_0=case_id_0, parity_lane=parity_lane, name=name), self.case() as patches:
                    self._case_growing_walk_runs_to_maxiter_and_restores_its_start(parity_lane, name, patches)

    def _case_growing_walk_runs_to_maxiter_and_restores_its_start(self, parity_lane, name, patches):
        formulations, norm, restored = _GROWING_CASES[name]
        _substitute(patches, formulations)
        with parity_default_device(parity_lane):
            result = _run(name, jnp.asarray(_START), maxiter=2.0)
            keys = ("x", "iterations", "norm", "singular", *restored)
            values = {key: host_array(getattr(result, key)) for key in keys}
        self.assertEqual(int(values["iterations"]), 2, name)
        self.assertFalse(bool(values["singular"]), name)
        np.testing.assert_allclose(values["norm"], norm, rtol=1e-12, err_msg=name)
        np.testing.assert_array_equal(values["x"], _START, err_msg=f"{name}: start not restored")
        for key, expected in restored.items():
            np.testing.assert_array_equal(values[key], expected, err_msg=f"{name} {key}")

    def test_divergence_factor_stops_a_growing_penalty_newton_walk(self):
        """Growing 19-fold per step, a factor of 1e3 stops after three steps, None runs to ``maxiter``; both restore."""
        # As in native's test_penalty_newton_divergence_factor (norm 6859 after three steps).
        formulations, _, restored = _GROWING_CASES["penalty-newton"]
        for factor, steps in ((1e3, 3), (None, 40)):
            with self.subTest(factor=factor), self.case() as patches:
                _substitute(patches, formulations)
                result = _run("penalty-newton", jnp.asarray(_START), maxiter=40.0, divergence_factor=factor)
                self.assertEqual(int(host_array(result.iterations)), steps)
                np.testing.assert_allclose(host_array(result.norm), 19.0**steps, rtol=1e-12)
                np.testing.assert_array_equal(host_array(result.x), _START)
                np.testing.assert_array_equal(host_array(result.gradient), restored["gradient"])
