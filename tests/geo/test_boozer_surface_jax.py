"""JAX ``JaxBoozerSurface`` and its device solvers against native ``BoozerSurface`` and SciPy."""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt._core.derivative import Derivative
    from simsopt.configs import get_data
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.geo.boozersurface import BoozerSurface
    from simsopt.geo.surfaceobjectives import BoozerResidual, Iotas, MajorRadius, NonQuasiSymmetricRatio
    from simsopt.geo.surfaceobjectives import PrincipalCurvature, ToroidalFlux, Volume
    from simsopt.geo.surfaceobjectives import boozer_surface_dexactresidual_dcoils_dcurrents_vjp
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
    from simsopt.objectives.utilities import forward_backward
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.backend.runtime import host_tree
    from simsopt_jax.core import boozer_solvers
    from simsopt_jax.core.boozer_problem import boozer_penalty_constraints, boozer_penalty_residual
    from simsopt_jax.core.boozer_solvers import boozer_penalty_bfgs, boozer_penalty_lbfgsb
    from simsopt_jax.core.quasi_newton import BFGS_MESSAGES, LBFGSB_MESSAGES, _bfgs_inverse_hessian_update, _cubicmin
    from simsopt_jax.core.quasi_newton import _quadmin, _zoom_trial, minimize_bfgs, minimize_lbfgsb
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_problem import boozer_problem
    from simsopt_jax_adapters.geo import boozer_surface as jax_boozer_surface
    from simsopt_jax_adapters.geo.boozer_surface import JaxBoozerResidual, JaxBoozerSurface
    from .surface_test_helpers import get_boozer_surface

from unittest_jax_support import assert_errors_shrink, assert_matches_native, disallow_host_transfers, host_array
from unittest_jax_support import jax_compilations, parity_default_device, parity_rng, place_float64

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial
import copy
import inspect
import unittest
from unittest import mock
from typing import cast
import warnings

import numpy as np
import scipy
from scipy.optimize import LbfgsInvHessProduct, OptimizeResult, least_squares, minimize
from scipy.optimize import _linesearch as scipy_linesearch


# Iterates and arrays agree to 1e-12 of the largest native entry (measured worst 2e-13, coil gradients of the
# objectives) and, without stellarator symmetry, where the BoozerExact system is nearly singular in one direction,
# to 1e-9 (measured 4e-11). BFGS and diverging Newton walks magnify round-off, so their tests cap the walks and state
# their own bounds. Solve tolerances sit above the residuals' round-off floor (native 9e-14, JAX 6e-14 here), below
# which native's decisions follow round-off.
_IOTA = -0.406
_RTOL = 1e-12
_RTOL_NONSYM = 1e-9
_WEIGHT = 100.0
# Scratch bytes the device quasi-Newton loops may add to the penalty's for their
# vectors and scalars, beyond their matrices.
_QUASI_NEWTON_SLACK = 2**20
_LANE_CASES = [dict(case_id_0=f"{lane}_parity", parity_lane=lane) for lane in ("cpu", "gpu")]
_STELLSYM_CASES = [dict(case_id_0="stellsym", stellsym=True), dict(case_id_0="nonsym", stellsym=False)]


@dataclass
class _Problem:
    """One of upstream's ``get_boozer_surface(converge=False)`` problems."""

    coils: list
    currents: list
    boozer: BoozerSurface | JaxBoozerSurface
    G0: float | None


def _G0(nfp: int, base_currents) -> float:
    """Upstream's starting ``G``: ``mu0`` times the summed ``|I|`` of the coils."""
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    return 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))


def scaled_direction(x0: np.ndarray, seed: int) -> np.ndarray:
    """A seeded normal direction moving each entry of ``x0`` by a fraction of its own size (of one, below one)."""
    return parity_rng(seed).standard_normal(x0.size) * np.maximum(np.abs(x0), 1.0)


def _problem(
    jax_field: bool, boozer_type: str = "exact", label: str = "Volume", *, stellsym: bool = True,
    optimize_G: bool = True, weight_inv_modB: bool = False, label_grid: tuple[int, int] | None = None,
    newton_tol: float | None = None, options: dict | None = None,
) -> _Problem:
    """Upstream's NCSX problem as a native ``BoozerSurface`` or a ``JaxBoozerSurface`` (``jax_field``).

    Without ``options``, verbose is off and ``weight_inv_modB`` and ``newton_tol`` (if given) are set; ``options``
    itself goes to ``JaxBoozerSurface`` and a copy to native, which fills it in place."""
    label_nphi, label_ntheta = label_grid or (None, None)
    bs, native = get_boozer_surface(
        label=label, nphi=label_nphi, ntheta=label_ntheta, boozer_type=boozer_type, optimize_G=optimize_G,
        converge=False, stellsym=stellsym, weight_inv_modB=weight_inv_modB if boozer_type == "ls" else False,
    )
    surface, label_object = native.surface, native.label
    # The first half-period contains the NCSX base currents, in their original order.
    base_currents = [coil.current for coil in bs.coils[:len(bs.coils) // (2 * surface.nfp)]]
    G0 = _G0(surface.nfp, base_currents) if optimize_G else None
    if options is None:
        options = {"verbose": False} if weight_inv_modB else {"verbose": False, "weight_inv_modB": False}
        if newton_tol is not None:
            options["newton_tol"] = newton_tol
    constraint_weight = None if boozer_type == "exact" else _WEIGHT
    # These parity tests keep the initial label, including AspectRatio (the helper uses 0.9 * J()).
    boozer = (
        JaxBoozerSurface(JaxBiotSavart(bs.coils), surface, label_object, label_object.J(), constraint_weight, options)
        if jax_field
        else BoozerSurface(bs, surface, label_object, label_object.J(), constraint_weight, dict(options))
    )
    return _Problem(bs.coils, base_currents, boozer, G0)


def _pair(*args, **kwargs) -> tuple[_Problem, _Problem]:
    return _problem(False, *args, **kwargs), _problem(True, *args, **kwargs)


def _assert_same_solve(jax_problem: _Problem, native_problem: _Problem, jax_res, native_res, arrays, name, rtol=_RTOL):
    """Native's result keys in order; iterations, success and ``G`` as natively; the surface, ``iota`` and ``arrays``
    to round-off; native's adjoint solve with the ``PLU`` (to round-off times the condition number)."""
    assert list(jax_res) == list(native_res), f"{name}: result keys {list(jax_res)} != native {list(native_res)}"
    for key in ("iter", "success"):
        if key in native_res:
            assert jax_res[key] == native_res[key], f"{name}: {key} {jax_res[key]} != native {native_res[key]}"
    assert (jax_res.get("G") is None) == (native_res.get("G") is None), f"{name}: G presence"
    assert_matches_native(
        jax_problem.boozer.surface.get_dofs(), native_problem.boozer.surface.get_dofs(), f"{name} surface", rtol
    )
    for key in ("iota", "G", *arrays):
        if native_res.get(key) is not None:
            assert_matches_native(jax_res[key], native_res[key], f"{name} {key}", rtol)
    if "PLU" in native_res:
        # The adjoint solve magnifies the matrices' round-off difference by their condition number.
        P, L, U = native_res["PLU"]
        jax_P, jax_L, jax_U = jax_res["PLU"]
        rhs = parity_rng(7).standard_normal(P.shape[0])
        assert_matches_native(forward_backward(jax_P, jax_L, jax_U, rhs), forward_backward(P, L, U, rhs),
                              f"{name} PLU solve", rtol * np.linalg.cond(P @ L @ U))


# --- BoozerExact ----------------------------------------------------------------------


_EXACT_CASES = {
    "stellsym-volume": (True, "Volume", None),
    "stellsym-flux-own-grid": (True, "ToroidalFlux", (51, 51)),
    "nonsym-volume": (False, "Volume", None),
    "nonsym-area-own-grid": (False, "Area", (31, 31)),
}


# --- BoozerLS ---------------------------------------------------------------------------


def _same_start(
    native: _Problem, jax_problem: _Problem, bfgs_maxiter: int, constraint_weight: float = _WEIGHT, *,
    tol: float = 1e-10, iota: float = _IOTA, limited_memory: bool = False, weight_inv_modB: bool = False,
):
    """A native BFGS iterate, given to both surfaces: ``(iota, G)``."""
    res = native.boozer.minimize_boozer_penalty_constraints_LBFGS(
        tol=tol, maxiter=bfgs_maxiter, constraint_weight=constraint_weight, iota=iota, G=native.G0,
        limited_memory=limited_memory, weight_inv_modB=weight_inv_modB,
    )
    jax_problem.boozer.surface.set_dofs(native.boozer.surface.get_dofs())
    native.boozer.recompute_bell()
    return res["iota"], res["G"]


def _singular_at_second_factorisation(patches, solver_name: str, native_solves_per_step: int):
    """Native's ``np.linalg.solve`` raising in the second step (after ``native_solves_per_step`` calls), and the JAX
    solver ``solver_name`` running one real step (``maxiter=1``) and stopping singular, as at an exactly zero pivot."""
    real_solve = np.linalg.solve
    calls = []

    def solve(matrix, rhs):
        calls.append(None)
        if len(calls) > native_solves_per_step:
            raise np.linalg.LinAlgError("Singular matrix")
        return real_solve(matrix, rhs)

    real_solver = getattr(jax_boozer_surface, solver_name)

    def solver(*args, **kwargs):
        arguments = inspect.signature(real_solver).bind(*args, **kwargs).arguments
        arguments["maxiter"] = place_float64(1.0, arguments["maxiter"])
        return replace(real_solver(**arguments), singular=jnp.asarray(True))

    patches.enter_context(mock.patch.object(np.linalg, "solve", solve))
    patches.enter_context(mock.patch.object(jax_boozer_surface, solver_name, solver))


def _least_squares_problem(jax_field: bool) -> _Problem:
    """Upstream's ``subtest_minimize_boozer_penalty_constraints_ls_manual`` problem (stellsym, ``optimize_G``)."""
    _, base_currents, axis, nfp, bs = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=5, ntor=5, stellsym=True, nfp=nfp, clamped_dims=[False, False, False],
        quadpoints_phi=np.linspace(0, 1 / nfp, 11, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 11, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1)
    flux = ToroidalFlux(surface, BiotSavart(bs.coils), nphi=51, ntheta=51)
    field = JaxBiotSavart(bs.coils) if jax_field else bs
    boozer = (JaxBoozerSurface if jax_field else BoozerSurface)(field, surface, flux, 0.1)
    return _Problem(bs.coils, base_currents, boozer, _G0(nfp, base_currents))


# --- native objectives and adjoints ---------------------------------------------


_NATIVE_OBJECTIVES = {
    "Iotas": lambda problem: Iotas(problem.boozer),
    "MajorRadius": lambda problem: MajorRadius(problem.boozer),
    "NonQuasiSymmetricRatio": lambda problem: NonQuasiSymmetricRatio(problem.boozer, BiotSavart(problem.coils)),
    "BoozerResidual": lambda problem: BoozerResidual(problem.boozer, BiotSavart(problem.coils)),
}


_OBJECTIVE_CASES = {
    "exact-volume": ("exact", "Volume", True, False),
    "exact-flux-own-grid": ("exact", "ToroidalFlux", True, False),
    "ls-weighted": ("ls", "Volume", True, True),
    "ls-G-from-currents": ("ls", "Volume", False, False),
}


# --- settings, compilation, transfers and the boundary --------------------------


def _toroidal_flux_adjoint_problem():
    """Solve the shared ToroidalFlux problem and build its adjoint objectives."""
    native, jax_problem = _pair("ls", "ToroidalFlux", label_grid=(31, 31))
    jax_objectives = {
        **_NATIVE_OBJECTIVES,
        "BoozerResidual": lambda problem: JaxBoozerResidual(problem.boozer, problem.boozer.biotsavart),
    }
    for problem in (native, jax_problem):
        problem.boozer.run_code(_IOTA, G=problem.G0)
    native_objectives = {name: make(native) for name, make in _NATIVE_OBJECTIVES.items()}
    return native, jax_problem, native_objectives, jax_objectives


class TestBoozerSurfaceJax(JaxTestCase):
    def test_exact_run_code_matches_native(self):
        """Exact run_code solves match native surface state, residuals and result metadata."""
        def check(name):
            stellsym, label, grid = _EXACT_CASES[name]
            native, jax_problem = _pair("exact", label, stellsym=stellsym, label_grid=grid, newton_tol=1e-10)
            native_res = native.boozer.run_code(_IOTA, G=native.G0)
            jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
            self.assertTrue(native_res["success"])
            self.assertGreater(native_res["iter"], 3)
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), name,
                               _RTOL if stellsym else _RTOL_NONSYM)
            np.testing.assert_array_equal(jax_res["mask"], native_res["mask"])
            self.assertEqual(jax_res["type"], "exact")
            self.assertIs(jax_res["s"], jax_problem.boozer.surface)
            self.assertTrue(np.max(np.abs(jax_res["residual"])) < 1e-11, "the converged residual is not at round-off")
            self.assertFalse(jax_problem.boozer.need_to_run_code)

        self.run_cases(check, [dict(name=name) for name in _EXACT_CASES])

    def test_exact_newton_at_maxiter_judges_its_final_iterate_as_natively(self):
        """At ``maxiter`` native judges success on the final iterate; fractional caps step below them, others don't."""
        # A walk whose last permitted step converges succeeds (maxiter=6); the surface holds the last iterate, the
        # residual and Jacobian are evaluated there. A nonpositive cap moves nothing.
        def check(case_id_0, stellsym):
            # The cap, native's step count and success at it.
            for maxiter, steps, success in ((2, 2, False), (6, 6, True), (0.5, 1, False), (0, 0, False),
                                            (-1, 0, False)):
                native, jax_problem = _pair("exact", stellsym=stellsym)
                start = native.boozer.surface.get_dofs()
                native_res = native.boozer.solve_residual_equation_exactly_newton(
                    tol=1e-10, maxiter=maxiter, iota=_IOTA, G=native.G0
                )
                jax_res = jax_problem.boozer.solve_residual_equation_exactly_newton(
                    tol=1e-10, maxiter=maxiter, iota=_IOTA, G=jax_problem.G0
                )
                self.assertEqual((native_res["iter"], native_res["success"]), (steps, success))
                rtol = _RTOL if stellsym else _RTOL_NONSYM
                _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), f"maxiter={maxiter}", rtol)
                # Converged residuals are round-off: compare on the scale of the terms (G |B| is 22 here).
                assert_matches_native(jax_res["residual"], native_res["residual"], f"maxiter={maxiter} residual", rtol,
                                      scale=22.0)
                if steps == 0:
                    np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), start)

        self.run_cases(check, _STELLSYM_CASES)

    def test_exact_newton_on_a_singular_system_raises_as_natively(self):
        """Without current the residual rows vanish; with the label off target native's solve raises before moving."""
        native, jax_problem = _pair("exact")
        for problem in (native, jax_problem):
            for current in problem.currents:
                current.local_full_x = np.zeros(1)
            problem.boozer.targetlabel *= 1.1
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(np.linalg.LinAlgError, "Singular matrix"):
                problem.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=5, iota=_IOTA, G=native.G0)
            np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)
            self.assertTrue(problem.boozer.need_to_run_code)

    def test_non_finite_solves_fail_as_natively(self):
        """A NaN ``G`` and a weighted zero-current penalty end in SciPy's ``lu`` refusal, at the start, as natively."""
        # A NaN G makes every BoozerExact step NaN: the walk runs to maxiter (a NaN norm never meets the tolerance) and
        # restores its start, where G is still NaN. A weighted BoozerLS penalty without current is 0/0: the Newton
        # loop does not start and lu refuses the Hessian before the surface moves.
        native, jax_problem = _pair("exact")
        for problem in (native, jax_problem):
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                problem.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=3, iota=_IOTA, G=np.nan)
            np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)

        native, jax_problem = _pair("ls", weight_inv_modB=True)
        for problem in (native, jax_problem):
            for current in problem.currents:
                current.local_full_x = np.zeros(1)
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11, maxiter=5, constraint_weight=_WEIGHT, iota=_IOTA, G=native.G0, weight_inv_modB=True
                )
            np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)

    def test_exact_newton_takes_G_from_the_currents(self):
        """Exact Newton seeds G from coil currents when no explicit initial G is supplied."""
        native, jax_problem = _pair("exact", optimize_G=False, newton_tol=1e-10)
        native_res = native.boozer.run_code(_IOTA)
        jax_res = jax_problem.boozer.run_code(_IOTA)
        _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), "G from currents")

    def test_ls_run_code_matches_native(self):
        """BFGS (or L-BFGS-B) then the penalty Newton: the polished solution, Newton step count and flags."""
        # Near convergence the quasi-Newton walk's last line searches follow round-off (its iteration count can
        # differ by one or two), so only the polished result is compared.
        def check(limited_memory, case_id_0, optimize_G, case_id_1, weight_inv_modB):
            native, jax_problem = _pair("ls", optimize_G=optimize_G, weight_inv_modB=weight_inv_modB, options={
                "verbose": False, "weight_inv_modB": weight_inv_modB, "limited_memory": limited_memory
            })
            native_res = native.boozer.run_code(_IOTA, G=native.G0)
            jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
            self.assertTrue(native_res["success"])
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("hessian",), "run_code")
            self.assertEqual(jax_res["type"], "ls")
            self.assertEqual(jax_res["weight_inv_modB"], weight_inv_modB)
            self.assertLessEqual(np.linalg.norm(jax_res["jacobian"]), 1e-11)

        self.run_cases(check, [
            dict(limited_memory=limited_memory, case_id_0=G_id, optimize_G=optimize_G, case_id_1=weight_id,
                 weight_inv_modB=weight_inv_modB)
            for limited_memory in (False, True) for G_id, optimize_G in (("G", True), ("G-from-currents", False))
            for weight_id, weight_inv_modB in (("weighted", True), ("unweighted", False))
        ])

    def test_bfgs_matches_native_up_to_maxiter(self):
        """Capped before convergence (a failed solve, kept as natively); iterates separate by 3e-13 at 15 steps."""
        # Round-off growth separates the walks: iterates by 3e-13 after 15 iterations, 1e-9 after 30 here.
        def check(case_id_0, stellsym, case_id_1, limited_memory):
            maxiter = 15
            native, jax_problem = _pair("ls", stellsym=stellsym)
            native_res, jax_res = (
                problem.boozer.minimize_boozer_penalty_constraints_LBFGS(
                    tol=1e-10, maxiter=maxiter, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0,
                    limited_memory=limited_memory, weight_inv_modB=False,
                )
                for problem in (native, jax_problem)
            )
            self.assertFalse(native_res["success"])
            self.assertEqual(native_res["iter"], maxiter)
            keys = ("nfev", "njev", "status", "message")
            self.assertEqual(tuple(jax_res["info"][key] for key in keys),
                             tuple(native_res["info"][key] for key in keys))
            # info is SciPy's OptimizeResult: its fields in its order, hess_inv
            # included. The steps carry the iterates' difference, and H magnifies
            # it (measured 2.1e-11 for sk, 6.3e-11 for H, of the largest entry).
            self.assertEqual(list(jax_res["info"]), list(native_res["info"]))
            jax_hess, native_hess = jax_res["info"].hess_inv, native_res["info"].hess_inv
            self.assertIs(type(jax_hess), type(native_hess))
            if limited_memory:
                assert_matches_native(jax_hess.sk, native_hess.sk, "L-BFGS-B hess_inv sk", 1e-10)
                assert_matches_native(jax_hess.yk, native_hess.yk, "L-BFGS-B hess_inv yk", 1e-10)
            else:
                assert_matches_native(jax_hess, native_hess, "BFGS hess_inv", 5e-10)
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("fun",), "BFGS", rtol=1e-11)
            # The Hessian magnifies the iterates' difference (measured 8e-11 of the largest entry).
            assert_matches_native(jax_res["gradient"], native_res["gradient"], "BFGS gradient", 1e-9)
            self.assertIs(jax_res["s"], jax_problem.boozer.surface)
            self.assertFalse(jax_problem.boozer.need_to_run_code)

        self.run_cases(check, [
            dict(**stellsym, case_id_1=method, limited_memory=limited_memory)
            for stellsym in _STELLSYM_CASES for method, limited_memory in (("BFGS", False), ("L-BFGS-B", True))
        ])

    def test_host_bfgs_runs_scipy_on_the_jitted_penalty(self):
        """With ``host_bfgs``, BFGS and L-BFGS-B are SciPy's on the jitted penalty bit for bit, via ``run_code`` too."""
        def check(limited_memory):
            maxiter, tol = 20, 1e-10
            jax_problem = _problem(True, "ls", options={
                "verbose": False, "weight_inv_modB": False, "limited_memory": limited_memory, "host_bfgs": True,
                "bfgs_maxiter": maxiter, "bfgs_tol": tol, "newton_maxiter": 0,
            })
            boozer = jax_problem.boozer
            start = boozer.surface.get_dofs()
            problem = boozer_problem(boozer.biotsavart, boozer.surface, boozer.label, boozer.targetlabel, _WEIGHT)

            def penalty(x):
                return host_tree(boozer_penalty_constraints(problem, place_float64(x, problem.targetlabel),
                                                            derivatives=1, optimize_G=True, weight_inv_modB=False))

            options = {"maxiter": maxiter, "gtol": tol}
            if limited_memory:
                options.update(maxcor=200, ftol=tol)
            expected = minimize(penalty, np.concatenate((start, [_IOTA, jax_problem.G0])), jac=True,
                                method="L-BFGS-B" if limited_memory else "BFGS", options=options)
            res = boozer.minimize_boozer_penalty_constraints_LBFGS(
                tol=tol, maxiter=maxiter, constraint_weight=_WEIGHT, iota=_IOTA, G=jax_problem.G0,
                limited_memory=limited_memory, weight_inv_modB=False, host_bfgs=True,
            )
            for key in ("x", "fun", "jac", "nit", "nfev", "njev", "status", "message"):
                np.testing.assert_array_equal(res["info"][key], expected[key],
                                              err_msg=f"host_bfgs {key} differs from SciPy's")
            # run_code reads the option; without Newton steps it ends at SciPy's endpoint.
            boozer.surface.set_dofs(start)
            boozer.recompute_bell()
            boozer.run_code(_IOTA, G=jax_problem.G0)
            np.testing.assert_array_equal(boozer.surface.get_dofs(), expected.x[:-2],
                                          err_msg="run_code did not run SciPy")

        self.run_cases(check, [dict(limited_memory=False), dict(limited_memory=True)])

    def test_device_quasi_newton_scratch_is_the_penalty_plus_its_state(self):
        """The compiled BFGS and L-BFGS-B loops need the penalty's scratch memory plus their state, never a multiple."""
        # BFGS: its inverse Hessian and update (measured 2 n^2 doubles on the CPU, 1.2 n^2 on the GPU, n = 75);
        # L-BFGS-B: its corrections and their small matrices (measured 3.8 MB on the CPU at maxcor 200).
        def check(parity_lane):
            with parity_default_device(parity_lane):
                jax_problem = _problem(True, "ls")
                boozer = jax_problem.boozer
                problem = boozer._problem(_WEIGHT)
                x = place_float64(boozer._decision_vector(_IOTA, jax_problem.G0), problem.targetlabel)
                scalar = place_float64(1.0, problem.targetlabel)
                options = {"optimize_G": True, "weight_inv_modB": False}

                def scratch(lowered):
                    return lowered.compile().memory_analysis().temp_size_in_bytes

                penalty = scratch(boozer_penalty_constraints.lower(problem, x, derivatives=1, **options))
                bfgs = scratch(boozer_penalty_bfgs.lower(problem, x, scalar, scalar, **options))
                maxcor = 200
                lbfgsb = scratch(boozer_penalty_lbfgsb.lower(problem, x, scalar, scalar, scalar, maxcor=maxcor,
                                                             **options))
            n, double = x.shape[0], 8
            self.assertLessEqual(bfgs, penalty + 3 * n * n * double + _QUASI_NEWTON_SLACK,
                                 f"BFGS scratch {bfgs} B exceeds the penalty's {penalty} B plus 3 n^2 doubles")
            self.assertLessEqual(lbfgsb, penalty + 2 * (2 * maxcor * n + 8 * maxcor**2) * double + _QUASI_NEWTON_SLACK,
                                 f"L-BFGS-B scratch {lbfgsb} B exceeds the penalty's {penalty} B plus its memory")

        self.run_cases(check, [dict(parity_lane="cpu"), dict(parity_lane="gpu")])

    def test_penalty_newton_matches_native(self):
        """From a BFGS iterate; ``stab`` shifts the Hessian of each step but not the returned one."""
        def check(stab):
            native, jax_problem = _pair("ls")
            iota, G = _same_start(native, jax_problem, 200)
            native_res, jax_res = (
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11, maxiter=20, constraint_weight=_WEIGHT, iota=iota, G=G, stab=stab, weight_inv_modB=False
                )
                for problem in (native, jax_problem)
            )
            self.assertTrue(native_res["success"])
            self.assertGreater(native_res["iter"], 1)
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("hessian",), f"stab={stab}")
            self.assertIs(jax_res["residual"], jax_res["jacobian"])

        self.run_cases(check, [dict(stab=0.0), dict(stab=1e-4)])

    def test_penalty_newton_diverges_and_restores_its_start_as_natively(self):
        """From an early BFGS iterate Newton diverges to ``maxiter``, fails and restores its start, as natively."""
        # Ending with a larger gradient norm than it started with, it restores its start and the derivatives there.
        # A two-step walk is compared with native's whatever it ends at.
        for maxiter in (2, 40):
            native, jax_problem = _pair("ls")
            iota, G = _same_start(native, jax_problem, 40)
            start = native.boozer.surface.get_dofs()
            native_res, jax_res = (
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11, maxiter=maxiter, constraint_weight=_WEIGHT, iota=iota, G=G, weight_inv_modB=False
                )
                for problem in (native, jax_problem)
            )
            self.assertFalse(native_res["success"])
            self.assertFalse(jax_res["success"])
            self.assertEqual(native_res["iter"], maxiter)
            self.assertEqual(jax_res["iter"], maxiter)
            if maxiter == 40:
                for problem, res in ((native, native_res), (jax_problem, jax_res)):
                    np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)
                    self.assertEqual((res["iota"], res["G"]), (iota, G))
            # Diverging steps magnify round-off (measured 1e-11 of the largest entry).
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian", "hessian"), f"maxiter={maxiter}",
                               1e-9 if maxiter == 2 else _RTOL)

    def test_penalty_newton_divergence_factor_as_natively(self):
        """``divergence_factor`` stops a diverging walk at native's step and leaves a converging one bit for bit."""
        # From an early BFGS iterate (where the walk diverges) a factor of 10 stops the walk before maxiter and the
        # start is restored. On a converging walk (from a later iterate) None, a factor that never fires and the
        # default call give the same iterates bit for bit, and native's.
        with self.subTest(walk="diverging"), self.case():
            self._case_divergence_factor_stops_a_diverging_walk()
        with self.subTest(walk="converging"), self.case():
            self._case_divergence_factor_leaves_a_converging_walk()

    @staticmethod
    def _newton_from_a_bfgs_iterate(bfgs_maxiter):
        """Both problems at native's BFGS iterate, the start's DOFs and the Newton options."""
        native, jax_problem = _pair("ls")
        iota, G = _same_start(native, jax_problem, bfgs_maxiter)
        options = dict(tol=1e-11, maxiter=40, constraint_weight=_WEIGHT, iota=iota, G=G, weight_inv_modB=False)
        return native, jax_problem, native.boozer.surface.get_dofs(), options

    def _case_divergence_factor_stops_a_diverging_walk(self):
        native, jax_problem, start, options = self._newton_from_a_bfgs_iterate(40)
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_newton(**options, divergence_factor=10.0)
            for problem in (native, jax_problem)
        )
        self.assertFalse(native_res["success"])
        self.assertLess(native_res["iter"], 40)
        np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), start)
        _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian", "hessian"), "factor=10")

    def _case_divergence_factor_leaves_a_converging_walk(self):
        native, jax_problem, start, options = self._newton_from_a_bfgs_iterate(200)
        native_res = native.boozer.minimize_boozer_penalty_constraints_newton(**options, divergence_factor=1e3)
        self.assertTrue(native_res["success"])
        jax_res = jax_problem.boozer.minimize_boozer_penalty_constraints_newton(**options, divergence_factor=None)
        # A converged gradient is round-off, so only the Hessian is compared.
        _assert_same_solve(jax_problem, native, jax_res, native_res, ("hessian",), "factor=None")
        solved = jax_problem.boozer.surface.get_dofs()
        for name, factor in (("factor=1e3", {"divergence_factor": 1e3}), ("default", {})):
            jax_problem.boozer.surface.set_dofs(start)
            jax_problem.boozer.recompute_bell()
            res = jax_problem.boozer.minimize_boozer_penalty_constraints_newton(**options, **factor)
            np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), solved, err_msg=name)
            self.assertEqual(res["iter"], jax_res["iter"], name)
            np.testing.assert_array_equal(res["hessian"], jax_res["hessian"], err_msg=name)

    def test_penalty_newton_with_a_non_finite_shift_restores_its_start_as_natively(self):
        """A NaN or infinite ``stab`` fails after one NaN step and restores the start with native's derivatives."""
        # Native's H + stab * I is NaN off the diagonal too (IEEE inf * 0), so the loop stops on the NaN gradient norm.
        # The reference derivatives come from a fresh native problem: native's own solve then raises in lu, because
        # its SurfaceXYZTensorFourier.d2volume_by_dcoeffdcoeff keeps the NaN of the stepped iterate.
        def check(case_id_0, stab):
            native, jax_problem = _pair("ls")
            start = native.boozer.surface.get_dofs()
            x0 = np.concatenate((start, [_IOTA, native.G0]))
            _, gradient, hessian = native.boozer.boozer_penalty_constraints_vectorized(
                x0, derivatives=2, constraint_weight=_WEIGHT, optimize_G=True, weight_inv_modB=False
            )
            res = jax_problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11, maxiter=2, constraint_weight=_WEIGHT, iota=_IOTA, G=jax_problem.G0, stab=stab,
                weight_inv_modB=False,
            )
            self.assertFalse(res["success"])
            self.assertEqual(res["iter"], 1)
            np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), start)
            self.assertEqual((res["iota"], res["G"]), (_IOTA, jax_problem.G0))
            assert_matches_native(res["jacobian"], gradient, f"stab={stab} gradient")
            assert_matches_native(res["hessian"], hessian, f"stab={stab} hessian")

        self.run_cases(check, [dict(case_id_0="nan", stab=np.nan), dict(case_id_0="inf", stab=np.inf)])

    def test_failed_scipy_solves_restore_their_start_as_natively(self):
        """Failed quasi-Newton or ``least_squares`` solves ending worse restore their start; lower ones keep theirs."""
        # SciPy's solver, and the device solver, is replaced by a failed solve that ends at x0 + delta (a larger or
        # non-finite penalty or residual norm) or, for delta=None, by the solver itself capped at two iterations or
        # evaluations (a lower one). BFGS runs with host_bfgs (LBFGS) and on the device.
        for solver in ("LBFGS", "BFGS", "L-BFGS-B", "ls"):
            for delta in (None, 1e-2, np.nan):
                with self.subTest(solver=solver, delta=delta), self.case() as patches:
                    self._case_failed_scipy_solves_restore_their_start_as_natively(solver, delta, patches)

    def _case_failed_scipy_solves_restore_their_start_as_natively(self, solver, delta, patches):
        def failed_minimize(fun, x, **kwargs):
            if delta is None:
                options = {**kwargs.pop("options"), "maxiter": 2}
                return minimize(fun, x, options=options, **kwargs)
            end = x + delta
            value, gradient = fun(end)
            return OptimizeResult(x=end, fun=value, jac=gradient, nit=1, success=False)

        def failed_device_solve(real_solver):
            def solve(*args, **kwargs):
                arguments = inspect.signature(real_solver).bind(*args, **kwargs).arguments
                arguments["maxiter"] = place_float64(2.0 if delta is None else 1.0, arguments["maxiter"])
                result = real_solver(**arguments)
                if delta is None:
                    return result
                end = arguments["x"] + delta
                value, gradient = boozer_penalty_constraints(
                    arguments["problem"], end, derivatives=1, optimize_G=arguments["optimize_G"],
                    weight_inv_modB=arguments["weight_inv_modB"],
                )
                return replace(result, x=end, fun=value, jac=gradient)

            return solve

        def failed_least_squares(fun, x, jac, **kwargs):
            if delta is None:
                return least_squares(fun, x, jac=jac, **{**kwargs, "max_nfev": 2})
            end = x + delta
            residual, jacobian = fun(end), jac(end)
            return OptimizeResult(x=end, fun=residual, grad=jacobian.T @ residual, jac=jacobian, status=0)

        native_name, native_replacement, jax_name, jax_replacement = {
            "LBFGS": ("minimize", failed_minimize, "minimize", failed_minimize),
            "BFGS": ("minimize", failed_minimize, "boozer_penalty_bfgs", failed_device_solve(boozer_penalty_bfgs)),
            "L-BFGS-B": (
                "minimize", failed_minimize, "boozer_penalty_lbfgsb", failed_device_solve(boozer_penalty_lbfgsb)
            ),
            "ls": ("least_squares", failed_least_squares, "_least_squares", failed_least_squares),
        }[solver]
        patches.enter_context(mock.patch(f"simsopt.geo.boozersurface.{native_name}", native_replacement))
        patches.enter_context(mock.patch.object(jax_boozer_surface, jax_name, jax_replacement))
        native, jax_problem = _pair("ls")
        start = native.boozer.surface.get_dofs()
        results = []
        for problem in (native, jax_problem):
            options = {"constraint_weight": _WEIGHT, "iota": _IOTA, "G": problem.G0, "weight_inv_modB": False}
            if solver != "ls":
                if solver == "LBFGS" and problem is jax_problem:
                    options["host_bfgs"] = True
                res = problem.boozer.minimize_boozer_penalty_constraints_LBFGS(
                    tol=1e-10, maxiter=10, limited_memory=solver == "L-BFGS-B", **options
                )
            else:
                res = problem.boozer.minimize_boozer_penalty_constraints_ls(tol=1e-12, maxiter=10, method="lm",
                                                                            **options)
            self.assertFalse(res["success"])
            moved = np.max(np.abs(problem.boozer.surface.get_dofs() - start))
            if delta is None:
                self.assertGreater(moved, 0.0, "a lower failed iterate was not kept")
            else:
                self.assertEqual(moved, 0.0, "a worse failed iterate was kept")
                self.assertEqual((res["iota"], res["G"]), (_IOTA, problem.G0))
            results.append(res)
        arrays = ("residual", "gradient", "jacobian") if solver == "ls" else ("fun", "gradient")
        _assert_same_solve(jax_problem, native, results[1], results[0], arrays, f"{solver} delta={delta}", 1e-10)

    def test_singular_step_after_a_step_leaves_the_last_iterate_as_natively(self):
        """A singular second factorisation raises with the surface at the first iterate, as native leaves it."""
        # Native moves the surface to every iterate before factorising there, so the solve raises LinAlgError with
        # need_to_run_code set (BoozerExact, penalty Newton and the damped Gauss-Newton). The JAX loop takes one real
        # step and its result is then marked singular: this checks what the adapter does with a reached iterate, not
        # the loop's detection (TestBoozerSolverLoopsJax below checks that). The starts are ones where the first step
        # lowers the norm, so the one-step loop keeps its iterate. The BoozerExact and penalty Newton steps at this
        # gradient norm solve twice and once.
        for method in ["exact", "newton", "manual"]:
            with self.subTest(method=method), self.case() as patches:
                self._case_singular_step_after_a_step_leaves_the_last_iterate_as_natively(method, patches)

    def _case_singular_step_after_a_step_leaves_the_last_iterate_as_natively(self, method, patches):
        native, jax_problem = _pair("exact" if method == "exact" else "ls")
        # From the raw start and early BFGS iterates the first penalty Newton step
        # raises the gradient norm; from this iterate it lowers it (2e-3 to 1e-6).
        iota, G = _same_start(native, jax_problem, 200) if method == "newton" else (_IOTA, native.G0)
        start = native.boozer.surface.get_dofs()
        solver_name, solves_per_step = {
            "exact": ("boozer_exact_newton", 2),
            "newton": ("boozer_penalty_newton", 1),
            "manual": ("boozer_penalty_gauss_newton", 1),
        }[method]
        _singular_at_second_factorisation(patches, solver_name, solves_per_step)
        for problem in (native, jax_problem):
            options = dict(tol=1e-11, maxiter=5, iota=iota, G=G)
            with self.assertRaisesRegex(np.linalg.LinAlgError, "Singular matrix"):
                if method == "exact":
                    problem.boozer.solve_residual_equation_exactly_newton(**options)
                elif method == "newton":
                    problem.boozer.minimize_boozer_penalty_constraints_newton(
                        **options, constraint_weight=_WEIGHT, weight_inv_modB=False
                    )
                else:
                    problem.boozer.minimize_boozer_penalty_constraints_ls(
                        **options, constraint_weight=_WEIGHT, weight_inv_modB=False, method="manual"
                    )
            self.assertTrue(problem.boozer.need_to_run_code)
        self.assertTrue(np.max(np.abs(native.boozer.surface.get_dofs() - start)) > 1e-6,
                        "the first step did not move the surface")
        assert_matches_native(jax_problem.boozer.surface.get_dofs(), native.boozer.surface.get_dofs(),
                              f"{method} surface")

    def test_least_squares_methods_match_native(self):
        """Upstream's problem from its BFGS start; Gauss-Newton's result is not stored, ``lm`` is capped at 8 calls."""
        # As natively the damped Gauss-Newton result is returned but not stored; at 8 evaluations lm's walk still
        # agrees to round-off.
        def check(method):
            native, jax_problem = _least_squares_problem(False), _least_squares_problem(True)
            constraint_weight = 1000.0 / (3 * 11 * 11)
            iota, G = _same_start(native, jax_problem, 700, constraint_weight, tol=1e-12, iota=-0.4,
                                  limited_memory=True, weight_inv_modB=True)
            native_res, jax_res = (
                problem.boozer.minimize_boozer_penalty_constraints_ls(
                    tol=1e-8, maxiter=50 if method == "manual" else 8, constraint_weight=constraint_weight, iota=iota,
                    G=G, method=method,
                )
                for problem in (native, jax_problem)
            )
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), method, 1e-10)
            # At the solution the residuals and J^T r are small differences of O(1) terms.
            assert_matches_native(jax_res["residual"], native_res["residual"], f"{method} residual", 1e-12, scale=1.0)
            assert_matches_native(jax_res["gradient"], native_res["gradient"], f"{method} gradient", 1e-11, scale=1.0)
            self.assertEqual(jax_res["success"], native_res["success"])
            self.assertIs(jax_res["s"], jax_problem.boozer.surface)
            if method == "manual":
                self.assertTrue(native_res["success"])
                self.assertTrue(jax_problem.boozer.need_to_run_code)
            else:
                self.assertEqual(jax_res["info"].nfev, native_res["info"].nfev)
                self.assertIs(jax_problem.boozer.res, jax_res)

        self.run_cases(check, [dict(method="manual"), dict(method="lm")])

    def test_penalty_residual_matches_native(self):
        """The least-squares formulation: native ``_get_residual_vector_and_jacobian`` with a label on its own grid."""
        def check(case_id_0, optimize_G, case_id_1, weight_inv_modB):
            native, jax_problem = _pair("ls", "ToroidalFlux", label_grid=(31, 31))
            boozer = jax_problem.boozer
            problem = boozer_problem(boozer.biotsavart, boozer.surface, boozer.label, boozer.targetlabel, 7.0)
            G0 = native.G0
            self.assertIsNotNone(G0)
            x = np.concatenate((boozer.surface.get_dofs(), [_IOTA, G0][: 1 + optimize_G]))
            expected = native.boozer._get_residual_vector_and_jacobian(x.copy(), 7.0, optimize_G, weight_inv_modB)
            placed = place_float64(x, problem.targetlabel)
            for derivatives in (0, 1):
                actual = boozer_penalty_residual(problem, placed, derivatives=derivatives, optimize_G=optimize_G,
                                                 weight_inv_modB=weight_inv_modB)
                for order in range(derivatives + 1):
                    assert_matches_native(actual[order], expected[order], f"penalty residual [{derivatives}][{order}]")

        self.run_cases(check, [
            dict(case_id_0=G_id, optimize_G=optimize_G, case_id_1=weight_id, weight_inv_modB=weight_inv_modB)
            for G_id, optimize_G in (("G", True), ("G-from-currents", False))
            for weight_id, weight_inv_modB in (("weighted", True), ("unweighted", False))
        ])

    def test_need_to_run_code_caches_results_as_natively(self):
        """Cached solves are reused and parent invalidation requests a fresh solve as natively."""
        jax_problem = _problem(True, "exact")
        boozer = jax_problem.boozer
        res = boozer.run_code(_IOTA, G=jax_problem.G0)
        self.assertIsNone(boozer.run_code(_IOTA, G=jax_problem.G0))
        for solve in (
            boozer.solve_residual_equation_exactly_newton, boozer.minimize_boozer_penalty_constraints_LBFGS,
            boozer.minimize_boozer_penalty_constraints_newton, boozer.minimize_boozer_penalty_constraints_ls,
        ):
            self.assertIs(solve(iota=0.3), res)
        jax_problem.currents[0].local_full_x = 1.01 * jax_problem.currents[0].local_full_x
        self.assertTrue(boozer.need_to_run_code, "a coil change does not mark the surface for a new solve")

    def test_native_objectives_use_the_jax_surface(self):
        """Native objectives on solved JAX surfaces give native values and coil gradients (via ``PLU`` and ``vjp``)."""
        # With a Volume label and G optimized or the currents fixed, JaxBoozerResidual is native BoozerResidual too,
        # also after a coil change that makes both re-solve. On a converged BoozerExact surface the Boozer residual is
        # round-off (J about 1e-29), so BoozerResidual is compared on capped BoozerExact solves instead
        # (test_boozer_residual_on_capped_boozer_exact_solves_matches_native).
        def check(name):
            boozer_type, label, optimize_G, weight_inv_modB = _OBJECTIVE_CASES[name]
            native, jax_problem = _pair(boozer_type, label, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
                                        label_grid=(51, 51) if label == "ToroidalFlux" else None)
            for problem in (native, jax_problem):
                problem.boozer.run_code(_IOTA, G=problem.G0)
            objectives = {
                objective_name: make for objective_name, make in _NATIVE_OBJECTIVES.items()
                if boozer_type == "ls" or objective_name != "BoozerResidual"
            }
            for objective_name, make in objectives.items():
                native_objective, jax_objective = make(native), make(jax_problem)
                assert_matches_native(jax_objective.J(), native_objective.J(), f"{name} {objective_name}")
                assert_matches_native(jax_objective.dJ(), native_objective.dJ(),
                                      f"{name} {objective_name} coil gradient")
            if boozer_type == "exact":
                return
            native_residual = BoozerResidual(native.boozer, BiotSavart(native.coils))
            jax_residual = JaxBoozerResidual(jax_problem.boozer, jax_problem.boozer.biotsavart)
            for change in ("solved", "after a coil change"):
                value = jax_residual.J()
                self.assertTrue(isinstance(value, np.float64), f"{name}: J is {type(value)}, native's is np.float64")
                assert_matches_native(value, native_residual.J(), f"{name} JaxBoozerResidual {change}")
                assert_matches_native(jax_residual.dJ(), native_residual.dJ(),
                                      f"{name} JaxBoozerResidual gradient {change}")
                for problem in (native, jax_problem):
                    problem.currents[0].local_full_x = 1.0001 * problem.currents[0].local_full_x
                self.assertTrue(jax_problem.boozer.need_to_run_code and jax_residual._J is None,
                                "a coil change left a stale value")

        self.run_cases(check, [dict(name=name) for name in _OBJECTIVE_CASES])

    def test_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(self):
        """On Area and AspectRatio labels (``G`` optimized) ``JaxBoozerResidual`` is native ``BoozerResidual``."""
        # Native BoozerResidual is the derivative of its value there. Both are compared at a shared state: a native
        # BFGS iterate given to both surfaces, stored with its Hessian by a penalty Newton call with maxiter=0 (these
        # labels' run_code walks follow round-off to different solutions, and the penalty Newton from the BFGS iterate
        # does not converge). The same solved surface isolates the objective's derivative from the ill-conditioned
        # Hessian's amplification of different solver round-off.
        def check(label):
            native, jax_problem = _pair("ls", label)
            iota, G = _same_start(native, jax_problem, 200)
            for problem in (native, jax_problem):
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11, maxiter=0, constraint_weight=_WEIGHT, iota=iota, G=G, weight_inv_modB=False
                )
            jax_residual = JaxBoozerResidual(jax_problem.boozer, jax_problem.boozer.biotsavart)
            # Native BoozerResidual on the same JAX surface (same res, PLU and vjp): the objective alone.
            on_jax_surface = BoozerResidual(jax_problem.boozer, BiotSavart(jax_problem.coils))
            assert_matches_native(jax_residual.J(), on_jax_surface.J(), f"{label} JaxBoozerResidual")
            assert_matches_native(jax_residual.dJ(), on_jax_surface.dJ(), f"{label} JaxBoozerResidual gradient")

        self.run_cases(check, [dict(label="Area"), dict(label="AspectRatio")])

    def test_boozer_residual_owns_its_surface_and_reads_changed_values_with_a_copied_field(self):
        """The residual's own surface copy, a copied field and new target/weight values, without recompilation."""
        # The residual copy uses the solved quadrature; a label sharing the solved DOFs uses its own grid. A copied
        # field follows later coil changes, and recomputing replaces private-copy edits.
        problem = _problem(True, "ls", label_grid=(31, 31))
        boozer = problem.boozer
        boozer.minimize_boozer_penalty_constraints_newton(
            maxiter=0, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0, weight_inv_modB=False
        )
        field = copy.copy(boozer.biotsavart)
        residual = JaxBoozerResidual(boozer, field)
        native = BoozerResidual(boozer, BiotSavart(problem.coils))
        self.assertIsNot(residual.surface.dofs, boozer.surface.dofs)
        self.assertIs(boozer.label.surface.dofs, boozer.surface.dofs)
        self.assertEqual(boozer.label.surface.quadpoints_phi.size, 31)
        np.testing.assert_array_equal(residual.surface.quadpoints_phi, boozer.surface.quadpoints_phi)
        np.testing.assert_array_equal(residual.surface.quadpoints_theta, boozer.surface.quadpoints_theta)
        solved_dofs = boozer.surface.get_dofs().copy()
        first = residual.J()
        assert_matches_native(first, native.J(), "private surface and label grids")
        assert_matches_native(residual.dJ(), native.dJ(), "copied field gradient")
        copied = copy.copy(residual)
        self.assertIsNot(copied.surface.dofs, residual.surface.dofs)
        self.assertIsNone(copied._J)
        self.assertIsNone(copied._dJ)
        self.assertEqual(copied.constraint_weight, residual.constraint_weight)
        assert_matches_native(copied.J(), native.J(), "copied objective")
        assert_matches_native(copied.dJ(), native.dJ(), "copied objective gradient")
        residual.surface.set_dofs(1.1 * solved_dofs)
        np.testing.assert_array_equal(boozer.surface.get_dofs(), solved_dofs)
        residual.constraint_weight = 2.0 * _WEIGHT
        native.constraint_weight = residual.constraint_weight
        boozer.targetlabel *= 1.01
        residual.recompute_bell()
        native.recompute_bell()
        with jax_compilations() as compilations:
            changed = residual.J()
            gradient = residual.dJ()
        self.assertEqual(compilations, [], "new residual target or weight recompiled the objective")
        self.assertNotEqual(changed, first)
        np.testing.assert_array_equal(residual.surface.get_dofs(), solved_dofs)
        assert_matches_native(changed, native.J(), "changed target and weight")
        assert_matches_native(gradient, native.dJ(), "changed target and weight gradient")
        problem.currents[0].local_full_x *= 1.001
        self.assertTrue(boozer.need_to_run_code)
        self.assertIsNone(residual._J)
        self.assertIsNone(residual._dJ)
        self.assertTrue(copied._J is None and copied._dJ is None, "a copied objective kept stale coil caches")
        boozer.minimize_boozer_penalty_constraints_newton(
            maxiter=0, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0, weight_inv_modB=False
        )
        assert_matches_native(residual.J(), native.J(), "copied field after a coil change")
        assert_matches_native(residual.dJ(), native.dJ(), "copied field gradient after a coil change")
        fresh = JaxBoozerResidual(boozer, boozer.biotsavart)
        assert_matches_native(copied.J(), fresh.J(), "copied objective after a coil change")
        assert_matches_native(copied.dJ(), fresh.dJ(), "copied objective gradient after a coil change")

    def test_boozer_residual_keeps_the_solved_surface_representation(self):
        """On XYZ and clamped tensor BoozerLS solves ``JaxBoozerResidual`` is native's on the solved surface itself."""
        # Volume label, G optimized, where native's gradient is the derivative. An unclamped tensor copy of these DOFs
        # is a different surface.
        _, base_currents, axis, nfp, bs = get_data("ncsx")
        G0 = _G0(nfp, base_currents)
        grid = dict(quadpoints_phi=np.linspace(0, 1 / nfp, 20, endpoint=False),
                    quadpoints_theta=np.linspace(0, 1, 20, endpoint=False))
        surfaces = {
            "SurfaceXYZFourier": SurfaceXYZFourier(mpol=3, ntor=3, stellsym=True, nfp=nfp, **grid),
            "clamped SurfaceXYZTensorFourier": SurfaceXYZTensorFourier(
                mpol=3, ntor=3, stellsym=True, nfp=nfp, clamped_dims=[True, False, False], **grid
            ),
        }
        for name, surface in surfaces.items():
            with self.subTest(surface=name), self.case():
                surface.fit_to_curve(axis, 0.1, flip_theta=True)
                label = Volume(surface)
                boozer = JaxBoozerSurface(JaxBiotSavart(bs.coils), surface, label, label.J(), _WEIGHT,
                                          {"verbose": False, "weight_inv_modB": False})
                boozer.minimize_boozer_penalty_constraints_newton(
                    maxiter=0, constraint_weight=_WEIGHT, iota=_IOTA, G=G0, weight_inv_modB=False
                )
                residual = JaxBoozerResidual(boozer, boozer.biotsavart)
                self.assertIs(type(residual.surface), type(surface))
                np.testing.assert_array_equal(residual.surface.gamma(), surface.gamma())
                unclamped = SurfaceXYZTensorFourier(mpol=3, ntor=3, stellsym=True, nfp=nfp, **grid)
                unclamped.set_dofs(surface.get_dofs())
                self.assertGreater(np.abs(unclamped.gamma() - surface.gamma()).max(), 0.1)
                on_solved_surface = BoozerResidual(boozer, BiotSavart(bs.coils))
                on_solved_surface.surface = surface
                assert_matches_native(residual.J(), on_solved_surface.J(), f"{name} JaxBoozerResidual")
                assert_matches_native(residual.dJ(), on_solved_surface.dJ(), f"{name} JaxBoozerResidual gradient")

    def test_ls_adjoint_with_a_toroidal_flux_label_is_the_derivative_of_the_solve(self):
        """With a ToroidalFlux label, JAX-surface coil gradients are central differences through native re-solves."""
        # The BoozerLS penalty depends on the coils through the label too; native's own adjoint
        # (boozer_surface_dlsqgrad_dcoils_vjp) drops that term and is not the derivative. BoozerResidual's value also
        # depends on the coils through the label: JaxBoozerResidual differentiates it, native BoozerResidual takes its
        # explicit derivative through B only. Re-solves are run_code from the solution; the centered differences then
        # converge to the adjoints at halving steps.
        native, jax_problem, native_objectives, jax_objectives = _toroidal_flux_adjoint_problem()
        jax_handle, native_handle = Iotas(jax_problem.boozer), Iotas(native.boozer)
        # Every free coil DOF moves, by a fraction of its own size.
        x0 = np.asarray(native_handle.x, dtype=np.float64)
        direction = scaled_direction(x0, 11)
        step = 1e-6

        def values(sign: float) -> dict[str, float]:
            native_handle.x = x0 + sign * step * direction
            return {name: objective.J() for name, objective in native_objectives.items()}

        solution = native.boozer.surface.get_dofs().copy()
        solved_iota, solved_G = native.boozer.res["iota"], native.boozer.res["G"]

        def restarted_values(sign):
            native.boozer.surface.set_dofs(solution)
            native.boozer.res["iota"], native.boozer.res["G"] = solved_iota, solved_G
            evaluated = values(sign)
            self.assertTrue(native.boozer.res["success"])
            return evaluated

        plus, minus = restarted_values(1.0), restarted_values(-1.0)
        native_handle.x = x0
        adjoints = {}
        for name, make in jax_objectives.items():
            central = (plus[name] - minus[name]) / (2 * step)
            with disallow_host_transfers():
                partials = make(jax_problem).dJ(partials=True)
            adjoint = adjoints[name] = partials(jax_handle) @ direction
            native_adjoint = native_objectives[name].dJ(partials=True)(native_handle) @ direction
            # Truncation and the re-solves' tolerances bound the agreement (measured 2e-9 to 2e-8 relative);
            # native's gradients are off by 1.8e-3 to 3e-2.
            self.assertTrue(abs(adjoint - central) <= 1e-7 * abs(central),
                            f"{name}: adjoint {adjoint} != difference {central}")
            self.assertTrue(abs(native_adjoint - central) > 1e-4 * abs(central),
                            f"{name}: native's adjoint matches the difference")

        errors = {name: [] for name in adjoints}
        for step in np.power(2.0, -np.arange(13, 18)):
            plus, minus = restarted_values(1.0), restarted_values(-1.0)
            for name, adjoint in adjoints.items():
                errors[name].append(abs((plus[name] - minus[name]) / (2 * step) - adjoint))
        for name, series in errors.items():
            assert_errors_shrink(self, series, 0.35, floor=1e-9, msg=name, initial=1e9)
        native_handle.x = x0
        native.boozer.surface.set_dofs(solution)

    def _free_current_adjoint_problem(self):
        """Solve the shared penalty problem with G derived from free currents."""
        native, jax_problem = _pair("ls")
        solve = dict(tol=1e-11, maxiter=40, constraint_weight=_WEIGHT, G=None, weight_inv_modB=False)
        start = native.boozer.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=1500, constraint_weight=_WEIGHT, iota=_IOTA, G=None, limited_memory=False,
            weight_inv_modB=False,
        )
        jax_problem.boozer.surface.set_dofs(native.boozer.surface.get_dofs())
        for problem in (native, jax_problem):
            problem.boozer.need_to_run_code = True
            res = problem.boozer.minimize_boozer_penalty_constraints_newton(iota=start["iota"], **solve)
            self.assertTrue(res["success"], "the penalty Newton with G from the currents did not converge")
        return native, jax_problem, solve

    def test_gradients_with_G_from_free_currents_are_the_derivatives_of_the_solve(self):
        """Solved with ``G=None`` and free currents, coil gradients are central differences through native re-solves."""
        # run_code refuses this; the solvers do not, and G follows the currents. The coil gradients of Iotas and
        # JaxBoozerResidual on a JaxBoozerSurface match differences through native re-solves (penalty Newton from the
        # solution); native's own drop dG/dI. The centered differences then converge to the adjoints at halving steps.
        native, jax_problem, solve = self._free_current_adjoint_problem()
        solution, iota = native.boozer.surface.get_dofs(), native.boozer.res["iota"]
        native_objectives = {
            "Iotas": Iotas(native.boozer), "BoozerResidual": BoozerResidual(native.boozer, BiotSavart(native.coils))
        }
        jax_objectives = {
            "Iotas": Iotas(jax_problem.boozer),
            "BoozerResidual": JaxBoozerResidual(jax_problem.boozer, jax_problem.boozer.biotsavart),
        }
        native_handle, jax_handle = native_objectives["Iotas"], jax_objectives["Iotas"]
        x0 = np.asarray(native_handle.x, dtype=np.float64)
        direction = scaled_direction(x0, 13)
        self.assertGreater(x0.size, 0)
        self.assertTrue(any(not current.dofs.all_fixed() for current in native.currents))
        # Central-difference truncation is 9e-6 relative at 1e-5 and 9e-8 at 1e-6 (Iotas), so the step is 1e-7.
        step = 1e-7

        def values(sign: float) -> dict[str, float]:
            native_handle.x = x0 + sign * step * direction
            native.boozer.surface.set_dofs(solution)
            native.boozer.minimize_boozer_penalty_constraints_newton(iota=iota, **solve)
            return {name: objective.J() for name, objective in native_objectives.items()}

        def restarted_values(sign):
            evaluated = values(sign)
            self.assertTrue(native.boozer.res["success"])
            return evaluated

        plus, minus = restarted_values(1.0), restarted_values(-1.0)
        native_handle.x = x0
        native.boozer.surface.set_dofs(solution)
        native.boozer.minimize_boozer_penalty_constraints_newton(iota=iota, **solve)
        adjoints = {}
        for name, objective in jax_objectives.items():
            central = (plus[name] - minus[name]) / (2 * step)
            with disallow_host_transfers():
                partials = objective.dJ(partials=True)
            adjoint = adjoints[name] = partials(jax_handle) @ direction
            native_adjoint = native_objectives[name].dJ(partials=True)(native_handle) @ direction
            # Truncation and the re-solves' tolerances bound the agreement (measured 6e-9 and 9e-11 relative);
            # native's gradients are off by 2.3e-4 and 7.1e-4.
            self.assertTrue(abs(adjoint - central) <= 1e-7 * abs(central),
                            f"{name}: adjoint {adjoint} != difference {central}")
            self.assertTrue(abs(native_adjoint - central) > 1e-4 * abs(central),
                            f"{name}: native's gradient matches the difference")

        errors = {name: [] for name in adjoints}
        for step in np.power(2.0, -np.arange(13, 18)):
            plus, minus = restarted_values(1.0), restarted_values(-1.0)
            for name, adjoint in adjoints.items():
                errors[name].append(abs((plus[name] - minus[name]) / (2 * step) - adjoint))
        for name, series in errors.items():
            assert_errors_shrink(self, series, 0.35, floor=1e-9, msg=name, initial=1e9)
        native_handle.x = x0
        native.boozer.surface.set_dofs(solution)

    def test_exact_adjoint_without_stellsym_matches_native(self):
        """Without stellsym the exact system has a ``z(0, 0)`` row after the label row: the ``vjp`` matches native's."""
        native, jax_problem = _pair("exact", "ToroidalFlux", stellsym=False, newton_tol=1e-10)
        for problem in (native, jax_problem):
            problem.boozer.run_code(_IOTA, G=problem.G0)
        res, jax_res = native.boozer.res, jax_problem.boozer.res
        lm = parity_rng(3).standard_normal(res["jacobian"].shape[0])
        expected = boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, native.boozer, res["iota"], res["G"])
        actual = jax_res["vjp"](lm, jax_problem.boozer, jax_res["iota"], jax_res["G"])
        assert_matches_native(actual(BiotSavart(jax_problem.coils)), expected(native.boozer.biotsavart),
                              "non-stellsym exact vjp", _RTOL_NONSYM)

    def test_boozer_residual_on_capped_boozer_exact_solves_matches_native(self):
        """``BoozerResidual`` after two BoozerExact steps on the JAX surface: native's value and zero coil gradient."""
        # Two Newton steps leave the residual above round-off; with and without stellarator symmetry and a ToroidalFlux
        # label, native BoozerResidual and JaxBoozerResidual on the JAX surface give native's value and coil gradient
        # (through the exact PLU and vjp). The residual depends on the coils and the surface only through the Newton
        # system's rows, which the adjoint holds fixed, so the gradient is zero: its explicit and adjoint terms cancel
        # (to 1e-15 of the explicit term here), and it is compared on the scale of those terms.
        def check(stellsym):
            native, jax_problem = _pair("exact", "ToroidalFlux", stellsym=stellsym)
            for problem in (native, jax_problem):
                problem.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=2, iota=_IOTA, G=problem.G0)
            rtol = _RTOL if stellsym else _RTOL_NONSYM
            native_residual = BoozerResidual(native.boozer, BiotSavart(native.coils))
            native_gradient = native_residual.dJ()
            field = native_residual.biotsavart
            # Native's B_vjp sums Derivatives, which type checkers read as an int.
            explicit_term = cast(Derivative, field.B_vjp(native_residual.dJ_by_dB()))
            explicit = np.max(np.abs(host_array(explicit_term(field), dtype=np.float64)))
            assert_matches_native(native_gradient, np.zeros_like(native_gradient), "native gradient", rtol, explicit)
            for name, residual in (
                ("BoozerResidual", BoozerResidual(jax_problem.boozer, BiotSavart(jax_problem.coils))),
                ("JaxBoozerResidual", JaxBoozerResidual(jax_problem.boozer, jax_problem.boozer.biotsavart)),
            ):
                assert_matches_native(residual.J(), native_residual.J(), name, rtol)
                assert_matches_native(residual.dJ(), native_gradient, f"{name} gradient", rtol, explicit)

        self.run_cases(check, [dict(stellsym=True), dict(stellsym=False)])

    def test_exact_adjoint_is_the_derivative_of_the_solve_taylor(self):
        """BoozerExact coil gradients are second-order limits of central differences through re-solves."""
        # A ToroidalFlux label puts coil dependence in the label row of the system too.
        problem = _problem(True, "exact", "ToroidalFlux", label_grid=(31, 31), newton_tol=1e-12)
        boozer = problem.boozer
        self.assertTrue(boozer.run_code(_IOTA, G=problem.G0)["success"])
        objectives = {"Iotas": Iotas(boozer), "MajorRadius": MajorRadius(boozer)}
        handle = objectives["Iotas"]
        # Every free coil DOF moves, by a fraction of its own size.
        x0 = np.asarray(handle.x, dtype=np.float64)
        direction = scaled_direction(x0, 17)
        gradients = {name: objective.dJ(partials=True)(handle) @ direction for name, objective in objectives.items()}
        solution = boozer.surface.get_dofs().copy()
        solved_iota, solved_G = boozer.res["iota"], boozer.res["G"]

        def values(x: np.ndarray) -> dict[str, float]:
            """The objectives after a Newton re-solve from the solution at coils ``x``."""
            boozer.surface.set_dofs(solution)
            boozer.res["iota"], boozer.res["G"] = solved_iota, solved_G
            handle.x = x
            evaluated = {name: objective.J() for name, objective in objectives.items()}
            self.assertTrue(boozer.res["success"], "the re-solve did not converge")
            return evaluated

        errors = {name: [] for name in objectives}
        for eps in 1e-2 * 0.5 ** np.arange(8):
            plus, minus = values(x0 + eps * direction), values(x0 - eps * direction)
            for name, gradient in gradients.items():
                errors[name].append(abs((plus[name] - minus[name]) / (2 * eps) - gradient))
        for name, gradient in gradients.items():
            # Central differences: the error falls by 4 per halving (measured 0.248 to 0.250).
            assert_errors_shrink(self, errors[name], 0.3, msg=name, initial=np.inf)
            # Measured 2.6e-5 (Iotas) and 2.9e-6 (MajorRadius) relative at the last step.
            self.assertLess(errors[name][-1], 1e-4 * abs(gradient), name)

    def test_fixed_surface_dofs_are_solved_for_as_natively(self):
        """Fixed surface DOFs are solved for as natively, with native's ``dlabel_dsurface`` in all of them."""
        # Native BoozerSurface solves for every surface DOF, fixed or free, with the label differentiated in all of
        # them, and so does the JAX solve: the penalty Newton from a BFGS iterate with a fixed DOF matches native's.
        native, jax_problem = _pair("ls")
        for problem in (native, jax_problem):
            problem.boozer.surface.fix("x(0,0)")
        iota, G = _same_start(native, jax_problem, 200)
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11, maxiter=20, constraint_weight=_WEIGHT, iota=iota, G=G, weight_inv_modB=False
            )
            for problem in (native, jax_problem)
        )
        self.assertTrue(native_res["success"])
        _assert_same_solve(jax_problem, native, jax_res, native_res, ("hessian",), "fixed DOFs")
        assert_matches_native(jax_problem.boozer.dlabel_dsurface(), native.boozer.dlabel_dsurface(),
                              "label gradient with fixed DOFs")

    def test_settings_are_read_at_every_solve(self):
        """The target, weight, options and coils in force at a solve are used; the caller's options are not modified."""
        # BoozerLS stops BFGS early and takes no Newton step, so its run_code result is deterministic.
        options = {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10}
        native, jax_problem = _pair("exact", options=options)
        ls_native, ls_jax = _pair("ls")
        for problem in (ls_native, ls_jax):
            problem.boozer.options.update(bfgs_maxiter=6, newton_maxiter=0)
        for problems in ((native, jax_problem), (ls_native, ls_jax)):
            for problem in problems:
                problem.boozer.run_code(_IOTA, G=problem.G0)
                problem.boozer.targetlabel *= 1.02
                problem.boozer.constraint_weight = None if problem.boozer.constraint_weight is None else 30.0
                problem.boozer.options["newton_maxiter"] = 2 if problem.boozer.boozer_type == "exact" else 0
                problem.currents[0].local_full_x = 1.002 * problem.currents[0].local_full_x
            native_res, jax_res = (
                problem.boozer.run_code(problem.boozer.res["iota"], G=problem.boozer.res["G"]) for problem in problems
            )
            self.assertFalse(native_res["success"])
            self.assertEqual(native_res["iter"], 2 if native_res["type"] == "exact" else 0)
            _assert_same_solve(problems[1], problems[0], jax_res, native_res, ("jacobian",), "new settings", 1e-10)
        self.assertEqual(options, {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10})

    def test_solves_preserve_the_field_point_buffer(self):
        """Solves, re-solves and coil adjoints keep the field's point buffer (native moves it to the surface)."""
        exact = _problem(True, "exact", newton_tol=1e-10)
        ls = _problem(True, "ls")
        ls.boozer.options.update(bfgs_maxiter=6, newton_maxiter=1)
        objectives = ((exact, Iotas(exact.boozer)), (ls, JaxBoozerResidual(ls.boozer, ls.boozer.biotsavart)))
        for problem, objective in objectives:
            field = problem.boozer.biotsavart
            points = parity_rng(5).uniform(-0.5, 0.5, (7, 3)) + [1.0, 0.0, 0.0]
            field.set_points(points)
            buffer, B = field.get_points_cart_ref(), np.asarray(field.B())
            problem.boozer.run_code(_IOTA, G=problem.G0)
            objective.dJ()
            # The coil change marks the surface for a re-solve, which the objective runs.
            current = problem.currents[0].local_full_x
            problem.currents[0].local_full_x = 1.002 * current
            objective.dJ()
            name = problem.boozer.boozer_type
            self.assertIs(field.get_points_cart_ref(), buffer, f"{name}: point buffer")
            np.testing.assert_array_equal(field.get_points_cart(), points, name)
            problem.currents[0].local_full_x = current
            np.testing.assert_array_equal(np.asarray(field.B()), B, name)

    def test_new_values_reuse_the_compiled_programs(self):
        """New coils, targets, weights, starting points, tolerances, caps and ``stab`` reuse every program."""

        def solve_all(exact_problem: _Problem, ls_problem: _Problem, settings: dict):
            exact, ls = exact_problem.boozer, ls_problem.boozer
            ls.constraint_weight = settings["weight"]
            res = exact.solve_residual_equation_exactly_newton(
                tol=settings["tol"], maxiter=settings["maxiter"], iota=settings["iota"], G=exact_problem.G0
            )
            Iotas(exact).dJ()
            for limited_memory in (True, False):
                ls.recompute_bell()
                start = ls.minimize_boozer_penalty_constraints_LBFGS(
                    tol=1e-10, maxiter=settings["maxiter"], constraint_weight=settings["weight"],
                    iota=settings["iota"], G=ls_problem.G0, limited_memory=limited_memory, weight_inv_modB=False,
                )
            for method in ("manual", "lm"):
                ls.recompute_bell()
                ls.minimize_boozer_penalty_constraints_ls(
                    tol=settings["tol"], maxiter=3, constraint_weight=settings["weight"], iota=start["iota"],
                    G=start["G"], method=method, weight_inv_modB=False,
                )
            ls.recompute_bell()
            ls.minimize_boozer_penalty_constraints_newton(
                tol=settings["tol"], maxiter=2, constraint_weight=settings["weight"], iota=start["iota"], G=start["G"],
                stab=settings["stab"], weight_inv_modB=False,
            )
            Iotas(ls).dJ()
            JaxBoozerResidual(ls, ls.biotsavart).dJ()
            for boozer in (exact, ls):
                boozer.recompute_bell()
            return res

        exact_problem, ls_problem = _problem(True, "exact"), _problem(True, "ls")
        first = solve_all(
            exact_problem, ls_problem, {"tol": 1e-10, "maxiter": 12, "iota": _IOTA, "weight": _WEIGHT, "stab": 0.0}
        )
        for problem in (exact_problem, ls_problem):
            problem.currents[0].local_full_x = 1.003 * problem.currents[0].local_full_x
            problem.boozer.surface.set_dofs(problem.boozer.surface.get_dofs() * 1.001)
            problem.boozer.targetlabel *= 1.01
        with jax_compilations() as compilations:
            second = solve_all(
                exact_problem, ls_problem, {"tol": 2e-10, "maxiter": 15, "iota": -0.41, "weight": 50.0, "stab": 1e-4}
            )
        self.assertEqual(compilations, [], "new values retraced or recompiled a solver")
        self.assertNotEqual(second["iota"], first["iota"])

    def test_solves_make_no_implicit_transfers(self):
        """Exact and penalty solves use explicit transfers under the runtime transfer guard."""
        def check(case_id_0, parity_lane, boozer_type):
            native = _problem(False, boozer_type)
            with parity_default_device(parity_lane):
                jax_problem = _problem(True, boozer_type)
                with disallow_host_transfers():
                    jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
                    jax_dJ = Iotas(jax_problem.boozer).dJ()
                    if boozer_type == "ls":
                        jax_residual_dJ = JaxBoozerResidual(jax_problem.boozer, jax_problem.boozer.biotsavart).dJ()
                        jax_problem.boozer.recompute_bell()
                        jax_problem.boozer.minimize_boozer_penalty_constraints_ls(
                            tol=1e-8, maxiter=2, constraint_weight=_WEIGHT, iota=jax_res["iota"], G=jax_res["G"],
                            method="lm", weight_inv_modB=False,
                        )
            coils = jax_problem.boozer.biotsavart.coil_set_spec()
            self.assertEqual({device.platform for leaf in jax.tree.leaves(coils) for device in leaf.devices()},
                             {parity_lane})
            native_res = native.boozer.run_code(_IOTA, G=native.G0)
            assert_matches_native(jax_res["iota"], native_res["iota"], f"{boozer_type} iota on {parity_lane}", 1e-10)
            assert_matches_native(jax_dJ, Iotas(native.boozer).dJ(), f"{boozer_type} Iotas gradient on {parity_lane}",
                                  1e-9)
            if boozer_type == "ls":
                native_residual_dJ = BoozerResidual(native.boozer, BiotSavart(native.coils)).dJ()
                assert_matches_native(jax_residual_dJ, native_residual_dJ,
                                      f"JaxBoozerResidual gradient on {parity_lane}", 1e-9)

        self.run_cases(check, [dict(**lane, boozer_type=kind) for lane in _LANE_CASES for kind in ("exact", "ls")])

    def test_boundary_refuses_unsupported_inputs(self):
        """Solver and residual construction reject unsupported surface classes."""
        _, _, axis, nfp, bs = get_data("ncsx")
        field = JaxBiotSavart(bs.coils)
        rz = SurfaceRZFourier(mpol=2, ntor=2, nfp=nfp)
        with self.assertRaisesRegex(Exception, "SurfaceXYZTensorFourier or SurfaceXYZFourier"):
            JaxBoozerSurface(field, rz, Volume(rz), 1.0)

        xyz = SurfaceXYZFourier(mpol=2, ntor=2, nfp=nfp, quadpoints_phi=np.linspace(0, 1 / nfp, 6, endpoint=False),
                                quadpoints_theta=np.linspace(0, 1, 7, endpoint=False))
        xyz.fit_to_curve(axis, 0.1, flip_theta=True)
        with self.assertRaisesRegex(RuntimeError, "SurfaceXYZTensorFourier"):
            JaxBoozerSurface(field, xyz, Volume(xyz), 1.0).solve_residual_equation_exactly_newton(iota=_IOTA)
        with self.assertRaisesRegex(TypeError, "JaxBiotSavart"):
            JaxBoozerSurface(bs, xyz, Volume(xyz), 1.0, 1.0).minimize_boozer_penalty_constraints_LBFGS(maxiter=1)
        with self.assertRaisesRegex(TypeError, "labels"):
            JaxBoozerSurface(field, xyz, PrincipalCurvature(xyz), 1.0, 1.0).minimize_boozer_penalty_constraints_LBFGS(
                maxiter=1
            )
        # As natively, G from the currents needs fixed currents for coil gradients.
        with self.assertRaises(AssertionError):
            JaxBoozerSurface(field, xyz, Volume(xyz), 1.0, 1.0).run_code(_IOTA)

        class Subclassed(SurfaceXYZFourier):
            pass

        subclassed = Subclassed(mpol=2, ntor=2, nfp=nfp)
        with self.assertRaisesRegex(TypeError, "JaxBoozerResidual supports"):
            JaxBoozerResidual(JaxBoozerSurface(field, subclassed, Volume(subclassed), 1.0, 1.0), field)


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
