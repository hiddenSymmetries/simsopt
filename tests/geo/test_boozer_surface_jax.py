"""JAX ``JaxBoozerSurface`` against native ``BoozerSurface``.

Each solver of native ``BoozerSurface`` (the BoozerExact Newton, BFGS and
L-BFGS-B, the penalty Newton, ``least_squares`` and its damped Gauss-Newton,
``run_code``) runs natively and in JAX from the same start on upstream's
NCSX problems (``tests/geo/surface_test_helpers.get_boozer_surface``), with
and without stellarator symmetry, ``optimize_G``, ``weight_inv_modB`` and
labels on their own grids: iteration counts, success flags, the surface,
``iota``, ``G``, the result arrays and the ``PLU`` solves agree. Failures at
``maxiter``, diverging Newton walks and other failed solves (which restore
their start when they end worse), singular systems and the
``need_to_run_code`` cache behave as natively. Native ``Iotas``,
``MajorRadius``, ``NonQuasiSymmetricRatio`` and ``BoozerResidual`` on a
``JaxBoozerSurface`` give native values and coil gradients for coil-independent
labels with optimized ``G``. ``JaxBoozerResidual`` also differentiates the
explicit ToroidalFlux and current-derived ``G`` terms that native misses,
checked through re-solves. Then: settings
read at every solve, no recompilation for new values, no implicit transfers,
fixed surface DOFs solved for as natively, the BoozerExact adjoint without
stellarator symmetry, ``BoozerResidual`` on BoozerExact solves,
``JaxBoozerResidual`` on ``SurfaceXYZFourier`` and clamped tensor solves, and
unsupported inputs. BFGS and L-BFGS-B run SciPy's
algorithms on the device by default; the ``host_bfgs`` option runs SciPy
itself on the jitted penalty, as before, and the device programs' scratch
memory is the penalty's plus their quasi-Newton state.

Tolerances sit above round-off: iterates and arrays agree to 1e-12 of the
largest native entry (measured worst 2e-13, coil gradients of the objectives)
and, without stellarator symmetry, where the BoozerExact system is nearly
singular in one direction, to 1e-9 (measured 4e-11). BFGS and diverging
Newton walks magnify round-off, so their tests cap the walks and state their
own bounds. The solves' tolerances are set above the round-off floor of the
residuals (native 9e-14, JAX 6e-14 on these problems), below which native's
decisions follow round-off.
"""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt._core.derivative import Derivative
    from simsopt.configs import get_data
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.geo.boozersurface import BoozerSurface
    from simsopt.geo.surfaceobjectives import (
        BoozerResidual,
        Iotas,
        MajorRadius,
        NonQuasiSymmetricRatio,
        PrincipalCurvature,
        ToroidalFlux,
        Volume,
        boozer_surface_dexactresidual_dcoils_dcurrents_vjp,
    )
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
    from simsopt.objectives.utilities import forward_backward
    from simsopt_jax.core.boozer_problem import (
        boozer_penalty_constraints,
        boozer_penalty_residual,
    )
    from simsopt_jax.core.boozer_solvers import boozer_penalty_bfgs, boozer_penalty_lbfgsb
    from simsopt_jax.runtime.host_boundary import disallow_host_transfers, host_tree
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_problem import boozer_problem
    from simsopt_jax_adapters.geo import boozer_surface as jax_boozer_surface
    from simsopt_jax_adapters.geo.boozer_surface import (
        JaxBoozerResidual,
        JaxBoozerSurface,
    )
    from .surface_test_helpers import get_boozer_surface

from unittest_jax_support import (
    assert_errors_shrink,
    assert_matches_native,
    host_array,
    jax_compilations,
    parity_default_device,
    parity_rng,
    place_float64,
    scaled_direction,
)

from dataclasses import dataclass, replace
import copy
import inspect
from unittest import mock
from typing import cast

import numpy as np
from scipy.optimize import OptimizeResult, least_squares, minimize


_IOTA = -0.406
_RTOL = 1e-12
_RTOL_NONSYM = 1e-9
_WEIGHT = 100.0
# Scratch bytes the device quasi-Newton loops may add to the penalty's for their
# vectors and scalars, beyond their matrices.
_QUASI_NEWTON_SLACK = 2**20


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


def _problem(
    jax_field: bool,
    boozer_type: str = "exact",
    label: str = "Volume",
    *,
    stellsym: bool = True,
    optimize_G: bool = True,
    weight_inv_modB: bool = False,
    label_grid: tuple[int, int] | None = None,
    newton_tol: float | None = None,
    options: dict | None = None,
) -> _Problem:
    """Upstream's problem on its own NCSX coils and surface, as a native
    ``BoozerSurface`` or, with ``jax_field``, a ``JaxBoozerSurface``. Without
    ``options``, the options turn ``verbose`` off and set ``weight_inv_modB``
    and ``newton_tol`` (if given); ``options`` itself goes to
    ``JaxBoozerSurface`` and a copy to native, which fills it in place."""
    label_nphi, label_ntheta = label_grid or (None, None)
    bs, native = get_boozer_surface(
        label=label,
        nphi=label_nphi,
        ntheta=label_ntheta,
        boozer_type=boozer_type,
        optimize_G=optimize_G,
        converge=False,
        stellsym=stellsym,
        weight_inv_modB=weight_inv_modB if boozer_type == "ls" else False,
    )
    surface, label_object = native.surface, native.label
    # The first half-period contains the NCSX base currents, in their original order.
    base_currents = [
        coil.current for coil in bs.coils[:len(bs.coils) // (2 * surface.nfp)]
    ]
    G0 = _G0(surface.nfp, base_currents) if optimize_G else None
    if options is None:
        options = (
            {"verbose": False}
            if weight_inv_modB
            else {"verbose": False, "weight_inv_modB": False}
        )
        if newton_tol is not None:
            options["newton_tol"] = newton_tol
    constraint_weight = None if boozer_type == "exact" else _WEIGHT
    # These parity tests keep the initial label, including AspectRatio (the helper uses 0.9 * J()).
    boozer = (
        JaxBoozerSurface(
            JaxBiotSavart(bs.coils),
            surface,
            label_object,
            label_object.J(),
            constraint_weight,
            options,
        )
        if jax_field
        else BoozerSurface(
            bs,
            surface,
            label_object,
            label_object.J(),
            constraint_weight,
            dict(options),
        )
    )
    return _Problem(bs.coils, base_currents, boozer, G0)


def _pair(*args, **kwargs) -> tuple[_Problem, _Problem]:
    return _problem(False, *args, **kwargs), _problem(True, *args, **kwargs)


def _assert_same_solve(
    jax_problem: _Problem,
    native_problem: _Problem,
    jax_res,
    native_res,
    arrays,
    name,
    rtol=_RTOL,
):
    """Native's result keys in native's order; iterations, success and ``G`` as
    natively; the surface, ``iota`` and the named arrays to round-off; and
    native's adjoint solve with the ``PLU`` (to round-off times the condition
    number)."""
    assert list(jax_res) == list(native_res), (
        f"{name}: result keys {list(jax_res)} != native {list(native_res)}"
    )
    for key in ("iter", "success"):
        if key in native_res:
            assert jax_res[key] == native_res[key], (
                f"{name}: {key} {jax_res[key]} != native {native_res[key]}"
            )
    assert (jax_res.get("G") is None) == (native_res.get("G") is None), (
        f"{name}: G presence"
    )
    assert_matches_native(
        jax_problem.boozer.surface.get_dofs(),
        native_problem.boozer.surface.get_dofs(),
        f"{name} surface",
        rtol,
    )
    for key in ("iota", "G", *arrays):
        if native_res.get(key) is not None:
            assert_matches_native(jax_res[key], native_res[key], f"{name} {key}", rtol)
    if "PLU" in native_res:
        # The adjoint solve magnifies the matrices' round-off difference by their condition number.
        P, L, U = native_res["PLU"]
        jax_P, jax_L, jax_U = jax_res["PLU"]
        rhs = parity_rng(7).standard_normal(P.shape[0])
        assert_matches_native(
            forward_backward(jax_P, jax_L, jax_U, rhs),
            forward_backward(P, L, U, rhs),
            f"{name} PLU solve",
            rtol * np.linalg.cond(P @ L @ U),
        )


# --- BoozerExact ----------------------------------------------------------------------


_EXACT_CASES = {
    "stellsym-volume": (True, "Volume", None),
    "stellsym-flux-own-grid": (True, "ToroidalFlux", (51, 51)),
    "nonsym-volume": (False, "Volume", None),
    "nonsym-area-own-grid": (False, "Area", (31, 31)),
}


# --- BoozerLS ---------------------------------------------------------------------------


def _same_start(
    native: _Problem,
    jax_problem: _Problem,
    bfgs_maxiter: int,
    constraint_weight: float = _WEIGHT,
    *,
    tol: float = 1e-10,
    iota: float = _IOTA,
    limited_memory: bool = False,
    weight_inv_modB: bool = False,
):
    """A native BFGS iterate, given to both surfaces: ``(iota, G)``."""
    res = native.boozer.minimize_boozer_penalty_constraints_LBFGS(
        tol=tol,
        maxiter=bfgs_maxiter,
        constraint_weight=constraint_weight,
        iota=iota,
        G=native.G0,
        limited_memory=limited_memory,
        weight_inv_modB=weight_inv_modB,
    )
    jax_problem.boozer.surface.set_dofs(native.boozer.surface.get_dofs())
    native.boozer.recompute_bell()
    return res["iota"], res["G"]


def _singular_at_second_factorisation(
    patches, solver_name: str, native_solves_per_step: int
):
    """Native's ``np.linalg.solve`` raising ``LinAlgError`` in the second step
    (after ``native_solves_per_step`` calls), and the JAX solver
    ``solver_name`` reporting that singular factorisation: its real loop runs
    one step (``maxiter=1``) and stops singular, as the loop does at an exactly
    zero pivot."""
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
    """Upstream's ``subtest_minimize_boozer_penalty_constraints_ls_manual``
    problem (stellarator-symmetric, ``optimize_G``)."""
    _, base_currents, axis, nfp, bs = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        clamped_dims=[False, False, False],
        quadpoints_phi=np.linspace(0, 1 / nfp, 11, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 11, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1)
    flux = ToroidalFlux(surface, BiotSavart(bs.coils), nphi=51, ntheta=51)
    field = JaxBiotSavart(bs.coils) if jax_field else bs
    boozer = (JaxBoozerSurface if jax_field else BoozerSurface)(
        field, surface, flux, 0.1
    )
    return _Problem(bs.coils, base_currents, boozer, _G0(nfp, base_currents))


# --- native objectives and adjoints ---------------------------------------------


_NATIVE_OBJECTIVES = {
    "Iotas": lambda problem: Iotas(problem.boozer),
    "MajorRadius": lambda problem: MajorRadius(problem.boozer),
    "NonQuasiSymmetricRatio": lambda problem: NonQuasiSymmetricRatio(
        problem.boozer, BiotSavart(problem.coils)
    ),
    "BoozerResidual": lambda problem: BoozerResidual(
        problem.boozer, BiotSavart(problem.coils)
    ),
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
        "BoozerResidual": lambda problem: JaxBoozerResidual(
            problem.boozer, problem.boozer.biotsavart
        ),
    }
    for problem in (native, jax_problem):
        problem.boozer.run_code(_IOTA, G=problem.G0)
    native_objectives = {
        name: make(native) for name, make in _NATIVE_OBJECTIVES.items()
    }
    return native, jax_problem, native_objectives, jax_objectives


class TestBoozerSurfaceJax(JaxTestCase):
    def test_exact_run_code_matches_native(self):
        """Exact run_code solves match native surface state, residuals and result metadata."""
        for name in _EXACT_CASES:
            with self.subTest(name=name), self.case():
                self._case_exact_run_code_matches_native(name)

    def _case_exact_run_code_matches_native(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        stellsym, label, grid = _EXACT_CASES[name]
        native, jax_problem = _pair(
            "exact", label, stellsym=stellsym, label_grid=grid, newton_tol=1e-10
        )
        native_res = native.boozer.run_code(_IOTA, G=native.G0)
        jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
        self.assertTrue(native_res["success"])
        self.assertGreater(native_res["iter"], 3)
        _assert_same_solve(
            jax_problem,
            native,
            jax_res,
            native_res,
            ("jacobian",),
            name,
            _RTOL if stellsym else _RTOL_NONSYM,
        )
        np.testing.assert_array_equal(jax_res["mask"], native_res["mask"])
        self.assertEqual(jax_res["type"], "exact")
        self.assertIs(jax_res["s"], jax_problem.boozer.surface)
        self.assertTrue(
            np.max(np.abs(jax_res["residual"])) < 1e-11,
            "the converged residual is not at round-off",
        )
        self.assertFalse(jax_problem.boozer.need_to_run_code)

    def test_exact_newton_at_maxiter_judges_its_final_iterate_as_natively(self):
        """At ``maxiter`` native judges success on the final iterate, so a walk
        whose last permitted step converges succeeds (``maxiter=6`` here); the
        surface holds the last iterate, the residual and Jacobian are evaluated
        there. A fractional cap allows steps while the count is below it; with a
        nonpositive cap nothing moves."""
        for case_id_0, stellsym in zip(
            ["stellsym", "nonsym"], [True, False], strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, stellsym=stellsym
            ), self.case():
                self._case_exact_newton_at_maxiter_judges_its_final_iterate_as_natively(stellsym)

    def _case_exact_newton_at_maxiter_judges_its_final_iterate_as_natively(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        # The cap, native's step count and success at it.
        for maxiter, steps, success in (
            (2, 2, False),
            (6, 6, True),
            (0.5, 1, False),
            (0, 0, False),
            (-1, 0, False),
        ):
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
            _assert_same_solve(
                jax_problem,
                native,
                jax_res,
                native_res,
                ("jacobian",),
                f"maxiter={maxiter}",
                rtol,
            )
            # Converged residuals are round-off: compare on the scale of the terms (G |B| is 22 here).
            assert_matches_native(
                jax_res["residual"],
                native_res["residual"],
                f"maxiter={maxiter} residual",
                rtol,
                scale=22.0,
            )
            if steps == 0:
                np.testing.assert_array_equal(
                    jax_problem.boozer.surface.get_dofs(), start
                )

    def test_exact_newton_on_a_singular_system_raises_as_natively(self):
        """Without current the field vanishes, so do the residual rows; with the
        label off its target, native's ``np.linalg.solve`` raises at the first
        step, before moving the surface."""
        native, jax_problem = _pair("exact")
        for problem in (native, jax_problem):
            for current in problem.currents:
                current.local_full_x = np.zeros(1)
            problem.boozer.targetlabel *= 1.1
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(np.linalg.LinAlgError, "Singular matrix"):
                problem.boozer.solve_residual_equation_exactly_newton(
                    tol=1e-10, maxiter=5, iota=_IOTA, G=native.G0
                )
            np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)
            self.assertTrue(problem.boozer.need_to_run_code)

    def test_non_finite_solves_fail_as_natively(self):
        """A NaN ``G`` makes every BoozerExact step NaN: as natively the walk runs
        to ``maxiter`` (a NaN norm never meets the tolerance), restores its start,
        where ``G`` is still NaN, and SciPy's ``lu`` refuses the Jacobian there. A weighted
        BoozerLS penalty without current is 0/0: the Newton loop does not start
        and ``lu`` refuses the Hessian before the surface moves."""
        native, jax_problem = _pair("exact")
        for problem in (native, jax_problem):
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                problem.boozer.solve_residual_equation_exactly_newton(
                    tol=1e-10, maxiter=3, iota=_IOTA, G=np.nan
                )
            np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)

        native, jax_problem = _pair("ls", weight_inv_modB=True)
        for problem in (native, jax_problem):
            for current in problem.currents:
                current.local_full_x = np.zeros(1)
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11,
                    maxiter=5,
                    constraint_weight=_WEIGHT,
                    iota=_IOTA,
                    G=native.G0,
                    weight_inv_modB=True,
                )
            np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)

    def test_exact_newton_takes_G_from_the_currents(self):
        """Exact Newton seeds G from coil currents when no explicit initial G is supplied."""
        native, jax_problem = _pair("exact", optimize_G=False, newton_tol=1e-10)
        native_res = native.boozer.run_code(_IOTA)
        jax_res = jax_problem.boozer.run_code(_IOTA)
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("jacobian",), "G from currents"
        )

    def test_ls_run_code_matches_native(self):
        """BFGS (or L-BFGS-B), then the penalty Newton. Near convergence the
        quasi-Newton walk's last line searches follow round-off (its iteration
        count can differ by one or two), so the polished solution, the Newton
        step count and flags are compared."""
        for limited_memory in (False, True):
            for case_id_0, optimize_G in zip(
                ["G", "G-from-currents"], [True, False], strict=True
            ):
                for case_id_1, weight_inv_modB in zip(
                    ["weighted", "unweighted"], [True, False], strict=True
                ):
                    with self.subTest(
                        limited_memory=limited_memory,
                        case_id_0=case_id_0,
                        optimize_G=optimize_G,
                        case_id_1=case_id_1,
                        weight_inv_modB=weight_inv_modB,
                    ), self.case():
                        self._case_ls_run_code_matches_native(
                            optimize_G, weight_inv_modB, limited_memory
                        )

    def _case_ls_run_code_matches_native(self, optimize_G, weight_inv_modB, limited_memory):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair(
            "ls",
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            options={
                "verbose": False,
                "weight_inv_modB": weight_inv_modB,
                "limited_memory": limited_memory,
            },
        )
        native_res = native.boozer.run_code(_IOTA, G=native.G0)
        jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
        self.assertTrue(native_res["success"])
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("hessian",), "run_code"
        )
        self.assertEqual(jax_res["type"], "ls")
        self.assertEqual(jax_res["weight_inv_modB"], weight_inv_modB)
        self.assertLessEqual(np.linalg.norm(jax_res["jacobian"]), 1e-11)

    def test_bfgs_matches_native_up_to_maxiter(self):
        """Capped before convergence (a failed solve, kept as natively). The two
        walks separate by round-off growth: iterates by 3e-13 after 15 iterations,
        1e-9 after 30 on this problem."""
        for case_id_0, stellsym in zip(
            ["stellsym", "nonsym"], [True, False], strict=True
        ):
            for case_id_1, limited_memory in zip(
                ["BFGS", "L-BFGS-B"], [False, True], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    stellsym=stellsym,
                    case_id_1=case_id_1,
                    limited_memory=limited_memory,
                ), self.case():
                    self._case_bfgs_matches_native_up_to_maxiter(
                        stellsym, limited_memory
                    )

    def _case_bfgs_matches_native_up_to_maxiter(
        self, stellsym, limited_memory
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        maxiter = 15
        native, jax_problem = _pair("ls", stellsym=stellsym)
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_LBFGS(
                tol=1e-10,
                maxiter=maxiter,
                constraint_weight=_WEIGHT,
                iota=_IOTA,
                G=problem.G0,
                limited_memory=limited_memory,
                weight_inv_modB=False,
            )
            for problem in (native, jax_problem)
        )
        self.assertFalse(native_res["success"])
        self.assertEqual(native_res["iter"], maxiter)
        keys = ("nfev", "njev", "status", "message")
        self.assertEqual(tuple(jax_res["info"][key] for key in keys), tuple(native_res["info"][key] for key in keys))
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
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("fun",), "BFGS", rtol=1e-11
        )
        # The Hessian magnifies the iterates' difference (measured 8e-11 of the largest entry).
        assert_matches_native(
            jax_res["gradient"], native_res["gradient"], "BFGS gradient", 1e-9
        )
        self.assertIs(jax_res["s"], jax_problem.boozer.surface)
        self.assertFalse(jax_problem.boozer.need_to_run_code)

    def test_host_bfgs_runs_scipy_on_the_jitted_penalty(self):
        """With ``host_bfgs``, BFGS and L-BFGS-B are SciPy's on the jitted
        penalty, the path before the device port, bit for bit; ``run_code``
        takes the option from ``options``."""
        for limited_memory in (False, True):
            with self.subTest(limited_memory=limited_memory), self.case():
                self._case_host_bfgs_runs_scipy_on_the_jitted_penalty(limited_memory)

    def _case_host_bfgs_runs_scipy_on_the_jitted_penalty(self, limited_memory):
        """Run one row with case-local objects released before runtime cleanup."""
        maxiter, tol = 20, 1e-10
        jax_problem = _problem(
            True,
            "ls",
            options={
                "verbose": False,
                "weight_inv_modB": False,
                "limited_memory": limited_memory,
                "host_bfgs": True,
                "bfgs_maxiter": maxiter,
                "bfgs_tol": tol,
                "newton_maxiter": 0,
            },
        )
        boozer = jax_problem.boozer
        start = boozer.surface.get_dofs()
        problem = boozer_problem(
            boozer.biotsavart, boozer.surface, boozer.label, boozer.targetlabel, _WEIGHT
        )

        def penalty(x):
            return host_tree(
                boozer_penalty_constraints(
                    problem,
                    place_float64(x, problem.targetlabel),
                    derivatives=1,
                    optimize_G=True,
                    weight_inv_modB=False,
                )
            )

        options = {"maxiter": maxiter, "gtol": tol}
        if limited_memory:
            options.update(maxcor=200, ftol=tol)
        expected = minimize(
            penalty,
            np.concatenate((start, [_IOTA, jax_problem.G0])),
            jac=True,
            method="L-BFGS-B" if limited_memory else "BFGS",
            options=options,
        )
        res = boozer.minimize_boozer_penalty_constraints_LBFGS(
            tol=tol,
            maxiter=maxiter,
            constraint_weight=_WEIGHT,
            iota=_IOTA,
            G=jax_problem.G0,
            limited_memory=limited_memory,
            weight_inv_modB=False,
            host_bfgs=True,
        )
        for key in ("x", "fun", "jac", "nit", "nfev", "njev", "status", "message"):
            np.testing.assert_array_equal(
                res["info"][key], expected[key], err_msg=f"host_bfgs {key} differs from SciPy's"
            )
        # run_code reads the option; without Newton steps it ends at SciPy's endpoint.
        boozer.surface.set_dofs(start)
        boozer.recompute_bell()
        boozer.run_code(_IOTA, G=jax_problem.G0)
        np.testing.assert_array_equal(
            boozer.surface.get_dofs(), expected.x[:-2], err_msg="run_code did not run SciPy"
        )

    def test_device_quasi_newton_scratch_is_the_penalty_plus_its_state(self):
        """The compiled BFGS and L-BFGS-B loops need the scratch memory of the
        penalty's value and gradient plus their state: BFGS's inverse Hessian
        and its update (measured 2 n^2 doubles on the CPU, 1.2 n^2 on the GPU,
        n = 75), L-BFGS-B's corrections and their small matrices (measured 3.8
        MB on the CPU at maxcor 200), never a multiple of the penalty's."""
        for parity_lane in ("cpu", "gpu"):
            with self.subTest(parity_lane=parity_lane), self.case():
                self._case_device_quasi_newton_scratch_is_the_penalty_plus_its_state(parity_lane)

    def _case_device_quasi_newton_scratch_is_the_penalty_plus_its_state(self, parity_lane):
        """Run one row with case-local objects released before runtime cleanup."""
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
            lbfgsb = scratch(
                boozer_penalty_lbfgsb.lower(problem, x, scalar, scalar, scalar, maxcor=maxcor, **options)
            )
        n, double = x.shape[0], 8
        self.assertLessEqual(
            bfgs,
            penalty + 3 * n * n * double + _QUASI_NEWTON_SLACK,
            f"BFGS scratch {bfgs} B exceeds the penalty's {penalty} B plus 3 n^2 doubles",
        )
        self.assertLessEqual(
            lbfgsb,
            penalty + 2 * (2 * maxcor * n + 8 * maxcor**2) * double + _QUASI_NEWTON_SLACK,
            f"L-BFGS-B scratch {lbfgsb} B exceeds the penalty's {penalty} B plus its memory",
        )

    def test_penalty_newton_matches_native(self):
        """From a BFGS iterate; ``stab`` shifts the Hessian of each step but not
        the returned one."""
        for stab in [0.0, 1e-4]:
            with self.subTest(stab=stab), self.case():
                self._case_penalty_newton_matches_native(stab)

    def _case_penalty_newton_matches_native(self, stab):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("ls")
        iota, G = _same_start(native, jax_problem, 200)
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11,
                maxiter=20,
                constraint_weight=_WEIGHT,
                iota=iota,
                G=G,
                stab=stab,
                weight_inv_modB=False,
            )
            for problem in (native, jax_problem)
        )
        self.assertTrue(native_res["success"])
        self.assertGreater(native_res["iter"], 1)
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("hessian",), f"stab={stab}"
        )
        self.assertIs(jax_res["residual"], jax_res["jacobian"])

    def test_penalty_newton_diverges_and_restores_its_start_as_natively(self):
        """From an early BFGS iterate the undamped Newton walk diverges: as natively
        it runs to ``maxiter``, fails and, ending with a larger gradient norm than
        it started with, restores its start and the derivatives there. A two-step
        walk is compared with native's whatever it ends at."""
        for maxiter in (2, 40):
            native, jax_problem = _pair("ls")
            iota, G = _same_start(native, jax_problem, 40)
            start = native.boozer.surface.get_dofs()
            native_res, jax_res = (
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11,
                    maxiter=maxiter,
                    constraint_weight=_WEIGHT,
                    iota=iota,
                    G=G,
                    weight_inv_modB=False,
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
            _assert_same_solve(
                jax_problem,
                native,
                jax_res,
                native_res,
                ("jacobian", "hessian"),
                f"maxiter={maxiter}",
                1e-9 if maxiter == 2 else _RTOL,
            )

    def test_penalty_newton_divergence_factor_as_natively(self):
        """``divergence_factor`` has native's meaning: from an early BFGS iterate
        (where the walk diverges) a factor of 10 stops the walk at native's step,
        before ``maxiter``, and the start is restored. On a converging walk (from
        a later iterate) None, a factor that never fires and the default call
        give the same iterates bit for bit, and native's."""
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
        """Run one row with case-local objects released before runtime cleanup."""
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
        """Run one row with case-local objects released before runtime cleanup."""
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
        """Native's ``H + stab * I`` is NaN off the diagonal too (IEEE ``inf * 0``),
        so the first step is NaN and the loop stops on the NaN gradient norm: as
        natively the solve fails after one step and restores its start, with
        native's penalty derivatives there (evaluated on a fresh native problem:
        native's own solve then raises in ``lu``, because its
        ``SurfaceXYZTensorFourier.d2volume_by_dcoeffdcoeff`` keeps the NaN of the
        stepped iterate after the surface is set back)."""
        for case_id_0, stab in zip(["nan", "inf"], [np.nan, np.inf], strict=True):
            with self.subTest(case_id_0=case_id_0, stab=stab), self.case():
                self._case_penalty_newton_with_a_non_finite_shift_restores_its_start_as_natively(
                    stab
                )

    def _case_penalty_newton_with_a_non_finite_shift_restores_its_start_as_natively(
        self, stab
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("ls")
        start = native.boozer.surface.get_dofs()
        x0 = np.concatenate((start, [_IOTA, native.G0]))
        _, gradient, hessian = native.boozer.boozer_penalty_constraints_vectorized(
            x0, derivatives=2, constraint_weight=_WEIGHT, optimize_G=True, weight_inv_modB=False
        )
        res = jax_problem.boozer.minimize_boozer_penalty_constraints_newton(
            tol=1e-11,
            maxiter=2,
            constraint_weight=_WEIGHT,
            iota=_IOTA,
            G=jax_problem.G0,
            stab=stab,
            weight_inv_modB=False,
        )
        self.assertFalse(res["success"])
        self.assertEqual(res["iter"], 1)
        np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), start)
        self.assertEqual((res["iota"], res["G"]), (_IOTA, jax_problem.G0))
        assert_matches_native(res["jacobian"], gradient, f"stab={stab} gradient")
        assert_matches_native(res["hessian"], hessian, f"stab={stab} hessian")

    def test_failed_scipy_solves_restore_their_start_as_natively(self):
        """A failed BFGS (``host_bfgs`` or on the device), L-BFGS-B (on the
        device) or ``least_squares`` solve that ends at a larger or non-finite
        penalty value or residual norm than at its start restores the start, as
        natively; one that ends lower keeps its iterate. SciPy's solver, and the
        device solver, is replaced by a failed solve that ends at ``x0 + delta``
        or, for ``delta=None``, by the solver itself capped at two iterations or
        evaluations."""
        for solver in ("LBFGS", "BFGS", "L-BFGS-B", "ls"):
            for delta in (None, 1e-2, np.nan):
                with self.subTest(solver=solver, delta=delta), self.case() as patches:
                    self._case_failed_scipy_solves_restore_their_start_as_natively(
                        solver, delta, patches
                    )

    def _case_failed_scipy_solves_restore_their_start_as_natively(
        self, solver, delta, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""

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
                    arguments["problem"],
                    end,
                    derivatives=1,
                    optimize_G=arguments["optimize_G"],
                    weight_inv_modB=arguments["weight_inv_modB"],
                )
                return replace(result, x=end, fun=value, jac=gradient)

            return solve

        def failed_least_squares(fun, x, jac, **kwargs):
            if delta is None:
                return least_squares(fun, x, jac=jac, **{**kwargs, "max_nfev": 2})
            end = x + delta
            residual, jacobian = fun(end), jac(end)
            return OptimizeResult(
                x=end, fun=residual, grad=jacobian.T @ residual, jac=jacobian, status=0
            )

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
            options = {
                "constraint_weight": _WEIGHT,
                "iota": _IOTA,
                "G": problem.G0,
                "weight_inv_modB": False,
            }
            if solver != "ls":
                if solver == "LBFGS" and problem is jax_problem:
                    options["host_bfgs"] = True
                res = problem.boozer.minimize_boozer_penalty_constraints_LBFGS(
                    tol=1e-10, maxiter=10, limited_memory=solver == "L-BFGS-B", **options
                )
            else:
                res = problem.boozer.minimize_boozer_penalty_constraints_ls(
                    tol=1e-12, maxiter=10, method="lm", **options
                )
            self.assertFalse(res["success"])
            moved = np.max(np.abs(problem.boozer.surface.get_dofs() - start))
            if delta is None:
                self.assertGreater(moved, 0.0, "a lower failed iterate was not kept")
            else:
                self.assertEqual(moved, 0.0, "a worse failed iterate was kept")
                self.assertEqual((res["iota"], res["G"]), (_IOTA, problem.G0))
            results.append(res)
        arrays = ("residual", "gradient", "jacobian") if solver == "ls" else ("fun", "gradient")
        _assert_same_solve(
            jax_problem, native, results[1], results[0], arrays, f"{solver} delta={delta}", 1e-10
        )

    def test_singular_step_after_a_step_leaves_the_last_iterate_as_natively(self):
        """The adapter's failure contract: native moves the surface to every
        iterate before factorising there, so when the second step's factorisation
        is singular the surface holds the first iterate, the solve raises
        ``LinAlgError`` and ``need_to_run_code`` stays set (BoozerExact, penalty
        Newton and the damped Gauss-Newton). The JAX loop takes one real step and
        its result is then marked singular, so this checks what the adapter does
        with a reached iterate, not the loop's detection of the singular step
        (``tests/core/test_boozer_solver_loops_jax.py`` checks that); the starts
        are ones where the first step lowers the norm, so that one-step loop keeps
        its iterate rather than restoring its start. The
        BoozerExact step and the penalty Newton step at this gradient norm solve
        twice and once."""
        for method in ["exact", "newton", "manual"]:
            with self.subTest(method=method), self.case() as patches:
                self._case_singular_step_after_a_step_leaves_the_last_iterate_as_natively(
                    method, patches
                )

    def _case_singular_step_after_a_step_leaves_the_last_iterate_as_natively(
        self, method, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("exact" if method == "exact" else "ls")
        # From the raw start and early BFGS iterates the first penalty Newton step
        # raises the gradient norm; from this iterate it lowers it (2e-3 to 1e-6).
        iota, G = (
            _same_start(native, jax_problem, 200)
            if method == "newton"
            else (_IOTA, native.G0)
        )
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
                        **options,
                        constraint_weight=_WEIGHT,
                        weight_inv_modB=False,
                        method="manual",
                    )
            self.assertTrue(problem.boozer.need_to_run_code)
        self.assertTrue(
            np.max(np.abs(native.boozer.surface.get_dofs() - start)) > 1e-6,
            "the first step did not move the surface",
        )
        assert_matches_native(
            jax_problem.boozer.surface.get_dofs(),
            native.boozer.surface.get_dofs(),
            f"{method} surface",
        )

    def test_least_squares_methods_match_native(self):
        """Upstream's test problem from its BFGS start; the damped Gauss-Newton
        result is returned but, as natively, not stored. ``lm`` is capped at 8
        evaluations, where its walk still agrees to round-off."""
        for method in ["manual", "lm"]:
            with self.subTest(method=method), self.case():
                self._case_least_squares_methods_match_native(method)

    def _case_least_squares_methods_match_native(self, method):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = (
            _least_squares_problem(False),
            _least_squares_problem(True),
        )
        constraint_weight = 1000.0 / (3 * 11 * 11)
        iota, G = _same_start(
            native,
            jax_problem,
            700,
            constraint_weight,
            tol=1e-12,
            iota=-0.4,
            limited_memory=True,
            weight_inv_modB=True,
        )
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_ls(
                tol=1e-8,
                maxiter=50 if method == "manual" else 8,
                constraint_weight=constraint_weight,
                iota=iota,
                G=G,
                method=method,
            )
            for problem in (native, jax_problem)
        )
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("jacobian",), method, 1e-10
        )
        # At the solution the residuals and J^T r are small differences of O(1) terms.
        assert_matches_native(
            jax_res["residual"],
            native_res["residual"],
            f"{method} residual",
            1e-12,
            scale=1.0,
        )
        assert_matches_native(
            jax_res["gradient"],
            native_res["gradient"],
            f"{method} gradient",
            1e-11,
            scale=1.0,
        )
        self.assertEqual(jax_res["success"], native_res["success"])
        self.assertIs(jax_res["s"], jax_problem.boozer.surface)
        if method == "manual":
            self.assertTrue(native_res["success"])
            self.assertTrue(jax_problem.boozer.need_to_run_code)
        else:
            self.assertEqual(jax_res["info"].nfev, native_res["info"].nfev)
            self.assertIs(jax_problem.boozer.res, jax_res)

    def test_penalty_residual_matches_native(self):
        """The least-squares methods' formulation: native
        ``_get_residual_vector_and_jacobian`` with a label on its own grid."""
        for case_id_0, optimize_G in zip(
            ["G", "G-from-currents"], [True, False], strict=True
        ):
            for case_id_1, weight_inv_modB in zip(
                ["weighted", "unweighted"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    optimize_G=optimize_G,
                    case_id_1=case_id_1,
                    weight_inv_modB=weight_inv_modB,
                ), self.case():
                    self._case_penalty_residual_matches_native(
                        optimize_G, weight_inv_modB
                    )

    def _case_penalty_residual_matches_native(
        self, optimize_G, weight_inv_modB
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("ls", "ToroidalFlux", label_grid=(31, 31))
        boozer = jax_problem.boozer
        problem = boozer_problem(
            boozer.biotsavart, boozer.surface, boozer.label, boozer.targetlabel, 7.0
        )
        G0 = native.G0
        self.assertIsNotNone(G0)
        x = np.concatenate((boozer.surface.get_dofs(), [_IOTA, G0][: 1 + optimize_G]))
        expected = native.boozer._get_residual_vector_and_jacobian(
            x.copy(), 7.0, optimize_G, weight_inv_modB
        )
        placed = place_float64(x, problem.targetlabel)
        for derivatives in (0, 1):
            actual = boozer_penalty_residual(
                problem,
                placed,
                derivatives=derivatives,
                optimize_G=optimize_G,
                weight_inv_modB=weight_inv_modB,
            )
            for order in range(derivatives + 1):
                assert_matches_native(
                    actual[order],
                    expected[order],
                    f"penalty residual [{derivatives}][{order}]",
                )

    def test_need_to_run_code_caches_results_as_natively(self):
        """Cached solves are reused and parent invalidation requests a fresh solve as natively."""
        jax_problem = _problem(True, "exact")
        boozer = jax_problem.boozer
        res = boozer.run_code(_IOTA, G=jax_problem.G0)
        self.assertIsNone(boozer.run_code(_IOTA, G=jax_problem.G0))
        for solve in (
            boozer.solve_residual_equation_exactly_newton,
            boozer.minimize_boozer_penalty_constraints_LBFGS,
            boozer.minimize_boozer_penalty_constraints_newton,
            boozer.minimize_boozer_penalty_constraints_ls,
        ):
            self.assertIs(solve(iota=0.3), res)
        jax_problem.currents[0].local_full_x = (
            1.01 * jax_problem.currents[0].local_full_x
        )
        self.assertTrue(
            boozer.need_to_run_code,
            "a coil change does not mark the surface for a new solve",
        )

    def test_native_objectives_use_the_jax_surface(self):
        """Native objectives on solved surfaces: values and coil gradients (through
        ``res['PLU']`` and ``res['vjp']``) as with native ``BoozerSurface``. With
        a Volume label and ``G`` optimized or the currents fixed,
        ``JaxBoozerResidual`` is native ``BoozerResidual`` too, also after a coil
        change that makes both re-solve. On a converged BoozerExact surface the
        Boozer residual is round-off (``J`` about 1e-29), so ``BoozerResidual`` is
        compared on capped BoozerExact solves instead
        (``test_boozer_residual_on_capped_boozer_exact_solves_matches_native``)."""
        for name in _OBJECTIVE_CASES:
            with self.subTest(name=name), self.case():
                self._case_native_objectives_use_the_jax_surface(name)

    def _case_native_objectives_use_the_jax_surface(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        boozer_type, label, optimize_G, weight_inv_modB = _OBJECTIVE_CASES[name]
        native, jax_problem = _pair(
            boozer_type,
            label,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
            label_grid=(51, 51) if label == "ToroidalFlux" else None,
        )
        for problem in (native, jax_problem):
            problem.boozer.run_code(_IOTA, G=problem.G0)
        objectives = {
            objective_name: make
            for objective_name, make in _NATIVE_OBJECTIVES.items()
            if boozer_type == "ls" or objective_name != "BoozerResidual"
        }
        for objective_name, make in objectives.items():
            native_objective, jax_objective = make(native), make(jax_problem)
            assert_matches_native(
                jax_objective.J(), native_objective.J(), f"{name} {objective_name}"
            )
            assert_matches_native(
                jax_objective.dJ(),
                native_objective.dJ(),
                f"{name} {objective_name} coil gradient",
            )
        if boozer_type == "exact":
            return
        native_residual = BoozerResidual(native.boozer, BiotSavart(native.coils))
        jax_residual = JaxBoozerResidual(
            jax_problem.boozer, jax_problem.boozer.biotsavart
        )
        for change in ("solved", "after a coil change"):
            value = jax_residual.J()
            self.assertTrue(
                isinstance(value, np.float64),
                f"{name}: J is {type(value)}, native's is np.float64",
            )
            assert_matches_native(
                value, native_residual.J(), f"{name} JaxBoozerResidual {change}"
            )
            assert_matches_native(
                jax_residual.dJ(),
                native_residual.dJ(),
                f"{name} JaxBoozerResidual gradient {change}",
            )
            for problem in (native, jax_problem):
                problem.currents[0].local_full_x = (
                    1.0001 * problem.currents[0].local_full_x
                )
            self.assertTrue(
                jax_problem.boozer.need_to_run_code and jax_residual._J is None,
                "a coil change left a stale value",
            )

    def test_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(
        self,
    ):
        """On Area and AspectRatio labels (``G`` optimized) native
        ``BoozerResidual`` is the derivative of its value, and
        ``JaxBoozerResidual`` agrees with it. Both are compared at a shared state:
        a native BFGS iterate given to both surfaces, stored with its Hessian by a
        penalty Newton call with ``maxiter=0`` (on this fixture these labels'
        ``run_code`` walks follow round-off to different solutions, and the
        penalty Newton from the BFGS iterate does not converge). Comparing on
        the same solved surface isolates the objective's derivative from the
        ill-conditioned Hessian's amplification of different solver round-off."""
        for label in ["Area", "AspectRatio"]:
            with self.subTest(label=label), self.case():
                self._case_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(
                    label
                )

    def _case_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(
        self, label
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("ls", label)
        iota, G = _same_start(native, jax_problem, 200)
        for problem in (native, jax_problem):
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11,
                maxiter=0,
                constraint_weight=_WEIGHT,
                iota=iota,
                G=G,
                weight_inv_modB=False,
            )
        jax_residual = JaxBoozerResidual(
            jax_problem.boozer, jax_problem.boozer.biotsavart
        )
        # Native BoozerResidual on the same JAX surface (same res, PLU and vjp): the objective alone.
        on_jax_surface = BoozerResidual(
            jax_problem.boozer, BiotSavart(jax_problem.coils)
        )
        assert_matches_native(
            jax_residual.J(), on_jax_surface.J(), f"{label} JaxBoozerResidual"
        )
        assert_matches_native(
            jax_residual.dJ(),
            on_jax_surface.dJ(),
            f"{label} JaxBoozerResidual gradient",
        )

    def test_boozer_residual_owns_its_surface_and_reads_changed_values_with_a_copied_field(
        self,
    ):
        """The residual copy uses solved quadrature; a label sharing the solved
        DOFs uses its own grid. A copied field follows later coil changes, and
        recomputing replaces private-copy edits and reads new target/weight values
        without recompilation."""
        problem = _problem(True, "ls", label_grid=(31, 31))
        boozer = problem.boozer
        boozer.minimize_boozer_penalty_constraints_newton(
            maxiter=0,
            constraint_weight=_WEIGHT,
            iota=_IOTA,
            G=problem.G0,
            weight_inv_modB=False,
        )
        field = copy.copy(boozer.biotsavart)
        residual = JaxBoozerResidual(boozer, field)
        native = BoozerResidual(boozer, BiotSavart(problem.coils))
        self.assertIsNot(residual.surface.dofs, boozer.surface.dofs)
        self.assertIs(boozer.label.surface.dofs, boozer.surface.dofs)
        self.assertEqual(boozer.label.surface.quadpoints_phi.size, 31)
        np.testing.assert_array_equal(
            residual.surface.quadpoints_phi, boozer.surface.quadpoints_phi
        )
        np.testing.assert_array_equal(
            residual.surface.quadpoints_theta, boozer.surface.quadpoints_theta
        )
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
        self.assertEqual(
            compilations, [], "new residual target or weight recompiled the objective"
        )
        self.assertNotEqual(changed, first)
        np.testing.assert_array_equal(residual.surface.get_dofs(), solved_dofs)
        assert_matches_native(changed, native.J(), "changed target and weight")
        assert_matches_native(
            gradient, native.dJ(), "changed target and weight gradient"
        )
        problem.currents[0].local_full_x *= 1.001
        self.assertTrue(boozer.need_to_run_code)
        self.assertIsNone(residual._J)
        self.assertIsNone(residual._dJ)
        self.assertTrue(
            copied._J is None and copied._dJ is None,
            "a copied objective kept stale coil caches",
        )
        boozer.minimize_boozer_penalty_constraints_newton(
            maxiter=0,
            constraint_weight=_WEIGHT,
            iota=_IOTA,
            G=problem.G0,
            weight_inv_modB=False,
        )
        assert_matches_native(
            residual.J(), native.J(), "copied field after a coil change"
        )
        assert_matches_native(
            residual.dJ(), native.dJ(), "copied field gradient after a coil change"
        )
        fresh = JaxBoozerResidual(boozer, boozer.biotsavart)
        assert_matches_native(
            copied.J(), fresh.J(), "copied objective after a coil change"
        )
        assert_matches_native(
            copied.dJ(), fresh.dJ(), "copied objective gradient after a coil change"
        )

    def test_boozer_residual_keeps_the_solved_surface_representation(self):
        """On a BoozerLS solve of a ``SurfaceXYZFourier`` and of a tensor surface
        with a clamped dimension, ``JaxBoozerResidual``'s copy has the solved
        geometry, and its value and coil gradient are native ``BoozerResidual``'s
        evaluated on the solved surface itself (Volume label, ``G`` optimized,
        where native's gradient is the derivative). An unclamped tensor copy of
        these DOFs is a different surface."""
        _, base_currents, axis, nfp, bs = get_data("ncsx")
        G0 = _G0(nfp, base_currents)
        grid = dict(
            quadpoints_phi=np.linspace(0, 1 / nfp, 20, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 20, endpoint=False),
        )
        surfaces = {
            "SurfaceXYZFourier": SurfaceXYZFourier(
                mpol=3, ntor=3, stellsym=True, nfp=nfp, **grid
            ),
            "clamped SurfaceXYZTensorFourier": SurfaceXYZTensorFourier(
                mpol=3,
                ntor=3,
                stellsym=True,
                nfp=nfp,
                clamped_dims=[True, False, False],
                **grid,
            ),
        }
        for name, surface in surfaces.items():
            with self.subTest(surface=name), self.case():
                surface.fit_to_curve(axis, 0.1, flip_theta=True)
                label = Volume(surface)
                boozer = JaxBoozerSurface(
                    JaxBiotSavart(bs.coils),
                    surface,
                    label,
                    label.J(),
                    _WEIGHT,
                    {"verbose": False, "weight_inv_modB": False},
                )
                boozer.minimize_boozer_penalty_constraints_newton(
                    maxiter=0,
                    constraint_weight=_WEIGHT,
                    iota=_IOTA,
                    G=G0,
                    weight_inv_modB=False,
                )
                residual = JaxBoozerResidual(boozer, boozer.biotsavart)
                self.assertIs(type(residual.surface), type(surface))
                np.testing.assert_array_equal(residual.surface.gamma(), surface.gamma())
                unclamped = SurfaceXYZTensorFourier(
                    mpol=3, ntor=3, stellsym=True, nfp=nfp, **grid
                )
                unclamped.set_dofs(surface.get_dofs())
                self.assertGreater(
                    np.abs(unclamped.gamma() - surface.gamma()).max(), 0.1
                )
                on_solved_surface = BoozerResidual(boozer, BiotSavart(bs.coils))
                on_solved_surface.surface = surface
                assert_matches_native(
                    residual.J(), on_solved_surface.J(), f"{name} JaxBoozerResidual"
                )
                assert_matches_native(
                    residual.dJ(),
                    on_solved_surface.dJ(),
                    f"{name} JaxBoozerResidual gradient",
                )

    def test_ls_adjoint_with_a_toroidal_flux_label_is_the_derivative_of_the_solve(self):
        """With a ToroidalFlux label the BoozerLS penalty depends on the coils
        through the label too. The coil gradients of native objectives on a
        ``JaxBoozerSurface`` are the central differences of their values through
        native re-solves (``run_code`` from the solution); native's own adjoint
        (``boozer_surface_dlsqgrad_dcoils_vjp``) drops the label term and is not.
        ``BoozerResidual``'s value also depends on the coils through the label:
        ``JaxBoozerResidual`` differentiates it; native ``BoozerResidual`` takes
        its explicit derivative through ``B`` only and is not the derivative.
        The centered differences then converge to the adjoints at halving steps."""
        native, jax_problem, native_objectives, jax_objectives = (
            _toroidal_flux_adjoint_problem()
        )
        jax_handle, native_handle = Iotas(jax_problem.boozer), Iotas(native.boozer)
        # Every free coil DOF moves, by a fraction of its own size.
        x0 = np.asarray(native_handle.x, dtype=np.float64)
        direction = scaled_direction(x0, 11)
        step = 1e-6

        def values(sign: float) -> dict[str, float]:
            native_handle.x = x0 + sign * step * direction
            return {
                name: objective.J() for name, objective in native_objectives.items()
            }

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
            native_adjoint = (
                native_objectives[name].dJ(partials=True)(native_handle) @ direction
            )
            # Truncation and the re-solves' tolerances bound the agreement (measured 2e-9 to 2e-8 relative);
            # native's gradients are off by 1.8e-3 to 3e-2.
            self.assertTrue(
                abs(adjoint - central) <= 1e-7 * abs(central),
                f"{name}: adjoint {adjoint} != difference {central}",
            )
            self.assertTrue(
                abs(native_adjoint - central) > 1e-4 * abs(central),
                f"{name}: native's adjoint matches the difference",
            )

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
        solve = dict(
            tol=1e-11,
            maxiter=40,
            constraint_weight=_WEIGHT,
            G=None,
            weight_inv_modB=False,
        )
        start = native.boozer.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10,
            maxiter=1500,
            constraint_weight=_WEIGHT,
            iota=_IOTA,
            G=None,
            limited_memory=False,
            weight_inv_modB=False,
        )
        jax_problem.boozer.surface.set_dofs(native.boozer.surface.get_dofs())
        for problem in (native, jax_problem):
            problem.boozer.need_to_run_code = True
            res = problem.boozer.minimize_boozer_penalty_constraints_newton(
                iota=start["iota"], **solve
            )
            self.assertTrue(
                res["success"],
                "the penalty Newton with G from the currents did not converge",
            )
        return native, jax_problem, solve

    def test_gradients_with_G_from_free_currents_are_the_derivatives_of_the_solve(self):
        """Solved directly with ``G=None`` and free currents (``run_code`` refuses
        this; the solvers do not), ``G`` follows the currents. The coil gradients
        of ``Iotas`` and ``JaxBoozerResidual`` on a ``JaxBoozerSurface`` are the
        central differences of the native objectives' values through native
        re-solves (penalty Newton from the solution); native's own drop
        ``dG/dI``. The centered differences then converge to the adjoints at
        halving steps."""
        native, jax_problem, solve = self._free_current_adjoint_problem()
        solution, iota = native.boozer.surface.get_dofs(), native.boozer.res["iota"]
        native_objectives = {
            "Iotas": Iotas(native.boozer),
            "BoozerResidual": BoozerResidual(native.boozer, BiotSavart(native.coils)),
        }
        jax_objectives = {
            "Iotas": Iotas(jax_problem.boozer),
            "BoozerResidual": JaxBoozerResidual(
                jax_problem.boozer, jax_problem.boozer.biotsavart
            ),
        }
        native_handle, jax_handle = native_objectives["Iotas"], jax_objectives["Iotas"]
        x0 = np.asarray(native_handle.x, dtype=np.float64)
        direction = scaled_direction(x0, 13)
        self.assertGreater(x0.size, 0)
        self.assertTrue(
            any(not current.dofs.all_fixed() for current in native.currents)
        )
        # Central-difference truncation is 9e-6 relative at 1e-5 and 9e-8 at 1e-6 (Iotas), so the step is 1e-7.
        step = 1e-7

        def values(sign: float) -> dict[str, float]:
            native_handle.x = x0 + sign * step * direction
            native.boozer.surface.set_dofs(solution)
            native.boozer.minimize_boozer_penalty_constraints_newton(iota=iota, **solve)
            return {
                name: objective.J() for name, objective in native_objectives.items()
            }

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
            native_adjoint = (
                native_objectives[name].dJ(partials=True)(native_handle) @ direction
            )
            # Truncation and the re-solves' tolerances bound the agreement (measured 6e-9 and 9e-11 relative);
            # native's gradients are off by 2.3e-4 and 7.1e-4.
            self.assertTrue(
                abs(adjoint - central) <= 1e-7 * abs(central),
                f"{name}: adjoint {adjoint} != difference {central}",
            )
            self.assertTrue(
                abs(native_adjoint - central) > 1e-4 * abs(central),
                f"{name}: native's gradient matches the difference",
            )

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
        """Without stellarator symmetry the BoozerExact system has a ``z(0, 0)``
        row after the label row: the JAX ``vjp`` matches native's, which takes the
        label multiplier second to last."""
        native, jax_problem = _pair(
            "exact", "ToroidalFlux", stellsym=False, newton_tol=1e-10
        )
        for problem in (native, jax_problem):
            problem.boozer.run_code(_IOTA, G=problem.G0)
        res, jax_res = native.boozer.res, jax_problem.boozer.res
        lm = parity_rng(3).standard_normal(res["jacobian"].shape[0])
        expected = boozer_surface_dexactresidual_dcoils_dcurrents_vjp(
            lm, native.boozer, res["iota"], res["G"]
        )
        actual = jax_res["vjp"](lm, jax_problem.boozer, jax_res["iota"], jax_res["G"])
        assert_matches_native(
            actual(BiotSavart(jax_problem.coils)),
            expected(native.boozer.biotsavart),
            "non-stellsym exact vjp",
            _RTOL_NONSYM,
        )

    def test_boozer_residual_on_capped_boozer_exact_solves_matches_native(self):
        """``BoozerResidual`` on a BoozerExact surface (no label penalty) after two
        Newton steps, where the residual is not yet round-off, with and without
        stellarator symmetry and a ToroidalFlux label: native ``BoozerResidual``
        and ``JaxBoozerResidual`` on the JAX surface give native's value and coil
        gradient (through the exact ``PLU`` and ``vjp``). The residual depends on
        the coils and the surface only through the Newton system's rows, which the
        adjoint holds fixed, so the coil gradient is zero: its explicit and adjoint
        terms cancel (to 1e-15 of the explicit term here), and it is compared on
        the scale of those terms."""
        for stellsym in (True, False):
            with self.subTest(stellsym=stellsym), self.case():
                self._case_boozer_residual_on_capped_boozer_exact_solves_matches_native(stellsym)

    def _case_boozer_residual_on_capped_boozer_exact_solves_matches_native(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("exact", "ToroidalFlux", stellsym=stellsym)
        for problem in (native, jax_problem):
            problem.boozer.solve_residual_equation_exactly_newton(
                tol=1e-10, maxiter=2, iota=_IOTA, G=problem.G0
            )
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

    def test_exact_adjoint_is_the_derivative_of_the_solve_taylor(self):
        """BoozerExact coil gradients are second-order limits of central differences through re-solves."""
        # A ToroidalFlux label puts coil dependence in the label row of the system too.
        problem = _problem(
            True, "exact", "ToroidalFlux", label_grid=(31, 31), newton_tol=1e-12
        )
        boozer = problem.boozer
        self.assertTrue(boozer.run_code(_IOTA, G=problem.G0)["success"])
        objectives = {"Iotas": Iotas(boozer), "MajorRadius": MajorRadius(boozer)}
        handle = objectives["Iotas"]
        # Every free coil DOF moves, by a fraction of its own size.
        x0 = np.asarray(handle.x, dtype=np.float64)
        direction = scaled_direction(x0, 17)
        gradients = {
            name: objective.dJ(partials=True)(handle) @ direction
            for name, objective in objectives.items()
        }
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
        """Native ``BoozerSurface`` solves for every surface DOF, fixed or free,
        with the label differentiated in all of them (``dlabel_dsurface``), and so
        does the JAX solve: the penalty Newton from a BFGS iterate with a fixed
        DOF matches native's; native ``BoozerResidual`` then reads the JAX
        surface's ``dlabel_dsurface``, which is native's."""
        native, jax_problem = _pair("ls")
        for problem in (native, jax_problem):
            problem.boozer.surface.fix("x(0,0)")
        iota, G = _same_start(native, jax_problem, 200)
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11,
                maxiter=20,
                constraint_weight=_WEIGHT,
                iota=iota,
                G=G,
                weight_inv_modB=False,
            )
            for problem in (native, jax_problem)
        )
        self.assertTrue(native_res["success"])
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("hessian",), "fixed DOFs"
        )
        assert_matches_native(
            jax_problem.boozer.dlabel_dsurface(),
            native.boozer.dlabel_dsurface(),
            "label gradient with fixed DOFs",
        )

    def test_settings_are_read_at_every_solve(self):
        """The target, weight, options and coils in force at a solve are used; the
        caller's options dictionary is not modified. BoozerLS stops BFGS early and
        takes no Newton step, so its ``run_code`` result is deterministic."""
        options = {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10}
        native, jax_problem = _pair("exact", options=options)
        ls_native, ls_jax = _pair("ls")
        for problem in (ls_native, ls_jax):
            problem.boozer.options.update(bfgs_maxiter=6, newton_maxiter=0)
        for problems in ((native, jax_problem), (ls_native, ls_jax)):
            for problem in problems:
                problem.boozer.run_code(_IOTA, G=problem.G0)
                problem.boozer.targetlabel *= 1.02
                problem.boozer.constraint_weight = (
                    None if problem.boozer.constraint_weight is None else 30.0
                )
                problem.boozer.options["newton_maxiter"] = (
                    2 if problem.boozer.boozer_type == "exact" else 0
                )
                problem.currents[0].local_full_x = (
                    1.002 * problem.currents[0].local_full_x
                )
            native_res, jax_res = (
                problem.boozer.run_code(
                    problem.boozer.res["iota"], G=problem.boozer.res["G"]
                )
                for problem in problems
            )
            self.assertFalse(native_res["success"])
            self.assertEqual(
                native_res["iter"], 2 if native_res["type"] == "exact" else 0
            )
            _assert_same_solve(
                problems[1],
                problems[0],
                jax_res,
                native_res,
                ("jacobian",),
                "new settings",
                1e-10,
            )
        self.assertEqual(
            options, {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10}
        )

    def test_solves_preserve_the_field_point_buffer(self):
        """Solves, re-solves and coil adjoints keep the field's point buffer (native moves it to the surface)."""
        exact = _problem(True, "exact", newton_tol=1e-10)
        ls = _problem(True, "ls")
        ls.boozer.options.update(bfgs_maxiter=6, newton_maxiter=1)
        objectives = (
            (exact, Iotas(exact.boozer)),
            (ls, JaxBoozerResidual(ls.boozer, ls.boozer.biotsavart)),
        )
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
        """New coils, targets, weights, starting points, tolerances, caps and
        ``stab`` reuse every solver and adjoint program."""

        def solve_all(exact_problem: _Problem, ls_problem: _Problem, settings: dict):
            exact, ls = exact_problem.boozer, ls_problem.boozer
            ls.constraint_weight = settings["weight"]
            res = exact.solve_residual_equation_exactly_newton(
                tol=settings["tol"],
                maxiter=settings["maxiter"],
                iota=settings["iota"],
                G=exact_problem.G0,
            )
            Iotas(exact).dJ()
            for limited_memory in (True, False):
                ls.recompute_bell()
                start = ls.minimize_boozer_penalty_constraints_LBFGS(
                    tol=1e-10,
                    maxiter=settings["maxiter"],
                    constraint_weight=settings["weight"],
                    iota=settings["iota"],
                    G=ls_problem.G0,
                    limited_memory=limited_memory,
                    weight_inv_modB=False,
                )
            for method in ("manual", "lm"):
                ls.recompute_bell()
                ls.minimize_boozer_penalty_constraints_ls(
                    tol=settings["tol"],
                    maxiter=3,
                    constraint_weight=settings["weight"],
                    iota=start["iota"],
                    G=start["G"],
                    method=method,
                    weight_inv_modB=False,
                )
            ls.recompute_bell()
            ls.minimize_boozer_penalty_constraints_newton(
                tol=settings["tol"],
                maxiter=2,
                constraint_weight=settings["weight"],
                iota=start["iota"],
                G=start["G"],
                stab=settings["stab"],
                weight_inv_modB=False,
            )
            Iotas(ls).dJ()
            JaxBoozerResidual(ls, ls.biotsavart).dJ()
            for boozer in (exact, ls):
                boozer.recompute_bell()
            return res

        exact_problem, ls_problem = _problem(True, "exact"), _problem(True, "ls")
        first = solve_all(
            exact_problem,
            ls_problem,
            {
                "tol": 1e-10,
                "maxiter": 12,
                "iota": _IOTA,
                "weight": _WEIGHT,
                "stab": 0.0,
            },
        )
        for problem in (exact_problem, ls_problem):
            problem.currents[0].local_full_x = 1.003 * problem.currents[0].local_full_x
            problem.boozer.surface.set_dofs(problem.boozer.surface.get_dofs() * 1.001)
            problem.boozer.targetlabel *= 1.01
        with jax_compilations() as compilations:
            second = solve_all(
                exact_problem,
                ls_problem,
                {
                    "tol": 2e-10,
                    "maxiter": 15,
                    "iota": -0.41,
                    "weight": 50.0,
                    "stab": 1e-4,
                },
            )
        self.assertEqual(compilations, [], "new values retraced or recompiled a solver")
        self.assertNotEqual(second["iota"], first["iota"])

    def test_solves_make_no_implicit_transfers(self):
        """Exact and penalty solves use explicit transfers under the runtime transfer guard."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for boozer_type in ["exact", "ls"]:
                with self.subTest(
                    case_id_0=case_id_0,
                    parity_lane=parity_lane,
                    boozer_type=boozer_type,
                ), self.case():
                    self._case_solves_make_no_implicit_transfers(
                        parity_lane, boozer_type
                    )

    def _case_solves_make_no_implicit_transfers(
        self, parity_lane, boozer_type
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native = _problem(False, boozer_type)
        with parity_default_device(parity_lane):
            jax_problem = _problem(True, boozer_type)
            with disallow_host_transfers():
                jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
                jax_dJ = Iotas(jax_problem.boozer).dJ()
                if boozer_type == "ls":
                    jax_residual_dJ = JaxBoozerResidual(
                        jax_problem.boozer, jax_problem.boozer.biotsavart
                    ).dJ()
                    jax_problem.boozer.recompute_bell()
                    jax_problem.boozer.minimize_boozer_penalty_constraints_ls(
                        tol=1e-8,
                        maxiter=2,
                        constraint_weight=_WEIGHT,
                        iota=jax_res["iota"],
                        G=jax_res["G"],
                        method="lm",
                        weight_inv_modB=False,
                    )
        coils = jax_problem.boozer.biotsavart.coil_set_spec()
        self.assertEqual(
            {
                device.platform
                for leaf in jax.tree.leaves(coils)
                for device in leaf.devices()
            },
            {parity_lane},
        )
        native_res = native.boozer.run_code(_IOTA, G=native.G0)
        assert_matches_native(
            jax_res["iota"],
            native_res["iota"],
            f"{boozer_type} iota on {parity_lane}",
            1e-10,
        )
        assert_matches_native(
            jax_dJ,
            Iotas(native.boozer).dJ(),
            f"{boozer_type} Iotas gradient on {parity_lane}",
            1e-9,
        )
        if boozer_type == "ls":
            native_residual_dJ = BoozerResidual(
                native.boozer, BiotSavart(native.coils)
            ).dJ()
            assert_matches_native(
                jax_residual_dJ,
                native_residual_dJ,
                f"JaxBoozerResidual gradient on {parity_lane}",
                1e-9,
            )

    def test_boundary_refuses_unsupported_inputs(self):
        """Solver and residual construction reject unsupported surface classes."""
        _, _, axis, nfp, bs = get_data("ncsx")
        field = JaxBiotSavart(bs.coils)
        rz = SurfaceRZFourier(mpol=2, ntor=2, nfp=nfp)
        with self.assertRaisesRegex(
            Exception, "SurfaceXYZTensorFourier or SurfaceXYZFourier"
        ):
            JaxBoozerSurface(field, rz, Volume(rz), 1.0)

        xyz = SurfaceXYZFourier(
            mpol=2,
            ntor=2,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1 / nfp, 6, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 7, endpoint=False),
        )
        xyz.fit_to_curve(axis, 0.1, flip_theta=True)
        with self.assertRaisesRegex(RuntimeError, "SurfaceXYZTensorFourier"):
            JaxBoozerSurface(
                field, xyz, Volume(xyz), 1.0
            ).solve_residual_equation_exactly_newton(iota=_IOTA)
        with self.assertRaisesRegex(TypeError, "JaxBiotSavart"):
            JaxBoozerSurface(
                bs, xyz, Volume(xyz), 1.0, 1.0
            ).minimize_boozer_penalty_constraints_LBFGS(maxiter=1)
        with self.assertRaisesRegex(TypeError, "labels"):
            JaxBoozerSurface(
                field, xyz, PrincipalCurvature(xyz), 1.0, 1.0
            ).minimize_boozer_penalty_constraints_LBFGS(maxiter=1)
        # As natively, G from the currents needs fixed currents for coil gradients.
        with self.assertRaises(AssertionError):
            JaxBoozerSurface(field, xyz, Volume(xyz), 1.0, 1.0).run_code(_IOTA)

        class Subclassed(SurfaceXYZFourier):
            pass

        subclassed = Subclassed(mpol=2, ntor=2, nfp=nfp)
        with self.assertRaisesRegex(TypeError, "JaxBoozerResidual supports"):
            JaxBoozerResidual(
                JaxBoozerSurface(field, subclassed, Volume(subclassed), 1.0, 1.0), field
            )
