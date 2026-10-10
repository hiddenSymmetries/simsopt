"""JAX ``JaxBoozerSurface`` against native ``BoozerSurface``.

Each solver of native ``BoozerSurface`` (the BoozerExact Newton, BFGS and
L-BFGS-B, the penalty Newton, ``least_squares`` and its damped Gauss-Newton,
``run_code``) runs natively and in JAX from the same start on upstream's
NCSX problems (``tests/geo/surface_test_helpers.get_boozer_surface``), with
and without stellarator symmetry, ``optimize_G``, ``weight_inv_modB`` and
labels on their own grids: iteration counts, success flags, the surface,
``iota``, ``G``, the result arrays and the ``PLU`` solves agree. Failures at
``maxiter``, diverging Newton walks, singular systems and the
``need_to_run_code`` cache behave as natively. Native ``Iotas``,
``MajorRadius``, ``NonQuasiSymmetricRatio`` and ``BoozerResidual`` on a
``JaxBoozerSurface`` give native values and coil gradients for coil-independent
labels with optimized ``G``. ``JaxBoozerResidual`` also differentiates the
explicit ToroidalFlux and current-derived ``G`` terms that native misses,
checked through re-solves. Then: settings
read at every solve, no recompilation for new values, no implicit transfers,
the documented differences (fixed DOFs, the BoozerExact adjoint without
stellarator symmetry) and unsupported inputs.

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
        boozer_surface_residual_dB,
    )
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
    from simsopt.objectives.utilities import forward_backward
    from simsopt_jax.core.boozer_problem import boozer_penalty_residual
    from simsopt_jax.runtime.host_boundary import disallow_host_transfers
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_problem import boozer_problem
    from simsopt_jax_adapters.geo import boozer_surface as jax_boozer_surface
    from simsopt_jax_adapters.geo.boozer_surface import (
        JaxBoozerResidual,
        JaxBoozerSurface,
    )
    from .surface_test_helpers import get_boozer_surface

from unittest_jax_support import (
    assert_matches_native,
    jax_compilations,
    parity_default_device,
    parity_rng,
    place_float64,
)

from dataclasses import dataclass, replace
import copy
import inspect
from contextlib import ExitStack
from unittest import mock
from typing import cast

import numpy as np


_IOTA = -0.406
_RTOL = 1e-12
_RTOL_NONSYM = 1e-9
_WEIGHT = 100.0


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
    objectives = {
        "Iotas": lambda problem: Iotas(problem.boozer),
        "MajorRadius": lambda problem: MajorRadius(problem.boozer),
        "NonQuasiSymmetricRatio": lambda problem: NonQuasiSymmetricRatio(
            problem.boozer, BiotSavart(problem.coils)
        ),
        "BoozerResidual": lambda problem: BoozerResidual(
            problem.boozer, BiotSavart(problem.coils)
        ),
    }
    jax_objectives = {
        **objectives,
        "BoozerResidual": lambda problem: JaxBoozerResidual(
            problem.boozer, problem.boozer.biotsavart
        ),
    }
    for problem in (native, jax_problem):
        problem.boozer.run_code(_IOTA, G=problem.G0)
    native_objectives = {name: make(native) for name, make in objectives.items()}
    return native, jax_problem, native_objectives, jax_objectives


class TestBoozerSurfaceJax(JaxTestCase):
    def test_exact_run_code_matches_native(self):
        """Exact run_code solves match native surface state, residuals and result metadata."""
        for name in _EXACT_CASES:
            with self.subTest(name=name), self.case() as patches:
                self._case_exact_run_code_matches_native(name, patches)

    def _case_exact_run_code_matches_native(self, name, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        stellsym, label, grid = _EXACT_CASES[name]
        native, jax_problem = _pair(
            "exact", label, stellsym=stellsym, label_grid=grid, newton_tol=1e-10
        )
        native_res = native.boozer.run_code(_IOTA, G=native.G0)
        jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
        self.assertTrue(
            native_res["success"] and native_res["iter"] > 3,
            'native_res["success"] and native_res["iter"] > 3',
        )
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
        self.assertTrue(
            jax_res["type"] == "exact" and jax_res["s"] is jax_problem.boozer.surface,
            'jax_res["type"] == "exact" and jax_res["s"] is jax_problem.boozer.surface',
        )
        self.assertTrue(
            np.max(np.abs(jax_res["residual"])) < 1e-11,
            "the converged residual is not at round-off",
        )
        self.assertTrue(
            not jax_problem.boozer.need_to_run_code,
            "not jax_problem.boozer.need_to_run_code",
        )

    def test_exact_newton_at_maxiter_fails_as_natively(self):
        """At ``maxiter`` native reports the norm checked before the last step, so
        the solve fails, also when that step converged (``maxiter=6`` here; PR
        #669 would report success); the surface holds the last iterate, the
        residual and Jacobian are evaluated there. With ``maxiter=0`` nothing
        moves."""
        for case_id_0, stellsym in zip(
            ["stellsym", "nonsym"], [True, False], strict=True
        ):
            with self.subTest(
                case_id_0=case_id_0, stellsym=stellsym
            ), self.case() as patches:
                self._case_exact_newton_at_maxiter_fails_as_natively(stellsym, patches)

    def _case_exact_newton_at_maxiter_fails_as_natively(self, stellsym, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        for maxiter in (2, 6, 0):
            native, jax_problem = _pair("exact", stellsym=stellsym)
            start = native.boozer.surface.get_dofs()
            native_res = native.boozer.solve_residual_equation_exactly_newton(
                tol=1e-10, maxiter=maxiter, iota=_IOTA, G=native.G0
            )
            jax_res = jax_problem.boozer.solve_residual_equation_exactly_newton(
                tol=1e-10, maxiter=maxiter, iota=_IOTA, G=jax_problem.G0
            )
            self.assertTrue(
                not native_res["success"] and native_res["iter"] == maxiter,
                'not native_res["success"] and native_res["iter"] == maxiter',
            )
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
            if maxiter == 0:
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
            self.assertTrue(
                problem.boozer.need_to_run_code, "problem.boozer.need_to_run_code"
            )

    def test_non_finite_solves_fail_as_natively(self):
        """A NaN ``G`` makes every BoozerExact step NaN: as natively the walk runs
        to ``maxiter`` (a NaN norm never meets the tolerance), the surface takes
        the NaN iterate and SciPy's ``lu`` refuses the final Jacobian. A weighted
        BoozerLS penalty without current is 0/0: the Newton loop does not start
        and ``lu`` refuses the Hessian before the surface moves."""
        native, jax_problem = _pair("exact")
        for problem in (native, jax_problem):
            with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                problem.boozer.solve_residual_equation_exactly_newton(
                    tol=1e-10, maxiter=3, iota=_IOTA, G=np.nan
                )
            self.assertTrue(
                np.all(np.isnan(problem.boozer.surface.get_dofs())),
                "np.all(np.isnan(problem.boozer.surface.get_dofs()))",
            )

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
        """BFGS, then the penalty Newton. Near convergence BFGS's last line
        searches follow round-off (its iteration count can differ by one or two),
        so the polished solution, the Newton step count and flags are compared."""
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
                ), self.case() as patches:
                    self._case_ls_run_code_matches_native(
                        optimize_G, weight_inv_modB, patches
                    )

    def _case_ls_run_code_matches_native(self, optimize_G, weight_inv_modB, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair(
            "ls", optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        native_res = native.boozer.run_code(_IOTA, G=native.G0)
        jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
        self.assertTrue(native_res["success"], 'native_res["success"]')
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("hessian",), "run_code"
        )
        self.assertTrue(
            jax_res["type"] == "ls" and jax_res["weight_inv_modB"] == weight_inv_modB,
            'jax_res["type"] == "ls" and jax_res["weight_inv_modB"] == weight_inv_modB',
        )
        self.assertTrue(
            np.linalg.norm(jax_res["jacobian"]) <= 1e-11,
            'np.linalg.norm(jax_res["jacobian"]) <= 1e-11',
        )

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
                ), self.case() as patches:
                    self._case_bfgs_matches_native_up_to_maxiter(
                        stellsym, limited_memory, patches
                    )

    def _case_bfgs_matches_native_up_to_maxiter(
        self, stellsym, limited_memory, patches
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
        self.assertTrue(
            not native_res["success"] and native_res["iter"] == maxiter,
            'not native_res["success"] and native_res["iter"] == maxiter',
        )
        self.assertTrue(
            jax_res["info"].nfev == native_res["info"].nfev,
            'jax_res["info"].nfev == native_res["info"].nfev',
        )
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("fun",), "BFGS", rtol=1e-11
        )
        # The Hessian magnifies the iterates' difference (measured 8e-11 of the largest entry).
        assert_matches_native(
            jax_res["gradient"], native_res["gradient"], "BFGS gradient", 1e-9
        )
        self.assertTrue(
            jax_res["s"] is jax_problem.boozer.surface
            and not jax_problem.boozer.need_to_run_code,
            'jax_res["s"] is jax_problem.boozer.surface and not jax_problem.boozer.need_to_run_code',
        )

    def test_penalty_newton_matches_native(self):
        """From a BFGS iterate; ``stab`` shifts the Hessian of each step but not
        the returned one."""
        for stab in [0.0, 1e-4]:
            with self.subTest(stab=stab), self.case() as patches:
                self._case_penalty_newton_matches_native(stab, patches)

    def _case_penalty_newton_matches_native(self, stab, patches):
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
        self.assertTrue(
            native_res["success"] and native_res["iter"] > 1,
            'native_res["success"] and native_res["iter"] > 1',
        )
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("hessian",), f"stab={stab}"
        )
        self.assertTrue(
            jax_res["residual"] is jax_res["jacobian"],
            'jax_res["residual"] is jax_res["jacobian"]',
        )

    def test_penalty_newton_diverges_and_keeps_its_iterate_as_natively(self):
        """From an early BFGS iterate the undamped Newton walk diverges: as natively
        (no divergence guard, no restore) it runs to ``maxiter``, fails and leaves
        the diverged iterate in the surface. The walk is chaotic, so only its
        outcome is compared; a two-step walk is compared iterate by iterate."""
        for maxiter in (2, 40):
            native, jax_problem = _pair("ls")
            iota, G = _same_start(native, jax_problem, 40)
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
            self.assertTrue(
                not native_res["success"] and not jax_res["success"],
                'not native_res["success"] and not jax_res["success"]',
            )
            self.assertTrue(
                native_res["iter"] == jax_res["iter"] == maxiter,
                'native_res["iter"] == jax_res["iter"] == maxiter',
            )
            if maxiter == 2:
                # Diverging steps magnify round-off (measured 1e-11 of the largest entry).
                _assert_same_solve(
                    jax_problem,
                    native,
                    jax_res,
                    native_res,
                    ("jacobian", "hessian"),
                    "two steps",
                    1e-9,
                )
            else:
                self.assertTrue(
                    np.linalg.norm(native_res["jacobian"]) > 1e3
                    and np.linalg.norm(jax_res["jacobian"]) > 1e3,
                    'np.linalg.norm(native_res["jacobian"]) > 1e3 and np.linalg.norm(jax_res["jacobian"]) > 1e3',
                )

    def test_penalty_newton_with_a_non_finite_shift_fails_as_natively(self):
        """Native's ``H + stab * I`` is NaN off the diagonal too (IEEE ``inf * 0``),
        so the first step is NaN, the loop stops on the NaN gradient norm and SciPy's
        ``lu`` refuses the Hessian: the surface keeps the NaN iterate."""
        for case_id_0, stab in zip(["nan", "inf"], [np.nan, np.inf], strict=True):
            with self.subTest(case_id_0=case_id_0, stab=stab), self.case() as patches:
                self._case_penalty_newton_with_a_non_finite_shift_fails_as_natively(
                    stab, patches
                )

    def _case_penalty_newton_with_a_non_finite_shift_fails_as_natively(
        self, stab, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("ls")
        for problem in (native, jax_problem):
            start = problem.boozer.surface.get_dofs()
            with self.assertRaisesRegex(ValueError, "infs or NaNs"):
                problem.boozer.minimize_boozer_penalty_constraints_newton(
                    tol=1e-11,
                    maxiter=2,
                    constraint_weight=_WEIGHT,
                    iota=_IOTA,
                    G=problem.G0,
                    stab=stab,
                    weight_inv_modB=False,
                )
            self.assertTrue(
                problem.boozer.need_to_run_code, "problem.boozer.need_to_run_code"
            )
            self.assertTrue(
                np.all(np.isnan(problem.boozer.surface.get_dofs()))
                and not np.any(np.isnan(start)),
                "np.all(np.isnan(problem.boozer.surface.get_dofs())) and not np.any(np.isnan(start))",
            )
        np.testing.assert_array_equal(
            jax_problem.boozer.surface.get_dofs(), native.boozer.surface.get_dofs()
        )

    def test_singular_step_after_a_step_leaves_the_last_iterate_as_natively(self):
        """The adapter's failure contract: native moves the surface to every
        iterate before factorising there, so when the second step's factorisation
        is singular the surface holds the first iterate, the solve raises
        ``LinAlgError`` and ``need_to_run_code`` stays set (BoozerExact, penalty
        Newton and the damped Gauss-Newton). The JAX loop takes one real step and
        its result is then marked singular, so this checks what the adapter does
        with a reached iterate, not the loop's detection of the singular step
        (``tests/jax/core/test_boozer_solver_loops_jax.py`` checks that). The
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
        start = native.boozer.surface.get_dofs()
        solver_name, solves_per_step = {
            "exact": ("boozer_exact_newton", 2),
            "newton": ("boozer_penalty_newton", 1),
            "manual": ("boozer_penalty_gauss_newton", 1),
        }[method]
        with ExitStack() as patch:
            _singular_at_second_factorisation(patch, solver_name, solves_per_step)
            for problem in (native, jax_problem):
                options = dict(tol=1e-11, maxiter=5, iota=_IOTA, G=problem.G0)
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
                self.assertTrue(
                    problem.boozer.need_to_run_code, "problem.boozer.need_to_run_code"
                )
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
            with self.subTest(method=method), self.case() as patches:
                self._case_least_squares_methods_match_native(method, patches)

    def _case_least_squares_methods_match_native(self, method, patches):
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
        self.assertTrue(
            jax_res["success"] == native_res["success"]
            and jax_res["s"] is jax_problem.boozer.surface,
            'jax_res["success"] == native_res["success"] and jax_res["s"] is jax_problem.boozer.surface',
        )
        if method == "manual":
            self.assertTrue(
                native_res["success"] and jax_problem.boozer.need_to_run_code,
                'native_res["success"] and jax_problem.boozer.need_to_run_code',
            )
        else:
            self.assertTrue(
                jax_res["info"].nfev == native_res["info"].nfev
                and jax_problem.boozer.res is jax_res,
                'jax_res["info"].nfev == native_res["info"].nfev and jax_problem.boozer.res is jax_res',
            )

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
                ), self.case() as patches:
                    self._case_penalty_residual_matches_native(
                        optimize_G, weight_inv_modB, patches
                    )

    def _case_penalty_residual_matches_native(
        self, optimize_G, weight_inv_modB, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, jax_problem = _pair("ls", "ToroidalFlux", label_grid=(31, 31))
        boozer = jax_problem.boozer
        problem = boozer_problem(
            boozer.biotsavart, boozer.surface, boozer.label, boozer.targetlabel, 7.0
        )
        G0 = native.G0
        self.assertTrue(G0 is not None, "G0 is not None")
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
        self.assertTrue(
            boozer.run_code(_IOTA, G=jax_problem.G0) is None,
            "boozer.run_code(_IOTA, G=jax_problem.G0) is None",
        )
        for solve in (
            boozer.solve_residual_equation_exactly_newton,
            boozer.minimize_boozer_penalty_constraints_LBFGS,
            boozer.minimize_boozer_penalty_constraints_newton,
            boozer.minimize_boozer_penalty_constraints_ls,
        ):
            self.assertTrue(solve(iota=0.3) is res, "solve(iota=0.3) is res")
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
        change that makes both re-solve; on BoozerExact both raise ``KeyError``
        (exact results have no ``weight_inv_modB``)."""
        for name in _OBJECTIVE_CASES:
            with self.subTest(name=name), self.case() as patches:
                self._case_native_objectives_use_the_jax_surface(name, patches)

    def _case_native_objectives_use_the_jax_surface(self, name, patches):
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
            "Iotas": lambda problem: Iotas(problem.boozer),
            "MajorRadius": lambda problem: MajorRadius(problem.boozer),
            "NonQuasiSymmetricRatio": lambda problem: NonQuasiSymmetricRatio(
                problem.boozer, BiotSavart(problem.coils)
            ),
        }
        if boozer_type == "ls":
            objectives["BoozerResidual"] = lambda problem: BoozerResidual(
                problem.boozer, BiotSavart(problem.coils)
            )
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
        native_residual = BoozerResidual(native.boozer, BiotSavart(native.coils))
        jax_residual = JaxBoozerResidual(
            jax_problem.boozer, jax_problem.boozer.biotsavart
        )
        if boozer_type == "exact":
            for objective in (native_residual, jax_residual):
                with self.assertRaisesRegex(KeyError, "weight_inv_modB"):
                    objective.J()
            return
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
            with self.subTest(label=label), self.case() as patches:
                self._case_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(
                    label, patches
                )

    def _case_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(
        self, label, patches
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
        self.assertTrue(
            residual.surface.dofs is not boozer.surface.dofs,
            "residual.surface.dofs is not boozer.surface.dofs",
        )
        self.assertTrue(
            boozer.label.surface.dofs is boozer.surface.dofs,
            "boozer.label.surface.dofs is boozer.surface.dofs",
        )
        self.assertTrue(
            boozer.label.surface.quadpoints_phi.size == 31,
            "boozer.label.surface.quadpoints_phi.size == 31",
        )
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
        self.assertTrue(
            copied.surface.dofs is not residual.surface.dofs,
            "copied.surface.dofs is not residual.surface.dofs",
        )
        self.assertTrue(
            copied._J is None and copied._dJ is None,
            "copied._J is None and copied._dJ is None",
        )
        self.assertTrue(
            copied.constraint_weight == residual.constraint_weight,
            "copied.constraint_weight == residual.constraint_weight",
        )
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
        self.assertTrue(
            compilations == [], "new residual target or weight recompiled the objective"
        )
        self.assertTrue(changed != first, "changed != first")
        np.testing.assert_array_equal(residual.surface.get_dofs(), solved_dofs)
        assert_matches_native(changed, native.J(), "changed target and weight")
        assert_matches_native(
            gradient, native.dJ(), "changed target and weight gradient"
        )
        problem.currents[0].local_full_x *= 1.001
        self.assertTrue(
            boozer.need_to_run_code and residual._J is None and residual._dJ is None,
            "boozer.need_to_run_code and residual._J is None and residual._dJ is None",
        )
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

    def test_ls_adjoint_with_a_toroidal_flux_label_is_the_derivative_of_the_solve(self):
        """With a ToroidalFlux label the BoozerLS penalty depends on the coils
        through the label too. The coil gradients of native objectives on a
        ``JaxBoozerSurface`` are the central differences of their values through
        native re-solves (``run_code`` from the solution); native's own adjoint
        (``boozer_surface_dlsqgrad_dcoils_vjp``) drops the label term and is not.
        ``BoozerResidual``'s value also depends on the coils through the label:
        ``JaxBoozerResidual`` differentiates it; native ``BoozerResidual`` takes
        its explicit derivative through ``B`` only and is not the derivative."""
        native, jax_problem, native_objectives, jax_objectives = (
            _toroidal_flux_adjoint_problem()
        )
        jax_handle, native_handle = Iotas(jax_problem.boozer), Iotas(native.boozer)
        # Every free coil DOF moves, by a fraction of its own size.
        x0 = np.asarray(native_handle.x, dtype=np.float64)
        direction = parity_rng(11).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        step = 1e-6

        def values(sign: float) -> dict[str, float]:
            native_handle.x = x0 + sign * step * direction
            return {
                name: objective.J() for name, objective in native_objectives.items()
            }

        plus, minus = values(1.0), values(-1.0)
        native_handle.x = x0
        for name, make in jax_objectives.items():
            central = (plus[name] - minus[name]) / (2 * step)
            with disallow_host_transfers():
                partials = make(jax_problem).dJ(partials=True)
            adjoint = partials(jax_handle) @ direction
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
        ``dG/dI``."""
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
        direction = parity_rng(13).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        self.assertTrue(
            x0.size > 0
            and any(not current.dofs.all_fixed() for current in native.currents),
            "x0.size > 0 and any(not current.dofs.all_fixed() for current in native.currents)",
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

        plus, minus = values(1.0), values(-1.0)
        native_handle.x = x0
        native.boozer.surface.set_dofs(solution)
        native.boozer.minimize_boozer_penalty_constraints_newton(iota=iota, **solve)
        for name, objective in jax_objectives.items():
            central = (plus[name] - minus[name]) / (2 * step)
            with disallow_host_transfers():
                partials = objective.dJ(partials=True)
            adjoint = partials(jax_handle) @ direction
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

    def test_exact_adjoint_without_stellsym_where_native_raises(self):
        """Native's BoozerExact ``vjp`` takes the label multiplier from the last
        entry and fails without stellarator symmetry, where the system has a
        ``z(0, 0)`` row too. The JAX ``vjp`` is checked against native's own pieces
        with the label multiplier second to last (simsopt PR #669's correction)."""
        native, jax_problem = _pair(
            "exact", "ToroidalFlux", stellsym=False, newton_tol=1e-10
        )
        for problem in (native, jax_problem):
            problem.boozer.run_code(_IOTA, G=problem.G0)
        res = native.boozer.res
        lm = parity_rng(3).standard_normal(res["jacobian"].shape[0])
        with self.assertRaises(ValueError):
            boozer_surface_dexactresidual_dcoils_dcurrents_vjp(
                lm, native.boozer, res["iota"], res["G"]
            )

        residual_dB = boozer_surface_residual_dB(
            native.boozer.surface, res["iota"], res["G"], native.boozer.biotsavart
        )
        self.assertTrue(residual_dB is not None, "residual_dB is not None")
        dres_dB = cast(tuple, residual_dB)[1]
        multipliers = np.zeros(res["mask"].shape)
        multipliers[res["mask"]] = lm[:-2]
        field_cotangent = np.sum(
            multipliers.reshape((-1, 3))[:, :, None] * dres_dB.reshape((-1, 3, 3)),
            axis=1,
        )
        expected = native.boozer.biotsavart.B_vjp(field_cotangent) + lm[
            -2
        ] * native.boozer.label.dJ(partials=True)(
            native.boozer.biotsavart, as_derivative=True
        )
        jax_res = jax_problem.boozer.res
        actual = jax_res["vjp"](lm, jax_problem.boozer, jax_res["iota"], jax_res["G"])
        assert_matches_native(
            actual(BiotSavart(jax_problem.coils)),
            expected(native.boozer.biotsavart),
            "non-stellsym exact vjp",
            _RTOL_NONSYM,
        )

    def test_fixed_surface_dofs_are_solved_for_as_with_every_dof_free(self):
        """Native ``BoozerSurface`` at 9e027eac3 cannot solve with fixed surface
        DOFs (its label gradient has the free DOFs only and does not fit; PR #669's
        ``dlabel_dsurface`` fixes that). The JAX solve treats every DOF as an
        unknown, as native does with every DOF free."""
        native, jax_problem = _pair("exact", newton_tol=1e-10)
        native.boozer.surface.fix("x(0,0)")
        with self.assertRaises(ValueError):
            native.boozer.run_code(_IOTA, G=native.G0)
        native.boozer.surface.unfix_all()
        native_res = native.boozer.run_code(_IOTA, G=native.G0)
        jax_problem.boozer.surface.fix("x(0,0)")
        jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
        _assert_same_solve(
            jax_problem, native, jax_res, native_res, ("jacobian",), "fixed DOFs"
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
            self.assertTrue(
                not native_res["success"]
                and native_res["iter"] == (2 if native_res["type"] == "exact" else 0),
                'not native_res["success"] and native_res["iter"] == (2 if native_res["type"] == "exact" else 0)',
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
        self.assertTrue(
            options
            == {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10},
            'options == {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10}',
        )

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
            start = ls.minimize_boozer_penalty_constraints_LBFGS(
                tol=1e-10,
                maxiter=settings["maxiter"],
                constraint_weight=settings["weight"],
                iota=settings["iota"],
                G=ls_problem.G0,
                limited_memory=False,
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
        self.assertTrue(
            compilations == [], "new values retraced or recompiled a solver"
        )
        self.assertTrue(
            second["iota"] != first["iota"], 'second["iota"] != first["iota"]'
        )

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
                ), self.case() as patches:
                    self._case_solves_make_no_implicit_transfers(
                        parity_lane, boozer_type, patches
                    )

    def _case_solves_make_no_implicit_transfers(
        self, parity_lane, boozer_type, patches
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
        self.assertTrue(
            {
                device.platform
                for leaf in jax.tree.leaves(coils)
                for device in leaf.devices()
            }
            == {parity_lane},
            "{device.platform for leaf in jax.tree.leaves(coils) for device in leaf.devices()} == {parity_lane}",
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
        """Solver construction rejects unsupported surface classes."""
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

    def test_ls_adjoint_with_a_toroidal_flux_label_is_the_derivative_of_the_solve_taylor(
        self,
    ):
        """Solved coil adjoints converge with the native centered Taylor stencil."""
        native, jax_problem, native_objectives, jax_objectives = (
            _toroidal_flux_adjoint_problem()
        )
        jax_handle, native_handle = Iotas(jax_problem.boozer), Iotas(native.boozer)
        # Every free coil DOF moves, by a fraction of its own size.
        x0 = np.asarray(native_handle.x, dtype=np.float64)
        direction = parity_rng(11).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
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
        for name, make in jax_objectives.items():
            central = (plus[name] - minus[name]) / (2 * step)
            with disallow_host_transfers():
                partials = make(jax_problem).dJ(partials=True)
            adjoint = partials(jax_handle) @ direction
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

        previous_errors = {name: 1e9 for name in jax_objectives}
        for step in np.power(2.0, -np.arange(13, 18)):
            plus, minus = restarted_values(1.0), restarted_values(-1.0)
            for name, make in jax_objectives.items():
                adjoint = make(jax_problem).dJ(partials=True)(jax_handle) @ direction
                error = abs((plus[name] - minus[name]) / (2 * step) - adjoint)
                self.assertLess(error, max(1e-9, 0.35 * previous_errors[name]), name)
                previous_errors[name] = error
        native_handle.x = x0
        native.boozer.surface.set_dofs(solution)

    def test_gradients_with_G_from_free_currents_are_the_derivatives_of_the_solve_taylor(
        self,
    ):
        """Solved coil adjoints converge with the native centered Taylor stencil."""
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
        direction = parity_rng(13).standard_normal(x0.size) * np.maximum(
            np.abs(x0), 1.0
        )
        self.assertTrue(
            x0.size > 0
            and any(not current.dofs.all_fixed() for current in native.currents),
            "x0.size > 0 and any(not current.dofs.all_fixed() for current in native.currents)",
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
        for name, objective in jax_objectives.items():
            central = (plus[name] - minus[name]) / (2 * step)
            with disallow_host_transfers():
                partials = objective.dJ(partials=True)
            adjoint = partials(jax_handle) @ direction
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

        previous_errors = {name: 1e9 for name in jax_objectives}
        for step in np.power(2.0, -np.arange(13, 18)):
            plus, minus = restarted_values(1.0), restarted_values(-1.0)
            for name, make in jax_objectives.items():
                adjoint = make.dJ(partials=True)(jax_handle) @ direction
                error = abs((plus[name] - minus[name]) / (2 * step) - adjoint)
                self.assertLess(error, max(1e-9, 0.35 * previous_errors[name]), name)
                previous_errors[name] = error
        native_handle.x = x0
        native.boozer.surface.set_dofs(solution)
