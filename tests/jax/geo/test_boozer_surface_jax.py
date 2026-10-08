"""JAX ``BoozerSurfaceJAX`` against native ``BoozerSurface``.

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
``BoozerSurfaceJAX`` give native values and coil gradients for coil-independent
labels with optimized ``G``. ``BoozerResidualJAX`` also differentiates the
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

from jax_test_support import (
    assert_matches_native,
    fixture_jax_runtime_guard,  # noqa: F401
    fixture_parity_lane,  # noqa: F401
    jax_compilations,
    parity_default_device,
    parity_rng,
    place_float64,
)

from dataclasses import dataclass, replace
import copy
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt.configs import get_data
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.geo.surfaceobjectives import (
    Area,
    AspectRatio,
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
from simsopt_jax_adapters.field import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_problem import boozer_problem
from simsopt_jax_adapters.geo import boozer_surface as jax_boozer_surface
from simsopt_jax_adapters.geo.boozer_surface import BoozerResidualJAX, BoozerSurfaceJAX

_IOTA = -0.406
_RTOL = 1e-12
_RTOL_NONSYM = 1e-9
_WEIGHT = 100.0


@dataclass
class _Problem:
    """One of upstream's ``get_boozer_surface(converge=False)`` problems."""

    coils: list
    currents: list
    boozer: BoozerSurface | BoozerSurfaceJAX
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
    ``BoozerSurface`` or, with ``jax_field``, a ``BoozerSurfaceJAX``. Without
    ``options``, the options turn ``verbose`` off and set ``weight_inv_modB``
    and ``newton_tol`` (if given); ``options`` itself goes to
    ``BoozerSurfaceJAX`` and a copy to native, which fills it in place."""
    _, base_currents, axis, nfp, bs = get_data("ncsx")
    G0 = _G0(nfp, base_currents) if optimize_G else None
    if not optimize_G:
        for coil in bs.coils:
            coil.current.fix_all()
    mpol = ntor = 6 if boozer_type == "exact" else 3
    nphi, ntheta = (2 * ntor + 1, 2 * mpol + 1) if boozer_type == "exact" else (20, 20)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / nfp, nphi, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    label_nphi, label_ntheta = label_grid or (None, None)
    labels = {
        "Volume": lambda: Volume(surface, nphi=label_nphi, ntheta=label_ntheta),
        "Area": lambda: Area(surface, nphi=label_nphi, ntheta=label_ntheta),
        "AspectRatio": lambda: AspectRatio(surface, nphi=label_nphi, ntheta=label_ntheta),
        "ToroidalFlux": lambda: ToroidalFlux(surface, BiotSavart(bs.coils), nphi=label_nphi, ntheta=label_ntheta),
    }
    label_object = labels[label]()
    if options is None:
        options = {"verbose": False} if weight_inv_modB else {"verbose": False, "weight_inv_modB": False}
        if newton_tol is not None:
            options["newton_tol"] = newton_tol
    constraint_weight = None if boozer_type == "exact" else _WEIGHT
    boozer = (
        BoozerSurfaceJAX(BiotSavartJAX(bs.coils), surface, label_object, label_object.J(), constraint_weight, options)
        if jax_field
        else BoozerSurface(bs, surface, label_object, label_object.J(), constraint_weight, dict(options))
    )
    return _Problem(bs.coils, base_currents, boozer, G0)


def _pair(*args, **kwargs) -> tuple[_Problem, _Problem]:
    return _problem(False, *args, **kwargs), _problem(True, *args, **kwargs)


def _assert_same_solve(jax_problem: _Problem, native_problem: _Problem, jax_res, native_res, arrays, name, rtol=_RTOL):
    """Native's result keys in native's order; iterations, success and ``G`` as
    natively; the surface, ``iota`` and the named arrays to round-off; and
    native's adjoint solve with the ``PLU`` (to round-off times the condition
    number)."""
    assert list(jax_res) == list(native_res), f"{name}: result keys {list(jax_res)} != native {list(native_res)}"
    for key in ("iter", "success"):
        if key in native_res:
            assert jax_res[key] == native_res[key], f"{name}: {key} {jax_res[key]} != native {native_res[key]}"
    assert (jax_res.get("G") is None) == (native_res.get("G") is None), f"{name}: G presence"
    assert_matches_native(jax_problem.boozer.surface.get_dofs(), native_problem.boozer.surface.get_dofs(), f"{name} surface", rtol)
    for key in ("iota", "G", *arrays):
        if native_res.get(key) is not None:
            assert_matches_native(jax_res[key], native_res[key], f"{name} {key}", rtol)
    if "PLU" in native_res:
        # The adjoint solve magnifies the matrices' round-off difference by their condition number.
        P, L, U = native_res["PLU"]
        jax_P, jax_L, jax_U = jax_res["PLU"]
        rhs = parity_rng(7).standard_normal(P.shape[0])
        assert_matches_native(
            forward_backward(jax_P, jax_L, jax_U, rhs), forward_backward(P, L, U, rhs), f"{name} PLU solve",
            rtol * np.linalg.cond(P @ L @ U),
        )


# --- BoozerExact ----------------------------------------------------------------------


_EXACT_CASES = {
    "stellsym-volume": (True, "Volume", None),
    "stellsym-flux-own-grid": (True, "ToroidalFlux", (51, 51)),
    "nonsym-volume": (False, "Volume", None),
    "nonsym-area-own-grid": (False, "Area", (31, 31)),
}


@pytest.mark.parametrize("name", _EXACT_CASES)
def test_exact_run_code_matches_native(name):
    stellsym, label, grid = _EXACT_CASES[name]
    native, jax_problem = _pair("exact", label, stellsym=stellsym, label_grid=grid, newton_tol=1e-10)
    native_res = native.boozer.run_code(_IOTA, G=native.G0)
    jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
    assert native_res["success"] and native_res["iter"] > 3
    _assert_same_solve(
        jax_problem, native, jax_res, native_res, ("jacobian",), name, _RTOL if stellsym else _RTOL_NONSYM
    )
    np.testing.assert_array_equal(jax_res["mask"], native_res["mask"])
    assert jax_res["type"] == "exact" and jax_res["s"] is jax_problem.boozer.surface
    assert np.max(np.abs(jax_res["residual"])) < 1e-11, "the converged residual is not at round-off"
    assert not jax_problem.boozer.need_to_run_code


@pytest.mark.parametrize("stellsym", [True, False], ids=["stellsym", "nonsym"])
def test_exact_newton_at_maxiter_fails_as_natively(stellsym):
    """At ``maxiter`` native reports the norm checked before the last step, so
    the solve fails, also when that step converged (``maxiter=6`` here; PR
    #669 would report success); the surface holds the last iterate, the
    residual and Jacobian are evaluated there. With ``maxiter=0`` nothing
    moves."""
    for maxiter in (2, 6, 0):
        native, jax_problem = _pair("exact", stellsym=stellsym)
        start = native.boozer.surface.get_dofs()
        native_res = native.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=maxiter, iota=_IOTA, G=native.G0)
        jax_res = jax_problem.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=maxiter, iota=_IOTA, G=jax_problem.G0)
        assert not native_res["success"] and native_res["iter"] == maxiter
        rtol = _RTOL if stellsym else _RTOL_NONSYM
        _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), f"maxiter={maxiter}", rtol)
        # Converged residuals are round-off: compare on the scale of the terms (G |B| is 22 here).
        assert_matches_native(jax_res["residual"], native_res["residual"], f"maxiter={maxiter} residual", rtol, scale=22.0)
        if maxiter == 0:
            np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), start)


def test_exact_newton_on_a_singular_system_raises_as_natively():
    """Without current the field vanishes, so do the residual rows; with the
    label off its target, native's ``np.linalg.solve`` raises at the first
    step, before moving the surface."""
    native, jax_problem = _pair("exact")
    for problem in (native, jax_problem):
        for current in problem.currents:
            current.local_full_x = np.zeros(1)
        problem.boozer.targetlabel *= 1.1
        start = problem.boozer.surface.get_dofs()
        with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
            problem.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=5, iota=_IOTA, G=native.G0)
        np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)
        assert problem.boozer.need_to_run_code


def test_non_finite_solves_fail_as_natively():
    """A NaN ``G`` makes every BoozerExact step NaN: as natively the walk runs
    to ``maxiter`` (a NaN norm never meets the tolerance), the surface takes
    the NaN iterate and SciPy's ``lu`` refuses the final Jacobian. A weighted
    BoozerLS penalty without current is 0/0: the Newton loop does not start
    and ``lu`` refuses the Hessian before the surface moves."""
    native, jax_problem = _pair("exact")
    for problem in (native, jax_problem):
        with pytest.raises(ValueError, match="infs or NaNs"):
            problem.boozer.solve_residual_equation_exactly_newton(tol=1e-10, maxiter=3, iota=_IOTA, G=np.nan)
        assert np.all(np.isnan(problem.boozer.surface.get_dofs()))

    native, jax_problem = _pair("ls", weight_inv_modB=True)
    for problem in (native, jax_problem):
        for current in problem.currents:
            current.local_full_x = np.zeros(1)
        start = problem.boozer.surface.get_dofs()
        with pytest.raises(ValueError, match="infs or NaNs"):
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11, maxiter=5, constraint_weight=_WEIGHT, iota=_IOTA, G=native.G0, weight_inv_modB=True
            )
        np.testing.assert_array_equal(problem.boozer.surface.get_dofs(), start)


def test_exact_newton_takes_G_from_the_currents():
    native, jax_problem = _pair("exact", optimize_G=False, newton_tol=1e-10)
    native_res = native.boozer.run_code(_IOTA)
    jax_res = jax_problem.boozer.run_code(_IOTA)
    _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), "G from currents")


# --- BoozerLS ---------------------------------------------------------------------------


@pytest.mark.parametrize("weight_inv_modB", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("optimize_G", [True, False], ids=["G", "G-from-currents"])
def test_ls_run_code_matches_native(optimize_G, weight_inv_modB):
    """BFGS, then the penalty Newton. Near convergence BFGS's last line
    searches follow round-off (its iteration count can differ by one or two),
    so the polished solution, the Newton step count and flags are compared."""
    native, jax_problem = _pair("ls", optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
    native_res = native.boozer.run_code(_IOTA, G=native.G0)
    jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
    assert native_res["success"]
    _assert_same_solve(jax_problem, native, jax_res, native_res, ("hessian",), "run_code")
    assert jax_res["type"] == "ls" and jax_res["weight_inv_modB"] == weight_inv_modB
    assert np.linalg.norm(jax_res["jacobian"]) <= 1e-11


@pytest.mark.parametrize("limited_memory", [False, True], ids=["BFGS", "L-BFGS-B"])
@pytest.mark.parametrize("stellsym", [True, False], ids=["stellsym", "nonsym"])
def test_bfgs_matches_native_up_to_maxiter(limited_memory, stellsym):
    """Capped before convergence (a failed solve, kept as natively). The two
    walks separate by round-off growth: iterates by 3e-13 after 15 iterations,
    1e-9 after 30 on this problem."""
    maxiter = 15
    native, jax_problem = _pair("ls", stellsym=stellsym)
    native_res, jax_res = (
        problem.boozer.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=maxiter, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0,
            limited_memory=limited_memory, weight_inv_modB=False,
        )
        for problem in (native, jax_problem)
    )
    assert not native_res["success"] and native_res["iter"] == maxiter
    assert jax_res["info"].nfev == native_res["info"].nfev
    _assert_same_solve(jax_problem, native, jax_res, native_res, ("fun",), "BFGS", rtol=1e-11)
    # The Hessian magnifies the iterates' difference (measured 8e-11 of the largest entry).
    assert_matches_native(jax_res["gradient"], native_res["gradient"], "BFGS gradient", 1e-9)
    assert jax_res["s"] is jax_problem.boozer.surface and not jax_problem.boozer.need_to_run_code


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
        tol=tol, maxiter=bfgs_maxiter, constraint_weight=constraint_weight, iota=iota, G=native.G0,
        limited_memory=limited_memory, weight_inv_modB=weight_inv_modB,
    )
    jax_problem.boozer.surface.set_dofs(native.boozer.surface.get_dofs())
    native.boozer.recompute_bell()
    return res["iota"], res["G"]


@pytest.mark.parametrize("stab", [0.0, 1e-4])
def test_penalty_newton_matches_native(stab):
    """From a BFGS iterate; ``stab`` shifts the Hessian of each step but not
    the returned one."""
    native, jax_problem = _pair("ls")
    iota, G = _same_start(native, jax_problem, 200)
    native_res, jax_res = (
        problem.boozer.minimize_boozer_penalty_constraints_newton(
            tol=1e-11, maxiter=20, constraint_weight=_WEIGHT, iota=iota, G=G, stab=stab, weight_inv_modB=False
        )
        for problem in (native, jax_problem)
    )
    assert native_res["success"] and native_res["iter"] > 1
    _assert_same_solve(jax_problem, native, jax_res, native_res, ("hessian",), f"stab={stab}")
    assert jax_res["residual"] is jax_res["jacobian"]


def test_penalty_newton_diverges_and_keeps_its_iterate_as_natively():
    """From an early BFGS iterate the undamped Newton walk diverges: as natively
    (no divergence guard, no restore) it runs to ``maxiter``, fails and leaves
    the diverged iterate in the surface. The walk is chaotic, so only its
    outcome is compared; a two-step walk is compared iterate by iterate."""
    for maxiter in (2, 40):
        native, jax_problem = _pair("ls")
        iota, G = _same_start(native, jax_problem, 40)
        native_res, jax_res = (
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11, maxiter=maxiter, constraint_weight=_WEIGHT, iota=iota, G=G, weight_inv_modB=False
            )
            for problem in (native, jax_problem)
        )
        assert not native_res["success"] and not jax_res["success"]
        assert native_res["iter"] == jax_res["iter"] == maxiter
        if maxiter == 2:
            # Diverging steps magnify round-off (measured 1e-11 of the largest entry).
            _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian", "hessian"), "two steps", 1e-9)
        else:
            assert np.linalg.norm(native_res["jacobian"]) > 1e3 and np.linalg.norm(jax_res["jacobian"]) > 1e3


@pytest.mark.parametrize("stab", [np.nan, np.inf], ids=["nan", "inf"])
def test_penalty_newton_with_a_non_finite_shift_fails_as_natively(stab):
    """Native's ``H + stab * I`` is NaN off the diagonal too (IEEE ``inf * 0``),
    so the first step is NaN, the loop stops on the NaN gradient norm and SciPy's
    ``lu`` refuses the Hessian: the surface keeps the NaN iterate."""
    native, jax_problem = _pair("ls")
    for problem in (native, jax_problem):
        start = problem.boozer.surface.get_dofs()
        with pytest.raises(ValueError, match="infs or NaNs"):
            problem.boozer.minimize_boozer_penalty_constraints_newton(
                tol=1e-11, maxiter=2, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0, stab=stab,
                weight_inv_modB=False,
            )
        assert problem.boozer.need_to_run_code
        assert np.all(np.isnan(problem.boozer.surface.get_dofs())) and not np.any(np.isnan(start))
    np.testing.assert_array_equal(jax_problem.boozer.surface.get_dofs(), native.boozer.surface.get_dofs())


def _singular_at_second_factorisation(monkeypatch, solver_name: str, native_solves_per_step: int):
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

    monkeypatch.setattr(np.linalg, "solve", solve)
    monkeypatch.setattr(jax_boozer_surface, solver_name, solver)


@pytest.mark.parametrize("method", ["exact", "newton", "manual"])
def test_singular_step_after_a_step_leaves_the_last_iterate_as_natively(monkeypatch, method):
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
    native, jax_problem = _pair("exact" if method == "exact" else "ls")
    start = native.boozer.surface.get_dofs()
    solver_name, solves_per_step = {
        "exact": ("boozer_exact_newton", 2),
        "newton": ("boozer_penalty_newton", 1),
        "manual": ("boozer_penalty_gauss_newton", 1),
    }[method]
    with monkeypatch.context() as patch:
        _singular_at_second_factorisation(patch, solver_name, solves_per_step)
        for problem in (native, jax_problem):
            options = dict(tol=1e-11, maxiter=5, iota=_IOTA, G=problem.G0)
            with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
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
            assert problem.boozer.need_to_run_code
    assert np.max(np.abs(native.boozer.surface.get_dofs() - start)) > 1e-6, "the first step did not move the surface"
    assert_matches_native(jax_problem.boozer.surface.get_dofs(), native.boozer.surface.get_dofs(), f"{method} surface")


def _least_squares_problem(jax_field: bool) -> _Problem:
    """Upstream's ``subtest_minimize_boozer_penalty_constraints_ls_manual``
    problem (stellarator-symmetric, ``optimize_G``)."""
    _, base_currents, axis, nfp, bs = get_data("ncsx")
    surface = SurfaceXYZTensorFourier(
        mpol=5, ntor=5, stellsym=True, nfp=nfp, clamped_dims=[False, False, False],
        quadpoints_phi=np.linspace(0, 1 / nfp, 11, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 11, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1)
    flux = ToroidalFlux(surface, BiotSavart(bs.coils), nphi=51, ntheta=51)
    field = BiotSavartJAX(bs.coils) if jax_field else bs
    boozer = (BoozerSurfaceJAX if jax_field else BoozerSurface)(field, surface, flux, 0.1)
    return _Problem(bs.coils, base_currents, boozer, _G0(nfp, base_currents))


@pytest.mark.parametrize("method", ["manual", "lm"])
def test_least_squares_methods_match_native(method):
    """Upstream's test problem from its BFGS start; the damped Gauss-Newton
    result is returned but, as natively, not stored. ``lm`` is capped at 8
    evaluations, where its walk still agrees to round-off."""
    native, jax_problem = _least_squares_problem(False), _least_squares_problem(True)
    constraint_weight = 1000.0 / (3 * 11 * 11)
    iota, G = _same_start(
        native, jax_problem, 700, constraint_weight, tol=1e-12, iota=-0.4, limited_memory=True, weight_inv_modB=True
    )
    native_res, jax_res = (
        problem.boozer.minimize_boozer_penalty_constraints_ls(
            tol=1e-8, maxiter=50 if method == "manual" else 8, constraint_weight=constraint_weight,
            iota=iota, G=G, method=method,
        )
        for problem in (native, jax_problem)
    )
    _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), method, 1e-10)
    # At the solution the residuals and J^T r are small differences of O(1) terms.
    assert_matches_native(jax_res["residual"], native_res["residual"], f"{method} residual", 1e-12, scale=1.0)
    assert_matches_native(jax_res["gradient"], native_res["gradient"], f"{method} gradient", 1e-11, scale=1.0)
    assert jax_res["success"] == native_res["success"] and jax_res["s"] is jax_problem.boozer.surface
    if method == "manual":
        assert native_res["success"] and jax_problem.boozer.need_to_run_code
    else:
        assert jax_res["info"].nfev == native_res["info"].nfev and jax_problem.boozer.res is jax_res


@pytest.mark.parametrize("weight_inv_modB", [True, False], ids=["weighted", "unweighted"])
@pytest.mark.parametrize("optimize_G", [True, False], ids=["G", "G-from-currents"])
def test_penalty_residual_matches_native(optimize_G, weight_inv_modB):
    """The least-squares methods' formulation: native
    ``_get_residual_vector_and_jacobian`` with a label on its own grid."""
    native, jax_problem = _pair("ls", "ToroidalFlux", label_grid=(31, 31))
    boozer = jax_problem.boozer
    problem = boozer_problem(boozer.biotsavart, boozer.surface, boozer.label, boozer.targetlabel, 7.0)
    G0 = native.G0
    assert G0 is not None
    x = np.concatenate((boozer.surface.get_dofs(), [_IOTA, G0][: 1 + optimize_G]))
    expected = native.boozer._get_residual_vector_and_jacobian(x.copy(), 7.0, optimize_G, weight_inv_modB)
    placed = place_float64(x, problem.target_label)
    for derivatives in (0, 1):
        actual = boozer_penalty_residual(
            problem, placed, derivatives=derivatives, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        for order in range(derivatives + 1):
            assert_matches_native(actual[order], expected[order], f"penalty residual [{derivatives}][{order}]")


def test_need_to_run_code_caches_results_as_natively():
    jax_problem = _problem(True, "exact")
    boozer = jax_problem.boozer
    res = boozer.run_code(_IOTA, G=jax_problem.G0)
    assert boozer.run_code(_IOTA, G=jax_problem.G0) is None
    for solve in (
        boozer.solve_residual_equation_exactly_newton,
        boozer.minimize_boozer_penalty_constraints_LBFGS,
        boozer.minimize_boozer_penalty_constraints_newton,
        boozer.minimize_boozer_penalty_constraints_ls,
    ):
        assert solve(iota=0.3) is res
    jax_problem.currents[0].local_full_x = 1.01 * jax_problem.currents[0].local_full_x
    assert boozer.need_to_run_code, "a coil change does not mark the surface for a new solve"


# --- native objectives and adjoints ---------------------------------------------


_OBJECTIVE_CASES = {
    "exact-volume": ("exact", "Volume", True, False),
    "exact-flux-own-grid": ("exact", "ToroidalFlux", True, False),
    "ls-weighted": ("ls", "Volume", True, True),
    "ls-G-from-currents": ("ls", "Volume", False, False),
}


@pytest.mark.parametrize("name", _OBJECTIVE_CASES)
def test_native_objectives_use_the_jax_surface(name):
    """Native objectives on solved surfaces: values and coil gradients (through
    ``res['PLU']`` and ``res['vjp']``) as with native ``BoozerSurface``. With
    a Volume label and ``G`` optimized or the currents fixed,
    ``BoozerResidualJAX`` is native ``BoozerResidual`` too, also after a coil
    change that makes both re-solve; on BoozerExact both raise ``KeyError``
    (exact results have no ``weight_inv_modB``)."""
    boozer_type, label, optimize_G, weight_inv_modB = _OBJECTIVE_CASES[name]
    native, jax_problem = _pair(
        boozer_type, label, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
        label_grid=(51, 51) if label == "ToroidalFlux" else None,
    )
    for problem in (native, jax_problem):
        problem.boozer.run_code(_IOTA, G=problem.G0)
    objectives = {
        "Iotas": lambda problem: Iotas(problem.boozer),
        "MajorRadius": lambda problem: MajorRadius(problem.boozer),
        "NonQuasiSymmetricRatio": lambda problem: NonQuasiSymmetricRatio(problem.boozer, BiotSavart(problem.coils)),
    }
    if boozer_type == "ls":
        objectives["BoozerResidual"] = lambda problem: BoozerResidual(problem.boozer, BiotSavart(problem.coils))
    for objective_name, make in objectives.items():
        native_objective, jax_objective = make(native), make(jax_problem)
        assert_matches_native(jax_objective.J(), native_objective.J(), f"{name} {objective_name}")
        assert_matches_native(jax_objective.dJ(), native_objective.dJ(), f"{name} {objective_name} coil gradient")
    native_residual = BoozerResidual(native.boozer, BiotSavart(native.coils))
    jax_residual = BoozerResidualJAX(jax_problem.boozer, jax_problem.boozer.biotsavart)
    if boozer_type == "exact":
        for objective in (native_residual, jax_residual):
            with pytest.raises(KeyError, match="weight_inv_modB"):
                objective.J()
        return
    for change in ("solved", "after a coil change"):
        value = jax_residual.J()
        assert isinstance(value, np.float64), f"{name}: J is {type(value)}, native's is np.float64"
        assert_matches_native(value, native_residual.J(), f"{name} BoozerResidualJAX {change}")
        assert_matches_native(jax_residual.dJ(), native_residual.dJ(), f"{name} BoozerResidualJAX gradient {change}")
        for problem in (native, jax_problem):
            problem.currents[0].local_full_x = 1.0001 * problem.currents[0].local_full_x
        assert jax_problem.boozer.need_to_run_code and jax_residual._J is None, "a coil change left a stale value"


@pytest.mark.parametrize("label", ["Area", "AspectRatio"])
def test_boozer_residual_jax_matches_native_for_labels_without_coil_dependence(label):
    """On Area and AspectRatio labels (``G`` optimized) native
    ``BoozerResidual`` is the derivative of its value, and
    ``BoozerResidualJAX`` agrees with it. Both are compared at a shared state:
    a native BFGS iterate given to both surfaces, stored with its Hessian by a
    penalty Newton call with ``maxiter=0`` (on this fixture these labels'
    ``run_code`` walks follow round-off to different solutions, and the
    penalty Newton from the BFGS iterate does not converge). Comparing on
    the same solved surface isolates the objective's derivative from the
    ill-conditioned Hessian's amplification of different solver round-off."""
    native, jax_problem = _pair("ls", label)
    iota, G = _same_start(native, jax_problem, 200)
    for problem in (native, jax_problem):
        problem.boozer.minimize_boozer_penalty_constraints_newton(
            tol=1e-11, maxiter=0, constraint_weight=_WEIGHT, iota=iota, G=G, weight_inv_modB=False
        )
    jax_residual = BoozerResidualJAX(jax_problem.boozer, jax_problem.boozer.biotsavart)
    # Native BoozerResidual on the same JAX surface (same res, PLU and vjp): the objective alone.
    on_jax_surface = BoozerResidual(jax_problem.boozer, BiotSavart(jax_problem.coils))
    assert_matches_native(jax_residual.J(), on_jax_surface.J(), f"{label} BoozerResidualJAX")
    assert_matches_native(jax_residual.dJ(), on_jax_surface.dJ(), f"{label} BoozerResidualJAX gradient")


def test_boozer_residual_owns_its_surface_and_reads_changed_values_with_a_copied_field():
    """The residual copy uses solved quadrature; a label sharing the solved
    DOFs uses its own grid. A copied field follows later coil changes, and
    recomputing replaces private-copy edits and reads new target/weight values
    without recompilation."""
    problem = _problem(True, "ls", label_grid=(31, 31))
    boozer = problem.boozer
    boozer.minimize_boozer_penalty_constraints_newton(
        maxiter=0, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0, weight_inv_modB=False
    )
    field = copy.copy(boozer.biotsavart)
    residual = BoozerResidualJAX(boozer, field)
    native = BoozerResidual(boozer, BiotSavart(problem.coils))
    assert residual.surface.dofs is not boozer.surface.dofs
    assert boozer.label.surface.dofs is boozer.surface.dofs
    assert boozer.label.surface.quadpoints_phi.size == 31
    np.testing.assert_array_equal(residual.surface.quadpoints_phi, boozer.surface.quadpoints_phi)
    np.testing.assert_array_equal(residual.surface.quadpoints_theta, boozer.surface.quadpoints_theta)
    solved_dofs = boozer.surface.get_dofs().copy()
    first = residual.J()
    assert_matches_native(first, native.J(), "private surface and label grids")
    assert_matches_native(residual.dJ(), native.dJ(), "copied field gradient")
    copied = copy.copy(residual)
    assert copied.surface.dofs is not residual.surface.dofs
    assert copied._J is None and copied._dJ is None
    assert copied.constraint_weight == residual.constraint_weight
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
    assert compilations == [], "new residual target or weight recompiled the objective"
    assert changed != first
    np.testing.assert_array_equal(residual.surface.get_dofs(), solved_dofs)
    assert_matches_native(changed, native.J(), "changed target and weight")
    assert_matches_native(gradient, native.dJ(), "changed target and weight gradient")
    problem.currents[0].local_full_x *= 1.001
    assert boozer.need_to_run_code and residual._J is None and residual._dJ is None
    assert copied._J is None and copied._dJ is None, "a copied objective kept stale coil caches"
    boozer.minimize_boozer_penalty_constraints_newton(
        maxiter=0, constraint_weight=_WEIGHT, iota=_IOTA, G=problem.G0, weight_inv_modB=False
    )
    assert_matches_native(residual.J(), native.J(), "copied field after a coil change")
    assert_matches_native(residual.dJ(), native.dJ(), "copied field gradient after a coil change")
    fresh = BoozerResidualJAX(boozer, boozer.biotsavart)
    assert_matches_native(copied.J(), fresh.J(), "copied objective after a coil change")
    assert_matches_native(copied.dJ(), fresh.dJ(), "copied objective gradient after a coil change")


def test_ls_adjoint_with_a_toroidal_flux_label_is_the_derivative_of_the_solve():
    """With a ToroidalFlux label the BoozerLS penalty depends on the coils
    through the label too. The coil gradients of native objectives on a
    ``BoozerSurfaceJAX`` are the central differences of their values through
    native re-solves (``run_code`` from the solution); native's own adjoint
    (``boozer_surface_dlsqgrad_dcoils_vjp``) drops the label term and is not.
    ``BoozerResidual``'s value also depends on the coils through the label:
    ``BoozerResidualJAX`` differentiates it; native ``BoozerResidual`` takes
    its explicit derivative through ``B`` only and is not the derivative."""
    native, jax_problem = _pair("ls", "ToroidalFlux", label_grid=(31, 31))
    objectives = {
        "Iotas": lambda problem: Iotas(problem.boozer),
        "MajorRadius": lambda problem: MajorRadius(problem.boozer),
        "NonQuasiSymmetricRatio": lambda problem: NonQuasiSymmetricRatio(problem.boozer, BiotSavart(problem.coils)),
        "BoozerResidual": lambda problem: BoozerResidual(problem.boozer, BiotSavart(problem.coils)),
    }
    jax_objectives = {
        **objectives,
        "BoozerResidual": lambda problem: BoozerResidualJAX(problem.boozer, problem.boozer.biotsavart),
    }
    for problem in (native, jax_problem):
        problem.boozer.run_code(_IOTA, G=problem.G0)
    native_objectives = {name: make(native) for name, make in objectives.items()}
    jax_handle, native_handle = Iotas(jax_problem.boozer), Iotas(native.boozer)
    # Every free coil DOF moves, by a fraction of its own size.
    x0 = np.asarray(native_handle.x, dtype=np.float64)
    direction = parity_rng(11).standard_normal(x0.size) * np.maximum(np.abs(x0), 1.0)
    step = 1e-6

    def values(sign: float) -> dict[str, float]:
        native_handle.x = x0 + sign * step * direction
        return {name: objective.J() for name, objective in native_objectives.items()}

    plus, minus = values(1.0), values(-1.0)
    native_handle.x = x0
    for name, make in jax_objectives.items():
        central = (plus[name] - minus[name]) / (2 * step)
        with disallow_host_transfers():
            partials = make(jax_problem).dJ(partials=True)
        adjoint = partials(jax_handle) @ direction
        native_adjoint = native_objectives[name].dJ(partials=True)(native_handle) @ direction
        # Truncation and the re-solves' tolerances bound the agreement (measured 2e-9 to 2e-8 relative);
        # native's gradients are off by 1.8e-3 to 3e-2.
        assert abs(adjoint - central) <= 1e-7 * abs(central), f"{name}: adjoint {adjoint} != difference {central}"
        assert abs(native_adjoint - central) > 1e-4 * abs(central), f"{name}: native's adjoint matches the difference"


def test_gradients_with_G_from_free_currents_are_the_derivatives_of_the_solve():
    """Solved directly with ``G=None`` and free currents (``run_code`` refuses
    this; the solvers do not), ``G`` follows the currents. The coil gradients
    of ``Iotas`` and ``BoozerResidualJAX`` on a ``BoozerSurfaceJAX`` are the
    central differences of the native objectives' values through native
    re-solves (penalty Newton from the solution); native's own drop
    ``dG/dI``."""
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
        assert res["success"], "the penalty Newton with G from the currents did not converge"
    solution, iota = native.boozer.surface.get_dofs(), native.boozer.res["iota"]
    native_objectives = {
        "Iotas": Iotas(native.boozer),
        "BoozerResidual": BoozerResidual(native.boozer, BiotSavart(native.coils)),
    }
    jax_objectives = {
        "Iotas": Iotas(jax_problem.boozer),
        "BoozerResidual": BoozerResidualJAX(jax_problem.boozer, jax_problem.boozer.biotsavart),
    }
    native_handle, jax_handle = native_objectives["Iotas"], jax_objectives["Iotas"]
    x0 = np.asarray(native_handle.x, dtype=np.float64)
    direction = parity_rng(13).standard_normal(x0.size) * np.maximum(np.abs(x0), 1.0)
    assert x0.size > 0 and any(not current.dofs.all_fixed() for current in native.currents)
    # Central-difference truncation is 9e-6 relative at 1e-5 and 9e-8 at 1e-6 (Iotas), so the step is 1e-7.
    step = 1e-7

    def values(sign: float) -> dict[str, float]:
        native_handle.x = x0 + sign * step * direction
        native.boozer.surface.set_dofs(solution)
        native.boozer.minimize_boozer_penalty_constraints_newton(iota=iota, **solve)
        return {name: objective.J() for name, objective in native_objectives.items()}

    plus, minus = values(1.0), values(-1.0)
    native_handle.x = x0
    native.boozer.surface.set_dofs(solution)
    native.boozer.minimize_boozer_penalty_constraints_newton(iota=iota, **solve)
    for name, objective in jax_objectives.items():
        central = (plus[name] - minus[name]) / (2 * step)
        with disallow_host_transfers():
            partials = objective.dJ(partials=True)
        adjoint = partials(jax_handle) @ direction
        native_adjoint = native_objectives[name].dJ(partials=True)(native_handle) @ direction
        # Truncation and the re-solves' tolerances bound the agreement (measured 6e-9 and 9e-11 relative);
        # native's gradients are off by 2.3e-4 and 7.1e-4.
        assert abs(adjoint - central) <= 1e-7 * abs(central), f"{name}: adjoint {adjoint} != difference {central}"
        assert abs(native_adjoint - central) > 1e-4 * abs(central), f"{name}: native's gradient matches the difference"


def test_exact_adjoint_without_stellsym_where_native_raises():
    """Native's BoozerExact ``vjp`` takes the label multiplier from the last
    entry and fails without stellarator symmetry, where the system has a
    ``z(0, 0)`` row too. The JAX ``vjp`` is checked against native's own pieces
    with the label multiplier second to last (simsopt PR #669's correction)."""
    native, jax_problem = _pair("exact", "ToroidalFlux", stellsym=False, newton_tol=1e-10)
    for problem in (native, jax_problem):
        problem.boozer.run_code(_IOTA, G=problem.G0)
    res = native.boozer.res
    lm = parity_rng(3).standard_normal(res["jacobian"].shape[0])
    with pytest.raises(ValueError):
        boozer_surface_dexactresidual_dcoils_dcurrents_vjp(lm, native.boozer, res["iota"], res["G"])

    residual_dB = boozer_surface_residual_dB(native.boozer.surface, res["iota"], res["G"], native.boozer.biotsavart)
    assert residual_dB is not None
    dres_dB = residual_dB[1]
    multipliers = np.zeros(res["mask"].shape)
    multipliers[res["mask"]] = lm[:-2]
    field_cotangent = np.sum(multipliers.reshape((-1, 3))[:, :, None] * dres_dB.reshape((-1, 3, 3)), axis=1)
    expected = native.boozer.biotsavart.B_vjp(field_cotangent) + lm[-2] * native.boozer.label.dJ(partials=True)(
        native.boozer.biotsavart, as_derivative=True
    )
    jax_res = jax_problem.boozer.res
    actual = jax_res["vjp"](lm, jax_problem.boozer, jax_res["iota"], jax_res["G"])
    assert_matches_native(actual(BiotSavart(jax_problem.coils)), expected(native.boozer.biotsavart), "non-stellsym exact vjp", _RTOL_NONSYM)


# --- settings, compilation, transfers and the boundary --------------------------


def test_fixed_surface_dofs_are_solved_for_as_with_every_dof_free():
    """Native ``BoozerSurface`` at 9e027eac3 cannot solve with fixed surface
    DOFs (its label gradient has the free DOFs only and does not fit; PR #669's
    ``dlabel_dsurface`` fixes that). The JAX solve treats every DOF as an
    unknown, as native does with every DOF free."""
    native, jax_problem = _pair("exact", newton_tol=1e-10)
    native.boozer.surface.fix("x(0,0)")
    with pytest.raises(ValueError):
        native.boozer.run_code(_IOTA, G=native.G0)
    native.boozer.surface.unfix_all()
    native_res = native.boozer.run_code(_IOTA, G=native.G0)
    jax_problem.boozer.surface.fix("x(0,0)")
    jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
    _assert_same_solve(jax_problem, native, jax_res, native_res, ("jacobian",), "fixed DOFs")


def test_settings_are_read_at_every_solve():
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
            problem.boozer.constraint_weight = None if problem.boozer.constraint_weight is None else 30.0
            problem.boozer.options["newton_maxiter"] = 2 if problem.boozer.boozer_type == "exact" else 0
            problem.currents[0].local_full_x = 1.002 * problem.currents[0].local_full_x
        native_res, jax_res = (
            problem.boozer.run_code(problem.boozer.res["iota"], G=problem.boozer.res["G"]) for problem in problems
        )
        assert not native_res["success"] and native_res["iter"] == (2 if native_res["type"] == "exact" else 0)
        _assert_same_solve(problems[1], problems[0], jax_res, native_res, ("jacobian",), "new settings", 1e-10)
    assert options == {"verbose": False, "weight_inv_modB": False, "newton_tol": 1e-10}


def test_new_values_reuse_the_compiled_programs():
    """New coils, targets, weights, starting points, tolerances, caps and
    ``stab`` reuse every solver and adjoint program."""

    def solve_all(exact_problem: _Problem, ls_problem: _Problem, settings: dict):
        exact, ls = exact_problem.boozer, ls_problem.boozer
        ls.constraint_weight = settings["weight"]
        res = exact.solve_residual_equation_exactly_newton(
            tol=settings["tol"], maxiter=settings["maxiter"], iota=settings["iota"], G=exact_problem.G0
        )
        Iotas(exact).dJ()
        start = ls.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=settings["maxiter"], constraint_weight=settings["weight"], iota=settings["iota"],
            G=ls_problem.G0, limited_memory=False, weight_inv_modB=False,
        )
        for method in ("manual", "lm"):
            ls.recompute_bell()
            ls.minimize_boozer_penalty_constraints_ls(
                tol=settings["tol"], maxiter=3, constraint_weight=settings["weight"], iota=start["iota"],
                G=start["G"], method=method, weight_inv_modB=False,
            )
        ls.recompute_bell()
        ls.minimize_boozer_penalty_constraints_newton(
            tol=settings["tol"], maxiter=2, constraint_weight=settings["weight"], iota=start["iota"],
            G=start["G"], stab=settings["stab"], weight_inv_modB=False,
        )
        Iotas(ls).dJ()
        BoozerResidualJAX(ls, ls.biotsavart).dJ()
        for boozer in (exact, ls):
            boozer.recompute_bell()
        return res

    exact_problem, ls_problem = _problem(True, "exact"), _problem(True, "ls")
    first = solve_all(exact_problem, ls_problem, {"tol": 1e-10, "maxiter": 12, "iota": _IOTA, "weight": _WEIGHT, "stab": 0.0})
    for problem in (exact_problem, ls_problem):
        problem.currents[0].local_full_x = 1.003 * problem.currents[0].local_full_x
        problem.boozer.surface.set_dofs(problem.boozer.surface.get_dofs() * 1.001)
        problem.boozer.targetlabel *= 1.01
    with jax_compilations() as compilations:
        second = solve_all(
            exact_problem, ls_problem, {"tol": 2e-10, "maxiter": 15, "iota": -0.41, "weight": 50.0, "stab": 1e-4}
        )
    assert compilations == [], "new values retraced or recompiled a solver"
    assert second["iota"] != first["iota"]


@pytest.mark.parametrize("boozer_type", ["exact", "ls"])
def test_solves_make_no_implicit_transfers(boozer_type, parity_lane):
    native = _problem(False, boozer_type)
    with parity_default_device(parity_lane):
        jax_problem = _problem(True, boozer_type)
        with disallow_host_transfers():
            jax_res = jax_problem.boozer.run_code(_IOTA, G=jax_problem.G0)
            jax_dJ = Iotas(jax_problem.boozer).dJ()
            if boozer_type == "ls":
                jax_residual_dJ = BoozerResidualJAX(jax_problem.boozer, jax_problem.boozer.biotsavart).dJ()
                jax_problem.boozer.recompute_bell()
                jax_problem.boozer.minimize_boozer_penalty_constraints_ls(
                    tol=1e-8, maxiter=2, constraint_weight=_WEIGHT, iota=jax_res["iota"], G=jax_res["G"],
                    method="lm", weight_inv_modB=False,
                )
    coils = jax_problem.boozer.biotsavart.coil_set_spec()
    assert {device.platform for leaf in jax.tree.leaves(coils) for device in leaf.devices()} == {parity_lane}
    native_res = native.boozer.run_code(_IOTA, G=native.G0)
    assert_matches_native(jax_res["iota"], native_res["iota"], f"{boozer_type} iota on {parity_lane}", 1e-10)
    assert_matches_native(jax_dJ, Iotas(native.boozer).dJ(), f"{boozer_type} Iotas gradient on {parity_lane}", 1e-9)
    if boozer_type == "ls":
        native_residual_dJ = BoozerResidual(native.boozer, BiotSavart(native.coils)).dJ()
        assert_matches_native(jax_residual_dJ, native_residual_dJ, f"BoozerResidualJAX gradient on {parity_lane}", 1e-9)


def test_boundary_refuses_unsupported_inputs():
    _, _, axis, nfp, bs = get_data("ncsx")
    field = BiotSavartJAX(bs.coils)
    rz = SurfaceRZFourier(mpol=2, ntor=2, nfp=nfp)
    with pytest.raises(Exception, match="SurfaceXYZTensorFourier or SurfaceXYZFourier"):
        BoozerSurfaceJAX(field, rz, Volume(rz), 1.0)

    xyz = SurfaceXYZFourier(mpol=2, ntor=2, nfp=nfp, quadpoints_phi=np.linspace(0, 1 / nfp, 6, endpoint=False),
                            quadpoints_theta=np.linspace(0, 1, 7, endpoint=False))
    xyz.fit_to_curve(axis, 0.1, flip_theta=True)
    with pytest.raises(RuntimeError, match="SurfaceXYZTensorFourier"):
        BoozerSurfaceJAX(field, xyz, Volume(xyz), 1.0).solve_residual_equation_exactly_newton(iota=_IOTA)
    with pytest.raises(TypeError, match="BiotSavartJAX"):
        BoozerSurfaceJAX(bs, xyz, Volume(xyz), 1.0, 1.0).minimize_boozer_penalty_constraints_LBFGS(maxiter=1)
    with pytest.raises(TypeError, match="labels"):
        BoozerSurfaceJAX(field, xyz, PrincipalCurvature(xyz), 1.0, 1.0).minimize_boozer_penalty_constraints_LBFGS(maxiter=1)
    # As natively, G from the currents needs fixed currents for coil gradients.
    with pytest.raises(AssertionError):
        BoozerSurfaceJAX(field, xyz, Volume(xyz), 1.0, 1.0).run_code(_IOTA)
