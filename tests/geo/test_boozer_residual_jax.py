"""JAX Boozer residual formulations against native ``BoozerSurface``.

``simsopt_jax.core.boozer_problem`` is compared with native
``boozer_surface_residual``, ``BoozerSurface.boozer_penalty_constraints_vectorized``,
``BoozerSurface.boozer_exact_constraints`` and the system of
``BoozerSurface.solve_residual_equation_exactly_newton`` (values, first and second
derivatives) for ``SurfaceRZFourier``, ``SurfaceXYZFourier`` and
``SurfaceXYZTensorFourier``, with and without stellarator symmetry, the
``Volume``, ``Area``, ``AspectRatio`` and ``ToroidalFlux`` labels (on their own
grids), ``weight_inv_modB`` and ``optimize_G`` on and off. Hessians are also
compared with central differences of native gradients; coil derivatives of the
derivatives (through the field's Taylor coefficients) with autodiff through the
field kernel, and the Hessians' nesting by their compiled memory. Then the
problem boundary: fixed DOFs, grids without a stellarator-symmetry mask, no
coils, singular cross-sections for AspectRatio, recompilation, implicit
transfers, zero fields and unsupported inputs.
"""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from simsopt.configs import get_data
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.geo.boozersurface import BoozerSurface
    from simsopt.geo.surfaceobjectives import (
        Area,
        AspectRatio,
        PrincipalCurvature,
        ToroidalFlux,
        Volume,
        boozer_surface_residual,
    )
    from simsopt.geo.surfacerzfourier import SurfaceRZFourier
    from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
    from simsopt_jax.backend import set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import boozer_problem as boozer_problem_module
    from simsopt_jax.core.boozer_problem import (
        BoozerProblem,
        boozer_exact_constraints,
        boozer_exact_residual,
        boozer_penalty_constraints,
        boozer_penalty_residual,
    )
    from simsopt_jax.core.boozer_problem import (
        boozer_surface_residual as jax_boozer_surface_residual,
    )
    from simsopt_jax.core.boozer_solvers import (
        boozer_exact_newton,
        boozer_penalty_bfgs,
        boozer_penalty_gauss_newton,
        boozer_penalty_lbfgsb,
        boozer_penalty_newton,
        boozer_residual_objective,
    )
    from simsopt_jax.runtime.host_boundary import disallow_host_transfers
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_problem import (
        boozer_exact_residual_rows,
        boozer_problem,
    )

from unittest_jax_support import (
    assert_errors_shrink,
    assert_matches_native,
    host_array,
    jax_compilations,
    parity_default_device,
    parity_rng,
    place_float64,
)

from collections import Counter
from dataclasses import dataclass, replace
from functools import partial
from unittest import mock
import re

import numpy as np


if JAX_IMPORT_ERROR is None:
    _NativeSurface = SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier
    _SURFACE_CLASSES = {
        "rz": SurfaceRZFourier,
        "xyz": SurfaceXYZFourier,
        "tensor": SurfaceXYZTensorFourier,
    }
_IOTA = -0.41
_G = 1.3
# Round-off bound relative to the largest native entry: native and JAX sum
# in different orders, and AspectRatio's native value uses det/inv where JAX
# uses the closed form of native's own derivative (measured worst: 1.5e-13).
_RTOL = 1e-12


@dataclass(frozen=True)
class _Case:
    surface: str
    stellsym: bool
    label: str
    optimize_G: bool
    weight_inv_modB: bool
    clamped: bool = False


@dataclass
class _Setup:
    surface: _NativeSurface
    bs: BiotSavart
    currents: list
    field: JaxBiotSavart
    label: Volume | Area | AspectRatio | ToroidalFlux
    target: float


def _surface(
    kind: str,
    stellsym: bool,
    nfp: int,
    clamped: bool = False,
    fit_grid: bool = False,
) -> _NativeSurface:
    """An ``mpol = ntor = 2`` native surface on a test grid or on the fit grid.

    Args:
        kind (str): ``"rz"``, ``"xyz"`` or ``"tensor"``.
        stellsym (bool): Whether the surface is stellarator symmetric.
        nfp (int): Number of field periods.
        clamped (bool): Clamp the tensor surface's ``x`` and ``z`` dimensions.
        fit_grid (bool): Use the full-period ``8 x 9`` fit grid of :func:`_fitted_surface`.

    Returns:
        SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier: The
        surface at zero DOFs.
    """
    if fit_grid:
        quadpoints_phi = np.linspace(0, 1 / nfp, 8, endpoint=False)
        quadpoints_theta = np.linspace(0, 1, 9, endpoint=False)
    elif kind == "tensor":
        # The grid of SurfaceXYZTensorFourier.get_stellsym_mask(), so BoozerExact works too.
        quadpoints_phi = np.linspace(0, 1 / nfp, 5, endpoint=False)
        quadpoints_theta = np.linspace(0, 1, 5, endpoint=False)
    else:
        quadpoints_phi = np.linspace(0, 1 / (2 * nfp), 4, endpoint=False)
        quadpoints_theta = np.linspace(0, 1, 6, endpoint=False)
    clamped_dims = {"clamped_dims": (clamped, False, clamped)} if kind == "tensor" else {}
    return _SURFACE_CLASSES[kind](
        mpol=2,
        ntor=2,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        **clamped_dims,
    )


def _fitted_surface(
    kind: str, stellsym: bool, nfp: int, axis, clamped: bool = False
) -> _NativeSurface:
    """A fit-grid surface fitted to a tube of minor radius 0.1 m about ``axis``.

    Args:
        kind (str): ``"rz"``, ``"xyz"`` or ``"tensor"``.
        stellsym (bool): Whether the surface is stellarator symmetric.
        nfp (int): Number of field periods.
        axis (Curve): Magnetic axis the tube follows.
        clamped (bool): Clamp the tensor surface's ``x`` and ``z`` dimensions.

    Returns:
        SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier: The
        fitted surface; its DOFs are the unique least-squares solution.

    Raises:
        AssertionError: If the fit grid leaves a DOF undetermined. Native's
            ``FullPivHouseholderQR`` then picks build-dependent free DOFs, as on
            the half-period ``4 x 6`` test grid of a non-stellarator-symmetric
            XYZ surface (rank 60 of 75); the ``8 x 9`` fit grid determines every
            DOF of every kind.
    """
    surface = _surface(kind, stellsym, nfp, clamped, fit_grid=True)
    ndofs = surface.get_dofs().size
    rank = np.linalg.matrix_rank(surface.dgamma_by_dcoeff().reshape(-1, ndofs))
    if rank != ndofs:
        raise AssertionError(f"{kind} fit grid determines {rank} of {ndofs} DOFs.")
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    return surface


def _assert_same_coil_pullbacks(test, pullbacks, cotangent, name):
    """The two coil pullbacks of ``cotangent`` agree leaf by leaf, as native values do."""
    leaves = [jax.tree.leaves(pullback(cotangent)[0]) for pullback in pullbacks]
    test.assertTrue(len(leaves[0]) == len(leaves[1]) > 0, "coil leaves")
    for leaf, (a, e) in enumerate(zip(*leaves, strict=True)):
        assert_matches_native(a, host_array(e), f"{name} coil VJP [{leaf}]")


def _setup(
    kind: str, stellsym: bool, label: str, clamped: bool = False, seed: int = 0
) -> _Setup:
    _, currents, axis, nfp, bs = get_data("ncsx")
    surface = _surface(kind, stellsym, nfp, clamped)
    dofs = _fitted_surface(kind, stellsym, nfp, axis, clamped).get_dofs()
    surface.set_dofs(dofs + 1e-3 * parity_rng(seed).standard_normal(dofs.size))
    labels = {
        "volume": lambda: Volume(surface, nphi=6, ntheta=7),
        "area": lambda: Area(surface),
        "aspect_ratio": lambda: AspectRatio(surface, nphi=6, ntheta=7),
        "toroidal_flux": lambda: ToroidalFlux(
            surface, BiotSavart(bs.coils), idx=1, nphi=6, ntheta=7
        ),
    }
    native_label = labels[label]()
    return _Setup(
        surface,
        bs,
        currents,
        JaxBiotSavart(bs.coils),
        native_label,
        1.01 * native_label.J(),
    )


def _native_boozer(
    setup: _Setup, constraint_weight: float | None = None
) -> BoozerSurface:
    """Native ``BoozerSurface``. Its constructor accepts only XYZ surfaces, but
    its formulations evaluate any surface: an RZ surface is assigned afterwards
    so that native's own code is what the JAX results are compared with."""
    if isinstance(setup.surface, SurfaceRZFourier):
        placeholder = SurfaceXYZTensorFourier(mpol=1, ntor=1, nfp=setup.surface.nfp)
        boozer = BoozerSurface(
            setup.bs, placeholder, Volume(placeholder), 1.0, constraint_weight
        )
        boozer.surface, boozer.label, boozer.targetlabel = (
            setup.surface,
            setup.label,
            setup.target,
        )
        return boozer
    return BoozerSurface(
        setup.bs, setup.surface, setup.label, setup.target, constraint_weight
    )


def _decision(surface: _NativeSurface, optimize_G: bool, multipliers=()) -> np.ndarray:
    return np.concatenate(
        (surface.get_dofs(), [_IOTA], [_G] if optimize_G else [], multipliers)
    )


def _outputs(result) -> tuple:
    """A native or JAX result as a tuple: values come bare, derivatives as tuples."""
    return result if isinstance(result, tuple) else (result,)


def _assert_all_native(actual, expected, name: str) -> None:
    actual, expected = _outputs(actual), _outputs(expected)
    assert len(actual) == len(expected), name
    for order, (a, e) in enumerate(zip(actual, expected)):
        assert_matches_native(a, e, f"{name}[{order}]")


# --- the four formulations against native ----------------------------------------


_RESIDUAL_CASES = {
    "rz-stellsym-weighted": _Case("rz", True, "volume", False, True),
    "rz-nonsym-G": _Case("rz", False, "volume", True, False),
    "xyz-stellsym-G-weighted": _Case("xyz", True, "volume", True, True),
    "xyz-nonsym": _Case("xyz", False, "volume", False, False),
    "tensor-stellsym-G": _Case("tensor", True, "volume", True, False),
    "tensor-clamped-nonsym-weighted": _Case(
        "tensor", False, "volume", False, True, clamped=True
    ),
}


_PENALTY_CASES = {
    "tensor-stellsym-flux-weighted": _Case(
        "tensor", True, "toroidal_flux", False, True
    ),
    "tensor-nonsym-volume-G": _Case("tensor", False, "volume", True, False),
    "xyz-stellsym-area-G-weighted": _Case("xyz", True, "area", True, True),
    "xyz-nonsym-aspect-ratio": _Case("xyz", False, "aspect_ratio", False, False),
    "rz-stellsym-volume-weighted": _Case("rz", True, "volume", False, True),
    "rz-nonsym-flux-G-weighted": _Case("rz", False, "toroidal_flux", True, True),
}


_EXACT_CONSTRAINT_CASES = {
    "tensor-stellsym-flux": _Case("tensor", True, "toroidal_flux", False, False),
    "tensor-nonsym-aspect-ratio-G": _Case("tensor", False, "aspect_ratio", True, False),
    "xyz-nonsym-volume-G": _Case("xyz", False, "volume", True, False),
    "rz-stellsym-area": _Case("rz", True, "area", False, False),
}


def _closed_form_mean_cross_sectional_area(surface: _NativeSurface) -> float:
    """Native's mean cross-sectional area in the closed form of its own
    ``dmean_cross_sectional_area_by_dcoeff``, which needs no ``det``/``inv`` of
    the cylindrical map and so stays defined where that map is singular.

    Args:
        surface (SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier):
            Native surface at its current DOFs.

    Returns:
        float: Mean cross-sectional area in square meters.
    """
    gamma, xphi, xtheta = surface.gamma(), surface.gammadash1(), surface.gammadash2()
    x, y = gamma[..., 0], gamma[..., 1]
    section = (
        xtheta[..., 2] * (x * xphi[..., 1] - y * xphi[..., 0])
        - xphi[..., 2] * (x * xtheta[..., 1] - y * xtheta[..., 0])
    ) / np.sqrt(x * x + y * y)
    return float(np.abs(np.mean(section)) / (2 * np.pi))


def _singular_section_surface() -> SurfaceXYZFourier:
    """A smooth surface whose cylindrical angle stops advancing with phi at
    every quadrature point (``J00 = 0`` in native ``mean_cross_sectional_area``)."""
    surface = SurfaceXYZFourier(
        mpol=1,
        ntor=1,
        nfp=1,
        quadpoints_phi=np.array([0.0, 0.2, 0.4, 0.6]),
        quadpoints_theta=np.array([0.125, 0.375, 0.625, 0.875]),
    )
    dofs, names = surface.get_dofs(), list(surface.local_full_dof_names)
    for name, value in (
        ("ys(0,1)", 1.0),
        ("ys(1,1)", 0.05),
        ("ys(1,-1)", -0.05),
        ("zs(0,1)", 0.07),
    ):
        dofs[names.index(name)] = value
    surface.set_dofs(dofs)
    return surface


# --- the problem boundary --------------------------------------------------------


def _evaluate_every_formulation(problem: BoozerProblem, x: np.ndarray, xl: np.ndarray):
    placed_x, placed_xl = (
        place_float64(x, problem.targetlabel),
        place_float64(xl, problem.targetlabel),
    )
    return jax.block_until_ready(
        {
            "residual": jax_boozer_surface_residual(
                problem, placed_x, derivatives=2, optimize_G=True
            ),
            "penalty": boozer_penalty_constraints(
                problem, placed_x, derivatives=2, optimize_G=True
            ),
            "exact constraints": boozer_exact_constraints(
                problem, placed_xl, derivatives=1
            ),
        }
    )


def _assert_every_formulation_native(
    results, setup: _Setup, weight: float, x, xl
) -> None:
    native_boozer = _native_boozer(setup, weight)
    setup.surface.set_dofs(x[:-2])
    _assert_all_native(
        results["residual"],
        boozer_surface_residual(setup.surface, x[-2], x[-1], setup.bs, derivatives=2),
        "residual",
    )
    _assert_all_native(
        results["penalty"],
        native_boozer.boozer_penalty_constraints_vectorized(
            x.copy(), derivatives=2, constraint_weight=weight, optimize_G=True
        ),
        "penalty",
    )
    _assert_all_native(
        results["exact constraints"],
        native_boozer.boozer_exact_constraints(xl.copy(), derivatives=1),
        "exact constraints",
    )


class TestBoozerResidualJax(JaxTestCase):
    def test_surface_residual_matches_native(self):
        """Boozer residuals, Jacobians and Hessians agree with native formulations."""
        for name in _RESIDUAL_CASES:
            with self.subTest(name=name), self.case():
                self._case_surface_residual_matches_native(name)

    def _case_surface_residual_matches_native(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        case = _RESIDUAL_CASES[name]
        setup = _setup(case.surface, case.stellsym, case.label, case.clamped)
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 1.0
        )
        x = place_float64(
            _decision(setup.surface, case.optimize_G), problem.targetlabel
        )
        for derivatives in (0, 1, 2):
            expected = boozer_surface_residual(
                setup.surface,
                _IOTA,
                _G if case.optimize_G else None,
                setup.bs,
                derivatives=derivatives,
                weight_inv_modB=case.weight_inv_modB,
            )
            result = jax_boozer_surface_residual(
                problem,
                x,
                derivatives=derivatives,
                optimize_G=case.optimize_G,
                weight_inv_modB=case.weight_inv_modB,
            )
            _assert_all_native(result, expected, f"{name} boozer_surface_residual")

    def test_penalty_constraints_match_native(self):
        """Value, gradient and Hessian at two constraint weights. Their difference
        isolates the label and ``z(0, 0)`` penalties, compared on their own scale."""
        for name in _PENALTY_CASES:
            with self.subTest(name=name), self.case():
                self._case_penalty_constraints_match_native(name)

    def _case_penalty_constraints_match_native(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        case = _PENALTY_CASES[name]
        setup = _setup(case.surface, case.stellsym, case.label)
        native_boozer = _native_boozer(setup, 1.0)
        x = _decision(setup.surface, case.optimize_G)
        results, expected = {}, {}
        # The weight difference must dominate the residual term it cancels:
        # at 700 the value of some cases is 1.5e4 times its label penalty, so
        # one ulp of the value exceeds 1e-12 of the difference.
        for weight in (0.7, 7e4):
            problem = boozer_problem(
                setup.field, setup.surface, setup.label, setup.target, weight
            )
            for derivatives in (0, 1, 2):
                expected[weight, derivatives] = (
                    native_boozer.boozer_penalty_constraints_vectorized(
                        x.copy(),
                        derivatives=derivatives,
                        constraint_weight=weight,
                        optimize_G=case.optimize_G,
                        weight_inv_modB=case.weight_inv_modB,
                    )
                )
                results[weight, derivatives] = boozer_penalty_constraints(
                    problem,
                    place_float64(x, problem.targetlabel),
                    derivatives=derivatives,
                    optimize_G=case.optimize_G,
                    weight_inv_modB=case.weight_inv_modB,
                )
                _assert_all_native(
                    results[weight, derivatives], expected[weight, derivatives], name
                )
        pairs = zip(
            _outputs(results[7e4, 2]),
            _outputs(results[0.7, 2]),
            _outputs(expected[7e4, 2]),
            _outputs(expected[0.7, 2]),
            strict=True,
        )
        for order, (heavy, light, native_heavy, native_light) in enumerate(pairs):
            assert_matches_native(
                host_array(heavy) - host_array(light),
                native_heavy - native_light,
                f"{name} label and z(0, 0) penalties[{order}]",
            )

    def test_exact_constraints_match_native(self):
        """``res`` and ``dres`` with zero and nonzero multipliers; the difference
        isolates the label and ``z(0, 0)`` derivatives, and the constraint rows and
        columns are compared on their own scale."""
        for name in _EXACT_CONSTRAINT_CASES:
            with self.subTest(name=name), self.case():
                self._case_exact_constraints_match_native(name)

    def _case_exact_constraints_match_native(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        case = _EXACT_CONSTRAINT_CASES[name]
        setup = _setup(case.surface, case.stellsym, case.label)
        native_boozer = _native_boozer(setup)
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        pairs = {}
        for multipliers in ((0.0, 0.0), (0.37, -0.23)):
            xl = _decision(setup.surface, case.optimize_G, multipliers)
            res, dres = map(
                np.asarray,
                _outputs(
                    native_boozer.boozer_exact_constraints(
                        xl.copy(), derivatives=1, optimize_G=case.optimize_G
                    )
                ),
            )
            placed_xl = place_float64(xl, problem.targetlabel)
            assert_matches_native(
                boozer_exact_constraints(
                    problem, placed_xl, optimize_G=case.optimize_G
                ),
                res,
                f"{name} res",
            )
            result = boozer_exact_constraints(
                problem, placed_xl, derivatives=1, optimize_G=case.optimize_G
            )
            _assert_all_native(result, (res, dres), name)
            assert_matches_native(result[0][-2:], res[-2:], f"{name} constraints")
            assert_matches_native(
                result[1][:-2, -2:], dres[:-2, -2:], f"{name} constraint gradients"
            )
            pairs[multipliers] = (
                host_array(result[0]),
                host_array(result[1]),
                res,
                dres,
            )
        zero, nonzero = pairs[(0.0, 0.0)], pairs[(0.37, -0.23)]
        assert_matches_native(
            nonzero[0] - zero[0],
            nonzero[2] - zero[2],
            f"{name} multiplier terms of res",
        )
        assert_matches_native(
            nonzero[1] - zero[1],
            nonzero[3] - zero[3],
            f"{name} multiplier terms of dres",
        )

    def test_exact_residual_matches_native(self):
        """The masked residual, label and ``z(0, 0)`` rows, and their Jacobian, as
        native ``solve_residual_equation_exactly_newton`` stacks them."""
        for label in ["volume", "toroidal_flux"]:
            for case_id_1, stellsym in zip(
                ["stellsym", "nonsym"], [True, False], strict=True
            ):
                with self.subTest(
                    label=label, case_id_1=case_id_1, stellsym=stellsym
                ), self.case():
                    self._case_exact_residual_matches_native(label, stellsym)

    def _case_exact_residual_matches_native(self, label, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        setup = _setup("tensor", stellsym, label)
        native_boozer = _native_boozer(setup)
        native = native_boozer.solve_residual_equation_exactly_newton(
            tol=0.0, maxiter=0, iota=_IOTA, G=_G
        )
        tail = [setup.label.J() - setup.target] + (
            [] if stellsym else [setup.surface.gamma()[0, 0, 2]]
        )
        b = np.concatenate((native["residual"][native["mask"]], tail))
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        rows = boozer_exact_residual_rows(setup.surface)
        x = place_float64(_decision(setup.surface, True), problem.targetlabel)
        assert_matches_native(
            boozer_exact_residual(problem, x, rows), b, "BoozerExact residual"
        )
        result = boozer_exact_residual(problem, x, rows, derivatives=1)
        _assert_all_native(result, (b, native["jacobian"]), "BoozerExact system")
        ntail = len(tail)
        assert_matches_native(result[0][-ntail:], b[-ntail:], "BoozerExact constraints")
        assert_matches_native(
            result[1][-ntail:],
            native["jacobian"][-ntail:],
            "BoozerExact constraint rows",
        )

    def test_exact_residual_jacobian_taylor(self):
        """The BoozerExact Jacobian, and its label and ``z(0, 0)`` rows on their own,
        match forward differences of the system at shrinking steps."""
        for label in ["volume", "toroidal_flux"]:
            for stellsym in [True, False]:
                with self.subTest(label=label, stellsym=stellsym), self.case():
                    self._case_exact_residual_jacobian_taylor(label, stellsym)

    def _case_exact_residual_jacobian_taylor(self, label, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        setup = _setup("tensor", stellsym, label)
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        rows = boozer_exact_residual_rows(setup.surface)
        x = _decision(setup.surface, True)
        direction = parity_rng(1).uniform(size=x.size) - 0.5

        def system(at):
            return host_array(
                boozer_exact_residual(problem, place_float64(at, problem.targetlabel), rows)
            )

        base, jacobian = (
            host_array(output)
            for output in boozer_exact_residual(
                problem, place_float64(x, problem.targetlabel), rows, derivatives=1
            )
        )
        analytic = jacobian @ direction
        # The Boozer rows dominate the norm of the whole system, so the label row
        # gets its own convergence check.
        label_row = rows.size
        differences = [
            (system(x + step * direction) - base) / step
            for step in np.power(2.0, -np.arange(7, 20))
        ]
        assert_errors_shrink(
            self,
            [np.linalg.norm(difference - analytic) for difference in differences],
            0.55,
            initial=1e9,
        )
        assert_errors_shrink(
            self,
            [
                abs(difference[label_row] - analytic[label_row])
                for difference in differences
            ],
            0.55,
            initial=1e9,
        )
        if not stellsym:
            # z(0, 0) is linear in the surface DOFs, so a forward difference equals
            # the Jacobian row up to the round-off of subtracting two z values.
            step = 2.0**-10
            shifted = system(x + step * direction)[-1]
            np.testing.assert_allclose(
                (shifted - base[-1]) / step,
                analytic[-1],
                rtol=0,
                atol=8 * np.finfo(float).eps * (abs(shifted) + abs(base[-1])) / step,
            )

    def test_unmasked_formulations_take_any_grid_and_the_mask_fails_as_natively(self):
        """On a stellarator-symmetric tensor grid that ``get_stellsym_mask()``
        rejects, native's penalty and exact constraints work and so do the JAX
        ones; only the masked BoozerExact system raises, with native's error."""
        _, _, axis, nfp, bs = get_data("ncsx")
        surface = SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1, 8, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 9, endpoint=False),
        )
        surface.fit_to_curve(axis, 0.1, flip_theta=True)
        label = Volume(surface)
        setup = _Setup(
            surface, bs, [], JaxBiotSavart(bs.coils), label, 1.01 * label.J()
        )
        native_boozer = _native_boozer(setup, 2.0)
        exact = boozer_problem(setup.field, surface, label, setup.target)
        xl = _decision(surface, True, (0.3, -0.1))
        _assert_all_native(
            boozer_exact_constraints(
                exact, place_float64(xl, exact.targetlabel), derivatives=1
            ),
            native_boozer.boozer_exact_constraints(xl.copy(), derivatives=1),
            "exact constraints",
        )
        penalty = boozer_problem(setup.field, surface, label, setup.target, 2.0)
        x = _decision(surface, False)
        _assert_all_native(
            boozer_penalty_constraints(
                penalty, place_float64(x, penalty.targetlabel), derivatives=2
            ),
            native_boozer.boozer_penalty_constraints_vectorized(
                x.copy(), derivatives=2, constraint_weight=2.0
            ),
            "penalty",
        )
        message = "require a specific set of quadrature points"
        with self.assertRaisesRegex(Exception, message):
            native_boozer.solve_residual_equation_exactly_newton(
                tol=0.0, maxiter=0, iota=_IOTA, G=_G
            )
        with self.assertRaisesRegex(Exception, message):
            boozer_exact_residual_rows(surface)

    def test_no_coils_give_native_G_and_residuals(self):
        """Without coils native takes ``G = 0`` and a zero field: the unweighted
        residuals vanish and only the label and ``z(0, 0)`` penalties remain."""
        setup = _setup("tensor", False, "volume")
        setup = replace(setup, bs=BiotSavart([]), field=JaxBiotSavart([]))
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 3.0
        )
        for optimize_G in (False, True):
            x = _decision(setup.surface, optimize_G)
            expected = boozer_surface_residual(
                setup.surface,
                _IOTA,
                _G if optimize_G else None,
                setup.bs,
                derivatives=2,
            )
            result = jax_boozer_surface_residual(
                problem,
                place_float64(x, problem.targetlabel),
                derivatives=2,
                optimize_G=optimize_G,
            )
            _assert_all_native(result, expected, "residual without coils")
        x = _decision(setup.surface, False)
        _assert_all_native(
            boozer_penalty_constraints(
                problem,
                place_float64(x, problem.targetlabel),
                derivatives=2,
                weight_inv_modB=False,
            ),
            _native_boozer(setup, 3.0).boozer_penalty_constraints_vectorized(
                x.copy(), derivatives=2, constraint_weight=3.0, weight_inv_modB=False
            ),
            "penalty without coils",
        )

    def test_aspect_ratio_label_matches_native_and_stays_finite_where_native_raises(
        self,
    ):
        """The aspect-ratio label matches native regular values and remains finite at a singular map."""
        setup = _setup("xyz", True, "aspect_ratio")
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        res = boozer_exact_constraints(
            problem,
            place_float64(
                _decision(setup.surface, True, (0.0, 0.0)), problem.targetlabel
            ),
        )
        assert_matches_native(
            res[-2], setup.label.J() - setup.target, "AspectRatio label"
        )

        surface = _singular_section_surface()
        label = AspectRatio(surface)
        with self.assertRaises(np.linalg.LinAlgError):
            label.J()
        problem = boozer_problem(setup.field, surface, label, 5.0, 1.0)
        x = _decision(surface, False)
        singular = _native_boozer(
            replace(setup, surface=surface, label=label, target=5.0), 1.0
        )
        with self.assertRaises(np.linalg.LinAlgError):
            singular.boozer_penalty_constraints_vectorized(x.copy(), derivatives=2)
        # Native with its mean cross-sectional area in the closed form of its own
        # derivatives is the reference: value, gradient and Hessian.
        with mock.patch.object(
            surface,
            "mean_cross_sectional_area",
            lambda: _closed_form_mean_cross_sectional_area(surface),
        ):
            expected = singular.boozer_penalty_constraints_vectorized(
                x.copy(), derivatives=2
            )
        _assert_all_native(
            boozer_penalty_constraints(
                problem, place_float64(x, problem.targetlabel), derivatives=2
            ),
            expected,
            "AspectRatio penalty at a singular cylindrical map",
        )

    def test_hessians_match_differences_of_native_derivatives(self):
        """Penalty and residual Hessians agree with central differences of native
        gradients and Jacobians, and forward differences converge to them at
        shrinking steps."""
        setup = _setup("xyz", False, "toroidal_flux")
        native_boozer = _native_boozer(setup, 5.0)
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 5.0
        )
        x = _decision(setup.surface, True)
        direction = parity_rng(2).standard_normal(x.size)
        step = 1e-6

        def native_gradient(at):
            return _outputs(
                native_boozer.boozer_penalty_constraints_vectorized(
                    at.copy(),
                    derivatives=1,
                    constraint_weight=5.0,
                    optimize_G=True,
                    weight_inv_modB=True,
                )
            )[1]

        def native_jacobian(at):
            setup.surface.set_dofs(at[:-2])
            return _outputs(
                boozer_surface_residual(
                    setup.surface, at[-2], at[-1], setup.bs, derivatives=1
                )
            )[1]

        hessian = host_array(
            boozer_penalty_constraints(
                problem,
                place_float64(x, problem.targetlabel),
                derivatives=2,
                optimize_G=True,
            )[2]
        )
        residual_hessians = host_array(
            jax_boozer_surface_residual(
                problem,
                place_float64(x, problem.targetlabel),
                derivatives=2,
                optimize_G=True,
            )[2]
        )
        for name, function, analytic in (
            ("penalty Hessian", native_gradient, hessian @ direction),
            ("residual Hessians", native_jacobian, residual_hessians @ direction),
        ):
            differences = (
                function(x + step * direction) - function(x - step * direction)
            ) / (2 * step)
            assert_matches_native(analytic, differences, name, rtol=1e-7)
            base = function(x)
            assert_errors_shrink(
                self,
                [
                    np.linalg.norm((function(x + h * direction) - base) / h - analytic)
                    for h in np.power(2.0, -np.arange(7, 20))
                ],
                0.55,
                msg=name,
                initial=1e9,
            )
        setup.surface.set_dofs(x[:-2])

    def test_fixed_surface_dofs_leave_the_derivatives_unchanged(self):
        """Derivatives are with respect to all surface DOFs, fixed ones included,
        as native ``dgamma_by_dcoeff`` and native's label gradient
        ``dlabel_dsurface``: native with the same DOFs fixed is the reference."""
        setup = _setup("tensor", True, "volume")
        names = list(setup.surface.local_full_dof_names)
        setup.surface.fix(names[0])
        setup.surface.fix(names[4])
        x = _decision(setup.surface, False)
        expected = _native_boozer(setup, 2.0).boozer_penalty_constraints_vectorized(
            x.copy(), derivatives=2, constraint_weight=2.0
        )
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 2.0
        )
        result = boozer_penalty_constraints(
            problem, place_float64(x, problem.targetlabel), derivatives=2
        )
        _assert_all_native(result, expected, "penalty with fixed DOFs")
        native_residual = boozer_surface_residual(
            setup.surface, _IOTA, None, setup.bs, derivatives=1
        )
        _assert_all_native(
            jax_boozer_surface_residual(
                problem,
                place_float64(x, problem.targetlabel),
                derivatives=1,
                optimize_G=False,
            ),
            native_residual,
            "residual with fixed DOFs",
        )

    def test_new_values_reuse_the_compiled_programs(self):
        """New surface DOFs, iota, G, multipliers, coil currents, target and weight
        reuse the programs; the old problem stays a snapshot."""
        setup = _setup("xyz", True, "toroidal_flux")
        first = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 3.0
        )
        x, xl = (
            _decision(setup.surface, True),
            _decision(setup.surface, True, (0.2, 0.1)),
        )
        first_results = _evaluate_every_formulation(first, x, xl)
        _assert_every_formulation_native(first_results, setup, 3.0, x, xl)
        first_x, first_xl = x, xl

        setup.surface.set_dofs(
            setup.surface.get_dofs()
            * (1 + 1e-3 * parity_rng(3).standard_normal(x.size - 2))
        )
        setup.currents[0].local_full_x = 1.1 * setup.currents[0].local_full_x
        setup.target *= 0.99
        x = _decision(setup.surface, True) * (
            1 + 1e-3 * parity_rng(4).standard_normal(x.size)
        )
        xl = np.concatenate((x, [-0.4, 0.3]))
        second = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 40.0
        )
        with jax_compilations() as compilations:
            second_results = _evaluate_every_formulation(second, x, xl)
            first_again = _evaluate_every_formulation(first, first_x, first_xl)
        self.assertEqual(
            compilations, [], "new values retraced or recompiled a formulation"
        )
        _assert_every_formulation_native(second_results, setup, 40.0, x, xl)
        self.assertFalse(
            np.allclose(
                host_array(first_results["penalty"][0]),
                host_array(second_results["penalty"][0]),
            )
        )
        # The first problem is a snapshot: the native changes leave it unchanged.
        for name, results in first_results.items():
            for order, (again, before) in enumerate(
                zip(first_again[name], results, strict=True)
            ):
                assert_matches_native(
                    again, host_array(before), f"first problem {name}[{order}]"
                )

    def test_formulations_make_no_implicit_transfers(self):
        """Boozer formulations accept placed inputs without implicit host/device transfers."""
        for case_id_0, parity_lane in zip(
            ("cpu_parity", "gpu_parity"), ("cpu", "gpu"), strict=True
        ):
            for case_id_1, stellsym in zip(
                ["stellsym", "nonsym"], [True, False], strict=True
            ):
                with self.subTest(
                    case_id_0=case_id_0,
                    parity_lane=parity_lane,
                    case_id_1=case_id_1,
                    stellsym=stellsym,
                ), self.case():
                    self._case_formulations_make_no_implicit_transfers(
                        parity_lane, stellsym
                    )

    def _case_formulations_make_no_implicit_transfers(
        self, parity_lane, stellsym
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        setup = _setup("tensor", stellsym, "toroidal_flux")
        x, xl = (
            _decision(setup.surface, True),
            _decision(setup.surface, True, (0.2, 0.1)),
        )
        for weight in (None, 4.0):
            with parity_default_device(parity_lane), disallow_host_transfers():
                problem = boozer_problem(
                    setup.field, setup.surface, setup.label, setup.target, weight
                )
                placed_x, placed_xl = (
                    place_float64(x, problem.targetlabel),
                    place_float64(xl, problem.targetlabel),
                )
                results = jax.block_until_ready(
                    [
                        jax_boozer_surface_residual(
                            problem, placed_x, derivatives=2, optimize_G=True
                        ),
                        boozer_exact_constraints(problem, placed_xl, derivatives=1),
                        boozer_exact_residual(
                            problem,
                            placed_x,
                            boozer_exact_residual_rows(setup.surface),
                            derivatives=1,
                        )
                        if weight is None
                        else boozer_penalty_constraints(
                            problem, placed_x, derivatives=2, optimize_G=True
                        ),
                    ]
                )
            native_boozer = _native_boozer(setup, weight)
            _assert_all_native(
                results[1],
                native_boozer.boozer_exact_constraints(xl.copy(), derivatives=1),
                "exact constraints",
            )
            if weight is not None:
                _assert_all_native(
                    results[2],
                    native_boozer.boozer_penalty_constraints_vectorized(
                        x.copy(),
                        derivatives=2,
                        constraint_weight=weight,
                        optimize_G=True,
                    ),
                    "penalty",
                )
            devices = {
                device for leaf in jax.tree.leaves(results) for device in leaf.devices()
            }
            self.assertEqual(devices, problem.targetlabel.devices())

    def test_zero_field_gives_the_native_values(self):
        """With every current zero, the weighted residual is 0/0 at every point:
        native and JAX return the same NaN pattern; unweighted values agree."""
        for weight_inv_modB in [False, True]:
            with self.subTest(weight_inv_modB=weight_inv_modB), self.case():
                self._case_zero_field_gives_the_native_values(weight_inv_modB)

    def _case_zero_field_gives_the_native_values(self, weight_inv_modB):
        """Run one row with case-local objects released before runtime cleanup."""
        setup = _setup("tensor", False, "volume")
        for current in setup.currents:
            current.local_full_x = np.zeros(1)
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 1.0
        )
        x = _decision(setup.surface, False)
        expected = _native_boozer(setup, 1.0).boozer_penalty_constraints_vectorized(
            x.copy(), derivatives=2, weight_inv_modB=weight_inv_modB
        )
        result = boozer_penalty_constraints(
            problem,
            place_float64(x, problem.targetlabel),
            derivatives=2,
            weight_inv_modB=weight_inv_modB,
        )
        for order, (actual, native) in enumerate(zip(result, expected)):
            actual, native = host_array(actual), np.asarray(native)
            np.testing.assert_array_equal(
                np.isnan(actual), np.isnan(native), err_msg=f"NaN pattern [{order}]"
            )
            self.assertEqual(np.all(np.isnan(native)), weight_inv_modB)
            finite = np.isfinite(native)
            assert_matches_native(
                actual[finite],
                native[finite],
                f"zero-field penalty[{order}]",
                rtol=_RTOL,
                scale=np.max(np.abs(native[finite]), initial=1.0),
            )

    def test_problem_boundary_refuses_unsupported_inputs(self):
        """The Boozer snapshot boundary rejects incompatible surfaces, fields and labels."""
        setup = _setup("xyz", True, "volume")
        with self.assertRaisesRegex(TypeError, "JaxBiotSavart"):
            boozer_problem(setup.bs, setup.surface, setup.label, setup.target)
        with self.assertRaisesRegex(TypeError, "labels"):
            boozer_problem(
                setup.field, setup.surface, PrincipalCurvature(setup.surface), 1.0
            )
        other = _surface("xyz", True, setup.surface.nfp)
        other.set_dofs(setup.surface.get_dofs())
        with self.assertRaisesRegex(ValueError, "sharing its DOFs"):
            boozer_problem(setup.field, setup.surface, Volume(other), 1.0)
        _, _, _, _, other_bs = get_data("ncsx")
        with self.assertRaisesRegex(ValueError, "field's coils"):
            boozer_problem(
                setup.field, setup.surface, ToroidalFlux(setup.surface, other_bs), 1.0
            )

        exact = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        self.assertIsNone(exact.constraint_weight)
        x = place_float64(_decision(setup.surface, False), exact.targetlabel)
        with self.assertRaisesRegex(ValueError, "constraint_weight"):
            boozer_penalty_constraints(exact, x)
        # As native solve_residual_equation_exactly_newton, only tensor surfaces have the exact system.
        with self.assertRaisesRegex(RuntimeError, "SurfaceXYZTensorFourier"):
            boozer_exact_residual_rows(setup.surface)
        tensor_setup = _setup("tensor", True, "volume")
        tensor_rows = boozer_exact_residual_rows(tensor_setup.surface)
        with self.assertRaisesRegex(RuntimeError, "SurfaceXYZTensorFourier"):
            boozer_exact_residual(
                exact,
                place_float64(_decision(setup.surface, True), exact.targetlabel),
                tensor_rows,
            )
        with self.assertRaisesRegex(ValueError, "decision vector"):
            boozer_penalty_constraints(
                replace(exact, constraint_weight=exact.targetlabel), x, optimize_G=True
            )
        # No order the field model does not carry, and only integer orders.
        weighted = replace(exact, constraint_weight=exact.targetlabel)
        tensor = boozer_problem(
            tensor_setup.field, tensor_setup.surface, tensor_setup.label, tensor_setup.target
        )
        tensor_x = place_float64(_decision(tensor_setup.surface, True), tensor.targetlabel)
        tensor_xl = place_float64(
            _decision(tensor_setup.surface, True, (0.1, 0.2)), tensor.targetlabel
        )
        unsupported = {
            "residual": [
                lambda order: jax_boozer_surface_residual(
                    exact, x, derivatives=order, optimize_G=False
                ),
                (-1, 3, 1.0, True),
            ],
            "penalty": [
                lambda order: boozer_penalty_constraints(weighted, x, derivatives=order),
                (-1, 3, 1.0, True),
            ],
            "exact constraints": [
                lambda order: boozer_exact_constraints(tensor, tensor_xl, derivatives=order),
                (-1, 2, 3, 0.0, False),
            ],
            "exact residual": [
                lambda order: boozer_exact_residual(
                    tensor, tensor_x, tensor_rows, derivatives=order
                ),
                (-1, 2, 3, 0.0, False),
            ],
        }
        for name, (formulation, orders) in unsupported.items():
            for order in orders:
                with self.subTest(name=name, derivatives=order), self.assertRaisesRegex(
                    ValueError, "derivatives must be"
                ):
                    formulation(order)

    def test_penalty_gradient_taylor(self):
        """The penalty gradient matches forward differences of the penalty at shrinking
        steps, as native ``test_boozer_penalty_constraints_gradient`` checks."""
        for name in _PENALTY_CASES:
            with self.subTest(name=name), self.case():
                self._case_penalty_gradient_taylor(name)

    def _case_penalty_gradient_taylor(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        case = _PENALTY_CASES[name]
        setup = _setup(case.surface, case.stellsym, case.label)
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 11.1232
        )
        x = _decision(setup.surface, case.optimize_G)
        direction = parity_rng(1).uniform(size=x.size) - 0.5

        def penalty(at, derivatives=0):
            return boozer_penalty_constraints(
                problem,
                place_float64(at, problem.targetlabel),
                derivatives=derivatives,
                optimize_G=case.optimize_G,
                weight_inv_modB=case.weight_inv_modB,
            )

        base, gradient = (host_array(output) for output in penalty(x, derivatives=1))
        analytic = gradient @ direction
        assert_errors_shrink(
            self,
            [
                abs((host_array(penalty(x + step * direction)) - base) / step - analytic)
                / abs(analytic)
                for step in np.power(2.0, -np.arange(7, 20))
            ],
            0.55,
            initial=1e9,
        )

    def test_exact_constraints_jacobian_taylor(self):
        """The Jacobian of the BoozerExact optimality conditions matches forward
        differences of ``res`` at shrinking steps, as native
        ``test_boozer_constrained_jacobian`` checks, with nonzero multipliers."""
        for name in _EXACT_CONSTRAINT_CASES:
            with self.subTest(name=name), self.case():
                self._case_exact_constraints_jacobian_taylor(name)

    def _case_exact_constraints_jacobian_taylor(self, name):
        """Run one row with case-local objects released before runtime cleanup."""
        case = _EXACT_CONSTRAINT_CASES[name]
        setup = _setup(case.surface, case.stellsym, case.label)
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        xl = _decision(setup.surface, case.optimize_G, (0.37, -0.23))
        direction = parity_rng(1).uniform(size=xl.size) - 0.5

        def constraints(at, derivatives=0):
            return boozer_exact_constraints(
                problem,
                place_float64(at, problem.targetlabel),
                derivatives=derivatives,
                optimize_G=case.optimize_G,
            )

        base, jacobian = (
            host_array(output) for output in constraints(xl, derivatives=1)
        )
        analytic = jacobian @ direction
        assert_errors_shrink(
            self,
            [
                np.linalg.norm(
                    (host_array(constraints(xl + step * direction)) - base) / step
                    - analytic
                )
                for step in np.power(2.0, -np.arange(7, 20))
            ],
            0.55,
            initial=1e9,
        )

    def test_coil_derivatives_through_the_taylor_model_match_the_field(self):
        """Coil derivatives of the residual Jacobian, the penalty gradient and
        Hessian and the BoozerExact Jacobian, which see the coils through the
        field's Taylor coefficients, equal those of the same derivatives taken
        through the field kernel itself."""
        setup = _setup("tensor", False, "toroidal_flux")
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 5.0
        )
        x = place_float64(_decision(setup.surface, False), problem.targetlabel)
        exact_x = place_float64(_decision(setup.surface, True), problem.targetlabel)
        rows = boozer_exact_residual_rows(setup.surface)

        def penalty(coils, point, derivatives):
            return boozer_penalty_constraints(
                replace(problem, coils=coils), point, derivatives=derivatives
            )

        def residual(coils, point, derivatives):
            return jax_boozer_surface_residual(
                replace(problem, coils=coils),
                point,
                derivatives=derivatives,
                optimize_G=False,
                weight_inv_modB=True,
            )

        def exact(coils, point, derivatives):
            return boozer_exact_residual(
                replace(problem, coils=coils), point, rows, derivatives=derivatives
            )

        # (Taylor model, through the field kernel) as functions of the coils.
        pairs = {
            "residual Jacobian": (
                lambda coils: residual(coils, x, 1)[1],
                lambda coils: jax.jacfwd(lambda p: residual(coils, p, 0)[0])(x),
            ),
            "penalty gradient": (
                lambda coils: penalty(coils, x, 1)[1],
                lambda coils: jax.grad(lambda p: penalty(coils, p, 0))(x),
            ),
            "penalty Hessian": (
                lambda coils: penalty(coils, x, 2)[2],
                lambda coils: jax.hessian(lambda p: penalty(coils, p, 0))(x),
            ),
            "BoozerExact Jacobian": (
                lambda coils: exact(coils, exact_x, 1)[1],
                lambda coils: jax.jacfwd(lambda p: exact(coils, p, 0))(exact_x),
            ),
        }
        rng = parity_rng(5)
        for name, (taylor, through_field) in pairs.items():
            with self.subTest(name=name):
                value, pullback = jax.vjp(taylor, problem.coils)
                expected_value, expected_pullback = jax.vjp(through_field, problem.coils)
                assert_matches_native(value, host_array(expected_value), name)
                cotangent = place_float64(
                    rng.standard_normal(value.shape), problem.targetlabel
                )
                _assert_same_coil_pullbacks(self, (pullback, expected_pullback), cotangent, name)

    def test_exact_constraint_values_refuse_derivatives_in_xl(self):
        """``boozer_exact_constraints(derivatives=0)`` is a gradient built with the
        field's first-order model, so its derivative in ``xl`` would miss
        ``d2B/dXdX``: forward, reverse and total derivatives through an ``xl``
        that depends on the coils raise and point to ``derivatives=1``. The
        guard that raises compiles to no operation."""
        setup = _setup("tensor", False, "toroidal_flux")
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        xl = place_float64(
            _decision(setup.surface, True, (0.37, -0.23)), problem.targetlabel
        )

        def constraints(point):
            return boozer_exact_constraints(problem, point)

        def through_the_coils(scale):
            coils = jax.tree.map(lambda leaf: scale * leaf, problem.coils)
            return boozer_exact_constraints(replace(problem, coils=coils), scale * xl)

        one = place_float64(1.0, problem.targetlabel)
        for name, derivative in (
            ("jacfwd in xl", lambda: jax.jacfwd(constraints)(xl)),
            ("jacrev in xl", lambda: jax.jacrev(constraints)(xl)),
            ("total derivative", lambda: jax.jacfwd(through_the_coils)(one)),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(TypeError, "derivatives=1"):
                derivative()

        # The function itself, traced afresh on every call: patching the guard
        # then changes what is traced without touching any compiled program.
        traced = boozer_exact_constraints.__wrapped__

        def operations():
            text = jit(lambda p, point: traced(p, point)).lower(problem, xl).as_text()
            return Counter(re.findall(r"\b(?:stablehlo|chlo|func)\.[a-z_]+", text))

        guarded = operations()
        with mock.patch.object(
            boozer_problem_module, "_not_differentiable_in_x", lambda value, refusal: value
        ):
            unguarded = operations()
            unguarded_jacobian = host_array(jax.jacfwd(partial(traced, problem))(xl))
        self.assertGreater(sum(guarded.values()), 0)
        self.assertEqual(guarded, unguarded, "the guard adds operations")
        # What the guard refuses is wrong: without it, the derivative in xl
        # misses d2B/dXdX and differs from the Jacobian of derivatives=1.
        jacobian = host_array(boozer_exact_constraints(problem, xl, derivatives=1)[1])
        self.assertGreater(
            np.max(np.abs(unguarded_jacobian - jacobian)), 1e-6 * np.max(np.abs(jacobian))
        )

    def test_derivative_outputs_refuse_derivatives_in_x(self):
        """Outputs requested with ``derivatives >= 1`` use the field's Taylor
        model of that order, so their derivative in ``x`` would miss the
        field's next derivative: every formulation raises instead."""
        setup = _setup("tensor", False, "toroidal_flux")
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 5.0
        )
        x = place_float64(_decision(setup.surface, True), problem.targetlabel)
        xl = place_float64(
            _decision(setup.surface, True, (0.37, -0.23)), problem.targetlabel
        )
        rows = boozer_exact_residual_rows(setup.surface)
        derivative_outputs = {
            "residual": (
                partial(jax_boozer_surface_residual, problem, derivatives=1, optimize_G=True),
                x,
            ),
            "penalty": (
                partial(boozer_penalty_constraints, problem, derivatives=1, optimize_G=True),
                x,
            ),
            "penalty residual": (
                partial(boozer_penalty_residual, problem, derivatives=1, optimize_G=True),
                x,
            ),
            "exact constraints": (
                partial(boozer_exact_constraints, problem, derivatives=1),
                xl,
            ),
            "exact residual": (
                partial(boozer_exact_residual, problem, residual_rows=rows, derivatives=1),
                x,
            ),
        }
        for name, (outputs, point) in derivative_outputs.items():
            with self.subTest(name=name), self.assertRaisesRegex(
                TypeError, r"\(derivatives=1\) is not differentiable in x"
            ):
                jax.jacfwd(outputs)(point)

    def test_exact_constraint_values_have_exact_coil_derivatives(self):
        """At a fixed ``xl``, the coil derivatives of
        ``boozer_exact_constraints(derivatives=0)`` (first and second order
        forward along one coil direction, jitted and vmapped over two steps,
        and a first-order reverse pullback) equal those of the same Lagrangian
        gradient taken through the field kernel itself, built here from the
        unweighted residual and constraints of :func:`boozer_exact_residual` at
        every residual row."""
        setup = _setup("tensor", False, "toroidal_flux")
        problem = boozer_problem(setup.field, setup.surface, setup.label, setup.target)
        xl = place_float64(
            _decision(setup.surface, True, (0.37, -0.23)), problem.targetlabel
        )
        nx = xl.shape[0] - 2
        npoints = setup.surface.gamma().size // 3
        every_row = jnp.arange(3 * npoints, dtype=jnp.int32)

        def reference(coils, point):
            moved = replace(problem, coils=coils)

            def lagrangian(x, multipliers):
                system = boozer_exact_residual(moved, x, every_row)
                residual, constraints = system[:-2], system[-2:]
                return 0.5 * jnp.sum(residual * residual) - multipliers @ constraints

            x, multipliers = point[:nx], point[nx:]
            stationarity = jax.grad(lagrangian)(x, multipliers)
            constraints = boozer_exact_residual(moved, x, every_row)[-2:]
            return jnp.concatenate((stationarity, constraints))

        def exact(coils, point):
            return boozer_exact_constraints(replace(problem, coils=coils), point)

        rng = parity_rng(7)
        direction = jax.tree.map(
            lambda leaf: place_float64(
                1e-2 * host_array(leaf) * rng.standard_normal(leaf.shape),
                problem.targetlabel,
            ),
            problem.coils,
        )

        def along(constraints):
            """Value, first and second derivatives along ``direction`` at ``t``."""

            def value(t):
                coils = jax.tree.map(lambda c, d: c + t * d, problem.coils, direction)
                return constraints(coils, xl)

            def first(t):
                return jax.jvp(value, (t,), (jnp.ones_like(t),))[1]

            def second(t):
                return jax.jvp(first, (t,), (jnp.ones_like(t),))[1]

            return lambda t: (value(t), first(t), second(t))

        steps = place_float64([0.0, 0.5], problem.targetlabel)
        actual = jax.vmap(jit(along(exact)))(steps)
        expected = jax.vmap(along(reference))(steps)
        for order, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert_matches_native(a, host_array(e), f"coil derivative of order {order}")

        cotangent = place_float64(rng.standard_normal(xl.shape), problem.targetlabel)

        pullbacks = [jax.vjp(partial(c, point=xl), problem.coils)[1] for c in (exact, reference)]
        _assert_same_coil_pullbacks(self, pullbacks, cotangent, "exact constraints")

    def test_formulations_cast_the_decision_vector_to_float64(self):
        """A float32 decision vector is evaluated in float64: every formulation
        and derivative, the residual objective with its ``dJ/dx``, and each
        solver's first iterate equal their values at the same numbers in
        float64."""
        setup = _setup("tensor", False, "volume")
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 3.0
        )
        rows = boozer_exact_residual_rows(setup.surface)
        # One step of each solver: tol 0 never stops it before maxiter 1.
        tol, one = jnp.asarray(0.0), jnp.asarray(1.0)
        options = {"optimize_G": True, "weight_inv_modB": True}
        formulations = {
            "residual": lambda x, xl: jax_boozer_surface_residual(
                problem, x, derivatives=2, optimize_G=True
            ),
            "penalty": lambda x, xl: boozer_penalty_constraints(
                problem, x, derivatives=2, optimize_G=True
            ),
            "penalty residual": lambda x, xl: boozer_penalty_residual(
                problem, x, derivatives=2, optimize_G=True
            ),
            "exact constraints": lambda x, xl: boozer_exact_constraints(
                problem, xl, derivatives=1
            ),
            "exact residual": lambda x, xl: boozer_exact_residual(
                problem, x, rows, derivatives=1
            ),
            "residual objective": lambda x, xl: boozer_residual_objective(
                problem, x, **options
            )[:2],
            "exact Newton": lambda x, xl: (
                boozer_exact_newton(problem, x, rows, tol, one).x,
            ),
            "penalty Newton": lambda x, xl: (
                boozer_penalty_newton(problem, x, tol, one, tol, **options).x,
            ),
            "Gauss-Newton": lambda x, xl: (
                boozer_penalty_gauss_newton(problem, x, tol, one, **options).x,
            ),
            "BFGS": lambda x, xl: (
                boozer_penalty_bfgs(problem, x, tol, one, **options).x,
            ),
            "L-BFGS-B": lambda x, xl: (
                boozer_penalty_lbfgsb(
                    problem, x, tol, tol, one, maxcor=3, **options
                ).x,
            ),
        }
        decision = _decision(setup.surface, True, (0.37, -0.23))
        rounded = place_float64(decision.astype(np.float32), problem.targetlabel)
        single = rounded.astype(jnp.float32)
        for name, formulation in formulations.items():
            with self.subTest(name=name):
                actual = formulation(single[:-2], single)
                expected = formulation(rounded[:-2], rounded)
                for order, (a, e) in enumerate(zip(actual, expected, strict=True)):
                    self.assertEqual(a.dtype, jnp.float64, f"{name}[{order}] dtype")
                    assert_matches_native(a, host_array(e), f"{name}[{order}]", rtol=1e-14)

    def test_second_derivatives_nest_forward_over_reverse(self):
        """The penalty Hessian and the BoozerExact KKT Jacobian compile, at
        1296 points and 435 unknowns, to temporaries below one
        ``(npoints, nx, nx)`` float64 array: forward over reverse needs a few
        ``(npoints, nx)`` tangents, while reverse over forward keeps every
        forward tangent for its reverse sweep (about 40 such arrays, 75 GiB).
        The backend's point chunks bound the field kernels' own temporaries."""
        set_backend("jax", device="cpu")
        _, _, axis, nfp, bs = get_data("ncsx")
        nphi = ntheta = 36
        surface = SurfaceXYZTensorFourier(
            mpol=8,
            ntor=8,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1 / nfp, nphi, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False),
        )
        surface.fit_to_curve(axis, 0.1, flip_theta=True)
        volume = Volume(surface)
        problem = boozer_problem(
            JaxBiotSavart(bs.coils), surface, volume, volume.J(), 1.0
        )
        x = place_float64(_decision(surface, True), problem.targetlabel)
        xl = place_float64(_decision(surface, True, (0.3, -0.2)), problem.targetlabel)
        nx = x.shape[0]
        bound = nphi * ntheta * nx * nx * 8
        compiled = {
            "penalty Hessian": boozer_penalty_constraints.lower(
                problem, x, derivatives=2, optimize_G=True
            ).compile(),
            "KKT Jacobian": boozer_exact_constraints.lower(
                problem, xl, derivatives=1
            ).compile(),
        }
        for name, program in compiled.items():
            with self.subTest(name=name):
                memory = program.memory_analysis()
                assert memory is not None, f"{name}: no memory analysis"
                temporaries = memory.temp_size_in_bytes
                self.assertTrue(
                    temporaries < bound,
                    f"{name}: {temporaries} bytes of temporaries >= {bound}",
                )
