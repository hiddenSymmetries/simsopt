"""JAX Boozer residual formulations against native ``BoozerSurface``.

``simsopt_jax.core.boozer_problem`` is compared with native
``boozer_surface_residual``, ``BoozerSurface.boozer_penalty_constraints_vectorized``,
``BoozerSurface.boozer_exact_constraints`` and the system of
``BoozerSurface.solve_residual_equation_exactly_newton`` (values, first and second
derivatives) for ``SurfaceRZFourier``, ``SurfaceXYZFourier`` and
``SurfaceXYZTensorFourier``, with and without stellarator symmetry, the
``Volume``, ``Area``, ``AspectRatio`` and ``ToroidalFlux`` labels (on their own
grids), ``weight_inv_modB`` and ``optimize_G`` on and off. The analytic kernels
are also compared with ``simsoptpp.boozer_residual_ds2`` directly, Hessians
with central differences of native gradients. Then the problem boundary: fixed
DOFs, grids without a stellarator-symmetry mask, no coils, singular
cross-sections for AspectRatio, recompilation, implicit transfers, zero fields
and unsupported inputs.
"""

from __future__ import annotations

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import simsoptpp as sopp
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
    from simsopt_jax.backend.dtypes import explicit_device_array
    from simsopt_jax.core.boozer_problem import (
        BoozerProblem,
        boozer_exact_constraints,
        boozer_exact_residual,
        boozer_penalty_constraints,
    )
    from simsopt_jax.core.boozer_problem import (
        boozer_surface_residual as jax_boozer_surface_residual,
    )
    from simsopt_jax.core.boozer_residual import BoozerPoints, boozer_least_squares
    from simsopt_jax.runtime.host_boundary import disallow_host_transfers
    from simsopt_jax_adapters.field import JaxBiotSavart
    from simsopt_jax_adapters.geo.boozer_problem import (
        boozer_exact_residual_rows,
        boozer_problem,
    )

from unittest_jax_support import (
    assert_matches_native,
    host_array,
    jax_compilations,
    parity_default_device,
    parity_rng,
    place_float64,
)

from dataclasses import dataclass, replace

import numpy as np


if JAX_IMPORT_ERROR is None:
    _NativeSurface = SurfaceRZFourier | SurfaceXYZFourier | SurfaceXYZTensorFourier
if JAX_IMPORT_ERROR is None:
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
    kind: str, stellsym: bool, nfp: int, clamped: bool = False
) -> _NativeSurface:
    if kind == "tensor":
        # The grid of SurfaceXYZTensorFourier.get_stellsym_mask(), so BoozerExact works too.
        return SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            stellsym=stellsym,
            nfp=nfp,
            quadpoints_phi=np.linspace(0, 1 / nfp, 5, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 5, endpoint=False),
            clamped_dims=(clamped, False, clamped),
        )
    return _SURFACE_CLASSES[kind](
        mpol=2,
        ntor=2,
        stellsym=stellsym,
        nfp=nfp,
        quadpoints_phi=np.linspace(0, 1 / (2 * nfp), 4, endpoint=False),
        quadpoints_theta=np.linspace(0, 1, 6, endpoint=False),
    )


def _setup(
    kind: str, stellsym: bool, label: str, clamped: bool = False, seed: int = 0
) -> _Setup:
    _, currents, axis, nfp, bs = get_data("ncsx")
    surface = _surface(kind, stellsym, nfp, clamped)
    surface.fit_to_curve(axis, 0.1, flip_theta=True)
    dofs = surface.get_dofs()
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


# --- the analytic kernels against simsoptpp -----------------------------------------


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
    def test_least_squares_kernel_matches_simsoptpp(self):
        """Boozer least-squares values and derivatives agree with simsoptpp for both G and weighting modes."""
        for optimize_G in [False, True]:
            for weight_inv_modB in [False, True]:
                with self.subTest(
                    optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
                ), self.case() as patches:
                    self._case_least_squares_kernel_matches_simsoptpp(
                        optimize_G, weight_inv_modB, patches
                    )

    def _case_least_squares_kernel_matches_simsoptpp(
        self, optimize_G, weight_inv_modB, patches
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        rng = parity_rng(1)
        nphi, ntheta, nsurface = 3, 4, 5
        shape = (nphi, ntheta)
        d2B = rng.normal(scale=0.13, size=shape + (3, 3, 3))
        native = {
            "B": rng.normal(size=shape + (3,)) + np.array([0.3, -0.2, 1.4]),
            "dB_by_dX": rng.normal(scale=0.2, size=shape + (3, 3)),
            "d2B_by_dXdX": 0.5 * (d2B + np.swapaxes(d2B, 2, 3)),
            "xphi": rng.normal(size=shape + (3,)),
            "xtheta": rng.normal(size=shape + (3,)),
            "dgamma_by_dcoeff": rng.normal(size=shape + (3, nsurface)),
            "dgammadash1_by_dcoeff": rng.normal(size=shape + (3, nsurface)),
            "dgammadash2_by_dcoeff": rng.normal(size=shape + (3, nsurface)),
        }
        points = BoozerPoints(
            **{
                name: explicit_device_array(
                    value.reshape((nphi * ntheta,) + value.shape[2:]), dtype=np.float64
                )
                for name, value in native.items()
            }
        )
        coefficient_derivatives = (
            native["dgamma_by_dcoeff"],
            native["dgammadash1_by_dcoeff"],
            native["dgammadash2_by_dcoeff"],
        )
        tangents = (native["xphi"], native["xtheta"])
        value, gradient = sopp.boozer_residual_ds(
            _G,
            _IOTA,
            native["B"],
            native["dB_by_dX"],
            *tangents,
            *coefficient_derivatives,
            weight_inv_modB,
        )
        second = sopp.boozer_residual_ds2(
            _G,
            _IOTA,
            native["B"],
            native["dB_by_dX"],
            native["d2B_by_dXdX"],
            *tangents,
            *coefficient_derivatives,
            weight_inv_modB,
        )
        kept = slice(None) if optimize_G else slice(0, nsurface + 1)
        expected = (
            sopp.boozer_residual(_G, _IOTA, *tangents, native["B"], weight_inv_modB),
            (value, gradient[kept]),
            (second[0], second[1][kept], second[2][kept, kept]),
        )
        for derivatives in (0, 1, 2):
            result = jax.jit(
                lambda p, d=derivatives: boozer_least_squares(
                    _G,
                    _IOTA,
                    p,
                    derivatives=d,
                    optimize_G=optimize_G,
                    weight_inv_modB=weight_inv_modB,
                )
            )(points)
            _assert_all_native(
                result, expected[derivatives], f"boozer_residual_ds{derivatives}"
            )

    def test_surface_residual_matches_native(self):
        """Boozer residuals, Jacobians and Hessians agree with native formulations."""
        for name in _RESIDUAL_CASES:
            with self.subTest(name=name), self.case() as patches:
                self._case_surface_residual_matches_native(name, patches)

    def _case_surface_residual_matches_native(self, name, patches):
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
            with self.subTest(name=name), self.case() as patches:
                self._case_penalty_constraints_match_native(name, patches)

    def _case_penalty_constraints_match_native(self, name, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        case = _PENALTY_CASES[name]
        setup = _setup(case.surface, case.stellsym, case.label)
        native_boozer = _native_boozer(setup, 1.0)
        x = _decision(setup.surface, case.optimize_G)
        results, expected = {}, {}
        for weight in (0.7, 700.0):
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
            _outputs(results[700.0, 2]),
            _outputs(results[0.7, 2]),
            _outputs(expected[700.0, 2]),
            _outputs(expected[0.7, 2]),
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
            with self.subTest(name=name), self.case() as patches:
                self._case_exact_constraints_match_native(name, patches)

    def _case_exact_constraints_match_native(self, name, patches):
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
                ), self.case() as patches:
                    self._case_exact_residual_matches_native(label, stellsym, patches)

    def _case_exact_residual_matches_native(self, label, stellsym, patches):
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
        previous_error = previous_label_error = 1e9
        for step in np.power(2.0, -np.arange(7, 20)):
            difference = (system(x + step * direction) - base) / step
            error = np.linalg.norm(difference - analytic)
            label_error = abs(difference[label_row] - analytic[label_row])
            self.assertLess(error, 0.55 * previous_error)
            self.assertLess(label_error, 0.55 * previous_label_error)
            previous_error, previous_label_error = error, label_error
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
        with self.assertRaises(np.linalg.LinAlgError):
            _native_boozer(
                replace(setup, surface=surface, label=label, target=5.0), 1.0
            ).boozer_penalty_constraints_vectorized(x.copy(), derivatives=2)
        for output in boozer_penalty_constraints(
            problem, place_float64(x, problem.targetlabel), derivatives=2
        ):
            self.assertTrue(
                np.all(np.isfinite(host_array(output))),
                "AspectRatio penalty is not finite",
            )

    def test_hessians_match_central_differences_of_native_derivatives(self):
        """Penalty and residual Hessians agree with central differences of native derivatives."""
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
        differences = (
            native_gradient(x + step * direction)
            - native_gradient(x - step * direction)
        ) / (2 * step)
        assert_matches_native(
            hessian @ direction, differences, "penalty Hessian", rtol=1e-7
        )

        residual_hessians = host_array(
            jax_boozer_surface_residual(
                problem,
                place_float64(x, problem.targetlabel),
                derivatives=2,
                optimize_G=True,
            )[2]
        )
        differences = (
            native_jacobian(x + step * direction)
            - native_jacobian(x - step * direction)
        ) / (2 * step)
        assert_matches_native(
            residual_hessians @ direction, differences, "residual Hessians", rtol=1e-7
        )

    def test_fixed_surface_dofs_leave_the_derivatives_unchanged(self):
        """Derivatives are with respect to all surface DOFs, fixed ones included,
        as native ``dgamma_by_dcoeff``. Native ``BoozerSurface`` at 9e027eac3 cannot
        evaluate a surface with fixed DOFs (its label gradient has the free DOFs
        only and does not fit), so the reference is native with every DOF free."""
        setup = _setup("tensor", True, "volume")
        x = _decision(setup.surface, False)
        expected = _native_boozer(setup, 2.0).boozer_penalty_constraints_vectorized(
            x.copy(), derivatives=2, constraint_weight=2.0
        )
        names = list(setup.surface.local_full_dof_names)
        setup.surface.fix(names[0])
        setup.surface.fix(names[4])
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
        self.assertTrue(
            compilations == [], "new values retraced or recompiled a formulation"
        )
        _assert_every_formulation_native(second_results, setup, 40.0, x, xl)
        self.assertTrue(
            not np.allclose(
                host_array(first_results["penalty"][0]),
                host_array(second_results["penalty"][0]),
            ),
            'not np.allclose(host_array(first_results["penalty"][0]), host_array(second_results["penalty"][0]))',
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
                ), self.case() as patches:
                    self._case_formulations_make_no_implicit_transfers(
                        parity_lane, stellsym, patches
                    )

    def _case_formulations_make_no_implicit_transfers(
        self, parity_lane, stellsym, patches
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
            self.assertTrue(
                devices == problem.targetlabel.devices(),
                "devices == problem.targetlabel.devices()",
            )

    def test_zero_field_gives_the_native_values(self):
        """With every current zero, the weighted residual is 0/0 at every point:
        native and JAX return the same NaN pattern; unweighted values agree."""
        for weight_inv_modB in [False, True]:
            with self.subTest(weight_inv_modB=weight_inv_modB), self.case() as patches:
                self._case_zero_field_gives_the_native_values(weight_inv_modB, patches)

    def _case_zero_field_gives_the_native_values(self, weight_inv_modB, patches):
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
            self.assertTrue(
                np.all(np.isnan(native)) == weight_inv_modB,
                "np.all(np.isnan(native)) == weight_inv_modB",
            )
            finite = np.isfinite(native)
            np.testing.assert_allclose(
                actual[finite],
                native[finite],
                rtol=0.0,
                atol=_RTOL * np.max(np.abs(native[finite]), initial=1.0),
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
        self.assertTrue(
            exact.constraint_weight is None, "exact.constraint_weight is None"
        )
        x = place_float64(_decision(setup.surface, False), exact.targetlabel)
        with self.assertRaisesRegex(ValueError, "constraint_weight"):
            boozer_penalty_constraints(exact, x)
        # As native solve_residual_equation_exactly_newton, only tensor surfaces have the exact system.
        with self.assertRaisesRegex(RuntimeError, "SurfaceXYZTensorFourier"):
            boozer_exact_residual_rows(setup.surface)
        tensor_rows = boozer_exact_residual_rows(
            _setup("tensor", True, "volume").surface
        )
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

    def test_hessians_taylor(self):
        """Boozer Hessians converge from native gradients and Jacobians at shrinking steps."""
        setup = _setup("xyz", False, "toroidal_flux")
        native = _native_boozer(setup, 5.0)
        problem = boozer_problem(
            setup.field, setup.surface, setup.label, setup.target, 5.0
        )
        x = _decision(setup.surface, True)
        direction = parity_rng(2).standard_normal(x.size)
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

        def gradient(at):
            return native.boozer_penalty_constraints_vectorized(
                at.copy(),
                derivatives=1,
                constraint_weight=5.0,
                optimize_G=True,
                weight_inv_modB=True,
            )[1]

        def jacobian(at):
            setup.surface.set_dofs(at[:-2])
            return _outputs(
                boozer_surface_residual(
                    setup.surface, at[-2], at[-1], setup.bs, derivatives=1
                )
            )[1]

        for function, analytic in (
            (gradient, hessian @ direction),
            (jacobian, residual_hessians @ direction),
        ):
            base = function(x)
            previous_error = 1e9
            for step in np.power(2.0, -np.arange(7, 20)):
                error = np.linalg.norm(
                    (function(x + step * direction) - base) / step - analytic
                )
                self.assertLess(error, 0.55 * previous_error)
                previous_error = error
        setup.surface.set_dofs(x[:-2])
