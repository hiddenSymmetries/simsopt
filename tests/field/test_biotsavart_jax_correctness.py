"""Native regression oracles for the JaxBiotSavart correctness review."""

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    from simsopt_jax.core import curve_gamma_and_gammadash_from_spec
    from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart
    from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve
from tempfile import TemporaryDirectory
from pathlib import Path

from typing import cast

import numpy as np

from simsopt import load
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import CurveXYZFourier, create_equally_spaced_curves
from simsopt.geo.curveperturbed import (
    CurvePerturbed,
    GaussianSampler,
    PerturbationSample,
)
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import (
    FrameRotation,
    FramedCurveCentroid,
    FramedCurveFrenet,
)


_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])
_FIELDS = ("B", "A", "dB_by_dX", "dA_by_dX")
_COMPOSED_WRAPPERS = (
    "shared_filament",
    "centroid_filament",
    "nested_perturbed",
    "perturbed_filament",
    "perturbed_shared_filament",
)


def _curves():
    return create_equally_spaced_curves(
        2, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32
    )


def _vjp(field, quantity, cotangent) -> Derivative:
    # Warm native caches, including the unit-current A cache.
    getattr(field, quantity)()
    if quantity == "dB_by_dX":
        return field.B_and_dB_vjp(np.zeros_like(_POINTS), cotangent)[1]
    if quantity == "dA_by_dX":
        return field.A_and_dA_vjp(np.zeros_like(_POINTS), cotangent)[1]
    return getattr(field, quantity + "_vjp")(cotangent)


def _gradient(derivative: Derivative, owner: Optimizable) -> np.ndarray:
    return cast(np.ndarray, derivative(owner))


def _cotangent(quantity):
    shape = (2, 3, 3) if "by_dX" in quantity else (2, 3)
    return np.linspace(-0.7, 0.9, np.prod(shape)).reshape(shape)


def _composed_native_field(wrapper, fixed):
    """Build the shared native oracle for composed-curve finite differences and Taylor tests."""
    curve = create_equally_spaced_curves(
        1,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.25,
        order=1,
        numquadpoints=24,
    )[0]
    current = Current(1e5)
    shared = "shared" in wrapper
    rotation = FrameRotation(
        curve.quadpoints,
        order=4 if shared else 1,
        dofs=curve.dofs if shared else None,
    )
    if not shared:
        rotation.local_full_x = np.array([0.2, -0.1, 0.05])
    if fixed:
        curve.fix(0)
        rotation.fix(rotation.local_dof_names[1])
        current.fix_all()
    sampler = GaussianSampler(curve.quadpoints, 1e-3, 0.2, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=np.random.default_rng(20261005))
    if wrapper == "nested_perturbed":
        wrapped = CurvePerturbed(CurvePerturbed(curve, sample), sample)
    else:
        frame = FramedCurveCentroid if "centroid" in wrapper else FramedCurveFrenet
        wrapped = CurveFilament(frame(curve, rotation), 0.01, 0.02)
        if wrapper.startswith("perturbed"):
            wrapped = CurvePerturbed(wrapped, sample)
    coils = [Coil(wrapped, current)]
    native = BiotSavart(coils).set_points(_POINTS)
    return native, curve, rotation


def _filament(frame, quadpoints):
    """Build a non-circular rotated filament on ``frame`` sampled at ``quadpoints``."""
    reference = create_equally_spaced_curves(
        1, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=24
    )[0]
    curve = CurveXYZFourier(quadpoints, 2)
    curve.x = reference.x + np.random.default_rng(20261011).normal(
        0.0, 0.03, reference.x.size
    )
    rotation = FrameRotation(quadpoints, 1)
    rotation.x = np.array([0.2, -0.1, 0.05])
    return CurveFilament(frame(curve, rotation), 0.05, 0.08)


def _jax_filament_geometry(frame, quadpoints):
    """Evaluate JAX filament ``gamma`` and ``gammadash`` at ``quadpoints``."""
    spec = curve_spec_from_adapter_curve(_filament(frame, quadpoints))
    gamma, gammadash = curve_gamma_and_gammadash_from_spec(spec)
    return np.asarray(gamma), np.asarray(gammadash)


class TestBiotsavartJaxCorrectness(JaxTestCase):
    def test_filament_gammadash_is_quadpoint_derivative(self):
        """JAX filament gammadash, built from the frame derivative kernels, is the
        quadpoint derivative of gamma and matches native for both frames."""
        for frame in [FramedCurveFrenet, FramedCurveCentroid]:
            with self.subTest(frame=frame.__name__), self.case():
                self._case_filament_gammadash_is_quadpoint_derivative(frame)

    def _case_filament_gammadash_is_quadpoint_derivative(self, frame):
        """Run one row with case-local objects released before runtime cleanup."""
        quadpoints = np.linspace(0.0, 1.0, 24, endpoint=False)
        gamma, gammadash = _jax_filament_geometry(frame, quadpoints)
        error = 1e6
        for exponent in range(5):
            step = 1e-2 * 0.5**exponent
            plus, _ = _jax_filament_geometry(frame, quadpoints + step)
            minus, _ = _jax_filament_geometry(frame, quadpoints - step)
            next_error = np.linalg.norm((plus - minus) / (2 * step) - gammadash)
            print(f"R10 {frame.__name__} step={step:.3e}: error={next_error:.3e}")
            self.assertLess(next_error, 0.3 * error)
            error = next_error
        native = _filament(frame, quadpoints)
        np.testing.assert_allclose(gamma, native.gamma(), rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            gammadash, native.gammadash(), rtol=1e-12, atol=1e-14
        )

    def test_affine_current_vjps_match_native_and_finite_difference(self):
        """Affine-current fields, owner partials and flat gradients agree with native and
        finite-difference references."""
        for scale in [1.0, -2.5]:
            for quantity in _FIELDS:
                with self.subTest(scale=scale, quantity=quantity), self.case():
                    self._case_affine_current_vjps_match_native_and_finite_difference(
                        scale, quantity
                    )

    def _case_affine_current_vjps_match_native_and_finite_difference(
        self, scale, quantity
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = _curves()
        for curve in curves:
            curve.fix_all()
        first, total = Current(3.0), Current(10.0)
        total.fix_all()
        coils = [
            Coil(curves[0], first),
            Coil(curves[1], scale * (total - first)),
        ]
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        cotangent = _cotangent(quantity)
        np.testing.assert_allclose(
            getattr(adapter, quantity)(),
            getattr(native, quantity)(),
            rtol=1e-12,
            atol=1e-14,
        )
        expected = _gradient(_vjp(native, quantity, cotangent), native)
        actual = _gradient(_vjp(adapter, quantity, cotangent), adapter)
        value = cast(np.ndarray, first.local_full_x).copy()
        step = 1e-3
        first.local_full_x = value + step
        plus = np.sum(getattr(native, quantity)() * cotangent)
        first.local_full_x = value - step
        minus = np.sum(getattr(native, quantity)() * cotangent)
        first.local_full_x = value
        finite_difference = (plus - minus) / (2 * step)
        pullback = {
            "B": adapter.B_pullback_native,
            "A": adapter.A_pullback_native,
            "dB_by_dX": adapter.dB_by_dX_pullback_native,
            "dA_by_dX": adapter.dA_by_dX_pullback_native,
        }[quantity](cotangent)
        flat = adapter.coil_cotangents_to_dofs_gradient(
            pullback.d_coil_arrays, pullback.coil_indices
        )
        print(
            f"R9 {quantity} scale={scale}: adapter={actual[0]:.16g} flat={float(flat[0]):.16g} native={expected[0]:.16g} fd={finite_difference:.16g}"
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(flat, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(actual, finite_difference, rtol=1e-9, atol=1e-14)
        native_partials = cast(
            Derivative,
            _vjp(native, quantity, cotangent)(native, as_derivative=True),
        )
        adapter_partials = cast(
            Derivative,
            _vjp(adapter, quantity, cotangent)(adapter, as_derivative=True),
        )
        for owner in native_partials.data:
            np.testing.assert_allclose(
                adapter_partials.data[owner],
                native_partials.data[owner],
                rtol=1e-11,
                atol=1e-13,
            )

    def test_derivative_retains_fixed_partials(self):
        """Field VJPs retain native partials for fixed owners while exposing the correct
        free gradient."""
        for fixed in ["partial", "curve", "all"]:
            for quantity in _FIELDS:
                with self.subTest(fixed=fixed, quantity=quantity), self.case():
                    self._case_derivative_retains_fixed_partials(fixed, quantity)

    def _case_derivative_retains_fixed_partials(self, fixed, quantity):
        """Run one row with case-local objects released before runtime cleanup."""
        curve = _curves()[0]
        current = Current(1e5)
        curve.fix(0)
        current.fix_all()
        if fixed != "partial":
            curve.fix_all()
        other = _curves()[1]
        if fixed == "all":
            other.fix_all()
        coils = [Coil(curve, current), Coil(other, Current(2e4))]
        if fixed == "all":
            coils[1].current.fix_all()
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        expected = cast(
            Derivative,
            _vjp(native, quantity, _cotangent(quantity))(native, as_derivative=True),
        )
        derivative = _vjp(adapter, quantity, _cotangent(quantity))
        actual = cast(Derivative, derivative(adapter, as_derivative=True))
        print(
            f"R5 {quantity} {fixed}: current={cast(np.ndarray, actual.data[current])[0]:.16g}/{cast(np.ndarray, expected.data[current])[0]:.16g} curve[0]={cast(np.ndarray, actual.data[curve])[0]:.16g}/{cast(np.ndarray, expected.data[curve])[0]:.16g}"
        )
        for owner in expected.data:
            np.testing.assert_allclose(
                actual.data[owner],
                expected.data[owner],
                rtol=1e-11,
                atol=1e-13,
            )
        np.testing.assert_allclose(
            _gradient(derivative, adapter),
            _gradient(_vjp(native, quantity, _cotangent(quantity)), native),
            rtol=1e-11,
            atol=1e-13,
        )

    def test_resampling_updates_captured_geometry_and_pullbacks(self):
        """Resampling perturbed curves refreshes field values and VJPs, including
        symmetry replicas."""
        for replicas in [False, True]:
            with self.subTest(replicas=replicas), self.case():
                self._case_resampling_updates_captured_geometry_and_pullbacks(replicas)

    def _case_resampling_updates_captured_geometry_and_pullbacks(self, replicas):
        """Run one row with case-local objects released before runtime cleanup."""
        curve = _curves()[0]
        sampler = GaussianSampler(curve.quadpoints, 1e-3, 0.2, n_derivs=1)
        sample = PerturbationSample(sampler, randomgen=np.random.default_rng(713))
        perturbed = CurvePerturbed(curve, sample)
        current = Current(1e5)
        coils = (
            coils_via_symmetries([perturbed], [current], 2, True)
            if replicas
            else [Coil(perturbed, current)]
        )
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        before = np.asarray(adapter.B()).copy()
        perturbed.resample()
        native.set_points(_POINTS)
        adapter.set_points(_POINTS)
        expected = native.B()
        actual = np.asarray(adapter.B())
        print(
            f"R6 replicas={replicas}: native_delta={np.max(np.abs(expected - before)):.16g} adapter_delta={np.max(np.abs(actual - before)):.16g} error={np.max(np.abs(actual - expected)):.16g}"
        )
        self.assertTrue(
            np.max(np.abs(expected - before)) > 1e-6,
            "np.max(np.abs(expected - before)) > 1e-6",
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(
            _gradient(adapter.B_vjp(_cotangent("B")), adapter),
            _gradient(cast(Derivative, native.B_vjp(_cotangent("B"))), native),
            rtol=1e-11,
            atol=1e-13,
        )

    def test_cartesian_points_are_owned_snapshots(self):
        """Mutating caller-owned Cartesian points leaves the adapter points and cached
        field equal to native snapshots."""
        storage = np.empty(_POINTS.size + 8, dtype=np.float64)
        offset = (-storage.ctypes.data % 64) // storage.itemsize
        points = storage[offset : offset + _POINTS.size].reshape(_POINTS.shape)
        points[:] = _POINTS
        coils = [Coil(_curves()[0], Current(1e5))]
        native = BiotSavart(coils).set_points(points)
        adapter = JaxBiotSavart(coils).set_points(points)
        adapter.B()
        points[:] += 0.3
        actual = adapter.get_points_cart()
        print(
            f"R8 point_error={np.max(np.abs(actual - native.get_points_cart())):.16g} field_error={np.max(np.abs(np.asarray(adapter.B()) - native.B())):.16g}"
        )
        np.testing.assert_array_equal(actual, native.get_points_cart())
        np.testing.assert_allclose(adapter.B(), native.B(), rtol=1e-12, atol=1e-14)

    def test_serialization_preserves_points_and_field(self):
        """GSON reconstruction and JSON loading preserve evaluation points and field
        values in either coordinate convention."""
        for coordinates in ["cart", "cyl"]:
            with self.subTest(coordinates=coordinates), self.case() as patches:
                self._case_serialization_preserves_points_and_field(
                    coordinates, patches
                )

    def _case_serialization_preserves_points_and_field(self, coordinates, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        tmp_path = Path(patches.enter_context(TemporaryDirectory()))
        coils = [Coil(_curves()[0], Current(1e5))]
        native, adapter = BiotSavart(coils), JaxBiotSavart(coils)
        for field in (native, adapter):
            getattr(field, "set_points_" + coordinates)(_POINTS)
        expected_points = native.get_points_cart()
        for name, field in (("native", native), ("adapter", adapter)):
            serialized = {}
            restored = type(field).from_dict(field.as_dict(serialized), serialized, {})
            filename = tmp_path / (name + ".json")
            field.save(filename, fmt="json")
            loaded = load(filename)
            print(
                f"R4 {coordinates} {name}: restored_points={restored.get_points_cart().tolist()}"
            )
            for result in (restored, loaded):
                np.testing.assert_array_equal(result.get_points_cart(), expected_points)
                np.testing.assert_allclose(
                    result.B(), native.B(), rtol=1e-12, atol=1e-14
                )

    def test_native_symmetry_curve_converts_for_fields_and_vjps(self):
        """Native symmetry Fourier curves yield matching JAX fields and VJPs with or
        without stellarator symmetry."""
        for stellsym in [False, True]:
            with self.subTest(stellsym=stellsym), self.case():
                self._case_native_symmetry_curve_converts_for_fields_and_vjps(stellsym)

    def _case_native_symmetry_curve_converts_for_fields_and_vjps(self, stellsym):
        """Run one row with case-local objects released before runtime cleanup."""
        curve = CurveXYZFourierSymmetries(64, 2, nfp=3, stellsym=stellsym, ntor=2)
        dofs = np.random.default_rng(1729).normal(0.0, 0.03, curve.local_full_dof_size)
        dofs[0] = 1.0
        curve.local_full_x = dofs
        coils = [Coil(curve, Current(1e5))]
        native = BiotSavart(coils).set_points(_POINTS)
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        for quantity in _FIELDS:
            actual, expected = (
                np.asarray(getattr(adapter, quantity)()),
                getattr(native, quantity)(),
            )
            print(
                f"R7 stellsym={stellsym} {quantity}: error={np.max(np.abs(actual - expected)):.16g}"
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
            np.testing.assert_allclose(
                _gradient(_vjp(adapter, quantity, _cotangent(quantity)), adapter),
                _gradient(_vjp(native, quantity, _cotangent(quantity)), native),
                rtol=1e-11,
                atol=1e-13,
            )

    def test_composed_curves_match_native_fields_owner_vjps_and_finite_difference(self):
        """Nested perturbed and filament curves match native fields, owner partials and
        directional finite differences."""
        for wrapper in _COMPOSED_WRAPPERS:
            for fixed in [False, True]:
                for quantity in _FIELDS:
                    with self.subTest(
                        wrapper=wrapper, fixed=fixed, quantity=quantity
                    ), self.case():
                        self._case_composed_curves_match_native_fields_owner_vjps_and_finite_difference(
                            wrapper, fixed, quantity
                        )

    def _case_composed_curves_match_native_fields_owner_vjps_and_finite_difference(
        self, wrapper, fixed, quantity
    ):
        """Run one row with case-local objects released before runtime cleanup."""
        native, curve, rotation = _composed_native_field(wrapper, fixed)
        expected_field = getattr(native, quantity)().copy()
        cotangent = _cotangent(quantity)
        expected_vjp = _vjp(native, quantity, cotangent)
        expected_gradient = _gradient(expected_vjp, native)
        expected_partials = cast(Derivative, expected_vjp(native, as_derivative=True))
        adapter = JaxBiotSavart(native.coils).set_points(_POINTS)
        np.testing.assert_allclose(
            getattr(adapter, quantity)(),
            expected_field,
            rtol=1e-12,
            atol=1e-14,
        )
        actual_vjp = _vjp(adapter, quantity, cotangent)
        actual_gradient = _gradient(actual_vjp, adapter)
        actual_partials = cast(Derivative, actual_vjp(adapter, as_derivative=True))
        np.testing.assert_allclose(
            actual_gradient,
            expected_gradient,
            rtol=1e-11,
            atol=1e-13,
        )
        self.assertTrue(
            curve in actual_partials.data,
            "curve in actual_partials.data",
        )
        if wrapper != "nested_perturbed":
            self.assertTrue(
                rotation in actual_partials.data,
                "rotation in actual_partials.data",
            )
        for owner in expected_partials.data:
            np.testing.assert_allclose(
                actual_partials.data[owner],
                expected_partials.data[owner],
                rtol=1e-11,
                atol=1e-13,
            )
        pullback = getattr(adapter, quantity + "_pullback_native")(cotangent)
        flat = adapter.coil_cotangents_to_dofs_gradient(
            pullback.d_coil_arrays,
            pullback.coil_indices,
        )
        np.testing.assert_allclose(flat, expected_gradient, rtol=1e-11, atol=1e-13)
        if quantity == "A":
            # The value half of the combined VJP must ignore the gradient cotangent.
            vgrad = _cotangent("dA_by_dX")
            combined_native = cast(
                Derivative,
                native.A_and_dA_vjp(cotangent, vgrad)[0](native, as_derivative=True),
            )
            combined_adapter = cast(
                Derivative,
                adapter.A_and_dA_vjp(cotangent, vgrad)[0](adapter, as_derivative=True),
            )
            for owner in expected_partials.data:
                np.testing.assert_allclose(
                    combined_adapter.data[owner],
                    combined_native.data[owner],
                    rtol=1e-11,
                    atol=1e-13,
                )
                np.testing.assert_allclose(
                    combined_adapter.data[owner],
                    actual_partials.data[owner],
                    rtol=1e-11,
                    atol=1e-13,
                )

        dofs = native.x.copy()
        direction = np.linspace(-0.3, 0.4, dofs.size)
        step = 1e-4
        objectives = []
        for multiple in (2, 1, -1, -2):
            native.x = dofs + multiple * step * direction
            perturbed_field = getattr(native, quantity)()
            np.testing.assert_allclose(
                getattr(adapter, quantity)(),
                perturbed_field,
                rtol=1e-12,
                atol=1e-14,
            )
            objectives.append(np.sum(perturbed_field * cotangent))
        native.x = dofs
        finite_difference = (
            -objectives[0] + 8 * objectives[1] - 8 * objectives[2] + objectives[3]
        ) / (12 * step)
        np.testing.assert_allclose(
            actual_gradient @ direction,
            finite_difference,
            rtol=1e-9,
            atol=1e-14,
        )

    def test_composed_curve_vjps_taylor(self):
        """Composed-curve VJPs converge under the native forward Taylor stencil."""
        for wrapper in _COMPOSED_WRAPPERS:
            for fixed in [False, True]:
                for quantity in _FIELDS:
                    with self.subTest(
                        wrapper=wrapper, fixed=fixed, quantity=quantity
                    ), self.case() as patches:
                        self._case_composed_curve_vjps_taylor(
                            wrapper, fixed, quantity, patches
                        )

    def _case_composed_curve_vjps_taylor(self, wrapper, fixed, quantity, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        native, curve, rotation = _composed_native_field(wrapper, fixed)
        cotangent = _cotangent(quantity)
        adapter = JaxBiotSavart(native.coils).set_points(_POINTS)
        actual_gradient = _gradient(_vjp(adapter, quantity, cotangent), adapter)

        dofs = native.x.copy()
        patches.callback(setattr, native, "x", dofs)
        direction = 1e-2 * np.random.RandomState(1).rand(dofs.size)
        baseline = np.sum(getattr(adapter, quantity)() * cotangent)
        analytic = actual_gradient @ direction
        error = 1e6
        for exponent in range(5, 10):
            step = 0.5**exponent
            adapter.x = dofs + step * direction
            value = np.sum(getattr(adapter, quantity)() * cotangent)
            next_error = np.linalg.norm((value - baseline) / step - analytic)
            self.assertLess(next_error, 0.55 * error)
            error = next_error

    def test_affine_current_vjps_shrinking_steps(self):
        """Affine-current VJPs retain linearity through five halved central steps."""
        for scale in (1.0, -2.5):
            for quantity in _FIELDS:
                with self.subTest(
                    scale=scale, quantity=quantity
                ), self.case() as patches:
                    self._case_affine_current_vjps_shrinking_steps(
                        scale, quantity, patches
                    )

    def _case_affine_current_vjps_shrinking_steps(self, scale, quantity, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = _curves()
        for curve in curves:
            curve.fix_all()
        first, total = Current(3.0), Current(10.0)
        total.fix_all()
        coils = [
            Coil(curves[0], first),
            Coil(curves[1], scale * (total - first)),
        ]
        adapter = JaxBiotSavart(coils).set_points(_POINTS)
        cotangent = _cotangent(quantity)
        dofs = adapter.x.copy()
        patches.callback(setattr, adapter, "x", dofs)
        analytic = _gradient(_vjp(adapter, quantity, cotangent), adapter)
        for exponent in range(5):
            step = 1e-3 * 0.5**exponent
            adapter.x = dofs + step
            plus = np.sum(getattr(adapter, quantity)() * cotangent)
            adapter.x = dofs - step
            minus = np.sum(getattr(adapter, quantity)() * cotangent)
            np.testing.assert_allclose(
                analytic,
                (plus - minus) / (2 * step),
                rtol=1e-9,
                atol=1e-14,
            )
