"""Native regression oracles for the BiotSavartJAX correctness review."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401
from typing import cast

import numpy as np
import pytest

from simsopt import load
from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt.geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from simsopt.geo.finitebuild import CurveFilament
from simsopt.geo.framedcurve import FrameRotation, FramedCurveFrenet
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1]])
_FIELDS = ("B", "A", "dB_by_dX", "dA_by_dX")


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


@pytest.mark.parametrize("quantity", _FIELDS)
@pytest.mark.parametrize("scale", [1.0, -2.5])
def test_affine_current_vjps_match_native_and_finite_difference(quantity, scale):
    curves = _curves()
    for curve in curves:
        curve.fix_all()
    first, total = Current(3.0), Current(10.0)
    total.fix_all()
    coils = [Coil(curves[0], first), Coil(curves[1], scale * (total - first))]
    native = BiotSavart(coils).set_points(_POINTS)
    adapter = BiotSavartJAX(coils).set_points(_POINTS)
    cotangent = _cotangent(quantity)
    np.testing.assert_allclose(getattr(adapter, quantity)(), getattr(native, quantity)(), rtol=1e-12, atol=1e-14)
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
    flat = adapter.coil_cotangents_to_dofs_gradient(pullback.d_coil_arrays, pullback.coil_indices)
    print(f"R9 {quantity} scale={scale}: adapter={actual[0]:.16g} flat={float(flat[0]):.16g} native={expected[0]:.16g} fd={finite_difference:.16g}")
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(flat, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(actual, finite_difference, rtol=1e-9, atol=1e-14)
    native_partials = cast(Derivative, _vjp(native, quantity, cotangent)(native, as_derivative=True))
    adapter_partials = cast(Derivative, _vjp(adapter, quantity, cotangent)(adapter, as_derivative=True))
    for owner in native_partials.data:
        np.testing.assert_allclose(adapter_partials.data[owner], native_partials.data[owner], rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("quantity", _FIELDS)
@pytest.mark.parametrize("fixed", ["partial", "curve", "all"])
def test_derivative_retains_fixed_partials(quantity, fixed):
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
    adapter = BiotSavartJAX(coils).set_points(_POINTS)
    expected = cast(Derivative, _vjp(native, quantity, _cotangent(quantity))(native, as_derivative=True))
    derivative = _vjp(adapter, quantity, _cotangent(quantity))
    actual = cast(Derivative, derivative(adapter, as_derivative=True))
    print(f"R5 {quantity} {fixed}: current={cast(np.ndarray, actual.data[current])[0]:.16g}/{cast(np.ndarray, expected.data[current])[0]:.16g} curve[0]={cast(np.ndarray, actual.data[curve])[0]:.16g}/{cast(np.ndarray, expected.data[curve])[0]:.16g}")
    for owner in expected.data:
        np.testing.assert_allclose(actual.data[owner], expected.data[owner], rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(_gradient(derivative, adapter), _gradient(_vjp(native, quantity, _cotangent(quantity)), native), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("replicas", [False, True])
def test_resampling_updates_captured_geometry_and_pullbacks(replicas):
    curve = _curves()[0]
    sampler = GaussianSampler(curve.quadpoints, 1e-3, 0.2, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=np.random.default_rng(713))
    perturbed = CurvePerturbed(curve, sample)
    current = Current(1e5)
    coils = coils_via_symmetries([perturbed], [current], 2, True) if replicas else [Coil(perturbed, current)]
    native = BiotSavart(coils).set_points(_POINTS)
    adapter = BiotSavartJAX(coils).set_points(_POINTS)
    before = np.asarray(adapter.B()).copy()
    perturbed.resample()
    native.set_points(_POINTS)
    adapter.set_points(_POINTS)
    expected = native.B()
    actual = np.asarray(adapter.B())
    print(f"R6 replicas={replicas}: native_delta={np.max(np.abs(expected-before)):.16g} adapter_delta={np.max(np.abs(actual-before)):.16g} error={np.max(np.abs(actual-expected)):.16g}")
    assert np.max(np.abs(expected - before)) > 1e-6
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(_gradient(adapter.B_vjp(_cotangent("B")), adapter), _gradient(cast(Derivative, native.B_vjp(_cotangent("B"))), native), rtol=1e-11, atol=1e-13)


def test_cartesian_points_are_owned_snapshots():
    storage = np.empty(_POINTS.size + 8, dtype=np.float64)
    offset = (-storage.ctypes.data % 64) // storage.itemsize
    points = storage[offset:offset + _POINTS.size].reshape(_POINTS.shape)
    points[:] = _POINTS
    coils = [Coil(_curves()[0], Current(1e5))]
    native = BiotSavart(coils).set_points(points)
    adapter = BiotSavartJAX(coils).set_points(points)
    adapter.B()
    points[:] += 0.3
    actual = adapter.get_points_cart()
    print(f"R8 point_error={np.max(np.abs(actual-native.get_points_cart())):.16g} field_error={np.max(np.abs(np.asarray(adapter.B())-native.B())):.16g}")
    np.testing.assert_array_equal(actual, native.get_points_cart())
    np.testing.assert_allclose(adapter.B(), native.B(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("coordinates", ["cart", "cyl"])
def test_serialization_preserves_points_and_field(tmp_path, coordinates):
    coils = [Coil(_curves()[0], Current(1e5))]
    native, adapter = BiotSavart(coils), BiotSavartJAX(coils)
    for field in (native, adapter):
        getattr(field, "set_points_" + coordinates)(_POINTS)
    expected_points = native.get_points_cart()
    for name, field in (("native", native), ("adapter", adapter)):
        serialized = {}
        restored = type(field).from_dict(field.as_dict(serialized), serialized, {})
        filename = tmp_path / (name + ".json")
        field.save(filename, fmt="json")
        loaded = load(filename)
        print(f"R4 {coordinates} {name}: restored_points={restored.get_points_cart().tolist()}")
        for result in (restored, loaded):
            np.testing.assert_array_equal(result.get_points_cart(), expected_points)
            np.testing.assert_allclose(result.B(), native.B(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("stellsym", [False, True])
def test_native_symmetry_curve_converts_for_fields_and_vjps(stellsym):
    curve = CurveXYZFourierSymmetries(64, 2, nfp=3, stellsym=stellsym, ntor=2)
    dofs = np.random.default_rng(1729).normal(0.0, 0.03, curve.local_full_dof_size)
    dofs[0] = 1.0
    curve.local_full_x = dofs
    coils = [Coil(curve, Current(1e5))]
    native = BiotSavart(coils).set_points(_POINTS)
    adapter = BiotSavartJAX(coils).set_points(_POINTS)
    for quantity in _FIELDS:
        actual, expected = np.asarray(getattr(adapter, quantity)()), getattr(native, quantity)()
        print(f"R7 stellsym={stellsym} {quantity}: error={np.max(np.abs(actual-expected)):.16g}")
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
        np.testing.assert_allclose(_gradient(_vjp(adapter, quantity, _cotangent(quantity)), adapter), _gradient(_vjp(native, quantity, _cotangent(quantity)), native), rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("quantity", ["B", "dB_by_dX"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("wrapper", [
    "shared_filament", "nested_perturbed", "perturbed_filament",
    "perturbed_shared_filament",
])
def test_composed_curves_match_native_fields_owner_vjps_and_finite_difference(
    quantity, fixed, wrapper,
):
    curve = create_equally_spaced_curves(
        1, 1, stellsym=False, R0=1.0, R1=0.25, order=1, numquadpoints=24,
    )[0]
    current = Current(1e5)
    shared = "shared" in wrapper
    rotation = FrameRotation(
        curve.quadpoints, order=4 if shared else 1,
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
        wrapped = CurveFilament(FramedCurveFrenet(curve, rotation), 0.01, 0.02)
        if wrapper.startswith("perturbed"):
            wrapped = CurvePerturbed(wrapped, sample)
    coils = [Coil(wrapped, current)]
    native = BiotSavart(coils).set_points(_POINTS)
    expected_field = getattr(native, quantity)().copy()
    cotangent = _cotangent(quantity)
    expected_vjp = _vjp(native, quantity, cotangent)
    expected_gradient = _gradient(expected_vjp, native)
    expected_partials = cast(Derivative, expected_vjp(native, as_derivative=True))
    adapter = BiotSavartJAX(coils).set_points(_POINTS)
    np.testing.assert_allclose(
        getattr(adapter, quantity)(), expected_field, rtol=1e-12, atol=1e-14,
    )
    actual_vjp = _vjp(adapter, quantity, cotangent)
    actual_gradient = _gradient(actual_vjp, adapter)
    actual_partials = cast(Derivative, actual_vjp(adapter, as_derivative=True))
    np.testing.assert_allclose(
        actual_gradient, expected_gradient, rtol=1e-11, atol=1e-13,
    )
    assert curve in actual_partials.data
    if wrapper != "nested_perturbed":
        assert rotation in actual_partials.data
    for owner in expected_partials.data:
        np.testing.assert_allclose(
            actual_partials.data[owner], expected_partials.data[owner],
            rtol=1e-11, atol=1e-13,
        )
    pullback = (
        adapter.B_pullback_native(cotangent) if quantity == "B"
        else adapter.dB_by_dX_pullback_native(cotangent)
    )
    flat = adapter.coil_cotangents_to_dofs_gradient(
        pullback.d_coil_arrays, pullback.coil_indices,
    )
    np.testing.assert_allclose(flat, expected_gradient, rtol=1e-11, atol=1e-13)

    dofs = native.x.copy()
    direction = np.linspace(-0.3, 0.4, dofs.size)
    step = 1e-4
    objectives = []
    for multiple in (2, 1, -1, -2):
        native.x = dofs + multiple * step * direction
        perturbed_field = getattr(native, quantity)()
        np.testing.assert_allclose(
            getattr(adapter, quantity)(), perturbed_field, rtol=1e-12, atol=1e-14,
        )
        objectives.append(np.sum(perturbed_field * cotangent))
    native.x = dofs
    finite_difference = (
        -objectives[0] + 8 * objectives[1] - 8 * objectives[2] + objectives[3]
    ) / (12 * step)
    np.testing.assert_allclose(
        actual_gradient @ direction, finite_difference, rtol=1e-9, atol=1e-14,
    )
