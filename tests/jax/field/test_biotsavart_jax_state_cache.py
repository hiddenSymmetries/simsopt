"""BiotSavartJAX reuses its coil state per DOF state and never serves a stale one.

The adapter builds grouped coil geometry once per coil-DOF state and caches
field values per state and point set. Every mutation simsopt notifies through
its recompute mechanism (DOF setters, fix/unfix, resample) and every point or
backend change must retire that state; native BiotSavart is the oracle.
"""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import copy
from typing import cast
import weakref

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from simsopt._core.derivative import Derivative
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves
from simsopt.geo.curveperturbed import CurvePerturbed, GaussianSampler, PerturbationSample
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.backend import invalidate_backend_cache
from simsopt_jax_adapters.field import biotsavart_backend as backend


_POINTS = np.array([[0.8, 0.1, 0.2], [1.1, -0.2, -0.1], [0.9, 0.3, -0.05]])
_OTHER_POINTS = np.array([[1.2, 0.05, 0.1], [0.7, -0.3, 0.0]])
_COTANGENT = np.arange(9, dtype=float).reshape(3, 3) / 11


def _base_curves(count=2):
    return create_equally_spaced_curves(
        count, 1, stellsym=False, R0=1.0, R1=0.25, order=2, numquadpoints=32
    )


def _fields(coils, points=_POINTS):
    field, native = backend.BiotSavartJAX(coils), BiotSavart(coils)
    field.set_points(points)
    native.set_points(points)
    return field, native


def _assert_field_matches_native(field, native):
    """B, dB/dX and the B VJP equal native BiotSavart's at the live state."""
    np.testing.assert_allclose(np.asarray(field.B()), native.B(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(field.dB_by_dX()), native.dB_by_dX(), rtol=1e-12, atol=1e-14
    )
    cotangent = _COTANGENT[: native.B().shape[0]]
    expected = cast(Derivative, native.B_vjp(cotangent))
    np.testing.assert_allclose(
        cast(np.ndarray, field.B_vjp(cotangent)(field)),
        cast(np.ndarray, expected(native)),
        rtol=1e-11, atol=1e-13,
    )


def _count_coil_state_builds(monkeypatch):
    builds = []
    build = backend._jitted_coil_set_spec_from_extraction_spec

    def counted_build(extraction_spec, coil_dofs):
        builds.append(1)
        return build(extraction_spec, coil_dofs)

    monkeypatch.setattr(backend, "_jitted_coil_set_spec_from_extraction_spec", counted_build)
    return builds


def test_evaluations_at_one_dof_state_build_coil_state_once(monkeypatch):
    curves = _base_curves()
    field, native = _fields(coils_via_symmetries(curves, [Current(1e5) for _ in curves], 2, True))
    builds = _count_coil_state_builds(monkeypatch)
    B_evaluations = []
    grouped_B = backend.grouped_biot_savart_B_from_spec

    def counted_grouped_B(points, coil_set_spec):
        B_evaluations.append(1)
        return grouped_B(points, coil_set_spec)

    monkeypatch.setattr(backend, "grouped_biot_savart_B_from_spec", counted_grouped_B)

    first = field.B()
    for _ in range(3):
        assert field.B() is first
    field.dB_by_dX()
    field.B_vjp(_COTANGENT)(field)
    field.dB_by_dcoilcurrents()
    field.coil_set_spec()
    assert (len(builds), len(B_evaluations)) == (1, 1), (
        "repeated evaluations at an unchanged coil-DOF state rebuilt the coil "
        f"spec {len(builds)} times and B {len(B_evaluations)} times"
    )

    field.x = field.x + 1e-3
    _assert_field_matches_native(field, native)
    assert len(builds) == 2


@pytest.mark.parametrize(
    "method_name,kernel",
    [
        ("dB_by_dcoilcurrents", backend.biot_savart_B),
        ("d2B_by_dXdcoilcurrents", backend.biot_savart_dB_by_dX),
        ("d3B_by_dXdXdcoilcurrents", backend.biot_savart_d2B_by_dXdX),
        ("dA_by_dcoilcurrents", backend.biot_savart_A),
        ("d2A_by_dXdcoilcurrents", backend.biot_savart_dA_by_dX),
        ("d3A_by_dXdXdcoilcurrents", backend.biot_savart_d2A_by_dXdX),
    ],
)
def test_per_current_outputs_reuse_arrays_and_follow_every_state_change(method_name, kernel):
    curves = _base_curves()
    currents = [Current(1e5), Current(-3e4)]
    field, _native = _fields([Coil(curve, current) for curve, current in zip(curves, currents)])
    evaluate = getattr(field, method_name)
    previous = evaluate()
    repeated = evaluate()
    assert repeated is not previous
    assert all(a is b for a, b in zip(previous, repeated, strict=True))
    repeated.clear()
    assert len(evaluate()) == len(field.coils), "caller list edits changed cached outputs"

    for mutation in ("free", "full", "parent", "current", "layout", "fixed", "points", "backend"):
        if mutation == "free":
            field.x = field.x * 1.01
        elif mutation == "full":
            field.full_x = field.full_x * .99
        elif mutation == "parent":
            curves[1].x = curves[1].x + 2e-3
        elif mutation == "current":
            currents[0].x = np.array([2.5e5])
        elif mutation == "layout":
            curves[0].fix(0)
        elif mutation == "fixed":
            curves[0].set(0, curves[0].get(0) + 5e-3)
        elif mutation == "points":
            field.set_points(_OTHER_POINTS)
        else:
            invalidate_backend_cache()
        actual = evaluate()
        expected = backend._per_coil_unit_field(field.get_points_cart_ref(), field.coil_set_spec(), kernel)
        assert all(a is not b for a, b in zip(actual, previous, strict=True)), mutation
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=mutation)
        assert all(a is b for a, b in zip(actual, evaluate(), strict=True)), mutation
        previous = actual


@pytest.mark.parametrize("setter", ["cart", "jax", "cyl", "spec", "clear"])
def test_point_changes_release_cached_outputs_and_preserve_coil_geometry(setter):
    field, _native = _fields([Coil(curve, Current(1e5)) for curve in _base_curves()])
    spec = field.coil_set_spec()
    field.B()
    field.dB_by_dX()
    field.B_and_dB()
    field.dB_by_dcoilcurrents()
    references = [weakref.ref(array) for array in jax.tree_util.tree_leaves(
        tuple(field._field_outputs.values()))]

    if setter == "cart":
        field.set_points(_OTHER_POINTS)
    elif setter == "jax":
        field.set_points(jnp.asarray(_OTHER_POINTS))
    elif setter == "cyl":
        field.set_points_cyl(np.array([[1.0, 0.3, 0.1], [0.9, -0.2, -0.1]]))
    elif setter == "spec":
        field.set_points_from_spec(backend.make_field_eval_spec(jnp.asarray(_OTHER_POINTS)))
    else:
        field.clear_points()

    assert all(reference() is None for reference in references), "obsolete outputs remain retained"
    assert field.coil_set_spec() is spec


def test_field_and_parent_dof_changes_refresh_the_field():
    curves = _base_curves()
    field, native = _fields(coils_via_symmetries(curves, [Current(1e5) for _ in curves], 2, True))
    _assert_field_matches_native(field, native)

    field.x = field.x * 1.01
    _assert_field_matches_native(field, native)

    curves[1].x = curves[1].x + 2e-3
    _assert_field_matches_native(field, native)

    field.full_x = field.full_x * 0.995
    _assert_field_matches_native(field, native)


def test_current_change_refreshes_the_field():
    curves = _base_curves()
    currents = [Current(1e5), Current(-3e4)]
    field, native = _fields([Coil(curve, current) for curve, current in zip(curves, currents)])
    _assert_field_matches_native(field, native)

    currents[0].x = np.array([2.5e5])
    _assert_field_matches_native(field, native)


def test_fixed_dof_change_refreshes_the_field():
    curves = _base_curves()
    field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
    _assert_field_matches_native(field, native)

    curves[0].fix(0)
    _assert_field_matches_native(field, native)
    curves[0].set(0, curves[0].get(0) + 5e-3)
    _assert_field_matches_native(field, native)


def test_set_points_refreshes_the_field():
    curves = _base_curves()
    field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
    _assert_field_matches_native(field, native)

    for points in (_OTHER_POINTS, _POINTS):
        field.set_points(points)
        native.set_points(points)
        _assert_field_matches_native(field, native)


def test_shared_dofs_refresh_every_coil_that_reads_them():
    curve = _base_curves(1)[0]
    shared = CurveXYZFourier(curve.quadpoints, curve.order, dofs=curve.dofs)
    field, native = _fields([Coil(curve, Current(1e5)), Coil(shared, Current(-4e4))])
    _assert_field_matches_native(field, native)

    shared.x = np.asarray(shared.x) + 3e-3
    np.testing.assert_array_equal(curve.x, shared.x)
    _assert_field_matches_native(field, native)


def test_current_sum_term_change_refreshes_the_field():
    curves = _base_curves()
    first, second = Current(1e5), Current(2e4)
    field, native = _fields([Coil(curves[0], first + second), Coil(curves[1], -2.5 * (first - second))])
    _assert_field_matches_native(field, native)

    second.x = np.array([-7e4])
    _assert_field_matches_native(field, native)


def test_resample_refreshes_the_field():
    base = _base_curves(1)[0]
    sampler = GaussianSampler(base.quadpoints, 1e-3, 0.2, n_derivs=1)
    sample = PerturbationSample(sampler, randomgen=np.random.default_rng(2024))
    perturbed = CurvePerturbed(base, sample)
    coils = [Coil(perturbed, Current(1e5))]
    field, native = _fields(coils)
    _assert_field_matches_native(field, native)

    before = np.asarray(field.B()).copy()
    perturbed.resample()
    _assert_field_matches_native(field, native)
    assert not np.array_equal(np.asarray(field.B()), before)

    # A sample redrawn without notification: native caches stay stale, the
    # adapter detects the replaced sample on its next evaluation.
    before = np.asarray(field.B()).copy()
    sample.resample()
    perturbed.invalidate_cache()
    _assert_field_matches_native(field, _fields(coils)[1])
    assert not np.array_equal(np.asarray(field.B()), before)


def test_backend_reconfiguration_rebuilds_the_coil_state(monkeypatch):
    curves = _base_curves()
    field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
    builds = _count_coil_state_builds(monkeypatch)
    field.B()
    field.B()
    assert len(builds) == 1

    invalidate_backend_cache()
    _assert_field_matches_native(field, native)
    assert len(builds) == 2


def test_owner_updates_fingerprint_fixed_dofs_once(monkeypatch):
    """Each owner update notifies the field; fixed DOFs are compared once, at the next read."""
    curves = _base_curves(3)
    field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
    curves[0].fix(0)
    _assert_field_matches_native(field, native)
    reads = []
    fingerprint = field._current_captured_coil_state_fingerprint

    def counted_fingerprint():
        reads.append(1)
        return fingerprint()

    monkeypatch.setattr(field, "_current_captured_coil_state_fingerprint", counted_fingerprint)
    for curve in curves:
        curve.x = np.asarray(curve.x) + 1e-3
    field.B()
    field.B_vjp(_COTANGENT)(field)
    assert len(reads) == 1, f"{len(reads)} fixed-DOF fingerprints for one evaluation"
    _assert_field_matches_native(field, native)


def _aligned_copy(points):
    """A C-contiguous, 64-byte-aligned copy, which CPU JAX placement aliases instead of copying."""
    raw = np.empty(points.size + 8)
    offset = (-raw.ctypes.data % 64) // raw.itemsize
    aligned = raw[offset:offset + points.size].reshape(points.shape)
    aligned[...] = points
    return aligned


def test_set_points_owns_a_jax_point_set_that_aliases_a_numpy_buffer():
    """Editing the caller's buffer after set_points moves neither the points nor B."""
    curves = _base_curves()
    field, native = _fields([Coil(curve, Current(1e5)) for curve in curves])
    buffer = _aligned_copy(_POINTS)
    points = jnp.asarray(buffer)
    if next(iter(points.devices())).platform == "cpu":
        assert np.shares_memory(np.asarray(points), buffer), "precondition: CPU placement aliases"
    field.set_points(points)
    field.B()

    buffer += 0.3

    np.testing.assert_array_equal(field.get_points_cart(), _POINTS)
    _assert_field_matches_native(field, native)


@pytest.mark.parametrize("make_copy", ["copy", "deepcopy_shared_coils"])
def test_copies_follow_later_dof_changes(make_copy):
    """A copy is a registered field over the coils: later DOF and layout changes reach it."""
    curves = _base_curves()
    currents = [Current(1e5), Current(-3e4)]
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    field, native = _fields(coils)
    field.B()
    field.B_vjp(_COTANGENT)(field)
    copied = (
        copy.copy(field) if make_copy == "copy"
        else copy.deepcopy(field, memo={id(field.coils): field.coils})
    )
    assert copied is not field
    assert all(mine is theirs for mine, theirs in zip(copied.coils, field.coils, strict=True))
    np.testing.assert_array_equal(copied.get_points_cart(), _POINTS)

    currents[0].x = np.array([2e5])
    curves[1].x = np.asarray(curves[1].x) + 2e-3
    _assert_field_matches_native(copied, native)
    _assert_field_matches_native(field, native)

    curves[0].fix(0)
    curves[0].set(0, curves[0].get(0) + 5e-3)
    assert copied.dof_size == field.dof_size
    _assert_field_matches_native(copied, native)


def test_deepcopy_needs_deep_copyable_coils():
    """simsopt coils are not deep-copyable, so neither is a field over them (as natively)."""
    curves = _base_curves()
    field, _native = _fields([Coil(curve, Current(1e5)) for curve in curves])
    with pytest.raises(TypeError):
        copy.deepcopy(field)
