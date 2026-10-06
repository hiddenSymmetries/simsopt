"""JAX coil-geometry penalties against the native curve objectives."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from collections.abc import Callable
from typing import cast

import jax
import numpy as np
import pytest

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    CurveXYZFourier,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt_jax_adapters.geo import (
    CurveCurveDistanceJAX,
    CurveLengthJAX,
    CurveSurfaceDistanceJAX,
    LpCurveCurvatureJAX,
    MeanSquaredCurvatureJAX,
)

# The thresholds make every penalty active at the test state.
_CC_THRESHOLD = 0.6
_CS_THRESHOLD = 0.4
_CURVATURE_THRESHOLD = 1.0


def _surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_nphi_ntheta(nphi=8, ntheta=8, nfp=2, range="half period")
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    return surface


def _curves(*, shared_curve: bool = False) -> tuple[list, list]:
    """Return base curves and all symmetric copies, with some base DOFs fixed.

    With ``shared_curve``, one more curve with its own quadrature grid shares
    the DOFs object of the second base curve. The grid is offset so that no
    point coincides with one of that curve (a zero distance has no gradient).
    """
    base = create_equally_spaced_curves(
        3, 2, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24
    )
    rng = np.random.default_rng(7)
    for curve in base:
        curve.x = curve.x + 0.03 * rng.standard_normal(curve.x.shape)
    base[0].fix("xc(0)")
    base[0].fix("zs(1)")
    curves = [
        coil.curve
        for coil in coils_via_symmetries(base, [Current(1.0) for _ in base], 2, True)
    ]
    if shared_curve:
        twin = CurveXYZFourier(np.linspace(0, 1, 30, endpoint=False) + 0.013, 3, dofs=base[1].dofs)
        curves.append(twin)
    return base, curves


def _single_curve_objectives(curve) -> list[tuple[str, Optimizable, Optimizable]]:
    return [
        ("CurveLength", CurveLength(curve), CurveLengthJAX(curve)),
        (
            "LpCurveCurvature",
            LpCurveCurvature(curve, 2, _CURVATURE_THRESHOLD),
            LpCurveCurvatureJAX(curve, 2, _CURVATURE_THRESHOLD),
        ),
        ("MeanSquaredCurvature", MeanSquaredCurvature(curve), MeanSquaredCurvatureJAX(curve)),
    ]


def _distance_objectives(curves, surface, num_basecurves) -> list[tuple[str, Optimizable, Optimizable]]:
    return [
        (
            "CurveCurveDistance",
            CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves),
            CurveCurveDistanceJAX(curves, _CC_THRESHOLD, num_basecurves=num_basecurves),
        ),
        (
            "CurveCurveDistance(downsample=2)",
            CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves, downsample=2),
            CurveCurveDistanceJAX(curves, _CC_THRESHOLD, num_basecurves=num_basecurves, downsample=2),
        ),
        (
            "CurveSurfaceDistance",
            CurveSurfaceDistance(curves, surface, _CS_THRESHOLD),
            CurveSurfaceDistanceJAX(curves, surface, _CS_THRESHOLD),
        ),
    ]


def _all_objectives(*, shared_curve: bool = False):
    base, curves = _curves(shared_curve=shared_curve)
    return (
        _single_curve_objectives(base[0])
        + _distance_objectives(curves, _surface(), len(base))
    )


def _partials(objective: Optimizable) -> Derivative:
    return cast(Callable[..., Derivative], objective.dJ)(partials=True)


def _assert_matches_native(name: str, native: Optimizable, adapter: Optimizable) -> None:
    native_value = float(native.J())
    assert native_value > 0.0, f"{name} must be active at the test state"
    np.testing.assert_allclose(adapter.J(), native_value, rtol=1e-12, atol=1e-14, err_msg=name)
    native_gradient = native.dJ()
    assert np.all(np.isfinite(native_gradient)), name
    np.testing.assert_allclose(
        adapter.dJ(), native_gradient, rtol=1e-11, atol=1e-13, err_msg=f"{name} free gradient"
    )
    native_partials, adapter_partials = _partials(native), _partials(adapter)
    assert set(adapter_partials.data) == set(native_partials.data), name
    for owner, expected in native_partials.data.items():
        # Fixed DOFs keep their partials, as in the native Derivative.
        np.testing.assert_allclose(
            adapter_partials.data[owner], expected, rtol=1e-11, atol=1e-13,
            err_msg=f"{name} partials of {owner.name}",
        )


@pytest.mark.parametrize("shared_curve", [False, True], ids=["symmetric_copies", "shared_dofs_twin"])
def test_penalties_match_native_values_gradients_and_partials(shared_curve):
    for name, native, adapter in _all_objectives(shared_curve=shared_curve):
        assert adapter.dof_names == native.dof_names, name
        _assert_matches_native(name, native, adapter)


def test_penalty_gradients_match_central_differences():
    rng = np.random.default_rng(3)
    for name, _, adapter in _all_objectives(shared_curve=True):
        x0 = np.array(adapter.x, dtype=float)
        direction = rng.standard_normal(x0.shape)
        step = 1e-6
        adapter.x = x0 + step * direction
        plus = adapter.J()
        adapter.x = x0 - step * direction
        minus = adapter.J()
        adapter.x = x0
        np.testing.assert_allclose(
            adapter.dJ() @ direction, (plus - minus) / (2 * step), rtol=1e-6, err_msg=name
        )


def test_penalties_follow_dof_changes_like_native():
    for name, native, adapter in _all_objectives():
        adapter.x = np.asarray(adapter.x) + 0.01
        _assert_matches_native(name, native, adapter)


def test_curve_surface_distance_depends_on_curves_only():
    base, curves = _curves()
    surface = _surface()
    native = CurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
    adapter = CurveSurfaceDistanceJAX(curves, surface, _CS_THRESHOLD)
    surface_names = set(surface.dof_names)
    assert adapter.dof_names == native.dof_names
    assert surface_names and not surface_names & set(adapter.dof_names)
    assert all(owner is not surface for owner in _partials(adapter).data)


def _segment(x_center, x_amplitude, numquadpoints=15):
    """``x = x_center + x_amplitude cos(2 pi t)``, ``y = z = 0``: zero tangent at ``t = 0``."""
    curve = CurveXYZFourier(numquadpoints, 1)
    curve.set("xc(0)", x_center)
    curve.set("xc(1)", x_amplitude)
    return curve


def _circle(center, radius, numquadpoints=20):
    curve = CurveXYZFourier(numquadpoints, 1)
    curve.set("xc(0)", center[0])
    curve.set("yc(0)", center[1])
    curve.set("zc(0)", center[2])
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    return curve


@pytest.mark.parametrize("threshold", [0.1, 5.0], ids=["no_candidates", "candidates"])
def test_shortest_distances_match_native(threshold):
    """Native scans all pairs only when no selected pair is a candidate.

    The close pair (2, 1) is outside the pairs ``j < num_basecurves = 1``: it
    sets the shortest distance only when the selected pairs are all far.
    """
    curves = [_circle((0.0, 0.0, 0.0), 1.0), _circle((5.0, 0.0, 0.0), 1.0),
              _circle((5.0, 0.0, 0.01), 1.0)]
    surface = _surface()
    pairs = (
        (CurveCurveDistance(curves, threshold, num_basecurves=1),
         CurveCurveDistanceJAX(curves, threshold, num_basecurves=1)),
        (CurveSurfaceDistance(curves, surface, threshold),
         CurveSurfaceDistanceJAX(curves, surface, threshold)),
    )
    for native, adapter in pairs:
        np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
    expected_curve_distance = 0.01 if threshold == 0.1 else pairs[0][0].shortest_distance()
    np.testing.assert_allclose(pairs[0][1].shortest_distance(), expected_curve_distance, rtol=1e-12)
    assert threshold == 0.1 or pairs[0][1].shortest_distance() > 1.0


def test_curve_curve_distance_follows_num_basecurves():
    base, curves = _curves()
    native = CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=1)
    adapter = CurveCurveDistanceJAX(curves, _CC_THRESHOLD, num_basecurves=1)
    _assert_matches_native("num_basecurves=1", native, adapter)
    value = adapter.J()
    native.num_basecurves = adapter.num_basecurves = len(base)
    # Native refreshes its candidate pairs only after a DOF change.
    native.recompute_bell()
    _assert_matches_native("num_basecurves=3", native, adapter)
    assert adapter.J() > value


def test_inactive_distance_terms_have_finite_zero_gradients():
    """A far curve with zero tangents adds J = 0 and a zero, finite gradient.

    The native objectives skip such curves; the dense JAX kernels evaluate them.
    """
    base, curves = _curves()
    point = _circle((50.0, 0.0, 0.0), 0.0)
    curves = [*curves, point]
    surface = _surface()
    for name, native, adapter in _distance_objectives(curves, surface, len(base)):
        _assert_matches_native(name, native, adapter)
        point_gradient = cast(np.ndarray, _partials(adapter)(point))
        assert np.all(np.isfinite(point_gradient)) and not np.any(point_gradient), name


def _assert_matches_native_including_nans(name: str, native: Optimizable, adapter: Optimizable) -> None:
    """Value, free gradient and partials equal to native, non-finite entries included."""
    np.testing.assert_allclose(adapter.J(), float(native.J()), rtol=1e-12, atol=1e-14, err_msg=name)
    np.testing.assert_allclose(adapter.dJ(), native.dJ(), rtol=1e-11, atol=1e-13, err_msg=name)
    native_partials, adapter_partials = _partials(native), _partials(adapter)
    assert set(adapter_partials.data) == set(native_partials.data), name
    for owner, expected in native_partials.data.items():
        np.testing.assert_allclose(
            adapter_partials.data[owner], expected, rtol=1e-11, atol=1e-13,
            err_msg=f"{name} partials of {owner.name}",
        )


def test_distance_terms_skip_exactly_the_native_non_candidates():
    """Pairs with no two points closer than the minimum distance are not evaluated.

    Identical circles and a constant curve on a surface sample have coincident
    points (a zero distance) and the constant curve a zero tangent; with a zero
    minimum distance they are no candidate pair, so value and gradient are 0.
    """
    surface = _surface()
    circle = _circle((1.0, 0.0, 0.0), 1.0)
    twin = _circle((1.0, 0.0, 0.0), 1.0)
    on_sample = _circle(tuple(surface.gamma()[0, 0]), 0.0)
    cases = (
        ("CurveCurveDistance", CurveCurveDistance([circle, twin], 0.0),
         CurveCurveDistanceJAX([circle, twin], 0.0)),
        ("CurveSurfaceDistance", CurveSurfaceDistance([circle, on_sample], surface, 0.0),
         CurveSurfaceDistanceJAX([circle, on_sample], surface, 0.0)),
    )
    for name, native, adapter in cases:
        _assert_matches_native_including_nans(name, native, adapter)
        assert adapter.J() == 0.0 and not np.any(adapter.dJ()), name


def test_distance_terms_evaluate_every_point_of_a_native_candidate():
    """Within a candidate pair every point is evaluated, as native: a zero tangent
    far from the other curve still gives native's non-finite gradient."""
    surface = _surface()
    sample = surface.gamma()[0, 0]
    # Near the sample at t = 1/2 (15 points: t = 7/15), zero tangent at t = 0, far away.
    segment = _segment(sample[0] + 1.1, 1.0)
    ring = _circle((float(sample[0]) + 0.12, 0.0, 0.15), 0.05)
    on_sample = _circle(tuple(sample), 0.0)
    cases = (
        ("CurveCurveDistance", CurveCurveDistance([ring, segment], 0.25),
         CurveCurveDistanceJAX([ring, segment], 0.25)),
        ("CurveSurfaceDistance", CurveSurfaceDistance([segment], surface, 0.25),
         CurveSurfaceDistanceJAX([segment], surface, 0.25)),
        ("CurveSurfaceDistance(coincident)", CurveSurfaceDistance([on_sample], surface, 0.1),
         CurveSurfaceDistanceJAX([on_sample], surface, 0.1)),
    )
    for name, native, adapter in cases:
        assert not np.all(np.isfinite(native.dJ())), f"{name}: the case must be singular natively"
        _assert_matches_native_including_nans(name, native, adapter)


# Minimum distances at which the squared distance of the two points rounds to the
# squared threshold: with and without fused multiply-adds the candidate test
# differs in the last bit (both directions; reviewer reproductions).
_TIE_CASES = (
    ((0.2, 0.5, 0.3), 0.6164414002968976),
    ((0.1, 0.4, 0.1), 0.42426406871192857),
)


@pytest.mark.parametrize("position, minimum_distance", _TIE_CASES, ids=["tie_a", "tie_b"])
def test_distance_candidates_at_the_threshold_are_the_native_ones(position, minimum_distance):
    """At a tie the candidate decision is ``simsoptpp``'s own, so the evaluated set,
    and with it the finite-zero or NaN gradient of these constant curves, is native's."""
    origin_curve = _circle((0.0, 0.0, 0.0), 0.0, numquadpoints=8)
    point_curve = _circle(position, 0.0, numquadpoints=8)
    origin_surface = SurfaceRZFourier.from_nphi_ntheta(nphi=4, ntheta=4)
    origin_surface.local_full_x = np.zeros_like(origin_surface.local_full_x)
    assert not np.any(origin_surface.gamma())
    cases = (
        ("CurveCurveDistance", CurveCurveDistance([origin_curve, point_curve], minimum_distance),
         CurveCurveDistanceJAX([origin_curve, point_curve], minimum_distance)),
        ("CurveSurfaceDistance", CurveSurfaceDistance([point_curve], origin_surface, minimum_distance),
         CurveSurfaceDistanceJAX([point_curve], origin_surface, minimum_distance)),
    )
    for name, native, adapter in cases:
        _assert_matches_native_including_nans(name, native, adapter)


def test_penalties_make_no_implicit_transfers():
    """J and dJ move data only through explicit transfers, also after a DOF change."""
    objectives = _all_objectives(shared_curve=True)
    for _, _, adapter in objectives:
        adapter.J()
        adapter.dJ()
    with jax.transfer_guard("disallow"):
        for _, _, adapter in objectives:
            adapter.J()
            adapter.dJ()
        for _, _, adapter in objectives:
            adapter.x = np.asarray(adapter.x) + 0.01
            adapter.J()
            adapter.dJ()
    for name, native, adapter in objectives:
        _assert_matches_native(name, native, adapter)
