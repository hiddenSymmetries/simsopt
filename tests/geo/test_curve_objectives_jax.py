"""JAX coil-geometry penalties against the native curve objectives."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from core.test_buffer_ownership import _host_array, make_execution_gate
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
    from simsopt_jax.backend import set_backend
    from simsopt_jax.core import curve_kernels
    from simsopt_jax.core.curve_kernels import (
        curve_curve_distance_penalty_pure,
        curve_length_from_incremental_arclength_pure,
        curvature_p_norm_from_kappa_pure,
        distance_candidate_pure,
        kappa_pure,
        mean_squared_curvature_pure,
        curve_surface_distance_penalty_pure,
    )
    from simsopt_jax_adapters.geo import (
        JaxCurveCurveDistance,
        JaxCurveLength,
        JaxCurveSurfaceDistance,
        JaxLpCurveCurvature,
        JaxMeanSquaredCurvature,
    )
from unittest import mock


from collections.abc import Callable
import os
import subprocess
import sys
from typing import cast

import numpy as np


# The thresholds make every penalty active at the test state.
_CC_THRESHOLD = 0.6
_CS_THRESHOLD = 0.4
_CURVATURE_THRESHOLD = 1.0



_DISTANCE_CHILD_SETUP = """
import sys
import jax
from jax._src import xla_bridge
import numpy as np
from simsopt.geo import (
    CurveCurveDistance, CurveSurfaceDistance, SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt_jax.backend import set_backend
from simsopt_jax_adapters.geo import JaxCurveCurveDistance, JaxCurveSurfaceDistance

assert not xla_bridge.backends_are_initialized()
curves = create_equally_spaced_curves(
    2, 1, stellsym=True, R0=1.0, R1=0.5, order=2, numquadpoints=12,
)
surface = SurfaceRZFourier.from_nphi_ntheta(nphi=4, ntheta=5)
surface.set_rc(0, 0, 1.0)
surface.set_rc(1, 0, 0.3)
surface.set_zs(1, 0, 0.3)
if sys.argv[1] == "curve_curve":
    adapter = JaxCurveCurveDistance(curves, 0.6)
    native = CurveCurveDistance(curves, 0.6)
else:
    adapter = JaxCurveSurfaceDistance(curves, surface, 0.4)
    native = CurveSurfaceDistance(curves, surface, 0.4)
"""


def _distance_child_environment():
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("JAX_", "SIMSOPT_"))
    }
    environment.update(
        JAX_PLATFORMS="cpu", JAX_ENABLE_COMPILATION_CACHE="false",
        OMP_NUM_THREADS="8", OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8",
    )
    return environment






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
        ("CurveLength", CurveLength(curve), JaxCurveLength(curve)),
        (
            "LpCurveCurvature",
            LpCurveCurvature(curve, 2, _CURVATURE_THRESHOLD),
            JaxLpCurveCurvature(curve, 2, _CURVATURE_THRESHOLD),
        ),
        ("MeanSquaredCurvature", MeanSquaredCurvature(curve), JaxMeanSquaredCurvature(curve)),
    ]


def _distance_objectives(curves, surface, num_basecurves) -> list[tuple[str, Optimizable, Optimizable]]:
    return [
        (
            "CurveCurveDistance",
            CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves),
            JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves),
        ),
        (
            "CurveCurveDistance(downsample=2)",
            CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves, downsample=2),
            JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves, downsample=2),
        ),
        (
            "CurveSurfaceDistance",
            CurveSurfaceDistance(curves, surface, _CS_THRESHOLD),
            JaxCurveSurfaceDistance(curves, surface, _CS_THRESHOLD),
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


def _assert_matches_native_including_nans(
    name: str, native: Optimizable, adapter: Optimizable, *, active: bool = False,
) -> None:
    """Value, free gradient and partials equal to native, non-finite entries included."""
    native_value = float(native.J())
    if active:
        assert native_value > 0.0, f"{name} must be active at the test state"
    np.testing.assert_allclose(adapter.J(), native_value, rtol=1e-12, atol=1e-14, err_msg=name)
    native_gradient = native.dJ()
    if active:
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


















# Minimum distances at which the squared distance of the two points rounds to the
# squared threshold: with and without fused multiply-adds the candidate test
# differs in the last bit (both directions; reviewer reproductions).
_TIE_CASES = (
    ((0.2, 0.5, 0.3), 0.6164414002968976),
    ((0.1, 0.4, 0.1), 0.42426406871192857),
)






class TestCurveObjectivesJax(JaxTestCase):
    def test_pure_kernels_snapshot_numpy_before_pending_evaluation(self):
        """Caller mutation cannot change queued length, curvature or candidate results."""
        cases = (
            (kappa_pure, ((65536, 3), (65536, 3))),
            (curvature_p_norm_from_kappa_pure, ((65536,), (65536, 3), (), ())),
            (mean_squared_curvature_pure, ((65536,), (65536, 3))),
            (distance_candidate_pure, ((32, 3), (32, 3), ())),
        )
        for kernel, shapes in cases:
            for operand_index in range(len(shapes)):
                for misaligned in (False, True):
                    with self.subTest(kernel=kernel.__name__, operand=operand_index, misaligned=misaligned), self.case():
                        self._case_pure_kernel_snapshot(kernel, shapes, operand_index, misaligned)

    def _case_pure_kernel_snapshot(self, kernel, shapes, operand_index, misaligned):
        """Queue real device work before mutating one host operand of a warm kernel."""
        host_inputs = [_host_array(shape or (1,), misaligned=misaligned).reshape(shape) for shape in shapes]
        if kernel is kappa_pure:
            host_inputs[1][:, 0] *= -1.0
        if kernel is distance_candidate_pure:
            host_inputs[1] += 0.2
        expected = np.asarray(kernel(*host_inputs)).copy()
        inputs: list[np.ndarray | jax.Array] = [jnp.asarray(value) for value in host_inputs]
        inputs[operand_index] = host_inputs[operand_index]
        gate = make_execution_gate()
        gate_index = 1 if operand_index == 0 else 0
        gate(jnp.asarray(host_inputs[gate_index])).block_until_ready()
        pending = gate(jnp.asarray(host_inputs[gate_index]))
        inputs[gate_index] = pending
        result = kernel(*inputs)
        self.assertFalse(result.is_ready(), "pure kernel finished before caller mutation")
        host_inputs[operand_index][...] = 0.0 if kernel is distance_candidate_pure else -7.0
        np.testing.assert_array_equal(result.block_until_ready(), expected)
        pending.block_until_ready()

    def test_length_kernel_snapshots_numpy_before_pending_evaluation(self):
        """Caller mutation cannot change a queued length.

        The length kernel has one operand, so no queued input can hold its evaluation; the
        test wraps its compiled formula so that it also waits for the execution gate."""
        compiled = curve_kernels._curve_length_from_incremental_arclength_pure
        held = jax.jit(lambda speed, ready: jnp.where(ready[0] > 0, compiled(speed), jnp.nan))
        for misaligned in (False, True):
            with self.subTest(misaligned=misaligned), self.case():
                speed = _host_array((65536,), misaligned=misaligned)
                gate = make_execution_gate()
                ready = gate(jnp.ones(1)).block_until_ready()
                expected = np.asarray(held(jnp.asarray(speed.copy()), ready)).copy()
                pending = gate(ready)
                with mock.patch.object(
                    curve_kernels, "_curve_length_from_incremental_arclength_pure", lambda value: held(value, pending)
                ):
                    result = curve_length_from_incremental_arclength_pure(speed)
                self.assertFalse(result.is_ready(), "length finished before caller mutation")
                speed[...] = -7.0
                np.testing.assert_array_equal(result.block_until_ready(), expected)

    def test_pure_kernels_support_nested_jit_and_grad(self):
        """Host boundaries preserve traced device values and differentiable formulas."""
        speed = jnp.arange(1.0, 9.0)
        derivative = jnp.stack((speed, speed * 0.5, speed * 0.25), axis=1)
        second = derivative.at[:, 0].multiply(-1.0)
        calls = (
            (curve_length_from_incremental_arclength_pure, (speed,)),
            (kappa_pure, (derivative, second)),
            (curvature_p_norm_from_kappa_pure, (speed, derivative, 2.0, 0.5)),
            (mean_squared_curvature_pure, (speed, derivative)),
            (distance_candidate_pure, (derivative, second, 0.5)),
        )
        for kernel, inputs in calls:
            with self.subTest(kernel=kernel.__name__):
                np.testing.assert_array_equal(jax.jit(kernel)(*inputs), kernel(*inputs))
                if kernel is not distance_candidate_pure:
                    gradient = jax.grad(lambda value: jnp.sum(kernel(value, *inputs[1:])))
                    np.testing.assert_allclose(jax.jit(gradient)(inputs[0]), gradient(inputs[0]), rtol=1e-14, atol=1e-14)

    def test_distance_kernels_snapshot_numpy_before_pending_evaluation(self):
        """Mutating each caller-owned geometry array cannot change a queued distance penalty."""
        for misaligned in [False, True]:
            for case_id_1, operand_index in zip(["positions1", "weights1", "positions2", "weights2"], range(4), strict=True):
                for case_id_2, kernel in zip(["curve_curve", "curve_surface"], [
    curve_curve_distance_penalty_pure, curve_surface_distance_penalty_pure,
], strict=True):
                    with self.subTest(misaligned=misaligned, case_id_1=case_id_1, operand_index=operand_index, case_id_2=case_id_2, kernel=kernel), self.case() as patches:
                        self._case_distance_kernels_snapshot_numpy_before_pending_evaluation(misaligned, operand_index, kernel, patches)

    def _case_distance_kernels_snapshot_numpy_before_pending_evaluation(self, misaligned, operand_index, kernel, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        execution_gate = make_execution_gate()
        host_inputs = [
            _host_array((32, 3), misaligned=misaligned) for _ in range(4)
        ]
        host_inputs[2] += 0.2
        inputs: list[np.ndarray | jax.Array] = list(host_inputs)
        expected = np.asarray(kernel(*inputs, 0.5, True)).copy()
        self.assertTrue(expected > 0, 'expected > 0')
        gate_index = 1 if operand_index == 0 else 0
        gate = execution_gate
        for index in range(4):
            if index != operand_index:
                inputs[index] = jnp.asarray(inputs[index])
        gate(inputs[gate_index]).block_until_ready()
        inputs[gate_index] = gate(inputs[gate_index])
        result = kernel(*inputs, 0.5, True)
        self.assertTrue(not result.is_ready(), "distance evaluation finished before caller mutation")
        host_inputs[operand_index][:] = -7.0
        np.testing.assert_array_equal(result.block_until_ready(), expected)

    def test_shortest_distance_before_backend_configuration_keeps_jax_uninitialized(self):
        """Shortest-distance queries match native without initializing a JAX backend."""
        for kind in ["curve_curve", "curve_surface"]:
            with self.subTest(kind=kind), self.case() as patches:
                self._case_shortest_distance_before_backend_configuration_keeps_jax_uninitialized(kind, patches)

    def _case_shortest_distance_before_backend_configuration_keeps_jax_uninitialized(self, kind, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        code = _DISTANCE_CHILD_SETUP + """
assert not xla_bridge.backends_are_initialized()
for _ in range(2):
    np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
assert not xla_bridge.backends_are_initialized(), "host shortest_distance initialized JAX"
set_backend("jax", device="cpu", intent="parity")
np.testing.assert_allclose(adapter.J(), native.J(), rtol=1e-12, atol=1e-14)
np.testing.assert_allclose(adapter.dJ(), native.dJ(), rtol=1e-11, atol=1e-13)
"""
        completed = subprocess.run(
            (sys.executable, "-c", code, kind), env=_distance_child_environment(),
            capture_output=True, text=True, timeout=300,
        )
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_distance_operand_cache_follows_default_device_scopes(self):
        """Changing default devices replaces operands while reusing host candidates."""
        for kind in ["curve_curve", "curve_surface"]:
            with self.subTest(kind=kind), self.case() as patches:
                self._case_distance_operand_cache_follows_default_device_scopes(kind, patches)

    def _case_distance_operand_cache_follows_default_device_scopes(self, kind, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        code = _DISTANCE_CHILD_SETUP + """
set_backend("jax", device="cpu", intent="parity")
first, second = jax.devices("cpu")[:2]
searches = []
search = adapter._candidates

def counted_search(samples):
    searches.append(None)
    return search(samples)

adapter._candidates = counted_search
for device in (first, second, first):
    with jax.default_device(device):
        _, operands = adapter._operands()
        _, reused = adapter._operands()
        assert reused is operands, "same-scope operands should be reused"
        assert all(array.devices() == {device} for array in jax.tree.leaves(operands)), device
        with jax.transfer_guard("disallow"):
            value, gradient = adapter.J(), adapter.dJ()
            shortest = adapter.shortest_distance()
    np.testing.assert_allclose(value, native.J(), rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(gradient, native.dJ(), rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(shortest, native.shortest_distance(), rtol=1e-14)
assert len(searches) == 1, "placement changes must reuse host candidates"
"""
        environment = _distance_child_environment()
        environment["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
        completed = subprocess.run(
            (sys.executable, "-c", code, kind), env=environment,
            capture_output=True, text=True, timeout=300,
        )
        self.assertTrue(completed.returncode == 0, completed.stdout + completed.stderr)

    def test_penalties_match_native_values_gradients_and_partials(self):
        """Geometry penalties match native values, free gradients and fixed partials."""
        for case_id_0, shared_curve in zip(["symmetric_copies", "shared_dofs_twin"], [False, True], strict=True):
            with self.subTest(case_id_0=case_id_0, shared_curve=shared_curve), self.case() as patches:
                self._case_penalties_match_native_values_gradients_and_partials(shared_curve, patches)

    def _case_penalties_match_native_values_gradients_and_partials(self, shared_curve, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        for name, native, adapter in _all_objectives(shared_curve=shared_curve):
            self.assertTrue(adapter.dof_names == native.dof_names, name)
            _assert_matches_native_including_nans(name, native, adapter, active=True)

    def test_penalty_gradients_match_central_differences(self):
        """Geometry penalty gradients match independent central differences."""
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

    def test_penalties_follow_dof_changes_like_native(self):
        """Geometry penalties follow shared DOF changes like their native counterparts."""
        for name, native, adapter in _all_objectives():
            adapter.x = np.asarray(adapter.x) + 0.01
            _assert_matches_native_including_nans(name, native, adapter, active=True)

    def test_curve_surface_distance_depends_on_curves_only(self):
        """Curve-surface distance exposes curve dependencies and excludes surface DOFs."""
        base, curves = _curves()
        surface = _surface()
        native = CurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
        adapter = JaxCurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
        surface_names = set(surface.dof_names)
        self.assertTrue(adapter.dof_names == native.dof_names, 'adapter.dof_names == native.dof_names')
        self.assertTrue(surface_names and not surface_names & set(adapter.dof_names), 'surface_names and not surface_names & set(adapter.dof_names)')
        self.assertTrue(all(owner is not surface for owner in _partials(adapter).data), 'all(owner is not surface for owner in _partials(adapter).data)')

    def test_shortest_distances_match_native(self):
        """Native scans all pairs only when no selected pair is a candidate.

            The close pair (2, 1) is outside the pairs ``j < num_basecurves = 1``: it
            sets the shortest distance only when the selected pairs are all far.
            """
        for case_id_0, threshold in zip(["no_candidates", "candidates"], [0.1, 5.0], strict=True):
            with self.subTest(case_id_0=case_id_0, threshold=threshold), self.case() as patches:
                self._case_shortest_distances_match_native(threshold, patches)

    def _case_shortest_distances_match_native(self, threshold, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        curves = [_circle((0.0, 0.0, 0.0), 1.0), _circle((5.0, 0.0, 0.0), 1.0),
                  _circle((5.0, 0.0, 0.01), 1.0)]
        surface = _surface()
        pairs = (
            (CurveCurveDistance(curves, threshold, num_basecurves=1),
             JaxCurveCurveDistance(curves, threshold, num_basecurves=1)),
            (CurveSurfaceDistance(curves, surface, threshold),
             JaxCurveSurfaceDistance(curves, surface, threshold)),
        )
        for native, adapter in pairs:
            np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
        expected_curve_distance = 0.01 if threshold == 0.1 else pairs[0][0].shortest_distance()
        np.testing.assert_allclose(pairs[0][1].shortest_distance(), expected_curve_distance, rtol=1e-12)
        self.assertTrue(threshold == 0.1 or pairs[0][1].shortest_distance() > 1.0, 'threshold == 0.1 or pairs[0][1].shortest_distance() > 1.0')

    def test_curve_curve_distance_follows_num_basecurves(self):
        """The base-curve count selects the same curve pairs as native."""
        base, curves = _curves()
        native = CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=1)
        adapter = JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=1)
        _assert_matches_native_including_nans("num_basecurves=1", native, adapter, active=True)
        value = adapter.J()
        native.num_basecurves = adapter.num_basecurves = len(base)
        # Native refreshes its candidate pairs only after a DOF change.
        native.recompute_bell()
        _assert_matches_native_including_nans("num_basecurves=3", native, adapter, active=True)
        self.assertTrue(adapter.J() > value, 'adapter.J() > value')

    def test_distance_evaluations_share_one_native_candidate_search(self):
        """Value, gradient and shortest-distance calls reuse one native search."""
        for case_id_0, objective_index in zip(["curve_curve", "downsampled", "curve_surface"], [0, 1, 2], strict=True):
            with self.subTest(case_id_0=case_id_0, objective_index=objective_index), self.case() as patches:
                self._case_distance_evaluations_share_one_native_candidate_search(objective_index, patches)

    def _case_distance_evaluations_share_one_native_candidate_search(self, objective_index, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        base, curves = _curves()
        name, native, adapter = _distance_objectives(curves, _surface(), len(base))[objective_index]
        searches = []
        search = adapter._candidates

        def counted_search(samples):
            searches.append(None)
            return search(samples)

        patches.enter_context(mock.patch.object(adapter, "_candidates", counted_search))
        for _ in range(2):
            _assert_matches_native_including_nans(name, native, adapter, active=True)
            np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
        self.assertTrue(len(searches) == 1, 'len(searches) == 1')

    def test_curve_curve_snapshot_follows_geometry_parameters_and_placement(self):
        """Geometry, distance settings and placement changes invalidate curve operands."""
        for change in ["free_dofs", "fixed_dofs", "threshold", "downsample", "base_count", "grid", "backend"]:
            with self.subTest(change=change), self.case() as patches:
                self._case_curve_curve_snapshot_follows_geometry_parameters_and_placement(change, patches)

    def _case_curve_curve_snapshot_follows_geometry_parameters_and_placement(self, change, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        base, curves = _curves()
        adapter = JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=len(base))
        adapter.J()
        searches = []
        search = adapter._candidates

        def counted_search(samples):
            searches.append(None)
            return search(samples)

        patches.enter_context(mock.patch.object(adapter, "_candidates", counted_search))
        if change == "free_dofs":
            base[0].x = np.asarray(base[0].x) + 0.01
        elif change == "fixed_dofs":
            base[0].set("xc(0)", base[0].get("xc(0)") + 0.01)
        elif change == "threshold":
            adapter.minimum_distance = 0.5
        elif change == "downsample":
            adapter.downsample = 3
        elif change == "base_count":
            adapter.num_basecurves = 1
        elif change == "grid":
            curves[0] = CurveXYZFourier(np.linspace(0, 1, 30, endpoint=False) + 0.013, 3, dofs=base[0].dofs)
        else:
            set_backend("jax", device="cpu", intent="fast")
        value, gradient, shortest = adapter.J(), adapter.dJ(), adapter.shortest_distance()
        self.assertTrue(len(searches) == (0 if change == "backend" else 1), 'len(searches) == (0 if change == "backend" else 1)')
        fresh = JaxCurveCurveDistance(
            curves, adapter.minimum_distance, adapter.num_basecurves, adapter.downsample,
        )
        np.testing.assert_array_equal(value, fresh.J())
        # A stale cache would be off by O(1); GPU reductions are not bitwise run to run, so the
        # gradient is compared at a few ulp.
        np.testing.assert_allclose(gradient, fresh.dJ(), rtol=1e-13, atol=0)
        np.testing.assert_array_equal(shortest, fresh.shortest_distance())

    def test_curve_surface_snapshot_follows_surface_and_operand_changes(self):
        """Surface geometry and operand changes invalidate the curve-surface snapshot."""
        for change in ["surface_dofs", "surface_grid", "normal_only", "tangent_only", "threshold"]:
            with self.subTest(change=change), self.case() as patches:
                self._case_curve_surface_snapshot_follows_surface_and_operand_changes(change, patches)

    def _case_curve_surface_snapshot_follows_surface_and_operand_changes(self, change, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        _, curves = _curves()
        surface = _surface()
        adapter = JaxCurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
        adapter.J()
        searches = []
        search = adapter._candidates

        def counted_search(samples):
            searches.append(None)
            return search(samples)

        patches.enter_context(mock.patch.object(adapter, "_candidates", counted_search))
        if change == "surface_dofs":
            surface.set_rc(1, 0, 0.35)
        elif change == "surface_grid":
            adapter.surface = SurfaceRZFourier.from_nphi_ntheta(nphi=9, ntheta=10, nfp=2, range="half period")
            adapter.surface.local_full_x = surface.local_full_x
        elif change == "normal_only":
            normal = surface.normal().copy() * 1.01
            patches.enter_context(mock.patch.object(surface, "normal", lambda: normal))
        elif change == "tangent_only":
            tangent = curves[0].gammadash().copy() * 1.01
            patches.enter_context(mock.patch.object(curves[0], "gammadash", lambda: tangent))
        else:
            adapter.minimum_distance = 0.3
        value, gradient, shortest = adapter.J(), adapter.dJ(), adapter.shortest_distance()
        self.assertTrue(len(searches) == (0 if change in ("normal_only", "tangent_only") else 1), 'len(searches) == (0 if change in ("normal_only", "tangent_only") else 1)')
        fresh = JaxCurveSurfaceDistance(curves, adapter.surface, adapter.minimum_distance)
        np.testing.assert_array_equal(value, fresh.J())
        # A stale cache would be off by O(1); GPU reductions are not bitwise run to run, so the
        # gradient is compared at a few ulp.
        np.testing.assert_allclose(gradient, fresh.dJ(), rtol=1e-13, atol=0)
        np.testing.assert_array_equal(shortest, fresh.shortest_distance())

    def test_inactive_distance_terms_have_finite_zero_gradients(self):
        """A far curve with zero tangents adds J = 0 and a zero, finite gradient.

            The native objectives skip such curves; the dense JAX kernels evaluate them.
            """
        base, curves = _curves()
        point = _circle((50.0, 0.0, 0.0), 0.0)
        curves = [*curves, point]
        surface = _surface()
        for name, native, adapter in _distance_objectives(curves, surface, len(base)):
            _assert_matches_native_including_nans(name, native, adapter, active=True)
            point_gradient = cast(np.ndarray, _partials(adapter)(point))
            self.assertTrue(np.all(np.isfinite(point_gradient)) and not np.any(point_gradient), name)

    def test_distance_terms_skip_exactly_the_native_non_candidates(self):
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
             JaxCurveCurveDistance([circle, twin], 0.0)),
            ("CurveSurfaceDistance", CurveSurfaceDistance([circle, on_sample], surface, 0.0),
             JaxCurveSurfaceDistance([circle, on_sample], surface, 0.0)),
        )
        for name, native, adapter in cases:
            _assert_matches_native_including_nans(name, native, adapter)
            self.assertTrue(adapter.J() == 0.0 and not np.any(adapter.dJ()), name)

    def test_distance_terms_evaluate_every_point_of_a_native_candidate(self):
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
             JaxCurveCurveDistance([ring, segment], 0.25)),
            ("CurveSurfaceDistance", CurveSurfaceDistance([segment], surface, 0.25),
             JaxCurveSurfaceDistance([segment], surface, 0.25)),
            ("CurveSurfaceDistance(coincident)", CurveSurfaceDistance([on_sample], surface, 0.1),
             JaxCurveSurfaceDistance([on_sample], surface, 0.1)),
        )
        for name, native, adapter in cases:
            self.assertTrue(not np.all(np.isfinite(native.dJ())), f"{name}: the case must be singular natively")
            _assert_matches_native_including_nans(name, native, adapter)

    def test_distance_candidates_at_the_threshold_are_the_native_ones(self):
        """At a tie the candidate decision is ``simsoptpp``'s own, so the evaluated set,
            and with it the finite-zero or NaN gradient of these constant curves, is native's."""
        for case_id_0, (position, minimum_distance) in zip(["tie_a", "tie_b"], _TIE_CASES, strict=True):
            with self.subTest(case_id_0=case_id_0, position=position, minimum_distance=minimum_distance), self.case() as patches:
                self._case_distance_candidates_at_the_threshold_are_the_native_ones(position, minimum_distance, patches)

    def _case_distance_candidates_at_the_threshold_are_the_native_ones(self, position, minimum_distance, patches):
        """Run one row with case-local objects released before runtime cleanup."""
        origin_curve = _circle((0.0, 0.0, 0.0), 0.0, numquadpoints=8)
        point_curve = _circle(position, 0.0, numquadpoints=8)
        origin_surface = SurfaceRZFourier.from_nphi_ntheta(nphi=4, ntheta=4)
        origin_surface.local_full_x = np.zeros_like(origin_surface.local_full_x)
        self.assertTrue(not np.any(origin_surface.gamma()), 'not np.any(origin_surface.gamma())')
        cases = (
            ("CurveCurveDistance", CurveCurveDistance([origin_curve, point_curve], minimum_distance),
             JaxCurveCurveDistance([origin_curve, point_curve], minimum_distance)),
            ("CurveSurfaceDistance", CurveSurfaceDistance([point_curve], origin_surface, minimum_distance),
             JaxCurveSurfaceDistance([point_curve], origin_surface, minimum_distance)),
        )
        for name, native, adapter in cases:
            _assert_matches_native_including_nans(name, native, adapter)

    def test_penalties_make_no_implicit_transfers(self):
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
            _assert_matches_native_including_nans(name, native, adapter, active=True)

    def test_penalty_gradients_taylor(self):
        """Active geometry penalties attain native shrinking-step convergence rates."""
        for index in range(6):
            with self.subTest(objective_index=index), self.case():
                self._case_penalty_gradients_taylor(index)

    def _case_penalty_gradients_taylor(self, index):
        """Use the native stencil, step interval and ratio bound for each penalty."""
        name, _, adapter = _all_objectives(shared_curve=True)[index]
        x0 = np.array(adapter.x, dtype=float)
        centered = index == 2
        scale = 1e-1 if centered else (1e-2 if index == 1 else 1e-3)
        direction = scale * np.random.RandomState(1).uniform(size=x0.shape)
        derivative = adapter.dJ() @ direction
        initial = adapter.J()
        self.assertGreater(abs(derivative), 1e-10, name)
        previous_error = 1e6
        stop = 10 if centered else (12 if index >= 3 else 15)
        bound = 0.3 if centered else (0.6 if index >= 3 else 0.55)
        for exponent in range(5, stop):
            step = 0.5 ** exponent
            adapter.x = x0 + step * direction
            plus = adapter.J()
            if centered:
                adapter.x = x0 - step * direction
                estimate = (plus - adapter.J()) / (2 * step)
            else:
                estimate = (plus - initial) / step
            error = abs(estimate - derivative)
            self.assertLess(error, bound * previous_error, f"{name}, step={step}")
            previous_error = error
        adapter.x = x0
