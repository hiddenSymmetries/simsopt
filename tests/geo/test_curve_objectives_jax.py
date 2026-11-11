"""JAX coil-geometry penalties against the native curve objectives."""

from __future__ import annotations

import itertools
import os
from functools import partial
from typing import cast
from unittest import mock

import numpy as np
from unittest_jax_support import JaxTestCase, optional_jax_imports, run_fresh_process
from unittest_jax_support import DISTANCE_TIE_CASES, assert_matches_native_including_nans, circle_curve
from unittest_jax_support import distinct_host_array, make_execution_gate, objective_partials, segment_curve

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.numpy as jnp
    from jax.extend.core import jaxprs_in_params
    from simsopt._core.optimizable import Optimizable
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import CurveCurveDistance, CurveLength, CurveSurfaceDistance, CurveXYZFourier, LpCurveCurvature
    from simsopt.geo import MeanSquaredCurvature, SurfaceRZFourier, create_equally_spaced_curves
    from simsopt_jax.backend import set_backend
    from simsopt_jax.backend.jit import jit
    from simsopt_jax.core import curve_kernels
    from simsopt_jax.core.curve_kernels import curve_curve_distance_penalty_pure, curve_surface_distance_penalty_pure
    from simsopt_jax.core.curve_kernels import curvature_p_norm_from_kappa_pure, distance_candidate_pure, kappa_pure
    from simsopt_jax.core.curve_kernels import curve_length_from_incremental_arclength_pure, mean_squared_curvature_pure
    from simsopt_jax_adapters.geo import JaxCurveCurveDistance, JaxCurveSurfaceDistance, curve_objectives
    from simsopt_jax_adapters.geo import JaxCurveLength, JaxLpCurveCurvature, JaxMeanSquaredCurvature


# The thresholds make every penalty active at the test state.
_CC_THRESHOLD, _CS_THRESHOLD, _CURVATURE_THRESHOLD = 0.6, 0.4, 1.0
_DISTANCE_CHILD_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fresh_process",
                                      "distance_adapters.py")


def _run_child(scenario, kind, **environment):
    """Run one scenario of the fresh-process child script, without the parent's JAX variables."""
    env = {name: value for name, value in os.environ.items() if not name.startswith("JAX_")}
    env.update(JAX_PLATFORMS="cpu", JAX_ENABLE_COMPILATION_CACHE="false", OMP_NUM_THREADS="8",
               OPENBLAS_NUM_THREADS="8", MKL_NUM_THREADS="8", **environment)
    return run_fresh_process(_DISTANCE_CHILD_SCRIPT, scenario, kind, env=env, timeout=300)


def _surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_nphi_ntheta(nphi=8, ntheta=8, nfp=2, range="half period")
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    return surface


def _curves(*, shared_curve: bool = False) -> tuple[list, list]:
    """Base curves, some DOFs fixed, and all symmetric copies; ``shared_curve`` adds a curve sharing the second base
    curve's DOFs on its own grid, offset so that no point coincides with one of that curve (no gradient at 0)."""
    base = create_equally_spaced_curves(3, 2, stellsym=True, R0=1.0, R1=0.5, order=3, numquadpoints=24)
    rng = np.random.default_rng(7)
    for curve in base:
        curve.x = curve.x + 0.03 * rng.standard_normal(curve.x.shape)
    base[0].fix("xc(0)")
    base[0].fix("zs(1)")
    curves = [coil.curve for coil in coils_via_symmetries(base, [Current(1.0) for _ in base], 2, True)]
    if shared_curve:
        curves.append(CurveXYZFourier(np.linspace(0, 1, 30, endpoint=False) + 0.013, 3, dofs=base[1].dofs))
    return base, curves


def _single_curve_objectives(curve) -> list[tuple[str, Optimizable, Optimizable]]:
    return [
        ("CurveLength", CurveLength(curve), JaxCurveLength(curve)),
        ("LpCurveCurvature", LpCurveCurvature(curve, 2, _CURVATURE_THRESHOLD),
         JaxLpCurveCurvature(curve, 2, _CURVATURE_THRESHOLD)),
        ("MeanSquaredCurvature", MeanSquaredCurvature(curve), JaxMeanSquaredCurvature(curve)),
    ]


def _distance_objectives(curves, surface, num_basecurves) -> list[tuple[str, Optimizable, Optimizable]]:
    return [
        ("CurveCurveDistance", CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves),
         JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves)),
        ("CurveCurveDistance(downsample=2)",
         CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves, downsample=2),
         JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=num_basecurves, downsample=2)),
        ("CurveSurfaceDistance", CurveSurfaceDistance(curves, surface, _CS_THRESHOLD),
         JaxCurveSurfaceDistance(curves, surface, _CS_THRESHOLD)),
    ]


def _all_objectives(*, shared_curve: bool = False):
    base, curves = _curves(shared_curve=shared_curve)
    return _single_curve_objectives(base[0]) + _distance_objectives(curves, _surface(), len(base))


def _separated_circles(num_far, twins):
    """Circles of radius 0.2 in the xy-plane, centred 1 m apart (0.6 m apart) on the x-axis, then twins: twin ``k``,
    an (x, quadrature count) pair, is a circle at ``x`` raised 0.05 m, so only twin-circle pairs lie within 0.1 m;
    ``num_far`` more circles follow on the x-axis beyond 100 m."""
    circles = [circle_curve((float(x), 0.0, 0.0), 0.2) for x, _ in twins]
    circles += [circle_curve((float(x), 0.0, 0.05), 0.2, numquadpoints=count) for x, count in twins]
    circles += [circle_curve((100.0 + k, 0.0, 0.0), 0.2) for k in range(num_far)]
    return circles


def _candidate_query_objectives(base, curves, surface, threshold):
    """Native and adapter curve-curve (``downsample=2``) and curve-surface objectives."""
    return (
        (CurveCurveDistance(curves, threshold, num_basecurves=len(base), downsample=2),
         JaxCurveCurveDistance(curves, threshold, num_basecurves=len(base), downsample=2)),
        (CurveSurfaceDistance(curves, surface, threshold), JaxCurveSurfaceDistance(curves, surface, threshold)),
    )


def _native_candidates(native):
    """A native objective's computed candidates as sorted index tuples."""
    return sorted(map(tuple, cast(list, native.candidates)))


def _point_pair_distance_shapes(adapter):
    """Sorted shapes of the batched point-pair distance arrays the adapter's program computes."""
    snapshot = adapter._snapshot()
    if isinstance(adapter, JaxCurveCurveDistance):
        total = partial(curve_objectives._curve_pair_penalty_total, batch_classes=snapshot.batch_classes,
                        downsample=adapter.downsample)
    else:
        total = partial(curve_objectives._curve_surface_penalty_total, batch_classes=snapshot.batch_classes)

    def shapes(jaxpr):
        for equation in jaxpr.eqns:
            if equation.primitive.name == "sqrt" and equation.outvars[0].aval.ndim == 3:
                yield equation.outvars[0].aval.shape
            for inner in jaxprs_in_params(equation.params):
                yield from shapes(inner)

    return sorted(shapes(jax.make_jaxpr(total)(*snapshot.operands).jaxpr))


# Native Taylor test of each penalty: (centered stencil, direction scale, stop exponent, error ratio bound).
_TAYLOR_SETTINGS = {
    "CurveLength": (False, 1e-3, 15, 0.55),
    "LpCurveCurvature": (False, 1e-2, 15, 0.55),
    "MeanSquaredCurvature": (True, 1e-1, 10, 0.3),
    "CurveCurveDistance": (False, 1e-3, 12, 0.6),
    "CurveCurveDistance(downsample=2)": (False, 1e-3, 12, 0.6),
    "CurveSurfaceDistance": (False, 1e-3, 12, 0.6),
}


class TestCurveObjectivesJax(JaxTestCase):
    """JAX length, curvature and distance penalties against the native curve objectives."""

    def test_pure_kernels_snapshot_numpy_before_pending_evaluation(self):
        """Caller mutation cannot change queued length, curvature or candidate results."""
        cases = ((kappa_pure, ((65536, 3), (65536, 3))),
                 (curvature_p_norm_from_kappa_pure, ((65536,), (65536, 3), (), ())),
                 (mean_squared_curvature_pure, ((65536,), (65536, 3))),
                 (distance_candidate_pure, ((32, 3), (32, 3), ())))
        for kernel, shapes in cases:
            for operand_index, misaligned in itertools.product(range(len(shapes)), (False, True)):
                with self.subTest(kernel=kernel.__name__, operand=operand_index, misaligned=misaligned), self.case():
                    self._case_pure_kernel_snapshot(kernel, shapes, operand_index, misaligned)

    def _case_pure_kernel_snapshot(self, kernel, shapes, operand_index, misaligned):
        """Queue real device work before mutating one host operand of a warm kernel."""
        host_inputs = [distinct_host_array(shape or (1,), misaligned=misaligned).reshape(shape) for shape in shapes]
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
        """Caller mutation cannot change a queued length; with one operand, no queued input can hold the evaluation,
        so the test wraps the compiled formula to also wait for the execution gate."""
        compiled = curve_kernels._curve_length_from_incremental_arclength_pure
        # No copy, so only the kernel boundary owns the buffer.
        held = jax.jit(lambda speed, ready: jnp.where(ready[0] > 0, compiled(speed), jnp.nan))  # noqa: TID251
        for misaligned in (False, True):
            with self.subTest(misaligned=misaligned), self.case():
                speed = distinct_host_array((65536,), misaligned=misaligned)
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
        calls = ((curve_length_from_incremental_arclength_pure, (speed,)), (kappa_pure, (derivative, second)),
                 (curvature_p_norm_from_kappa_pure, (speed, derivative, 2.0, 0.5)),
                 (mean_squared_curvature_pure, (speed, derivative)),
                 (distance_candidate_pure, (derivative, second, 0.5)))
        for kernel, inputs in calls:
            with self.subTest(kernel=kernel.__name__):
                np.testing.assert_array_equal(jit(kernel)(*inputs), kernel(*inputs))
                if kernel is not distance_candidate_pure:
                    gradient = jax.grad(lambda value: jnp.sum(kernel(value, *inputs[1:])))
                    np.testing.assert_allclose(jit(gradient)(inputs[0]), gradient(inputs[0]), rtol=1e-14, atol=1e-14)

    def test_pure_kernels_evaluate_float32_inputs_in_double_like_native(self):
        """float32 geometry is evaluated in float64, as native does; the scales make a float32 evaluation overflow (max
        3.4e38): |gamma'|^3 at |gamma'| ~ 1e13, the sum of speeds ~ 1e38, and kappa^2 at kappa ~ 1e20."""
        rng = np.random.default_rng(9)
        speed = 1e38 * (1.0 + rng.random(16))
        tangent = 1e13 * rng.standard_normal((16, 3))
        normal = 1e13 * rng.standard_normal((16, 3))
        kappa = 1e20 * (1.0 + rng.random(16))
        cases = ((curve_length_from_incremental_arclength_pure, (speed,)), (kappa_pure, (tangent, normal)),
                 (curvature_p_norm_from_kappa_pure, (kappa, tangent, 2.0, 1e19)),
                 (mean_squared_curvature_pure, (kappa, tangent)))
        for kernel, inputs in cases:
            with self.subTest(kernel=kernel.__name__):
                single = tuple(np.asarray(value, dtype=np.float32) for value in inputs)
                expected = np.asarray(kernel(*(value.astype(np.float64) for value in single)))
                actual = kernel(*single)
                self.assertEqual(actual.dtype, np.float64)
                self.assertTrue(np.all(np.isfinite(expected)))
                np.testing.assert_array_equal(np.asarray(actual), expected)

    def test_distance_kernels_snapshot_numpy_before_pending_evaluation(self):
        """Mutating each caller-owned geometry array cannot change a queued distance penalty."""
        kernels = (curve_curve_distance_penalty_pure, curve_surface_distance_penalty_pure)
        self.run_cases(self._case_distance_kernels_snapshot_numpy_before_pending_evaluation, [
            dict(misaligned=misaligned, operand_index=operand_index, kernel=kernel)
            for misaligned in (False, True) for operand_index in range(4) for kernel in kernels])

    def _case_distance_kernels_snapshot_numpy_before_pending_evaluation(self, misaligned, operand_index, kernel):
        gate = make_execution_gate()
        host_inputs = [distinct_host_array((32, 3), misaligned=misaligned) for _ in range(4)]
        host_inputs[2] += 0.2
        inputs: list[np.ndarray | jax.Array] = list(host_inputs)
        expected = np.asarray(kernel(*inputs, 0.5, True)).copy()
        self.assertGreater(expected, 0)
        gate_index = 1 if operand_index == 0 else 0
        for index in range(4):
            if index != operand_index:
                inputs[index] = jnp.asarray(inputs[index])
        gate(inputs[gate_index]).block_until_ready()
        inputs[gate_index] = gate(inputs[gate_index])
        result = kernel(*inputs, 0.5, True)
        self.assertFalse(result.is_ready(), "distance evaluation finished before caller mutation")
        host_inputs[operand_index][:] = -7.0
        np.testing.assert_array_equal(result.block_until_ready(), expected)

    def test_shortest_distance_before_backend_configuration_keeps_jax_uninitialized(self):
        """Shortest-distance queries match native without initializing a JAX backend."""
        for kind in ["curve_curve", "curve_surface"]:
            with self.subTest(kind=kind), self.case():
                completed = _run_child("uninitialized", kind)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_distance_operand_cache_follows_default_device_scopes(self):
        """Changing default devices replaces operands while reusing host candidates."""
        for kind in ["curve_curve", "curve_surface"]:
            with self.subTest(kind=kind), self.case():
                completed = _run_child("device_scopes", kind, XLA_FLAGS="--xla_force_host_platform_device_count=2")
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_penalties_match_native_values_gradients_and_partials(self):
        """Geometry penalties match native values, free gradients and fixed partials."""
        self.run_cases(self._case_penalties_match_native_values_gradients_and_partials,
                       [dict(shared_curve=False), dict(shared_curve=True)])

    def _case_penalties_match_native_values_gradients_and_partials(self, shared_curve):
        for name, native, adapter in _all_objectives(shared_curve=shared_curve):
            self.assertEqual(adapter.dof_names, native.dof_names, name)
            assert_matches_native_including_nans(name, native, adapter, active=True)

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
            np.testing.assert_allclose(adapter.dJ() @ direction, (plus - minus) / (2 * step), rtol=1e-6, err_msg=name)

    def test_penalties_follow_dof_changes_like_native(self):
        """Geometry penalties follow shared DOF changes like their native counterparts."""
        for name, native, adapter in _all_objectives():
            adapter.x = np.asarray(adapter.x) + 0.01
            assert_matches_native_including_nans(name, native, adapter, active=True)

    def test_curve_surface_distance_depends_on_curves_only(self):
        """Curve-surface distance exposes curve dependencies and excludes surface DOFs."""
        _, curves = _curves()
        surface = _surface()
        native = CurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
        adapter = JaxCurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
        surface_names = set(surface.dof_names)
        self.assertEqual(adapter.dof_names, native.dof_names)
        self.assertTrue(surface_names)
        self.assertFalse(surface_names & set(adapter.dof_names))
        self.assertTrue(all(owner is not surface for owner in objective_partials(adapter).data))

    def test_shortest_distances_match_native(self):
        """Native scans all pairs only when no selected pair is a candidate: the close pair (2, 1) is outside the pairs
        ``j < num_basecurves = 1`` and sets the shortest distance only when the selected pairs are all far."""
        self.run_cases(self._case_shortest_distances_match_native, [dict(threshold=0.1), dict(threshold=5.0)])

    def _case_shortest_distances_match_native(self, threshold):
        curves = [circle_curve((0.0, 0.0, 0.0), 1.0), circle_curve((5.0, 0.0, 0.0), 1.0),
                  circle_curve((5.0, 0.0, 0.01), 1.0)]
        surface = _surface()
        pairs = (
            (CurveCurveDistance(curves, threshold, num_basecurves=1),
             JaxCurveCurveDistance(curves, threshold, num_basecurves=1)),
            (CurveSurfaceDistance(curves, surface, threshold), JaxCurveSurfaceDistance(curves, surface, threshold)),
        )
        for native, adapter in pairs:
            np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
        expected_curve_distance = 0.01 if threshold == 0.1 else pairs[0][0].shortest_distance()
        np.testing.assert_allclose(pairs[0][1].shortest_distance(), expected_curve_distance, rtol=1e-12)
        if threshold != 0.1:
            self.assertGreater(pairs[0][1].shortest_distance(), 1.0)

    def test_curve_curve_distance_follows_num_basecurves(self):
        """The base-curve count selects the same curve pairs as native."""
        base, curves = _curves()
        native = CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=1)
        adapter = JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=1)
        assert_matches_native_including_nans("num_basecurves=1", native, adapter, active=True)
        value = adapter.J()
        native.num_basecurves = adapter.num_basecurves = len(base)
        # Native refreshes its candidate pairs only after a DOF change.
        native.recompute_bell()
        assert_matches_native_including_nans("num_basecurves=3", native, adapter, active=True)
        self.assertGreater(adapter.J(), value)

    def test_downsampled_distance_of_grids_with_one_sampled_count_matches_native(self):
        """Grids of 24 and 23 points both give 12 samples at stride 2 and gradients reach every point: the adapter
        stacks full grids and samples inside the differentiated program, so the two grids are separate classes."""
        base, curves = _curves()
        odd_twin = CurveXYZFourier(np.linspace(0, 1, 23, endpoint=False) + 0.013, 3, dofs=base[1].dofs)
        curves = curves + [odd_twin]
        native = CurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=len(base), downsample=2)
        adapter = JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=len(base), downsample=2)
        assert_matches_native_including_nans("downsample=2 with 24- and 23-point grids", native, adapter, active=True)

    def test_distance_evaluations_share_one_native_candidate_search(self):
        """Value, gradient and shortest-distance calls reuse one native search."""
        for case_id_0, objective_index in zip(["curve_curve", "downsampled", "curve_surface"], [0, 1, 2], strict=True):
            with self.subTest(case_id_0=case_id_0, objective_index=objective_index), self.case() as patches:
                self._case_distance_evaluations_share_one_native_candidate_search(objective_index, patches)

    def _case_distance_evaluations_share_one_native_candidate_search(self, objective_index, patches):
        base, curves = _curves()
        name, native, adapter = _distance_objectives(curves, _surface(), len(base))[objective_index]
        search = patches.enter_context(mock.patch.object(adapter, "_candidates", wraps=adapter._candidates))
        for _ in range(2):
            assert_matches_native_including_nans(name, native, adapter, active=True)
            np.testing.assert_allclose(adapter.shortest_distance(), native.shortest_distance(), rtol=1e-14)
        self.assertEqual(search.call_count, 1)

    def _assert_snapshot_matches_fresh(self, adapter, search, expected_searches, fresh):
        """J, dJ and the shortest distance after a change equal those of ``fresh()``, a newly built adapter."""
        value, gradient, shortest = adapter.J(), adapter.dJ(), adapter.shortest_distance()
        self.assertEqual(search.call_count, expected_searches)
        fresh = fresh()
        np.testing.assert_array_equal(value, fresh.J())
        # A stale cache would be off by O(1); GPU reductions are not bitwise run to run, so the
        # gradient is compared at a few ulp.
        np.testing.assert_allclose(gradient, fresh.dJ(), rtol=1e-13, atol=0)
        np.testing.assert_array_equal(shortest, fresh.shortest_distance())

    def test_curve_curve_snapshot_follows_geometry_parameters_and_placement(self):
        """Geometry, distance settings and placement changes invalidate curve operands."""
        for change in ["free_dofs", "fixed_dofs", "threshold", "downsample", "base_count", "grid", "backend"]:
            with self.subTest(change=change), self.case() as patches:
                self._case_curve_curve_snapshot_follows_geometry_parameters_and_placement(change, patches)

    def _case_curve_curve_snapshot_follows_geometry_parameters_and_placement(self, change, patches):
        base, curves = _curves()
        adapter = JaxCurveCurveDistance(curves, _CC_THRESHOLD, num_basecurves=len(base))
        adapter.J()
        search = patches.enter_context(mock.patch.object(adapter, "_candidates", wraps=adapter._candidates))
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
            set_backend("jax", device="cpu")
        self._assert_snapshot_matches_fresh(adapter, search, 0 if change == "backend" else 1, lambda: (
            JaxCurveCurveDistance(curves, adapter.minimum_distance, adapter.num_basecurves, adapter.downsample)))

    def test_curve_surface_snapshot_follows_surface_and_operand_changes(self):
        """Surface geometry and operand changes invalidate the curve-surface snapshot."""
        for change in ["surface_dofs", "surface_grid", "normal_only", "tangent_only", "threshold"]:
            with self.subTest(change=change), self.case() as patches:
                self._case_curve_surface_snapshot_follows_surface_and_operand_changes(change, patches)

    def _case_curve_surface_snapshot_follows_surface_and_operand_changes(self, change, patches):
        _, curves = _curves()
        surface = _surface()
        adapter = JaxCurveSurfaceDistance(curves, surface, _CS_THRESHOLD)
        adapter.J()
        search = patches.enter_context(mock.patch.object(adapter, "_candidates", wraps=adapter._candidates))
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
        self._assert_snapshot_matches_fresh(
            adapter, search, 0 if change in ("normal_only", "tangent_only") else 1,
            lambda: JaxCurveSurfaceDistance(curves, adapter.surface, adapter.minimum_distance))

    def test_inactive_distance_terms_have_finite_zero_gradients(self):
        """A far curve with zero tangents adds J = 0 and a zero, finite gradient (native skips such curves; the dense
        JAX kernels evaluate them)."""
        base, curves = _curves()
        point = circle_curve((50.0, 0.0, 0.0), 0.0)
        curves = [*curves, point]
        surface = _surface()
        for name, native, adapter in _distance_objectives(curves, surface, len(base)):
            assert_matches_native_including_nans(name, native, adapter, active=True)
            point_gradient = cast(np.ndarray, objective_partials(adapter)(point))
            self.assertTrue(np.all(np.isfinite(point_gradient)) and not np.any(point_gradient), name)

    def test_distance_terms_skip_exactly_the_native_non_candidates(self):
        """Pairs with no two points closer than the minimum distance are not evaluated: identical circles, and a
        constant curve (zero tangent) on a surface sample, have coincident points (a zero distance); with a zero
        minimum distance they are no candidate pair, so value and gradient are 0."""
        surface = _surface()
        circle = circle_curve((1.0, 0.0, 0.0), 1.0)
        twin = circle_curve((1.0, 0.0, 0.0), 1.0)
        on_sample = circle_curve(tuple(surface.gamma()[0, 0]), 0.0)
        cases = (
            ("CurveCurveDistance", CurveCurveDistance([circle, twin], 0.0), JaxCurveCurveDistance([circle, twin], 0.0)),
            ("CurveSurfaceDistance", CurveSurfaceDistance([circle, on_sample], surface, 0.0),
             JaxCurveSurfaceDistance([circle, on_sample], surface, 0.0)),
        )
        for name, native, adapter in cases:
            assert_matches_native_including_nans(name, native, adapter)
            self.assertTrue(adapter.J() == 0.0 and not np.any(adapter.dJ()), name)

    def test_distance_programs_scale_with_candidates_not_pairs(self):
        """Only native candidates are evaluated, in power-of-two padded batches per pair of quadrature classes, however
        many far curves are added, and padding adds nothing: three 20-point twins form one batch padded to 4 pairs,
        the 24-point twin one of 1; the curve near the surface is the only surface candidate."""
        surface = _surface()
        twins = [(0, 20), (1, 20), (2, 20), (3, 24)]
        for num_far in (0, 12):
            with self.subTest(num_far=num_far), self.case():
                curves = _separated_circles(num_far, twins)
                pair_native, pair_adapter = CurveCurveDistance(curves, 0.1), JaxCurveCurveDistance(curves, 0.1)
                assert_matches_native_including_nans("CurveCurveDistance", pair_native, pair_adapter, active=True)
                self.assertEqual(len(pair_adapter.candidates), len(twins))
                self.assertEqual(_point_pair_distance_shapes(pair_adapter), [(1, 24, 20), (4, 20, 20)])
                near = [circle_curve((0.0, 0.0, 0.0), 1.0), *curves[2 * len(twins):]]
                surface_native = CurveSurfaceDistance(near, surface, 0.4)
                surface_adapter = JaxCurveSurfaceDistance(near, surface, 0.4)
                assert_matches_native_including_nans("CurveSurfaceDistance", surface_native, surface_adapter,
                                                      active=True)
                self.assertEqual(_point_pair_distance_shapes(surface_adapter), [(1, 20, 64)])

    def test_distance_programs_are_reused_by_candidate_sets_of_equal_shape(self):
        """Candidate sets with the same quadrature-class pairs and padded batch lengths share the compiled J and dJ
        programs, whichever rows they hold and in whichever order the class pairs first occur: curves 0 to 5 have 20,
        24, 24, 20, 20 and 24 points; the first layout raises curve 3 above curve 0 and curve 5 above curve 1, the
        second curve 2 above curve 1 and curve 4 above curve 3 (one 20- and one 24-point pair, in opposite orders)."""
        counts = (20, 24, 24, 20, 20, 24)
        programs = (curve_objectives._curve_pair_penalty, curve_objectives._curve_pair_penalty_grad)
        program_counts = []
        for close in (((3, 0), (5, 1)), ((2, 1), (4, 3))):
            above = dict(close)
            curves = [circle_curve((float(above.get(k, k)), 0.0, 0.05 if k in above else 0.0), 0.2, numquadpoints=count)
                      for k, count in enumerate(counts)]
            native, adapter = CurveCurveDistance(curves, 0.1), JaxCurveCurveDistance(curves, 0.1)
            assert_matches_native_including_nans("CurveCurveDistance", native, adapter, active=True)
            self.assertEqual(sorted(adapter.candidates), sorted(close))
            program_counts.append([program._cache_size() for program in programs])
        self.assertEqual(program_counts[1], program_counts[0])

    def test_distance_terms_without_candidates_do_no_jax_work(self):
        """With no native candidate the value is 0 and the gradient native's zeros, without placing geometry or running
        a JAX program."""
        surface = _surface()
        curves = _separated_circles(3, [])
        cases = (
            ("CurveCurveDistance", CurveCurveDistance(curves, 0.1), JaxCurveCurveDistance(curves, 0.1)),
            ("CurveSurfaceDistance", CurveSurfaceDistance(curves, surface, 0.1),
             JaxCurveSurfaceDistance(curves, surface, 0.1)),
        )
        forbidden = mock.Mock(side_effect=AssertionError("JAX work without a candidate"))
        for name in ("_class_stacked_geometry", "_curve_pair_penalty", "_curve_pair_penalty_grad",
                     "_curve_surface_penalty", "_curve_surface_penalty_grad"):
            self.patches.enter_context(mock.patch.object(curve_objectives, name, forbidden))
        for name, native, adapter in cases:
            assert_matches_native_including_nans(name, native, adapter)
            self.assertEqual(adapter.J(), 0.0)
            self.assertFalse(np.any(adapter.dJ()), name)

    def test_distance_operands_are_released_without_candidates(self):
        """Once no pair is a candidate, the placed operands of the last candidate state are dropped rather than held
        until the next one."""
        surface = _surface()
        curves = [circle_curve((0.0, 0.0, 0.0), 1.0), circle_curve((0.0, 0.0, 0.05), 1.0)]
        adapters = (JaxCurveCurveDistance(curves, 0.1), JaxCurveSurfaceDistance(curves, surface, 0.4))
        for adapter in adapters:
            self.assertGreater(adapter.J(), 0.0)
            self.assertIsNotNone(adapter._distance_snapshot)
        for k, curve in enumerate(curves):
            curve.x = circle_curve((0.0, 0.0, 50.0 + 5.0 * k), 1.0).x
        for adapter in adapters:
            self.assertEqual(adapter.J(), 0.0)
            self.assertIsNone(adapter._distance_snapshot)

    def test_distance_candidate_queries_match_native(self):
        """``candidates``, ``compute_candidates()`` and ``shortest_distance_among_candidates()`` match native, without
        candidates too, and follow a DOF change that adds candidates to both searches; ``simsoptpp`` returns the
        candidates in no fixed order."""
        base, curves = _curves()
        surface = _surface()
        for threshold in (0.05, _CC_THRESHOLD, 1.0):
            with self.subTest(threshold=threshold), self.case():
                for native, adapter in _candidate_query_objectives(base, curves, surface, threshold):
                    self.assertIsNone(adapter.compute_candidates())
                    native.compute_candidates()
                    self.assertEqual(sorted(adapter.candidates), _native_candidates(native))
                    self.assertEqual(adapter.shortest_distance_among_candidates(),
                                     native.shortest_distance_among_candidates())
        objectives = _candidate_query_objectives(base, curves, surface, _CC_THRESHOLD)
        before = []
        for native, adapter in objectives:
            native.compute_candidates()
            before.append(_native_candidates(native))
            self.assertEqual(sorted(adapter.candidates), before[-1])
        base[0].x = np.asarray(base[0].x) + 0.2
        for (native, adapter), earlier in zip(objectives, before, strict=True):
            native.compute_candidates()
            self.assertNotEqual(_native_candidates(native), earlier)
            self.assertEqual(sorted(adapter.candidates), _native_candidates(native))

    def test_assigned_distance_candidates_act_as_native(self):
        """Assigning ``candidates`` replaces the search as on native: an empty list gives J = 0 and the all-pairs
        shortest distance, a subset only its terms, None the search again; a DOF change discards the assignment."""
        base, curves = _curves()
        surface = _surface()
        for native, adapter in _candidate_query_objectives(base, curves, surface, _CC_THRESHOLD):
            name = type(native).__name__
            native.compute_candidates()
            for pairs in ([], _native_candidates(native)[:1], None):
                with self.subTest(objective=name, pairs=pairs), self.case():
                    native.candidates = pairs
                    adapter.candidates = pairs
                    assert_matches_native_including_nans(name, native, adapter)
                    self.assertEqual(adapter.shortest_distance(), native.shortest_distance())
                    self.assertEqual(sorted(cast(list, adapter.candidates)), _native_candidates(native))
            native.candidates = adapter.candidates = []
            base[0].x = np.asarray(base[0].x) + 0.01
            native.compute_candidates()
            self.assertTrue(native.candidates, name)
            self.assertEqual(sorted(adapter.candidates), _native_candidates(native))
            base[0].x = np.asarray(base[0].x) - 0.01

    def test_a_notification_discards_assigned_distance_candidates(self):
        """As on native, a recompute notification discards assigned candidates even when the geometry is unchanged:
        reassigning a curve's own DOFs brings back the searched pairs and their nonzero penalty."""
        base, curves = _curves()
        surface = _surface()
        for native, adapter in _candidate_query_objectives(base, curves, surface, _CC_THRESHOLD):
            name = type(native).__name__
            with self.subTest(objective=name), self.case():
                native.candidates = adapter.candidates = []
                self.assertEqual(adapter.J(), 0.0)
                base[0].x = np.array(base[0].x)
                native.compute_candidates()
                self.assertTrue(native.candidates, name)
                self.assertEqual(sorted(adapter.candidates), _native_candidates(native))
                assert_matches_native_including_nans(name, native, adapter, active=True)

    def test_distance_terms_evaluate_every_point_of_a_native_candidate(self):
        """Within a candidate pair every point is evaluated, as native: a zero tangent far from the other curve still
        gives native's non-finite gradient."""
        surface = _surface()
        sample = surface.gamma()[0, 0]
        # Near the sample at t = 1/2 (15 points: t = 7/15), zero tangent at t = 0, far away.
        segment = segment_curve(sample[0] + 1.1, 1.0)
        ring = circle_curve((float(sample[0]) + 0.12, 0.0, 0.15), 0.05)
        on_sample = circle_curve(tuple(sample), 0.0)
        cases = (
            ("CurveCurveDistance", CurveCurveDistance([ring, segment], 0.25),
             JaxCurveCurveDistance([ring, segment], 0.25)),
            ("CurveSurfaceDistance", CurveSurfaceDistance([segment], surface, 0.25),
             JaxCurveSurfaceDistance([segment], surface, 0.25)),
            ("CurveSurfaceDistance(coincident)", CurveSurfaceDistance([on_sample], surface, 0.1),
             JaxCurveSurfaceDistance([on_sample], surface, 0.1)),
        )
        for name, native, adapter in cases:
            self.assertFalse(np.all(np.isfinite(native.dJ())), f"{name}: the case must be singular natively")
            assert_matches_native_including_nans(name, native, adapter)

    def test_distance_candidates_at_the_threshold_are_the_native_ones(self):
        """At a tie the candidate decision is ``simsoptpp``'s own, so the evaluated set, and with it the finite-zero or
        NaN gradient of these constant curves, is native's."""
        self.run_cases(self._case_distance_candidates_at_the_threshold_are_the_native_ones,
                       [dict(position=position, minimum_distance=distance)
                        for position, distance in DISTANCE_TIE_CASES])

    def _case_distance_candidates_at_the_threshold_are_the_native_ones(self, position, minimum_distance):
        origin_curve = circle_curve((0.0, 0.0, 0.0), 0.0, numquadpoints=8)
        point_curve = circle_curve(position, 0.0, numquadpoints=8)
        origin_surface = SurfaceRZFourier.from_nphi_ntheta(nphi=4, ntheta=4)
        origin_surface.local_full_x = np.zeros_like(origin_surface.local_full_x)
        self.assertFalse(np.any(origin_surface.gamma()))
        cases = (
            ("CurveCurveDistance", CurveCurveDistance([origin_curve, point_curve], minimum_distance),
             JaxCurveCurveDistance([origin_curve, point_curve], minimum_distance)),
            ("CurveSurfaceDistance", CurveSurfaceDistance([point_curve], origin_surface, minimum_distance),
             JaxCurveSurfaceDistance([point_curve], origin_surface, minimum_distance)),
        )
        for name, native, adapter in cases:
            assert_matches_native_including_nans(name, native, adapter)

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
            assert_matches_native_including_nans(name, native, adapter, active=True)

    def test_penalty_gradients_taylor(self):
        """Active geometry penalties attain native shrinking-step convergence rates."""
        for name in _TAYLOR_SETTINGS:
            with self.subTest(objective=name), self.case():
                self._case_penalty_gradients_taylor(name)

    def _case_penalty_gradients_taylor(self, name):
        """Use the native stencil, step interval and ratio bound for each penalty."""
        adapter = {key: objective for key, _, objective in _all_objectives(shared_curve=True)}[name]
        centered, scale, stop, bound = _TAYLOR_SETTINGS[name]
        x0 = np.array(adapter.x, dtype=float)
        direction = scale * np.random.RandomState(1).uniform(size=x0.shape)
        derivative = adapter.dJ() @ direction
        initial = adapter.J()
        self.assertGreater(abs(derivative), 1e-10, name)
        previous_error = 1e6
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
