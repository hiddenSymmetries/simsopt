"""The gradgradB VJP of JaxBiotSavart against native Hessian finite differences, and its point-tiled kernel, on CPU."""

from __future__ import annotations

from unittest_jax_support import JaxTestCase, optional_jax_imports
from unittest_jax_support import assert_tree_equal, distinct_host_array, host_snapshot, make_execution_gate
from contextlib import ExitStack
from typing import Protocol, cast

import numpy as np

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax
    import jax.extend.backend
    import jax.numpy as jnp
    from simsopt_jax.backend import set_backend
    from simsopt_jax.core import biotsavart as core
    from simsopt_jax.core.field import grouped_biot_savart_d2B_by_dXdX_from_spec
    from simsopt_jax_adapters.field.biotsavart_backend import JaxBiotSavart

    # Native simsopt imports jax too.
    from simsopt._core.derivative import Derivative
    from simsopt._core.optimizable import Optimizable
    from simsopt.configs import get_ncsx_data
    from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
    from simsopt.geo import create_equally_spaced_curves

_DIRECT_RTOL, _DIRECT_ATOL = (1e-10, 1e-12)
_FD_RTOL = 1e-07
_FIXED_CASES = [dict(fixed=fixed, case=case) for fixed in ["free", "partial", "all"] for case in ["one", "ncsx"]]
_STEPS = [0.5**exponent for exponent in range(5, 10)]


class _CompiledCache(Protocol):
    """``jax.jit`` wrappers expose their compiled-program count as ``_cache_size``."""

    def _cache_size(self) -> int: ...


def _ncsx_case(case, fixed):
    """The NCSX coils (or only the first), with free, partly or all fixed DOFs, at five points, with three seeds."""
    curves, currents, _ = get_ncsx_data()
    coils = coils_via_symmetries(curves, currents, 3, True)
    if case == "one":
        coils, curves, currents = ([coils[0]], curves[:1], currents[:1])
    if fixed == "partial":
        for curve in curves:
            curve.fix(curve.local_dof_names[0])
        currents[0].fix_all()
    elif fixed == "all":
        for owner in [*curves, *currents]:
            owner.fix_all()
    rng = np.random.default_rng(649)
    phi = rng.uniform(0, 2 * np.pi, 5)
    points = np.stack((1.5 * np.cos(phi), 1.5 * np.sin(phi), rng.uniform(-0.1, 0.1, 5)), axis=1)
    seeds = tuple(rng.standard_normal((len(points),) + (3,) * rank) for rank in (1, 2, 3))
    native = BiotSavart(coils).set_points(points)
    field = JaxBiotSavart(coils).set_points(points)
    return (field, native, curves, currents, seeds)


def _contracted(native, points, seed):
    # Native fields do not receive recompute notifications from all-fixed graphs.
    native.set_points(points)
    return np.sum(native.d2B_by_dXdX() * seed)


def _central_difference(native, owner: Optimizable, seed, step, directions=None):
    """Central differences of the contracted Hessian along ``directions`` (default: each DOF)."""
    original = cast(np.ndarray, owner.local_full_x).copy()
    points = native.get_points_cart().copy()
    if directions is None:
        directions = np.eye(original.size)
    gradient = np.empty(len(directions))
    for index, direction in enumerate(directions):
        owner.local_full_x = original + step * direction
        plus = _contracted(native, points, seed)
        owner.local_full_x = original - step * direction
        minus = _contracted(native, points, seed)
        gradient[index] = (plus - minus) / (2 * step)
    owner.local_full_x = original
    native.set_points(points)
    return gradient


class TestBiotsavartGradgradbVjp(JaxTestCase):
    """The gradgradB VJP of ``JaxBiotSavart`` and of its kernel."""

    def test_gradgradb_vjp_contracted_hessian_shrinking_steps(self):
        """Contracted Hessian VJPs converge for curve DOFs and stay linear in current."""
        self.run_cases(self._check_shrinking_steps, _FIXED_CASES)

    def _check_shrinking_steps(self, fixed, case):
        set_backend("jax", device="cpu")
        field, native, curves, currents, seeds = _ncsx_case(case, fixed)
        partials = cast(Derivative, field.B_and_dB_and_d2B_vjp(*seeds)[2](field, as_derivative=True))
        points = native.get_points_cart().copy()
        with ExitStack() as patches:
            patches.callback(native.set_points, points)
            for owner in [*curves, *currents]:
                original = cast(np.ndarray, owner.local_full_x).copy()
                patches.callback(setattr, owner, "local_full_x", original)
                if isinstance(owner, Current):
                    # Relative current perturbations avoid subtracting nearly equal
                    # multi-coil fields when the current is of order 1e5 amperes.
                    direction = 1e-2 * original
                    analytic = float(np.asarray(partials.data[owner]) @ direction)
                    # Current dependence is affine, so no truncation ratio is expected.
                    for step in _STEPS:
                        np.testing.assert_allclose(
                            analytic, _central_difference(native, owner, seeds[2], step, direction[None])[0],
                            rtol=_FD_RTOL, atol=1e-12 * abs(analytic),
                            err_msg=f"{case}/{fixed}: current Hessian contraction at step {step}")
                else:
                    # Extend the native B/dB VJP forward stencil to contracted d2B.
                    baseline = np.sum(native.d2B_by_dXdX() * seeds[2])
                    direction = 1e-2 * np.random.RandomState(1).rand(original.size)
                    analytic = np.asarray(partials.data[owner]) @ direction
                    error = 1e6
                    for step in _STEPS:
                        owner.local_full_x = original + step * direction
                        value = _contracted(native, points, seeds[2])
                        next_error = np.linalg.norm((value - baseline) / step - analytic)
                        self.assertLess(next_error, 0.55 * error,
                                        f"{case}/{fixed}: {type(owner).__name__} Hessian contraction at step {step}")
                        error = next_error
                owner.local_full_x = original
                native.set_points(points)

    def test_gradgradb_vjp_matches_native_curve_and_current_central_differences(self):
        """Hessian VJPs match native finite differences for free and fixed coil DOFs."""
        self.run_cases(self._check_central_differences, _FIXED_CASES)

    def _check_central_differences(self, fixed, case):
        set_backend("jax", device="cpu")
        field, native, curves, currents, seeds = _ncsx_case(case, fixed)
        self.assertFalse(np.array_equal(seeds[2], seeds[2].swapaxes(1, 2)), "symmetric Hessian seed")
        np.testing.assert_allclose(field.d2B_by_dXdX(), native.d2B_by_dXdX(), rtol=_DIRECT_RTOL, atol=_DIRECT_ATOL)
        derivatives = field.B_and_dB_and_d2B_vjp(*seeds)
        self.assertEqual(len(derivatives), 3)
        for value in derivatives:
            self.assertIsInstance(value, Derivative)
        derivative = derivatives[2]
        partials = cast(Derivative, derivative(field, as_derivative=True))
        for owner in [*curves, *currents]:
            step = 0.1 if isinstance(owner, Current) else 2e-05
            coarse = _central_difference(native, owner, seeds[2], step)
            fine = _central_difference(native, owner, seeds[2], step / 2)
            # Extrapolate central differences to remove their leading O(step^2) error.
            expected = (4 * fine - coarse) / 3
            np.testing.assert_allclose(
                np.asarray(partials.data[owner]), expected, rtol=_FD_RTOL, atol=1e-12 * float(np.max(np.abs(expected))),
                err_msg=f"{case}/{fixed}: full {type(owner).__name__} cotangents")
        pullback = field.d2B_by_dXdX_pullback_native(seeds[2])
        projected = field.coil_cotangents_to_dofs_gradient(pullback.d_coil_arrays, pullback.coil_indices)
        gradient = np.asarray(derivative(field))
        np.testing.assert_allclose(projected, gradient, rtol=1e-12, atol=1e-14)
        self.assertEqual(len(gradient), len(native.x))

    def test_first_two_slots_equal_existing_B_and_dB_vjp(self):
        """Adding the Hessian slot preserves both existing field and gradient VJPs."""
        self.run_cases(self._check_first_two_slots, [dict(case="one"), dict(case="ncsx")])

    def _check_first_two_slots(self, case):
        field, _, _, _, seeds = _ncsx_case(case, "partial")
        expected = field.B_and_dB_vjp(*seeds[:2])
        actual = field.B_and_dB_and_d2B_vjp(*seeds)
        for previous, added in zip(expected, actual[:2], strict=True):
            np.testing.assert_array_equal(previous(field), added(field))
            for owner, value in previous.data.items():
                np.testing.assert_array_equal(value, added.data[owner])

    def test_tiled_hessian_vjp_matches_dense_tile_zero_with_multiple_groups(self):
        """Tiled, untiled and public pullbacks match jax.vjp of the grouped forward Hessian, with two groups."""
        self.run_cases(self._check_tiled, [dict(point_count=count) for count in [0, 1, 4, 5, 8, 12, 13, 37]])

    def _check_tiled(self, point_count):
        curves = create_equally_spaced_curves(2, 1, stellsym=False, R0=1.0, R1=0.3, order=2, numquadpoints=12)
        curves[1] = create_equally_spaced_curves(2, 1, stellsym=False, R0=1.0, R1=0.3, order=2, numquadpoints=16)[1]
        field = JaxBiotSavart([Coil(curves[0], Current(100000.0)), Coil(curves[1], Current(-20000.0))])
        rng = np.random.default_rng(649)
        field.set_points(rng.uniform(-0.2, 0.2, (point_count, 3)) + np.array([0.9, 0, 0]))
        seed = jnp.asarray(rng.standard_normal((point_count, 3, 3, 3)))
        expected = field._group_pullback_native(core._make_d2B_vjp_kernel(0), seed).d_coil_arrays
        direct = field._field_pullback_native(grouped_biot_savart_d2B_by_dXdX_from_spec, seed).d_coil_arrays
        tiled = field._group_pullback_native(core._make_d2B_vjp_kernel(4), seed).d_coil_arrays
        pullback = field.d2B_by_dXdX_pullback_native(seed)
        self.assertEqual(pullback.coil_indices, field.coil_set_spec().coil_index_lists())
        leaves = (jax.tree.leaves(tree) for tree in (pullback.d_coil_arrays, tiled, expected, direct))
        for public, actual, untiled, dense in zip(*leaves, strict=True):
            np.testing.assert_allclose(actual, untiled, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(actual, dense, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(public, dense, rtol=1e-12, atol=1e-14)

    def test_nonsymmetric_seed_contracts_native_ordered_hessian_current_gradient(self):
        """Mixed spatial partials commute; this oracle detects component-axis mixups."""
        set_backend("jax", device="cpu")
        field, native, _, currents, _ = _ncsx_case("one", "free")
        center = field.coils[0].curve.gamma()[0]
        # Far-field vacuum Hessians can be symmetric even in their component axis.
        # Near-coil finite quadrature distinguishes component/spatial ordering.
        points = center + np.array([[0.02, 0.01, -0.025], [-0.01, 0.025, 0.02], [0.02, -0.02, 0.015]])
        field.set_points(points)
        native.set_points(points)
        seed = np.random.default_rng(649).standard_normal((len(points), 3, 3, 3))
        self.assertFalse(np.array_equal(seed, seed.swapaxes(1, 2)), "symmetric Hessian seed")
        hessian = native.d2B_by_dXdX()
        # B is linear in the single coil's current, so this is an independent
        # exact current-gradient oracle with explicit [point, d1, d2, component] axes.
        expected = np.einsum("pijc,pijc->", seed, hessian) / currents[0].get_value()
        derivative = field.B_and_dB_and_d2B_vjp(np.zeros((len(points), 3)), np.zeros((len(points), 3, 3)), seed)[2]
        actual = derivative.data[currents[0]][0]
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12 * abs(expected))

    def test_new_seeds_reuse_compiled_hessian_vjp(self):
        """Changing the Hessian seed reuses compilation and scales the cotangents."""
        set_backend("jax", device="cpu")
        field, _, _, _, seeds = _ncsx_case("one", "free")
        kernel = cast(_CompiledCache, core._make_d2B_vjp_kernel(core.HESSIAN_VJP_POINT_CHUNK_SIZE["cpu"]))
        first = field.d2B_by_dXdX_pullback_native(seeds[2])
        jax.block_until_ready(first.d_coil_arrays)
        compiled_count = kernel._cache_size()
        self.assertEqual(compiled_count, 1)
        second = field.d2B_by_dXdX_pullback_native(-2 * seeds[2])
        jax.block_until_ready(second.d_coil_arrays)
        self.assertEqual(kernel._cache_size(), compiled_count)
        leaves = zip(jax.tree.leaves(second.d_coil_arrays), jax.tree.leaves(first.d_coil_arrays), strict=True)
        for actual, original in leaves:
            np.testing.assert_allclose(actual, -2 * original, rtol=1e-13, atol=1e-14)

    def test_hessian_vjp_evaluates_float32_inputs_in_float64(self):
        """Float32 inputs give the float64 cotangents of their float64 values."""
        set_backend("jax", device="cpu")
        rng = np.random.default_rng(649)
        # 13 points: three four-point tiles plus a one-point tail.
        inputs = (0.2 * rng.standard_normal((13, 3)), rng.standard_normal((13, 3, 3, 3)),
                  rng.standard_normal((2, 12, 3)) + 3, rng.standard_normal((2, 12, 3)), np.array([1e5, -2e4]))
        single = tuple(value.astype(np.float32) for value in inputs)
        for name, pullback in [
            ("four-point tiles", core._make_d2B_vjp_kernel(4)),
            ("public", core.biot_savart_d2B_by_dXdX_vjp),
            ("public inside jax.jit", jax.jit(core.biot_savart_d2B_by_dXdX_vjp)),
        ]:
            expected = pullback(*(value.astype(np.float64) for value in single))
            actual = pullback(*single)
            for observed, reference in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
                self.assertEqual(observed.dtype, jnp.float64, name)
                np.testing.assert_allclose(observed, reference, rtol=1e-12,
                                           atol=1e-14 * float(np.max(np.abs(reference))), err_msg=name)

    def test_hessian_vjp_owns_numpy_inputs(self):
        """Caller mutation after dispatch cannot alter pending Hessian cotangents."""
        cases = [dict(pending_input=name, misaligned=flag) for name in ["points", "seed"] for flag in [False, True]]
        self.run_cases(self._check_owns_numpy_inputs, cases)

    def _check_owns_numpy_inputs(self, pending_input, misaligned):
        execution_gate = make_execution_gate()
        set_backend("jax", device="cpu")
        shapes = ((13, 3), (13, 3, 3, 3), (1, 12, 3), (1, 12, 3), (1,))
        sources = [distinct_host_array(shape, misaligned=misaligned) for shape in shapes]
        sources[0] *= 0.2
        expected = host_snapshot(core.biot_savart_d2B_by_dXdX_vjp(*sources))
        index = 0 if pending_input == "points" else 1
        device_input = jnp.asarray(sources[index].copy())
        execution_gate(device_input).block_until_ready()
        inputs = list(sources)
        inputs[index] = execution_gate(device_input)
        result = core.biot_savart_d2B_by_dXdX_vjp(*inputs)
        pending = any((not leaf.is_ready() for leaf in jax.tree.leaves(result)))
        self.assertTrue(pending, "pullback completed before mutation")
        for source in sources:
            source[:] = -7.0
        assert_tree_equal(result, expected)

    def test_hessian_vjp_point_tile_follows_the_platform_of_the_call(self):
        """The public pullback compiles only at the tile (CPU 128, GPU 2048) of its call's platform."""
        self.assertEqual(dict(core.HESSIAN_VJP_POINT_CHUNK_SIZE), {"cpu": 128, "gpu": 2048})
        self.assertEqual(core.POINT_CHUNK_SIZE, 1024)
        rows = [("cpu", None)]
        if {"cuda", "gpu"} & set(jax.extend.backend.backends()):
            rows += [("gpu", None), ("cpu", "gpu"), ("gpu", "cpu")]
        self.run_cases(self._check_platform_tile, [dict(default=default, committed=device) for default, device in rows])

    def _check_platform_tile(self, default, committed):
        """The call runs on a committed input's device, else on the default one; the forward tile is separate."""
        set_backend("jax", device=default)
        rng = np.random.default_rng(649)
        points = 0.2 * rng.standard_normal((5, 3))
        if committed is not None:
            points = jax.device_put(points, jax.devices(committed)[0])
        core.biot_savart_d2B_by_dXdX_vjp(points, rng.standard_normal((5, 3, 3, 3)), rng.standard_normal((1, 12, 3)) + 3,
                                         rng.standard_normal((1, 12, 3)), np.array([1e5]))
        platform = committed or default
        for name, tile in core.HESSIAN_VJP_POINT_CHUNK_SIZE.items():
            kernel = cast(_CompiledCache, core._make_d2B_vjp_kernel(tile))
            message = f"{name} tile compiled for a call on {platform}"
            self.assertEqual(kernel._cache_size(), int(name == platform), message)
