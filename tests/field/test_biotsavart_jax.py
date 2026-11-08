"""
Parity tests for the JAX Biot-Savart implementation.

Validates against:
1. Analytical on-axis field of a circular current loop.
2. Maxwell's equation ∇·B = 0 (trace of dB/dX).
3. C++ reference from the installed simsoptpp extension.
"""

from unittest_jax_support import JAX_IMPORT_ERROR, JaxTestCase, optional_jax_imports

with optional_jax_imports():
    import simsopt_jax  # noqa: F401
    import jax  # noqa: F401
    import jax.numpy as jnp
    from simsopt_jax_adapters.field.biotsavart_backend import (
        JaxBiotSavart,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import (
        _per_coil_unit_field,
    )
    from simsopt_jax.core.specs import CoilGroupSpec, GroupedCoilSetSpec
    from simsopt_jax.core import biotsavart as core_biotsavart
    from simsopt_jax.backend import get_field_kernel_tuning
    from simsopt_jax.backend.runtime import invalidate_backend_cache
    from simsopt_jax.runtime.host_boundary import disallow_host_transfers


import os
from unittest import mock

import numpy as np
from typing import cast
from simsopt._core.derivative import Derivative


from simsopt.configs import get_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.field.coil import Coil, Current
from simsopt.geo.curvexyzfourier import CurveXYZFourier


if JAX_IMPORT_ERROR is None:
    biot_savart_B = core_biotsavart.biot_savart_B
    biot_savart_dB_by_dX = core_biotsavart.biot_savart_dB_by_dX
    biot_savart_A = core_biotsavart.biot_savart_A
    biot_savart_dA_by_dX = core_biotsavart.biot_savart_dA_by_dX


_DIRECT_KERNEL_TOLS = {
    "rtol": 1e-10,
    "atol": 1e-12,
}


_DERIVATIVE_HEAVY_TOLS = {
    "first_derivative_rtol": 1e-08,
    "first_derivative_atol": 1e-10,
    "second_derivative_rtol": 1e-10,
    "second_derivative_atol": 1e-12,
}


def _ncsx_biotsavart_parity_fixture():

    curves, currents_objs, _, nfp, _ = get_data("ncsx")
    coils = coils_via_symmetries(curves, currents_objs, nfp, stellsym=True)
    bs = BiotSavart(coils)

    npoints = 50
    rng = np.random.default_rng(42)
    points_np = rng.standard_normal((npoints, 3)) * 0.3
    points_np[:, 0] += 1.0  # shift near torus

    bs.set_points(points_np)
    gammas_np = np.array([coil.curve.gamma() for coil in coils])
    gds_np = np.array([coil.curve.gammadash() for coil in coils])
    currents_np = np.array([coil.current.get_value() for coil in coils])
    return bs, points_np, gammas_np, gds_np, currents_np


def _native_taylor_fixture():
    """Coil and points of the native ``tests/field/test_biotsavart.py`` Taylor tests."""
    curve = CurveXYZFourier(200, 3)
    coeffs = curve.dofs_matrix
    coeffs[1][0] = 1.0
    coeffs[1][1] = 0.5
    coeffs[2][2] = 0.5
    curve.set_dofs(np.concatenate(coeffs))
    rng = np.random.default_rng(1)
    points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
    points += 0.001 * (rng.random(points.shape) - 0.5)
    return Coil(curve, Current(1e4)), points


def _cart_points_to_cyl(points):
    return np.ascontiguousarray(
        np.stack(
            (
                np.sqrt(points[:, 0] * points[:, 0] + points[:, 1] * points[:, 1]),
                np.arctan2(points[:, 1], points[:, 0]),
                points[:, 2],
            ),
            axis=1,
        )
    )


def _assert_cylindrical_points_match_cpu(jax_field, cpu_field):
    np.testing.assert_allclose(
        np.asarray(jax_field.get_points_cyl()),
        np.asarray(cpu_field.get_points_cyl()),
        rtol=0.0,
        atol=1.0e-15,
    )


def _assert_cylindrical_accessors_match_cpu(jax_field, cpu_field):
    _assert_cylindrical_points_match_cpu(jax_field, cpu_field)
    np.testing.assert_allclose(
        np.asarray(jax_field.AbsB()),
        np.asarray(cpu_field.AbsB()),
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )
    assert np.asarray(jax_field.AbsB()).shape == np.asarray(cpu_field.AbsB()).shape
    for method_name in ("B_cyl", "A_cyl"):
        np.testing.assert_allclose(
            np.asarray(getattr(jax_field, method_name)()),
            np.asarray(getattr(cpu_field, method_name)()),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
    np.testing.assert_allclose(
        np.asarray(jax_field.GradAbsB_cyl()),
        np.asarray(cpu_field.GradAbsB_cyl()),
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
    )


class TestBiotSavartJaxPhysics(JaxTestCase):
    """Field properties that hold independently of the native implementation."""

    def test_circular_loop_on_axis_field_is_analytic(self):
        """On the axis of a circular loop of radius R carrying I,
        B = (0, 0, mu0 I R^2 / (2 (R^2 + z^2)^(3/2))).

        The integrand is constant along the loop for points on its axis, so the
        quadrature is exact and only round-off remains.
        """
        radius, current = 1.3, 2.0e5
        curve = CurveXYZFourier(64, 1)
        curve.x = np.array(
            [radius if name in ("xc(1)", "ys(1)") else 0.0 for name in curve.local_dof_names]
        )
        z = np.array([-2.0, -0.4, 0.0, 0.25, 3.0])
        points = np.stack((np.zeros_like(z), np.zeros_like(z), z), axis=1)
        field = JaxBiotSavart([Coil(curve, Current(current))]).set_points(points)
        mu0 = 4.0e-7 * np.pi
        expected = np.zeros_like(points)
        expected[:, 2] = mu0 * current * radius**2 / (2.0 * (radius**2 + z**2) ** 1.5)
        np.testing.assert_allclose(
            np.asarray(field.B()), expected, rtol=1e-14, atol=1e-14 * expected[:, 2].max()
        )

    def test_field_is_divergence_free(self):
        """The trace of dB/dX vanishes to round-off of its largest entry at each
        point, around the NCSX magnetic axis."""
        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        field = JaxBiotSavart(list(bs._coils)).set_points(points_np)
        dB = np.asarray(field.dB_by_dX())
        divergence = np.trace(dB, axis1=1, axis2=2)
        self.assertLess(
            np.max(np.abs(divergence) / np.max(np.abs(dB), axis=(1, 2))), 1e-13
        )


class TestBiotSavartJaxCppParity(JaxTestCase):
    """Compare against the installed C++ simsoptpp kernel."""

    def test_kernel_parity_ncsx(self):
        """Raw JAX kernels match native ``BiotSavart`` on the NCSX parity fixture."""
        direct = (_DIRECT_KERNEL_TOLS["rtol"], _DIRECT_KERNEL_TOLS["atol"])
        first_derivative = (
            _DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        for quantity, kernel, (rtol, atol) in (
            ("B", biot_savart_B, direct),
            ("A", biot_savart_A, direct),
            ("dB_by_dX", biot_savart_dB_by_dX, first_derivative),
            ("dA_by_dX", biot_savart_dA_by_dX, first_derivative),
        ):
            with self.subTest(quantity=quantity), self.case():
                self._case_kernel_parity_ncsx(quantity, kernel, rtol, atol)

    def _case_kernel_parity_ncsx(self, quantity, kernel, rtol, atol):
        """Run one row with case-local objects released before runtime cleanup."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        expected = getattr(bs, quantity)()

        actual = kernel(
            jnp.array(points_np),
            jnp.array(gammas_np),
            jnp.array(gds_np),
            jnp.array(currents_np),
        )

        np.testing.assert_allclose(np.array(actual), expected, rtol=rtol, atol=atol)

    def test_cylindrical_public_accessors_parity_ncsx(self):
        """``JaxBiotSavart`` owns cylindrical public accessors on its boundary."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = _cart_points_to_cyl(points_np)
        points_cyl[:, 1] += 2.0 * np.pi
        bs.set_points_cyl(points_cyl)

        bs_jax = JaxBiotSavart(list(bs._coils))
        self.assertTrue(
            bs_jax.set_points_cyl(points_cyl) is bs_jax,
            "bs_jax.set_points_cyl(points_cyl) is bs_jax",
        )
        self.assertTrue(
            bs_jax.set_points_cart(points_np) is bs_jax,
            "bs_jax.set_points_cart(points_np) is bs_jax",
        )
        self.assertTrue(
            bs_jax.set_points_cyl(points_cyl) is bs_jax,
            "bs_jax.set_points_cyl(points_cyl) is bs_jax",
        )

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cylindrical_public_accessors_use_cached_phi_basis_ncsx(self):
        """``B_cyl`` / ``A_cyl`` / ``GradAbsB_cyl`` use cached cylindrical phi."""

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cyl = np.ascontiguousarray(
            np.array(
                [
                    [0.0, 1.234, 0.1],
                    [0.0, -1.5, -0.2],
                    [0.7, 2.0 * np.pi + 0.4, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cyl(points_cyl)

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points_cyl(points_cyl)

        _assert_cylindrical_accessors_match_cpu(bs_jax, bs)

    def test_cartesian_public_accessors_normalize_cylindrical_phi_ncsx(self):
        """Cartesian point conversion follows C++ ``get_points_cyl_impl``."""

        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        points_cart = np.ascontiguousarray(
            np.array(
                [
                    [0.5, -0.5, 0.1],
                    [-0.5, -0.25, -0.2],
                    [0.0, 0.0, 0.3],
                ],
                dtype=np.float64,
            )
        )
        bs.set_points_cart(points_cart)

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points_cart(points_cart)

        _assert_cylindrical_points_match_cpu(bs_jax, bs)

    def test_nonfinite_point_leaves_other_cylindrical_angles_native(self):
        """A nonfinite point does not change the cylindrical angle of finite points,
        for Cartesian input (wrap to [0, 2*pi)) and cylindrical input (fmod)."""
        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        bs_jax = JaxBiotSavart(list(bs._coils))
        for setter, points in (
            ("set_points_cart", [[1.0, -1.0, 0.0], [np.inf, 0.0, 0.0]]),
            ("set_points_cyl", [[1.0, 7.0, 0.0], [1.0, np.nan, 0.0]]),
        ):
            with self.subTest(setter=setter):
                points = np.ascontiguousarray(points, dtype=np.float64)
                getattr(bs, setter)(points)
                getattr(bs_jax, setter)(points)
                np.testing.assert_array_equal(
                    bs_jax.get_points_cyl()[0], bs.get_points_cyl()[0]
                )

    def test_default_evaluation_point_is_origin_like_native(self):
        """Without set_points the field is evaluated at one point at the origin."""
        bs, _, _, _, _ = _ncsx_biotsavart_parity_fixture()
        native = BiotSavart(list(bs._coils))
        bs_jax = JaxBiotSavart(list(bs._coils))
        np.testing.assert_array_equal(bs_jax.get_points_cart(), np.zeros((1, 3)))
        np.testing.assert_allclose(
            np.asarray(bs_jax.B()),
            native.B(),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_empty_coil_set_GradAbsB_is_nan_like_native(self):
        """With no coils |B| is identically zero and its gradient is native's 0/0."""
        _, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        native = BiotSavart([]).set_points(points_np)
        bs_jax = JaxBiotSavart([]).set_points(points_np)
        np.testing.assert_array_equal(np.asarray(bs_jax.B()), native.B())
        self.assertTrue(np.all(np.isnan(native.GradAbsB())))
        np.testing.assert_array_equal(np.asarray(bs_jax.GradAbsB()), native.GradAbsB())
        # The NaN is built on device, so it also holds under the strict guard.
        bs_jax.set_points(points_np)
        with disallow_host_transfers():
            grad_abs_B = bs_jax.GradAbsB()
        np.testing.assert_array_equal(np.asarray(grad_abs_B), native.GradAbsB())

    def test_B_vjp_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""

        bs_cpu, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        coils = list(bs_cpu._coils)

        bs_jax = JaxBiotSavart(coils)
        bs_jax.set_points(points_np)

        v = np.asarray(bs_cpu.B(), dtype=np.float64).copy()
        deriv_cpu = cast(Derivative, bs_cpu.B_vjp(v))
        deriv_jax = bs_jax.B_vjp(v)

        for coil in coils:
            np.testing.assert_allclose(
                np.asarray(deriv_jax(coil)),
                np.asarray(deriv_cpu(coil)),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
                err_msg=(
                    "JaxBiotSavart.B_vjp() does not match BiotSavart.B_vjp() "
                    "on the NCSX parity fixture"
                ),
            )

    def test_coil_chunked_kernels_and_pullback_match_native_ncsx(self):
        """Coil chunking leaves B, dB/dX and the B pullback at native values.

        The 18 NCSX coils form two chunks of 10 and 8 (the exact two-chunk
        path), or two chunks of 7 and a 4-coil tail (the padded loop).
        """
        for chunk_size in (10, 7):
            with self.subTest(chunk_size=chunk_size), self.case() as patches:
                patches.enter_context(
                    mock.patch.dict(
                        os.environ, {"SIMSOPT_JAX_COIL_CHUNK_SIZE": str(chunk_size)}
                    )
                )
                invalidate_backend_cache()
                self._case_coil_chunked_kernels_and_pullback_match_native_ncsx(
                    chunk_size
                )

    def _case_coil_chunked_kernels_and_pullback_match_native_ncsx(self, chunk_size):
        """Run one row with case-local objects released before runtime cleanup."""
        bs, points_np, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        self.assertEqual(get_field_kernel_tuning().coil_chunk_size, chunk_size)
        self.assertEqual(len(currents_np), 18)
        np.testing.assert_allclose(
            np.asarray(biot_savart_B(points_np, gammas_np, gds_np, currents_np)),
            bs.B(),
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )
        np.testing.assert_allclose(
            np.asarray(biot_savart_dB_by_dX(points_np, gammas_np, gds_np, currents_np)),
            bs.dB_by_dX(),
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        bs_jax = JaxBiotSavart(list(bs._coils)).set_points(points_np)
        v = np.asarray(bs.B(), dtype=np.float64).copy()
        expected = cast(Derivative, bs.B_vjp(v))
        actual = bs_jax.B_vjp(v)
        for coil in bs._coils:
            np.testing.assert_allclose(
                np.asarray(actual(coil)),
                np.asarray(expected(coil)),
                rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
                atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
            )

    def test_adapter_derivative_parity_ncsx(self):
        """``JaxBiotSavart`` spatial derivatives match native ``BiotSavart`` on the NCSX
        parity fixture."""
        first_derivative = (
            _DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )
        second_derivative = (
            _DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            _DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )
        for quantity, (rtol, atol) in (
            ("dA_by_dX", first_derivative),
            ("d2B_by_dXdX", second_derivative),
            ("d2A_by_dXdX", second_derivative),
        ):
            with self.subTest(quantity=quantity), self.case():
                self._case_adapter_derivative_parity_ncsx(quantity, rtol, atol)

    def _case_adapter_derivative_parity_ncsx(self, quantity, rtol, atol):
        """Run one row with case-local objects released before runtime cleanup."""
        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        expected = getattr(bs, quantity)()

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points(points_np)
        actual = getattr(bs_jax, quantity)()

        np.testing.assert_allclose(np.array(actual), expected, rtol=rtol, atol=atol)

    def test_GradAbsB_taylor_ncsx(self):
        """GradAbsB matches central differences of AbsB with second-order convergence
        under shrinking point steps."""

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points(points_np)
        grad = np.asarray(bs_jax.GradAbsB())
        for direction in np.eye(3):
            with self.subTest(direction=direction):
                deriv = grad @ direction
                err = 1e6
                for i in range(5):
                    eps = 1e-3 * 0.5**i
                    bs_jax.set_points(points_np + eps * direction)
                    plus = np.asarray(bs_jax.AbsB())[:, 0]
                    bs_jax.set_points(points_np - eps * direction)
                    minus = np.asarray(bs_jax.AbsB())[:, 0]
                    new_err = np.linalg.norm((plus - minus) / (2 * eps) - deriv)
                    self.assertLess(new_err, 0.3 * err)
                    err = new_err


class TestJaxBiotSavartSpatialTaylor(JaxTestCase):
    """Shrinking-step Taylor tests of the ``JaxBiotSavart`` spatial derivatives.

    Mirrors the native ``tests/field/test_biotsavart.py`` checks on its coil
    and points: steps ``0.5**i`` for ``i`` in ``5..9``, a one-sided difference
    with error ratio 0.55 for first derivatives and a central difference of
    the first derivative with error ratio 0.30 for second derivatives.
    """

    def _assert_spatial_taylor(self, value_name, derivative_name, *, central, ratio):
        """Assert ``derivative[:, k]`` is the shrinking-step limit along ``e_k``."""
        coil, points_np = _native_taylor_fixture()
        bs_jax = JaxBiotSavart([coil])
        bs_jax.set_points(points_np)
        value0 = np.asarray(getattr(bs_jax, value_name)())
        derivative = np.asarray(getattr(bs_jax, derivative_name)())
        for k, direction in enumerate(np.eye(3)):
            with self.subTest(direction=k):
                expected = derivative[:, k]
                err = 1e6
                for i in range(5, 10):
                    eps = 0.5**i
                    bs_jax.set_points(points_np + eps * direction)
                    plus = np.asarray(getattr(bs_jax, value_name)())
                    if central:
                        bs_jax.set_points(points_np - eps * direction)
                        minus = np.asarray(getattr(bs_jax, value_name)())
                        estimate = (plus - minus) / (2.0 * eps)
                    else:
                        estimate = (plus - value0) / eps
                    new_err = np.linalg.norm(estimate - expected)
                    self.assertLess(new_err, ratio * err)
                    err = new_err

    def test_dB_by_dX_taylor(self):
        """dB_by_dX is the first-order shrinking-step limit of B along each axis."""
        self._assert_spatial_taylor("B", "dB_by_dX", central=False, ratio=0.55)

    def test_d2B_by_dXdX_taylor(self):
        """d2B_by_dXdX is the central shrinking-step limit of dB_by_dX along each axis."""
        self._assert_spatial_taylor("dB_by_dX", "d2B_by_dXdX", central=True, ratio=0.30)

    def test_dA_by_dX_taylor(self):
        """dA_by_dX is the first-order shrinking-step limit of A along each axis."""
        self._assert_spatial_taylor("A", "dA_by_dX", central=False, ratio=0.55)

    def test_d2A_by_dXdX_taylor(self):
        """d2A_by_dXdX is the central shrinking-step limit of dA_by_dX along each axis."""
        self._assert_spatial_taylor("dA_by_dX", "d2A_by_dXdX", central=True, ratio=0.30)


# About 45 ulp. The measured worst case on these configurations is about
# 2e-15 at the integrand and 1.4e-15 at the kernel.
_ROUND_OFF = 1e-14


class TestFusedBAndDBAutodiffReference(JaxTestCase):
    def test_fused_B_and_dB_matches_autodiff_of_B(self):
        """Claim: the hand-written fused B + dB/dX integrand, kept as a mirror of
        native ``biot_savart_kernel`` because it is faster, equals the B integrand
        and its ``jax.jacfwd`` in the evaluation point to round-off at every
        quadrature node, and the tiled ``biot_savart_B_and_dB`` equals
        ``biot_savart_B`` and its ``jax.jacfwd`` kernel ``biot_savart_dB_by_dX``.

        Configurations: NCSX coils at points around the magnetic axis, at points
        1-10 mm from a coil, and at points about 10 m away. Integrand errors are
        relative to the node's term size, |gammadash| / r**2 for B and
        |gammadash| / r**3 for dB, so cancellation in the cross product does not
        inflate them; kernel errors are relative to the largest entry at each
        point.
        """
        _bs, axis_points, gammas_np, gds_np, currents_np = (
            _ncsx_biotsavart_parity_fixture()
        )
        gammas = jnp.asarray(gammas_np)
        gammadashs = jnp.asarray(gds_np)
        currents = jnp.asarray(currents_np)
        tangent = gds_np[0, 0] / np.linalg.norm(gds_np[0, 0])
        offset = np.cross(tangent, [0.0, 0.0, 1.0])
        offset /= np.linalg.norm(offset)
        configurations = {
            "around the axis": axis_points,
            "near a coil": gammas_np[0, 0] + np.outer([1e-3, 3e-3, 1e-2], offset),
            "far away": np.array([[10.0, 3.0, -2.0], [-7.0, 8.0, 5.0]]),
        }

        def B_integrand(x):
            return core_biotsavart._biot_savart_B_integrand(x, gammas, gammadashs)

        def autodiff_integrand(x):
            # jacfwd yields [..., component, direction]; the fused integrand
            # puts the derivative direction first.
            return B_integrand(x), jnp.swapaxes(jax.jacfwd(B_integrand)(x), -1, -2)

        def fused_integrand(x):
            return core_biotsavart._biot_savart_B_and_dB_integrand(
                x, gammas, gammadashs
            )

        for name, points_np in configurations.items():
            with self.subTest(configuration=name):
                points = jnp.asarray(points_np)
                B_fused, dB_fused = jax.vmap(fused_integrand)(points)
                B_auto, dB_auto = jax.vmap(autodiff_integrand)(points)
                distance = np.linalg.norm(
                    points_np[:, None, None, :] - gammas_np[None], axis=-1
                )
                B_scale = np.linalg.norm(gds_np, axis=-1)[None] / distance**2
                B_error = np.max(np.abs(np.asarray(B_fused - B_auto)), axis=-1)
                dB_error = np.max(np.abs(np.asarray(dB_fused - dB_auto)), axis=(-2, -1))
                self.assertLess(np.max(B_error / B_scale), _ROUND_OFF)
                self.assertLess(np.max(dB_error * distance / B_scale), _ROUND_OFF)

                B_kernel, dB_kernel = core_biotsavart.biot_savart_B_and_dB(
                    points, gammas, gammadashs, currents
                )
                B_reference = np.asarray(
                    biot_savart_B(points, gammas, gammadashs, currents)
                )
                dB_reference = np.asarray(
                    biot_savart_dB_by_dX(points, gammas, gammadashs, currents)
                )
                self.assertLess(
                    np.max(
                        np.max(np.abs(np.asarray(B_kernel) - B_reference), axis=-1)
                        / np.max(np.abs(B_reference), axis=-1)
                    ),
                    _ROUND_OFF,
                )
                self.assertLess(
                    np.max(
                        np.max(np.abs(np.asarray(dB_kernel) - dB_reference), axis=(-2, -1))
                        / np.max(np.abs(dB_reference), axis=(-2, -1))
                    ),
                    _ROUND_OFF,
                )


class TestBiotSavartJaxCppCoilCurrentParity(JaxTestCase):
    """Per-coil field and spatial derivative parity against native Biot-Savart."""

    @staticmethod
    def _assert_coil_current_list_parity(cache_method, list_method, *, rtol, atol):

        bs, points_np, _, _, _ = _ncsx_biotsavart_parity_fixture()
        # Populate the matching C++ fieldcache entries before pulling the
        # per-coil list, so ordering is deterministic.
        getattr(bs, cache_method)()
        cpu_list = getattr(bs, list_method)()

        bs_jax = JaxBiotSavart(list(bs._coils))
        bs_jax.set_points(points_np)
        jax_list = getattr(bs_jax, list_method)()

        assert len(jax_list) == len(cpu_list)
        for k, (j_entry, c_entry) in enumerate(zip(jax_list, cpu_list)):
            np.testing.assert_allclose(
                np.array(j_entry),
                c_entry,
                rtol=rtol,
                atol=atol,
                err_msg=f"coil {k}",
            )

    def test_dB_by_dcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "B",
            "dB_by_dcoilcurrents",
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_per_coil_unit_field_vectorizes_within_quadrature_group(self):
        """Unit-current fields vectorize per quadrature group and return values in
        public coil order."""
        points = jnp.asarray([[0.0, 0.0, 0.0], [0.25, -0.5, 1.0]], dtype=jnp.float64)
        group0_gammas = jnp.arange(18, dtype=jnp.float64).reshape(2, 3, 3)
        group0_gammadashs = group0_gammas + 0.5
        group1_gammas = jnp.arange(9, dtype=jnp.float64).reshape(1, 3, 3) - 2.0
        group1_gammadashs = group1_gammas - 0.25
        coil_set_spec = GroupedCoilSetSpec(
            groups=(
                CoilGroupSpec(
                    gammas=group0_gammas,
                    gammadashs=group0_gammadashs,
                    currents=jnp.asarray([3.0, 4.0], dtype=jnp.float64),
                    coil_indices=(2, 0),
                ),
                CoilGroupSpec(
                    gammas=group1_gammas,
                    gammadashs=group1_gammadashs,
                    currents=jnp.asarray([5.0], dtype=jnp.float64),
                    coil_indices=(1,),
                ),
            )
        )
        calls = []

        def kernel(kernel_points, gammas, gammadashs, currents):
            calls.append(gammas.shape)
            value = jnp.sum(gammas) + jnp.sum(gammadashs) + jnp.sum(currents)
            return jnp.broadcast_to(value, kernel_points.shape)

        results = _per_coil_unit_field(points, coil_set_spec, kernel)

        self.assertTrue(len(calls) == 2, "len(calls) == 2")
        self.assertTrue(
            calls == [(1, 3, 3), (1, 3, 3)], "calls == [(1, 3, 3), (1, 3, 3)]"
        )
        self.assertTrue(len(results) == 3, "len(results) == 3")
        np.testing.assert_allclose(
            np.asarray(results[0]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[1]))
                + np.sum(np.asarray(group0_gammadashs[1]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[1]),
            np.broadcast_to(
                np.sum(np.asarray(group1_gammas[0]))
                + np.sum(np.asarray(group1_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )
        np.testing.assert_allclose(
            np.asarray(results[2]),
            np.broadcast_to(
                np.sum(np.asarray(group0_gammas[0]))
                + np.sum(np.asarray(group0_gammadashs[0]))
                + 1.0,
                points.shape,
            ),
        )

    def test_dA_by_dcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "A",
            "dA_by_dcoilcurrents",
            rtol=_DIRECT_KERNEL_TOLS["rtol"],
            atol=_DIRECT_KERNEL_TOLS["atol"],
        )

    def test_d2B_by_dXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "dB_by_dX",
            "d2B_by_dXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d2A_by_dXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "dA_by_dX",
            "d2A_by_dXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
        )

    def test_d3B_by_dXdXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "d2B_by_dXdX",
            "d3B_by_dXdXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )

    def test_d3A_by_dXdXdcoilcurrents_parity_ncsx(self):
        """Native Biot-Savart parity on identical coils, points and cotangents."""
        self._assert_coil_current_list_parity(
            "d2A_by_dXdX",
            "d3A_by_dXdXdcoilcurrents",
            rtol=_DERIVATIVE_HEAVY_TOLS["second_derivative_rtol"],
            atol=_DERIVATIVE_HEAVY_TOLS["second_derivative_atol"],
        )


class TestJaxBiotSavartCoilStateToken(JaxTestCase):
    """Traceable runtime cache invalidation is keyed to coil DOF state."""

    @staticmethod
    def _make_two_basic_coils():

        coils = []
        for current_amp in (1.0e6, -1.0e6):
            curve = CurveXYZFourier(quadpoints=16, order=1)
            curve.x = np.array(
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
            coils.append(Coil(curve, Current(current_amp)))
        return coils

    @staticmethod
    def _coil_arrays_in_original_order(coil_set_spec):
        coil_count = sum(len(group.coil_indices) for group in coil_set_spec.groups)
        gammas = [None] * coil_count
        gammadashs = [None] * coil_count
        currents = [None] * coil_count
        for group in coil_set_spec.groups:
            for position, coil_index in enumerate(group.coil_indices):
                gammas[coil_index] = group.gammas[position]
                gammadashs[coil_index] = group.gammadashs[position]
                currents[coil_index] = group.currents[position]
        return gammas, gammadashs, currents

    @staticmethod
    def _assert_per_coil_entries_equal(actual_entries, expected_entries):
        assert len(actual_entries) == len(expected_entries)
        for actual, expected in zip(actual_entries, expected_entries, strict=True):
            np.testing.assert_allclose(
                np.asarray(actual),
                np.asarray(expected),
                rtol=0.0,
                atol=0.0,
            )

    def test_live_coil_set_spec_matches_explicit_dofs_and_native_coils(self):
        """The cached live spec is the explicit-DOF reconstruction of the native coils."""
        coils = self._make_two_basic_coils()
        bs_jax = JaxBiotSavart(coils)

        live = self._coil_arrays_in_original_order(bs_jax.coil_set_spec())
        explicit = self._coil_arrays_in_original_order(
            bs_jax.coil_set_spec_from_dofs(bs_jax.x)
        )
        native = (
            [coil.curve.gamma() for coil in coils],
            [coil.curve.gammadash() for coil in coils],
            [coil.current.get_value() for coil in coils],
        )

        for live_entries, explicit_entries, native_entries in zip(
            live, explicit, native, strict=True
        ):
            for actual, from_dofs, expected in zip(
                live_entries, explicit_entries, native_entries, strict=True
            ):
                np.testing.assert_allclose(
                    cast(jax.Array, actual),
                    cast(jax.Array, from_dofs),
                    rtol=1e-14,
                    atol=1e-15,
                )
                np.testing.assert_allclose(
                    cast(jax.Array, actual),
                    expected,
                    rtol=1e-13,
                    atol=1e-14,
                )

    def test_biotsavart_extraction_spec_changes_only_for_captured_dof_contract(self):
        """Free-DOF changes reuse extraction specs; layout or captured fixed-DOF
        changes replace them."""
        coils = self._make_two_basic_coils()
        field = JaxBiotSavart(list(coils))
        curve = coils[0].curve
        initial_spec = field.coil_dof_extraction_spec()
        initial_value = float(curve.local_full_x[0])

        curve.set(0, initial_value + 1.0e-3)
        self.assertTrue(
            field.coil_dof_extraction_spec() is initial_spec,
            "field.coil_dof_extraction_spec() is initial_spec",
        )

        curve.fix(0)
        fixed_spec = field.coil_dof_extraction_spec()
        self.assertTrue(
            fixed_spec is not initial_spec, "fixed_spec is not initial_spec"
        )

        curve.set(0, initial_value + 2.0e-3)
        self.assertTrue(
            field.coil_dof_extraction_spec() is not fixed_spec,
            "field.coil_dof_extraction_spec() is not fixed_spec",
        )
