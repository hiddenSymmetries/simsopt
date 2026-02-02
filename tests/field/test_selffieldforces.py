import unittest
import logging

import numpy as np
from scipy import constants
from scipy.interpolate import interp1d
from scipy.special import ellipk, ellipe

from simsopt.field import (
    Coil, RegularizedCoil, CircularRegularizedCoil, RectangularRegularizedCoil, 
    Current, coils_via_symmetries,
    _coil_coil_inductances_pure,
    _coil_coil_inductances_inv_pure,
    _induced_currents_pure,
    NetFluxes,
    B2Energy,
    LpCurveTorque,
    SquaredMeanTorque,
    LpCurveForce,
    SquaredMeanForce,
)
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.configs import get_data
from simsopt.geo import CurveXYZFourier, JaxCurveXYZFourier, CurvePlanarFourier, JaxCurvePlanarFourier
from simsopt.field.selffield import (
    _rectangular_xsection_k,
    _rectangular_xsection_delta,
    regularization_circ,
    regularization_rect,
)

logger = logging.getLogger(__name__)


class SpecialFunctionsTests(unittest.TestCase):
    """
    Test the functions that are specific to the reduced model for rectangular
    cross-section coils.
    """

    def test_k_square(self):
        """Check value of k for a square cross-section."""
        truth = 2.556493222766492
        np.testing.assert_allclose(_rectangular_xsection_k(0.3, 0.3), truth)
        np.testing.assert_allclose(_rectangular_xsection_k(2.7, 2.7), truth)

    def test_delta_square(self):
        """Check value of delta for a square cross-section."""
        truth = 0.19985294779417703
        np.testing.assert_allclose(_rectangular_xsection_delta(0.3, 0.3), truth)
        np.testing.assert_allclose(_rectangular_xsection_delta(2.7, 2.7), truth)

    def test_symmetry(self):
        """k and delta should be unchanged if a and b are swapped."""
        d = 0.01  # Geometric mean of a and b
        for ratio in [0.1, 3.7]:
            a = d * ratio
            b = d / ratio
            np.testing.assert_allclose(
                _rectangular_xsection_delta(a, b), _rectangular_xsection_delta(b, a)
            )
            np.testing.assert_allclose(
                _rectangular_xsection_k(a, b), _rectangular_xsection_k(b, a)
            )

    def test_limits(self):
        """Check limits of k and delta for a >> b and b >> a."""
        ratios = [1.1e6, 2.2e4, 3.5e5]
        xs = [0.2, 1.0, 7.3]
        for ratio in ratios:
            for x in xs:
                # a >> b
                b = x
                a = b * ratio
                np.testing.assert_allclose(_rectangular_xsection_k(a, b), (7.0 / 6) + np.log(a / b), rtol=1e-3)
                np.testing.assert_allclose(_rectangular_xsection_delta(a, b), a / (b * np.exp(3)), rtol=1e-3)

                # b >> a
                a = x
                b = ratio * a
                np.testing.assert_allclose(_rectangular_xsection_k(a, b), (7.0 / 6) + np.log(b / a), rtol=1e-3)
                np.testing.assert_allclose(_rectangular_xsection_delta(a, b), b / (a * np.exp(3)), rtol=1e-3)

    def test_regularization_circ(self):
        """Test regularization_circ function."""
        a = 0.01
        reg_circ = regularization_circ(a)
        expected = a**2 / np.sqrt(np.e)
        np.testing.assert_allclose(reg_circ, expected, rtol=1e-10)
        
        # Test with different radius
        a2 = 0.05
        reg_circ2 = regularization_circ(a2)
        expected2 = a2**2 / np.sqrt(np.e)
        np.testing.assert_allclose(reg_circ2, expected2, rtol=1e-10)
        
        # Verify scaling: should scale as a^2
        np.testing.assert_allclose(reg_circ2 / reg_circ, (a2 / a)**2, rtol=1e-10)

    def test_regularization_rect(self):
        """Test regularization_rect function."""
        a = 0.01
        b = 0.023
        
        # Test square cross-section
        reg_rect_square = regularization_rect(a, a)
        reg_circ_equiv = regularization_circ(a)
        # For square, should be close to circular with same area
        # (not exact, but should be similar order of magnitude)
        assert reg_rect_square > 0
        assert reg_circ_equiv > 0
        
        # Test rectangular cross-section
        reg_rect = regularization_rect(a, b)
        expected_delta = _rectangular_xsection_delta(a, b)
        expected = a * b * expected_delta
        np.testing.assert_allclose(reg_rect, expected, rtol=1e-10)
        
        # Test symmetry: should be same if a and b are swapped
        reg_rect_swapped = regularization_rect(b, a)
        np.testing.assert_allclose(reg_rect, reg_rect_swapped, rtol=1e-10)
        
        # Test with different dimensions
        a2 = 0.02
        b2 = 0.03
        reg_rect2 = regularization_rect(a2, b2)
        assert reg_rect2 > 0
        # Should scale roughly with area
        assert reg_rect2 > reg_rect  # Larger dimensions should give larger regularization


class CoilForcesTest(unittest.TestCase):

    def test_circular_coil(self):
        """Check whether B_reg and hoop force on a circular-centerline coil are correct."""
        R0 = 1.7
        I = 1e5
        a = 0.01
        b = 0.023
        order = 1
        R1 = 40.0
        R2 = 3.0
        d = 5.0

        # Analytic field has only a z component
        B_reg_analytic_circ = constants.mu_0 * I / (4 * np.pi * R0) * (np.log(8 * R0 / a) - 3 / 4)
        force_analytic_circ = B_reg_analytic_circ * I

        # For two concentric circular coils, only "analytic" for R1 >> R0
        Lij_analytic = constants.mu_0 * np.pi * R0 ** 2 / (2 * R1)
        # self_inductance_analytic = constants.mu_0 * R0 * (np.log(8 * R0 / a) - 7.0 / 4.0)

        # For two coils that share a common axis
        k = np.sqrt(4.0 * R0 * R2 / ((R0 + R2) ** 2 + d ** 2))
        Lij_analytic2 = constants.mu_0 * np.sqrt(R0 * R2) * (
            (2 / k - k) * ellipk(k ** 2) - (2 / k) * ellipe(k ** 2)
        )

        # Eq (98) in Landreman Hurwitz Antonsen:
        B_reg_analytic_rect = constants.mu_0 * I / (4 * np.pi * R0) * (
            np.log(8 * R0 / np.sqrt(a * b)) + 13.0 / 12 - _rectangular_xsection_k(a, b) / 2
        )
        force_analytic_rect = B_reg_analytic_rect * I

        # Very large number of quadrature points is required to accurately compute the self-inductance
        # so need to implement Siena's fast quadrature scheme. Not testing the self-inductance here 
        # for that reason.
        for N_quad in [500]:
            for downsample in [2, 3, 4]:

                # Create a circle of radius R0 in the x-y plane:
                curve = CurveXYZFourier(N_quad, order)
                curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
                curve2 = CurveXYZFourier(N_quad, order)
                curve2.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
                phi = 2 * np.pi * curve.quadpoints

                current = Current(I)
                coil = RegularizedCoil(curve, current, regularization_circ(a))

                # Check the case of circular cross-section:
                B_reg_test = coil.B_regularized()
                np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_circ)
                np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

                # Test self-force for circular coil
                force_test = coil.self_force()
                np.testing.assert_allclose(force_test[:, 0], force_analytic_circ * np.cos(phi))
                np.testing.assert_allclose(force_test[:, 1], force_analytic_circ * np.sin(phi))
                np.testing.assert_allclose(force_test[:, 2], 0.0)

                # Test self-torque for circular coil (should be zero by symmetry)
                # Analytic torque for a perfect circle is zero
                torque_test = coil.torque([coil])
                np.testing.assert_allclose(torque_test, 0.0, atol=1e-9)

                normal = [0, 0, 1]
                alpha = np.arcsin(normal[1])
                delta = np.arccos(normal[2] / np.cos(alpha))
                curve = CurvePlanarFourier(N_quad, 0)
                dofs = np.zeros(8)
                dofs[0] = R0
                dofs[1] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
                dofs[2] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
                dofs[3] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
                dofs[4] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
                # Now specify the center
                dofs[5] = 0.0
                dofs[6] = 0.0
                dofs[7] = 0.0
                curve.set_dofs(dofs)

                # Make concentric coil with larger radius
                curve2 = CurvePlanarFourier(N_quad, 0)
                dofs2 = np.zeros(8)
                dofs2[0] = R1
                dofs2[1] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
                dofs2[2] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
                dofs2[3] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
                dofs2[4] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
                # Now specify the center
                dofs2[5] = 0.0
                dofs2[6] = 0.0
                dofs2[7] = 0.0
                curve2.set_dofs(dofs2)

                # Make circular coil with shared axis
                curve3 = CurvePlanarFourier(N_quad * 2, 0)
                dofs3 = np.zeros(8)
                dofs3[0] = R2
                dofs3[1] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)    
                dofs3[2] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
                dofs3[3] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
                dofs3[4] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
                # Now specify the center
                dofs3[5] = 0.0
                dofs3[6] = 0.0
                dofs3[7] = d
                curve3.set_dofs(dofs3)

                Lij = _coil_coil_inductances_pure(
                    np.array([curve.gamma(), curve2.gamma()]),
                    np.array([curve.gammadash(), curve2.gammadash()]),
                    downsample=downsample,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                np.testing.assert_allclose(Lij[1, 0], Lij_analytic, rtol=1e-2)
                # np.testing.assert_allclose(Lij[0, 0], self_inductance_analytic, rtol=1e-2)

                Lij_no_downsample = _coil_coil_inductances_pure(
                    np.array([curve.gamma(), curve2.gamma()]),
                    np.array([curve.gammadash(), curve2.gammadash()]),
                    downsample=1,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                # Only off-diagonal will agree because self-inductance accuracy needs many more quadrature points
                np.testing.assert_allclose(Lij_no_downsample[1, 0], Lij[1, 0], rtol=1e-2)

                # Test rectangular cross section for a << R
                Lij_rect = _coil_coil_inductances_pure(
                    np.array([curve.gamma(), curve2.gamma()]),
                    np.array([curve.gammadash(), curve2.gammadash()]),
                    downsample=downsample,
                    regularizations=np.array([regularization_rect(a, b), regularization_rect(a, b)]),
                )
                np.testing.assert_allclose(Lij_rect[1, 0], Lij[1, 0], rtol=1e-2)  # rectangular is not so different from circular

                Lij_rect_no_downsample = _coil_coil_inductances_pure(
                    np.array([curve.gamma(), curve2.gamma()]),
                    np.array([curve.gammadash(), curve2.gammadash()]),
                    downsample=1,
                    regularizations=np.array([regularization_rect(a, b), regularization_rect(a, b)]),
                )
                np.testing.assert_allclose(Lij_rect_no_downsample[1, 0], Lij_rect[1, 0], rtol=1e-2)

                # retry but swap the coils
                Lji = _coil_coil_inductances_pure(
                    np.array([curve2.gamma(), curve.gamma()]),
                    np.array([curve2.gammadash(), curve.gammadash()]),
                    downsample=downsample,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                print(Lij)
                print(Lji)
                assert np.allclose(Lji[1, 0], Lij[0, 1])
                assert np.allclose(Lji[0, 0], Lij[1, 1])
                assert np.allclose(Lji[1, 1], Lij[0, 0])
                np.testing.assert_allclose(Lji[1, 0], Lij_analytic, rtol=1e-2)

                # now test coils with shared axis
                Lij3 = _coil_coil_inductances_pure(
                    [curve.gamma(), curve3.gamma()],
                    [curve.gammadash(), curve3.gammadash()],
                    downsample=downsample,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                np.testing.assert_allclose(Lij3[1, 0], Lij_analytic2, rtol=1e-2)

                Lij3_no_downsample = _coil_coil_inductances_pure(
                    [curve.gamma(), curve3.gamma()],
                    [curve.gammadash(), curve3.gammadash()],
                    downsample=1,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                np.testing.assert_allclose(Lij3[1, 0], Lij3_no_downsample[1, 0], rtol=1e-2)

                # This function is really for passive coils 
                # but just checking we can compute the induced currents correctly
                induced_currents_test = _induced_currents_pure(
                    np.array([curve.gamma()]),
                    np.array([curve.gammadash()]),
                    np.array([curve2.gamma()]),
                    np.array([curve2.gammadash()]),
                    np.array([1e6]),
                    downsample=downsample,
                    regularizations=np.array([regularization_circ(a)]),
                )
                assert np.all(np.abs(induced_currents_test) > 1e3)

                # Test cholesky computation of the inverse works on simple case
                Lij_inv = _coil_coil_inductances_inv_pure(
                    [curve.gamma(), curve3.gamma()],
                    [curve.gammadash(), curve3.gammadash()],
                    downsample=downsample,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                assert np.allclose(np.linalg.inv(Lij3), Lij_inv)

                # Check the case of rectangular cross-section:
                coil_rect = RegularizedCoil(curve, current, regularization_rect(a, b))
                B_reg_test = coil_rect.B_regularized()
                np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_rect)
                np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

                force_test = coil_rect.self_force()
                np.testing.assert_allclose(force_test[:, 0], force_analytic_rect * np.cos(phi))
                np.testing.assert_allclose(force_test[:, 1], force_analytic_rect * np.sin(phi))
                np.testing.assert_allclose(force_test[:, 2], 0.0)

                # Test self-torque for rectangular cross-section coil (should also be zero by symmetry)
                # Analytic torque for a perfect rectangle is zero
                torque_test_rect = coil_rect.torque([coil_rect])
                np.testing.assert_allclose(torque_test_rect, 0.0, atol=1e-9)

                # --- Two concentric circular coils: test mutual torque ---
                # Both in xy-plane, same center, different radii
                curve_inner = CurveXYZFourier(N_quad, order)
                curve_inner.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
                curve_outer = CurveXYZFourier(N_quad, order)
                curve_outer.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R1
                current_inner = Current(I)
                current_outer = Current(I)
                coil_inner = RegularizedCoil(curve_inner, current_inner, regularization_circ(a))
                coil_outer = RegularizedCoil(curve_outer, current_outer, regularization_circ(a))
                # Compute torque on each coil due to both
                torque_inner = coil_inner.torque([coil_inner, coil_outer])
                torque_outer = coil_outer.torque([coil_inner, coil_outer])
                # By symmetry, both should be zero
                np.testing.assert_allclose(torque_inner, 0.0, atol=1e-8)
                np.testing.assert_allclose(torque_outer, 0.0, atol=1e-8)

                # --- LpCurveTorque objective should also be zero ---
                obj1 = LpCurveTorque(coil_inner, coil_outer, p=2.0, threshold=0.0)
                val1 = obj1.J()
                np.testing.assert_allclose(val1, 0.0, atol=1e-1)
                # Outer as group 1, inner as group 2
                obj2 = LpCurveTorque(coil_outer, coil_inner, p=2.0, threshold=0.0)
                val2 = obj2.J()
                np.testing.assert_allclose(val2, 0.0, atol=1e-1)

                # --- Net force on each coil should also be zero ---
                net_force_inner = np.sum(coil_inner.force([coil_inner, coil_outer]), axis=0)
                net_force_outer = np.sum(coil_outer.force([coil_inner, coil_outer]), axis=0)
                np.testing.assert_allclose(net_force_inner, 0.0, atol=1e-6)
                np.testing.assert_allclose(net_force_outer, 0.0, atol=1e-6)

                # --- Two circular coils, separated along z but sharing a common axis: torque should be zero ---
                coil_z1 = RegularizedCoil(curve2, Current(I), regularization_circ(a))
                coil_z2 = RegularizedCoil(curve3, Current(I), regularization_circ(a))
                np.testing.assert_allclose(curve3.centroid(), [0, 0, 5], atol=1e-10)
                np.testing.assert_allclose(curve2.centroid(), [0, 0, 0], atol=1e-10)
                torque_z1 = coil_z1.torque([coil_z1, coil_z2])
                torque_z2 = coil_z2.torque([coil_z1, coil_z2])
                np.testing.assert_allclose(np.sum(torque_z1, axis=0), 0.0, atol=1e-8)
                np.testing.assert_allclose(np.sum(torque_z2, axis=0), 0.0, atol=1e-8)

                # --- JAX CurveXYZFourier: check equivalence ---
                jax_curve = JaxCurveXYZFourier(N_quad, order)
                jax_curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
                jax_coil = RegularizedCoil(jax_curve, current, regularization_circ(a))
                # B_regularized_circ
                B_reg_jax = jax_coil.B_regularized()
                np.testing.assert_allclose(B_reg_jax, B_reg_test, rtol=1e-2, atol=1e-12)
                # self_force_circ
                force_jax = jax_coil.self_force()
                np.testing.assert_allclose(force_jax, force_test, rtol=1e-2, atol=1e-12)
                # self-torque
                torque_jax = jax_coil.torque([jax_coil])
                np.testing.assert_allclose(torque_jax, torque_test, rtol=1e-2, atol=1e-10)

                # --- JAX CurvePlanarFourier: check equivalence ---
                jax_curve_p = JaxCurvePlanarFourier(N_quad, 0)
                jax_curve_p.set_dofs(dofs)
                jax_curve2_p = JaxCurvePlanarFourier(N_quad, 0)
                jax_curve2_p.set_dofs(dofs2)
                jax_curve3_p = JaxCurvePlanarFourier(N_quad * 2, 0)
                jax_curve3_p.set_dofs(dofs3)
                # Check JAX and non-JAX curves are equivalent
                np.testing.assert_allclose(jax_curve_p.gamma(), curve.gamma(), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(jax_curve_p.gammadash(), curve.gammadash(), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(jax_curve2_p.gamma(), curve2.gamma(), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(jax_curve2_p.gammadash(), curve2.gammadash(), rtol=1e-12, atol=1e-12)
                # Inductance
                Lij_jax = _coil_coil_inductances_pure(
                    np.array([jax_curve_p.gamma(), jax_curve2_p.gamma()]),
                    np.array([jax_curve_p.gammadash(), jax_curve2_p.gammadash()]),
                    downsample=downsample,
                    regularizations=np.array([regularization_circ(a), regularization_circ(a)]),
                )
                np.testing.assert_allclose(Lij_jax, Lij, rtol=1e-2, atol=1e-12)
                # Rectangular cross section
                Lij_rect_jax = _coil_coil_inductances_pure(
                    np.array([jax_curve_p.gamma(), jax_curve2_p.gamma()]),
                    np.array([jax_curve_p.gammadash(), jax_curve2_p.gammadash()]),
                    downsample=downsample,
                    regularizations=np.array([regularization_rect(a, b), regularization_rect(a, b)]),
                )
                np.testing.assert_allclose(Lij_rect_jax, Lij_rect, rtol=1e-2, atol=1e-12)
                # B_regularized_rect
                jax_coil_rect = RegularizedCoil(jax_curve_p, current, regularization_rect(a, b))
                B_reg_rect_jax = jax_coil_rect.B_regularized()
                np.testing.assert_allclose(B_reg_rect_jax, B_reg_test, rtol=1e-2, atol=1e-12)
                # self_force_rect
                force_rect_jax = jax_coil_rect.self_force()
                np.testing.assert_allclose(force_rect_jax, force_test, rtol=1e-2, atol=1e-12)
                # self-torque rect
                torque_rect_jax = jax_coil_rect.torque([jax_coil_rect])
                np.testing.assert_allclose(torque_rect_jax, torque_test_rect, rtol=1e-2, atol=1e-10)
            
            current = Current(I)
            _ = Coil(curve, current)

            # Check the case of circular cross-section:
            coil_circ = CircularRegularizedCoil(curve, current, a)
            B_reg_test = coil_circ.B_regularized()
            np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_circ)
            np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

            force_test = coil_circ.self_force()
            np.testing.assert_allclose(force_test[:, 0], force_analytic_circ * np.cos(phi))
            np.testing.assert_allclose(force_test[:, 1], force_analytic_circ * np.sin(phi))
            np.testing.assert_allclose(force_test[:, 2], 0.0)

            # Check the case of rectangular cross-section:
            coil_rect = RectangularRegularizedCoil(curve, current, a, b)
            B_reg_test = coil_rect.B_regularized()
            np.testing.assert_allclose(B_reg_test[:, 2], B_reg_analytic_rect)
            np.testing.assert_allclose(B_reg_test[:, 0:2], 0)

            force_test = coil_rect.self_force()
            np.testing.assert_allclose(force_test[:, 0], force_analytic_rect * np.cos(phi))
            np.testing.assert_allclose(force_test[:, 1], force_analytic_rect * np.sin(phi))
            np.testing.assert_allclose(force_test[:, 2], 0.0)

    def test_force_convergence(self):
        """Check that the self-force is approximately independent of the number of quadrature points"""
        points_per_periods = [8, 4, 2, 7, 5]
        for j, points_per_period in enumerate(points_per_periods):
            base_curves, base_currents, ma, nfp, bs = get_data("hsx", points_per_period=points_per_period)
            curve = base_curves[0]
            I = 1.5e3
            a = 0.01
            coil = CircularRegularizedCoil(curve, Current(I), a)
            force = coil.self_force()
            max_force = np.max(np.abs(force))
            if j == 0:
                interpolant = interp1d(curve.quadpoints, force, axis=0)
                max_force_ref = max_force
            else:
                np.testing.assert_allclose(force, interpolant(curve.quadpoints), atol=max_force_ref / 60)

    def test_hsx_coil(self):
        """Compare self-force for HSX coil 1 to result from CoilForces.jl"""
        base_curves, base_currents, ma, nfp, bs  = get_data("hsx")
        assert len(base_curves[0].quadpoints) == 160
        I = 150e3
        a = 0.01
        b = 0.023
        coil_circ = CircularRegularizedCoil(base_curves[0], Current(I), a)
        coil_rect = RectangularRegularizedCoil(base_curves[0], Current(I), a, b)

        # Case of circular cross-section

        # The data from CoilForces.jl were generated with the command
        # CoilForces.reference_HSX_force_for_simsopt_test()
        F_x_benchmark = np.array(
            [-15624.06752062059, -21673.892879345873, -27805.92218896322, -33138.2025931857, -36514.62850757798, -37154.811045050716, -35224.36483811566, -31790.6909934216, -28271.570764376913, -25877.063414550663, -25275.54000792784, -26426.552957555898, -28608.08732785721, -30742.66146788618, -31901.1192650387, -31658.2982018783, -30115.01252455622, -27693.625158453917, -24916.97602450875, -22268.001550194127, -20113.123569572494, -18657.02934190755, -17925.729621918534, -17787.670352261383, -18012.98424762069, -18355.612668419068, -18631.130455525174, -18762.19098176415, -18778.162916012046, -18776.500656205895, -18866.881771744567, -19120.832848894337, -19543.090214569205, -20070.954769137115, -20598.194181114803, -21013.020202255055, -21236.028702664324, -21244.690600996386, -21076.947768954156, -20815.355048694666, -20560.007956111527, -20400.310604802795, -20393.566682281307, -20554.83647318684, -20858.986285059094, -21253.088938981215, -21675.620708707665, -22078.139271497712, -22445.18444801059, -22808.75225496607, -23254.130115531163, -23913.827617806084, -24946.957266144746, -26504.403695291898, -28685.32300927181, -31495.471071978012, -34819.49374359714, -38414.82789487393, -41923.29333627555, -44885.22293635466, -46749.75134352123, -46917.59025432583, -44896.50887106118, -40598.462003586974, -34608.57105847433, -28108.332731765862, -22356.321253373, -18075.405570497107, -15192.820251877345, -13027.925896696135, -10728.68775277632, -7731.104577216556, -4026.458734812997, -67.65800705092924, 3603.7480987311537, 6685.7274727329805, 9170.743233515725, 11193.25631660189, 12863.446736995473, 14199.174999621611, 15157.063376046968, 15709.513692788054, 15907.086239630167, 15889.032882713132, 15843.097529146156, 15944.109516240991, 16304.199171854023, 16953.280592130628, 17852.57440796256, 18932.066168700923, 20133.516941300426, 21437.167716977303, 22858.402963585464, 24417.568974489524, 26100.277202379944, 27828.811426061613, 29459.771430218898, 30813.7836860175, 31730.62350657151, 32128.502820609796, 32038.429339023023, 31593.803847403953, 30979.028723505002, 30362.077268204735, 29840.850204702965, 29422.877198133527, 29042.28057709125, 28604.02774189412, 28036.121314230902, 27327.793860493435, 26538.11580899982, 25773.01411179288, 25142.696104375616, 24718.6066327647, 24507.334842447635, 24451.10991168722, 24454.085831995577, 24423.536258237124, 24308.931868210013, 24122.627773352768, 23933.764307662732, 23838.57162949479, 23919.941100154054, 24212.798983180386, 24689.158548635372, 25269.212310785344, 25854.347267952628, 26368.228758087153, 26787.918123459167, 27150.79244000832, 27533.348289627098, 28010.279752667528, 28611.021858534772, 29293.073660468486, 29946.40958260143, 30430.92513540546, 30631.564524187717, 30503.197269324868, 30080.279217014842, 29444.6938562621, 28667.38229651914, 27753.348490269695, 26621.137071620036, 25137.82866539427, 23205.371963209964, 20853.92976118877, 18273.842305983166, 15753.018584850472, 13562.095187201534, 11864.517807863573, 10688.16332321768, 9935.766441264674, 9398.023223792645, 8766.844594289494, 7680.841209848606, 5824.4042671660145, 3040.702284846631, -630.2054351866387, -5035.57692055936, -10048.785939525675]
        )

        F_x_test = coil_circ.self_force()[:, 0]
        np.testing.assert_allclose(F_x_benchmark, F_x_test, rtol=1e-9, atol=0)

        # Case of rectangular cross-section

        F_x_benchmark = np.array(
            [-15905.20099921593, -22089.84960387874, -28376.348489470365, -33849.08438046449, -37297.138833218974, -37901.3580214951, -35838.71064362283, -32228.643120480687, -28546.9118841109, -26046.96628692484, -25421.777194138715, -26630.791911489407, -28919.842325785943, -31157.40078884933, -32368.19957740524, -32111.184287572887, -30498.330514718982, -27974.45692852191, -25085.400672446423, -22334.49737678633, -20104.78648017159, -18610.931535243944, -17878.995292047493, -17767.35330442759, -18030.259902092654, -18406.512856357545, -18702.39969540496, -18838.862854941028, -18849.823944445518, -18840.62799920807, -18928.85330885538, -19191.02138695175, -19632.210519767978, -20185.474968977625, -20737.621297822592, -21169.977809582055, -21398.747768091078, -21400.62658689198, -21216.133558586924, -20932.595132161085, -20655.60793743372, -20479.40191077005, -20464.28582628529, -20625.83431400738, -20936.962932518098, -21341.067527434556, -21772.38656616101, -22178.862986210577, -22542.999300185398, -22897.045487538875, -23329.342412912913, -23978.387795050137, -25011.595805992223, -26588.8272541588, -28816.499234411625, -31703.566987071903, -35132.3971671138, -38852.71510558583, -42494.50815372789, -45583.48852415488, -47551.1577527285, -47776.415427331594, -45743.97982645536, -41354.37991615283, -35210.20495138465, -28540.23742988024, -22654.55869049082, -18301.96907423793, -15401.963398143102, -13243.762349314706, -10939.450828758423, -7900.820612170931, -4120.028225769904, -72.86209546891608, 3674.253747922276, 6809.0803070326565, 9328.115750414787, 11374.122069162511, 13062.097330371573, 14409.383808494194, 15369.251684718018, 15911.988418337934, 16090.021555975769, 16048.21613878066, 15981.151899412167, 16068.941633738388, 16425.88464448961, 17080.88532516404, 17992.129241265648, 19086.46631302506, 20304.322975363317, 21627.219065732254, 23073.563938875737, 24666.38845701993, 26391.47816311481, 28167.521012668185, 29843.93199662863, 31232.367301229497, 32164.969954389788, 32556.923587447265, 32442.446350951064, 31963.284032424053, 31314.01211399212, 30670.79551082286, 30135.039340095944, 29712.330052677768, 29330.71025802117, 28887.8200773726, 28306.412420411067, 27574.83013193789, 26755.843397583598, 25961.936385889934, 25310.01540139794, 24875.789463354584, 24666.066357125907, 24619.136261928328, 24632.619408002214, 24607.413073397413, 24489.503028993608, 24292.044623409187, 24088.74651990258, 23982.195361428472, 24060.929104794097, 24362.6460843878, 24858.082439252874, 25462.457564195745, 26070.50973682213, 26600.547196554344, 27028.01270305341, 27393.03996450607, 27777.872708277075, 28263.357416931998, 28882.7902495421, 29593.307386932454, 30279.887846398404, 30794.507327329207, 31014.791285198782, 30892.485429183558, 30464.50108998591, 29819.03800239511, 29033.577206319136, 28116.32127507844, 26983.626000124084, 25495.394951521277, 23544.852551314456, 21157.350595114454, 18526.131317622883, 15948.394109661942, 13705.248433750054, 11967.480036214449, 10766.293968812726, 10004.685998499026, 9470.706025372589, 8849.607342610005, 7769.149525451194, 5902.017638994769, 3084.6416074691333, -641.878548205229, -5119.944566458021, -10221.371299891642]
        )

        F_x_test = coil_rect.self_force()[:, 0]
        np.testing.assert_allclose(F_x_benchmark, F_x_test, rtol=1e-9, atol=0)

    def test_coil_force_requires_regularized_coil(self):
        """Test that coil_force raises an error when given a Coil instead of RegularizedCoil"""
        nfp = 3
        ncoils = 4
        I = 1.7e4

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)

        # Methods should raise an AttributeError when called on a regular Coil
        with self.assertRaises(AttributeError):
            coils[0].force(coils)

        with self.assertRaises(AttributeError):
            coils[0].torque(coils)

        with self.assertRaises(AttributeError):
            coils[0].net_force(coils)

        with self.assertRaises(AttributeError):
            coils[0].net_torque(coils)

    def test_net_force_and_torque(self):
        """Test coil_net_force and coil_net_torque functions."""
        nfp = 3
        ncoils = 4
        I = 1.7e4
        regularization = regularization_circ(0.05)

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True,
                                     regularizations=[regularization] * ncoils)

        # Test coil_net_force: should be the integral of pointwise forces
        target_coil = coils[0]
        source_coils = coils
        
        # Compute pointwise forces
        pointwise_forces = target_coil.force(source_coils)
        
        # Compute net force using the method
        net_force = target_coil.net_force(source_coils)
        
        # Compute net force manually by integrating pointwise forces
        gammadash = target_coil.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_force_manual = np.sum(gammadash_norm * pointwise_forces, axis=0) / gammadash.shape[0]
        
        np.testing.assert_allclose(net_force, net_force_manual, rtol=1e-10,
                                   err_msg="net_force should match manual integration")
        
        # Test coil_net_torque: should be the integral of pointwise torques
        pointwise_torques = target_coil.torque(source_coils)
        
        # Compute net torque using the method
        net_torque = target_coil.net_torque(source_coils)
        
        # Compute net torque manually by integrating pointwise torques
        net_torque_manual = np.sum(gammadash_norm * pointwise_torques, axis=0) / gammadash.shape[0]
        
        np.testing.assert_allclose(net_torque, net_torque_manual, rtol=1e-10,
                                   err_msg="net_torque should match manual integration")
        
        # Test with rectangular regularization
        regularization_rect_val = regularization_rect(0.01, 0.023)
        coils_rect = coils_via_symmetries(base_curves, base_currents, nfp, True,
                                          regularizations=[regularization_rect_val] * ncoils)
        
        target_coil_rect = coils_rect[0]
        source_coils_rect = coils_rect
        
        # Test coil_net_force with rectangular regularization: should be the integral of pointwise forces
        pointwise_forces_rect = target_coil_rect.force(source_coils_rect)
        net_force_rect = target_coil_rect.net_force(source_coils_rect)
        
        # Compute net force manually by integrating pointwise forces
        gammadash_rect = target_coil_rect.curve.gammadash()
        gammadash_norm_rect = np.linalg.norm(gammadash_rect, axis=1)[:, None]
        net_force_manual_rect = np.sum(gammadash_norm_rect * pointwise_forces_rect, axis=0) / gammadash_rect.shape[0]
        
        np.testing.assert_allclose(net_force_rect, net_force_manual_rect, rtol=1e-10,
                                   err_msg="net_force with rectangular regularization should match manual integration")
        
        # Test coil_net_torque with rectangular regularization: should be the integral of pointwise torques
        pointwise_torques_rect = target_coil_rect.torque(source_coils_rect)
        net_torque_rect = target_coil_rect.net_torque(source_coils_rect)
        
        # Compute net torque manually by integrating pointwise torques
        net_torque_manual_rect = np.sum(gammadash_norm_rect * pointwise_torques_rect, axis=0) / gammadash_rect.shape[0]
        
        np.testing.assert_allclose(net_torque_rect, net_torque_manual_rect, rtol=1e-10,
                                   err_msg="net_torque with rectangular regularization should match manual integration")
        
        # Net force and torque should be finite and reasonable
        assert np.all(np.isfinite(net_force_rect))
        assert np.all(np.isfinite(net_torque_rect))
        assert np.linalg.norm(net_force_rect) > 0  # Should be non-zero for interacting coils
        assert np.linalg.norm(net_torque_rect) > 0  # Should be non-zero for interacting coils

    def test_force_objectives(self):
        """Check whether objective function matches function for export"""
        nfp = 3
        ncoils = 4
        I = 1.7e4

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True, 
                                     regularizations=[regularization_circ(0.05)] * ncoils)
        # Test B2Energy
        objective = B2Energy(coils).J()

        # Test LpCurveForce
        p = 2.5
        threshold = 1e-3  # Threshold in MN/m (equivalent to 1.0e3 N/m)
        objective = float(LpCurveForce(coils[0], coils, p=p, threshold=threshold).J())
        dJ = LpCurveForce(coils[0], coils, p=p, threshold=threshold).dJ()
        np.testing.assert_allclose(dJ.shape, (ncoils * len(coils[0].x),))

        # Now compute the objective a different way, using the independent
        # coil_force function
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        force_norm_N_per_m = np.linalg.norm(coils[0].force(coils), axis=1)
        force_norm_MN_per_m = force_norm_N_per_m / 1e6  # Convert to MN/m
        print("force_norm mean:", np.mean(force_norm_N_per_m), "max:", np.max(force_norm_N_per_m))
        objective_alt = (1 / p) * np.sum(np.maximum(force_norm_MN_per_m - threshold, 0)**p * gammadash_norm) / np.shape(gammadash_norm)[0]

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        # Test SquaredMeanForce
        objective = float(SquaredMeanForce(coils[0], coils).J())
        dJ = SquaredMeanForce(coils[0], coils).dJ()
        np.testing.assert_allclose(dJ.shape, (ncoils * len(coils[0].x),))

        # Now compute the objective a different way, using the independent
        # force method
        # SquaredMeanForce computes: ||mean_force||^2 where mean_force = sum(force * gammadash_norm) / npts
        forces = coils[0].force(coils)
        net_force_N_per_m = np.sum(forces * gammadash_norm[:, None], axis=0) / len(gammadash_norm)
        net_force_MN_per_m = net_force_N_per_m / 1e6  # Convert to MN/m
        objective_alt = np.linalg.norm(net_force_MN_per_m) ** 2

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        # Test SquaredMeanForce
        p = 2.5
        objective = SquaredMeanForce(coils[0], coils).J()

        # Now compute the objective a different way, using the independent
        # force method
        # SquaredMeanForce computes: ||mean_force||^2 where mean_force = sum(force * gammadash_norm) / npts
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        forces = coils[0].force(coils)
        net_force_N_per_m = np.sum(forces * gammadash_norm[:, None], axis=0) / len(gammadash_norm)
        net_force_MN_per_m = net_force_N_per_m / 1e6  # Convert to MN/m
        objective_alt = np.linalg.norm(net_force_MN_per_m) ** 2

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        # Test SquaredMeanForce vs SquaredMeanForce
        p = 2.5
        objective = 0.0
        objective2 = 0.0
        objective3 = 0.0
        objective_mixed = 0.0
        objective_direct = 0.0
        for i in range(len(coils)):
            objective += float(SquaredMeanForce(coils[i], coils).J())
            objective2 += float(SquaredMeanForce(coils[i], coils, downsample=2).J())
            objective3 += float(SquaredMeanForce(coils[i], coils, downsample=3).J())
            # Forces are in N/m, convert to MN/m before squaring
            net_force_N_per_m = np.sum(coils[i].force(coils) * gammadash_norm[:, None], axis=0) / gammadash_norm.shape[0]
            net_force_MN_per_m = net_force_N_per_m / 1e6
            objective_mixed += np.linalg.norm(net_force_MN_per_m) ** 2
            net_force_direct_N_per_m = coils[i].net_force(coils)
            net_force_direct_MN_per_m = net_force_direct_N_per_m / 1e6
            objective_direct += np.linalg.norm(net_force_direct_MN_per_m) ** 2

        print("objective:", objective, "mixed:", objective_mixed)
        np.testing.assert_allclose(objective, objective_mixed, rtol=1e-6)

        print("objective:", objective, "direct:", objective_direct)
        np.testing.assert_allclose(objective, objective_direct, rtol=1e-6)

        print("objective:", objective, "downsampled:", objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-6)

        print("objective:", objective, "downsampled further:", objective3)
        np.testing.assert_allclose(objective, objective3, rtol=1e-3)

        # # Test LpCurveForce
        threshold = 1e-3  # Threshold in MN/m (equivalent to 1.0e3 N/m)
        objective = 0.0
        objective2 = 0.0
        objective3 = 0.0
        objective_alt = 0.0
        for i in range(len(coils)):
            objective += float(LpCurveForce(coils[i], coils, p=p, threshold=threshold).J())
            objective2 += float(LpCurveForce(coils[i], coils, p=p, threshold=threshold, downsample=2).J())
            objective3 += float(LpCurveForce(coils[i], coils, p=p, threshold=threshold, downsample=3).J())
            force_norm_N_per_m = np.linalg.norm(coils[i].force(coils), axis=1)
            force_norm_MN_per_m = force_norm_N_per_m / 1e6  # Convert to MN/m
            gammadash_norm = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
            objective_alt += (1 / p) * np.sum(np.maximum(force_norm_MN_per_m - threshold, 0)**p * gammadash_norm) / gammadash_norm.shape[0]

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        print("objective:", objective, "objective2:", objective2, "diff:", objective - objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-2)

        print("objective:", objective, "objective3:", objective3, "diff:", objective - objective3)
        np.testing.assert_allclose(objective, objective3, rtol=1e-2)

        # Scramble the orientations so the torques are nonzero
        for i in range(len(base_curves)):
            x_new = base_curves[i].x
            x_new[3] += 0.1
            base_curves[i].x = x_new

        objective = float(SquaredMeanTorque(coils[0], coils).J())

        # Now compute the objective a different way, using the independent
        # coil_force function
        gammadash_norm = np.linalg.norm(coils[0].curve.gammadash(), axis=1)
        torques_N = coils[0].torque(coils)
        net_torque_N = np.sum(torques_N * gammadash_norm[:, None], axis=0) / gammadash_norm.shape[0]
        net_torque_MN = net_torque_N / 1e6  # Convert to MN
        objective_alt = np.linalg.norm(net_torque_MN, axis=-1) ** 2

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-2)

        objective = 0.0
        objective2 = 0.0
        objective3 = 0.0
        objective_alt = 0.0
        objective_direct = 0.0
        for i in range(len(coils)):
            objective += float(SquaredMeanTorque(coils[i], coils).J())
            objective2 += float(SquaredMeanTorque(coils[i], coils, downsample=2).J())
            objective3 += float(SquaredMeanTorque(coils[i], coils, downsample=3).J())
            gammadash_norm = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
            net_torque_N = np.sum(coils[i].torque(coils) * gammadash_norm[:, None], axis=0) / gammadash_norm.shape[0]
            net_torque_MN = net_torque_N / 1e6
            objective_alt += np.linalg.norm(net_torque_MN) ** 2
            net_torque_direct_N = coils[i].net_torque(coils)
            net_torque_direct_MN = net_torque_direct_N / 1e6
            objective_direct += np.linalg.norm(net_torque_direct_MN) ** 2

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-2)

        print("objective:", objective, "objective_direct:", objective_direct, "diff:", objective - objective_direct)
        np.testing.assert_allclose(objective, objective_direct, rtol=1e-2)

        print("objective:", objective, "downsampled:", objective2, "diff:", objective - objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-2)

        print("objective:", objective, "downsampled further:", objective3, "diff:", objective - objective3)
        np.testing.assert_allclose(objective, objective3, rtol=1e-2)

        # Test LpCurveTorque
        objective = 0.0
        objective2 = 0.0
        objective3 = 0.0
        objective_alt = 0.0
        threshold = 0.0
        objective_mixed = 0.0
        for i in range(len(coils)):
            objective += float(LpCurveTorque(coils[i], coils, p=p, threshold=threshold).J())
            objective2 += float(LpCurveTorque(coils[i], coils, p=p, threshold=threshold, downsample=2).J())
            objective3 += float(LpCurveTorque(coils[i], coils, p=p, threshold=threshold, downsample=3).J())
            torque_norm_N = np.linalg.norm(coils[i].torque(coils), axis=1)
            torque_norm_MN = torque_norm_N / 1e6  # Convert to MN
            gammadash_norm = np.linalg.norm(coils[i].curve.gammadash(), axis=1)
            objective_alt += (1 / p) * np.sum(np.maximum(torque_norm_MN - threshold, 0)**p * gammadash_norm) / gammadash_norm.shape[0]

        print("objective:", objective, "objective_alt:", objective_alt, "diff:", objective - objective_alt)
        np.testing.assert_allclose(objective, objective_alt, rtol=1e-6)

        print("objective:", objective, "downsampled:", objective2, "diff:", objective - objective2)
        np.testing.assert_allclose(objective, objective2, rtol=1e-4)

        print("objective:", objective, "downsampled further:", objective3, "diff:", objective - objective3)
        np.testing.assert_allclose(objective, objective3, rtol=1e-2)

    def test_force_and_torque_objectives_with_different_quadpoints(self):
        """Check that force and torque objectives work with coils having different numbers of quadrature points."""
        I = 1.7e4
        # Create two coils with different numbers of quadrature points
        curve1 = CurveXYZFourier(30, 1)
        curve1.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 1.0
        curve2 = CurveXYZFourier(50, 1)
        curve2.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 1.2
        curve3 = CurveXYZFourier(70, 1)
        curve3.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 0.8
        current1 = Current(I)
        current2 = Current(I)
        current3 = Current(I)
        coil1 = RegularizedCoil(curve1, current1, regularization_circ(0.05))
        coil2 = RegularizedCoil(curve2, current2, regularization_circ(0.05))
        coil3 = RegularizedCoil(curve3, current3, regularization_circ(0.05))
        coils = [coil1, coil2, coil3]
        # LpCurveForce (threshold in MN/m)
        threshold = 1e-3  # Threshold in MN/m (equivalent to 1.0e3 N/m)
        val = LpCurveForce(coil1, coil2, p=2.5, threshold=threshold).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveForce(coil1, [coil2, coil3], p=2.5, threshold=threshold).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveForce(coil1, coils, p=2.5, threshold=threshold).J()
        self.assertTrue(np.isfinite(val))
        # SquaredMeanForce
        val = SquaredMeanForce(coil1, coil3).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanForce([coil1, coil2], coil3).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanForce(coil1, coils).J()
        self.assertTrue(np.isfinite(val))
        # LpCurveTorque (threshold in MN)
        threshold = 1e-3  # Threshold in MN (equivalent to 1.0e3 N)
        val = LpCurveTorque(coil1, coils, p=2.5, threshold=threshold).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveTorque([coil1, coil2], coils, p=2.5, threshold=threshold).J()
        self.assertTrue(np.isfinite(val))
        val = LpCurveTorque(coil3, coil1, p=2.5, threshold=threshold).J()
        self.assertTrue(np.isfinite(val))
        # SquaredMeanTorque
        val = SquaredMeanTorque(coil1, coils).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanTorque(coil3, [coil1, coil2]).J()
        self.assertTrue(np.isfinite(val))
        val = SquaredMeanTorque(coil3, coils).J()
        self.assertTrue(np.isfinite(val))

    def test_Taylor(self):
        """
        Perform Taylor tests for a variety of coil force and torque objectives to verify the correctness of their derivatives.

        This test numerically checks the accuracy of the analytic derivatives (gradients) of several objective functions
        (e.g., net flux, B^2 energy, L^p force/torque, squared mean force/torque) used in coil optimization. It does so by:

        - Sweeping over different numbers of coils, field periods (nfp), stellarator symmetry options, regularization types, and downsampling factors.
        - For each configuration, constructing two sets of coils and computing the objective and its derivative.
        - Performing a finite-difference Taylor test: perturbing the parameters in a random direction, evaluating the objective at small steps, and comparing the finite-difference estimate of the derivative to the analytic value.
        - Asserting that the relative error decreases by at least a factor of 0.5 as the step size decreases, indicating correct derivative implementation.
        - Plotting the error decay for all objectives and parameter sweeps.

        A test passes if the Taylor error decreases rapidly (ideally quadratically) as the step size shrinks, confirming the correctness of the gradient implementation for all tested objectives and configurations.
        """
        import matplotlib.pyplot as plt
        ncoils_list = [2]
        nfp_list = [1, 2, 3]
        stellsym_list = [False, True]
        p_list = [2.5]
        threshold_list = [0.0, 1e-3]
        downsample_list = [1, 2]
        jax_flag_list = [False, True]
        numquadpoints_list = [20]
        I = 1.7e5
        a = 0.05
        b = 0.05
        np.random.seed(1234)
        regularization_types = [
            ("circular", lambda: regularization_circ(a)),
            ("rectangular", lambda: regularization_rect(a, b)),
        ]
        all_errors = []
        all_labels = []
        all_eps = []
        for ncoils in ncoils_list:
            for nfp in nfp_list:
                for stellsym in stellsym_list:
                    for p in p_list:
                        for threshold in threshold_list:
                            for reg_name, reg_func in regularization_types:
                                regularization = reg_func()
                                for downsample in downsample_list:
                                    for use_jax_curve in jax_flag_list:
                                        for numquadpoints in numquadpoints_list:
                                            base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym, numquadpoints=numquadpoints, use_jax_curve=use_jax_curve)
                                            base_curves2 = create_equally_spaced_curves(ncoils, nfp, stellsym, numquadpoints=numquadpoints, use_jax_curve=use_jax_curve)
                                            base_currents = [Current(I) for _ in range(ncoils)]
                                            coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym, regularizations=[regularization] * ncoils)
                                            for ii in range(ncoils):
                                                base_curves2[ii].x = base_curves2[ii].x + np.ones(len(base_curves2[ii].x)) * 0.1
                                            coils2 = coils_via_symmetries(base_curves2, base_currents, nfp, stellsym, regularizations=[regularization] * ncoils)
                                            objectives = [
                                                sum([NetFluxes(coils[i], coils2) for i in range(len(coils))]),
                                                B2Energy(coils + coils2, downsample=downsample),
                                                LpCurveTorque(coils, coils2, p=p, threshold=threshold, downsample=downsample),
                                                SquaredMeanTorque(coils, coils2, downsample=downsample),
                                                LpCurveForce(coils, coils2, p=p, threshold=threshold, downsample=downsample),
                                                SquaredMeanForce(coils, coils2, downsample=downsample),
                                            ]
                                            dofs = np.copy(LpCurveTorque(coils, coils2, p=p, threshold=threshold, downsample=downsample).x)
                                            h = np.ones_like(dofs)
                                            for J in objectives:
                                                print(f"ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, p={p}, threshold={threshold}, reg={reg_name}, downsample={downsample}, objective={type(J).__name__}")
                                                J.x = dofs  # Need to reset Jf.x for each objective
                                                dJ = J.dJ()
                                                deriv = np.sum(dJ * h)
                                                errors = []
                                                epsilons = []
                                                label = f"{type(J).__name__}, ncoils={ncoils}, nfp={nfp}, stellsym={stellsym}, p={getattr(J, 'p', p)}, threshold={getattr(J, 'threshold', threshold)}, reg={reg_name}, downsample={downsample}"
                                                for i in range(10, 16):
                                                    eps = 0.5**i
                                                    J.x = dofs + eps * h
                                                    Jp = J.J()
                                                    J.x = dofs - eps * h
                                                    Jm = J.J()
                                                    deriv_est = (Jp - Jm) / (2 * eps)
                                                    if np.abs(deriv) < 1e-8:
                                                        err_new = np.abs(deriv_est - deriv)  # compute absolute error instead
                                                    else:
                                                        err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
                                                    # Check error decrease by at least a factor of 0.5
                                                    if len(errors) > 0 and err_new > 1e-10:
                                                        print(f"err: {err_new}, jac: {np.abs(deriv)}, jac_est: {np.abs(deriv_est)}, ratio: {(err_new + 1e-12) / (errors[-1] + 1e-12)}")
                                                        assert err_new < 0.5 * errors[-1], f"Error did not decrease by factor 0.5: prev={errors[-1]}, curr={err_new}"
                                                    errors.append(err_new)
                                                    epsilons.append(eps)
                                                all_errors.append(errors)
                                                all_labels.append(label)
                                                all_eps.append(epsilons)
        # Plot all errors
        plt.figure(figsize=(14, 8))
        for errors, label, epsilons in zip(all_errors, all_labels, all_eps):
            plt.loglog(epsilons, errors, marker='o', label=label)
        plt.xlabel('eps')
        plt.ylabel('Relative Taylor error')
        plt.title('Taylor test errors for all objectives and parameter sweeps')
        plt.legend(fontsize=6, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('taylor_errors.png')

    def test_objectives_time(self):
        import time
        import matplotlib.pyplot as plt
        import numpy as np

        nfp = 3
        I = 1.7e4

        p = 2.5
        threshold = 1e-3  # Threshold in MN/m or MN (equivalent to 1.0e3 N/m or N)
        regularization = regularization_circ(0.05)

        # List of objective classes to test
        objective_classes = [
            "LpCurveForce",
            "LpCurveForce (one sum)",
            "LpCurveTorque",
            "LpCurveTorque (one sum)",
            "SquaredMeanForce",
            "SquaredMeanForce (one sum)",
            "SquaredMeanTorque",
            "SquaredMeanTorque (one sum)",
        ]

        ncoils_list = [2, 4]
        runtimes_J = np.zeros((len(objective_classes), len(ncoils_list)))
        runtimes_dJ = np.zeros((len(objective_classes), len(ncoils_list)))
        compile_times_J = np.zeros((len(objective_classes), len(ncoils_list)))
        compile_times_dJ = np.zeros((len(objective_classes), len(ncoils_list)))

        for idx_n, ncoils in enumerate(ncoils_list):
            print(f"\n--- Timing tests for ncoils = {ncoils} ---")
            base_curves = create_equally_spaced_curves(ncoils, nfp, True)
            base_currents = [Current(I) for j in range(ncoils)]
            coils = coils_via_symmetries(base_curves, base_currents, nfp, True, regularizations=[regularization] * ncoils)
            base_curves2 = create_equally_spaced_curves(ncoils, nfp, True)
            for i in range(ncoils):
                base_curves2[i].x = base_curves2[i].x + np.ones(len(base_curves2[i].x)) * 0.01
            coils2 = coils_via_symmetries(base_curves2, base_currents, nfp, True)
            for c in coils:
                c.regularization = regularization

            # Prepare objectives for each class
            # LpCurveForce, LpCurveTorque, SquaredMeanForce, SquaredMeanTorque: sum over all coils
            # Mixed objectives are faster if coils are split evenly into two groups
            objectives = [
                LpCurveForce(coils, coils2, p=p, threshold=threshold, downsample=2),
                LpCurveTorque(coils, coils2, p=p, threshold=threshold, downsample=2),
                SquaredMeanForce(coils, coils2, downsample=2),
                SquaredMeanTorque(coils, coils2, downsample=2),
            ]

            # Compilation time (first call)
            print("Timing compilation (first call):")
            for i, (obj, obj_label) in enumerate(zip(objectives, objective_classes)):
                t1 = time.time()
                obj.J()
                t2 = time.time()
                compile_times_J[i, idx_n] = t2 - t1
                print(f'{obj_label}: Compilation (J) took {t2 - t1:.6f} seconds')
                t1 = time.time()
                obj.dJ()
                t2 = time.time()
                compile_times_dJ[i, idx_n] = t2 - t1
                print(f'{obj_label}: Compilation (dJ) took {t2 - t1:.6f} seconds')

                # Run time (second call)
                print("Timing run (second call):")
                t1 = time.time()
                obj.J()
                t2 = time.time()
                runtimes_J[i, idx_n] = t2 - t1
                print(f'{obj_label}: Run (J) took {t2 - t1:.6f} seconds')
                t1 = time.time()
                obj.dJ()
                t2 = time.time()
                runtimes_dJ[i, idx_n] = t2 - t1
                print(f'{obj_label}: Run (dJ) took {t2 - t1:.6f} seconds')

        # Optionally, plot the results
        plt.figure(figsize=(10, 7))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'x', '+']
        colors = plt.cm.tab10.colors
        for i, label in enumerate(objective_classes):
            plt.semilogy(ncoils_list, runtimes_J[i], marker=markers[i % len(markers)], color=colors[i % len(colors)], label=f"{label} J()")
            plt.semilogy(ncoils_list, runtimes_dJ[i], marker=markers[i % len(markers)], linestyle='--', color=colors[i % len(colors)], label=f"{label} dJ()")
        plt.xlabel("Number of coils")
        plt.ylabel("Run time (s)")
        plt.title("Objective run times as a function of number of coils")
        plt.legend(fontsize=8)
        plt.grid(True, which='both', ls='--')
        plt.tight_layout()
        plt.savefig("objective_runtimes_semilogy.png")
        print("Run times saved to objective_runtimes_semilogy.png")

    def test_regularized_coil_requirement(self):
        """Test that force, torque, and energy objectives require RegularizedCoil objects."""
        # Create regular Coil objects (not RegularizedCoil)
        curve = CurveXYZFourier(20, 1)
        curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * 1.0
        current = Current(1.7e4)
        coil = Coil(curve, current)
        
        # Create RegularizedCoil for comparison
        regularization = regularization_circ(0.05)
        reg_coil = RegularizedCoil(curve, current, regularization)
        
        # Test that regular Coil objects raise ValueError for force/torque/energy objectives
        threshold = 1e-3
        with self.assertRaises(ValueError):
            LpCurveForce(coil, [reg_coil], p=2.5, threshold=threshold)
        
        # Net forces do not include a self-force, so no issue!
        SquaredMeanForce(coil, [reg_coil])
        SquaredMeanForce(coil, reg_coil)  # should not raise ValueError
        SquaredMeanTorque(coil, [reg_coil])
        SquaredMeanTorque(coil, reg_coil)  # should not raise ValueError
        
        with self.assertRaises(ValueError):
            LpCurveTorque(coil, [reg_coil], p=2.5, threshold=threshold)
        
        # Net torques do not include a self-torque, so no issue!
        SquaredMeanTorque(coil, [reg_coil])
        
        with self.assertRaises(ValueError):
            B2Energy([coil, reg_coil])
        
        # Test that RegularizedCoil objects work fine
        try:
            LpCurveForce(reg_coil, [reg_coil], p=2.5, threshold=threshold)
            SquaredMeanForce(reg_coil, [reg_coil])
            LpCurveTorque(reg_coil, [reg_coil], p=2.5, threshold=threshold)
            SquaredMeanTorque(reg_coil, [reg_coil])
            B2Energy([reg_coil])
        except ValueError:
            self.fail("RegularizedCoil objects should not raise ValueError")

    def test_lpcurveforces_taylor_test(self):
        """Verify that dJ matches finite differences of J"""
        # The Fourier spectrum of the NCSX coils is truncated - we don't need the
        # actual coil shapes from the experiment, just a few nonzero dofs.

        base_curves, base_currents, axis, nfp, bs = get_data("ncsx", coil_order=2)
        regularization = regularization_circ(0.05)
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True,
                                     regularizations=[regularization] * len(base_curves))
        
        threshold = 1e-3  # Threshold in MN/m (equivalent to 1.0e3 N/m)
        J = LpCurveForce(coils[0], coils, p=2.5, threshold=threshold)
        dJ = J.dJ()
        deriv = np.sum(dJ * np.ones_like(J.x))
        dofs = J.x
        h = np.ones_like(dofs)
        err = 100
        for i in range(10, 18):
            eps = 0.5**i
            J.x = dofs + eps * h
            Jp = J.J()
            J.x = dofs - eps * h
            Jm = J.J()
            deriv_est = (Jp - Jm) / (2 * eps)
            err_new = np.abs(deriv_est - deriv) / np.abs(deriv)
            print("test_lpcurveforces_taylor_test i:", i, "deriv_est:", deriv_est, "deriv:", deriv, "err_new:", err_new, "err:", err, "ratio:", err_new / err)
            np.testing.assert_array_less(err_new, 0.31 * err)
            err = err_new

    def test_circular_regularized_coil_subclass(self):
        """Test that CircularRegularizedCoil works with the new method-based API."""
        R0 = 1.7
        I = 10000
        a = 0.01
        order = 1
        N_quad = 23

        # Create a circle of radius R0 in the x-y plane:
        curve = CurveXYZFourier(N_quad, order)
        curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
        
        current = Current(I)
        coil = CircularRegularizedCoil(curve, current, a)
        
        # Test that it's a RegularizedCoil
        self.assertIsInstance(coil, RegularizedCoil)
        self.assertIsInstance(coil, CircularRegularizedCoil)
        
        # Test that regularization was computed correctly
        expected_reg = regularization_circ(a)
        np.testing.assert_allclose(coil.regularization, expected_reg, rtol=1e-10)
        self.assertEqual(coil.a, a)
        
        # Test B_regularized method
        B_reg = coil.B_regularized()
        self.assertEqual(B_reg.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(B_reg)))
        
        # Test self_force method
        force = coil.self_force()
        self.assertEqual(force.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(force)))
        
        # Verify it matches creating a new coil
        coil_alt = CircularRegularizedCoil(curve, current, a)
        force_alt = coil_alt.self_force()
        np.testing.assert_allclose(force, force_alt, rtol=1e-10)
        
        # Test force method with other coils
        nfp = 3
        ncoils = 4
        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        
        # Test force method
        force_from_others = coil.force(coils)
        self.assertEqual(force_from_others.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(force_from_others)))
        
        # Test net_force method
        net_force = coil.net_force(coils)
        self.assertEqual(net_force.shape, (3,))
        self.assertTrue(np.all(np.isfinite(net_force)))
        
        # Test torque method
        torque = coil.torque(coils)
        self.assertEqual(torque.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(torque)))
        
        # Test net_torque method
        net_torque = coil.net_torque(coils)
        self.assertEqual(net_torque.shape, (3,))
        self.assertTrue(np.all(np.isfinite(net_torque)))
        
        # Verify it matches using RegularizedCoil directly
        reg_val = regularization_circ(a)
        coil_reg = RegularizedCoil(curve, current, reg_val)
        force_reg = coil_reg.force(coils)
        net_force_reg = coil_reg.net_force(coils)
        torque_reg = coil_reg.torque(coils)
        net_torque_reg = coil_reg.net_torque(coils)
        
        np.testing.assert_allclose(force_from_others, force_reg, rtol=1e-10)
        np.testing.assert_allclose(net_force, net_force_reg, rtol=1e-10)
        np.testing.assert_allclose(torque, torque_reg, rtol=1e-10)
        np.testing.assert_allclose(net_torque, net_torque_reg, rtol=1e-10)

    def test_rectangular_regularized_coil_subclass(self):
        """Test that RectangularRegularizedCoil works with the new method-based API."""
        R0 = 1.7
        I = 10000
        a = 0.01
        b = 0.023
        order = 1
        N_quad = 23

        # Create a circle of radius R0 in the x-y plane:
        curve = CurveXYZFourier(N_quad, order)
        curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0., 0.]) * R0
        
        current = Current(I)
        coil = RectangularRegularizedCoil(curve, current, a, b)
        
        # Test that it's a RegularizedCoil
        self.assertIsInstance(coil, RegularizedCoil)
        self.assertIsInstance(coil, RectangularRegularizedCoil)
        
        # Test that regularization was computed correctly
        expected_reg = regularization_rect(a, b)
        np.testing.assert_allclose(coil.regularization, expected_reg, rtol=1e-10)
        self.assertEqual(coil.a, a)
        self.assertEqual(coil.b, b)
        
        # Test B_regularized method
        B_reg = coil.B_regularized()
        self.assertEqual(B_reg.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(B_reg)))
        
        # Test self_force method
        force = coil.self_force()
        self.assertEqual(force.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(force)))
        
        # Verify it matches creating a new coil
        coil_alt = RectangularRegularizedCoil(curve, current, a, b)
        force_alt = coil_alt.self_force()
        np.testing.assert_allclose(force, force_alt, rtol=1e-10)
        
        # Test force method with other coils
        nfp = 3
        ncoils = 4
        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        
        # Test force method
        force_from_others = coil.force(coils)
        self.assertEqual(force_from_others.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(force_from_others)))
        
        # Test net_force method
        net_force = coil.net_force(coils)
        self.assertEqual(net_force.shape, (3,))
        self.assertTrue(np.all(np.isfinite(net_force)))
        
        # Test torque method
        torque_from_others = coil.torque(coils)
        self.assertEqual(torque_from_others.shape, (N_quad, 3))
        self.assertTrue(np.all(np.isfinite(torque_from_others)))
        
        # Test net_torque method
        net_torque = coil.net_torque(coils)
        self.assertEqual(net_torque.shape, (3,))
        self.assertTrue(np.all(np.isfinite(net_torque)))
        
        # Verify all methods match using RegularizedCoil directly
        reg_val = regularization_rect(a, b)
        coil_reg = RegularizedCoil(curve, current, reg_val)
        force_reg = coil_reg.force(coils)
        net_force_reg = coil_reg.net_force(coils)
        torque_reg = coil_reg.torque(coils)
        net_torque_reg = coil_reg.net_torque(coils)
        
        np.testing.assert_allclose(force_from_others, force_reg, rtol=1e-10)
        np.testing.assert_allclose(net_force, net_force_reg, rtol=1e-10)
        np.testing.assert_allclose(torque_from_others, torque_reg, rtol=1e-10)
        np.testing.assert_allclose(net_torque, net_torque_reg, rtol=1e-10)

    def test_regularized_coil_methods_comprehensive(self):
        """Comprehensive test of all RegularizedCoil methods."""
        nfp = 3
        ncoils = 4
        I = 1.7e4
        regularization = regularization_circ(0.05)

        base_curves = create_equally_spaced_curves(ncoils, nfp, True)
        base_currents = [Current(I) for j in range(ncoils)]
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True,
                                     regularizations=[regularization] * ncoils)

        target_coil = coils[0]
        source_coils = coils
        
        # Test B_regularized returns correct shape
        B_reg = target_coil.B_regularized()
        n_points = len(target_coil.curve.quadpoints)
        self.assertEqual(B_reg.shape, (n_points, 3))
        self.assertTrue(np.all(np.isfinite(B_reg)))
        
        # Test self_force returns correct shape
        self_force = target_coil.self_force()
        self.assertEqual(self_force.shape, (n_points, 3))
        self.assertTrue(np.all(np.isfinite(self_force)))
        
        # Test force returns correct shape
        force = target_coil.force(source_coils)
        self.assertEqual(force.shape, (n_points, 3))
        self.assertTrue(np.all(np.isfinite(force)))
        
        # Test net_force returns correct shape
        net_force = target_coil.net_force(source_coils)
        self.assertEqual(net_force.shape, (3,))
        self.assertTrue(np.all(np.isfinite(net_force)))
        
        # Test torque returns correct shape
        torque = target_coil.torque(source_coils)
        self.assertEqual(torque.shape, (n_points, 3))
        self.assertTrue(np.all(np.isfinite(torque)))
        
        # Test net_torque returns correct shape
        net_torque = target_coil.net_torque(source_coils)
        self.assertEqual(net_torque.shape, (3,))
        self.assertTrue(np.all(np.isfinite(net_torque)))
        
        # Test that force includes both self and mutual contributions
        # Force should be non-zero for interacting coils
        self.assertTrue(np.linalg.norm(force) > 0)
        
        # Test that net_force is consistent with integrating force
        gammadash = target_coil.curve.gammadash()
        gammadash_norm = np.linalg.norm(gammadash, axis=1)[:, None]
        net_force_manual = np.sum(gammadash_norm * force, axis=0) / gammadash.shape[0]
        np.testing.assert_allclose(net_force, net_force_manual, rtol=1e-10)
        
        # Test that torque is perpendicular to force (torque = r x force)
        gamma = target_coil.curve.gamma()
        center = target_coil.curve.centroid()
        r = gamma - center
        torque_manual = np.cross(r, force)
        np.testing.assert_allclose(torque, torque_manual, rtol=1e-10)
        
        # Test that net_torque is consistent with integrating torque
        net_torque_manual = np.sum(gammadash_norm * torque, axis=0) / gammadash.shape[0]
        np.testing.assert_allclose(net_torque, net_torque_manual, rtol=1e-10)


if __name__ == '__main__':
    unittest.main()
