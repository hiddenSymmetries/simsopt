import unittest
import numpy as np

from simsopt.configs import get_w7x_data, get_Cary_Hanson_field
from simsopt.field import BiotSavart, coils_via_symmetries, find_periodic_field_line
from simsopt.field.periodic_field_lines import _integrate_field_line, _pseudospectral_residual, _pseudospectral_jacobian
from simsopt.util.spectral_diff_matrix import spectral_diff_matrix

def _get_w7x_field():
    base_curves, base_currents, ma = get_w7x_data()
    coils = coils_via_symmetries(base_curves, base_currents, 5, True)
    return BiotSavart(coils)

class Tests(unittest.TestCase):
    def test_integrate_field_line(self):
        base_curves, base_currents, ma = get_w7x_data()
        coils = coils_via_symmetries(base_curves, base_currents, 5, True)
        field = BiotSavart(coils)

        # The initial condition is (approximately) on the magnetic axis:
        R0 = 5.95
        Z0 = 0.0
        Delta_phi = np.pi / 5  # Only half a field period

        for current_signs in [1, -1]:
            for c in base_currents:
                c.x = current_signs * c.x

            # Integrate the field line
            R, Z = _integrate_field_line(field, R0, Z0, Delta_phi)

            # Check that the final coordinates are as expected
            np.testing.assert_allclose(R, 5.207, atol=1e-3)
            np.testing.assert_allclose(Z, 0, atol=1e-3)

    def test_find_periodic_field_line_2D(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 5.9
        Z0 = 0.1

        # Find the magnetic axis:
        m = 1
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0)
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, atol=1e-8)
        np.testing.assert_allclose(Z, 0, atol=1e-7)

        # Find the magnetic axis at the half-period plane:
        R0 = 5.3
        Z0 = 0.1
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, half_period=True)
        print("R, Z", R, Z)
        np.testing.assert_allclose(R, 5.20481580547662, atol=1e-8)
        np.testing.assert_allclose(Z, 0, atol=1e-8)

        # Now find one of the island chains:
        m = 5
        R0 = 5.5
        Z0 = 0.87
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0)
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        np.testing.assert_allclose(Z, 0.875629267473603, atol=0.003)

    @unittest.skip
    def test_find_periodic_field_line_1D(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 5.9

        # Find the magnetic axis:
        m = 1
        R, Z = find_periodic_field_line(field, nfp, m, R0, method="1D R")
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, atol=1e-8)
        np.testing.assert_allclose(Z, 0, atol=1e-3)

        # # Now find one of the island chains:
        # m = 5
        # R0 = 5.5
        # Z0 = 0.87
        # R, Z = find_periodic_field_line(field, nfp, m, R0, Z0)
        # print("R, Z", R, Z)

        # # Check that the final coordinates are as expected
        # np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        # np.testing.assert_allclose(Z, 0.875629267473603, atol=0.003)

    def test_pseudospectral_jacobian(self):
        """Compare the analytic Jacobian to finite differences."""
        # Load the W7-X field:
        field = _get_w7x_field()
        nfp = 5

        nphi = 21
        phimax = 2 * np.pi / nfp
        phi = np.linspace(0, phimax, nphi, endpoint=False)
        D = spectral_diff_matrix(nphi, xmin=0, xmax=phimax)
        R0 = 5.9 + 0.1 * np.cos(nfp * phi)
        Z0 = 0.2 + 0.1 * np.sin(nfp * phi)
        x = np.concatenate((R0, Z0))
        analytic_jacobian = _pseudospectral_jacobian(x, nphi, D, phi, field)
        finite_diff_jacobian = np.zeros((2 * nphi, 2 * nphi))
        delta = 1e-6
        for j in range(2 * nphi):
            x1 = np.copy(x)
            x1[j] += delta
            r1 = _pseudospectral_residual(x1, nphi, D, phi, field)

            x2 = np.copy(x)
            x2[j] -= delta
            r2 = _pseudospectral_residual(x2, nphi, D, phi, field)

            finite_diff_jacobian[:, j] = (r1 - r2) / (2 * delta)

        np.testing.assert_allclose(analytic_jacobian, finite_diff_jacobian, rtol=1e-7, atol=0)

    def test_find_periodic_field_line_pseudospectral(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 5.9
        Z0 = 0.1

        # Find the magnetic axis:
        m = 1
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, method="pseudospectral")
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, rtol=3e-5)
        np.testing.assert_allclose(Z, 0, atol=1e-8)

        # Find the magnetic axis at the half-period plane:
        R0 = 5.3
        Z0 = 0.1
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, half_period=True, method="pseudospectral")
        print("R, Z", R, Z)
        np.testing.assert_allclose(R, 5.20481580547662, rtol=3e-5)
        np.testing.assert_allclose(Z, 0, atol=1e-8)

        # # Now find one of the island chains:
        # m = 5
        # R0 = 5.5
        # Z0 = 0.87
        # R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, method="pseudospectral")
        # print("R, Z", R, Z)

        # # Check that the final coordinates are as expected
        # np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        # np.testing.assert_allclose(Z, 0.875629267473603, atol=0.003)

    def test_Hanson_Cary_1984(self):
        # First try the non-optimized coils:
        coils, field = get_Cary_Hanson_field("1984", optimized=False)

        # Find the magnetic axis:
        R0 = 1.0
        Z0 = 0
        nfp = 5
        m = 1
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, method="pseudospectral")
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 0.9834328716279733, rtol=1e-7)
        np.testing.assert_allclose(Z, 0, atol=1e-8)

        # Repeat with the optimized coils:
        coils, field = get_Cary_Hanson_field("1984", optimized=True)

        # Find the magnetic axis:
        R0 = 1.0
        Z0 = 0
        nfp = 5
        m = 1
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, method="pseudospectral")
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 0.955022421271663, rtol=1e-7)
        np.testing.assert_allclose(Z, 0, atol=1e-8)
