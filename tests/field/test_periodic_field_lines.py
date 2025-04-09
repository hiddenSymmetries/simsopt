import unittest
import numpy as np

from simsopt.configs import get_w7x_data
from simsopt.field import BiotSavart, coils_via_symmetries, find_periodic_field_line

from simsopt.field.periodic_field_lines import _integrate_field_line

class Tests(unittest.TestCase):
    def test_integrate_field_line(self):
        # Load the W7-X field:
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
        # Load the W7-X field:
        base_curves, base_currents, ma = get_w7x_data()
        nfp = 5
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        field = BiotSavart(coils)

        # Initial guess:
        R0 = 5.9
        Z0 = 0.1

        # Find the magnetic axis:
        m = 1
        R, Z = find_periodic_field_line(field, R0, Z0, nfp, m)
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, atol=1e-8)
        np.testing.assert_allclose(Z, 0, atol=1e-3)

        # Now find one of the island chains:
        m = 5
        R0 = 5.5
        Z0 = 0.87
        R, Z = find_periodic_field_line(field, R0, Z0, nfp, m)
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        np.testing.assert_allclose(Z, 0.875629267473603, atol=0.003)

    def test_find_periodic_field_line_psueodspectral(self):
        # Load the W7-X field:
        base_curves, base_currents, ma = get_w7x_data()
        nfp = 5
        coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
        field = BiotSavart(coils)

        # Initial guess:
        R0 = 5.9
        Z0 = 0.1

        # Find the magnetic axis:
        m = 1
        R, Z = find_periodic_field_line(field, R0, Z0, nfp, m, method="pseudospectral")
        print("R, Z", R, Z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, rtol=3e-5)
        np.testing.assert_allclose(Z, 0, atol=1e-8)

        # # Now find one of the island chains:
        # m = 5
        # R0 = 5.5
        # Z0 = 0.87
        # R, Z = find_periodic_field_line(field, R0, Z0, nfp, m, method="pseudospectral")
        # print("R, Z", R, Z)

        # # Check that the final coordinates are as expected
        # np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        # np.testing.assert_allclose(Z, 0.875629267473603, atol=0.003)
