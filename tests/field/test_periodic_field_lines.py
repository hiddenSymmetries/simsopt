import unittest
import numpy as np

from simsopt.configs import get_w7x_data, get_Cary_Hanson_field
from simsopt.field import BiotSavart, coils_via_symmetries, find_periodic_field_line, PeriodicFieldLine
from simsopt.field.periodic_field_lines import (
    _integrate_field_line,
    _integrate_field_line_cyl,
    _pseudospectral_residual,
    _pseudospectral_jacobian,
)
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
        z0 = 0.0
        Delta_phi = np.pi / 5  # Only half a field period

        for current_signs in [1, -1]:
            for c in base_currents:
                c.x = current_signs * c.x

            # Integrate the field line
            R, z = _integrate_field_line(field, R0, z0, Delta_phi)

            # Check that the final coordinates are as expected
            np.testing.assert_allclose(R, 5.207, atol=1e-3)
            np.testing.assert_allclose(z, 0, atol=1e-3)

    def test_integrate_field_line_cyl(self):
        field = _get_w7x_field()

        # The initial condition is not too far from the magnetic axis:
        R0 = 5.95
        z0 = 0.1
        Delta_phi = np.pi

        for phi0 in [0, 0.9]:
            for nphi in [1, 2, 3]:
                R1, z1 = _integrate_field_line(field, R0, z0, Delta_phi, phi0=phi0, nphi=nphi)
                R2, z2 = _integrate_field_line_cyl(field, R0, z0, Delta_phi, phi0=phi0, nphi=nphi)
                print("diffs:", R1 - R2, z1 - z2)
                np.testing.assert_allclose(R1, R2, atol=1e-8)
                np.testing.assert_allclose(z1, z2, atol=1e-8)

    def test_find_periodic_field_line_2D(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 5.9
        z0 = 0.1

        # Find the magnetic axis:
        m = 1
        R, z = find_periodic_field_line(field, nfp, m, R0, z0)
        print("R, z", R, z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, atol=1e-8)
        np.testing.assert_allclose(z, 0, atol=1e-7)

        # Find the magnetic axis at the half-period plane:
        R0 = 5.3
        z0 = 0.1
        R, z = find_periodic_field_line(field, nfp, m, R0, z0, half_period=True)
        print("R, z", R, z)
        np.testing.assert_allclose(R, 5.20481580547662, atol=1e-8)
        np.testing.assert_allclose(z, 0, atol=1e-8)

        # Now find one of the island chains:
        m = 5
        R0 = 5.5
        z0 = 0.87
        R, z = find_periodic_field_line(field, nfp, m, R0, z0)
        print("R, z", R, z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        np.testing.assert_allclose(z, 0.875629267473603, atol=0.003)

    def test_find_periodic_field_line_1D(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 6

        # Find the magnetic axis:
        m = 1
        R, z = find_periodic_field_line(field, nfp, m, R0, method="1D R")
        print("R, z", R, z)
        np.testing.assert_allclose(R, 5.949141380504241, atol=1e-8)
        np.testing.assert_allclose(z, 0, atol=1e-7)

        R, z = find_periodic_field_line(field, nfp, m, R0, method="1D z")
        print("R, z", R, z)
        np.testing.assert_allclose(R, 5.949141380504241, atol=1e-8)
        np.testing.assert_allclose(z, 0, atol=1e-7)

        # Find the magnetic axis at the half-period plane:
        R0 = 5.3
        R, z = find_periodic_field_line(field, nfp, m, R0, method="1D R", half_period=True)
        print("R, z", R, z)
        np.testing.assert_allclose(R, 5.20481580547662, atol=1e-8)
        np.testing.assert_allclose(z, 0, atol=1e-7)

        R, z = find_periodic_field_line(field, nfp, m, R0, method="1D z", half_period=True)
        print("R, z", R, z)
        np.testing.assert_allclose(R, 5.20481580547662, atol=1e-8)
        np.testing.assert_allclose(z, 0, atol=1e-7)


        # # Now find one of the island chains:
        # m = 5
        # R0 = 5.5
        # z0 = 0.87
        # R, z = find_periodic_field_line(field, nfp, m, R0, z0)
        # print("R, z", R, z)

        # # Check that the final coordinates are as expected
        # np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        # np.testing.assert_allclose(z, 0.875629267473603, atol=0.003)

    def test_pseudospectral_jacobian(self):
        """Compare the analytic Jacobian to finite differences."""
        # Load the W7-X field:
        field = _get_w7x_field()
        nfp = 5

        nphi = 21
        phimax = 2 * np.pi / nfp
        phi = np.linspace(0, phimax, nphi, endpoint=False)
        D = spectral_diff_matrix(nphi, xmin=0, xmax=phimax)
        for force_z0 in [False, True]:
            R0 = 5.9 + 0.1 * np.cos(nfp * phi)
            z0 = 0.2 + 0.1 * np.sin(nfp * phi)
            x = np.concatenate((R0, z0))
            analytic_jacobian = _pseudospectral_jacobian(x, nphi, D, phi, field, force_z0=force_z0)
            finite_diff_jacobian = np.zeros((2 * nphi, 2 * nphi))
            delta = 1e-6
            for j in range(2 * nphi):
                x1 = np.copy(x)
                x1[j] += delta
                r1 = _pseudospectral_residual(x1, nphi, D, phi, field, force_z0=force_z0)

                x2 = np.copy(x)
                x2[j] -= delta
                r2 = _pseudospectral_residual(x2, nphi, D, phi, field, force_z0=force_z0)

                finite_diff_jacobian[:, j] = (r1 - r2) / (2 * delta)

            np.testing.assert_allclose(analytic_jacobian, finite_diff_jacobian, rtol=1e-7, atol=0)

    def test_find_periodic_field_line_pseudospectral(self):
        field = _get_w7x_field()
        nfp = 5

        for method in ["pseudospectral", "pseudospectral z0"]:
            # Initial guess:
            R0 = 5.9
            z0 = 0.1

            # Find the magnetic axis:
            m = 1
            R, z = find_periodic_field_line(field, nfp, m, R0, z0, method=method)
            print("R, z", R[0], z[0])

            # Check that the final coordinates are as expected
            np.testing.assert_allclose(R[0], 5.949141380504241, rtol=3e-5)
            np.testing.assert_allclose(z[0], 0, atol=1e-8)

            # Find the magnetic axis at the half-period plane:
            R0 = 5.3
            z0 = 0.1
            R, z = find_periodic_field_line(field, nfp, m, R0, z0, half_period=True, method=method)
            print("R, z", R[0], z[0])
            np.testing.assert_allclose(R[0], 5.20481580547662, rtol=3e-5)
            np.testing.assert_allclose(z[0], 0, atol=1e-8)

        # # Now find one of the island chains:
        # m = 5
        # R0 = 5.5
        # z0 = 0.87
        # R, z = find_periodic_field_line(field, nfp, m, R0, z0, method="pseudospectral")
        # print("R, z", R, z)

        # # Check that the final coordinates are as expected
        # np.testing.assert_allclose(R, 5.4490101687346115, rtol=1e-3)
        # np.testing.assert_allclose(z, 0.875629267473603, atol=0.003)

    def test_find_periodic_field_line_1D_optimization(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 5.9

        # Find the magnetic axis:
        m = 1
        R, z = find_periodic_field_line(field, nfp, m, R0, method="1D optimization")
        print("R, z", R, z)

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R, 5.949141380504241, rtol=3e-5)
        np.testing.assert_allclose(z, 0, atol=1e-8)

        # Find the magnetic axis at the half-period plane:
        R0 = 5.3
        R, z = find_periodic_field_line(field, nfp, m, R0, half_period=True, method="1D optimization")
        print("R, z", R, z)
        np.testing.assert_allclose(R, 5.20481580547662, rtol=3e-5)
        np.testing.assert_allclose(z, 0, atol=1e-8)

    def test_Hanson_Cary_1984(self):
        # First try the non-optimized coils:
        coils, field = get_Cary_Hanson_field("1984", optimized=False)

        # Find the magnetic axis:
        R0 = 1.0
        z0 = 0
        nfp = 5
        m = 1
        R, z = find_periodic_field_line(field, nfp, m, R0, z0, method="pseudospectral")
        print("R, z", R[0], z[0])

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R[0], 0.9834328716279733, rtol=1e-7)
        np.testing.assert_allclose(z[0], 0, atol=1e-8)

        # Repeat with the optimized coils:
        coils, field = get_Cary_Hanson_field("1984", optimized=True)

        # Find the magnetic axis:
        R0 = 1.0
        z0 = 0
        nfp = 5
        m = 1
        R, z = find_periodic_field_line(field, nfp, m, R0, z0, method="pseudospectral")
        print("R, z", R[0], z[0])

        # Check that the final coordinates are as expected
        np.testing.assert_allclose(R[0], 0.955022421271663, rtol=1e-7)
        np.testing.assert_allclose(z[0], 0, atol=1e-8)

    def test_find_periodic_field_line_class(self):
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 6.0

        # Find the magnetic axis:
        m = 1
        R1, z1 = find_periodic_field_line(field, nfp, m, R0, method="1D z")

        pfl = PeriodicFieldLine(field, nfp, m, R0, method="1D z")
        np.testing.assert_allclose(pfl.R0, R1, atol=1e-8)
        np.testing.assert_allclose(pfl.z0, z1, atol=1e-8)

        # Check that the integral is insensitive to the number of points:
        integral1 = pfl.integral_A_dl()
        pfl2 = PeriodicFieldLine(field, nfp, m, R0, method="1D z", nphi=1234)
        integral2 = pfl2.integral_A_dl()
        np.testing.assert_allclose(integral1, integral2, rtol=2e-11)

        # Repeat with one of the islands.
        R0 = 6.2
        m = 5
        R1, z1 = find_periodic_field_line(field, nfp, m, R0, method="1D optimization")
        np.testing.assert_array_less(6.2, R1)  # Make sure we get the island and not the axis
        pfl = PeriodicFieldLine(field, nfp, m, R0, method="1D optimization", nphi=501)
        np.testing.assert_allclose(pfl.R0, R1, atol=1e-8)
        np.testing.assert_allclose(pfl.z0, z1, atol=1e-8)

        # Check that the integral is insensitive to the number of points:
        integral1 = pfl.integral_A_dl()
        pfl2 = PeriodicFieldLine(field, nfp, m, R0, method="1D optimization", nphi=1234)
        integral2 = pfl2.integral_A_dl()
        np.testing.assert_allclose(integral1, integral2, rtol=1e-10)

        try:
            import pyevtk
        except ImportError:
            # pyevtk not installed so skipping vtk export test
            return
        
        pyevtk.__path__  # To prevent ruff from complaining that pyevtk is not used
        pfl.to_vtk("/tmp/periodic_field_line")

    def test_d_r_d_phi_for_integral_A_dl(self):
        """Check the calculation of d_r_d_phi for the integral of A dl."""
        field = _get_w7x_field()
        nfp = 5

        # Initial guess:
        R0 = 6.0

        # Find the magnetic axis:
        m = 1
        R1, z1 = find_periodic_field_line(field, nfp, m, R0, method="1D z")

        pfl = PeriodicFieldLine(field, nfp, m, R0, method="1D z", nphi=500)
        integral, d_r_d_phi = pfl._integral_A_dl()
        dphi = pfl.phi[1] - pfl.phi[0]

        x = pfl.R * np.cos(pfl.phi)
        y = pfl.R * np.sin(pfl.phi)
        d_x_d_phi_alt = (x[2:] - x[:-2]) / (2 * dphi)
        d_y_d_phi_alt = (y[2:] - y[:-2]) / (2 * dphi)
        d_z_d_phi_alt = (pfl.z[2:] - pfl.z[:-2]) / (2 * dphi)

        if False:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 8))
            n_rows = 3
            n_cols = 1

            plt.subplot(n_rows, n_cols, 1)
            plt.plot(d_r_d_phi[1:-1, 0])
            plt.plot(d_x_d_phi_alt, ':')

            plt.subplot(n_rows, n_cols, 2)
            plt.plot(d_r_d_phi[1:-1, 1])
            plt.plot(d_y_d_phi_alt, ':')

            plt.subplot(n_rows, n_cols, 3)
            plt.plot(d_r_d_phi[1:-1, 2])
            plt.plot(d_z_d_phi_alt, ':')

            plt.tight_layout()
            plt.show()

        atol = 0.003
        rtol = 1e-10
        np.testing.assert_allclose(d_r_d_phi[1:-1, 0], d_x_d_phi_alt, atol=atol, rtol=rtol)
        np.testing.assert_allclose(d_r_d_phi[1:-1, 1], d_y_d_phi_alt, atol=atol, rtol=rtol)
        np.testing.assert_allclose(d_r_d_phi[1:-1, 2], d_z_d_phi_alt, atol=atol, rtol=rtol)
        
    def test_get_R_z_at_phi(self):
        field = _get_w7x_field()
        nfp = 5

        m = 1
        R0 = 6.0
        pfl = PeriodicFieldLine(field, nfp, m, R0, method="1D z")
        for j, phi in enumerate(pfl.phi):
            R, z = pfl.get_R_z_at_phi(phi)
            np.testing.assert_allclose(R, pfl.R[j])
            np.testing.assert_allclose(z, pfl.z[j], atol=1e-8)

            R, z = pfl.get_R_z_at_phi(phi + 2 * np.pi / nfp)
            np.testing.assert_allclose(R, pfl.R[j])
            np.testing.assert_allclose(z, pfl.z[j], atol=1e-7)

            R, z = pfl.get_R_z_at_phi(phi - 2 * np.pi / nfp)
            np.testing.assert_allclose(R, pfl.R[j])
            np.testing.assert_allclose(z, pfl.z[j], atol=1e-7)