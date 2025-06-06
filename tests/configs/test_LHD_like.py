import unittest
import numpy as np

from simsopt.configs import get_LHD_like_data
from simsopt.field import (
    Current,
    Coil,
    BiotSavart,
    find_periodic_field_line,
    get_magnetic_axis,
)
from simsopt.geo import curves_to_vtk
import simsoptpp as sopp


class Tests(unittest.TestCase):
    def test_axis(self):
        coils, currents, ma = get_LHD_like_data()
        curves_to_vtk(coils, "LHD_coils")
        coils = [
            Coil(curve, Current(current)) for curve, current in zip(coils, currents)
        ]
        field = BiotSavart(coils)
        nfp = 10

        # Find the magnetic axis:
        m = 1
        nphi = 21
        R0 = np.full(nphi, 3.8)  # Initial guess
        z0 = np.full(nphi, 0.1)
        R, z = find_periodic_field_line(
            field, nfp, m, R0, z0, method="pseudospectral", nphi=nphi
        )
        print(f"LHD_like axis location: R = {R}, z = {z}")
        np.testing.assert_allclose(R[0], 3.6299180124750916)
        np.testing.assert_allclose(z[0], 0.0, atol=1e-8)

        curve = get_magnetic_axis(field, nfp, 3.6, order=6)
        print("dofs:")
        for val in curve.x:
            print(f"{val},")
        for val, name in zip(curve.x, curve.local_dof_names):
            print(f"{name}: {val}")

    def test_axis2(self):
        """
        If we trace a field line starting from the expected magnetic axis, it should
        match the purported axis.
        """
        coils, currents, axis = get_LHD_like_data()
        # Flip the sign of current so B points towards +phi. Otherwise
        # fieldline_tracing traces towards -phi.
        currents = -np.array(currents)
        coils = [
            Coil(curve, Current(current)) for curve, current in zip(coils, currents)
        ]
        field = BiotSavart(coils)

        axis_gamma = axis.gamma()
        expected_R_axis = 3.629918012474283
        _, res_phi_hit = sopp.fieldline_tracing(
            field,
            [expected_R_axis, 0, 0],
            tmax=10.0,
            tol=1e-10,
            phis=axis.quadpoints * 2 * np.pi,
            stopping_criteria=[],
        )
        # At each phi, compare xyz from fieldline_tracing to the expected axis:
        n_checks = 0
        for item in res_phi_hit:
            np.testing.assert_allclose(
                item[2:],
                axis_gamma[int(item[1]), :],
                atol=1e-9,
            )
            n_checks += 1
        # Make sure tmax was sufficient to check all points:
        np.testing.assert_array_less(len(axis.quadpoints), n_checks)
