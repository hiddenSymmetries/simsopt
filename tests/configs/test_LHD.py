import unittest
import numpy as np

from simsopt.configs import get_LHD_like_data
from simsopt.field import Current, Coil, BiotSavart, find_periodic_field_line
from simsopt.geo import curves_to_vtk

class Tests(unittest.TestCase):
    def test_axis(self):
        coils, currents, ma = get_LHD_like_data()
        curves_to_vtk(coils, "LHD_coils")
        coils = [
            Coil(curve, Current(current))
            for curve, current in zip(coils, currents)
        ]
        field = BiotSavart(coils)
        nfp = 10

        # Find the magnetic axis:
        m = 1
        nphi = 21
        R0 = np.full(nphi, 3.8)  # Initial guess
        Z0 = np.full(nphi, 0.1)
        R, Z = find_periodic_field_line(field, nfp, m, R0, Z0, method="pseudospectral", nphi=nphi)
        print(f"LHD_like axis location: R = {R}, Z = {Z}")
        np.testing.assert_allclose(R, 3.6299180124750916)
        np.testing.assert_allclose(Z, 0.0, atol=1e-12)
