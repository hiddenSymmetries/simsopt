import numpy as np
import unittest

from simsopt.geo.magneticfieldclasses import ToroidalField

class Testing(unittest.TestCase):

    def test_toroidal_field(self):
        R0test    = 1.3
        B0test    = 0.8
        pointVar  = 1e-2
        npoints   = 10
        # point locations
        points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
        points   += pointVar * (np.random.rand(*points.shape)-0.5)
        # Bfield from class
        Bfield    = ToroidalField(R0test,B0test)
        Bfield.set_points(points)
        B1        = Bfield.B()
        dB1_by_dX = Bfield.dB_by_dX()
        # Bfield analytical
        B2        = np.array([(B0test*R0test/(point[0]**2+point[1]**2))*np.array([-point[1], point[0], 0.]) for point in points])
        dB2_by_dX = np.array([(B0test*R0test/((point[0]**2+point[1]**2)**2))*np.array([[2*point[0]*point[1], point[1]**2-point[0]**2, 0],[point[1]**2-point[0]**2, -2*point[0]*point[1], 0],[0,0,0]]) for point in points])
        # Verify
        assert np.allclose(B1, B2)
        assert np.allclose(dB1_by_dX, dB2_by_dX)

if __name__ == "__main__":
    unittest.main()
