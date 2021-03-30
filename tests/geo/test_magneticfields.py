import numpy as np
import unittest

from simsopt.geo.magneticfieldclasses import ToroidalField

class Testing(unittest.TestCase):

	def test_toroidal_field(self):
		R0test    = 1.3
		B0test    = 0.8
		pointtest = 0.5
		rtol      = 1e-4
		npoints   = 30
		# point locations
		points    = np.asarray(npoints * [[pointtest, pointtest, 1]])
		points   += rtol * (np.random.rand(*points.shape)-0.5)
		# Bfield from class
		Bfield    = ToroidalField(R0test,B0test)
		Bfield.set_points(points)
		B1        = Bfield.B()
		dB1_by_dX = Bfield.dB_by_dX()
		# Bfield analytical
		B2        = B0test*R0test*np.array([-1, 1, 0.])/(2*pointtest)
		dB2_by_dX = B0test*R0test*np.array([[1,0,0],[0,-1,0],[0,0,0]])/pointtest
		# Verify
		assert np.allclose(B1, B2,rtol=rtol)
		assert np.linalg.norm(np.concatenate(np.concatenate([dB2_by_dX-db1 for db1 in dB1_by_dX]))) < 5*npoints**0.45*rtol

if __name__ == "__main__":
	unittest.main()
