import numpy as np
import unittest

from simsopt.geo.magneticfieldclasses import ToroidalField, ScalarPotentialRZMagneticField, CircularCoil
from simsopt.geo.magneticfield import MagneticFieldSum
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.biotsavart import BiotSavart

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

    def test_sum_Bfields(self):
        # Set up helical field
        coils     = [CurveHelical(101, 2, 5, 2, 1., 0.3) for i in range(2)]
        coils[0].set_dofs(np.concatenate(([np.pi,0],[0,0])))
        coils[1].set_dofs(np.concatenate(([0    ,0],[0,0])))
        currents  = [-2.1e5,2.1e5]
        Bhelical  = BiotSavart(coils, currents)
        # Set up toroidal field
        Btoroidal = ToroidalField(1.,1.)
        # Set up sum of the two
        Btotal    = MagneticFieldSum(Bhelical,Btoroidal)
        # Evaluate at a given point
        points    = np.array([[1.1,0.9,0.3]])
        Bhelical.set_points(points)
        Btoroidal.set_points(points)
        Btotal.set_points(points)
        # Verify
        assert np.allclose(Bhelical.B()+Btoroidal.B(),Btotal.B())
        assert np.allclose(Bhelical.dB_by_dX()+Btoroidal.dB_by_dX(),Btotal.dB_by_dX())

    def test_scalarpotential_Bfield(self):
        # Set up magnetic field scalar potential
        PhiStr = "0.1*phi+0.2*R*Z+0.3*Z*phi+0.4*R**2+0.5*Z**2"
        # Define set of points
        pointVar  = 1e-2
        npoints   = 20
        points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
        points   += pointVar * (np.random.rand(*points.shape)-0.5)
        # Set up scalar potential B
        Bscalar = ScalarPotentialRZMagneticField(PhiStr)
        Bscalar.set_points(points)
        B1        = np.array(Bscalar.B())
        dB1_by_dX = np.array(Bscalar.dB_by_dX())
        # Analytical Formula for B
        rphiz     = [[np.sqrt(np.power(point[0],2) + np.power(point[1],2)),np.arctan2(point[1],point[0]),point[2]] for point in points]
        B2        = np.array([[0.2*point[2]+0.8*point[0],(0.1+0.3*point[2])/point[0],0.2*point[0]+0.3*point[1]+point[2]] for point in rphiz])
        dB2_by_dX = np.array([
            [[0.8*np.cos(point[1]),-(np.cos(point[1])/point[0]**2)*(0.1+0.3*point[2]),0.2*np.cos(point[1])-0.3*np.sin(point[1])/point[0]],
             [0.8*np.sin(point[1]),-(np.sin(point[1])/point[0]**2)*(0.1+0.3*point[2]),0.2*np.sin(point[1])+0.3*np.cos(point[1])/point[0]],
             [0.2, 0.3/point[0], 1]] for point in rphiz])
        # Verify
        assert np.allclose(B1,B2)
        assert np.allclose(dB1_by_dX,dB2_by_dX)

    def test_circularcoil_Bfield(self):
        Bfield = CircularCoil(I=1e7, r0=1)
        points=np.array([[1e-10,0,0.]])
        Bfield.set_points(points)
        assert np.allclose(Bfield.B(),[[0,0,2*np.pi]])

if __name__ == "__main__":
    unittest.main()
