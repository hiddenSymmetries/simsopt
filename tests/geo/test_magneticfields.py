from simsopt.geo.magneticfieldclasses import ToroidalField, ScalarPotentialRZMagneticField, CircularCoilXY
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.magneticfield import MagneticFieldSum
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.biotsavart import BiotSavart

import numpy as np
import unittest

class Testing(unittest.TestCase):

    def test_toroidal_field(self):
        R0test    = 1.3
        B0test    = 0.8
        pointVar  = 1e-2
        npoints   = 20
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
        # Verify that divergence is zero
        assert (dB1_by_dX[:,0,0]+dB1_by_dX[:,1,1]+dB1_by_dX[:,2,2]==np.zeros((npoints))).all()
        assert (dB2_by_dX[:,0,0]+dB2_by_dX[:,1,1]+dB2_by_dX[:,2,2]==np.zeros((npoints))).all()
        # Verify that, as a vacuum field, grad B=grad grad phi so that grad_i B_j = grad_j B_i
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        transpGradB2 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(dB1_by_dX,transpGradB1)
        assert np.allclose(dB2_by_dX,transpGradB2)

    def test_sum_Bfields(self):
        pointVar  = 1e-1
        npoints   = 20
        points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
        points   += pointVar * (np.random.rand(*points.shape)-0.5)
        # Set up helical field
        coils     = [CurveHelical(101, 2, 5, 2, 1., 0.3) for i in range(2)]
        coils[0].set_dofs(np.concatenate(([np.pi/2,0],[0,0])))
        coils[1].set_dofs(np.concatenate(([0    ,0],[0,0])))
        currents  = [-2.1e5,2.1e5]
        Bhelical  = BiotSavart(coils, currents)
        # Set up toroidal fields
        Btoroidal1 = ToroidalField(1.,1.)
        Btoroidal2 = ToroidalField(1.2,0.1)
        # Set up sum of the three in two different ways
        Btotal1 = MagneticFieldSum([Bhelical,Btoroidal1,Btoroidal2])
        Btotal2 = Bhelical+Btoroidal1+Btoroidal2
        # Evaluate at a given point
        Bhelical.set_points(points)
        Btoroidal1.set_points(points)
        Btoroidal2.set_points(points)
        Btotal1.set_points(points)
        Btotal2.set_points(points)
        # Verify
        assert np.allclose(Btotal1.B(),Btotal2.B())
        assert np.allclose(Bhelical.B()+Btoroidal1.B()+Btoroidal2.B(),Btotal1.B())
        assert np.allclose(Btotal1.dB_by_dX(),Btotal2.dB_by_dX())
        assert np.allclose(Bhelical.dB_by_dX()+Btoroidal1.dB_by_dX()+Btoroidal2.dB_by_dX(),Btotal1.dB_by_dX())

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

    def test_circularcoilXY_Bfield(self):
        current = 1e7
        radius  = 1.0
        pointVar  = 1e-2
        npoints   = 20
        Bfield  = CircularCoilXY(I=current, r0=radius)
        ## verify the field at the center of the coil
        points  = np.array([[1e-10,0,0.]])
        Bfield.set_points(points)
        assert np.allclose(Bfield.B(),[[0,0,2*np.pi]])
        ## compare to biosavart(circular_coil)
        # at these points
        points    = np.asarray(npoints * [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]])
        points   += pointVar * (np.random.rand(*points.shape)-0.5)
        # verify with a x^2+y^2=radius^2 circular coil
        coils = [CurveRZFourier(300, 1, 1, True)]
        coils[0].set_dofs([radius,0,0])
        Bcircular = BiotSavart(coils, [current])
        Bfield.set_points(points)
        Bcircular.set_points(points)
        assert np.allclose(Bfield.B(),Bcircular.B())

    def test_helicalcoil_Bfield(self):
        point = [[-1.41513202e-03,  8.99999382e-01, -3.14473221e-04]]
        field = [[-0.00101961,0.20767292,-0.00224908]]
        derivative = [[[0.47545098,0.01847397,1.10223595],[0.01847426,-2.66700072,0.01849548],[1.10237535,0.01847085,2.19154973]]]
        coils     = [CurveHelical(100, 2, 5, 2, 1., 0.3) for i in range(2)]
        coils[0].set_dofs(np.concatenate(([0,0],[0,0])))
        coils[1].set_dofs(np.concatenate(([np.pi/2,0],[0,0])))
        currents  = [-3.07e5,3.07e5]
        Bhelical  = BiotSavart(coils, currents)
        Bhelical.set_points(point)
        assert np.allclose(Bhelical.B(),field)
        assert np.allclose(Bhelical.dB_by_dX(),derivative)

if __name__ == "__main__":
    unittest.main()