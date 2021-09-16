from simsopt.field.magneticfieldclasses import ToroidalField, \
    ScalarPotentialRZMagneticField, CircularCoil, Dommaschk, \
    Reiman, sympy_found, InterpolatedField, PoloidalField
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.coil import coils_via_symmetries, Coil, Current
from simsopt.util.zoo import get_ncsx_data

import numpy as np
import unittest

try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False


class Testing(unittest.TestCase):

    def test_toroidal_field(self):
        R0test = 1.3
        B0test = 0.8
        pointVar = 1e-2
        npoints = 20
        # point locations
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += pointVar * (np.random.rand(*points.shape)-0.5)
        # Bfield from class
        Bfield = ToroidalField(R0test, B0test)
        Bfield.set_points(points)
        B1 = Bfield.B()
        dB1_by_dX = Bfield.dB_by_dX()
        # Bfield analytical
        B2 = np.array([(B0test*R0test/(point[0]**2+point[1]**2))*np.array([-point[1], point[0], 0.]) for point in points])
        dB2_by_dX = np.array([(B0test*R0test/((point[0]**2+point[1]**2)**2))*np.array([[2*point[0]*point[1], point[1]**2-point[0]**2, 0], [point[1]**2-point[0]**2, -2*point[0]*point[1], 0], [0, 0, 0]]) for point in points])
        # Verify
        assert np.allclose(B1, B2)
        assert np.allclose(dB1_by_dX, dB2_by_dX)
        # Verify that divergence is zero
        assert (dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2] == np.zeros((npoints))).all()
        assert (dB2_by_dX[:, 0, 0]+dB2_by_dX[:, 1, 1]+dB2_by_dX[:, 2, 2] == np.zeros((npoints))).all()
        # Verify that, as a vacuum field, grad B=grad grad phi so that grad_i B_j = grad_j B_i
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        transpGradB2 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(dB1_by_dX, transpGradB1)
        assert np.allclose(dB2_by_dX, transpGradB2)
        # Verify values of the vector potential
        Afield1 = Bfield.A()
        newA1 = np.array([[B0test*R0test*point[0]*point[2]/(point[0]**2+point[1]**2), B0test*R0test*point[1]*point[2]/(point[0]**2+point[1]**2), 0] for point in points])
        assert np.allclose(Afield1, newA1)
        # Verify that curl of magnetic vector potential is the toroidal magnetic field
        dA1_by_dX = Bfield.dA_by_dX()
        newB1 = np.array([[dA1bydX[2, 1]-dA1bydX[1, 2], dA1bydX[0, 2]-dA1bydX[2, 0], dA1bydX[1, 0]-dA1bydX[0, 1]] for dA1bydX in dA1_by_dX])
        assert np.allclose(B1, newB1)
        # Verify symmetry of the Hessians
        GradGradB1 = Bfield.d2B_by_dXdX()
        GradGradA1 = Bfield.d2A_by_dXdX()
        transpGradGradB1 = np.array([[gradgradB1.T for gradgradB1 in gradgradB]for gradgradB in GradGradB1])
        transpGradGradA1 = np.array([[gradgradA1.T for gradgradA1 in gradgradA]for gradgradA in GradGradA1])
        assert np.allclose(GradGradB1, transpGradGradB1)
        assert np.allclose(GradGradA1, transpGradGradA1)

    def test_sum_Bfields(self):
        pointVar = 1e-1
        npoints = 20
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += pointVar * (np.random.rand(*points.shape)-0.5)
        # Set up helical field
        curves = [CurveHelical(101, 2, 5, 2, 1., 0.3) for i in range(2)]
        curves[0].set_dofs(np.concatenate(([np.pi/2, 0], [0, 0])))
        curves[1].set_dofs(np.concatenate(([0, 0], [0, 0])))
        currents = [-2.1e5, 2.1e5]
        Bhelical = BiotSavart([
            Coil(curves[0], Current(currents[0])),
            Coil(curves[1], Current(currents[1]))])
        # Set up toroidal fields
        Btoroidal1 = ToroidalField(1., 1.)
        Btoroidal2 = ToroidalField(1.2, 0.1)
        # Set up sum of the three in two different ways
        Btotal1 = MagneticFieldSum([Bhelical, Btoroidal1, Btoroidal2])
        Btotal2 = Bhelical+Btoroidal1+Btoroidal2
        Btotal3 = Btoroidal1+Btoroidal2
        # Evaluate at a given point
        Bhelical.set_points(points)
        Btoroidal1.set_points(points)
        Btoroidal2.set_points(points)
        Btotal1.set_points(points)
        Btotal2.set_points(points)
        Btotal3.set_points(points)
        # Verify
        assert np.allclose(Btotal1.B(), Btotal2.B())
        assert np.allclose(Bhelical.B()+Btoroidal1.B()+Btoroidal2.B(), Btotal1.B())
        assert np.allclose(Btotal1.dB_by_dX(), Btotal2.dB_by_dX())
        assert np.allclose(Bhelical.dB_by_dX()+Btoroidal1.dB_by_dX()+Btoroidal2.dB_by_dX(), Btotal1.dB_by_dX())

        assert np.allclose(Btoroidal1.d2B_by_dXdX()+Btoroidal2.d2B_by_dXdX(), Btotal3.d2B_by_dXdX())
        assert np.allclose(Btoroidal1.A()+Btoroidal2.A(), Btotal3.A())
        assert np.allclose(Btoroidal1.dA_by_dX()+Btoroidal2.dA_by_dX(), Btotal3.dA_by_dX())
        assert np.allclose(Btoroidal1.d2A_by_dXdX()+Btoroidal2.d2A_by_dXdX(), Btotal3.d2A_by_dXdX())

    @unittest.skipIf(not sympy_found, "Sympy not found")
    def test_scalarpotential_Bfield(self):
        # Set up magnetic field scalar potential
        PhiStr = "0.1*phi+0.2*R*Z+0.3*Z*phi+0.4*R**2+0.5*Z**2"
        # Define set of points
        pointVar = 1e-1
        npoints = 20
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += pointVar * (np.random.rand(*points.shape)-0.5)
        # Set up scalar potential B
        Bscalar = ScalarPotentialRZMagneticField(PhiStr)
        Bscalar.set_points(points)
        B1 = np.array(Bscalar.B())
        dB1_by_dX = np.array(Bscalar.dB_by_dX())
        # Analytical Formula for B
        rphiz = [[np.sqrt(np.power(point[0], 2) + np.power(point[1], 2)), np.arctan2(point[1], point[0]), point[2]] for point in points]
        B2 = np.array([[0.2*point[2]+0.8*point[0], (0.1+0.3*point[2])/point[0], 0.2*point[0]+0.3*point[1]+point[2]] for point in rphiz])
        dB2_by_dX = np.array([
            [[0.8*np.cos(point[1]), -(np.cos(point[1])/point[0]**2)*(0.1+0.3*point[2]), 0.2*np.cos(point[1])-0.3*np.sin(point[1])/point[0]],
             [0.8*np.sin(point[1]), -(np.sin(point[1])/point[0]**2)*(0.1+0.3*point[2]), 0.2*np.sin(point[1])+0.3*np.cos(point[1])/point[0]],
             [0.2, 0.3/point[0], 1]] for point in rphiz])
        # Verify
        assert np.allclose(B1, B2)
        assert np.allclose(dB1_by_dX, dB2_by_dX)

    def test_circularcoil_Bfield(self):
        current = 1.2e7
        radius = 1.12345
        center = [0.12345, 0.6789, 1.23456]
        pointVar = 1e-1
        npoints = 1
        ## verify the field at the center of a coil in the xy plane
        Bfield = CircularCoil(I=current, r0=radius)
        points = np.array([[1e-10, 0, 0.]])
        Bfield.set_points(points)
        assert np.allclose(Bfield.B(), [[0, 0, current/1e7*2*np.pi/radius]])
        # Verify that divergence is zero
        dB1_by_dX = Bfield.dB_by_dX()
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))
        # Verify that, as a vacuum field, grad B=grad grad phi so that grad_i B_j = grad_j B_i
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(dB1_by_dX, transpGradB1)
        ### compare to biosavart(circular_coil)
        ## at these points
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += pointVar * (np.random.rand(*points.shape)-0.5)
        ## verify with a x^2+z^2=radius^2 circular coil
        normal = [np.pi/2, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], radius, 0., center[1], 0., 0., center[2], 0., radius])
        Bcircular = BiotSavart([Coil(curve, Current(current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))
        assert np.allclose(dB1_by_dX, transpGradB1)
        # use normal = [0, 1, 0]
        normal = [0, 1, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], radius, 0., center[1], 0., 0., center[2], 0., radius])
        Bcircular = BiotSavart([Coil(curve, Current(current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))
        assert np.allclose(dB1_by_dX, transpGradB1)
        ## verify with a y^2+z^2=radius^2 circular coil
        normal = [np.pi/2, np.pi/2]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, 0., center[1], radius, 0., center[2], 0., radius])
        Bcircular = BiotSavart([Coil(curve, Current(current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient
        Bfield.set_points([[0.1, 0.2, 0.3]])
        Afield = Bfield.A()
        assert np.allclose(Afield, [[0, 5.15786, -2.643056]])
        # use normal=[1,0,0]
        normal = [1, 0, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, 0., center[1], radius, 0., center[2], 0., radius])
        Bcircular = BiotSavart([Coil(curve, Current(current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient
        ## verify with a x^2+y^2=radius^2 circular coil
        center = [0, 0, 0]
        normal = [0, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, radius, center[1], radius, 0., center[2], 0., 0.])
        Bcircular = BiotSavart([Coil(curve, Current(current))])
        curve2 = CurveRZFourier(300, 1, 1, True)
        curve2.set_dofs([radius, 0, 0])
        Bcircular2 = BiotSavart([Coil(curve2, Current(current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        Bcircular2.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.B(), Bcircular2.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular2.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient
        # use normal = [0, 0, 1]
        center = [0, 0, 0]
        normal = [0, 0, 1]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, radius, center[1], radius, 0., center[2], 0., 0.])
        Bcircular = BiotSavart([Coil(curve, Current(current))])
        curve2 = CurveRZFourier(300, 1, 1, True)
        curve2.set_dofs([radius, 0, 0])
        Bcircular2 = BiotSavart([Coil(curve, Current(current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        Bcircular2.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.B(), Bcircular2.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular2.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient

    def test_helicalcoil_Bfield(self):
        point = [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]]
        field = [[-0.00101961, 0.20767292, -0.00224908]]
        derivative = [[[0.47545098, 0.01847397, 1.10223595], [0.01847426, -2.66700072, 0.01849548], [1.10237535, 0.01847085, 2.19154973]]]
        curves = [CurveHelical(100, 2, 5, 2, 1., 0.3) for i in range(2)]
        curves[0].set_dofs(np.concatenate(([0, 0], [0, 0])))
        curves[1].set_dofs(np.concatenate(([np.pi/2, 0], [0, 0])))
        currents = [-3.07e5, 3.07e5]
        Bhelical = BiotSavart([
            Coil(curves[0], Current(currents[0])),
            Coil(curves[1], Current(currents[1]))])
        Bhelical.set_points(point)
        assert np.allclose(Bhelical.B(), field)
        assert np.allclose(Bhelical.dB_by_dX(), derivative)

    def test_Dommaschk(self):
        mn = [[10, 2], [15, 3]]
        coeffs = [[-2.18, -2.18], [25.8, -25.8]]
        Bfield = Dommaschk(mn=mn, coeffs=coeffs)
        Bfield.set_points([[0.9231, 0.8423, -0.1123]])
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify B
        assert np.allclose(Bfield.B(), [[-1.72696, 3.26173, -2.22013]])
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, np.array([[-59.9602, 8.96793, -24.8844], [8.96793, 49.0327, -18.4131], [-24.8844, -18.4131, 10.9275]]))

    def test_BifieldMultiply(self):
        scalar = 1.2345
        pointVar = 1e-1
        npoints = 20
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += pointVar * (np.random.rand(*points.shape)-0.5)
        ## Multiply by left side
        Bfield1 = ToroidalField(1.23498, 0.012389)
        Bfield2 = scalar*ToroidalField(1.23498, 0.012389)
        Bfield1.set_points(points)
        Bfield2.set_points(points)
        # Verify B
        assert np.allclose(Bfield2.B(), scalar*np.array(Bfield1.B()))
        assert np.allclose(Bfield2.dB_by_dX(), scalar*np.array(Bfield1.dB_by_dX()))
        assert np.allclose(Bfield2.d2B_by_dXdX(), scalar*np.array(Bfield1.d2B_by_dXdX()))
        # Verify A
        assert np.allclose(Bfield2.A(), scalar*np.array(Bfield1.A()))
        assert np.allclose(Bfield2.dA_by_dX(), scalar*np.array(Bfield1.dA_by_dX()))
        assert np.allclose(Bfield2.d2A_by_dXdX(), scalar*np.array(Bfield1.d2A_by_dXdX()))
        ## Multiply by right side
        Bfield1 = ToroidalField(1.91784391874, 0.2836482)
        Bfield2 = ToroidalField(1.91784391874, 0.2836482)*scalar
        Bfield1.set_points(points)
        Bfield2.set_points(points)
        # Verify B
        assert np.allclose(Bfield2.B(), scalar*np.array(Bfield1.B()))
        assert np.allclose(Bfield2.dB_by_dX(), scalar*np.array(Bfield1.dB_by_dX()))
        assert np.allclose(Bfield2.d2B_by_dXdX(), scalar*np.array(Bfield1.d2B_by_dXdX()))
        # Verify A
        assert np.allclose(Bfield2.A(), scalar*np.array(Bfield1.A()))
        assert np.allclose(Bfield2.dA_by_dX(), scalar*np.array(Bfield1.dA_by_dX()))
        assert np.allclose(Bfield2.d2A_by_dXdX(), scalar*np.array(Bfield1.d2A_by_dXdX()))

    def test_Reiman(self):
        iota0 = 0.15
        iota1 = 0.38
        k = [6]
        epsilonk = [0.01]
        # point locations
        pointVar = 1e-1
        npoints = 20
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += pointVar * (np.random.rand(*points.shape)-0.5)
        # Bfield from class
        Bfield = Reiman(iota0=iota0, iota1=iota1, k=k, epsilonk=epsilonk)
        Bfield.set_points(points)
        B1 = np.array(Bfield.B())
        # Check that div(B)=0
        dB1 = Bfield.dB_by_dX()
        assert np.allclose(dB1[:, 0, 0]+dB1[:, 1, 1]+dB1[:, 2, 2], np.zeros((npoints)))
        # Bfield analytical
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        Bx = (y*np.sqrt(x**2 + y**2) + x*z*(0.15 + 0.38*((-1 + np.sqrt(x**2 + y**2))**2 + z**2) - 
                                            0.06*((-1 + np.sqrt(x**2 + y**2))**2 + z**2)**2*np.cos(np.arctan2(y, x) - 6*np.arctan(z/(-1 + np.sqrt(x**2 + y**2))))) + 
              0.06*x*(1 - np.sqrt(x**2 + y**2))*((-1 + np.sqrt(x**2 + y**2))**2 + z**2)**2 *
              np.sin(np.arctan2(y, x) - 6*np.arctan(z/(-1 + np.sqrt(x**2 + y**2)))))/(x**2 + y**2)
        By = (-1.*x*np.sqrt(x**2 + y**2) + y*z*(0.15 + 0.38*((-1 + np.sqrt(x**2 + y**2))**2 + z**2) - 
                                                0.06*((-1 + np.sqrt(x**2 + y**2))**2 + z**2)**2*np.cos(np.arctan2(y, x) - 6*np.arctan(z/(-1 + np.sqrt(x**2 + y**2))))) + 
              0.06*y*(1 - np.sqrt(x**2 + y**2))*((-1 + np.sqrt(x**2 + y**2))**2 + z**2)**2 *
              np.sin(np.arctan2(y, x) - 6*np.arctan(z/(-1 + np.sqrt(x**2 + y**2)))))/(x**2 + y**2)
        Bz = (-((-1 + np.sqrt(x**2 + y**2))*(0.15 + 0.38*((-1 + np.sqrt(x**2 + y**2))**2 + z**2) - 
                                             0.06*((-1 + np.sqrt(x**2 + y**2))**2 + z**2)**2*np.cos(np.arctan2(y, x) - 6*np.arctan(z/(-1 + np.sqrt(x**2 + y**2)))))) - 
              0.06*z*((-1 + np.sqrt(x**2 + y**2))**2 + z**2)**2*np.sin(np.arctan2(y, x) - 6*np.arctan(z/(-1 + np.sqrt(x**2 + y**2)))))/np.sqrt(x**2 + y**2)
        B2 = np.array(np.vstack((Bx, By, Bz)).T)
        assert np.allclose(B1, B2)
        # Derivative
        points = [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]]
        Bfield.set_points(points)
        dB1 = np.array(Bfield.dB_by_dX()[0])
        dB2 = np.array([[1.68810242e-03, -1.11110794e+00, 3.11091859e-04],
                        [2.57225263e-06, -1.69487835e-03, -1.98320069e-01],
                        [-2.68700789e-04, 1.70889034e-01, 6.77592533e-06]])
        assert np.allclose(dB1, dB2)

    def subtest_reiman_dBdX_taylortest(self, idx):
        iota0 = 0.15
        iota1 = 0.38
        k = [6]
        epsilonk = [0.01]
        bs = Reiman(iota0=iota0, iota1=iota1, k=k, epsilonk=epsilonk)
        points = np.asarray(17 * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        points += 0.001 * (np.random.rand(*points.shape)-0.5)
        bs.set_points(points)
        B0 = bs.B()[idx]
        dB = bs.dB_by_dX()[idx]
        for direction in [np.asarray((1., 0, 0)), np.asarray((0, 1., 0)), np.asarray((0, 0, 1.))]:
            deriv = dB.T.dot(direction)
            err = 1e6
            for i in range(5, 10):
                eps = 0.5**i
                bs.set_points(points + eps * direction)
                Beps = bs.B()[idx]
                deriv_est = (Beps-B0)/(eps)
                new_err = np.linalg.norm(deriv-deriv_est)
                assert new_err < 0.55 * err
                err = new_err

    def test_reiman_dBdX_taylortest(self):
        for idx in [0, 16]:
            with self.subTest(idx=idx):
                self.subtest_reiman_dBdX_taylortest(idx)

    def test_interpolated_field_close_with_symmetries(self):
        R0test = 1.5
        B0test = 0.8
        B0 = ToroidalField(R0test, B0test)

        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        btotal = bs + B0
        n = 12
        rmin = 1.5
        rmax = 1.7
        rsteps = n
        phimin = 0
        phimax = 2*np.pi/nfp
        phisteps = n*32//nfp
        zmin = 0.
        zmax = 0.1
        zsteps = n//2
        bsh = InterpolatedField(
            btotal, 4, [rmin, rmax, rsteps], [phimin, phimax, phisteps], [zmin, zmax, zsteps],
            True, nfp=nfp, stellsym=True)
        N = 1000
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(rmax-rmin) + rmin
        points[:, 1] = points[:, 1]*(nfp*phimax-phimin) + phimin
        points[:, 2] = points[:, 2]*(2*zmax) - zmax
        btotal.set_points_cyl(points)
        dB = btotal.GradAbsB()
        B = btotal.B()
        dBc = btotal.GradAbsB_cyl()
        Bc = btotal.B_cyl()
        bsh.set_points_cyl(points)
        Bh = bsh.B()
        dBh = bsh.GradAbsB()
        Bhc = bsh.B_cyl()
        dBhc = bsh.GradAbsB_cyl()
        assert np.allclose(B, Bh, rtol=1e-3)
        assert np.allclose(dB, dBh, rtol=1e-3)
        assert np.allclose(Bc, Bhc, rtol=1e-3)
        assert np.allclose(dBc, dBhc, rtol=1e-3)

    def test_interpolated_field_close_no_sym(self):
        R0test = 1.5
        B0test = 0.8
        B0 = ToroidalField(R0test, B0test)

        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        btotal = bs + B0
        n = 8
        rmin = 1.5
        rmax = 1.7
        rsteps = n
        phimin = 0
        phimax = 2*np.pi
        phisteps = n*16
        zmin = -0.1
        zmax = 0.1
        zsteps = n
        bsh = InterpolatedField(
            btotal, 4, [rmin, rmax, rsteps], [phimin, phimax, phisteps], [zmin, zmax, zsteps],
            True)
        N = 100
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(rmax-rmin) + rmin
        points[:, 1] = points[:, 1]*(phimax-phimin) + phimin
        points[:, 2] = points[:, 2]*(zmax-zmin) + zmin
        btotal.set_points_cyl(points)
        dB = btotal.GradAbsB()
        B = btotal.B()
        dBc = btotal.GradAbsB_cyl()
        Bc = btotal.B_cyl()
        bsh.set_points_cyl(points)
        Bh = bsh.B()
        dBh = bsh.GradAbsB()
        Bhc = bsh.B_cyl()
        dBhc = bsh.GradAbsB_cyl()
        assert np.allclose(B, Bh, rtol=1e-2)
        assert np.allclose(dB, dBh, rtol=1e-2)
        assert np.allclose(Bc, Bhc, rtol=1e-2)
        assert np.allclose(dBc, dBhc, rtol=1e-2)

    def test_interpolated_field_convergence_rate(self):
        R0test = 1.5
        B0test = 0.8
        B0 = ToroidalField(R0test, B0test)

        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        old_err_1 = 1e6
        old_err_2 = 1e6
        btotal = bs + B0

        for n in [4, 8, 16]:
            rmin = 1.5
            rmax = 1.7
            rsteps = n
            phimin = 0
            phimax = 2*np.pi
            phisteps = n*16
            zmin = -0.1
            zmax = 0.1
            zsteps = n
            bsh = InterpolatedField(btotal, 2, [rmin, rmax, rsteps], [phimin, phimax, phisteps], [zmin, zmax, zsteps], True)
            err_1 = np.mean(bsh.estimate_error_B(1000))
            err_2 = np.mean(bsh.estimate_error_GradAbsB(1000))
            print(err_1, err_2)
            assert err_1 < 0.6**3 * old_err_1
            assert err_2 < 0.6**3 * old_err_2
            old_err_1 = err_1
            old_err_2 = err_2

    def test_get_set_points_cyl_cart(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)

        points_xyz = np.asarray([[0.5, 0.6, 0.7]])
        points_rphiz = np.zeros_like(points_xyz)
        points_rphiz[:, 0] = np.linalg.norm(points_xyz[:, 0:2], axis=1)
        points_rphiz[:, 1] = np.mod(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]), 2*np.pi)
        points_rphiz[:, 2] = points_xyz[:, 2]
        bs.set_points_cyl(points_rphiz)
        # import IPython; IPython.embed()
        # import sys; sys.exit()
        assert np.allclose(bs.get_points_cyl(), points_rphiz)
        assert np.allclose(bs.get_points_cart(), points_xyz)

        bs.set_points_cart(points_xyz)
        assert np.allclose(bs.get_points_cyl(), points_rphiz)
        assert np.allclose(bs.get_points_cart(), points_xyz)

    @unittest.skipIf(not pyevtk_found, "pyevtk not found")
    def test_to_vtk(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        bs.to_vtk('/tmp/bfield')

    def test_poloidal_field(self):
        B0 = 1.1
        R0 = 1.2
        q = 1.3
        # point locations
        points = [
            [-1.41513202e-3, 8.99999382e-1, -3.14473221e-4],
            [0.1231, 2.4123, 0.002341],
        ]
        # Bfield from class
        Bfield = PoloidalField(R0=R0, B0=B0, q=q)
        Bfield.set_points(points)
        B1 = Bfield.B()
        dB1 = Bfield.dB_by_dX()
        print(dB1)
        B1_analytical = [
            [-3.48663e-7, 0.000221744, -0.211538],
            [-0.0000841262, -0.00164856, 0.85704]
        ]
        dB1_analytical = [
            [[0.000246381, 3.87403e-7, 0.00110872],
             [3.87403e-7, 6.0914e-10, -0.705127],
             [-0.00110872, 0.705127, 0]],
            [[-0.000681623, 0.0000347833, -0.035936],
             [0.0000347833, -1.775e-6, -0.704212],
             [0.035936, 0.704212, 0]]
        ]
        assert np.allclose(B1, B1_analytical)
        assert np.allclose(dB1, dB1_analytical)


if __name__ == "__main__":
    unittest.main()
