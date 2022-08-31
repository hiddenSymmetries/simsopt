import unittest
import json

import numpy as np
from monty.json import MontyEncoder, MontyDecoder
try:
    import pyevtk
    pyevtk_found = True
except ImportError:
    pyevtk_found = False

from simsopt.field.magneticfieldclasses import ToroidalField, \
    ScalarPotentialRZMagneticField, CircularCoil, Dommaschk, \
    Reiman, sympy_found, InterpolatedField, PoloidalField, \
    WindingSurfaceField
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.magneticfield import MagneticFieldSum
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart, CurrentPotential, CurrentPotentialFourier
from simsopt.field.coil import coils_via_symmetries, Coil, Current
from simsopt.configs.zoo import get_ncsx_data
from scipy.special import ellipk, ellipe
from pathlib import Path
from scipy.io import netcdf_file
import matplotlib.pyplot as plt
np.random.seed(100)

TEST_DIR = Path(__file__).parent / ".." / "test_files"

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

        field_json_str = json.dumps(Bfield, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        B1_regen = Bfield_regen.B()
        self.assertTrue(np.allclose(B1, B1_regen))

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

        B1 = Btotal1.B()
        B2 = Btotal2.B()

        # Verify serialization works
        field_json_str = json.dumps(Btotal2, cls=MontyEncoder)
        Btotal_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(B2, Btotal_regen.B()))

        # Verify
        assert np.allclose(B1, B2)
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

        # Verify serialization works
        field_json_str = json.dumps(Bscalar, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(B1, np.array(Bfield_regen.B())))

        # Analytical Formula for B
        rphiz = [[np.sqrt(np.power(point[0], 2) + np.power(point[1], 2)), np.arctan2(point[1], point[0]), point[2]] for point in points]
        B2 = np.array([[0.2*point[2]+0.8*point[0], (0.1+0.3*point[2])/point[0], 0.2*point[0]+0.3*point[1]+point[2]] for point in rphiz])
        # Convert to Cartesian coordinates
        r = np.sqrt(np.power(points[:, 0], 2) + np.power(points[:, 1], 2))
        phi = np.arctan2(points[:, 1], points[:, 0])
        z = points[:, 2]
        B2_cart = np.zeros_like(B2)
        # Bx = Br cos(phi) - Bphi sin(phi)
        B2_cart[:, 0] = B2[:, 0] * np.cos(phi) - B2[:, 1] * np.sin(phi)
        # By = Br sin(phi) + Bphi cos(phi)
        B2_cart[:, 1] = B2[:, 0] * np.sin(phi) + B2[:, 1] * np.cos(phi)
        B2_cart[:, 2] = B2[:, 2]
        dB2_by_dX = np.array([
            [[0.8*np.cos(point[1]), -(np.cos(point[1])/point[0]**2)*(0.1+0.3*point[2]), 0.2*np.cos(point[1])-0.3*np.sin(point[1])/point[0]],
             [0.8*np.sin(point[1]), -(np.sin(point[1])/point[0]**2)*(0.1+0.3*point[2]), 0.2*np.sin(point[1])+0.3*np.cos(point[1])/point[0]],
             [0.2, 0.3/point[0], 1]] for point in rphiz])
        dBxdx = dB1_by_dX[:, 0, 0]
        dBxdy = dB1_by_dX[:, 1, 0]
        dBxdz = dB1_by_dX[:, 2, 0]
        dBydx = dB1_by_dX[:, 0, 1]
        dBydy = dB1_by_dX[:, 1, 1]
        dBydz = dB1_by_dX[:, 2, 1]
        dB1_by_dX_cyl = np.zeros_like(dB2_by_dX)
        dcosphidx = -points[:, 0]**2/r**3 + 1/r
        dsinphidx = -points[:, 0]*points[:, 1]/r**3
        dcosphidy = -points[:, 0]*points[:, 1]/r**3
        dsinphidy = -points[:, 1]**2/r**3 + 1/r
        Bx = B1[:, 0]
        By = B1[:, 1]
        # Br = Bx cos(phi) + By sin(phi)
        dB1_by_dX_cyl[:, 0, 0] = dBxdx * np.cos(phi) + Bx * dcosphidx + dBydx * np.sin(phi) \
            + By * dsinphidx
        dB1_by_dX_cyl[:, 1, 0] = dBxdy * np.cos(phi) + Bx * dcosphidy + dBydy * np.sin(phi) \
            + By * dsinphidy
        dB1_by_dX_cyl[:, 2, 0] = dBxdz * np.cos(phi) + dBydz * np.sin(phi)
        # Bphi = - sin(phi) Bx + cos(phi) By
        dB1_by_dX_cyl[:, 0, 1] = - dBxdx * np.sin(phi) - Bx * dsinphidx + dBydx * np.cos(phi) \
            + By * dcosphidx
        dB1_by_dX_cyl[:, 1, 1] = - dBxdy * np.sin(phi) - Bx * dsinphidy + dBydy * np.cos(phi) \
            + By * dcosphidy
        dB1_by_dX_cyl[:, 2, 1] = - dBxdz * np.sin(phi) + dBydz * np.cos(phi)
        dB1_by_dX_cyl[:, :, 2] = dB1_by_dX[:, :, 2]
        # Verify
        assert np.allclose(B1, B2_cart)
        assert np.allclose(dB1_by_dX_cyl, dB2_by_dX)

        # Check for divergence-free condition for dipole field
        # Set up magnetic field scalar potential
        PhiStr = "Z/(R*R + Z*Z)**(3/2)"
        # Set up scalar potential B
        Bscalar = ScalarPotentialRZMagneticField(PhiStr)
        Bscalar.set_points(points)
        dB1_by_dX = np.array(Bscalar.dB_by_dX())
        divB = dB1_by_dX[:, 0, 0] + dB1_by_dX[:, 1, 1] + dB1_by_dX[:, 2, 2]
        assert np.allclose(np.abs(divB), 0)

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

        # Verify serialization works
        field_json_str = json.dumps(Bfield, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(Bfield.B(), Bfield_regen.B()))

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
        normal = [np.pi/2, np.pi/2]
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
        normal = [0, np.pi/2]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, 0., center[1], radius, 0., center[2], 0., radius])
        Bcircular = BiotSavart([Coil(curve, Current(-current))])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient
        Bfield.set_points(np.asarray([[0.1, 0.2, 0.3]]))
        Afield = Bfield.A()
        assert np.allclose(Afield, [[0, 5.15785, -2.643056]])
        # use normal=[1,0,0]
        normal = [1, 0, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, 0., center[1], radius, 0., center[2], 0., radius])
        Bcircular = BiotSavart([Coil(curve, Current(-current))])
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
        ## Test with results from coilpy
        radius = 1.2345
        center = np.array([0.123, 1.456, 2.789])
        current = 1E6
        points = np.array([[2.987, 1.654, 0.321]])
        angle = 0.123
        field = CircularCoil(r0=radius, center=center, I=current, normal=[np.pi/2, -angle])
        field.set_points(points)
        assert np.allclose(field.B(), [[-1.29465197e-02, 2.56216948e-05, 3.70911295e-03]])
        angle = 0.982
        field = CircularCoil(r0=radius, center=center, I=current, normal=[np.pi/2, -angle])
        field.set_points(points)
        assert np.allclose(field.B(), [[-0.00916089, 0.00677598, 0.00294619]])
        angle = 2.435
        field = CircularCoil(r0=radius, center=center, I=current, normal=[np.pi/2, -angle])
        field.set_points(points)
        assert np.allclose(field.B(), [[0.01016974, 0.00629875, -0.00220838]])
        ## Random test
        radius = 1.2345
        center = np.array([0.123, 1.456, 2.789])
        current = 1E6
        points = np.array([[2.987, 1.654, 0.321]])
        angle = 2.435

        field = CircularCoil(r0=radius, center=center, I=current, normal=[np.pi/2, -angle])
        field.set_points(points)
        np.allclose(field.B(), [[0.01016974, 0.00629875, -0.00220838]])

    def test_circularcoil_Bfield_toroidal_arrangement(self):
        # This makes N_coils with centered at major radius R_m
        # each coil has N_turns which are evenly spaced between a1 and a2.
        R_m = 0.3048
        N_coils = 30

        N_turns = 3
        a1 = 10 / 2 * 0.0254
        a2 = 19.983 / 2 * 0.0254
        r_array = np.linspace(a1, a2, N_turns)
        I_amp = 433 * (33/N_turns)

        phi_ax = np.linspace(0, 2*np.pi, N_coils, endpoint=False) + (np.pi/N_coils)
        for xyz in range(3):
            # xyz = 0: Coil centers and eval points in the x-y plane.
            # xyz = 1: Coil centers and eval points in the y-z plane.
            # xyz = 2: Coil centers and eval points in the z-x plane.
            coils = []
            for j in np.arange(N_coils):

                for a_m in r_array:

                    phi = phi_ax[j]
                    if xyz == 0:
                        R0 = R_m * np.array([np.cos(phi), np.sin(phi), 0])
                        n1 = np.array([-np.sin(phi), np.cos(phi), 0])
                    elif xyz == 1:
                        R0 = R_m * np.array([0, np.cos(phi), np.sin(phi)])
                        n1 = np.array([0, -np.sin(phi), np.cos(phi)])
                    elif xyz == 2:
                        R0 = R_m * np.array([np.sin(phi), 0, np.cos(phi)])
                        n1 = np.array([np.cos(phi), 0, -np.sin(phi)])

                    B = CircularCoil(I=I_amp, r0=a_m, center=R0, normal=n1)
                    coils.append(B)

            B_field = sum(coils)

            ### setup target points
            N_points = 100
            ax = np.linspace(0, 2*np.pi, N_points, endpoint=False)

            if xyz == 0:
                points = R_m * np.array([np.cos(ax), np.sin(ax), 0*ax]).T
            elif xyz == 1:
                points = R_m * np.array([0*ax, np.cos(ax), np.sin(ax)]).T
            elif xyz == 2:
                points = R_m * np.array([np.sin(ax), 0*ax, np.cos(ax)]).T

            points = np.ascontiguousarray(points)

            B_field.set_points(points)

            ### evaluate
            Bout = B_field.B()

            #bx,by,bz = Bout.T
            bx, by, bz = np.nan_to_num(Bout).T      # maps NaN (which should not occur if running correctly) to 0
            bmag = np.sqrt(bx*bx + by*by + bz*bz)
            np.testing.assert_allclose(bmag, 0.281279, rtol=3e-05, atol=1e-5)

    def test_helicalcoil_Bfield(self):
        point = np.asarray([[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
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

        # Verify serialization works
        field_json_str = json.dumps(Bhelical, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(Bhelical.B(), Bfield_regen.B()))

    def test_Dommaschk(self):
        mn = [[10, 2], [15, 3]]
        coeffs = [[-2.18, -2.18], [25.8, -25.8]]
        Bfield = Dommaschk(mn=mn, coeffs=coeffs)
        point = np.asarray([[0.9231, 0.8423, -0.1123]])
        Bfield.set_points(point)
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify B
        B = Bfield.B()
        assert np.allclose(B, [[-1.72696, 3.26173, -2.22013]])
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, np.array([[-59.9602, 8.96793, -24.8844], [8.96793, 49.0327, -18.4131], [-24.8844, -18.4131, 10.9275]]))
        # Verify serialization works
        field_json_str = json.dumps(Bfield, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(B, Bfield_regen.B()))

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
        # Verify serialization works
        field_json_str = json.dumps(Bfield2, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(Bfield2.B(), Bfield_regen.B()))

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
        # Verify serialization works
        field_json_str = json.dumps(Bfield, cls=MontyEncoder)
        Bfield_regen = json.loads(field_json_str, cls=MontyDecoder)
        self.assertTrue(np.allclose(B1, Bfield_regen.B()))
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
        points = np.asarray([[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
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
        assert np.allclose(dBc, dBhc, rtol=1e-2, atol=1e-5)

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

        points_xyz = np.asarray([[0.5, 0.6, 0.7], [0.4, 0.1, 0.6]])
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

        f_contig = np.asfortranarray(points_xyz)
        bsbs = 2*bs
        with self.assertRaises(ValueError):
            bsbs.set_points_cart(f_contig)
        with self.assertRaises(ValueError):
            bsbs.set_points_cyl(f_contig)
        with self.assertRaises(ValueError):
            bsbs.set_points_cart(f_contig.flatten())
        with self.assertRaises(ValueError):
            bsbs.set_points_cyl(f_contig.flatten())

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
        points = np.asarray([
            [-1.41513202e-3, 8.99999382e-1, -3.14473221e-4],
            [0.1231, 2.4123, 0.002341],
        ])
        # Bfield from class
        Bfield = PoloidalField(R0=R0, B0=B0, q=q)
        Bfield.set_points(points)
        B1 = Bfield.B()
        dB1 = Bfield.dB_by_dX()
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

    def test_windingsurface_calculation(self):
        # Make a circular cross section, high-aspect ratio current loop in the Z = 0 plane
        # Following approximate analytic solution in Jackson problem 5.32 for a circular
        # cross section wire with major radius a, minor radius b, a >> b.
        filename = 'test_files/input.circular_tokamak_aspect_100'
        nphi = 32
        ntheta = 16
        winding_surface = SurfaceRZFourier.from_vmec_input(filename, range="full torus", nphi=nphi, ntheta=ntheta)

        # Make CurrentPotential class from this winding surface with 1 amp toroidal current
        ### Note, it appears we must pass the same quadpoints for now
        current_potential = CurrentPotentialFourier(winding_surface, net_poloidal_current_amperes=0, net_toroidal_current_amperes=-1, nfp=winding_surface.nfp, quadpoints_phi=winding_surface.quadpoints_phi, quadpoints_theta=winding_surface.quadpoints_theta)

        # compute the Bfield from this current loop at some nearby random points
        Bfield = WindingSurfaceField(current_potential)
        gamma = winding_surface.gamma().reshape((-1, 3))
        nx = 100
        ny = 32
        N = nx * ny

        # find the edge of the plasma
        b = np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 2] ** 2)) * np.ones(nx)
        phi = winding_surface.quadpoints_phi

        # approximation error ~ sqrt(eps ** 2) / a ~ 1e-10
        eps = 1e-18
        x = np.zeros(N)
        y = np.zeros(N)
        for i in range(nx):
            for j in range(ny):
                x[i * ny + j] = (b[i] + eps) * np.cos(phi[j])
                y[i * ny + j] = (b[i] + eps) * np.sin(phi[j])
        z = np.zeros(N)

        # Now have bunch of points that are right outside the plasma surface
        points = np.array([x, y, z]).T
        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        A_predict = Bfield.A()

        # calculate the Bfield analytically in spherical coordinates
        mu_fac = 2e-7
        r = np.sqrt(z ** 2 + x ** 2 + y ** 2)
        Aphi = mu_fac * (np.log(8 * 200 / r) - 2)

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        phi_points = np.arctan2(points[:, 1], points[:, 0])
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        print(A_predict, A_analytic, A_predict.shape, A_analytic.shape)
        assert np.allclose(A_predict, A_analytic)

    def test_windingsurface_exact(self):
        # Make an infinitesimally thin current loop in the Z = 0 plane
        # Following approximate analytic solution in Jackson 5.37
        nphi = 128
        ntheta = 8

        # Make winding surface with major radius = 1, no minor radius
        winding_surface = SurfaceRZFourier(nphi=nphi, ntheta=ntheta)
        for i in range(winding_surface.mpol + 1):
            for j in range(-winding_surface.ntor, winding_surface.ntor + 1):
                winding_surface.set_rc(i, j, 0.0)
                winding_surface.set_zs(i, j, 0.0)
        winding_surface.set_rc(0, 0, 1.0)
        winding_surface.set_rc(1, 0, 1e-8)  # current loop must have finite width for simsopt
        winding_surface.set_zs(1, 0, 1e-8)  # current loop must have finite width for simsopt

        # Make CurrentPotential class from this winding surface with 1 amp toroidal current
        ### Note, it appears we must pass the same quadpoints for now
        current_potential = CurrentPotentialFourier(winding_surface, net_poloidal_current_amperes=0, net_toroidal_current_amperes=-1, quadpoints_phi=winding_surface.quadpoints_phi, quadpoints_theta=winding_surface.quadpoints_theta)

        # compute the Bfield from this current loop at some points
        Bfield = WindingSurfaceField(current_potential)
        N = 1000
        phi = winding_surface.quadpoints_phi

        # Check that the full expression is correct
        points = (np.random.rand(N, 3) - 0.5) * 10
        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        A_predict = Bfield.A()

        # calculate the Bfield analytically in spherical coordinates
        mu_fac = 1e-7

        # See Jackson 5.37 for the vector potential in terms of the elliptic integrals
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        k = np.sqrt(4 * r * np.sin(theta) / (1 + r ** 2 + 2 * r * np.sin(theta)))

        # Note scipy is very annoying... scipy function ellipk(k^2)
        # is equivalent to what Jackson calls ellipk(k) so call it with k^2
        Aphi = mu_fac * (4 / np.sqrt(1 + r ** 2 + 2 * r * np.sin(theta))) * ((2 - k ** 2) * ellipk(k ** 2) - 2 * ellipe(k ** 2)) / k ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        phi_points = np.arctan2(points[:, 1], points[:, 0])
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic_elliptic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        assert np.allclose(A_predict, A_analytic_elliptic)

        # Now check that the far-field looks like a dipole
        points = (np.random.rand(N, 3) + 1) * 1000
        gamma = winding_surface.gamma().reshape((-1, 3))
        print(np.min(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)), np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2)))
        print(winding_surface.area(), winding_surface.volume())

        Bfield.set_points(np.ascontiguousarray(points))
        B_predict = Bfield.B()
        A_predict = Bfield.A()

        # calculate the Bfield analytically in spherical coordinates
        mu_fac = 1e-7

        # See Jackson 5.37 for the vector potential in terms of the elliptic integrals
        r = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
        theta = np.arctan2(np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2), points[:, 2])
        k = np.sqrt(4 * r * np.sin(theta) / (1 + r ** 2 + 2 * r * np.sin(theta)))

        # Note scipy is very annoying... scipy function ellipk(k^2)
        # is equivalent to what Jackson calls ellipk(k) so call it with k^2
        Aphi = mu_fac * (4 / np.sqrt(1 + r ** 2 + 2 * r * np.sin(theta))) * ((2 - k ** 2) * ellipk(k ** 2) - 2 * ellipe(k ** 2)) / k ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        phi_points = np.arctan2(points[:, 1], points[:, 0])
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic_elliptic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        assert np.allclose(A_predict, A_analytic_elliptic)

        # double check with vector potential of a dipole
        Aphi = np.pi * mu_fac * np.sin(theta) / r ** 2

        # convert A_analytic to Cartesian
        Ax = np.zeros(len(Aphi))
        Ay = np.zeros(len(Aphi))
        for i in range(N):
            Ax[i] = - np.sin(phi_points[i]) * Aphi[i]
            Ay[i] = np.cos(phi_points[i]) * Aphi[i]
        A_analytic = np.array([Ax, Ay, np.zeros(len(Aphi))]).T

        assert np.allclose(A_predict, A_analytic)

    def test_winding_surface_regcoil(self):
        # This compares the normal field from regcoil with that computed from
        # WindingSurface for a W7-X configuration 

        stellsym = True
        filename = TEST_DIR / 'regcoil_out.w7x.nc'
        f = netcdf_file(filename, 'r')
        Bnormal_regcoil = f.variables['Bnormal_total'][()][-1, :, :]

        r_plasma = f.variables['r_plasma'][()]
        r_coil = f.variables['r_coil'][()]
        rmnc_plasma  = f.variables['rmnc_plasma'][()]
        zmns_plasma  = f.variables['zmns_plasma'][()]
        xm_plasma = f.variables['xm_plasma'][()]
        xn_plasma = f.variables['xn_plasma'][()]
        nfp = f.variables['nfp'][()]
        ntheta_plasma = f.variables['ntheta_plasma'][()]
        nzeta_plasma = f.variables['nzeta_plasma'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        net_poloidal_current_amperes = f.variables['net_poloidal_current_Amperes'][()]
        net_toroidal_current_amperes = f.variables['net_toroidal_current_Amperes'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        mpol_plasma = int(np.max(xm_plasma))
        ntor_plasma = int(np.max(xn_plasma)/nfp)

        rmnc_coil = f.variables['rmnc_coil'][()]
        zmns_coil = f.variables['zmns_coil'][()]
        xm_coil = f.variables['xm_coil'][()]
        xn_coil = f.variables['xn_coil'][()]
        ntheta_coil = f.variables['ntheta_coil'][()]
        nzeta_coil = f.variables['nzeta_coil'][()]
        single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][-1, :]
        mpol_coil = int(np.max(xm_coil))
        ntor_coil = int(np.max(xn_coil)/nfp)

        s_plasma = SurfaceRZFourier(nfp=nfp, ntheta=ntheta_plasma, nphi=nzeta_plasma,
                             mpol=mpol_plasma, ntor=ntor_plasma, stellsym=stellsym, range="field period")
        s_plasma.set_dofs(0*s_plasma.get_dofs())
        for im in range(len(xm_plasma)):
            s_plasma.set_rc(xm_plasma[im], int(xn_plasma[im]/nfp), rmnc_plasma[im])
            s_plasma.set_zs(xm_plasma[im], int(xn_plasma[im]/nfp), zmns_plasma[im])

        assert np.allclose(r_plasma[0:nzeta_plasma,:,:],s_plasma.gamma())

        s_coil = SurfaceRZFourier(nfp=nfp, ntheta=ntheta_coil, nphi=nzeta_coil*nfp,
                             mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym, range="full torus")
        s_coil.set_dofs(0*s_coil.get_dofs())
        for im in range(len(xm_coil)):
            s_coil.set_rc(xm_coil[im], int(xn_coil[im]/nfp), rmnc_coil[im])
            s_coil.set_zs(xm_coil[im], int(xn_coil[im]/nfp), zmns_coil[im])

        assert np.allclose(r_coil,s_coil.gamma())

        cp = CurrentPotentialFourier(s_coil, nfp=nfp, stellsym=stellsym, mpol=mpol_potential,
                                     ntor=ntor_potential, nphi=nfp*nzeta_coil, ntheta=ntheta_coil,
                                     net_poloidal_current_amperes=net_poloidal_current_amperes,
                                     net_toroidal_current_amperes=net_toroidal_current_amperes, range="full torus")
        for im in range(len(xm_potential)):
            cp.set_phis(xm_potential[im], int(xn_potential[im]/nfp), single_valued_current_potential_mn[im])

        Bfield = WindingSurfaceField(cp)
        points = s_plasma.gamma().reshape((int(len(s_plasma.gamma().flatten())/3),3))

        Bfield.set_points(points)
        B = Bfield.B()

        normal = s_plasma.unitnormal().reshape((int(len(s_plasma.gamma().flatten())/3),3))
        Bnormal = np.sum(B*normal,axis=1).reshape(np.shape(s_plasma.gamma()[:,:,0]))

        Bnormal_average = np.mean(np.abs(Bnormal))

        print(np.max(np.abs(Bnormal.flatten()/Bnormal_average- Bnormal_regcoil.flatten()/Bnormal_average)))

        assert np.allclose(Bnormal.flatten()/Bnormal_average, Bnormal_regcoil.flatten()/Bnormal_average)

if __name__ == "__main__":
    unittest.main()
