import json
import unittest
import tempfile
import numpy as np
from pathlib import Path

from monty.tempfile import ScratchDir
from scipy.io import netcdf_file
try:
    import sympy
except ImportError:
    sympy = None

try:
    import pyevtk
except ImportError:
    pyevtk = None

from simsopt._core.json import SIMSON, GSONDecoder, GSONEncoder
from simsopt.configs import get_ncsx_data
from simsopt.field import (BiotSavart, CircularCoil, Coil, Current,
                           DipoleField, Dommaschk, InterpolatedField,
                           MagneticFieldSum, PoloidalField, Reiman,
                           ScalarPotentialRZMagneticField, ToroidalField,
                           coils_via_symmetries, MirrorModel)
from simsopt.objectives import SquaredFlux
from simsopt.geo import (CurveHelical, CurveRZFourier, CurveXYZFourier,
                         PermanentMagnetGrid, SurfaceRZFourier, CurvePlanarFourier,
                         JaxCurvePlanarFourier, create_equally_spaced_curves)
from simsoptpp import dipole_field_Bn

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


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

        field_json_str = json.dumps(SIMSON(Bfield), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
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
        curves = [CurveHelical(101, 1, 5, 2, 1., 0.3) for i in range(2)]
        curves[0].x = [np.pi / 2, 0, 0]
        curves[1].x = [0, 0, 0]
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
        field_json_str = json.dumps(SIMSON(Btotal2), cls=GSONEncoder)
        Btotal_regen = json.loads(field_json_str, cls=GSONDecoder)
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

    @unittest.skipIf(sympy is None, "Sympy not found")
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
        field_json_str = json.dumps(SIMSON(Bscalar), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(B1, np.array(Bfield_regen.B())))

        # Analytical Formula for B
        rphiz = [[np.sqrt(np.power(point[0], 2) + np.power(point[1], 2)), np.arctan2(point[1], point[0]), point[2]] for point in points]
        B2 = np.array([[0.2*point[2]+0.8*point[0], (0.1+0.3*point[2])/point[0], 0.2*point[0]+0.3*point[1]+point[2]] for point in rphiz])
        # Convert to Cartesian coordinates
        r = np.sqrt(np.power(points[:, 0], 2) + np.power(points[:, 1], 2))
        phi = np.arctan2(points[:, 1], points[:, 0])
        B2_cart = np.zeros_like(B2)
        B2_cart[:, 0] = B2[:, 0] * np.cos(phi) - B2[:, 1] * np.sin(phi)
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
        field_json_str = json.dumps(SIMSON(Bfield), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(Bfield.B(), Bfield_regen.B()))

        def compare_gammas(circular_coil, general_coil):
            # Verify that the gamma values are the same, up to a shift in the
            # array index.
            gamma1 = general_coil.curve.gamma()
            gamma2 = circular_coil.gamma(len(curve.quadpoints))
            if general_coil.current.get_value() * circular_coil.I < 0:
                # Currents are opposite sign, so the direction of the points
                # will be reversed.
                gamma1 = np.flipud(gamma1)

            index = np.argmin(np.linalg.norm(gamma1[0, None] - gamma2, axis=1))
            gamma3 = np.roll(gamma2, -index, axis=0)
            np.testing.assert_allclose(gamma1, gamma3, atol=1e-14)

        # Verify that divergence is zero
        dB1_by_dX = Bfield.dB_by_dX()
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))
        # Verify that, as a vacuum field, grad B=grad grad phi so that grad_i B_j = grad_j B_i
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(dB1_by_dX, transpGradB1)
        ### compare to biosavart(circular_coil)
        ## at these points
        points = np.asarray(npoints * [[-1.41513202e-03, 8.99999382e-01, -3.14473221e-04]])
        np.random.seed(0)
        points += pointVar * (np.random.rand(*points.shape)-0.5)

        ## verify with a x^2+z^2=radius^2 circular coil
        normal = [np.pi/2, np.pi/2]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], radius, 0., center[1], 0., 0., center[2], 0., radius])
        general_coil = Coil(curve, Current(current))
        Bcircular = BiotSavart([general_coil])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))
        assert np.allclose(dB1_by_dX, transpGradB1)
        compare_gammas(Bfield, general_coil)

        # use normal = [0, 1, 0]
        normal = [0, 1, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], radius, 0., center[1], 0., 0., center[2], 0., radius])
        general_coil = Coil(curve, Current(current))
        Bcircular = BiotSavart([general_coil])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))
        assert np.allclose(dB1_by_dX, transpGradB1)
        compare_gammas(Bfield, general_coil)

        ## verify with a y^2+z^2=radius^2 circular coil
        normal = [0, np.pi/2]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, 0., center[1], radius, 0., center[2], 0., radius])
        general_coil = Coil(curve, Current(-current))
        Bcircular = BiotSavart([general_coil])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient
        compare_gammas(Bfield, general_coil)

        # one points
        Bfield.set_points(np.asarray([[0.1, 0.2, 0.3]]))
        Afield = Bfield.A()
        assert np.allclose(Afield, [[0, 5.15785, -2.643056]])

        # two points
        Bfield.set_points(np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))
        Afield = Bfield.A()
        assert np.allclose(Afield, [[0, 5.15785, -2.643056], [0, 5.15785, -2.643056]])

        # three points
        Bfield.set_points(np.asarray([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]))
        Afield = Bfield.A()
        assert np.allclose(Afield, [[0, 5.15785, -2.643056], [0, 5.15785, -2.643056], [0, 5.15785, -2.643056]])

        # use normal=[1,0,0]
        normal = [1, 0, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, 0., center[1], radius, 0., center[2], 0., radius])
        general_coil = Coil(curve, Current(-current))
        Bcircular = BiotSavart([general_coil])
        Bfield = CircularCoil(I=current, r0=radius, normal=normal, center=center)
        Bfield.set_points(points)
        Bcircular.set_points(points)
        dB1_by_dX = Bfield.dB_by_dX()
        transpGradB1 = [dBdx.T for dBdx in dB1_by_dX]
        assert np.allclose(Bfield.B(), Bcircular.B())
        assert np.allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX())
        assert np.allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)))  # divergence
        assert np.allclose(dB1_by_dX, transpGradB1)  # symmetry of the gradient
        compare_gammas(Bfield, general_coil)

        ## verify with a x^2+y^2=radius^2 circular coil
        center = [0, 0, 0]
        normal = [0, 0]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, radius, center[1], radius, 0., center[2], 0., 0.])
        general_coil = Coil(curve, Current(current))
        Bcircular = BiotSavart([general_coil])
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
        compare_gammas(Bfield, general_coil)

        # use normal = [0, 0, 1]
        center = [0, 0, 0]
        normal = [0, 0, 1]
        curve = CurveXYZFourier(300, 1)
        curve.set_dofs([center[0], 0, radius, center[1], radius, 0., center[2], 0., 0.])
        general_coil = Coil(curve, Current(current))
        Bcircular = BiotSavart([general_coil])
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
        
        # use basic normal and verify CircularCoil against CurvePlanarFourier class
        normal = [0, 0, 1]
        alpha = np.arcsin(normal[1])
        delta = np.arccos(normal[2] / np.cos(alpha))
        center = [0, 0, 0]
        order = 1
        ppp = 300
        curve = CurvePlanarFourier(order*ppp, order)
        dofs = np.zeros(10)
        dofs[0] = radius
        dofs[1] = 0.0
        dofs[2] = 0.0
        dofs[3] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[4] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[5] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
        dofs[6] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
        # Now specify the center
        dofs[7] = center[0]
        dofs[8] = center[1]
        dofs[9] = center[2]
        curve.set_dofs(dofs)
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
        np.testing.assert_allclose(Bfield.B(), Bcircular.B(), atol=1e-10,rtol=1e-10, err_msg="Bfield and analytic Bcircular should be identical")
        np.testing.assert_allclose(Bfield.B(), Bcircular2.B(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular2 should be identical")
        np.testing.assert_allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular should have the same dB_by_dX")
        np.testing.assert_allclose(Bfield.dB_by_dX(), Bcircular2.dB_by_dX(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular2 should have the same dB_by_dX")
        np.testing.assert_allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)), atol=1e-10, rtol=1e-10, err_msg="Divergence should be zero")  # divergence
        np.testing.assert_allclose(dB1_by_dX, transpGradB1, atol=1e-10, rtol=1e-10, err_msg="Symmetry of the gradient should be preserved")  # symmetry of the gradient

        # Repeat the above test with JaxCurvePlanarFourier
        curve = JaxCurvePlanarFourier(order*ppp, order)
        dofs = np.zeros(10)
        dofs[0] = radius
        dofs[1] = 0.0
        dofs[2] = 0.0
        dofs[3] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[4] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[5] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
        dofs[6] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
        # Now specify the center
        dofs[7] = center[0]
        dofs[8] = center[1]
        dofs[9] = center[2]
        curve.set_dofs(dofs)
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
        np.testing.assert_allclose(Bfield.B(), Bcircular.B(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular should be identical")
        np.testing.assert_allclose(Bfield.B(), Bcircular2.B(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular2 should be identical")
        np.testing.assert_allclose(Bfield.dB_by_dX(), Bcircular.dB_by_dX(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular should have the same dB_by_dX")
        np.testing.assert_allclose(Bfield.dB_by_dX(), Bcircular2.dB_by_dX(), atol=1e-10, rtol=1e-10, err_msg="Bfield and analytic Bcircular2 should have the same dB_by_dX")
        np.testing.assert_allclose(dB1_by_dX[:, 0, 0]+dB1_by_dX[:, 1, 1]+dB1_by_dX[:, 2, 2], np.zeros((npoints)), atol=1e-10, rtol=1e-10, err_msg="Divergence should be zero")  # divergence
        np.testing.assert_allclose(dB1_by_dX, transpGradB1, atol=1e-10, rtol=1e-10, err_msg="Symmetry of the gradient should be preserved")  # symmetry of the gradient

        # use random normal and verify against CurvePlanarFourier class
        normal = np.random.rand(3)
        normal = normal / np.sqrt(np.sum(normal ** 2, axis=-1))
        alpha = np.arcsin(-normal[1])
        delta = np.arccos(normal[2] / np.cos(alpha))
        center = [0, 0, 0]
        order = 1
        ppp = 300
        curve = CurvePlanarFourier(order*ppp, order)
        dofs = np.zeros(10)
        dofs[0] = radius
        dofs[1] = 0.0
        dofs[2] = 0.0
        dofs[3] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[4] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[5] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
        dofs[6] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
        # Now specify the center
        dofs[7] = center[0]
        dofs[8] = center[1]
        dofs[9] = center[2]
        curve.set_dofs(dofs)
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

        # use random normal and verify against CurvePlanarFourier class
        normal = np.random.rand(3)
        normal = normal / np.sqrt(np.sum(normal ** 2, axis=-1))
        alpha = np.arcsin(-normal[1])
        delta = np.arccos(normal[2] / np.cos(alpha))
        center = [0, 0, 0]
        order = 1
        ppp = 300
        curve = JaxCurvePlanarFourier(order*ppp, order)
        dofs = np.zeros(10)
        dofs[0] = radius
        dofs[1] = 0.0
        dofs[2] = 0.0
        dofs[3] = np.cos(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[4] = np.sin(alpha / 2.0) * np.cos(delta / 2.0)
        dofs[5] = np.cos(alpha / 2.0) * np.sin(delta / 2.0)
        dofs[6] = -np.sin(alpha / 2.0) * np.sin(delta / 2.0)
        # Now specify the center
        dofs[7] = center[0]
        dofs[8] = center[1]
        dofs[9] = center[2]
        curve.set_dofs(dofs)
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
        np.testing.assert_allclose(field.B(), [[0.01016974, 0.00629875, -0.00220838]], rtol=1e-6)
        # test coil location
        np.testing.assert_allclose(field.gamma(points=4), [[1.3575, 1.456, 2.789], [0.123, center[1]+radius*np.cos(-angle), center[2]-radius*np.sin(-angle)],
                                                           [-1.1115, 1.456, 2.789], [0.123, center[1]-radius*np.cos(-angle), center[2]+radius*np.sin(-angle)]])
        with ScratchDir("."):
            for close in [True, False]:
                field.to_vtk('test', close=close)

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
        curves = [CurveHelical(100, 1, 5, 2, 1., 0.3) for i in range(2)]
        curves[0].x = [0, 0, 0]
        curves[1].x =[np.pi / 2, 0, 0]
        currents = [-3.07e5, 3.07e5]
        Bhelical = BiotSavart([
            Coil(curves[0], Current(currents[0])),
            Coil(curves[1], Current(currents[1]))])
        Bhelical.set_points(point)
        assert np.allclose(Bhelical.B(), field)
        assert np.allclose(Bhelical.dB_by_dX(), derivative)

        # Verify serialization works
        field_json_str = json.dumps(SIMSON(Bhelical), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
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
        field_json_str = json.dumps(SIMSON(Bfield), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(B, Bfield_regen.B()))

        #Field configuration from Dommaschk paper equation number (40)
        mn = [[5, 2], [5, 4], [5, 10]]
        coeffs = [[1.4, 1.4], [19.25, 0], [5.10e10, 5.10e10]]
        Bfield = Dommaschk(mn=mn, coeffs=coeffs)
        point = np.asarray([[0.71879008, 0.76265643, 0.0745]])
        Bfield.set_points(point)
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        B = Bfield.B()
        assert np.allclose(B, [[-0.7094243, 0.65632967, -0.125321]])
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, np.array([[0.90663628, 0.5078183, -0.55436901],
                                            [0.5078183, 0.27261978, -0.66073972],
                                            [-0.55436901, -0.66073972, -1.17925605]]))
        #Test field
        mn = [[3, 2], [6, 4], [2, 11]]
        coeffs = [[1.4, 1.4], [19.25, 0], [5.10e10, 5.10e10]]
        Bfield = Dommaschk(mn=mn, coeffs=coeffs)
        point = np.asarray([[0.77066908, -0.61182119, 0.1057]])
        Bfield.set_points(point)
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        B = Bfield.B()
        assert np.allclose(B, [[0.55674279, 0.83401312, -0.121491]])
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, np.array([[0.11538721234011184, -0.7518405857812525, -0.6107605261251816],
                                            [-0.7518410735861303, 1.0695191900989125, 0.14110885184619465],
                                            [-0.6107606676662055, 0.1411086735566982, -1.18491]]))
        #Test field 2
        mn = [[5, 0], [10, 10], [15, 19]]
        coeffs = [[1.4, 1.4], [5.10e10, 5.10e10], [9e20, 0]]
        Bfield = Dommaschk(mn=mn, coeffs=coeffs)
        point = np.asarray([[0.06660615, -0.93924128, 0.16]])
        Bfield.set_points(point)
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        B = Bfield.B()
        assert np.allclose(B, [[3.90161959, -1.87151853, 0.0119783]])
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, np.array([[39.394312086253024, 14.061725133810995, 0.1684479703125076],
                                            [14.061729381899355, -40.23304445668633, -0.40810476986895994],
                                            [0.16844815337021118, -0.4081047568874514, 0.838733]]))
        # Verify serialization works
        field_json_str = json.dumps(SIMSON(Bfield), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(Bfield.B(), Bfield_regen.B()))

    def test_MirrorModel(self):
        """
        For MirrorModel, compare to reference values from Rogerio Jorge's
        Mathematica notebook.
        """
        Bfield = MirrorModel(B0=6.51292, gamma=0.124904, Z_m=0.98)
        point = np.asarray([[0.9231, 0.8423, -0.1123]])
        Bfield.set_points(point)
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify B
        B = Bfield.B()
        assert np.allclose(B, [[0.172472, 0.157375, 0.551171]])
        assert np.allclose(transpGradB, np.array([[0.18684, 0, -1.66368], [0, 0.18684, -1.51805], [0, 0, -0.373679]]))
        # Verify serialization works
        field_json_str = json.dumps(SIMSON(Bfield), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
        self.assertTrue(np.allclose(B, Bfield_regen.B()))

    def test_DipoleField_single_dipole(self):
        m = np.array([0.5, 0.5, 0.5])
        m_loc = np.array([0.1, -0.1, 1]).reshape(1, 3)
        field_loc = np.array([1, 0.2, 0.5]).reshape(1, 3)
        Bfield = DipoleField(m_loc, m, stellsym=False, coordinate_flag='cartesian')
        Bfield.set_points(field_loc)
        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify B
        assert np.allclose(Bfield.B(), 1e-7 * np.array([[0.260891, -0.183328, -0.77562]]))
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]]))
        # Verify A
        assert np.allclose(Bfield.A(), 1e-7 * np.array([[-0.324349, 0.567611, -0.243262]]))
        # Verify gradA
        gradA = np.array(Bfield.dA_by_dX())
        assert np.allclose(gradA, 1e-7 * np.array([[0.76151796, -0.151597, -0.0176294], [-0.92722, -0.444219, 0.3349286], [0.1657024, 0.5958156, -0.31730]]))

    def test_DipoleField_multiple_dipoles(self):
        Ndipoles = 100
        m = np.ravel(np.outer(np.ones(Ndipoles), np.array([0.5, 0.5, 0.5])))
        m_loc = np.outer(np.ones(Ndipoles), np.array([0.1, -0.1, 1]))
        field_loc = np.outer(np.ones(1001), np.array([1, 0.2, 0.5]))
        Bfield = DipoleField(m_loc, m, stellsym=False, coordinate_flag='cartesian')
        Bfield.set_points(field_loc)
        B_simsopt = Bfield.B()
        B_correct = Ndipoles * 1e-7 * np.array([0.260891, -0.183328, -0.77562])
        # Verify B
        assert np.allclose(B_simsopt, B_correct)

        gradB_simsopt = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])

        gradB = np.array(Bfield.dB_by_dX())
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, gradB_simsopt, atol=1e-4)
        # Verify A
        assert np.allclose(Bfield.A(), Ndipoles * 1e-7 * np.array([[-0.324349, 0.567611, -0.243262]]), atol=1e-4)
        # Verify gradA
        gradA = np.array(Bfield.dA_by_dX())
        assert np.allclose(gradA, Ndipoles * 1e-7 * np.array([[0.76151796, -0.151597, -0.0176294], [-0.92722, -0.444219, 0.3349286], [0.1657024, 0.5958156, -0.31730]]), atol=1e-4)

        # Save to vtk
        with ScratchDir("."):
            Bfield._toVTK('test')

    def test_DipoleField_multiple_points_multiple_dipoles(self):
        Ndipoles = 101
        m = np.ravel(np.outer(np.ones(Ndipoles), np.array([0.5, 0.5, 0.5])))
        m_loc = np.outer(np.ones(Ndipoles), np.array([0.1, -0.1, 1]))
        field_loc = np.array([[1, 0.2, 0.5], [-1, 0.5, 0.0], [0.1, 0.5, 0.5]])
        Bfield = DipoleField(m_loc, m, coordinate_flag='cartesian')
        Bfield.set_points(field_loc)
        B_simsopt = Bfield.B()
        A_simsopt = Bfield.A()
        B_correct = Ndipoles * 1e-7 * np.array([[0.260891, -0.183328, -0.77562], [0.11238748, -0.248857, 0.0911378], [0.0, -0.73980, -1.307552]])
        A_correct = Ndipoles * 1e-7 * np.array([[-0.324349, 0.567611, -0.243262], [-0.194174, -0.0121359, 0.20631], [-1.15443, 0.524742, 0.62969]])
        # Verify B
        assert np.allclose(B_simsopt, B_correct, atol=1e-4)
        # Verify B
        assert np.allclose(A_simsopt, A_correct, atol=1e-4)

        field_loc = np.array([[1, 0.2, 0.5], [1, 0.2, 0.5], [1, 0.2, 0.5]])
        gradB = np.array(Bfield.dB_by_dX())
        gradB_simsopt = np.zeros((3, 3, 3))
        gradB_simsopt[0, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        gradB_simsopt[1, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        gradB_simsopt[2, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, gradB_simsopt, atol=1e-4)

        # Repeat in cylindrical coords
        Bfield = DipoleField(m_loc, m, coordinate_flag='cylindrical')
        Bfield.set_points(field_loc)
        B_simsopt = Bfield.B()
        A_simsopt = Bfield.A()
        B_correct = Ndipoles * 1e-7 * np.array([[0.260891, -0.183328, -0.77562], [0.11238748, -0.248857, 0.0911378], [0.0, -0.73980, -1.307552]])
        A_correct = Ndipoles * 1e-7 * np.array([[-0.324349, 0.567611, -0.243262], [-0.194174, -0.0121359, 0.20631], [-1.15443, 0.524742, 0.62969]])
        # Verify B
        assert np.allclose(B_simsopt, B_correct, atol=1e-4)
        # Verify B
        assert np.allclose(A_simsopt, A_correct, atol=1e-4)

        field_loc = np.array([[1, 0.2, 0.5], [1, 0.2, 0.5], [1, 0.2, 0.5]])
        gradB = np.array(Bfield.dB_by_dX())
        gradB_simsopt = np.zeros((3, 3, 3))
        gradB_simsopt[0, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        gradB_simsopt[1, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        gradB_simsopt[2, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, gradB_simsopt, atol=1e-4)

        # Repeat with toroidal orientation
        Bfield = DipoleField(m_loc, m, coordinate_flag='toroidal')
        Bfield.set_points(field_loc)
        B_simsopt = Bfield.B()
        A_simsopt = Bfield.A()
        B_correct = Ndipoles * 1e-7 * np.array([[0.260891, -0.183328, -0.77562], [0.11238748, -0.248857, 0.0911378], [0.0, -0.73980, -1.307552]])
        A_correct = Ndipoles * 1e-7 * np.array([[-0.324349, 0.567611, -0.243262], [-0.194174, -0.0121359, 0.20631], [-1.15443, 0.524742, 0.62969]])
        # Verify B
        assert np.allclose(B_simsopt, B_correct, atol=1e-4)
        # Verify B
        assert np.allclose(A_simsopt, A_correct, atol=1e-4)

        field_loc = np.array([[1, 0.2, 0.5], [1, 0.2, 0.5], [1, 0.2, 0.5]])
        gradB = np.array(Bfield.dB_by_dX())
        gradB_simsopt = np.zeros((3, 3, 3))
        gradB_simsopt[0, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        gradB_simsopt[1, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        gradB_simsopt[2, :, :] = Ndipoles * 1e-7 * np.array([[0.03678574, 0.40007205, 1.8716069], [0.40007205, 1.085255, 0.27131429], [1.8716069, 0.27131429, -1.122044]])
        transpGradB = np.array([dBdx.T for dBdx in gradB])
        # Verify gradB is symmetric and its value
        assert np.allclose(gradB, transpGradB)
        assert np.allclose(gradB, gradB_simsopt, atol=1e-4)

    def test_pmopt_dipoles(self):
        """
        Test that A * m in the permanent magnet optimizer class
        agrees with SquaredFlux function using Bn from the DipoleField
        class, with range of different plasma surfaces with different
        values of field-period symmetry.
        """
        nphi = 8
        ntheta = 8
        file_tests = ["input.LandremanPaul2021_QA", "input.W7-X_standard_configuration",
                      "input.LandremanPaul2021_QH_reactorScale_lowres",
                      "input.circular_tokamak", "input.rotating_ellipse"]

        for filename in file_tests:
            sfilename = TEST_DIR / filename
            if filename[:4] == 'wout':
                s = SurfaceRZFourier.from_wout(sfilename, range="half period", nphi=nphi, ntheta=ntheta)
                s_inner = SurfaceRZFourier.from_wout(sfilename, range="half period", nphi=nphi, ntheta=ntheta)
                s_outer = SurfaceRZFourier.from_wout(sfilename, range="half period", nphi=nphi, ntheta=ntheta)
            else:
                s = SurfaceRZFourier.from_vmec_input(sfilename, range="half period", nphi=nphi, ntheta=ntheta)
                s_inner = SurfaceRZFourier.from_vmec_input(sfilename, range="half period", nphi=nphi, ntheta=ntheta)
                s_outer = SurfaceRZFourier.from_vmec_input(sfilename, range="half period", nphi=nphi, ntheta=ntheta)
            # Make the inner and outer surfaces by extending the plasma surface
            s_inner.extend_via_projected_normal(0.1)
            s_outer.extend_via_projected_normal(0.2)

            base_curves = create_equally_spaced_curves(2, s.nfp, stellsym=True, R0=0.5, R1=1.0, order=2)
            base_currents = [Current(1e5) for i in range(2)]
            coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
            bs = BiotSavart(coils)
            bs.set_points(s.gamma().reshape((-1, 3)))
            Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
            with ScratchDir("."):
                pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                    s, Bn, s_inner, s_outer)
            dipoles = np.random.rand(pm_opt.ndipoles * 3)
            pm_opt.m = dipoles
            b_dipole = DipoleField(pm_opt.dipole_grid_xyz,
                                   pm_opt.m,
                                   nfp=s.nfp,
                                   stellsym=s.stellsym,
                                   coordinate_flag=pm_opt.coordinate_flag,
                                   m_maxima=pm_opt.m_maxima)
            b_dipole.set_points(s.gamma().reshape((-1, 3)))
            # check Bn
            Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
            Ngrid = nphi * ntheta
            Bn_Am = (pm_opt.A_obj.dot(pm_opt.m) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms)
            assert np.allclose(Bn_Am.reshape(nphi, ntheta), np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))
            # check <Bn>
            B_opt = np.mean(np.abs(pm_opt.A_obj.dot(dipoles) - pm_opt.b_obj) * np.sqrt(Ngrid / Nnorms))
            B_dipole_field = np.mean(np.abs(np.sum((bs.B() + b_dipole.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
            # Bn_dipole_only = np.sum(b_dipole.B().reshape(-1, 3) * s.unitnormal().reshape(-1, 3), axis=1)
            assert np.isclose(B_opt, B_dipole_field)
            A_dipole = dipole_field_Bn(s.gamma().reshape(-1, 3),
                                       pm_opt.dipole_grid_xyz,
                                       s.unitnormal().reshape(-1, 3),
                                       s.nfp, s.stellsym,
                                       pm_opt.b_obj, pm_opt.coordinate_flag)
            # Rescale
            A_dipole = A_dipole.reshape(Ngrid, pm_opt.ndipoles * 3)
            Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
            for i in range(A_dipole.shape[0]):
                A_dipole[i, :] = A_dipole[i, :] * np.sqrt(Nnorms[i] / Ngrid)
            ATb = A_dipole.T @ pm_opt.b_obj
            assert np.allclose(A_dipole, pm_opt.A_obj)
            assert np.allclose(ATb, pm_opt.ATb)
            # check integral Bn^2
            f_B_Am = 0.5 * np.linalg.norm(pm_opt.A_obj.dot(dipoles) - pm_opt.b_obj, ord=2) ** 2
            f_B = SquaredFlux(s, b_dipole, -Bn).J()
            assert np.isclose(f_B, f_B_Am)

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
        field_json_str = json.dumps(SIMSON(Bfield2), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
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
        field_json_str = json.dumps(SIMSON(Bfield), cls=GSONEncoder)
        Bfield_regen = json.loads(field_json_str, cls=GSONDecoder)
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

    def test_cyl_versions(self):
        R0test = 1.5
        B0test = 0.8
        B0 = ToroidalField(R0test, B0test)

        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        btotal = bs + B0
        rmin = 1.5
        rmax = 1.7
        phimin = 0
        phimax = 2*np.pi/nfp
        zmax = 0.1
        N = 1000
        points = np.random.uniform(size=(N, 3))
        points[:, 0] = points[:, 0]*(rmax-rmin) + rmin
        points[:, 1] = points[:, 1]*(nfp*phimax-phimin) + phimin
        points[:, 2] = points[:, 2]*(2*zmax) - zmax
        btotal.set_points_cyl(points)

        dB = btotal.GradAbsB()
        B = btotal.B()
        A = btotal.A()
        dB_cyl = btotal.GradAbsB_cyl()
        B_cyl = btotal.B_cyl()
        A_cyl = btotal.A_cyl()

        for j in range(N):
            phi = points[j, 1]
            rotation = np.array([[np.cos(phi), np.sin(phi), 0],
                                [-np.sin(phi), np.cos(phi), 0],
                                [0, 0, 1]])
            np.testing.assert_allclose(rotation @ B[j, :], B_cyl[j, :])
            np.testing.assert_allclose(rotation @ dB[j, :], dB_cyl[j, :])
            np.testing.assert_allclose(rotation @ A[j, :], A_cyl[j, :])

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
        bsh = InterpolatedField(btotal, 4, [rmin, rmax, rsteps], [phimin, phimax, phisteps],
                                [zmin, zmax, zsteps], True)
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
            bsh = InterpolatedField(btotal, 2, [rmin, rmax, rsteps], [phimin, phimax, phisteps],
                                    [zmin, zmax, zsteps], True)
            err_1 = np.mean(bsh.estimate_error_B(1000))
            err_2 = np.mean(bsh.estimate_error_GradAbsB(1000))
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

    @unittest.skipIf(pyevtk is None, "pyevtk not found")
    def test_to_vtk(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        bs.to_vtk('/tmp/bfield')

    def subtest_to_mgrid(self, include_potential):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = Path(tmpdir) / "mgrid.bfield.nc"
            bs.to_mgrid(filename, nfp=nfp, include_potential=include_potential)

            # Compare the B data in the file to a separate evaluation here
            with netcdf_file(filename, mmap=False) as f:
                rmin = f.variables["rmin"][()]
                rmax = f.variables["rmax"][()]
                zmin = f.variables["zmin"][()]
                zmax = f.variables["zmax"][()]
                nr = f.variables["ir"][()]
                nphi = f.variables["kp"][()]
                nz = f.variables["jz"][()]
                Br = f.variables["br_001"][()]
                Bphi = f.variables["bp_001"][()]
                Bz = f.variables["bz_001"][()]
                assert nr == f.dimensions["rad"]
                assert nphi == f.dimensions["phi"]
                assert nz == f.dimensions["zee"]
                assert Br.shape == (nphi, nz, nr)
                assert Bphi.shape == (nphi, nz, nr)
                assert Bz.shape == (nphi, nz, nr)
                if include_potential:
                    Ar = f.variables["ar_001"][()]
                    Aphi = f.variables["ap_001"][()]
                    Az = f.variables["az_001"][()]
                    assert Ar.shape == (nphi, nz, nr)
                    assert Aphi.shape == (nphi, nz, nr)
                    assert Az.shape == (nphi, nz, nr)
                r = np.linspace(rmin, rmax, nr)
                phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
                z = np.linspace(zmin, zmax, nz)
                for jr in range(nr):
                    for jphi in range(nphi):
                        for jz in range(nz):
                            bs.set_points_cyl(np.array([[r[jr], phi[jphi], z[jz]]]))
                            np.testing.assert_allclose(Br[jphi, jz, jr], bs.B_cyl()[0, 0])
                            np.testing.assert_allclose(Bphi[jphi, jz, jr], bs.B_cyl()[0, 1])
                            np.testing.assert_allclose(Bz[jphi, jz, jr], bs.B_cyl()[0, 2])
                            if include_potential:
                                np.testing.assert_allclose(Ar[jphi, jz, jr], bs.A_cyl()[0, 0])
                                np.testing.assert_allclose(Aphi[jphi, jz, jr], bs.A_cyl()[0, 1])
                                np.testing.assert_allclose(Az[jphi, jz, jr], bs.A_cyl()[0, 2])

    def test_to_mgrid(self):
        for include_potential in [True, False]:
            with self.subTest(include_potential=include_potential):
                self.subtest_to_mgrid(include_potential)

    def test_poloidal_field(self):
        B0 = 1.1
        R0 = 1.2
        q = 1.3
        # point locations
        points = np.asarray([[-1.41513202e-3, 8.99999382e-1, -3.14473221e-4],
                             [0.1231, 2.4123, 0.002341]])
        # Bfield from class
        Bfield = PoloidalField(R0=R0, B0=B0, q=q)
        Bfield.set_points(points)
        B1 = Bfield.B()
        dB1 = Bfield.dB_by_dX()
        B1_analytical = [[-3.48663e-7, 0.000221744, -0.211538],
                         [-0.0000841262, -0.00164856, 0.85704]]
        dB1_analytical = [[[0.000246381, 3.87403e-7, 0.00110872],
                           [3.87403e-7, 6.0914e-10, -0.705127],
                           [-0.00110872, 0.705127, 0]],
                          [[-0.000681623, 0.0000347833, -0.035936],
                           [0.0000347833, -1.775e-6, -0.704212],
                           [0.035936, 0.704212, 0]]]
        assert np.allclose(B1, B1_analytical)
        assert np.allclose(dB1, dB1_analytical)


if __name__ == "__main__":
    unittest.main()
