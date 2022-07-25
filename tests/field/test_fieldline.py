import unittest
import logging
import numpy as np

from simsopt.field.magneticfieldclasses import ToroidalField, PoloidalField, InterpolatedField, UniformInterpolationRule
from simsopt.field.tracing import compute_fieldlines, particles_to_vtk, plot_poincare_data
from simsopt.field.biotsavart import BiotSavart
from simsopt.configs.zoo import get_ncsx_data
from simsopt.field.coil import coils_via_symmetries, Coil, Current
from simsopt.geo.curvehelical import CurveHelical
from simsopt.geo.curvexyzfourier import CurveXYZFourier
import simsoptpp as sopp

logging.basicConfig()

try:
    import pyevtk
    with_evtk = True
except ImportError:
    with_evtk = False


def validate_phi_hits(phi_hits, nphis):
    """
    Assert that we are hitting the phi planes in the correct order.
    For the toroidal field, we should always keep increasing in phi.
    """
    for i in range(len(phi_hits)-1):
        this_idx = int(phi_hits[i][1])
        next_idx = int(phi_hits[i+1][1])
        if not next_idx == (this_idx + 1) % nphis:
            return False
    return True


class FieldlineTesting(unittest.TestCase):

    def test_poincare_toroidal(self):
        logger = logging.getLogger('simsopt.field.tracing')
        logger.setLevel(1)
        # Test a toroidal magnetic field with no rotational transform
        R0test = 1.3
        B0test = 0.8
        Bfield = ToroidalField(R0test, B0test)
        nlines = 10
        R0 = [1.1 + i*0.1 for i in range(nlines)]
        Z0 = [0 for i in range(nlines)]
        nphis = 10
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            Bfield, R0, Z0, tmax=100, phis=phis, stopping_criteria=[])
        for i in range(nlines):
            assert np.allclose(res_tys[i][:, 3], 0.)
            assert np.allclose(np.linalg.norm(res_tys[i][:, 1:3], axis=1), R0[i])
            assert validate_phi_hits(res_phi_hits[i], nphis)
        if with_evtk:
            particles_to_vtk(res_tys, '/tmp/fieldlines')

    def test_poincare_tokamak(self):
        # Test a simple circular tokamak geometry that
        # consists of a superposition of a purely toroidal
        # and a purely poloidal magnetic field
        R0test = 1.0
        B0test = 1.0
        qtest = 3.2
        Bfield = ToroidalField(R0test, B0test)+PoloidalField(R0test, B0test, qtest)
        nlines = 4
        R0 = [1.05 + i*0.02 for i in range(nlines)]
        Z0 = [0 for i in range(nlines)]
        nphis = 4
        phis = np.linspace(0, 2*np.pi, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            Bfield, R0, Z0, tmax=10, phis=phis, stopping_criteria=[])
        # Check that Poincare plot is a circle in the R,Z plane with R centered at R0
        rtest = [[np.sqrt((np.sqrt(res_tys[i][j][1]**2+res_tys[i][j][2]**2)-R0test)**2+res_tys[i][j][3]**2)-R0[i]+R0test for j in range(len(res_tys[i]))] for i in range(len(res_tys))]
        assert [np.allclose(rtest[i], 0., rtol=1e-5, atol=1e-5) for i in range(nlines)]

    def test_poincare_plot(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        n = 10
        rrange = (1.0, 1.9, n)
        phirange = (0, 2*np.pi/nfp, n*2)
        zrange = (0, 0.4, n)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(2),
            rrange, phirange, zrange, True, nfp=3, stellsym=True
        )
        nlines = 4
        r0 = np.linalg.norm(ma.gamma()[0, :2])
        z0 = ma.gamma()[0, 2]
        R0 = [r0 + i*0.01 for i in range(nlines)]
        Z0 = [z0 for i in range(nlines)]
        nphis = 4
        phis = np.linspace(0, 2*np.pi/nfp, nphis, endpoint=False)
        res_tys, res_phi_hits = compute_fieldlines(
            bsh, R0, Z0, tmax=1000, phis=phis, stopping_criteria=[])
        try:
            import matplotlib  # noqa
            plot_poincare_data(res_phi_hits, phis, '/tmp/fieldlines.png')
        except ImportError:
            pass

    def test_poincare_ncsx_known(self):
        curves, currents, ma = get_ncsx_data()
        nfp = 3
        coils = coils_via_symmetries(curves, currents, nfp, True)
        bs = BiotSavart(coils)
        R0 = [np.linalg.norm(ma.gamma()[0, :2])]
        Z0 = [ma.gamma()[0, 2]]
        phis = np.arctan2(ma.gamma()[:, 1], ma.gamma()[:, 0])
        res_tys, res_phi_hits = compute_fieldlines(
            bs, R0, Z0, tmax=10, phis=phis, stopping_criteria=[])
        for i in range(len(phis)-1):
            assert np.linalg.norm(ma.gamma()[i+1, :] - res_phi_hits[0][i, 2:5]) < 1e-4

    def test_poincare_caryhanson(self):
        # Test with a known magnetic field - optimized Cary&Hanson configuration
        # with a magnetic axis at R=0.9413. Field created using the Biot-Savart
        # solver given a set of two helical coils created using the CurveHelical
        # class. The total magnetic field is a superposition of a helical and
        # a toroidal magnetic field.
        curves = [CurveHelical(200, 2, 5, 2, 1., 0.3) for i in range(2)]
        curves[0].set_dofs(np.concatenate(([np.pi/2, 0.2841], [0, 0])))
        curves[1].set_dofs(np.concatenate(([0, 0], [0, 0.2933])))
        currents = [3.07e5, -3.07e5]
        Btoroidal = ToroidalField(1.0, 1.0)
        Bhelical = BiotSavart([
            Coil(curves[0], Current(currents[0])),
            Coil(curves[1], Current(currents[1]))])
        bs = Bhelical+Btoroidal
        ma = CurveXYZFourier(300, 1)
        magnetic_axis_radius = 0.9413
        ma.set_dofs([0, 0, magnetic_axis_radius, 0, magnetic_axis_radius, 0, 0, 0, 0])
        R0 = [np.linalg.norm(ma.gamma()[0, :2])]
        Z0 = [ma.gamma()[0, 2]]        
        phis = np.arctan2(ma.gamma()[:, 1], ma.gamma()[:, 0])
        res_tys, res_phi_hits = compute_fieldlines(
            bs, R0, Z0, tmax=2, phis=phis, stopping_criteria=[])

        for i in range(len(res_phi_hits[0])):
            assert np.linalg.norm(ma.gamma()[i+1, :] - res_phi_hits[0][i, 2:5]) < 2e-3
