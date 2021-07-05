from simsopt.field.magneticfieldclasses import ToroidalField, InterpolatedField, UniformInterpolationRule
from simsopt.field.tracing import compute_fieldlines, particles_to_vtk, plot_poincare_data
from simsopt.field.biotsavart import BiotSavart
from simsopt.util.zoo import get_ncsx_data
from simsopt.geo.coilcollection import CoilCollection
import simsoptpp as sopp
import unittest
import numpy as np
import logging
logging.basicConfig()
try:
    import pyevtk
    with_evtk = True
except ImportError:
    with_evtk = False
with_boost = sopp.with_boost()


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

    @unittest.skipIf(not with_boost, "boost not found")
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

    @unittest.skipIf(not with_boost, "boost not found")
    def test_poincare_plot(self):
        coils, currents, ma = get_ncsx_data(Nt_coils=15)
        nfp = 3
        stellarator = CoilCollection(coils, currents, nfp, True)
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        n = 10
        rrange = (1.0, 1.9, n)
        phirange = (0, 2*np.pi, n*6)
        zrange = (-0.4, 0.4, n)
        bsh = InterpolatedField(
            bs, UniformInterpolationRule(2),
            rrange, phirange, zrange, True
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

    @unittest.skipIf(not with_boost, "boost not found")
    def test_poincare_ncsx_known(self):
        coils, currents, ma = get_ncsx_data(Nt_coils=25)
        nfp = 3
        stellarator = CoilCollection(coils, currents, nfp, True)
        currents = [-c for c in currents]
        bs = BiotSavart(stellarator.coils, stellarator.currents)
        R0 = [np.linalg.norm(ma.gamma()[0, :2])]
        Z0 = [ma.gamma()[0, 2]]
        phis = np.arctan2(ma.gamma()[:, 1], ma.gamma()[:, 0])
        res_tys, res_phi_hits = compute_fieldlines(
            bs, R0, Z0, tmax=10, phis=phis, stopping_criteria=[])
        for i in range(len(phis)-1):
            assert np.linalg.norm(ma.gamma()[i+1, :] - res_phi_hits[0][i, 2:5]) < 1e-4
