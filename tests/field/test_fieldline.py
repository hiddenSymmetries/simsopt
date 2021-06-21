from simsopt.field.magneticfieldclasses import ToroidalField
from simsopt.field.tracing import compute_fieldlines, particles_to_vtk
import unittest
import numpy as np
import logging
logging.basicConfig()


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

    def test_poincare(self):
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
        particles_to_vtk(res_tys, '/tmp/fieldlines')
