import unittest
import numpy as np
import os
import logging
from scipy.io import netcdf_file
try:
    import booz_xform
except ImportError as e:
    booz_xform = None 

try:
    import vmec
except ImportError as e:
    vmec = None 

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None

from simsopt._core.optimizable import Optimizable
if MPI is not None:
    from simsopt.mhd.boozer import Boozer, Quasisymmetry  # , booz_xform_found
    from simsopt.mhd.vmec import Vmec  # , vmec_found
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)


class MockBoozXform():
    """
    This class exists only for testing the Quasisymmetry class.  It
    returns similar data to the real Booz_xform class, but without doing a
    real calculation.
    """

    def __init__(self, mpol, ntor, nfp):
        mnmax = (ntor * 2 + 1) * mpol + ntor + 1
        xm = np.zeros(mnmax)
        xn = np.zeros(mnmax)
        xn[:ntor + 1] = np.arange(ntor + 1)
        for m in range(1, mpol + 1):
            index = ntor + 1 + (ntor * 2 + 1) * (m - 1)
            xm[index: index + (ntor * 2 + 1)] = m
            xn[index: index + (ntor * 2 + 1)] = np.arange(-ntor, ntor + 1)
        self.xm_b = xm
        self.xn_b = xn * nfp
        self.mnmax_b = mnmax
        self.nfp = nfp
        arr1 = np.arange(1.0, mnmax + 1) * 10
        arr2 = arr1 + 1
        arr2[0] = 100
        self.bmnc_b = np.stack((arr1, arr2)).transpose()
        print('bmnc_b:')
        print(self.bmnc_b)
        # print('booz_xform_found:', booz_xform_found)


class MockBoozer(Optimizable):
    """
    This class exists only for testing the Quasisymmetry class.  It
    returns similar data to the real Boozer class, but without doing a
    real calculation.
    """

    def __init__(self, mpol, ntor, nfp):
        self.bx = MockBoozXform(mpol, ntor, nfp)
        self.s_to_index = {0: 0, 1: 1}
        self.mpi = None
        super().__init__()

    def register(self, s):
        pass

    def run(self):
        pass


@unittest.skipIf(MPI is None, "mpi4py python package is not found")
class QuasisymmetryTests(unittest.TestCase):
    def test_quasisymmetry_residuals(self):
        """
        Verify that the correct residual vectors are returned for QA, QP, and QH
        """
        b = MockBoozer(3, 2, 4)
        # xm:    [  0  0  0  1  1  1  1  1  2   2   2   2   2   3   3   3   3   3]
        # xn:    [  0  4  8 -8 -4  0  4  8 -8  -4   0   4   8  -8  -4   0   4   8]
        # bmnc: [[ 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
        # bmnc:  [100 21 31 41 51 61 71 81 91 101 111 121 131 141 151 161 171 181]]

        # QA
        s = 0; q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18])
        s = 1; q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [.21, .31, .41, .51, .71, .81, .91, 1.01, 1.21, 1.31, 1.41, 1.51, 1.71, 1.81])
        s = (0, 1); q = Quasisymmetry(b, s, 1, 0, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18,
                                           .21, .31, .41, .51, .71, .81, .91, 1.01, 1.21, 1.31, 1.41, 1.51, 1.71, 1.81])

        # QP
        s = 0
        q = Quasisymmetry(b, s, 0, 1, "B00", "even")
        np.testing.assert_allclose(q.J(), [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # QH
        q = Quasisymmetry(b, s, 1, 1, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18])

        # QH, opposite "chirality"
        q = Quasisymmetry(b, s, 1, -1, "B00", "even")
        np.testing.assert_allclose(q.J(), [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18])

    @unittest.skipIf(booz_xform is None, "booz_xform python package not found")
    def test_boozer_register(self):
        b1 = Boozer(None)
        self.assertEqual(b1.s, set())
        # Try registering a single surface:
        qs11 = Quasisymmetry(b1, 0.5, 1, 1)
        self.assertEqual(b1.s, {0.5})
        # Register another surface:
        qs12 = Quasisymmetry(b1, 0.75, 1, 0)
        self.assertEqual(b1.s, {0.5, 0.75})
        # Register the same surface:
        qs13 = Quasisymmetry(b1, 0.75, 1, 0)
        self.assertEqual(b1.s, {0.5, 0.75})
        # Register two surfaces:
        qs14 = Quasisymmetry(b1, [0.1, 0.2], 1, 0)
        self.assertEqual(b1.s, {0.1, 0.2, 0.5, 0.75})
        # Register two surfaces, with a copy:
        qs15 = Quasisymmetry(b1, {0.2, 0.3}, 1, 0)
        self.assertEqual(b1.s, {0.1, 0.2, 0.3, 0.5, 0.75})

    @unittest.skipIf((booz_xform is None) or (vmec is None),
                     "vmec or booz_xform python package not found")
    def test_boozer_circular_tokamak(self):
        v = Vmec(os.path.join(TEST_DIR, "input.circular_tokamak"))
        b = Boozer(v, mpol=48, ntor=0)

        # Register a QA target at s = 0.5:
        qs1 = Quasisymmetry(b, 0.5, 1, 0)
        self.assertEqual(b._calls, 0)
        self.assertEqual(b.s, {0.5})
        residuals1 = qs1.J()
        self.assertEqual(b._calls, 1)
        # The residuals should be empty because axisymmetry implies QA
        self.assertEqual(len(residuals1), 0)
        # The VMEC config has 17 full-grid surfaces, so 16 half-grid
        # surfaces, so s=0.5 is close to the half-grid surface 7:
        np.testing.assert_allclose(b.bx.compute_surfs, [7])
        self.assertEqual(b.s_to_index, {0.5: 0})

        # Register a QP target at s = 1:
        qs2 = Quasisymmetry(b, 1.0, 0, 1)
        self.assertEqual(b._calls, 1)
        self.assertEqual(b.s, {0.5, 1.0})
        residuals2 = qs2.J()
        self.assertEqual(b._calls, 2)
        np.testing.assert_allclose(b.bx.compute_surfs, [7, 15])
        self.assertEqual(b.s_to_index, {0.5: 0, 1.0: 1})
        # Evaluating qs1 again should not cause booz_xform to run:
        residuals1 = qs1.J()
        self.assertEqual(b._calls, 2)        
        self.assertEqual(len(residuals1), 0)
        # All the modes except m=0 should contribute to the qs2 residuals:
        bmnc = b.bx.bmnc_b
        np.testing.assert_allclose(bmnc[1:, 1] / bmnc[0, 1], residuals2)

        # Register a QH target on the same pair of surfaces:
        qs3 = Quasisymmetry(b, [0.5, 1.0], 1, 1)
        residuals3 = qs3.J()
        np.testing.assert_allclose(b.bx.compute_surfs, [7, 15])
        self.assertEqual(b.s_to_index, {0.5: 0, 1.0: 1})
        # All modes except m=0 should contribute to the residuals:
        residuals3a = bmnc[1:, 0] / bmnc[0, 0]
        residuals3b = bmnc[1:, 1] / bmnc[0, 1]
        np.testing.assert_allclose(np.concatenate((residuals3a, residuals3b)),
                                   residuals3)

        # Register a target on a surface very close to a previously
        # registered surface:
        # Register a QP target at s = 1:
        s = 0.9999
        qs4 = Quasisymmetry(b, s, 0, 1)
        self.assertEqual(b.s, {0.5, s, 1.0})
        residuals4 = qs4.J()
        np.testing.assert_allclose(b.bx.compute_surfs, [7, 15])
        self.assertEqual(b.s_to_index, {0.5: 0, 1.0: 1, s: 1})

        # Compare to reference boozmn*.nc file
        f = netcdf_file(os.path.join(TEST_DIR, "boozmn_circular_tokamak.nc"),
                        mmap=False)
        bmnc_ref = f.variables["bmnc_b"][()].transpose()
        atol = 1e-12
        rtol = 1e-12
        np.testing.assert_allclose(bmnc[:, 0], bmnc_ref[:, 7],
                                   atol=atol, rtol=rtol)
        np.testing.assert_allclose(bmnc[:, 1], bmnc_ref[:, 15],
                                   atol=atol, rtol=rtol)

    @unittest.skipIf((booz_xform is None) or (vmec is None),
                     "vmec or booz_xform python package not found")
    def test_boozer_li383(self):
        v = Vmec(os.path.join(TEST_DIR, "wout_li383_low_res_reference.nc"))
        b = Boozer(v, mpol=32, ntor=16)
        qs1 = Quasisymmetry(b, [0.0, 1.0], 1, 0)
        residuals = qs1.J()
        np.testing.assert_allclose(b.bx.compute_surfs, [0, 14])
        self.assertEqual(b.s_to_index, {0.0: 0, 1.0: 1})
        bmnc = b.bx.bmnc_b

        # Compare to a reference boozmn*.nc file created by standalone
        # booz_xform:
        f = netcdf_file(os.path.join(TEST_DIR, "boozmn_li383_low_res.nc"),
                        mmap=False)
        bmnc_ref = f.variables["bmnc_b"][()].transpose()
        f.close()
        atol = 1e-12
        rtol = 1e-12
        np.testing.assert_allclose(bmnc[:, 0], bmnc_ref[:, 0],
                                   atol=atol, rtol=rtol)
        np.testing.assert_allclose(bmnc[:, 1], bmnc_ref[:, -1],
                                   atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
