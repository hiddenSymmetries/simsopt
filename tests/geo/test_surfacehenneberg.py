import unittest
import os
import logging
import json
from pathlib import Path

import numpy as np
from monty.json import MontyEncoder, MontyDecoder

from simsopt.geo.surfacehenneberg import SurfaceHenneberg

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import vmec
    vmec_found = True
except ImportError:
    vmec_found = False

if (MPI is not None) and vmec_found:
    from simsopt.mhd.vmec import Vmec
else:
    Vmec = None

TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


class SurfaceHennebergTests(unittest.TestCase):

    def test_names(self):
        """
        Check that the names of the dofs are set correctly.
        """
        surf = SurfaceHenneberg(nfp=1, alpha_fac=1, mmax=2, nmax=1)
        names_correct = ['R0nH(0)', 'R0nH(1)',
                         'Z0nH(1)',
                         'bn(0)', 'bn(1)',
                         'rhomn(0,1)',
                         'rhomn(1,-1)', 'rhomn(1,0)', 'rhomn(1,1)',
                         'rhomn(2,-1)', 'rhomn(2,0)', 'rhomn(2,1)']
        self.assertEqual(surf.local_dof_names, names_correct)

    def test_set_get_dofs(self):
        """
        If the dofs are set to random numbers, and then you get the dofs,
        you should get what you set.
        """
        for mmax in range(1, 4):
            for nmax in range(4):
                surf = SurfaceHenneberg(nfp=1, alpha_fac=1, mmax=mmax, nmax=nmax)
                self.assertEqual(len(surf.local_dof_names), len(surf.get_dofs()))

                # R0nH has nmax+1
                # Z0nH has nmax
                # bn has nmax + 1
                # rhomn for m=0 has nmax
                # rhomn for m>0 has 2 * nmax + 1
                nvals = nmax + 1 \
                    + nmax \
                    + nmax + 1 \
                    + nmax \
                    + mmax * (2 * nmax + 1)
                vals = np.random.rand(nvals) - 0.5
                surf.set_dofs(vals)
                np.testing.assert_allclose(surf.get_dofs(), vals)

    def test_set_get_rhomn(self):
        """
        Test get_rhomn() and set_rhomn(). If you set a particular mode and
        then get that mode, you should get what you set.
        """
        for mmax in range(1, 4):
            for nmax in range(4):
                surf = SurfaceHenneberg(nfp=2, alpha_fac=-1, mmax=mmax, nmax=nmax)
                for m in range(mmax + 1):
                    nmin = -nmax
                    if m == 0:
                        nmin = 1
                    for n in range(nmin, nmax + 1):
                        val = np.random.rand() - 0.5
                        surf.set_rhomn(m, n, val)
                        self.assertAlmostEqual(surf.get_rhomn(m, n), val)

    def test_fixed_range(self):
        """
        Test the fixed_range() function.
        """
        surf = SurfaceHenneberg(nfp=1, alpha_fac=1, mmax=2, nmax=1)
        # Order of elements:
        names_correct = ['R0nH(0)', 'R0nH(1)',
                         'Z0nH(1)',
                         'bn(0)', 'bn(1)',
                         'rhomn(0,1)',
                         'rhomn(1,-1)', 'rhomn(1,0)', 'rhomn(1,1)',
                         'rhomn(2,-1)', 'rhomn(2,0)', 'rhomn(2,1)']
        surf.fixed_range(20, 20, True)
        np.testing.assert_equal(surf.local_dofs_free_status, [False]*12)
        surf.fixed_range(20, 20, False)
        np.testing.assert_equal(surf.local_dofs_free_status, [True]*12)
        surf.fixed_range(0, 0, True)
        np.testing.assert_equal(surf.local_dofs_free_status,
                                [False, True, True, True, True, True,
                                 True, True, True, True, True, True])
        surf.fixed_range(1, 0, True)
        np.testing.assert_equal(surf.local_dofs_free_status,
                                [False, True, True, False, True, True,
                                 True, False, True, True, True, True])
        surf.fixed_range(2, 0, True)
        np.testing.assert_equal(surf.local_dofs_free_status,
                                [False, True, True, False, True, True,
                                 True, False, True, True, False, True])
        surf.local_fix_all()
        surf.fixed_range(0, 1, False)
        np.testing.assert_equal(surf.local_dofs_free_status,
                                [True, True, True, False, False, True,
                                 False, False, False, False, False, False])
        surf.fixed_range(1, 1, False)
        np.testing.assert_equal(surf.local_dofs_free_status,
                                [True, True, True, True, True, True,
                                 True, True, True, False, False, False])

    def test_indexing(self):
        """
        Check that set_rhomn() sets the expected element of the dofs vector.
        """
        mmax = 3
        nmax = 2
        surf = SurfaceHenneberg(nfp=2, alpha_fac=1, mmax=mmax, nmax=nmax)
        for n in range(nmax + 1):
            surf.R0nH[n] = 1000.0 + n
            surf.bn[n] = 2000.0 + n
        for n in range(1, nmax + 1):
            surf.Z0nH[n] = 3000.0 + n
        for m in range(mmax + 1):
            nmin = -nmax
            if m == 0:
                nmin = 1
            for n in range(nmin, nmax + 1):
                surf.set_rhomn(m, n, m * 100 + n)

        x = surf.get_dofs()
        np.testing.assert_allclose(
            x,
            [1.000e+03, 1.001e+03, 1.002e+03, 3.001e+03, 3.002e+03, 2.000e+03,
             2.001e+03, 2.002e+03, 1.000e+00, 2.000e+00, 9.800e+01, 9.900e+01,
             1.000e+02, 1.010e+02, 1.020e+02, 1.980e+02, 1.990e+02, 2.000e+02,
             2.010e+02, 2.020e+02, 2.980e+02, 2.990e+02, 3.000e+02, 3.010e+02,
             3.020e+02])
        self.assertListEqual(
            surf.local_dof_names,
            ['R0nH(0)', 'R0nH(1)', 'R0nH(2)', 'Z0nH(1)', 'Z0nH(2)', 'bn(0)',
             'bn(1)', 'bn(2)', 'rhomn(0,1)', 'rhomn(0,2)', 'rhomn(1,-2)',
             'rhomn(1,-1)', 'rhomn(1,0)', 'rhomn(1,1)', 'rhomn(1,2)',
             'rhomn(2,-2)', 'rhomn(2,-1)', 'rhomn(2,0)', 'rhomn(2,1)',
             'rhomn(2,2)', 'rhomn(3,-2)', 'rhomn(3,-1)', 'rhomn(3,0)',
             'rhomn(3,1)', 'rhomn(3,2)'])

    def test_axisymm(self):
        """
        Verify to_RZFourier() for an axisymmetric surface with circular
        cross-section.
        """
        R0 = 1.5
        a = 0.3
        for nfp in range(1, 3):
            for alpha_fac in [-1, 0, 1]:
                for mmax in range(1, 3):
                    for nmax in range(3):
                        surfH = SurfaceHenneberg(nfp=nfp, alpha_fac=alpha_fac,
                                                 mmax=mmax, nmax=nmax)
                        self.assertEqual(surfH.num_dofs(), len(surfH.get_dofs()))
                        surfH.R0nH[0] = R0
                        surfH.bn[0] = a
                        surfH.set_rhomn(1, 0, a)
                        surf = surfH.to_RZFourier()
                        for m in range(surf.mpol + 1):
                            for n in range(-surf.ntor, surf.ntor + 1):
                                if m == 0 and n == 0:
                                    self.assertAlmostEqual(surf.get_rc(m, n), R0)
                                    self.assertAlmostEqual(surf.get_zs(m, n), 0.0)
                                elif m == 1 and n == 0:
                                    self.assertAlmostEqual(surf.get_rc(m, n), a)
                                    self.assertAlmostEqual(surf.get_zs(m, n), a)
                                else:
                                    self.assertAlmostEqual(surf.get_rc(m, n), 0.0)
                                    self.assertAlmostEqual(surf.get_zs(m, n), 0.0)

    @unittest.skipIf(Vmec is None, "Valid Python interface to VMEC not found")
    def test_from_RZFourier(self):
        """
        Take a VMEC boundary in the original RZFourier representation,
        convert it to the Henneberg representation, and convert it
        back to the RZFourier representation. Coordinate-independent
        quantities like the volume and area should not change. If the
        result is converted to the Henneberg representation yet again,
        the degrees of freedom should be unchanged from the earlier
        Henneberg representation.
        """
        # Prepare some arbitrary points for testing gamma_lin:
        nlist = 15
        theta_list = np.linspace(-0.2, 0.8, nlist)
        phi_list = theta_list ** 2 - 0.5

        # Try a QI, QA, and a QH:
        files = ['input.W7-X_standard_configuration', 'input.cfqs_2b40', 'input.NuhrenbergZille_1988_QHS']
        alpha_facs = [1, 1, -1]

        for test_file, alpha_fac in zip(files, alpha_facs):
            vmec = Vmec(os.path.join(TEST_DIR, test_file))
            surf1 = vmec.boundary
            surf2 = SurfaceHenneberg.from_RZFourier(surf1, alpha_fac)
            # Make sure that the graph framework dofs are sync-ed with
            # the rc/zs arrays:
            np.testing.assert_allclose(surf2.x, surf2.get_dofs())
            self.assertEqual(surf2.num_dofs(), len(surf2.get_dofs()))
            surf3 = surf2.to_RZFourier()
            # Make sure that the graph framework dofs are sync-ed with
            # the rc/zs arrays:
            np.testing.assert_allclose(surf3.x, surf3.get_dofs())

            # Test gamma_lin:
            gamma2 = np.zeros((nlist, 3))
            gamma3 = np.zeros((nlist, 3))
            surf2.gamma_lin(gamma2, phi_list, theta_list)
            surf3.gamma_lin(gamma3, phi_list, theta_list)
            np.testing.assert_allclose(gamma2, gamma3, atol=1e-13, rtol=1e-13)

            # Test other surface functions:
            np.testing.assert_allclose(surf2.gamma(), surf3.gamma(), atol=1e-13, rtol=1e-13)
            np.testing.assert_allclose(surf2.gammadash2(), surf3.gammadash2(), atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(surf2.gammadash1(), surf3.gammadash1(), atol=1e-11, rtol=1e-11)
            np.testing.assert_allclose(surf1.volume(), surf2.volume(), atol=0, rtol=1e-3)
            np.testing.assert_allclose(surf1.volume(), surf3.volume(), atol=0, rtol=1e-3)
            np.testing.assert_allclose(surf1.area(), surf2.area(), atol=0, rtol=1e-3)
            np.testing.assert_allclose(surf1.area(), surf3.area(), atol=0, rtol=1e-3)
            surf4 = SurfaceHenneberg.from_RZFourier(surf3, alpha_fac, mmax=surf2.mmax, nmax=surf2.nmax)
            np.testing.assert_allclose(surf2.R0nH, surf4.R0nH, atol=1e-10, rtol=1e-4)
            np.testing.assert_allclose(surf2.Z0nH, surf4.Z0nH, atol=1e-12, rtol=1e-4)
            np.testing.assert_allclose(surf2.bn, surf4.bn, atol=1e-12, rtol=1e-4)
            np.testing.assert_allclose(surf2.rhomn, surf4.rhomn, atol=1e-8, rtol=1e-4)

    @unittest.skipIf(Vmec is None, "Valid Python interface to VMEC not found")
    def test_vmec(self):
        """
        If we run VMEC using a given boundary shape, then convert the
        boundary to the Henneberg representation and convert back and
        run VMEC again, the rotational transform should be almost
        identical.
        """
        vmec = Vmec(os.path.join(TEST_DIR, 'input.cfqs_2b40'))
        vmec.run()
        iota1 = vmec.wout.iotaf

        vmec.boundary = SurfaceHenneberg.from_RZFourier(vmec.boundary, 1)
        vmec.need_to_run_code = True
        vmec.run()
        iota2 = vmec.wout.iotaf

        logger.info(f'iota1: {iota1}')
        logger.info(f'iota2: {iota2}')
        logger.info(f'diff: {iota1 - iota2}')
        np.testing.assert_allclose(iota1, iota2, atol=1e-3, rtol=1e-3)
        # But if the 2 iota profiles are _exactly_ the same, vmec must
        # not have actually used the converted boundary.
        self.assertTrue(np.max(np.abs(iota1 - iota2)) > 1e-12)

    def test_serialization(self):
        R0 = 1.5
        a = 0.3
        for nfp in range(1, 3):
            for alpha_fac in [-1, 0, 1]:
                for mmax in range(1, 3):
                    for nmax in range(3):
                        surfH = SurfaceHenneberg(nfp=nfp, alpha_fac=alpha_fac,
                                                 mmax=mmax, nmax=nmax)
                        surfH.R0nH[0] = R0
                        surfH.bn[0] = a
                        surfH.set_rhomn(1, 0, a)
                        surfH.local_full_x = surfH.get_dofs()
                        surf_str = json.dumps(surfH, cls=MontyEncoder)
                        surfH_regen = json.loads(surf_str, cls=MontyDecoder)
                        self.assertAlmostEqual(surfH.area(), surfH_regen.area(),
                                               places=4)
                        self.assertAlmostEqual(surfH.volume(), surfH_regen.volume(),
                                               places=3)


if __name__ == "__main__":
    unittest.main()
