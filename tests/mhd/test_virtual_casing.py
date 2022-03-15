import unittest
import logging
import os
import numpy as np
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.virtual_casing import VirtualCasing
from . import TEST_DIR
try:
    import virtual_casing
except ImportError:
    virtual_casing = None

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


@unittest.skipIf(virtual_casing is None, "virtual_casing python package not installed")
class VirtualCasingTests(unittest.TestCase):
    def test_nphi_multiple_of_2_nfp(self):
        """
        nphi must be a multiple of 2 * nfp
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs.nc')
        vmec = Vmec(filename)

        with self.assertRaises(ValueError):
            vc = VirtualCasing.from_vmec(vmec, nphi=151, ntheta=20)
        with self.assertRaises(ValueError):
            vc = VirtualCasing.from_vmec(vmec, nphi=150, ntheta=20)
        vc = VirtualCasing.from_vmec(vmec, nphi=152, ntheta=20)

    def test_different_initializations(self):
        """
        Verify the virtual casing object can be initialized from a Vmec
        object, from a Vmec input file, or from a Vmec wout file.
        """
        filename = os.path.join(TEST_DIR, 'input.li383_low_res')
        vc = VirtualCasing.from_vmec(filename, nphi=72, ntheta=10)

        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs.nc')
        vc = VirtualCasing.from_vmec(filename, nphi=80, ntheta=10)

        vmec = Vmec(filename)
        vc = VirtualCasing.from_vmec(vmec, nphi=80, ntheta=10)

    def test_bnorm_benchmark(self):
        """
        Verify that the virtual_casing module by Malhotra et al gives
        results that match a reference calculation by the old fortran
        BNORM code.
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs.nc')
        bnorm_filename = os.path.join(TEST_DIR, 'bnorm.20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs')

        vmec = Vmec(filename)
        factor = 2
        vc = VirtualCasing.from_vmec(vmec, nphi=factor * 152, ntheta=factor * 20)

        nfp = vmec.wout.nfp
        theta, phi = np.meshgrid(2 * np.pi * vc.theta, 2 * np.pi * vc.phi)
        B_internal_normal_bnorm = np.zeros((vc.nphi, vc.ntheta))

        # Read BNORM output file:
        with open(bnorm_filename, 'r') as f:
            lines = f.readlines()

        for line in lines:
            splitline = line.split()
            if len(splitline) != 3:
                continue
            m = int(splitline[0])
            n = int(splitline[1])
            amplitude = float(splitline[2])
            B_internal_normal_bnorm += amplitude * np.sin(m * theta + n * nfp * phi)
            # To see that it should be (mu+nv) rather than (mu-nv) in the above line, you can examine
            # BNORM/Sources/bn_fouri.f (where the arrays in the bnorm files are computed)
            # or NESCOIL/Sources/bnfld.f (where bnorm files are read)

        # The BNORM code divides Bnormal by curpol. Undo this scaling now:
        curpol = (2 * np.pi / nfp) * (1.5 * vmec.wout.bsubvmnc[0, -1] - 0.5 * vmec.wout.bsubvmnc[0, -2])
        B_internal_normal_bnorm *= -curpol

        difference = B_internal_normal_bnorm - vc.B_internal_normal
        avg = 0.5 * (B_internal_normal_bnorm + vc.B_internal_normal)
        rms = np.sqrt(np.mean(avg ** 2))
        rel_difference = difference / rms
        logger.info(f'root mean squared of B_internal_normal: {rms}')
        logger.info('Diff between BNORM and virtual_casing: '
                    f'abs={np.max(np.abs(difference))}, rel={np.max(np.abs(rel_difference))}')
        np.testing.assert_allclose(B_internal_normal_bnorm, vc.B_internal_normal, atol=0.006)

        if 0:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 7))
            nrows = 2
            ncols = 2
            contours = np.linspace(-0.2, 0.2, 25)

            plt.subplot(nrows, ncols, 1)
            plt.contourf(phi, theta, B_internal_normal_bnorm, contours)
            plt.colorbar()
            plt.xlabel('phi')
            plt.ylabel('theta')
            plt.title('B_internal_normal from BNORM')

            plt.subplot(nrows, ncols, 2)
            plt.contourf(phi, theta, vc.B_internal_normal, contours)
            plt.colorbar()
            plt.xlabel('phi')
            plt.ylabel('theta')
            plt.title('B_internal_normal from virtual_casing')

            plt.subplot(nrows, ncols, 3)
            plt.contourf(phi, theta, B_internal_normal_bnorm - vc.B_internal_normal, 25)
            plt.colorbar()
            plt.xlabel('phi')
            plt.ylabel('theta')
            plt.title('Difference')

            plt.tight_layout()
            plt.show()

    def test_save_load(self):
        """
        Save a calculation, then load it into a different object. The
        fields of the objects should all match.
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs.nc')
        vc1 = VirtualCasing.from_vmec(filename, nphi=152, ntheta=20, filename='vcasing.nc')
        vc2 = VirtualCasing.load('vcasing.nc')
        variables = ['ntheta', 'nphi', 'theta', 'phi', 'gamma', 'unit_normal', 'B_total', 'B_internal', 'B_internal_normal']
        for variable in variables:
            variable1 = eval('vc1.' + variable)
            variable2 = eval('vc2.' + variable)
            logger.info(f'Variable {variable} in vc1 is {variable1} and in vc2 is {variable2}')
            np.testing.assert_allclose(variable1, variable2)

    def test_plot(self):
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs.nc')
        vc = VirtualCasing.from_vmec(filename, nphi=152, ntheta=20)
        vc.plot(show=False)
