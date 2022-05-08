import unittest
import logging
import os
import numpy as np
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.virtual_casing import VirtualCasing
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from . import TEST_DIR
try:
    import virtual_casing
except ImportError:
    virtual_casing = None

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import vmec
    vmec_found = True
except ImportError:
    vmec_found = False

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

variables = [
    'src_nphi', 'src_ntheta', 'src_phi', 'src_theta', 'trgt_nphi',
    'trgt_ntheta', 'trgt_phi', 'trgt_theta', 'gamma', 'unit_normal',
    'B_total', 'B_external', 'B_external_normal'
]


@unittest.skipIf(
    (virtual_casing is None) or
    (MPI is None) or (not vmec_found),
    "Need virtual_casing, mpi4py, and vmec python packages")
class VirtualCasingVmecTests(unittest.TestCase):

    def test_different_initializations(self):
        """
        Verify the virtual casing object can be initialized from a Vmec
        object, from a Vmec input file, or from a Vmec wout file.
        """
        filename = os.path.join(TEST_DIR, 'input.li383_low_res')
        vc = VirtualCasing.from_vmec(filename, src_nphi=8)

        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vc = VirtualCasing.from_vmec(filename, src_nphi=9, src_ntheta=10)

        vmec = Vmec(filename)
        vc = VirtualCasing.from_vmec(vmec, src_nphi=10)


class VirtualCasingTests(unittest.TestCase):

    def test_bnorm_benchmark(self):
        for use_stellsym in [True, False]:
            with self.subTest(use_stellsym=use_stellsym):
                self.subtest_bnorm_benchmark(use_stellsym)

    def subtest_bnorm_benchmark(self, use_stellsym):
        """
        Verify that the virtual_casing module by Malhotra et al gives
        results that match a reference calculation by the old fortran
        BNORM code.
        """

        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        bnorm_filename = os.path.join(TEST_DIR, 'bnorm.20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs')

        vmec = Vmec(filename)
        nphi_fac = 1 if use_stellsym else 2
        vc = VirtualCasing.from_vmec(vmec, src_nphi=25 * nphi_fac,
                                     trgt_nphi=32, trgt_ntheta=32, use_stellsym=use_stellsym)

        nfp = vmec.wout.nfp
        theta, phi = np.meshgrid(2 * np.pi * vc.trgt_theta, 2 * np.pi * vc.trgt_phi)
        B_external_normal_bnorm = np.zeros((vc.trgt_nphi, vc.trgt_ntheta))

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
            B_external_normal_bnorm += amplitude * np.sin(m * theta + n * nfp * phi)
            # To see that it should be (mu+nv) rather than (mu-nv) in the above line, you can examine
            # BNORM/Sources/bn_fouri.f (where the arrays in the bnorm files are computed)
            # or NESCOIL/Sources/bnfld.f (where bnorm files are read)

        # The BNORM code divides Bnormal by curpol. Undo this scaling now:
        curpol = (2 * np.pi / nfp) * (1.5 * vmec.wout.bsubvmnc[0, -1] - 0.5 * vmec.wout.bsubvmnc[0, -2])
        B_external_normal_bnorm *= curpol

        difference = B_external_normal_bnorm - vc.B_external_normal
        avg = 0.5 * (B_external_normal_bnorm + vc.B_external_normal)
        rms = np.sqrt(np.mean(avg ** 2))
        rel_difference = difference / rms
        logger.info(f'root mean squared of B_external_normal: {rms}')
        logger.info('Diff between BNORM and virtual_casing: '
                    f'abs={np.max(np.abs(difference))}, rel={np.max(np.abs(rel_difference))}')
        np.testing.assert_allclose(B_external_normal_bnorm, vc.B_external_normal, atol=0.0061)

        if 0:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 7))
            nrows = 2
            ncols = 2
            contours = np.linspace(-0.2, 0.2, 25)

            plt.subplot(nrows, ncols, 1)
            plt.contourf(phi, theta, B_external_normal_bnorm, contours)
            plt.colorbar()
            plt.xlabel('phi')
            plt.ylabel('theta')
            plt.title('B_external_normal from BNORM')

            plt.subplot(nrows, ncols, 2)
            plt.contourf(phi, theta, vc.B_external_normal, contours)
            plt.colorbar()
            plt.xlabel('phi')
            plt.ylabel('theta')
            plt.title('B_external_normal from virtual_casing')

            plt.subplot(nrows, ncols, 3)
            plt.contourf(phi, theta, B_external_normal_bnorm - vc.B_external_normal, 25)
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
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vc1 = VirtualCasing.from_vmec(filename, src_nphi=11, src_ntheta=12, trgt_nphi=13, trgt_ntheta=11, filename='vcasing.nc')
        vc2 = VirtualCasing.load('vcasing.nc')
        for variable in variables:
            variable1 = eval('vc1.' + variable)
            variable2 = eval('vc2.' + variable)
            logger.info(f'Variable {variable} in vc1 is {variable1} and in vc2 is {variable2}')
            np.testing.assert_allclose(variable1, variable2)

    def test_plot(self):
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vc = VirtualCasing.from_vmec(filename, src_nphi=8, src_ntheta=9)
        vc.plot(show=False)
        # Now cover the case in which an axis is provided:
        import matplotlib.pyplot as plt
        fig, ax0 = plt.subplots()
        ax1 = vc.plot(ax=ax0, show=False)
        assert ax1 is ax0

    def test_vacuum(self):
        """
        For a vacuum field, B_internal should be 0.
        """
        filename = os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc')
        vmec = Vmec(filename)
        vc = VirtualCasing.from_vmec(vmec, src_nphi=32)
        logger.info(f'ntheta: {vc.src_ntheta}  B_external.shape: {vc.B_external.shape}')
        logger.info(f'max(|B_external - B_total|): {np.max(np.abs(vc.B_external - vc.B_total))}')
        logger.info(f'max(|B_external_normal|): {np.max(np.abs(vc.B_external_normal))}')
        np.testing.assert_allclose(vc.B_external, vc.B_total, atol=0.02)
        np.testing.assert_allclose(vc.B_external_normal, 0, atol=0.001)

    def test_stellsym(self):
        """
        B_external_normal should obey stellarator symmetry.
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vmec = Vmec(filename)
        nfp = vmec.wout.nfp
        src_nphi = 48
        src_ntheta = 12
        vc = VirtualCasing.from_vmec(vmec, src_nphi=src_nphi, src_ntheta=src_ntheta, use_stellsym=False)
        Bn_flipped = -np.rot90(np.rot90(vc.B_external_normal))
        Bn_flipped = np.roll(np.roll(Bn_flipped, 1, axis=0), 1, axis=1)
        logger.info(f'max diff in B_external_normal: {np.max(np.abs(vc.B_external_normal - Bn_flipped))}')
        np.testing.assert_allclose(vc.B_external_normal, Bn_flipped, atol=1e-10)
        vc_ss = VirtualCasing.from_vmec(vmec, src_nphi=src_nphi//4, src_ntheta=src_ntheta, use_stellsym=True)
        # pick indices so that the quadrature point of `field period` grid
        # match those on the `half period` grid and then compare bfield values there
        idxs = list(range(1, vc.trgt_nphi//2, 2))
        np.testing.assert_allclose(vc.trgt_phi[idxs], vc_ss.trgt_phi)
        logger.info(f'max diff in B_external_normal: {np.max(np.abs(vc.B_external_normal[idxs, :] - vc_ss.B_external_normal))}')
        np.testing.assert_allclose(vc.B_external_normal[idxs, :], vc_ss.B_external_normal, atol=1e-5)
        """
        import matplotlib.pyplot as plt
        plt.subplot(1, 2, 1)
        plt.contourf(vc.B_external_normal)
        plt.subplot(1, 2, 2)
        plt.contourf(Bn_flipped)
        plt.tight_layout()
        plt.show()
        """
