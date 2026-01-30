import unittest
import logging
import os

import numpy as np
from monty.tempfile import ScratchDir
try:
    import virtual_casing
except ImportError:
    virtual_casing = None
try:
    import matplotlib
except ImportError:
    matplotlib = None

try:
    from mpi4py import MPI
except:
    MPI = None

try:
    import vmec as vmec_mod
except ImportError:
    vmec_mod = None

from simsopt.mhd.vmec import Vmec
from simsopt.mhd.virtual_casing import VirtualCasing
from simsopt.field import VirtualCasingField
from simsopt.geo import SurfaceRZFourier
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

variables = [
    'src_nphi', 'src_ntheta', 'src_phi', 'src_theta', 'trgt_nphi',
    'trgt_ntheta', 'trgt_phi', 'trgt_theta', 'gamma', 'unit_normal',
    'B_total', 'B_external', 'B_external_normal'
]


def is_ci_environment():
    """Check if running in CI environment (skip plotting in CI)."""
    return os.environ.get('CI', 'false').lower() == 'true' or \
           os.environ.get('GITHUB_ACTIONS', 'false').lower() == 'true' or \
           os.environ.get('TRAVIS', 'false').lower() == 'true' or \
           os.environ.get('CIRCLECI', 'false').lower() == 'true'


@unittest.skipIf(
    (virtual_casing is None) or
    (MPI is None) or (vmec_mod is None),
    "Need virtual_casing, mpi4py, and vmec python packages")
class VirtualCasingVmecTests(unittest.TestCase):

    def test_different_initializations(self):
        """
        Verify the virtual casing object can be initialized from a Vmec
        object, from a Vmec input file, or from a Vmec wout file.
        """
        with ScratchDir("."):
            filename = os.path.join(TEST_DIR, 'input.li383_low_res')
            VirtualCasing.from_vmec(filename, src_nphi=8)

            filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
            VirtualCasing.from_vmec(filename, src_nphi=9, src_ntheta=10)

            vmec = Vmec(filename)
            VirtualCasing.from_vmec(vmec, src_nphi=10)

    @unittest.skipIf(matplotlib is None, "Need matplotlib for this test")
    def test_run_vmec_boundary_scan(self):
        """
        Test VirtualCasingField with boundary scan over different s_bound values.
        """
        import matplotlib.pyplot as plt

        with ScratchDir("."):
            s_bounds = np.linspace(0.3, 0.7, 3)
            digits = 3
            residuals = []
            for s_bound in s_bounds:
                print('s_bound: ', s_bound)
                # Run the full flux equilibrium with Vmec.
                vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m'))
                # vmec.indata.am_aux_s[1] = s_bound
                # vmec.indata.ac_aux_s[1] = s_bound
                vmec.run()

                # Define a half boundary surface at s=0.5
                # and run Vmec with scaled pressure and current.
                boundary_half = SurfaceRZFourier.from_wout(vmec.output_file, s=s_bound)
                vmec_half = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m_half'))
                vmec_half.boundary = boundary_half
                vmec_half.indata.phiedge = s_bound * vmec.indata.phiedge
                vmec_half.run()

                # Resolution on the plasma boundary surface:
                # nphi is the number of grid points in 1/2 a field period.
                nphi = 40
                ntheta = 40

                # Resolution for the virtual casing calculation:
                vc_src_nphi = 80
                vc_src_ntheta = 80

                # Setup VirtualCasingField from half-flux equilibrium
                vc = VirtualCasingField.from_vmec(vmec_half.output_file, src_ntheta=vc_src_ntheta,
                                                src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta,
                                                digits=digits, on_surface_tol=0.01)

                # Compute off-surface magnetic field on full flux surface
                surf = SurfaceRZFourier.from_wout(vmec.output_file, nphi=nphi, ntheta=ntheta, range='half period')

                vc.set_points(surf.gamma().reshape((-1, 3)))
                B = vc.B().reshape((nphi, ntheta, 3))
                Bn = np.sum(B * surf.unitnormal(), axis=2)

                # Compare with on-surface calculation of normal field from full flux surface
                vc = VirtualCasing.from_vmec(vmec.output_file, src_ntheta=vc_src_ntheta, src_nphi=vc_src_nphi,
                                            trgt_nphi=nphi, trgt_ntheta=ntheta)

                print('mpol: ', vmec.indata.mpol)
                print('residual: ', np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

                residuals.append(np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

            if not is_ci_environment():
                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasingField Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, -vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasing Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T + vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("Difference in Bn")

                plt.figure()
                plt.loglog(s_bounds, residuals, marker='o')
                plt.xlabel('s_bound')
                plt.ylabel('Residual Norm')
                plt.savefig('boundary_scan.png')
                # plt.show()
                plt.close('all')

            # Verify residuals are reasonable
            for residual in residuals:
                self.assertLess(residual, 1.0, "Residual should be less than 1")

            # Todo: add robust check that the residual is converging to zero with resolution
            # Right now this is not the case! 

    @unittest.skipIf(matplotlib is None, "Need matplotlib for this test")
    def test_run_vmec_grid_scan(self):
        """
        Test VirtualCasingField with grid resolution scan.
        """
        import matplotlib.pyplot as plt

        with ScratchDir("."):
            nphis = []
            residuals = []

            # Run the full flux equilibrium with Vmec.
            vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m'))
            vmec.run()

            # Define a half boundary surface at s=0.5
            # and run Vmec with scaled pressure and current.
            boundary_half = SurfaceRZFourier.from_wout(vmec.output_file, s=0.5)
            vmec_half = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m_half'))
            vmec_half.boundary = boundary_half
            vmec_half.indata.phiedge = 0.5 * vmec.indata.phiedge
            vmec_half.run()

            # Compute toroidal current profiles
            mu0 = 4 * np.pi * 1e-7
            It_full = vmec.wout.signgs * 2 * np.pi * vmec.wout.bsubumnc[0, 1::] / mu0
            It_half = vmec_half.wout.signgs * 2 * np.pi * vmec_half.wout.bsubumnc[0, 1::] / mu0
            flux_full = 0.5 * (vmec.wout.phi[0:-1] + vmec.wout.phi[1::])
            flux_half = 0.5 * (vmec_half.wout.phi[0:-1] + vmec_half.wout.phi[1::])

            if not is_ci_environment():
                plt.figure()
                plt.axhline(-1.7425E+05, linestyle='--', color='black', label='Target Current')
                plt.plot(flux_full, It_full, label='VMEC Full')
                plt.plot(flux_half, It_half, '--', label='VMEC Half')
                plt.legend()
                plt.xlabel('Toroidal Flux')
                plt.ylabel('Toroidal Current')

                plt.figure()
                plt.plot(vmec.wout.phi, vmec.wout.presf, label='VMEC Full')
                plt.plot(vmec_half.wout.phi, vmec_half.wout.presf, '--', label='VMEC Half')
                plt.legend()
                plt.xlabel('Toroidal Flux')
                plt.ylabel('Pressure')

                # plt.show()
                plt.close('all')

            for resolution in [2, 4, 6]:
                # Resolution on the plasma boundary surface:
                # nphi is the number of grid points in 1/2 a field period.
                nphi = resolution * 10
                ntheta = nphi
                nphis.append(nphi)

                # Resolution for the virtual casing calculation:
                vc_src_nphi = nphi
                vc_src_ntheta = nphi

                # Setup VirtualCasingField from half flux equilibrium
                vc = VirtualCasingField.from_vmec(vmec_half.output_file, 
                                                on_surface_tol=0.01,
                                                src_ntheta=vc_src_ntheta,
                                                src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

                # Compute off-surface magnetic field on full flux surface
                surf = SurfaceRZFourier.from_wout(vmec.output_file, nphi=nphi, ntheta=ntheta, range='half period')

                vc.set_points(surf.gamma().reshape((-1, 3)))
                B = vc.B().reshape((nphi, ntheta, 3))
                Bn = np.sum(B * surf.unitnormal(), axis=2)

                # Compare with on-surface calculation of normal field from full flux surface
                vc = VirtualCasing.from_vmec(vmec.output_file, src_ntheta=vc_src_ntheta, src_nphi=vc_src_nphi,
                                            trgt_nphi=nphi, trgt_ntheta=ntheta)

                print('residual: ', np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

                residuals.append(np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

            if not is_ci_environment():
                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasingField Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, -vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasing Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T + vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("Difference in Bn")

                plt.figure()
                plt.loglog(nphis, residuals, marker='o')
                plt.xlabel('nphi')
                plt.ylabel('Residual Norm')
                plt.savefig('grid_scan.png')
                # plt.show()
                plt.close('all')

            # Verify convergence: residual should decrease with resolution
            # for i in range(1, len(residuals)):
            #     self.assertLess(residuals[i], residuals[i - 1],
            #                     f"Residual should decrease: {residuals[i]} < {residuals[i - 1]}")

    @unittest.skipIf(matplotlib is None, "Need matplotlib for this test")
    def test_run_vmec_mpol_scan(self):
        """
        Test VirtualCasingField with mpol scan.
        """
        import matplotlib.pyplot as plt

        with ScratchDir("."):
            mpols = []
            residuals = []
            for resolution in [6, 7, 8, 9]:
                # Run the full flux equilibrium with Vmec.
                vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m'))
                vmec.indata.mpol = resolution
                vmec.indata.ntor = resolution
                vmec.run()

                mpols.append(resolution)

                # Define a half boundary surface at s=0.5
                # and run Vmec with scaled pressure and current.
                boundary_half = SurfaceRZFourier.from_wout(vmec.output_file, s=0.5)
                vmec_half = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m_half'))
                vmec_half.indata.mpol = resolution
                vmec_half.indata.ntor = resolution
                vmec_half.boundary = boundary_half
                vmec_half.indata.phiedge = 0.5 * vmec.indata.phiedge
                vmec_half.run()

                # Resolution on the plasma boundary surface:
                # nphi is the number of grid points in 1/2 a field period.
                nphi = resolution * 3
                ntheta = nphi

                # Resolution for the virtual casing calculation:
                vc_src_nphi = nphi
                vc_src_ntheta = nphi

                # Setup VirtualCasingField from half flux equilibrium
                vc = VirtualCasingField.from_vmec(vmec_half.output_file, src_ntheta=vc_src_ntheta,
                                                src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

                # Compute off-surface magnetic field on full flux surface
                surf = SurfaceRZFourier.from_wout(vmec.output_file, nphi=nphi, ntheta=ntheta, range='half period')

                vc.set_points(surf.gamma().reshape((-1, 3)))
                B = vc.B().reshape((nphi, ntheta, 3))
                Bn = np.sum(B * surf.unitnormal(), axis=2)

                # Compare with on-surface calculation of normal field from full flux surface
                vc = VirtualCasing.from_vmec(vmec.output_file, src_ntheta=vc_src_ntheta, src_nphi=vc_src_nphi,
                                            trgt_nphi=nphi, trgt_ntheta=ntheta)

                print('mpol: ', vmec.indata.mpol)
                print('residual: ', np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

                residuals.append(np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

            if not is_ci_environment():
                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasingField Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, -vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasing Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T + vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("Difference in Bn")

                plt.figure()
                plt.loglog(mpols, residuals, marker='o')
                plt.xlabel('mpol')
                plt.ylabel('Residual Norm')
                plt.savefig('mpol_scan.png')
                # plt.show()
                plt.close('all')

            # Verify residuals are reasonable
            for residual in residuals:
                self.assertLess(residual, 1.0, "Residual should be less than 1")

    @unittest.skipIf(matplotlib is None, "Need matplotlib for this test")
    def test_run_vmec_ns_scan(self):
        """
        Test VirtualCasingField with ns (radial resolution) scan.
        """
        import matplotlib.pyplot as plt

        prefactor = 25
        with ScratchDir("."):
            nss = []
            residuals = []
            for resolution in [1, 2, 3]:
                # Run the full flux equilibrium with Vmec.
                vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m'))
                vmec.indata.ns_array[3] = resolution * prefactor
                vmec.run()

                nss.append(prefactor * resolution)

                # Define a half boundary surface at s=0.5
                # and run Vmec with scaled pressure and current.
                boundary_half = SurfaceRZFourier.from_wout(vmec.output_file, s=0.5)
                vmec_half = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m_half'))
                vmec_half.indata.ns_array[3] = resolution * prefactor
                vmec_half.boundary = boundary_half
                vmec_half.indata.phiedge = 0.5 * vmec.indata.phiedge
                vmec_half.run()

                # Resolution on the plasma boundary surface:
                # nphi is the number of grid points in 1/2 a field period.
                nphi = 20
                ntheta = nphi

                # Resolution for the virtual casing calculation:
                vc_src_nphi = 80
                vc_src_ntheta = 80

                # Setup VirtualCasingField from half flux equilibrium
                vc = VirtualCasingField.from_vmec(vmec_half.output_file, src_ntheta=vc_src_ntheta,
                                                src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta)

                # Compute off-surface magnetic field on full flux surface
                surf = SurfaceRZFourier.from_wout(vmec.output_file, nphi=nphi, ntheta=ntheta, range='half period')

                vc.set_points(surf.gamma().reshape((-1, 3)))
                B = vc.B().reshape((nphi, ntheta, 3))
                Bn = np.sum(B * surf.unitnormal(), axis=2)

                # Compare with on-surface calculation of normal field from full flux surface
                vc = VirtualCasing.from_vmec(vmec.output_file, src_ntheta=vc_src_ntheta, src_nphi=vc_src_nphi,
                                            trgt_nphi=nphi, trgt_ntheta=ntheta)

                print('ns: ', vmec.indata.ns_array)
                print('residual: ', np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

                residuals.append(np.linalg.norm(Bn.T + vc.B_external_normal.T) / np.linalg.norm(Bn.T))

            if not is_ci_environment():
                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasingField Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, -vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("VirtualCasing Bn")

                plt.figure()
                plt.contourf(surf.quadpoints_phi, surf.quadpoints_theta, Bn.T + vc.B_external_normal.T, cmap='RdBu')
                plt.xlabel('phi')
                plt.ylabel('theta')
                plt.colorbar()
                plt.title("Difference in Bn")

                plt.figure()
                plt.loglog(nss, residuals, marker='o')
                plt.xlabel('ns')
                plt.ylabel('Residual Norm')
                plt.savefig('ns_scan.png')
                # plt.show()
                plt.close('all')

            # Verify residuals are reasonable
            for residual in residuals:
                self.assertLess(residual, 1.0, "Residual should be less than 1")


@unittest.skipIf(
    (virtual_casing is None),
    "Need virtual_casing python package installed to run VirtualCasingTests")
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
            plt.savefig('vacuum_test.png')
            plt.close('all')

    def test_save_load(self):
        """
        Save a calculation, then load it into a different object. The
        fields of the objects should all match.
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        with ScratchDir("."):
            vc1 = VirtualCasing.from_vmec(filename, src_nphi=11, src_ntheta=12, trgt_nphi=13, trgt_ntheta=11, filename='vcasing.nc')
            vc2 = VirtualCasing.load('vcasing.nc')
            for variable in variables:
                variable1 = eval('vc1.' + variable)
                variable2 = eval('vc2.' + variable)
                logger.info(f'Variable {variable} in vc1 is {variable1} and in vc2 is {variable2}')
                np.testing.assert_allclose(variable1, variable2)

    @unittest.skipIf(
        (matplotlib is None),
        "Need matplotlib python package to test VirtualCasing plot")
    def test_plot(self):
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        with ScratchDir("."):
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
        with ScratchDir("."):
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


@unittest.skipIf(
    (virtual_casing is None),
    "Need virtual_casing python package installed to run VirtualCasingFieldTests")
class VirtualCasingFieldTests(unittest.TestCase):
    """
    Tests for VirtualCasingField, verifying it produces consistent results
    with VirtualCasing and converges as resolution increases.
    
    Note: VirtualCasingField now precomputes on-surface B values during 
    initialization, making on-surface evaluation instant. Off-surface 
    evaluation uses compute_external_B_offsurf which is slower but handles
    arbitrary points. Evaluating very close to the surface can be slow due
    to singular integrals.
    """

    def test_VirtualCasingField_convergence_off_surface(self):
        """
        Test that as resolution increases, the B field computed
        from VirtualCasingField converges at off-surface points.
        We compare results at different resolutions and verify 
        that the difference decreases.
        
        Uses off-surface evaluation (0.2m offset) to test the 
        compute_external_B_offsurf pathway.
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vmec = Vmec(filename)
        
        # Fixed target resolution for comparison
        # Need nphi >= 2*ntor+1 = 17 for extend_via_normal with ntor=8
        trgt_nphi = 18
        trgt_ntheta = 18
        digits = 3
        
        # Create evaluation surface offset from plasma boundary
        eval_surf = SurfaceRZFourier.from_wout(filename, nphi=trgt_nphi, ntheta=trgt_ntheta, range="half period")
        eval_surf.extend_via_normal(0.2)  # 20cm offset to avoid singularity
        gamma = eval_surf.gamma()
        
        # Test convergence at different source resolutions
        resolutions = [16, 32, 64]
        B_results = []
        
        for src_nphi in resolutions:
            src_ntheta = src_nphi
            vc_field = VirtualCasingField.from_vmec(vmec, src_nphi=src_nphi, src_ntheta=src_ntheta,
                                                     trgt_nphi=trgt_nphi, trgt_ntheta=trgt_ntheta, digits=digits)
            vc_field.set_points(gamma.reshape((-1, 3)))
            B = vc_field.B().reshape((trgt_nphi, trgt_ntheta, 3))
            B_results.append(B)
        
        # Verify that differences decrease as resolution increases
        B_diff_low = np.max(np.abs(B_results[0] - B_results[1]))
        B_diff_high = np.max(np.abs(B_results[1] - B_results[2]))
        
        logger.info(f'B diff (res {resolutions[0]} vs {resolutions[1]}): {B_diff_low}')
        logger.info(f'B diff (res {resolutions[1]} vs {resolutions[2]}): {B_diff_high}')
        
        # Verify convergence: higher resolution should give smaller differences
        self.assertLess(B_diff_high, B_diff_low, 
                        "B field should converge as resolution increases")

    def test_VirtualCasingField_off_surface(self):
        """
        Test that VirtualCasingField can evaluate B at points off the surface.
        The field should be smooth and reasonable at nearby points.
        
        Compares on-surface evaluation (using precomputed values) with
        off-surface evaluation (using compute_external_B_offsurf).
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vmec = Vmec(filename)
        nfp = vmec.wout.nfp
        
        # Higher resolution now that on-surface is instant
        # Need nphi >= 2*ntor+1 = 17 for extend_via_normal with ntor=8
        src_nphi = 32
        trgt_nphi = 32  # Must be >= 17 for extend_via_normal
        trgt_ntheta = 32
        digits = 2
        
        # Create VirtualCasingField
        vc_field = VirtualCasingField.from_vmec(vmec, src_nphi=src_nphi, 
                                                 trgt_nphi=trgt_nphi, trgt_ntheta=trgt_ntheta, digits=digits)
        
        # Get on-surface points from the target grid (where precomputed values exist)
        # Note: vc_field.gamma is the SOURCE grid, but precomputed B values are on trgt_gamma
        gamma_on_surf = vc_field.trgt_gamma
        
        # Create off-surface evaluation points (0.2m offset)
        surf_extended = SurfaceRZFourier.from_wout(filename, nphi=src_nphi, ntheta=src_nphi, range="half period")
        surf_extended.extend_via_normal(0.2)  # 20cm outward
        gamma_off_surf = surf_extended.gamma()
        
        # Evaluate B on surface (uses precomputed values - instant)
        vc_field.set_points(gamma_on_surf.reshape((-1, 3)))
        B_on_surf = vc_field.B()
        modB_on_surf = np.linalg.norm(B_on_surf, axis=-1)
        
        # Evaluate B off surface (uses compute_external_B_offsurf)
        vc_field.set_points(gamma_off_surf.reshape((-1, 3)))
        B_off_surf = vc_field.B()
        print('B_off_surf: ', B_off_surf)
        modB_off_surf = np.linalg.norm(B_off_surf, axis=-1)
        
        # The field magnitude should be finite and positive at all points
        self.assertTrue(np.all(np.isfinite(B_on_surf)), "B should be finite on surface")
        self.assertTrue(np.all(np.isfinite(B_off_surf)), "B should be finite off surface")
        self.assertTrue(np.all(modB_on_surf > 0), "B magnitude should be positive on surface")
        self.assertTrue(np.all(modB_off_surf > 0), "B magnitude should be positive off surface")
        
        # The field should have changed (not be exactly the same)
        # but should be of similar magnitude
        avg_modB_on = np.mean(modB_on_surf)
        avg_modB_off = np.mean(modB_off_surf)
        logger.info(f'Average |B| on surface: {avg_modB_on}')
        logger.info(f'Average |B| off surface: {avg_modB_off}')
        
        # Field strength should be within a reasonable range
        # Note: The field can decay significantly off-surface, so we allow 
        # up to a factor of 100 change
        self.assertGreater(avg_modB_off, avg_modB_on / 100, 
                           "B magnitude off surface should not be too small")
        self.assertLess(avg_modB_off, avg_modB_on * 100,
                        "B magnitude off surface should not be too large")

    def test_VirtualCasingField_use_stellsym_false(self):
        """
        Test that use_stellsym=False works even when vmec.wout.lasym=False
        (i.e., when the equilibrium IS stellarator symmetric).
        
        With use_stellsym=False, the calculation uses a full field period
        instead of a half field period.
        """
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vmec = Vmec(filename)
        
        # Verify the equilibrium is stellarator symmetric
        self.assertFalse(vmec.wout.lasym, "Test requires stellarator symmetric equilibrium")
        
        src_nphi = 16
        trgt_nphi = 16
        trgt_ntheta = 16
        digits = 2
        
        # Create with use_stellsym=False (full field period)
        vc_field = VirtualCasingField.from_vmec(
            vmec, src_nphi=src_nphi, 
            trgt_nphi=trgt_nphi, trgt_ntheta=trgt_ntheta,
            use_stellsym=False, digits=digits
        )
        
        # Verify we can evaluate B at some points
        gamma = vc_field.gamma
        vc_field.set_points(gamma.reshape((-1, 3)))
        B = vc_field.B()
        
        # B should be finite and non-zero
        self.assertTrue(np.all(np.isfinite(B)), "B should be finite")
        self.assertTrue(np.all(np.linalg.norm(B, axis=-1) > 0), "B magnitude should be positive")
        
        logger.info(f"use_stellsym=False test passed, avg |B| = {np.mean(np.linalg.norm(B, axis=-1)):.4f}")

    def test_VirtualCasingField_different_npoints_near_surface(self):
        """
        Test evaluation at a different number of points than the target grid,
        testing both near-surface (nearest-neighbor lookup) and far-from-surface
        (compute_external_B_offsurf) cases.
        """
        import warnings
        
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vmec = Vmec(filename)
        
        # Use small grid for speed
        src_nphi = 80
        trgt_nphi = 32
        trgt_ntheta = 32
        digits = 1
        
        vc_field = VirtualCasingField.from_vmec(
            vmec, src_nphi=src_nphi,
            trgt_nphi=trgt_nphi, trgt_ntheta=trgt_ntheta, digits=digits
        )
        
        # Use trgt_gamma (not gamma) since get_close_mask compares against trgt_gamma
        gamma_full = vc_field.trgt_gamma  # Already shape (n_trgt, 3)
        
        # Test 1: Far points should NOT trigger nearest-neighbor warning
        gamma_far = gamma_full[:1].copy()  # Just 1 point for speed
        gamma_far[:, 0] *= 2.0  # Move far from surface
        gamma_far[:, 1] *= 2.0
        
        vc_field.set_points(gamma_far)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            B_far = vc_field.B()
            
            warning_messages = [str(warning.message) for warning in w]
            has_nn_warning = any("nearest-neighbor" in msg.lower() for msg in warning_messages)
            self.assertFalse(has_nn_warning,
                f"Far points should NOT trigger nearest-neighbor warning. Got: {warning_messages}")
        
        self.assertTrue(np.all(np.isfinite(B_far)), "B should be finite for far points")

        # Test 2: Near-surface points SHOULD trigger nearest-neighbor warning
        # Use just 1 point for speed
        gamma_near = gamma_full[:1].copy()
        gamma_near += 0.001 * np.random.randn(*gamma_near.shape)  # Small perturbation
        
        vc_field.set_points(gamma_near)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            B_near = vc_field.B()
            
            warning_messages = [str(warning.message) for warning in w]
            has_nn_warning = any("nearest-neighbor" in msg.lower() for msg in warning_messages)
            self.assertTrue(has_nn_warning,
                f"Near-surface points SHOULD trigger nearest-neighbor warning. Got: {warning_messages}")
        
        self.assertTrue(np.all(np.isfinite(B_near)), "B should be finite for near-surface points")
        
        logger.info("Different n_points warning test passed")

    def test_VirtualCasingField_compute_max_grid_spacing(self):
        """
        Test the _compute_max_grid_spacing static method.
        
        This method computes the maximum distance between any grid point
        and its nearest neighbor, which is used to set a safe minimum
        for on_surface_tol.
        """
        # Test 1: Regular 2D grid
        # Create a 3x3 grid with spacing 1.0 in x and y
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 2])
        xx, yy = np.meshgrid(x, y)
        grid_2d = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(9)])
        
        max_spacing = VirtualCasingField._compute_max_grid_spacing(grid_2d)
        # For a regular grid, max spacing should be 1.0 (horizontal/vertical neighbors)
        self.assertAlmostEqual(max_spacing, 1.0, places=5,
            msg=f"Expected max_spacing=1.0 for regular grid, got {max_spacing}")
        
        # Test 2: Non-uniform grid
        # Create a grid where one point is farther from its neighbors
        grid_nonuniform = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [5, 0, 0],  # This point is 3 units from nearest neighbor
        ])
        
        max_spacing = VirtualCasingField._compute_max_grid_spacing(grid_nonuniform)
        # Max spacing should be 3.0 (distance from [5,0,0] to [2,0,0])
        self.assertAlmostEqual(max_spacing, 3.0, places=5,
            msg=f"Expected max_spacing=3.0 for non-uniform grid, got {max_spacing}")
        
        # Test 3: 3D grid
        grid_3d = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        
        max_spacing = VirtualCasingField._compute_max_grid_spacing(grid_3d)
        # All points are 1.0 from origin, so max spacing is 1.0
        self.assertAlmostEqual(max_spacing, 1.0, places=5,
            msg=f"Expected max_spacing=1.0 for 3D grid, got {max_spacing}")
        
        logger.info("_compute_max_grid_spacing test passed")

    def test_VirtualCasingField_lasym_raises_error(self):
        """
        Test that VirtualCasingField raises RuntimeError when vmec.wout.lasym=True
        (i.e., when the equilibrium is NOT stellarator symmetric).
        """
        from unittest.mock import patch
        
        filename = os.path.join(TEST_DIR, 'wout_20220102-01-053-003_QH_nfp4_aspect6p5_beta0p05_iteratedWithSfincs_reference.nc')
        vmec = Vmec(filename)
        
        # Mock lasym to be True (non-stellarator symmetric)
        with patch.object(vmec.wout, 'lasym', True):
            with self.assertRaises(RuntimeError) as context:
                VirtualCasingField.from_vmec(vmec, src_nphi=16, digits=2)
            
            self.assertIn("stellarator symmetry", str(context.exception).lower())
        
        logger.info("lasym RuntimeError test passed")

    @unittest.skipIf(matplotlib is None, "Need matplotlib for this test")
    def test_VirtualCasingField_mpol_nphi_scan(self):
        """
        Test VirtualCasingField convergence over mpol and src_nphi.
        
        Creates a 2D scan over VMEC Fourier resolution (mpol) and virtual casing
        source resolution (src_nphi), producing a contour plot of errors.
        Uses fixed digits=3.
        
        Uses li383_half equilibrium for VirtualCasingField (off-surface B computation)
        and compares with VirtualCasing on the full li383 surface.
        """
        import matplotlib.pyplot as plt
        
        with ScratchDir("."):
            s_bound = 0.5
            digits = 3  # Fixed digits for all calculations
            
            # Resolution on the plasma boundary surface (target grid)
            nphi = 32
            ntheta = 32
            
            # Scan parameters
            mpol_values = [5, 7, 9]
            src_nphi_values = [10, 20, 40, 80]
            
            # 2D array to store residuals: [mpol_idx, nphi_idx]
            residuals_2d = np.zeros((len(mpol_values), len(src_nphi_values)))
            
            for i_mpol, mpol in enumerate(mpol_values):
                ntor = mpol  # Keep ntor = mpol for simplicity
                
                print(f'\n=== mpol={mpol}, ntor={ntor} ===')
                
                # Run the full flux equilibrium with Vmec
                vmec = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m'))
                vmec.indata.mpol = mpol
                vmec.indata.ntor = ntor
                vmec.indata.am_aux_s[1] = s_bound
                vmec.indata.ac_aux_s[1] = s_bound
                vmec.indata.ns_array[3] = 100
                vmec.run()
                
                # Define a half boundary surface at s=0.5
                # and run Vmec with scaled pressure and current
                boundary_half = SurfaceRZFourier.from_wout(vmec.output_file, s=s_bound)
                vmec_half = Vmec(os.path.join(TEST_DIR, 'input.li383_1.4m_half'))
                vmec_half.boundary = boundary_half
                vmec_half.indata.mpol = mpol
                vmec_half.indata.ntor = ntor
                vmec_half.indata.phiedge = s_bound * vmec.indata.phiedge
                # vmec_half.indata.ns_array[3] = 100
                vmec_half.run()
                
                # Create evaluation surface (full flux surface boundary)
                surf = SurfaceRZFourier.from_wout(vmec.output_file, nphi=nphi, ntheta=ntheta, range='half period')
                gamma = surf.gamma()
                unit_normal = surf.unitnormal()
                
                for i_nphi, src_nphi in enumerate(src_nphi_values):
                    src_ntheta = src_nphi  # Keep source grid square
                    
                    # Setup VirtualCasingField from half-flux equilibrium
                    vc_field = VirtualCasingField.from_vmec(
                        vmec_half.output_file, src_nphi=src_nphi, src_ntheta=src_ntheta,
                        trgt_nphi=nphi, trgt_ntheta=ntheta, digits=digits
                    )
                    
                    # Compute B field on full flux surface
                    vc_field.set_points(gamma.reshape((-1, 3)))
                    B = vc_field.B().reshape((nphi, ntheta, 3))
                    Bn = np.sum(B * unit_normal, axis=2)
                    
                    # Get reference Bn from VirtualCasing on full flux surface
                    vc_ref = VirtualCasing.from_vmec(
                        vmec.output_file, src_nphi=src_nphi, src_ntheta=src_ntheta,
                        trgt_nphi=nphi, trgt_ntheta=ntheta, digits=digits
                    )
                    
                    # Compute residual: Bn from VirtualCasingField should equal -B_external_normal from VirtualCasing
                    residual = np.linalg.norm(Bn.T + vc_ref.B_external_normal.T) / np.linalg.norm(Bn.T)
                    residuals_2d[i_mpol, i_nphi] = residual
                    
                    print(f'  src_nphi={src_nphi}: residual={residual:.6e}')
                    logger.info(f"mpol={mpol}, src_nphi={src_nphi}: residual = {residual:.6e}")
            
            if not is_ci_environment():
                # Create contour plot of errors vs mpol and src_nphi
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Use log scale for residuals
                log_residuals = np.log10(residuals_2d)
                
                # Create meshgrid for contour plot
                MPOL, NPHI = np.meshgrid(mpol_values, src_nphi_values, indexing='ij')
                
                # Contour plot
                levels = np.linspace(log_residuals.min(), log_residuals.max(), 20)
                cf = ax.contourf(MPOL, NPHI, log_residuals, levels=levels, cmap='viridis')
                cs = ax.contour(MPOL, NPHI, log_residuals, levels=levels[::2], colors='white', linewidths=0.5)
                ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
                
                cbar = plt.colorbar(cf, ax=ax)
                cbar.set_label('log10(Residual Norm)')
                
                ax.set_xlabel('mpol (VMEC Fourier resolution)')
                ax.set_ylabel('src_nphi (Virtual Casing resolution)')
                ax.set_title('VirtualCasingField Error: mpol vs src_nphi')
                
                # Mark scan points
                ax.scatter(MPOL.flatten(), NPHI.flatten(), c='red', s=20, zorder=5)
                
                plt.tight_layout()
                plt.savefig('mpol_nphi_scan.png', dpi=150)
                
                # Also create line plots for each mpol value
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                for i_mpol, mpol in enumerate(mpol_values):
                    ax2.loglog(src_nphi_values, residuals_2d[i_mpol, :], 
                              marker='o', label=f'mpol={mpol}')
                ax2.set_xlabel('src_nphi')
                ax2.set_ylabel('Residual Norm')
                ax2.set_title('VirtualCasingField Convergence vs Resolution')
                ax2.legend()
                ax2.grid(True)
                plt.tight_layout()
                plt.savefig('mpol_nphi_scan_lines.png', dpi=150)
                
                plt.show()
                plt.close('all')
            
            # Verify residuals are reasonable
            self.assertTrue(np.all(residuals_2d < 1.0), "All residuals should be less than 1")
        
        logger.info("mpol-nphi scan test passed")
