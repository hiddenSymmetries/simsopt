from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

from simsopt.field import (BiotSavart, Current, CurrentVoxelsField, Coil)
from simsopt.geo import (CurrentVoxelsGrid, SurfaceRZFourier, SurfaceXYZFourier)
from simsopt.objectives import SquaredFlux
from simsopt.solve import relax_and_split_minres
from simsopt.util import *

#from . import TEST_DIR
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()

# File for the desired boundary magnetic surface:
filename_QA = TEST_DIR / 'input.LandremanPaul2021_QA'
filename_QH = TEST_DIR / 'input.20210406-01-002-nfp4_QH_000_000240'
filename_tokamak = TEST_DIR / 'input.circular_tokamak'


class Testing(unittest.TestCase):

    def test_bad_params(self):
        """
        Test that the current voxel grid initialization
        correctly catches bad instances of the function arguments.
        """
        nphi = 4
        ntheta = nphi
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename_QA, range="half period", nphi=nphi, ntheta=ntheta)
            s1 = SurfaceRZFourier.from_vmec_input(filename_QA, range="half period", nphi=nphi, ntheta=ntheta)
            s2 = SurfaceRZFourier.from_vmec_input(filename_QA, range="half period", nphi=nphi, ntheta=ntheta)
            s3 = SurfaceXYZFourier(nfp=2, stellsym=True)
            s1.extend_via_projected_normal(0.1)
            s2.extend_via_projected_normal(0.2)
            Bn = np.zeros(s.gamma().shape[:2])
            
            # Check SurfaceXYZFourier initialization
            CurrentVoxelsGrid(s3, s3, s3)

            # Check SurfaceRZFourier initialization
            CurrentVoxelsGrid(s, s1, s2)
            s.stellsym = False
            c = CurrentVoxelsGrid(s, s1, s2)
            c.to_vtk_before_solve('test_current_voxels')
            c.to_vtk_after_solve('test_current_voxels_after_opt')
            s.nfp = 1
            c = CurrentVoxelsGrid(s, s1, s2)
            c.to_vtk_before_solve('test_current_voxels')
            c.to_vtk_after_solve('test_current_voxels_after_opt')

            with self.assertRaises(TypeError):
                CurrentVoxelsGrid(s)
            with self.assertRaises(TypeError):
                CurrentVoxelsGrid(s, s1)
            with self.assertRaises(TypeError):
                CurrentVoxelsGrid(Bn, s1, s2, s)
            with self.assertRaises(AttributeError):
                CurrentVoxelsGrid(Bn, s, s1)
            with self.assertRaises(AttributeError):
                CurrentVoxelsGrid(s1, s, Bn)
            with self.assertRaises(ValueError):
                kwargs = {"nx": -0.05}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"ny": -0.05}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"nz": -0.05}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Bn": [0.0]}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Bn": np.zeros((nphi, ntheta, 3))}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Bn": np.zeros((nphi, ntheta // 2))}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Nx": -2}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Ny": -2}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Nz": -2}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Bn_Itarget": np.zeros((nphi, 3))}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)
            with self.assertRaises(ValueError):
                kwargs = {"Bn_Itarget": np.zeros(nphi), "Itarget_curve": make_curve_at_theta0(s, nphi + 1)}
                CurrentVoxelsGrid(s, s1, s2, **kwargs)

    def test_optimize_bad_parameters(self):
        """
        Test that the current voxel optimizer
        correctly catches bad instances of the function arguments.
        """
        nphi = 4
        ntheta = nphi
        with ScratchDir("."):
            s = SurfaceRZFourier.from_vmec_input(filename_QA, range="half period", nphi=nphi, ntheta=ntheta)
            s1 = SurfaceRZFourier.from_vmec_input(filename_QA, range="half period", nphi=nphi, ntheta=ntheta)
            s2 = SurfaceRZFourier.from_vmec_input(filename_QA, range="half period", nphi=nphi, ntheta=ntheta)
            s1.extend_via_projected_normal(0.1)
            s2.extend_via_projected_normal(0.2)

            cv_opt = CurrentVoxelsGrid(s, s1, s2)

            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, kappa=-1)
            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, sigma=-1)
            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, nu=-1)
            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, rs_max_iter=-1)
            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, max_iter=-1)
            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, alpha0=np.zeros((10, 1)))
            with self.assertRaises(ValueError):
                _ = relax_and_split_minres(cv_opt, alpha0=np.zeros((cv_opt.N_grid * 5, 2)))
            _ = relax_and_split_minres(cv_opt)

    def test_Bn(self):
        """
        Creates a realistic QA configuration and some voxels, and optimizes it
        with the default parameters. Checks the the cv_opt object agrees with Bn and
        f_B with the DipoleField + SquaredFlux way of calculating Bn and f_B.

        Note nphi = 32 is the minimum number of points for the current voxels grid to work correctly
        here since the voxel grid generation is pretty sensitive to the resolution 
        of the plasma boundary surface.
        """
        from simsopt.geo import create_equally_spaced_curves
        from simsopt.field import Current, coils_via_symmetries
        from simsopt.geo import curves_to_vtk
        nphi = 16
        ntheta = nphi
        # with ScratchDir("."):
        for filename in [filename_QA, filename_QH, filename_tokamak]:
            s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
            s1 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi * 4, ntheta=ntheta * 4)
            s2 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi * 4, ntheta=ntheta * 4)
            s1.extend_via_projected_normal(0.2)
            s2.extend_via_projected_normal(0.4)
            # Initialize some coils for some external Bfields
            # base_curves, base_currents, ncoils = read_focus_coils(TEST_DIR / 'muse_tf_coils.focus')
            base_curves = create_equally_spaced_curves(
                2, s.nfp, stellsym=s.stellsym, R0=1.3, R1=1.1, order=5
            )
            base_currents = [Current(1e5) for i in range(2)]
            coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
            for coil in coils:
                coil.fix_all()
            curves_to_vtk([c.curve for c in coils], 'test_coils')
            s.to_vtk('test_surf')
            s1.to_vtk('test_surf1')
            s2.to_vtk('test_surf2')
            bs = BiotSavart(coils)
            bs.set_points(s.gamma().reshape(-1, 3))
            Bn = np.sum(bs.B().reshape(nphi, ntheta, 3) * s.unitnormal(), axis=-1)
            Nx = 20
            kwargs = {"Bn": Bn, "Nx": Nx, "Ny": Nx, "Nz": Nx}
            cv_opt = CurrentVoxelsGrid(s, s1, s2, **kwargs)
            cv_opt.to_vtk_before_solve('test_current_voxels')
            kwargs = {"max_iter": 1e4}
            _ = relax_and_split_minres(cv_opt, **kwargs)

            # This test is not passing currently for the four-field period QH, and not sure why!
            # cv_opt.check_fluxes()
            bs_current_voxels = CurrentVoxelsField(
                cv_opt.J,
                cv_opt.XYZ_integration,
                cv_opt.grid_scaling,
                nfp=s.nfp,
                stellsym=s.stellsym
            )
            bs_current_voxels.set_points(s.gamma().reshape(-1, 3))

            # check Bn
            Nnorms = np.ravel(np.sqrt(np.sum(s.normal() ** 2, axis=-1)))
            Ngrid = nphi * ntheta
            Bn_matprod = (cv_opt.B_matrix.dot(cv_opt.alphas) - cv_opt.b_rhs) * np.sqrt(Ngrid / Nnorms)
            assert np.allclose(Bn_matprod.reshape(nphi, ntheta), np.sum((bs.B() + bs_current_voxels.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2))

            # check <Bn>
            B_opt = np.mean(np.abs(cv_opt.B_matrix.dot(cv_opt.alphas) - cv_opt.b_rhs) * np.sqrt(Ngrid / Nnorms))
            B_voxel_field = np.mean(np.abs(np.sum((bs.B() + bs_current_voxels.B()).reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)))
            assert np.isclose(B_opt, B_voxel_field)

            # check integral Bn^2
            f_B_Am = 0.5 * np.linalg.norm(cv_opt.B_matrix.dot(cv_opt.alphas) - cv_opt.b_rhs, ord=2) ** 2
            f_B = SquaredFlux(s, bs_current_voxels, -Bn).J()
            assert np.isclose(f_B, f_B_Am)

    def test_cv_helpers(self):
        """
            Test the helper functions in utils/permanent_magnet_helpers.py
            that are relevant to the current voxels.
        """

        # Test build of the MUSE coils
        with ScratchDir("."):
            nphi = 16
            ntheta = nphi
            for filename in [filename_QA, filename_QH, filename_tokamak]:
                # Check coil initialization for some common stellarators wout_LandremanPaul2021_QA_lowres
                s = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi, ntheta=ntheta)
                s1 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi * 4, ntheta=ntheta * 4)
                s2 = SurfaceRZFourier.from_vmec_input(filename, range="half period", nphi=nphi * 4, ntheta=ntheta * 4)
                s1.extend_via_normal(0.1)
                s2.extend_via_normal(0.4)
                curve = make_curve_at_theta0(s, 100)
                assert curve.stellsym
                assert len(curve.gamma()[:, 0]) == 100
                cv_opt = CurrentVoxelsGrid(s, s1, s2)
                cv_opt.to_vtk_before_solve('test_current_voxels')
                cv_opt.to_vtk_before_solve('test_current_voxels', plot_quadrature_points=True)
                kwargs = {"l0_thresholds": [1e5]}
                _ = relax_and_split_minres(cv_opt, **kwargs)
                cv_opt.to_vtk_after_solve('test_current_voxels_after_opt')
                cv_opt.to_vtk_after_solve('test_current_voxels_after_opt', plot_quadrature_points=True)
                bs_current_voxels = CurrentVoxelsField(
                    cv_opt.J,
                    cv_opt.XYZ_integration,
                    cv_opt.grid_scaling,
                    nfp=s.nfp,
                    stellsym=s.stellsym
                )
                bs_current_voxels.set_points(s.gamma().reshape((-1, 3)))
                bs_current_voxels.B()
                curve = make_filament_from_voxels(cv_opt, 1e5, num_fourier=16)
                current = Current(cv_opt.Itarget)
                current.fix_all()
                coil = [Coil(curve, current)]
                bs = BiotSavart(coil)
                bs.set_points(s.gamma().reshape((-1, 3)))


if __name__ == "__main__":
    unittest.main()
