from pathlib import Path
import unittest

import numpy as np

from simsopt.solve.current_voxels_optimization import prox_group_l0, compute_J
from simsopt.solve import relax_and_split_minres
from simsopt.util import *
from simsopt.geo import SurfaceRZFourier, CurrentVoxelsGrid
from simsopt.field import BiotSavart


class Testing(unittest.TestCase):

    def test_prox(self):
        m = np.random.rand(3000, 5)
        threshold = 1e5 
        m_thresholded = prox_group_l0(m, threshold, 3000, 5)
        m_thresholded = m_thresholded[~np.isclose(np.linalg.norm(m_thresholded, axis=1), 0.0), :]
        assert np.all(np.linalg.norm(m_thresholded, axis=1) >= 1e5)

    def test_compute_J(self):
        nphi = 8
        ntheta = nphi
        coff = 0.1
        poff = 0.05
        input_name = 'input.LandremanPaul2021_QA_lowres'

        # Read in the plasma equilibrium file
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

        # Make the inner and outer surfaces by extending the plasma surface
        s_inner.extend_via_normal(poff)
        s_outer.extend_via_normal(poff + coff)
        cv_opt = CurrentVoxelsGrid(s, s_inner, s_outer) 
        alphas = np.ones((cv_opt.N_grid, 5))
        compute_J(cv_opt, alphas, np.zeros((cv_opt.N_grid, 5)))
        assert cv_opt.J.shape == (cv_opt.N_grid, cv_opt.nx * cv_opt.ny * cv_opt.nz, 3)
        assert cv_opt.J_sparse.shape == (cv_opt.N_grid, cv_opt.nx * cv_opt.ny * cv_opt.nz, 3)
        assert np.allclose(cv_opt.J_sparse, 0.0)
        assert np.allclose(cv_opt.J, np.sum(cv_opt.Phi, axis=0))

    def test_algorithms(self):
        """ 
            Test the relax and split algorithm for solving
            the current voxel problem.
        """
        nphi = 8
        ntheta = 8
        coff = 0.1
        poff = 0.05
        input_name = 'input.LandremanPaul2021_QA_lowres'

        # Read in the plasma equilibrium file
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

        # Make the inner and outer surfaces by extending the plasma surface
        s_inner.extend_via_normal(poff)
        s_outer.extend_via_normal(poff + coff)

        # optimize the currents in the TF coils just so we have some external Bfields
        base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s)
        bs = BiotSavart(coils)
        bs = coil_optimization(s, bs, base_curves, curves)
        bs.set_points(s.gamma().reshape((-1, 3)))
        Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

        cv_opt = CurrentVoxelsGrid(
            s, s_inner, s_outer
        )

        # Test we get the trivial solution in sigma = 0 or kappa = infty limits
        kwargs = {"kappa": 1e20, "print_iter": 1}
        _ = relax_and_split_minres(cv_opt, **kwargs)
        assert np.allclose(cv_opt.alphas, 0.0)
        kwargs = {"sigma": 0.0, "print_iter": 1}
        _ = relax_and_split_minres(cv_opt, **kwargs)
        assert np.allclose(cv_opt.alphas, 0.0)

        kwargs_geo = {"Bn": Bnormal}
        cv_opt = CurrentVoxelsGrid(
            s, s_inner, s_outer, **kwargs_geo
        )

        # Set some hyperparameters for the optimization
        kwargs = {"kappa": 1e-3} 
        mdict1 = relax_and_split_minres(cv_opt)
        kwargs['precondition'] = True
        mdict2 = relax_and_split_minres(cv_opt)
        assert mdict1['f0'][-1] == mdict2['f0'][-1]
        assert mdict1['fB'][-1] == mdict2['fB'][-1]
        assert mdict1['fC'][-1] == mdict2['fC'][-1]
        assert mdict1['fI'][-1] == mdict2['fI'][-1]
        assert mdict1['fK'][-1] == mdict2['fK'][-1]
        assert mdict1['fminres'][-1] == mdict2['fminres'][-1]

        kwargs['nu'] = 1e2
        kwargs['alpha0'] = mdict2['alpha_opt']
        kwargs['max_iter'] = 100
        kwargs['max_iter_RS'] = 20
        l0_thresholds = np.linspace(1e4, 1e5)
        kwargs['l0_thresholds'] = l0_thresholds
        _ = relax_and_split_minres(cv_opt)
        w = cv_opt.w[~np.isclose(np.linalg.norm(cv_opt.w, axis=-1), 0.0), :]
        assert np.all(np.linalg.norm(w, axis=-1) > l0_thresholds[-1])


if __name__ == "__main__":
    unittest.main()
