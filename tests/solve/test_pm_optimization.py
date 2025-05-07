from pathlib import Path
import unittest

import numpy as np
from monty.tempfile import ScratchDir

import simsoptpp as sopp
from simsopt.solve.permanent_magnet_optimization import prox_l0, prox_l1
from simsopt.solve.permanent_magnet_optimization import setup_initial_condition
from simsopt.solve import relax_and_split, GPMO
from simsopt.util import *
from simsopt.geo import SurfaceRZFourier, PermanentMagnetGrid
from simsopt.field import BiotSavart


class Testing(unittest.TestCase):

    def test_prox(self):
        m = np.random.rand(3000)
        mmax = np.ones(1000)
        reg_l0 = 0.5
        nu = 0.5
        m_thresholded = prox_l0(m, mmax, reg_l0, nu)
        m_thresholded = m_thresholded[~np.isclose(m_thresholded, 0.0)]
        assert np.all(m_thresholded >= 0.5)
        nu = 1
        m_thresholded = prox_l1(m, mmax, reg_l0, nu)
        assert np.linalg.norm(m_thresholded) < np.linalg.norm(m)

    def test_MwPGP(self):
        """ 
            Test the MwPGP algorithm for solving the convex
            part of the permanent magnet problem. 
        """
        ndipoles = 100
        nquad = 512
        max_iter = 100
        m_maxima = np.random.rand(ndipoles) * 10
        m0 = np.zeros((ndipoles, 3))
        b = np.random.rand(nquad)
        A = np.random.rand(nquad, ndipoles, 3)
        ATA = np.tensordot(A, A, axes=([1, 1]))
        alpha = 2.0 / np.linalg.norm(ATA.reshape(nquad * 3, nquad * 3), ord=2)
        ATb = np.tensordot(A, b, axes=([0, 0]))
        with ScratchDir("."):
            MwPGP_hist, RS_hist, m_hist, dipoles = sopp.MwPGP_algorithm(
                A_obj=A, b_obj=b, ATb=ATb, m_proxy=m0, m0=m0, m_maxima=m_maxima,
                alpha=alpha, nu=1e100, epsilon=1e-4, max_iter=max_iter,  # verbose=True,
                reg_l0=0.0, reg_l1=0.0, reg_l2=0.0)
            m_hist = np.array(m_hist)
            assert dipoles.shape == (ndipoles, 3)
            assert m_hist.shape == (ndipoles, 3, 21)

    def test_algorithms(self):
        """ 
            Test the relax and split algorithm for solving
            the permanent magnet problem. Test the GPMO
            algorithm variants in the limit that they should
            all produce the same solution.
        """
        nphi = 8  # nphi = ntheta >= 64 needed for accurate full-resolution runs
        ntheta = 8
        dr = 0.04  # cylindrical bricks with radial extent 4 cm
        coff = 0.1  # PM grid starts offset ~ 10 cm from the plasma surface
        poff = 0.05  # PM grid end offset ~ 15 cm from the plasma surface
        input_name = 'input.LandremanPaul2021_QA_lowres'

        # Read in the plasma equilibrium file
        TEST_DIR = (Path(__file__).parent / ".." / ".." / "tests" / "test_files").resolve()
        surface_filename = TEST_DIR / input_name
        s = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_inner = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)
        s_outer = SurfaceRZFourier.from_vmec_input(surface_filename, range="half period", nphi=nphi, ntheta=ntheta)

        # Make the inner and outer surfaces by extending the plasma surface
        s_inner.extend_via_projected_normal(poff)
        s_outer.extend_via_projected_normal(poff + coff)

        # optimize the currents in the TF coils
        with ScratchDir("."):
            base_curves, curves, coils = initialize_coils('qa', TEST_DIR, s)
            bs = BiotSavart(coils)
            bs = coil_optimization(s, bs, base_curves, curves)
            bs.set_points(s.gamma().reshape((-1, 3)))
            Bnormal = np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)

            kwargs_geo = {"dr": dr}
            pm_opt = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
                s, Bnormal, s_inner, s_outer, **kwargs_geo
            )
            setup_initial_condition(pm_opt, np.zeros(pm_opt.ndipoles * 3))

            reg_l0 = 0.05  # Threshold off magnets with 5% or less strength
            nu = 1e10  # how strongly to make proxy variable w close to values in m

            # Rescale the hyperparameters and then add contributions to ATA and ATb
            reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

            # Set some hyperparameters for the optimization
            kwargs = initialize_default_kwargs()
            kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
            kwargs['max_iter'] = 40  # Number of iterations to take in a convex step
            kwargs['max_iter_RS'] = 20  # Number of total iterations of the relax-and-split algorithm
            kwargs['reg_l0'] = reg_l0
            relax_and_split(pm_opt, **kwargs)
            w = pm_opt.m_proxy[~np.isclose(pm_opt.m_proxy, 0.0)]
            assert np.all(np.abs(w) >= reg_l0 * pm_opt.m_maxima[0])

            # Try again with more aggressive thresholding
            reg_l0 = 0.5  # Threshold off magnets with 50% or less strength
            nu = 1e10  # how strongly to make proxy variable w close to values in m

            # Rescale the hyperparameters and then add contributions to ATA and ATb
            reg_l0, _, _, nu = pm_opt.rescale_for_opt(reg_l0, 0.0, 0.0, nu)

            # Set some hyperparameters for the optimization
            kwargs = initialize_default_kwargs()
            kwargs['nu'] = nu  # Strength of the "relaxation" part of relax-and-split
            kwargs['reg_l0'] = reg_l0
            relax_and_split(pm_opt, **kwargs)
            w = pm_opt.m_proxy[~np.isclose(pm_opt.m_proxy, 0.0)]
            assert np.all(np.abs(w) >= reg_l0 * pm_opt.m_maxima[0])
            kwargs['reg_l1'] = reg_l0
            with self.assertRaises(ValueError):
                relax_and_split(pm_opt, **kwargs)
            kwargs['reg_l0'] = 0.0
            kwargs['epsilon_RS'] = 1e5
            relax_and_split(pm_opt, **kwargs)

            # Test that all the GPMO variants return the same solutions
            # in various limits.
            kwargs = initialize_default_kwargs('GPMO')
            with self.assertRaises(ValueError):
                GPMO(pm_opt, algorithm='baseline', **kwargs)
            kwargs['nhistory'] = 10
            kwargs['K'] = 10
            errors1, Bn_errors1, m_history1 = GPMO(pm_opt, algorithm='baseline', **kwargs)
            m1 = pm_opt.m
            ndipoles = pm_opt.ndipoles
            pol_vector_x = np.zeros((ndipoles, 3))
            pol_vector_x[:, 0] = 1.0
            pol_vector_y = np.zeros((ndipoles, 3))
            pol_vector_y[:, 1] = 1.0
            pol_vector_z = np.zeros((ndipoles, 3))
            pol_vector_z[:, 2] = 1.0
            pol_vectors = np.transpose(np.array([pol_vector_x, pol_vector_y, pol_vector_z]), [1, 0, 2])
            pm_opt.pol_vectors = pol_vectors
            errors2, Bn_errors2, m_history2 = GPMO(pm_opt, algorithm='ArbVec', **kwargs)
            m2 = pm_opt.m
            assert np.allclose(m1, m2)
            assert np.allclose(errors1, errors2)
            assert np.allclose(Bn_errors1, Bn_errors2)
            assert np.allclose(m_history1, m_history2)
            kwargs['Nadjacent'] = 1
            kwargs['dipole_grid_xyz'] = pm_opt.dipole_grid_xyz
            errors3, Bn_errors3, m_history3 = GPMO(pm_opt, algorithm='multi', **kwargs)
            m3 = pm_opt.m
            assert np.allclose(m1, m3)
            assert np.allclose(errors1, errors3)
            assert np.allclose(Bn_errors1, Bn_errors3)
            assert np.allclose(m_history1, m_history3)
            kwargs['backtracking'] = 500
            kwargs['max_nMagnets'] = 1000
            errors4, Bn_errors4, m_history4 = GPMO(pm_opt, algorithm='backtracking', **kwargs)
            m4 = pm_opt.m
            assert np.allclose(m1, m4)
            assert np.allclose(errors1, errors4)
            assert np.allclose(Bn_errors1, Bn_errors4)
            assert np.allclose(m_history1, m_history4)

            # Note: ArbVec_backtracking history arrays contain one additional
            # entry at the beginning for the initialized solution

            errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            m5 = pm_opt.m
            assert np.allclose(m1, m5)
            assert np.allclose(errors1, errors5[1:])
            assert np.allclose(Bn_errors1, Bn_errors5[1:])
            assert np.allclose(m_history1, m_history5[:, :, 1:])
            with self.assertRaises(ValueError):
                pm_opt.coordinate_flag = 'cylindrical'
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            with self.assertRaises(NotImplementedError):
                errors5, Bn_errors5, m_history5 = GPMO(pm_opt, algorithm='random_name', **kwargs)

            kwargs['m_init'] = pm_opt.m.reshape([-1, 3])
            pm_opt.coordinate_flag = 'cartesian'
            errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)
            # Note: when K = n_history, m_history[:,:,-1] will be zeros
            assert np.allclose(m_history5[:, :, -2], m_history6[:, :, 0])
            with self.assertRaises(ValueError):
                kwargs['m_init'] = m_history6[:-1, :, -1]
                errors6, Bn_errors6, m_history6 = GPMO(pm_opt, algorithm='ArbVec_backtracking', **kwargs)


if __name__ == "__main__":
    unittest.main()
