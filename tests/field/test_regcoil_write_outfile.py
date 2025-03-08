import unittest
from simsopt.geo import Surface, SurfaceRZFourier
from matplotlib import pyplot as plt
import numpy as np
from simsoptpp import WindingSurfaceBn_REGCOIL
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.objectives import SquaredFlux
from simsopt.geo import SurfaceRZFourier
from simsopt.field import BiotSavart, CurrentPotential, CurrentPotentialFourier, CurrentPotentialSolve
from scipy.special import ellipk, ellipe
from pathlib import Path
from scipy.io import netcdf_file
np.random.seed(100)

TEST_DIR = Path(__file__).parent / ".." / "test_files"


class Testing(unittest.TestCase):

    def test_regcoil_write(self):
        """
            This function tests the SIMSOPT routine that writes
            REGCOIL outfiles for backwards compatability.
        """
        for fname in ['regcoil_out.w7x_infty.nc', 'regcoil_out.li383_infty.nc']:  # , 'regcoil_out.near_axis_asym.nc', 'regcoil_out.near_axis.nc', 'regcoil_out.w7x.nc', 'regcoil_out.li383.nc']:
            filename = TEST_DIR / fname
            cpst = CurrentPotentialSolve.from_netcdf(filename)

            for ilambda in range(1, 3):
                # Load in big list of variables from REGCOIL to check agree with SIMSOPT
                f = netcdf_file(filename, 'r')
                Bnormal_regcoil_total = f.variables['Bnormal_total'][()][ilambda, :, :]
                Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
                Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
                r_plasma = f.variables['r_plasma'][()]
                r_coil = f.variables['r_coil'][()]
                nzeta_plasma = f.variables['nzeta_plasma'][()]
                nzeta_coil = f.variables['nzeta_coil'][()]
                ntheta_coil = f.variables['ntheta_coil'][()]
                nfp = f.variables['nfp'][()]
                stellsym = f.variables['symmetry_option'][()]
                ntheta_plasma = f.variables['ntheta_plasma'][()]
                K2_regcoil = f.variables['K2'][()][ilambda, :, :]
                lambda_regcoil = f.variables['lambda'][()][ilambda]
                b_rhs_regcoil = f.variables['RHS_B'][()]
                k_rhs_regcoil = f.variables['RHS_regularization'][()]
                single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][ilambda, :]
                xm_potential = f.variables['xm_potential'][()]
                xn_potential = f.variables['xn_potential'][()]
                theta_coil = f.variables['theta_coil'][()]
                zeta_coil = f.variables['zeta_coil'][()]
                f_B_regcoil = 0.5 * f.variables['chi2_B'][()][ilambda]
                f_K_regcoil = 0.5 * f.variables['chi2_K'][()][ilambda]
                norm_normal_plasma = f.variables['norm_normal_plasma'][()]
                current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()][ilambda, :, :]
                f.close()

                # Compare optimized dofs
                cp = cpst.current_potential

                # when lambda -> infinity, the L1 and L2 regularized problems should agree
                optimized_phi_mn_lasso, f_B_lasso, f_K_lasso, _, _ = cpst.solve_lasso(lam=lambda_regcoil)
                optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_regcoil)
                assert np.allclose(single_valued_current_potential_mn, optimized_phi_mn)
                assert np.isclose(f_B_lasso, f_B)
                assert np.allclose(optimized_phi_mn_lasso, optimized_phi_mn)

                # Test that current potential solve class correctly writes REGCOIL outfiles
            cpst.write_regcoil_out(filename='simsopt_' + fname)
            g = netcdf_file('simsopt_' + fname, 'r')
            f = netcdf_file(filename, 'r')
            for ilambda in range(2):
                print(filename, ilambda)
                Bnormal_regcoil_total = f.variables['Bnormal_total'][()][ilambda + 1, :, :]
                Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
                Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
                r_plasma = f.variables['r_plasma'][()]
                r_coil = f.variables['r_coil'][()]
                nzeta_plasma = f.variables['nzeta_plasma'][()]
                nzeta_coil = f.variables['nzeta_coil'][()]
                ntheta_coil = f.variables['ntheta_coil'][()]
                nfp = f.variables['nfp'][()]
                ntheta_plasma = f.variables['ntheta_plasma'][()]
                K2_regcoil = f.variables['K2'][()][ilambda + 1, :, :]
                b_rhs_regcoil = f.variables['RHS_B'][()]
                k_rhs_regcoil = f.variables['RHS_regularization'][()]
                lambda_regcoil = f.variables['lambda'][()][ilambda + 1]
                b_rhs_regcoil = f.variables['RHS_B'][()]
                k_rhs_regcoil = f.variables['RHS_regularization'][()]
                single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][ilambda + 1, :]
                xm_plasma = f.variables['xm_plasma'][()]
                xn_plasma = f.variables['xn_plasma'][()]
                xm_coil = f.variables['xm_coil'][()]
                xn_coil = f.variables['xn_coil'][()]
                xm_potential = f.variables['xm_potential'][()]
                xn_potential = f.variables['xn_potential'][()]
                theta_coil = f.variables['theta_coil'][()]
                zeta_coil = f.variables['zeta_coil'][()]
                f_B_regcoil = f.variables['chi2_B'][()][ilambda + 1]
                f_K_regcoil = f.variables['chi2_K'][()][ilambda + 1]
                norm_normal_plasma = f.variables['norm_normal_plasma'][()]
                current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()][ilambda + 1, :, :]
                assert np.allclose(single_valued_current_potential_mn, g.variables['single_valued_current_potential_mn'][()][ilambda, :])
                assert np.allclose(Bnormal_regcoil_total, g.variables['Bnormal_total'][()][ilambda, :, :])
                assert np.allclose(Bnormal_from_plasma_current, g.variables['Bnormal_from_plasma_current'][()])
                assert np.allclose(Bnormal_from_net_coil_currents, g.variables['Bnormal_from_net_coil_currents'][()])
                assert np.allclose(nzeta_plasma, g.variables['nzeta_plasma'][()])
                assert np.allclose(nzeta_coil, g.variables['nzeta_coil'][()])
                assert np.allclose(ntheta_coil, g.variables['ntheta_coil'][()])
                assert np.allclose(nfp, g.variables['nfp'][()])
                assert np.allclose(ntheta_plasma, g.variables['ntheta_plasma'][()])
                assert np.allclose(xm_plasma, g.variables['xm_plasma'][()])
                assert np.allclose(xn_plasma, g.variables['xn_plasma'][()])
                assert np.allclose(xm_coil, g.variables['xm_coil'][()])
                assert np.allclose(xm_potential, g.variables['xm_potential'][()])
                assert np.allclose(xn_potential, g.variables['xn_potential'][()])
                assert np.allclose(theta_coil, g.variables['theta_coil'][()])
                assert np.allclose(zeta_coil, g.variables['zeta_coil'][()])
                assert np.allclose(r_coil, g.variables['r_coil'][()])
                assert np.allclose(r_plasma, g.variables['r_plasma'][()])
                assert np.allclose(K2_regcoil, g.variables['K2'][()][ilambda, :, :])
                assert (K2_regcoil.shape == g.variables['K2_l1'][()][ilambda, :, :].shape)
                assert np.allclose(lambda_regcoil, g.variables['lambda'][()][ilambda])
                assert np.allclose(lambda_regcoil, g.variables['lambda_l1'][()][ilambda])
                assert np.allclose(b_rhs_regcoil, g.variables['RHS_B'][()])
                assert np.allclose(k_rhs_regcoil, g.variables['RHS_regularization'][()])
                assert np.allclose(f_B_regcoil, g.variables['chi2_B'][()][ilambda])
                assert np.allclose(f_B_regcoil, g.variables['chi2_B_l1'][()][ilambda])
                assert np.allclose(f_K_regcoil, g.variables['chi2_K'][()][ilambda])
                assert np.allclose(norm_normal_plasma, g.variables['norm_normal_plasma'][()])
                assert np.allclose(current_potential_thetazeta, g.variables['single_valued_current_potential_thetazeta'][()][ilambda, :, :])
                assert np.allclose(current_potential_thetazeta, g.variables['single_valued_current_potential_thetazeta_l1'][()][ilambda, :, :])
            g.close()


if __name__ == "__main__":
    unittest.main()
