import unittest
import numpy as np
from simsopt.field import CurrentPotentialSolve
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
                f = netcdf_file(filename, 'r', mmap=False)
                Bnormal_regcoil_total = f.variables['Bnormal_total'][()][ilambda, :, :]
                Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
                Bnormal_from_net_coil_currents = f.variables['Bnormal_from_net_coil_currents'][()]
                r_plasma = f.variables['r_plasma'][()]
                r_coil = f.variables['r_coil'][()]
                nzeta_plasma = f.variables['nzeta_plasma'][()]
                nzeta_coil = f.variables['nzeta_coil'][()]
                ntheta_coil = f.variables['ntheta_coil'][()]
                nfp = f.variables['nfp'][()]
                _stellsym = f.variables['symmetry_option'][()]
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
                _cp = cpst.current_potential

                # when lambda -> infinity, the L1 and L2 regularized problems should agree
                optimized_phi_mn_lasso, f_B_lasso, f_K_lasso, _, _ = cpst.solve_lasso(lam=lambda_regcoil)
                optimized_phi_mn, f_B, f_K = cpst.solve_tikhonov(lam=lambda_regcoil)
                np.testing.assert_allclose(single_valued_current_potential_mn, optimized_phi_mn, err_msg=f"{fname} ilambda={ilambda}: phi_mn (Tikhonov) mismatch")
                np.testing.assert_allclose(f_B_lasso, f_B, err_msg=f"{fname} ilambda={ilambda}: f_B (Lasso) != f_B (Tikhonov)")
                np.testing.assert_allclose(optimized_phi_mn_lasso, optimized_phi_mn, err_msg=f"{fname} ilambda={ilambda}: phi_mn (Lasso) != phi_mn (Tikhonov)")

            # Test that current potential solve class correctly writes REGCOIL outfiles
            cpst.write_regcoil_out(filename='simsopt_' + fname)
            g = netcdf_file('simsopt_' + fname, 'r', mmap=False)
            f = netcdf_file(filename, 'r', mmap=False)
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
                _xn_coil = f.variables['xn_coil'][()]
                xm_potential = f.variables['xm_potential'][()]
                xn_potential = f.variables['xn_potential'][()]
                theta_coil = f.variables['theta_coil'][()]
                zeta_coil = f.variables['zeta_coil'][()]
                f_B_regcoil = f.variables['chi2_B'][()][ilambda + 1]
                f_K_regcoil = f.variables['chi2_K'][()][ilambda + 1]
                norm_normal_plasma = f.variables['norm_normal_plasma'][()]
                current_potential_thetazeta = f.variables['single_valued_current_potential_thetazeta'][()][ilambda + 1, :, :]
                np.testing.assert_allclose(single_valued_current_potential_mn, g.variables['single_valued_current_potential_mn'][()][ilambda, :], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written phi_mn mismatch")
                np.testing.assert_allclose(Bnormal_regcoil_total, g.variables['Bnormal_total'][()][ilambda, :, :], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written Bnormal_total mismatch")
                np.testing.assert_allclose(Bnormal_from_plasma_current, g.variables['Bnormal_from_plasma_current'][()], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written Bnormal_from_plasma_current mismatch")
                np.testing.assert_allclose(Bnormal_from_net_coil_currents, g.variables['Bnormal_from_net_coil_currents'][()], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written Bnormal_from_net_coil_currents mismatch")
                np.testing.assert_allclose(nzeta_plasma, g.variables['nzeta_plasma'][()], atol=1e-12, err_msg=f"{fname}: written nzeta_plasma mismatch")
                np.testing.assert_allclose(nzeta_coil, g.variables['nzeta_coil'][()], atol=1e-12, err_msg=f"{fname}: written nzeta_coil mismatch")
                np.testing.assert_allclose(ntheta_coil, g.variables['ntheta_coil'][()], atol=1e-12, err_msg=f"{fname}: written ntheta_coil mismatch")
                np.testing.assert_allclose(nfp, g.variables['nfp'][()], atol=1e-12, err_msg=f"{fname}: written nfp mismatch")
                np.testing.assert_allclose(ntheta_plasma, g.variables['ntheta_plasma'][()], atol=1e-12, err_msg=f"{fname}: written ntheta_plasma mismatch")
                np.testing.assert_allclose(xm_plasma, g.variables['xm_plasma'][()], atol=1e-12, err_msg=f"{fname}: written xm_plasma mismatch")
                np.testing.assert_allclose(xn_plasma, g.variables['xn_plasma'][()], atol=1e-12, err_msg=f"{fname}: written xn_plasma mismatch")
                np.testing.assert_allclose(xm_coil, g.variables['xm_coil'][()], atol=1e-12, err_msg=f"{fname}: written xm_coil mismatch")
                np.testing.assert_allclose(xm_potential, g.variables['xm_potential'][()], atol=1e-12, err_msg=f"{fname}: written xm_potential mismatch")
                np.testing.assert_allclose(xn_potential, g.variables['xn_potential'][()], atol=1e-12, err_msg=f"{fname}: written xn_potential mismatch")
                np.testing.assert_allclose(theta_coil, g.variables['theta_coil'][()], atol=1e-12, err_msg=f"{fname}: written theta_coil mismatch")
                np.testing.assert_allclose(zeta_coil, g.variables['zeta_coil'][()], atol=1e-12, err_msg=f"{fname}: written zeta_coil mismatch")
                np.testing.assert_allclose(r_coil, g.variables['r_coil'][()], atol=1e-12, err_msg=f"{fname}: written r_coil mismatch")
                np.testing.assert_allclose(r_plasma, g.variables['r_plasma'][()], atol=1e-12, err_msg=f"{fname}: written r_plasma mismatch")
                np.testing.assert_allclose(K2_regcoil, g.variables['K2'][()][ilambda, :, :], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written K2 mismatch")
                assert (K2_regcoil.shape == g.variables['K2_l1'][()][ilambda, :, :].shape), f"{fname} ilambda={ilambda}: K2_l1 shape mismatch: {K2_regcoil.shape} vs {g.variables['K2_l1'][()][ilambda, :, :].shape}"
                np.testing.assert_allclose(lambda_regcoil, g.variables['lambda'][()][ilambda], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written lambda mismatch")
                np.testing.assert_allclose(lambda_regcoil, g.variables['lambda_l1'][()][ilambda], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written lambda_l1 mismatch")
                np.testing.assert_allclose(b_rhs_regcoil, g.variables['RHS_B'][()], atol=1e-12, err_msg=f"{fname}: written RHS_B mismatch")
                np.testing.assert_allclose(k_rhs_regcoil, g.variables['RHS_regularization'][()], atol=1e-12, err_msg=f"{fname}: written RHS_regularization mismatch")
                np.testing.assert_allclose(f_B_regcoil, g.variables['chi2_B'][()][ilambda], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written chi2_B mismatch")
                np.testing.assert_allclose(f_B_regcoil, g.variables['chi2_B_l1'][()][ilambda], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written chi2_B_l1 mismatch")
                np.testing.assert_allclose(f_K_regcoil, g.variables['chi2_K'][()][ilambda], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written chi2_K mismatch")
                np.testing.assert_allclose(norm_normal_plasma, g.variables['norm_normal_plasma'][()], atol=1e-12, err_msg=f"{fname}: written norm_normal_plasma mismatch")
                np.testing.assert_allclose(current_potential_thetazeta, g.variables['single_valued_current_potential_thetazeta'][()][ilambda, :, :], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written current_potential_thetazeta mismatch")
                np.testing.assert_allclose(current_potential_thetazeta, g.variables['single_valued_current_potential_thetazeta_l1'][()][ilambda, :, :], atol=1e-12, err_msg=f"{fname} ilambda={ilambda}: written current_potential_thetazeta_l1 mismatch")
            g.close()


if __name__ == "__main__":
    unittest.main()
