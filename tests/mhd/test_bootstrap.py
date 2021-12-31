# Test ideas:
# * Reproduce fig 6 from Redl paper
# * Evaluate Redl's <j dot B> for Wistell-A and LDRD QA configs for which I already have self-consistent SFINCS calculations

import unittest
import logging
import os
import numpy as np
from simsopt.mhd.bootstrap import compute_trapped_fraction, compute_trapped_fraction_booz, \
    j_dot_B_Redl, vmec_j_dot_B_Redl, VmecRedlBootstrapMismatch
from simsopt.mhd.profiles import ProfilePolynomial
from simsopt.mhd.vmec import Vmec
from simsopt.mhd.boozer import Boozer
from simsopt.util.constants import ELEMENTARY_CHARGE
from . import TEST_DIR

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)


class BootstrapTests(unittest.TestCase):
    def test_compute_trapped_fraction(self):
        ns = 2
        ntheta = 15
        nphi = 7
        nfp = 3
        modB = np.zeros((ntheta, nphi, ns))
        sqrtg = np.zeros((ntheta, nphi, ns))

        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi, theta = np.meshgrid(phi1d, theta1d)

        sqrtg[:, :, 0] = 10.0
        sqrtg[:, :, 1] = -25.0

        modB[:, :, 0] = 13.0 + 2.6 * np.cos(theta)
        modB[:, :, 1] = 9.0 + 3.7 * np.sin(theta - nfp * phi)

        Bmin, Bmax, epsilon, fsa_b2, fsa_1overB, f_t = compute_trapped_fraction(modB, sqrtg)
        # The average of (b0 + b1 cos(theta))^2 is b0^2 + (1/2) * b1^2
        np.testing.assert_allclose(fsa_b2, [13.0 ** 2 + 0.5 * 2.6 ** 2, 9.0 ** 2 + 0.5 * 3.7 ** 2])
        np.testing.assert_allclose(fsa_1overB, [1 / np.sqrt(13.0 ** 2 - 2.6 ** 2), 1 / np.sqrt(9.0 ** 2 - 3.7 ** 2)])
        np.testing.assert_allclose(Bmin, [13.0 - 2.6, 9.0 - 3.7], rtol=1e-4)
        np.testing.assert_allclose(Bmax, [13.0 + 2.6, 9.0 + 3.7], rtol=1e-4)
        np.testing.assert_allclose(epsilon, [2.6 / 13.0, 3.7 / 9.0], rtol=1e-3)

    def test_trapped_fraction_Kim(self):
        """
        Compare the trapped fraction to eq (C18) in Kim, Diamond, &
        Groebner, Physics of Fluids B 3, 2050 (1991)
        """
        ns = 50
        ntheta = 100
        nphi = 3
        B0 = 7.5
        epsilon_in = np.linspace(0, 1, ns, endpoint=False)  # Avoid divide-by-0 when epsilon=1
        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        nfp = 3
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        #phi1d = np.array([0])
        phi, theta = np.meshgrid(phi1d, theta1d)
        modB = np.zeros((ntheta, nphi, ns))
        sqrtg = np.zeros((ntheta, nphi, ns))
        for js in range(ns):
            # Eq (A6)
            modB[:, :, js] = B0 / (1 + epsilon_in[js] * np.cos(theta))
            # For Jacobian, use eq (A7) for the theta dependence,
            # times an arbitrary overall scale factor
            sqrtg[:, :, js] = 6.7 * (1 + epsilon_in[js] * np.cos(theta))
        Bmin, Bmax, epsilon_out, fsa_B2, fsa_1overB, f_t = compute_trapped_fraction(modB, sqrtg)

        f_t_Kim = 1.46 * np.sqrt(epsilon_in) - 0.46 * epsilon_in  # Eq (C18) in Kim et al

        np.testing.assert_allclose(Bmin, B0 / (1 + epsilon_in))
        np.testing.assert_allclose(Bmax, B0 / (1 - epsilon_in))
        np.testing.assert_allclose(epsilon_in, epsilon_out)
        # Eq (A8):
        np.testing.assert_allclose(fsa_B2, B0 * B0 / np.sqrt(1 - epsilon_in ** 2), rtol=1e-6)
        np.testing.assert_allclose(f_t, f_t_Kim, rtol=0.1, atol=0.07)  # We do not expect precise agreement. 
        np.testing.assert_allclose(fsa_1overB, (2 + epsilon_in ** 2) / (2 * B0))

        if False:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            #plt.plot(epsilon_in, f_t, label='simsopt')
            #plt.plot(epsilon_in, f_t_Kim, label='Kim')
            plt.plot(np.sqrt(epsilon_in), f_t, label='simsopt')
            plt.plot(np.sqrt(epsilon_in), f_t_Kim, label='Kim')
            plt.xlabel('sqrt(epsilon)')
            plt.title('Trapped fraction $f_t$')
            plt.legend(loc=0)

            plt.subplot(1, 2, 2)
            #plt.plot(epsilon_in, (f_t_Kim - f_t) / (1 - f_t))
            plt.plot(epsilon_in, f_t_Kim - f_t)
            plt.title('Relative difference in $f_c$')
            plt.xlabel('epsilon')
            #plt.plot(epsilon_in, epsilon_out, 'r')
            #plt.plot(epsilon_in, epsilon_in, ':k')
            plt.show()

    def test_compute_trapped_fraction_booz_tokamak(self):
        """
        """
        filename = os.path.join(TEST_DIR, 'wout_ITERModel_reference.nc')
        vmec = Vmec(filename)
        booz = Boozer(vmec, mpol=32, ntor=0)
        booz.register([0.1, 1.0])
        Bmin_b, Bmax_b, epsilon_b, fsa_B2_b, fsa_1overB_b, f_t_b = compute_trapped_fraction_booz(booz, 0, 64)

        ne = ProfilePolynomial(5.0e20 * np.array([1, -0.9]))
        Te = ProfilePolynomial(8e3 * np.array([1, -0.9]))
        Ti = Te
        Zeff = 1
        helicity_N = 0
        jdotB, details = vmec_j_dot_B_Redl(vmec, booz.bx.s_b, ne, Te, Ti, Zeff, helicity_N)
        #Bmin_v, Bmax_v, epsilon_v, fsa_B2_v, fsa_1overB_v, f_t_v = compute_trapped_fraction(vmec, 0, 20)
        logging.info(f'booz.s: {booz.s}  booz.s_used: {booz.s_used}  booz.bx.s_b: {booz.bx.s_b}')
        logging.info(f'Bmin:  vmec={details.Bmin}  booz={Bmin_b}  diff={details.Bmin-Bmin_b}')
        logging.info(f'Bmax:  vmec={details.Bmax}  booz={Bmax_b}  diff={details.Bmax-Bmax_b}')
        logging.info(f'epsilon:  vmec={details.epsilon}  booz={epsilon_b}  diff={details.epsilon-epsilon_b}')
        logging.info(f'fsa_B2:  vmec={details.fsa_B2}  booz={fsa_B2_b}  diff={details.fsa_B2-fsa_B2_b}')
        logging.info(f'fsa_1overB:  vmec={details.fsa_1overB}  booz={fsa_1overB_b}  diff={details.fsa_1overB-fsa_1overB_b}')
        logging.info(f'f_t:  vmec={details.f_t}  booz={f_t_b}  diff={details.f_t-f_t_b}')
        np.testing.assert_allclose(details.Bmin, Bmin_b, rtol=1e-6)
        np.testing.assert_allclose(details.Bmax, Bmax_b, rtol=1e-6)
        np.testing.assert_allclose(details.epsilon, epsilon_b, rtol=1e-6)
        np.testing.assert_allclose(details.fsa_B2, fsa_B2_b, rtol=1e-6)
        np.testing.assert_allclose(details.fsa_1overB, fsa_1overB_b, rtol=1e-6)
        np.testing.assert_allclose(details.f_t, f_t_b, rtol=1e-6)

    def test_compute_trapped_fraction_booz_QH(self):
        """
        """
        filename = os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc')
        vmec = Vmec(filename)
        booz = Boozer(vmec, mpol=16, ntor=16)
        booz.register([0.1, 1.0])
        helicity_n = -1
        Bmin_b, Bmax_b, epsilon_b, fsa_B2_b, fsa_1overB_b, f_t_b = compute_trapped_fraction_booz(booz, helicity_n, 64)

        ne = ProfilePolynomial(5.0e20 * np.array([1, -0.9]))
        Te = ProfilePolynomial(8e3 * np.array([1, -0.9]))
        Ti = Te
        Zeff = 1
        helicity_N = helicity_n * vmec.wout.nfp
        jdotB, details = vmec_j_dot_B_Redl(vmec, booz.bx.s_b, ne, Te, Ti, Zeff, helicity_N)
        #Bmin_v, Bmax_v, epsilon_v, fsa_B2_v, fsa_1overB_v, f_t_v = compute_trapped_fraction(vmec, 0, 20)
        logging.info(f'booz.s: {booz.s}  booz.s_used: {booz.s_used}  booz.bx.s_b: {booz.bx.s_b}')
        logging.info(f'Bmin:  vmec={details.Bmin}  booz={Bmin_b}  diff={details.Bmin-Bmin_b}')
        logging.info(f'Bmax:  vmec={details.Bmax}  booz={Bmax_b}  diff={details.Bmax-Bmax_b}')
        logging.info(f'epsilon:  vmec={details.epsilon}  booz={epsilon_b}  diff={details.epsilon-epsilon_b}')
        logging.info(f'fsa_B2:  vmec={details.fsa_B2}  booz={fsa_B2_b}  diff={details.fsa_B2-fsa_B2_b}')
        logging.info(f'fsa_1overB:  vmec={details.fsa_1overB}  booz={fsa_1overB_b}  diff={details.fsa_1overB-fsa_1overB_b}')
        logging.info(f'f_t:  vmec={details.f_t}  booz={f_t_b}  diff={details.f_t-f_t_b}')
        np.testing.assert_allclose(details.Bmin, Bmin_b, rtol=1e-2)
        np.testing.assert_allclose(details.Bmax, Bmax_b, rtol=1e-2)
        np.testing.assert_allclose(details.epsilon, epsilon_b, rtol=1e-2)
        np.testing.assert_allclose(details.fsa_B2, fsa_B2_b, rtol=1e-2)
        np.testing.assert_allclose(details.fsa_1overB, fsa_1overB_b, rtol=1e-2)
        np.testing.assert_allclose(details.f_t, f_t_b, rtol=1e-2)

    def test_Redl_second_pass(self):
        """
        A second pass through coding up the equations from Redl et al,
        Phys Plasmas (2021) to make sure I didn't make transcription
        errors.
        """
        # Make up some arbitrary functions to use for input:
        ne = ProfilePolynomial(1.0e20 * np.array([1, 0, -0.8]))
        Te = ProfilePolynomial(25e3 * np.array([1, -0.9]))
        Ti = ProfilePolynomial(20e3 * np.array([1, -0.9]))
        Zeff = ProfilePolynomial(np.array([1.5, 0.5]))
        s = np.linspace(0, 1, 20)
        s = s[1:]  # Avoid divide-by-0 on axis
        helicity_N = 4
        G = 32.0 - s
        iota = 0.95 - 0.7 * s
        R = 6.0 - 0.1 * s
        epsilon = 0.3 * np.sqrt(s)
        f_t = 1.46 * np.sqrt(epsilon)
        psi_edge = 68 / (2 * np.pi)

        # Evaluate profiles on the s grid:
        ne_s = ne(s)
        Te_s = Te(s)
        Ti_s = Ti(s)
        Zeff_s = Zeff(s)
        ni_s = ne_s / Zeff_s
        d_ne_d_s = ne.dfds(s)
        d_Te_d_s = Te.dfds(s)
        d_Ti_d_s = Ti.dfds(s)

        # Sauter eq (18d)-(18e):
        ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_s) / Te_s)
        ln_Lambda_ii = 30.0 - np.log((Zeff_s ** 3) * np.sqrt(ni_s) / (Ti_s ** 1.5))

        # Sauter eq (18b)-(18c):
        nu_e = abs(R * (6.921e-18) * ne_s * Zeff_s * ln_Lambda_e \
                   / ((iota - helicity_N) * (Te_s ** 2) * (epsilon ** 1.5)))
        nu_i = abs(R * (4.90e-18) * ni_s * (Zeff_s ** 4) * ln_Lambda_ii \
                   / ((iota - helicity_N) * (Ti_s ** 2) * (epsilon ** 1.5)))

        # Redl eq (11):
        X31 = f_t / (1 + 0.67 * (1 - 0.7 * f_t) * np.sqrt(nu_e) / (0.56 + 0.44 * Zeff_s) \
                     + (0.52 + 0.086 * np.sqrt(nu_e)) * (1 + 0.87 * f_t) * nu_e / (1 + 1.13 * np.sqrt(Zeff_s - 1)))

        # Redl eq (10):
        Zfac = Zeff_s ** 1.2 - 0.71
        L31 = (1 + 0.15 / Zfac) * X31 - 0.22 / Zfac * (X31 ** 2) \
            + 0.01 / Zfac * (X31 ** 3) + 0.06 / Zfac * (X31 ** 4)

        # Redl eq (14):
        X32e = f_t / (1 + 0.23 * (1 - 0.96 * f_t) * np.sqrt(nu_e / Zeff_s) \
                      + 0.13 * (1 - 0.38 * f_t) * nu_e / (Zeff_s ** 2) \
                      * (np.sqrt(1 + 2 * np.sqrt(Zeff_s - 1))
                         + f_t * f_t * np.sqrt((0.075 + 0.25 * ((Zeff_s - 1) ** 2)) * nu_e)))

        # Redl eq (13):
        F32ee = (0.1 + 0.6 * Zeff_s) / (Zeff_s * (0.77 + 0.63 * (1 + (Zeff_s - 1) ** 1.1))) * (X32e - X32e ** 4) \
            + 0.7 / (1 + 0.2 * Zeff_s) * (X32e ** 2 - X32e ** 4 - 1.2 * (X32e ** 3 - X32e ** 4)) \
            + 1.3 / (1 + 0.5 * Zeff_s) * (X32e ** 4)

        # Redl eq (16)
        X32ei = f_t / (1 + 0.87 * (1 + 0.39 * f_t) * np.sqrt(nu_e) / (1 + 2.95 * ((Zeff_s - 1) ** 2)) \
                       + 1.53 * (1 - 0.37 * f_t) * nu_e * (2 + 0.375 * (Zeff_s - 1)))

        # Redl eq (15)
        F32ei = -(0.4 + 1.93 * Zeff_s) / (Zeff_s * (0.8 + 0.6 * Zeff_s)) * (X32ei - X32ei ** 4) \
            + 5.5 / (1.5 + 2 * Zeff_s) * (X32ei ** 2 - X32ei ** 4 - 0.8 * (X32ei ** 3 - X32ei ** 4)) \
            - 1.3 / (1 + 0.5 * Zeff_s) * (X32ei ** 4)

        # Redl eq (12)
        L32 = F32ei + F32ee

        # Redl eq (20):
        alpha0 = -(0.62 + 0.055 * (Zeff_s - 1)) * (1 - f_t) \
            / ((0.53 + 0.17 * (Zeff_s - 1)) * (1 - (0.31 - 0.065 * (Zeff_s - 1)) * f_t - 0.25 * (f_t ** 2)))

        # Redl eq (21):
        alpha = ((alpha0 + 0.7 * Zeff_s * np.sqrt(f_t * nu_i)) / (1 + 0.18 * np.sqrt(nu_i)) - 0.002 * (nu_i ** 2) * (f_t ** 6)) \
            / (1 + 0.004 * (nu_i ** 2) * (f_t ** 6))

        # Redl eq (19):
        L34 = L31

        Te_J = Te_s * ELEMENTARY_CHARGE
        Ti_J = Ti_s * ELEMENTARY_CHARGE
        d_Te_d_s_J = d_Te_d_s * ELEMENTARY_CHARGE
        d_Ti_d_s_J = d_Ti_d_s * ELEMENTARY_CHARGE
        pe = ne_s * Te_J
        pi = ni_s * Ti_J
        p = pe + pi
        Rpe = pe / p
        d_ni_d_s = d_ne_d_s / Zeff_s
        d_p_d_s = ne_s * d_Te_d_s_J + Te_J * d_ne_d_s + ni_s * d_Ti_d_s_J + Ti_J * d_ni_d_s
        # Equation from the bottom of the right column of Sauter errata:
        factors = -G * pe / (psi_edge * (iota - helicity_N))
        dnds_term = factors * L31 * p / pe * ((Te_J * d_ne_d_s + Ti_J * d_ni_d_s) / p)
        dTeds_term = factors * (L31 * p / pe * (ne_s * d_Te_d_s_J) / p + L32 * (d_Te_d_s_J / Te_J))
        dTids_term = factors * (L31 * p / pe * (ni_s * d_Ti_d_s_J) / p + L34 * alpha * (1 - Rpe) / Rpe * (d_Ti_d_s_J / Ti_J))
        jdotB_pass2 = -G * pe * (L31 * p / pe * (d_p_d_s / p) + L32 * (d_Te_d_s_J / Te_J) + L34 * alpha * (1 - Rpe) / Rpe * (d_Ti_d_s_J / Ti_J)) / (psi_edge * (iota - helicity_N))

        jdotB, details = j_dot_B_Redl(s, ne, Te, Ti, Zeff, G, R, iota, epsilon, f_t, psi_edge, helicity_N)

        atol = 1e-13
        rtol = 1e-13
        np.testing.assert_allclose(details.nu_e_star, nu_e, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.nu_i_star, nu_i, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.X31, X31, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.L31, L31, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.X32e, X32e, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.F32ee, F32ee, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.X32ei, X32ei, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.F32ei, F32ei, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.L32, L32, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.alpha, alpha, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.dTeds_term, dTeds_term, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.dTids_term, dTids_term, atol=atol, rtol=rtol)
        np.testing.assert_allclose(details.dnds_term, dnds_term, atol=atol, rtol=rtol)
        np.testing.assert_allclose(jdotB, jdotB_pass2, atol=atol, rtol=rtol)

    def test_Redl_figures_2_3(self):
        """
        Make sure the implementation here can roughly recover the plots
        from figures 2 and 3 in the Redl paper.
        """
        for Zeff in [1, 1.8]:
            target_nu_e_star = 4.0e-5
            target_nu_i_star = 1.0e-5
            # Make up some profiles
            ne = ProfilePolynomial([1.0e17])
            Te = ProfilePolynomial([1.0e5])
            Ti_over_Te = np.sqrt(4.9 * Zeff * Zeff * target_nu_e_star / (6.921 * target_nu_i_star))
            Ti = ProfilePolynomial([1.0e5 * Ti_over_Te])
            s = np.linspace(0, 1, 100) ** 4
            s = s[1:]  # Avoid divide-by-0 on axis
            helicity_N = 0
            epsilon = np.sqrt(s)
            f_t = 1.46 * np.sqrt(epsilon) - 0.46 * epsilon
            psi_edge = 68 / (2 * np.pi)
            G = 32.0 - s  # Doesn't matter
            R = 5.0 + 0.1 * s  # Doesn't matter
            # Redl uses fixed values of nu_e*. To match this, I'll use
            # a contrived iota profile that is chosen just to give the
            # desired nu_*.

            ne_s = ne(s)
            Te_s = Te(s)
            Zeff_s = Zeff

            # Sauter eq (18d):
            ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_s) / Te_s)

            # Sauter eq (18b), but without the iota factor:
            nu_e_without_iota = R * (6.921e-18) * ne_s * Zeff_s * ln_Lambda_e \
                / ((Te_s ** 2) * (epsilon ** 1.5))

            iota = nu_e_without_iota / target_nu_e_star
            # End of determining the qR profile that gives the desired nu*.

            jdotB, details = j_dot_B_Redl(s, ne, Te, Ti, Zeff, G, R, iota,
                                          epsilon, f_t, psi_edge, helicity_N)

            if False:
                # Make a plot, matching the axis ranges of Redl's
                # figures 2 and 3 as best as possible.
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6.5, 5.5))
                nrows = 2
                ncols = 2
                xlim = [-0.05, 1.05]

                plt.subplot(nrows, ncols, 1)
                plt.semilogy(f_t, details.nu_e_star, label=r'$\nu_{e*}$')
                plt.semilogy(f_t, details.nu_i_star, label=r'$\nu_{i*}$')
                plt.xlabel('f_t')
                plt.legend(loc=0, fontsize=8)

                plt.subplot(nrows, ncols, 2)
                plt.plot(f_t, details.L31, 'g')
                plt.title('L31')
                plt.xlabel('f_t')
                plt.xlim(xlim)
                plt.ylim(-0.05, 1.05)

                plt.subplot(nrows, ncols, 3)
                plt.plot(f_t, details.L32, 'g')
                plt.title('L32')
                plt.xlabel('f_t')
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-0.25, 0.025)
                else:
                    plt.ylim(-0.2, 0.05)

                plt.subplot(nrows, ncols, 4)
                plt.plot(f_t, details.alpha, 'g')
                plt.title('alpha')
                plt.xlabel('f_t')
                plt.xlim(xlim)
                plt.ylim(-1.25, 0.04)

                plt.tight_layout()
                plt.show()

            # Make sure L31, L32, and alpha are within the right range:
            np.testing.assert_array_less(details.L31, 1.05)
            np.testing.assert_array_less(0, details.L31)
            np.testing.assert_array_less(details.L32, 0.01)
            np.testing.assert_array_less(-0.25, details.L32)
            np.testing.assert_array_less(details.alpha, 0.05)
            np.testing.assert_array_less(-1.2, details.alpha)
            if Zeff > 1:
                np.testing.assert_array_less(-1.0, details.alpha)
            assert details.L31[0] < 0.1
            assert details.L31[-1] > 0.9
            assert details.L32[0] > -0.05
            assert details.L32[-1] > -0.05
            if Zeff == 0:
                assert np.min(details.L32) < -0.2
                assert details.alpha[0] < -1.05
            else:
                assert np.min(details.L32) < -0.13
                assert details.alpha[0] < -0.9
            assert details.alpha[-1] > -0.1 

    def test_Redl_figures_4_5(self):
        """
        Make sure the implementation here can roughly recover the plots
        from figures 4 and 5 in the Redl paper.
        """
        for Zeff in [1, 1.8]:
            n_nu_star = 30
            n_f_t = 3
            target_nu_stars = 10.0 ** np.linspace(-4, 4, n_nu_star)
            f_ts = np.array([0.24, 0.45, 0.63])
            L31s = np.zeros((n_nu_star, n_f_t))
            L32s = np.zeros((n_nu_star, n_f_t))
            alphas = np.zeros((n_nu_star, n_f_t))
            nu_e_stars = np.zeros((n_nu_star, n_f_t))
            nu_i_stars = np.zeros((n_nu_star, n_f_t))
            for j_nu_star, target_nu_star in enumerate(target_nu_stars):
                target_nu_e_star = target_nu_star
                target_nu_i_star = target_nu_star
                # Make up some profiles
                ne = ProfilePolynomial([1.0e17])
                Te = ProfilePolynomial([1.0e5])
                Ti_over_Te = np.sqrt(4.9 * Zeff * Zeff * target_nu_e_star / (6.921 * target_nu_i_star))
                Ti = ProfilePolynomial([1.0e5 * Ti_over_Te])
                s = np.ones(n_f_t)
                helicity_N = 0
                G = 32.0 - s  # Doesn't matter
                R = 5.0 + 0.1 * s  # Doesn't matter
                epsilon = s  # Doesn't matter
                f_t = f_ts
                psi_edge = 68 / (2 * np.pi)
                # Redl uses fixed values of nu_e*. To match this, I'll use
                # a contrived iota profile that is chosen just to give the
                # desired nu_*.

                ne_s = ne(s)
                Te_s = Te(s)
                Zeff_s = Zeff

                # Sauter eq (18d):
                ln_Lambda_e = 31.3 - np.log(np.sqrt(ne_s) / Te_s)

                # Sauter eq (18b), but without the q = 1/iota factor:
                nu_e_without_iota = R * (6.921e-18) * ne_s * Zeff_s * ln_Lambda_e \
                    / ((Te_s ** 2) * (epsilon ** 1.5))

                iota = nu_e_without_iota / target_nu_e_star
                # End of determining the qR profile that gives the desired nu*.

                jdotB, details = j_dot_B_Redl(s, ne, Te, Ti, Zeff, G, R, iota,
                                              epsilon, f_t, psi_edge, helicity_N)

                L31s[j_nu_star, :] = details.L31
                L32s[j_nu_star, :] = details.L32
                alphas[j_nu_star, :] = details.alpha
                nu_e_stars[j_nu_star, :] = details.nu_e_star
                nu_i_stars[j_nu_star, :] = details.nu_i_star
                #np.testing.assert_allclose(details.nu_e_star, target_nu_e_star)
                #np.testing.assert_allclose(details.nu_i_star, target_nu_i_star)

            if False:
                # Make a plot, matching the axis ranges of Redl's
                # figures 4 and 5 as best as possible.
                import matplotlib.pyplot as plt
                plt.figure(figsize=(6.5, 5.5))
                nrows = 2
                ncols = 2
                xlim = [3.0e-5, 1.5e4]

                plt.subplot(nrows, ncols, 2)
                for j in range(n_f_t):
                    plt.semilogx(nu_e_stars[:, j], L31s[:, j], label=f'f_t={f_ts[j]}')
                plt.legend(loc=0, fontsize=8)
                plt.title(f'L31, Zeff={Zeff}')
                plt.xlabel('nu_{*e}')
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-0.05, 0.85)
                else:
                    plt.ylim(-0.05, 0.75)

                plt.subplot(nrows, ncols, 3)
                for j in range(n_f_t):
                    plt.semilogx(nu_e_stars[:, j], L32s[:, j], label=f'f_t={f_ts[j]}')
                plt.legend(loc=0, fontsize=8)
                plt.title(f'L32, Zeff={Zeff}')
                plt.xlabel('nu_{*e}')
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-0.26, 0.21)
                else:
                    plt.ylim(-0.18, 0.2)

                plt.subplot(nrows, ncols, 4)
                for j in range(n_f_t):
                    plt.semilogx(nu_i_stars[:, j], alphas[:, j], label=f'f_t={f_ts[j]}')
                plt.legend(loc=0, fontsize=8)
                plt.title(f'alpha, Zeff={Zeff}')
                plt.xlabel('nu_{*i}')
                plt.xlim(xlim)
                if Zeff == 1:
                    plt.ylim(-1.1, 2.2)
                else:
                    plt.ylim(-1.1, 2.35)

                plt.tight_layout()
                plt.show()

            # Make sure L31, L32, and alpha are within the right range:
            if Zeff == 1:
                np.testing.assert_array_less(L31s, 0.71)
                np.testing.assert_array_less(0, L31s)
                np.testing.assert_array_less(L31s[-1, :], 1.0e-5)
                np.testing.assert_array_less(L32s, 0.2)
                np.testing.assert_array_less(-0.23, L32s)
                np.testing.assert_array_less(L32s[-1, :], 3.0e-5)
                np.testing.assert_array_less(-3.0e-5, L32s[-1, :])
                np.testing.assert_array_less(L32s[0, :], -0.17)
                np.testing.assert_array_less(alphas, 1.2)
                np.testing.assert_array_less(alphas[0, :], -0.58)
                np.testing.assert_array_less(-1.05, alphas)
                np.testing.assert_array_less(0.8, np.max(alphas, axis=0))
                np.testing.assert_array_less(L31s[:, 0], 0.33)
                assert L31s[0, 0] > 0.3
                np.testing.assert_array_less(L31s[0, 1], 0.55)
                assert L31s[0, 1] > 0.51
                np.testing.assert_array_less(L31s[0, 2], 0.7)
                assert L31s[0, 2] > 0.68
            else:
                np.testing.assert_array_less(L31s, 0.66)
                np.testing.assert_array_less(0, L31s)
                np.testing.assert_array_less(L31s[-1, :], 1.5e-5)
                np.testing.assert_array_less(L32s, 0.19)
                np.testing.assert_array_less(-0.15, L32s)
                np.testing.assert_array_less(L32s[-1, :], 5.0e-5)
                np.testing.assert_array_less(0, L32s[-1, :])
                np.testing.assert_array_less(L32s[0, :], -0.11)
                np.testing.assert_array_less(alphas, 2.3)
                np.testing.assert_array_less(alphas[0, :], -0.4)
                np.testing.assert_array_less(-0.9, alphas)
                np.testing.assert_array_less(1.8, np.max(alphas, axis=0))
                np.testing.assert_array_less(L31s[:, 0], 0.27)
                assert L31s[0, 0] > 0.24
                np.testing.assert_array_less(L31s[0, 1], 0.49)
                assert L31s[0, 1] > 0.45
                np.testing.assert_array_less(L31s[0, 2], 0.66)
                assert L31s[0, 2] > 0.63

    def test_Redl_sfincs_tokamak_benchmark(self):
        """
        Compare the Redl <j dot B> to a SFINCS calculation for a tokamak.

        The SFINCS calculation is on Matt Landreman's laptop in
        /Users/mattland/Box Sync/work21/20211225-01-sfincs_tokamak_bootstrap_for_Redl_benchmark
        """
        ne = ProfilePolynomial(5.0e20 * np.array([1, 0, 0, 0, -1]))
        Te = ProfilePolynomial(8e3 * np.array([1, -1]))
        Ti = Te
        Zeff = 1
        surfaces = np.linspace(0.01, 0.99, 99)
        helicity_N = 0
        filename = os.path.join(TEST_DIR, 'wout_ITERModel_reference.nc')
        vmec = Vmec(filename)
        jdotB, details = vmec_j_dot_B_Redl(vmec, surfaces, ne, Te, Ti, Zeff, helicity_N, plot=False)

        jdotB_sfincs = np.array([-577720.30718026, -737097.14851563, -841877.1731213, -924690.37927967,
                                 -996421.14965534, -1060853.54247997, -1120000.15051496, -1175469.30096585,
                                 -1228274.42232883, -1279134.94084881, -1328502.74017954, -1376746.08281939,
                                 -1424225.7135264, -1471245.54499716, -1518022.59582135, -1564716.93168823,
                                 -1611473.13548435, -1658436.14166984, -1705743.3966606, -1753516.75354018,
                                 -1801854.51072685, -1850839.64964612, -1900546.86009713, -1951047.40424607,
                                 -2002407.94774638, -2054678.30773555, -2107880.19135161, -2162057.48184046,
                                 -2217275.94462326, -2273566.0131982, -2330938.65226651, -2389399.44803491,
                                 -2448949.45267694, -2509583.82212581, -2571290.69542303, -2634050.8642164,
                                 -2697839.22372799, -2762799.43321187, -2828566.29269343, -2895246.32116721,
                                 -2962784.4499046, -3031117.70888815, -3100173.19345621, -3169866.34773162,
                                 -3240095.93569359, -3310761.89170199, -3381738.85511963, -3452893.53199984,
                                 -3524079.68661978, -3595137.36934266, -3665892.0942594, -3736154.01439094,
                                 -3805717.10211429, -3874383.57672975, -3941857.83476556, -4007909.78112076,
                                 -4072258.58610167, -4134609.67635966, -4194641.53357309, -4252031.55214378,
                                 -4306378.23970987, -4357339.70206557, -4404503.4788238, -4447364.62100875,
                                 -4485559.83633318, -4518524.39965094, -4545649.39588513, -4566517.96382113,
                                 -4580487.82371991, -4586917.13595789, -4585334.66419017, -4574935.10788554,
                                 -4555027.85929442, -4524904.81212564, -4483819.87563906, -4429820.99499252,
                                 -4364460.04545626, -4285813.80804979, -4193096.44129549, -4085521.92933703,
                                 -3962389.92116629, -3822979.23919869, -3666751.38186485, -3493212.37971975,
                                 -3302099.71461769, -3093392.43317121, -2867475.54470152, -2625108.03673121,
                                 -2367586.06128219, -2096921.32817857, -1815701.10075496, -1527523.11762782,
                                 -1237077.39816553, -950609.3080458, -677002.74349353, -429060.85924996,
                                 -224317.60933134, -82733.32462396, -22233.12804732])

        # The relative error is a bit larger at s \approx 1, where the
        # absolute magnitude is quite small, so drop those points.
        np.testing.assert_allclose(jdotB[:-5], jdotB_sfincs[:-5], rtol=0.1)

        if False:
            import matplotlib.pyplot as plt
            plt.plot(surfaces, jdotB, '.-', label='Redl')
            plt.plot(surfaces, jdotB_sfincs, '.-', label='sfincs')
            plt.xlabel('s')
            plt.title('J dot B')
            plt.legend(loc=0)
            plt.show()

        if False:
            import matplotlib.pyplot as plt
            nrows = 2
            ncols = 2
            for j in range(4):
                plt.subplot(nrows, ncols, j + 1)
                plt.contourf(details.phi1d, details.theta1d, details.modB[:, :, j], 25)
                plt.ylabel('theta')
                plt.xlabel('phi')
                plt.title(f'modB, s={surfaces[j]}')
                plt.colorbar()
            plt.tight_layout()
            plt.show()

    @unittest.skip("")
    def test_Redl_sfincs_WistellA(self):
        """
        Compare the Redl <j dot B> to a SFINCS calculation for Wistell-A.
        """
        ne = ProfilePolynomial(0.9e20 * np.array([1, 0, 0, 0, 0, -1]))
        #Te = ProfilePolynomial(1.3e3 * np.array([1, -1]))
        #Te = ProfilePolynomial(2.5e3 * np.array([1, -1]))
        Te = ProfilePolynomial(3.5e3 * np.array([1, -1]))
        Ti = Te
        Zeff = 1
        helicity_N = 4
        #filename = os.path.join(TEST_DIR, 'wout_ITERModel_reference.nc')
        #filename = '/Users/mattland/Box Sync/work19/20191229-01-wistell_self_consistent_current_profiles/wout_wista_15_beta_0.71.00000.nc'
        #filename = '/Users/mattland/Box Sync/work19/20191229-01-wistell_self_consistent_current_profiles/wout_wista_15_beta_1.39.00000.nc'
        filename = '/Users/mattland/Box Sync/work19/20191229-01-wistell_self_consistent_current_profiles/wout_wista_15_beta_1.97.00000.nc'
        vmec = Vmec(filename)
        surfaces = vmec.s_full_grid[1:-1]  # Drop points on the axis and edge to avoid divide-by-0
        jdotB, details = vmec_j_dot_B_Redl(vmec, surfaces, ne, Te, Ti, Zeff, helicity_N, plot=False)

        # The current profile in this vmec file was computed using SFINCS:
        jdotB_sfincs = vmec.wout.jdotb[1:-1]

        # The relative error is a bit larger at s \approx 1, where the
        # absolute magnitude is quite small, so drop those points.
        #np.testing.assert_allclose(jdotB[:-5], jdotB_sfincs[:-5], rtol=0.1)

        if True:
            import matplotlib.pyplot as plt
            plt.plot(surfaces, jdotB, '.-', label='Redl')
            plt.plot(surfaces, jdotB_sfincs, '.-', label='sfincs')
            plt.legend(loc=0)
            plt.xlabel('s')
            plt.title('J dot B')
            plt.show()

    def test_Redl_sfincs_precise_QA(self):
        """
        Compare the Redl <j dot B> to a SFINCS calculation for the precise QA.

        The SFINCS calculation is on cobra in
        /ptmp/mlan/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/20211226-01-012_QA_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces
        """
        ne = ProfilePolynomial(4.13e20 * np.array([1, 0, 0, 0, 0, -1]))
        Te = ProfilePolynomial(12.0e3 * np.array([1, -1]))
        Ti = Te
        Zeff = 1
        helicity_N = 0
        # filename = os.path.join(TEST_DIR, 'wout_new_QA_aScaling.nc')  # High resolution
        filename = os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc')
        vmec = Vmec(filename)
        surfaces = np.linspace(0.025, 0.975, 39)
        jdotB, details = vmec_j_dot_B_Redl(vmec, surfaces, ne, Te, Ti, Zeff, helicity_N, plot=False)

        jdotB_sfincs = np.array([-2164875.78234086, -3010997.004258, -3586912.40439179, -4025873.78974165,
                                 -4384855.40656673, -4692191.91608418, -4964099.33007648, -5210508.61474677,
                                 -5442946.68999908, -5657799.82786579, -5856450.57370037, -6055808.19817868,
                                 -6247562.80014873, -6431841.43078959, -6615361.81912527, -6793994.01503932,
                                 -6964965.34953497, -7127267.47873969, -7276777.92844458, -7409074.62499181,
                                 -7518722.07866914, -7599581.37772525, -7644509.67670812, -7645760.36382036,
                                 -7594037.38147436, -7481588.70786642, -7299166.08742784, -7038404.20002745,
                                 -6691596.45173419, -6253955.52847633, -5722419.58059673, -5098474.47777983,
                                 -4390147.20699043, -3612989.71633149, -2793173.34162084, -1967138.17518374,
                                 -1192903.42248978, -539990.088677, -115053.37380415])

        if False:
            import matplotlib.pyplot as plt
            plt.plot(surfaces, jdotB, '.-', label='Redl')
            plt.plot(surfaces, jdotB_sfincs, '.-', label='sfincs')
            plt.legend(loc=0)
            plt.xlabel('s')
            plt.ylabel('J dot B [SI units]')
            plt.title('Bootstrap current for the precise QA (Landreman & Paul (2021))')
            plt.show()

        np.testing.assert_allclose(jdotB[1:-1], jdotB_sfincs[1:-1], rtol=0.1)

    def test_Redl_sfincs_precise_QH(self):
        """
        Compare the Redl <j dot B> to a SFINCS calculation for the precise QH.

        The SFINCS calculation is on cobra in
        /ptmp/mlan/20211226-01-sfincs_for_precise_QS_for_Redl_benchmark/20211226-01-019_QH_Ntheta25_Nzeta39_Nxi60_Nx7_manySurfaces
        """
        ne = ProfilePolynomial(4.13e20 * np.array([1, 0, 0, 0, 0, -1]))
        Te = ProfilePolynomial(12.0e3 * np.array([1, -1]))
        Ti = Te
        Zeff = 1
        helicity_N = -4
        #filename = os.path.join(TEST_DIR, 'wout_new_QH_aScaling.nc')  # High resolution
        filename = os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QH_reactorScale_lowres_reference.nc')
        vmec = Vmec(filename)
        surfaces = np.linspace(0.025, 0.975, 39)
        jdotB, details = vmec_j_dot_B_Redl(vmec, surfaces, ne, Te, Ti, Zeff, helicity_N, plot=False)

        jdotB_sfincs = np.array([-1086092.9561775, -1327299.73501589, -1490400.04894085, -1626634.32037339,
                                 -1736643.64671843, -1836285.33939607, -1935027.3099312, -2024949.13178129,
                                 -2112581.50178861, -2200196.92359437, -2289400.72956248, -2381072.32897262,
                                 -2476829.87345286, -2575019.97938908, -2677288.45525839, -2783750.09013764,
                                 -2894174.68898196, -3007944.74771214, -3123697.37793226, -3240571.57445779,
                                 -3356384.98579004, -3468756.64908024, -3574785.02500657, -3671007.37469685,
                                 -3753155.07811322, -3816354.48636373, -3856198.2242986, -3866041.76391937,
                                 -3839795.40512069, -3770065.26594065, -3649660.76253605, -3471383.501417,
                                 -3228174.23182819, -2914278.54799143, -2525391.54652021, -2058913.26485519,
                                 -1516843.60879267, -912123.395174, -315980.89711036])

        if False:
            import matplotlib.pyplot as plt
            plt.plot(surfaces, jdotB, '.-', label='Redl')
            plt.plot(surfaces, jdotB_sfincs, '.-', label='sfincs')
            plt.legend(loc=0)
            plt.xlabel('s')
            plt.ylabel('J dot B [SI units]')
            plt.title('Bootstrap current for the precise QH (Landreman & Paul (2021))')
            plt.show()

        np.testing.assert_allclose(jdotB, jdotB_sfincs, rtol=0.1)

    def test_VmecRedlBootstrapMismatch_1(self):
        """
        The VmecRedlBootstrapMismatch objective function should be
        approximately 1.0 if the vmec configuration is a vacuum
        field. (It will not be exactly 1.0 since there is numerical
        noise in vmec's jdotb.)
        """
        ne = ProfilePolynomial(5.0e20 * np.array([1, 0, 0, 0, -1]))
        Te = ProfilePolynomial(8e3 * np.array([1, -1]))
        Ti = Te
        Zeff = 1
        helicity_N = 0
        filename = os.path.join(TEST_DIR, 'wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc')
        vmec = Vmec(filename)
        obj = VmecRedlBootstrapMismatch(vmec, ne, Te, Ti, Zeff, helicity_N)
        obj1 = obj.J()
        logging.info(f'test_VmecRedlBootstrapMismatch_1: objective is {obj1:.3e} '
                     f'and should be approximately 1.0. Diff: {obj1 - 1.0}')
        np.testing.assert_allclose(obj1, 1.0, rtol=1e-6)

    def test_VmecRedlBootstrapMismatch_independent_of_ns(self):
        """
        Confirm that the VmecRedlBootstrapMismatch objective function is
        approximately independent of the radial resolution ns.
        """
        ne = ProfilePolynomial(5.0e20 * np.array([1, 0, 0, 0, -1]))
        Te = ProfilePolynomial(8e3 * np.array([1, -1]))
        Ti = Te
        Zeff = 1
        helicity_N = 0
        filename = os.path.join(TEST_DIR, 'input.ITERModel')
        vmec = Vmec(filename)
        obj = VmecRedlBootstrapMismatch(vmec, ne, Te, Ti, Zeff, helicity_N, nphi=3)
        # Resolution 1:
        vmec.indata.ns_array[:3] = [13, 25, 0]
        vmec.indata.ftol_array[:3] = [1e-20, 1e-15, 0]
        vmec.indata.niter_array[:3] = [500, 2000, 0]
        obj1 = obj.J()
        # Resolution 2:
        vmec.indata.ns_array[:3] = [13, 25, 51]
        vmec.indata.ftol_array[:3] = [1e-20, 1e-15, 1e-15]
        vmec.indata.niter_array[:3] = [500, 800, 2000]
        vmec.need_to_run_code = True
        obj2 = obj.J()
        rel_diff = (obj1 - obj2) / (0.5 * (obj1 + obj2))
        logging.info(f'resolution1: {obj1:.4e}  resolution2: {obj2:.4e}  rel diff: {rel_diff}')
        np.testing.assert_allclose(obj1, obj2, rtol=0.003)

