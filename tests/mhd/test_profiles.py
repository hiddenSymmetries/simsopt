import unittest
import logging
import numpy as np
from simsopt.mhd.profiles import ProfilePolynomial, compute_trapped_fraction

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG)


class ProfilesTests(unittest.TestCase):
    def test_polynomial(self):
        """
        Test the value and derivative of a polynomial profile
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        s = np.linspace(0, 1, 10)
        np.testing.assert_allclose(prof.f(s), 3 * (1 - s ** 3))
        np.testing.assert_allclose(prof.dfds(s), 3 * (- 3 * s ** 2))

    def test_plot(self):
        """
        Test the plotting function.
        """
        # Function: f(s) = 3 * (1 - s^3)
        prof = ProfilePolynomial([3, 0, 0, -3])
        prof.plot(show=False)


class RedlBootstrapTests(unittest.TestCase):
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

        Bmin, Bmax, epsilon, fsa_b2, f_t = compute_trapped_fraction(modB, sqrtg)
        # The average of (b0 + b1 cos(theta))^2 is b0^2 + (1/2) * b1^2
        np.testing.assert_allclose(fsa_b2, [13.0 ** 2 + 0.5 * 2.6 ** 2, 9.0 ** 2 + 0.5 * 3.7 ** 2])
        np.testing.assert_allclose(Bmin, [13.0 - 2.6, 9.0 - 3.7], rtol=1e-4)
        np.testing.assert_allclose(Bmax, [13.0 + 2.6, 9.0 + 3.7], rtol=1e-4)
        np.testing.assert_allclose(epsilon, [2.6 / 13.0, 3.7 / 9.0], rtol=1e-3)
        print('f_t:', f_t)

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
        Bmin, Bmax, epsilon_out, fsa_B2, f_t = compute_trapped_fraction(modB, sqrtg)

        f_t_Kim = 1.46 * np.sqrt(epsilon_in) - 0.46 * epsilon_in  # Eq (C18) in Kim et al

        np.testing.assert_allclose(Bmin, B0 / (1 + epsilon_in))
        np.testing.assert_allclose(Bmax, B0 / (1 - epsilon_in))
        np.testing.assert_allclose(epsilon_in, epsilon_out)
        # Eq (A8):
        np.testing.assert_allclose(fsa_B2, B0 * B0 / np.sqrt(1 - epsilon_in ** 2), rtol=1e-6)
        np.testing.assert_allclose(f_t, f_t_Kim, rtol=0.1, atol=0.07)  # We do not expect precise agreement. 

        if False:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(14, 5))
            plt.subplot(1, 2, 1)
            plt.plot(epsilon_in, f_t, label='simsopt')
            plt.plot(epsilon_in, f_t_Kim, label='Kim')
            plt.xlabel('epsilon')
            plt.title('Trapped fraction $f_t$')
            plt.legend(loc=0)

            plt.subplot(1, 2, 2)
            #plt.plot(epsilon_in, (f_t_Kim - f_t) / (1 - f_t))
            plt.plot(epsilon_in, f_t_Kim - f_t)
            plt.title('Relative difference in $f_c$')
            #plt.plot(epsilon_in, epsilon_out, 'r')
            #plt.plot(epsilon_in, epsilon_in, ':k')
            plt.show()
