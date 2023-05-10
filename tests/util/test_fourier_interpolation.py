#!/usr/bin/env python3

import unittest
import numpy as np
from numpy.testing import assert_allclose
from simsopt.util.fourier_interpolation import fourier_interpolation, sin_coeff, cos_coeff
from math import pi


class FourierInterpolationTests(unittest.TestCase):

    def test_constant(self):
        """
        If data is constant, the interpolated values should be equal to that constant
        """
        for ndata in range(2, 10):
            for const in [-0.4, 1.3]:
                data = np.full(ndata, const)
                for nrequest in range(1, 5):
                    xrequest = np.random.rand(nrequest) * 20 - 10
                    y = fourier_interpolation(data, xrequest)
                    #print('data:', data, ', xrequest:', xrequest, ', y:', y)
                    np.testing.assert_allclose(y, np.full(nrequest, const), rtol=1e-10, atol=1e-10)

    def test_single_mode(self):
        """
        Verify that we can interpolate a single Fourier mode exactly
        """
        phase = 0.7
        for ndata in range(3, 10):
            xdata = np.linspace(0, 2 * np.pi, ndata, endpoint=False)
            for const in [-0.4, 1.3]:
                for amp in [-0.6, 2.1]:
                    data = const + amp * np.cos(xdata + phase)
                    for nrequest in range(1, 5):
                        xrequest = np.random.rand(nrequest) * 20 - 10
                        y = fourier_interpolation(data, xrequest)
                        #print('data:', data, ', xrequest:', xrequest, ', y:', y)
                        np.testing.assert_allclose(y, const + amp * np.cos(xrequest + phase), rtol=1e-10, atol=1e-10)

    def test_multiple_modes(self):
        """
        Verify that we can interpolate a sum of Fourier modes exactly
        """
        for nmodes in range(5):
            # Make sure we have enough sample points to resolve all the modes
            for ndata in range(nmodes * 2 + 1, 20):
                xdata = np.linspace(0, 2 * np.pi, ndata, endpoint=False)
                amps = np.random.rand(nmodes) * 4 - 2
                phases = np.random.rand(nmodes) * 20 - 10
                data = np.zeros(ndata)
                for n in range(nmodes):
                    data += amps[n] * np.cos(n * xdata + phases[n])

                for nrequest in range(1, 5):
                    xrequest = np.random.rand(nrequest) * 20 - 10
                    y = fourier_interpolation(data, xrequest)

                    yexpected = np.zeros(nrequest)
                    for n in range(nmodes):
                        yexpected += amps[n] * np.cos(n * xrequest + phases[n])

                    #print('data:', data, ', xrequest:', xrequest, ', y:', y)
                    np.testing.assert_allclose(y, yexpected, rtol=1e-10, atol=1e-10)


def sawtooth(x, L=1):
    x = np.mod(x, 2*L)
    return x/L


def triangle(x, L=1):
    x = np.mod(x, 2*L)
    return np.where(x < L, x/L, -x/L+2) 


def square_wave(x, L=1):
    x = np.mod(x, 2*L)
    return np.where(x < L, -1, 1)


def x_square(x, L=1):
    return (x-L)**2


class TestFourierCoefficients(unittest.TestCase):

    def setUp(self):
        self.L = 5
        self.f1 = lambda x: sawtooth(x, self.L)
        self.f2 = lambda x: triangle(x, self.L)
        self.f3 = lambda x: square_wave(x, self.L)
        self.f4 = lambda x: x_square(x, self.L)

    def test_sin_coeff_sawtooth(self):
        for n in range(1, 30):
            self.assertAlmostEqual(sin_coeff(self.f1, n, self.L), -2/(n*np.pi), places=6)
            self.assertAlmostEqual(sin_coeff(self.f2, n, self.L), 0, places=6)
            self.assertAlmostEqual(sin_coeff(self.f3, n, self.L), 2*((-1)**n-1)/(n*pi), places=4)
            self.assertAlmostEqual(sin_coeff(self.f4, n, self.L), 0, places=6)

    def test_cos_coeff(self):
        for n in range(1, 30):
            self.assertAlmostEqual(cos_coeff(self.f1, n, self.L), 0, places=6)
            self.assertAlmostEqual(cos_coeff(self.f2, n, self.L), 2*(((-1)**n-1)/(n*pi)**2), places=6)
            self.assertAlmostEqual(cos_coeff(self.f3, n, self.L), 0, places=6)
            self.assertAlmostEqual(cos_coeff(self.f4, n, self.L), 4*self.L**2/(n*pi)**2, places=6)

    def test_cos_coeff_order0(self):
        n = 0
        self.assertAlmostEqual(cos_coeff(self.f1, n, self.L), 1, places=6)
        self.assertAlmostEqual(cos_coeff(self.f2, n, self.L), 0.5, places=6)
        self.assertAlmostEqual(cos_coeff(self.f3, n, self.L), 0, places=6)
        self.assertAlmostEqual(cos_coeff(self.f4, n, self.L), self.L**2/3, places=6)


if __name__ == "__main__":
    unittest.main()
