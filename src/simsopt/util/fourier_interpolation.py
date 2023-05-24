# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains a subroutine for spectrally accurate interpolation of
data that is known on a uniform grid in a periodic domain.
"""

import numpy as np

# Get machine precision
eps = np.finfo(float).eps


def fourier_interpolation(fk, x):
    """
    Interpolate data that is known on a uniform grid in [0, 2pi).

    This routine is based on the
    matlab routine fourint.m in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html

    Args:
        fk:  Vector of y-coordinates of data, at equidistant points
             x(k) = (k-1)*2*pi/N,  k = 1...N
        x:   Vector of x-values where interpolant is to be evaluated.

    Returns:
        Array of length ``len(x)`` with the interpolated values.
    """

    N = len(fk)
    M = len(x)

    # Compute equidistant points
    xk = (np.arange(N) * 2 * np.pi) / N

    # Weights for trig interpolation
    w = (-1.0) ** np.arange(0, N)

    D = 0.5 * (np.outer(x, np.ones(N)) - np.outer(np.ones(M), xk))

    if np.mod(N, 2) == 0:
        # Formula for N even
        D = 1 / np.tan(D + eps * (D == 0))
    else:
        # Formula for N odd
        D = 1 / np.sin(D + eps * (D == 0))

    # Evaluate interpolant as matrix-vector products
    return np.dot(D, w * fk) / np.dot(D, w)
