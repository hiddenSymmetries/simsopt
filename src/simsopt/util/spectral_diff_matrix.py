#!/usr/bin/env python3

"""
This module contains a subroutine for making spectral differentiation matrices.
"""

import numpy as np
from scipy.linalg import toeplitz

def spectral_diff_matrix(n, xmin=0, xmax=2*np.pi):
    """
    Return the spectral differentiation matrix for n grid points
    on the periodic domain [xmax, xmax). This routine is based on the
    matlab code in the DMSuite package by S.C. Reddy and J.A.C. Weideman, available at
    http://www.mathworks.com/matlabcentral/fileexchange/29
    or here:
    http://dip.sun.ac.za/~weideman/research/differ.html  
    """

    h = 2 * np.pi / n
    kk = np.arange(1, n)
    n1 = int(np.floor((n - 1) / 2))
    n2 = int(np.ceil((n - 1) / 2))
    if np.mod(n, 2) == 0:
        topc = 1 / np.tan(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, -np.flip(topc[0:n1])))
    else:
        topc = 1 / np.sin(np.arange(1, n2 + 1) * h / 2)
        temp = np.concatenate((topc, np.flip(topc[0:n1])))

    col1 = np.concatenate(([0], 0.5 * ((-1) ** kk) * temp))
    row1 = -col1
    D = 2 * np.pi / (xmax - xmin) * toeplitz(col1, r=row1)
    return D
