# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides a class that handles the transformation to
Boozer coordinates, and an optimization target for quasisymmetry.
"""

import logging
import os.path
import numpy as np

booz_xform_found = True
try:
    import booz_xform
except:
    booz_xform_found = False

from simsopt.core import Optimizable

logger = logging.getLogger(__name__)

class Boozer(Optimizable):
    """
    This class handles the transformation to Boozer coordinates.
    """
    def __init__(self, equil, mpol=None, ntor=None):
        """
        Constructor
        """
        if not booz_xform_found:
            raise RuntimeError("To use a Boozer object, the booz_xform package"
                               "must be installed. Run 'pip install -v booz_xform'")
        self.equil = equil
        self.depends_on = ["equil"]
        self.mpol = mpol
        self.ntor = ntor
        self.bx = booz_xform.Booz_xform()
        self.need_to_run_code = True

class Quasisymmetry(Optimizable):
    """
    This class is used to compute the departure from quasisymmetry on
    a given flux surface based on the Boozer spectrum.
    """
    def __init__(self,
                 boozer: Boozer,
                 s: float,
                 m: int,
                 n: int,
                 normalization: str = "B00",
                 weight: str = "even") -> None:
        """
        Constructor

        Args:
            boozer: A Boozer object on which the calculation will be based.
            s: The normalized toroidal magnetic flux for the flux surface to analyze. Should be in the range [0, 1].
            m: The departure from symmetry B(m * theta - nfp * n * zeta) will be reported.
            n: The departure from symmetry B(m * theta - nfp * n * zeta) will be reported.
            normalization: A uniform normalization applied to all bmnc harmonics.
            weight: An option for a m- or n-dependent weight to be applied to the bmnc amplitudes.
        """
        self.boozer = boozer
        self.s = s
        self.m = m
        self.n = n
        self.normalization = normalization
        self.weight = weight
        self.depends_on = ['boozer']

    def J(self) -> float:
        """
        Carry out the calculation of the quasisymmetry error.
        """
        # The next line is the expensive part of the calculation:
        bmnc = self.boozer.bmnc(self.s)
        xm = self.boozer.xm
        xn = self.boozer.xn / self.boozer.nfp

        if self.m != 0 and self.m != 1:
            raise ValueError("m for quasisymmetry should be 0 or 1.")

        # Find the indices of the symmetric modes:
        if self.n == 0:
            # Quasi-axisymmetry
            symmetric = (xn == 0)
            
        elif self.m == 0:
            # Quasi-poloidal symmetry
            symmetric = (xm == 0)
            
        else:
            # Quasi-helical symmetry
            symmetric = (xm * self.n + xn * self.m == 0)
            # Stellopt takes the "and" of this with mod(xm, self.m),
            # which does not seem necessary since self.m must be 1 to
            # get here.
        nonsymmetric = np.logical_not(symmetric)

        # Scale all bmnc modes so the average |B| is 1 or close to 1:
        
        if self.normalization == "B00":
            # Normalize by the (m,n) = (0,0) mode amplitude:
            assert xm[0] == 0
            assert xn[0] == 0
            bnorm = bmnc[0]

        elif self.normalization == "symmetric":
            # Normalize by sqrt(sum_{symmetric modes} B{m,n}^2)
            temp = bmnc[symmetric]
            bnorm = np.sqrt(np.dot(temp, temp))
            
        else:
            raise ValueError("Unrecognized value for normalization in Quasisymmetry")
        
        bmnc /= bnorm

        # Apply any weight that depends on m and/or n:
            
        if self.weight == "even":
            # Evenly weight each bmnc mode. Normalize by the m=n=0 mode on that surface.
            return bmnc[nonsymmetric]
        
        elif self.weight == "stellopt":
            # Stellopt applies a m-dependent radial weight:
            s_used = self.s # This line may need to be changed
            rad_sigma = np.full_like(xm, s_used * s_used)
            rad_sigma[xm < 3] = s_used
            rad_sigma[xm == 3] = s_used ** 1.5
            temp = bmnc / rad_sigma
            return temp[nonsymmetric]

        else:
            raise ValueError("Unrecognized value for weight in Quasisymmetry")
            
