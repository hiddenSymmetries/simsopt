# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
from typing import Optional
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    from desc.equilibrium import Equilibrium
    from desc.geometry import FourierRZToroidalSurface
    from desc.profiles import PowerSeriesProfile
except ImportError as e:
    Equilibrium = None
    logger.debug(str(e))

from .._core.optimizable import Optimizable
from .._core.util import Struct, ObjectiveFailure
from ..geo.surfacerzfourier import SurfaceRZFourier

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None


def desc_to_simsopt_surf(desc_surf):
    simsopt_surf = SurfaceRZFourier(
        nfp=desc_surf.NFP,
        mpol=desc_surf.M,
        ntor=desc_surf.N,
        stellsym=desc_surf.sym
        )
    
    if not simsopt_surf.stellsym:
        raise NotImplementedError('DESC class only implemented for stellarator symmetric surfaces')

    for lmn, mode in zip(desc_surf.R_basis.modes, desc_surf.R_lmn):
        mm = lmn[1]
        nn = lmn[2]
        mm = lmn[1]
        nn = lmn[2]
        if mm<0:
            continue
        if mm==0 and nn<0:
            continue
        name = f'rc({mm},{nn})'
        simsopt_surf.set( name, mode )
    for lmn, mode in zip(desc_surf.Z_basis.modes, desc_surf.Z_lmn):
        mm = lmn[1]
        nn = lmn[2]
        mm = lmn[1]
        nn = lmn[2]
        if mm<0:
            continue
        if mm==0 and nn<0:
            continue
        name = f'zs({mm},{nn})'
        simsopt_surf.set( name, mode )

    return simsopt_surf



class Desc(Optimizable):
    """
    Class to use DESC as equilibrium solver

    Args:
        filename: VMEC or DESC input file in which the boundary is defined
        M: Poloidal resolution. By default, set to resolution of boundary
        N: Toroidal resolution. By default, set to resolution of boundary
    """
    def __init__(self,
                 filename: str = None,
                 M: int = None,
                 N: int = None,
                 L: int = None,
                 psi: int = 1
                 ):
        
        # Read surface - we use DESC reading method to accept both DESC and VMEC input files
        self._desc_surf = FourierRZToroidalSurface.from_input_file( filename )

        # Transform surface to a SurfaceRZFourier
        self._simsopt_surf = desc_to_simsopt_surf( self._desc_surf )


        
        # Set spectral resolution
        if M is None:
            self._M = self.desc_surf.M
        else:
            self._M = M

        if N is None:
            self._N = self.desc_surf.N
        else:
            self._M = N

        if L is None:
            self._L = self.M # Default in DESC if spectral_indexing=ANSI
        else:
            self._L = L

        # Set toroidal flux
        self._psi = psi

        # Define equilibrium
        self.equilibrium = Equilibrium( 
            L=self._L,
            M=self._M,
            N=self._N,
            surface=self._desc_surf,
            Psi=self._psi
        )

        # Set recompute bell to True
        self.need_to_run_code = True

    # Resolution parameters are properties
    @property
    def M(self):
        return self._M
    @property
    def N(self):
        return self._N
    @property
    def L(self):
        return self._L
    @property
    def psi(self):
        return self._psi
    
    # Everytime resolution is changed, change it both internally and for the self.equilibrium instance. Then, need to run code.
    @M.setter
    def M(self, M):
        if M is not self._M:
            self._M = M
            self.equilibrium.change_resolution(M=M)
            self.need_to_run_code = True
    @M.setter
    def N(self, N):
        if N is not self._N:
            self._N = N
            self.equilibrium.change_resolution(N=N)
            self.need_to_run_code = True
    @L.setter
    def L(self, L):
        if L is not self._L:
            self._L = L
            self.equilibrium.change_resolution(L=L)
            self.need_to_run_code = True
    @psi.setter
    def psi(self, psi):
        if psi is not self._psi:
            self._psi = psi
            self.equilibrium.psi = psi
            self.need_to_run_code = True


    def run():
        pass


