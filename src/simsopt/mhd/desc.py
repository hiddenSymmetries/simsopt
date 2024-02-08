# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""
This module provides a class that handles the VMEC equilibrium code.
"""

import logging
import os.path
from .._core.types import RealArray
from typing import Union, Iterable
from typing import Optional
from datetime import datetime
from .._core.derivative import derivative_dec, Derivative
import jax

import numpy as np

logger = logging.getLogger(__name__)

try:
    from mpi4py import MPI
except ImportError as e:
    MPI = None
    logger.debug(str(e))

try:
    from desc.equilibrium import Equilibrium
except ImportError as e:
    Equilibrium = None
    logger.debug(str(e))

try:
    from desc.geometry import FourierRZToroidalSurface
except ImportError as e:
    FourierRZToroidalSurface = None
    logger.debug(str(e))

try:
    from desc.profiles import PowerSeriesProfile
except ImportError as e:
    PowerSeriesProfile = None
    logger.debug(str(e))

try:
    from desc.vmec_utils import ptolemy_identity_fwd, ptolemy_identity_rev
except ImportError as e:
    ptolemy_identity_fwd = None
    ptolemy_identity_rev = None
    logger.debug(str(e))


from desc.continuation import solve_continuation_automatic
from desc.grid import Grid
from desc.objectives import ObjectiveFunction, RotationalTransform, ForceBalance, QuasisymmetryBoozer, QuasisymmetryTwoTerm, QuasisymmetryTripleProduct
from desc.optimize import ProximalProjection

from .._core.optimizable import Optimizable
from .._core.util import Struct, ObjectiveFailure
from ..geo.surfacerzfourier import SurfaceRZFourier

if MPI is not None:
    from ..util.mpi import MpiPartition
else:
    MpiPartition = None

__all__ = ['Desc', 'DescRotationalTransform']


def desc_to_simsopt_surf(desc_surf, simsopt_surf):
    if ptolemy_identity_rev is None:
        raise RuntimeError('desc.ptolemy_identity_rev needs to be available for transform a DESC surface to a SurfaceRZFourier')
    

    # R modes
    mm, nn, rs, rc = ptolemy_identity_rev(desc_surf.R_basis.modes[:,1], desc_surf.R_basis.modes[:,2], desc_surf.R_lmn )
    for m, n, rmnc, rmns in zip(mm,nn,rc[0],rs[0]):
        simsopt_surf.set(f'rc({m},{n})', rmnc)
        if not(m==0 and n==0) and not simsopt_surf.stellsym:
            simsopt_surf.set(f'rs({m},{n})', rmns)

    # Z modes
    mm, nn, zs, zc = ptolemy_identity_rev(desc_surf.Z_basis.modes[:,1], desc_surf.Z_basis.modes[:,2], desc_surf.Z_lmn )
    for m, n, zmnc, zmns in zip(mm,nn,zc[0],zs[0]):
        if not simsopt_surf.stellsym:
            simsopt_surf.set(f'rc({m},{n})', zmnc)
        if not(m==0 and n==0):
            simsopt_surf.set(f'zs({m},{n})', zmns)
    

def simsopt_to_desc_surf(simsopt_surf):
    rc = simsopt_surf.rc.reshape((-1,))[simsopt_surf.ntor:]
    rs = simsopt_surf.rs.reshape((-1,))[simsopt_surf.ntor:]
    zc = simsopt_surf.zc.reshape((-1,))[simsopt_surf.ntor:]
    zs = simsopt_surf.zs.reshape((-1,))[simsopt_surf.ntor:]

    nm = rc.size
    m_0 = simsopt_surf.m[:nm]
    n_0 = simsopt_surf.n[:nm]
    m_1, n_1, r = ptolemy_identity_fwd(m_0, n_0, rs, rc)
    _,   _,   z = ptolemy_identity_fwd(m_0, n_0, zs, zc)

    mm = [int(m) for m in m_1]
    nn = [int(n) for n in n_1]

    return FourierRZToroidalSurface(
        R_lmn = r.reshape((-1,)),
        Z_lmn = z.reshape((-1,)),
        modes_R = np.array([mm,nn]).transpose(),
        NFP = simsopt_surf.nfp,
        sym = simsopt_surf.stellsym,
        rho = 1
    )



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
        self._boundary = SurfaceRZFourier(
            nfp=self._desc_surf.NFP,
            mpol=self._desc_surf.M,
            ntor=self._desc_surf.N,
            stellsym=self._desc_surf.sym
            )
        desc_to_simsopt_surf( self._desc_surf, self._boundary )
        
        # Set spectral resolution
        if M is None:
            self._M = self._desc_surf.M
        else:
            self._M = M

        if N is None:
            self._N = self._desc_surf.N
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
        self.continuation = True

        # Initialize Optimizable
        x0 = self.get_dofs()
        fixed = np.full(len(x0), True)
        names = []
        super().__init__(x0=x0, fixed=fixed, names=names, depends_on=[self._boundary])

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
    
    
    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        if boundary is not self._boundary:
            logging.debug('Replacing surface in boundary setter')
            self.remove_parent(self._boundary)
            self._boundary = boundary
            self.append_parent(boundary)
            self.need_to_run_code = True
    
    # Everytime resolution is changed, change it both internally and for the self.equilibrium instance. Then, need to run code.
    @M.setter
    def M(self, M):
        if M is not self._M:
            self._M = M
            self.equilibrium.change_resolution(M=M)
            self._simsopt_surf.change_resolution(mpol=M, ntor=self.N)
            self.need_to_run_code = True
    @M.setter
    def N(self, N):
        if N is not self._N:
            self._N = N
            self.equilibrium.change_resolution(N=N)
            self._simsopt_surf.change_resolution(mpol=self.M, ntor=N)
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

    def get_dofs(self):
        return []
    
    def set_dofs(self, x):
        self.need_to_run_code = True

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run(self, **kargs):
        if not self.need_to_run_code:
            logger.info("run() called but no need to re-run DESC.")
            return

        # Transfer boundary dofs to DESC
        self.equilibrium.surface = simsopt_to_desc_surf( self.boundary )

        # Run DESC
        if self.continuation:
            # If this is the first tiem we run this equilibrium, need to use continuation method to ensure good convergence
            eqf = solve_continuation_automatic(self.equilibrium.copy(), **kargs)
            self.equilibrium = eqf[-1]
            self.continuation = False 
        else:
            # Otherwise, start from former equilibrium
            self.equilibrium, _ = self.equilibrium.solve(copy=True)


        # No need to rerun,
        self.need_to_run_code = False

    def volume(self):
        return self.boundary.volume()
    
    def aspect_ratio(self):
        return self.boundary.aspect_ratio()


def desc_to_simsopt_derivatives(eq, derivatives):
    # Transform DESC derivatives into VMEC format
    mm = eq.surface.R_basis.modes[:,1]
    nn = eq.surface.R_basis.modes[:,2]
    _, _, rs, rc = ptolemy_identity_rev(mm, nn, derivatives['Rb_lmn'] )
    mm = eq.surface.Z_basis.modes[:,1]
    nn = eq.surface.Z_basis.modes[:,2]
    _, _, zs, zc = ptolemy_identity_rev(mm, nn, derivatives['Zb_lmn'] )

    # Order derivatives as required by simsopt
    if eq.sym:
        return np.append(rc[0], zs[0][1:])
    else:
        return np.append(rc[0], rs[0][1:], zc[0], zs[0][1:])
    


class _DescObjective(Optimizable):
    def __init__(self, desc:Desc, objective):
        self.desc = desc
        self.objective = objective


        constraint = ObjectiveFunction(ForceBalance(self.desc.equilibrium))
        self.prox = ProximalProjection(self.objective, constraint, self.desc.equilibrium)

        self.prox.build()

        super().__init__(depends_on=[desc])

    def J(self):
        self.desc.run()

        logger.debug(f'Evaluating roational transform constraint from DESC')
        x = self.prox.x(self.desc.equilibrium)
        return self.prox.compute_unscaled(x)
    
    @derivative_dec
    def dJ(self):
        x = self.prox.x(self.desc.equilibrium)
        jac = self.prox.jac_scaled(x)
        derivatives = jax.vmap(self.prox.unpack_state, (0, None))(jac, False)[0]

        return Derivative({
            self.desc.boundary: desc_to_simsopt_derivatives( self.desc.equilibrium, derivatives)
            }
            )


class DescRotationalTransform(_DescObjective):
    def __init__(self, desc:Desc, **kwargs):
        objective = ObjectiveFunction(
            RotationalTransform(desc.equilibrium, **kwargs)
            )
        super().__init__(desc=desc, objective=objective)

class DescQSBoozer(_DescObjective):
    def __init__(self, desc:Desc, **kwargs):
        objective = ObjectiveFunction(
            QuasisymmetryBoozer(
            eq=desc.equilibrium, **kwargs
            )
        )
        super().__init__(desc=desc, objective=objective)

class DescQuasisymmetryTwoTerm(_DescObjective):
    def __init__(self, desc:Desc, **kwargs):
        objective = ObjectiveFunction(
            QuasisymmetryTwoTerm(
            eq=desc.equilibrium, **kwargs
            )
        )
        super().__init__(desc=desc, objective=objective)

class DescQuasisymmetryTripleProduct(_DescObjective):
    def __init__(self, desc:Desc, **kwargs):
        objective = ObjectiveFunction(
            QuasisymmetryTripleProduct(
            eq=desc.equilibrium, **kwargs
            )
        )
        super().__init__(desc=desc, objective=objective)











    




