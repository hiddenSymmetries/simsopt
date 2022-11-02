import logging

import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)

__all__ = ["SurfaceNewQuadPoints"]


class SurfaceNewQuadPoints(sopp.SurfaceNewQuadPoints, Surface):
    r"""
    ``SurfaceNewQuadPoints`` is a surface wrapper class that uses the
    DOFS of the parent surface class with new quadrature points.

    For more information about the arguments ``quadpoints_phi``, and
    ``quadpoints_theta``, see the general documentation on :ref:`surfaces`.
    Instead of supplying the quadrature point arrays along :math:`\phi` and
    :math:`\theta` directions, one could also specify the number of
    quadrature points for :math:`\phi` and :math:`\theta` using the
    class method :py:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`.

    Args:
        surface: Surface object whose DOFs are used
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self, surface, quadpoints_phi=None, quadpoints_theta=None):

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints()

        sopp.SurfaceNewQuadPoints.__init__(self, surface, quadpoints_phi,
                                           quadpoints_theta)
        self._surface = surface
        Surface.__init__(self, depends_on=[surface])

    def to_RZFourier(self):
        rz_fourier = self._surface.to_RZFourier()
        nfp = rz_fourier.nfp
        stellsym = rz_fourier.stellsym
        mpol = rz_fourier.mpol
        ntor = rz_fourier.ntor
        return SurfaceRZFourier(nfp=nfp, stellsym=stellsym, mpol=mpol, ntor=ntor,
                                quadpoints_phi=self.quadpoints_phi,
                                quadpoints_theta=self.quadpoints_theta)

