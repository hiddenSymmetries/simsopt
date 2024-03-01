import numpy as np
from monty.json import MSONable, MontyDecoder, MontyEncoder

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec

__all__ = ['SquaredFlux']


class SquaredFlux(Optimizable):

    r"""
    Objective representing the quadratic flux of a field on a surface, that is

    .. math::
        \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
    """

    def __init__(self, surface, field, target=None):
        self.surface = surface
        self.target = target
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)
        Btarget = self.target if self.target is not None else []
        return sopp.integral_BdotN(Bcoil, Btarget, n)

    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1./absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)
        if self.target is not None:
            B_n = (Bcoil_n - self.target)
        else:
            B_n = Bcoil_n
        dJdB = (B_n[..., None] * unitn * absn[..., None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)

    def as_dict(self) -> dict:
        return MSONable.as_dict(self)

    @classmethod
    def from_dict(cls, d):
        decoder = MontyDecoder()
        surface = decoder.process_decoded(d["surface"])
        field = decoder.process_decoded(d["field"])
        target = decoder.process_decoded(d["target"])
        return cls(surface, field, target)
