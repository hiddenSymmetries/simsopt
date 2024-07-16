import numpy as np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.derivative import derivative_dec


__all__ = ['SquaredFlux']


class SquaredFlux(Optimizable):

    r"""
    Objective representing quadratic-flux-like quantities, useful for stage-2
    coil optimization. Several variations are available, which can be selected
    using the ``definition`` argument. For ``definition="quadratic flux"`` 
    (the default), the objective is defined as

    .. math::
        J = \frac12 \int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds,

    where :math:`\mathbf{n}` is the surface unit normal vector and
    :math:`B_T` is an optional (zero by default) target value for the
    magnetic field. Also :math:`\int_{S} ds` indicates a surface integral.
    For ``definition="normalized"``, the objective is defined as

    .. math::
        J = \frac12 \frac{\int_{S} (\mathbf{B}\cdot \mathbf{n} - B_T)^2 ds}
                         {\int_{S} |\mathbf{B}|^2 ds}.

    For ``definition="local"``, the objective is defined as

    .. math::
        J = \frac12 \int_{S} \frac{(\mathbf{B}\cdot \mathbf{n} - B_T)^2}{|\mathbf{B}|^2} ds.

    The definition ``"quadratic flux"`` has the advantage of simplicity, and it
    is used in other contexts such as REGCOIL. However for stage-2 optimization,
    the optimizer can "cheat", lowering this objective by reducing the magnitude
    of the field. The definitions ``"normalized"`` and ``"local"`` close this loophole.

    Args:
        surface: A :obj:`simsopt.geo.surface.Surface` object on which to compute the flux
        field: A :obj:`simsopt.field.magneticfield.MagneticField` for which to compute the flux.
        target: A ``nphi x ntheta`` numpy array containing target values for the flux. Here 
          ``nphi`` and ``ntheta`` correspond to the number of quadrature points on `surface` 
          in ``phi`` and ``theta`` direction.
        definition: A string to select among the definitions above. The
          available options are ``"quadratic flux"``, ``"normalized"``, and ``"local"``.
    """


    def __init__(self, surface, field, Btarget=None):

        self.surface = surface
        if target is not None:
            self.target = np.ascontiguousarray(target)
        else:
            self.target = np.zeros(self.surface.normal().shape[:2])
        self.field = field
        xyz = self.surface.gamma()
        self.field.set_points(xyz.reshape((-1, 3)))
        if definition not in ["quadratic flux", "normalized", "local"]:
            raise ValueError("Unrecognized option for 'definition'.")
        self.definition = definition
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[field])

    def J(self):
        n = self.surface.normal()
        Bcoil = self.field.B().reshape(n.shape)

        return sopp.integral_BdotN(Bcoil, self.Btarget, n)


    @derivative_dec
    def dJ(self):
        n = self.surface.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1. / absn)[:, :, None]
        Bcoil = self.field.B().reshape(n.shape)
        Bcoil_n = np.sum(Bcoil*unitn, axis=2)

        B_n = (Bcoil_n - self.Btarget)
        dJdB = (B_n[..., None] * unitn * absn[..., None])/absn.size
        dJdB = dJdB.reshape((-1, 3))
        return self.field.B_vjp(dJdB)

