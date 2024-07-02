import numpy as np

import simsoptpp as sopp
from .curve import Curve

__all__ = ['CurveCWSFourier']


class CurveCWSFourier(sopp.CurveCWSFourier, Curve):
    r"""
    ``CurveCWSFourier`` is a curve parameterized to a surface, with the given parameterization:

    .. math::
        \theta(t) &= theta_{l} t + \sum_{m=0}^{\text{order}} theta_{c,m}\cos(m t) + \sum_{m=1}^{\text{order}} theta_{s,m}\sin(m t) \\
        \phi(t) &= phi_{l} t + \sum_{m=0}^{\text{order}} phi_{c,m}\cos(m t) + \sum_{m=1}^{\text{order}} phi_{s,m}\sin(m t) \\
    
    where :math:`t \in [0, 1]` is the curve parameter, and :math:`\theta` and :math:`\phi` are the poloidal and toroidal angles
    respectively. The curve is parameterized to a winding surface of type ``SurfaceRZFourier``.

    Note that for :math:`m=0` we skip the :math:m=0` term for the sin terms.

    Args:
        mpol: The number of poloidal modes of the winding surface.
        ntor: The number of toroidal modes of the winding surface.
        idofs: The dofs of the winding surface.
        quadpoints: The number of quadrature points to use for the curve.
        order: The order of the Fourier series.
        nfp: The number of field periods.
        stellsym: The stellarator symmetry of the winding surface.
        dofs: The dofs of the curve. If not provided, the dofs are initialized to zero. Not working yet - dofs must be defined outside of the constructor using `set_dofs` method.
    """

    def __init__(self, mpol, ntor, idofs, quadpoints, order, nfp, stellsym, dofs=None):
        if not np.isscalar(quadpoints):
            quadpoints = len(quadpoints)
        sopp.CurveCWSFourier.__init__(self, mpol, ntor, idofs, quadpoints, order, nfp, stellsym)
        if dofs is None:
            Curve.__init__(self, external_dof_setter=CurveCWSFourier.set_dofs_impl,
                           x0=self.get_dofs())
        else:
            Curve.__init__(self, external_dof_setter=CurveCWSFourier.set_dofs_impl,
                           dofs=dofs)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveCWSFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveCWSFourier.set_dofs(self, dofs)

    def get_dofs_surface(self):
        """
        This function returns the number of the dofs associated with the Coil Winding Surface
        """
        return np.asarray(sopp.CurveCWSFourier.get_dofs_surface(self))

    def num_dofs_surface(self):
        """
        This function returns the number of the dofs associated with the Coil Winding Surface
        """
        return sopp.CurveCWSFourier.num_dofs_surface(self)

    # @classmethod
    # def from_dict(cls, d, serial_objs_dict, recon_objs):
    #     quadpoints = GSONDecoder().process_decoded(d['quadpoints'], serial_objs_dict, recon_objs)
    #     curve = cls(mpol=d["mpol"], ntor=d["ntor"], idofs=d["idofs"], quadpoints=len(quadpoints),
    #                 order=d["order"], nfp=d["nfp"], stellsym=d["stellsym"])
    #     curve.local_full_x = d["x"]
    #     return curve
