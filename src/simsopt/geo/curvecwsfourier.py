import numpy as np

import simsoptpp as sopp
from .._core.json import GSONDecoder
from .curve import Curve

__all__ = ['CurveCWSFourier']


class CurveCWSFourier(sopp.CurveCWSFourier, Curve):
    r"""
    ``CurveCWSFourier`` is a curve parameterized to a surface, with the given parameterization:

    .. math::
        \theta(t) &= theta_{l} t + \sum_{m=0}^{\text{order}} theta_{c,m}\cos(m t) + \sum_{m=1}^{\text{order}} theta_{s,m}\sin(m t) \\
        \theta(t) &= phi_{l} t + \sum_{m=0}^{\text{order}} phi_{c,m}\cos(m t) + \sum_{m=1}^{\text{order}} phi_{s,m}\sin(m t) \\
    """

    def __init__(self, mpol, ntor, idofs, numquadpoints, order, nfp, stellsym):
        #numquadpoints = list(np.linspace(0, 1./nfp, numquadpoints, endpoint=False))
        sopp.CurveCWSFourier.__init__(self, mpol, ntor, idofs, numquadpoints, order, nfp, stellsym)
        Curve.__init__(self, external_dof_setter=CurveCWSFourier.set_dofs_impl, x0=self.get_dofs())

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

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        quadpoints = GSONDecoder().process_decoded(d['quadpoints'], serial_objs_dict, recon_objs)
        curve = cls(quadpoints,
                    d["order"],
                    d["nfp"],
                    d["stellsym"])
        curve.local_full_x = d["x0"]
        return curve
