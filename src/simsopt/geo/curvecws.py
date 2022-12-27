import numpy as np

import simsoptpp as sopp
from .._core.json import GSONDecoder
from .curve import Curve

__all__ = ['CurveCWS']


class CurveCWS(sopp.CurveCWS, Curve):
    r"""
    ``CurveCWS`` is a curve parameterized to a surface, with the given parameterization:

    .. math::
        \theta(t) &= theta_{l} t + \sum_{m=0}^{\text{order}} theta_{c,m}\cos(m t) + \sum_{m=1}^{\text{order}} theta_{s,m}\sin(m t) \\
        \theta(t) &= phi_{l} t + \sum_{m=0}^{\text{order}} phi_{c,m}\cos(m t) + \sum_{m=1}^{\text{order}} phi_{s,m}\sin(m t) \\
    """

    def __init__(self, quadpoints, order, nfp, stellsym):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1./nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveCWSFourier.__init__(self, quadpoints, order, nfp, stellsym)
        Curve.__init__(self, external_dof_setter=CurveCWS.set_dofs_impl,
                       x0=self.get_dofs())

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveCWS.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveCWS.set_dofs(self, dofs)

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        quadpoints = GSONDecoder().process_decoded(d['quadpoints'], serial_objs_dict, recon_objs)
        curve = cls(quadpoints,
                    d["order"],
                    d["nfp"],
                    d["stellsym"])
        curve.local_full_x = d["x0"]
        return curve
