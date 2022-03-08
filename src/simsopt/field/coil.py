from simsopt._core.graph_optimizable import Optimizable
from simsopt._core.derivative import Derivative
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
import simsoptpp as sopp
from math import pi
import numpy as np


class Coil(sopp.Coil, Optimizable):
    """
    A :obj:`Coil` combines a :obj:`~simsopt.geo.curve.Curve` and a
    :obj:`Current` and is used as input for a
    :obj:`~simsopt.field.biotsavart.BiotSavart` field.
    """

    def __init__(self, curve, current):
        self.__curve = curve
        self.__current = current
        sopp.Coil.__init__(self, curve, current)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[curve, current])

    def vjp(self, v_gamma, v_gammadash, v_current):
        return self.curve.dgamma_by_dcoeff_vjp(v_gamma) \
            + self.curve.dgammadash_by_dcoeff_vjp(v_gammadash) \
            + self.current.vjp(v_current)

    def plot(self, **kwargs):
        """
        Plot the coil's curve. This method is just shorthand for calling
        the :obj:`~simsopt.geo.curve.Curve.plot()` function on the
        underlying Curve. All arguments are passed to
        :obj:`simsopt.geo.curve.Curve.plot()`
        """
        return self.curve.plot(**kwargs)


class Current(sopp.Current, Optimizable):
    """
    An optimizable object that wraps around a single scalar degree of
    freedom. It represents the electric current in a coil, or in a set
    of coils that are constrained to use the same current.
    """

    def __init__(self, current):
        sopp.Current.__init__(self, current)
        Optimizable.__init__(self, external_dof_setter=sopp.Current.set_dofs,
                             x0=self.get_dofs())

    def vjp(self, v_current):
        return Derivative({self: v_current})

    def __neg__(self):
        return ScaledCurrent(self, -1.)


class ScaledCurrent(sopp.ScaledCurrent, Optimizable):
    """
    Scales :mod:`Current` by a factor. To be used for example to flip currents
    for stellarator symmetric coils.
    """

    def __init__(self, basecurrent, scale):
        self.__basecurrent = basecurrent
        sopp.ScaledCurrent.__init__(self, basecurrent, scale)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[basecurrent])

    def vjp(self, v_current):
        return self.__basecurrent.vjp(self.scale * v_current)

    def __neg__(self):
        return ScaledCurrent(self, -1.)


def coils_via_symmetries(curves, currents, nfp, stellsym):
    """
    Take a list of ``n`` curves and return ``n * nfp * (1+int(stellsym))``
    ``Coil`` objects obtained by applying rotations and flipping corresponding
    to ``nfp`` fold rotational symmetry and optionally stellarator symmetry.
    """

    assert len(curves) == len(currents)
    flip_list = [False, True] if stellsym else [False]
    coils = []
    for k in range(0, nfp):
        for flip in flip_list:
            for i in range(len(curves)):
                if k == 0 and not flip:
                    coils.append(Coil(curves[i], currents[i]))
                else:
                    rotcurve = RotatedCurve(curves[i], 2*pi*k/nfp, flip)
                    current = ScaledCurrent(currents[i], -1.) if flip else currents[i]
                    coils.append(Coil(rotcurve, current))
    return coils
