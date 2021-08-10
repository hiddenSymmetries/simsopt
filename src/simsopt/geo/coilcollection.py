from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.geo.curve import RotatedCurve
from simsopt.field.biotsavart import Coil, ScaledCurrent
from math import pi
import numpy as np


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


def create_equally_spaced_curves(ncurves, nfp, stellsym, R0=1.0, R1=0.5, order=6, PPP=15):
    """
    Create ``ncurves`` curves that will result in equally spaced coils after applying
    ``coils_via_symmetries``. Example that creates 4 base curves, that are then 
    rotated 3 times and flipped for stellarator symmetry:

    .. code-block::

        base_curves = create_equally_spaced_curves(4, 3, stellsym=True)
        base_currents = [Current(1e5) for c in base_curves]
        coils = coils_via_symmetries(base_curves, base_currents, 3, stellsym=True)

    """
    curves = []
    for i in range(ncurves):
        curve = CurveXYZFourier(order*PPP, order)
        d = curve.x
        d[0] = R0
        d[1] = R1
        d[2*(2*order+1)+2] = R1
        curve.x = d
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncoils)
        curve = RotatedCurve(curve, angle, False)
        curves.append(curve)
    return curves
