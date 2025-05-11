import numpy as np
from scipy.constants import mu_0
from ..geo.curvehelical import CurveHelical
from ..field.biotsavart import BiotSavart
from ..field.coil import Current, Coil
from ..field.magneticfieldclasses import ToroidalField

__all__ = ["get_Cary_Hanson_field"]

def get_Cary_Hanson_field(config, nquadpoints=400, optimized=False):
    """
    Return the coil configurations discussed in the following papers:

    `Hanson & Cary (1984) <https://doi.org/10.1063/1.864692>`__
    and
    `Cary & Hanson (1986) <https://doi.org/10.1063/1.865539>`__

    These configurations have two helical coils, each of which lies on a
    circular-cross-section axisymmetric torus. A purely toroidal field is added.

    Options for the ``config`` argument are
    ``"1984"``, ``"0.015"``, ``"0.021"``, ``"0.029"``, ``"0.0307"``,
    and ``"Geraldini"``.
    
    
    """
    nfp = 5
    ell = 2
    R0 = 1
    a = 0.3

    if config == "1984":
        order = 3
    elif config in ["0.0307", "Geraldini"]:
        order = 1
    else:
        order = 2

    if config in ["1984", "0.021"]:
        current = 0.021
    elif config == "0.029":
        current = 0.029
    elif config in ["0.0307", "Geraldini"]:
        current = 0.0307
    elif config == "0.015":
        current = 0.015
    else:
        raise ValueError(f"Unknown config: {config}")

    curves = [
        CurveHelical(nquadpoints, order, nfp, ell, R0, a),
        CurveHelical(nquadpoints, order, nfp, ell, R0, a),
    ]
    factor = 4 * np.pi / mu_0
    currents = [
        (-factor) * Current(current),
        factor * Current(current),
    ]
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    curves[1].set("A_0", np.pi / 2)

    if optimized:
        if config == "1984":
            # Values from Table 1
            curves[0].set("B_1", 0.243960)
            curves[0].set("B_2", 0.026240)
            curves[0].set("B_3", 0.000856)

            curves[1].set("A_1", 0.224859)
            curves[1].set("A_3", -0.000856)
            curves[1].set("B_2", -0.026490)

        elif config == "0.0307":
            curves[0].set("B_1", 0.298089)  # B(coil 1, mode m=1)
            curves[1].set("A_1", 0.298089)  # A(coil 2, mode m=1)

        elif config == "Geraldini":
            # Values from page 33.
            # Did Alessandro have the two numbers swapped? I can only reproduce
            # the good Poincare plot if I swap his two values.
            curves[0].set("B_1", 0.3414)  # p[1] = B_{-,1}
            curves[1].set("A_1", 0.3066)  # p[0] = A_{+,1}

    field = BiotSavart(coils) + ToroidalField(1.0, 1.0)

    return coils, field
