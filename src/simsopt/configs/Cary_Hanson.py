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

    These configurations have two helical coils, each of which lies on a
    circular-cross-section axisymmetric torus.

    Options for the ``config`` argument are
    ``"1984"``, ``"0.015"``, ``"0.021"``, ``"0.029"``, ``"0.0307"``,
    and ``"Geraldini"``.
    
    
    """
    nfp = 5
    l0 = 2
    R0 = 1
    a = 0.3

    if config == "1984":
        order = 4
    elif config in ["0.0307", "Geraldini"]:
        order = 2
    else:
        order = 3

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
        CurveHelical(nquadpoints, order, nfp, l0, R0, a),
        CurveHelical(nquadpoints, order, nfp, l0, R0, a),
    ]
    factor = 4 * np.pi / mu_0
    currents = [
        (-factor) * Current(current),
        factor * Current(current),
    ]
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    x0 = curves[0].x
    x1 = curves[1].x
    x1[0] = np.pi / 2

    if optimized:
        if config == "1984":
            x0[5] = 0.243960
            x0[6] = 0.026240
            x0[7] = 0.000856

            x1[1] = 0.224859
            x1[3] = -0.000856
            x1[6] = -0.026490

        elif config == "0.0307":
            x0[3] = 0.298089  # B(coil 1, mode m=1)
            x1[1] = 0.298089  # A(coil 2, mode m=1)

        elif config == "Geraldini":
            # Did Alessandro have the two numbers swapped? I can only reproduce
            # the good Poincare plot if I swap his two values.
            x0[3] = 0.3066  # p[1] = B_{-,1}
            x1[1] = 0.3414  # p[0] = A_{+,1}

            x0[3] = 0.3414  # p[1] = B_{-,1}
            x1[1] = 0.3066  # p[0] = A_{+,1}

    curves[0].x = x0
    curves[1].x = x1

    field = BiotSavart(coils) + ToroidalField(1.0, 1.0)

    return coils, field
