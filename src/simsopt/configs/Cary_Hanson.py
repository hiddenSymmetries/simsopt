import numpy as np
from scipy.constants import mu_0
from ..geo.curvehelical import CurveHelical
from ..field.biotsavart import BiotSavart
from ..field.coil import Current, Coil
from ..field.magneticfieldclasses import ToroidalField

__all__ = ["get_Cary_Hanson_field"]

def get_Cary_Hanson_field(config, nquadpoints=400, optimized=False):
    """
    Return the coil configurations discussed in the following papers, as
    examples of optimizing coil shapes using Greene's residues to reduce
    magnetic islands and increase the volume of good flux surfaces:

    * `Hanson & Cary, "Elimination of stochasticity in stellarators", Physics of
      Fluids 27, 767 (1984) <https://doi.org/10.1063/1.864692>`__,

    * `Cary & Hanson, "Stochasticity reduction", Physics of Fluids 29, 2464 (1986)
      <https://doi.org/10.1063/1.865539>`__, and

    * `Geraldini, Landreman, & Paul, "An adjoint method for determining the
      sensitivity of island size to magnetic field variations", Journal of Plasma
      Physics 87, 905870302 (2021)
      <https://doi.org/10.1017/S0022377821000428>`__.

    These configurations have two helical coils, each of which lies on a
    circular-cross-section axisymmetric torus. A purely toroidal field is added.
    The helical coils are represented using the :obj:`~simsopt.geo.CurveHelical`
    class.

    Options for the ``config`` argument are ``"1984"``, ``"0.015"``,
    ``"0.021"``, ``"0.029"``, ``"0.0307"``, and ``"Geraldini"``. The setting
    ``"1984"`` corresponds to both the un-optimized and optimized configurations
    from the 1984 paper by Hanson & Cary. The setting ``"Geraldini"``
    corresponds to both the un-optimized and optimized configurations from the
    2021 paper by Geraldini, Landreman, & Paul. The other settings refer to the
    configurations from the 1986 paper by Cary & Hanson.

    Each configuration has an un-optimized and an optimized version. In the
    un-optimized version, the helical coil shapes follow a straight winding law,
    meaning that the geometric poloidal angle is an affine function of the
    standard toroidal angle. In the optimized version, the helical coil shapes
    are modified by adding periodic functions of the toroidal angle to the
    geometric poloidal angle, which reduces the size of magnetic islands and
    increases the volume of good flux surfaces. The un-optimized ``"0.0307"``
    and ``"Geraldini"`` configurations are identical. Also, the un-optimized
    ``"1984"`` and ``"0.021"`` configurations are identical.
    
    Args:
        config (str): Which configuration to return.
        nquadpoints (int): Number of quadrature points for each of the helical curves.
        optimized (bool): If True, use the optimized values for the coefficients
            of the Fourier series. If False, return the unoptimized coil shapes
            with a straight winding law.
    
    Returns:
        A 2-element tuple containing the list of coils and the magnetic field.
        The coils are represented as a list of :obj:`~simsopt.field.Coil` objects, and the
        magnetic field is represented as a :obj:`~simsopt.field.MagneticField` object.
    """
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
        raise ValueError(f"Unknown configuration: {config}")

    nfp = 5
    ell = 2
    R0 = 1
    a = 0.3

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
            # Values from Table 1 in the 1984 paper by Hanson & Cary.
            curves[0].set("B_1", 0.243960)
            curves[0].set("B_2", 0.026240)
            curves[0].set("B_3", 0.000856)

            curves[1].set("A_1", 0.224859)
            curves[1].set("A_3", -0.000856)
            curves[1].set("B_2", -0.026490)

        elif config == "0.0307":
            # Values from Table 1 in the 1986 paper by Cary & Hanson.
            curves[0].set("B_1", 0.298089)  # B(coil 1, mode m=1)
            curves[1].set("A_1", 0.298089)  # A(coil 2, mode m=1)

        elif config == "0.029":
            # Values from Table 1 in the 1986 paper by Cary & Hanson.
            curves[0].set("B_1", 0.144648)  # B(coil 1, mode m=1)
            curves[1].set("A_1", 0.254564)  # A(coil 2, mode m=1)
            curves[0].set("B_2", 0.049151)  # B(coil 1, mode m=2)
            curves[1].set("A_2", -0.035467)  # A(coil 2, mode m=2)

        elif config == "0.021":
            # Values from Table 1 in the 1986 paper by Cary & Hanson.
            curves[0].set("B_1", 0.243960)  # B(coil 1, mode m=1)
            curves[1].set("A_1", 0.224859)  # A(coil 2, mode m=1)
            curves[0].set("B_2", 0.026241)  # B(coil 1, mode m=2)
            curves[1].set("A_2", -0.026490)  # A(coil 2, mode m=2)

        elif config == "0.015":
            # Values from Table 1 in the 1986 paper by Cary & Hanson.
            curves[0].set("B_1", 0.351093)  # B(coil 1, mode m=1)
            curves[1].set("A_1", 0.269967)  # A(coil 2, mode m=1)
            curves[0].set("B_2", 0.030902)  # B(coil 1, mode m=2)
            curves[1].set("A_2", -0.016579)  # A(coil 2, mode m=2)

        elif config == "Geraldini":
            # Values from page 33 of Geraldini's paper.
            # Did Alessandro have the two numbers swapped? I can only reproduce
            # the good Poincare plot if I swap his two values.
            curves[0].set("B_1", 0.3414)  # p[1] = B_{-,1}
            curves[1].set("A_1", 0.3066)  # p[0] = A_{+,1}

    field = BiotSavart(coils) + ToroidalField(1.0, 1.0)

    return coils, field
