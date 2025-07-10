import numpy as np
from ..geo.curve import RotatedCurve
from ..geo.curvexyzfourier import CurveXYZFourier
from ..geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries
from ..geo.curverzfourier import CurveRZFourier

__all__ = ['get_LHD_like_data']

def get_LHD_like_data(numquadpoints_circular=400, numquadpoints_helical=1000, numquadpoints_axis=30):
    """Return the coils and axis for an LHD-like configuration.

    This coil set is a single-filament approximation of the coils in LHD, the
    Large Helical Device in Japan. Each filament corresponds to the center of
    the winding pack of the real finite-thickness LHD coils. The coil currents
    correspond to the configuration in which the major radius of the magnetic
    axis is 3.6 m.

    This configuration has 6 circular coils and 2 helical coils. In the
    lists of curves and currents, the order is OVU, OVL, ISU, ISL, IVU, IVL,
    helical1, helical2. Here, U and L indicate upper and lower, OV and IV indicate
    the outer and inner vertical field coils, and IS indicates the inner shaping
    coils.

    These coils were generated from data generously provided by Yasuhiro Suzuki.
    They produce a configuration similar to that used in Suzuki, Y., K. Y.
    Watanabe, and S. Sakakibara. "Theoretical studies of equilibrium beta limit
    in LHD plasmas." Physics of Plasmas 27, 10 (2020).
    
    Typical usage::

        from simsopt.configs import get_LHD_like_data
        from simsopt.field import BiotSavart, Current, Coil
        
        coils, currents, axis = get_LHD_like_data()
        coils = [
            Coil(curve, Current(current)) for curve, current in zip(coils, currents)
        ]
        field = BiotSavart(coils)

    Args:
        numquadpoints_circular: number of quadrature points for the circular coils
        numquadpoints_helical: number of quadrature points for helical coils
        numquadpoints_axis: number of quadrature points for the magnetic axis

    Returns:
        3 element tuple containing the curves, currents, and the magnetic axis.
    """
    nfp = 5  # Even though LHD has nfp=10 overall, each helical coil by itself has nfp=5.
    order_circular = 1
    order_helical = 6
    stellsym = True
    ntor = 1  # Number of toroidal turns for each helical coil to bite its tail
    curves = [
        CurveXYZFourier(numquadpoints_circular, order_circular),
        CurveXYZFourier(numquadpoints_circular, order_circular),
        CurveXYZFourier(numquadpoints_circular, order_circular),
        CurveXYZFourier(numquadpoints_circular, order_circular),
        CurveXYZFourier(numquadpoints_circular, order_circular),
        CurveXYZFourier(numquadpoints_circular, order_circular),
        CurveXYZFourierSymmetries(numquadpoints_helical, order_helical, nfp, stellsym, ntor),
    ]
    curves.append(RotatedCurve(curves[-1], phi=np.pi / 5, flip=False))

    # Set the shape of the first pair of circular coils, OVU and OVL:
    R = 5.55
    Z = 1.55
    curves[0].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
    curves[1].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

    # Set the shape of the second pair of circular coils, ISU and ISL:
    R = 2.82
    Z = 2.0
    curves[2].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
    curves[3].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

    # Set the shape of the third pair of circular coils, IVU and IVL:
    R = 1.8
    Z = 0.8
    curves[4].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
    curves[5].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

    # Set the shape of the helical coils:
    curves[6].x = [3.850062473963758, 0.9987505207248398, 0.049916705720487310, 0.0012492189452854780, 1.0408856336378722e-05, 0, 0, 0, 0, 0, -1.0408856336392461e-05, 0, 0, -0.9962526034072403, -0.049958346351996670, -0.0012486983723145407, -2.082291883655196e-05, 0, 0]

    currents = [2824400.0, 2824400.0, 682200.0, 682200.0, -2940000.0, -2940000.0, -5400000.0, -5400000.0]

    axis = CurveRZFourier(
        quadpoints=numquadpoints_axis,
        order=6,
        nfp=10,
        stellsym=True,
    )
    axis.x = [
        3.591808210975107,
        0.03794646194915659,
        0.00016372996351568552,
        -3.8324273652135154e-07,
        -7.090559083982798e-09,
        -7.966967131848883e-11,
        -5.308175230062491e-13,
        -0.03663986968740222,
        -0.00016230047363370836,
        4.326127544845136e-07,
        1.1123540323857856e-08,
        6.833905523642707e-11,
        4.612346787214785e-13,
    ]

    return curves, currents, axis