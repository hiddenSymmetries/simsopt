import numpy as np
from ..geo.curve import RotatedCurve
from ..geo.curvexyzfourier import CurveXYZFourier
from ..geo.curvexyzfouriersymmetries import CurveXYZFourierSymmetries

__all__ = ['get_LHD_like_data']

def get_LHD_like_data(numquadpoints_circular=400, numquadpoints_helical=1000):
    """
    Return the coil configurations for the LHD-like configuration, used by Chris
    Smiet and Todd Elder.

    This configuration has 6 vertical field coils and 2 helical coils. It is not
    exactly the LHD configuration, but it is similar.

    The magnetic axis is not yet available.

    Args:
        numquadpoints_circular: number of quadrature points for the circular (vertical field) coils
        numquadpoints_helical: number of quadrature points for helical coils

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

    # Set the shape of the first pair of vertical field coils
    R = 5.55
    Z = 1.55
    curves[0].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
    curves[1].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

    # Set the shape of the second pair of vertical field coils
    R = 2.82
    Z = 2.0
    curves[2].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
    curves[3].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

    # Set the shape of the third pair of vertical field coils
    R = 1.8
    Z = 0.8
    curves[4].x = [0, 0, R, 0, R, 0, +Z, 0, 0]
    curves[5].x = [0, 0, R, 0, R, 0, -Z, 0, 0]

    # Set the shape of the helical coils:
    curves[6].x = [3.850062473963758, 0.9987505207248398, 0.049916705720487310, 0.0012492189452854780, 1.0408856336378722e-05, 0, 0, 0, 0, 0, -1.0408856336392461e-05, 0, 0, -0.9962526034072403, -0.049958346351996670, -0.0012486983723145407, -2.082291883655196e-05, 0, 0]

    currents = [2824400.0, 2824400.0, 682200.0, 682200.0, -2940000.0, -2940000.0, -5400000.0, -5400000.0]

    return curves, currents, None