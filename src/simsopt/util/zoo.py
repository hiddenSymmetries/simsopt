import numpy as np
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.coil import Current

from pathlib import Path
THIS_DIR = (Path(__file__).parent).resolve()


def get_ncsx_data(Nt_coils=25, Nt_ma=10, ppp=10):
    """
    Get a configuration that corresponds to the modular coils of the NCSX experiment (circular coils are not included).
    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
    """
    filename = THIS_DIR / 'NCSX.dat'
    curves = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
    nfp = 3
    currents = [Current(c) for c in [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]]
    cR = [
        1.471415400740515, 0.1205306261840785, 0.008016125223436036, -0.000508473952304439,
        -0.0003025251710853062, -0.0001587936004797397, 3.223984137937924e-06, 3.524618949869718e-05,
        2.539719080181871e-06, -9.172247073731266e-06, -5.9091166854661e-06, -2.161311017656597e-06,
        -5.160802127332585e-07, -4.640848016990162e-08, 2.649427979914062e-08, 1.501510332041489e-08,
        3.537451979994735e-09, 3.086168230692632e-10, 2.188407398004411e-11, 5.175282424829675e-11,
        1.280947310028369e-11, -1.726293760717645e-11, -1.696747733634374e-11, -7.139212832019126e-12,
        -1.057727690156884e-12, 5.253991686160475e-13]
    sZ = [
        0.06191774986623827, 0.003997436991295509, -0.0001973128955021696, -0.0001892615088404824,
        -2.754694372995494e-05, -1.106933185883972e-05, 9.313743937823742e-06, 9.402864564707521e-06,
        2.353424962024579e-06, -1.910411249403388e-07, -3.699572817752344e-07, -1.691375323357308e-07,
        -5.082041581362814e-08, -8.14564855367364e-09, 1.410153957667715e-09, 1.23357552926813e-09,
        2.484591855376312e-10, -3.803223187770488e-11, -2.909708414424068e-11, -2.009192074867161e-12,
        1.775324360447656e-12, -7.152058893039603e-13, -1.311461207101523e-12, -6.141224681566193e-13,
        -6.897549209312209e-14]

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
    ma.rc[:] = cR[0:(Nt_ma+1)]
    ma.zs[:] = sZ[0:Nt_ma]
    return (curves, currents, ma)
