import numpy as np
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt.field.coil import Current

from pathlib import Path
THIS_DIR = (Path(__file__).parent).resolve()

__all__ = ['get_ncsx_data', 'get_hsx_data', 'get_giuliani_data']


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
    ma.x = ma.get_dofs()
    return (curves, currents, ma)


def get_hsx_data(Nt_coils=16, Nt_ma=10, ppp=10):
    """
    Get a configuration that corresponds to the modular coils of the HSX experiment.

    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
    """
    filename = THIS_DIR / 'HSX.dat'
    curves = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
    nfp = 4
    currents = [Current(c) for c in [-1.500725500000000e+05, -1.500725500000000e+05, -1.500725500000000e+05, -1.500725500000000e+05, -1.500725500000000e+05, -1.500725500000000e+05]]
    cR = [1.221168734647426701e+00, 2.069298947130969735e-01, 1.819037041932574511e-02, 4.787659822787012774e-05,
          -3.394778038757981920e-05, 4.051690884402789139e-05, 1.066865447680375597e-05, -1.418831703321225589e-05, 
          2.041664078576817539e-05, 2.407340923216046553e-05, -1.281275289727263035e-05, -2.712941403326357315e-05, 
          1.828622086757983125e-06, 1.945955315401206440e-05, 1.409134021563425399e-05, 4.572199318143535127e-06, 
          3.136573559452139703e-07, -3.918158977823491866e-07, -2.204187636324686728e-07, -4.532041599796651056e-08, 
          2.878479243210971143e-08, 2.102768080992785704e-08, -1.267816940685333911e-08, -2.268541399245120326e-08, 
          -8.015316098897114283e-09, 6.401201778979550964e-09]
    sZ = [1.670393448410154857e-01, 1.638250511845155272e-02, 1.656424673977177490e-04, -1.506417857585283353e-04, 
          8.367238367133577161e-05, -1.386982370447437845e-05, -7.536154112897463947e-06, -1.533108076767641072e-05, 
          -9.966838351213697000e-06, 2.561158318745738406e-05, -1.212668371257951164e-06, -1.476513099369021112e-05, 
          -3.716380502156798402e-06, 3.381104573944970371e-06, 2.605458694352088474e-06, 5.701177408478323677e-07, 
          -1.056254779440627595e-07, -1.112799365280694501e-07, -5.381768314066269919e-08, -1.484193645281248712e-08, 
          1.160936870766209295e-08, 1.466392841646290274e-08, 1.531935984912975004e-09, -6.857347022910395347e-09, 
          -4.082678667917087128e-09]

    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
    ma.rc[:] = cR[0:(Nt_ma+1)]
    ma.zs[:] = sZ[0:Nt_ma]
    ma.x = ma.get_dofs()
    return (curves, currents, ma)


def get_giuliani_data(Nt_coils=16, Nt_ma=10, ppp=10, length=18, nsurfaces=5):
    """

    This example simply loads the coils after the nine stage optimization runs discussed in
   
       A. Giuliani, F. Wechsung, M. Landreman, G. Stadler, A. Cerfon, Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasi-symmetry. Journal of Plasma Physics.

    Args:
        Nt_coils: order of the curves representing the coils.
        Nt_ma: order of the curve representing the magnetic axis.
        ppp: point-per-period: number of quadrature points per period

    Returns: 3 element tuple containing the coils, currents, and the magnetic axis.
    """
    assert length in [18, 20, 22, 24]
    assert nsurfaces in [5, 9]

    filename = THIS_DIR / f'GIULIANI_length{length}_nsurfaces{nsurfaces}'
    curves = CurveXYZFourier.load_curves_from_file(filename.with_suffix('.curves'), order=Nt_coils, ppp=ppp)
    currents = [Current(c) for c in np.loadtxt(filename.with_suffix('.currents'))]
    ma_dofs = np.loadtxt(filename.with_suffix('.ma'))
    cR = ma_dofs[:26]
    sZ = ma_dofs[26:]
    nfp = 2
    
    numpoints = Nt_ma*ppp+1 if ((Nt_ma*ppp) % 2 == 0) else Nt_ma*ppp
    ma = CurveRZFourier(numpoints, Nt_ma, nfp, True)
    ma.rc[:] = cR[:(Nt_ma+1)]
    ma.zs[:] = sZ[:Nt_ma]
    ma.x = ma.get_dofs()
    return (curves, currents, ma)

