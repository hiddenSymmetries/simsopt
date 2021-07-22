import numpy as np
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.geo.curverzfourier import CurveRZFourier
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from pathlib import Path
TEST_DIR = (Path(__file__).parent / ".." / "test_files").resolve()


def get_ncsx_data(Nt_coils=25, Nt_ma=10, ppp=10):
    filename = TEST_DIR / 'NCSX_test_data' / 'NCSX_coil_coeffs.dat'
    coils = CurveXYZFourier.load_curves_from_file(filename, order=Nt_coils, ppp=ppp)
    nfp = 3
    currents = [6.52271941985300E+05, 6.51868569367400E+05, 5.37743588647300E+05]
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
    return (coils, currents, ma)


def get_surface(surfacetype, stellsym, phis=None, thetas=None, ntor=5, mpol=5):
    nfp = 3
    nphi = 11 if surfacetype == "SurfaceXYZTensorFourier" else 15
    ntheta = 11 if surfacetype == "SurfaceXYZTensorFourier" else 15

    if phis is None:
        phis = np.linspace(0, 1/nfp, nphi, endpoint=False)
    if thetas is None:
        if surfacetype == "SurfaceXYZTensorFourier":
            thetas = np.linspace(0, 1, ntheta, endpoint=False)
        else:
            thetas = np.linspace(0, 1/(1. + int(stellsym)), ntheta, endpoint=False)

    if surfacetype == "SurfaceXYZFourier":
        s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                              quadpoints_phi=phis, quadpoints_theta=thetas)
    elif surfacetype == "SurfaceRZFourier":
        s = SurfaceRZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                             quadpoints_phi=phis, quadpoints_theta=thetas)
    elif surfacetype == "SurfaceXYZTensorFourier":
        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                                    clamped_dims=[False, False, False],
                                    quadpoints_phi=phis, quadpoints_theta=thetas
                                    )
    else:
        raise Exception("surface type not implemented")
    return s


def get_exact_surface():
    filename_X = TEST_DIR / 'NCSX_test_data'/'X.dat'
    filename_Y = TEST_DIR / 'NCSX_test_data'/'Y.dat'
    filename_Z = TEST_DIR / 'NCSX_test_data'/'Z.dat'
    X = np.loadtxt(filename_X)
    Y = np.loadtxt(filename_Y)
    Z = np.loadtxt(filename_Z)
    xyz = np.concatenate((X[:, :, None], Y[:, :, None], Z[:, :, None]), axis=2)
    ntor = 16
    mpol = 10

    nfp = 1
    stellsym = False
    nphi = 33
    ntheta = 21

    phis = np.linspace(0, 1, nphi, endpoint=False)
    thetas = np.linspace(0, 1, ntheta, endpoint=False)
    s = SurfaceXYZFourier(mpol=mpol, ntor=ntor, nfp=nfp, stellsym=stellsym,
                          quadpoints_phi=phis, quadpoints_theta=thetas)
    s.least_squares_fit(xyz)

    return s
