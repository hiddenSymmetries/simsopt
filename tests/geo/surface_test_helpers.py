from pathlib import Path

import numpy as np

from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

TEST_DIR = Path(__file__).parent / ".." / "test_files"


def get_surface(surfacetype, stellsym, phis=None, thetas=None, mpol=5, ntor=5,
                nphi=None, ntheta=None, full=False, nfp=3):
    if nphi is None:
        nphi = 11 if surfacetype == "SurfaceXYZTensorFourier" else 15
    if ntheta is None:
        ntheta = 11 if surfacetype == "SurfaceXYZTensorFourier" else 15

    if phis is None:
        phis = np.linspace(0, 1/nfp, nphi, endpoint=False)
    if thetas is None:
        if (surfacetype == "SurfaceXYZTensorFourier" or full == True):
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
