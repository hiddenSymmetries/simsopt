from pathlib import Path

import numpy as np

from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier

TEST_DIR = Path(__file__).parent / ".." / "test_files"


def get_surface(surfacetype, stellsym, phis=None, thetas=None, mpol=5, ntor=5,
                nphi=None, ntheta=None, full=False):
    nfp = 3
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


def get_boozer_surface():
    from simsopt.util.zoo import get_ncsx_data
    from simsopt.field.coil import coils_via_symmetries
    from simsopt.field.biotsavart import BiotSavart
    from simsopt.geo.surfaceobjectives import Volume
    
    base_curves, base_currents, ma = get_ncsx_data()
    coils = coils_via_symmetries(base_curves, base_currents, 3, True)
    bs = BiotSavart(coils)
    current_sum = sum(abs(c.current.get_value()) for c in coils)
    G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
    
    ## RESOLUTION DETAILS OF SURFACE ON WHICH WE OPTIMIZE FOR QA
    mpol = 6  
    ntor = 6  
    stellsym = True
    nfp = 3
    
    from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
    from simsopt.geo.boozersurface import BoozerSurface

    phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
    thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
    s = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.fit_to_curve(ma, 0.1, flip_theta=True)
    iota = -0.406
    
    vol = Volume(s)
    vol_target = vol.J()
    
    ## COMPUTE THE SURFACE
    boozer_surface = BoozerSurface(bs, s, vol, vol_target)
    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G0)
    print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}")
    
    return bs, boozer_surface
