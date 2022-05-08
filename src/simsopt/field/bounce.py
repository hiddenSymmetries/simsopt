from math import sqrt
import numpy as np
import simsoptpp as sopp
import logging
from simsopt.field.magneticfield import MagneticField
from simsopt.field.sampling import draw_uniform_on_curve, draw_uniform_on_surface
from simsopt.geo.surface import SurfaceClassifier
from simsopt.util.constants import ALPHA_PARTICLE_MASS, ALPHA_PARTICLE_CHARGE, FUSION_ALPHA_PARTICLE_ENERGY
from nptyping import NDArray, Float
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp, quad, quad_explain
import warnings
warnings.catch_warnings()

logger = logging.getLogger(__name__)

def bounce_averaged_jpar(field, s, lam, nfp, nmax, theta0, root_tol=1e-8, tol=1e-8, argmin_tol=1e-3, adjust=False, nzeta=1000):
    """

    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 0, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 1, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = np.asarray(bouncer_try)
    # Check that a consecutive pair of bounce pairs is obtained
    if (len(bouncel_try)>0 and len(bouncer_try)>0):
        bouncel_min = np.min(bouncel_try)
        bouncer_min = np.min(bouncer_try[bouncer_try>bouncel_min])

        # Compute bounce integrals
        integrals = sopp.bounce_integral([bouncel_min],[bouncer_min], field, s, theta0, lam, nfp, True,
                False, False, False, False, False, False, adjust=adjust, tol=tol)

        return integrals[:,0], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find bounce point pair.")

def bounce_averaged_psi_dot(field, s, lam, nfp, nmax, theta0, root_tol=1e-8, tol=1e-8, argmin_tol=1e-3, adjust=False, nzeta=1000):
    """

    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 0, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 1, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = np.asarray(bouncer_try)
    # Check that a consecutive pair of bounce pairs is obtained
    if (len(bouncel_try)>0 and len(bouncer_try)>0):
        bouncel_min = np.min(bouncel_try)
        bouncer_min = np.min(bouncer_try[bouncer_try>bouncel_min])

        # Compute bounce integrals
        integrals = sopp.bounce_integral([bouncel_min],[bouncer_min], field, s, theta0, lam, nfp, False,
                True, False, False, False, False, False, adjust=adjust, tol=tol)

        return integrals[:,1], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find bounce point pair.")

def bounce_averaged_alpha_dot(field, s, lam, nfp, nmax, theta0, root_tol=1e-8, tol=1e-8, argmin_tol=1e-3, adjust=False, nzeta=1000):
    """

    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 0, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 1, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = np.asarray(bouncer_try)
    # Check that a consecutive pair of bounce pairs is obtained
    if (len(bouncel_try)>0 and len(bouncer_try)>0):
        bouncel_min = np.min(bouncel_try)
        bouncer_min = np.min(bouncer_try[bouncer_try>bouncel_min])

        # Compute bounce integrals
        integrals = sopp.bounce_integral([bouncel_min],[bouncer_min], field, s, theta0, lam, nfp, False,
                False, True, False, False, False, False, adjust=adjust, tol=tol)

        return integrals[:,2], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find bounce point pair.")

def bounce_averaged_alpha_dot(field, s, lam, nfp, nmax, theta0, root_tol=1e-8, tol=1e-8, argmin_tol=1e-3, adjust=False, nzeta=1000):
    """

    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 0, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = sopp.find_bounce_points(field, s, theta0, lam, nfp, 1, nmax, root_tol=root_tol, argmin_tol=argmin_tol, nzeta=nzeta)
    bouncer_try = np.asarray(bouncer_try)
    # Check that a consecutive pair of bounce pairs is obtained
    if (len(bouncel_try)>0 and len(bouncer_try)>0):
        bouncel_min = np.min(bouncel_try)
        bouncer_min = np.min(bouncer_try[bouncer_try>bouncel_min])

        # Compute bounce integrals
        integrals = sopp.bounce_integral([bouncel_min],[bouncer_min], field, s, theta0, lam, nfp, False,
                False, True, False, False, False, False, adjust=adjust, tol=tol)

        return integrals[:,2], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find bounce point pair.")

def eps_eff(field, s, nlam, nalpha, nzeta, nfp=1, nmin=1, nmax=10, norm=2,
    digits=4,nmin_tol=1e-3,nstep=1, step_size=1e-3, tol=1e-8):
    """
    nmin: minimum number of toroidal transits
    nmax: maximum number of toroidal transits
    norm: normalization option
    """
    assert norm in [1,2]
    import time

    timem1 = time.time()

    points = np.zeros((1, 3))
    points[:, 0] = s
    field.set_points(points)
    iota = field.iota()[0,0]

    theta0 = np.linspace(0, 2*np.pi, nalpha, endpoint=False)
    zeta = np.linspace(0, 2*np.pi/(nfp), nzeta, endpoint=False)
    [theta02d,zeta2d] = np.meshgrid(theta0,zeta)
    dtheta = theta0[1]-theta0[0]
    dzeta = zeta[1]-zeta[0]

    points = np.zeros((len(theta02d.flatten()), 3))
    points[:, 0] = s
    points[:, 1] = theta02d.flatten()
    points[:, 2] = zeta2d.flatten()
    field.set_points(points)
    modB = field.modB()[:,0]
    minB = np.min(modB)
    maxB = np.max(modB)
    if (norm==2):
        Bref = maxB

    R = field.R()[:, 0]
    dRdtheta = field.dRdtheta()[:, 0]
    dRdzeta = field.dRdzeta()[:, 0]
    dRdpsi = field.dRds()[:, 0]/field.psi0
    dZdtheta = field.dZdtheta()[:, 0]
    dZdzeta = field.dZdzeta()[:, 0]
    dZdpsi = field.dZds()[:, 0]/field.psi0
    nu = field.nu()[:, 0]
    dnudtheta = field.dnudtheta()[:, 0]
    dnudzeta = field.dnudzeta()[:, 0]
    dnudpsi = field.dnuds()[:, 0]/field.psi0

    phi = zeta2d.flatten() - nu
    dphidpsi = - dnudpsi
    dphidtheta = - dnudtheta
    dphidzeta = 1 - dnudzeta
    dXdtheta = dRdtheta * np.cos(phi) - R * np.sin(phi) * dphidtheta
    dYdtheta = dRdtheta * np.sin(phi) + R * np.cos(phi) * dphidtheta
    dXdzeta = dRdzeta * np.cos(phi) - R * np.sin(phi) * dphidzeta
    dYdzeta = dRdzeta * np.sin(phi) + R * np.cos(phi) * dphidzeta

    gthetatheta = dXdtheta**2 + dYdtheta**2 + dZdtheta**2
    gthetazeta = dXdtheta*dXdzeta + dYdtheta*dYdzeta + dZdtheta*dZdzeta
    gzetazeta = dXdzeta**2 + dYdzeta**2 + dZdzeta**2

    sqrtg = (field.G()[:,0]+field.iota()[:,0]*field.I()[:,0])/(modB*modB)
    norm_grad_psi = (gthetatheta*gzetazeta - gthetazeta**2)**(1/2) / sqrtg

    vprime = np.sum(sqrtg)*dtheta*dzeta
    norm_grad_psi_average = np.sum(sqrtg*norm_grad_psi)*dtheta*dzeta/vprime

    lam = np.linspace(1/maxB,1/minB,nlam)
    dlam = (lam[1]-lam[0])*np.ones_like(lam)
    dlam[0] *= 0.5
    dlam[-1] *= 0.5

    time0 = time.time()

    int = 0
    for il in range(len(lam)):
        alpha0 = 0
        time1 = time.time()
        bouncel = 0
        bouncer = bouncel + 2*np.pi*nmin/nfp
        integrals = sopp.bounce_integral(bouncel, bouncer, field, s,
            alpha0, lam[il], nfp, False, False, False, True, False, True,
            False,step_size=step_size,tol=tol)
        I = integrals[:,3]
        dKdalpha = integrals[:,5]
        this_int = np.sum(dKdalpha[I!=0]**2/I[I!=0])*dlam[il]/lam[il]
        ntransits = nmin

        # Now add nstep transits at a time until convergence
        for i in range(nmin+nstep,nmax+1,nstep):
            alpha0 = 2*np.pi*ntransits*iota/nfp
            bouncel = 0
            bouncer = bouncel + 2*np.pi*nstep/nfp
            integrals = sopp.bounce_integral(bouncel, bouncer, field, s,
                alpha0, lam[il], nfp, False, False, False, True, False, True,
                False,step_size=step_size,tol=tol)
            d_I = integrals[:,3]
            d_dKdalpha = integrals[:,5]
            new_int = np.sum(d_dKdalpha[d_I!=0]**2/d_I[d_I!=0])*dlam[il]/lam[il]
            this_int += new_int
            ntransits += nstep
            if (new_int <= nmin_tol*this_int):
                break

        # Normalize by vprime at the correct number of transits
        alpha0 = 0.
        vprime = sopp.vprime(field,s,alpha0,nzeta,nfp,ntransits,step_size,digits)
        int += this_int/vprime

    int *= np.pi/(2*np.sqrt(2)*norm_grad_psi_average**2)
    # Compute normalization factor
    points = np.zeros((len(theta02d.flatten()), 3))
    points[:, 0] = 0
    points[:, 1] = theta02d.flatten()
    points[:, 2] = zeta2d.flatten()
    field.set_points(points)
    Rref = np.mean(field.R())
    if (norm == 1):
        Bref = np.mean(field.B())

    return int*(Bref*Rref)**2

    return indexl, indexr, bounce, zeta
