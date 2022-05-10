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
import math
warnings.catch_warnings()

logger = logging.getLogger(__name__)

def bounce_averaged_jpar(field, s, lam, nfp, nmax, alpha0, root_tol=1e-8,
        tol=1e-8, nzeta=1000, step_size=1e-3):
    """
    Computes the parallel adiabatic invariant (divided by v)

    .. math:

        \oint dl \, \sqrt(1 - \lambda B)

    where integration is performed along a field line and :math:`\lambda = v_{\perp}^2/(v^2 B)`
    and integration is performed along a field line between bounce points where
    :math:`\lambda B = 1`.

    field: The :class:`BoozerMagneticField` instance
    s (float): normalized toroidal flux for evaluation
    lam (double): value of lamda = vperp^2/(v^2*B) (between 1/Bmin and 1/Bmax)
    nfp (int): number of field periods
    nmax (int): maximum number of toroidal field periods between left and right
        bounce points and number of toroidal field periods for which left bounce
        points will be evaluated
    alpha0 (double): field line label alpha0 = theta - iota*zeta where bounce integral evaluated
    root_tol (float): tolerance parameter for left bounce point solve. If this
        parameter is too small, root solve will fail and integral segment will be skipped.
    tol (float): tolerance parameter for integration along a field line
    nzeta (int): number of gridpoints in toroidal angle for detection of
        left bounce points. If this number is too small, then bounce point may
        not be detected.
    step_size (float): step size in toroidal angle where the observer is called to determine
        if a right bounce point has been obtained during field line integration.
        If this parameter is too large, the right bounce may be stepped over.
    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, alpha0, lam, nfp, 0, nmax, root_tol=root_tol, nzeta=nzeta)
    bouncel_try = np.asarray(bouncel_try)
    if (len(bouncel_try)>0):
        bouncer_try = bouncel_try + 2*np.pi*nmax/nfp

        # Compute bounce integrals
        integrals = sopp.bounce_integral(bouncel_try,bouncer_try, field, s, alpha0, lam, nfp, True,
                False, False, False, False, False, False, adjust=True, tol=tol, step_size=step_size)

        return integrals[:,0], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find left bounce point.")

def bounce_averaged_psi_dot(field, s, lam, nfp, nmax, alpha0, root_tol=1e-8,
        tol=1e-8, nzeta=1000, step_size=1e-3):
    """
    Computes the bounce integral of the radial magnetic drift (divided by m v/q)

    .. math:

        \oint dl \, \sqrt{g}\left((2 - \lamda B)(I*dB/d\zeta-G*dB/d\theta)/(\sqrt(1 - \lamda B)2B)\right)

    where integration is performed along a field line and :math:`\lambda = v_{\perp}^2/(v^2 B)`
    and integration is performed along a field line between bounce points where
    :math:`\lambda B = 1`.

    field: The :class:`BoozerMagneticField` instance
    s (float): normalized toroidal flux for evaluation
    lam (double): value of lamda = vperp^2/(v^2*B) (between 1/Bmin and 1/Bmax)
    nfp (int): number of field periods
    nmax (int): maximum number of toroidal field periods between left and right
        bounce points and number of toroidal field periods for which left bounce
        points will be evaluated
    alpha0 (double): field line label alpha0 = theta - iota*zeta where bounce integral evaluated
    root_tol (float): tolerance parameter for left bounce point solve. If this
        parameter is too small, root solve will fail and integral segment will be skipped.
    tol (float): tolerance parameter for integration along a field line
    nzeta (int): number of gridpoints in toroidal angle for detection of
        left bounce points. If this number is too small, then bounce point may
        not be detected.
    step_size (float): step size in toroidal angle where the observer is called to determine
        if a right bounce point has been obtained during field line integration.
        If this parameter is too large, the right bounce may be stepped over.
    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, alpha0, lam, nfp, 0, nmax, root_tol=root_tol, nzeta=nzeta)
    bouncel_try = np.asarray(bouncel_try)
    if (len(bouncel_try)>0):
        bouncer_try = bouncel_try + 2*np.pi*nmax/nfp

        # Compute bounce integrals
        integrals = sopp.bounce_integral(bouncel_try,bouncer_try, field, s, alpha0, lam, nfp, False,
                True, False, False, False, False, False, adjust=True, tol=tol, step_size=step_size)

        return integrals[:,1], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find bounce point pair.")

def bounce_averaged_alpha_dot(field, s, lam, nfp, nmax, alpha0, root_tol=1e-8,
        tol=1e-8, nzeta=1000, step_size=1e-3):
    """
    Computes the bounce integral of the alpha magnetic drift (divided by m v/q)

    .. math:
        \oint dl \, \sqrt{g}\left( (2 - \lambda B) A1/(\sqrt(1 - \lambda B)2B) \
            + \sqrt(1 - \lambda B) A2) \right)

    where

    .. math:
        A1 = K (-\iota dB/d\theta - dB/d\zeta)
            + I (-d\iota/d\psi\zeta dB/d\zeta + \iota dB/d\psi)
            + G (dB/d\psi + \zeta d\iota/d\psi  dB/d\theta)
        A2 = -\iota dI/d\psi - dG/d\psi + dK/d\zeta + \iota dK/d\theta

    and integration is performed along a field line and :math:`\lambda = v_{\perp}^2/(v^2 B)`
    and integration is performed along a field line between bounce points where
    :math:`\lambda B = 1`.

    field: The :class:`BoozerMagneticField` instance
    s (float): normalized toroidal flux for evaluation
    lam (double): value of lamda = vperp^2/(v^2*B) (between 1/Bmin and 1/Bmax)
    nfp (int): number of field periods
    nmax (int): maximum number of toroidal field periods between left and right
        bounce points and number of toroidal field periods for which left bounce
        points will be evaluated
    alpha0 (double): field line label alpha0 = theta - iota*zeta where bounce integral evaluated
    root_tol (float): tolerance parameter for left bounce point solve. If this
        parameter is too small, root solve will fail and integral segment will be skipped.
    tol (float): tolerance parameter for integration along a field line
    nzeta (int): number of gridpoints in toroidal angle for detection of
        left bounce points. If this number is too small, then bounce point may
        not be detected.
    step_size (float): step size in toroidal angle where the observer is called to determine
        if a right bounce point has been obtained during field line integration.
        If this parameter is too large, the right bounce may be stepped over.
    """
    # Compute bounce points
    bouncel_try = sopp.find_bounce_points(field, s, alpha0, lam, nfp, 0, nmax, root_tol=root_tol, nzeta=nzeta)
    bouncel_try = np.asarray(bouncel_try)
    if (len(bouncel_try)>0):
        bouncer_try = bouncel_try + 2*np.pi*nmax/nfp

        # Compute bounce integrals
        integrals = sopp.bounce_integral(bouncel_try,bouncer_try, field, s, alpha0, lam, nfp, False,
                False, True, False, False, False, False, adjust=True, tol=tol, step_size=step_size)

        return integrals[:,2], integrals[:,7], integrals[:,8]
    else:
        raise RuntimeError("Unable to find bounce point pair.")

def eps_eff(field, s, nfp, nlam=30, ntheta=100, nzeta=100, nmin=100, nmax=1000,
     nstep=100, nmin_tol=1e-3, step_size=1e-3, tol=1e-8, nmax_bounce=5, root_tol=1e-8,
     nzeta_bounce=1000, norm=2):
    """
    Performs calculation of the effective ripple on a prescribe surface of a
    given :class:`BoozerMagneticField` instance.

    .. math::

        \eps_{\text{eff}}^{3/2} = \frac{\pi (B_{ref}R_{ref})^2}{2\sqrt{2}\langle|\nabla \psi|\rangle^2}\sum_i \int_{1/B_{\max}}^{1/B_{\min}} \frac{\left(dK_i/d\alpha\right)^2}{I_i V_i \lambda} \, d\lambda
        R_{ref} = \frac{\int_0^{2\pi} \int_0^{2\pi} R \, d \theta d\zeta}{\int_0^{2\pi} \int_0^{2\pi} \, d \theta d\zeta}

    See V. V. Nemov et al., Phys. Plasmas 6, 4622 (1999). http://dx.doi.org/10.1063/1.873749

    Here the summation is taken over wells along a field line. For norm = 1:

    ... math::

        B_{ref} = \frac{\int_0^{2\pi} \int_0^{2\pi} B \, d \theta d\zeta}{\int_0^{2\pi} \int_0^{2\pi} \, d \theta d\zeta}

    and for norm = 2, B_ref is taken to be the maximum over the surface

    field: The :class:`BoozerMagneticField` instance
    s (float): normalized toroidal flux for evaluation
    nfp (int): number of field periods
    nlam (int): number of gridpoints in lamda = vperp^2/(v^2*B) between 1/Bmin and 1/Bmax
    ntheta (int): number of poloidal angle grid points for performing surface integrals in normalization factors
    nzeta (int): number of toroidal angle grid points for performing surface integrals in normalization factors
    nmin (int): minimum number of field period transits for field line integration
    nmax (int): maximum number of field period transits for field line integration
    nstep (int): number of field period transits is increased from nmin to nmax by increments of
        nstep until the nmin_tol parameter is achieved
    nmin_tol (float): tolerance parameter for number of toroidal transits. Integration along field line
        is terminated when the relative change in the integral is less than nmin_tol
    step_size (float): step size in toroidal angle where the observer is called to determine
        if a right bounce point has been obtained during field line integration.
        If this parameter is too large, the right bounce may be stepped over.
    tol (float): tolerance parameter for integration along a field line
    nmax_bounce (int): maximum number of toroidal field periods between left and right
        bounce points
    nzeta_bounce (int): number of gridpoints in toroidal angle for detection of
        left bounce points. If this number is too small, then bounce point may
        not be detected.
    root_tol (float): tolerance parameter for left bounce point solve. If this
        parameter is too small, root solve will fail and integral segment will be skipped.
    norm (int): normalization option. Must be 1 or 2. If = 1, then normalizing field
        strength is taken to be the average over the Boozer angles. If = 2, then
        normalizing field strength is taken to be the maximum on the surface.
    """
    assert norm in [1,2]

    points = np.zeros((1, 3))
    points[:, 0] = s
    field.set_points(points)
    iota = field.iota()[0,0]

    theta0 = np.linspace(0, 2*np.pi, ntheta, endpoint=False)
    zeta = np.linspace(0, 2*np.pi/nfp, nzeta, endpoint=False)
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

    norm_grad_psi_average = np.sum(sqrtg*norm_grad_psi)/np.sum(sqrtg)

    lam = np.linspace(1/maxB,1/minB,nlam)
    dlam = (lam[1]-lam[0])*np.ones_like(lam)
    dlam[0] *= 0.5
    dlam[-1] *= 0.5

    int = 0
    for il in range(len(lam)):
        alpha0 = 0
        bouncel = sopp.find_bounce_points(field, s, alpha0, lam[il], nfp, 0, nmin,
            root_tol=root_tol, nzeta=nzeta_bounce)
        if (len(bouncel)==0):
            continue
        bouncel = np.asarray(bouncel)
        bouncer = bouncel + 2*np.pi*nmax_bounce/nfp

        integrals = sopp.bounce_integral(bouncel, bouncer, field, s,
            alpha0, lam[il], nfp, False, False, False, True, False, True,
            False,step_size=step_size,tol=tol,adjust=True)
        I = integrals[:,3]
        dKdalpha = integrals[:,5]
        this_int = np.sum(dKdalpha[I!=0]**2/I[I!=0])*dlam[il]/lam[il]
        ntransits = nmin

        # Now add nstep transits at a time until convergence
        for i in range(nmin+nstep,nmax+1,nstep):
            alpha0 = 2*np.pi*ntransits*iota/nfp
            ntransits += nstep

            bouncel = sopp.find_bounce_points(field, s, alpha0, lam[il], nfp, 0, nmin,
                root_tol=root_tol, nzeta=nzeta_bounce)
            if (len(bouncel)==0):
                continue
            bouncel = np.asarray(bouncel)
            bouncer = bouncel + 2*np.pi*nmax_bounce/nfp
            integrals = sopp.bounce_integral(bouncel, bouncer, field, s,
                alpha0, lam[il], nfp, False, False, False, True, False, True,
                False,step_size=step_size,tol=tol,adjust=True)

            d_I = integrals[:,3]
            d_dKdalpha = integrals[:,5]
            new_int = np.sum(d_dKdalpha[d_I!=0]**2/d_I[d_I!=0])*dlam[il]/lam[il]
            this_int += new_int
            if (new_int <= nmin_tol*this_int):
                break

        # Normalize by vprime at the correct number of transits
        alpha0 = 0.
        vprime = sopp.vprime(field,s,alpha0,nfp,ntransits,step_size)
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
        Bref = np.mean(field.modB())

    return int*(Bref*Rref)**2

    return indexl, indexr, bounce, zeta
