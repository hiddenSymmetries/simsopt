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

def find_sigma_minus(field, s, ngrid, nfp, rtol=1e-3, tmax = 100, toroidal=False, theta0=0, zeta0=0):
    """
    This assumes that curves of b dot grad B = 0 close poloidally.
    1. First find points on theta = theta0 curves that satisfy b dot grad B = 0. Each
       of these correspond with an initial condition for a curve corresponding to
       the intersection of Sigma^- with this flux surface.
    2. Now integrate with respect to theta to obtain zeta(theta) such that
       bdotgradB = 0.
    """
    points = np.zeros((ngrid,3))
    points[:,0] = s
    # Contours close toroidally
    if (toroidal):
        thetas = np.linspace(0,2*np.pi,ngrid)
        points[:,1] = thetas
        points[:,2] = zeta0
    else:
        zetas = np.linspace(0,2*np.pi/nfp,ngrid)
        points[:,1] = thet0
        points[:,2] = zetas
    field.set_points(points)

    bdotgradB = (field.iota()[:,0]*field.dmodBdtheta()[:,0] + field.dmodBdzeta()[:,0])/field.modB()[:,0]

    # Find points in zeta that bracket b dot grad B = 0.
    points_left = []
    points_right = []
    for i in range(ngrid-1):
        # Find points that bracket bdotgradB = 0
        if ((bdotgradB[i] > 0) == (bdotgradB[i+1] < 0)):
            if toroidal:
                points_left.append(thetas[i])
                points_right.append(thetas[i+1])
            else:
                points_left.append(zetas[i])
                points_right.append(zetas[i+1])

    point = np.zeros((1,3))
    point[:,0] = s
    def bdotgradBf(param):
        if (toroidal):
            point[:,1] = param
            point[:,2] = zeta0
        else:
            point[:,1] = theta0
            point[:,2] = param
        field.set_points(point)
        return field.iota()[0,0]*field.dmodBdtheta()[0,0] + field.dmodBdzeta()[0,0]

    def rhs(t,y):
        point[:,1] = y[0]
        point[:,2] = y[1]
        field.set_points(point)
        d2modBdtheta2 = field.d2modBdtheta2()[0,0]
        d2modBdzeta2 = field.d2modBdzeta2()[0,0]
        d2modBdthetadzeta = field.d2modBdthetadzeta()[0,0]
        iota = field.iota()[0,0]
        G = field.G()[0,0]
        I = field.I()[0,0]
        modB = field.modB()[0,0]
        sqrtginv = modB*modB/(G+iota*I)
        dBdotgradBdtheta = iota*d2modBdtheta2 + d2modBdthetadzeta
        dBdotgradBdzeta = iota*d2modBdthetadzeta + d2modBdzeta2
        dthetadt = dBdotgradBdzeta/np.sqrt(dBdotgradBdzeta**2 + dBdotgradBdtheta**2)
        dzetadt = -dBdotgradBdtheta/np.sqrt(dBdotgradBdzeta**2 + dBdotgradBdtheta**2)
        return [dthetadt,dzetadt]

    def event1(t, y):
        theta = y[0]
        zeta = y[1]
        if t>0:
            if (theta-theta0 > 0):
                # Cross in positive direction
                return   theta-theta0 - 2*np.pi
            elif (theta-theta0 < 0):
                return -(theta-theta0) - 2*np.pi
            else:
                return 1
        else:
            return 1

    def event2(t, y):
        theta = y[0]
        zeta = y[1]
        if (t>0):
            if (zeta - sol.root > 0):
                return  zeta-sol.root - 2*np.pi/nfp
            elif (zeta-sol.root < 0):
                return -(zeta-sol.root) - 2*np.pi/nfp
            else:
                return 1
        else:
            return 1

    event1.terminal = True
    event1.direction = +1
    event2.terminal = True
    event2.direction = +1

    point = np.zeros((1,3))
    point[:,0] = s

    # Now iterate over possible brackets and perform a root solve
    t_span = (0,tmax)
    sigmaminus_spline = []
    sigmaminus = []
    t_events = []
    y = np.zeros((2,))
    for i in range(len(points_left)):
        print('Point left: ', i)
        sol = root_scalar(bdotgradBf, x0=points_left[i], x1=points_right[i], rtol=rtol)
        print('Converged: ',sol.converged)
        print('root: ', sol.root)

        if (toroidal):
            point[:,1] = sol.root
            point[:,2] = zeta0
        else:
            point[:,1] = theta0
            point[:,2] = sol.root
        field.set_points(point)
        bdotgradB = field.iota()[:,0] * field.dmodBdtheta()[:,0] + field.dmodBdzeta()[:,0]
        print('BdotgradB: ',bdotgradB)
        bdotgrad2B = field.iota()[:,0]**2 * field.d2modBdtheta2()[:,0] + field.d2modBdzeta2()[:,0] \
            + 2 * field.iota()[:,0] * field.d2modBdthetadzeta()[:,0]
        print('bdotgrad2B: ',bdotgrad2B)
        modB = field.modB()[0,0]

        if (toroidal):
            point[:,1] = points_left[i]
            point[:,2] = zeta0
        else:
            point[:,1] = theta0
            point[:,2] = points_left[i]
        field.set_points(point)
        modBleft = field.modB()[0,0]
        print('modB: ', modB)
        print('modBleft: ',modBleft)
        if (bdotgrad2B < 0 or modB > modBleft):
            y[0:1] = point[0,1:2]
            sol = solve_ivp(rhs, t_span, y, dense_output=True, rtol=rtol, atol=rtol, events=[event1,event2])
            # print(np.shape(sol.t))
            # print(np.shape(sol.y))
            print(sol.message)

            output = np.zeros((len(sol.t),3))
            output[:,0] = sol.t
            output[:,1] = sol.y[0]
            output[:,2] = sol.y[1]
            sigmaminus_spline.append(sol.sol)
            sigmaminus.append(output)
            t_events.append(sol.t_events)

    return sigmaminus_spline, sigmaminus, t_events

def gamma_c(field, s, nlam, nalpha, nzeta, nfp=1, nmax=10, limit=50, eps=1e-8):

    theta0 = np.linspace(0, 2*np.pi, nalpha, endpoint=False)
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
        Bnorm = maxB

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

    import time
    int = 0
    for il in range(len(lam)):
        for ia in range(len(theta0)):
            time1 = time.time()
            _, _, psidot, _ = bounce_integral(field, s, theta0[ia], nzeta, lam[il], mode='psidot', nfp=nfp, nmax=nmax, limit=limit, eps=eps)
            time2 = time.time()
            print('Time for psidot: ',time2-time1)
            _, _, alphadot, _ = bounce_integral(field, s, theta0[ia], nzeta, lam[il], mode='alphadot', nfp=nfp, nmax=nmax, limit=limit, eps=eps)
            time3 = time.time()
            print('Time for alphadot: ',time3-time2)
            _, _, tau, _ = bounce_integral(field, s, theta0[ia], nzeta, lam[il], mode='tau', nfp=nfp, nmax=nmax, limit=limit, eps=eps)
            time4 = time.time()
            print('Time for tau: ',time4-time3)
            psidot = np.asarray(psidot)
            alphadot = np.asarray(alphadot)
            tau = np.asarray(tau)
            gammac = (2/np.pi)*np.arctan2(psidot,np.abs(alphadot))
            int += np.sum(tau*gamma_c**2)*dtheta*dlam[il]

    int *= np.pi/(16*np.sqrt(2)*vprime)

def eps_eff(field, s, nlam, nalpha, nzeta, nfp=1, nmin=1, nmax=10, ntransitmax=1, limit=50, norm=2, eps=1e-8, maxp1=50,
    step_size=1e-3,digits=4,tol=1e-3,nmin_tol=1e-3,nstep=1):
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
    # print('Time before loop: ', time0-timem1)

    int = 0
    for il in range(len(lam)):
        # for i in range(nalpha):
            # alpha0 = i*2*np.pi*iota
        # for ia in range(len(theta0)):
        # alpha0 = theta0[ia]
        alpha0 = 0
        time1 = time.time()
        integrals = sopp.bounce_integral(field, s,
            alpha0, nzeta, lam[il], nfp, nmin, ntransitmax, False, False, False, True, False, True,
            False,step_size,digits,tol)
        I = integrals[:,3]
        # /(2*np.pi*nmax/nfp)
        dKdalpha = integrals[:,5]
        this_int = np.sum(dKdalpha[I!=0]**2/I[I!=0])*dlam[il]/lam[il]
        ntransits = nmin

        # Now add nstep transits at a time until convergence
        for i in range(nmin+nstep,nmax+1,nstep):
            alpha0 = 2*np.pi*ntransits/nfp
            integrals = sopp.bounce_integral(field, s,
                alpha0, nzeta, lam[il], nfp, nstep, ntransitmax, False, False, False, True, False, True,
                False,step_size,digits,tol)
            d_I = integrals[:,3]
            d_dKdalpha = integrals[:,5]
            new_int = np.sum(d_dKdalpha[d_I!=0]**2/d_I[d_I!=0])*dlam[il]/lam[il]
            this_int += new_int
            ntransits += nstep
            if (new_int <= nmin_tol*this_int):
                # print('Terminated at i = ',i)
                break

        # Normalize by vprime at the correct number of transits
        alpha0 = 0.
        vprime = sopp.vprime(field,s,alpha0,nzeta,nfp,ntransits,step_size,digits)
        int += this_int/vprime

        # /(2*np.pi*nmax/nfp)
        # indexl1, indexr1, dKdalpha, _ = bounce_integral(field, s, alpha0, nzeta,
        #     lam[il], mode='dkhatdalpha', nfp=nfp, nmax=nmax, limit=limit, eps=eps, maxp1=maxp1)
        # time2 = time.time()
        # print('Time for dKdalpha: ', )
        # indexl2, indexr2, I, _ = bounce_integral(field, s, alpha0, nzeta, lam[il],
        #     mode='ihat', nfp=nfp, nmax=nmax, limit=limit, eps=eps, maxp1=maxp1)
        # dKdalpha = np.asarray(dKdalpha)
        # I = np.asarray(I)
        # time3 = time.time()
        # indexl1 = np.asarray(indexl1)
        # indexl2 = np.asarray(indexl2)
        # indexr1 = np.asarray(indexr1)
        # indexr2 = np.asarray(indexr2)

        # dKdalpha = np.asarray(dKdalpha)
        # assert(len(dKdalpha)==len(I))
        # assert(np.allclose(indexl1,indexl2))
        # assert(np.allclose(indexr1,indexr2))
        # for i in range(1,len(indexl1)):
        #     assert(indexr1[i-1]<indexl1[i])
        # if (np.any(I == 0)):
        #     print('indexl: ', indexl1)
        #     print('indexr: ', indexr1)
        #     print('I = 0')
        #     print('dKdalpha: ', dKdalpha)
        #     print('lam: ',lam[il])
        #     print('theta0: ',theta0[ia])
        #     print('s: ',s)

        # print('Time for I: ',time3-time2)
        # int += np.sum(dKdalpha**2/I)*dtheta*dlam[il]/lam[il]
        # print('dKdalpha: ', dKdalpha)
        # print('I: ',I)
        # print('il: ', il)

        # for in in range(nmin+1,nmax):
        #             alpha0 = 0
        #             time1 = time.time()
        #             integrals = sopp.bounce_integral(field, s,
        #                 alpha0, nzeta, lam[il], nfp, nmin, False, False, False, True, False, True,
        #                 False,step_size,digits,tol)
        #             I = integrals[:,3]


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

def bounce_integral(field, s, theta0, nzeta, lam, mode='jpar', nfp=1, nmax=10,
        limit=50, eps=1e-8, maxp1=50):
    """
    nmax: maximum number of toroidal periods to integrate
    type: What kind of bounce integral is performed. Can be 'Jpar', 'psidot', 'alphadot'
    """
    mode = mode.lower()
    assert mode in ['jpar', 'psidot', 'alphadot', 'ihat', 'khat','dkhatdalpha','tau']

    # Compute flux functions on this surface
    points = np.zeros((1, 3))
    points[:, 0] = s
    points[:, 1] = 0
    points[:, 2] = 0
    field.set_points(points)
    iota = field.iota()[0, 0]
    jacfac = (field.G() + field.iota()*field.I())[0, 0]
    iota = field.iota()[0,0]

    # Construct grids
    zeta = np.linspace(0, 2*np.pi*nmax/(nfp), nzeta, endpoint=True)

    # Compute modB on this grid
    points = np.zeros((len(zeta), 3))
    points[:, 0] = s
    points[:, 1] = theta0 + iota * zeta
    points[:, 2] = zeta
    field.set_points(points)
    modB = field.modB().reshape(np.shape(zeta))

    # For each lambda and theta use modB grid to find potential bounce points in zeta
    points = np.zeros((1, 3))
    points[:, 0] = s

    def modBf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.modB()[0, 0]

    def dmodBdthetaf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dmodBdtheta()[0, 0]

    def dmodBdzetaf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dmodBdzeta()[0, 0]

    def dmodBdpsif(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dmodBds()[0, 0]/field.psi0

    def If(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.I()[0, 0]

    def Gf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.G()[0, 0]

    def diotadpsif(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.diotads()[0, 0]/field.psi0

    def dIdpsif(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dIds()[0, 0]/field.psi0

    def dGdpsif(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dGds()[0, 0]/field.psi0

    def Kf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.K()[0, 0]

    def dKdthetaf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dKdtheta()[0, 0]

    def dKdzetaf(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dKdzeta()[0, 0]

    def dKdpsif(zeta):
        points[:, 1] = theta0 + iota * zeta
        points[:, 2] = zeta
        field.set_points(points)
        return field.dKds()[0, 0]/field.psi0

    def integrand(zeta, state):
        if (lam*modBf(zeta) > 1):
            int = 0
        else:
            int = 1
        return int

    def integrand_quad(zeta, zetal, zetar):
        if (lam*modBf(zeta) > 1):
            int = 0
        # elif (modBf(zeta) == 0):
        #     int = 0
        else:
            if (mode == 'jpar'):
                int = np.sqrt(1 - lam*modBf(zeta))*jacfac/modBf(zeta)
            elif (mode == 'psidot'): # singular
                modB = modBf(zeta)
                dmodBdtheta = dmodBdthetaf(zeta)
                dmodBdzeta = dmodBdzetaf(zeta)
                I = If(zeta, theta0)
                G = Gf(zeta, theta0)
                int = (2 - lam*modB)*(I*dmodBdzeta-G*dmodBdtheta)/(np.sqrt(1 - lam*modB)*2*modB*modB)
            elif (mode == 'alphadot'): # singular
                modB = modBf(zeta)
                dmodBdtheta = dmodBdthetaf(zeta)
                dmodBdzeta = dmodBdzetaf(zeta)
                dmodBdpsi = dmodBdpsif(zeta)
                I = If(zeta)
                G = Gf(zeta)
                dIdpsi = dIdpsif(zeta)
                dGdpsi = dGdpsif(zeta)
                diotadpsi = diotadpsif(zeta)
                K = Kf(zeta)
                dKdtheta = dKdthetaf(zeta)
                dKdzeta = dKdzetaf(zeta)
                fac1 = K * (-iota*dmodBdtheta - dmodBdzeta) \
                    + I * (-diotadpsi*zeta*dmodBdzeta + iota*dmodBdpsi) \
                    + G * (dmodBdpsi + diotadpsi*zeta*dmodBdtheta)
                fac2 = -dIdpsi*iota - dGdpsi + dKdzeta + iota*dKdtheta
                int = (2 - lam*modB)*fac1/(np.sqrt(1 - lam*modB)*2*modB*modB) \
                    + np.sqrt(1 - lam*modB)*fac2/modB
            elif (mode == 'ihat'):
                modB = modBf(zeta)
                if (zeta == zetal):
                    int = (1/(modB*np.sqrt(np.abs(zeta-zetar))))*jacfac/modB
                elif (zeta == zetar):
                    int = (1/(modB*np.sqrt(np.abs(zeta-zetal))))*jacfac/modB
                else:
                    int = (np.sqrt(np.abs(1 - lam*modB))/(modB*np.sqrt(np.abs((zeta-zetal)*(zeta-zetar)))))*jacfac/modB
            elif (mode == 'khat'):
                modB = modBf(zeta)
                int = (np.sqrt(1 - lam*modB)**3/modB)*jacfac/modB
            elif (mode == 'dkhatdalpha'):
                modB = modBf(zeta)
                dmodBdalpha = dmodBdthetaf(zeta)
                if (zeta == zetal):
                    int = 1/np.sqrt(np.abs((zeta-zetar)))*dmodBdalpha*(-1.5*lam \
                              -2*(1 - lam*modB)/modB)*jacfac/(modB*modB)
                elif (zeta == zetar):
                    int = 1/np.sqrt(np.abs((zeta-zetal)))*dmodBdalpha*(-1.5*lam \
                              -2*(1 - lam*modB)/modB)*jacfac/(modB*modB)
                else:
                    int = np.sqrt(np.abs(1 - lam*modB))/np.sqrt(np.abs((zeta-zetal)*(zeta-zetar)))*dmodBdalpha*(-1.5*lam \
                              -2*(1 - lam*modB)/modB)*jacfac/(modB*modB)
            elif (mode == 'tau'):
                int = (1/np.sqrt(1 - lam*modB))*jacfac/modB
            else:
                raise ValueError('Incorrect value of mode')
        return int

    def event(zeta, state):
        return modBf(zeta)-1/lam

    event.terminal = True
    event.direction = +1.

    import time

    # Find points such that modB brackets 1/lam on either side and
    # left point has larger modB, right points has smaller modB
    # This yields the set of potential left bounce points
    bouncel = np.argwhere((modB[0:-1] > 1/lam)*(modB[1::] < 1/lam))

    indexl = []
    indexr = []
    bounce = []
    indexl_try = []
    for ir in range(len(bouncel)):
        zetal = zeta[bouncel[ir]]
        zetar = zeta[bouncel[ir]+1]
        assert(modBf(zetal)-1/lam > 0)
        assert(modBf(zetar)-1/lam < 0)
        assert(zetar > zetal)

        # time1 = time.time()
        sol = root_scalar(lambda zeta: modBf(zeta)-1/lam, bracket=[zeta[bouncel[ir]], zeta[bouncel[ir]+1]])

        # time2 = time.time()
        # print('Time for root_scalar: ',time2-time1)
        # Check that field is decreasing along field line
        points[:, 1] = theta0 + iota * sol.root
        points[:, 2] = sol.root
        field.set_points(points)
        Bprime = field.dmodBdzeta() + iota*field.dmodBdtheta()
        # Only consider left bounce points
        if Bprime <= 0:
            indexl_try.append(sol.root)

    # For each of these left bounce points, perform integral until right bounce point is reached
    for ir in range(len(indexl_try)):
        # Check that this left bounce point is to the right of the previous right bounce point
        if (len(indexr)>0):
            if (indexl_try[ir]<=indexr[-1]):
                continue
        Jpar = solve_ivp(integrand, [indexl_try[ir], indexl_try[ir]+nmax*2*np.pi], [0], events=event)
        # A termination event ocurred, i.e., right bounce point is found
        if (Jpar.status == 1 and Jpar.t_events[-1]-indexl_try[ir] > 0):
            indexr.append(Jpar.t_events[-1][0])
            indexl.append(indexl_try[ir])
            # assert()

            # Need to perform integration using quad to take care of the singularity
            # These are the integrals with sqrt()
            # if (mode=='jpar' or mode=='dkhatdalpha' or mode=='khat' or mode=='ihat'):
            #     y, abserr = quad(lambda zeta:
            #         integrand_quad(zeta)/np.sqrt((zeta-indexl_try[ir])*(Jpar.t_events[-1][0]-zeta)), indexl_try[ir],
            #                      Jpar.t_events[-1][0], points=(indexl_try[ir],
            #                      Jpar.t_events[-1][0]),
            #                      full_output=0, limit=limit,
            #                      weight='alg', wvar=(1/2,1/2))
            # else:
            # time1 = time.time()
            try:
                y, abserr = quad(integrand_quad, indexl_try[ir],
                                 Jpar.t_events[-1][0],
                                 # points=(indexl_try[ir],
                                 # Jpar.t_events[-1][0]),
                                 full_output=0, limit=limit,
                                 weight='alg', wvar=(1/2,1/2),
                                 args=(indexl_try[ir],Jpar.t_events[-1][0]),
                                 maxp1=maxp1)
                # y, abserr = quad(integrand_quad, indexl_try[ir],
                #                  Jpar.t_events[-1][0], points=(indexl_try[ir],
                #                  Jpar.t_events[-1][0]), epsabs=0, epsrel=eps,
                #                  full_output=0, limit=limit)
            except:
                print('Exception ocurred during integration')
                print('theta0 = ',theta0)
                print('s = ',s)
                print('lam = ',lam)
                print('integral = ',y)
            # time2 = time.time()
            # print('Time for quad: ',time2-time1)
            bounce.append(y)

    return indexl, indexr, bounce, zeta
