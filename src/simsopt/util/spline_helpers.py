import numpy as np
import matplotlib.pyplot as plt
from simsopt.util.mpi import MpiPartition

mpi = MpiPartition()
mpi.write()

def print_dofs_nicely(surf, lb=None, ub=None):
    if lb is None and ub is None:
        dofs_lb_ub = list(zip(surf.x, surf.lower_bounds, surf.upper_bounds))
    else:
        dofs_lb_ub = list(zip(surf.x, lb, ub))
    dofs_dict = dict(zip(surf.dof_names, dofs_lb_ub))
    print("{:<30} {:<20} {:<20} {:<20}".format('dof','value','lower bound', 'upper bound'))
    for k, v in dofs_dict.items():
        val, lb, ub = v
        print("{:<30} {:<20} {:<20} {:<20}".format(k, val, lb, ub))

def b_p(t, p, x, i=None):
    '''
    Compute the B-spline basis function B_pi.

    ## Inputs:
    t : knot vector \\
    p : degree \\
    x : point at which to evaluate \\
    i : basis of interest (if None, returns all )

    ## Outputs:
    If i is None, an array of shape (x, k) is returned, which consists of the kth p-order B-Spline basis function computed on the array x.
    If i is an integer, an array of shape (x) is returned, which consists of the ith p-order B-Spline basis function computed on the array x.
    '''
    b = []

    for deg in range(0, p+1):
        if deg == 0:
            # x = x[(x >= t[0]) & (x <= t[-1])]
            x1d = x.copy()
            t1d = t.copy()
            t = np.outer(np.ones(len(x)), t)
            x = np.expand_dims(x, 0)
            b0 = ((x.T >= t[:, :-1]) & (x.T < t[:, 1:]))
            b0[np.isclose(x1d, t1d[-1]), np.isclose(t1d[1:], t1d[-1])] = 1 # accounting for evaluation at rightmost knot
            b.append(b0)
        else:
            l_term_n = (x.T - t[:, :-deg-1])
            l_term_d = (t[:, deg:-1] - t[:, :-deg-1])
            l_term = b[-1][:, :-1]* np.divide(l_term_n, l_term_d, out=np.zeros_like(l_term_d), where=l_term_d != 0)

            r_term_n = (t[:, deg+1:] - x.T)
            r_term_d = (t[:, deg+1:] - t[:, 1:-deg])
            r_term = b[-1][:, 1:]* np.divide(r_term_n, r_term_d, out=np.zeros_like(r_term_d), where=r_term_d != 0)

            b.append(l_term + r_term)

    if i is None:
        return b[-1]
    else:
        return b[-1][:, i]

def b_p_deriv(t, p, x):
    '''
    Compute the B-spline basis function B_pi and its first derivative dB_pi/dx,
    using the standard identity

        dB_i,p/dx = p * (B_i,p-1/(t[i+p]-t[i]) - B_i+1,p-1/(t[i+p+1]-t[i+1]))

    b_p's Cox-de Boor recursion already computes the degree-(p-1) basis on its
    way to degree p, and the knot-difference denominators above are exactly
    the ones it computes for its own final (degree-p) recursion step -- so
    this is b_p with one addition, not a separate derivation. Requires p >= 1.

    ## Inputs / Outputs: same convention as b_p (i=None, i.e. all basis
    functions), but returns a (basis, derivative) tuple, each of shape (x, k).
    '''
    b = []
    tt = t
    xx = x
    for deg in range(0, p+1):
        if deg == 0:
            x1d = xx.copy()
            t1d = tt.copy()
            tt = np.outer(np.ones(len(xx)), tt)
            xx = np.expand_dims(xx, 0)
            b0 = ((xx.T >= tt[:, :-1]) & (xx.T < tt[:, 1:]))
            b0[np.isclose(x1d, t1d[-1]), np.isclose(t1d[1:], t1d[-1])] = 1
            b.append(b0)
        else:
            l_term_n = (xx.T - tt[:, :-deg-1])
            l_term_d = (tt[:, deg:-1] - tt[:, :-deg-1])
            l_term = b[-1][:, :-1] * np.divide(l_term_n, l_term_d, out=np.zeros_like(l_term_d), where=l_term_d != 0)

            r_term_n = (tt[:, deg+1:] - xx.T)
            r_term_d = (tt[:, deg+1:] - tt[:, 1:-deg])
            r_term = b[-1][:, 1:] * np.divide(r_term_n, r_term_d, out=np.zeros_like(r_term_d), where=r_term_d != 0)

            b.append(l_term + r_term)

    bp_minus1 = b[-2]
    l_d = (tt[:, p:-1] - tt[:, :-p-1])
    r_d = (tt[:, p+1:] - tt[:, 1:-p])
    deriv = p * (
        np.divide(bp_minus1[:, :-1], l_d, out=np.zeros_like(l_d), where=l_d != 0)
        - np.divide(bp_minus1[:, 1:], r_d, out=np.zeros_like(r_d), where=r_d != 0)
    )
    return b[-1], deriv

def uniform_knots(n, p, domain=2*np.pi):
    '''
    Periodic, uniformly-spaced knot vector of degree p for a control array
    built by tiling p points from a closed loop of n+1 points onto each
    side (length n+1+2p). Counterpart to chord_length_knots with the same
    shape/convention (length n+3p+2, knots[2p]==0, knots[n+2p+1]==domain),
    for when point spacing shouldn't influence the parametrization. Only
    exactly reflection-symmetric for odd p.
    '''
    interval = domain / (n + 1)
    k = np.arange(-2 * p, n + 2 * p + 2)
    knots_wide = k * interval
    n_needed = (n + 1 + 2 * p) + p + 1
    start = (len(knots_wide) - n_needed) // 2
    return knots_wide[start:start + n_needed]

def chord_length_knots(points, p, domain=2*np.pi):
    '''
    Periodic, chord-length-parametrized knot vector of degree p for a
    control array built by tiling p points from points[0..n] onto each
    side (length n+1+2p; see uniform_knots for the shape/convention this
    matches). Knot spacing follows the actual Euclidean distance between
    consecutive points (wrapping points[n] back to points[0]) rather than
    assuming uniform spacing (Piegl & Tiller, "The NURBS Book"), adapted
    for a periodic/closed curve. Only exactly reflection-symmetric for
    odd p.

    points: (n+1, dim) array of control points, one full period, NOT
    including the tiled copies (those are built separately, same as for
    the control points themselves).
    '''
    n = len(points) - 1
    closed = np.concatenate([points, points[:1]], axis=0)  # (n+2, dim): +1 closing gap
    gaps = np.linalg.norm(np.diff(closed, axis=0), axis=1)  # (n+1,)
    total = np.sum(gaps)
    if total <= 0:
        # degenerate (coincident points) -- fall back to uniform rather than divide by zero
        gaps = np.ones(n + 1)
        total = n + 1

    # cumulative chord length, normalized to [0, domain]: t_core[0]=0, ..., t_core[n+1]=domain
    t_core = np.concatenate([[0.0], np.cumsum(gaps)]) * (domain / total)

    # periodic extension, wide enough for 2 tiles on each side, then keep
    # exactly the knots needed for the p-tiled-both-sides control array
    k = np.arange(-2 * p, n + 2 * p + 2)
    m = k % (n + 1)
    shift = (k - m) // (n + 1)
    knots_wide = t_core[m] + shift * domain

    n_needed = (n + 1 + 2 * p) + p + 1
    start = (len(knots_wide) - n_needed) // 2
    return knots_wide[start:start + n_needed]

def rot_matrix_2d(theta):
    return np.array(
        [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ]
    )

def alan_plot(rbc, rbs, zbc, zbs, ntheta, nzeta, M, N, nfp, ax=None, poincare=True):
    if ax is None:
        fig = plt.figure("3D Surface Plot")
        fig.patch.set_facecolor('white')
        ax = fig.add_subplot(projection='3d',azim=0, elev=90)
    xn = np.arange(-N,N+1,1)
    xm = np.arange(0,M+1,1)

    ntheta = 200
    nzeta = 9
    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi/nfp,num=nzeta) + 2*np.pi/nfp
    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)

    if poincare:
        fig = plt.figure("Poincare Plots",figsize=(14,7))
        fig.patch.set_facecolor('white')

        R = np.zeros((ntheta,nzeta))
        Z = np.zeros((ntheta,nzeta))

        for i in range(rbc.shape[0]):
            for j in range(rbc.shape[1]):
                if rbc[i,j] !=0 or zbs[i,j] != 0:
                    angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                    R = R + rbc[i,j]*np.cos(angle)#/(np.abs(i) + np.abs(j))
                    Z = Z + zbs[i,j]*np.sin(angle)#/(np.abs(i) + np.abs(j))
                if rbs[i,j] !=0 or zbc[i,j]:
                    angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                    R = R + rbs[i,j]*np.sin(angle)
                    Z = Z + zbc[i,j]*np.cos(angle)
        numCols = 5
        numRows = 2
        plotNum = 1
        zeta = np.linspace(0,2 * np.pi/nfp,num=nzeta,endpoint=True)

        plt.subplot(numRows,numCols,plotNum)
        #plt.subplot(1,1,1)
        plotNum += 1
        for ind in range(nzeta):
            plt.subplot(numRows,numCols,ind+1)
            plt.title(r'$\phi =$' + str(zeta[ind]))
            plt.gca().set_aspect('equal',adjustable='box')

            plt.plot(R[:,ind], Z[:,ind], '-')
            plt.plot(R[:,ind], Z[:,ind], '-')
            plt.plot(R[:,ind], Z[:,ind], '-')
            plt.plot(R[:,ind], Z[:,ind], '-')
        plt.gca().set_aspect('equal',adjustable='box')
        plt.xlabel('R')
        plt.ylabel('Z')

    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi,num=nzeta) + 2*np.pi/nfp

    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)

    ntheta = 200
    nzeta = 200
    theta1D = np.linspace(0,2*np.pi,num=ntheta)
    zeta1D = np.linspace(0,2*np.pi,num=nzeta) + 2*np.pi/nfp
    zeta2D, theta2D = np.meshgrid(zeta1D,theta1D)
    R = np.zeros((ntheta,nzeta))
    Z = np.zeros((ntheta,nzeta))

    for i in range(rbc.shape[0]):
        for j in range(rbc.shape[1]):
            if rbc[i,j] !=0 or zbs[i,j] != 0:
                angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                R = R + rbc[i,j]*np.cos(angle)#/(np.abs(i) + np.abs(j))
                Z = Z + zbs[i,j]*np.sin(angle)#/(np.abs(i) + np.abs(j))
            if rbs[i,j] !=0 or zbc[i,j]:
                angle = xm[j]*theta2D - xn[i]*zeta2D*nfp
                R = R + rbs[i,j]*np.sin(angle)
                Z = Z + zbc[i,j]*np.cos(angle)
    X = R * np.cos(zeta2D)
    Y = R * np.sin(zeta2D)

    ax.plot_surface(X, Y, Z)
    ax.set_box_aspect((1, 1, 1))
    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)
    ax.set_zlim(-1, 1)

def vol_from_boundary(surf: "SurfaceRZFourier", nu, nv):  # noqa: F821 -- deliberate string forward-ref, not imported to avoid a circular import (simsopt.geo -> this module -> simsopt.geo)
    '''
    Compute volume enclosed by a boundary given with VMEC Fourier coefficients.
    Uses a clever trick based on Gauss' identity:
    ∫∫∫ div(A) dV = ∫∫ A . n dA
    Choose some A whose divergence is unity, in this case (in cylindrical coordinates)
    (0, 0, z). Then,
    V = ∫∫∫dV = ∫∫ zn_z dA.

    :param rbc: RBC coefficients, given in (n,m) format
    :param zbs: ZBS coefficients, given in (n,m) format
    :param M: Max poloidal mode number
    :param N: Max toroidal mode number
    :param nu: Number of points to take in u for quadrature
    :param nv: Number of points to take in v for quadrature
    :param nfp: Number of field periods
    '''
    rbc = surf.rc.T
    zbs = surf.zs.T
    M = surf.mpol
    N = surf.ntor
    nfp = surf.nfp

    u_1d = np.linspace(0, 2*np.pi, nu, endpoint=True)
    v_1d = np.linspace(0, 2*np.pi, nv, endpoint=True)
    v_grid, u_grid = np.meshgrid(v_1d, u_1d)
    cosnmuz = np.array(
        [[np.cos(m*u_grid - n*(nfp*v_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )
    sinnmuz = np.array(
        [[np.sin(m*u_grid - n*(nfp*v_grid)) for m in range(0, M+1)] for n in range(-N, N+1)],
    )
    m = np.array(
        [[m for m in range(0, M+1)] for n in range(-N, N+1)]
    )

    R_uz = np.einsum('nm,nmuz->uz', rbc, cosnmuz)
    Z_uz = np.einsum('nm,nmuz->uz', zbs, sinnmuz)
    duR_uz = np.einsum('nm,nmuz->uz', -m*rbc, sinnmuz)

    integrand = Z_uz * R_uz * duR_uz
    res = (u_1d[1]-u_1d[0])*(v_1d[1]-v_1d[0])*np.sum(integrand)

    return res