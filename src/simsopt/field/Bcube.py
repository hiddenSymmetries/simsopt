
import numpy as np
import itertools
import time

mu0 = 4*np.pi*10**-7
dim = np.array([1,1,1])

#FOR CUBIC MAGNETS

__all__ = ['Pd', 'iterate_over_corners',
           'Hd_i_prime', 'B_direct', 'Bn_direct',
           'gd_i', 'Acube', 'Bn_fromMat'
           ]

def Pd(phi,theta): #goes from global to local
    return np.array([
        [np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)],
        [np.sin(phi), np.cos(phi), np.zeros(len(theta))],
        [-np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    ])

def iterate_over_corners(corner, x, y, z):
    i,j,k = corner
    summa = (-1)**(i+j+k)
    rijk = np.sqrt(x[i, :, :]**2 + y[j, :, :]**2 + z[k, :, :]**2)

    atan_xy = np.arctan2(y[j, :, :]*x[i, :, :],z[k, :, :]*rijk)
    atan_xz = np.arctan2(z[k, :, :]*x[i, :, :],y[j, :, :]*rijk)
    atan_yz = np.arctan2(z[k, :, :]*y[j, :, :],x[i, :, :]*rijk)
    log_x = np.log(x[i, :, :] + rijk)
    log_y = np.log(y[j, :, :] + rijk)
    log_z = np.log(z[k, :, :] + rijk)
    
    h = summa * np.array([
        [atan_xy + atan_xz, log_z, log_y],
        [log_z, atan_xy + atan_yz, log_x],
        [log_y, log_x, atan_xz + atan_yz]
    ])

    return h

def Hd_i_prime(r, dims):
    xp = r[:, :, 0]
    yp = r[:, :, 1]
    zp = r[:, :, 2]
    
    x = np.array([xp + dims[0]/2, xp - dims[0]/2])
    y = np.array([yp + dims[1]/2, yp - dims[1]/2])
    z = np.array([zp + dims[2]/2, zp - dims[2]/2])
    
    lst = np.array([0, 1])

    corners = [np.array(corn) for corn in itertools.product(lst, repeat=3)]
    H = np.sum([iterate_over_corners(corner, x, y, z) for corner in corners], axis=0)/(4*np.pi)
    H = np.transpose(H, axes=([2, 3, 0, 1]))
    return H

def B_direct(points, magPos, m, dims, phiThetas):

    P = Pd(phiThetas[:, 0], phiThetas[:, 1])
    P = np.transpose(P, axes=[2, 0, 1])
    r_loc = np.sum(P[None, :, :, :] * (points[:, None, :] - magPos[None, :, :])[:, :, None, :], axis=-1)

    tx = np.heaviside(dims[0]/2 - np.abs(r_loc[:, :, 0]),0.5)
    ty = np.heaviside(dims[1]/2 - np.abs(r_loc[:, :, 1]),0.5)
    tz = np.heaviside(dims[2]/2 - np.abs(r_loc[:, :, 2]),0.5)       
    tm = 2*tx*ty*tz
            
    Pm = np.sum(P * m[:, None, :], axis=-1)
    tm_Pm = tm[:, :, None] * Pm[None, :, :]
    H_pm = np.sum(Hd_i_prime(r_loc, dims) * Pm[None, :, None, :], axis=-1)

    # Double sum because we are rotating by P and then summing over all the magnet locations
    B = mu0 * np.sum(np.sum(P[None, :, :, :] * (H_pm + tm_Pm)[:, :, None, :], axis=-1), axis=1) / np.prod(dims)

    return B

def Bn_direct(points, magPos, m, norms, dims, phiThetas): #solve Bnorm using analytic formula
    N = len(points)
    B = B_direct(points, magPos, m, dims, phiThetas)
    Bn = np.sum(B * norms, axis=-1)

    return Bn

def gd_i(r_loc, n_i_loc, dims): #for matrix formulation
    tx = np.heaviside(dims[None, None, 0]/2 - np.abs(r_loc[:, :, 0]), 0.5)
    ty = np.heaviside(dims[None, None, 1]/2 - np.abs(r_loc[:, :, 1]), 0.5)
    tz = np.heaviside(dims[None, None, 2]/2 - np.abs(r_loc[:, :, 2]), 0.5)       
    tm = 2*tx*ty*tz

    # Need eye on tm
    return mu0 * np.sum((Hd_i_prime(r_loc, dims) + tm[:, :, None, None] * np.eye(3)[None, None, :, :]) * n_i_loc[:, :, None, :], axis=-1)

def Acube(points, magPos, norms, dims, phiThetas):
    N = len(points)
    print('beginning Acube calc: ')
    t1 = time.time()
    P = Pd(phiThetas[:, 0], phiThetas[:, 1])
    P = np.transpose(P, axes=[2, 0, 1])
    r_loc = np.sum(P[None, :, :, :] * (points[:, None, :] - magPos[None, :, :])[:, :, None, :], axis=-1)
    n_loc = np.sum(P[None, :, :, :] * norms[:, None, None, :], axis=-1)
    A = gd_i(r_loc, n_loc, dims).reshape(N, -1)
    t2 = time.time()
    print('Acube calc took t = ', t2 - t1, ' s')
    return A / np.prod(dims)

def Bn_fromMat(points, magPos, m, norms, dims, phiThetas): #solving Bnorm using matrix formulation
    A = Acube(points, magPos, norms, dims, phiThetas)
    ms = np.concatenate(m)
    return A @ ms

