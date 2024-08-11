import numpy as np
import sympy as sp
import itertools

mu0 = 4*np.pi*10**-7
dim = np.array([1,1,1])

#FOR CUBIC MAGNETS

__all__ = ['Pd', 'iterate_over_corners',
           'Hd_i_prime', 'B_direct', 'Bn_direct',
           'gd_i', 'Acube', 'Bn_fromMat', 'Bdip_direct',
           'Bndip_direct', 'g_dip', 'Adip', 'Bndip_fromMat'
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
    # print('H shape = ', H.shape)
    return H

def B_direct(points, magPos, m, dims, phiThetas):

    P = Pd(phiThetas[:, 0], phiThetas[:, 1])
    P = np.transpose(P, axes=[2, 0, 1])
    r_loc = np.sum(P[None, :, :, :] * (points[:, None, :] - magPos[None, :, :])[:, :, None, :], axis=-1)

    tx = np.heaviside(dims[0]/2 - np.abs(r_loc[:, :, 0]),0.5)
    ty = np.heaviside(dims[1]/2 - np.abs(r_loc[:, :, 1]),0.5)
    tz = np.heaviside(dims[2]/2 - np.abs(r_loc[:, :, 2]),0.5)       
    tm = 2*tx*ty*tz
            
    # print(Hd_i_prime(r_loc, dims).shape, P.shape, tm.shape, m.shape)
    Pm = np.sum(P * m[:, None, :], axis=-1)
    tm_Pm = tm[:, :, None] * Pm[None, :, :]
    H_pm = np.sum(Hd_i_prime(r_loc, dims) * Pm[None, :, None, :], axis=-1)

    # Double sum because we are rotating by P and then summing over all the magnet locations
    B = mu0 * np.sum(np.sum(P[None, :, :, :] * (H_pm + tm_Pm)[:, :, None, :], axis=-1), axis=1) / np.prod(dims)
    
    # B = np.zeros((N,3))
    # for n in range(N):
    #     for d in range(D):
    #         P = Pd(phiThetas[d,0],phiThetas[d,1])
    #         r_loc = P @ (points[n] - magPos[d])

    #         tx = np.heaviside(dims[0]/2 - np.abs(r_loc[0]),0.5)
    #         ty = np.heaviside(dims[1]/2 - np.abs(r_loc[1]),0.5)
    #         tz = np.heaviside(dims[2]/2 - np.abs(r_loc[2]),0.5)       
    #         tm = 2*tx*ty*tz
            
    #         B[n] += mu0 * P.T @ (Hd_i_prime(r_loc,dims) @ (P @ m[d]) + tm*P@m[d]) / np.prod(dims)

    return B

def Bn_direct(points, magPos, m, norms, dims, phiThetas): #solve Bnorm using analytic formula
    N = len(points)
    B = B_direct(points, magPos, m, dims, phiThetas)
    Bn = np.sum(B * norms, axis=-1)
    # Bn = np.zeros(N)
    # for n in range(N):
    #     Bn[n] = B[n] @ norms[n]
    return Bn

def gd_i(r_loc, n_i_loc, dims): #for matrix formulation
    tx = np.heaviside(dims[None, None, 0]/2 - np.abs(r_loc[:, :, 0]), 0.5)
    ty = np.heaviside(dims[None, None, 1]/2 - np.abs(r_loc[:, :, 1]), 0.5)
    tz = np.heaviside(dims[None, None, 2]/2 - np.abs(r_loc[:, :, 2]), 0.5)       
    tm = 2*tx*ty*tz
    # print(tx.shape, tm.shape, n_i_loc.shape, Hd_i_prime(r_loc, dims).shape)

    # Need eye on tm
    return mu0 * np.sum((Hd_i_prime(r_loc, dims) + tm[:, :, None, None] * np.eye(3)[None, None, :, :]) * n_i_loc[:, :, None, :], axis=-1)

def Acube(points, magPos, norms, dims, phiThetas):
    import time
    N = len(points)
    # D = len(magPos)
    
    # A = np.zeros((N, 3 * D))
    print('beginning Acube calc: ')
    t1 = time.time()
    P = Pd(phiThetas[:, 0], phiThetas[:, 1])
    P = np.transpose(P, axes=[2, 0, 1])
    r_loc = np.sum(P[None, :, :, :] * (points[:, None, :] - magPos[None, :, :])[:, :, None, :], axis=-1)
    n_loc = np.sum(P[None, :, :, :] * norms[:, None, None, :], axis=-1)
    A = gd_i(r_loc, n_loc, dims).reshape(N, -1)
    # for n in range(N):
    #     for d in range(D):
    #         P = Pd(phiThetas[d,0],phiThetas[d,1])
    #         r_loc = P @ (points[n] - magPos[d])
    #         n_loc = P @ norms[n]
            
    #         g = P.T @ gd_i(r_loc,n_loc,dims)#P.T to make g global
    #         A[n,3*d : 3*d + 3] = g

            # assert (N,3*D) == A.shape
    t2 = time.time()
    print('Acube calc took t = ', t2 - t1, ' s')
    return A / np.prod(dims)

def Bn_fromMat(points, magPos, m, norms, dims, phiThetas): #solving Bnorm using matrix formulation
    A = Acube(points, magPos, norms, dims, phiThetas)
    ms = np.concatenate(m)
    return A @ ms


#FOR DIPOLES

def Bdip_direct(points, magPos, m):
    N = len(points)
    D = len(m)
    
    B = np.zeros((N,3))
    for n in range(N):
        for d in range(D):
            r = points[n] - magPos[d]
            B[n] += mu0/(4*np.pi) * (3*m[d]@r * r/np.linalg.norm(r)**5 - m[d]/np.linalg.norm(r)**3)
    return B

def Bndip_direct(points, magPos, m, norms): #solve Bnorm using analytic formula
    N = len(points)
    B = Bdip_direct(points, magPos, m)
    Bn = np.zeros(N)

    for n in range(N):
        Bn[n] = B[n] @ norms[n]
    return Bn

def g_dip(r,n): #for matrix formulation
    return mu0/(4*np.pi) * (3*r@n * r/np.linalg.norm(r)**5 - n/np.linalg.norm(r)**3)

def Adip(points, magPos, norms):
    N = len(points)
    D = len(magPos)
    
    A = np.zeros((N,3*D))
    for n in range(N):
        for d in range(D):
            r = (points[n] - magPos[d])            
            g = g_dip(r,norms[n])
            
            A[n,3*d] = g[0]
            A[n,3*d+1] = g[1]
            A[n,3*d+2] = g[2]
    return A

def Bndip_fromMat(points, magPos, m, norms): #solve Bnorm using matrix formulation
    A = Adip(points, magPos, norms)
    ms = np.concatenate(m)
    return A @ ms

