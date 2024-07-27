import numpy as np
import sympy as sp
import itertools
import time

mu0 = 4*np.pi*10**-7
dim = np.array([1,1,1])

#FOR CUBIC MAGNETS

def Pd(phi,theta): #goes from global to local
    return np.array([
        [np.cos(theta)*np.cos(phi), -np.cos(theta)*np.sin(phi), np.sin(theta)],
        [np.sin(phi), np.cos(phi), 0],
        [-np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    ])

def iterate_over_corners(corner, x, y, z):
    i,j,k = corner
    summa = (-1)**(i+j+k)
    rijk = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)

    h = np.zeros((3,3))
    h[0] = summa * np.array([np.arctan2(y[j]*x[i],z[k]*rijk) + np.arctan2(z[k]*x[i],y[j]*rijk), np.log(z[k] + rijk), np.log(y[j] + rijk)])
    h[1] = summa * np.array([np.log(z[k] + rijk), np.arctan2(x[i]*y[j],z[k]*rijk) + np.arctan2(z[k]*y[j],x[i]*rijk), np.log(x[i] + rijk)])
    h[2] = summa * np.array([np.log(y[j] + rijk), np.log(x[i] + rijk), np.arctan2(x[i]*z[k],y[j]*rijk) + np.arctan2(y[j]*z[k],x[i]*rijk)])

    return h

def Hd_i_prime(r, dims):
    H = np.zeros((3,3))

    xp, yp, zp = r
    
    x = np.array([xp + dims[0]/2, xp - dims[0]/2])
    y = np.array([yp + dims[1]/2, yp - dims[1]/2])
    z = np.array([zp + dims[2]/2, zp - dims[2]/2])
    
    lst = np.array([0,1])

    corners = [np.array(corn) for corn in itertools.product(lst, repeat=3)]
    H += np.sum(list(map(lambda corner: iterate_over_corners(corner, x, y, z), corners)), axis=0)
    return H/(4*np.pi)

def B_direct(points, magPos, M, dims, phiThetas):
    N = len(points)
    D = len(M)
    
    B = np.zeros((N,3))
    for n in range(N):
        for d in range(D):
            P = Pd(phiThetas[d,0],phiThetas[d,1])
            r_loc = P @ (points[n] - magPos[d])

            tx = np.heaviside(dims[0]/2 - np.abs(r_loc[0]),0.5)
            ty = np.heaviside(dims[1]/2 - np.abs(r_loc[1]),0.5)
            tz = np.heaviside(dims[2]/2 - np.abs(r_loc[2]),0.5)       
            tm = 2*tx*ty*tz
            
            B[n] += mu0 * P.T @ (Hd_i_prime(r_loc,dims) @ (P @ M[d]) + tm*P@M[d])

    return B

def Bn_direct(points, magPos, M, norms, dims, phiThetas): #solve Bnorm using analytic formula
    N = len(points)
    B = B_direct(points, magPos, M, dims, phiThetas)
    Bn = np.zeros(N)
    for n in range(N):
        Bn[n] = B[n] @ norms[n]
    return Bn

def gd_i(r_loc, n_i_loc, dims): #for matrix formulation
    tx = np.heaviside(dims[0]/2 - np.abs(r_loc[0]),0.5)
    ty = np.heaviside(dims[1]/2 - np.abs(r_loc[1]),0.5)
    tz = np.heaviside(dims[2]/2 - np.abs(r_loc[2]),0.5)       
    tm = 2*tx*ty*tz

    return mu0 * (Hd_i_prime(r_loc,dims).T + tm*np.eye(3)) @ n_i_loc

def Acube(points, magPos, norms, dims, phiThetas):
    print(points.shape, magPos.shape, norms.shape, phiThetas.shape)
    N = len(points)
    D = len(magPos)
    
    A = np.zeros((N,3*D))
    print('N, D =', N, D)
    t1 = time.time()
    for n in range(N):
        for d in range(D):
            P = Pd(phiThetas[d,0],phiThetas[d,1])
            r_loc = P @ (points[n] - magPos[d])
            n_loc = P @ norms[n]
            
            g = P.T@gd_i(r_loc,n_loc,dims)#P.T to make g global
            A[n,3*d : 3*d + 3] = g

            assert (N,3*D) == A.shape
    t2 = time.time()
    print('t = ', t2 - t1)
    return A

def Bn_fromMat(points, magPos, M, norms, dims, phiThetas): #solving Bnorm using matrix formulation
    A = Acube(points, magPos, norms, dims, phiThetas)
    Ms = np.concatenate(M)
    return A @ Ms


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

