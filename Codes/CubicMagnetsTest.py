
import numpy as np

mu0 = 4*np.pi*10**-7
dim = np.array([1,1,1])

#transformation matrix
def Pd(phi,theta): #goes from local to global
    return np.array([
        [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), -np.sin(theta)],
        [-np.sin(phi), np.cos(phi), 0],
        [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    ])

def Hd_i(r, dims, P):
    H = np.zeros((3,3))
    
    r_local = P.T@r #get r into local coordinates
    xp, yp, zp = r_local
    
    x = np.array([xp + dims[0]/2, xp - dims[0]/2])
    y = np.array([yp + dims[1]/2, yp - dims[1]/2])
    z = np.array([zp + dims[2]/2, zp - dims[2]/2])
    
    lst = np.array([0,1])
    for i in lst:
        for j in lst:
            for k in lst:
                summa = (-1)**(i+j+k)
                rijk = np.sqrt(x[i]**2 + y[j]**2 + z[k]**2)
                        
                H[0] += summa * np.array([np.arctan(x[i]*rijk/(y[j]*z[k])), np.log(z[k] + rijk), np.log(y[j] + rijk)])
                H[1] += summa * np.array([np.log(z[k] + rijk), np.arctan(y[j]*rijk/(x[i]*z[k])), np.log(x[i] + rijk)])
                H[2] += summa * np.array([np.log(y[j] + rijk), np.log(x[i] + rijk), np.arctan(z[k]*rijk/(x[i]*y[j]))])
    return H

def g_d(r, n_i, P, dims):
    H_d = Hd_i(r, dim, P)
    n_i_local = P.T@n_i #global to local
    return H_d.T@n_i_local + n_i_local

def constructA(points, norms, phiThetas, magLocs, dims):
    assert len(points) == len(norms)
    D = len(magLocs)
    N = len(points)

    A = np.zeros((N,3*D))

    for d in range(D):
        P = Pd(phiThetas[d][0], phiThetas[d][1])
        rd = points - magLocs[d]
        for i in range(N):
            g = g_d(rd[i], norms[i], P, dims)
            A[i][3*d] = g[0]
            A[i][3*d+1] = g[1]
            A[i][3*d+2] = g[2]

    return A

#changing M from global to local
def AdotM(A,Ms,phiThetas):
    D = len(Ms)
    Ms_loc = np.zeros((D,3))
    
    for d in range(D):
        P = Pd(phiThetas[d][0],phiThetas[d][1])
        Ms_loc[d] = P.T@Ms[d] #global to local

    Ms_loc = np.concatenate(Ms_loc)
    return A@Ms_loc

#changing gs from local to global
#P.T does the inverse rotation
def constructAglobal(points, norms, phiThetas, magLocs, dims):
    assert len(points) == len(norms)
    D = len(magLocs)
    N = len(points)

    A = np.zeros((N,3*D))

    for d in range(D):
        P = Pd(phiThetas[d][0], phiThetas[d][1])
        rd = points - magLocs[d]
        for i in range(N):
            g = g_d(rd[i], norms[i], P, dims)
            g_global = P@g #local to global
            A[i][3*d] = g_global[0]
            A[i][3*d+1] = g_global[1]
            A[i][3*d+2] = g_global[2]

    return A

def SolveBnorm(points, norms, phiThetas, method, M, magLocations, dimensions = dim):
    if method == '1':
        A = mu0 * constructA(points, norms, phiThetas, magLocations, dimensions)
        Bn = AdotM(A,M,phiThetas)
    elif method == '2':
        A = mu0 * constructAglobal(points, norms, phiThetas, magLocations, dimensions)
        Ms = np.concatenate(M)
        Bn = np.dot(A,Ms)
    else:
        print('Error: method ' + str(method) + ' does not exist')
    return Bn

#phiThetas = phi theta coordinates of magnets in radians (magPos will be redundant when we introduce torus, will be changed)
#points = points at which to evaluate B normal
#magPos = locations of magnets in the global coordinate system
#norms = normal vectors of plasma at each point

# m determines M vectors
# V = np.prod(dim)
# M = m * V

#solve for B directly, without A matrix
def Bnorm_cube_simple(points, norms, phiThetas, M, magLocs, dims):
    assert len(points) == len(norms)
    D = len(magLocs)
    N = len(norms)
    B = np.zeros((N,3))
    Bnorm = np.zeros(N)

    for n in range(N):
        rn = points[n] - magLocs
        mags = np.zeros((D,3))
        for d in range(D):
            P = Pd(phiThetas[d][0], phiThetas[d][1])
            mags[d] = (Hd_i(rn[d], dims, P) + np.identity(3)) @ M[d]
        B[n] = mu0 * np.sum(mags,0)
        Bnorm[n] = B[n]@norms[n]
    return Bnorm, B


print(Bnorm_cube_simple(np.array([[30,-30,10]]),np.array([[0,0,1]]),np.array([[0,0]]),np.array([[0,0,1]]),np.array([[0,0,0]]),np.array([1,1,1]))[1])
            


