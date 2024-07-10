import numpy as np
    
mu0 = 4*np.pi*10**-7

#what to do with square root term?
def g_d(r_i, n_i):
    return mu0/(4*np.pi) * (3*r_i@n_i/np.linalg.norm(r_i)**5 * r_i - n_i/np.linalg.norm(r_i)**3)

def g_d_sqrt(r_i, n_i, dphi_i, dtheta_i):
    return mu0/(4*np.pi) * (3*r_i@n_i/np.linalg.norm(r_i)**5 * r_i - n_i/np.linalg.norm(r_i)**3) * np.sqrt(dphi_i*dtheta_i*np.linalg.norm(n_i))

def constructA(points, norms, magLocs, dims):
    assert len(points) == len(norms)
    D = len(magLocs)
    N = len(points)

    A = np.zeros((N,3*D))

    for d in range(D):
        r = points - magLocs[d]
        for i in range(N):
            gd_i = g_d(r[i], norms[i])
            A[i][3*d] = gd_i[0]
            A[i][3*d+1] = gd_i[1]
            A[i][3*d+2] = gd_i[2]

    return A

def Bnorm_solve(points, norms, magLocs, dims, m):
    A = constructA(points, norms, magLocs, dims)
    ms = np.concatenate(m)
    return A@ms

#solve directly, without A matrix
def Bnorm_dip_simple(points, dip_locs, m, norms): 
    D = len(dip_locs)
    N = len(points)

    B = np.zeros((N,3))
    Bnorm = np.zeros(N)

    for n in range(N):
        mag = np.zeros((D,3))
        r = points[n] - dip_locs
        
        for d in range(D):
            mag[d] = (3*m[d]@r[d]/(np.linalg.norm(r[d])**5)) * r[d] - m[d]/(np.linalg.norm(r[d])**3)

        magSum = np.sum(mag,0)
        B[n] = (mu0 / (4*np.pi)) * magSum
        Bnorm[n] = B[n]@norms[n]
        
    return Bnorm, B

print(Bnorm_dip_simple(np.array([[3000,-30,10000]]),np.array([[0,0,0]]),np.array([[0,0,1]]),np.array([[0,0,1]]))[0])

