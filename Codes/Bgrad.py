import numpy as np
import sympy as sp
from Bcube__7_1_24 import Pdsym


#FOR CUBE
mu0 = 4*np.pi*10**-7
dim = np.array([1,1,1])


def grad_r_H(r_loc, dims, P):

    dH = np.zeros((3,3,3))
 
    x = np.array([r_loc[0] + dims[0]/2, r_loc[0] - dims[0]/2])
    y = np.array([r_loc[1] + dims[1]/2, r_loc[1] - dims[1]/2])
    z = np.array([r_loc[2] + dims[2]/2, r_loc[2] - dims[2]/2])
    
    lst = np.array([0,1])

    for s in range(3):
        for row in range(3):
            for col in range(3):
                for i in lst:
                    for j in lst:
                        for k in lst:
                            summa = (-1)**(i+j+k)
                            
                            rp = np.array([x[i],y[j],z[k]])
                            drp = np.array([P[0][s],P[1][s],P[2][s]])
                            
                            rijk = np.sqrt(rp[0]**2 + rp[1]**2 + rp[2]**2)
                            drijk = (rp[0]*drp[0]+rp[1]*drp[1]+rp[2]*drp[2])/rijk
                            
                            if row == col:
                                dH[row,col,s] += summa * ( (rp[row-1]*rijk*(drp[row]*rp[row-2]+rp[row]*drp[row-2]) - rp[row-2]*rp[row]*(drp[row-1]*rijk+rp[row-1]*drijk))/((rp[row-1]*rijk)**2 + (rp[row-2]*rp[row])**2) + (rp[row-2]*rijk*(drp[row]*rp[row-1]+rp[row]*drp[row-1]) - rp[row-1]*rp[row]*(drp[row-2]*rijk+rp[row-2]*drijk))/((rp[row-2]*rijk)**2 + (rp[row-1]*rp[row])**2) )
                            else:
                                if 2*row-col > 2:
                                    index = 2*row-col - 3
                                else:
                                    index = 2*row-col
                                dH[row,col,s] += summa * 1/(rp[index]+rijk) * (drp[index]+drijk)
    return dH/(4*np.pi)


def gradr_Bcube(points, magPos, M, phiThetas, dims):
    D = len(M)
    N = len(points)

    dB = np.zeros((N,3,3))
    for n in range(N):
        for d in range(D):
            P = Pdsym(phiThetas[d][0],phiThetas[d][1])
            r_loc = P @ (points[n] - magPos[d])
            drH = grad_r_H(r_loc, dims, P)
            dB[n] += mu0 * P.T @ (drH @ M[d]) @ P
    return dB


#FOR DIPOLE

def gradr_Bdip(points, magPos, m):
    D = len(m)
    N = len(points)
    hat = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    dB = np.zeros((N,3,3))
    for n in range(N):
        for d in range(D):
            r = points[n] - magPos[d]
            rmag = np.linalg.norm(r)
            for s in range(3):
                grad = 3*mu0/(4*np.pi) * (((m[d,s]*r+(m[d]@r)*hat[s])*rmag**2 - 5*r[s]*r*(m[d]@r))/rmag**7 + m[d]*r[s]/rmag**5)
                dB[n,:,s] += grad

    return dB

