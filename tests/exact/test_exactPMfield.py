import unittest
import numpy as np

import sys
sys.path.append('/Users/willhoffman/simsopt/Codes')
from Bcube_nonVec import *
from simsoptpp import dipole_field_Bn, dipole_field_B, dipole_field_dB, dipole_field_dA


def simBn(points, magPos, M, norms):
    N = len(norms)    
    B = dipole_field_B(points, magPos, M)    
    Bn = np.zeros(N)
    for n in range(N):
        Bn[n] = B[n] @ norms[n]

    return Bn
    

class Testing(unittest.TestCase):

    def test_B_field_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])

        Bcube = B_direct(point, magPos, M, dims, phiTheta)
        Bdip = Bdip_direct(point, magPos, M)
        Bsim_dip = dipole_field_B(point, magPos, M)

        assert np.allclose(Bcube, Bdip)
        assert np.allclose(Bdip, Bsim_dip)
        assert np.allclose(Bcube, Bsim_dip)

    def test_Bn_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        Bncube = Bn_direct(point, magPos, M, norm, dims, phiTheta)
        Bndip = Bndip_direct(point, magPos, M, norm)
        Bnsim_dip = simBn(point, magPos, M, norm)

        assert np.allclose(Bncube, Bndip)
        assert np.allclose(Bnsim_dip, Bndip)
        assert np.allclose(Bnsim_dip, Bncube)

    def test_Amatrix_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        N = len(norm)
        D = len(magPos)

        A_sim = dipole_field_Bn(point, magPos, norm, 1, 0, np.array([0,0,0])).reshape(N,3*D)
        A_dip = Adip(point, magPos, norm)
        A_cube = Acube(point, magPos, norm, dims, phiTheta)

        assert np.allclose(A_sim, A_cube)
        assert np.allclose(A_dip, A_cube)
        assert np.allclose(A_dip, A_sim)

    def test_matrixFormulation_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        N = len(norm)
        D = len(magPos)

        BncubeMAT = Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
        BndipMAT = Bndip_fromMat(point, magPos, M, norm)
        Bncube = Bn_direct(point, magPos, M, norm, dims, phiTheta)
        Bndip = Bndip_direct(point, magPos, M, norm)
        BnsimMAT = dipole_field_Bn(point, magPos, norm, 1, 0, np.array([0,0,0])).reshape(N,3*D) @ np.concatenate(M)

        assert np.allclose(BncubeMAT, BndipMAT)
        assert np.allclose(BncubeMAT, Bncube)
        assert np.allclose(BndipMAT, Bndip)
        assert np.allclose(BncubeMAT, BnsimMAT)
        assert np.allclose(BndipMAT, BnsimMAT)

    def test_Bgrad_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])

        dB_cube = gradr_Bcube(point, magPos, M, phiTheta, dims)
        dB_dip = gradr_Bdip(point, magPos, M)
        dBsim_dip = dipole_field_dB(point, magPos, M)

        assert np.allclose(dB_cube, dB_dip)
        assert np.allclose(dB_dip, dBsim_dip)
        assert np.allclose(dB_cube, dBsim_dip)

        # include test for vector potential and vector potential gradient

    def test_100_vs_dipSimsopt(self): #100 magnets, 100 points, random location (but sufficiently far), magnetization and orientation. Testing everything       
        pos_points = np.random.uniform(150,1500, size = (100,3))
        signs = np.random.choice([-1,1], size = (100,3))
        points = pos_points * signs
        
        magPos = np.random.uniform(-10,10, size = (100,3))
        M = np.random.uniform(-2,2,size = (100,3))
        phiThetas = np.random.uniform(0,2*np.pi, size = (100,3))
        norms = np.random.uniform(-1,1, size = (100,3))
        dims = np.array([1,1,1])

        Bcube = B_direct(points, magPos, M, dims, phiThetas)
        Bsim_dip = dipole_field_B(points, magPos, M)
        assert np.allclose(Bcube, Bsim_dip)

        Bncube = Bn_direct(points, magPos, M, norms, dims, phiThetas)
        Bnsim_dip = simBn(points, magPos, M, norms)
        assert np.allclose(Bncube, Bnsim_dip)

        N = len(points)
        D = len(magPos)
        
        A_sim = dipole_field_Bn(points, magPos, norms, 1, 0, np.array([0,0,0])).reshape(N,3*D) #what about w/ nfp neq 0? b neq vec(0)?
        A_cube = Acube(points, magPos, norms, dims, phiThetas)
        assert np.allclose(A_sim, A_cube)

        BncubeMAT = Bn_fromMat(points, magPos, M, norms, dims, phiThetas)
        BnsimMAT = dipole_field_Bn(points, magPos, norms, 1, 0, np.array([0,0,0])).reshape(N,3*D) @ np.concatenate(M)
        assert np.allclose(BncubeMAT, BnsimMAT)
        
        dB_cube = gradr_Bcube(points, magPos, M, phiThetas, dims)
        dBsim_dip = dipole_field_dB(points, magPos, M)
        assert np.allclose(dB_cube, dBsim_dip)

        #lastly add vector potential check and vector potential gradient check

if __name__ == '__main__':
    unittest.main()

    