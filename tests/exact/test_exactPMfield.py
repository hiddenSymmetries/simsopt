import unittest
import numpy as np
import sympy as sp

import sys
sys.path.append('/Users/willhoffman/simsopt/Codes')
from Bcube__7_1_24 import Pdsym, B_direct, Bn_direct, Bn_fromMat, Bdip_direct, Bndip_direct, Bndip_fromMat, Adip, Acube
from Bgrad__7_1_24 import gradr_Bcube, gradr_Bdip
from simsoptpp import dipole_field_Bn, dipole_field_B, dipole_field_dB


def simBn(points, magPos, M, norms):
    N = len(norms)
    
    B = dipole_field_B(points, magPos, M)
    Bs = np.sum(dipole_field_Bn(points, magPos, norms, 1, 0, np.array([0,0,0])), 1)
    assert (B == Bs).all()
    
    Bn = np.zeros(N)
    for n in range(N):
        Bn[n] = B[n] @ norms[n]

    return Bn
    

class Testing(unittest.TestCase):

    def test_B_field_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[sp.rad(32),sp.rad(144)]])
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
        phiTheta = np.array([[sp.rad(32),sp.rad(144)]])
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
        phiTheta = np.array([[sp.rad(32),sp.rad(144)]])
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
        phiTheta = np.array([[sp.rad(32),sp.rad(144)]])
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
        phiTheta = np.array([[sp.rad(32),sp.rad(144)]])
        dims = np.array([1,1,1])

        dB_cube = gradr_Bcube(point, magPos, M, phiTheta, dims)
        dB_dip = gradr_Bdip(point, magPos, M)
        dBsim_dip = dipole_field_dB(point, magPos, M)

        assert np.allclose(dB_cube, dB_dip)
        assert np.allclose(dB_dip, dBsim_dip)
        assert np.allclose(dB_cube, dBsim_dip)

if __name__ == '__main__':
    unittest.main()

    