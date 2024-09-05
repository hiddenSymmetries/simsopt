import unittest
import numpy as np

import sys
sys.path.append('/Users/willhoffman/simsopt/Codes')
import Bcube_nonVec as floop
import simsoptpp as sopp

point = np.array([[0,0,100]])
magPos = np.array([[0,0,0]])
M = np.array([[0,0,1]])
phiTheta = np.array([[np.radians(32),np.radians(144)]])
dims = np.array([1,1,1])
norm = np.array([[0,0,1]])

import time

Bcube4 = floop.B_direct(point, magPos, M, dims, phiTheta)
t0 = time.time()
BcubeC = sopp.B_direct(point, magPos, M, dims, phiTheta)
t1 = time.time()
print('c++ took', t1 - t0, 'seconds')
print('B python = ', Bcube4)
print('B c++ = ', BcubeC)

A_cube4 = floop.Acube(point, magPos, norm, dims, phiTheta)
A_cubeC = sopp.Acube(point, magPos, norm, dims, phiTheta)
print('A python = ', A_cube4)
print('A c++ = ', A_cubeC)

BncubeMAT4 = floop.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
BncubeMATC = sopp.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
print('Bn_mat python = ', BncubeMAT4)
print('Bn_mat c++ = ', BncubeMATC)

"""
class Testing(unittest.TestCase):

    def Htest(self):
        r = np.array([0,0,100])
        dims = np.array([1,1,1])

        Hcube4 = floop.Hd_i_prime(r, dims)
        HcubeC = sopp.Hd_i_prime(r, dims)
        assert np.allclose(Hcube4, HcubeC)

    def test_B_field_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])

        Bcube4 = floop.B_direct(point, magPos, M, dims, phiTheta)
        BcubeC = sopp.B_direct(point, magPos, M, dims, phiTheta)
        assert np.allclose(Bcube4, BcubeC)

    
    def test_Bn_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        Bncube4 = floop.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BncubeC = sopp.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        assert np.allclose(Bncube4, BncubeC)

    
    def test_Amatrix_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        A_cube4 = floop.Acube(point, magPos, norm, dims, phiTheta)
        A_cubeC = sopp.Acube(point, magPos, norm, dims, phiTheta)
        assert np.allclose(A_cube4, A_cubeC)


    def test_matrixFormulation_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        BncubeMAT4 = floop.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
        Bncube4 = floop.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BncubeMATV = sopp.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
        BncubeC = sopp.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        assert np.allclose(BncubeMAT4, Bncube4)
        assert np.allclose(BncubeMATV, BncubeC)
        assert np.allclose(BncubeMATV, BncubeC)


    def test_100_vs_dipSimsopt(self): #100 magnets, 100 points, random location (but sufficiently far), magnetization and orientation. Testing everything       
        pos_points = np.random.uniform(150,1500, size = (100,3))
        signs = np.random.choice([-1,1], size = (100,3))
        points = pos_points * signs
        
        magPos = np.random.uniform(-10,10, size = (100,3))
        M = np.random.uniform(-2,2,size = (100,3))
        phiThetas = np.random.uniform(0,2*np.pi, size = (100,3))
        norms = np.random.uniform(-1,1, size = (100,3))
        dims = np.array([1,1,1])

        Bcube4 = floop.B_direct(points, magPos, M, dims, phiThetas)
        BcubeC = sopp.B_direct(points, magPos, M, dims, phiThetas)
        assert np.allclose(Bcube4, BcubeC)

        Bncube4 = floop.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        BncubeC = sopp.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        assert np.allclose(Bncube4, BncubeC)

        A_cube4 = floop.Acube(points, magPos, norms, dims, phiThetas)
        A_cubeC = sopp.Acube(points, magPos, norms, dims, phiThetas)
        assert np.allclose(A_cube4, A_cubeC)

        BncubeMAT4 = floop.Bn_fromMat(points, magPos, M, norms, dims, phiThetas)
        BncubeMATV = sopp.Bn_fromMat(points, magPos, M, norms, dims, phiThetas)
        assert np.allclose(BncubeMAT4, BncubeMATV)


if __name__ == '__main__':
    unittest.main()
"""
