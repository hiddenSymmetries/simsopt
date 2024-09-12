import unittest
import numpy as np

import sys
sys.path.append('/Users/willhoffman/simsopt/Codes')
import Bcube_nonVec as floop
import simsopt.field as vec

point = np.array([[0,0,100]])
magPos = np.array([[0,0,0]])
M = np.array([[0,0,1]])
phiTheta = np.array([[np.radians(32),np.radians(144)]])
dims = np.array([1,1,1])

Bcube4 = floop.B_direct(point, magPos, M, dims, phiTheta)
BcubeV = vec.B_direct(point, magPos, M, dims, phiTheta)
print(Bcube4)
print(BcubeV)

class Testing(unittest.TestCase):

    def test_B_field_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])

        Bcube4 = floop.B_direct(point, magPos, M, dims, phiTheta)
        BcubeV = vec.B_direct(point, magPos, M, dims, phiTheta)
        assert np.allclose(Bcube4, BcubeV, atol = 1e-15)

    
    def test_Bn_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        Bncube4 = floop.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BncubeV = vec.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        assert np.allclose(Bncube4, BncubeV, atol = 1e-15)

    
    def test_Amatrix_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        A_cube4 = floop.Acube(point, magPos, norm, dims, phiTheta)
        A_cubeV = vec.Acube(point, magPos, norm, dims, phiTheta)
        assert np.allclose(A_cube4, A_cubeV, atol = 1e-15)


    def test_matrixFormulation_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        BncubeMAT4 = floop.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
        Bncube4 = floop.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BncubeMATV = vec.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
        BncubeV = vec.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        assert np.allclose(BncubeMAT4, Bncube4, atol = 1e-15)
        assert np.allclose(BncubeMATV, BncubeV, atol = 1e-15)
        assert np.allclose(BncubeMATV, BncubeV, atol = 1e-15)


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
        BcubeV = vec.B_direct(points, magPos, M, dims, phiThetas)
        assert np.allclose(Bcube4, BcubeV, atol = 1e-15)

        Bncube4 = floop.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        BncubeV = vec.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        assert np.allclose(Bncube4, BncubeV, atol = 1e-15)

        A_cube4 = floop.Acube(points, magPos, norms, dims, phiThetas)
        A_cubeV = vec.Acube(points, magPos, norms, dims, phiThetas)
        assert np.allclose(A_cube4, A_cubeV, atol = 1e-15)

        BncubeMAT4 = floop.Bn_fromMat(points, magPos, M, norms, dims, phiThetas)
        BncubeMATV = vec.Bn_fromMat(points, magPos, M, norms, dims, phiThetas)
        assert np.allclose(BncubeMAT4, BncubeMATV, atol = 1e-15)


if __name__ == '__main__':
    unittest.main()
    