import unittest
import numpy as np

import simsopt.field as pycub
import simsoptpp as sopp

# note test_exactPMfield has been changed to check c++ code against existing simsopt dipole functions
# unless doing something specific with python code, this test is obsolete
class Testing(unittest.TestCase):

    def Ptest(self):
        phiTheta = np.array([[np.radians(32),np.radians(144)]])

        P4 = pycub.Pd(phiTheta[0,0], phiTheta[0,1])
        PC = sopp.Pd(phiTheta[0,0], phiTheta[0,1])
        assert np.allclose(P4, PC, atol = 1e-15)
    
    def Htest(self):
        r = np.array([0,0,100])
        dims = np.array([1,1,1])

        Hcube4 = pycub.Hd_i_prime(r, dims)
        HcubeC = sopp.Hd_i_prime(r, dims)
        assert np.allclose(Hcube4, HcubeC, atol = 1e-15)

    def test_B_field_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])

        Bcube4 = pycub.B_direct(point, magPos, M, dims, phiTheta)
        BcubeC = sopp.B_direct(point, magPos, M, dims, phiTheta)
        assert np.allclose(Bcube4, BcubeC, atol = 1e-15)

    
    def test_Bn_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        Bncube4 = pycub.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BncubeC = sopp.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        assert np.allclose(Bncube4, BncubeC, atol = 1e-15)

    
    def test_Amatrix_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        A_cube4 = pycub.Acube(point, magPos, norm, dims, phiTheta)
        A_cubeC = sopp.Acube(point, magPos, norm, dims, phiTheta, 1, 0)
        assert np.allclose(A_cube4, A_cubeC, atol = 1e-15)


    def test_matrixFormulation_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        BncubeMAT4 = pycub.Bn_fromMat(point, magPos, M, norm, dims, phiTheta)
        Bncube4 = pycub.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BncubeMATV = sopp.Bn_fromMat(point, magPos, M, norm, dims, phiTheta, 1, 0)
        BncubeC = sopp.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        assert np.allclose(BncubeMAT4, Bncube4, atol = 1e-15)
        assert np.allclose(BncubeMATV, BncubeC, atol = 1e-15)
        assert np.allclose(BncubeMATV, BncubeC, atol = 1e-15)


    def test_100_vs_dipSimsopt(self): #100 magnets, 100 points, random location (but sufficiently far), magnetization and orientation. Testing everything       
        pos_points = np.random.uniform(150,1500, size = (100,3))
        signs = np.random.choice([-1,1], size = (100,3))
        points = pos_points * signs
        
        magPos = np.random.uniform(-10,10, size = (100,3))
        M = 1e10 * np.random.uniform(-2,2,size = (100,3))
        phiThetas = np.random.uniform(0,2*np.pi, size = (100,3))
        norms = np.random.uniform(-1,1, size = (100,3))
        dims = np.array([1,1,1])

        Bcube4 = pycub.B_direct(points, magPos, M, dims, phiThetas)
        BcubeC = sopp.B_direct(points, magPos, M, dims, phiThetas)
        assert np.allclose(Bcube4, BcubeC, atol = 1e-15)

        Bncube4 = pycub.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        BncubeC = sopp.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        assert np.allclose(Bncube4, BncubeC, atol = 1e-15)

        A_cube4 = pycub.Acube(points, magPos, norms, dims, phiThetas)
        A_cubeC = sopp.Acube(points, magPos, norms, dims, phiThetas, 1, 0)
        assert np.allclose(A_cube4, A_cubeC, atol = 1e-15)

        BncubeMAT4 = pycub.Bn_fromMat(points, magPos, M, norms, dims, phiThetas)
        BncubeMATV = sopp.Bn_fromMat(points, magPos, M, norms, dims, phiThetas, 1, 0)
        assert np.allclose(BncubeMAT4, BncubeMATV, atol = 1e-15)


if __name__ == '__main__':
    unittest.main()
