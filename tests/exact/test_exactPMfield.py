import unittest
import numpy as np

import simsopt.field as pycub #note gradient functions are not vectorized
import simsoptpp as sopp

def simBn(points, magPos, M, norms):
    N = len(norms)    
    B = sopp.dipole_field_B(points, magPos, M)    
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

        Bcube = sopp.B_direct(point, magPos, M, dims, phiTheta)
        Bsim_dip = sopp.dipole_field_B(point, magPos, M)
        assert np.allclose(Bcube, Bsim_dip, atol = 1e-15)

    def test_Bn_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        Bncube = sopp.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        Bnsim_dip = simBn(point, magPos, M, norm)
        assert np.allclose(Bncube, Bnsim_dip, atol = 1e-15)

    def test_Amatrix_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        N = len(norm)
        D = len(magPos)

        A_sim = sopp.dipole_field_Bn(point, magPos, norm, 1, 0, np.array([0,0,0])).reshape(N,3*D)
        A_cube = sopp.Acube(point, magPos, norm, dims, phiTheta, 1, 0)
        assert np.allclose(A_sim, A_cube, atol = 1e-15)

    def test_matrixFormulation_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])
        norm = np.array([[0,0,1]])

        N = len(norm)
        D = len(magPos)

        BncubeMAT = sopp.Bn_fromMat(point, magPos, M, norm, dims, phiTheta, 1, 0)
        Bncube = sopp.Bn_direct(point, magPos, M, norm, dims, phiTheta)
        BnsimMAT = sopp.dipole_field_Bn(point, magPos, norm, 1, 0, np.array([0,0,0])).reshape(N,3*D) @ np.concatenate(M)
        assert np.allclose(BncubeMAT, Bncube, atol = 1e-15)
        assert np.allclose(BncubeMAT, BnsimMAT, atol = 1e-15)

    def test_Bgrad_single(self):
        point = np.array([[0,0,100]])
        magPos = np.array([[0,0,0]])
        M = np.array([[0,0,1]])
        phiTheta = np.array([[np.radians(32),np.radians(144)]])
        dims = np.array([1,1,1])

        dB_cube = pycub.gradr_Bcube(point, magPos, M, phiTheta, dims)
        dB_dip = pycub.gradr_Bdip(point, magPos, M)
        dBsim_dip = sopp.dipole_field_dB(point, magPos, M)
        assert np.allclose(dB_cube, dB_dip, atol = 1e-15)
        assert np.allclose(dB_dip, dBsim_dip, atol = 1e-15)
        assert np.allclose(dB_cube, dBsim_dip, atol = 1e-15)

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

        Bcube = sopp.B_direct(points, magPos, M, dims, phiThetas)
        Bsim_dip = sopp.dipole_field_B(points, magPos, M)
        assert np.allclose(Bcube, Bsim_dip, atol = 1e-15)

        Bncube = sopp.Bn_direct(points, magPos, M, norms, dims, phiThetas)
        Bnsim_dip = simBn(points, magPos, M, norms)
        assert np.allclose(Bncube, Bnsim_dip, atol = 1e-15)

        N = len(points)
        D = len(magPos)
        
        A_sim = sopp.dipole_field_Bn(points, magPos, norms, 1, 0, np.array([0,0,0])).reshape(N,3*D) #what about w/ nfp neq 0? b neq vec(0)?
        A_cube = sopp.Acube(points, magPos, norms, dims, phiThetas, 1, 0)
        assert np.allclose(A_sim, A_cube, atol = 1e-15)

        BncubeMAT = sopp.Bn_fromMat(points, magPos, M, norms, dims, phiThetas, 1, 0)
        BnsimMAT = sopp.dipole_field_Bn(points, magPos, norms, 1, 0, np.array([0,0,0])).reshape(N,3*D) @ np.concatenate(M)
        assert np.allclose(BncubeMAT, BnsimMAT, atol = 1e-15)
        
        dB_cube = pycub.gradr_Bcube(points, magPos, M, phiThetas, dims)
        dBsim_dip = sopp.dipole_field_dB(points, magPos, M)
        assert np.allclose(dB_cube, dBsim_dip, atol = 1e-15)

        #lastly add vector potential check and vector potential gradient check

if __name__ == '__main__':
    unittest.main()

    
