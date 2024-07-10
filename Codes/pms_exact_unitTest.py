import numpy as np
import sympy as sp
from Bgrad__7_1_24 import gradr_Bcube, gradr_Bdip
from Bcube__7_1_24 import B_direct, Bn_direct, Bn_fromMat, Bdip_direct, Bndip_direct, Bndip_fromMat
import matplotlib.pyplot as plt

mu0 = 4*np.pi*10**-7
dim = np.array([1,1,1])


def UnitTest(a,b,type):
    if (abs(a-b) <= 0.01 * (abs(a) + 1e-20)).all() and (abs(a-b) <= 0.01 * (abs(b) + 1e-20)).all():
        print(str(type) + ' test PASSED')
    else:
        print(str(type) + ' test FAILED')


points = np.array([[0,0,100]])
magPos = np.array([[0,0,0]])
norms = np.array([[0,0,1]])
M = np.array([[0,0,1]])
phithetas = np.array([[sp.rad(16),sp.rad(134)]])

cubeNorm = Bn_direct(points, magPos, M, norms, dim, phithetas)
cubeNorm_fromMat = Bn_fromMat(points, magPos, M, norms, dim, phithetas)
dipNorm = Bndip_direct(points, magPos, M, norms)
dipNorm_fromMat = Bndip_fromMat(points, magPos, M, norms)
print('Bnorm for cube = ',cubeNorm_fromMat)
print('Bnorm for dipole = ',dipNorm_fromMat)
UnitTest(cubeNorm_fromMat,dipNorm_fromMat,'Bnorm_mat')



cubeB = B_direct(points, magPos, M, dim, phithetas)
dipB = Bdip_direct(points, magPos, M)
print('B for cube = ',cubeB)
print('B for dipole = ',dipB)
UnitTest(cubeB,dipB,'B field')



dCube = gradr_Bcube(points, magPos, M, phithetas, dim)
dDip = gradr_Bdip(points, magPos, M)
print('grad B for cube = ',dCube)
print('grad B for dipole = ',dDip)
UnitTest(dCube,dDip,'grad B')



#Bgrad Plots
dim = np.array([1,1,1])
ran = np.linspace(-2,2,1000)
M = np.array([[0,0,1]])
phiTheta = np.array([[sp.rad(0),sp.rad(90)]])
cubeGrad = []
dipGrad = []
Brem = mu0*np.linalg.norm(M[0])
for r in ran:
    point = np.array([[0,0,r]])
    cubeGrad.append(gradr_Bcube(point,magPos,M,phiTheta,dim)[0][2][2]/Brem)
    dipGrad.append(gradr_Bdip(point,magPos,M)[0][2][2]/Brem)

plt.plot(ran/(dim[2]/2),cubeGrad,c='orange',label='B grad cube')
plt.plot(ran/(dim[2]/2),dipGrad,c='purple',label='B grad dip')

#Bfield Plots
field_strength_dip = []
field_strength_cub = []
for r in ran:
    axial_dist = np.array([[0,0,r]])
    field_strength_dip.append(Bdip_direct(axial_dist,magPos,M)[0][2]/Brem)
    field_strength_cub.append(B_direct(axial_dist,magPos,M,dim,phiTheta)[0][2]/Brem)


plt.plot(ran/(dim[2]/2),field_strength_dip,c='b',label='B dip')
plt.plot(ran/(dim[2]/2),field_strength_cub,c='r',label='B cube')
plt.axvline(-1,c='black',linestyle='--')
plt.axvline(1,c='black',linestyle='--')
plt.ylim(-2.5,2.5)
plt.ylabel('B Field Strength')
plt.xlabel('Axial Distance From Magnet')
plt.title('Magnet Field Strength and Gradient vs On-Axis Distance')
plt.legend()
plt.grid()
#plt.show()

