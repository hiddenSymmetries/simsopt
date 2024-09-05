import numpy as np
import simsoptpp as sopp
import sys
sys.path.append('/Users/willhoffman/simsopt/Codes')
import Bcube_nonVec as floop
import simsopt.field as vec
import time

N = 1000
D = 10000

pos_points = np.random.uniform(150,1500, size = (N,3))
signs = np.random.choice([-1,1], size = (N,3))
points = pos_points * signs

magPos = np.random.uniform(-10,10, size = (D,3))
M = np.random.uniform(-2,2,size = (D,3))
phiThetas = np.random.uniform(0,2*np.pi, size = (D,3))
norms = np.random.uniform(-1,1, size = (N,3))
dims = np.array([1,1,1])

# t41 = time.time()
# B4 = floop.B_direct(points, magPos, M, dims, phiThetas)
# t42 = time.time()
# print('for loop B_direct took t = ', t42 - t41,' s')

tv1 = time.time()
Bv = vec.B_direct(points, magPos, M, dims, phiThetas)
tv2 = time.time()
print('vectorized B_direct took t = ', tv2 - tv1,' s')

tc1 = time.time()
Bc = sopp.B_direct(points, magPos, M, dims, phiThetas)
tc2 = time.time()
print('c++ B_direct took t = ', tc2 - tc1,' s')