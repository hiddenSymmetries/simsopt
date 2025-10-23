
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# rep = 'bulkGPMO'
rep = 'bulk_RS_1k'

pick = 3

# data = pd.read_csv(rep+'/objectives.csv', delimiter = ',')
data = pd.read_csv(rep+'/obj_all.csv', delimiter = ',')
dip_dip = data['dipole_dipole']
dip_ex = data['dipole_exact']
ex_ex = data['exact_exact']
ex_dip = data['exact_dipole']

fex = ex_ex[pick]
fex_d = ex_dip[pick]
fdip_d = dip_dip[pick]
fdip = dip_ex[pick]

A_ex = np.load(rep+f'/Anb_objs/Aexact{pick}.npy')
b_ex = np.load(rep+f'/Anb_objs/bexact{pick}.npy')
A_dip = np.load(rep+f'/Anb_objs/Adip{pick}.npy')
b_dip = np.load(rep+f'/Anb_objs/bdip{pick}.npy')

print(fex)
print(fdip)

dip_m = np.load(rep+f'/opt_mvecs/dip{pick}.npy')
ex_m = np.load(rep+f'/opt_mvecs/exact{pick}.npy')

lamb = 0 # values output by the bulk runs are just fB, the errors in the magnetic field, not the full optimization
# objective with regularization, so we shouldn't model it either

def numerical_grad(m_ex, m_dip, f_ex, f_dip): # slope from a point on fex to a point on fdip
    dm = m_ex - m_dip
    dfdm = np.zeros_like(dm)
    dmsq = np.dot(dm,dm)
    
    dfdm = (f_ex-f_dip)/dmsq * dm
    print(dfdm.shape)
    return dfdm

# def analytic_grad(m, A, b, lamb=lamb, algorithm='GPMO'): # slope of either fex or fdip itself, not the same thing as numerical
#     if algorithm == 'GPMO':
#         dfdm = np.dot(A.T, (np.dot(A,m)-b)) + lamb * m
#     if algorithm == 'RS':
#         dfdm = np.dot(A.T, (np.dot(A,m)-b))
#     return dfdm

def fB(m, A, b, lamb=lamb):
    x = A @ m.T - b[:, None]
    print('b shape ',b.shape)
    print('A shape ',A.shape)
    print('m shape ',m.shape)
    print('x shape is ',x.shape)
    return 0.5 * np.sum((A @ m.T - b[:, None])**2, axis=0)

dfdm = numerical_grad(ex_m, dip_m, fex, fdip)

eps = 1e-2
density = 10000
dm = ex_m - dip_m
t = np.linspace(1.0 + eps, 0.0 - eps, density)
# m_range = dip_m[None,:] + np.linspace(0.0 - eps, 1.0 + eps, density)[:,None] * dm[None,:]
m_range = dip_m[None,:] + t[:,None] * dm[None,:]

fB_ex = fB(m_range, A_ex, b_ex)
fB_dip = fB(m_range, A_dip, b_dip)
f_line = fdip + dfdm @ (m_range - dip_m).T

# print(fB_ex[-1], fex)
# print(fB_dip[0], fdip)

mopt_dip = m_range[np.argmin(fB_dip)]
mopt_ex = m_range[np.argmin(fB_ex)]

print('mopts are ', mopt_ex, mopt_dip)
assert all(mopt_ex == mopt_dip)

plt.plot(t, fB_ex, label = 'exact fB')
plt.plot(t, fB_dip, label = 'dipole fB')
plt.plot(t, f_line, label = 'line')
plt.scatter(1.0, fex, label = 'exact_exact')
plt.scatter(1.0, fex_d, label = 'exact_dipole')
plt.scatter(0.0, fdip_d, label = 'dipole_dipole')
plt.scatter(0.0, fdip, label = 'dipole_exact')

plt.legend()
plt.grid(True)
plt.show()

