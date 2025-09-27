
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dims = np.array([0.0576, 0.0601, 0.0656])

data = pd.read_csv('bulk_RS_1k/obj_all.csv', delimiter = ',')

dist_ratio = data['poff'] / np.max(dims)

ex_ex = data['exact_exact']
ex_dip = data['exact_dipole']
dip_dip = data['dipole_dipole']
dip_ex = data['dipole_exact']

plt.plot(dist_ratio, ex_ex, marker = '.', label = 'exact')
plt.plot(dist_ratio, dip_ex, marker = '.', label = 'dipole')
plt.legend()
plt.grid(True)
plt.xlabel(r'distance ratio, $\frac{R}{d}$')
plt.ylabel(r'Sum of Squared Errors, $\|Ax-b\|^2$')
plt.yscale('log')
plt.title('Exact and Dipole Solutions vs Grid Distance for a 100-Iteration Relax-and-Split')

ax = plt.gca()
ax.invert_xaxis()

plt.show()
