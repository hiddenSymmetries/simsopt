#from turbulence.GX_io import GX_Runner
#from simsopt.turbulence import GX_io
from simsopt.turbulence.GX_io import GX_Output

import numpy as np
import matplotlib.pyplot as plt
import sys

'''
This script scans of GX output,
assumes an X-axis geometry manipulation

and makes plots of the EPA (exponential moving average)
using sqrt(EPV) (exponential movine variance) as an errorbar

for a handfull of tau parameters.

30 August 2022
Tony M. Qian
'''

f_list = []

for f in sys.argv[1:]:

    if f.find('restart') > 0:
        continue

    f_list.append(f)



gx_outs = [GX_Output(f) for f in f_list]
q_med = np.array([ g.median_estimator() for g in gx_outs ])

plt.figure()

#X = np.linspace(0,1, len(f_list))
X = np.linspace(-1,1.5,20)


colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8']

j = 0
for tau in [200,100,50]:

    q_avg, q_var = np.transpose( [ g.exponential_window_estimator(tau=tau) for g in gx_outs ] )
    dq = np.sqrt(q_var) # converts a varience to standard deviation

    col = colors[j]
    plt.plot(X,q_avg,col+'o-',label=rf'EMA $\tau = {tau}$')
    plt.fill_between(X, q_avg-dq, q_avg+dq, alpha=0.3, color=col)

    j += 1
#plt.errorbar(X,q_avg,yerr=dq,fmt='.-',label='exponential moving average');

plt.plot(X,q_med,'kx',label='median of medians'); 
plt.ylabel('GX Nonlinear Heat Flux')
plt.xlabel('Boundary ZBS (m=0, n=1)')
plt.legend(fontsize=10)
plt.show()

import pdb
pdb.set_trace()

