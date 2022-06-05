import numpy as np
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset
#from scipy.io import netcdf as nc

'''
    This script takes a list of GX outputs,
    and computes qavg from q(t) for each.

    It uses three methods
    1) median of medians
    2) exponential moving average
    3) histogram

    usage: python qflux.py [list of .nc files]')

    Updated 5 June 2022
    tqian@pppl.gov
'''



### pull GX data
data = []
fins = []
for fname in sys.argv[1:]:
#for fname in fins:

    try:

        f = Dataset(fname, mode='r')
        #f = nc.netcdf_file(fname, 'r')  # scipy.io.netcdf only reads netcdf3
        data.append(f)
        fins.append(fname)
        print(' read', fname)
    
    except:
        print(' error reading', fname)
        print(' usage: python qflux.py [list of .nc files]')

fig, axs = plt.subplots(1,3,figsize=(12,4))
cols = ['C0','C1','C2','C3','C4','C5','C6','C7']

j = 0
flux = []
for f in data:

    time = f.variables['time'][:]
    qflux = f.groups['Fluxes'].variables['qflux'][:,0]
    # check for nans
    if ( np.isnan(qflux).any() ):
         print('  nans found')
         qflux = np.nan_to_num(qflux)
    # fix nan_to_num here
    axs[0].plot(time,qflux,'.-',label=fins[j])

    # median of a sliding median
    N = len(qflux)
    arr_med = [ np.median( qflux[::-1][:k] ) for k in np.arange(1,N)] 
    med = np.median( arr_med )
    axs[0].axhline(med, color=cols[j%8], ls='--')
    axs[0].plot(time[::-1][:-1],arr_med)
    flux.append(med)

    # make histogram
    axs[2].hist(qflux,bins=50)
    axs[2].grid()
    j+=1

arr = []
median_estimator = []
for i in np.arange(N):
    #arr.append( np.median( [qflux[:i][::-1][:k] for k in np.arange(1,i)] ) )
    #arr.append( np.median( [qflux[:i][::-1][:k] for k in np.arange(1,i)] ) )
    #median_estimator.append( np.median(arr) )
    median_estimator.append( np.median(qflux[:i]) )



print(flux)
plt.suptitle(fins[0])
#plt.yscale('log')
axs[0].set_ylabel('qflux')
axs[0].set_xlabel('time')
axs[0].grid()
#axs[0].legend()
#plt.show()


# Bill's method
tau = 50
t0 = 0
qavg = 0
var_qavg = 0

Q_avg = []
Var_Q_avg = []

for k in np.arange(N):

    q = qflux[k]
    t = time [k]

    gamma = (t - t0)/tau
    alpha = np.e**( - gamma)
    delta = q - qavg

    qavg = alpha * qavg + q * (1 - alpha)
    var_qavg = alpha * ( var_qavg + (1-alpha)* delta**2)
    t0 = t

    #print(qavg, var_qavg)
    Q_avg.append(qavg)
    Var_Q_avg.append(var_qavg)

axs[1].plot(Q_avg, label='avg')
axs[1].plot(Var_Q_avg, label='var')
axs[1].plot(qflux, label='signal')
axs[1].legend()
axs[1].grid()

plt.tight_layout()
plt.show()

