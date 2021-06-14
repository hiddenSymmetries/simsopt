#!/usr/bin/env python3

#import mpi4py
#mpi4py.rc(initialize=False, finalize=False)
import sys
sys.path.append('../../modules/VMEC')
from mpi4py import MPI
import numpy as np
from vmec_class import VMEC

"""
Perform several runs with the VMEC python wrapper
while changing a particular surface Fourier coefficient.
"""

# initial settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Hello from rank %s of %s." % (rank, size))
#print(MPI.Is_initialized())
#print(MPI.Is_finalized())

fcomm = comm.py2f()
path = os.path.join(os.path.dirname(__file__), 'inputs', 'input.QAS')
if rank == 0:
    verbose = True
else:
    verbose = False
QAS = VMEC(input_file=path, verbose=verbose, comm=fcomm)

# save original data
rbc = np.copy(QAS.indata.rbc)
zbs = np.copy(QAS.indata.zbs)
raxis_cc = np.copy(QAS.indata.raxis_cc)
zaxis_cs = np.copy(QAS.indata.zaxis_cs)

# run & load
QAS.run(iseq=rank)
QAS.load()
if rank == 0:
    print('First run status: ', QAS.success, QAS.ictrl)
    print('rmnc[0,-1] = ', QAS.wout.rmnc[0, -1])

# second run
comm.Barrier()
if rank == 0:
    print('\nBegin the second run with RBC(0,0)=1.5')

# reset data
QAS.indata.rbc = np.copy(rbc)
QAS.indata.zbs = np.copy(zbs)
QAS.indata.raxis_cc = np.copy(raxis_cc)
QAS.indata.zaxis_cs = np.copy(zaxis_cs)

# revise data, run, check
QAS.indata.rbc[101, 0] = 1.5
QAS.reinit()
QAS.run(iseq=rank)
QAS.load()
if rank == 0:
    print('Second run status: ', QAS.success, QAS.ictrl)
    print('rmnc[0,-1] = ', QAS.wout.rmnc[0, -1])

# third run
comm.Barrier()
if rank == 0:
    print('\nBegin the third run with RBC(0,0) = original value and muted')

# reset data
QAS.indata.rbc = np.copy(rbc)
QAS.indata.zbs = np.copy(zbs)
QAS.indata.raxis_cc = np.copy(raxis_cc)
QAS.indata.zaxis_cs = np.copy(zaxis_cs)
QAS.verbose = False
QAS.reinit()

# run and load
QAS.run(iseq=rank)
QAS.load()
if rank == 0:
    print('Third run status: ', QAS.success, QAS.ictrl)
    print('rmnc[0,-1] = ', QAS.wout.rmnc[0, -1])
    print('Finished!')
