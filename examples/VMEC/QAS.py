#import mpi4py
#mpi4py.rc(initialize=False, finalize=False)
import sys
sys.path.append('../../modules/VMEC')
from mpi4py import MPI
import numpy as np
from vmec_class import VMEC

# initial settings
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print("Hello from rank %s of %s." % (rank, size))
#print(MPI.Is_initialized())
#print(MPI.Is_finalized())

fcomm = comm.py2f()
path = 'input.QAS'
QAS = VMEC(input_file=path, verbose=True, comm=fcomm)
for i in range(size):
    comm.Barrier()
    if rank == i:
        print('the first run', rank, QAS.ictrl)
#QAS.run(mode='input',iseq=rank)
QAS.run(iseq=rank)
for i in range(size):
    comm.Barrier()
    if rank == i:
        print('the second run', rank, QAS.ictrl)
QAS.reinit()
QAS.run(iseq=rank)
for i in range(size):
    comm.Barrier()
    if rank == i:
        print('the third run', rank, QAS.ictrl)
QAS.reinit()
QAS.run(iseq=rank)
print('exit', rank, QAS.ictrl, ', success: ', QAS.success)

