import builtins
import os
import sys 

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    verbose = (comm.rank == 0)
except ImportError:
    comm = None
    verbose = True

def print(*args, **kwargs):
    r"""
    Overloaded print function to force flushing of stdout.
    Only proc0 prints to stdout. 
    """
    if verbose:
        builtins.print(*args, **kwargs)
        os.fsync(sys.stdout)