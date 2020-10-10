"""
"""

import numpy as np
from mpi4py import MPI
import logging

STOP = 0
CALCULATE_F = 1
CALCULATE_JAC = 2

logger = logging.getLogger(__name__)

def proc0():
    """
    Return True if this processor has rank 0 in MPI_COMM_WORLD, else
    return False.
    """
    return MPI.COMM_WORLD.Get_rank() == 0

def mobilize_workers(x, action):
    logger.debug('[{}] mobilize_workers, action={}, x={}'.format(MPI.COMM_WORLD.Get_rank(), action, x))
    if not proc0():
        raise RuntimeError('Only proc 0 should call mobilize_workers()')

    comm = MPI.COMM_WORLD

    # First, notify workers that we will be doing a calculation:
    if action != CALCULATE_F and action != CALCULATE_JAC:
        raise ValueError('action must be either CALCULATE_F or CALCULATE_JAC')
    comm.bcast(action, root=0)

    # Next, broadcast the state vector to workers:
    comm.bcast(x, root=0)

def stop_workers():
    logger.debug('[{}] stop_workers'.format(MPI.COMM_WORLD.Get_rank()))
    if not proc0():
        raise RuntimeError('Only proc 0 should call stop_workers()')

    comm = MPI.COMM_WORLD
    data = STOP
    comm.bcast(data, root=0)

def worker_loop(dofs):
    logger.debug('[{}] entering worker_loop'.format(MPI.COMM_WORLD.Get_rank()))
    if proc0():
        raise RuntimeError('Proc 0 should not call worker_loop()')

    comm = MPI.COMM_WORLD
    # x is a buffer for receiving the state vector:
    x = np.empty(dofs.nparams, dtype='d')

    while True:
        # Wait for proc 0 to send us something:
        data = None
        data = comm.bcast(data, root=0)
        logger.debug('[{}] worker_loop received {}'.format(MPI.COMM_WORLD.Get_rank(), data))
        if data == STOP:
            logger.debug('[{}] worker_loop stopping'.format(MPI.COMM_WORLD.Get_rank()))
            break
        
        # If we make it here, we must be doing a calculation, so
        # receive the state vector:
        # mpi4py has separate bcast and Bcast functions!!
        #comm.Bcast(x, root=0)
        x = comm.bcast(x, root=0)
        logger.debug('[{}] worker_loop x={}'.format(MPI.COMM_WORLD.Get_rank(), x))
        dofs.set(x)

        if data == CALCULATE_F:
            dofs.f()
        elif data == CALCULATE_JAC:
            dofs.jac()
        else:
            raise ValueError('Unexpected data in worker_loop')
        
    logger.debug('[{}] worker_loop end'.format(MPI.COMM_WORLD.Get_rank()))
