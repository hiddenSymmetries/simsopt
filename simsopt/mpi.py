"""
This module contains functions related to MPI parallelization.
"""

import numpy as np
from mpi4py import MPI
import logging

STOP = 0
CALCULATE_F = 1
CALCULATE_JAC = 2

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

def proc0():
    """
    Return True if this processor has rank 0 in MPI_COMM_WORLD, else
    return False.
    """
    return MPI.COMM_WORLD.Get_rank() == 0

def mobilize_workers(x, action):
    logger.debug('mobilize_workers, action={}, x={}'.format(action, x))
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
    logger.debug('stop_workers')
    if not proc0():
        raise RuntimeError('Only proc 0 should call stop_workers()')

    comm = MPI.COMM_WORLD
    data = STOP
    comm.bcast(data, root=0)

def worker_loop(dofs):
    logger.debug('entering worker_loop')
    if proc0():
        raise RuntimeError('Proc 0 should not call worker_loop()')

    comm = MPI.COMM_WORLD
    # x is a buffer for receiving the state vector:
    x = np.empty(dofs.nparams, dtype='d')

    while True:
        # Wait for proc 0 to send us something:
        data = None
        data = comm.bcast(data, root=0)
        logger.debug('worker_loop received {}'.format(data))
        if data == STOP:
            break
        
        # If we make it here, we must be doing a calculation, so
        # receive the state vector:
        # mpi4py has separate bcast and Bcast functions!!
        #comm.Bcast(x, root=0)
        x = comm.bcast(x, root=0)
        logger.debug('worker_loop x={}'.format(x))
        dofs.set(x)

        if data == CALCULATE_F:
            dofs.f()
        elif data == CALCULATE_JAC:
            dofs.jac()
        else:
            raise ValueError('Unexpected data in worker_loop')
        
    logger.debug('worker_loop end')

class MpiPartition():
    def __init__(self, ngroups=None, comm_world=MPI.COMM_WORLD):
        self.comm_world = comm_world
        self.rank_world = comm_world.Get_rank()
        self.nprocs_world = comm_world.Get_size()
        self.proc0_world = (self.rank_world == 0)
        
        if ngroups is None:
            ngroups = self.nprocs_world
        # Force ngroups to be in the range [1, nprocs_world]
        if ngroups < 1:
            ngroups = 1
            logger.info('Raising ngroups to 1')
        if ngroups > self.nprocs_world:
            ngroups = self.nprocs_world
            logger.info('Lowering ngroups to {}'.format(ngroups))
        self.ngroups = ngroups

        self.group = int(np.floor((self.rank_world * ngroups) / self.nprocs_world))

        # Set up the "groups" communicator:
        self.comm_groups = self.comm_world.Split(color=self.group, key=self.rank_world)
        self.rank_groups = self.comm_groups.Get_rank()
        self.nprocs_groups = self.comm_groups.Get_size()
        self.proc0_groups = (self.rank_groups == 0)

        # Set up the "leaders" communicator:
        if self.proc0_groups:
            color = 0
        else:
            color = MPI.UNDEFINED
        self.comm_leaders = self.comm_world.Split(color=color, key=self.rank_world)
        if self.proc0_groups:
            self.rank_leaders = self.comm_leaders.Get_rank()
            self.nprocs_leaders = self.comm_leaders.Get_size()
        else:
            # We are not allowed to query the rank from procs that are
            # not members of comm_leaders.
            self.rank_leaders = -1
            self.nprocs_leaders = -1

    def write(self):
        """ Dump info about the MPI configuration """
        columns = ["rank_world","nprocs_world","group","ngroups","rank_groups","nprocs_groups","rank_leaders","nprocs_leaders"]
        data = [self.rank_world , self.nprocs_world , self.group , self.ngroups , self.rank_groups , self.nprocs_groups , self.rank_leaders , self.nprocs_leaders]

        # Each processor sends their data to proc0_world, and
        # proc0_world writes the result to the file in order.
        if self.proc0_world:
            # Print header row
            width = max(len(s) for s in columns) + 1
            print(",".join(s.rjust(width) for s in columns))
            print(",".join(str(s).rjust(width) for s in data))
            for tag in range(1, self.nprocs_world):
                data = self.comm_world.recv(tag=tag)
                print(",".join(str(s).rjust(width) for s in data))
        else:
            tag = self.rank_world
            self.comm_world.send(data, 0, tag)
            
