# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains functions related to MPI parallelization.
"""

import numpy as np
from mpi4py import MPI
import logging

STOP = 0
CALCULATE_F = 1
CALCULATE_JAC = 2
CALCULATE_FD_JAC = 3

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

class MpiPartition():
    def __init__(self, ngroups=None, comm_world=MPI.COMM_WORLD):
        self.together = True
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
            
    def mobilize_leaders(self, x):
        logger.debug('mobilize_leaders, x={}'.format(x))
        if not self.proc0_world:
            raise RuntimeError('Only proc 0 should call mobilize_leaders()')

        self.comm_leaders.bcast(CALCULATE_FD_JAC, root=0)

        # Next, broadcast the state vector to leaders:
        self.comm_leaders.bcast(x, root=0)

    def mobilize_workers(self, x, action):
        logger.debug('mobilize_workers, action={}, x={}'.format(action, x))
        if not self.proc0_groups:
            raise RuntimeError('Only group leaders should call mobilize_workers()')

        # First, notify workers that we will be doing a calculation:
        if action != CALCULATE_F and action != CALCULATE_JAC:
            raise ValueError('action must be either CALCULATE_F or CALCULATE_JAC')
        self.comm_groups.bcast(action, root=0)

        # Next, broadcast the state vector to workers:
        self.comm_groups.bcast(x, root=0)

    def stop_leaders(self):
        logger.debug('stop_leaders')
        if not self.proc0_world:
            raise RuntimeError('Only proc0_world should call stop_leaders()')

        data = STOP
        self.comm_leaders.bcast(data, root=0)
        self.together = True

    def stop_workers(self):
        logger.debug('stop_workers')
        if not self.proc0_groups:
            raise RuntimeError('Only proc0_groups should call stop_workers()')

        data = STOP
        self.comm_groups.bcast(data, root=0)
        self.together = True

    def leaders_loop(self, dofs):
        if self.proc0_world:
            logger.debug('proc0_world bypassing leaders_loop')
            return
        
        logger.debug('entering leaders_loop')

        # x is a buffer for receiving the state vector:
        x = np.empty(dofs.nparams, dtype='d')

        while True:
            # Wait for proc 0 to send us something:
            data = None
            data = self.comm_leaders.bcast(data, root=0)
            logger.debug('leaders_loop received {}'.format(data))
            if data == STOP:
                # Tell workers to stop
                #self.comm_groups.bcast(STOP, root=0)
                break

            # If we make it here, we must be doing a fd_jac_par
            # calculation, so receive the state vector: mpi4py has
            # separate bcast and Bcast functions!!  comm.Bcast(x,
            # root=0)
            x = self.comm_leaders.bcast(x, root=0)
            logger.debug('leaders_loop x={}'.format(x))
            dofs.set(x)

            dofs.fd_jac_par(self)

        logger.debug('leaders_loop end')

    def worker_loop(self, dofs):
        self.together = False
        
        if self.proc0_groups:
            logger.debug('bypassing worker_loop since proc0_groups')
            return
        
        logger.debug('entering worker_loop')

        # x is a buffer for receiving the state vector:
        x = np.empty(dofs.nparams, dtype='d')

        while True:
            # Wait for the group leader to send us something:
            data = None
            data = self.comm_groups.bcast(data, root=0)
            logger.debug('worker_loop worker received {}'.format(data))
            if data == STOP:
                break

            # If we make it here, we must be doing a calculation, so
            # receive the state vector:
            # mpi4py has separate bcast and Bcast functions!!
            #comm.Bcast(x, root=0)
            x = self.comm_groups.bcast(x, root=0)
            logger.debug('worker_loop worker x={}'.format(x))
            dofs.set(x)

            # We don't store or do anything with f() or jac(), because
            # the group leader will handle that.
            if data == CALCULATE_F:
                dofs.f()
            elif data == CALCULATE_JAC:
                dofs.jac()
            else:
                raise ValueError('Unexpected data in worker_loop')

        logger.debug('worker_loop end')
        self.together = True

