# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains the MpiPartition class.

This module should be completely self-contained, depending only on
mpi4py and numpy, not on any other simsopt components.
"""

import numpy as np
from mpi4py import MPI
import logging

STOP = 0

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

class MpiPartition():
    """
    This module contains functions related to dividing up the set of
    MPI processors into groups, each of which can work together.
    """
    def __init__(self, ngroups=None, comm_world=MPI.COMM_WORLD):
        self.is_apart = False
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
            
    def mobilize_leaders(self, action_const):
        logger.debug('mobilize_leaders, action_const={}'.format(action_const))
        if not self.proc0_world:
            raise RuntimeError('Only proc0_world should call mobilize_leaders()')

        self.comm_leaders.bcast(action_const, root=0)

    def mobilize_workers(self, action_const):
        logger.debug('mobilize_workers, action_const={}'.format(action_const))
        if not self.proc0_groups:
            raise RuntimeError('Only group leaders should call mobilize_workers()')

        self.comm_groups.bcast(action_const, root=0)

    def stop_leaders(self):
        logger.debug('stop_leaders')
        if not self.proc0_world:
            raise RuntimeError('Only proc0_world should call stop_leaders()')

        data = STOP
        self.comm_leaders.bcast(data, root=0)

    def stop_workers(self):
        logger.debug('stop_workers')
        if not self.proc0_groups:
            raise RuntimeError('Only proc0_groups should call stop_workers()')

        data = STOP
        self.comm_groups.bcast(data, root=0)

    def leaders_loop(self, action):
        """
        actions should be a dict where the keys are possible integer
        constants, and the values are callable functions that are
        called when the corresponding key is sent from
        mobilize_leaders.
        """
        if self.proc0_world:
            logger.debug('proc0_world bypassing leaders_loop')
            return
        
        logger.debug('entering leaders_loop')

        while True:
            # Wait for proc 0 to send us something:
            data = None
            data = self.comm_leaders.bcast(data, root=0)
            logger.debug('leaders_loop received {}'.format(data))
            if data == STOP:
                # Tell workers to stop
                break

            # Call the requested function:
            action(self, data)

        logger.debug('leaders_loop end')

    def worker_loop(self, action):
        """
        actions should be a dict where the keys are possible integer
        constants, and the values are callable functions that are
        called when the corresponding key is sent from
        mobilize_workers.
        """
        if self.proc0_groups:
            logger.debug('bypassing worker_loop since proc0_groups')
            return
        
        logger.debug('entering worker_loop')

        while True:
            # Wait for the group leader to send us something:
            data = None
            data = self.comm_groups.bcast(data, root=0)
            logger.debug('worker_loop worker received {}'.format(data))
            if data == STOP:
                break

            # Call the requested function:
            action(self, data)

        logger.debug('worker_loop end')

    def apart(self, leaders_action, workers_action):
        """
        Send workers and group leaders off to their respective loops to
        wait for instructions from their group leader or
        proc0_world, respectively.
        """
        self.is_apart = True
        if self.proc0_world:
            pass
        elif self.proc0_groups:
            self.leaders_loop(leaders_action)
        else:
            self.worker_loop(workers_action)

    def together(self):
        """
        Bring workers and group leaders back from their respective loops.
        """
        if self.proc0_world:
            self.stop_leaders() # Proc0_world stops the leaders.

        if self.proc0_groups:
            self.stop_workers() # All group leaders stop their workers.

        self.is_apart = False
