import unittest
import numpy as np
from simsopt.mpi import MpiPartition
from mpi4py import MPI

class MpiPartitionTests(unittest.TestCase):
    def test_ngroups1(self):
        """
        Verify that all quantities make sense when ngroups = 1.
        """
        rank_world = MPI.COMM_WORLD.Get_rank()
        nprocs = MPI.COMM_WORLD.Get_size()
        m = MpiPartition(ngroups=1)
        
        self.assertEqual(m.ngroups, 1)
        
        self.assertEqual(m.rank_world, rank_world)
        self.assertEqual(m.rank_groups, rank_world)
        self.assertEqual(m.rank_leaders, 0 if rank_world==0 else -1)

        self.assertEqual(m.nprocs_world, nprocs)
        self.assertEqual(m.nprocs_groups, nprocs)
        self.assertEqual(m.nprocs_leaders, 1 if rank_world==0 else -1)

        self.assertEqual(m.proc0_world, rank_world==0)
        self.assertEqual(m.proc0_groups, rank_world==0)
        m.write()
