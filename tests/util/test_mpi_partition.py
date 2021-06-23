import logging
import unittest
import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None
if MPI is not None:
    from simsopt.util.mpi import MpiPartition

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)
logger = logging.getLogger(__name__)


@unittest.skipIf(MPI is None, "Requires mpi4py")
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
        self.assertEqual(m.rank_leaders, 0 if rank_world == 0 else -1)

        self.assertEqual(m.nprocs_world, nprocs)
        self.assertEqual(m.nprocs_groups, nprocs)
        self.assertEqual(m.nprocs_leaders, 1 if rank_world == 0 else -1)

        self.assertEqual(m.proc0_world, rank_world == 0)
        self.assertEqual(m.proc0_groups, rank_world == 0)
        m.write()

    def test_ngroups_max(self):
        """
        Verify that all quantities make sense when ngroups >= nprocs_world
        and ngroups = None.
        """
        rank_world = MPI.COMM_WORLD.Get_rank()
        nprocs = MPI.COMM_WORLD.Get_size()

        for shift in range(-1, 3):
            if shift == -1:
                ngroups = None
            else:
                ngroups = nprocs + shift

            m = MpiPartition(ngroups=ngroups)
            self.assertEqual(m.ngroups, nprocs)

            self.assertEqual(m.rank_world, rank_world)
            self.assertEqual(m.rank_groups, 0)
            self.assertEqual(m.rank_leaders, rank_world)

            self.assertEqual(m.nprocs_world, nprocs)
            self.assertEqual(m.nprocs_groups, 1)
            self.assertEqual(m.nprocs_leaders, nprocs)

            self.assertEqual(m.proc0_world, rank_world == 0)
            self.assertTrue(m.proc0_groups)

    def test_ngroups_scan(self):
        """
        Verify that all quantities make sense when ngroups >= nprocs_world
        and ngroups = None.
        """
        rank_world = MPI.COMM_WORLD.Get_rank()
        nprocs = MPI.COMM_WORLD.Get_size()

        for ngroups in range(-1, nprocs+3):
            m = MpiPartition(ngroups=ngroups)

            self.assertGreaterEqual(m.ngroups, 1)
            self.assertLessEqual(m.ngroups, nprocs)

            self.assertEqual(m.rank_world, rank_world)
            self.assertGreaterEqual(m.rank_groups, 0)
            self.assertLess(m.rank_groups, nprocs)

            self.assertEqual(m.nprocs_world, nprocs)
            self.assertGreaterEqual(m.nprocs_groups, 1)
            self.assertLessEqual(m.nprocs_groups, nprocs)

            self.assertEqual(m.proc0_world, rank_world == 0)

            if m.proc0_groups:
                self.assertGreaterEqual(m.rank_leaders, 0)
                self.assertLessEqual(m.rank_leaders, nprocs)
                self.assertGreaterEqual(m.nprocs_leaders, 1)
                self.assertLessEqual(m.nprocs_leaders, nprocs)
            else:
                self.assertEqual(m.rank_leaders, -1)
                self.assertEqual(m.nprocs_leaders, -1)

            # The sizes of the worker groups should be relatively
            # even, with a difference of no more than 1 between the
            # largest and the smallest.
            if m.proc0_world:
                group_sizes = np.zeros(nprocs, dtype='i')
                group_sizes[0] = m.nprocs_groups
                for j in range(1, nprocs):
                    group_sizes[j] = m.comm_world.recv(tag=j)
                print('group_sizes:', group_sizes)
                self.assertLessEqual(np.max(group_sizes) - np.min(group_sizes), 1)
            else:
                m.comm_world.send(m.nprocs_groups, 0, tag=m.rank_world)
        m.write()


