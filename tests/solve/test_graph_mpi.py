import logging
import unittest

import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None

from simsopt._core.graph_optimizable import Optimizable
from simsopt.objectives.graph_least_squares import LeastSquaresProblem
if MPI is not None:
    from simsopt.util.mpi import MpiPartition
    from simsopt.solve.graph_mpi import fd_jac_mpi, least_squares_mpi_solve

#logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFunction1(Optimizable):
    def __init__(self):
        x = np.array([1.2, 0.9, -0.4])
        fixed = np.full(3, False)
        super().__init__(x0=x, fixed=fixed)

    def J(self):
        return np.exp(self.full_x[0] ** 2 - np.exp(self.full_x[1]) \
                      + np.sin(self.full_x[2]))

    return_fn_map = {'J': J}


class TestFunction2(Optimizable):
    def __init__(self):
        x = np.array([1.2, 0.9])
        fixed = np.full(2, False)
        super().__init__(x0=x, fixed=fixed)

    def f0(self):
        return np.exp(0 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def f1(self):
        return np.exp(1 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def f2(self):
        return np.exp(2 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def f3(self):
        return np.exp(3 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    return_fn_map = {'f0': f0, 'f1': f1, 'f2': f2, 'f3': f3}


class TestFunction3(Optimizable):
    """
    This is the Rosenbrock function again, but with some unnecessary
    MPI communication added in order to test optimization with MPI.
    """

    def __init__(self, comm):
        x = [0., 0.]
        self.comm = comm
        self.dummy = 42
        logger.debug("inside test function 3 init")
        super().__init__(x0=x)

    def f0(self):
        # Do some random MPI stuff just for the sake of testing.
        self.comm.barrier()
        self.comm.bcast(self.full_x)
        return self.full_x[0] - 1

    def f1(self):
        # Do some random MPI stuff just for the sake of testing.
        self.comm.bcast(self.dummy)
        self.comm.barrier()
        return self.full_x[0] ** 2 - self.full_x[1]

    return_fn_map = {'f0': f0, 'f1': f1}


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

    @unittest.skip
    def test_fd_jac(self):
        """
        Test the parallel finite-difference Jacobian calculation.
        """
        for ngroups in range(1, 4):
            logger.debug('ngroups={}'.format(ngroups))
            mpi = MpiPartition(ngroups=ngroups)
            o = TestFunction1()
            logger.debug('About to do worker loop 1')
            jac, xs, evals = fd_jac_mpi(o, mpi, centered=False, eps=1e-7)
            jac_reference = np.array([[5.865176283537110e-01, -6.010834349701177e-01, 2.250910244305793e-01]])
            if mpi.proc0_world:
                np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            # o.set_dofs(np.array([1.2, 0.9, -0.4]))
            o.x = np.array([1.2, 0.9, -0.4])
            #jac = d.fd_jac(centered=False, eps=1e-7)
            #np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

            # Repeat with centered differences
            #o.set_dofs(np.array([1.2, 0.9, -0.4]))
            o.x = np.array([1.2, 0.9, -0.4])
            logger.debug('About to do worker loop 2')
            jac, xs, evals = fd_jac_mpi(o, mpi, centered=True, eps=1e-7)
            jac_reference = np.array([[5.865175337071982e-01, -6.010834789627051e-01, 2.250910093037906e-01]])
            if mpi.proc0_world:
                np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            # o.set_dofs(np.array([1.2, 0.9, -0.4]))
            o.x = np.array([1.2, 0.9, -0.4])
            # jac = o.fd_jac(centered=True, eps=1e-7)
            # np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

            # Now try a case with different nparams and nfuncs.
            o = TestFunction2()
            logger.debug('About to do worker loop 3')
            # jac, xs, evals = fd_jac_mpi(o, mpi, centered=False, eps=1e-7)
            # jac_reference = np.array([[8.657715439008840e-01, -8.872724499564555e-01],
            #                           [2.353411054922816e+00, -2.411856577788640e+00],
            #                           [6.397234502131255e+00, -6.556105911492693e+00],
            #                           [1.738948636642590e+01, -1.782134355643450e+01]])
            # if mpi.proc0_world:
            #     np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            # o.set_dofs(np.array([1.2, 0.9]))
            o.x = np.array([1.2, 0.9])
            # jac = d.fd_jac(centered=False, eps=1e-7)
            # np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

            # Repeat with centered differences
            # o.set_dofs(np.array([1.2, 0.9]))
            logger.debug('About to do worker loop 4')
            # jac, xs, evals = fd_jac_mpi(d, mpi, centered=True, eps=1e-7)
            # jac_reference = np.array([[8.657714037352271e-01, -8.872725151820582e-01],
            #                           [2.353410674116319e+00, -2.411856754314101e+00],
            #                           [6.397233469623842e+00, -6.556106388888594e+00],
            #                           [1.738948351093228e+01, -1.782134486205678e+01]])
            # if mpi.proc0_world:
            #     np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            # o.set_dofs(np.array([1.2, 0.9]))
            o.x = np.array([1.2, 0.9])
            # jac = d.fd_jac(centered=True, eps=1e-7)
            # np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

    def test_parallel_optimization(self):
        """
        Test a full least-squares optimization.
        """
        rank_world = MPI.COMM_WORLD.Get_rank()
        logger.info(f"rank world is {rank_world}")
        nprocs = MPI.COMM_WORLD.Get_size()
        for ngroups in range(1, 4):
            #for grad in [True, False]:
            mpi = MpiPartition(ngroups=ngroups)
            o = TestFunction3(mpi.comm_groups)
            term1 = (o.f0, 0, 1)
            term2 = (o.f1, 0, 1)
            prob = LeastSquaresProblem.from_tuples([term1, term2])
            least_squares_mpi_solve(prob, mpi)  # , grad=grad)
            self.assertAlmostEqual(prob.full_x[0], 1)
            self.assertAlmostEqual(prob.full_x[1], 1)
