import logging
import unittest

import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None

from simsopt._core.optimizable import Optimizable
from simsopt.objectives.functions import Beale
from simsopt.objectives.least_squares import LeastSquaresProblem
if MPI is not None:
    from simsopt.util.mpi import MpiPartition
    from simsopt.solve.mpi import least_squares_mpi_solve

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

    def __init__(self, comm, x=[0, 0]):
        self.comm = comm
        self.dummy = 42
        self.f0_call_cnt = 0
        self.f1_call_cnt = 0
        logger.debug("inside test function 3 init")
        super().__init__(x0=x)

    def f0(self):
        # Do some random MPI stuff just for the sake of testing.
        self.comm.barrier()
        self.comm.bcast(self.local_full_x)
        self.f0_call_cnt += 1
        print(f"x is {self.local_full_x}")
        print(f"TestFunction3.f0 called {self.f0_call_cnt} times")
        return self.local_full_x[0] - 1

    def f1(self):
        # Do some random MPI stuff just for the sake of testing.
        self.comm.bcast(self.dummy)
        self.comm.barrier()
        self.f1_call_cnt += 1
        print(f"x is {self.local_full_x}")
        print(f"TestFunction3.f1 called {self.f1_call_cnt} times")
        return self.local_full_x[0] ** 2 - self.local_full_x[1]

    return_fn_map = {'f0': f0, 'f1': f1}


@unittest.skipIf(MPI is None, "Requires mpi4py")
class MPISolveTests(unittest.TestCase):

    def test_parallel_optimization_without_grad(self):
        """
        Test a full least-squares optimization.
        """
        for ngroups in range(1, 4):
            mpi = MpiPartition(ngroups=ngroups)
            o = TestFunction3(mpi.comm_groups)
            term1 = (o.f0, 0, 1)
            term2 = (o.f1, 0, 1)
            prob = LeastSquaresProblem.from_tuples([term1, term2])
            least_squares_mpi_solve(prob, mpi, grad=False)
            self.assertAlmostEqual(prob.x[0], 1)
            self.assertAlmostEqual(prob.x[1], 1)

    def test_parallel_optimization_with_grad(self):
        """
        Test a full least-squares optimization.
        """
        for ngroups in range(1, 4):
            for abs_step in [0, 1.0e-7]:
                # Only try rel_step=0 if abs_step is positive:
                rel_steps = [0, 1.0e-7]
                if abs_step == 0:
                    rel_steps = [1.0e-7]

                for rel_step in rel_steps:
                    for diff_method in ["forward", "centered"]:
                        logger.debug(f'ngroups={ngroups} abs_step={abs_step} ' \
                                     f'rel_step={rel_step} diff_method={diff_method}')
                        mpi = MpiPartition(ngroups=ngroups)
                        o = TestFunction3(mpi.comm_groups)
                        term1 = (o.f0, 0, 1)
                        term2 = (o.f1, 0, 1)
                        prob = LeastSquaresProblem.from_tuples([term1, term2])
                        # Set initial condition different from 0,
                        # because otherwise abs_step=0 causes step
                        # size to be 0.
                        prob.x = [-0.1, 0.2]
                        least_squares_mpi_solve(prob, mpi, grad=True,
                                                diff_method=diff_method,
                                                abs_step=abs_step,
                                                rel_step=rel_step)
                        self.assertAlmostEqual(prob.x[0], 1)
                        self.assertAlmostEqual(prob.x[1], 1)

