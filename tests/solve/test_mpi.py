import logging
import unittest
import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None
from simsopt._core.dofs import Dofs
from simsopt.objectives.functions import Beale
from simsopt.objectives.least_squares import LeastSquaresProblem

from simsopt.util.mpi import log
if MPI:
    from simsopt.util.mpi import MpiPartition  # , log
    from simsopt.solve.mpi import fd_jac_mpi, least_squares_mpi_solve


#log(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TestFunction1():
    def __init__(self):
        self.x = np.array([1.2, 0.9, -0.4])
        self.fixed = np.full(3, False)

    def get_dofs(self):
        return self.x

    def set_dofs(self, x):
        self.x = x

    def J(self):
        return np.exp(self.x[0] ** 2 - np.exp(self.x[1]) + np.sin(self.x[2]))


class TestFunction2():
    def __init__(self):
        self.x = np.array([1.2, 0.9])
        self.fixed = np.full(2, False)

    def get_dofs(self):
        return self.x

    def set_dofs(self, x):
        self.x = x

    def f0(self):
        return np.exp(0 + self.x[0] ** 2 - np.exp(self.x[1]))

    def f1(self):
        return np.exp(1 + self.x[0] ** 2 - np.exp(self.x[1]))

    def f2(self):
        return np.exp(2 + self.x[0] ** 2 - np.exp(self.x[1]))

    def f3(self):
        return np.exp(3 + self.x[0] ** 2 - np.exp(self.x[1]))


class TestFunction3:
    """
    This is the Rosenbrock function again, but with some unnecessary
    MPI communication added in order to test optimization with MPI.
    """

    def __init__(self, comm):
        self.comm = comm
        self.x = [0., 0.]
        self.dummy = 42

    def get_dofs(self):
        return self.x

    def set_dofs(self, x):
        self.x = x

    def f0(self):
        # Do some random MPI stuff just for the sake of testing.
        self.comm.barrier()
        self.comm.bcast(self.x)
        return self.x[0] - 1

    def f1(self):
        # Do some random MPI stuff just for the sake of testing.
        self.comm.bcast(self.dummy)
        self.comm.barrier()
        return self.x[0] ** 2 - self.x[1]


@unittest.skipIf(MPI is None, "MPI is missing")
class SolveMpiTests(unittest.TestCase):
    def test_fd_jac_eval_points(self):
        """
        Check that fd_jac_mpi is evaluating the residual functions at the
        expected locations.
        """
        for ngroups in range(1, 2):
            mpi = MpiPartition(ngroups)
            b = Beale()  # Any Optimizable object with 2 d.o.f.'s will do.

            # First examine 1-sided differences
            prob = LeastSquaresProblem([(b, 0, 1)], diff_method="forward",
                                       abs_step=1e-6, rel_step=1e-2)

            b.set_dofs([0, 0.2])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[0.0, 1e-6, 0],
                                   [0.2, 0.2, 0.202]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            b.set_dofs([0, 0])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[0.0, 1e-6, 0],
                                   [0.0, 0.0, 1e-6]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            b.set_dofs([-3, -4])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[-3.0, -2.97, -3.0],
                                   [-4.0, -4.0, -3.96]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            b.set_dofs([3e-7, 4e-7])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[3e-7, 1.3e-6, 3.0e-7],
                                   [4e-7, 4.0e-7, 1.4e-6]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            # Now examine centered differences
            prob = LeastSquaresProblem([(b, 0, 1)], diff_method="centered",
                                       abs_step=1e-6, rel_step=1e-2)

            b.set_dofs([0, 0.2])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[1e-6, -1e-6, 0, 0],
                                   [0.2, 0.2, 0.202, 0.198]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            b.set_dofs([0, 0])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[1e-6, -1e-6, 0, 0],
                                   [0, 0, 1e-6, -1e-6]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            b.set_dofs([-3, -4])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[-2.97, -3.03, -3.00, -3.00],
                                   [-4.00, -4.00, -3.96, -4.04]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

            b.set_dofs([3e-7, 4e-7])
            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi)
            xs_correct = np.array([[1.3e-6, -0.7e-6, 3.0e-7, 3.00e-7],
                                   [4.0e-7, 4.00e-7, 1.4e-6, -0.6e-6]])
            if mpi.proc0_groups:
                np.testing.assert_allclose(xs, xs_correct)

    def test_fd_jac(self):
        """
        Test the parallel finite-difference Jacobian calculation.
        """
        abs_step = 1.0e-7
        rel_step = 0
        for ngroups in range(1, 4):
            logger.debug('ngroups={}'.format(ngroups))
            mpi = MpiPartition(ngroups=ngroups)
            o = TestFunction1()
            d = Dofs([o], diff_method="forward", abs_step=abs_step, rel_step=rel_step)
            logger.debug('About to do worker loop 1')
            jac, xs, evals = fd_jac_mpi(d, mpi)
            jac_reference = np.array([[5.865176283537110e-01, -6.010834349701177e-01, 2.250910244305793e-01]])
            if mpi.proc0_world:
                np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            o.set_dofs(np.array([1.2, 0.9, -0.4]))
            jac = d.fd_jac()
            np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

            # Repeat with centered differences
            o.set_dofs(np.array([1.2, 0.9, -0.4]))
            logger.debug('About to do worker loop 2')
            d.diff_method = "centered"
            jac, xs, evals = fd_jac_mpi(d, mpi)
            jac_reference = np.array([[5.865175337071982e-01, -6.010834789627051e-01, 2.250910093037906e-01]])
            if mpi.proc0_world:
                np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            o.set_dofs(np.array([1.2, 0.9, -0.4]))
            jac = d.fd_jac()
            np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

            # Now try a case with different nparams and nfuncs.
            o = TestFunction2()
            d = Dofs([o.f0, o.f1, o.f2, o.f3], diff_method="forward",
                     abs_step=abs_step, rel_step=rel_step)
            logger.debug('About to do worker loop 3')
            jac, xs, evals = fd_jac_mpi(d, mpi)
            jac_reference = np.array([[8.657715439008840e-01, -8.872724499564555e-01],
                                      [2.353411054922816e+00, -2.411856577788640e+00],
                                      [6.397234502131255e+00, -6.556105911492693e+00],
                                      [1.738948636642590e+01, -1.782134355643450e+01]])
            if mpi.proc0_world:
                np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            o.set_dofs(np.array([1.2, 0.9]))
            jac = d.fd_jac()
            np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

            # Repeat with centered differences
            o.set_dofs(np.array([1.2, 0.9]))
            d.diff_method = "centered"
            logger.debug('About to do worker loop 4')
            jac, xs, evals = fd_jac_mpi(d, mpi)
            jac_reference = np.array([[8.657714037352271e-01, -8.872725151820582e-01],
                                      [2.353410674116319e+00, -2.411856754314101e+00],
                                      [6.397233469623842e+00, -6.556106388888594e+00],
                                      [1.738948351093228e+01, -1.782134486205678e+01]])
            if mpi.proc0_world:
                np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)
            # While we're at it, also test the serial FD Jacobian:
            o.set_dofs(np.array([1.2, 0.9]))
            jac = d.fd_jac()
            np.testing.assert_allclose(jac, jac_reference, rtol=1e-13, atol=1e-13)

    def test_fd_jac_abs_rel_steps(self):
        """
        Confirm that the parallel finite difference gradient gives nearly
        the same result regardless of whether absolute or relative
        steps are used.
        """
        rtol = 1e-6
        atol = 1e-6
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
                        o = TestFunction1()
                        d = Dofs([o], diff_method=diff_method,
                                 abs_step=abs_step, rel_step=rel_step)
                        logger.debug('About to do worker loop 1')
                        jac, xs, evals = fd_jac_mpi(d, mpi)
                        jac_reference = np.array([[5.865175337071982e-01, -6.010834789627051e-01, 2.250910093037906e-01]])
                        if mpi.proc0_world:
                            np.testing.assert_allclose(jac, jac_reference, rtol=rtol, atol=atol)
                        # While we're at it, also test the serial FD Jacobian:
                        jac = d.fd_jac()
                        np.testing.assert_allclose(jac, jac_reference, rtol=rtol, atol=atol)

                        # Now try a case with different nparams and nfuncs.
                        o = TestFunction2()
                        d = Dofs([o.f0, o.f1, o.f2, o.f3], diff_method=diff_method,
                                 abs_step=abs_step, rel_step=rel_step)
                        logger.debug('About to do worker loop 2')
                        jac, xs, evals = fd_jac_mpi(d, mpi)
                        jac_reference = np.array([[8.657714037352271e-01, -8.872725151820582e-01],
                                                  [2.353410674116319e+00, -2.411856754314101e+00],
                                                  [6.397233469623842e+00, -6.556106388888594e+00],
                                                  [1.738948351093228e+01, -1.782134486205678e+01]])
                        if mpi.proc0_world:
                            np.testing.assert_allclose(jac, jac_reference, rtol=rtol, atol=atol)
                        # While we're at it, also test the serial FD Jacobian:
                        jac = d.fd_jac()
                        np.testing.assert_allclose(jac, jac_reference, rtol=rtol, atol=atol)

    def test_parallel_optimization_without_grad(self):
        """
        Test a full least-squares optimization.
        """
        for ngroups in range(1, 4):
            mpi = MpiPartition(ngroups=ngroups)
            o = TestFunction3(mpi.comm_groups)
            term1 = (o.f0, 0, 1)
            term2 = (o.f1, 0, 1)
            prob = LeastSquaresProblem([term1, term2])
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
                        prob = LeastSquaresProblem([term1, term2], diff_method=diff_method,
                                                   abs_step=abs_step, rel_step=rel_step)
                        # Set initial condition different from 0,
                        # because otherwise abs_step=0 causes step
                        # size to be 0.
                        prob.x = [-0.1, 0.2]
                        least_squares_mpi_solve(prob, mpi)
                        self.assertAlmostEqual(prob.x[0], 1)
                        self.assertAlmostEqual(prob.x[1], 1)

