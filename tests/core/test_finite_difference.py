import logging
import unittest

import numpy as np
try:
    from mpi4py import MPI
except:
    MPI = None

from simsopt._core.optimizable import Optimizable, make_optimizable
from simsopt._core.finite_difference import FiniteDifference
if MPI is not None:
    from simsopt.util.mpi import MpiPartition
    from simsopt._core.finite_difference import MPIFiniteDifference

logger = logging.getLogger(__name__)


class TestFunction1(Optimizable):
    def __init__(self):
        x = np.array([1.2, 0.9, -0.4])
        fixed = np.full(3, False)
        super().__init__(x0=x, fixed=fixed)

    def J(self):
        return np.exp(self.full_x[0] ** 2 - np.exp(self.full_x[1]) \
                      + np.sin(self.full_x[2]))

    def dJ(self):
        jac = self.J() * np.asarray([2*self.full_x[0],
                                     -np.exp(self.full_x[1]),
                                     np.cos(self.full_x[2])])
        return np.expand_dims(jac, axis=0)

    return_fn_map = {'J': J}


class TestFunction2(Optimizable):
    def __init__(self):
        x = np.array([1.2, 0.9])
        fixed = np.full(2, False)
        super().__init__(x0=x, fixed=fixed)

    def f0(self):
        return np.exp(0 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def df0(self):
        jac = self.f0() * np.array([2*self.full_x[0], -np.exp(self.full_x[1])])
        return np.expand_dims(jac, axis=0)

    def f1(self):
        return np.exp(1 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def df1(self):
        jac = self.f1() * np.array([2*self.full_x[0], -np.exp(self.full_x[1])])
        return np.expand_dims(jac, axis=0)

    def f2(self):
        return np.exp(2 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def df2(self):
        jac = self.f2() * np.array([2*self.full_x[0], -np.exp(self.full_x[1])])
        return np.expand_dims(jac, axis=0)

    def f3(self):
        return np.exp(3 + self.full_x[0] ** 2 - np.exp(self.full_x[1]))

    def df3(self):
        jac = self.f3() * np.array([2*self.full_x[0], -np.exp(self.full_x[1])])
        return np.expand_dims(jac, axis=0)

    return_fn_map = {'f0': f0, 'f1': f1, 'f2': f2, 'f3': f3}


class TestFunction3(Optimizable):
    """
    This is the Rosenbrock function again, but with some unnecessary
    MPI communication added in order to test optimization with MPI.
    """

    def __init__(self, comm, x=[0.0, 0.0]):
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


class FiniteDifferenceTests(unittest.TestCase):
    def test_jac(self):
        o = TestFunction1()
        fd = FiniteDifference(o.J, diff_method="forward", abs_step=1e-7)
        jac = fd.jac()
        jac_ref = o.dJ()
        np.testing.assert_allclose(jac, jac_ref, rtol=1e-7, atol=1e-7)

        o.x = 2 * np.array([1.2, 0.9, -0.4])
        jac_ref = o.dJ()
        jac = fd.jac()
        np.testing.assert_allclose(jac, jac_ref, rtol=1e-6, atol=1e-6)

        jac = fd.jac(3 * np.array([1.2, 0.9, -0.4]))
        o.x = 3 * np.array([1.2, 0.9, -0.4])
        jac_ref = o.dJ()
        np.testing.assert_allclose(jac, jac_ref, rtol=1e-6, atol=1e-6)

        # Now try a case with different nparams and nfuncs.
        o = TestFunction2()
        anlt_jac = np.concatenate((o.df0(), o.df1(), o.df2(), o.df3()))

        # Using temporary optimization to test the same above
        opt = make_optimizable(lambda x: [x.f0(), x.f1(), x.f2(), x.f3()], o)

        fd = FiniteDifference(opt.J, diff_method="forward", abs_step=1e-7)
        fd_jac = fd.jac()
        np.testing.assert_allclose(fd_jac, anlt_jac, rtol=1e-6, atol=1e-6)

        opt.x = 2 * np.array([1.2, 0.9])
        anlt_jac = np.concatenate((o.df0(), o.df1(), o.df2(), o.df3()))
        fd_jac = fd.jac()
        np.testing.assert_allclose(fd_jac, anlt_jac, rtol=1e-6, atol=1e-6)

        # Repeat with centered differences
        fd = FiniteDifference(opt.J, diff_method="centered", abs_step=1e-7)
        fd_jac = fd.jac(3 * np.array([1.2, 0.9]))
        opt.x = 3 * np.array([1.2, 0.9])
        anlt_jac = np.concatenate((o.df0(), o.df1(), o.df2(), o.df3()))
        np.testing.assert_allclose(fd_jac, anlt_jac, rtol=1e-6, atol=1e-6)

    def test_fd_jac_abs_rel_steps(self):
        """
        Confirm that the parallel finite difference gradient gives nearly
        the same result regardless of whether absolute or relative
        steps are used.
        """
        rtol = 1e-6
        atol = 1e-6
        for abs_step in [0, 1.0e-7]:
            # Only try rel_step=0 if abs_step is positive:
            rel_steps = [0, 1.0e-7]
            if abs_step == 0:
                rel_steps = [1.0e-7]

            for rel_step in rel_steps:
                for diff_method in ["forward", "centered"]:
                    logger.debug(f'abs_step={abs_step} '
                                 f'rel_step={rel_step} diff_method={diff_method}')
                    o = TestFunction1()
                    fd = FiniteDifference(o.J, diff_method="forward",
                                          abs_step=abs_step, rel_step=rel_step)
                    jac = fd.jac()
                    jac_ref = o.dJ()
                    np.testing.assert_allclose(jac, jac_ref,
                                               rtol=rtol, atol=atol)

                    # Now try a case with different nparams and nfuncs.
                    o = TestFunction2()
                    anlt_jac = np.concatenate(
                        (o.df0(), o.df1(), o.df2(), o.df3()))

                    # Using temporary optimization to test the same above
                    opt = make_optimizable(
                        lambda x: [x.f0(), x.f1(), x.f2(), x.f3()], o)

                    fd = FiniteDifference(opt.J, diff_method=diff_method,
                                          abs_step=abs_step, rel_step=rel_step)
                    fd_jac = fd.jac()
                    np.testing.assert_allclose(fd_jac, anlt_jac, rtol=1e-6,
                                               atol=1e-6)


@unittest.skipIf(MPI is None, "Requires mpi4py")
class MPIFiniteDifferenceTests(unittest.TestCase):
    def test_jac_mpi(self):
        """
        Test the parallel finite-difference Jacobian calculation.
        """
        for ngroups in range(1, 4):
            logger.debug('ngroups={}'.format(ngroups))
            mpi = MpiPartition(ngroups=ngroups)
            o = TestFunction1()
            print(f'output of o  is  {o.J()}')
            jac_ref = o.dJ()
            print(f'analyticjacobian is  {o.dJ()}')
            fd = MPIFiniteDifference(o.J, mpi, diff_method="forward",
                                     abs_step=1e-7)
            fd.mpi_apart()
            fd.init_log()
            logger.debug('About to do worker loop 1')
            if mpi.proc0_world:
                jac = fd.jac()
                np.testing.assert_allclose(jac, jac_ref, rtol=1e-7, atol=1e-7)
            mpi.together()
            if mpi.proc0_world:
                fd.log_file.close()

            # Use context manager
            with MPIFiniteDifference(o.J, mpi, diff_method="forward", abs_step=1e-7) as fd:
                if mpi.proc0_world:
                    jac = fd.jac()
                    np.testing.assert_allclose(jac, jac_ref, rtol=1e-7, atol=1e-7)

            # Repeat with centered differences
            fd = MPIFiniteDifference(o.J, mpi, diff_method="centered",
                                     abs_step=1e-7)
            fd.mpi_apart()
            fd.init_log()
            logger.debug('About to do worker loop 2')
            if mpi.proc0_world:
                jac = fd.jac()
                np.testing.assert_allclose(jac, jac_ref, rtol=1e-7, atol=1e-7)
            mpi.together()
            if mpi.proc0_world:
                fd.log_file.close()

            # Now try a case with different nparams and nfuncs.
            o = TestFunction2()
            anlt_jac = np.concatenate((o.df0(), o.df1(), o.df2(), o.df3()))

            # Using temporary optimization to test the same above
            opt = make_optimizable(lambda x: [x.f0(), x.f1(), x.f2(), x.f3()], o)

            fd = MPIFiniteDifference(opt.J, mpi, diff_method="forward",
                                     abs_step=1e-7)
            logger.debug('About to do worker loop 2')
            fd.mpi_apart()
            fd.init_log()
            if mpi.proc0_world:
                fd_jac = fd.jac()
                np.testing.assert_allclose(fd_jac, anlt_jac,
                                           rtol=1e-6, atol=1e-6)
            mpi.together()
            if mpi.proc0_world:
                fd.log_file.close()
