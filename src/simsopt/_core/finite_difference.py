# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides Jacobian evaluated with finite difference scheme
"""

from __future__ import annotations

import logging
import traceback
import collections
from time import time
from datetime import datetime
from typing import Callable, Sequence
from numbers import Real

import numpy as np
try:
    # We import mpi4py here rather than mpi4py.MPI so MPI is not
    # initialized, since initializing MPI is disallowed on login nodes
    # for some HPC systems.
    import mpi4py
except ImportError:
    mpi4py = None

from .types import RealArray
from .dev import SimsoptRequires
from .optimizable import Optimizable
from .util import finite_difference_steps

logger = logging.getLogger(__name__)

__all__ = ['FiniteDifference']


class FiniteDifference:
    """
    Provides Jacobian evaluated with finite difference scheme.
    Supplies a method named jac to be used with optimizers. Use
    the initialization to customize the finite difference scheme
    """

    def __init__(self, func: Callable,
                 x0: RealArray = None,
                 abs_step: Real = 1.0e-7,
                 rel_step: Real = 0.0,
                 diff_method: str = "forward") -> None:

        try:
            if not isinstance(func.__self__, Optimizable):
                raise TypeError("Function supplied should be a method of Optimizable")
        except:
            raise TypeError("Function supplied should be a method of Optimizable")

        self.fn = func
        self.opt = func.__self__

        self.abs_step = abs_step
        self.rel_step = rel_step
        if diff_method not in ['centered', 'forward']:
            raise ValueError(f"Finite difference method {diff_method} not implemented. "
                             "Supported methods are 'centered' and 'forward'.")
        self.diff_method = diff_method

        self.x0 = np.asarray(x0) if x0 is not None else x0

        self.jac_size = None

    def jac(self, x: RealArray = None) -> RealArray:
        if x is not None:
            self.x0 = np.asarray(x)
        x0 = self.x0 if self.x0 is not None else self.opt.x
        opt_x0 = self.opt.x

        if self.jac_size is None:
            out = self.fn()
            if not isinstance(out, (np.ndarray, collections.abc.Sequence)):
                out = [out]
            self.jac_size = (len(out), self.opt.dof_size)

        jac = np.zeros(self.jac_size)
        steps = finite_difference_steps(x0, abs_step=self.abs_step,
                                        rel_step=self.rel_step)
        if self.diff_method == "centered":
            # Centered differences:
            for j in range(len(x0)):
                x = np.copy(x0)

                x[j] = x0[j] + steps[j]
                self.opt.x = x
                fplus = np.asarray(self.fn())

                x[j] = x0[j] - steps[j]
                self.opt.x = x
                fminus = np.asarray(self.fn())

                jac[:, j] = (fplus - fminus) / (2 * steps[j])

        elif self.diff_method == "forward":
            # 1-sided differences
            self.opt.x = x0
            f0 = np.asarray(self.fn())
            for j in range(len(x0)):
                x = np.copy(x0)
                x[j] = x0[j] + steps[j]
                self.opt.x = x
                fplus = np.asarray(self.fn())

                jac[:, j] = (fplus - f0) / steps[j]

        # Set the opt.x to the original x
        self.opt.x = opt_x0

        return jac


@SimsoptRequires(mpi4py is not None, "MPIFiniteDifference requires mpi4py")
class MPIFiniteDifference:
    """
    Provides Jacobian evaluated with finite difference scheme.
    Use MPI to parallelize the function evaluations needed for the
    finite difference scheme.
    Supplies a method named jac to be used with optimizers. Use
    the initialization to customize the finite difference scheme
    """

    def __init__(self, func: Callable,
                 mpi,  # Specifying the type MpiPartition here would require initializing MPI
                 x0: RealArray = None,
                 abs_step: Real = 1.0e-7,
                 rel_step: Real = 0.0,
                 diff_method: str = "forward",
                 log_file: Union[str, typing.IO] = "jac_log") -> None:

        try:
            if not isinstance(func.__self__, Optimizable):
                raise TypeError(
                    "Function supplied should be a method of Optimizable")
        except:
            raise TypeError(
                "Function supplied should be a method of Optimizable")

        self.fn = func
        self.mpi = mpi
        self.opt = func.__self__

        self.abs_step = abs_step
        self.rel_step = rel_step
        if diff_method not in ['centered', 'forward']:
            raise ValueError(
                f"Finite difference method {diff_method} not implemented. "
                "Supported methods are 'centered' and 'forward'.")
        self.diff_method = diff_method
        self.log_file = log_file
        self.new_log_file = False
        self.log_header_written = False

        x0 = np.asarray(x0) if x0 is not None else x0
        self.x0 = x0 if x0 else self.opt.x

        self.jac_size = None
        self.eval_cnt = 1

    def __enter__(self):
        self.mpi_apart()
        self.init_log()
        return self

    def mpi_apart(self):
        self.mpi.apart(lambda mpi, data: self.mpi_leaders_task(),
                       lambda mpi, data: self.mpi_workers_task())

    def init_log(self):
        if self.mpi.proc0_world:
            if isinstance(self.log_file, str):
                datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                log_file = self.log_file + "_" + datestr + ".dat"
                self.log_file = open(log_file, 'w')
                self.new_log_file = True
        self.start_time = time()

    def __exit__(self, exc_type, exc_value, tb):
        self.mpi.together()
        if self.mpi.proc0_world and self.new_log_file:
            self.log_file.close()

    # Called by MPI leaders
    def _jac(self, x: RealArray = None):
        # Use shortcuts for class variables
        opt = self.opt
        mpi = self.mpi
        if not mpi.is_apart:
            mpi.worker_loop(lambda mpi, data: self.mpi_workers_task())
        if not mpi.proc0_groups:  # This condition shouldn't  be  triggered
            return (None, None, None)

        if x is not None:
            opt.x = x

        logger.info('Beginning parallel finite difference gradient calculation')

        x0 = np.copy(opt.x)
        nparams = opt.dof_size
        # Make sure all leaders have the same x0.
        mpi.comm_leaders.Bcast(x0)
        logger.info(f'nparams: {nparams}')
        logger.info(f'x0:  {x0}')

        # Set up the list of parameter values to try
        steps = finite_difference_steps(x0, abs_step=self.abs_step,
                                        rel_step=self.rel_step)
        mpi.comm_leaders.Bcast(steps)
        diff_method = mpi.comm_leaders.bcast(self.diff_method)
        if diff_method == "centered":
            nevals_jac = 2 * nparams
            xs = np.zeros((nparams, nevals_jac))
            for j in range(nparams):
                xs[:, 2 * j] = x0[:]  # I don't think I need np.copy(), but not 100% sure.
                xs[j, 2 * j] = x0[j] + steps[j]
                xs[:, 2 * j + 1] = x0[:]
                xs[j, 2 * j + 1] = x0[j] - steps[j]
        else:  # diff_method == "forward":
            # 1-sided differences
            nevals_jac = nparams + 1
            xs = np.zeros((nparams, nevals_jac))
            xs[:, 0] = x0[:]
            for j in range(nparams):
                xs[:, j + 1] = x0[:]
                xs[j, j + 1] = x0[j] + steps[j]

        evals = None
        # nvals = None # Work on this later
        if not mpi.proc0_world:
            # All procs other than proc0_world should initialize evals before
            # the nevals_jac loop, since they may not have any evals.
            self.jac_size = np.zeros(2, dtype=np.int32)
            self.jac_size = mpi.comm_leaders.bcast(self.jac_size)
            evals = np.zeros((self.jac_size[0], nevals_jac))
        # Do the hard work of evaluating the functions.
        logger.info(f'size of evals is ({self.jac_size[0]}, {nevals_jac})')

        ARB_VAL = 100
        for j in range(nevals_jac):
            # Handle only this group's share of the work:
            if np.mod(j, mpi.ngroups) == mpi.rank_leaders:
                mpi.mobilize_workers(ARB_VAL)
                x = xs[:, j]
                mpi.comm_groups.bcast(x, root=0)
                opt.x = x
                out = np.asarray(self.fn())

                if evals is None and mpi.proc0_world:
                    self.jac_size = mpi.comm_leaders.bcast(self.jac_size)
                    evals = np.zeros((self.jac_size[0], nevals_jac))

                evals[:, j] = out
                # evals[:, j] = np.array([f() for f in dofs.funcs])

        # Combine the results from all groups:
        evals = mpi.comm_leaders.reduce(evals, op=mpi4py.MPI.SUM, root=0)

        if not mpi.is_apart:
            mpi.stop_workers()
        # Only proc0_world will actually have the Jacobian.
        if not mpi.proc0_world:
            return (None, None, None)

        # Use the evals to form the Jacobian
        jac = np.zeros(self.jac_size)
        if diff_method == "centered":
            for j in range(nparams):
                jac[:, j] = (evals[:, 2 * j] - evals[:, 2 * j + 1]) / (
                    2 * steps[j])
        else:  # diff_method == "forward":
            # 1-sided differences:
            for j in range(nparams):
                jac[:, j] = (evals[:, j + 1] - evals[:, 0]) / steps[j]

        # Weird things may happen if we do not reset the state vector
        # to x0:
        opt.x = x0
        return jac, xs, evals

    def mpi_leaders_task(self, *args):
        """
            This function is called by group leaders when
            MpiPartition.leaders_loop() receives a signal to do something.

            We have to take a "data" argument, but there is only 1 task we
            would do, so we don't use it.
            """
        logger.debug('mpi leaders task')

        # x is a buffer for receiving the state vector:
        x = np.empty(self.opt.dof_size, dtype='d')
        # If we make it here, we must be doing a fd_jac_par
        # calculation, so receive the state vector: mpi4py has
        # separate bcast and Bcast functions!!  comm.Bcast(x,
        # root=0)
        x = self.mpi.comm_leaders.bcast(x, root=0)
        logger.debug(f'mpi leaders loop x={x}')
        self.opt.x = x
        self._jac()

    def mpi_workers_task(self, *args):
        """
            Note: func is a method of opt.
            """
        logger.debug('mpi workers task')

        # x is a buffer for receiving the state vector:
        x = np.empty(self.opt.dof_size, dtype='d')
        # If we make it here, we must be doing a fd_jac_par
        # calculation, so receive the state vector: mpi4py has
        # separate bcast and Bcast functions!!  comm.Bcast(x, root=0)
        x = self.mpi.comm_groups.bcast(x, root=0)
        logger.debug(f'worker loop worker x={x}')
        self.opt.x = x

        # We don't store or do anything with f() or jac(), because
        # the group leader will handle that.
        try:
            return self.fn()
        except:
            logger.warning("Exception caught by worker during residual "
                           "evaluation in worker loop")
            traceback.print_exc()  # Print traceback

    # Call to jac function is made in proc0
    def jac(self, x: RealArray = None, *args, **kwargs):
        """
        Called by proc0
        """

        ARB_VAL = 100
        logger.debug("Entering jac evaluation")

        if self.jac_size is None:  # Do one evaluation of code
            if x is None:
                x = self.x0
            self.mpi.mobilize_workers(ARB_VAL)
            self.mpi.comm_groups.bcast(x, root=0)
            self.opt.x = x
            out = self.fn()
            if not isinstance(out, (np.ndarray, collections.abc.Sequence)):
                out = np.array([out])
            else:
                out = np.asarray(out)
            self.jac_size = np.array((len(out), self.opt.dof_size),
                                     dtype=np.int32)

        self.mpi.mobilize_leaders(ARB_VAL)  # Any value not equal to STOP
        self.mpi.comm_leaders.bcast(x, root=0)

        jac, xs, evals = self._jac(x)
        logger.debug(f'jac is {jac}')

        # Write to the log file:
        logfile = self.log_file
        if not self.log_header_written:
            logfile.write(f'Problem type:\nleast_squares\nnparams:\n{len(x)}\n')
            logfile.write('function_evaluation,seconds')
            for j in range(len(x)):
                logfile.write(f',x({j})')
            logfile.write('\n')
            self.log_header_written = True
        nevals = evals.shape[1]
        for j in range(nevals):
            del_t = time() - self.start_time
            j_eval = j + self.eval_cnt - 1
            logfile.write(f'{j_eval:6d},{del_t:12.4e}')
            for xj in xs[:, j]:
                logfile.write(f',{xj:24.16e}')
            logfile.write('\n')
            logfile.flush()

        self.eval_cnt += nevals

        return jac
