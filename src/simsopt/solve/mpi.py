# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides two main functions, 
:meth:`~simsopt.solve.mpi_solve.fd_jac_mpi()`
and
:meth:`~simsopt.solve.mpi_solve.least_squares_mpi_solve()`.
Also included are some functions that help in
the operation of these main functions.
"""

import logging
from datetime import datetime
from time import time
import traceback

import numpy as np
from scipy.optimize import least_squares

try:
    from mpi4py import MPI
except ImportError as err:
    MPI = None

from .._core.dofs import Dofs
from ..util.mpi import MpiPartition
from ..util.dev import deprecated
from .._core.util import finite_difference_steps
from ..objectives.least_squares import LeastSquaresProblem
from .graph_mpi import least_squares_mpi_solve as glsmpi

logger = logging.getLogger(__name__)

# Constants for signaling to workers what task to do:
CALCULATE_F = 1
CALCULATE_JAC = 2
CALCULATE_FD_JAC = 3


def _mpi_leaders_task(mpi, dofs, data):
    """
    This function is called by group leaders when
    MpiPartition.leaders_loop() receives a signal to do something.

    We have to take a "data" argument, but there is only 1 task we
    would do, so we don't use it.
    """
    logger.debug('mpi leaders task')

    # x is a buffer for receiving the state vector:
    x = np.empty(dofs.nparams, dtype='d')
    # If we make it here, we must be doing a fd_jac_par
    # calculation, so receive the state vector: mpi4py has
    # separate bcast and Bcast functions!!  comm.Bcast(x,
    # root=0)
    x = mpi.comm_leaders.bcast(x, root=0)
    logger.debug(f'mpi leaders loop x={x}')
    dofs.set(x)
    fd_jac_mpi(dofs, mpi)


def _mpi_workers_task(mpi, dofs, data):
    """
    This function is called by worker processes when
    MpiPartition.workers_loop() receives a signal to do something.
    """
    logger.debug('mpi workers task')

    # x is a buffer for receiving the state vector:
    x = np.empty(dofs.nparams, dtype='d')
    # If we make it here, we must be doing a fd_jac_par
    # calculation, so receive the state vector: mpi4py has
    # separate bcast and Bcast functions!!  comm.Bcast(x,
    # root=0)
    x = mpi.comm_groups.bcast(x, root=0)
    logger.debug('worker loop worker x={}'.format(x))
    dofs.set(x)

    # We don't store or do anything with f() or jac(), because
    # the group leader will handle that.
    if data == CALCULATE_F:
        try:
            dofs.f()
        except:
            logger.info("Exception caught by worker during dofs.f()")
            traceback.print_exc()  # Print traceback

    elif data == CALCULATE_JAC:
        try:
            dofs.jac()
        except:
            logger.info("Exception caught by worker during dofs.jac()")
            traceback.print_exc()  # Print traceback

    else:
        raise ValueError('Unexpected data in worker_loop')


def fd_jac_mpi(dofs: Dofs,
               mpi: MpiPartition,
               x: np.ndarray = None
               ) -> tuple:
    """
    Compute the finite-difference Jacobian of the functions in dofs
    with respect to all non-fixed degrees of freedom. Parallel
    function evaluations will be used.

    The attribues ``abs_step`', ``rel_step``, and ``diff_method`` of
    the ``Dofs`` object will be queried and used to set the finite
    difference step sizes, using
    :func:`simsopt._core.util.finite_difference_steps()`.

    If the argument x is not supplied, the Jacobian will be
    evaluated for the present state vector. If x is supplied, then
    first get_dofs() will be called for each object to set the
    global state vector to x.

    There are 2 ways to call this function. In method 1, all procs
    (including workers) call this function (so mpi.is_apart is
    False). In this case, the worker loop will be started
    automatically. In method 2, the worker loop has already been
    started before this function is called, as would be the case
    in least_squares_mpi_solve(). Then only the group leaders
    call this function.

    Args:
        dofs: The map from :math:`\mathbb{R}^n \\to \mathbb{R}^m` for which you
          want to compute the Jacobian.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` object, storing
          the information about how the pool of MPI processes is
          divided into worker groups.
        x: The 1D state vector at which you wish to evaluate the Jacobian.
          If ``None``, the Jacobian will be evaluated at the present
          state vector.

    Returns: 
        tuple containing

        - **jac** (*numpy.ndarray*) -- The Jacobian matrix.
        - **xmat** (*numpy.ndarray*) -- A matrix, the columns of which give
          all the values of x at which the functions were evaluated.
        - **fmat** (*numpy.ndarray*) -- A matrix, the columns of which give
          the corresponding values of the functions.
    """
    if MPI is None:
        raise RuntimeError("fd_jac_mpi requires the mpi4py package.")

    apart_at_start = mpi.is_apart
    if not apart_at_start:
        mpi.worker_loop(lambda mpi2, data: _mpi_workers_task(mpi2, dofs, data))
    if not mpi.proc0_groups:
        return (None, None, None)

    # Only group leaders execute this next section.

    if x is not None:
        dofs.set(x)

    logger.info('Beginning parallel finite difference gradient calculation for functions ' + str(dofs.funcs))

    x0 = dofs.x
    # Make sure all leaders have the same x0.
    mpi.comm_leaders.Bcast(x0)
    logger.info('  nparams: {}, nfuncs: {}'.format(dofs.nparams, dofs.nfuncs))
    logger.info('  x0: ' + str(x0))

    # Set up the list of parameter values to try
    steps = finite_difference_steps(x0, abs_step=dofs.abs_step, rel_step=dofs.rel_step)
    mpi.comm_leaders.Bcast(steps)
    diff_method = mpi.comm_leaders.bcast(dofs.diff_method)
    if diff_method == "centered":
        nevals_jac = 2 * dofs.nparams
        xs = np.zeros((dofs.nparams, nevals_jac))
        for j in range(dofs.nparams):
            xs[:, 2 * j] = x0[:]  # I don't think I need np.copy(), but not 100% sure.
            xs[j, 2 * j] = x0[j] + steps[j]
            xs[:, 2 * j + 1] = x0[:]
            xs[j, 2 * j + 1] = x0[j] - steps[j]
    elif diff_method == "forward":
        # 1-sided differences
        nevals_jac = dofs.nparams + 1
        xs = np.zeros((dofs.nparams, nevals_jac))
        xs[:, 0] = x0[:]
        for j in range(dofs.nparams):
            xs[:, j + 1] = x0[:]
            xs[j, j + 1] = x0[j] + steps[j]
    else:
        raise ValueError("diff_method must be 'centered' or 'forward'")

    # proc0_world will be responsible for detecting nvals, since
    #proc0_world always does at least 1 function evaluation. Other
    #procs cannot be trusted to evaluate nvals because they may
    #not have any function evals, in which case they never create
    #"evals", so the MPI reduce would fail.

    #evals = np.zeros((dofs.nfuncs, nevals_jac))
    evals = None
    if not mpi.proc0_world:
        # All procs other than proc0_world should initialize evals
        # before the nevals_jac loop, since they may not have any
        # evals.
        dofs.nvals = mpi.comm_leaders.bcast(dofs.nvals)
        evals = np.zeros((dofs.nvals, nevals_jac))
    # Do the hard work of evaluating the functions.
    for j in range(nevals_jac):
        # Handle only this group's share of the work:
        if np.mod(j, mpi.ngroups) == mpi.rank_leaders:
            mpi.mobilize_workers(CALCULATE_F)
            x = xs[:, j]
            mpi.comm_groups.bcast(x, root=0)
            dofs.set(x)

            f = dofs.f()

            if evals is None and mpi.proc0_world:
                dofs.nvals = mpi.comm_leaders.bcast(dofs.nvals)
                evals = np.zeros((dofs.nvals, nevals_jac))

            evals[:, j] = f
            #evals[:, j] = np.array([f() for f in dofs.funcs])

    # Combine the results from all groups:
    evals = mpi.comm_leaders.reduce(evals, op=MPI.SUM, root=0)

    if not apart_at_start:
        mpi.stop_workers()

    # Only proc0_world will actually have the Jacobian.
    if not mpi.proc0_world:
        return (None, None, None)

    # Use the evals to form the Jacobian
    jac = np.zeros((dofs.nvals, dofs.nparams))
    if diff_method == "centered":
        for j in range(dofs.nparams):
            jac[:, j] = (evals[:, 2 * j] - evals[:, 2 * j + 1]) / (2 * steps[j])
    elif diff_method == "forward":
        # 1-sided differences:
        for j in range(dofs.nparams):
            jac[:, j] = (evals[:, j + 1] - evals[:, 0]) / steps[j]
    else:
        assert False, "Program should not get here"

    # Weird things may happen if we do not reset the state vector
    # to x0:
    dofs.set(x0)
    return jac, xs, evals


@deprecated(replacement=glsmpi,
            message="This class has been deprecated from v0.6.0 and will be "
                    "deleted from future versions of simsopt. Use graph "
                    "framework to define the optimization problem. Use "
                    "simsopt.objectives.graph_least_squares.LeastSquaresProblem"
                    " class in conjunction with"
                    " simsopt.solve.graph_mpi.least_squares_mpi_solve")
def least_squares_mpi_solve(prob: LeastSquaresProblem,
                            mpi: MpiPartition,
                            grad: bool = None,
                            **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    MPI. All MPI processes (including group leaders and workers)
    should call this function.

    Args:
        prob: An instance of LeastSquaresProblem, defining the objective function(s) and parameter space.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` object, storing
          the information about how the pool of MPI processes is
          divided into worker groups.
        grad: Whether to use a gradient-based optimization algorithm, as opposed to a gradient-free algorithm.
          If unspecified, a gradient-based algorithm will be used if ``prob`` has gradient information available,
          otherwise a gradient-free algorithm will be used by default. If you set ``grad=True`` for a problem in
          which gradient information is not available, finite-difference gradients will be used.
        kwargs: Any arguments to pass to 
          `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_. 
          For instance,
          you can supply ``max_nfev=100`` to set the maximum number of function evaluations (not counting
          finite-difference gradient evaluations) to 100. Or, you can supply ``method`` to choose the optimization algorithm.
    """

    if MPI is None:
        raise RuntimeError("least_squares_mpi_solve requires the mpi4py package.")

    logger.info("Beginning solve.")
    prob._init()
    if grad is None:
        grad = prob.dofs.grad_avail

    x = np.copy(prob.x)  # For use in Bcast later.

    logfile = None
    logfile_started = False
    residuals_file = None
    nevals = 0
    start_time = time()

    def _f_proc0(x):
        """
        This function is used for least_squares_mpi_solve.  It is similar
        to LeastSquaresProblem.f(), except this version is called only by
        proc 0 while workers are in the worker loop.
        """
        logger.debug("Entering _f_proc0")
        mpi.mobilize_workers(CALCULATE_F)
        # Send workers the state vector:
        mpi.comm_groups.bcast(x, root=0)
        logger.debug("Past bcast in _f_proc0")

        f_unshifted = prob.dofs.f(x)
        f_shifted = prob.f_from_unshifted(f_unshifted)
        objective_val = prob.objective_from_shifted_f(f_shifted)

        nonlocal logfile_started, logfile, residuals_file, nevals

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not logfile_started:
            # Initialize log file
            logfile_started = True
            datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            filename = "simsopt_" + datestr + ".dat"
            logfile = open(filename, 'w')
            logfile.write("Problem type:\nleast_squares\nnparams:\n{}\n".format(prob.dofs.nparams))
            logfile.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                logfile.write(",x({})".format(j))
            logfile.write(",objective_function")
            logfile.write("\n")

            filename = "residuals_" + datestr + ".dat"
            residuals_file = open(filename, 'w')
            residuals_file.write("Problem type:\nleast_squares\nnparams:\n{}\n".format(prob.dofs.nparams))
            residuals_file.write("function_evaluation,seconds")
            for j in range(prob.dofs.nparams):
                residuals_file.write(",x({})".format(j))
            residuals_file.write(",objective_function")
            for j in range(prob.dofs.nvals):
                residuals_file.write(",F({})".format(j))
            residuals_file.write("\n")

        logfile.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            logfile.write(",{:24.16e}".format(xj))
        logfile.write(",{:24.16e}".format(objective_val))
        logfile.write("\n")
        logfile.flush()

        residuals_file.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
        for xj in x:
            residuals_file.write(",{:24.16e}".format(xj))
        residuals_file.write(",{:24.16e}".format(objective_val))
        for fj in f_unshifted:
            residuals_file.write(",{:24.16e}".format(fj))
        residuals_file.write("\n")
        residuals_file.flush()

        nevals += 1
        return f_shifted

    def _jac_proc0(x):
        """
        This function is used for least_squares_mpi_solve.  It is similar
        to LeastSquaresProblem.jac, except this version is called only by
        proc 0 while workers are in the worker loop.
        """
        if prob.dofs.grad_avail:
            # proc0_world calling mobilize_workers will mobilize only group 0.
            mpi.mobilize_workers(CALCULATE_JAC)
            # Send workers the state vector:
            mpi.comm_groups.bcast(x, root=0)

            return prob.jac(x)

        else:
            # Evaluate Jacobian using fd_jac_mpi
            mpi.mobilize_leaders(CALCULATE_FD_JAC)
            # Send leaders the state vector:
            mpi.comm_leaders.bcast(x, root=0)

            jac, xs, evals = fd_jac_mpi(prob.dofs, mpi, x)

            # Write function evaluations to the files
            nonlocal logfile_started, logfile, residuals_file, nevals
            nevals_jac = evals.shape[1]
            for j in range(nevals_jac):
                objective_val = prob.objective_from_unshifted_f(evals[:, j])

                logfile.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
                for xj in xs[:, j]:
                    logfile.write(",{:24.16e}".format(xj))
                logfile.write(",{:24.16e}".format(objective_val))
                logfile.write("\n")
                logfile.flush()

                residuals_file.write("{:6d},{:12.4e}".format(nevals, time() - start_time))
                for xj in xs[:, j]:
                    residuals_file.write(",{:24.16e}".format(xj))
                residuals_file.write(",{:24.16e}".format(objective_val))
                for fj in evals[:, j]:
                    residuals_file.write(",{:24.16e}".format(fj))
                residuals_file.write("\n")
                residuals_file.flush()

                nevals += 1

            return prob.scale_dofs_jac(jac)

    # Send group leaders and workers into their respective loops:
    leaders_action = lambda mpi2, data: _mpi_leaders_task(mpi, prob.dofs, data)
    workers_action = lambda mpi2, data: _mpi_workers_task(mpi, prob.dofs, data)
    mpi.apart(leaders_action, workers_action)

    if mpi.proc0_world:
        # proc0_world does this block, running the optimization.
        x0 = np.copy(prob.dofs.x)
        #print("x0:",x0)
        # Call scipy.optimize:
        if grad:
            logger.info("Using derivatives")
            print("Using derivatives")
            result = least_squares(_f_proc0, x0, verbose=2, jac=_jac_proc0, **kwargs)
        else:
            logger.info("Using derivative-free method")
            print("Using derivative-free method")
            result = least_squares(_f_proc0, x0, verbose=2, **kwargs)

        logger.info("Completed solve.")
        x = result.x

        logfile.close()
        residuals_file.close()

    # Stop loops for workers and group leaders:
    mpi.together()

    logfile_started = False
    logger.info("Completed solve.")

    # Finally, make sure all procs get the optimal state vector.
    mpi.comm_world.Bcast(x)
    logger.debug('After Bcast, x={}'.format(x))
    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.dofs.set(x)
