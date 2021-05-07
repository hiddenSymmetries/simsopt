# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides two main functions, fd_jac_mpi and
least_squares_mpi_solve. Also included are some functions that help in
the operation of these main functions.
"""

import logging
from datetime import datetime
from time import time

import numpy as np
from scipy.optimize import least_squares

try:
    from mpi4py import MPI
except ImportError as err:
    MPI = None

from .._core.graph_optimizable import Optimizable
from ..util.mpi import MpiPartition
from ..util.types import RealArray
from ..objectives.graph_least_squares import LeastSquaresProblem

logger = logging.getLogger(__name__)

# Constants for signaling to workers what task to do:
CALCULATE_F = 1
CALCULATE_JAC = 2
CALCULATE_FD_JAC = 3


def _mpi_leaders_task(mpi: MpiPartition,
                      prob: Optimizable,
                      data: int):
    """
    This function is called by group leaders when
    MpiPartition.leaders_loop() receives a signal to do something.

    We have to take a "data" argument, but there is only 1 task we
    would do, so we don't use it.

    Args:
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` object, storing
          the information about how the pool of MPI processes is
          divided into worker groups.
        prob: Optimizable object
        data: Dummy argument for this function
    """
    logger.debug('mpi leaders task')

    # x is a buffer for receiving the state vector:
    x = np.empty(prob.dof_size, dtype='d')
    # If we make it here, we must be doing a fd_jac_par
    # calculation, so receive the state vector: mpi4py has
    # separate bcast and Bcast functions!!  comm.Bcast(x,
    # root=0)
    x = mpi.comm_leaders.bcast(x, root=0)
    logger.debug(f'mpi leaders loop x={x}')
    prob.x = x
    fd_jac_mpi(prob, mpi)


def _mpi_workers_task(mpi: MpiPartition,
                      prob: Optimizable,
                      data: str):
    """
    This function is called by worker processes when
    MpiPartition.workers_loop() receives a signal to do something.

    Args:
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` object, storing
          the information about how the pool of MPI processes is
          divided into worker groups.
        prob: Optimizable object
        data: Integer with a value from 1 to 3
    """
    logger.debug('mpi workers task')

    # x is a buffer for receiving the state vector:
    x = np.empty(prob.dof_size, dtype='d')
    # If we make it here, we must be doing a fd_jac_par
    # calculation, so receive the state vector: mpi4py has
    # separate bcast and Bcast functions!!  comm.Bcast(x, root=0)
    x = mpi.comm_groups.bcast(x, root=0)
    logger.debug('worker loop worker x={}'.format(x))
    prob.x = x

    # We don't store or do anything with f() or jac(), because
    # the group leader will handle that.
    if data == CALCULATE_F:
        try:
            prob.objective()
        except:
            logger.info("Exception caught by worker during dofs.f()")

    elif data == CALCULATE_JAC:
        try:
            prob.jac()
        except:
            logger.info("Exception caught by worker during dofs.jac()")

    else:
        raise ValueError('Unexpected data in worker_loop')


def fd_jac_mpi(prob: Optimizable,
               mpi: MpiPartition,
               x: RealArray = None,
               eps: float = 1e-7,
               centered: bool = False):
    """
    Compute the finite-difference Jacobian of the functions in dofs
    with respect to all non-fixed degrees of freedom. Parallel
    function evaluations will be used.

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
        prob: The map from :math:`\mathbb{R}^n \\to \mathbb{R}^m` for which you
          want to compute the Jacobian.
        mpi: A :obj:`simsopt.util.mpi.MpiPartition` object, storing
          the information about how the pool of MPI processes is
          divided into worker groups.
        x: The 1D state vector at which you wish to evaluate the Jacobian.
          If ``None``, the Jacobian will be evaluated at the present
          state vector.
        eps: Step size for finite differences.
        centered: If ``True``, centered finite differences will be used.
          If ``false``, one-sided finite differences will be used.

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
        mpi.worker_loop(lambda mpi2, data: _mpi_workers_task(mpi2, prob, data))
    if not mpi.proc0_groups:
        return (None, None, None)

    # Only group leaders execute this next section.

    if x is not None:
        prob.x = x

    logger.info(f'Beginning parallel finite difference gradient calculation')

    x0 = prob.x
    # Make sure all leaders have the same x0.
    mpi.comm_leaders.Bcast(x0)
    #logger.info('  nparams: {}, nfuncs: {}'.format(prob.dof_size, dofs.nfuncs))
    logger.info(f'nparams: {prob.dof_size}')
    logger.info(f'x0: {x0}')

    # Set up the list of parameter values to try
    if centered:
        nevals_jac = 2 * prob.dof_size
        xs = np.zeros((prob.dof_size, nevals_jac))
        for j in range(prob.dof_size):
            xs[:, 2 * j] = x0[:]  # I don't think I need np.copy(), but not 100% sure.
            xs[j, 2 * j] = x0[j] + eps
            xs[:, 2 * j + 1] = x0[:]
            xs[j, 2 * j + 1] = x0[j] - eps
    else:
        # 1-sided differences
        nevals_jac = prob.dof_size + 1
        xs = np.zeros((prob.dof_size, nevals_jac))
        xs[:, 0] = x0[:]
        for j in range(prob.dof_size):
            xs[:, j + 1] = x0[:]
            xs[j, j + 1] = x0[j] + eps

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
        nvals = mpi.comm_leaders.bcast(prob.dof_size)
        evals = np.zeros((nvals, nevals_jac))
    # Do the hard work of evaluating the functions.
    for j in range(nevals_jac):
        # Handle only this group's share of the work:
        if np.mod(j, mpi.ngroups) == mpi.rank_leaders:
            mpi.mobilize_workers(CALCULATE_F)
            x = xs[:, j]
            mpi.comm_groups.bcast(x, root=0)
            prob.x = x

            try:
                f = prob.objective()
            except:
                logger.info("Exception caught during function evaluation")
                f = np.full(prob.dofs.nvals, 1.0e12)

            if evals is None and mpi.proc0_world:
                nvals = mpi.comm_leaders.bcast(prob.dof_size)
                evals = np.zeros((nvals, nevals_jac))

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
    jac = np.zeros((prob.dof_size, prob.dof_size))
    if centered:
        for j in range(prob.dof_size):
            jac[:, j] = (evals[:, 2 * j] - evals[:, 2 * j + 1]) / (2 * eps)
    else:
        # 1-sided differences:
        for j in range(prob.dof_size):
            jac[:, j] = (evals[:, j + 1] - evals[:, 0]) / eps

    # Weird things may happen if we do not reset the state vector
    # to x0:
    prob.x = x0
    return jac, xs, evals


def least_squares_mpi_solve(prob: LeastSquaresProblem,
                            mpi: MpiPartition,
                            grad: bool = None,
                            **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    MPI. All MPI processes (including group leaders and workers)
    should call this function.

    Args:
        prob: Optimizable object defining the objective function(s) and
              parameter space.
        mpi: A MpiPartition object, storing the information about how
             the pool of MPI processes is divided into worker groups.
        grad: Whether to use a gradient-based optimization algorithm, as
              opposed to a gradient-free algorithm. If unspecified, a
              gradient-based algorithm will be used if ``prob`` has gradient
              information available, otherwise a gradient-free algorithm
              will be used by default. If you set ``grad=True`` for a problem
              in which gradient information is not available,
              finite-difference gradients will be used.
        kwargs: Any arguments to pass to
                `scipy.optimize.least_squares <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>`_.
                For instance, you can supply ``max_nfev=100`` to set
                the maximum number of function evaluations (not counting
                finite-difference gradient evaluations) to 100. Or, you
                can supply ``method`` to choose the optimization algorithm.
    """
    if MPI is None:
        raise RuntimeError(
            "least_squares_mpi_solve requires the mpi4py package.")

    logger.info("Beginning solve.")
    #prob._init()
    #if grad is None:
    #    grad = prob.grad_avail

    x = np.copy(prob.x)  # For use in Bcast later.

    objective_file = None
    residuals_file = None
    datalog_started = False
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

        try:
            #f_unshifted = prob.dofs.f(x)
            unweighted_residuals = prob.unweighted_residuals(x)
            logger.debug(f"unweighted residuals:\n {unweighted_residuals}")
            residuals = prob.residuals()
            logger.debug(f"residuals:\n {residuals}")
        except:
            #f_unshifted = np.full(prob.dofs.nvals, 1.0e12)
            unweighted_residuals = np.full(prob.get_parent_return_fns_no(),
                                           1.0e12)
            residuals = np.full(prob.get_parent_return_fns_no(), 1.0e12)
            logger.info("Exception caught during function evaluation.")

        #f_shifted = prob.f_from_unshifted(f_unshifted)
        #objective_val = prob.objective_from_shifted_f(f_shifted)
        objective_val = prob.objective()

        nonlocal datalog_started, objective_file, residuals_file, nevals

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not datalog_started:
            # Initialize log file
            datalog_started = True
            datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            objective_file = open(f"objective_{datestr}.dat", 'w')
            objective_file.write("Problem type:\nleast_squares")
            objective_file.write("function_evaluation,seconds")

            residuals_file = open(f"residuals_{datestr}.dat", 'w')
            residuals_file.write("Problem type:\nleast_squares")
            residuals_file.write("function_evaluation,seconds")

            for j in range(prob.dof_size):
                objective_file.write(f",x({j})")
            objective_file.write(",objective_function\n")

            for j in range(prob.dof_size):
                residuals_file.write(f",x({j})")
            residuals_file.write(",objective_function")
            for j in range(prob.dof_size):
                residuals_file.write(f",F({j})")
            residuals_file.write("\n")

        del_t = time() - start_time
        objective_file.write(f"{nevals:6d},{del_t:12.4e}")
        for xj in x:
            objective_file.write(f",{xj:24.16e}")
        objective_file.write(f",{objective_val:24.16e}\n")
        objective_file.flush()

        residuals_file.write(f"{nevals:6d},{del_t:12.4e}")
        for xj in x:
            residuals_file.write(f",{xj:24.16e}")
        residuals_file.write(f",{objective_val:24.16e}")
        for fj in unweighted_residuals:
            residuals_file.write(f",{fj:24.16e}")
        residuals_file.write("\n")
        residuals_file.flush()

        nevals += 1
        return residuals

    def _jac_proc0(x):
        """
        This function is used for least_squares_mpi_solve.  It is similar
        to LeastSquaresProblem.jac, except this version is called only by
        proc 0 while workers are in the worker loop.
        """
        return None
        # TODO: reimplement this after jac evaluation is implemented
        if prob.grad_avail:
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

            jac, xs, evals = fd_jac_mpi(prob, mpi, x)

            # Write function evaluations to the files
            nonlocal datalog_started, objective_file, residuals_file, nevals
            nevals_jac = evals.shape[1]
            for j in range(nevals_jac):
                objective_val = prob.objective_from_unshifted_f(evals[:, j])

                del_t = time() - start_time
                objective_file.write(f"{nevals:6d},{del_t:12.4e}")
                for xj in xs[:, j]:
                    objective_file.write(f",{xj:24.16e}")
                objective_file.write(f",{objective_val:24.16e}")
                objective_file.write("\n")
                objective_file.flush()

                residuals_file.write(f"{nevals:6d},{del_t:12.4e}")
                for xj in xs[:, j]:
                    residuals_file.write(f",{xj:24.16e}")
                residuals_file.write(f",{objective_val:24.16e}")
                for fj in evals[:, j]:
                    residuals_file.write(f",{fj:24.16e}")
                residuals_file.write("\n")
                residuals_file.flush()

                nevals += 1

            return prob.scale_dofs_jac(jac)

    # Send group leaders and workers into their respective loops:
    leaders_action = lambda mpi2, data: _mpi_leaders_task(mpi, prob, data)
    workers_action = lambda mpi2, data: _mpi_workers_task(mpi, prob, data)
    mpi.apart(leaders_action, workers_action)

    if mpi.proc0_world:
        # proc0_world does this block, running the optimization.
        x0 = np.copy(prob.x)
        #print("x0:",x0)
        # Call scipy.optimize:
        if grad:
            logger.info("Using derivative free method despite derivatives given")
            #result = least_squares(_f_proc0, x0, verbose=2, jac=_jac_proc0, **kwargs)
            result = least_squares(_f_proc0, x0, verbose=2, **kwargs)
        else:
            logger.info("Using derivative-free method")
            result = least_squares(_f_proc0, x0, verbose=2, **kwargs)

        logger.info("Completed solve.")
        x = result.x

        objective_file.close()
        residuals_file.close()

    # Stop loops for workers and group leaders:
    mpi.together()

    datalog_started = False
    logger.info("Completed solve.")

    # Finally, make sure all procs get the optimal state vector.
    mpi.comm_world.Bcast(x)
    logger.debug(f'After Bcast, x={x}')
    # Set Parameters to their values for the optimum
    prob.x = x

