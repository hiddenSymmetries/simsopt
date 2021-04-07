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

logger = logging.getLogger(__name__)

# Constants for signaling to workers what task to do:
CALCULATE_F = 1
CALCULATE_JAC = 2
CALCULATE_FD_JAC = 3

def mpi_leaders_task(mpi, dofs, data):
    """
    This function is called by group leaders when
    MpiPartition.leaders_loop() receives a signal to do something.

    We have to take a "data" argument, but there is only 1 task we
    would do, so we don't use it.
    """
    logger.debug('mpi_leaders_task')

    # x is a buffer for receiving the state vector:
    x = np.empty(dofs.nparams, dtype='d')
    # If we make it here, we must be doing a fd_jac_par
    # calculation, so receive the state vector: mpi4py has
    # separate bcast and Bcast functions!!  comm.Bcast(x,
    # root=0)
    x = mpi.comm_leaders.bcast(x, root=0)
    logger.debug('mpi_leaders_loop x={}'.format(x))
    dofs.set(x)
    fd_jac_mpi(dofs, mpi)


def mpi_workers_task(mpi, dofs, data):
    """
    This function is called by worker processes when
    MpiPartition.workers_loop() receives a signal to do something.
    """
    logger.debug('mpi_workers_task')

    # x is a buffer for receiving the state vector:
    x = np.empty(dofs.nparams, dtype='d')
    # If we make it here, we must be doing a fd_jac_par
    # calculation, so receive the state vector: mpi4py has
    # separate bcast and Bcast functions!!  comm.Bcast(x,
    # root=0)
    x = mpi.comm_groups.bcast(x, root=0)
    logger.debug('worker_loop worker x={}'.format(x))
    dofs.set(x)

    # We don't store or do anything with f() or jac(), because
    # the group leader will handle that.
    if data == CALCULATE_F:
        try:
            dofs.f()
        except:
            logger.info("Exception caught by worker during dofs.f()")

    elif data == CALCULATE_JAC:
        try:
            dofs.jac()
        except:
            logger.info("Exception caught by worker during dofs.jac()")

    else:
        raise ValueError('Unexpected data in worker_loop')


def fd_jac_mpi(dofs, mpi, x=None, eps=1e-7, centered=False):
    """
    Compute the finite-difference Jacobian of the functions in dofs
    with respect to all non-fixed degrees of freedom. Parallel
    function evaluations will be used.

    If the argument x is not supplied, the Jacobian will be
    evaluated for the present state vector. If x is supplied, then
    first get_dofs() will be called for each object to set the
    global state vector to x.

    The mpi argument should be an MpiPartition.

    There are 2 ways to call this function. In method 1, all procs
    (including workers) call this function (so mpi.is_apart is
    False). In this case, the worker loop will be started
    automatically. In method 2, the worker loop has already been
    started before this function is called, as would be the case
    in least_squares_mpi_solve(). Then only the group leaders
    call this function.

    This function returns a 3-tuple. The first entry is the
    Jacobian. The second entry is a matrix, the columns of which give
    all the values of x at which the functions were evaluated. The
    third entry is a matrix, the colums of which give the
    corresponding values of the functions.
    """
    if MPI is None:
        raise RuntimeError("fd_jac_mpi requires the mpi4py package.")

    apart_at_start = mpi.is_apart
    if not apart_at_start:
        mpi.worker_loop(lambda mpi2, data: mpi_workers_task(mpi2, dofs, data))
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
    if centered:
        nevals_jac = 2 * dofs.nparams
        xs = np.zeros((dofs.nparams, nevals_jac))
        for j in range(dofs.nparams):
            xs[:, 2 * j] = x0[:] # I don't think I need np.copy(), but not 100% sure.
            xs[j, 2 * j] = x0[j] + eps
            xs[:, 2 * j + 1] = x0[:]
            xs[j, 2 * j + 1] = x0[j] - eps
    else:
        # 1-sided differences
        nevals_jac = dofs.nparams + 1
        xs = np.zeros((dofs.nparams, nevals_jac))
        xs[:, 0] = x0[:]
        for j in range(dofs.nparams):
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
            
            try:
                f = dofs.f()
            except:
                logger.info("Exception caught during function evaluation")
                f = np.full(prob.dofs.nvals, 1.0e12)
                
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
    if centered:
        for j in range(dofs.nparams):
            jac[:, j] = (evals[:, 2 * j] - evals[:, 2 * j + 1]) / (2 * eps)
    else:
        # 1-sided differences:
        for j in range(dofs.nparams):
            jac[:, j] = (evals[:, j + 1] - evals[:, 0]) / eps

    # Weird things may happen if we do not reset the state vector
    # to x0:
    dofs.set(x0)
    return jac, xs, evals


def least_squares_mpi_solve(prob, mpi, grad=None, **kwargs):
    """
    Solve a nonlinear-least-squares minimization problem using
    MPI. All MPI processes (including group leaders and workers)
    should call this function.

    prob should be an instance of LeastSquaresProblem.

    mpi should be an instance of MpiPartition.

    kwargs allows you to pass any arguments to scipy.optimize.minimize.
    """
    if MPI is None:
        raise RuntimeError("least_squares_mpi_solve requires the mpi4py package.")
    
    logger.info("Beginning solve.")
    prob._init()
    if grad is None:
        grad = prob.dofs.grad_avail

    x = np.copy(prob.x) # For use in Bcast later.

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

        try:
            f_unshifted = prob.dofs.f(x)
        except:
            f_unshifted = np.full(prob.dofs.nvals, 1.0e12)
            logger.info("Exception caught during function evaluation.")

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

    # End of _f_proc0

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

    # End of _jac_proc0
    
    # Send group leaders and workers into their respective loops:
    leaders_action = lambda mpi2, data: mpi_leaders_task(mpi, prob.dofs, data)
    workers_action = lambda mpi2, data: mpi_workers_task(mpi, prob.dofs, data)
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

