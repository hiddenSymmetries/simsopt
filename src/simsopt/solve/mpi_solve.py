# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides two main functions, fd_jac_mpi and
least_squares_mpi_solve. Also included are some functions that help in
the operation of these main functions.
"""

from mpi4py import MPI
import numpy as np
from scipy.optimize import least_squares
import logging
from .dofs import Dofs
from .util import isnumber
from .optimizable import function_from_user, Target

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

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
        dofs.f()
    elif data == CALCULATE_JAC:
        dofs.jac()
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
    """

    apart_at_start = mpi.is_apart
    if not apart_at_start:
        mpi.worker_loop(lambda mpi2, data: mpi_workers_task(mpi2, dofs, data))
    if not mpi.proc0_groups:
        return

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
        nevals = 2 * dofs.nparams
        xs = np.zeros((dofs.nparams, nevals))
        for j in range(dofs.nparams):
            xs[:, 2 * j] = x0[:] # I don't think I need np.copy(), but not 100% sure.
            xs[j, 2 * j] = x0[j] + eps
            xs[:, 2 * j + 1] = x0[:]
            xs[j, 2 * j + 1] = x0[j] - eps
    else:
        # 1-sided differences
        nevals = dofs.nparams + 1
        xs = np.zeros((dofs.nparams, nevals))
        xs[:, 0] = x0[:]
        for j in range(dofs.nparams):
            xs[:, j + 1] = x0[:]
            xs[j, j + 1] = x0[j] + eps

    # proc0_world will be responsible for detecting nvals, since
    #proc0_world always does at least 1 function evaluation. Other
    #procs cannot be trusted to evaluate nvals because they may
    #not have any function evals, in which case they never create
    #"evals", so the MPI reduce would fail.

    #evals = np.zeros((dofs.nfuncs, nevals))
    evals = None
    if not mpi.proc0_world:
        # All procs other than proc0_world should initialize evals
        # before the nevals loop, since they may not have any
        # evals.
        dofs.nvals = mpi.comm_leaders.bcast(dofs.nvals)
        evals = np.zeros((dofs.nvals, nevals))
    # Do the hard work of evaluating the functions.
    for j in range(nevals):
        # Handle only this group's share of the work:
        if np.mod(j, mpi.ngroups) == mpi.rank_leaders:
            mpi.mobilize_workers(CALCULATE_F)
            x = xs[:, j]
            mpi.comm_groups.bcast(x, root=0)
            dofs.set(x)
            f = dofs.f()
            if evals is None and mpi.proc0_world:
                dofs.nvals = mpi.comm_leaders.bcast(dofs.nvals)
                evals = np.zeros((dofs.nvals, nevals))
            evals[:, j] = f
            #evals[:, j] = np.array([f() for f in dofs.funcs])

    # Combine the results from all groups:
    evals = mpi.comm_leaders.reduce(evals, op=MPI.SUM, root=0)

    if not apart_at_start:
        mpi.stop_workers()

    # Only proc0_world will actually have the Jacobian.
    if not mpi.proc0_world:
        return None

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
    return jac


def _f_proc0(x, prob, mpi):
    """
    This function is used for least_squares_mpi_solve.  It is similar
    to LeastSquaresProblem.f, except this version is called only by
    proc 0 while workers are in the worker loop.
    """
    mpi.mobilize_workers(CALCULATE_F)
    # Send workers the state vector:
    mpi.comm_groups.bcast(x, root=0)
    
    return prob.f(x)


def _jac_proc0(x, prob, mpi):
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

        return prob.scale_dofs_jac(fd_jac_mpi(prob.dofs, mpi, x))


def least_squares_mpi_solve(prob, mpi, grad=None):
    """
    Solve a nonlinear-least-squares minimization problem using
    MPI. All MPI processes (including group leaders and workers)
    should call this function.

    prob should be an instance of LeastSquaresProblem.

    mpi should be an instance of MpiPartition.
    """
    logger.info("Beginning solve.")
    prob._init()
    if grad is None:
        grad = prob.dofs.grad_avail

    x = np.copy(prob.x) # For use in Bcast later.

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
            result = least_squares(_f_proc0, x0, verbose=2, jac=_jac_proc0, args=(prob, mpi))
        else:
            logger.info("Using derivative-free method")
            print("Using derivative-free method")
            result = least_squares(_f_proc0, x0, verbose=2, args=(prob, mpi))

        logger.info("Completed solve.")
        x = result.x
        
    # Stop loops for workers and group leaders:
    mpi.together()

    # Finally, make sure all procs get the optimal state vector.
    mpi.comm_world.Bcast(x)
    logger.debug('After Bcast, x={}'.format(x))
    #print("optimum x:",result.x)
    #print("optimum residuals:",result.fun)
    #print("optimum cost function:",result.cost)
    # Set Parameters to their values for the optimum
    prob.dofs.set(x)

