# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains functions related to MPI parallelization.
"""

import numpy as np
from mpi4py import MPI
import logging
#from simsopt.core.mpi_solve import fd_jac_mpi
from scipy.optimize import least_squares

STOP = 0
CALCULATE_F = 1
CALCULATE_JAC = 2
CALCULATE_FD_JAC = 3

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

class MpiPartition():
    def __init__(self, ngroups=None, comm_world=MPI.COMM_WORLD):
        self.is_apart = False
        self.comm_world = comm_world
        self.rank_world = comm_world.Get_rank()
        self.nprocs_world = comm_world.Get_size()
        self.proc0_world = (self.rank_world == 0)
        
        if ngroups is None:
            ngroups = self.nprocs_world
        # Force ngroups to be in the range [1, nprocs_world]
        if ngroups < 1:
            ngroups = 1
            logger.info('Raising ngroups to 1')
        if ngroups > self.nprocs_world:
            ngroups = self.nprocs_world
            logger.info('Lowering ngroups to {}'.format(ngroups))
        self.ngroups = ngroups

        self.group = int(np.floor((self.rank_world * ngroups) / self.nprocs_world))

        # Set up the "groups" communicator:
        self.comm_groups = self.comm_world.Split(color=self.group, key=self.rank_world)
        self.rank_groups = self.comm_groups.Get_rank()
        self.nprocs_groups = self.comm_groups.Get_size()
        self.proc0_groups = (self.rank_groups == 0)

        # Set up the "leaders" communicator:
        if self.proc0_groups:
            color = 0
        else:
            color = MPI.UNDEFINED
        self.comm_leaders = self.comm_world.Split(color=color, key=self.rank_world)
        if self.proc0_groups:
            self.rank_leaders = self.comm_leaders.Get_rank()
            self.nprocs_leaders = self.comm_leaders.Get_size()
        else:
            # We are not allowed to query the rank from procs that are
            # not members of comm_leaders.
            self.rank_leaders = -1
            self.nprocs_leaders = -1

    def write(self):
        """ Dump info about the MPI configuration """
        columns = ["rank_world","nprocs_world","group","ngroups","rank_groups","nprocs_groups","rank_leaders","nprocs_leaders"]
        data = [self.rank_world , self.nprocs_world , self.group , self.ngroups , self.rank_groups , self.nprocs_groups , self.rank_leaders , self.nprocs_leaders]

        # Each processor sends their data to proc0_world, and
        # proc0_world writes the result to the file in order.
        if self.proc0_world:
            # Print header row
            width = max(len(s) for s in columns) + 1
            print(",".join(s.rjust(width) for s in columns))
            print(",".join(str(s).rjust(width) for s in data))
            for tag in range(1, self.nprocs_world):
                data = self.comm_world.recv(tag=tag)
                print(",".join(str(s).rjust(width) for s in data))
        else:
            tag = self.rank_world
            self.comm_world.send(data, 0, tag)
            
    def mobilize_leaders(self, x):
        logger.debug('mobilize_leaders, x={}'.format(x))
        if not self.proc0_world:
            raise RuntimeError('Only proc 0 should call mobilize_leaders()')

        self.comm_leaders.bcast(CALCULATE_FD_JAC, root=0)

        # Next, broadcast the state vector to leaders:
        self.comm_leaders.bcast(x, root=0)

    def mobilize_workers(self, x, action):
        logger.debug('mobilize_workers, action={}, x={}'.format(action, x))
        if not self.proc0_groups:
            raise RuntimeError('Only group leaders should call mobilize_workers()')

        # First, notify workers that we will be doing a calculation:
        if action != CALCULATE_F and action != CALCULATE_JAC:
            raise ValueError('action must be either CALCULATE_F or CALCULATE_JAC')
        self.comm_groups.bcast(action, root=0)

        # Next, broadcast the state vector to workers:
        self.comm_groups.bcast(x, root=0)

    def stop_leaders(self):
        logger.debug('stop_leaders')
        if not self.proc0_world:
            raise RuntimeError('Only proc0_world should call stop_leaders()')

        data = STOP
        self.comm_leaders.bcast(data, root=0)

    def stop_workers(self):
        logger.debug('stop_workers')
        if not self.proc0_groups:
            raise RuntimeError('Only proc0_groups should call stop_workers()')

        data = STOP
        self.comm_groups.bcast(data, root=0)

    def leaders_loop(self, dofs):
        if self.proc0_world:
            logger.debug('proc0_world bypassing leaders_loop')
            return
        
        logger.debug('entering leaders_loop')

        # x is a buffer for receiving the state vector:
        x = np.empty(dofs.nparams, dtype='d')

        while True:
            # Wait for proc 0 to send us something:
            data = None
            data = self.comm_leaders.bcast(data, root=0)
            logger.debug('leaders_loop received {}'.format(data))
            if data == STOP:
                # Tell workers to stop
                #self.comm_groups.bcast(STOP, root=0)
                break

            # If we make it here, we must be doing a fd_jac_par
            # calculation, so receive the state vector: mpi4py has
            # separate bcast and Bcast functions!!  comm.Bcast(x,
            # root=0)
            x = self.comm_leaders.bcast(x, root=0)
            logger.debug('leaders_loop x={}'.format(x))
            dofs.set(x)

            fd_jac_mpi(dofs, self)

        logger.debug('leaders_loop end')

    def worker_loop(self, dofs):
        if self.proc0_groups:
            logger.debug('bypassing worker_loop since proc0_groups')
            return
        
        logger.debug('entering worker_loop')

        # x is a buffer for receiving the state vector:
        x = np.empty(dofs.nparams, dtype='d')

        while True:
            # Wait for the group leader to send us something:
            data = None
            data = self.comm_groups.bcast(data, root=0)
            logger.debug('worker_loop worker received {}'.format(data))
            if data == STOP:
                break

            # If we make it here, we must be doing a calculation, so
            # receive the state vector:
            # mpi4py has separate bcast and Bcast functions!!
            #comm.Bcast(x, root=0)
            x = self.comm_groups.bcast(x, root=0)
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

        logger.debug('worker_loop end')

    def apart(self, dofs):
        """
        Send workers and group leaders off to their respective loops to
        wait for instructions from their group leader or
        proc0_world, respectively.
        """
        self.is_apart = True
        if self.proc0_world:
            pass
        elif self.proc0_groups:
            self.leaders_loop(dofs)
        else:
            self.worker_loop(dofs)

    def together(self):
        """
        Bring workers and group leaders back from their respective loops.
        """
        if self.proc0_world:
            self.stop_leaders() # Proc0_world stops the leaders.

        if self.proc0_groups:
            self.stop_workers() # All group leaders stop their workers.

        self.is_apart = False

        
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
        mpi.worker_loop(dofs)
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
            mpi.mobilize_workers(xs[:, j], CALCULATE_F)
            dofs.set(xs[:, j])
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
    mpi.mobilize_workers(x, CALCULATE_F)
    return prob.f(x)


def _jac_proc0(x, prob, mpi):
    """
    This function is used for least_squares_mpi_solve.  It is similar
    to LeastSquaresProblem.jac, except this version is called only by proc 0 while workers
    are in the worker loop.
    """
    if prob.dofs.grad_avail:
        # proc0_world calling mobilize_workers will mobilize only group 0.
        mpi.mobilize_workers(x, CALCULATE_JAC)
        return prob.jac(x)
    else:
        # Evaluate Jacobian using fd_jac_mpi
        mpi.mobilize_leaders(x)
        return prob.scale_dofs_jac(fd_jac_mpi(prob.dofs, mpi, x))


def least_squares_mpi_solve(prob, mpi, grad=None):
    """
    Solve a nonlinear-least-squares minimization problem using MPI.

    prob should be an instance of LeastSquaresProblem.

    mpi should be an instance of MpiPartition.
    """
    logger.info("Beginning solve.")
    prob._init()
    if grad is None:
        grad = prob.dofs.grad_avail

    x = np.copy(prob.x) # For use in Bcast later.

    # Send group leaders and workers into their respective loops:
    mpi.apart(prob.dofs)

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

