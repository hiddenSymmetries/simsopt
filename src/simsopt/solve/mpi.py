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
import traceback

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.optimize import NonlinearConstraint, LinearConstraint

try:
    from mpi4py import MPI
except ImportError as err:
    MPI = None

from .._core.optimizable import Optimizable
from ..util.mpi import MpiPartition
from .._core.finite_difference import MPIFiniteDifference
from ..objectives.least_squares import LeastSquaresProblem
from ..objectives.constrained import ConstrainedProblem

logger = logging.getLogger(__name__)

# Constants for signaling to workers what task to do:
CALCULATE_F = 1
CALCULATE_JAC = 2
CALCULATE_FD_JAC = 3
CALCULATE_NLC = 4

__all__ = ['least_squares_mpi_solve', 'constrained_mpi_solve']


def _mpi_workers_task(mpi: MpiPartition,
                      prob: Optimizable):
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
    logger.debug(f'worker loop worker x={x}')
    prob.x = x

    # We don't store or do anything with f() or jac(), because
    # the group leader will handle that.
    try:
        prob.unweighted_residuals()
    except:
        logger.warning("Exception caught by worker during residual "
                       "evaluation in worker loop")
        traceback.print_exc()  # Print traceback


def least_squares_mpi_solve(prob: LeastSquaresProblem,
                            mpi: MpiPartition,
                            grad: bool = False,
                            abs_step: float = 1.0e-7,
                            rel_step: float = 0.0,
                            diff_method: str = "forward",
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
             a gradient-free algorithm
             will be used by default. If you set ``grad=True``
             finite-difference gradients will be used.
        abs_step: Absolute step size for finite difference jac evaluation
        rel_step: Relative step size for finite difference jac evaluation
        diff_method: Differentiation strategy. Options are "centered", and
             "forward". If ``centered``, centered finite differences will
             be used. If ``forward``, one-sided finite differences will
             be used. Else, error is raised.
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
            unweighted_residuals = prob.unweighted_residuals(x)
            logger.debug(f"unweighted residuals in _f_proc0:\n {unweighted_residuals}")
            residuals = prob.residuals()
            logger.debug(f"residuals in _f_proc0:\n {residuals}")
        except:
            unweighted_residuals = np.full(prob.parent_return_fns_no, 1.0e12)
            residuals = np.full(prob.parent_return_fns_no, 1.0e12)
            logger.info("Exception caught during function evaluation.")

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
            objective_file.write(f"Problem type:\nleast_squares\nnparams:\n{prob.dof_size}\n")
            objective_file.write("function_evaluation,seconds")

            residuals_file = open(f"residuals_{datestr}.dat", 'w')
            residuals_file.write(f"Problem type:\nleast_squares\nnparams:\n{prob.dof_size}\n")
            residuals_file.write("function_evaluation,seconds")

            for j in range(prob.dof_size):
                objective_file.write(f",x({j})")
            objective_file.write(",objective_function\n")

            for j in range(prob.dof_size):
                residuals_file.write(f",x({j})")
            residuals_file.write(",objective_function")
            for j in range(len(residuals)):
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
        logger.debug(f"residuals are {residuals}")
        return residuals

    # For MPI finite difference gradient, get the worker and leader action from
    # MPIFiniteDifference
    if grad:
        with MPIFiniteDifference(prob.residuals, mpi, abs_step=abs_step,
                                 rel_step=rel_step, diff_method=diff_method) as fd:
            if mpi.proc0_world:
                # proc0_world does this block, running the optimization.
                x0 = np.copy(prob.x)
                logger.info("Using finite difference method implemented in "
                            "SIMSOPT for evaluating gradient")
                result = least_squares(_f_proc0, x0, jac=fd.jac, verbose=2,
                                       **kwargs)

    else:
        leaders_action = lambda mpi, data: None
        workers_action = lambda mpi, data: _mpi_workers_task(mpi, prob)
        # Send group leaders and workers into their respective loops:
        mpi.apart(leaders_action, workers_action)

        if mpi.proc0_world:
            # proc0_world does this block, running the optimization.
            x0 = np.copy(prob.x)
            logger.info("Using derivative-free method")
            result = least_squares(_f_proc0, x0, verbose=2, **kwargs)

        # Stop loops for workers and group leaders:
        mpi.together()

    if mpi.proc0_world:
        x = result.x

        objective_file.close()
        residuals_file.close()

    datalog_started = False
    logger.info("Completed solve.")

    # Finally, make sure all procs get the optimal state vector.
    mpi.comm_world.Bcast(x)
    logger.debug(f'After Bcast, x={x}')
    # Set Parameters to their values for the optimum
    prob.x = x


def _constrained_mpi_workers_task(mpi: MpiPartition,
                                  prob: Optimizable,
                                  data: int):
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
    logger.debug(f'worker loop worker x={x}')
    prob.x = x

    # We don't store or do anything with f() or jac(), because
    # the group leader will handle that.
    if data == CALCULATE_F:
        try:
            prob.objective()
        except:
            logger.warning("Exception caught by worker during objective"
                           "evaluation in worker loop")
            traceback.print_exc()  # Print traceback
    elif data == CALCULATE_NLC:
        try:
            prob.nonlinear_constraints()
        except:
            logger.warning("Exception caught by worker during constraint"
                           "evaluation in worker loop")
            traceback.print_exc()  # Print traceback


def constrained_mpi_solve(prob: ConstrainedProblem,
                          mpi: MpiPartition,
                          grad: bool = False,
                          abs_step: float = 1.0e-7,
                          rel_step: float = 0.0,
                          diff_method: str = "forward",
                          opt_method: str = "SLSQP",
                          options: dict = None):
    r"""
    Solve a constrained minimization problem using
    MPI. All MPI processes (including group leaders and workers)
    should call this function.

    Args:
        prob: :obj:`~simsopt.objectives.ConstrainedProblem` object defining the
            objective function, parameter space, and constraints.
        mpi: A MpiPartition object, storing the information about how
            the pool of MPI processes is divided into worker groups.
        grad: Whether to use a gradient-based optimization algorithm, as
            opposed to a gradient-free algorithm. If unspecified, a
            a gradient-free algorithm
            will be used by default. If you set ``grad=True``
            finite-difference gradients will be used.
        abs_step: Absolute step size for finite difference jac evaluation
        rel_step: Relative step size for finite difference jac evaluation
        diff_method: Differentiation strategy. Options are ``"centered"`` and
            ``"forward"``. If ``"centered"``, centered finite differences will
            be used. If ``"forward"``, one-sided finite differences will
            be used. For other values, an error is raised.
        opt_method: Constrained solver to use: One of ``"SLSQP"``,
            ``"trust-constr"``, or ``"COBYLA"``. Use ``"COBYLA"`` for
            derivative-free optimization. See
            `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_
            for a description of the methods.
        options: dict, ``options`` keyword which is passed to
            `scipy.optimize.minimize <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize>`_.
    """
    if MPI is None:
        raise RuntimeError(
            "cosntrained_mpi_solve requires the mpi4py package.")
    logger.info("Beginning solve.")

    x = np.copy(prob.x)  # For use in Bcast later.

    objective_file = None
    constraint_file = None
    objective_datalog_started = False
    constraint_datalog_started = False
    n_objective_evals = 0
    n_constraint_evals = 0
    start_time = time()

    def _f_proc0(x):
        """
        This function is used for constrained_mpi_solve.  It is called only by
        proc 0 while workers are in the worker loop.
        """
        logger.debug("Entering _f_proc0")
        mpi.mobilize_workers(CALCULATE_F)
        # Send workers the state vector:
        mpi.comm_groups.bcast(x, root=0)
        logger.debug("Past bcast in _f_proc0")

        try:
            objective_val = prob.objective(x)
            logger.debug(f"objective in _f_proc0:\n {objective_val}")
        except:
            objective_val = prob.fail
            logger.info("Exception caught during function evaluation.")

        nonlocal objective_datalog_started, objective_file, n_objective_evals

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not objective_datalog_started:
            # Initialize log file
            objective_datalog_started = True
            datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            objective_file = open(f"objective_{datestr}.dat", 'w')
            objective_file.write(f"Problem type:\nconstrained\nnparams:\n{prob.dof_size}\n")
            objective_file.write("function_evaluation,seconds")

            for j in range(prob.dof_size):
                objective_file.write(f",x({j})")
            objective_file.write(",objective_function\n")

        del_t = time() - start_time
        objective_file.write(f"{n_objective_evals:6d},{del_t:12.4e}")
        for xj in x:
            objective_file.write(f",{xj:24.16e}")
        objective_file.write(f",{objective_val:24.16e}\n")
        objective_file.flush()

        n_objective_evals += 1
        logger.debug(f"objective is {objective_val}")
        return objective_val

    # wrap the constraints for logging
    def _nlc_proc0(x):
        """
        This function is used for constrained_mpi_solve.  It is called only by
        proc 0 while workers are in the worker loop.
        """
        logger.debug("Entering _nlc_proc0")
        mpi.mobilize_workers(CALCULATE_NLC)
        # Send workers the state vector:
        mpi.comm_groups.bcast(x, root=0)
        logger.debug("Past bcast in _nlc_proc0")

        try:
            constraint_val = prob.nonlinear_constraints(x)
            logger.debug(f"constraints in _nlc_proc0:\n {constraint_val}")
        except:
            constraint_val = np.full(prob.nvals, 1.0e12)
            logger.info("Exception caught during function evaluation.")

        nonlocal constraint_datalog_started, constraint_file, n_constraint_evals

        # Since the number of terms is not known until the first
        # evaluation of the objective function, we cannot write the
        # header of the output file until this first evaluation is
        # done.
        if not constraint_datalog_started:
            # Initialize log file
            constraint_datalog_started = True
            datestr = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            constraint_file = open(f"constraint_{datestr}.dat", 'w')
            constraint_file.write(f"Problem type:\nconstrained\nnparams:\n{prob.dof_size}\n")
            constraint_file.write("function_evaluation,seconds")

            for j in range(prob.dof_size):
                constraint_file.write(f",x({j})")
            constraint_file.write(",constraint_function\n")
            for j in range(len(constraint_val)):
                constraint_file.write(f",F({j})")
            constraint_file.write("\n")

        del_t = time() - start_time
        constraint_file.write(f"{n_constraint_evals:6d},{del_t:12.4e}")
        for xj in x:
            constraint_file.write(f",{xj:24.16e}")
        for fj in constraint_val:
            constraint_file.write(f",{fj:24.16e}")
        constraint_file.write("\n")
        constraint_file.flush()

        n_constraint_evals += 1
        logger.debug(f"constraints are {constraint_val}")
        return constraint_val

    # prepare bounds
    bounds = list(zip(*prob.bounds))

    # prepare linear constraints
    constraints = []
    if prob.has_lc:
        constraints.append(LinearConstraint(prob.A_lc, lb=prob.l_lc, ub=prob.u_lc))

    # For MPI finite difference gradient, get the worker and leader action from
    # MPIFiniteDifference
    if grad:
        with MPIFiniteDifference(prob.all_funcs, mpi, abs_step=abs_step,
                                 rel_step=rel_step, diff_method=diff_method) as fd:

            def obj_jac(x):
                # dummy wrapper for batch finite difference
                return fd.jac(x)[0]

            if mpi.proc0_world:
                if prob.has_nlc:
                    def nlc_jac(x):
                        # dummy wrapper for batch finite difference
                        return fd.jac(x)[1:]
                    nlc = NonlinearConstraint(_nlc_proc0, lb=-np.inf, ub=0.0, jac=nlc_jac)
                    constraints.append(nlc)

                # proc0_world does this block, running the optimization.
                x0 = np.copy(prob.x)
                logger.info("Using finite difference method implemented in "
                            "SIMSOPT for evaluating gradient")
                result = minimize(_f_proc0, x0, jac=obj_jac,
                                  bounds=bounds, constraints=constraints,
                                  method=opt_method, options=options)

    else:

        leaders_action = lambda mpi, data: None
        workers_action = lambda mpi, data: _constrained_mpi_workers_task(mpi, prob, data)
        # Send group leaders and workers into their respective loops:
        mpi.apart(leaders_action, workers_action)

        if mpi.proc0_world:
            # proc0_world does this block, running the optimization.
            if prob.has_nlc:
                nlc = NonlinearConstraint(_nlc_proc0, lb=-np.inf, ub=0.0)
                constraints.append(nlc)
            x0 = np.copy(prob.x)
            logger.info("Using derivative-free method")
            result = minimize(_f_proc0, x0,
                              bounds=bounds, constraints=constraints,
                              method=opt_method, options=options)

        # Stop loops for workers and group leaders:
        mpi.together()

    if mpi.proc0_world:
        x = result.x

        objective_file.close()
        if prob.has_nlc:
            constraint_file.close()

    datalog_started = False
    logger.info("Completed solve.")

    # Finally, make sure all procs get the optimal state vector.
    mpi.comm_world.Bcast(x)
    logger.debug(f'After Bcast, x={x}')
    # Set Parameters to their values for the optimum
    prob.x = x
