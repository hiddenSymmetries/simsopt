# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the collect_dofs() function, used by least-squares
and general optimization problems.
"""

import numpy as np
import types
import logging
from mpi4py import MPI
from .util import unique
from .optimizable import function_from_user
from .mpi import CALCULATE_F

logger = logging.getLogger('[{}]'.format(MPI.COMM_WORLD.Get_rank()) + __name__)

def get_owners(obj, owners_so_far=[]):
    """
    Given an object, return a list of objects that own any
    degrees of freedom, including both the input object and any of its
    dependendents, if there are any.
    """
    owners = [obj]
    # If the 'depends_on' attribute does not exist, assume obj does
    # not depend on the dofs of any other objects.
    if hasattr(obj, 'depends_on'):
        for j in obj.depends_on:
            subobj = getattr(obj, j)
            if subobj in owners_so_far:
                raise RuntimeError('Circular dependency detected among the objects')
            owners += get_owners(subobj, owners_so_far=owners)
    return owners
    

class Dofs():
    """
    This class holds data related to the vector of degrees of freedom
    that have been combined from multiple optimizable objects, keeping
    only the non-fixed dofs.
    """
    def __init__(self, funcs):
        """
        Given a list of optimizable functions, 

        funcs: A list/set/tuple of callable functions.

        returns: an object with the following attributes:
        x: A 1D numpy vector of variable dofs.

        dof_owners: A vector, with each element pointing to the object whose
        set_dofs() function should be called to update the corresponding
        dof.

        all_owners: A list of all objects involved in computing funcs,
        including those that do not directly own any of the non-fixed
        dofs.

        indices: A vector of ints, with each element giving the index in
        the owner's set_dofs method corresponding to this dof.

        names: A list of strings to identify each of the dofs.
        """
        
        # Convert all user-supplied function-like things to actual functions:
        funcs = [function_from_user(f) for f in funcs]
        
        # First, get a list of the objects and any objects they depend on:
        all_owners = []
        for j in funcs:
            all_owners += get_owners(j.__self__)

        # Eliminate duplicates, preserving order:
        all_owners = unique(all_owners)

        # Go through the objects, looking for any non-fixed dofs:
        x = []
        dof_owners = []
        indices = []
        mins = []
        maxs = []
        names = []
        fixed_merged = []
        for owner in all_owners:
            ox = owner.get_dofs()
            ndofs = len(ox)
            # If 'fixed' is not present, assume all dofs are not fixed
            if hasattr(owner, 'fixed'):
                fixed = list(owner.fixed)
            else:
                fixed = [False] * ndofs
            fixed_merged += fixed
            
            # Check for bound constraints:
            if hasattr(owner, 'mins'):
                omins = owner.mins
            else:
                omins = np.full(ndofs, np.NINF)
            if hasattr(owner, 'maxs'):
                omaxs = owner.maxs
            else:
                omaxs = np.full(ndofs, np.Inf)

            # Check for names:
            if hasattr(owner, 'names'):
                onames = [name + ' of ' + str(owner) for name in owner.names]
            else:
                onames = ['x[{}] of {}'.format(k, owner) for k in range(ndofs)]

            for jdof in range(ndofs):
                if not fixed[jdof]:
                    x.append(ox[jdof])
                    dof_owners.append(owner)
                    indices.append(jdof)
                    names.append(onames[jdof])
                    mins.append(omins[jdof])
                    maxs.append(omaxs[jdof])

        # Now repeat the process we just went through, but for only a
        # single element of funcs. The results will be needed to
        # handle gradient information.
        func_dof_owners = []
        func_indices = []
        func_fixed = []
        # For the global dof's, we store the dof_owners and indices
        # only for non-fixed dofs. But for the dof's associated with
        # each func, we store the dof_owners and indices for all dofs,
        # even if they are fixed. It turns out that this is the
        # information needed to convert the individual function
        # gradients into the global Jacobian.
        for func in funcs:
            owners = get_owners(func.__self__)
            f_dof_owners = []
            f_indices = []
            f_fixed = []
            for owner in owners:
                ox = owner.get_dofs()
                ndofs = len(ox)
                # If 'fixed' is not present, assume all dofs are not fixed
                if hasattr(owner, 'fixed'):
                    fixed = list(owner.fixed)
                else:
                    fixed = [False] * ndofs
                f_fixed += fixed

                for jdof in range(ndofs):
                    f_dof_owners.append(owner)
                    f_indices.append(jdof)
                    #if not fixed[jdof]:
                    #    f_dof_owners.append(owner)
                    #    f_indices.append(jdof)
            func_dof_owners.append(f_dof_owners)
            func_indices.append(f_indices)
            func_fixed.append(f_fixed)
            
        # Check whether derivative information is available:
        grad_avail = True
        grad_funcs = []
        for func in funcs:
            # Check whether a gradient function exists:
            owner = func.__self__
            grad_func_name = 'd' + func.__name__
            if not hasattr(owner, grad_func_name):
                grad_avail = False
                break
            # If we get here, a gradient function exists.
            grad_funcs.append(getattr(owner, grad_func_name))

        self.funcs = funcs
        self.nfuncs = len(funcs)
        self.nparams = len(x)
        self.dof_owners = dof_owners
        self.indices = np.array(indices)
        self.names = names
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.all_owners = all_owners
        self.func_dof_owners = func_dof_owners
        self.func_indices = func_indices
        self.func_fixed = func_fixed
        self.grad_avail = grad_avail
        self.grad_funcs = grad_funcs

    @property
    def x(self):
        """
        Call get_dofs() for each object, to assemble an up-to-date version
        of the state vector.
        """
        x = np.zeros(self.nparams)
        for owner in self.dof_owners:
            # In the next line, we make sure to cast the type to a
            # float. Otherwise get_dofs might return an array with
            # integer type.
            objx = np.array(owner.get_dofs(), dtype=np.dtype(float))
            for j in range(self.nparams):
                if self.dof_owners[j] == owner:
                    x[j] = objx[self.indices[j]]
        return x

    def f(self, x=None):
        """
        Return the vector of function values. Result is a 1D numpy array.

        If the argument x is not supplied, the functions will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        if x is not None:
            self.set(x)

        return np.array([f() for f in self.funcs])
    
    def jac(self, x=None):
        """
        Return the Jacobian, i.e. the gradients of all the functions that
        were originally supplied to Dofs(). Result is a 2D numpy
        array.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first set_dofs() will be called for each object to set the
        global state vector to x.
        """
        if not self.grad_avail:
            raise RuntimeError('Gradient information is not available for this Dofs()')

        if x is not None:
            self.set(x)
        
        results = np.zeros((self.nfuncs, self.nparams))
        # Loop over the rows of the Jacobian, i.e. over the functions
        # that were originally provided to Dofs():
        for jfunc in range(self.nfuncs):
            # Get the gradient of this particular function with
            # respect to all of it's dofs, which is a different set
            # from the global dofs:
            grad = np.array(self.grad_funcs[jfunc]()).flatten()
            # Above, we cast to a np.array and flatten() to be a bit
            # forgiving in case the user provides something other than
            # a plain 1D numpy array.
            
            # Match up the global dofs with the dofs for this particular gradient function:
            for jdof in range(self.nparams):
                for jgrad in range(len(self.func_indices[jfunc])):
                    # A global dof matches a dof for this function if the owners and indices both match:
                    if self.dof_owners[jdof] == self.func_dof_owners[jfunc][jgrad] and self.indices[jdof] == self.func_indices[jfunc][jgrad]:
                        results[jfunc, jdof] = grad[jgrad]
                        # If we find a match, we can exit the innermost loop:
                        break
                        
        #print('finite-difference Jacobian:')
        #fd_jac = self.fd_jac()
        #print(fd_jac)
        #print('analytic Jacobian:')
        #print(results)
        #print('difference:')
        #print(fd_jac - results)
        return results
            
    def set(self, x):
        """
        Call set_dofs() for each object, given a global state vector x.
        """
        # Idea behind the following loops: call set_dofs exactly once
        # once for each object, in case that improves performance at
        # all for the optimizable objects.
        for owner in self.all_owners:
            # In the next line, we make sure to cast the type to a
            # float. Otherwise get_dofs might return an array with
            # integer type.
            objx = np.array(owner.get_dofs(), dtype=np.dtype(float))
            for j in range(self.nparams):
                if self.dof_owners[j] == owner:
                    objx[self.indices[j]] = x[j]
            owner.set_dofs(objx)

    def fd_jac(self, x=None, eps=1e-7):
        """
        Compute the finite-difference Jacobian of the functions with
        respect to all non-fixed degrees of freedom. A
        centered-difference approximation is used, with step size eps.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first get_dofs() will be called for each object to set the
        global state vector to x.
        """

        if x is not None:
            self.set(x)
        
        logger.info('Beginning finite difference gradient calculation for functions ' + str(self.funcs))

        x0 = self.x
        logger.info('  nparams: {}, nfuncs: {}'.format(self.nparams, self.nfuncs))
        logger.info('  x0: ' + str(x0))

        jac = np.zeros((self.nfuncs, self.nparams))
        for j in range(self.nparams):
            x = np.copy(x0)

            x[j] = x0[j] + eps
            self.set(x)
            fplus = np.array([f() for f in self.funcs])

            x[j] = x0[j] - eps
            self.set(x)
            fminus = np.array([f() for f in self.funcs])

            # Centered differences:
            jac[:, j] = (fplus - fminus) / (2 * eps)

        return jac


    def fd_jac_par(self, mpi, x=None, eps=1e-7, centered=False):
        """
        Compute the finite-difference Jacobian of the functions with
        respect to all non-fixed degrees of freedom. Parallel function
        evaluations will be used.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first get_dofs() will be called for each object to set the
        global state vector to x.

        The mpi argument should be an MpiPartition.

        There are 2 ways to call this function. In method 1, all procs
        (including workers) call this function (so mpi.together is
        True). In this case, the worker loop will be started
        automatically. In method 2, the worker loop has already been
        started before this function is called, as would be the case
        in LeastSquaresProblem.solve(). Then only the group leaders
        call this function.
        """

        together_at_start = mpi.together
        if mpi.together:
            mpi.worker_loop(self)
        if not mpi.proc0_groups:
            return
        
        # Only group leaders execute this next section.
        
        if x is not None:
            self.set(x)
        
        logger.info('Beginning parallel finite difference gradient calculation for functions ' + str(self.funcs))

        x0 = self.x
        # Make sure all leaders have the same x0.
        mpi.comm_leaders.Bcast(x0)
        logger.info('  nparams: {}, nfuncs: {}'.format(self.nparams, self.nfuncs))
        logger.info('  x0: ' + str(x0))

        # Set up the list of parameter values to try
        if centered:
            nevals = 2 * self.nparams
            xs = np.zeros((self.nparams, nevals))
            for j in range(self.nparams):
                xs[:, 2 * j] = x0[:] # I don't think I need np.copy(), but not 100% sure.
                xs[j, 2 * j] = x0[j] + eps
                xs[:, 2 * j + 1] = x0[:]
                xs[j, 2 * j + 1] = x0[j] - eps
        else:
            # 1-sided differences
            nevals = self.nparams + 1
            xs = np.zeros((self.nparams, nevals))
            xs[:, 0] = x0[:]
            for j in range(self.nparams):
                xs[:, j + 1] = x0[:]
                xs[j, j + 1] = x0[j] + eps
        
        evals = np.zeros((self.nfuncs, nevals))
        # Do the hard work of evaluating the functions.
        for j in range(nevals):
            # Handle only this group's share of the work:
            if np.mod(j, mpi.ngroups) == mpi.rank_leaders:
                mpi.mobilize_workers(xs[:, j], CALCULATE_F)
                self.set(xs[:, j])
                evals[:, j] = np.array([f() for f in self.funcs])

        # Combine the results from all groups:
        evals = mpi.comm_leaders.reduce(evals, op=MPI.SUM, root=0)

        if together_at_start:
            mpi.stop_workers()
        
        # Only proc0_world will actually have the Jacobian.
        if not mpi.proc0_world:
            return None

        # Use the evals to form the Jacobian
        jac = np.zeros((self.nfuncs, self.nparams))
        if centered:
            for j in range(self.nparams):
                jac[:, j] = (evals[:, 2 * j] - evals[:, 2 * j + 1]) / (2 * eps)
        else:
            # 1-sided differences:
            for j in range(self.nparams):
                jac[:, j] = (evals[:, j + 1] - evals[:, 0]) / eps

        return jac

