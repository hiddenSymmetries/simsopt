# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides the Dofs class.

This module should not depend on anything involving communication
(e.g. MPI) or on specific types of optimization problems.
"""

import logging
from typing import Union

import numpy as np

from .optimizable import function_from_user
from .util import unique, ObjectiveFailure, finite_difference_steps
from ..util.dev import deprecated

logger = logging.getLogger(__name__)


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


@deprecated(message="This class has been deprecated from v0.6.0 and will be "
                    "deleted from future versions of simsopt. Use graph "
                    "framework to define the optimization problem. To use graph"
                    "framework use simsopt._core.graph_optimizable.Optimizable "
                    "class.",
            category=DeprecationWarning)
class Dofs:
    """
    This class holds data related to the vector of degrees of freedom
    that have been combined from multiple optimizable objects, keeping
    only the non-fixed dofs.

    For the meaning of ``abs_step`` and ``rel_step``, see
    :func:`simsopt._core.util.finite_difference_steps()`.

    Args:
        funcs: A list, tuple, or set of optimizable functions.
        fail: Should be None, a large positive float, or NaN. If not
          None, any ObjectiveFailure excpetions raised will be caught
          and the corresponding residual values will be replaced by this
          value.
        abs_step: Absolute step size for finite differences.
        rel_step: Relative step size for finite differences.
        diff_method: Method to use if and when finite differences are
          evaluated. Should be ``forward`` or ``centered``.

    Attributes:
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

    def __init__(self,
                 funcs,
                 fail: Union[None, float] = 1.0e12,
                 abs_step: float = 1.0e-7,
                 rel_step: float = 0.0,
                 diff_method: str = "centered"
                 ):

        self.fail = fail
        self.abs_step = abs_step
        self.rel_step = rel_step
        self.diff_method = diff_method

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
                    # if not fixed[jdof]:
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
        self.nvals = None  # We won't know this until the first function eval.
        self.nvals_per_func = np.full(self.nfuncs, 0)
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

        # Autodetect whether the functions return scalars or vectors.
        # For now let's do this on every function eval for
        # simplicity. Maybe there is some speed advantage to only
        # doing it the first time (if self.nvals is None.)
        val_list = []
        failed = False
        for j, func in enumerate(self.funcs):
            try:
                f = func()
            except ObjectiveFailure:
                logger.info("Function evaluation failed")
                failed = True
                if self.fail is None:
                    raise
                # As soon as any functions fail, don't bother
                # evaluating the rest:
                break

            if isinstance(f, (np.ndarray, list, tuple)):
                self.nvals_per_func[j] = len(f)
                val_list.append(np.array(f))
            else:
                self.nvals_per_func[j] = 1
                val_list.append(np.array([f]))

        if failed:
            if self.nvals is None:
                # This case occurs if there is a failure on the first
                # function evaluation, so we do not yet know how many
                # residuals to return.
                raise RuntimeError("Objective failed on first function evaluation")

            return np.full(self.nvals, self.fail)

        else:
            logger.debug('Detected nvals_per_func={}'.format(self.nvals_per_func))
            self.nvals = np.sum(self.nvals_per_func)
            return np.concatenate(val_list)

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

        # grads = [np.array(f()) for f in self.grad_funcs]

        start_indices = np.full(self.nfuncs, 0)
        end_indices = np.full(self.nfuncs, 0)

        # First, evaluate all the gradient functions, and autodetect
        # how many rows there are in the gradient for each function.
        grads = []
        for j in range(self.nfuncs):
            grad = np.array(self.grad_funcs[j]())
            # Above, we cast to a np.array to be a bit forgiving in
            # case the user provides something other than a plain 1D
            # or 2D numpy array. Previously I also had flatten() for
            # working with Florian's simsgeo function; this may not
            # work now without the flatten.

            # Make sure grad is a 2D array (like the Jacobian)
            if grad.ndim == 1:
                # In this case, I should perhaps handle the rare case
                # of a function from R^1 -> R^n with n > 1.
                grad2D = grad.reshape((1, len(grad)))
            elif grad.ndim == 2:
                grad2D = grad
            else:
                raise ValueError('gradient should be 1D or 2D')

            grads.append(grad2D)
            this_nvals = grad2D.shape[0]
            if self.nvals_per_func[j] > 0:
                assert self.nvals_per_func[j] == this_nvals, \
                    "Number of rows in gradient is not consistent with number of entries in the function"
            else:
                self.nvals_per_func[j] = this_nvals

            if j > 0:
                start_indices[j] = end_indices[j - 1]
            end_indices[j] = start_indices[j] + this_nvals

        self.nvals = np.sum(self.nvals_per_func)

        results = np.zeros((self.nvals, self.nparams))
        # Loop over the rows of the Jacobian, i.e. over the functions
        # that were originally provided to Dofs():
        for jfunc in range(self.nfuncs):
            start_index = start_indices[jfunc]
            end_index = end_indices[jfunc]
            grad = grads[jfunc]

            # Match up the global dofs with the dofs for this particular gradient function:
            for jdof in range(self.nparams):
                for jgrad in range(len(self.func_indices[jfunc])):
                    # A global dof matches a dof for this function if the owners and indices both match:
                    if self.dof_owners[jdof] == self.func_dof_owners[jfunc][jgrad] and self.indices[jdof] == \
                            self.func_indices[jfunc][jgrad]:
                        results[start_index:end_index, jdof] = grad[:, jgrad]
                        # If we find a match, we can exit the innermost loop:
                        break

        # print('finite-difference Jacobian:')
        # fd_jac = self.fd_jac()
        # print(fd_jac)
        # print('analytic Jacobian:')
        # print(results)
        # print('difference:')
        # print(fd_jac - results)
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

    def fd_jac(self,
               x: np.ndarray = None
               ) -> np.ndarray:
        """
        Compute the finite-difference Jacobian of the functions with
        respect to all non-fixed degrees of freedom. Either a 1-sided
        or centered-difference approximation can be used.

        The attributes ``abs_step``, ``rel_step``, and ``diff_method`` of
        the ``Dofs`` object will be used in this calculation. For the
        meaning of ``abs_step`` and ``rel_step``, see
        :func:`simsopt._core.util.finite_difference_steps()`.

        If the argument x is not supplied, the Jacobian will be
        evaluated for the present state vector. If x is supplied, then
        first get_dofs() will be called for each object to set the
        global state vector to x.

        No parallelization is used here.

        Args:
            x: The state vector at which you wish to compute the Jacobian.
        """

        if x is not None:
            self.set(x)

        logger.info('Beginning finite difference gradient calculation for functions ' + str(self.funcs))

        x0 = self.x
        steps = finite_difference_steps(x0, abs_step=self.abs_step, rel_step=self.rel_step)
        logger.info('  nparams: {}, nfuncs: {}, nvals: {}'.format(self.nparams, self.nfuncs, self.nvals))
        logger.info('  x0: ' + str(x0))

        # Handle the rare case in which nparams==0, so the Jacobian
        # has size (nvals, 0):
        if self.nparams == 0:
            if self.nvals is None:
                # We don't know nvals yet. In this case, we could
                # either do a function eval to determine it, or
                # else return a 2d array of size (1,0), which is
                # probably the wrong size. For safety, let's do a
                # function eval to determine nvals.
                self.f()
            jac = np.zeros((self.nvals, self.nparams))
            return jac

        if self.diff_method == "centered":
            # Centered differences:
            jac = None
            for j in range(self.nparams):
                x = np.copy(x0)

                x[j] = x0[j] + steps[j]
                self.set(x)
                # fplus = np.array([f() for f in self.funcs])
                fplus = self.f()
                if jac is None:
                    # After the first function evaluation, we now know
                    # the size of the Jacobian.
                    jac = np.zeros((self.nvals, self.nparams))

                x[j] = x0[j] - steps[j]
                self.set(x)
                fminus = self.f()

                jac[:, j] = (fplus - fminus) / (2 * steps[j])

        elif self.diff_method == "forward":
            # 1-sided differences
            f0 = self.f()
            jac = np.zeros((self.nvals, self.nparams))
            for j in range(self.nparams):
                x = np.copy(x0)
                x[j] = x0[j] + steps[j]
                self.set(x)
                fplus = self.f()

                jac[:, j] = (fplus - f0) / steps[j]

        else:
            raise ValueError(f"Finite difference method {self.diff_method} " \
                             "not implemented. Available methods are " \
                             "'centered' or 'forward'")

        # Weird things may happen if we do not reset the state vector
        # to x0:
        self.set(x0)
        return jac
