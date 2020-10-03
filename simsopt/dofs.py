"""
This module provides the collect_dofs() function, used by least-squares
and general optimization problems.
"""

import numpy as np
import types
import logging
from .util import unique
from .optimizable import function_from_user

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
        
        self.logger = logging.getLogger(__name__)

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

    @property
    def jac(self):
        """
        Return the Jacobian, i.e. the gradients of all the functions that were originally
        supplied to Dofs(). Result is a 2D numpy array.
        """
        if not self.grad_avail:
            raise RuntimeError('Gradient information is not available for this Dofs()')

        results = np.zeros((self.nfuncs, self.nparams))
        # Loop over the rows of the Jacobian, i.e. over the functions
        # that were originally provided to Dofs():
        for jfunc in range(self.nfuncs):
            # Get the gradient of this particular function with
            # respect to all of it's dofs, which is a different set
            # from the global dofs:
            grad = self.grad_funcs[jfunc]()
            
            # Match up the global dofs with the dofs for this particular gradient function:
            for jdof in range(self.nparams):
                for jgrad in range(len(self.func_indices[jfunc])):
                    # A global dof matches a dof for this function if the owners and indices both match:
                    if self.dof_owners[jdof] == self.func_dof_owners[jfunc][jgrad] and self.indices[jdof] == self.func_indices[jfunc][jgrad]:
                        results[jfunc, jdof] = grad[jgrad]
                        # If we find a match, we can exit the innermost loop:
                        break
                        
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

    def fd_jac(self, eps=1e-7):
        """
        Compute the finite-difference Jacobian of the functions with
        respect to all non-fixed degrees of freedom. A
        centered-difference approximation is used, with step size eps.
        """

        self.logger.info('Beginning finite difference gradient calculation for functions ' + str(self.funcs))

        x0 = self.x
        self.logger.info('  nparams: {}, nfuncs: {}'.format(self.nparams, self.nfuncs))
        self.logger.info('  x0: ' + str(x0))

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

