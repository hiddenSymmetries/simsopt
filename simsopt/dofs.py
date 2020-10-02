"""
This module provides the collect_dofs() function, used by least-squares
and general optimization problems.
"""

import numpy as np
from .util import unique
import types

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
        for owner in all_owners:
            ox = owner.get_dofs()
            ndofs = len(ox)
            # If 'fixed' is not present, assume all dofs are not fixed
            if hasattr(owner, 'fixed'):
                fixed = owner.fixed
            else:
                fixed = np.full(ndofs, False)

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

        self.nparams = len(x)
        self.dof_owners = dof_owners
        self.indices = np.array(indices)
        self.names = names
        self.mins = np.array(mins)
        self.maxs = np.array(maxs)
        self.all_owners = all_owners

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
    
    def set(self, x):
        """
        Call set_dofs() for each object, given a state vector x.
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

