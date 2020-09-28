"""
This module provides the collect_dofs() function, used by least-squares
and general optimization problems.
"""

import numpy as np
from .util import unique, Struct

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
            if j in owners_so_far:
                raise RuntimeError('Circular dependency detected among the objects')
            owners += get_owners(j, owners_so_far=owners)
    return owners
    

def collect_dofs(funcs):
    """Given a list of optimizable functions, 

    funcs: A list/set/tuple of callable functions.

    returns: an object with the following attributes:
    x: A 1D numpy vector of variable dofs.
    
    owners: A vector, with each element pointing to the object whose
    set_dofs() function should be called to update the corresponding
    dof.

    indices: A vector of ints, with each element giving the index in
    the owner's set_dofs method corresponding to this dof.

    names: A list of strings to identify each of the dofs.
    """

    # First, get a list of the objects and any objects they depend on:
    owner_list = []
    for j in funcs:
        owner_list += get_owners(j.__self__)

    # Eliminate duplicates, preserving order:
    owner_list = unique(owner_list)

    # Go through the objects, looking for any non-fixed dofs:
    x = []
    owners = []
    indices = []
    mins = []
    maxs = []
    names = []
    for owner in owner_list:
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
            onames = owner.names
        else:
            onames = ['x[{}] of {}'.format(k, owner) for k in range(ndofs)]

        for jdof in range(ndofs):
            if not fixed[jdof]:
                x.append(ox[jdof])
                owners.append(owner)
                indices.append(jdof)
                names.append(onames[jdof])
                mins.append(omins[jdof])
                maxs.append(omaxs[jdof])

    # Finally, package results for return
    results = Struct()
    
    results.x = np.array(x)
    results.owners = owners
    results.indices = np.array(indices)
    results.names = names
    results.mins = np.array(mins)
    results.maxs = np.array(maxs)
    
    return results
