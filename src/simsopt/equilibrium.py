"""
This module provides an abstract class for numerically representing a
magnetohydrodynamic (MHD) equilibrium.
"""

# To handle free-boundary vs fixed-boundary, we could have two
# subclasses of Equilibrium, or we could have a bool attribute.

from simsopt import Parameter, Surface

class Equilibrium:
    """
    This abstract class is for numerically representing a
    magnetohydrodynamic (MHD) equilibrium.
    """
    def __init__(self):
        """
        Constructor
        """
        self.nfp = Parameter(1, min=1)
        self.stelsym = Parameter(True)
        self.boundary = Surface(nfp=self.nfp.val, stelsym=self.stelsym.val)

    def __repr__(self):
        """
        Print the object in an informative way.
        """
        return "Equilibrium instance (nfp=" + str(self.nfp.val) + \
            " stelsym=" + str(self.stelsym.val) + ")"
