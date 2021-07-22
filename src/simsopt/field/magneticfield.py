import numpy as np
import simsoptpp as sopp


class MagneticField(sopp.MagneticField):

    '''
    Generic class that represents any magnetic field from which each magnetic
    field class inherits. The usage of ``MagneticField`` is as follows:

    .. code-block::

        bfield = BiotSavart(coils, currents) # An instance of a MagneticField
        points = ... # points is a (n, 3) numpy array
        bfield.set_points(points)
        B = bfield.B() # returns the Magnetic field at `points`
        dA = bfield.dA_by_dX() # returns the gradient of the potential of the field at `points`

    ``MagneticField`` has a cache to avoid repeated calculations.
    To clear this cache, call the `clear_cached_properties()` function.
    The cache is automatically cleard when ``set_points`` is called.

    '''

    def __init__(self):
        sopp.MagneticField.__init__(self)

    def clear_cached_properties(self):
        """Clear the cache."""
        sopp.MagneticField.invalidate_cache(self)

    def __add__(self, other):
        """Add two magnetic fields."""
        return MagneticFieldSum([self, other])

    def __mul__(self, other):
        """Multiply a field with a scalar."""
        return MagneticFieldMultiply(other, self)

    def __rmul__(self, other):
        """Multiply a field with a scalar."""
        return MagneticFieldMultiply(other, self)

    def to_vtk(self, filename, nr=10, nphi=10, nz=10, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5):
        """Export the field evaluated on a regular grid for visualisation with e.g. Paraview."""
        from pyevtk.hl import gridToVTK
        rs = np.linspace(rmin, rmax, nr, endpoint=True)
        phis = np.linspace(0, 2*np.pi, nphi, endpoint=True)
        zs = np.linspace(zmin, zmax, nz, endpoint=True)

        R, Phi, Z = np.meshgrid(rs, phis, zs)
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        Z = Z

        RPhiZ = np.zeros((R.size, 3))
        RPhiZ[:, 0] = R.flatten()
        RPhiZ[:, 1] = Phi.flatten()
        RPhiZ[:, 2] = Z.flatten()

        self.set_points_cyl(RPhiZ)
        vals = self.B().reshape((R.shape[0], R.shape[1], R.shape[2], 3))
        contig = np.ascontiguousarray
        gridToVTK(filename, X, Y, Z, pointData={"B": (contig(vals[..., 0]), contig(vals[..., 1]), contig(vals[..., 2]))})


class MagneticFieldMultiply(MagneticField):
    """
    Class used to multiply a magnetic field by a scalar.  It takes as input a
    MagneticField class and a scalar and multiplies B, A and their derivatives
    by that value.
    """

    def __init__(self, scalar, Bfield):
        MagneticField.__init__(self)
        self.scalar = scalar
        self.Bfield = Bfield

    def _set_points_cb(self):
        self.Bfield.set_points_cart(self.get_points_cart_ref())

    def _B_impl(self, B):
        B[:] = self.scalar*self.Bfield.B()

    def _dB_by_dX_impl(self, dB):
        dB[:] = self.scalar*self.Bfield.dB_by_dX()

    def _d2B_by_dXdX_impl(self, ddB):
        ddB[:] = self.scalar*self.Bfield.d2B_by_dXdX()

    def _A_impl(self, A):
        A[:] = self.scalar*self.Bfield.A()

    def _dA_by_dX_impl(self, dA):
        dA[:] = self.scalar*self.Bfield.dA_by_dX()

    def _d2A_by_dXdX_impl(self, ddA):
        ddA[:] = self.scalar*self.Bfield.d2A_by_dXdX()


class MagneticFieldSum(MagneticField):
    """
    Class used to sum two or more magnetic field together.  It can either be
    called directly with a list of magnetic fields given as input and outputing
    another magnetic field with B, A and its derivatives added together or it
    can be called by summing magnetic fields classes as Bfield1 + Bfield1
    """

    def __init__(self, Bfields):
        MagneticField.__init__(self)
        self.Bfields = Bfields

    def _set_points_cb(self):
        for bf in self.Bfields:
            bf.set_points_cart(self.get_points_cart_ref())

    def _B_impl(self, B):
        B[:] = np.sum([bf.B() for bf in self.Bfields], axis=0)

    def _dB_by_dX_impl(self, dB):
        dB[:] = np.sum([bf.dB_by_dX() for bf in self.Bfields], axis=0)

    def _d2B_by_dXdX_impl(self, ddB):
        ddB[:] = np.sum([bf.d2B_by_dXdX() for bf in self.Bfields], axis=0)

    def _A_impl(self, A):
        A[:] = np.sum([bf.A() for bf in self.Bfields], axis=0)

    def _dA_by_dX_impl(self, dA):
        dA[:] = np.sum([bf.dA_by_dX() for bf in self.Bfields], axis=0)

    def _d2A_by_dXdX_impl(self, ddA):
        ddA[:] = np.sum([bf.d2A_by_dXdX() for bf in self.Bfields], axis=0)
