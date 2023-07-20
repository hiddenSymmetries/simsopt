import numpy as np

import simsoptpp as sopp
from .._core.optimizable import Optimizable
from .._core.json import GSONDecoder
from .mgrid import MGrid

__all__ = ['MagneticField', 'MagneticFieldSum', 'MagneticFieldMultiply']


class MagneticField(sopp.MagneticField, Optimizable):

    '''
    Generic class that represents any magnetic field from which each magnetic
    field class inherits. The usage of ``MagneticField`` is as follows:

    .. code-block::

        bfield = BiotSavart(coils) # An instance of a MagneticField
        points = ... # points is a (n, 3) numpy array
        bfield.set_points(points)
        B = bfield.B() # returns the Magnetic field at `points`
        dA = bfield.dA_by_dX() # returns the gradient of the potential of the field at `points`

    ``MagneticField`` has a cache to avoid repeated calculations.
    To clear this cache manually, call the `clear_cached_properties()` function.
    The cache is automatically cleared when ``set_points`` is called or one of the dependencies
    changes.

    '''

    def set_points(self, xyz):
        return self.set_points_cart(xyz)

    def set_points_cart(self, xyz):
        if len(xyz.shape) != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz array should have shape (n, 3), but has shape {xyz.shape}")
        if not xyz.flags['C_CONTIGUOUS']:
            raise ValueError("xyz array should be C contiguous. Consider using `numpy.ascontiguousarray`.")
        return sopp.MagneticField.set_points_cart(self, xyz)

    def set_points_cyl(self, rphiz):
        if len(rphiz.shape) != 2 or rphiz.shape[1] != 3:
            raise ValueError(f"rphiz array should have shape (n, 3), but has shape {rphiz.shape}")
        if not rphiz.flags['C_CONTIGUOUS']:
            raise ValueError("rphiz array should be C contiguous. Consider using `numpy.ascontiguousarray`.")
        return sopp.MagneticField.set_points_cyl(self, rphiz)

    def __init__(self, **kwargs):
        sopp.MagneticField.__init__(self)
        Optimizable.__init__(self, **kwargs)

    def clear_cached_properties(self):
        """Clear the cache."""
        sopp.MagneticField.invalidate_cache(self)

    def recompute_bell(self, parent=None):
        if np.any(self.dofs_free_status):
            self.clear_cached_properties()

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

    def to_mgrid(self, filename, nr=10, nphi=4, nz=12, rmin=1.0, rmax=2.0, zmin=-0.5, zmax=0.5, nfp=1):
        """Export the field to the mgrid format for free boundary calculations.

        The field data is represented as a single "current group". For
        free-boundary vmec, the "extcur" array should have a single nonzero
        element, set to 1.0.

        In the future, we may want to implement multiple current groups.

        Args:
            filename: Name of the NetCDF file to save.
            nr: Number of grid points in the major radius dimension.
            nphi: Number of planes in the toroidal angle.
            nz: Number of grid points in the z coordinate.
            rmin: Minimum value of major radius for the grid.
            rmax: Maximum value of major radius for the grid.
            zmin: Minimum value of z for the grid.
            zmax: Maximum value of z for the grid.
            nfp: Number of field periods.
        """

        rs = np.linspace(rmin, rmax, nr, endpoint=True)
        phis = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        zs = np.linspace(zmin, zmax, nz, endpoint=True)

        Phi, Z, R = np.meshgrid(phis, zs, rs, indexing='ij')
        X = R * np.cos(Phi)
        Y = R * np.sin(Phi)
        Z = Z

        RPhiZ = np.zeros((R.size, 3))
        RPhiZ[:, 0] = R.flatten()
        RPhiZ[:, 1] = Phi.flatten()  
        RPhiZ[:, 2] = Z.flatten()

        # get field on the grid
        self.set_points_cyl(RPhiZ)
        B = self.B_cyl()

        # shape the components
        br, bp, bz = B.T
        br_3 = br.reshape((nphi, nz, nr))
        bp_3 = bp.reshape((nphi, nz, nr))
        bz_3 = bz.reshape((nphi, nz, nr))

        mgrid = MGrid(nfp=nfp,
                      nr=nr, nz=nz, nphi=nphi,
                      rmin=rmin, rmax=rmax, zmin=zmin, zmax=zmax)
        mgrid.add_field_cylindrical(br_3, bp_3, bz_3, name='simsopt_coils')  

        mgrid.write(filename)


class MagneticFieldMultiply(MagneticField):
    """
    Class used to multiply a magnetic field by a scalar.  It takes as input a
    MagneticField class and a scalar and multiplies B, A and their derivatives
    by that value.
    """

    def __init__(self, scalar, Bfield):
        MagneticField.__init__(self, depends_on=[Bfield])
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

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        Bfield = decoder.process_decoded(d["Bfield"], serial_objs_dict, recon_objs)
        field = cls(d["scalar"], Bfield)
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field.set_points_cart(xyz)
        return field


class MagneticFieldSum(MagneticField):
    """
    Class used to sum two or more magnetic field together.  It can either be
    called directly with a list of magnetic fields given as input and outputing
    another magnetic field with B, A and its derivatives added together or it
    can be called by summing magnetic fields classes as Bfield1 + Bfield1
    """

    def __init__(self, Bfields):
        MagneticField.__init__(self, depends_on=Bfields)
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

    def B_vjp(self, v):
        return sum([bf.B_vjp(v) for bf in self.Bfields if np.any(bf.dofs_free_status)])

    def as_dict(self, serial_objs_dict) -> dict:
        d = super().as_dict(serial_objs_dict=serial_objs_dict)
        d["points"] = self.get_points_cart()
        return d

    @classmethod
    def from_dict(cls, d, serial_objs_dict, recon_objs):
        decoder = GSONDecoder()
        Bfields = decoder.process_decoded(d["Bfields"], serial_objs_dict, recon_objs)
        field_sum = cls(Bfields)
        xyz = decoder.process_decoded(d["points"], serial_objs_dict, recon_objs)
        field_sum.set_points_cart(xyz)
        return field_sum



