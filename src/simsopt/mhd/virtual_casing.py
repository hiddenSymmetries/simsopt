# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides routines for interacting with the
``virtual_casing`` package by D Malhotra et al, for e.g. computing the
magnetic field on a surface due to currents inside the surface.

For details of the algorithm, see
D Malhotra, A J Cerfon, M O'Neil, and E Toler,
"Efficient high-order singular quadrature schemes in magnetic fusion",
Plasma Physics and Controlled Fusion 62, 024004 (2020).
"""

import os
import logging
from datetime import datetime
import numpy as np
from scipy.io import netcdf_file
from scipy.signal import resample
from .vmec_diagnostics import B_cartesian
from .vmec import Vmec
from ..geo.surfacerzfourier import SurfaceRZFourier
from ..geo.surface import best_nphi_over_ntheta

logger = logging.getLogger(__name__)


def resample_2D(arr, d0, d1):
    """
    Given a 2D array, use Fourier interpolation to resample the data in both dimensions.

    Args:
        arr: An input 2D array.
        d0: The desired dimension along axis 0 of the returned array.
        d1: The desired dimension along axis 1 of the returned array.

    Returns:
        A 2D array of size ``(d0, d1)`` with the resampled data.
    """
    return resample(resample(arr, d0), d1, axis=1)


class VirtualCasing:
    r"""
    Use the virtual casing principle to compute the contribution to
    the total magnetic field due to current inside a bounded surface.

    Usually, an instance of this class is created using the
    :func:`from_vmec()` class method, which also drives the
    computationally demanding part of the calculation (solving the
    integral equation).  In the future, interfaces to other
    equilibrium codes may be added.

    In the standard 2-stage approach to stellarator optimization, the
    virtual casing calculation is run once, at the end of stage 1
    (optimizing the plasma shape), with the result provided as input
    to stage 2 (optimizing the coil shapes). In this case, you can use
    the :func:`save()` function or the ``filename`` argument of
    :func:`from_vmec()` to save the results of the virtual casing
    calculation.  These saved results can then be loaded in later
    using the :func:`load()` class method, when needed for solving the
    stage-2 problem.

    A common situation is that you may wish to use a different number
    of grid points in :math:`\theta` and :math:`\phi` on the surface
    for the stage-2 optimization compared to the number of grid points
    used for the virtual casing calculation. In this situation, the
    :func:`resample()` function is provided to interpolate the virtual
    casing results onto whatever grid you wish to use for the stage-2
    problem.

    To set the grid resolutions ``nphi`` and ``ntheta``, it can be
    convenient to use the function
    :func:`simsopt.geo.surface.best_nphi_over_ntheta`.

    An instance of this class has the following attributes. For all
    vector quantites, Cartesian coordinates are used, corresponding to
    array dimensions of size 3:

    - ``nphi``: The number of grid points in the toroidal angle :math:`\phi`, for the full torus.
    - ``ntheta``: The number of grid points in the poloidal angle :math:`\theta`.
    - ``phi``: An array of size ``(nphi,)`` with the grid points of :math:`\phi`.
    - ``theta``: An array of size ``(ntheta,)`` with the grid points of :math:`\theta`.
    - ``gamma``: An array of size ``(nphi, ntheta, 3)`` with the position vector on the surface.
    - ``unit_normal``: An array of size ``(nphi, ntheta, 3)`` with the unit normal vector on the surface.
    - ``B_total``: An array of size ``(nphi, ntheta, 3)`` with the total magnetic field vector on the surface.
    - ``B_internal``: An array of size ``(nphi, ntheta, 3)`` with the contribution
      to the magnetic field due to current inside the surface.
    - ``B_internal_normal``: An array of size ``(nphi, ntheta)`` with the contribution
      to the magnetic field due to current inside the surface, taking just the component
      normal to the surface.
    """

    @classmethod
    def from_vmec(cls, vmec, nphi, ntheta=None, digits=6, filename="auto"):
        """
        Given a :obj:`~simsopt.mhd.vmec.Vmec` object, compute the
        contribution to the total magnetic field due to currents inside
        the plasma.

        This function requires the python ``virtual_casing`` package to be
        installed.

        The argument ``nphi`` refers to the number of points around the
        full torus. It must be a multiple of ``2 * nfp``, so there is an
        integer number of points per half field period.

        To set the grid resolutions ``nphi`` and ``ntheta``, it can be
        convenient to use the function
        :func:`simsopt.geo.surface.best_nphi_over_ntheta`. This is
        done automatically if you omit the ``ntheta`` argument.

        For now, this routine only works for stellarator symmetry.

        Args:
            vmec: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`, or the name of a
              Vmec ``input.*`` or ``wout*`` file.
            nphi: Number of grid points toroidally for the calculation.
            ntheta: Number of grid points poloidally for the calculation. If ``None``,
              the number of grid points will be calculated automatically using
              :func:`simsopt.geo.surface.best_nphi_over_ntheta()` to minimize
              the grid anisotropy, given the specified ``nphi``.
            digits: Approximate number of digits of precision for the calculation.
            filename: If not ``None``, the results of the virtual casing calculation
              will be saved in this file. For the default value of ``"auto"``, the
              filename will automatically be set to ``"vcasing_<extension>.nc"``
              where ``<extension>`` is the string associated with Vmec input and output
              files, analogous to the Vmec output file ``"wout_<extension>.nc"``.
        """
        import virtual_casing as vc_module

        if not isinstance(vmec, Vmec):
            vmec = Vmec(vmec)

        vmec.run()
        nfp = vmec.wout.nfp
        if vmec.wout.lasym:
            raise RuntimeError('virtual casing presently only works for stellarator symmetry')
        if nphi % (2 * nfp) != 0:
            raise ValueError(f'nphi must be a multiple of 2 * nfp. nphi={nphi}, nfp={nfp}')

        if ntheta is None:
            ntheta = int(nphi / best_nphi_over_ntheta(vmec.boundary))
            logger.debug(f'new ntheta: {ntheta}')

        # The requested nphi and ntheta may not match the quadrature
        # points in vmec.boundary, and the range may not be "full torus",
        # so generate a SurfaceRZFourier with the desired resolution:
        surf = SurfaceRZFourier(mpol=vmec.wout.mpol, ntor=vmec.wout.ntor, nfp=vmec.wout.nfp,
                                nphi=nphi, ntheta=ntheta, range="full torus")
        for jmn in range(vmec.wout.mnmax):
            surf.set_rc(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.rmnc[jmn, -1])
            surf.set_zs(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.zmns[jmn, -1])

        gamma = surf.gamma()
        unit_normal = surf.unitnormal()
        Bxyz = B_cartesian(vmec, nphi=nphi, ntheta=ntheta, range="full torus")
        logger.debug(f'gamma.shape: {gamma.shape}')
        logger.debug(f'unit_normal.shape: {unit_normal.shape}')
        logger.debug(f'Bxyz[0].shape: {Bxyz[0].shape}')

        # virtual_casing wants all input arrays to be 1D. The order is
        # {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}
        # where Nt is toroidal (not theta!) and Np is poloidal (not phi!)
        gamma1d = np.zeros(nphi * ntheta * 3)
        B1d = np.zeros(nphi * ntheta * 3)
        B3d = np.zeros((nphi, ntheta, 3))
        for jxyz in range(3):
            gamma1d[jxyz * nphi * ntheta: (jxyz + 1) * nphi * ntheta] = gamma[:, :, jxyz].flatten(order='C')
            B1d[jxyz * nphi * ntheta: (jxyz + 1) * nphi * ntheta] = Bxyz[jxyz].flatten(order='C')
            B3d[:, :, jxyz] = Bxyz[jxyz]

        """
        # Check order:
        index = 0
        for jxyz in range(3):
            for jphi in range(nphi):
                for jtheta in range(ntheta):
                    np.testing.assert_allclose(gamma1d[index], gamma[jphi, jtheta, jxyz])
                    np.testing.assert_allclose(B1d[index], Bxyz[jxyz][jphi, jtheta])
                    index += 1
        """

        vcasing = vc_module.VirtualCasing()
        vcasing.set_surface(nphi, ntheta, gamma1d)
        vcasing.set_accuracy(digits)
        # This next line launches the main computation:
        Bexternal = np.array(vcasing.compute_external_B(B1d))

        # Unpack 1D array results:
        Binternal1d = B1d - Bexternal
        Binternal3d = np.zeros((nphi, ntheta, 3))
        for jxyz in range(3):
            Binternal3d[:, :, jxyz] = Binternal1d[jxyz * nphi * ntheta: (jxyz + 1) * nphi * ntheta].reshape((nphi, ntheta), order='C')

        """
        # Check order:
        index = 0
        for jxyz in range(3):
            for jphi in range(nphi):
                for jtheta in range(ntheta):
                    np.testing.assert_allclose(Binternal1d[index], Binternal3d[jphi, jtheta, jxyz])
                    index += 1
        """

        Btotal_normal = np.sum(B3d * unit_normal, axis=2)
        Binternal_normal = np.sum(Binternal3d * unit_normal, axis=2)

        vc = cls()
        vc.ntheta = ntheta
        vc.nphi = nphi
        vc.theta = surf.quadpoints_theta
        vc.phi = surf.quadpoints_phi
        vc.B_total = B3d
        vc.gamma = gamma
        vc.unit_normal = unit_normal
        vc.B_internal = Binternal3d
        vc.B_internal_normal = Binternal_normal

        if filename is not None:
            if filename == 'auto':
                directory, basefile = os.path.split(vmec.output_file)
                filename = os.path.join(directory, 'vcasing' + basefile[4:])
                logger.debug(f'New filename: {filename}')
            vc.save(filename)

        return vc

    def save(self, filename="vcasing.nc"):
        """
        Save the results of a virtual casing calculation in a NetCDF file.

        Args:
            filename: Name of the file to create.
        """
        f = netcdf_file(filename, 'w')
        f.history = 'This file created by simsopt on ' + datetime.now().strftime("%B %d %Y, %H:%M:%S")
        f.createDimension('ntheta', self.ntheta)
        f.createDimension('nphi', self.nphi)
        f.createDimension('xyz', 3)

        ntheta = f.createVariable('ntheta', 'i', tuple())
        ntheta.assignValue(self.ntheta)
        ntheta.description = 'Number of grid points in the poloidal angle theta'
        ntheta.units = 'Dimensionless'

        nphi = f.createVariable('nphi', 'i', tuple())
        nphi.assignValue(self.nphi)
        nphi.description = 'Number of grid points in the toroidal angle phi, covering the full torus'
        nphi.units = 'Dimensionless'

        theta = f.createVariable('theta', 'd', ('ntheta',))
        theta[:] = self.theta
        theta.description = 'Grid points in the poloidal angle theta. Note that theta extends over [0, 1) not [0, 2pi).'
        theta.units = 'Dimensionless'

        phi = f.createVariable('phi', 'd', ('nphi',))
        phi[:] = self.phi
        phi.description = 'Grid points in the toroidal angle phi. Note that phi extends over [0, 1) not [0, 2pi).'
        phi.units = 'Dimensionless'

        gamma = f.createVariable('gamma', 'd', ('nphi', 'ntheta', 'xyz'))
        gamma[:, :, :] = self.gamma
        gamma.description = 'Position vector on the boundary surface'
        gamma.units = 'meter'

        unit_normal = f.createVariable('unit_normal', 'd', ('nphi', 'ntheta', 'xyz'))
        unit_normal[:, :, :] = self.unit_normal
        unit_normal.description = 'Unit-length normal vector on the boundary surface'
        unit_normal.units = 'Dimensionless'

        B_total = f.createVariable('B_total', 'd', ('nphi', 'ntheta', 'xyz'))
        B_total[:, :, :] = self.B_total
        B_total.description = 'Total magnetic field vector on the surface, including currents both inside and outside of the surface'
        B_total.units = 'Tesla'

        B_internal = f.createVariable('B_internal', 'd', ('nphi', 'ntheta', 'xyz'))
        B_internal[:, :, :] = self.B_internal
        B_internal.description = 'Contribution to the magnetic field vector on the surface due only to currents inside the surface'
        B_internal.units = 'Tesla'

        B_internal_normal = f.createVariable('B_internal_normal', 'd', ('nphi', 'ntheta'))
        B_internal_normal[:, :] = self.B_internal_normal
        B_internal_normal.description = 'Component of B_internal normal to the surface'
        B_internal_normal.units = 'Tesla'

        f.close()

    @classmethod
    def load(cls, filename):
        """
        Load in the results of a previous virtual casing calculation,
        previously saved in NetCDF format.

        Args:
            filename: Name of the file to load.
        """
        vc = cls()
        f = netcdf_file(filename, mmap=False)
        for key, val in f.variables.items():
            val2 = val[()]  # Convert to numpy array
            vc.__setattr__(key, val2)
        return vc

    def resample(self, ntheta, nphi):
        """
        Return a new ``VirtualCasing`` object in which all of the data
        have been resampled in the poloidal and toroidal angles.
        This is useful if you wish to use a different surface resolution
        for the stage-2 coil optimization compared to the resolution
        used for the virtual casing calculation.

        Args:
            ntheta: New number of grid points in the poloidal angle.
            nphi: New number of grid points in the toroidal angle (including all field periods).

        Returns:
            A new ``VirtualCasing`` object.
        """
        newvc = VirtualCasing()
        newvc.ntheta = ntheta
        newvc.nphi = nphi
        newvc.theta = np.linspace(0, 1, ntheta, endpoint=False)
        newvc.phi = np.linspace(0, 1, nphi, endpoint=False)
        newvc.B_internal_normal = resample_2D(self.B_internal_normal, nphi, ntheta)
        # Vector fields on the surface:
        variables = ['gamma', 'unit_normal', 'B_total', 'B_internal']
        for variable in variables:
            oldvar = eval('self.' + variable)
            newvar = np.zeros((nphi, ntheta, 3))
            for j in range(3):
                newvar[:, :, j] = resample_2D(oldvar[:, :, j], nphi, ntheta)
            newvc.__setattr__(variable, newvar)
        return newvc

    def plot(self, show=True):
        """
        Plot ``B_internal_normal``, the component normal to the surface of
        the magnetic field generated by currents inside the surface.
        This routine requires ``matplotlib``.

        Args:
            show: Whether to call matplotlib's ``show()`` function.
        """
        import matplotlib.pyplot as plt
        plt.contourf(self.phi, self.theta, self.B_internal_normal.T, 25)
        plt.xlabel(r'$\phi$')
        plt.ylabel(r'$\theta$')
        plt.title('B_internal_normal [Tesla]')
        plt.colorbar()
        plt.tight_layout()
        if show:
            plt.show()
