# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides routines for interacting with the
``virtual_casing`` package by D Malhotra et al, for e.g. computing the
magnetic field on a surface due to currents outside the surface.

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

from .vmec_diagnostics import B_cartesian
from .vmec import Vmec
from ..geo.surfacerzfourier import SurfaceRZFourier
from ..geo.surface import best_nphi_over_ntheta

logger = logging.getLogger(__name__)

__all__ = ['VirtualCasing']


class VirtualCasing:
    r"""
    Use the virtual casing principle to compute the contribution to
    the total magnetic field due to current outside a bounded surface.

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

    In order to compute the external field accurately, one requires
    fairly high grid resolution. For the stage-2 problem however, a
    lower resolution is often sufficient. To deal with this, we
    consider two grids, one denoted by the prefix ``src_`` and the
    other one denoted by ``trgt_``. ``src_nphi`` and ``src_ntheta``
    refer to the resolution of the grid for the input data, i.e. the
    total magnetic field and shape of the surface. ``trgt_nphi`` and
    ``trgt_ntheta`` refer to the resolution of the grid where the
    external field is computed, i.e. the output of the virtual casing
    calculation that is provided as input to the stage 2 problem).

    To set the grid resolutions ``src_nphi`` and ``src_ntheta``, it can be
    convenient to use the function
    :func:`simsopt.geo.surface.best_nphi_over_ntheta`.

    An instance of this class has the following attributes. For all
    vector quantites, Cartesian coordinates are used, corresponding to
    array dimensions of size 3:

    - ``src_nphi``: The number of grid points in the toroidal angle :math:`\phi`, for a half field period (or a full field period if use_stellsym=False)
    - ``src_ntheta``: The number of grid points in the poloidal angle :math:`\theta`.
    - ``src_phi``: An array of size ``(src_nphi,)`` with the grid points of :math:`\phi`.
    - ``src_theta``: An array of size ``(src_ntheta,)`` with the grid points of :math:`\theta`.
    - ``trgt_nphi``: The number of grid points in the toroidal angle :math:`\phi`, for a half field period (or a full field period if use_stellsym=False)
    - ``trgt_ntheta``: The number of grid points in the poloidal angle :math:`\theta`.
    - ``trgt_phi``: An array of size ``(trgt_nphi,)`` with the grid points of :math:`\phi`.
    - ``trgt_theta``: An array of size ``(trgt_ntheta,)`` with the grid points of :math:`\theta`.

    - ``gamma``: An array of size ``(src_nphi, src_ntheta, 3)`` with the position vector on the surface.
    - ``B_total``: An array of size ``(src_nphi, src_ntheta, 3)`` with the total magnetic field vector on the surface.
    - ``unit_normal``: An array of size ``(trgt_nphi, trgt_ntheta, 3)`` with the unit normal vector on the surface.
    - ``B_external``: An array of size ``(trgt_nphi, trgt_ntheta, 3)`` with the contribution
      to the magnetic field due to current outside the surface.
    - ``B_external_normal``: An array of size ``(trgt_nphi, trgt_ntheta)`` with the contribution
      to the magnetic field due to current outside the surface, taking just the component
      normal to the surface.

    The :math:`\phi` and :math:`\theta` grids for these data are both
    uniformly spaced, and are the same as for
    :obj:`~simsopt.geo.surface.Surface` classes with ``range="half
    period"`` or ``range="full period"``, for the case of
    stellarator-symmetry or non-stellarator-symmetry respectively.
    (See the description of the ``range`` parameter in the
    documentation on :ref:`surfaces`.)  For the usual case of
    stellarator symmetry, all the virtual casing data are given on
    half a field period. There is no grid point at :math:`\phi=0`,
    rather the grid is shifted in :math:`\phi` by half the grid
    spacing. Thus, the ``src_phi`` grid is ``np.linspace(1 / (2 * nfp
    * src_nphi), (src_nphi - 0.5) / (src_nphi * nfp), src_nphi)``
    (recalling the simsopt convention that :math:`\phi` and :math:`\theta` have period 1,
    not :math:`2\pi`). For a non-stellarator-symmetric calculation,
    the ``src_phi`` grid is ``np.linspace(0, 1 / nfp, src_nphi,
    endpoint=False)``.  The ``trgt_phi`` grid follows the same logic as
    the ``src_phi`` grid.  Note that for stellarator symmetry, if
    ``src_nphi != trgt_nphi``, then the shift (i.e. first grid point)
    in ``src_phi`` and ``trgt_phi`` will be different. For both
    stellarator symmetry and non-stellarator-symmetry, the
    ``src_theta`` grid is ``np.linspace(0, 1, src_ntheta,
    endpoint=False)``, and the ``trgt_theta`` grid is the same but with
    ``trgt_ntheta``.

    In particular, ``B_external_normal`` is given on the grid that
    would be naturally used for stage-2 coil optimization, so no
    resampling is required.
    """

    @classmethod
    def from_vmec(cls, vmec, src_nphi, src_ntheta=None, trgt_nphi=None, trgt_ntheta=None, use_stellsym=True, digits=6, filename="auto"):
        """
        Given a :obj:`~simsopt.mhd.vmec.Vmec` object, compute the contribution
        to the total magnetic field due to currents outside the plasma.

        This function requires the python ``virtual_casing`` package to be
        installed.

        The argument ``src_nphi`` refers to the number of points around a half
        field period if stellarator symmetry is exploited, or a full field
        period if not.

        To set the grid resolutions ``src_nphi`` and ``src_ntheta``, it can be
        convenient to use the function
        :func:`simsopt.geo.surface.best_nphi_over_ntheta`. This is
        done automatically if you omit the ``src_ntheta`` argument.

        For now, this routine only works for stellarator symmetry.

        Args:
            vmec: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`, or the name of a
              Vmec ``input.*`` or ``wout*`` file.
            src_nphi: Number of grid points toroidally for the input of the calculation.
            src_ntheta: Number of grid points poloidally for the input of the calculation. If ``None``,
              the number of grid points will be calculated automatically using
              :func:`simsopt.geo.surface.best_nphi_over_ntheta()` to minimize
              the grid anisotropy, given the specified ``nphi``.
            trgt_nphi: Number of grid points toroidally for the output of the calculation.
              If unspecified, ``src_nphi`` will be used.
            trgt_ntheta: Number of grid points poloidally for the output of the calculation.
              If unspecified, ``src_ntheta`` will be used.
            use_stellsym: whether to exploit stellarator symmetry in the calculation.
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
        stellsym = (not bool(vmec.wout.lasym)) and use_stellsym
        if vmec.wout.lasym:
            raise RuntimeError('virtual casing presently only works for stellarator symmetry')

        if src_ntheta is None:
            src_ntheta = int((1+int(stellsym)) * nfp * src_nphi / best_nphi_over_ntheta(vmec.boundary))
            logger.info(f'new src_ntheta: {src_ntheta}')

        # The requested nphi and ntheta may not match the quadrature
        # points in vmec.boundary, and the range may not be "full torus",
        # so generate a SurfaceRZFourier with the desired resolution:
        if stellsym:
            ran = "half period"
        else:
            ran = "field period"
        surf = SurfaceRZFourier.from_nphi_ntheta(mpol=vmec.wout.mpol, ntor=vmec.wout.ntor, nfp=nfp,
                                                 nphi=src_nphi, ntheta=src_ntheta, range=ran)
        for jmn in range(vmec.wout.mnmax):
            surf.set_rc(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.rmnc[jmn, -1])
            surf.set_zs(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.zmns[jmn, -1])
        Bxyz = B_cartesian(vmec, nphi=src_nphi, ntheta=src_ntheta, range=ran)
        gamma = surf.gamma()
        logger.debug(f'gamma.shape: {gamma.shape}')
        logger.debug(f'Bxyz[0].shape: {Bxyz[0].shape}')

        if trgt_nphi is None:
            trgt_nphi = src_nphi
        if trgt_ntheta is None:
            trgt_ntheta = src_ntheta
        trgt_surf = SurfaceRZFourier.from_nphi_ntheta(mpol=vmec.wout.mpol, ntor=vmec.wout.ntor, nfp=nfp,
                                                      nphi=trgt_nphi, ntheta=trgt_ntheta, range=ran)
        trgt_surf.x = surf.x

        unit_normal = trgt_surf.unitnormal()
        logger.debug(f'unit_normal.shape: {unit_normal.shape}')

        # virtual_casing wants all input arrays to be 1D. The order is
        # {x11, x12, ..., x1Np, x21, x22, ... , xNtNp, y11, ... , z11, ...}
        # where Nt is toroidal (not theta!) and Np is poloidal (not phi!)
        gamma1d = np.zeros(src_nphi * src_ntheta * 3)
        B1d = np.zeros(src_nphi * src_ntheta * 3)
        B3d = np.zeros((src_nphi, src_ntheta, 3))
        for jxyz in range(3):
            gamma1d[jxyz * src_nphi * src_ntheta: (jxyz + 1) * src_nphi * src_ntheta] = gamma[:, :, jxyz].flatten(order='C')
            B1d[jxyz * src_nphi * src_ntheta: (jxyz + 1) * src_nphi * src_ntheta] = Bxyz[jxyz].flatten(order='C')
            B3d[:, :, jxyz] = Bxyz[jxyz]

        """
        # Check order:
        index = 0
        for jxyz in range(3):
            for jphi in range(src_nphi):
                for jtheta in range(src_ntheta):
                    np.testing.assert_allclose(gamma1d[index], gamma[jphi, jtheta, jxyz])
                    np.testing.assert_allclose(B1d[index], Bxyz[jxyz][jphi, jtheta])
                    index += 1
        """

        vcasing = vc_module.VirtualCasing()
        vcasing.setup(
            digits, nfp, stellsym,
            src_nphi, src_ntheta, gamma1d,
            src_nphi, src_ntheta,
            trgt_nphi, trgt_ntheta)
        # This next line launches the main computation:
        Bexternal1d = np.array(vcasing.compute_external_B(B1d))

        # Unpack 1D array results:
        Bexternal3d = np.zeros((trgt_nphi, trgt_ntheta, 3))
        for jxyz in range(3):
            Bexternal3d[:, :, jxyz] = Bexternal1d[jxyz * trgt_nphi * trgt_ntheta: (jxyz + 1) * trgt_nphi * trgt_ntheta].reshape((trgt_nphi, trgt_ntheta), order='C')

        """
        # Check order:
        index = 0
        for jxyz in range(3):
            for jphi in range(trgt_nphi):
                for jtheta in range(trgt_ntheta):
                    np.testing.assert_allclose(Bexternal1d[index], Bexternal3d[jphi, jtheta, jxyz])
                    index += 1
        """

        Bexternal_normal = np.sum(Bexternal3d * unit_normal, axis=2)

        vc = cls()
        vc.src_ntheta = src_ntheta
        vc.src_nphi = src_nphi
        vc.src_theta = surf.quadpoints_theta
        vc.src_phi = surf.quadpoints_phi

        vc.trgt_ntheta = trgt_ntheta
        vc.trgt_nphi = trgt_nphi
        vc.trgt_theta = trgt_surf.quadpoints_theta
        vc.trgt_phi = trgt_surf.quadpoints_phi

        vc.nfp = nfp
        vc.B_total = B3d
        vc.gamma = gamma
        vc.unit_normal = unit_normal
        vc.B_external = Bexternal3d
        vc.B_external_normal = Bexternal_normal

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
        with netcdf_file(filename, 'w') as f:
            f.history = 'This file created by simsopt on ' + datetime.now().strftime("%B %d %Y, %H:%M:%S")
            f.createDimension('src_ntheta', self.src_ntheta)
            f.createDimension('src_nphi', self.src_nphi)
            f.createDimension('trgt_ntheta', self.trgt_ntheta)
            f.createDimension('trgt_nphi', self.trgt_nphi)
            f.createDimension('xyz', 3)

            src_ntheta = f.createVariable('src_ntheta', 'i', tuple())
            src_ntheta.assignValue(self.src_ntheta)
            src_ntheta.description = 'Number of grid points in the poloidal angle theta for source B field and surface shape'
            src_ntheta.units = 'Dimensionless'

            trgt_ntheta = f.createVariable('trgt_ntheta', 'i', tuple())
            trgt_ntheta.assignValue(self.trgt_ntheta)
            trgt_ntheta.description = 'Number of grid points in the poloidal angle theta for resulting B_external'
            trgt_ntheta.units = 'Dimensionless'

            src_nphi = f.createVariable('src_nphi', 'i', tuple())
            src_nphi.assignValue(self.src_nphi)
            src_nphi.description = 'Number of grid points in the toroidal angle phi for source B field and surface shape'
            src_nphi.units = 'Dimensionless'

            trgt_nphi = f.createVariable('trgt_nphi', 'i', tuple())
            trgt_nphi.assignValue(self.trgt_nphi)
            trgt_nphi.description = 'Number of grid points in the toroidal angle phi for resulting B_external'
            trgt_nphi.units = 'Dimensionless'

            nfp = f.createVariable('nfp', 'i', tuple())
            nfp.assignValue(self.nfp)
            nfp.description = 'Periodicity in toroidal direction'
            nfp.units = 'Dimensionless'

            src_theta = f.createVariable('src_theta', 'd', ('src_ntheta',))
            src_theta[:] = self.src_theta
            src_theta.description = 'Grid points in the poloidal angle theta for source B field and surface shape. Note that theta extends over [0, 1) not [0, 2pi).'
            src_theta.units = 'Dimensionless'

            trgt_theta = f.createVariable('trgt_theta', 'd', ('trgt_ntheta',))
            trgt_theta[:] = self.trgt_theta
            trgt_theta.description = 'Grid points in the poloidal angle theta for resulting B_external. Note that theta extends over [0, 1) not [0, 2pi).'
            trgt_theta.units = 'Dimensionless'

            src_phi = f.createVariable('src_phi', 'd', ('src_nphi',))
            src_phi[:] = self.src_phi
            src_phi.description = 'Grid points in the toroidal angle phi for source B field and surface shape. Note that phi extends over [0, 1) not [0, 2pi).'
            src_phi.units = 'Dimensionless'

            trgt_phi = f.createVariable('trgt_phi', 'd', ('trgt_nphi',))
            trgt_phi[:] = self.trgt_phi
            trgt_phi.description = 'Grid points in the toroidal angle phi for resulting B_external. Note that phi extends over [0, 1) not [0, 2pi).'
            trgt_phi.units = 'Dimensionless'

            gamma = f.createVariable('gamma', 'd', ('src_nphi', 'src_ntheta', 'xyz'))
            gamma[:, :, :] = self.gamma
            gamma.description = 'Position vector on the boundary surface'
            gamma.units = 'meter'

            unit_normal = f.createVariable('unit_normal', 'd', ('trgt_nphi', 'trgt_ntheta', 'xyz'))
            unit_normal[:, :, :] = self.unit_normal
            unit_normal.description = 'Unit-length normal vector on the boundary surface'
            unit_normal.units = 'Dimensionless'

            B_total = f.createVariable('B_total', 'd', ('src_nphi', 'src_ntheta', 'xyz'))
            B_total[:, :, :] = self.B_total
            B_total.description = 'Total magnetic field vector on the surface, including currents both inside and outside of the surface'
            B_total.units = 'Tesla'

            B_external = f.createVariable('B_external', 'd', ('trgt_nphi', 'trgt_ntheta', 'xyz'))
            B_external[:, :, :] = self.B_external
            B_external.description = 'Contribution to the magnetic field vector on the surface due only to currents outside the surface'
            B_external.units = 'Tesla'

            B_external_normal = f.createVariable('B_external_normal', 'd', ('trgt_nphi', 'trgt_ntheta'))
            B_external_normal[:, :] = self.B_external_normal
            B_external_normal.description = 'Component of B_external normal to the surface'
            B_external_normal.units = 'Tesla'

    @classmethod
    def load(cls, filename):
        """
        Load in the results of a previous virtual casing calculation,
        previously saved in NetCDF format.

        Args:
            filename: Name of the file to load.
        """
        vc = cls()
        with netcdf_file(filename, mmap=False) as f:
            for key, val in f.variables.items():
                val2 = val[()]  # Convert to numpy array
                vc.__setattr__(key, val2)
        return vc

    def plot(self, ax=None, show=True):
        """
        Plot ``B_external_normal``, the component normal to the surface of
        the magnetic field generated by currents outside the surface.
        This routine requires ``matplotlib``.

        Args:
            ax: The axis object on which to plot. This argument is useful when plotting multiple
              objects on the same axes. If equal to the default ``None``, a new axis will be created.
            show: Whether to call matplotlib's ``show()`` function.

        Returns:
            An axis which could be passed to a further call to matplotlib if desired.
        """
        import matplotlib.pyplot as plt
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        contours = ax.contourf(self.trgt_phi, self.trgt_theta, self.B_external_normal.T, 25)
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\theta$')
        ax.set_title('B_external_normal [Tesla]')
        fig.colorbar(contours)
        fig.tight_layout()
        if show:
            plt.show()
        return ax
