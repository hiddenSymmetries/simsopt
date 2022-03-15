# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module provides routines for interacting with the virtual_casing package,
for e.g. computing the magnetic field on a surface due to currents inside the
surface.
"""

import logging
import numpy as np
from .vmec_diagnostics import B_cartesian
from ..geo.surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)

# TODO:
# * Check that quadpoints_phi is "full torus"
# * Check that nphi is a multiple of nfp?
# * Allow for different ntheta & nphi compared to the vmec.boundary object
# * Routine for finding the optimal ntheta:nphi ratio
# * Separate routine to save data in a file
# * plotting routine
# * Tests:
#   - Compare to reference BNORM calculation for d23_t4
#   - For a vacuum config, B_internal should be 0
#   - Should have nfp and stellarator symmetry


class VirtualCasing:
    """
    Use the virtual casing principle to compute the contribution to
    the total magnetic field due to current inside a bounded surface.
    """

    @classmethod
    def from_vmec(cls, vmec, nphi, ntheta, digits=6, filename=None):
        """
        Given a :obj:`~simsopt.mhd.vmec.Vmec` object, compute the
        contribution to the total magnetic field due to currents inside
        the plasma.

        This function requires the python ``virtual_casing`` package to be
        installed.

        The argument ``nphi`` refers to the number of points around the
        full torus. It must be a multiple of ``2 * nfp``, so there is an
        integer number of points per half field period.

        For now, this routine only works for stellarator symmetry.

        Args:
            filename: If not ``None``, the Bnormal data will be saved in this file.
        """
        import virtual_casing as vc_module

        vmec.run()
        nfp = vmec.wout.nfp
        if vmec.wout.lasym:
            raise RuntimeError('virtual casing presently only works for stellarator symmetry')
        if nphi % 2 * nfp != 0:
            raise ValueError('nphi must be a multiple of 2 * nfp')

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
            vc.save(filename)

        return vc

    def save(self, filename="vcasing.nc"):
        pass

    @classmethod
    def load(filename):
        pass

    def resample(self, ntheta, nphi):
        pass
