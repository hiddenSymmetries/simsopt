import logging
from typing import Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
#import matplotlib.pyplot as plt

import simsoptpp as sopp
from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier
from .._core.types import RealArray

logger = logging.getLogger(__name__)

__all__ = ['SurfaceHenneberg']


class SurfaceHenneberg(sopp.Surface, Surface):
    r"""
    This class represents a toroidal surface using the
    parameterization in Henneberg, Helander, and Drevlak, Journal of
    Plasma Physics 87, 905870503 (2021). The main benefit of this
    representation is that there is no freedom in the poloidal angle,
    i.e. :math:`\theta` is uniquely defined, in contrast to other
    parameterizations like
    :obj:`~.surfacerzfourier.SurfaceRZFourier`. Stellarator symmetry
    is assumed.

    In this representation by Henneberg et al, the cylindrical
    coordinates :math:`(R,\phi,Z)` are written in terms of a unique
    poloidal angle :math:`\theta` as follows:

    .. math::
        R(\theta,\phi) = R_0^H(\phi) + \rho(\theta,\phi) \cos(\alpha\phi) - \zeta(\theta,\phi) \sin(\alpha\phi), \\
        Z(\theta,\phi) = Z_0^H(\phi) + \rho(\theta,\phi) \sin(\alpha\phi) + \zeta(\theta,\phi) \cos(\alpha\phi),

    where

    .. math::
        R_0^H(\phi) &=& \sum_{n=0}^{nmax} R_{0,n}^H \cos(n_{fp} n \phi), \\
        Z_0^H(\phi) &=& \sum_{n=1}^{nmax} Z_{0,n}^H \sin(n_{fp} n \phi), \\
        \zeta(\theta,\phi) &=& \sum_{n=0}^{nmax} b_n \cos(n_{fp} n \phi) \sin(\theta - \alpha \phi), \\
        \rho(\theta,\phi) &=& \sum_{n,m} \rho_{n,m} \cos(m \theta - n_{fp} n \phi - \alpha \phi).

    The continuous degrees of freedom are :math:`\{\rho_{m,n}, b_n,
    R_{0,n}^H, Z_{0,n}^H\}`.  These variables correspond to the
    attributes ``rhomn``, ``bn``, ``R0nH``, and ``Z0nH`` respectively,
    which are all numpy arrays.  There is also a discrete degree of
    freedom :math:`\alpha` which should be :math:`\pm n_{fp}/2` where
    :math:`n_{fp}` is the number of field periods. The attribute
    ``alpha_fac`` corresponds to :math:`2\alpha/n_{fp}`, so
    ``alpha_fac`` is either 1, 0, or -1. Using ``alpha_fac = 0`` is
    appropriate for axisymmetry, while values of 1 or -1 are
    appropriate for a stellarator, depending on the handedness of the
    rotating elongation.

    For :math:`R_{0,n}^H` and :math:`b_n`, :math:`n` is 0 or any
    positive integer up through ``nmax`` (inclusive).  For
    :math:`Z_{0,n}^H`, :math:`n` is any positive integer up through
    ``nmax``.  For :math:`\rho_{m,n}`, :math:`m` is an integer from 0
    through ``mmax`` (inclusive). For positive values of :math:`m`,
    :math:`n` can be any integer from ``-nmax`` through ``nmax``.  For
    :math:`m=0`, :math:`n` is restricted to integers from 1 through
    ``nmax``.  Note that we exclude the element of :math:`\rho_{m,n}`
    with :math:`m=n=0`, because this degree of freedom is already
    represented in :math:`R_{0,0}^H`.

    For the 2D array ``rhomn``, functions :func:`set_rhomn()` and
    :func:`get_rhomn()` are provided for convenience so you can specify
    ``n``, since the corresponding array index is shifted by
    ``nmax``. There are no corresponding functions for the 1D arrays
    ``R0nH``, ``Z0nH``, and ``bn`` since these arrays all have a first
    index corresponding to ``n=0``.

    For more information about the arguments ``quadpoints_phi``, and
    ``quadpoints_theta``, see the general documentation on :ref:`surfaces`.
    Instead of supplying the quadrature point arrays along :math:`\phi` and
    :math:`\theta` directions, one could also specify the number of
    quadrature points for :math:`\phi` and :math:`\theta` using the
    class method :py:meth:`~simsopt.geo.surface.Surface.from_nphi_ntheta`.

    Args:
        nfp: The number of field periods.
        alpha_fac: Should be +1 or -1 for a stellarator, depending on the handedness
          by which the elongation rotates, or 0 for axisymmetry.
        mmax: Maximum poloidal mode number included.
        nmax: Maximum toroidal mode number included, divided by ``nfp``.
        quadpoints_phi: Set this to a list or 1D array to set the :math:`\phi_j` grid points directly.
        quadpoints_theta: Set this to a list or 1D array to set the :math:`\theta_j` grid points directly.
    """

    def __init__(self,
                 nfp: int = 1,
                 alpha_fac: int = 1,
                 mmax: int = 1,
                 nmax: int = 0,
                 quadpoints_phi: RealArray = None,
                 quadpoints_theta: RealArray = None
                 ):

        if alpha_fac > 1 or alpha_fac < -1:
            raise ValueError('alpha_fac must be 1, 0, or -1')

        self.nfp = nfp
        self.alpha_fac = alpha_fac
        self.mmax = mmax
        self.nmax = nmax
        self.stellsym = True
        self.allocate()

        if quadpoints_theta is None:
            quadpoints_theta = Surface.get_theta_quadpoints()
        if quadpoints_phi is None:
            quadpoints_phi = Surface.get_phi_quadpoints(nfp=nfp)

        sopp.Surface.__init__(self, quadpoints_phi, quadpoints_theta)
        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.R0nH[0] = 1.0
        self.bn[0] = 0.1
        self.set_rhomn(1, 0, 0.1)

        Surface.__init__(self, x0=self.get_dofs(), names=self._make_names(),
                         external_dof_setter=SurfaceHenneberg.set_dofs_impl)

    def __repr__(self):
        return f"{self.name} (nfp={self.nfp}, alpha_fac={self.alpha_fac}, " \
            + f"mmax={self.mmax}, nmax={self.nmax})"

    def allocate(self):
        """
        Create the arrays for the continuous degrees of freedom. Also set
        the names of the dofs.
        """
        logger.debug("Allocating SurfaceHenneberg")
        # Note that for simpicity, the Z0nH array contains an element
        # for n=0 even though this element is always 0. Similarly, the
        # rhomn array has some elements for (m=0, n<0) even though
        # these elements are always zero.

        self.R0nH = np.zeros(self.nmax + 1)
        self.Z0nH = np.zeros(self.nmax + 1)
        self.bn = np.zeros(self.nmax + 1)

        self.ndim = 2 * self.nmax + 1
        myshape = (self.mmax + 1, self.ndim)
        self.rhomn = np.zeros(myshape)

    def _make_names(self):
        names = []
        for n in range(self.nmax + 1):
            names.append('R0nH(' + str(n) + ')')
        for n in range(1, self.nmax + 1):
            names.append('Z0nH(' + str(n) + ')')
        for n in range(self.nmax + 1):
            names.append('bn(' + str(n) + ')')
        # Handle m = 0 modes in rho_mn:
        for n in range(1, self.nmax + 1):
            names.append('rhomn(0,' + str(n) + ')')
        # Handle m > 0 modes in rho_mn:
        for m in range(1, self.mmax + 1):
            for n in range(-self.nmax, self.nmax + 1):
                names.append('rhomn(' + str(m) + ',' + str(n) + ')')
        return names

    def _validate_mn(self, m, n):
        r"""
        Check whether given (m, n) values are allowed for :math:`\rho_{m,n}`.
        """
        if m < 0:
            raise ValueError(f'm must be >= 0, but m = {m}')
        if m > self.mmax:
            raise ValueError(f'm must be <= mmax, but m = {m}')
        if m == 0 and n < 1:
            raise ValueError(f'For m=0, n must be >= 1, but n = {n}')
        if n > self.nmax:
            raise ValueError(f'n must be <= nmax, but n = {n}')
        if n < -self.nmax:
            raise ValueError(f'n must be >= -nmax, but n = {n}')

    def get_rhomn(self, m, n):
        r"""
        Return a particular :math:`\rho_{m,n}` coefficient.
        """
        self._validate_mn(m, n)
        return self.rhomn[m, n + self.nmax]

    def set_rhomn(self, m, n, val):
        r"""
        Set a particular :math:`\rho_{m,n}` coefficient.
        """
        self._validate_mn(m, n)
        self.rhomn[m, n + self.nmax] = val
        self.invalidate_cache()

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        return np.concatenate((self.R0nH, self.Z0nH[1:], self.bn,
                               self.rhomn[0, self.nmax + 1:],
                               np.reshape(self.rhomn[1:, :], (self.mmax * (2 * self.nmax + 1),), order='C')))

    def set_dofs(self, dofs):
        self.local_x = dofs

    def num_dofs(self):
        """
        Return the number of degrees of freedom.
        """
        ndofs = self.nmax + 1  # R0nH
        ndofs += self.nmax  # Z0nH
        ndofs += self.nmax + 1  # b0n
        ndofs += self.nmax  # rhomn for m = 0
        ndofs += self.mmax * (2 * self.nmax + 1)  # rhomn for m > 0

        return ndofs

    def set_dofs_impl(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        n = self.num_dofs()
        if len(v) != n:
            raise ValueError('Input vector should have ' + str(n) + \
                             ' elements but instead has ' + str(len(v)))

        index = 0
        nvals = self.nmax + 1
        self.R0nH = v[index: index + nvals]
        index += nvals

        nvals = self.nmax
        self.Z0nH[1:] = v[index: index + nvals]
        index += nvals

        nvals = self.nmax + 1
        self.bn = v[index: index + nvals]
        index += nvals

        nvals = self.nmax
        self.rhomn[0, self.nmax + 1:] = v[index: index + nvals]
        index += nvals

        self.rhomn[1:, :] = np.reshape(v[index:], (self.mmax, 2 * self.nmax + 1), order='C')

    def fixed_range(self, mmax, nmax, fixed=True):
        """
        Set the ``fixed`` property for a range of ``m`` and ``n`` values.

        All modes with ``m <= mmax`` and ``|n| <= nmax`` will have
        their fixed property set to the value of the ``fixed``
        parameter. Note that ``mmax`` and ``nmax`` are included.

        Both ``mmax`` and ``nmax`` must be >= 0.

        For any value of ``mmax``, the ``fixed`` properties of
        ``R0nH``, ``Z0nH``, and ``rhomn`` are set. The ``fixed``
        properties of ``bn`` are set only if ``mmax > 0``. In other
        words, the ``bn`` modes are treated as having ``m=1``.
        """
        if mmax < 0:
            raise ValueError('mmax must be >= 0')
        if mmax > self.mmax:
            mmax = self.mmax
        if nmax < 0:
            raise ValueError('nmax must be >= 0')
        if nmax > self.nmax:
            nmax = self.nmax

        fn = self.fix if fixed else self.unfix

        for n in range(nmax + 1):
            fn(f'R0nH({n})')
        for n in range(1, nmax + 1):
            fn(f'Z0nH({n})')
        if mmax > 0:
            for n in range(nmax + 1):
                fn(f'bn({n})')

        for m in range(mmax + 1):
            nmin_to_use = -nmax
            if m == 0:
                nmin_to_use = 1
            for n in range(nmin_to_use, nmax + 1):
                fn(f'rhomn({m},{n})')

    def to_RZFourier(self):
        """
        Return a :obj:`~.surfacerzfourier.SurfaceRZFourier` object with the identical shape. This
        routine implements eq (4.5)-(4.6) in the Henneberg paper, plus
        m=0 terms for R0 and Z0.
        """
        mpol = self.mmax
        ntor = self.nmax + 1  # More modes are needed in the SurfaceRZFourier because some indices are shifted by +/- 2*alpha.
        s = SurfaceRZFourier(nfp=self.nfp, stellsym=True, mpol=mpol, ntor=ntor)
        s.rc[:] = 0.0
        s.zs[:] = 0.0

        # Set Rmn.
        # Handle the 1d arrays (R0nH, bn):
        for nprime in range(self.nmax + 1):
            n = nprime
            # Handle the R0nH term:
            s.set_rc(0, n, s.get_rc(0, n) + self.R0nH[n])
            # Handle the b_n term:
            s.set_rc(1, n, s.get_rc(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{-n} term:
            n = -nprime
            s.set_rc(1, n, s.get_rc(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{n-2alpha} term:
            n = nprime + self.alpha_fac
            s.set_rc(1, n, s.get_rc(1, n) - 0.25 * self.bn[nprime])
            # Handle the b_{-n+2alpha} term:
            n = -nprime + self.alpha_fac
            s.set_rc(1, n, s.get_rc(1, n) - 0.25 * self.bn[nprime])
        # Handle the 2D rho terms:
        for m in range(self.mmax + 1):
            nmin = -self.nmax
            if m == 0:
                nmin = 1
            for nprime in range(nmin, self.nmax + 1):
                # Handle the rho_{m, -n} term:
                n = -nprime
                s.set_rc(m, n, s.get_rc(m, n) + 0.5 * self.get_rhomn(m, nprime))
                # Handle the rho_{m, -n+2alpha} term:
                n = -nprime + self.alpha_fac
                s.set_rc(m, n, s.get_rc(m, n) + 0.5 * self.get_rhomn(m, nprime))

        # Set Zmn.
        # Handle the 1d arrays (Z0nH, bn):
        for nprime in range(self.nmax + 1):
            n = nprime
            # Handle the Z0nH term:
            s.set_zs(0, n, s.get_zs(0, n) - self.Z0nH[n])
            # Handle the b_n term:
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{-n} term:
            n = -nprime
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{n-2alpha} term:
            n = nprime + self.alpha_fac
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
            # Handle the b_{-n+2alpha} term:
            n = -nprime + self.alpha_fac
            s.set_zs(1, n, s.get_zs(1, n) + 0.25 * self.bn[nprime])
        # Handle the 2D rho terms:
        for m in range(self.mmax + 1):
            nmin = -self.nmax
            if m == 0:
                nmin = 1
            for nprime in range(nmin, self.nmax + 1):
                # Handle the rho_{m, -n} term:
                n = -nprime
                s.set_zs(m, n, s.get_zs(m, n) + 0.5 * self.get_rhomn(m, nprime))
                # Handle the rho_{m, -n+2alpha} term:
                n = -nprime + self.alpha_fac
                s.set_zs(m, n, s.get_zs(m, n) - 0.5 * self.get_rhomn(m, nprime))

        return s

    @classmethod
    def from_RZFourier(cls,
                       surf,
                       alpha_fac: int,
                       mmax: Union[int, None] = None,
                       nmax: Union[int, None] = None,
                       ntheta: Union[int, None] = None,
                       nphi: Union[int, None] = None):
        """
        Convert a :obj:`~.surfacerzfourier.SurfaceRZFourier` surface to a
        :obj:`SurfaceHenneberg` surface.

        Args:
            surf: The :obj:`~.surfacerzfourier.SurfaceRZFourier` object to convert.
            mmax: Maximum poloidal mode number to include in the new surface. If ``None``,
              the value ``mpol`` from the old surface will be used.
            nmax: Maximum toroidal mode number to include in the new surface. If ``None``,
              the value ``ntor`` from the old surface will be used.
            ntheta: Number of grid points in the poloidal angle used for the transformation.
              If ``None``, the value ``3 * ntheta`` will be used.
            nphi: Number of grid points in the toroidal angle used for the transformation.
              If ``None``, the value ``3 * nphi`` will be used.
        """
        if not surf.stellsym:
            raise RuntimeError('SurfaceHenneberg.from_RZFourier method only '
                               'works for stellarator symmetric surfaces')
        if mmax is None:
            mmax = surf.mpol
        if nmax is None:
            nmax = surf.ntor
        if ntheta is None:
            ntheta = mmax * 3
        if nphi is None:
            nphi = nmax * 3
        logger.info(f'Beginning conversion with mmax={mmax}, nmax={nmax}, ntheta={ntheta}, nphi={nphi}')
        nfp = surf.nfp
        theta = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        alpha = 0.5 * nfp * alpha_fac

        # Initialize arrays to store quantities in real-space:
        R0_realsp = np.zeros(nphi)
        Z0_realsp = np.zeros(nphi)
        b_realsp = np.zeros(nphi)
        rho_realsp = np.zeros((ntheta, nphi))

        def b_min(theta, phi0, cosaphi, sinaphi):
            """
            This function is minimized as part of finding b.
            """
            R = 0
            Z = 0
            for m in range(surf.mpol + 1):
                for n in range(-surf.ntor, surf.ntor + 1):
                    angle = m * theta - n * nfp * phi0
                    R += surf.get_rc(m, n) * np.cos(angle)
                    Z += surf.get_zs(m, n) * np.sin(angle)
            return Z * cosaphi - R * sinaphi

        def b_max(theta, phi0, cosaphi, sinaphi):
            return -b_min(theta, phi0, cosaphi, sinaphi)

        # An independent transformation is performed at each grid point in phi:
        for jphi, phi0 in enumerate(phi):
            logger.debug(f'Transforming jphi={jphi} of {nphi}')
            cosaphi = np.cos(alpha * phi0)
            sinaphi = np.sin(alpha * phi0)

            # Find the max and min of the surface in the zeta direction:
            opt_result = minimize_scalar(b_min, args=(phi0, cosaphi, sinaphi), tol=1e-12)
            min_for_b = opt_result.fun

            opt_result = minimize_scalar(b_max, args=(phi0, cosaphi, sinaphi), tol=1e-12)
            max_for_b = -opt_result.fun

            b = 0.5 * (max_for_b - min_for_b)
            Q = 0.5 * (max_for_b + min_for_b)

            R = np.zeros(ntheta)
            Z = np.zeros(ntheta)
            d_Z_d_theta = np.zeros(ntheta)
            d_R_d_theta = np.zeros(ntheta)
            for m in range(surf.mpol + 1):
                for n in range(-surf.ntor, surf.ntor + 1):
                    angle = m * theta - n * nfp * phi0
                    R += surf.get_rc(m, n) * np.cos(angle)
                    Z += surf.get_zs(m, n) * np.sin(angle)
                    d_Z_d_theta += surf.get_zs(m, n) * m * np.cos(angle)
                    d_R_d_theta += surf.get_rc(m, n) * m * (-np.sin(angle))

            # Now compute the new theta for each grid point in the old
            # theta.  This mostly amount to taking an arcsin, but we
            # must be careful to assign points to the proper half of
            # [0, 2pi], since arcsin by itself only returns values in
            # the range [-pi/2, pi/2].
            d_Z_rot_d_theta = d_Z_d_theta * cosaphi - d_R_d_theta * sinaphi
            # Copy the first element to the end, for periodicity:
            d_Z_rot_d_theta_circ = np.concatenate((d_Z_rot_d_theta, [d_Z_rot_d_theta[0]]))
            sign_flips = d_Z_rot_d_theta_circ[1:] * d_Z_rot_d_theta_circ[:-1]
            sign_flip_indices = [j for j in range(ntheta) if sign_flips[j] < 0]
            if len(sign_flip_indices) != 2:
                logger.warning(f'A number of sign flips other than 2 detected for jphi={jphi}: sign_flip_indices={sign_flip_indices}.' \
                               ' This may mean the surface cannot be represented in Henneberg form.' \
                               f' sign_flips={sign_flips}')

            temp = (Z * cosaphi - R * sinaphi - Q) / b
            if np.any(temp > 1):
                # Going outside [-1, 1] by ~ roundoff is okay, but
                # warn if we are much farther than that.
                if np.any(temp > 1 + 1.0e-12):
                    logger.warning(f'Argument of arcsin exceeds 1: {temp[temp > 1] - 1}')
                temp[temp > 1] = 1.0
            if np.any(temp < -1):
                if np.any(temp < -1 - 1.0e-12):
                    logger.warning(f'Argument of arcsin is below -1: {temp[temp < -1] + 1}')
                temp[temp < -1] = -1.0

            arcsin_term = np.arcsin(temp)
            mask = d_Z_rot_d_theta < 0
            arcsin_term[mask] = np.pi - arcsin_term[mask]
            mask = arcsin_term < 0
            arcsin_term[mask] = arcsin_term[mask] + 2 * np.pi
            theta_H = arcsin_term + alpha * phi0

            # Copy arrays 3 times, so endpoints are interpolated correctly:
            theta_H_3 = np.concatenate((theta_H - 2 * np.pi, theta_H, theta_H + 2 * np.pi))
            R_3 = np.concatenate((R, R, R))
            Z_3 = np.concatenate((Z, Z, Z))
            R_interp = interp1d(theta_H_3, R_3, kind='cubic')
            Z_interp = interp1d(theta_H_3, Z_3, kind='cubic')

            R_H = R_interp(theta)
            Z_H = Z_interp(theta)

            avg_R = np.mean(R_H)
            avg_Z = np.mean(Z_H)

            R0H = avg_R * cosaphi * cosaphi + avg_Z * sinaphi * cosaphi - Q * sinaphi
            Z0H = avg_R * cosaphi * sinaphi + avg_Z * sinaphi * sinaphi + Q * cosaphi

            R0_realsp[jphi] = R0H
            Z0_realsp[jphi] = Z0H
            b_realsp[jphi] = b
            rho_realsp[:, jphi] = (R_H - R0H) * cosaphi + (Z_H - Z0H) * sinaphi

        surf_H = cls(nfp=nfp, alpha_fac=alpha_fac, mmax=mmax, nmax=nmax)
        # Now convert from real-space to Fourier space.
        # Start with the 0-frequency terms:
        surf_H.R0nH[0] = np.mean(R0_realsp)
        surf_H.bn[0] = np.mean(b_realsp)
        Z00H = np.mean(Z0_realsp)
        logger.info(f'n=0 term of Z0nH: {Z00H} (should be ~ 0)')
        assert np.abs(Z00H) < 1.0e-6
        rho00 = np.mean(rho_realsp)
        logger.info(f'm=n=0 term of rho_mn: {rho00} (should be ~ 0)')
        assert np.abs(rho00) < 1.0e-6

        # Now handle 1D arrays:
        for n in range(1, nmax + 1):
            cosnphi_fac = np.cos(n * nfp * phi) / nphi
            sinnphi_fac = np.sin(n * nfp * phi) / nphi
            surf_H.R0nH[n] = 2 * np.sum(R0_realsp * cosnphi_fac)
            surf_H.Z0nH[n] = 2 * np.sum(Z0_realsp * sinnphi_fac)
            surf_H.bn[n] = 2 * np.sum(b_realsp * cosnphi_fac)

        # Transform rho:
        phi2d, theta2d = np.meshgrid(phi, theta)
        #print('phi2d.shape:', phi2d.shape)
        for m in range(mmax + 1):
            nmin = -nmax
            if m == 0:
                nmin = 1
            for n in range(nmin, nmax + 1):
                # Eq above (4.5):
                angle = m * theta2d + (n * nfp - alpha) * phi2d
                surf_H.set_rhomn(m, n, 2 * np.sum(rho_realsp * np.cos(angle)) / (ntheta * nphi))

        """
        plt.figure()
        plt.contourf(phi2d, theta2d, rho_realsp, 25)
        plt.colorbar()
        plt.xlabel('phi')
        plt.ylabel('theta')
        plt.title('rho_realsp')
        plt.show()
        """

        # Check that the inverse-transform of the transform gives back
        # the original arrays, approximately:
        b_alt = np.zeros(nphi)
        R0_alt = np.zeros(nphi)
        Z0_alt = np.zeros(nphi)
        for n in range(nmax + 1):
            b_alt += surf_H.bn[n] * np.cos(n * nfp * phi)
            R0_alt += surf_H.R0nH[n] * np.cos(n * nfp * phi)
            Z0_alt += surf_H.Z0nH[n] * np.sin(n * nfp * phi)

        print('b_realsp:', b_realsp)
        print('b_alt:   ', b_alt)
        print('bn:', surf_H.bn)
        print('Diff in b:', np.max(np.abs(b_alt - b_realsp)))
        print('Diff in R0:', np.max(np.abs(R0_alt - R0_realsp)))
        print('Diff in Z0:', np.max(np.abs(Z0_alt - Z0_realsp)))

        rho_alt = np.zeros((ntheta, nphi))
        for m in range(mmax + 1):
            nmin = -nmax
            if m == 0:
                nmin = 1
            for n in range(nmin, nmax + 1):
                angle = m * theta2d + (n * nfp - alpha) * phi2d
                rho_alt += surf_H.get_rhomn(m, n) * np.cos(angle)
        #print('rho_realsp:', rho_realsp)
        #print('rho_alt:   ', rho_alt)
        print('Diff in rho:', np.max(np.abs(rho_realsp - rho_alt)))

        surf_H.local_full_x = surf_H.get_dofs()
        return surf_H

    def gamma_lin(self, data, quadpoints_phi, quadpoints_theta):
        """
        Evaluate the position vector on the surface in Cartesian
        coordinates, for a list of (phi, theta) points.
        """
        # I prefer to work with angles that go up to 2pi rather than 1.
        theta = quadpoints_theta * 2 * np.pi
        phi = quadpoints_phi * 2 * np.pi
        nfp = self.nfp
        shape = phi.shape
        R0H = np.zeros(shape)
        Z0H = np.zeros(shape)
        b = np.zeros(shape)
        rho = np.zeros(shape)
        alpha = 0.5 * nfp * self.alpha_fac
        for n in range(self.nmax + 1):
            cosangle = np.cos(nfp * n * phi)
            R0H += self.R0nH[n] * cosangle
            b += self.bn[n] * cosangle
        for n in range(1, self.nmax + 1):
            sinangle = np.sin(nfp * n * phi)
            Z0H += self.Z0nH[n] * sinangle
        for m in range(self.mmax + 1):
            nmin = -self.nmax
            if m == 0:
                nmin = 1
            for n in range(nmin, self.nmax + 1):
                cosangle = np.cos(m * theta + nfp * n * phi - alpha * phi)
                rho += self.get_rhomn(m, n) * cosangle
        zeta = b * np.sin(theta - alpha * phi)
        sinaphi = np.sin(alpha * phi)
        cosaphi = np.cos(alpha * phi)
        R = R0H + rho * cosaphi - zeta * sinaphi
        Z = Z0H + rho * sinaphi + zeta * cosaphi
        data[:, 0] = R * np.cos(phi)
        data[:, 1] = R * np.sin(phi)
        data[:, 2] = Z

    def gamma_impl(self, data, quadpoints_phi, quadpoints_theta):
        """
        Evaluate the position vector on the surface in Cartesian
        coordinates, for a tensor product grid of points in theta and
        phi.
        """
        nphi = len(quadpoints_phi)
        ntheta = len(quadpoints_theta)
        phi2d, theta2d = np.meshgrid(quadpoints_phi, quadpoints_theta)
        data1d = np.zeros((nphi * ntheta, 3))
        self.gamma_lin(data1d,
                       np.reshape(phi2d, (nphi * ntheta,)),
                       np.reshape(theta2d, (nphi * ntheta,)))
        for xyz in range(3):
            data[:, :, xyz] = np.reshape(data1d[:, xyz], (ntheta, nphi)).T

    def gammadash1_impl(self, data):
        """
        Evaluate the derivative of the position vector with respect to the
        toroidal angle phi.
        """
        # I prefer to work with angles that go up to 2pi rather than 1.
        theta1D = self.quadpoints_theta * 2 * np.pi
        phi1D = self.quadpoints_phi * 2 * np.pi
        nphi = len(phi1D)
        ntheta = len(theta1D)
        nfp = self.nfp
        R0H = np.zeros(nphi)
        Z0H = np.zeros(nphi)
        b = np.zeros(nphi)
        rho = np.zeros((ntheta, nphi))
        d_R0H_d_phi = np.zeros(nphi)
        d_Z0H_d_phi = np.zeros(nphi)
        d_b_d_phi = np.zeros(nphi)
        d_rho_d_phi = np.zeros((ntheta, nphi))
        phi, theta = np.meshgrid(phi1D, theta1D)
        alpha = 0.5 * nfp * self.alpha_fac
        for n in range(self.nmax + 1):
            angle = nfp * n * phi1D
            cosangle = np.cos(angle)
            sinangle = np.sin(angle)
            R0H += self.R0nH[n] * cosangle
            b += self.bn[n] * cosangle
            d_R0H_d_phi -= self.R0nH[n] * sinangle * nfp * n
            d_b_d_phi -= self.bn[n] * sinangle * nfp * n
            if n > 0:
                Z0H += self.Z0nH[n] * sinangle
                d_Z0H_d_phi += self.Z0nH[n] * cosangle * nfp * n
        for m in range(self.mmax + 1):
            nmin = -self.nmax
            if m == 0:
                nmin = 1
            for n in range(nmin, self.nmax + 1):
                angle = m * theta + nfp * n * phi - alpha * phi
                cosangle = np.cos(angle)
                sinangle = np.sin(angle)
                rho += self.get_rhomn(m, n) * cosangle
                d_rho_d_phi -= self.get_rhomn(m, n) * sinangle * (nfp * n - alpha)
        R0H2D = np.kron(R0H, np.ones((ntheta, 1)))
        Z0H2D = np.kron(Z0H, np.ones((ntheta, 1)))
        b2D = np.kron(b, np.ones((ntheta, 1)))
        zeta = b2D * np.sin(theta - alpha * phi)
        d_R0H2D_d_phi = np.kron(d_R0H_d_phi, np.ones((ntheta, 1)))
        d_Z0H2D_d_phi = np.kron(d_Z0H_d_phi, np.ones((ntheta, 1)))
        d_b2D_d_phi = np.kron(d_b_d_phi, np.ones((ntheta, 1)))
        d_zeta_d_phi = d_b2D_d_phi * np.sin(theta - alpha * phi) \
            + b2D * np.cos(theta - alpha * phi) * (-alpha)
        sinaphi = np.sin(alpha * phi)
        cosaphi = np.cos(alpha * phi)
        R = R0H2D + rho * cosaphi - zeta * sinaphi
        Z = Z0H2D + rho * sinaphi + zeta * cosaphi
        d_R_d_phi = d_R0H2D_d_phi + d_rho_d_phi * cosaphi + rho * (-alpha * sinaphi) \
            - d_zeta_d_phi * sinaphi - zeta * (alpha * cosaphi)
        d_Z_d_phi = d_Z0H2D_d_phi + d_rho_d_phi * sinaphi + rho * (alpha * cosaphi) \
            + d_zeta_d_phi * cosaphi + zeta * (-alpha * sinaphi)
        # Insert factors of 2pi since theta here is 2pi times the theta used for d/dtheta
        data[:, :, 0] = 2 * np.pi * (d_R_d_phi * np.cos(phi) - R * np.sin(phi)).T
        data[:, :, 1] = 2 * np.pi * (d_R_d_phi * np.sin(phi) + R * np.cos(phi)).T
        data[:, :, 2] = 2 * np.pi * d_Z_d_phi.T

    def gammadash2_impl(self, data):
        """
        Evaluate the derivative of the position vector with respect to
        theta.
        """
        # I prefer to work with angles that go up to 2pi rather than 1.
        theta1D = self.quadpoints_theta * 2 * np.pi
        phi1D = self.quadpoints_phi * 2 * np.pi
        nphi = len(phi1D)
        ntheta = len(theta1D)
        b = np.zeros(nphi)
        d_rho_d_theta = np.zeros((ntheta, nphi))
        phi, theta = np.meshgrid(phi1D, theta1D)
        alpha = 0.5 * self.nfp * self.alpha_fac
        for n in range(self.nmax + 1):
            cosangle = np.cos(self.nfp * n * phi1D)
            b += self.bn[n] * cosangle
        for m in range(self.mmax + 1):
            nmin = -self.nmax
            if m == 0:
                nmin = 1
            for n in range(nmin, self.nmax + 1):
                sinangle = np.sin(m * theta + self.nfp * n * phi - alpha * phi)
                d_rho_d_theta -= self.get_rhomn(m, n) * m * sinangle
        b2D = np.kron(b, np.ones((ntheta, 1)))
        d_zeta_d_theta = b2D * np.cos(theta - alpha * phi)
        sinaphi = np.sin(alpha * phi)
        cosaphi = np.cos(alpha * phi)
        d_R_d_theta = d_rho_d_theta * cosaphi - d_zeta_d_theta * sinaphi
        d_Z_d_theta = d_rho_d_theta * sinaphi + d_zeta_d_theta * cosaphi
        # Insert factors of 2pi since theta here is 2pi times the theta used for d/dtheta
        data[:, :, 0] = (2 * np.pi * d_R_d_theta * np.cos(phi)).T
        data[:, :, 1] = (2 * np.pi * d_R_d_theta * np.sin(phi)).T
        data[:, :, 2] = 2 * np.pi * d_Z_d_theta.T

    def as_dict(self) -> dict:
        d = super().as_dict()
        d["alpha_fac"] = self.alpha_fac
        d["mmax"] = self.mmax
        d["nmax"] = self.nmax
        return d

    @classmethod
    def from_dict(cls, d):
        surf = cls(nfp=d["nfp"], alpha_fac=d["alpha_fac"],
                   mmax=d["mmax"], nmax=d["nmax"],
                   quadpoints_phi=d["quadpoints_phi"],
                   quadpoints_theta=d["quadpoints_theta"])
        surf.local_full_x = d["x0"]
        return surf

