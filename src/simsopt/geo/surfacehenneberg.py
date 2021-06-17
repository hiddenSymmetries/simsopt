import logging
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

from .surface import Surface
from .surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)


class SurfaceHenneberg(Surface):
    r"""
    This class represents a toroidal surface using the
    parameterization in Henneberg, Helander, and Drevlak,
    arXiv:2105.00768 (2021).

    In the implementation here, stellarator symmetry is assumed.

    The continuous degrees of freedom are :math:`\{\rho_{m,n}, b_n,
    R_{0,n}^H, Z_{0,n}^H\}`.  These variables correspond to the
    attributes ``rhomn``, ``bn``, ``R0nH``, and ``Z0nH`` respectively,
    which are all numpy arrays.  There is also a discrete degree of
    freedom :math:`\alpha` which should be :math:`\pm n_{fp}/2` where
    :math:`n_{fp}` is the number of field periods. The attribute
    ``alpha_fac`` corresponds to :math:`2\alpha/n_{fp}`, so
    ``alpha_fac`` is either 1 or -1.

    For :math:`R_{0,n}^H` and :math:`b_n`, :math:`n` is 0 or any
    positive integer up through ``ntor`` (inclusive).  For
    :math:`Z_{0,n}^H`, :math:`n` is any positive integer up through
    ``ntor``.  For :math:`\rho_{m,n}`, :math:`m` is an integer from 0
    through ``mpol`` (inclusive). For positive values of :math:`m`,
    :math:`n` can be any integer from ``-ntor`` through ``ntor``.  For
    :math:`m=0`, :math:`n` is restricted to integers from 1 through
    ``ntor``.  Note that we exclude the element of :math:`\rho_{m,n}`
    with :math:`m=n=0`, because this degree of freedom is already
    represented in :math:`R_{0,0}^H`.

    For the 2D array ``rhomn``, functions :func:`set_rhomn()` and
    :func:`get_rhomn()` are provided for convenience so you can specify
    ``n``, since the corresponding array index is shifted by
    ``nmax``. There are no corresponding functions for the 1D arrays
    ``R0nH``, ``Z0nH``, and ``bn`` since these arrays all have a first
    index corresponding to ``n=0``.
    """

    def __init__(self, nfp, alpha_fac, mmax, nmax):
        if alpha_fac != 1 and alpha_fac != -1:
            raise ValueError('alpha_fac must be 1 or -1')

        self.nfp = nfp
        self.alpha_fac = alpha_fac
        self.mmax = mmax
        self.nmax = nmax
        self.stellsym = True
        self.allocate()
        self.recalculate = True

        # Initialize to an axisymmetric torus with major radius 1m and
        # minor radius 0.1m
        self.R0nH[0] = 1.0
        self.bn[0] = 0.1
        self.set_rhomn(1, 0, 0.1)
        Surface.__init__(self)

    def __repr__(self):
        return "SurfaceHenneberg " + str(hex(id(self))) + " (nfp=" + \
            str(self.nfp) + ", alpha=" + str(self.alpha_fac * 0.5) \
            + ", mmax=" + str(self.mmax) + ", nmax=" + str(self.nmax) + ")"

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

        self.names = []
        for n in range(self.nmax + 1):
            self.names.append('R0nH(' + str(n) + ')')
        for n in range(1, self.nmax + 1):
            self.names.append('Z0nH(' + str(n) + ')')
        for n in range(self.nmax + 1):
            self.names.append('bn(' + str(n) + ')')
        # Handle m = 0 modes in rho_mn:
        for n in range(1, self.nmax + 1):
            self.names.append('rhomn(0,' + str(n) + ')')
        # Handle m > 0 modes in rho_mn:
        for m in range(1, self.mmax + 1):
            for n in range(-self.nmax, self.nmax + 1):
                self.names.append('rhomn(' + str(m) + ',' + str(n) + ')')

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
        return self.rhomn[m, n - self.nmax]

    def set_rhomn(self, m, n, val):
        r"""
        Set a particular :math:`\rho_{m,n}` coefficient.
        """
        self._validate_mn(m, n)
        self.rhomn[m, n - self.nmax] = val
        self.recalculate = True

    def get_dofs(self):
        """
        Return a 1D numpy array with all the degrees of freedom.
        """
        return np.concatenate((self.R0nH, self.Z0nH[1:], self.bn,
                               self.rhomn[0, self.nmax + 1:],
                               np.reshape(self.rhomn[1:, :], (self.mmax * (2 * self.nmax + 1),), order='F')))

    def set_dofs(self, v):
        """
        Set the shape coefficients from a 1D list/array
        """

        n = len(self.get_dofs())
        if len(v) != n:
            raise ValueError('Input vector should have ' + str(n) + \
                             ' elements but instead has ' + str(len(v)))

        # Check whether any elements actually change:
        if np.all(np.abs(self.get_dofs() - np.array(v)) == 0):
            logger.info('set_dofs called, but no dofs actually changed')
            return

        logger.info('set_dofs called, and at least one dof changed')
        self.recalculate = True

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

        self.rhomn[1:, :] = np.reshape(v[index:], (self.mmax, 2 * self.nmax + 1), order='F')

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

        for n in range(nmax + 1):
            self.set_fixed(f'R0nH({n})', fixed)
        for n in range(1, nmax + 1):
            self.set_fixed(f'Z0nH({n})', fixed)
        if mmax > 0:
            for n in range(nmax + 1):
                self.set_fixed(f'bn({n})', fixed)

        for m in range(mmax + 1):
            nmin_to_use = -nmax
            if m == 0:
                nmin_to_use = 1
            for n in range(nmin_to_use, nmax + 1):
                self.set_fixed(f'rhomn({m},{n})', fixed)

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
                       mmax=None,
                       nmax=None,
                       ntheta=None,
                       nphi=None):
        """
        Convert a :obj:`~.surfacerzfourier.SurfaceRZFourier` surface to a
        :obj:`SurfaceHenneberg` surface.

        Args:
            surf: The :obj:`~.surfacerzfourier.SurfaceRZFourier` object to convert.
            mmax: 
        """
        if not surf.stellsym:
            raise RuntimeError('SurfaceHenneberg.from_RZFourier() only works for stellarator symmetric surfaces')
        if mmax is None:
            mmax = surf.mpol
        if nmax is None:
            nmax = surf.ntor
        if ntheta is None:
            ntheta = mmax * 4
        if nphi is None:
            nphi = nmax * 4
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

        for jphi, phi0 in enumerate(phi):
            logger.debug(f'Transforming jphi={jphi} of {nphi}')
            cosaphi = np.cos(alpha * phi0)
            sinaphi = np.sin(alpha * phi0)

            opt_result = minimize_scalar(b_min, args=(phi0, cosaphi, sinaphi))
            min_for_b = opt_result.fun

            opt_result = minimize_scalar(b_max, args=(phi0, cosaphi, sinaphi))
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

            d_Z_rot_d_theta = d_Z_d_theta * cosaphi - d_R_d_theta * sinaphi

            # Old method:
            """
            theta_H = np.arcsin((Z * cosaphi - R * sinaphi - Q) / b) + alpha * phi0
            # Copy the first element to the end, for periodicity:
            d_Z_rot_d_theta_circ = np.concatenate((d_Z_rot_d_theta, [d_Z_rot_d_theta[0]]))
            sign_flips = d_Z_rot_d_theta_circ[1:] * d_Z_rot_d_theta_circ[:-1]
            sign_flip_indices = [j for j in range(ntheta) if sign_flips[j] <= 0]
            if len(sign_flip_indices) != 2:
                raise RuntimeError(f'More than 2 sign flips detected for jphi={jphi}: sign_flip_indices={sign_flip_indices}')
            #print('Number of sign flips:', np.sum(sign_flips <= 0))
            #print('sign flip indices:', sign_flip_indices)
            theta_H[sign_flip_indices[0] + 1 : sign_flip_indices[1] + 1] = np.pi - theta_H[sign_flip_indices[0] + 1:sign_flip_indices[1] + 1] + 2 * alpha * phi0
            theta_H[sign_flip_indices[1] + 1:] += 2 * np.pi
            """

            # New method:
            arcsin_term = np.arcsin((Z * cosaphi - R * sinaphi - Q) / b)
            # arcsin returns values in the range [-pi/2, pi/2]
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
        print('phi2d.shape:', phi2d.shape)
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

        # Check transforms:        
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
        print('rho_realsp:', rho_realsp)
        print('rho_alt:   ', rho_alt)
        print('Diff in rho:', np.max(np.abs(rho_realsp - rho_alt)))

        return surf_H

    def area_volume(self):
        """
        Compute the surface area and the volume enclosed by the surface.
        """
        if self.recalculate:
            logger.info('Running calculation of area and volume')
        else:
            logger.info('area_volume called, but no need to recalculate')
            return

        self.recalculate = False

        # Delegate to the area and volume calculations of SurfaceRZFourier():
        s = self.to_RZFourier()
        self._area = s.area()
        self._volume = s.volume()

    def area(self):
        """
        Return the area of the surface.
        """
        self.area_volume()
        return self._area

    def volume(self):
        """
        Return the volume of the surface.
        """
        self.area_volume()
        return self._volume

