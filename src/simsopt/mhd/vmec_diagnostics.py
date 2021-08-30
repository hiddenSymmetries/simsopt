# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains functions that can postprocess VMEC output.
"""

import logging
from typing import Union
import numpy as np
from scipy.interpolate import interp1d
from .._core.optimizable import Optimizable
from .vmec import Vmec
from .._core.util import Struct
from ..util.types import RealArray
logger = logging.getLogger(__name__)


class QuasisymmetryRatioResidual(Optimizable):
    r"""
    This class provides a measure of the deviation from quasisymmetry,
    one that can be computed without Boozer coordinates.  This metric
    is based on the fact that for quasisymmetry, the ratio

    .. math::
        (\vec{B}\times\nabla B \cdot\nabla\psi) / (\vec{B} \cdot\nabla B)

    is constant on flux surfaces.

    Specifically, this class represents the objective function

    .. math::
        f = \sum_{s_j} w_j \left< \left[ \frac{1}{B^3} \left( (N - \iota M)\vec{B}\times\nabla B\cdot\nabla\psi - (MG+NI)\vec{B}\cdot\nabla B \right) \right]^2 \right>

    where the sum is over a set of flux surfaces with normalized
    toroidal flux :math:`s_j`, the coefficients :math:`w_j` are
    user-supplied weights, :math:`\left< \ldots \right>` denotes a
    flux surface average, :math:`G(s)` is :math:`\mu_0/(2\pi)` times
    the poloidal current outside the surface, :math:`I(s)` is
    :math:`\mu_0/(2\pi)` times the toroidal current inside the
    surface, :math:`\mu_0` is the permeability of free space,
    :math:`2\pi\psi` is the toroidal flux, and :math:`(M,N)` are
    user-supplied integers that specify the desired helicity of
    symmetry. If the magnetic field is quasisymmetric, so
    :math:`B=B(\psi,\chi)` where :math:`\chi=M\vartheta - N\varphi`
    where :math:`(\vartheta,\varphi)` are the poloidal and toroidal
    Boozer angles, then :math:`\vec{B}\times\nabla B\cdot\nabla\psi
    \to -(MG+NI)(\vec{B}\cdot\nabla\varphi)\partial B/\partial \chi`
    and :math:`\vec{B}\cdot\nabla B \to (-N+\iota
    M)(\vec{B}\cdot\nabla\varphi)\partial B/\partial \chi`, implying
    the metric :math:`f` vanishes. The flux surface average is
    discretized using a uniform grid in the VMEC poloidal and toroidal
    angles :math:`(\theta,\phi)`. In this case :math:`f` can be
    written as a finite sum of squares:

    .. math::
        f = \sum_{s_j, \theta_j, \phi_j} R(s_j, \theta_k, \phi_{\ell})^2

    where the :math:`\phi_{\ell}` grid covers a single field period.
    Here, each residual term is

    .. math::
        R(\theta, \phi) = \sqrt{w_j \frac{n_{fp} \Delta_{\theta} \Delta_{\phi}}{V'}|\sqrt{g}|}
        \frac{1}{B^3} \left( (N-\iota M)\vec{B}\times\nabla B\cdot\nabla\psi - (MG+NI)\vec{B}\cdot\nabla B \right).

    Here, :math:`n_{fp}` is the number of field periods,
    :math:`\Delta_{\theta}` and :math:`\Delta_{\phi}` are the spacing
    of grid points in the poloidal and toroidal angles,
    :math:`\sqrt{g} = 1/(\nabla s\cdot\nabla\theta \times
    \nabla\phi)` is the Jacobian of the :math:`(s,\theta,\phi)`
    coordinates, and :math:`V' = \int_0^{2\pi} d\theta \int_0^{2\pi}d\phi |\sqrt{g}| = dV/d\psi`
    where :math:`V` is the volume enclosed by a flux surface.

    Args:
        vmec: A :obj:`simsopt.mhd.vmec.Vmec` object from which the
          quasisymmetry error will be calculated.
        s: Value of normalized toroidal flux at which you want the
          quasisymmetry error evaluated, or a list of values. Each
          value must be in the interval [0, 1], with 0 corresponding
          to the magnetic axis and 1 to the VMEC plasma boundary.
          This parameter corresponds to :math:`s_j` above.
        m: Desired poloidal mode number :math:`M` in the magnetic field
          strength :math:`B`, so
          :math:`B = B(s, M \vartheta - n_{fp} \hat{N} \varphi)`
          where :math:`\vartheta` and :math:`\varphi` are Boozer angles.
        n: Desired toroidal mode number :math:`\hat{N} = N / n_{fp}` in the magnetic field
          strength :math:`B`, so
          :math:`B = B(s, M \vartheta - n_{fp} \hat{N} \varphi)`
          where :math:`\vartheta` and :math:`\varphi` are Boozer angles.
          Note that the supplied value of ``n`` will be multiplied by
          the number of field periods :math:`n_{fp}`, so typically
          ``n`` should be +1 or -1 for quasi-helical symmetry.
        weights: The list of weights :math:`w_j` for each flux surface.
          If ``None``, a weight of :math:`w_j=1` will be used for
          all surfaces.
        ntheta: Number of grid points in :math:`\theta` used to
          discretize the flux surface average.
        nphi: Number of grid points per field period in :math:`\phi` used to
          discretize the flux surface average.
    """

    def __init__(self,
                 vmec: Vmec,
                 s: Union[float, RealArray],
                 m: int = 1,
                 n: int = 0,
                 weights: RealArray = None,
                 ntheta: int = 63,
                 nphi: int = 64) -> None:

        self.vmec = vmec
        self.depends_on = ["vmec"]
        self.ntheta = ntheta
        self.nphi = nphi
        self.m = m
        self.n = n

        # Make sure s is a list:
        try:
            self.s = list(s)
        except:
            self.s = [s]

        if weights is None:
            self.weights = np.ones(len(self.s))
        else:
            self.weights = weights
        assert len(self.weights) == len(self.s)

    def get_dofs(self):
        return np.array([])

    def set_dofs(self, x):
        self.need_to_run_code = True

    def compute(self):
        """
        Compute the quasisymmetry metric. This function returns an object
        that contains (as attributes) all the intermediate quantities
        for the calculation. Users do not need to call this function
        for optimization; instead the :func:`residuals()` function can be
        used. However, this function can be useful if users wish to
        inspect the quantities going into the calculation.
        """
        self.vmec.run()
        if self.vmec.wout.lasym:
            raise RuntimeError('Quasisymmetry class cannot yet handle non-stellarator-symmetric configs')

        logger.debug('Evaluating quasisymmetry residuals')
        ns = len(self.s)
        ntheta = self.ntheta
        nphi = self.nphi
        nfp = self.vmec.wout.nfp
        d_psi_d_s = -self.vmec.wout.phi[-1] / (2 * np.pi)

        # First, interpolate in s to get the quantities we need on the surfaces we need.
        method = 'linear'

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.iotas[1:], fill_value="extrapolate")
        iota = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bvco[1:], fill_value="extrapolate")
        G = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.buco[1:], fill_value="extrapolate")
        I = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.gmnc[:, 1:], fill_value="extrapolate")
        gmnc = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bmnc[:, 1:], fill_value="extrapolate")
        bmnc = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsubumnc[:, 1:], fill_value="extrapolate")
        bsubumnc = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsubvmnc[:, 1:], fill_value="extrapolate")
        bsubvmnc = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsupumnc[:, 1:], fill_value="extrapolate")
        bsupumnc = interp(self.s)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsupvmnc[:, 1:], fill_value="extrapolate")
        bsupvmnc = interp(self.s)

        theta1d = np.linspace(0, 2 * np.pi, ntheta, endpoint=False)
        phi1d = np.linspace(0, 2 * np.pi / nfp, nphi, endpoint=False)
        phi2d, theta2d = np.meshgrid(phi1d, theta1d)
        phi3d = phi2d.reshape((1, ntheta, nphi))
        theta3d = theta2d.reshape((1, ntheta, nphi))

        myshape = (ns, ntheta, nphi)
        modB = np.zeros(myshape)
        d_B_d_theta = np.zeros(myshape)
        d_B_d_phi = np.zeros(myshape)
        sqrtg = np.zeros(myshape)
        bsubu = np.zeros(myshape)
        bsubv = np.zeros(myshape)
        bsupu = np.zeros(myshape)
        bsupv = np.zeros(myshape)
        residuals3d = np.zeros(myshape)
        for jmn in range(len(self.vmec.wout.xm_nyq)):
            m = self.vmec.wout.xm_nyq[jmn]
            n = self.vmec.wout.xn_nyq[jmn]
            angle = m * theta3d - n * phi3d
            cosangle = np.cos(angle)
            sinangle = np.sin(angle)
            modB += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
            d_B_d_theta += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), -m * sinangle)
            d_B_d_phi += np.kron(bmnc[jmn, :].reshape((ns, 1, 1)), n * sinangle)
            sqrtg += np.kron(gmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
            bsubu += np.kron(bsubumnc[jmn, :].reshape((ns, 1, 1)), cosangle)
            bsubv += np.kron(bsubvmnc[jmn, :].reshape((ns, 1, 1)), cosangle)
            bsupu += np.kron(bsupumnc[jmn, :].reshape((ns, 1, 1)), cosangle)
            bsupv += np.kron(bsupvmnc[jmn, :].reshape((ns, 1, 1)), cosangle)

        B_dot_grad_B = bsupu * d_B_d_theta + bsupv * d_B_d_phi
        B_cross_grad_B_dot_grad_psi = d_psi_d_s * (bsubu * d_B_d_phi - bsubv * d_B_d_theta) / sqrtg

        dtheta = theta1d[1] - theta1d[0]
        dphi = phi1d[1] - phi1d[0]
        V_prime = nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))
        # Check that we can evaluate the flux surface average <1> and the result is 1:
        assert np.sum(np.abs(np.sqrt((1 / V_prime) * nfp * dtheta * dphi * np.sum(sqrtg, axis=(1, 2))) - 1)) < 1e-12

        nn = self.n * nfp
        for js in range(ns):
            residuals3d[js, :, :] = np.sqrt(self.weights[js] * nfp * dtheta * dphi / V_prime[js] * sqrtg[js, :, :]) \
                * (B_cross_grad_B_dot_grad_psi[js, :, :] * (nn - iota[js] * self.m) \
                   - B_dot_grad_B[js, :, :] * (self.m * G[js] + nn * I[js])) \
                / (modB[js, :, :] ** 3)

        residuals1d = residuals3d.reshape((ns * ntheta * nphi,))
        profile = np.sum(residuals3d * residuals3d, axis=(1, 2))
        total = np.sum(residuals1d * residuals1d)

        # Form a structure with all the intermediate data as attributes:
        results = Struct()
        variables = ['ns', 'ntheta', 'nphi', 'dtheta', 'dphi', 'nfp', 'V_prime', 'theta1d', 'phi1d',
                     'theta2d', 'phi2d', 'theta3d', 'phi3d', 'd_psi_d_s', 'B_dot_grad_B',
                     'B_cross_grad_B_dot_grad_psi', 'modB', 'd_B_d_theta', 'd_B_d_phi', 'sqrtg',
                     'bsubu', 'bsubv', 'bsupu', 'bsupv', 'G', 'I', 'iota',
                     'residuals3d', 'residuals1d', 'profile', 'total']
        for v in variables:
            results.__setattr__(v, eval(v))

        logger.debug('Done evaluating quasisymmetry residuals')
        return results

    def residuals(self):
        """
        Evaluate the quasisymmetry metric in terms of a 1D numpy vector of
        residuals, corresponding to :math:`R` in the documentation
        for this class. This is the function to use when forming a
        least-squares objective function.
        """
        results = self.compute()
        return results.residuals1d

    def profile(self):
        """
        Return the quasisymmetry metric in terms of a 1D radial
        profile. The residuals :math:`R` are squared and summed over
        theta and phi, but not over s. The total quasisymmetry error
        :math:`f` returned by the :func:`total()` function is the sum
        of the values in the profile returned by this function.
        """
        results = self.compute()
        return results.profile

    def total(self):
        """
        Evaluate the quasisymmetry metric in terms of the scalar total
        :math:`f`.
        """
        results = self.compute()
        return results.total

