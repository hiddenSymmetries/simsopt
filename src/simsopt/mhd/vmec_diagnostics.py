# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the LGPL License

"""
This module contains functions that can postprocess VMEC output.
"""

import logging
from typing import Union

import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.optimize import newton

from .vmec import Vmec
from .._core.util import Struct
from .._core.optimizable import Optimizable
from .._core.types import RealArray
from ..geo.surfaceobjectives import parameter_derivatives
from ..geo.surface import Surface
from ..geo.surfacerzfourier import SurfaceRZFourier

logger = logging.getLogger(__name__)

__all__ = ['QuasisymmetryRatioResidual', 'IotaTargetMetric', 'IotaWeighted',
           'WellWeighted']


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
        surfaces: Value of normalized toroidal flux at which you want the
          quasisymmetry error evaluated, or a list of values. Each
          value must be in the interval [0, 1], with 0 corresponding
          to the magnetic axis and 1 to the VMEC plasma boundary.
          This parameter corresponds to :math:`s_j` above.
        helicity_m: Desired poloidal mode number :math:`M` in the magnetic field
          strength :math:`B`, so
          :math:`B = B(s, M \vartheta - n_{fp} \hat{N} \varphi)`
          where :math:`\vartheta` and :math:`\varphi` are Boozer angles.
        helicity_n: Desired toroidal mode number :math:`\hat{N} = N / n_{fp}` in the magnetic field
          strength :math:`B`, so
          :math:`B = B(s, M \vartheta - n_{fp} \hat{N} \varphi)`
          where :math:`\vartheta` and :math:`\varphi` are Boozer angles.
          Note that the supplied value of ``helicity_n`` will be multiplied by
          the number of field periods :math:`n_{fp}`, so typically
          ``helicity_n`` should be +1 or -1 for quasi-helical symmetry.
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
                 surfaces: Union[float, RealArray],
                 helicity_m: int = 1,
                 helicity_n: int = 0,
                 weights: RealArray = None,
                 ntheta: int = 63,
                 nphi: int = 64) -> None:

        self.vmec = vmec
        #self.depends_on = ["vmec"]
        self.ntheta = ntheta
        self.nphi = nphi
        self.helicity_m = helicity_m
        self.helicity_n = helicity_n

        # Make sure surfaces is a list:
        try:
            self.surfaces = list(surfaces)
        except:
            self.surfaces = [surfaces]

        if weights is None:
            self.weights = np.ones(len(self.surfaces))
        else:
            self.weights = weights
        assert len(self.weights) == len(self.surfaces)
        super().__init__(depends_on=[vmec])

    # def recompute_bell(self, parent=None):
    #     self.need_to_run_code = True

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
        ns = len(self.surfaces)
        ntheta = self.ntheta
        nphi = self.nphi
        nfp = self.vmec.wout.nfp
        d_psi_d_s = -self.vmec.wout.phi[-1] / (2 * np.pi)

        # First, interpolate in s to get the quantities we need on the surfaces we need.
        method = 'linear'

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.iotas[1:], fill_value="extrapolate")
        iota = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bvco[1:], fill_value="extrapolate")
        G = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.buco[1:], fill_value="extrapolate")
        I = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.gmnc[:, 1:], fill_value="extrapolate")
        gmnc = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bmnc[:, 1:], fill_value="extrapolate")
        bmnc = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsubumnc[:, 1:], fill_value="extrapolate")
        bsubumnc = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsubvmnc[:, 1:], fill_value="extrapolate")
        bsubvmnc = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsupumnc[:, 1:], fill_value="extrapolate")
        bsupumnc = interp(self.surfaces)

        interp = interp1d(self.vmec.s_half_grid, self.vmec.wout.bsupvmnc[:, 1:], fill_value="extrapolate")
        bsupvmnc = interp(self.surfaces)

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

        nn = self.helicity_n * nfp
        for js in range(ns):
            residuals3d[js, :, :] = np.sqrt(self.weights[js] * nfp * dtheta * dphi / V_prime[js] * sqrtg[js, :, :]) \
                * (B_cross_grad_B_dot_grad_psi[js, :, :] * (nn - iota[js] * self.helicity_m) \
                   - B_dot_grad_B[js, :, :] * (self.helicity_m * G[js] + nn * I[js])) \
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


def B_cartesian(vmec,
                quadpoints_phi=None,
                quadpoints_theta=None,
                range=Surface.RANGE_FULL_TORUS,
                nphi=None,
                ntheta=None):
    r"""
    Computes Cartesian vector components of the magnetic field on the
    Vmec boundary.  The results are returned on a grid in the Vmec
    toroidal and poloidal angles. This routine is required to compute
    adjoint-based shape gradients and for the virtual casing
    calculation.

    There are two ways to define the grid points in the poloidal and
    toroidal angles on which the field is returned.  The default
    option, if ``quadpoints_phi``, ``quadpoints_theta``, ``nphi``, and
    ``ntheta`` are all unspecified, is to use the quadrature grid
    associated with the ``Surface`` object attached to
    ``vmec.boundary``.  The second option is that you can specify
    custom ``phi`` and ``theta`` grids using the arguments
    ``quadpoints_phi``, ``quadpoints_theta``, ``nphi``, ``ntheta``,
    and ``range``, exactly as when initializing a ``Surface`` object.
    For more details, see the documentation on :ref:`surfaces`.  Note
    that both angles go up to 1, not :math:`2\pi`.

    For now, this routine only works for stellarator symmetry.

    Args:
        vmec: instance of Vmec

    Returns:
        Tuple containing ``(Bx, By, Bz)``. Each of these three entries is a
        2D array of size ``(numquadpoints_phi, numquadpoints_theta)``
        containing the Cartesian component of the magnetic field on the Vmec boundary surface.
    """
    vmec.run()
    nfp = vmec.wout.nfp
    if vmec.wout.lasym:
        raise RuntimeError('B_cartesian presently only works for stellarator symmetry')

    if nphi is None and quadpoints_phi is None:
        phi1D_1 = vmec.boundary.quadpoints_phi
    elif quadpoints_phi is None:
        phi1D_1 = Surface.get_phi_quadpoints(range=range, nphi=nphi, nfp=vmec.wout.nfp)
    else:
        phi1D_1 = quadpoints_phi

    if ntheta is None and quadpoints_theta is None:
        theta1D_1 = vmec.boundary.quadpoints_theta
    elif quadpoints_theta is None:
        theta1D_1 = Surface.get_theta_quadpoints(ntheta=ntheta)
    else:
        theta1D_1 = quadpoints_theta

    theta1D = np.array(theta1D_1) * 2 * np.pi
    phi1D = np.array(phi1D_1) * 2 * np.pi

    theta, phi = np.meshgrid(theta1D, phi1D)

    # Get the tangent vectors using the gammadash1/2 functions from SurfaceRZFourier:
    surf = SurfaceRZFourier(mpol=vmec.wout.mpol, ntor=vmec.wout.ntor, nfp=vmec.wout.nfp,
                            quadpoints_phi=phi1D_1, quadpoints_theta=theta1D_1)
    for jmn in np.arange(vmec.wout.mnmax):
        surf.set_rc(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.rmnc[jmn, -1])
        surf.set_zs(int(vmec.wout.xm[jmn]), int(vmec.wout.xn[jmn] / nfp), vmec.wout.zmns[jmn, -1])
    dgamma1 = surf.gammadash1()
    dgamma2 = surf.gammadash2()

    bsupumnc = 1.5 * vmec.wout.bsupumnc[:, -1] - 0.5 * vmec.wout.bsupumnc[:, -2]
    bsupvmnc = 1.5 * vmec.wout.bsupvmnc[:, -1] - 0.5 * vmec.wout.bsupvmnc[:, -2]
    angle = vmec.wout.xm_nyq[:, None, None] * theta[None, :, :] \
        - vmec.wout.xn_nyq[:, None, None] * phi[None, :, :]
    Bsupu = np.sum(bsupumnc[:, None, None] * np.cos(angle), axis=0)
    Bsupv = np.sum(bsupvmnc[:, None, None] * np.cos(angle), axis=0)

    Bx = (Bsupv * dgamma1[:, :, 0] + Bsupu * dgamma2[:, :, 0])/(2*np.pi)
    By = (Bsupv * dgamma1[:, :, 1] + Bsupu * dgamma2[:, :, 1])/(2*np.pi)
    Bz = (Bsupv * dgamma1[:, :, 2] + Bsupu * dgamma2[:, :, 2])/(2*np.pi)

    return Bx, By, Bz


class IotaTargetMetric(Optimizable):
    r"""
    IotaTargetMetric is a class that computes a metric quantifying the
    deviation of the rotational transform :math:`\iota` in from a
    prescribed target profile in a Vmec equilibrium:

    .. math::
        J = \frac{1}{2} \int ds \, (\iota - \iota_{target})^2

    where the integral is over the normalized toroidal flux :math:`s`,
    and the function :math:`\iota_{target}(s)` corresponds to the
    argument ``iota_target``. This class also can compute the
    derivatives of :math:`J` using an adjoint method.

    Args:
        vmec : instance of Vmec
        iota_target : function handle which takes a single argument, s,
            the normalized toroidal flux, and returns the target rotational
            transform.
        adjoint_epsilon : sets the amplitude of the toroidal
            current perturbation required for the adjoint solve.
    """

    def __init__(self, vmec, iota_target, adjoint_epsilon=1.e-1):
        self.vmec = vmec
        self.boundary = vmec.boundary
        self.iota_target = iota_target
        self.adjoint_epsilon = adjoint_epsilon
        super().__init__(depends_on=[vmec])

    def J(self):
        """
        Computes the quantity :math:`J` described in the class definition.
        """
        # if self.vmec.runnable:
        #     self.vmec.need_to_run_code = True
        self.vmec.run()
        return 0.5 * np.sum((self.vmec.wout.iotas[1::]
                             - self.iota_target(self.vmec.s_half_grid))**2) * self.vmec.ds

    def dJ(self):
        """
        Computes derivatives of :math:`J` with respect to surface
        parameters using an adjoint method.
        """
        if self.vmec.indata.ncurr != 1:
            raise RuntimeError('''dJ cannot be computed without
                running vmec with ncurr = 1''')

        shape_gradient = self.shape_gradient()
        return parameter_derivatives(self.vmec.boundary, shape_gradient)

    def shape_gradient(self):
        r"""
        Computes the shape gradient of the quantity :math:`J` described in
        the class definition.  For a perturbation to the surface
        :math:`\delta \vec{x}`, the resulting perturbation to the
        objective function is

        .. math::
          \delta J(\delta \vec{x}) = \int d^2 x \, G \delta \vec{x} \cdot \vec{n}

        where the integral is over the VMEC boundary surface,
        :math:`G` is the shape gradient, and :math:`\vec{n}` is the
        unit normal.

        Returns:
            :math:`G` : 2d array of size (numquadpoints_phi,numquadpoints_theta)
        """
        self.vmec.run()

        Bx0, By0, Bz0 = B_cartesian(self.vmec)

        mu0 = 4*np.pi*1e-7
        It_half = self.vmec.wout.signgs * 2*np.pi * self.vmec.wout.bsubumnc[0, 1::] / mu0
        ac_aux_f_prev = np.copy(self.vmec.indata.ac_aux_f)
        ac_aux_s_prev = np.copy(self.vmec.indata.ac_aux_s)
        pcurr_type_prev = np.copy(self.vmec.indata.pcurr_type)
        curtor_prev = np.copy(self.vmec.indata.curtor)

        perturbation = (self.vmec.wout.iotas[1::]-self.iota_target(self.vmec.s_half_grid)) \
            / (self.vmec.wout.phi[-1]*self.vmec.wout.signgs/(2*np.pi))

        # Perturbed toroidal current profile
        It_new = It_half + self.adjoint_epsilon*perturbation
        curtor = 1.5*It_new[-1] - 0.5*It_new[-2]
        self.vmec.indata.ac_aux_f = -1.*np.ones_like(self.vmec.indata.ac_aux_f)
        self.vmec.indata.ac_aux_s = -1.*np.ones_like(self.vmec.indata.ac_aux_s)
        self.vmec.indata.ac_aux_f[0:self.vmec.wout.ns-1] = It_new
        self.vmec.indata.ac_aux_s[0:self.vmec.wout.ns-1] = self.vmec.s_half_grid
        self.vmec.indata.curtor = curtor
        self.vmec.indata.pcurr_type = b'line_segment_I'
        self.vmec.need_to_run_code = True

        self.vmec.run()

        It_half = self.vmec.wout.signgs * 2*np.pi * self.vmec.wout.bsubumnc[0, 1::] / mu0

        Bx, By, Bz = B_cartesian(self.vmec)

        # Reset input values
        self.vmec.indata.ac_aux_f = ac_aux_f_prev
        self.vmec.indata.ac_aux_s = ac_aux_s_prev
        self.vmec.indata.pcurr_type = pcurr_type_prev
        self.vmec.indata.curtor = curtor_prev
        self.vmec.need_to_run_code = True

        deltaB_dot_B = ((Bx-Bx0)*Bx0 + (By-By0)*By0 + (Bz-Bz0)*Bz0)/self.adjoint_epsilon

        return deltaB_dot_B/(2*np.pi*mu0)


class IotaWeighted(Optimizable):
    r"""
    Computes a weighted average of the rotational transform for a VMEC
    configuration.  The quantity computed is defined by

    .. math::
        J = \frac{ \int ds \, \iota(s) w(s)}
                 { \int ds \, w(s)}

    where :math:`w(s)` is a prescribed weight function, corresponding
    to the argument ``weight_function``. This class also can compute the
    derivatives of :math:`J` using an adjoint method.

    Args:
        vmec : instance of Vmec
        weight_function : function handle which takes a single argument, s,
            the normalized toroidal flux
        adjoint_epsilon : sets the amplitude of the toroidal
            current perturbation required for the adjoint solve.
    """

    def __init__(self, vmec, weight_function, adjoint_epsilon=1.e-1):
        self.vmec = vmec
        self.boundary = vmec.boundary
        self.weight_function = weight_function
        self.adjoint_epsilon = adjoint_epsilon
        # self.depends_on = ["boundary"]
        super().__init__(depends_on=[vmec])

    def J(self):
        """
        Computes the quantity :math:`J` described in the class definition.
        """
        self.vmec.run()
        return np.sum(self.weight_function(self.vmec.s_half_grid) * self.vmec.wout.iotas[1:]) \
            / np.sum(self.weight_function(self.vmec.s_half_grid))

    def dJ(self):
        """
        Computes derivatives of :math:`J` with respect to surface
        parameters using an adjoint method.
        """
        if self.vmec.indata.ncurr != 1:
            raise RuntimeError('''dJ cannot be computed without
                running vmec with ncurr = 1''')

        shape_gradient = self.shape_gradient()
        return parameter_derivatives(self.vmec.boundary, shape_gradient)

    def shape_gradient(self):
        r"""
        Computes the shape gradient of the quantity :math:`J` described in
        the class definition.  For a perturbation to the surface
        :math:`\delta \vec{x}`, the resulting perturbation to the
        objective function is

        .. math::
          \delta J(\delta \vec{x}) = \int d^2 x \, G \delta \vec{x} \cdot \vec{n}

        where the integral is over the VMEC boundary surface,
        :math:`G` is the shape gradient, and :math:`\vec{n}` is the
        unit normal.

        Returns:
            :math:`G` : 2d array of size (numquadpoints_phi,numquadpoints_theta)
        """
        self.vmec.run()

        Bx0, By0, Bz0 = B_cartesian(self.vmec)

        mu0 = 4*np.pi*1e-7
        It_half = self.vmec.wout.signgs * 2*np.pi * self.vmec.wout.bsubumnc[0, 1::] / mu0
        ac_aux_f_prev = np.copy(self.vmec.indata.ac_aux_f)
        ac_aux_s_prev = np.copy(self.vmec.indata.ac_aux_s)
        pcurr_type_prev = np.copy(self.vmec.indata.pcurr_type)
        curtor_prev = np.copy(self.vmec.indata.curtor)

        perturbation = self.weight_function(self.vmec.s_half_grid)

        # Perturbed toroidal current profile
        It_new = It_half + self.adjoint_epsilon*perturbation
        curtor = 1.5*It_new[-1] - 0.5*It_new[-2]
        self.vmec.indata.ac_aux_f = -1.*np.ones_like(self.vmec.indata.ac_aux_f)
        self.vmec.indata.ac_aux_s = -1.*np.ones_like(self.vmec.indata.ac_aux_s)
        self.vmec.indata.ac_aux_f[0:self.vmec.wout.ns-1] = It_new
        self.vmec.indata.ac_aux_s[0:self.vmec.wout.ns-1] = self.vmec.s_half_grid
        self.vmec.indata.curtor = curtor
        self.vmec.indata.pcurr_type = b'line_segment_I'
        self.vmec.need_to_run_code = True

        self.vmec.run()

        It_half = self.vmec.wout.signgs * 2*np.pi * self.vmec.wout.bsubumnc[0, 1::] / mu0

        Bx, By, Bz = B_cartesian(self.vmec)

        # Reset input values
        self.vmec.indata.ac_aux_f = ac_aux_f_prev
        self.vmec.indata.ac_aux_s = ac_aux_s_prev
        self.vmec.indata.pcurr_type = pcurr_type_prev
        self.vmec.indata.curtor = curtor_prev
        self.vmec.need_to_run_code = True

        deltaB_dot_B = ((Bx-Bx0)*Bx0 + (By-By0)*By0 + (Bz-Bz0)*Bz0)/self.adjoint_epsilon

        return deltaB_dot_B/(mu0*self.vmec.ds*self.vmec.wout.phi[-1]*self.vmec.wout.signgs*np.sum(self.weight_function(self.vmec.s_half_grid)))


class WellWeighted(Optimizable):
    r"""
    WellWeighted is a class that computes a measure of magnetic well
    for a vmec equilibrium. The magnetic well measure is

    .. math::
        J = \frac{ \int ds \, V'(s) [w_1(s) - w_2(s)]}
            { \int ds \, V'(s) [w_1(s) + w_2(s)]},

    where :math:`w_1(s)` and :math:`w_2(s)` correspond to the
    arguments ``weight_function1`` and ``weight_function2``, and
    :math:`V(s)` is the volume enclosed by the flux surface with
    normalized toroidal flux :math:`s`.  Typically, :math:`w_1` would
    be peaked on the edge while :math:`w_2` would be peaked on the
    axis, such that :math:`J < 0` corresonds to :math:`V''(s) < 0`,
    which is favorable for stability.

    This class also provides calculations of the derivatives of
    :math:`J` using an adjoint method.

    Args:
        vmec : instance of Vmec
        weight_function1 : function handle which takes a single argument, s,
            the normalized toroidal flux
        weight_function2 : function handle which takes a single argument, s,
            the normalized toroidal flux
        adjoint_epsilon : sets the amplitude of the toroidal
            current perturbation required for the adjoint solve.
    """

    def __init__(self, vmec, weight_function1, weight_function2, adjoint_epsilon=1.e-1):
        self.vmec = vmec
        self.boundary = vmec.boundary
        self.weight_function1 = weight_function1
        self.weight_function2 = weight_function2
        self.adjoint_epsilon = adjoint_epsilon
        # self.depends_on = ["boundary"]
        super().__init__(depends_on=[vmec])

    def J(self):
        """
        Computes the quantity :math:`J` described in the class definition.
        """
        self.vmec.run()
        return np.sum((self.weight_function1(self.vmec.s_half_grid)-self.weight_function2(self.vmec.s_half_grid)) * self.vmec.wout.vp[1:]) \
            / np.sum((self.weight_function1(self.vmec.s_half_grid)+self.weight_function2(self.vmec.s_half_grid)) * self.vmec.wout.vp[1:])

    def dJ(self):
        """
        Computes derivatives of :math:`J` with respect to surface
        parameters using an adjoint method.
        """

        self.vmec.need_to_run_code = True
        shape_gradient = self.shape_gradient()
        return parameter_derivatives(self.vmec.boundary, shape_gradient)

    def shape_gradient(self):
        r"""
        Computes the shape gradient of the quantity :math:`J` described in
        the class definition.  For a perturbation to the surface
        :math:`\delta \vec{x}`, the resulting perturbation to the
        objective function is

        .. math::
          \delta J(\delta \vec{x}) = \int d^2 x \, G \delta \vec{x} \cdot \vec{n}

        where the integral is over the VMEC boundary surface,
        :math:`G` is the shape gradient, and :math:`\vec{n}` is the
        unit normal.

        Returns:
            :math:`G` : 2d array of size (numquadpoints_phi,numquadpoints_theta)
        """
        self.vmec.run()

        Bx0, By0, Bz0 = B_cartesian(self.vmec)

        mu0 = 4*np.pi*1e-7
        am_aux_f_prev = np.copy(self.vmec.indata.am_aux_f)
        am_aux_s_prev = np.copy(self.vmec.indata.am_aux_s)
        pmass_type_prev = np.copy(self.vmec.indata.pmass_type)

        pres = self.vmec.wout.pres[1::]
        weight1 = self.weight_function1(self.vmec.s_half_grid) - self.weight_function2(self.vmec.s_half_grid)
        weight2 = self.weight_function1(self.vmec.s_half_grid) + self.weight_function2(self.vmec.s_half_grid)
        numerator = np.sum(weight1 * self.vmec.wout.vp[1::])
        denominator = np.sum(weight2 * self.vmec.wout.vp[1::])
        fW = numerator/denominator
        perturbation = (weight1 - fW * weight2) / (denominator * self.vmec.ds * 4 * np.pi * np.pi)

        # Perturbed pressure profile
        pres_new = pres + self.adjoint_epsilon*perturbation

        self.vmec.indata.am_aux_f = -1.*np.ones_like(self.vmec.indata.am_aux_f)
        self.vmec.indata.am_aux_s = -1.*np.ones_like(self.vmec.indata.am_aux_s)
        self.vmec.indata.am_aux_f[0:self.vmec.wout.ns-1] = pres_new
        self.vmec.indata.am_aux_s[0:self.vmec.wout.ns-1] = self.vmec.s_half_grid
        self.vmec.indata.pmass_type = b'cubic_spline'
        self.vmec.need_to_run_code = True

        self.vmec.run()

        Bx, By, Bz = B_cartesian(self.vmec)

        # Reset input values
        self.vmec.indata.am_aux_f = am_aux_f_prev
        self.vmec.indata.am_aux_s = am_aux_s_prev
        self.vmec.indata.pmass_type = pmass_type_prev
        self.vmec.need_to_run_code = True

        deltaB_dot_B = ((Bx-Bx0)*Bx0 + (By-By0)*By0 + (Bz-Bz0)*Bz0)/self.adjoint_epsilon

        return deltaB_dot_B/(mu0) + perturbation[-1]


def vmec_splines(vmec):
    """
    Initialize radial splines for a VMEC equilibrium.

    Args:
        vmec: An instance of :obj:`simsopt.mhd.vmec.Vmec`.

    Returns:
        A structure with the splines as attributes.
    """
    vmec.run()
    results = Struct()

    rmnc = []
    zmns = []
    lmns = []
    d_rmnc_d_s = []
    d_zmns_d_s = []
    d_lmns_d_s = []
    for jmn in range(vmec.wout.mnmax):
        rmnc.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.rmnc[jmn, :]))
        zmns.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.zmns[jmn, :]))
        lmns.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.lmns[jmn, 1:]))
        d_rmnc_d_s.append(rmnc[-1].derivative())
        d_zmns_d_s.append(zmns[-1].derivative())
        d_lmns_d_s.append(lmns[-1].derivative())

    gmnc = []
    bmnc = []
    bsupumnc = []
    bsupvmnc = []
    bsubsmns = []
    bsubumnc = []
    bsubvmnc = []
    d_bmnc_d_s = []
    d_bsupumnc_d_s = []
    d_bsupvmnc_d_s = []
    for jmn in range(vmec.wout.mnmax_nyq):
        gmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.gmnc[jmn, 1:]))
        bmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bmnc[jmn, 1:]))
        bsupumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupumnc[jmn, 1:]))
        bsupvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsupvmnc[jmn, 1:]))
        # Note that bsubsmns is on the full mesh, unlike the other components:
        bsubsmns.append(InterpolatedUnivariateSpline(vmec.s_full_grid, vmec.wout.bsubsmns[jmn, :]))
        bsubumnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubumnc[jmn, 1:]))
        bsubvmnc.append(InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.bsubvmnc[jmn, 1:]))
        d_bmnc_d_s.append(bmnc[-1].derivative())
        d_bsupumnc_d_s.append(bsupumnc[-1].derivative())
        d_bsupvmnc_d_s.append(bsupvmnc[-1].derivative())

    # Handle 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.pres[1:])
    results.d_pressure_d_s = results.pressure.derivative()
    results.iota = InterpolatedUnivariateSpline(vmec.s_half_grid, vmec.wout.iotas[1:])
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = vmec.wout.phi[-1]
    variables = ['Aminor_p', 'mnmax', 'xm', 'xn', 'mnmax_nyq', 'xm_nyq', 'xn_nyq', 'nfp']
    for v in variables:
        results.__setattr__(v, eval('vmec.wout.' + v))

    variables = ['rmnc', 'zmns', 'lmns', 'd_rmnc_d_s', 'd_zmns_d_s', 'd_lmns_d_s',
                 'gmnc', 'bmnc', 'd_bmnc_d_s', 'bsupumnc', 'bsupvmnc', 'd_bsupumnc_d_s', 'd_bsupvmnc_d_s',
                 'bsubsmns', 'bsubumnc', 'bsubvmnc']
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


def vmec_fieldlines(vs, s, alpha, theta1d=None, phi1d=None, phi_center=0, plot=False, show=True):
    r"""
    Compute field lines in a vmec configuration, and compute many
    geometric quantities of interest along the field lines. In
    particular, this routine computes the geometric quantities that
    enter the gyrokinetic equation.

    One of the tasks performed by this function is to convert between
    the poloidal angles :math:`\theta_{vmec}` and
    :math:`\theta_{pest}`. The latter is the angle in which the field
    lines are straight when used in combination with the standard
    toroidal angle :math:`\phi`. Note that all angles in this function
    have period :math:`2\pi`, not period 1.

    For the inputs and outputs of this function, a field line label
    coordinate is defined by

    .. math::

        \alpha = \theta_{pest} - \iota (\phi - \phi_{center}).

    Here, :math:`\phi_{center}` is a constant, usually 0, which can be
    set to a nonzero value if desired so the magnetic shear
    contribution to :math:`\nabla\alpha` vanishes at a toroidal angle
    different than 0.  Also, wherever the term ``psi`` appears in
    variable names in this function and the returned arrays, it means
    :math:`\psi =` the toroidal flux divided by :math:`2\pi`, so

    .. math::

        \vec{B} = \nabla\psi\times\nabla\theta_{pest} + \iota\nabla\phi\times\nabla\psi = \nabla\psi\times\nabla\alpha.

    To specify the parallel extent of the field lines, you can provide
    either a grid of :math:`\theta_{pest}` values or a grid of
    :math:`\phi` values. If you specify both or neither, ``ValueError``
    will be raised.

    Most of the arrays that are computed have shape ``(ns, nalpha,
    nl)``, where ``ns`` is the number of flux surfaces, ``nalpha`` is the
    number of field lines on each flux surface, and ``nl`` is the number
    of grid points along each field line. In other words, ``ns`` is the
    size of the input ``s`` array, ``nalpha`` is the size of the input
    ``alpha`` array, and ``nl`` is the size of the input ``theta1d`` or
    ``phi1d`` array. The output arrays are returned as attributes of the
    returned object. Many intermediate quantities are included, such
    as the Cartesian components of the covariant and contravariant
    basis vectors. Some of the most useful of these output arrays are (all with SI units):

    - ``phi``: The standard toroidal angle :math:`\phi`.
    - ``theta_vmec``: VMEC's poloidal angle :math:`\theta_{vmec}`.
    - ``theta_pest``: The straight-field-line angle :math:`\theta_{pest}` associated with :math:`\phi`.
    - ``modB``: The magnetic field magnitude :math:`|B|`.
    - ``B_sup_theta_vmec``: :math:`\vec{B}\cdot\nabla\theta_{vmec}`.
    - ``B_sup_phi``: :math:`\vec{B}\cdot\nabla\phi`.
    - ``B_cross_grad_B_dot_grad_alpha``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\alpha`.
    - ``B_cross_grad_B_dot_grad_psi``: :math:`\vec{B}\times\nabla|B|\cdot\nabla\psi`.
    - ``B_cross_kappa_dot_grad_alpha``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\alpha`,
      where :math:`\vec{\kappa}=\vec{b}\cdot\nabla\vec{b}` is the curvature and :math:`\vec{b}=|B|^{-1}\vec{B}`.
    - ``B_cross_kappa_dot_grad_psi``: :math:`\vec{B}\times\vec{\kappa}\cdot\nabla\psi`.
    - ``grad_alpha_dot_grad_alpha``: :math:`|\nabla\alpha|^2 = \nabla\alpha\cdot\nabla\alpha`.
    - ``grad_alpha_dot_grad_psi``: :math:`\nabla\alpha\cdot\nabla\psi`.
    - ``grad_psi_dot_grad_psi``: :math:`|\nabla\psi|^2 = \nabla\psi\cdot\nabla\psi`.
    - ``iota``: The rotational transform :math:`\iota`. This array has shape ``(ns,)``.
    - ``shat``: The magnetic shear :math:`\hat s= (x/q) (d q / d x)` where 
      :math:`x = \mathrm{Aminor_p} \, \sqrt{s}` and :math:`q=1/\iota`. This array has shape ``(ns,)``.

    The following normalized versions of these quantities used in the
    gyrokinetic codes ``stella``, ``gs2``, and ``GX`` are also
    returned: ``bmag``, ``gbdrift``, ``gbdrift0``, ``cvdrift``,
    ``cvdrift0``, ``gds2``, ``gds21``, and ``gds22``, along with
    ``L_reference`` and ``B_reference``.  Instead of ``gradpar``, two
    variants are returned, ``gradpar_theta_pest`` and ``gradpar_phi``,
    corresponding to choosing either :math:`\theta_{pest}` or
    :math:`\phi` as the parallel coordinate.

    The value(s) of ``s`` provided as input need not coincide with the
    full grid or half grid in VMEC, as spline interpolation will be
    used radially.

    The implementation in this routine is similar to the one in the
    gyrokinetic code ``stella``.

    Example usage::

        import numpy as np
        from simsopt.mhd.vmec import Vmec
        from simsopt.mhd.vmec_diagnostics import vmec_fieldlines

        v = Vmec('wout_li383_1.4m.nc')
        theta = np.linspace(-np.pi, np.pi, 50)
        fl = vmec_fieldlines(v, 0.5, 0, theta1d=theta)
        print(fl.B_cross_grad_B_dot_grad_alpha)

    Args:
        vs: Either an instance of :obj:`simsopt.mhd.vmec.Vmec`
          or the structure returned by :func:`vmec_splines`.
        s: Values of normalized toroidal flux on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        alpha: Values of the field line label :math:`\alpha` on which to construct the field lines.
          You can give a single number, or a list or numpy array.
        theta1d: 1D array of :math:`\theta_{pest}` values, setting the grid points
          along the field line and the parallel extent of the field line.
        phi1d: 1D array of :math:`\phi` values, setting the grid points along the
          field line and the parallel extent of the field line.
        phi_center: :math:`\phi_{center}`, an optional shift to the toroidal angle
          in the definition of :math:`\alpha`.
        plot: Whether to create a plot of the main geometric quantities. Only one field line will
          be plotted, corresponding to the leading elements of ``s`` and ``alpha``.
        show: Only matters if ``plot==True``. Whether to call matplotlib's ``show()`` function
          after creating the plot.
    """
    # If given a Vmec object, convert it to vmec_splines:
    if isinstance(vs, Vmec):
        vs = vmec_splines(vs)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError('You cannot specify both theta and phi')
    if (theta1d is None) and (phi1d is None):
        raise ValueError('You must specify either theta or phi')
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Shorthand:
    mnmax = vs.mnmax
    xm = vs.xm
    xn = vs.xn
    mnmax_nyq = vs.mnmax_nyq
    xm_nyq = vs.xm_nyq
    xn_nyq = vs.xn_nyq

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    # shat = (r/q)(dq/dr) where r = a sqrt(s)
    #      = - (r/iota) (d iota / d r) = -2 (s/iota) (d iota / d s)
    shat = (-2 * s / iota) * d_iota_d_s

    rmnc = np.zeros((ns, mnmax))
    zmns = np.zeros((ns, mnmax))
    lmns = np.zeros((ns, mnmax))
    d_rmnc_d_s = np.zeros((ns, mnmax))
    d_zmns_d_s = np.zeros((ns, mnmax))
    d_lmns_d_s = np.zeros((ns, mnmax))

    ######## CAREFUL!!###########################################################
    # Everything here and in vmec_splines is designed for up-down symmetric eqlbia
    # When we start optimizing equilibria with lasym = "True"
    # we should edit this as well as vmec_splines 
    lmnc = np.zeros((ns, mnmax))
    lasym = False

    for jmn in range(mnmax):
        rmnc[:, jmn] = vs.rmnc[jmn](s)
        zmns[:, jmn] = vs.zmns[jmn](s)
        lmns[:, jmn] = vs.lmns[jmn](s)
        d_rmnc_d_s[:, jmn] = vs.d_rmnc_d_s[jmn](s)
        d_zmns_d_s[:, jmn] = vs.d_zmns_d_s[jmn](s)
        d_lmns_d_s[:, jmn] = vs.d_lmns_d_s[jmn](s)

    gmnc = np.zeros((ns, mnmax_nyq))
    bmnc = np.zeros((ns, mnmax_nyq))
    d_bmnc_d_s = np.zeros((ns, mnmax_nyq))
    bsupumnc = np.zeros((ns, mnmax_nyq))
    bsupvmnc = np.zeros((ns, mnmax_nyq))
    bsubsmns = np.zeros((ns, mnmax_nyq))
    bsubumnc = np.zeros((ns, mnmax_nyq))
    bsubvmnc = np.zeros((ns, mnmax_nyq))
    for jmn in range(mnmax_nyq):
        gmnc[:, jmn] = vs.gmnc[jmn](s)
        bmnc[:, jmn] = vs.bmnc[jmn](s)
        d_bmnc_d_s[:, jmn] = vs.d_bmnc_d_s[jmn](s)
        bsupumnc[:, jmn] = vs.bsupumnc[jmn](s)
        bsupvmnc[:, jmn] = vs.bsupvmnc[jmn](s)
        bsubsmns[:, jmn] = vs.bsubsmns[jmn](s)
        bsubumnc[:, jmn] = vs.bsubumnc[jmn](s)
        bsubvmnc[:, jmn] = vs.bsubvmnc[jmn](s)

    theta_pest = np.zeros((ns, nalpha, nl))
    phi = np.zeros((ns, nalpha, nl))

    if theta1d is None:
        # We are given phi. Compute theta_pest:
        for js in range(ns):
            phi[js, :, :] = phi1d[None, :]
            theta_pest[js, :, :] = alpha[:, None] + iota[js] * (phi1d[None, :] - phi_center)
    else:
        # We are given theta_pest. Compute phi:
        for js in range(ns):
            theta_pest[js, :, :] = theta1d[None, :]
            phi[js, :, :] = phi_center + (theta1d[None, :] - alpha[:, None]) / iota[js]

    def residual(theta_v, phi0, theta_p_target, jradius):
        """
        This function is used for computing an array of values of theta_vmec that
        give a desired theta_pest array.
        """
        return theta_p_target - (theta_v + np.sum(lmns[js, :, None] * np.sin(xm[:, None] * theta_v - xn[:, None] * phi0), axis=0))

    theta_vmec = np.zeros((ns, nalpha, nl))
    for js in range(ns):
        for jalpha in range(nalpha):
            theta_guess = theta_pest[js, jalpha, :]
            solution = newton(
                residual,
                x0=theta_guess,
                x1=theta_guess + 0.1,
                args=(phi[js, jalpha, :], theta_pest[js, jalpha, :], js),
            )
            theta_vmec[js, jalpha, :] = solution

    # Now that we know theta_vmec, compute all the geometric quantities
    angle = xm[:, None, None, None] * theta_vmec[None, :, :, :] - xn[:, None, None, None] * phi[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm[:, None, None, None] * cosangle
    ncosangle = xn[:, None, None, None] * cosangle
    msinangle = xm[:, None, None, None] * sinangle
    nsinangle = xn[:, None, None, None] * sinangle
    # Order of indices in cosangle and sinangle: mn, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn
    R = np.einsum('ij,jikl->ikl', rmnc, cosangle)
    d_R_d_s = np.einsum('ij,jikl->ikl', d_rmnc_d_s, cosangle)
    d_R_d_theta_vmec = -np.einsum('ij,jikl->ikl', rmnc, msinangle)
    d_R_d_phi = np.einsum('ij,jikl->ikl', rmnc, nsinangle)

    Z = np.einsum('ij,jikl->ikl', zmns, sinangle)
    d_Z_d_s = np.einsum('ij,jikl->ikl', d_zmns_d_s, sinangle)
    d_Z_d_theta_vmec = np.einsum('ij,jikl->ikl', zmns, mcosangle)
    d_Z_d_phi = -np.einsum('ij,jikl->ikl', zmns, ncosangle)

    d_lambda_d_s = np.einsum('ij,jikl->ikl', d_lmns_d_s, sinangle)
    d_lambda_d_theta_vmec = np.einsum('ij,jikl->ikl', lmns, mcosangle)
    d_lambda_d_phi = -np.einsum('ij,jikl->ikl', lmns, ncosangle)

    # Now handle the Nyquist quantities:
    angle = xm_nyq[:, None, None, None] * theta_vmec[None, :, :, :] - xn_nyq[:, None, None, None] * phi[None, :, :, :]
    cosangle = np.cos(angle)
    sinangle = np.sin(angle)
    mcosangle = xm_nyq[:, None, None, None] * cosangle
    ncosangle = xn_nyq[:, None, None, None] * cosangle
    msinangle = xm_nyq[:, None, None, None] * sinangle
    nsinangle = xn_nyq[:, None, None, None] * sinangle

    sqrt_g_vmec = np.einsum('ij,jikl->ikl', gmnc, cosangle)
    modB = np.einsum('ij,jikl->ikl', bmnc, cosangle)
    d_B_d_s = np.einsum('ij,jikl->ikl', d_bmnc_d_s, cosangle)
    d_B_d_theta_vmec = -np.einsum('ij,jikl->ikl', bmnc, msinangle)
    d_B_d_phi = np.einsum('ij,jikl->ikl', bmnc, nsinangle)

    B_sup_theta_vmec = np.einsum('ij,jikl->ikl', bsupumnc, cosangle)
    B_sup_phi = np.einsum('ij,jikl->ikl', bsupvmnc, cosangle)
    B_sub_s = np.einsum('ij,jikl->ikl', bsubsmns, sinangle)
    B_sub_theta_vmec = np.einsum('ij,jikl->ikl', bsubumnc, cosangle)
    B_sub_phi = np.einsum('ij,jikl->ikl', bsubvmnc, cosangle)
    B_sup_theta_pest = iota[:, None, None] * B_sup_phi

    sqrt_g_vmec_alt = R * (d_Z_d_s * d_R_d_theta_vmec - d_R_d_s * d_Z_d_theta_vmec)

    # Note the minus sign. psi in the straight-field-line relation seems to have opposite sign to vmec's phi array.
    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # *********************************************************************
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    # X = R * cos(phi):   
    d_X_d_theta_vmec = d_R_d_theta_vmec * cosphi
    d_X_d_phi = d_R_d_phi * cosphi - R * sinphi
    d_X_d_s = d_R_d_s * cosphi
    # Y = R * sin(phi):
    d_Y_d_theta_vmec = d_R_d_theta_vmec * sinphi
    d_Y_d_phi = d_R_d_phi * sinphi + R * cosphi
    d_Y_d_s = d_R_d_s * sinphi

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_s_X = (d_Y_d_theta_vmec * d_Z_d_phi - d_Z_d_theta_vmec * d_Y_d_phi) / sqrt_g_vmec
    grad_s_Y = (d_Z_d_theta_vmec * d_X_d_phi - d_X_d_theta_vmec * d_Z_d_phi) / sqrt_g_vmec
    grad_s_Z = (d_X_d_theta_vmec * d_Y_d_phi - d_Y_d_theta_vmec * d_X_d_phi) / sqrt_g_vmec

    grad_theta_vmec_X = (d_Y_d_phi * d_Z_d_s - d_Z_d_phi * d_Y_d_s) / sqrt_g_vmec
    grad_theta_vmec_Y = (d_Z_d_phi * d_X_d_s - d_X_d_phi * d_Z_d_s) / sqrt_g_vmec
    grad_theta_vmec_Z = (d_X_d_phi * d_Y_d_s - d_Y_d_phi * d_X_d_s) / sqrt_g_vmec

    grad_phi_X = (d_Y_d_s * d_Z_d_theta_vmec - d_Z_d_s * d_Y_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Y = (d_Z_d_s * d_X_d_theta_vmec - d_X_d_s * d_Z_d_theta_vmec) / sqrt_g_vmec
    grad_phi_Z = (d_X_d_s * d_Y_d_theta_vmec - d_Y_d_s * d_X_d_theta_vmec) / sqrt_g_vmec
    # End of dual relations.

    # *********************************************************************
    # Compute the Cartesian components of other quantities we need:
    # *********************************************************************

    grad_psi_X = grad_s_X * edge_toroidal_flux_over_2pi
    grad_psi_Y = grad_s_Y * edge_toroidal_flux_over_2pi
    grad_psi_Z = grad_s_Z * edge_toroidal_flux_over_2pi

    # Form grad alpha = grad (theta_vmec + lambda - iota * phi)
    grad_alpha_X = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_X
    grad_alpha_Y = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_Y
    grad_alpha_Z = (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) * grad_s_Z

    grad_alpha_X += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_X + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_X
    grad_alpha_Y += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Y + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_Y
    grad_alpha_Z += (1 + d_lambda_d_theta_vmec) * grad_theta_vmec_Z + (-iota[:, None, None] + d_lambda_d_phi) * grad_phi_Z

    grad_B_X = d_B_d_s * grad_s_X + d_B_d_theta_vmec * grad_theta_vmec_X + d_B_d_phi * grad_phi_X
    grad_B_Y = d_B_d_s * grad_s_Y + d_B_d_theta_vmec * grad_theta_vmec_Y + d_B_d_phi * grad_phi_Y
    grad_B_Z = d_B_d_s * grad_s_Z + d_B_d_theta_vmec * grad_theta_vmec_Z + d_B_d_phi * grad_phi_Z

    B_X = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_X_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_X_d_theta_vmec) / sqrt_g_vmec
    B_Y = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Y_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Y_d_theta_vmec) / sqrt_g_vmec
    B_Z = edge_toroidal_flux_over_2pi * ((1 + d_lambda_d_theta_vmec) * d_Z_d_phi + (iota[:, None, None] - d_lambda_d_phi) * d_Z_d_theta_vmec) / sqrt_g_vmec

    # *********************************************************************
    # For gbdrift, we need \vect{B} cross grad |B| dot grad alpha.
    # For cvdrift, we also need \vect{B} cross grad s dot grad alpha.
    # Let us compute both of these quantities 2 ways, and make sure the two
    # approaches give the same answer (within some tolerance).
    # *********************************************************************

    B_cross_grad_s_dot_grad_alpha = (B_sub_phi * (1 + d_lambda_d_theta_vmec) \
                                     - B_sub_theta_vmec * (d_lambda_d_phi - iota[:, None, None])) / sqrt_g_vmec

    B_cross_grad_s_dot_grad_alpha_alternate = 0 \
        + B_X * grad_s_Y * grad_alpha_Z \
        + B_Y * grad_s_Z * grad_alpha_X \
        + B_Z * grad_s_X * grad_alpha_Y \
        - B_Z * grad_s_Y * grad_alpha_X \
        - B_X * grad_s_Z * grad_alpha_Y \
        - B_Y * grad_s_X * grad_alpha_Z

    B_cross_grad_B_dot_grad_alpha = 0 \
        + (B_sub_s * d_B_d_theta_vmec * (d_lambda_d_phi - iota[:, None, None]) \
           + B_sub_theta_vmec * d_B_d_phi * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) \
           + B_sub_phi * d_B_d_s * (1 + d_lambda_d_theta_vmec) \
           - B_sub_phi * d_B_d_theta_vmec * (d_lambda_d_s - (phi - phi_center) * d_iota_d_s[:, None, None]) \
           - B_sub_theta_vmec * d_B_d_s * (d_lambda_d_phi - iota[:, None, None]) \
           - B_sub_s * d_B_d_phi * (1 + d_lambda_d_theta_vmec)) / sqrt_g_vmec

    B_cross_grad_B_dot_grad_alpha_alternate = 0 \
        + B_X * grad_B_Y * grad_alpha_Z \
        + B_Y * grad_B_Z * grad_alpha_X \
        + B_Z * grad_B_X * grad_alpha_Y \
        - B_Z * grad_B_Y * grad_alpha_X \
        - B_X * grad_B_Z * grad_alpha_Y \
        - B_Y * grad_B_X * grad_alpha_Z

    grad_alpha_dot_grad_alpha = grad_alpha_X * grad_alpha_X + grad_alpha_Y * grad_alpha_Y + grad_alpha_Z * grad_alpha_Z

    grad_alpha_dot_grad_psi = grad_alpha_X * grad_psi_X + grad_alpha_Y * grad_psi_Y + grad_alpha_Z * grad_psi_Z

    grad_psi_dot_grad_psi = grad_psi_X * grad_psi_X + grad_psi_Y * grad_psi_Y + grad_psi_Z * grad_psi_Z

    B_cross_grad_B_dot_grad_psi = (B_sub_theta_vmec * d_B_d_phi - B_sub_phi * d_B_d_theta_vmec) / sqrt_g_vmec * edge_toroidal_flux_over_2pi

    B_cross_kappa_dot_grad_psi = B_cross_grad_B_dot_grad_psi / modB

    mu_0 = 4 * np.pi * (1.0e-7)
    B_cross_kappa_dot_grad_alpha = B_cross_grad_B_dot_grad_alpha / modB + mu_0 * d_pressure_d_s[:, None, None] / edge_toroidal_flux_over_2pi

    # stella / gs2 / gx quantities:

    L_reference = vs.Aminor_p
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    bmag = modB / B_reference

    gradpar_theta_pest = L_reference * B_sup_theta_pest / modB

    gradpar_phi = L_reference * B_sup_phi / modB

    gds2 = grad_alpha_dot_grad_alpha * L_reference * L_reference * s[:, None, None]

    gds21 = grad_alpha_dot_grad_psi * shat[:, None, None] / B_reference

    gds22 = grad_psi_dot_grad_psi * shat[:, None, None] * shat[:, None, None] / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])

    # temporary fix. Please see issue #238 and the discussion therein
    gbdrift = -1 * 2 * B_reference * L_reference * L_reference * sqrt_s[:, None, None] * B_cross_grad_B_dot_grad_alpha / (modB * modB * modB) * toroidal_flux_sign

    gbdrift0 = B_cross_grad_B_dot_grad_psi * 2 * shat[:, None, None] / (modB * modB * modB * sqrt_s[:, None, None]) * toroidal_flux_sign

    # temporary fix. Please see issue #238 and the discussion therein
    cvdrift = gbdrift - 2 * B_reference * L_reference * L_reference * sqrt_s[:, None, None] * mu_0 * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (edge_toroidal_flux_over_2pi * modB * modB)

    cvdrift0 = gbdrift0

    # Package results into a structure to return:
    results = Struct()
    variables = ['ns', 'nalpha', 'nl', 's', 'iota', 'd_iota_d_s', 'd_pressure_d_s', 'shat',
                 'alpha', 'theta1d', 'phi1d', 'phi', 'theta_pest',
                 'd_lambda_d_s', 'd_lambda_d_theta_vmec', 'd_lambda_d_phi', 'sqrt_g_vmec', 'sqrt_g_vmec_alt',
                 'theta_vmec', 'modB', 'd_B_d_s', 'd_B_d_theta_vmec', 'd_B_d_phi', 'B_sup_theta_vmec', 'B_sup_theta_pest', 'B_sup_phi',
                 'B_sub_s', 'B_sub_theta_vmec', 'B_sub_phi', 'edge_toroidal_flux_over_2pi', 'sinphi', 'cosphi',
                 'R', 'd_R_d_s', 'd_R_d_theta_vmec', 'd_R_d_phi', 'Z', 'd_Z_d_s', 'd_Z_d_theta_vmec', 'd_Z_d_phi',
                 'd_X_d_theta_vmec', 'd_X_d_phi', 'd_X_d_s', 'd_Y_d_theta_vmec', 'd_Y_d_phi', 'd_Y_d_s',
                 'grad_s_X', 'grad_s_Y', 'grad_s_Z', 'grad_theta_vmec_X', 'grad_theta_vmec_Y', 'grad_theta_vmec_Z',
                 'grad_phi_X', 'grad_phi_Y', 'grad_phi_Z', 'grad_psi_X', 'grad_psi_Y', 'grad_psi_Z',
                 'grad_alpha_X', 'grad_alpha_Y', 'grad_alpha_Z', 'grad_B_X', 'grad_B_Y', 'grad_B_Z',
                 'B_X', 'B_Y', 'B_Z',
                 'B_cross_grad_s_dot_grad_alpha', 'B_cross_grad_s_dot_grad_alpha_alternate',
                 'B_cross_grad_B_dot_grad_alpha', 'B_cross_grad_B_dot_grad_alpha_alternate',
                 'B_cross_grad_B_dot_grad_psi', 'B_cross_kappa_dot_grad_psi', 'B_cross_kappa_dot_grad_alpha',
                 'grad_alpha_dot_grad_alpha', 'grad_alpha_dot_grad_psi', 'grad_psi_dot_grad_psi',
                 'L_reference', 'B_reference', 'toroidal_flux_sign',
                 'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gds2', 'gds21', 'gds22', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0']

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(13, 7))
        nrows = 4
        ncols = 5
        variables = ['modB', 'B_sup_theta_pest', 'B_sup_phi', 'B_cross_grad_B_dot_grad_alpha', 'B_cross_grad_B_dot_grad_psi',
                     'B_cross_kappa_dot_grad_alpha', 'B_cross_kappa_dot_grad_psi',
                     'grad_alpha_dot_grad_alpha', 'grad_alpha_dot_grad_psi', 'grad_psi_dot_grad_psi',
                     'bmag', 'gradpar_theta_pest', 'gradpar_phi', 'gbdrift', 'gbdrift0', 'cvdrift', 'cvdrift0', 'gds2', 'gds21', 'gds22']
        for j, variable in enumerate(variables):
            plt.subplot(nrows, ncols, j + 1)
            plt.plot(phi[0, 0, :], eval(variable + '[0, 0, :]'))
            plt.xlabel('Standard toroidal angle $\phi$')
            plt.title(variable)

        plt.figtext(0.5, 0.995, f's={s[0]}, alpha={alpha[0]}', ha='center', va='top')
        plt.tight_layout()
        if show:
            plt.show()

    for v in variables:
        results.__setattr__(v, eval(v))

    return results
