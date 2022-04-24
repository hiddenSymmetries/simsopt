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

from .vmec import Vmec
from .._core.util import Struct
from .._core.graph_optimizable import Optimizable
from ..util.types import RealArray
from ..geo.surfaceobjectives import parameter_derivatives

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


def B_cartesian(vmec):
    r"""
    Computes Cartesian vector components of magnetic field on boundary
    on a grid in the vmec toroidal and poloidal angles. This is
    required to compute adjoint-based shape gradients.

    Args:
        vmec : instance of Vmec

    Returns: 3 element tuple containing (Bx, By, Bz)
        Bx, By, Bz : 2d arrays of size (numquadpoints_phi,numquadpoints_theta)
            defining Cartesian components of magnetic field on vmec.boundary
    """
    dgamma1 = vmec.boundary.gammadash1()
    dgamma2 = vmec.boundary.gammadash2()

    theta1D = vmec.boundary.quadpoints_theta * 2 * np.pi
    phi1D = vmec.boundary.quadpoints_phi * 2 * np.pi
    nphi = len(phi1D)
    ntheta = len(theta1D)
    theta, phi = np.meshgrid(theta1D, phi1D)

    vmec.run()
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
