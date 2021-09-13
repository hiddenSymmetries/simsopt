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
from ..util.types import RealArray
from .._core.optimizable import Optimizable
from ..geo.surfaceobjectives import parameter_derivatives

logger = logging.getLogger(__name__)


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
        self.depends_on = ["boundary"]

    def J(self):
        """
        Computes the quantity :math:`J` described in the class definition.
        """
        self.vmec.need_to_run_code = True
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
        self.depends_on = ["boundary"]
        self.ancestors = [self.boundary]

    def J(self):
        """
        Computes the quantity :math:`J` described in the class definition.
        """
        self.vmec.need_to_run_code = True
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
        # return parameter_derivatives(self.vmec.boundary, shape_gradient)
        s = self.vmec.boundary
        return s.dgamma_by_dcoeff_vjp(s.normal() * shape_gradient[:, :, None])

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
        self.depends_on = ["boundary"]

    def J(self):
        """
        Computes the quantity :math:`J` described in the class definition.
        """

        self.vmec.need_to_run_code = True
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
