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
    """
    Computes Cartesian vector components of magnetic field on boundary
    on a grid in the vmec toroidal and poloidal angles. This is
    required to compute adjoint-based shape gradients.
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
    """
    IotaTargetMetric is a class that computes the metric,

        0.5 * \int ds \, (iota - iota_target)**2

    from a vmec equilibrium. Its derivatives can also be computed with an adjoint
    method.

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
        Computes a metric quantifying the deviation of iota from a prescribed
        target value,

            0.5 * \int ds \, (iota - iota_target)**2
        """
        self.vmec.need_to_run_code = True
        self.vmec.run()
        return 0.5 * np.sum((self.vmec.wout.iotas[1::]
                             - self.iota_target(self.vmec.s_half_grid))**2) * self.vmec.ds

    def dJ(self):
        """
        Computes derivatives of J wrt surface parameters using
        an adjoint method.
        """
        if self.vmec.indata.ncurr != 1:
            raise RuntimeError('''dJ cannot be computed without
                running vmec with ncurr = 1''')

        shape_gradient = self.shape_gradient()
        return parameter_derivatives(self.vmec.boundary, shape_gradient)

    def shape_gradient(self):
        """
        Computes shape gradient of J, defined as S where,

            \delta f(\delta x) = \int d^2 x \, S \delta x \cdot n,

        is the perturbation to the objective function corresponding to the
        perturbation of the surface, \delta x.
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
    """
    Computes a weighted average of the rotational transform defined by
    the prescribed weight_function,

        \int ds \, iota * weight_function(s) / \int ds \, weight_function(s)

    from a vmec equilibrium. [an adjoint method is also available, but not
    yet implemented]

    Args:
        vmec : instance of Vmec
        weight_function : function handle which takes a single argument, s,
            the normalized toroidal flux
    """

    def __init__(self, vmec, weight_function):
        self.vmec = vmec
        self.boundary = vmec.boundary
        self.weight_function = weight_function
        self.depends_on = ["boundary"]

    def J(self):
        """
        Computes a weighted average of the rotational transform defined by
        the prescribed weight_function, 

            \int ds \, iota * weight_function(s) / \int ds \, weight_function(s)
        """
        self.vmec.need_to_run_code = True
        self.vmec.run()
        return np.sum(self.weight_function(self.vmec.s_half_grid) * self.vmec.wout.iotas[1:]) \
            / np.sum(self.weight_function(self.vmec.s_half_grid))


class WellWeighted(Optimizable):
    """
    WellWeighted is a class that computes the objective function,

        f = \int ds \, V'(s) * (weight_function1(s) - weight_function2(s))
            / \int ds \, V'(s) (weight_function1(s) + weight_function2(s)),

    from a vmec equilibrium. Its derivatives can also be computed with an adjoint
    method.

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
        Computed a weighted average of the vacuum magnetic well defined by
        the prescribed weight_functions:

        f = \int ds \, V'(s) * (weight_function1(s) - weight_function2(s))
            / \int ds \, V'(s) (weight_function1(s) + weight_function2(s))

        For example, weight_function1 could be peaked on the edge while
        weight_function2 could be peaked on the axis such that f < 0 corresonds
        to V''(s) < 0.
        """

        self.vmec.need_to_run_code = True
        self.vmec.run()
        return np.sum((self.weight_function1(self.vmec.s_half_grid)-self.weight_function2(self.vmec.s_half_grid)) * self.vmec.wout.vp[1:]) \
            / np.sum((self.weight_function1(self.vmec.s_half_grid)+self.weight_function2(self.vmec.s_half_grid)) * self.vmec.wout.vp[1:])

    def dJ(self):
        """
        Computes derivatives of J wrt surface parameters using
        an adjoint method.
        """

        self.vmec.need_to_run_code = True
        shape_gradient = self.shape_gradient()
        return parameter_derivatives(self.vmec.boundary, shape_gradient)

    def shape_gradient(self):
        """
        Computes shape gradient of J, defined as S where,

        \delta f(\delta x) = \int d^2 x \, S \delta x \cdot n,

        is the perturbation to the objective function corresponding to the
        perturbation of the surface, \delta x.
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
