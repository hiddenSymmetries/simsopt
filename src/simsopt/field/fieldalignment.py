"""Implements optimization of the angle between the magnetic field and the ReBCO c-axis."""
import math
import pickle
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart, Coil
from simsopt.field.selffield import B_regularized, regularization_rect, regularization_circ, B_regularized_pure
from simsopt.geo import FramedCurve
from simsopt.geo.framedcurve import rotated_frenet_frame
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def vector_matrix_product(matrix, vector):
    """Product between matrix and a vector with one free axis (e.g. along a coil)"""
    return jnp.einsum('ijk, ik -> ij', matrix, vector)


def inner(vector_1, vector_2):
    """Dot Product between two vectors with one free axis (e.g. along a coil)"""
    return jnp.sum(vector_1*vector_2, axis=1)


def rotation(angle, axis, vector):
    """Rotates a 3D vector around an axis vector"""

    n_quad = angle.shape[0]

    rotation_matrix = jnp.array([[[jnp.cos(angle[i])+axis[i, 0]**2*(1-jnp.cos(angle[i])),
                                  axis[i, 0]*axis[i, 1] *
                                  (1-jnp.cos(angle[i])) -
                                  axis[i, 2]*jnp.sin(angle[i]),
                                  axis[i, 0]*axis[i, 2]*(1-jnp.cos(angle[i])) + axis[i, 1]*jnp.sin(angle[i])],

                                 [axis[i, 1]*axis[i, 0]*(1-jnp.cos(angle[i])) + axis[i, 2]*jnp.sin(angle[i]),
                                  jnp.cos(angle[i]) + axis[i, 1]**2 *
                                  (1-jnp.cos(angle[i])),
                                  axis[i, 1]*axis[i, 2]*(1-jnp.cos(angle[i]))-axis[i, 0]*jnp.sin(angle[i])],

                                 [axis[i, 0]*axis[i, 2]*(1-jnp.cos(angle[i])) - axis[i, 1]*jnp.sin(angle[i]),
                                  axis[i, 1]*axis[i, 2] *
                                  (1-jnp.cos(angle[i])) +
                                  axis[i, 0]*jnp.sin(angle[i]),
                                  jnp.cos(angle[i]) + axis[i, 2]**2*(1-jnp.cos(angle[i]))]] for i in range(n_quad)])

    return vector_matrix_product(rotation_matrix, vector)


def c_axis_pure(t, b):
    """Implements THEVA-tape-like c-axis"""
    n_quad = t.shape[0]
    rotation_degrees = 60*jnp.ones(n_quad)
    rotation_radians = jnp.radians(rotation_degrees)
    c_axis = rotation(rotation_radians, t, b)

    return c_axis


def c_axis_angle_pure(coil, B):
    """Angle between the magnetic field and the c-axis of the REBCO Tape"""
    t, _, b = coil.curve.rotated_frame()
    b_norm = jnp.einsum("ij, i->ij", B, 1/jnp.linalg.norm(B, axis=1))

    c_axis = c_axis_pure(t, b)
    angle = jnp.arccos(inner(c_axis, b_norm))

    return angle


def critical_current_pure(gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b):

    regularization = regularization_rect(a, b)
    tangent, normal, binormal = rotated_frenet_frame(
        gamma, gammadash, gammadashdash, alpha)

    # Fit parameters for reduced Kim-like model of the critical current (doi:10.1088/0953-2048/24/6/065005)
    xi = -0.3
    k = 0.7
    B_0 = 42.6e-3
    Ic_0 = 1

    field = B_regularized_pure(
        gamma, gammadash, gammadashdash, quadpoints, current, regularization)
    B_proj = field - inner(field, tangent)[:, None] * tangent

    B_perp = inner(B_proj, normal)
    B_par = inner(B_proj, binormal)

    Ic = Ic_0*(jnp.sqrt(k**2 * B_par**2 + B_perp**2) / B_0)**xi
    return Ic


def critical_current(framedcoil, a, b, JANUS=True):
    """Critical current on a coil assuming a homogeneous current distribution. Replace with analytical model"""
    Ic = critical_current_pure(framedcoil.curve.curve.gamma(), framedcoil.curve.curve.gammadash(), framedcoil.curve.curve.gammadashdash(
    ), framedcoil.curve.rotation.alpha(framedcoil.curve.quadpoints), framedcoil.curve.quadpoints, framedcoil.current.current, a, b)
    return Ic


def critical_current_obj_pure(gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b):
    Ic = critical_current_pure(
        gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b)
    obj = np.min(Ic)
    return obj


def critical_current_obj(framedcoil, a, b, JANUS=True):
    """Objective for field alignement optimization: Target minimum of the critical current along the coil"""
    return np.min(critical_current(framedcoil, a, b))


class CrtitcalCurrentOpt(Optimizable):
    """Optimizable class to optimize the critical on a ReBCO coil"""

    def __init__(self, framedcoil, coils, a=0.05):
        self.coil = coil
        self.curve = coil.curve
        self.coils = coils
        self.a = a
        self.B_ext = BiotSavart(coils).set_points(self.curve.gamma()).B()
        self.B_self = 0
        self.B = 0
        self.alpha = coil.curve.rotation.alpha(coil.curve.quadpoints)
        self.quadpoints = coil.curve.quadpoints
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash, alpha, quadpoints, current, a,
                         b: critical_current_obj_pure(gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b))

        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, current, phi, phidash, B_ext: grad(
            self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, current, phi, phidash, B_ext))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, current, phi, phidash, B_ext: grad(
            self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, current, phi, phidash, B_ext))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, current, phi, phidash, B_ext: grad(
            self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, current, phi, phidash, B_ext))

        super().__init__(depends_on=[coil])

    def J(self):
        gamma = self.coil.curve.gamma()
        d1gamma = self.coil.curve.gammadash()
        d2gamma = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        phi = self.coil.curve.quadpoints
        phidash = self.coil.curve.quadpoints
        B_ext = self.B_ext
        return self.J_jax(gamma, d1gamma, d2gamma, current, phi, phidash, B_ext)

    @derivative_dec
    def dJ(self):
        gamma = self.coil.curve.gamma()
        d1gamma = self.coil.curve.gammadash()
        d2gamma = self.coil.curve.gammadashdash()
        current = self.coil.current.get_value()
        phi = self.coil.curve.quadpoints
        phidash = self.coil.curve.quadpoints
        B_ext = self.B_ext

        grad0 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, phidash, B_ext)
        grad1 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, phidash, B_ext)
        grad2 = self.thisgrad0(gamma, d1gamma, d2gamma,
                               current, phi, phidash, B_ext)

        return self.coil.curve.dgamma_by_dcoeff_vjp(grad0) + self.coil.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.coil.curve.dgammadashdash_by_dcoeff_vjp(grad2)

    return_fn_map = {'J': J, 'dJ': dJ}
