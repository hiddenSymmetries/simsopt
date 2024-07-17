"""Implements optimization of the angle between the magnetic field and the ReBCO c-axis."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart
from simsopt.field.selffield import regularization_rect, B_regularized_pure
from simsopt.geo.framedcurve import rotated_centroid_frame
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = ['CriticalCurrentOpt']

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
    tangent, normal, binormal = rotated_centroid_frame(
        gamma, gammadash, alpha)

    # Fit parameters for reduced Kim-like model of the critical current (doi:10.1088/0953-2048/24/6/065005)
    xi = -0.7
    k = 0.3
    B_0 = 42.6e-3
    Ic_0 = 1  # 1.3e11,

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
    Ic0 = 1  # QUESTION: Why is Ic0 equal to 1 here?
    Ic = critical_current_pure(
        gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b)
    obj = jnp.sum(jnp.maximum(Ic-Ic0, 0)**2)
    return obj


def critical_current_obj(framedcoil, a, b, JANUS=True):
    """Objective for field alignement optimization: Target minimum of the critical current along the coil"""
    obj = critical_current_obj_pure(framedcoil.curve.curve.gamma, framedcoil.curve.curve.d1gamma, framedcoil.curve.curve.d2gamma,
                                    framedcoil.curve.rotation.alpha(framedcoil.curve.quadpoints), framedcoil.curve.quadpoints, framedcoil.current.current, a, b)
    return obj


class CriticalCurrentOpt(Optimizable):
    """Optimizable class to optimize the critical on a ReBCO coil
    
    Args:
    -----
     - framedcoil. simsopt.field.Coil object, where framedcoil.curve is an instance of the simsopt.geo.framedcurve class. The critical current will be evaluated for this coil.
     - coils (list). List of all other coils in the system
     - a (float, default: 0.05). Conductor stack width ?
     - b (float, default: 0.05). Conductor stack height ?
    """

    def __init__(self, framedcoil, coils, a=0.05, b=0.05):
        self.coil = framedcoil
        self.curve = framedcoil.curve
        self.quadpoints = self.coil.curve.curve.quadpoints
        self.coils = coils
        self.a = a
        self.b = b
        self.B_ext = BiotSavart(coils).set_points(
            framedcoil.curve.curve.gamma()).B()
        self.B_self = 0
        self.B = 0
        self.alpha = framedcoil.curve.rotation.alpha(
            framedcoil.curve.quadpoints
            )
        self.quadpoints = framedcoil.curve.quadpoints
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash, alpha, current: critical_current_obj_pure(gamma, gammadash, gammadashdash, alpha, self.quadpoints, current, self.a, self.b))

        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, alpha, current: grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, alpha, current))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, alpha, current: grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, alpha, current))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, alpha, current: grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, alpha, current))
        self.thisgrad3 = jit(lambda gamma, gammadash, gammadashdash, alpha, current: grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, alpha, current))
        self.thisgrad4 = jit(lambda gamma, gammadash, gammadashdash, alpha, current: grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, alpha, current))

        super().__init__(depends_on=[framedcoil])

    def J(self):
        gamma = self.coil.curve.curve.gamma()
        d1gamma = self.coil.curve.curve.gammadash()
        d2gamma = self.coil.curve.curve.gammadashdash()
        current = self.coil.current.get_value()
        #phidash = self.coil.curve.curve.quadpoints
        alpha = self.coil.curve.rotation.alpha(self.quadpoints)

        #B_ext = self.B_ext
        return self.J_jax(gamma, d1gamma, d2gamma, alpha, current)

    @derivative_dec
    def dJ(self):
        gamma = self.coil.curve.curve.gamma()
        d1gamma = self.coil.curve.curve.gammadash()
        d2gamma = self.coil.curve.curve.gammadashdash()
        current = self.coil.current.get_value()
        alpha = self.coil.curve.rotation.alpha(self.quadpoints)

        #B_ext = self.B_ext

        grad0 = self.thisgrad0(gamma, d1gamma, d2gamma, alpha, current)
        grad1 = self.thisgrad1(gamma, d1gamma, d2gamma, alpha, current)
        grad2 = self.thisgrad2(gamma, d1gamma, d2gamma, alpha, current)
        grad3 = self.thisgrad3(gamma, d1gamma, d2gamma, alpha, current)
        #grad4 = self.thisgrad4(gamma, d1gamma, d2gamma, alpha, current)

        return self.coil.curve.curve.dgamma_by_dcoeff_vjp(grad0) \
            + self.coil.curve.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.coil.curve.curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self.coil.curve.rotation.dalpha_by_dcoeff_vjp(self.quadpoints, grad3) \
            #+ self.coil.current.vjp(grad4)

    return_fn_map = {'J': J, 'dJ': dJ}
