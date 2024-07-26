"""Implements optimization of the angle between the magnetic field and the ReBCO c-axis."""
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart
from simsopt._core.derivative import Derivative
from simsopt.field.selffield import regularization_rect, B_regularized_pure
from simsopt.geo.framedcurve import rotated_centroid_frame
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi

__all__ = ['CriticalCurrentOpt', 'critical_current']

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


def critical_current_pure(gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b, Bext):

    regularization = regularization_rect(a, b)
    tangent, normal, binormal = rotated_centroid_frame(
        gamma, gammadash, alpha
        )

    # Evaluate field
    field = B_regularized_pure(
        gamma, gammadash, gammadashdash, quadpoints, current, regularization) + Bext
    B_proj = field - inner(field, tangent)[:, None] * tangent
    B_perp = inner(B_proj, normal)
    B_par = inner(B_proj, binormal)

    # Fit parameters for reduced Kim-like model of the critical current (doi:10.1088/0953-2048/24/6/065005)
    xi = -0.7
    k = 0.3
    B_0 = 42.6e-3
    Ic_0 = 1  # 1.3e11,


    Ic = Ic_0*(jnp.sqrt(k**2 * B_par**2 + B_perp**2) / B_0)**xi
    return Ic


def critical_current(framedcoil, a, b, Bext, JANUS=True):
    """Critical current on a coil assuming a homogeneous current distribution. Replace with analytical model"""
    Ic = critical_current_pure(
        framedcoil.framedcurve.curve.gamma(), 
        framedcoil.framedcurve.curve.gammadash(), 
        framedcoil.framedcurve.curve.gammadashdash(), 
        framedcoil.framedcurve.rotation.alpha(framedcoil.framedcurve.curve.quadpoints), 
        framedcoil.framedcurve.curve.quadpoints, 
        framedcoil.current.get_value(), 
        a, b, Bext
    )
    return Ic


def critical_current_obj_pure(gamma, gammadash, gammadashdash, alpha, quadpoints, current, a, b, Bext):
    Ic0 = 0  # QUESTION: Why is Ic0 equal to 1 here?
    Ic = critical_current_pure(
        gamma, 
        gammadash, 
        gammadashdash, 
        alpha, 
        quadpoints, 
        current, 
        a, b, 
        Bext
    )
    return jnp.sum(jnp.maximum(Ic-Ic0, 0)**2)


def critical_current_obj(framedcoil, a, b, Bext, JANUS=True):
    """Objective for field alignement optimization: Target minimum of the critical current along the coil"""
    obj = critical_current_obj_pure(
        framedcoil.framedcurve.curve.gamma(), 
        framedcoil.framedcurve.curve.d1gamma(), 
        framedcoil.framedcurve.curve.d2gamma(),
        framedcoil.framedcurve.rotation.alpha(framedcoil.framedcurve.curve.quadpoints),
        framedcoil.framedcurve.curve.quadpoints, 
        framedcoil.framedcurrent.get_value(), 
        a, b, Bext
    )
    return obj


class CriticalCurrentOpt(Optimizable):
    """Optimizable class to optimize the critical on a ReBCO coil
    
    Args:
    -----
     - framedcoil. simsopt.field.Coil object, where framedcoil.curve is an instance of the simsopt.geo.framedcurve class. The critical current will be evaluated for this coil.
     - biotsavart_ext. simsopt.field.BiotSavart object. Contains all other coils in the system. Should not include the coil represented by framedcoil.
     - a (float, default: 0.05). Conductor stack width ?
     - b (float, default: 0.05). Conductor stack height ?
    """

    def __init__(self, framedcoil, biotsavart_ext, a=0.05, b=0.05):
        self.coil = framedcoil
        self.framedcurve = framedcoil.framedcurve
        self.quadpoints = self.coil.curve.curve.quadpoints
        
        self.a = a
        self.b = b

        self.B_ext = biotsavart_ext
        self.B_ext.set_points(self.framedcurve.curve.gamma().reshape((-1,3)))

        self.quadpoints = framedcoil.framedcurve.curve.quadpoints
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: critical_current_obj_pure(gamma, gammadash, gammadashdash, alpha, self.quadpoints, current, self.a, self.b, Bext))

        self.thisgrad0 = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: grad(self.J_jax, argnums=0)(gamma, gammadash, gammadashdash, alpha, current, Bext))
        self.thisgrad1 = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: grad(self.J_jax, argnums=1)(gamma, gammadash, gammadashdash, alpha, current, Bext))
        self.thisgrad2 = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: grad(self.J_jax, argnums=2)(gamma, gammadash, gammadashdash, alpha, current, Bext))
        self.thisgrad3 = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: grad(self.J_jax, argnums=3)(gamma, gammadash, gammadashdash, alpha, current, Bext))
        self.thisgrad4 = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: grad(self.J_jax, argnums=4)(gamma, gammadash, gammadashdash, alpha, current, Bext))
        self.thisgrad5 = jit(lambda gamma, gammadash, gammadashdash, alpha, current, Bext: grad(self.J_jax, argnums=5)(gamma, gammadash, gammadashdash, alpha, current, Bext))

        super().__init__(depends_on=[framedcoil, biotsavart_ext])

    def J(self):
        gamma = self.coil.framedcurve.curve.gamma()
        d1gamma = self.coil.framedcurve.curve.gammadash()
        d2gamma = self.coil.framedcurve.curve.gammadashdash()
        alpha = self.coil.framedcurve.rotation.alpha(self.quadpoints)
        current = self.coil.current.get_value()

        Bext = self.B_ext.B()
        return self.J_jax(gamma, d1gamma, d2gamma, alpha, current, Bext)

    @derivative_dec
    def dJ(self):
        gamma = self.coil.framedcurve.curve.gamma()
        d1gamma = self.coil.framedcurve.curve.gammadash()
        d2gamma = self.coil.framedcurve.curve.gammadashdash()
        current = self.coil.current.get_value()
        alpha = self.coil.framedcurve.rotation.alpha(self.quadpoints)

        B_ext = self.B_ext.B()

        grad0 = self.thisgrad0(gamma, d1gamma, d2gamma, alpha, current, B_ext)
        grad1 = self.thisgrad1(gamma, d1gamma, d2gamma, alpha, current, B_ext)
        grad2 = self.thisgrad2(gamma, d1gamma, d2gamma, alpha, current, B_ext)
        grad3 = self.thisgrad3(gamma, d1gamma, d2gamma, alpha, current, B_ext)
        grad4 = self.thisgrad4(gamma, d1gamma, d2gamma, alpha, current, B_ext)
        grad5 = self.thisgrad5(gamma, d1gamma, d2gamma, alpha, current, B_ext)

        out = self.coil.framedcurve.curve.dgamma_by_dcoeff_vjp(grad0) \
            + self.coil.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.coil.framedcurve.curve.dgammadashdash_by_dcoeff_vjp(grad2) \
            + self.coil.framedcurve.rotation.dalpha_by_dcoeff_vjp(self.quadpoints, grad3) \
            + self.coil.current.vjp(grad4) \
            + self.B_ext.B_vjp(grad5)

        return out
    
    return_fn_map = {'J': J, 'dJ': dJ}
