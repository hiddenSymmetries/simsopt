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


def critical_current_pure(gamma, gammadash, alpha, Bself, Bext, model=None):
    tangent, normal, binormal = rotated_centroid_frame(
        gamma, gammadash, alpha
        )

    # Evaluate field
    field = Bself + Bext
    B_proj = field - inner(field, tangent)[:, None] * tangent
    B_perp = inner(B_proj, normal)
    B_par = inner(B_proj, binormal)

    # Fit parameters for reduced Kim-like model of the critical current (doi:10.1088/0953-2048/24/6/065005)
    if model is None:
        xi = -0.7
        k = 0.3
        B_0 = 42.6e-3
        Ic_0 = 1  # 1.3e11,

        Ic = Ic_0*(jnp.sqrt(k**2 * B_par**2 + B_perp**2) / B_0)**xi
    else:
        Ic = model(B_perp, B_par, B_0)

    #TODO: modify call signature everywhere
    #TODO: make a file with different critical current models
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


def critical_current_obj_pure(gamma, gammadash, alpha, Bself, Bext, p=10, model=None):
    # TODO: What is this?!
    Ic0 = 1 

    Ic = critical_current_pure(
        gamma, gammadash, alpha, Bself, Bext, model
    )

    # TODO: clarify this thing.
    # Step 1: Maximize critical current for one winding.    
    # Step 1: P-norm approx of the critical current minimum
    return jnp.mean(Ic**(1./p))**p
    
    # Step 2: Penalize current above critical current, given temperature and number of winds
    # Ic is the current in the coil
    # Ic0 is the critical current along the coil
    #return jnp.sum(jnp.maximum(Ic-Ic0, 0)**2)


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

    def __init__(self, self_field, frame, biotsavart_ext, model=None, p=4):
        self.self_field = self_field
        self._curve = self_field._curve #shorthand
        self._current = self_field._current #shorthand
        self.frame = frame
        self.B_ext = biotsavart_ext
        self.B_ext.set_points(self.self_field.get_points_cart())
        self.model = model
        self.p = p

        self.J_jax = jit(lambda gamma, gammadash, alpha, Bself, Bext: critical_current_obj_pure(gamma, gammadash, alpha, Bself, Bext, p=self.p, model=self.model))

        self.thisgrad0 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=0)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad1 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=1)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad2 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=2)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad3 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=3)(gamma, gammadash, alpha, Bself, Bext))
        self.thisgrad4 = jit(lambda gamma, gammadash, alpha, Bself, Bext: grad(self.J_jax, argnums=4)(gamma, gammadash, alpha, Bself, Bext))

        super().__init__(depends_on=[self_field, frame, biotsavart_ext])

    def J(self):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        alpha = self.frame.rotation.alpha(self._curve.quadpoints)
        Bself = self.self_field.B()
        Bext = self.B_ext.B()

        return self.J_jax(gamma, gammadash, alpha, Bself, Bext)

    @derivative_dec
    def dJ(self):
        gamma = self._curve.gamma()
        gammadash = self._curve.gammadash()
        alpha = self.frame.rotation.alpha(self._curve.quadpoints)
        Bself = self.self_field.B()
        Bext = self.B_ext.B()

        grad0 = self.thisgrad0(gamma, gammadash, alpha, Bself, Bext)
        grad1 = self.thisgrad1(gamma, gammadash, alpha, Bself, Bext)
        grad2 = self.thisgrad2(gamma, gammadash, alpha, Bself, Bext)
        grad3 = self.thisgrad3(gamma, gammadash, alpha, Bself, Bext)
        grad4 = self.thisgrad4(gamma, gammadash, alpha, Bself, Bext)

        out = self._curve.dgamma_by_dcoeff_vjp(grad0) \
            + self._curve.dgammadash_by_dcoeff_vjp(grad1) \
            + self.frame.rotation.dalpha_by_dcoeff_vjp(self._curve.quadpoints, grad2) \
            + self.self_field.B_vjp(grad3) \
            + self.B_ext.B_vjp(grad4)

        # out = self._curve.dgamma_by_dcoeff_vjp(grad0) \
        #     + self._curve.dgammadash_by_dcoeff_vjp(grad1) \
        #     + self.frame.rotation.dalpha_by_dcoeff_vjp(self._curve.quadpoints, grad2)\
        #     + self.self_field.B_vjp(grad3)

        return out
    
    return_fn_map = {'J': J, 'dJ': dJ}
