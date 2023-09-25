"""Implements optimization of the angle between the magnetic field and the ReBCO c-axis."""
import math
import pickle
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart, Coil
from simsopt.field.selffield import field_on_coils
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.finitebuild import rotated_centroid_frame

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
    t, n, b = rotated_centroid_frame(coil.curve.gamma(
    ), coil.curve.gammadash(), coil.curve.rotation.alpha(coil.curve.quadpoints))
    b_norm = jnp.einsum("ij, i->ij", B, 1/jnp.linalg.norm(B, axis=1))

    c_axis = c_axis_pure(t, b)
    angle = jnp.arccos(inner(c_axis, b_norm))
    return angle


def critical_current_obj(coil, JANUS=True):
    field = field_on_coils(coil)
    angle = c_axis_angle_pure(coil, field)
    file_path = "/Users/paulhuslage/PhD/epos-simsopt/src/simsopt/examples/3_Advanced/inputs/Ic_angle_field_interpolation_with units.pck"
    B_norm = np.linalg.norm(field, axis=1)

    with open(file_path, "rb") as file:
        # Load the contents of the .pck file
        data = pickle.load(file)

    ic_interpolation = data['Ic_interpolation']
    ic_interpolation_reversed = data['Ic_interpolation_reversed']

    if JANUS:
        Ic = [ic_interpolation_reversed([a, f]) for a, f in zip(angle, B_norm)]
    else:
        Ic = [ic_interpolation([a, f]) for a, f in zip(angle, B_norm)]

    return np.min(Ic)


def critical_current(coil, JANUS=True):
    field = field_on_coils(coil)
    angle = c_axis_angle_pure(coil, field)
    file_path = "/Users/paulhuslage/PhD/epos-simsopt/src/simsopt/examples/3_Advanced/inputs/Ic_angle_field_interpolation_with units.pck"
    B_norm = np.linalg.norm(field, axis=1)

    with open(file_path, "rb") as file:
        # Load the contents of the .pck file
        data = pickle.load(file)

    ic_interpolation = data['Ic_interpolation']
    ic_interpolation_reversed = data['Ic_interpolation_reversed']

    if JANUS:
        Ic = [ic_interpolation_reversed([a, f]) for a, f in zip(angle, B_norm)]
    else:
        Ic = [ic_interpolation([a, f]) for a, f in zip(angle, B_norm)]

    return Ic


def critical_current_pure(coil, JANUS=True):
    field = field_on_coils(coil)
    angle = c_axis_angle_pure(coil, field)
    file_path = "/Users/paulhuslage/PhD/epos-simsopt/src/simsopt/examples/3_Advanced/inputs/Ic_angle_field_interpolation_with units.pck"
    B_norm = np.linalg.norm(field, axis=1)

    with open(file_path, "rb") as file:
        # Load the contents of the .pck file
        data = pickle.load(file)

    ic_interpolation = data['Ic_interpolation']
    ic_interpolation_reversed = data['Ic_interpolation_reversed']

    if JANUS:
        Ic = [ic_interpolation_reversed([a, f]) for a, f in zip(angle, B_norm)]
    else:
        Ic = [ic_interpolation([a, f]) for a, f in zip(angle, B_norm)]

    return Ic


class CrtitcalCurrentOpt(Optimizable):
    """Optimizable class to optimize the critical on a ReBCO coil"""

    def __init__(self, coil, coils, a=0.05):
        self.coil = coil
        self.curve = coil.curve
        self.coils = coils
        self.a = a
        self.B_ext = BiotSavart(coils).set_points(self.curve.gamma()).B()
        self.B_self = 0
        self.B = 0
        self.alpha = coil.curve.rotation.alpha(coil.curve.quadpoints)
        self.J_jax = jit(lambda gamma, gammadash, gammadashdash,
                         current, phi, phidash, B_ext: force_opt_pure(gamma, gammadash, gammadashdash,
                                                                      current, phi, phidash, B_ext))

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


def field_alignment_pure(curve, field):
    angle = c_axis_angle_pure(curve, field)
    Ic = 0
    return Ic


def grad_field_alignment_pure(curve, field):
    angle = c_axis_angle_pure(curve, field)
    return 0


class FieldAlignment(Optimizable):

    def init(self, curve, field):
        Optimizable.__init__(self, depends_on=[curve])
        self.curve = curve
        self.field = field
        # new_dofs = np.zeros(len(dofs))
        # for i in range(len(dofs)):
        # new_dofs[i] = dofs[i]

    def J(self):

        return field_alignment_pure(self.curve, self.field)

    @derivative_dec
    def dJ(self):
        # return Derivative({self: lambda x0: gradfunction(x0, self.order, self.curves, self.n)})
        return grad_field_alignment_pure(self.curve, self.field)
