"""Implements the force on a coil in its own magnetic field and the field of other coils."""
import math
from scipy import constants
import numpy as np
import jax.numpy as jnp
from jax import grad
from simsopt.field import BiotSavart, Coil
from simsopt.geo.jit import jit
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

Biot_savart_prefactor = constants.mu_0 / 4 / np.pi


def field_on_coils_pure(gamma, gammadash, gammadashdash, phi, phidash, current,  a=0.05):
    """Regularized field for optimization"""
    I = current
    n_quad = phidash.shape[0]

    A = (1 / jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**3 * jnp.cross(gammadashdash, gammadash)) * (
        1 + jnp.log(a / (8 * math.e**0.25 * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis])))
    b_int_phi_dash = jnp.zeros((n_quad, n_quad, 3))

    for i in range(len(phidash)):
        b_int_phi_dash.at[i].set((jnp.cross(gammadash[i], (gamma-gamma[i]))) / (jnp.linalg.norm(gamma - gamma[i])**2 + a/jnp.sqrt(math.e)) ** 1.5
                                 + (jnp.cross(gammadashdash, gammadash) * (1 - jnp.cos(phidash[i] - phi)[:, jnp.newaxis]) / (2 * (
                                     1 - jnp.cos(phidash[i] - phi)) * jnp.linalg.norm(gammadash)**2 + a**2/jnp.sqrt(math.e))[:, jnp.newaxis]))

    integral = jnp.trapz(b_int_phi_dash, phidash, axis=0)

    b_reg = I * Biot_savart_prefactor * (integral + A)

    return b_reg


def field_on_coils_rect_pure(gamma, gammadash, gammadashdash, phi, phidash, current, a=0.05, b=0.03):
    """Regularized field for optimization of a coil with rectangular cross section"""
    I = current
    n_quad = phidash.shape[0]

    k = (4 * b) / (3 * a) * jnp.arctan(a/b) + (4*a)/(3*b)*jnp.arctan(b/a) + \
        (b**2)/(6*a**2)*jnp.log(b/a) + (a**2)/(6*b**2)*jnp.log(a/b) - \
        (a**4 - 6*a**2*b**2 + b**4)/(6*a**2*b**2)*jnp.log(a/b+b/a)

    delta = jnp.exp(-25/6 + k)
    A = (jnp.cross(gammadash, gammadashdash)) / (2*jnp.linalg.norm(gammadash, axis=1)
                                                 [:, jnp.newaxis]**3) * (-2 + jnp.log(64 / (delta * a * b) * jnp.linalg.norm(gammadash, axis=1)[:, jnp.newaxis]**2))

    b_int_phi_dash = jnp.zeros((n_quad, n_quad, 3))

    for i in range(len(phidash)):
        b_int_phi_dash.at[i].set((jnp.cross(gammadash[i], (gamma-gamma[i]))) / (jnp.linalg.norm(gamma - gamma[i])**2 + delta + a * b) ** 1.5
                                 + (jnp.cross(gammadashdash, gammadash) * (1 - jnp.cos(phidash[i] - phi)[:, jnp.newaxis]) / ((
                                     2 - 2 * jnp.cos(phidash[i] - phi)) * jnp.linalg.norm(gammadash)**2 + delta * a * b)[:, jnp.newaxis]))

    integral = jnp.trapz(b_int_phi_dash, phidash, axis=0)

    b_reg = I * Biot_savart_prefactor * (integral + A)

    return b_reg


def field_from_other_coils_pure(gamma, curves, currents, a=0.05):
    """field on one coil from the other coils"""
    coils = [Coil(curve, current) for curve, current in zip(curves, currents)]
    b_ext = BiotSavart(coils)
    b_ext.set_points(gamma)
    return b_ext.B()


def coil_force_pure(B, I, t, a=0.05):
    """force on coil for optimization"""
    force = jnp.cross(I*t, B)
    return force


@jit
def force_opt_pure(gamma, gammadash, gammadashdash,
                   current, phi, phidash, b_ext):
    """Cost function for force optimization. Optimize for peak self force on the coil (so far)"""
    t = gammadash / jnp.linalg.norm(gammadash)
    b_self = field_on_coils_rect_pure(
        gamma, gammadash, gammadashdash, phi, phidash, current)
    b_tot = b_self + b_ext
    force = coil_force_pure(b_tot, current, t)
    f_norm = jnp.linalg.norm(force, axis=1)
    result = jnp.max(f_norm)
    # result = jnp.sum(f_norm)
    return result


class ForceOpt(Optimizable):
    """Optimizabke class to optimize forces on a coil"""
    def __init__(self, coil, coils, a=0.05):
        self.coil = coil
        self.curve = coil.curve
        self.coils = coils
        self.a = a
        self.B_ext = BiotSavart(coils).set_points(self.curve.gamma()).B()
        self.B_self = 0
        self.B = 0
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
