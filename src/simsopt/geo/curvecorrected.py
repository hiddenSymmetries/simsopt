import numpy as np
import simsoptpp as sopp
from simsopt.geo.curve import Curve
from simsopt._core.derivative import Derivative
from jax.numpy import sin, cos, asarray
from jax import vjp
from .jit import jit

@jit
def apply_curve_correction_jax(shiftangles, g):
    shift = shiftangles[:3]
    alpha = shiftangles[3]
    beta = shiftangles[4]
    gamma = shiftangles[5]
    yaw = asarray([
        [+cos(alpha), -sin(alpha), 0],
        [+sin(alpha), +cos(alpha), 0],
        [0, 0, 1]])
    pitch = asarray([
        [+cos(beta), 0, +sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, +cos(beta)]])
    roll = asarray([
        [1, 0, 0],
        [0, +cos(gamma), -sin(gamma)],
        [0, +sin(gamma), +cos(gamma)]])

    res = g @ (yaw @ pitch @ roll) + shift[None, :]
    return res


correction_vjp0 = jit(lambda shiftangles, gamma, v: vjp(lambda x: apply_curve_correction_jax(x, gamma), shiftangles)[1](v)[0])
correction_vjp1 = jit(lambda shiftangles, gamma, v: vjp(lambda y: apply_curve_correction_jax(shiftangles, y), gamma)[1](v)[0])

@jit
def apply_curve_correction_noshift_jax(shiftangles, g):
    alpha = shiftangles[3]
    beta = shiftangles[4]
    gamma = shiftangles[5]
    yaw = asarray([
        [+cos(alpha), -sin(alpha), 0],
        [+sin(alpha), +cos(alpha), 0],
        [0, 0, 1]])
    pitch = asarray([
        [+cos(beta), 0, +sin(beta)],
        [0, 1, 0],
        [-sin(beta), 0, +cos(beta)]])
    roll = asarray([
        [1, 0, 0],
        [0, +cos(gamma), -sin(gamma)],
        [0, +sin(gamma), +cos(gamma)]])

    res = g @ (yaw @ pitch @ roll)
    return res


correction_noshift_vjp0 = jit(lambda shiftangles, gamma, v: vjp(lambda x: apply_curve_correction_noshift_jax(x, gamma), shiftangles)[1](v)[0])
correction_noshift_vjp1 = jit(lambda shiftangles, gamma, v: vjp(lambda y: apply_curve_correction_noshift_jax(shiftangles, y), gamma)[1](v)[0])

class CurveCorrected(sopp.Curve, Curve):

    def __init__(self, curve):
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, x0=np.zeros((6, )), depends_on=[curve])

    def recompute_bell(self, parent=None):
        self.invalidate_cache()

    def gamma_impl(self, gamma, quadpoints):
        self.curve.gamma_impl(gamma, quadpoints)
        gamma[:] = apply_curve_correction_jax(self.local_full_x, gamma)

    def gammadash_impl(self, gammadash):
        gammadash[:] = apply_curve_correction_noshift_jax(self.local_full_x, self.curve.gammadash())

    def gammadashdash_impl(self, gammadashdash):
        gammadashdash[:] = apply_curve_correction_noshift_jax(self.local_full_x, self.curve.gammadashdash())

    def gammadashdashdash_impl(self, gammadashdashdash):
        gammadashdashdash[:] = apply_curve_correction_noshift_jax(self.local_full_x, self.curve.gammadashdashdash())

    def dgamma_by_dcoeff_vjp(self, v):
        dself = Derivative({self: correction_vjp0(self.local_full_x, self.curve.gamma(), v)})
        dcurve = self.curve.dgamma_by_dcoeff_vjp(correction_vjp1(self.local_full_x, self.curve.gamma(), v))
        return dself + dcurve

    def dgammadash_by_dcoeff_vjp(self, v):
        dself = Derivative({self: correction_noshift_vjp0(self.local_full_x, self.curve.gammadash(), v)})
        dcurve = self.curve.dgammadash_by_dcoeff_vjp(correction_noshift_vjp1(self.local_full_x, self.curve.gammadash(), v))
        return dself + dcurve

    def dgammadashdash_by_dcoeff_vjp(self, v):
        dself = Derivative({self: correction_noshift_vjp0(self.local_full_x, self.curve.gammadashdash(), v)})
        dcurve = self.curve.dgammadashdash_by_dcoeff_vjp(correction_noshift_vjp1(self.local_full_x, self.curve.gammadashdash(), v))
        return dself + dcurve

    def dgammadashdashdash_by_dcoeff_vjp(self, v):
        dself = Derivative({self: correction_noshift_vjp0(self.local_full_x, self.curve.gammadashdashdash(), v)})
        dcurve = self.curve.dgammadashdashdash_by_dcoeff_vjp(correction_noshift_vjp1(self.local_full_x, self.curve.gammadashdashdash(), v))
        return dself + dcurve
