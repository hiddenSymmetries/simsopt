import numpy as np
import jax.numpy as jnp
from jax import vjp, jvp, grad
import simsoptpp as sopp
from simsopt.geo.jit import jit
from simsopt.geo import ZeroRotation, Curve
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo.curveobjectives import Lp_torsion_pure

class LPBinormalCurvatureStrainPenalty(Optimizable):
    def __init__(self, framedcurve, width=1e-3, p=2, threshold=0):
        self.framedcurve = framedcurve
        self.strain = StrainOpt(framedcurve,width)
        self.width = width
        self.p = p 
        self.threshold = threshold 
        self.J_jax = jit(lambda binorm, gammadash: Lp_torsion_pure(binorm, gammadash, p, threshold))
        self.grad0 = jit(lambda binorm, gammadash: grad(self.J_jax, argnums=0)(binorm, gammadash))
        self.grad1 = jit(lambda binorm, gammadash: grad(self.J_jax, argnums=1)(binorm, gammadash))
        super().__init__(depends_on=[framedcurve])

    def J(self):
        return self.J_jax(self.strain.binormal_curvature_strain(),self.framedcurve.curve.gammadash())

    @derivative_dec
    def dJ(self):
        grad0 = self.grad0(self.strain.binormal_curvature_strain(),self.framedcurve.curve.gammadash())
        grad1 = self.grad1(self.strain.binormal_curvature_strain(),self.framedcurve.curve.gammadash())
        vjp0 = self.strain.binormstrain_vjp(self.framedcurve.frame_binormal_curvature(),self.width,grad0)
        return self.framedcurve.dframe_binormal_curvature_by_dcoeff_vjp(vjp0) \
             + self.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1)
    
    return_fn_map = {'J': J, 'dJ': dJ}

class LPTorsionalStrainPenalty(Optimizable):

    def __init__(self, framedcurve, width=1e-3, p=2, threshold=0):
        self.framedcurve = framedcurve
        self.strain = StrainOpt(framedcurve,width)
        self.width = width
        self.p = p 
        self.threshold = threshold 
        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p, threshold))
        self.grad0 = jit(lambda torsion, gammadash: grad(
            self.J_jax, argnums=0)(torsion, gammadash))
        self.grad1 = jit(lambda torsion, gammadash: grad(
            self.J_jax, argnums=1)(torsion, gammadash))
        super().__init__(depends_on=[framedcurve])

    def J(self):
        return self.J_jax(self.strain.torsional_strain(),self.framedcurve.curve.gammadash())

    @derivative_dec
    def dJ(self):
        grad0 = self.grad0(self.strain.torsional_strain(),self.framedcurve.curve.gammadash())
        grad1 = self.grad1(self.strain.torsional_strain(),self.framedcurve.curve.gammadash())
        vjp0 = self.strain.torstrain_vjp(self.framedcurve.frame_torsion(),self.width,grad0)
        return self.framedcurve.dframe_torsion_by_dcoeff_vjp(vjp0) \
             + self.framedcurve.curve.dgammadash_by_dcoeff_vjp(grad1)
    
    return_fn_map = {'J': J, 'dJ': dJ}

class StrainOpt(Optimizable):

    def __init__(self, framedcurve, width=1e-3):
        self.framedcurve = framedcurve
        self.width = width
        self.torstrain_jax = jit(lambda torsion, width: torstrain_pure(
            torsion, width))
        self.binormstrain_jax = jit(lambda binorm, width: binormstrain_pure(
            binorm, width))
        self.torstrain_vjp = jit(lambda torsion, width, v: vjp(
            lambda g: torstrain_pure(g, width), torsion)[1](v)[0])
        self.binormstrain_vjp = jit(lambda binorm, width, v: vjp(
            lambda g: binormstrain_pure(g, width), binorm)[1](v)[0])

        super().__init__(depends_on=[framedcurve])

    def torsional_strain(self):
        return self.torstrain_jax(self.framedcurve.frame_torsion(),self.width)

    def binormal_curvature_strain(self):
        return self.binormstrain_jax(self.framedcurve.frame_binormal_curvature(),self.width)

@jit
def torstrain_pure(torsion, width):
    return torsion**2 * width**2 / 12

@jit
def binormstrain_pure(binorm, width):
    return (width / 2) * jnp.abs(binorm)
