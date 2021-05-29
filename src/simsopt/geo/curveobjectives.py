from jax import grad, vjp
import jax.numpy as jnp
from .jit import jit

from .._core.optimizable import Optimizable


@jit
def curve_length_pure(l):
    return jnp.mean(l)


class CurveLength(Optimizable):

    def __init__(self, curve):
        self.curve = curve
        self.thisgrad = jit(lambda l: grad(curve_length_pure)(l))
        self.depends_on = ["curve"]

    def J(self):
        return curve_length_pure(self.curve.incremental_arclength())

    def dJ(self):
        return self.curve.dincremental_arclength_by_dcoeff_vjp(self.thisgrad(self.curve.incremental_arclength()))


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):

    def __init__(self, curve, p, desired_length=None):
        self.curve = curve
        self.depends_on = ["curve"]
        if desired_length is None:
            self.desired_kappa = 0
        else:
            radius = desired_length/(2*pi)
            self.desired_kappa = 1/radius

        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, self.desired_kappa))
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))

    def J(self):
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    def dJ(self):
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)


@jit
def Lp_torsion_pure(torsion, gammadash, p):
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.abs(torsion)**p * arc_length)


class LpCurveTorsion(Optimizable):

    def __init__(self, curve, p):
        self.curve = curve
        self.depends_on = ["curve"]

        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p))
        self.thisgrad0 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.thisgrad1 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))

    def J(self):
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    def dJ(self):
        grad0 = self.thisgrad0(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)


def distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1) * jnp.linalg.norm(l2, axis=1)
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])


class MinimumDistance(Optimizable):

    def __init__(self, curves, minimum_distance):
        self.curves = curves
        self.depends_on = ["curves"]
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gamma1, l1, gamma2, l2: distance_pure(gamma1, l1, gamma2, l2, minimum_distance))
        self.thisgrad0 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2))
        self.thisgrad1 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2))
        self.thisgrad2 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2))
        self.thisgrad3 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2))

    def J(self):
        res = 0
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            for j in range(i):
                gamma2 = self.curves[j].gamma()
                l2 = self.curves[j].gammadash()
                res += self.J_jax(gamma1, l1, gamma2, l2)
        return res

    def dJ(self):
        dgamma_by_dcoeff_vjp_vecs = [None for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [None for c in self.curves]
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            for j in range(i):
                gamma2 = self.curves[j].gamma()
                l2 = self.curves[j].gammadash()

                temp = self.thisgrad0(gamma1, l1, gamma2, l2)
                if dgamma_by_dcoeff_vjp_vecs[i] is None:
                    dgamma_by_dcoeff_vjp_vecs[i] = temp
                else:
                    dgamma_by_dcoeff_vjp_vecs[i] += temp

                temp = self.thisgrad1(gamma1, l1, gamma2, l2)
                if dgammadash_by_dcoeff_vjp_vecs[i] is None:
                    dgammadash_by_dcoeff_vjp_vecs[i] = temp
                else:
                    dgammadash_by_dcoeff_vjp_vecs[i] += temp

                temp = self.thisgrad2(gamma1, l1, gamma2, l2)
                if dgamma_by_dcoeff_vjp_vecs[j] is None:
                    dgamma_by_dcoeff_vjp_vecs[j] = temp
                else:
                    dgamma_by_dcoeff_vjp_vecs[j] += temp
                temp = self.thisgrad3(gamma1, l1, gamma2, l2)
                if dgammadash_by_dcoeff_vjp_vecs[j] is None:
                    dgammadash_by_dcoeff_vjp_vecs[j] = temp
                else:
                    dgammadash_by_dcoeff_vjp_vecs[j] += temp

        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return res
