import numpy as np
from jax import grad, vjp
import jax.numpy as jnp
from .jit import jit

from .._core.graph_optimizable import Optimizable
from .._core.derivative import Derivative, derivative_dec


@jit
def curve_length_pure(l):
    """
    This function is used in a Python+Jax implementation of the curve length formula.
    """
    return jnp.mean(l)


class CurveLength(Optimizable):
    r"""
    CurveLength is a class that computes the length of a curve, i.e.

    .. math::
        J = \int_{\text{curve}}~dl.

    """

    def __init__(self, curve):
        self.curve = curve
        self.thisgrad = jit(lambda l: grad(curve_length_pure)(l))
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return curve_length_pure(self.curve.incremental_arclength())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """

        return self.curve.dincremental_arclength_by_dcoeff_vjp(
            self.thisgrad(self.curve.incremental_arclength()))

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_curvature_pure(kappa, gammadash, p, desired_kappa):
    """
    This function is used in a Python+Jax implementation of the curvature penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.maximum(kappa-desired_kappa, 0)**p * arc_length)


class LpCurveCurvature(Optimizable):
    r"""
    This class computes a penalty term based on the :math:`L_p` norm
    of the curve's curvature, and penalizes where the local curve curvature exceeds a threshold


    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \text{max}(\kappa - \kappa_0, 0)^p ~dl

    where :math:`\kappa_0` is a threshold curvature.  When ``desired_length`` is
    provided, :math:`\kappa_0 = 2 \pi / \text{desired_length}`, otherwise :math:`\kappa_0=0`.
    """

    def __init__(self, curve, p, desired_length=None):
        self.curve = curve
        super().__init__(depends_on=[curve])
        if desired_length is None:
            self.desired_kappa = 0
        else:
            radius = desired_length/(2*np.pi)
            self.desired_kappa = 1/radius

        self.J_jax = jit(lambda kappa, gammadash: Lp_curvature_pure(kappa, gammadash, p, self.desired_kappa))
        self.thisgrad0 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=0)(kappa, gammadash))
        self.thisgrad1 = jit(lambda kappa, gammadash: grad(self.J_jax, argnums=1)(kappa, gammadash))

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.kappa(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.kappa(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.kappa(), self.curve.gammadash())
        return self.curve.dkappa_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


@jit
def Lp_torsion_pure(torsion, gammadash, p):
    """
    This function is used in a Python+Jax implementation of the formula for the torsion penalty term.
    """
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return (1./p)*jnp.mean(jnp.abs(torsion)**p * arc_length)


class LpCurveTorsion(Optimizable):
    r"""
    LpCurveTorsion is a class that computes a penalty term based on the :math:`L_p` norm
    of the curve's torsion:


    .. math::
        J = \frac{1}{p} \int_{\text{curve}} \tau^p ~dl.

    """

    def __init__(self, curve, p):
        self.curve = curve

        self.J_jax = jit(lambda torsion, gammadash: Lp_torsion_pure(torsion, gammadash, p))
        self.thisgrad0 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=0)(torsion, gammadash))
        self.thisgrad1 = jit(lambda torsion, gammadash: grad(self.J_jax, argnums=1)(torsion, gammadash))
        super().__init__(depends_on=[curve])

    def J(self):
        """
        This returns the value of the quantity.
        """
        return self.J_jax(self.curve.torsion(), self.curve.gammadash())

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        grad0 = self.thisgrad0(self.curve.torsion(), self.curve.gammadash())
        grad1 = self.thisgrad1(self.curve.torsion(), self.curve.gammadash())
        return self.curve.dtorsion_by_dcoeff_vjp(grad0) + self.curve.dgammadash_by_dcoeff_vjp(grad1)

    return_fn_map = {'J': J, 'dJ': dJ}


def distance_pure(gamma1, l1, gamma2, l2, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the distance formula.
    """
    dists = jnp.sqrt(jnp.sum((gamma1[:, None, :] - gamma2[None, :, :])**2, axis=2))
    alen = jnp.linalg.norm(l1, axis=1) * jnp.linalg.norm(l2, axis=1)
    return jnp.sum(alen * jnp.maximum(minimum_distance-dists, 0)**2)/(gamma1.shape[0]*gamma2.shape[0])


class MinimumDistance(Optimizable):
    r"""
    MinimumDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} \sum_{j = 1}^{i-1} d_{i,j}

    where 

    .. math::
        d_{i,j} = \int_{\text{curve}_i} \int_{\text{curve}_j} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{r}_j \|_2)^2 ~dl_j ~dl_i\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{r}_j` are points on coils :math:`i` and :math:`j`, respectively.
    :math:`d_\min` is a desired threshold minimum intercoil distance.  This penalty term is zero when the points on coil :math:`i` and 
    coil :math:`j` lie more than :math:`d_\min` away from one another, for :math:`i, j \in \{1, \cdots, \text{num_coils}\}`

    """

    def __init__(self, curves, minimum_distance):
        self.curves = curves
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gamma1, l1, gamma2, l2: distance_pure(gamma1, l1, gamma2, l2, minimum_distance))
        self.thisgrad0 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=0)(gamma1, l1, gamma2, l2))
        self.thisgrad1 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=1)(gamma1, l1, gamma2, l2))
        self.thisgrad2 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=2)(gamma1, l1, gamma2, l2))
        self.thisgrad3 = jit(lambda gamma1, l1, gamma2, l2: grad(self.J_jax, argnums=3)(gamma1, l1, gamma2, l2))
        super().__init__(depends_on=curves)

    def recompute_bell(self, parent=None):
        from scipy.spatial import KDTree
        self.trees = []
        for i in range(len(self.curves)):
            self.trees.append(KDTree(self.curves[i].gamma()))

    def J(self):
        """
        This returns the value of the quantity.
        """
        res = 0
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            tree1 = self.trees[i]
            for j in range(i):
                tree2 = self.trees[j]
                # check whether there are any points that are actually closer
                # than minimum_distance
                dists = tree1.sparse_distance_matrix(tree2, self.minimum_distance)
                if len(dists) == 0:
                    continue

                gamma2 = self.curves[j].gamma()
                l2 = self.curves[j].gammadash()
                res += self.J_jax(gamma1, l1, gamma2, l2)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        for i in range(len(self.curves)):
            gamma1 = self.curves[i].gamma()
            l1 = self.curves[i].gammadash()
            tree1 = self.trees[i]
            for j in range(i):
                tree2 = self.trees[j]
                dists = tree1.sparse_distance_matrix(tree2, self.minimum_distance)
                # check whether there are any points that are actually closer
                # than minimum_distance
                if len(dists) == 0:
                    continue
                gamma2 = self.curves[j].gamma()
                l2 = self.curves[j].gammadash()

                dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gamma1, l1, gamma2, l2)
                dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gamma1, l1, gamma2, l2)
                dgamma_by_dcoeff_vjp_vecs[j] += self.thisgrad2(gamma1, l1, gamma2, l2)
                dgammadash_by_dcoeff_vjp_vecs[j] += self.thisgrad3(gamma1, l1, gamma2, l2)

        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}
