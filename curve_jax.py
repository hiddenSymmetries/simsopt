"""
Pure JAX implementation of CurveXYZFourier as a JAX pytree.

DOF layout (per coordinate x, y, z):
    [c0, s1, c1, s2, c2, ..., s_order, c_order]
where ci = cosine coefficient for mode i, si = sine coefficient for mode i.
Full dofs array has shape (3*(2*order+1),).

Pytree leaves  (traced): quadpoints, dofs
Pytree aux     (static): order
"""

import jax
import jax.numpy as jnp


@jax.tree_util.register_pytree_node_class
class CurveXYZFourierJax:
    """
    JAX pytree implementation of CurveXYZFourier.

    The curve is parameterised by phi in [0, 1) and represented as:

        x(phi) = xc0 + sum_{j=1}^{order} [ xs_j sin(2*pi*j*phi) + xc_j cos(2*pi*j*phi) ]
        y(phi) = yc0 + ...
        z(phi) = zc0 + ...

    Args:
        quadpoints: 1-D array of sample points in [0, 1), shape (nquad,).
        dofs:       Fourier coefficients, shape (3*(2*order+1),), stored as
                    [xc0, xs1, xc1, ..., xs_order, xc_order,
                     yc0, ys1, yc1, ...,
                     zc0, zs1, zc1, ...].
        order:      int, maximum Fourier mode number (static).
    """

    def __init__(self, quadpoints, dofs, order: int):
        self.quadpoints = jnp.asarray(quadpoints, dtype=float)
        self.dofs = jnp.asarray(dofs, dtype=float)
        self.order = order  # static -- lives in aux_data, not in leaves

    # ------------------------------------------------------------------ #
    # JAX pytree protocol                                                   #
    # ------------------------------------------------------------------ #

    def tree_flatten(self):
        children = (self.quadpoints, self.dofs)
        aux_data = self.order
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        quadpoints, dofs = children
        return cls(quadpoints, dofs, aux_data)

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _basis(self):
        """
        Pre-compute sin/cos basis matrices.

        Returns:
            jrange : integer modes, shape (order,)
            sjp    : sin(2*pi*j*phi), shape (order, nquad)
            cjp    : cos(2*pi*j*phi), shape (order, nquad)
        """
        jrange = jnp.arange(1, self.order + 1)          # (order,)
        phi = 2.0 * jnp.pi * self.quadpoints             # (nquad,)
        jp = jrange[:, None] * phi[None, :]              # (order, nquad)
        return jrange, jnp.sin(jp), jnp.cos(jp)

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def gamma(self):
        """
        Curve position.

        Returns:
            Array of shape (nquad, 3).
        """
        k = 2 * self.order + 1
        jrange, sjp, cjp = self._basis()
        coords = [
            self.dofs[i * k]
            + jnp.sum(
                self.dofs[i * k + 2 * jrange - 1, None] * sjp
                + self.dofs[i * k + 2 * jrange,     None] * cjp,
                axis=0,
            )
            for i in range(3)
        ]
        return jnp.stack(coords, axis=-1)

    def _gammadash(self):
        """
        First derivative d(gamma)/d(phi), shape (nquad, 3).
        """
        k = 2 * self.order + 1
        jrange, sjp, cjp = self._basis()
        twopij = 2.0 * jnp.pi * jrange                  # (order,)
        coords = [
            jnp.sum(
                  self.dofs[i * k + 2 * jrange - 1, None] * twopij[:, None] * cjp
                - self.dofs[i * k + 2 * jrange,     None] * twopij[:, None] * sjp,
                axis=0,
            )
            for i in range(3)
        ]
        return jnp.stack(coords, axis=-1)

    def _gammadashdash(self):
        """
        Second derivative d2(gamma)/d(phi)2, shape (nquad, 3).
        """
        k = 2 * self.order + 1
        jrange, sjp, cjp = self._basis()
        twopij2 = (2.0 * jnp.pi * jrange) ** 2          # (order,)
        coords = [
            jnp.sum(
                - self.dofs[i * k + 2 * jrange - 1, None] * twopij2[:, None] * sjp
                - self.dofs[i * k + 2 * jrange,     None] * twopij2[:, None] * cjp,
                axis=0,
            )
            for i in range(3)
        ]
        return jnp.stack(coords, axis=-1)

    def _gammadashdashdash(self):
        """
        Third derivative d3(gamma)/d(phi)3, shape (nquad, 3).
        """
        k = 2 * self.order + 1
        jrange, sjp, cjp = self._basis()
        twopij3 = (2.0 * jnp.pi * jrange) ** 3          # (order,)
        coords = [
            jnp.sum(
                - self.dofs[i * k + 2 * jrange - 1, None] * twopij3[:, None] * cjp
                + self.dofs[i * k + 2 * jrange,     None] * twopij3[:, None] * sjp,
                axis=0,
            )
            for i in range(3)
        ]
        return jnp.stack(coords, axis=-1)

    def kappa(self):
        """
        Curvature kappa = ||gamma' x gamma''|| / ||gamma'||^3.

        Returns:
            Array of shape (nquad,).
        """
        d1 = self._gammadash()
        d2 = self._gammadashdash()
        return (
            jnp.linalg.norm(jnp.cross(d1, d2), axis=1)
            / jnp.linalg.norm(d1, axis=1) ** 3
        )

    def torsion(self):
        """
        Torsion tau = (gamma' x gamma'') . gamma''' / ||gamma' x gamma''||^2.

        Returns:
            Array of shape (nquad,).
        """
        d1 = self._gammadash()
        d2 = self._gammadashdash()
        d3 = self._gammadashdashdash()
        cross = jnp.cross(d1, d2)                        # (nquad, 3)
        return jnp.sum(cross * d3, axis=1) / jnp.sum(cross ** 2, axis=1)

    def frenet_frame(self):
        """
        Frenet-Serret frame (t, n, b) at each quadrature point.

            t = gamma' / |gamma'|
            n = t' / |t'|   (derivative w.r.t. arclength)
            b = t x n

        Returns:
            Tuple (t, n, b), each an array of shape (nquad, 3).
        """
        d1 = self._gammadash()
        d2 = self._gammadashdash()

        arclength = jnp.linalg.norm(d1, axis=1)          # (nquad,)

        t = d1 / arclength[:, None]

        # Derivative of t w.r.t. phi (chain rule on gamma'/|gamma'|):
        #   tdash = gamma''/|gamma'| - (gamma'·gamma'' / |gamma'|^3) * gamma'
        inner_d1_d2 = jnp.sum(d1 * d2, axis=1)           # (nquad,)
        tdash = (
            d2 / arclength[:, None]
            - (inner_d1_d2 / arclength ** 3)[:, None] * d1
        )

        n = tdash / jnp.linalg.norm(tdash, axis=1)[:, None]
        b = jnp.cross(t, n)
        return t, n, b
