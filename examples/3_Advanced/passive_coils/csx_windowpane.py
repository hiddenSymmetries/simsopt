from simsopt.geo import JaxCurve
import jax.numpy as jnp
import numpy as np


def pure(dofs, points, order, Rvessel):
    zc = dofs[0:order+1]
    zs = dofs[order+1:2*order+1]
    phic = dofs[2*order+1:3*order+2]
    phis = dofs[3*order+2:4*order+2]

    z = jnp.zeros((len(points),))
    phi = jnp.zeros((len(points),))
    for j in range(order+1):
        if j > 0:
            z = z + (zs[j-1] * jnp.sin(2*jnp.pi*j*points))
            phi = phi + (phis[j-1] * jnp.sin(2*jnp.pi*j*points))
        z = z + (zc[j] * jnp.cos(2*jnp.pi*j*points))
        phi = phi + (phic[j] * jnp.cos(2*jnp.pi*j*points))

    gamma = jnp.zeros((len(points), 3))
    gamma = gamma.at[:, 1].set(Rvessel * jnp.cos(phi))
    gamma = gamma.at[:, 2].set(Rvessel * jnp.sin(phi))
    gamma = gamma.at[:, 0].set(z)
    return gamma


class csx_windowpane_curve_constant_R(JaxCurve):
    """
    OrientedCurveXYZFourier is a translated and rotated Curve.
    """

    def __init__(self, quadpoints, order, Rvessel, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)

        self.order = order
        self.Rvessel = Rvessel
        def gamma(dofs, points): return pure(dofs, points, self.order, self.Rvessel)

        self.coefficients = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]
        if dofs is None:
            super().__init__(quadpoints, gamma, x0=np.concatenate(self.coefficients),
                             external_dof_setter=csx_windowpane_curve_constant_R.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, gamma, dofs=dofs,
                             external_dof_setter=csx_windowpane_curve_constant_R.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 2*(2*self.order+1)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:self.order+1]              # zc
        self.coefficients[1][:] = dofs[self.order+1:2*self.order+1]   # zs
        self.coefficients[2][:] = dofs[2*self.order+1:3*self.order+2]  # phi_c
        self.coefficients[3][:] = dofs[3*self.order+2:4*self.order+2]  # phi_s

    def _make_names(self):
        dofs_name = []
        for c in ['z', 'phi']:
            dofs_name += [f'{c}c(0)']
            for even_or_odd in ['c', 's']:
                for j in range(1, self.order+1):
                    dofs_name += [f'{c}{even_or_odd}({j})']

        return dofs_name


def pure2(dofs, points, order, Zvessel):
    xc = dofs[0:order+1]
    xs = dofs[order+1:2*order+1]
    yc = dofs[2*order+1:3*order+2]
    ys = dofs[3*order+2:4*order+2]

    x = jnp.zeros((len(points),))
    y = jnp.zeros((len(points),))
    for j in range(order+1):
        if j > 0:
            x = x + (xs[j-1] * jnp.sin(2*jnp.pi*j*points))
            y = y + (ys[j-1] * jnp.sin(2*jnp.pi*j*points))
        x = x + (xc[j] * jnp.cos(2*jnp.pi*j*points))
        y = y + (yc[j] * jnp.cos(2*jnp.pi*j*points))

    gamma = jnp.zeros((len(points), 3))
    gamma = gamma.at[:, 1].set(x)
    gamma = gamma.at[:, 2].set(y)
    gamma = gamma.at[:, 0].set(Zvessel)

    return gamma


class csx_windowpane_curve_constant_Z(JaxCurve):
    """
    OrientedCurveXYZFourier is a translated and rotated Curve.
    """

    def __init__(self, quadpoints, order, Zvessel, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = jnp.linspace(0, 1, quadpoints, endpoint=False)

        self.order = order
        self.Zvessel = Zvessel
        def gamma(dofs, points): return pure2(dofs, points, self.order, self.Zvessel)

        self.coefficients = [np.zeros((order+1,)), np.zeros((order,)), np.zeros((order+1,)), np.zeros((order,))]
        if dofs is None:
            super().__init__(quadpoints, gamma, x0=np.concatenate(self.coefficients),
                             external_dof_setter=csx_windowpane_curve_constant_Z.set_dofs_impl,
                             names=self._make_names())
        else:
            super().__init__(quadpoints, gamma, dofs=dofs,
                             external_dof_setter=csx_windowpane_curve_constant_Z.set_dofs_impl,
                             names=self._make_names())

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 2*(2*self.order+1)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        self.coefficients[0][:] = dofs[0:self.order+1]                # xc
        self.coefficients[1][:] = dofs[self.order+1:2*self.order+1]   # xs
        self.coefficients[2][:] = dofs[2*self.order+1:3*self.order+2]  # yc
        self.coefficients[3][:] = dofs[3*self.order+2:4*self.order+2]  # ys

    def _make_names(self):
        dofs_name = []
        for c in ['x', 'y']:
            dofs_name += [f'{c}c(0)']
            for even_or_odd in ['c', 's']:
                for j in range(1, self.order+1):
                    dofs_name += [f'{c}{even_or_odd}({j})']

        return dofs_name
