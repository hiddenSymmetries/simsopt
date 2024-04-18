import jax.numpy as jnp
import numpy as np
from .curve import JaxCurve

__all__ = ['CurveXYZHelical']


def jaxXYZHelicalFouriercurve_pure(dofs, quadpoints, order, nfp, stellsym):
    
    if stellsym:
        xc = dofs[:order+1]
        ys = dofs[order+1:2*order+1]
        zs = dofs[2*order+1:]
        
        theta, m = jnp.meshgrid(quadpoints, jnp.arange(order+1), indexing='ij')
        xhat = np.sum(xc[None, :] * jnp.cos(2 * jnp.pi * nfp*m*theta), axis=1)
        yhat = np.sum(ys[None, :] * jnp.sin(2 * jnp.pi * nfp*m[:, 1:]*theta[:, 1:]), axis=1)
        
        x = jnp.cos(2*jnp.pi*quadpoints) * xhat - jnp.sin(2*jnp.pi*quadpoints) * yhat
        y = jnp.sin(2*jnp.pi*quadpoints) * xhat + jnp.cos(2*jnp.pi*quadpoints) * yhat
        z = jnp.sum(zs[None, :] * jnp.sin(2*jnp.pi*nfp * m[:, 1:]*theta[:, 1:]), axis=1)
    else:
        xc = dofs[0        :   order+1]
        xs = dofs[order+1  : 2*order+1]
        yc = dofs[2*order+1: 3*order+2]
        ys = dofs[3*order+2: 4*order+2]
        zc = dofs[4*order+2: 5*order+3]
        zs = dofs[5*order+3:          ]
        
        theta, m = jnp.meshgrid(quadpoints, jnp.arange(order+1), indexing='ij')
        xhat = np.sum(xc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(xs[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)
        yhat = np.sum(yc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(ys[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)
        
        x = jnp.cos(2*jnp.pi*quadpoints) * xhat - jnp.sin(2*jnp.pi*quadpoints) * yhat
        y = jnp.sin(2*jnp.pi*quadpoints) * xhat + jnp.cos(2*jnp.pi*quadpoints) * yhat
        z = np.sum(zc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(zs[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)

    gamma = jnp.zeros((len(quadpoints),3))
    gamma = gamma.at[:, 0].add(x)
    gamma = gamma.at[:, 1].add(y)
    gamma = gamma.at[:, 2].add(z)
    return gamma


class CurveXYZHelical(JaxCurve):
    r'''A curve representation for a helical coil that does not lie on a torus.  The coordinates of the curve are given by:
        .. math::
            \hat x(\theta) &= x_{c, 0} + \sum_{m=1}^{\text{order}} x_{c,m} cos(2pi n_{\text{fp}} m \theta)\\
            \hat y(\theta) &=            \sum_{m=1}^{\text{order}} y_{s,m} sin(2pi n_{\text{fp}} m \theta)\\
            x(\theta) &= \hat x(\theta)  \cos(2pi \theta) - \hat y(\theta)  \sin(2pi \theta)\\
            y(\theta) &= \hat x(\theta)  \sin(2pi \theta) + \hat y(\theta)  \cos(2pi \theta)\\
            z(\theta) &= \sum_{i=1}^{\text{order}} z_{s,m} sin(2pi n_{\text{fp}} \theta)
        if the coil is stellarator symmetric.  When the coil is not stellarator symmetric, the formulas above
        become
        .. math::
            \hat x(\theta) &= x_{c, 0} + \sum_{m=1}^{\text{order}} x_{c, m} cos(2pi m \theta) +  x_{s, m} sin(2pi m \theta)\\
            \hat y(\theta) &= y_{c, 0} + \sum_{m=1}^{\text{order}} y_{c, m} cos(2pi m \theta) +  y_{s, m} sin(2pi m \theta)\\
            x(\theta) &= \hat x(\theta)  \cos(2pi \theta) - \hat y(\theta)  \sin(2pi \theta)\\
            y(\theta) &= \hat x(\theta)  \sin(2pi \theta) + \hat y(\theta)  \cos(2pi \theta)\\
            z(\theta) &= z_{c, 0} + \sum_{m=1}^{\text{order}} z_{c, m} cos(2pi \theta) + z_{s, m} sin(2pi \theta)
        
        Args:
            quadpoints: number of grid points/resolution along the curve,
            order:  how many Fourier harmonics to include in the Fourier representation,
            nfp: discrete rotational symmetry number, 
            stellsym: stellaratory symmetry if True, not stellarator symmetric otherwise.
        '''

    def __init__(self, quadpoints, order, nfp, stellsym, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        pure = lambda dofs, points: jaxXYZHelicalFouriercurve_pure(
            dofs, points, order, nfp, stellsym)
        
        self.order = order
        self.nfp = nfp
        self.stellsym = stellsym
        self.coefficients = np.zeros(self.num_dofs())
        if "dofs" not in kwargs:
            if "x0" not in kwargs:
                kwargs["x0"] = self.coefficients
            else:
                self.set_dofs_impl(kwargs["x0"])

        super().__init__(quadpoints, pure, names=self._make_names(order), **kwargs)

    def _make_names(self, order):
        if self.stellsym:
            x_cos_names = [f'xc({i})' for i in range(0, order + 1)]
            x_names = x_cos_names
            y_sin_names = [f'ys({i})' for i in range(1, order + 1)]
            y_names = y_sin_names
            z_sin_names = [f'zs({i})' for i in range(1, order + 1)]
            z_names = z_sin_names
        else:
            x_names = ['xc(0)']
            x_cos_names = [f'xc({i})' for i in range(1, order + 1)]
            x_sin_names = [f'xs({i})' for i in range(1, order + 1)]
            x_names += x_cos_names + x_sin_names
            y_names = ['yc(0)']
            y_cos_names = [f'yc({i})' for i in range(1, order + 1)]
            y_sin_names = [f'ys({i})' for i in range(1, order + 1)]
            y_names += y_cos_names + y_sin_names
            z_names = ['zc(0)']
            z_cos_names = [f'zc({i})' for i in range(1, order + 1)]
            z_sin_names = [f'zs({i})' for i in range(1, order + 1)]
            z_names += z_cos_names + z_sin_names

        return x_names + y_names + z_names

    def num_dofs(self):
        return (self.order+1) + self.order + self.order if self.stellsym else 3*(2*self.order+1)

    def get_dofs(self):
        return self.coefficients

    def set_dofs_impl(self, dofs):
        self.coefficients[:] = dofs[:]

