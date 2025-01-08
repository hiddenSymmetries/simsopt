import numpy as np
import jax.numpy as jnp

import simsoptpp as sopp
from .curve import Curve, JaxCurve

__all__ = ['CurveRZFourier', 'JaxCurveRZFourier']


class CurveRZFourier(sopp.CurveRZFourier, Curve):
    r"""
    ``CurveRZFourier`` is a curve that is represented in cylindrical
    coordinates using the following Fourier series:

    .. math::
       r(\phi) &= \sum_{m=0}^{\text{order}} r_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(n_{\text{fp}} m \phi) \\
       z(\phi) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(n_{\text{fp}} m \phi)

    If ``stellsym = True``, then the :math:`\sin` terms for :math:`r` and the :math:`\cos` terms for :math:`z` are zero.

    For the ``stellsym = False`` case, the dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, z_{c,0},....]

    or in the ``stellsym = True`` case they are stored

    .. math::
       [r_{c,0},...,r_{c,order},z_{s,1},...,z_{s,order}]
    """

    def __init__(self, quadpoints, order, nfp, stellsym, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1./nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveRZFourier.__init__(self, quadpoints, order, nfp, stellsym)
        if dofs is None:
            Curve.__init__(self, external_dof_setter=CurveRZFourier.set_dofs_impl,
                           x0=self.get_dofs())
        else:
            Curve.__init__(self, external_dof_setter=CurveRZFourier.set_dofs_impl,
                           dofs=dofs)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveRZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveRZFourier.set_dofs(self, dofs)



def jaxCurveRZFourier_pure(dofs, quadpoints, order, nfp, stellsym):
    """The gamma() method for the JaxCurveRZFourier class. The code is
    written as a 'pure' jax function and is ammenable to autodifferentation.

    Args:
        dofs (array): 1D-array of dofs. Should be ordered as in the JaxCurveRZFourier
            class.
        quadpoints (array): 1D-array of quadrature points in [0, 1/nfp].
        order (int): Maximum mode number of the expansion.
        nfp (int): Number of field periods.
        stellsym (bool): True for stellarator symmetry, False otherwise.

    Returns:
        gamma: (nphi, 3) jax numpy array of points on the curve.
    """

    phi1d = 2 * jnp.pi * quadpoints
    phi, m = jnp.meshgrid(phi1d, jnp.arange(order + 1), indexing='ij')

    if stellsym:
        rc = dofs[:order+1]
        zs = dofs[order+1:]

        r = jnp.sum(rc[None, :] * jnp.cos(m * nfp * phi), axis=1)
        z = jnp.sum(zs[None, :] * jnp.sin(m[:, 1:] * nfp * phi[:, 1:]), axis=1)

    else:
        rc = dofs[0        :   order+1]
        rs = dofs[order+1  : 2*order+1]
        zc = dofs[2*order+1: 3*order+2]
        zs = dofs[3*order+2:          ]
        
        r = (jnp.sum(rc[None, :] * jnp.cos(m * nfp * phi), axis=1) +
             jnp.sum(rs[None, :] * jnp.sin(m[:, 1:] * nfp * phi[:, 1:]), axis=1)
             )    
        z = (jnp.sum(zc[None, :] * jnp.cos(m * nfp * phi), axis=1) +
             jnp.sum(zs[None, :] * jnp.sin(m[:, 1:] * nfp * phi[:, 1:]), axis=1)
             )

    x = r * jnp.cos(phi1d)
    y = r * jnp.sin(phi1d)

    gamma = jnp.zeros((len(quadpoints),3))
    gamma = gamma.at[:, 0].add(x)
    gamma = gamma.at[:, 1].add(y)
    gamma = gamma.at[:, 2].add(z)
    return gamma


class JaxCurveRZFourier(JaxCurve):
    r'''A ``CurveRZFourier`` that is based off of the ``JaxCurve`` class. 
    ``CurveRZFourier``is a curve that is represented in cylindrical coordinates using the following Fourier series:

    .. math::
       r(\phi) &= \sum_{m=0}^{\text{order}} r_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} r_{s,m}\sin(n_{\text{fp}} m \phi) \\
       z(\phi) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(n_{\text{fp}} m \phi) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(n_{\text{fp}} m \phi)

    If ``stellsym = True``, then the :math:`\sin` terms for :math:`r` and the :math:`\cos` terms for :math:`z` are zero.

    For the ``stellsym = False`` case, the dofs are stored in the order

    .. math::
       [r_{c,0}, \cdots, r_{c,\text{order}}, r_{s,1}, \cdots, r_{s,\text{order}}, z_{c,0},....]

    or in the ``stellsym = True`` case they are stored

    .. math::
       [r_{c,0},...,r_{c,order},z_{s,1},...,z_{s,order}]

    Example usage of JaxCurve's can be found in examples/2_Intermediate/jax_curve_example.py.

    Args:
        quadpoints (int, array): Either the number of quadrature points or a 
            1D-array of quadrature points.
        order (int): Maximum mode number of the expansion.
        nfp (int): Number of field periods.
        stellsym (bool): True for stellarator symmetry, False otherwise.
    '''

    def __init__(self, quadpoints, order, nfp, stellsym, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1./nfp, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        pure = lambda dofs, points: jaxCurveRZFourier_pure(
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
            r_cos_names = [f'rc({i})' for i in range(0, order + 1)]
            r_names = r_cos_names
            z_sin_names = [f'zs({i})' for i in range(1, order + 1)]
            z_names = z_sin_names
        else:
            r_names = ['rc(0)']
            r_cos_names = [f'rc({i})' for i in range(1, order + 1)]
            r_sin_names = [f'rs({i})' for i in range(1, order + 1)]
            r_names += r_cos_names + r_sin_names
            z_names = ['zc(0)']
            z_cos_names = [f'zc({i})' for i in range(1, order + 1)]
            z_sin_names = [f'zs({i})' for i in range(1, order + 1)]
            z_names += z_cos_names + z_sin_names

        return r_names + z_names

    def num_dofs(self):
        return (self.order+1) + self.order if self.stellsym else 4*self.order+2

    def get_dofs(self):
        return self.coefficients

    def set_dofs_impl(self, dofs):
        self.coefficients[:] = dofs[:]