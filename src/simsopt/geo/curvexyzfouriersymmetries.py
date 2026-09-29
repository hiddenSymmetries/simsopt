import jax.numpy as jnp
import numpy as np
from scipy.interpolate import CubicSpline
from .curve import JaxCurve
from .curverzfourier import CurveRZFourier
from math import gcd

__all__ = ['CurveXYZFourierSymmetries']


def jaxXYZFourierSymmetriescurve_pure(dofs, quadpoints, order, nfp, stellsym, ntor):

    theta, m = jnp.meshgrid(quadpoints, jnp.arange(order + 1), indexing='ij')

    if stellsym:
        xc = dofs[:order+1]
        ys = dofs[order+1:2*order+1]
        zs = dofs[2*order+1:]

        xhat = np.sum(xc[None, :] * jnp.cos(2 * jnp.pi * nfp*m*theta), axis=1)
        yhat = np.sum(ys[None, :] * jnp.sin(2 * jnp.pi * nfp*m[:, 1:]*theta[:, 1:]), axis=1)

        z = jnp.sum(zs[None, :] * jnp.sin(2*jnp.pi*nfp * m[:, 1:]*theta[:, 1:]), axis=1)
    else:
        xc = dofs[0: order+1]
        xs = dofs[order+1: 2*order+1]
        yc = dofs[2*order+1: 3*order+2]
        ys = dofs[3*order+2: 4*order+2]
        zc = dofs[4*order+2: 5*order+3]
        zs = dofs[5*order+3:]

        xhat = np.sum(xc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(xs[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)
        yhat = np.sum(yc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(ys[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)

        z = np.sum(zc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(zs[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)

    angle = 2 * jnp.pi * quadpoints * ntor
    x = jnp.cos(angle) * xhat - jnp.sin(angle) * yhat
    y = jnp.sin(angle) * xhat + jnp.cos(angle) * yhat

    gamma = jnp.zeros((len(quadpoints), 3))
    gamma = gamma.at[:, 0].add(x)
    gamma = gamma.at[:, 1].add(y)
    gamma = gamma.at[:, 2].add(z)
    return gamma


class CurveXYZFourierSymmetries(JaxCurve):
    r'''A curve representation that allows for stellarator and discrete rotational symmetries.  This class can be used to
    represent a helical coil that does not lie on a torus.  The coordinates of the curve are given by:

    .. math::
        x(\theta) &= \hat x(\theta)  \cos(2 \pi \theta n_{\text{tor}}) - \hat y(\theta)  \sin(2 \pi \theta n_{\text{tor}})\\
        y(\theta) &= \hat x(\theta)  \sin(2 \pi \theta n_{\text{tor}}) + \hat y(\theta)  \cos(2 \pi \theta n_{\text{tor}})\\
        z(\theta) &= \sum_{m=1}^{\text{order}} z_{s,m} \sin(2 \pi n_{\text{fp}} m \theta)

    where

    .. math::
        \hat x(\theta) &= x_{c, 0} + \sum_{m=1}^{\text{order}} x_{c,m} \cos(2 \pi n_{\text{fp}} m \theta)\\
        \hat y(\theta) &=            \sum_{m=1}^{\text{order}} y_{s,m} \sin(2 \pi n_{\text{fp}} m \theta)\\


    if the coil is stellarator symmetric.  When the coil is not stellarator symmetric, the formulas above
    become

    .. math::
        x(\theta) &= \hat x(\theta)  \cos(2 \pi \theta n_{\text{tor}}) - \hat y(\theta)  \sin(2 \pi \theta n_{\text{tor}})\\
        y(\theta) &= \hat x(\theta)  \sin(2 \pi \theta n_{\text{tor}}) + \hat y(\theta)  \cos(2 \pi \theta n_{\text{tor}})\\
        z(\theta) &= z_{c, 0} + \sum_{m=1}^{\text{order}} \left[ z_{c, m} \cos(2 \pi n_{\text{fp}} m \theta) + z_{s, m} \sin(2 \pi n_{\text{fp}} m \theta) \right]

    where

    .. math::
        \hat x(\theta) &= x_{c, 0} + \sum_{m=1}^{\text{order}} \left[ x_{c, m} \cos(2 \pi n_{\text{fp}} m \theta) +  x_{s, m} \sin(2 \pi n_{\text{fp}} m \theta) \right] \\
        \hat y(\theta) &= y_{c, 0} + \sum_{m=1}^{\text{order}} \left[ y_{c, m} \cos(2 \pi n_{\text{fp}} m \theta) +  y_{s, m} \sin(2 \pi n_{\text{fp}} m \theta) \right] \\

    Args:
        quadpoints: number of grid points/resolution along the curve,
        order:  how many Fourier harmonics to include in the Fourier representation,
        nfp: discrete rotational symmetry number, 
        stellsym: stellaratory symmetry if True, not stellarator symmetric otherwise,
        ntor: the number of times the curve wraps toroidally before biting its tail. Note,
              it is assumed that nfp and ntor are coprime.  If they are not coprime,
              then then the curve actually has nfp_new:=nfp // gcd(nfp, ntor),
              and ntor_new:=ntor // gcd(nfp, ntor).  The operator `//` is integer division.
              To avoid confusion, we assert that ntor and nfp are coprime at instantiation.
    '''

    def __init__(self, quadpoints, order, nfp, stellsym, ntor=1, **kwargs):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        def pure(dofs, points): return jaxXYZFourierSymmetriescurve_pure(
            dofs, points, order, nfp, stellsym, ntor)

        if gcd(ntor, nfp) != 1:
            raise Exception('nfp and ntor must be coprime')

        self.order = order
        self.nfp = nfp
        self.stellsym = stellsym
        self.ntor = ntor
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

    def to_RZFourier(self, order=None, quadpoints=None, nfp=None, n_samples=None):
        r"""
        Represent this curve as a :class:`~simsopt.geo.CurveRZFourier`, which
        is parametrized by the toroidal angle, :math:`\phi = 2\pi\theta`.

        This is only possible for a curve that goes around the torus once
        (``ntor=1``, or ``ntor=-1`` for a curve that runs towards decreasing
        :math:`\phi`) with :math:`\phi` monotonic along the curve, such as a
        magnetic axis. The curve is sampled over one field period, :math:`R`
        and :math:`Z` are interpolated as periodic functions of :math:`\phi`,
        and the result is fitted with :meth:`least_squares_fit`. The
        :class:`~simsopt.geo.CurveRZFourier` has the same ``stellsym`` as this curve.

        Args:
            order (int, optional): order of the CurveRZFourier. Defaults to ``self.order``.
            quadpoints (int or array, optional): quadrature points of the
                CurveRZFourier. Defaults to ``4*(2*order+1)*nfp`` points on [0, 1).
            nfp (int, optional): number of field periods of the CurveRZFourier.
                Defaults to ``self.nfp``. A multiple of ``self.nfp`` can be given
                if the curve has more symmetry than its representation.
            n_samples (int, optional): number of points at which this curve is
                sampled over one field period for the interpolation. Defaults to
                ``100*(2*self.order+1)``.

        Returns:
            CurveRZFourier: the curve in cylindrical representation.
        """
        if abs(self.ntor) != 1:
            raise ValueError(f"Only a curve that goes around the torus once can be represented as a CurveRZFourier, "
                             f"but this curve has ntor={self.ntor}.")
        if nfp is None:
            nfp = self.nfp
        if nfp % self.nfp != 0:
            raise ValueError(f"nfp={nfp} must be a multiple of the curve's nfp={self.nfp}.")
        if order is None:
            order = self.order
        if quadpoints is None:
            quadpoints = 4*(2*order + 1)*nfp
        if n_samples is None:
            n_samples = 100*(2*self.order + 1)
        period = 2*np.pi/self.nfp

        samples = CurveXYZFourierSymmetries(np.linspace(0, 1/self.nfp, n_samples, endpoint=False), self.order,
                                            self.nfp, self.stellsym, ntor=self.ntor, x0=self.x).gamma()
        phi = np.unwrap(np.arctan2(samples[:, 1], samples[:, 0]))
        if self.ntor < 0:
            samples, phi = samples[::-1], phi[::-1]
        if np.any(np.diff(phi) <= 0):
            raise ValueError("The toroidal angle is not monotonic along the curve, so it cannot be represented as a CurveRZFourier.")
        R = np.linalg.norm(samples[:, :2], axis=1)
        Z = samples[:, 2]
        # close the period for the periodic spline
        phi = np.append(phi, phi[0] + period)
        R_of_phi = CubicSpline(phi, np.append(R, R[0]), bc_type='periodic', extrapolate='periodic')
        Z_of_phi = CubicSpline(phi, np.append(Z, Z[0]), bc_type='periodic', extrapolate='periodic')

        curve = CurveRZFourier(quadpoints, order, nfp, self.stellsym)
        phi_quad = 2*np.pi*np.asarray(curve.quadpoints)
        R_quad = R_of_phi(phi_quad)
        curve.least_squares_fit(np.column_stack((R_quad*np.cos(phi_quad), R_quad*np.sin(phi_quad), Z_of_phi(phi_quad))))
        return curve
