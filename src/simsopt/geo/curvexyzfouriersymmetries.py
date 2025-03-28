import jax.numpy as jnp
from jax import jvp
import numpy as np
from .curve import JaxCurve, Curve
from .jit import jit
import simsoptpp as sopp
from math import gcd

__all__ = ['CurveXYZFourierSymmetries', 'CurveXYZFourierSymmetriesPartner']


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
        xc = dofs[0        :   order+1]
        xs = dofs[order+1  : 2*order+1]
        yc = dofs[2*order+1: 3*order+2]
        ys = dofs[3*order+2: 4*order+2]
        zc = dofs[4*order+2: 5*order+3]
        zs = dofs[5*order+3:          ]
        
        xhat = np.sum(xc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(xs[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)
        yhat = np.sum(yc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(ys[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)
        
        z = np.sum(zc[None, :] * jnp.cos(2*jnp.pi*nfp*m*theta), axis=1) + np.sum(zs[None, :] * jnp.sin(2*jnp.pi*nfp*m[:, 1:]*theta[:, 1:]), axis=1)

    angle = 2 * jnp.pi * quadpoints * ntor
    x = jnp.cos(angle) * xhat - jnp.sin(angle) * yhat
    y = jnp.sin(angle) * xhat + jnp.cos(angle) * yhat

    gamma = jnp.zeros((len(quadpoints),3))
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
        pure = lambda dofs, points: jaxXYZFourierSymmetriescurve_pure(
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


class CurveXYZFourierSymmetriesPartner(sopp.Curve, Curve):
    """
    A class that represents a stellarator-symmetric flipped copy of another XYZFourierSymmetries object.

    DO NOT USE THIS FOR FANCY COIL OPTIMIZATION, I have not implemented all the vjps and jvps for this class.
    """
    def __init__(self, curve: CurveXYZFourierSymmetries):
        self.order = curve.order
        self.nfp = curve.nfp
        if not curve.stellsym:
            raise Exception('The CurveXYZFourierSymmetriesPartner can only be applied to stellarator-symmetric CurveXYZFourierSymmetries objects')
        self.stellsym = curve.stellsym
        self.ntor = curve.ntor
        self.curve = curve
        sopp.Curve.__init__(self, curve.quadpoints)
        Curve.__init__(self, depends_on=[curve])
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)
        self.gamma_pure = lambda dofs, points: jaxXYZFourierSymmetriescurve_pure(
            dofs, points, self.order, self.nfp, self.stellsym, self.ntor)
        
        self.gamma_jax = jit(lambda dofs: self.gamma_pure(dofs, points))
        self.gamma_impl_jax = jit(lambda dofs, p: self.gamma_pure(dofs, p))

        self.gammadash_pure = lambda x, q: jvp(lambda p: self.gamma_pure(x, p), (q,), (ones,))[1]
        self.gammadash_jax = jit(lambda x: self.gammadash_pure(x, points))

        self.gammadashdash_pure = lambda x, q: jvp(lambda p: self.gammadash_pure(x, p), (q,), (ones,))[1]
        self.gammadashdash_jax = jit(lambda x: self.gammadashdash_pure(x, points))

        self.gammadashdashdash_pure = lambda x, q: jvp(lambda p: self.gammadashdash_pure(x, p), (q,), (ones,))[1]
        self.gammadashdashdash_jax = jit(lambda x: self.gammadashdashdash_pure(x, points))

    
    @property
    def coefficients(self):
        coefficients = np.copy(self.curve.coefficients)
        # flip the sign of odd coefficients
        coefficients[1:self.order+1:2] = -coefficients[1:self.order+1:2]
        coefficients[self.order+1:self.order*2+1:2] = -coefficients[self.order+1:self.order*2+1:2]
        coefficients[self.order*2+1::2] = -coefficients[self.order*2+1::2]
        return coefficients


    def get_dofs(self):
        """
        RotatedCurve does not have any dofs of its own.
        This function returns null array
        """
        return np.array([])

    def set_dofs_impl(self, d):
        """
        RotatedCurve does not have any dofs of its own.
        This function does nothing.
        """
        pass

    def num_dofs(self):
        """
        This function returns the number of dofs associated to the curve.
        """
        return self.curve.num_dofs()
    
    def gamma(self):
        """
        This function returns the gamma values of the curve at the given quadpoints.
        """
        return self.gamma_impl_jax(self.coefficients, self.quadpoints)
    
    def gamma_impl(self, gamma, quadpoints):
        """
        This function returns the gamma values of the curve at the given quadpoints.
        """
        gamma[:,:] = self.gamma_impl_jax(self.coefficients, self.quadpoints)
    
    def gammadash_impl(self, gammadash):
        r"""
        This function returns :math:`\Gamma'(\varphi)`, where :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadash[:, :] = self.gammadash_jax(self.coefficients)
        
    def gammadashdash_impl(self, gammadashdash):
        r"""
        This function returns :math:`\Gamma''(\varphi)`, and :math:`\Gamma` are the x, y, z coordinates
        of the curve.
        """

        gammadashdash[:, :] = self.gammadashdash_jax(self.coefficients)
    

    


