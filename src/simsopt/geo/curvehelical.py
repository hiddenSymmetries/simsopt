import jax.numpy as jnp
from math import pi
import numpy as np
from .curve import JaxCurve

__all__ = ['CurveHelical']


def jaxHelicalfouriercurve_pure(dofs, quadpoints, order, n0, l0, R0, r0):
    A = dofs[:order]
    B = dofs[order:]
    phi = quadpoints*2*pi*l0
    m, phiV = jnp.meshgrid(jnp.arange(order), phi)
    AcosArray = jnp.sum(A*jnp.cos(m*phiV*n0/l0), axis=1)
    BsinArray = jnp.sum(B*jnp.sin(m*phiV*n0/l0), axis=1)
    eta = n0*phi/l0+AcosArray+BsinArray
    gamma = jnp.zeros((len(quadpoints), 3))
    gamma = gamma.at[:, 0].add((R0 + r0 * jnp.cos(eta)) * jnp.cos(phi))
    gamma = gamma.at[:, 1].add((R0 + r0 * jnp.cos(eta)) * jnp.sin(phi))
    gamma = gamma.at[:, 2].add(-r0 * jnp.sin(eta))
    return gamma


class CurveHelical(JaxCurve):
    r'''Curve representation of a helical coil.
    The helical coil positions are specified by a poloidal angle eta that is a function of the toroidal angle phi with Fourier coefficients :math:`A_k` and :math:`B_k`.
    The poloidal angle is represented as 

    .. math:: 
        \eta = m_0 \phi/l_0 + \sum_k A_k \cos(n_0 \phi k/l_0) + B_k \sin(n_0 \phi k/l_0)

    Args:
        quadpoints: number of grid points/resolution along the curve;
        order:  number of fourier coefficients, i.e., the length of the array A (or B);
        n0:  toroidal periodicity/number of field periods
        l0:  number of :math:`2\pi` turns in :math:`\phi`
        R0:  major radius
        r0:  minor radius
    '''

    def __init__(self, quadpoints, order, n0, l0, R0, r0):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        pure = lambda dofs, points: jaxHelicalfouriercurve_pure(
            dofs, points, order, n0, l0, R0, r0)
        self.order = order
        self.n0 = n0
        self.l0 = l0
        self.R0 = R0
        self.r0 = r0
        self.coefficients = [np.zeros((order,)), np.zeros((order,))]
        super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients))

    def num_dofs(self):
        return 2*self.order

    def get_dofs(self):
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        order = int(len(dofs)/2)
        for i in range(2):
            self.coefficients[i] = dofs[i*order:(i+1)*order]

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["quadpoints"] = list(self.quadpoints)
        d["order"] = self.order
        d["n0"] = self.n0
        d["l0"] = self.l0
        d["R0"] = self.R0
        d["r0"] = self.r0
        d["x0"] = list(self.local_full_x)
        return d

    @classmethod
    def from_dict(cls, d):
        curve = cls(d["quadpoints"], d["order"], d["n0"], d["l0"], d["R0"],
                    d["r0"])
        curve.local_full_x = d["x0"]
        return curve
