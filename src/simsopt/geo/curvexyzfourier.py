from math import pi
from itertools import chain

import numpy as np
import jax.numpy as jnp
from monty.json import MontyDecoder

from .curve import Curve, JaxCurve
import simsoptpp as sopp

__all__ = ['CurveXYZFourier', 'JaxCurveXYZFourier']


class CurveXYZFourier(sopp.CurveXYZFourier, Curve):

    r"""
       ``CurveXYZFourier`` is a curve that is represented in Cartesian
       coordinates using the following Fourier series:

        .. math::
           x(\theta) &= \sum_{m=0}^{\text{order}} x_{c,m}\cos(m\theta) + \sum_{m=1}^{\text{order}} x_{s,m}\sin(m\theta) \\
           y(\theta) &= \sum_{m=0}^{\text{order}} y_{c,m}\cos(m\theta) + \sum_{m=1}^{\text{order}} y_{s,m}\sin(m\theta) \\
           z(\theta) &= \sum_{m=0}^{\text{order}} z_{c,m}\cos(m\theta) + \sum_{m=1}^{\text{order}} z_{s,m}\sin(m\theta)

       The dofs are stored in the order

        .. math::
           [x_{c,0}, x_{s,1}, x_{c,1},\cdots x_{s,\text{order}}, x_{c,\text{order}},y_{c,0},y_{s,1},y_{c,1},\cdots]

    """

    def __init__(self, quadpoints, order):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveXYZFourier.__init__(self, quadpoints, order)
        Curve.__init__(self, x0=self.get_dofs(), names=self._make_names(order),
                       external_dof_setter=CurveXYZFourier.set_dofs_impl)

    def _make_names(self, order):
        x_names = ['xc(0)']
        x_cos_names = [f'xc({i})' for i in range(1, order + 1)]
        x_sin_names = [f'xs({i})' for i in range(1, order + 1)]
        x_names += list(chain.from_iterable(zip(x_sin_names, x_cos_names)))
        y_names = ['yc(0)']
        y_cos_names = [f'yc({i})' for i in range(1, order + 1)]
        y_sin_names = [f'ys({i})' for i in range(1, order + 1)]
        y_names += list(chain.from_iterable(zip(y_sin_names, y_cos_names)))
        z_names = ['zc(0)']
        z_cos_names = [f'zc({i})' for i in range(1, order + 1)]
        z_sin_names = [f'zs({i})' for i in range(1, order + 1)]
        z_names += list(chain.from_iterable(zip(z_sin_names, z_cos_names)))

        return x_names + y_names + z_names

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.asarray(sopp.CurveXYZFourier.get_dofs(self))

    def set_dofs(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        self.local_x = dofs
        sopp.CurveXYZFourier.set_dofs(self, dofs)

    @staticmethod
    def load_curves_from_file(filename, order=None, ppp=20, delimiter=','):
        """
        This function loads a file containing Fourier coefficients for several coils.
        The file is expected to have :mod:`6*num_coils` many columns, and :mod:`order+1` many rows.
        The columns are in the following order,

            sin_x_coil1, cos_x_coil1, sin_y_coil1, cos_y_coil1, sin_z_coil1, cos_z_coil1, sin_x_coil2, cos_x_coil2, sin_y_coil2, cos_y_coil2, sin_z_coil2, cos_z_coil2,  ...

        """
        coil_data = np.loadtxt(filename, delimiter=delimiter)

        assert coil_data.shape[1] % 6 == 0
        assert order <= coil_data.shape[0]-1

        num_coils = coil_data.shape[1]//6
        coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]
        for ic in range(num_coils):
            dofs = coils[ic].dofs
            dofs[0][0] = coil_data[0, 6*ic + 1]
            dofs[1][0] = coil_data[0, 6*ic + 3]
            dofs[2][0] = coil_data[0, 6*ic + 5]
            for io in range(0, min(order, coil_data.shape[0]-1)):
                dofs[0][2*io+1] = coil_data[io+1, 6*ic + 0]
                dofs[0][2*io+2] = coil_data[io+1, 6*ic + 1]
                dofs[1][2*io+1] = coil_data[io+1, 6*ic + 2]
                dofs[1][2*io+2] = coil_data[io+1, 6*ic + 3]
                dofs[2][2*io+1] = coil_data[io+1, 6*ic + 4]
                dofs[2][2*io+2] = coil_data[io+1, 6*ic + 5]
            coils[ic].local_x = np.concatenate(dofs)
        return coils

    def as_dict(self) -> dict:
        d = {}
        d["@class"] = self.__class__.__name__
        d["@module"] = self.__class__.__module__
        d["quadpoints"] = list(self.quadpoints)
        d["order"] = self.order
        d["x0"] = list(self.local_full_x)
        return d

    @classmethod
    def from_dict(cls, d):
        curve = cls(d["quadpoints"], d["order"])
        curve.local_full_x = d["x0"]
        return curve


def jaxfouriercurve_pure(dofs, quadpoints, order):
    k = len(dofs)//3
    coeffs = [dofs[:k], dofs[k:(2*k)], dofs[(2*k):]]
    points = quadpoints
    gamma = jnp.zeros((len(points), 3))
    for i in range(3):
        gamma = gamma.at[:, i].add(coeffs[i][0])
        for j in range(1, order+1):
            gamma = gamma.at[:, i].add(coeffs[i][2 * j - 1] * jnp.sin(2 * pi * j * points))
            gamma = gamma.at[:, i].add(coeffs[i][2 * j] * jnp.cos(2 * pi * j * points))
    return gamma


class JaxCurveXYZFourier(JaxCurve):

    """
    A Python+Jax implementation of the CurveXYZFourier class.  There is
    actually no reason why one should use this over the C++ implementation in
    :mod:`simsoptpp`, but the point of this class is to illustrate how jax can be used
    to define a geometric object class and calculate all the derivatives (both
    with respect to dofs and with respect to the angle :math:`\theta`) automatically.
    """

    def __init__(self, quadpoints, order):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        pure = lambda dofs, points: jaxfouriercurve_pure(dofs, points, order)
        self.order = order
        self.coefficients = [np.zeros((2*order+1,)), np.zeros((2*order+1,)), np.zeros((2*order+1,))]
        super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients))

    def num_dofs(self):
        """
        This function returns the number of dofs associated to this object.
        """
        return 3*(2*self.order+1)

    def get_dofs(self):
        """
        This function returns the dofs associated to this object.
        """
        return np.concatenate(self.coefficients)

    def set_dofs_impl(self, dofs):
        """
        This function sets the dofs associated to this object.
        """
        counter = 0
        for i in range(3):
            self.coefficients[i][0] = dofs[counter]
            counter += 1
            for j in range(1, self.order+1):
                self.coefficients[i][2*j-1] = dofs[counter]
                counter += 1
                self.coefficients[i][2*j] = dofs[counter]
                counter += 1

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["quadpoints"] = list(self.quadpoints)
        d["order"] = self.order
        d["x0"] = list(self.local_full_x)
        return d

    @classmethod
    def from_dict(cls, d):
        curve = cls(d["quadpoints"], d["order"])
        curve.local_full_x = d["x0"]
        return curve
