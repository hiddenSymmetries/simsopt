from itertools import chain

import numpy as np
import jax.numpy as jnp
from scipy.fft import rfft

from .curve import Curve, JaxCurve
from .coilio import get_data_from_makegrid, get_data_from_xyzfile
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

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = list(np.linspace(0, 1, quadpoints, endpoint=False))
        elif isinstance(quadpoints, np.ndarray):
            quadpoints = list(quadpoints)
        sopp.CurveXYZFourier.__init__(self, quadpoints, order)
        if dofs is None:
            Curve.__init__(self, x0=self.get_dofs(), names=self._make_names(order),
                           external_dof_setter=CurveXYZFourier.set_dofs_impl)
        else:
            Curve.__init__(self, dofs=dofs,
                           external_dof_setter=CurveXYZFourier.set_dofs_impl)

    def _make_names(self, order):
        """
        This function returns the names of the dofs associated to this object.
        Args:
            order (int): Order of the Fourier series.

        Returns:
            List of dof names.
        """
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
            dofs = coils[ic].dofs_matrix
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
    
    @staticmethod
    def load_curves_from_makegrid_file(filename: str, order: int, ppp=20, group_names=None):
        """
        This function loads a Makegrid input file containing the Cartesian
        coordinates for several coils and finds the corresponding Fourier
        coefficients through an fft. The format is described at
        https://princetonuniversity.github.io/STELLOPT/MAKEGRID

        Args:
            filename: file to load.
            order: maximum mode number in the Fourier series. 
            ppp: points-per-period: number of quadrature points per period.
            group_names: List of coil group names (str). If not 'None', only get coils in coil groups that are in the list.

        Returns:
            A list of ``CurveXYZFourier`` objects.
        """
        curve_data = get_data_from_makegrid(filename, group_names=group_names)  # read file
        if len(curve_data) == 0:
            raise ValueError(f"No curves found in file {filename}. Check the file format and group names.")
        
        num_coils = len(curve_data)

        coils = [CurveXYZFourier(order*ppp, order) for i in range(num_coils)]  # create empty coils
        for curve_datum, coil in zip(curve_data, coils):
            coil.x = CurveXYZFourier.fft_coil_positions(curve_datum, order, ppp)  # ft the data from file. 

        return coils
    
    @staticmethod
    def load_curves_from_xyz_file(filename: str, order: int, ppp=20, group_names = None):
        """
        This function loads a Makegrid input file containing the Cartesian
        coordinates for several coils and finds the corresponding Fourier
        coefficients through an fft. The format is described at
        https://princetonuniversity.github.io/STELLOPT/MAKEGRID

        Args:
            filename: file to load.
            order: maximum mode number in the Fourier series. 
            ppp: points-per-period: number of quadrature points per period.
            group_names: List of coil group names (str). If not 'None', only get coils in coil groups that are in the list.

        Returns:
            A list of ``CurveXYZFourier`` objects.
        """
        curve_datum = get_data_from_xyzfile(filename)  # read file
        if len(curve_datum) == 0:
            raise ValueError(f"No curves found in file {filename}. Check the file format and group names.")

        coil = CurveXYZFourier(order*ppp, order)  # create empty coil
        coil.x = CurveXYZFourier.fft_coil_positions(curve_datum, order, ppp)  # ft the data from file. 

        return [coil,]

    
    @staticmethod
    def fft_coil_positions(coil_positions, order, ppp=20):
        """
        This function computes the Fourier coefficients for a set of coil positions.
        
        Args:
            coil_positions: An Nx3-array containing coordinates along each coil.
            order: The maximum mode number in the Fourier series.
        
        Returns:
            an array of Fourier coefficents in the order of the degrees-of-freedom
            of the CurveXYZFourier class (x_c0, x_s1, x_c1, ..., y_c0, y_s1, y_c1, ..., z_c0, z_s1, z_c1, ...).
            
        """
        assert len(coil_positions) >= 2 * order, "Not enough points to compute Fourier coefficients."
        
        xArr, yArr, zArr = np.transpose(coil_positions)

        fourier_lists = []
        for x in [xArr, yArr, zArr]:
            assert len(x) >= 2 * order  # the order of the f:ft is limited by the number of samples
            ft = 2* rfft(x) / len(x)
            coeffs = np.empty((2*order+1,))  # empty array to hold the coefficients
            coeffs[0::2] = ft[:order+1].real  # cosine coefficients are the real parts even indices
            coeffs[1::2] = - ft[1:order+1].imag  # rest is sine coefficient, they are negative imaginary, skip the first one
            coeffs[0] /= 2   # divide the 0th coefficient by 2, as it is only half the amplitude in the rfft
            fourier_lists.append(coeffs)
        
        # fourier_lists now contains the fourier coefficients for the x, y, and z coordinates
        # in the order of the degrees-of-freedom of the CurveXYZFourier class
        fourier_coeffs = np.concatenate(fourier_lists)
        assert len(fourier_coeffs) == 3 * (2 * order + 1), \
            f"Expected {3 * (2 * order + 1)} coefficients, got {len(fourier_coeffs)}."
        return fourier_coeffs

        

def jaxfouriercurve_pure(dofs, quadpoints, order):
    """
    This pure function returns the curve position vector in XYZ coordinates..

    Args:
        dofs (array, shape (ndofs,)): Array of dofs.
        quadpoints (array, shape (N, 3)): Array of quadrature points.
        order (int): Order of the Fourier series.

    Returns:
        Array of curve points, shape (N, 3)
    """
    k = jnp.shape(dofs)[0]//3
    coeffs = [dofs[:k], dofs[k:(2*k)], dofs[(2*k):]]
    points = 2 * np.pi * quadpoints
    jrange = jnp.arange(1, order + 1)
    jp = jrange[:, None] * points[None, :]
    sjp = jnp.sin(jp)
    cjp = jnp.cos(jp)
    gamma_x = coeffs[0][0] + jnp.sum(coeffs[0][2 * jrange - 1, None] * sjp
                                     + coeffs[0][2 * jrange, None] * cjp, axis=0)
    gamma_y = coeffs[1][0] + jnp.sum(coeffs[1][2 * jrange - 1, None] * sjp
                                     + coeffs[1][2 * jrange, None] * cjp, axis=0)
    gamma_z = coeffs[2][0] + jnp.sum(coeffs[2][2 * jrange - 1, None] * sjp
                                     + coeffs[2][2 * jrange, None] * cjp, axis=0)
    return jnp.stack((gamma_x, gamma_y, gamma_z), axis=-1)


class JaxCurveXYZFourier(JaxCurve):

    """
    A Python+Jax implementation of the CurveXYZFourier class. This is an autodiff
    compatible version of the same CurveXYZFourier class in the C++ implementation in
    :mod:`simsoptpp`. The point of this class is to illustrate how jax can be used
    to define a geometric object class and calculate all the derivatives (both
    with respect to dofs and with respect to the angle :math:`\theta`) automatically.

    Args:
        quadpoints (array): Array of quadrature points.
        order (int): Order of the Fourier series.
        dofs (array): Array of dofs.
    """

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)

        def pure(dofs, points): return jaxfouriercurve_pure(dofs, points, order)
        self.order = order
        self.coefficients = [np.zeros((2*order+1,)), np.zeros((2*order+1,)), np.zeros((2*order+1,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             names=self._make_names(order),
                             external_dof_setter=JaxCurveXYZFourier.set_dofs_impl)
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
                             names=self._make_names(order),
                             external_dof_setter=JaxCurveXYZFourier.set_dofs_impl)

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

    def _make_names(self, order):
        """
        This function returns the names of the dofs associated to this object.

        Args:
            order (int): Order of the Fourier series.

        Returns:
            List of dof names.
        """
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