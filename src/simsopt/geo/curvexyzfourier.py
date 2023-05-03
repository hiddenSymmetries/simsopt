from math import pi
from itertools import chain

import numpy as np
import jax.numpy as jnp
from scipy import interpolate

from .curve import Curve, JaxCurve
from .._core.json import GSONDecoder
import simsoptpp as sopp
from simsopt.util.fourier_interpolation import sin_coeff, cos_coeff

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
    def load_curves_from_file(filename, order=None, ppp=20, delimiter=',',Cartesian=False, accuracy = 500):
        """
        This function loads a file containing Fourier coefficients for several coils.
        The file is expected to have :mod:`6*num_coils` many columns, and :mod:`order+1` many rows.
        The columns are in the following order,

            sin_x_coil1, cos_x_coil1, sin_y_coil1, cos_y_coil1, sin_z_coil1, cos_z_coil1, sin_x_coil2, cos_x_coil2, sin_y_coil2, cos_y_coil2, sin_z_coil2, cos_z_coil2,  ...

        """
        if Cartesian:
            with open(filename, 'r') as f:
                allCoilsValues = f.read().splitlines()[3:] 
        
            coilN=0
            coilPos=[[]]
            for nVals in range(len(allCoilsValues)):
                vals=allCoilsValues[nVals].split()
                try:
                    floatVals = [float(nVals) for nVals in vals][0:3]
                    coilPos[coilN].append(floatVals)
                except:
                    try:
                        floatVals = [float(nVals) for nVals in vals[0:3]][0:3]
                        coilPos[coilN].append(floatVals)
                        coilN=coilN+1
                        coilPos.append([])
                    except:
                        break
        
            coilPos = coilPos[:-1]
            
            coil_data = []
            
            for curve in coilPos:
                xArr=[i[0] for i in curve]
                yArr=[i[1] for i in curve]
                zArr=[i[2] for i in curve]

                L = [0 for i in range(len(xArr))]
                for itheta in range(1,len(xArr)): 
                    dx = xArr[itheta]-xArr[itheta-1]
                    dy = yArr[itheta]-yArr[itheta-1]
                    dz = zArr[itheta]-zArr[itheta-1]
                    dL = np.sqrt(dx*dx+dy*dy+dz*dz)
                    L[itheta]=L[itheta-1]+dL

                L = np.array(L)*2*pi/L[-1] 
            
                xf  = interpolate.CubicSpline(L,xArr) #use the CubicSpline method with periodic bc instead? would require closing the line.
                yf  = interpolate.CubicSpline(L,yArr)
                zf  = interpolate.CubicSpline(L,zArr)
                
                order_interval = range(order+1)
                curvesFourierXS=[sin_coeff(xf,j,accuracy) for j in order_interval]
                curvesFourierXC=[cos_coeff(xf,j,accuracy) for j in order_interval]
                curvesFourierYS=[sin_coeff(yf,j,accuracy) for j in order_interval]
                curvesFourierYC=[cos_coeff(yf,j,accuracy) for j in order_interval]
                curvesFourierZS=[sin_coeff(zf,j,accuracy) for j in order_interval]
                curvesFourierZC=[cos_coeff(zf,j,accuracy) for j in order_interval]
                
                coil_data.append(np.concatenate([curvesFourierXS,curvesFourierXC,curvesFourierYS,curvesFourierYC,curvesFourierZS,curvesFourierZC]))
        
            coil_data = np.asarray(coil_data)
            coil_data = coil_data.reshape(6*len(coilPos),order+1) #There are 6*order coefficients per coil
            coil_data = np.transpose(coil_data)
        else:  
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

    def __init__(self, quadpoints, order, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        pure = lambda dofs, points: jaxfouriercurve_pure(dofs, points, order)
        self.order = order
        self.coefficients = [np.zeros((2*order+1,)), np.zeros((2*order+1,)), np.zeros((2*order+1,))]
        if dofs is None:
            super().__init__(quadpoints, pure, x0=np.concatenate(self.coefficients),
                             external_dof_setter=JaxCurveXYZFourier.set_dofs_impl)
        else:
            super().__init__(quadpoints, pure, dofs=dofs,
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
