import numpy as np

from ..geo.curve import Curve
from .._core.derivative import Derivative
from .magneticfield import MagneticField


class DipoleField(MagneticField):
    r"""
    Computes the MagneticField induced by N dipoles. This is very simple but needs to be
    a type MagneticField class for using the other simsopt functionality.
    The field is given by

    .. math::

        B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{i=1}^{N} (\frac{3\mathbf{r}_i\cdot \mathbf{m}_i}{|\mathbf{r}_i|^5}\mathbf{r}_i - \frac{\mathbf{m}_i}{|\mathbf{r}_i|^3}) 

    where :math:`\mu_0=4\pi 10^{-7}` is the magnetic constant and :math:\mathbf{r_i} = \mathbf{x} - \mathbf{x}^{dipole}_i is the vector between the field evaluation point and the dipole i position. 

    Args:
        pm_opt: A PermanentMagnetOptimizer object that has already been optimized.
    """

    def __init__(self, pm_opt):
        self.ndipoles = pm_opt.ndipoles
        self.m = pm_opt.m
        self.dipole_grid = pm_opt.dipole_grid

    def compute_B(self):
        r"""
        Compute the magnetic field of N dipole fields

        .. math::
             
             B(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{i=1}^{N} (\frac{3\mathbf{r}_i\cdot \mathbf{m}_i}{|\mathbf{r}_i|^5}\mathbf{r}_i - \frac{\mathbf{m}_i}{|\mathbf{r}_i|^3}) 

        """
        # Get field evaluation points in cylindrical coordinates
        field_points = self.get_points_cyl_ref()
        nfieldpoints = len(field_points)
        mu0_fac = 1e-7
        # reshape the dipoles (hopefully correctly)
        m_vec = self.m.reshape(self.ndipoles, 3)
        rdotm = np.zeros((self.ndipoles, 3))
        B_field = np.zeros((nfieldpoints, 3))
        for i in range(nfieldpoints):
            r_vector_i = self.dipole_grid - field_points[i, :] # size ndipoles x 3
            rdotm = np.sum(r_vector_i * m_vec, axis=-1)
            radial_term = self.dipole_grid[:, 0] ** 2 + field_points[i, 0] ** 2
            angular_term = - 2 * self.dipole_grid[:, 0] * field_points[i, 0] * np.cos(self.dipole_grid[:, 1] - field_points[i, 1])
            axial_term = (self.dipole_grid[:, 2] - field_points[i, 2]) ** 2
            rmag_i = np.sqrt(radial_term + angular_term + axial_term)
            first_term = 3 * np.sum(rdotm * r_vector_i / rmag_i ** 5, axis=0)
            second_term = - np.sum(m_vec / rmag_i ** 3, axis=0)
            B_field[i, :] = first_term + second_term
        self._B = B_field * mu0_fac 
        return self

    
    def compute_A(self):
        r"""
        Compute the potential of N dipole fields

        .. math::

            A(\mathbf{x}) = \frac{\mu_0}{4\pi} \sum_{i=1}^{N} \frac{\mathbf{m}_i \times \mathbf{r}_i}{|\mathbf{r}_i|^3} 

        """
        # Get field evaluation points in cylindrical coordinates
        field_points = self.get_points_cyl_ref()
        nfieldpoints = len(field_points)
        mu0_fac = 1e-7
        # reshape the dipoles (hopefully correctly)
        m_vec = self.m.reshape(self.ndipoles, 3)
        rxm = np.zeros((self.ndipoles, 3))
        vec_potential = np.zeros((nfieldpoints, 3))
        for i in range(nfieldpoints):
            r_vector_i = self.dipole_grid - field_points[i, :] # size ndipoles x 3
            rxm[:, 0] = r_vector_i[:, 1] * m_vec[:, 2] -  r_vector_i[:, 2] * m_vec[:, 1]
            rxm[:, 1] = r_vector_i[:, 2] * m_vec[:, 0] -  r_vector_i[:, 0] * m_vec[:, 2]
            rxm[:, 2] = r_vector_i[:, 0] * m_vec[:, 1] -  r_vector_i[:, 1] * m_vec[:, 0]
            radial_term = self.dipole_grid[:, 0] ** 2 + field_points[i, 0] ** 2
            angular_term = - 2 * self.dipole_grid[:, 0] * field_points[i, 0] * np.cos(self.dipole_grid[:, 1] - field_points[i, 1])
            axial_term = (self.dipole_grid[:, 2] - field_points[i, 2]) ** 2
            rmag_i = np.sqrt(radial_term + angular_term + axial_term)
            vec_potential[i, :] = np.sum(rxm / rmag_i ** 3, axis=0)
        self._A = vec_potential * mu0_fac 
        return self

    def _A_impl(self, A):
        self.compute_A()
        A[:] = self._A

    def _B_impl(self, B):
        self.compute_B()
        B[:] = self._B
