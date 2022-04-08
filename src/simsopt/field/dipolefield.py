import numpy as np

# import simsoptpp as sopp
from ..geo.curve import Curve
from .._core.derivative import Derivative
from .magneticfield import MagneticField


#class DipoleField(sopp.DipoleField, MagneticField):
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
        m: Solution for the dipoles using the pm_opt object. 
    """

    def __init__(self, pm_opt, m):
        MagneticField.__init__(self)
        self.ndipoles = pm_opt.ndipoles
        self.m = m
        self.m_vec = self.m.reshape(self.ndipoles, 3)
        self.dipole_grid = pm_opt.dipole_grid.T
        # sopp.DipoleField.__init__(self)

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
        B_field = np.zeros((nfieldpoints, 3))
        dipole_grid_r = np.outer(np.ones(nfieldpoints), self.dipole_grid[:, 0])
        dipole_grid_phi = np.outer(np.ones(nfieldpoints), self.dipole_grid[:, 1])
        dipole_grid_z = np.outer(np.ones(nfieldpoints), self.dipole_grid[:, 2])
        field_points_r = np.outer(field_points[:, 0], np.ones(self.ndipoles))
        field_points_phi = np.outer(field_points[:, 1], np.ones(self.ndipoles))
        field_points_z = np.outer(field_points[:, 2], np.ones(self.ndipoles))
        rmag = np.outer(np.sqrt((dipole_grid_r ** 2 + field_points_r ** 2
                                 ) - 2 * dipole_grid_r * field_points_r * np.cos(
            dipole_grid_phi - field_points_phi
        ) + (dipole_grid_z - field_points_z) ** 2
        ),
            np.ones(3)
        )

        for i in range(nfieldpoints):
            r_vector_i = self.dipole_grid - np.outer(
                np.ones(self.ndipoles), field_points[i, :]
            )  # size ndipoles x 3
            rdotm = np.outer(np.sum(r_vector_i * self.m_vec, axis=-1), np.ones(3))
            #radial_term = self.dipole_grid[:, 0] ** 2 + field_points[i, 0] ** 2
            #angular_term = - 2 * self.dipole_grid[:, 0] * field_points[i, 0] * np.cos(self.dipole_grid[:, 1] - field_points[i, 1])
            #axial_term = (self.dipole_grid[:, 2] - field_points[i, 2]) ** 2
            #rmag_i = np.outer(np.sqrt(radial_term + angular_term + axial_term), np.ones(3))
            B_field[i, :] = np.sum(
                (3 * rdotm * r_vector_i / rmag[i, :] ** 5
                 ) - self.m_vec / rmag[i, :] ** 3, axis=0)
        self._B = B_field * mu0_fac 
        return self

    def dB_by_dX(self):
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
        vec_potential = np.zeros((nfieldpoints, 3))
        dipole_grid_r = np.outer(np.ones(nfieldpoints), self.dipole_grid[:, 0])
        dipole_grid_phi = np.outer(np.ones(nfieldpoints), self.dipole_grid[:, 1])
        dipole_grid_z = np.outer(np.ones(nfieldpoints), self.dipole_grid[:, 2])
        field_points_r = np.outer(field_points[:, 0], np.ones(self.ndipoles))
        field_points_phi = np.outer(field_points[:, 1], np.ones(self.ndipoles))
        field_points_z = np.outer(field_points[:, 2], np.ones(self.ndipoles))
        rmag = np.sqrt((dipole_grid_r ** 2 + field_points_r ** 2
                        ) - 2 * dipole_grid_r * field_points_r * np.cos(
            dipole_grid_phi - field_points_phi
        ) + (dipole_grid_z - field_points_z) ** 2
        )
        for i in range(nfieldpoints):
            r_vector_i = self.dipole_grid - field_points[i, :]  # size ndipoles x 3
            rxm = np.cross(r_vector_i, m_vec, axis=-1)
            # rxm[:, 0] = r_vector_i[:, 1] * self.m_vec[:, 2] -  r_vector_i[:, 2] * self.m_vec[:, 1]
            # rxm[:, 1] = r_vector_i[:, 2] * self.m_vec[:, 0] -  r_vector_i[:, 0] * self.m_vec[:, 2]
            # rxm[:, 2] = r_vector_i[:, 0] * self.m_vec[:, 1] -  r_vector_i[:, 1] * self.m_vec[:, 0]
            # rmag_i = np.sqrt( radial_term + angular_term + axial_term)
            vec_potential[i, :] = np.sum(rxm / rmag[i, :] ** 3, axis=0)

        self._A = vec_potential * mu0_fac 
        return self

    def _A_impl(self, A):
        self.compute_A()
        A[:] = self._A

    def _B_impl(self, B):
        self.compute_B()
        B[:] = self._B

    def _dB_by_dX_impl(self, dB_by_dX):
        self.compute_dB_by_dX()
        dB_by_dX[:] = self._dB

    def _dA_by_dX_impl(self, dA_by_dX):
        self.compute_A(compute_derivatives=1)
        dA_by_dX[:] = self._dA

