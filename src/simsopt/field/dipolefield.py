import numpy as np
from simsopt.field.magneticfield import MagneticField
import simsoptpp as sopp


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
        self.m_vec = m.reshape(self.ndipoles, 3)
        self.dipole_grid = pm_opt.dipole_grid.T
        #dipole_grid_z = pm_opt.dipole_grid[2, :]
        #phi = pm_opt.plasma_boundary.quadpoints_phi
        #dipole_grid_x = np.zeros(len(dipole_grid_z))
        #dipole_grid_y = np.zeros(len(dipole_grid_z))
        #running_tally = 0
        #for i in range(pm_opt.nphi):
        #    radii = np.ravel(np.array(pm_opt.final_rz_grid[i])[:, 0])
        #    len_radii = len(radii)
        #    dipole_grid_x[running_tally:running_tally + len_radii] = radii * np.cos(phi[i])
        #    dipole_grid_y[running_tally:running_tally + len_radii] = radii * np.sin(phi[i])
        #    running_tally += len_radii
        #self.dipole_grid = np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).t
        #print(self.dipole_grid.shape)
    
    def _B_impl(self, B):
        points = self.get_points_cart_ref()
        B[:] = sopp.dipole_field_B(points, self.dipole_grid, self.m_vec)

    def _dB_by_dX_impl(self, dB):
        points = self.get_points_cart_ref()
        dB[:] = sopp.dipole_field_dB(points, self.dipole_grid, self.m_vec)


