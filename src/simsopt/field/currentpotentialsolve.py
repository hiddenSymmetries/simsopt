import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from simsopt.geo.surface import Surface

__all__ = ["CurrentPotentialSolveTikhonov"]


class CurrentPotentialSolveTikhonov:
    def __init__(self, winding_surface_field, lam=0):
        self.winding_surface_field = winding_surface_field
        self.lam = lam
        self.current_potential = winding_surface_field.current_potential
        self.winding_surface = self.current_potential.winding_surface
        self.ntheta_coil = len(self.current_potential.quadpoints_theta)
        self.nphi_coil = len(self.current_potential.quadpoints_phi)
        self.ndofs = self.current_potential.num_dofs()

    def solve(self):
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        norm_normal = np.linalg.norm(normal, axis=2)
        self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, norm_normal)
        self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, norm_normal)
        print(K_matrix)
        print(K_rhs)
        dofs = np.linalg.solve(K_matrix, K_rhs)
        return dofs
