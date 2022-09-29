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

    def solve(self, normal_plasma, quadpoints_plasma, Bnormal_plasma):
        """
           Bnormal_plasma is the Bnormal component coming from both plasma currents
           and other external coils, so only B^{GI} (Eq. A7) and B^{SV} need to be
           calculated and set up for the least squares solve. 
        """
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        quadpoints_coil = self.winding_surface.gamma().reshape(-1, 3)
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        norm_normal = np.linalg.norm(normal, axis=2)
        self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, norm_normal)
        self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, norm_normal)

        B_matrix = sopp.winding_surface_field_Bn(quadpoints_plasma, quadpoints_coil, normal_plasma, normal, self.winding_surface.nfp, self.winding_surface.stellsym, self.winding_surface.quadpoints_phi, self.winding_surface.quadpoints_theta, self.ndofs, self.current_potential.m, self.current_potential.n)
        B_GI = sopp.winding_surface_field_Bn_GI(quadpoints_plasma, quadpoints_coil, normal_plasma, normal, self.winding_surface.nfp, self.winding_surface.stellsym, self.winding_surface.quadpoints_phi, self.winding_surface.quadpoints_theta, self.ndofs, self.current_potential.m, self.current_potential.n)
        b_rhs = sopp.winding_surface_field_B_rhs(quadpoints_plasma, quadpoints_coil, normal_plasma, normal, self.winding_surface.nfp, self.winding_surface.stellsym, self.winding_surface.quadpoints_phi, self.winding_surface.quadpoints_theta, self.ndofs, self.current_potential.m, self.current_potential.n)
        #self.current_potential.B_rhs_impl_helper(B_rhs, dg1, dg2, norm_normal)
        print(K_matrix)
        print(K_rhs)
        print(B_matrix, B_matrix.shape)
        ### TODO: multiply B_matrix and B_rhs by the grid spacing constants Delta phi Delta theta
        ### from both the plasma quadrature grid and the coil quadrature grid
        print(b_rhs)
        dofs = np.linalg.solve(B_matrix, b_rhs)
        #dofs = np.linalg.solve(B_matrix + lam * K_matrix, B_rhs + lam * K_rhs)
        return dofs
