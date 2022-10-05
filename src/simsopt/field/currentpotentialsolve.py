import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from simsopt.geo.surface import Surface

__all__ = ["CurrentPotentialSolveTikhonov"]


class CurrentPotentialSolveTikhonov:
    def __init__(self, cp):
        self.current_potential = cp
        self.winding_surface = self.current_potential.winding_surface
        self.ntheta_coil = len(self.current_potential.quadpoints_theta)
        self.nphi_coil = len(self.current_potential.quadpoints_phi)
        self.ndofs = self.current_potential.num_dofs()

    def K_rhs_impl(self, K_rhs):
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, normal)
        K_rhs *= self.winding_surface.quadpoints_theta[1]*self.winding_surface.quadpoints_phi[1]/self.winding_surface.nfp

    def K_rhs(self):
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        self.K_rhs_impl(K_rhs)
        return K_rhs

    def K_matrix_impl(self, K_matrix):
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, normal)
        K_matrix *= self.winding_surface.quadpoints_theta[1]*self.winding_surface.quadpoints_phi[1]/self.winding_surface.nfp

    def K_matrix(self):
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        self.K_matrix_impl(K_matrix)
        return K_matrix

    def solve(self, plasma_surface, Bnormal_plasma, lam=0):
        """
           Bnormal_plasma is the Bnormal component coming from both plasma currents
           and other external coils, so only B^{GI} (Eq. A7) and B^{SV} need to be
           calculated and set up for the least squares solve.
        """
        normal_plasma = plasma_surface.normal().reshape(-1, 3)
        quadpoints_plasma = plasma_surface.gamma().reshape(-1, 3)
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        quadpoints_coil = self.winding_surface.gamma().reshape(-1, 3)
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()

        # G = self.current_potential.Phidash1()
        # I = self.current_potential.Phidash2()
        G = self.current_potential.net_poloidal_current_amperes
        I = self.current_potential.net_toroidal_current_amperes

        normal = self.winding_surface.normal()
        norm_normal = np.linalg.norm(normal, axis=2)
        self.K_matrix_impl(K_matrix)
        # self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, norm_normal)
        # self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, norm_normal)

        # print(K_rhs)

        theta = self.winding_surface.quadpoints_theta
        phi_mesh, theta_mesh = np.meshgrid(self.winding_surface.quadpoints_phi, theta, indexing='ij')
        #phi_mesh, theta_mesh = np.meshgrid(self.winding_surface.quadpoints_phi, theta, indexing='ij')
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)
        normal = self.winding_surface.normal().reshape(-1, 3)

        gj, B_matrix = sopp.winding_surface_field_Bn(quadpoints_plasma, quadpoints_coil, normal_plasma, normal, self.winding_surface.stellsym, phi_mesh, theta_mesh, self.ndofs, self.current_potential.m, self.current_potential.n)

        B_GI = sopp.winding_surface_field_Bn_GI(quadpoints_plasma, quadpoints_coil, normal_plasma, phi_mesh, theta_mesh, G, I, dg1, dg2)

        # scale everything by the grid spacings
        dzeta_plasma = (plasma_surface.quadpoints_phi[1] - plasma_surface.quadpoints_phi[0]) * 2 * np.pi
        dtheta_plasma = (plasma_surface.quadpoints_theta[1] - plasma_surface.quadpoints_theta[0]) * 2 * np.pi
        dzeta_coil = (self.winding_surface.quadpoints_phi[1] - self.winding_surface.quadpoints_phi[0]) * 2 * np.pi
        dtheta_coil = (self.winding_surface.quadpoints_theta[1] - self.winding_surface.quadpoints_theta[0]) * 2 * np.pi

        B_GI = B_GI * dzeta_coil * dtheta_coil
        # set up RHS of optimization
        print('B_GI = ', B_GI)
        print('Bnormal_plasma = ', Bnormal_plasma)
        b_rhs = np.zeros(self.ndofs)
        for i in range(self.ndofs):
            b_rhs[i] = - (B_GI + Bnormal_plasma) @ gj[:, i]

        b_rhs = b_rhs * dzeta_plasma * dtheta_plasma * dzeta_coil * dtheta_coil
        B_matrix = B_matrix * dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2

        # print('B_matrix = ', B_matrix)
        print('b_matrix = ', b_rhs)
        dofs, _, _, _ = np.linalg.lstsq(B_matrix, b_rhs)
        #dofs = np.linalg.solve(B_matrix + lam * K_matrix, B_rhs + lam * K_rhs)
        return dofs
