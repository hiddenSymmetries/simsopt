import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from simsopt.geo import Surface, SurfaceRZFourier 
from simsopt.field.currentpotential import CurrentPotentialFourier
from scipy.io import netcdf_file

__all__ = ["CurrentPotentialSolveTikhonov"]


class CurrentPotentialSolveTikhonov:
    def __init__(self, cp):
        self.current_potential = cp
        self.winding_surface = self.current_potential.winding_surface
        self.ntheta_coil = len(self.current_potential.quadpoints_theta)
        self.nphi_coil = len(self.current_potential.quadpoints_phi)
        self.ndofs = self.current_potential.num_dofs()

    @classmethod
    def from_netcdf(cls, filename: str):
        """
        Initialize a CurrentPotentialSolveTikhonov using a CurrentPotentialFourier
        from a regcoil netcdf output file.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
        """
        f = netcdf_file(filename, 'r')
        nfp = f.variables['nfp'][()]
        mpol_potential = f.variables['mpol_potential'][()]
        ntor_potential = f.variables['ntor_potential'][()]
        net_poloidal_current_amperes = f.variables['net_poloidal_current_Amperes'][()]
        net_toroidal_current_amperes = f.variables['net_toroidal_current_Amperes'][()]
        xm_potential = f.variables['xm_potential'][()]
        xn_potential = f.variables['xn_potential'][()]
        symmetry_option = f.variables['symmetry_option'][()]
        if symmetry_option == 1:
            stellsym = True
        else:
            stellsym = False
        rmnc_coil = f.variables['rmnc_coil'][()]
        zmns_coil = f.variables['zmns_coil'][()]
        if ('rmns_coil' in f.variables and 'zmnc_coil' in f.variables):
            rmns_coil = f.variables['rmns_coil'][()]
            zmnc_coil = f.variables['zmnc_coil'][()]
            if np.all(zmnc_coil == 0) and np.all(rmns_coil == 0):
                stellsym_surf = True
            else:
                stellsym_surf = False
        else:
            rmns_coil = np.zeros_like(rmnc_coil)
            zmnc_coil = np.zeros_like(zmns_coil)
            stellsym_surf = True
        xm_coil = f.variables['xm_coil'][()]
        xn_coil = f.variables['xn_coil'][()]
        ntheta_coil = f.variables['ntheta_coil'][()]
        nzeta_coil = f.variables['nzeta_coil'][()]
        single_valued_current_potential_mn = f.variables['single_valued_current_potential_mn'][()][-1, :]
        mpol_coil = int(np.max(xm_coil))
        ntor_coil = int(np.max(xn_coil)/nfp)

        s_coil = SurfaceRZFourier(nfp=nfp,mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf)
        s_coil = s_coil.from_nphi_ntheta(nfp=nfp, ntheta=ntheta_coil, nphi=nzeta_coil*nfp,
                                         mpol=mpol_coil, ntor=ntor_coil, stellsym=stellsym_surf, range='full torus')
        s_coil.set_dofs(0*s_coil.get_dofs())
        for im in range(len(xm_coil)):
            s_coil.set_rc(xm_coil[im], int(xn_coil[im]/nfp), rmnc_coil[im])
            s_coil.set_zs(xm_coil[im], int(xn_coil[im]/nfp), zmns_coil[im])

        cp = CurrentPotentialFourier(s_coil, mpol=mpol_potential, ntor=ntor_potential,
                                     net_poloidal_current_amperes=net_poloidal_current_amperes,
                                     net_toroidal_current_amperes=net_toroidal_current_amperes,
                                     stellsym=stellsym)
        for im in range(len(xm_potential)):
            cp.set_phis(xm_potential[im], int(xn_potential[im]/nfp), single_valued_current_potential_mn[im])

        return cls(cp)

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

        G = self.current_potential.net_poloidal_current_amperes
        I = self.current_potential.net_toroidal_current_amperes

        normal = self.winding_surface.normal()
        norm_normal = np.linalg.norm(normal, axis=2)
        self.K_matrix_impl(K_matrix)
        self.K_rhs_impl(K_rhs)

        theta = self.winding_surface.quadpoints_theta
        phi_mesh, theta_mesh = np.meshgrid(self.winding_surface.quadpoints_phi, theta, indexing='ij')
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)
        normal = self.winding_surface.normal().reshape(-1, 3)

        gj, B_matrix = sopp.winding_surface_field_Bn(quadpoints_plasma, quadpoints_coil, normal_plasma, normal, self.winding_surface.stellsym, phi_mesh, theta_mesh, self.ndofs, self.current_potential.m, self.current_potential.n)

        print(dg2.reshape(-1, 3).shape, quadpoints_plasma.shape, quadpoints_coil.shape, normal_plasma.shape, phi_mesh.shape, theta_mesh.shape, G, I)
        B_GI = sopp.winding_surface_field_Bn_GI(quadpoints_plasma, quadpoints_coil, normal_plasma, phi_mesh, theta_mesh, G, I, dg1.reshape(-1, 3), dg2.reshape(-1, 3))

        # scale everything by the grid spacings
        dzeta_plasma = (plasma_surface.quadpoints_phi[1] - plasma_surface.quadpoints_phi[0])
        dtheta_plasma = (plasma_surface.quadpoints_theta[1] - plasma_surface.quadpoints_theta[0])
        dzeta_coil = (self.winding_surface.quadpoints_phi[1] - self.winding_surface.quadpoints_phi[0])
        dtheta_coil = (self.winding_surface.quadpoints_theta[1] - self.winding_surface.quadpoints_theta[0])

        #gj = gj * dzeta_coil * dtheta_coil
        B_GI = B_GI * dzeta_coil * dtheta_coil
        # set up RHS of optimization
        print('B_GI = ', B_GI, B_GI.shape)
        print('Bnormal_plasma = ', Bnormal_plasma, Bnormal_plasma.shape)
        b_rhs = np.zeros(self.ndofs)
        for i in range(self.ndofs):
            b_rhs[i] = - (B_GI + Bnormal_plasma) @ gj[:, i]

        b_rhs = b_rhs * dzeta_plasma * dtheta_plasma * dzeta_coil * dtheta_coil
        B_matrix = B_matrix * dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2

        # print('B_matrix = ', B_matrix)
        print('b_matrix = ', b_rhs)
        # dofs, _, _, _ = np.linalg.lstsq(B_matrix, b_rhs)
        dofs = np.linalg.solve(B_matrix + lam * K_matrix, b_rhs + lam * K_rhs)
        return dofs
