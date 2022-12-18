import numpy as np
from .._core.optimizable import DOFs, Optimizable
import simsoptpp as sopp
from sklearn.linear_model import Lasso
from simsopt.geo import Surface, SurfaceRZFourier
from simsopt.field.currentpotential import CurrentPotentialFourier
from scipy.io import netcdf_file
from simsopt.field.magneticfieldclasses import WindingSurfaceField

__all__ = ["CurrentPotentialSolve"]


class CurrentPotentialSolve:
    """
    Current Potential Solve object is designed for performing
    the winding surface coil optimization. We provide functionality
    for the REGCOIL (tikhonov regularization) and L1 norm (Lasso)
    variants of the problem. 

    Args:
        cp: CurrentPotential class object containing the winding surface.
        plasma_surface: The plasma surface to optimize Bnormal over.
        Bnormal_plasma: Bnormal coming from plasma currents.
        B_GI: Bnormal coming from the net coil currents. 
    """

    def __init__(self, cp, plasma_surface, Bnormal_plasma, B_GI):
        self.current_potential = cp
        self.winding_surface = self.current_potential.winding_surface
        self.ntheta_coil = len(self.current_potential.quadpoints_theta)
        self.nphi_coil = len(self.current_potential.quadpoints_phi)
        self.ndofs = self.current_potential.num_dofs()
        self.plasma_surface = plasma_surface
        self.Bnormal_plasma = Bnormal_plasma
        self.B_GI = B_GI

    @classmethod
    def from_netcdf(cls, filename: str):
        """
        Initialize a CurrentPotentialSolve using a CurrentPotentialFourier
        from a regcoil netcdf output file. The single_valued_current_potential_mn
        are set to zero.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
        """
        f = netcdf_file(filename, 'r')
        nfp = f.variables['nfp'][()]
        Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
        rmnc_plasma = f.variables['rmnc_plasma'][()]
        zmns_plasma = f.variables['zmns_plasma'][()]
        xm_plasma = f.variables['xm_plasma'][()]
        xn_plasma = f.variables['xn_plasma'][()]
        mpol_plasma = int(np.max(xm_plasma))
        ntor_plasma = int(np.max(xn_plasma)/nfp)
        ntheta_plasma = f.variables['ntheta_plasma'][()]
        nzeta_plasma = f.variables['nzeta_plasma'][()]
        if ('rmns_plasma' in f.variables and 'zmnc_plasma' in f.variables):
            rmns_plasma = f.variables['rmns_plasma'][()]
            zmnc_plasma = f.variables['zmnc_plasma'][()]
            if np.all(zmnc_plasma == 0) and np.all(rmns_plasma == 0):
                stellsym_plasma_surf = True
            else:
                stellsym_plasma_surf = False
        else:
            rmns_plasma = np.zeros_like(rmnc_plasma)
            zmnc_plasma = np.zeros_like(zmns_plasma)
            stellsym_plasma_surf = True
        f.close()

        cp = CurrentPotentialFourier.from_netcdf(filename)

        s_plasma = SurfaceRZFourier(nfp=nfp,
                                    mpol=mpol_plasma, ntor=ntor_plasma, stellsym=stellsym_plasma_surf)
        s_plasma = s_plasma.from_nphi_ntheta(nfp=nfp, ntheta=ntheta_plasma, nphi=nzeta_plasma,
                                             mpol=mpol_plasma, ntor=ntor_plasma, stellsym=stellsym_plasma_surf, range="field period")
        s_plasma.set_dofs(0*s_plasma.get_dofs())
        for im in range(len(xm_plasma)):
            s_plasma.set_rc(xm_plasma[im], int(xn_plasma[im]/nfp), rmnc_plasma[im])
            s_plasma.set_zs(xm_plasma[im], int(xn_plasma[im]/nfp), zmns_plasma[im])
            if not stellsym_plasma_surf:
                s_plasma.set_rs(xm_plasma[im], int(xn_plasma[im]/nfp), rmns_plasma[im])
                s_plasma.set_zc(xm_plasma[im], int(xn_plasma[im]/nfp), zmnc_plasma[im])

        cp_copy = CurrentPotentialFourier.from_netcdf(filename)
        Bfield = WindingSurfaceField(cp_copy)
        points = s_plasma.gamma().reshape(-1, 3)
        Bfield.set_points(points)
        B = Bfield.B()
        normal = s_plasma.unitnormal().reshape(-1, 3)
        B_GI_winding_surface = np.sum(B*normal, axis=1)
        return cls(cp, s_plasma, np.ravel(Bnormal_from_plasma_current), B_GI_winding_surface)

    def K_rhs(self):
        """
            Wrapper function for computing the right-hand-side part
            that corresponds to the L2 norm of the current density. 
        """
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, normal)
        K_rhs *= self.winding_surface.quadpoints_theta[1]*self.winding_surface.quadpoints_phi[1]/self.winding_surface.nfp
        return K_rhs

    def K_matrix(self):
        """
            Wrapper function for computing the matrix corresponding to the
            L2 norm of the current density. 
        """
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, normal)
        K_matrix *= self.winding_surface.quadpoints_theta[1]*self.winding_surface.quadpoints_phi[1]/self.winding_surface.nfp
        return K_matrix

    def B_matrix_and_rhs(self):
        """
            Compute the matrix and right-hand-side corresponding the Bnormal part of
            the optimization.
        """
        plasma_surface = self.plasma_surface
        normal = self.winding_surface.normal().reshape(-1, 3)
        Bnormal_plasma = self.Bnormal_plasma
        normal_plasma = plasma_surface.normal().reshape(-1, 3)
        points_plasma = plasma_surface.gamma().reshape(-1, 3)
        points_coil = self.winding_surface.gamma().reshape(-1, 3)
        theta = self.winding_surface.quadpoints_theta
        phi_mesh, theta_mesh = np.meshgrid(self.winding_surface.quadpoints_phi, theta, indexing='ij')
        phi_mesh = np.ravel(phi_mesh)
        theta_mesh = np.ravel(theta_mesh)
        gj, B_matrix = sopp.winding_surface_field_Bn(points_plasma,
                                                     points_coil, normal_plasma, normal, self.winding_surface.stellsym,
                                                     phi_mesh, theta_mesh, self.ndofs, self.current_potential.m, self.current_potential.n,
                                                     self.winding_surface.nfp)
        B_GI = self.B_GI

        # set up RHS of optimization
        b_rhs = - np.ravel(B_GI + Bnormal_plasma) @ gj
        dzeta_plasma = (plasma_surface.quadpoints_phi[1] - plasma_surface.quadpoints_phi[0])
        dtheta_plasma = (plasma_surface.quadpoints_theta[1] - plasma_surface.quadpoints_theta[0])
        dzeta_coil = (self.winding_surface.quadpoints_phi[1] - self.winding_surface.quadpoints_phi[0])
        dtheta_coil = (self.winding_surface.quadpoints_theta[1] - self.winding_surface.quadpoints_theta[0])
        # scale bmatrix and b_rhs by factors of the grid spacing
        b_rhs = b_rhs * dzeta_plasma * dtheta_plasma * dzeta_coil * dtheta_coil
        B_matrix = B_matrix * dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2
        return b_rhs, B_matrix

    def solve_tikhonov(self, lam=0):
        """
            Solve the REGCOIL problem -- winding surface optimization with
            the L2 norm. This is tested against REGCOIL runs extensively in
            tests/field/test_regcoil.py.
        """
        K_matrix = self.K_matrix()
        K_rhs = self.K_rhs()
        b_rhs, B_matrix = self.B_matrix_and_rhs()
        phi_mn_opt = np.linalg.solve(B_matrix + lam * K_matrix, b_rhs + lam * K_rhs)
        self.current_potential.set_dofs(phi_mn_opt)
        return phi_mn_opt

    def solve_lasso(self, lam=0):
        """
            Solve the Lasso problem -- winding surface optimization with
            the L1 norm, which should tend to allow stronger current 
            filaments to form than the L2. 
        """
        K_matrix = self.K_matrix()
        K_rhs = self.K_rhs()
        b_rhs, B_matrix = self.B_matrix_and_rhs()

        # note taking sqrt(lam) to be similar amoun of regularization as the L2
        solver = Lasso(alpha=np.sqrt(lam))
        solution = solver.fit(B_matrix + np.sqrt(lam) * K_matrix, b_rhs + np.sqrt(lam) * K_rhs)
        phi_mn_opt = solution.coef_
        self.current_potential.set_dofs(phi_mn_opt)
        return phi_mn_opt