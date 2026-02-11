import numpy as np
import os
import time
import warnings
from scipy.io import netcdf_file

TIMING = os.environ.get('SIMSOPT_TIMING', '').lower() in ('1', 'true', 'yes')
from scipy.interpolate import interp2d
from simsopt.field.magneticfieldclasses import WindingSurfaceField
from simsopt.geo import SurfaceRZFourier
from simsopt.field.currentpotential import CurrentPotentialFourier
import simsoptpp as sopp

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

    def __init__(self, cp, plasma_surface, Bnormal_plasma):
        
        if np.max(plasma_surface.quadpoints_phi)>=1/plasma_surface.nfp:
            raise AttributeError('winding_surface must contain only one field period.')

        self.current_potential = cp
        self.winding_surface = self.current_potential.winding_surface
        self.ndofs = self.current_potential.num_dofs()
        self.plasma_surface = plasma_surface
        self.ntheta_plasma = len(self.plasma_surface.quadpoints_theta)
        self.nzeta_plasma = len(self.plasma_surface.quadpoints_phi)
        self.ntheta_coil = len(self.current_potential.quadpoints_theta)
        self.nzeta_coil = len(self.current_potential.quadpoints_phi)
        # Calculating B_GI
        cp_no_phi_sv = CurrentPotentialFourier(
            cp.winding_surface, mpol=cp.mpol, ntor=cp.ntor,
            net_poloidal_current_amperes=cp.net_poloidal_current_amperes,
            net_toroidal_current_amperes=cp.net_toroidal_current_amperes,
            quadpoints_phi=cp.quadpoints_phi,
            quadpoints_theta=cp.quadpoints_theta,
            stellsym=cp.stellsym
        )
        Bfield = WindingSurfaceField(cp_no_phi_sv)
        points = plasma_surface.gamma().reshape(-1, 3)
        Bfield.set_points(points)
        B_GI_vector = Bfield.B()
        normal = plasma_surface.unitnormal().reshape(-1, 3)
        B_GI_winding_surface = np.sum(B_GI_vector*normal, axis=1)
        # Permitting Bnormal_plasma to be a scalar
        if np.isscalar(Bnormal_plasma):
            Bnormal_plasma = Bnormal_plasma * np.ones(normal.shape[0])
        else:
            # If Bnormal is not a scalar, try reshaping it into
            # the proper shape
            try:
                Bnormal_plasma = Bnormal_plasma.reshape(normal.shape[0])
            except:
                raise ValueError('The shape of Bnormal_plasma does not match with the quadrature points of plasma_surface.')
        self.Bnormal_plasma = Bnormal_plasma
        self.B_GI = B_GI_winding_surface
        # Save list of results for each L2 or L1 winding surface
        # optimization performed with this class object
        self.ilambdas_l2 = []
        self.dofs_l2 = []
        self.current_potential_l2 = []
        self.K2s_l2 = []
        self.fBs_l2 = []
        self.fKs_l2 = []
        self.ilambdas_l1 = []
        self.dofs_l1 = []
        self.current_potential_l1 = []
        self.K2s_l1 = []
        self.fBs_l1 = []
        self.fKs_l1 = []
        warnings.warn(
            "Beware: the f_B (also called chi^2_B) computed from the "
            "CurrentPotentialSolve class will be slightly different than "
            "the f_B computed using SquaredFlux with the BiotSavart law "
            "implemented in WindingSurfaceField. This is because the "
            "optimization formulation and the full BiotSavart calculation "
            "are discretized in different ways. This disagreement will "
            "worsen at low regularization, but improve with higher "
            "resolution on the plasma and coil surfaces. "
        )

    @classmethod
    def from_netcdf(cls, filename: str, plasma_ntheta_res=1.0, plasma_nzeta_res=1.0, coil_ntheta_res=1.0, coil_nzeta_res=1.0):
        """
        Initialize a CurrentPotentialSolve using a CurrentPotentialFourier
        from a regcoil netcdf output file. The single_valued_current_potential_mn
        are set to zero.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
            plasma_ntheta_res: The resolution of the plasma surface in the theta direction.
            plasma_nzeta_res: The resolution of the plasma surface in the zeta direction.
            coil_ntheta_res: The resolution of the coil surface in the theta direction.
            coil_nzeta_res: The resolution of the coil surface in the zeta direction.

        Returns:
            cls: The CurrentPotentialSolve object.

        Notes:
            This function initializes a CurrentPotentialSolve object from a regcoil netcdf output file.
            The CurrentPotentialFourier object is initialized from the file, and the Bnormal_from_plasma_current
            is read from the file. The plasma surface is initialized from the file, and the coil surface is
            initialized from the file. The CurrentPotentialSolve object is returned.
        """
        f = netcdf_file(filename, 'r', mmap=False)
        nfp = f.variables['nfp'][()]
        Bnormal_from_plasma_current = f.variables['Bnormal_from_plasma_current'][()]
        rmnc_plasma = f.variables['rmnc_plasma'][()]
        zmns_plasma = f.variables['zmns_plasma'][()]
        xm_plasma = f.variables['xm_plasma'][()]
        xn_plasma = f.variables['xn_plasma'][()]
        mpol_plasma = int(np.max(xm_plasma))
        ntor_plasma = int(np.max(xn_plasma)/nfp)
        ntheta_plasma = int(f.variables['ntheta_plasma'][()] * plasma_ntheta_res)
        nzeta_plasma = int(f.variables['nzeta_plasma'][()] * plasma_nzeta_res)
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

        cp = CurrentPotentialFourier.from_netcdf(filename, coil_ntheta_res, coil_nzeta_res)

        s_plasma = SurfaceRZFourier(
            nfp=nfp,
            mpol=mpol_plasma,
            ntor=ntor_plasma,
            stellsym=stellsym_plasma_surf
        )
        s_plasma = s_plasma.from_nphi_ntheta(
            nfp=nfp, ntheta=ntheta_plasma,
            nphi=nzeta_plasma,
            mpol=mpol_plasma, ntor=ntor_plasma,
            stellsym=stellsym_plasma_surf, range="field period"
        )

        # Need to interpolate Bnormal_from_plasma if increasing resolution
        if plasma_ntheta_res > 1.0 or plasma_nzeta_res > 1.0:
            warnings.warn(
                "User specified to increase the plasma surface resolution, but is "
                "reading the CurrentPotential object from a netcdf file. Therefore, "
                "the Bnormal_from_plasma_current will be interpolated to the "
                "high resolution grid, which may not be very accurate!"
            )
            quadpoints_phi = np.linspace(0, 1 / ((int(stellsym_plasma_surf) + 1) * nfp), f.variables['nzeta_plasma'][()] + 1, endpoint=True)
            quadpoints_theta = np.linspace(0, 1, f.variables['ntheta_plasma'][()] + 1, endpoint=True)
            quadpoints_phi = quadpoints_phi[:-1]
            quadpoints_theta = quadpoints_theta[:-1]
            quadpoints_phi, quadpoints_theta = np.meshgrid(quadpoints_phi, quadpoints_theta, indexing='ij')
            Bnormal_from_plasma_current = interp2d(
                quadpoints_phi, quadpoints_theta, Bnormal_from_plasma_current,
                kind='cubic'
            )
            Bnormal_from_plasma_current = Bnormal_from_plasma_current(
                s_plasma.quadpoints_phi,
                s_plasma.quadpoints_theta
            )
        f.close()
        s_plasma.set_dofs(0 * s_plasma.get_dofs())
        for im in range(len(xm_plasma)):
            s_plasma.set_rc(xm_plasma[im], int(xn_plasma[im] / nfp), rmnc_plasma[im])
            s_plasma.set_zs(xm_plasma[im], int(xn_plasma[im] / nfp), zmns_plasma[im])
            if not stellsym_plasma_surf:
                s_plasma.set_rs(xm_plasma[im], int(xn_plasma[im] / nfp), rmns_plasma[im])
                s_plasma.set_zc(xm_plasma[im], int(xn_plasma[im] / nfp), zmnc_plasma[im])
        return cls(cp, s_plasma, np.ravel(Bnormal_from_plasma_current))

    def write_regcoil_out(self, filename: str):
        """
        Take optimized CurrentPotentialSolve class and save it to a regcoil-style
        outfile for backwards compatability with other stellarator codes.

        Args:
            filename: Name of the ``regcoil_out.*.nc`` file to read.
        """
        f = netcdf_file(filename, 'w')
        f.history = 'Created for writing a SIMSOPT-optimized winding surface and current potential to a regcoil-style output file'

        scalars = ['nfp', 'mpol_plasma', 'ntor_plasma', 'mpol_potential', 'ntor_potential', 'symmetry_option', 'net_poloidal_current_Amperes', 'net_toroidal_current_Amperes', 'ntheta_plasma', 'nzeta_plasma', 'ntheta_coil', 'nzeta_coil']
        s = self.plasma_surface
        w = self.winding_surface
        G = self.current_potential.net_poloidal_current_amperes
        I = self.current_potential.net_toroidal_current_amperes
        scalar_variables = [s.nfp, s.mpol, s.ntor, w.mpol, w.ntor, s.stellsym + 1, G, I, self.ntheta_plasma, self.nzeta_plasma, self.ntheta_coil, self.nzeta_coil / s.nfp]
        for i_scalar, scalar_name in enumerate(scalars):
            f.createDimension(scalar_name, 1)
            if 'amperes' not in scalar_name:
                var = f.createVariable(scalar_name, 'i', (scalar_name,))
                var.units = 'dimensionless'
            else:
                var = f.createVariable(scalar_name, 'f', (scalar_name,))
                var.units = 'Amperes'
            var[:] = scalar_variables[i_scalar]

        # go through and compute all the rmnc, rmns for the plasma surface
        nfp = s.nfp
        xn_plasma = s.n * nfp
        xm_plasma = s.m
        rmnc = np.zeros(len(xm_plasma))
        zmns = np.zeros(len(xm_plasma))
        zmnc = np.zeros(len(xm_plasma))
        rmns = np.zeros(len(xm_plasma))
        for im in range(len(xm_plasma)):
            rmnc[im] = s.get_rc(xm_plasma[im], int(xn_plasma[im]/nfp))
            zmns[im] = s.get_zs(xm_plasma[im], int(xn_plasma[im]/nfp))
            if not s.stellsym:
                rmns[im] = s.get_rs(xm_plasma[im], int(xn_plasma[im]/nfp))
                zmnc[im] = s.get_zc(xm_plasma[im], int(xn_plasma[im]/nfp))

        rmnc_plasma = np.copy(rmnc)
        rmns_plasma = np.copy(rmns)
        zmns_plasma = np.copy(zmns)
        zmnc_plasma = np.copy(zmnc)

        # go through and compute all the rmnc, rmns for the coil surface
        xn_potential = self.current_potential.n * w.nfp
        xm_potential = self.current_potential.m
        xn_coil = w.n * w.nfp
        xm_coil = w.m
        rmnc = np.zeros(len(xm_coil))
        rmns = np.zeros(len(xm_coil))
        zmnc = np.zeros(len(xm_coil))
        zmns = np.zeros(len(xm_coil))
        nfp = w.nfp
        for im in range(len(xm_coil)):
            rmnc[im] = w.get_rc(xm_coil[im], int(xn_coil[im]/nfp))
            zmns[im] = w.get_zs(xm_coil[im], int(xn_coil[im]/nfp))
            if not w.stellsym:
                rmns[im] = w.get_rs(xm_coil[im], int(xn_coil[im]/nfp))
                zmnc[im] = w.get_zc(xm_coil[im], int(xn_coil[im]/nfp))

        # get the RHS b vector in the optimization
        RHS_B, _ = self.B_matrix_and_rhs()

        # Define geometric objects and then compute all the Bnormals
        points = s.gamma().reshape(-1, 3)
        normal = s.normal().reshape(-1, 3)
        ws_points = w.gamma().reshape(-1, 3)
        ws_normal = w.normal().reshape(-1, 3)
        dtheta_coil = w.quadpoints_theta[1]
        dzeta_coil = w.quadpoints_phi[1]
        Bnormal_totals = []
        Bnormal_totals_l1 = []
        if len(self.ilambdas_l2) > 0:
            for i, ilambda in enumerate(self.ilambdas_l2):
                # current_potential_l2 stores 1 field period; tile to full torus for WindingSurfaceBn_REGCOIL
                cp_full = np.tile(self.current_potential_l2[i], (nfp, 1))
                Bnormal_regcoil_sv = sopp.WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, cp_full, normal) * dtheta_coil * dzeta_coil
                Bnormal_totals.append((Bnormal_regcoil_sv + self.B_GI + self.Bnormal_plasma).reshape(self.ntheta_plasma, self.nzeta_plasma))
        if len(self.ilambdas_l1) > 0:
            for i, ilambda in enumerate(self.ilambdas_l1):
                # current_potential_l1 stores 1 field period; tile to full torus for WindingSurfaceBn_REGCOIL
                cp_full = np.tile(self.current_potential_l1[i], (nfp, 1))
                Bnormal_regcoil_sv = sopp.WindingSurfaceBn_REGCOIL(points, ws_points, ws_normal, cp_full, normal) * dtheta_coil * dzeta_coil
                Bnormal_totals_l1.append((Bnormal_regcoil_sv + self.B_GI + self.Bnormal_plasma).reshape(self.ntheta_plasma, self.nzeta_plasma))

        vectors = ['Bnormal_from_plasma_current', 'Bnormal_from_net_coil_currents',
                   'rmnc_plasma', 'rmns_plasma', 'zmns_plasma', 'zmnc_plasma',
                   'rmnc_coil', 'rmns_coil', 'zmns_coil', 'zmnc_coil',
                   'xm_plasma', 'xn_plasma', 'xm_coil', 'xn_coil',
                   'xm_potential', 'xn_potential',
                   'r_plasma', 'r_coil',
                   'theta_coil', 'zeta_coil',
                   'RHS_B', 'RHS_regularization',
                   'norm_normal_plasma', 'norm_normal_coil',
                   'single_valued_current_potential_mn', 'single_valued_current_potential_thetazeta',
                   'current_potential',
                   'K2', 'lambda', 'chi2_B', 'chi2_K', 'Bnormal_total',
                   'single_valued_current_potential_mn_l1', 'single_valued_current_potential_thetazeta_l1',
                   'current_potential_l1',
                   'K2_l1', 'lambda_l1', 'chi2_B_l1', 'chi2_K_l1', 'Bnormal_total_l1'
                   ]

        # Define the full plasma surface and few other geometric quantities
        quadpoints_phi = np.linspace(0, 1, self.nzeta_plasma * nfp + 1, endpoint=True)
        quadpoints_theta = np.linspace(0, 1, self.ntheta_plasma + 1, endpoint=True)
        quadpoints_phi = quadpoints_phi[:-1]
        quadpoints_theta = quadpoints_theta[:-1]
        sf = SurfaceRZFourier(
            nfp=s.nfp,
            mpol=s.mpol,
            ntor=s.ntor,
            stellsym=True,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta
        )
        sf.set_dofs(0 * sf.get_dofs())
        for im in range(len(xm_plasma)):
            sf.set_rc(xm_plasma[im], int(xn_plasma[im] / s.nfp), rmnc_plasma[im])
            sf.set_zs(xm_plasma[im], int(xn_plasma[im] / s.nfp), zmns_plasma[im])
            if not sf.stellsym:
                sf.set_rs(xm_plasma[im], int(xn_plasma[im] / s.nfp), rmns_plasma[im])
                sf.set_zc(xm_plasma[im], int(xn_plasma[im] / s.nfp), zmnc_plasma[im])

        norm_normal_plasma = np.linalg.norm(s.normal(), axis=-1) / (2 * np.pi * 2 * np.pi)
        norm_normal_coil = np.linalg.norm(w.normal(), axis=-1) / (2 * np.pi * 2 * np.pi)

        current_potential = []
        for i in range(len(self.current_potential_l2)):
            current_potential.append(self.current_potential_l2[i] + self.current_potential.current_potential_secular[:self.nzeta_coil // nfp, :])

        current_potential_l1 = []
        for i in range(len(self.current_potential_l1)):
            current_potential_l1.append(self.current_potential_l1[i] + self.current_potential.current_potential_secular[:self.nzeta_coil // nfp, :])

        # Define all the vectors we need to save
        vector_variables = [self.Bnormal_plasma.reshape(self.ntheta_plasma, self.nzeta_plasma),
                            self.B_GI.reshape(self.ntheta_plasma, self.nzeta_plasma),
                            rmnc_plasma, rmns_plasma, zmns_plasma, zmnc_plasma,
                            rmnc, rmns, zmns, zmnc,
                            xm_plasma[:(len(xm_plasma)) // 2 + 1], xn_plasma[:(len(xm_plasma)) // 2 + 1],
                            xm_coil[:(len(xm_coil)) // 2 + 1], xn_coil[:(len(xm_coil)) // 2 + 1],
                            xm_potential, xn_potential,
                            sf.gamma(),
                            w.gamma(),
                            w.quadpoints_theta * 2 * np.pi,
                            w.quadpoints_phi[:self.nzeta_coil // w.nfp] * 2 * np.pi,
                            RHS_B, self.K_rhs(), norm_normal_plasma, norm_normal_coil,
                            np.array(self.dofs_l2), np.array(self.current_potential_l2),
                            np.array(current_potential),
                            np.array(self.K2s_l2)[:, :self.nzeta_coil // w.nfp, :], 
                            np.array(self.ilambdas_l2),
                            2 * np.array(self.fBs_l2), 2 * np.array(self.fKs_l2), 
                            np.array(Bnormal_totals),
                            np.array(self.dofs_l1), np.array(self.current_potential_l1),
                            np.array(current_potential_l1),
                            np.array(self.K2s_l1)[:, :self.nzeta_coil // w.nfp, :], np.array(self.ilambdas_l1),
                            2 * np.array(self.fBs_l1), 2 * np.array(self.fKs_l1), np.array(Bnormal_totals_l1)
                            ]

        # Loop through and save all the vector variables
        for i_vector, vector_name in enumerate(vectors):
            vector_shape = vector_variables[i_vector].shape
            shape_tuple = (vector_name + '0', )
            for j, vshape in enumerate(vector_shape):
                f.createDimension(vector_name + str(j), vshape)
                if j > 0:
                    shape_tuple = shape_tuple + (vector_name + str(j),)
            var = f.createVariable(vector_name, 'f', shape_tuple)
            if 'Bnormal' in vector_name:
                var.units = 'Tesla'
            elif 'chi2_B' in vector_name:
                var.units = 'Tesla^2 m^2'
            elif 'chi2_K' in vector_name:
                var.units = 'Ampere^2 / m^2'
            elif 'thetazeta' in vector_name:
                var.units = 'Ampere / m'
            elif 'rmn' in vector_name or 'zmn' in vector_name or 'r_' in vector_name or 'norm_' in vector_name:
                var.units = 'm'
            elif 'lambda' in vector_name:  # lambda * chi2_K must have same units as chi2_B
                var.units = 'Tesla^2 * m^4 / Ampere^2'
            else:
                var.units = 'dimensionless'
            var[:] = vector_variables[i_vector]

        f.close()

    def K_rhs_impl(self, K_rhs):
        """
        Implied function for the K rhs for the REGCOIL problem.

        Args:
            K_rhs: The K rhs to be computed.
        """
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_rhs_impl_helper(K_rhs, dg1, dg2, normal)
        K_rhs *= self.winding_surface.quadpoints_theta[1] * self.winding_surface.quadpoints_phi[1] / self.winding_surface.nfp

    def K_rhs(self):
        """
        Compute the K rhs for the REGCOIL problem.

        Args:
            None

        Returns:
            K_rhs: The K rhs for the REGCOIL problem.
        """
        K_rhs = np.zeros((self.current_potential.num_dofs(),))
        self.K_rhs_impl(K_rhs)
        return K_rhs

    def K_matrix_impl(self, K_matrix):
        """
        Implied function for the K matrix for the REGCOIL problem.

        Args:
            K_matrix: The K matrix to be computed.
        """
        dg1 = self.winding_surface.gammadash1()
        dg2 = self.winding_surface.gammadash2()
        normal = self.winding_surface.normal()
        self.current_potential.K_matrix_impl_helper(K_matrix, dg1, dg2, normal)
        K_matrix *= self.winding_surface.quadpoints_theta[1] * self.winding_surface.quadpoints_phi[1] / self.winding_surface.nfp

    def K_matrix(self):
        """
        Compute the K matrix for the REGCOIL problem.
        
        Args:
            None

        Returns:
            K_matrix: The K matrix for the REGCOIL problem.
        """
        K_matrix = np.zeros((self.current_potential.num_dofs(), self.current_potential.num_dofs()))
        self.K_matrix_impl(K_matrix)
        return K_matrix

    def B_matrix_and_rhs(self):
        """
            Compute the matrices and right-hand-side corresponding the Bnormal part of
            the optimization, for both the Tikhonov and Lasso optimizations.
        """
        t0 = time.perf_counter() if TIMING else 0
        plasma_surface = self.plasma_surface
        normal = self.winding_surface.normal().reshape(-1, 3)
        Bnormal_plasma = self.Bnormal_plasma
        normal_plasma = plasma_surface.normal().reshape(-1, 3)
        points_plasma = plasma_surface.gamma().reshape(-1, 3)
        points_coil = self.winding_surface.gamma().reshape(-1, 3)
        theta = self.winding_surface.quadpoints_theta
        phi_mesh, theta_mesh = np.meshgrid(self.winding_surface.quadpoints_phi, theta, indexing='ij')
        zeta_coil = np.ravel(phi_mesh)
        theta_coil = np.ravel(theta_mesh)
        if TIMING:
            print(f"        [B_matrix_and_rhs] setup (gamma, normal, mesh): {time.perf_counter()-t0:.3f}s")

        if self.winding_surface.stellsym:
            ndofs_half = self.current_potential.num_dofs()
        else:
            ndofs_half = self.current_potential.num_dofs() // 2

        # Compute terms for the REGCOIL (L2) problem
        t0 = time.perf_counter() if TIMING else 0
        contig = np.ascontiguousarray
        gj, B_matrix = sopp.winding_surface_field_Bn(contig(points_plasma), contig(points_coil), contig(normal_plasma), contig(normal), self.winding_surface.stellsym, contig(zeta_coil), contig(theta_coil), self.current_potential.num_dofs(), contig(self.current_potential.m[:ndofs_half]), contig(self.current_potential.n[:ndofs_half]), self.winding_surface.nfp)
        if TIMING:
            print(f"        [B_matrix_and_rhs] winding_surface_field_Bn: {time.perf_counter()-t0:.3f}s")
        B_GI = self.B_GI

        # set up RHS of optimization
        t0 = time.perf_counter() if TIMING else 0
        b_rhs = - np.ravel(B_GI + Bnormal_plasma) @ gj
        dzeta_plasma = (plasma_surface.quadpoints_phi[1] - plasma_surface.quadpoints_phi[0])
        dtheta_plasma = (plasma_surface.quadpoints_theta[1] - plasma_surface.quadpoints_theta[0])
        dzeta_coil = (self.winding_surface.quadpoints_phi[1] - self.winding_surface.quadpoints_phi[0])
        dtheta_coil = (self.winding_surface.quadpoints_theta[1] - self.winding_surface.quadpoints_theta[0])

        # scale bmatrix and b_rhs by factors of the grid spacing
        b_rhs = b_rhs * dzeta_plasma * dtheta_plasma * dzeta_coil * dtheta_coil
        B_matrix = B_matrix * dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2
        normN = np.linalg.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1)
        self.gj = gj * np.sqrt(dzeta_plasma * dtheta_plasma * dzeta_coil ** 2 * dtheta_coil ** 2)
        self.b_e = - np.sqrt(normN * dzeta_plasma * dtheta_plasma) * (B_GI + Bnormal_plasma)
        if TIMING:
            print(f"        [B_matrix_and_rhs] RHS scaling: {time.perf_counter()-t0:.3f}s")

        t0 = time.perf_counter() if TIMING else 0
        normN = np.linalg.norm(self.winding_surface.normal().reshape(-1, 3), axis=-1)
        dr_dzeta = self.winding_surface.gammadash1().reshape(-1, 3)
        dr_dtheta = self.winding_surface.gammadash2().reshape(-1, 3)
        if TIMING:
            print(f"        [B_matrix_and_rhs] gammadash1/2 (K2 prep): {time.perf_counter()-t0:.3f}s")
        G = self.current_potential.net_poloidal_current_amperes
        I = self.current_potential.net_toroidal_current_amperes

        normal_coil = self.winding_surface.normal().reshape(-1, 3)
        m = self.current_potential.m[:ndofs_half]
        n = self.current_potential.n[:ndofs_half]
        nfp = self.winding_surface.nfp

        contig = np.ascontiguousarray

        # Compute terms for the Lasso (L1) problem
        t0 = time.perf_counter() if TIMING else 0
        d, fj = sopp.winding_surface_field_K2_matrices(
            contig(dr_dzeta), contig(dr_dtheta), contig(normal_coil), self.winding_surface.stellsym,
            contig(zeta_coil), contig(theta_coil), self.ndofs, contig(m), contig(n), nfp, G, I
        )
        self.fj = fj * 2 * np.pi * np.sqrt(dzeta_coil * dtheta_coil)
        self.d = d * 2 * np.pi * np.sqrt(dzeta_coil * dtheta_coil)
        if TIMING:
            print(f"        [B_matrix_and_rhs] winding_surface_field_K2_matrices: {time.perf_counter()-t0:.3f}s")
        return b_rhs, B_matrix

    def solve_tikhonov(self, lam=0, record_history=True):
        """
            Solve the REGCOIL problem -- winding surface optimization with
            the L2 norm. This is tested against REGCOIL runs extensively in
            tests/field/test_regcoil.py.

            Args:
                lam: Regularization parameter for the Tikhonov regularization.
                record_history: Whether to record the history of the optimization.

            Returns:
                phi_mn_opt: The optimized winding surface potential.
                f_B: The value of the Bnormal loss term.
                f_K: The value of the K loss term.

            Notes:
                This function solves the REGCOIL problem with Tikhonov regularization.
                The regularization term is added to the Bnormal and K matrices to
                prevent overfitting. The optimization is performed using a least-squares
                solve. The history of the optimization is recorded if record_history is True.
                The history is recorded in the ilambdas_l2, dofs_l2, current_potential_l2,
                fBs_l2, and fKs_l2 lists.
        """
        t0 = time.perf_counter() if TIMING else 0
        K_matrix = self.K_matrix()
        if TIMING:
            print(f"    [solve_tikhonov] K_matrix: {time.perf_counter()-t0:.3f}s")
        t0 = time.perf_counter() if TIMING else 0
        K_rhs = self.K_rhs()
        if TIMING:
            print(f"    [solve_tikhonov] K_rhs: {time.perf_counter()-t0:.3f}s")
        t0 = time.perf_counter() if TIMING else 0
        b_rhs, B_matrix = self.B_matrix_and_rhs()
        if TIMING:
            print(f"    [solve_tikhonov] B_matrix_and_rhs: {time.perf_counter()-t0:.3f}s")

        # least-squares solve
        t0 = time.perf_counter() if TIMING else 0
        phi_mn_opt = np.linalg.solve(B_matrix + lam * K_matrix, b_rhs + lam * K_rhs)
        if TIMING:
            print(f"    [solve_tikhonov] linalg.solve: {time.perf_counter()-t0:.3f}s")
        self.current_potential.set_dofs(phi_mn_opt)

        # Get other matrices for direct computation of fB and fK loss terms
        nfp = self.plasma_surface.nfp
        normN = np.linalg.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1)
        A_times_phi = self.gj @ phi_mn_opt / np.sqrt(normN)
        b_e = self.b_e
        Ak_times_phi = self.fj @ phi_mn_opt
        f_B = 0.5 * np.linalg.norm(A_times_phi - b_e) ** 2 * nfp
        # extra normN factor needed here because fj and d don't have it
        # K^2 has 1/normn^2 factor, the sum over the winding surface has factor of normn,
        # for total factor of 1/normn
        normN = np.linalg.norm(self.winding_surface.normal().reshape(-1, 3), axis=-1)
        f_K = 0.5 * np.linalg.norm((Ak_times_phi - self.d) / np.sqrt(normN[:, None])) ** 2

        if record_history:
            self.ilambdas_l2.append(lam)
            self.dofs_l2.append(phi_mn_opt)
            # REGCOIL only uses 1 / 2 nfp of the winding surface
            self.current_potential_l2.append(np.copy(self.current_potential.Phi()[:self.nzeta_coil // nfp, :]))
            self.fBs_l2.append(f_B)
            self.fKs_l2.append(f_K)
            K2 = np.sum(self.current_potential.K() ** 2, axis=2)
            self.K2s_l2.append(K2)
        return phi_mn_opt, f_B, f_K

    def solve_lasso(self, lam=0, max_iter=1000, acceleration=True):
        """
            Solve the Lasso problem -- winding surface optimization with
            the L1 norm, which should tend to allow stronger current
            filaments to form than the L2. There are a couple changes to make:

            1. Need to define new optimization variable z = A_k * phi_mn - b_k
               so that optimization becomes
               ||AA_k^{-1} * z - (b - A * A_k^{-1} * b_k)||_2^2 + alpha * ||z||_1
               which is the form required to use the Lasso pre-built optimizer
               from sklearn (which actually works poorly) or the optimizer used
               here (proximal gradient descent for LASSO, also called ISTA or FISTA).
            2. The alpha term should be similar amount of regularization as the L2
               so we rescale lam -> sqrt(lam) since lam is used for the (L2 norm)^2
               loss term used for Tikhonov regularization.


            We use the FISTA algorithm but you could use scikit-learn's Lasso
            optimizer too. Like any gradient-based algorithm, both FISTA
            and Lasso will work very poorly at low regularization since convergence
            goes like the condition number of the fB matrix. In both cases,
            this can be addressed by using the exact Tikhonov solution (obtained
            with a matrix inverse instead of a gradient-based optimization) as
            an initial guess to the optimizers.
        """
        # Set up some matrices
        t0 = time.perf_counter() if TIMING else 0
        _, _ = self.B_matrix_and_rhs()
        if TIMING:
            print(f"    [solve_lasso] B_matrix_and_rhs: {time.perf_counter()-t0:.3f}s")
        t0 = time.perf_counter() if TIMING else 0
        normN = np.linalg.norm(self.plasma_surface.normal().reshape(-1, 3), axis=-1)
        ws_normN = np.linalg.norm(self.winding_surface.normal().reshape(-1, 3), axis=-1)
        A_matrix = self.gj
        for i in range(self.gj.shape[0]):
            A_matrix[i, :] *= (1.0 / np.sqrt(normN[i]))
        b_e = self.b_e
        fj = self.fj / np.sqrt(ws_normN)[:, None, None]
        d = self.d / np.sqrt(ws_normN)[:, None]
        Ak_matrix = fj.reshape(fj.shape[0] * 3, fj.shape[-1])
        d = np.ravel(d)
        nfp = self.plasma_surface.nfp

        # Ak is non-square so pinv required. Careful with rcond parameter
        Ak_inv = np.linalg.pinv(Ak_matrix, rcond=1e-10)
        A_new = A_matrix @ Ak_inv
        b_new = b_e - A_new @ d
        if TIMING:
            print(f"    [solve_lasso] A_matrix setup + pinv: {time.perf_counter()-t0:.3f}s")
        # print('Checking Ak_inv computed with pinv = ', Ak_inv)

        # rescale the l1 regularization
        l1_reg = lam
        # l1_reg = np.sqrt(lam)

        # if alpha << 1, want to use initial guess from the Tikhonov solve,
        # which is exact since it comes from a matrix inverse.
        t0 = time.perf_counter() if TIMING else 0
        phi0, _, _, = self.solve_tikhonov(lam=lam, record_history=False)
        if TIMING:
            print(f"    [solve_lasso] solve_tikhonov (initial guess): {time.perf_counter()-t0:.3f}s")

        # L1 norm here should already include the contributions from the winding surface discretization
        # and factor of 1 / ws_normN from the K, cancelling the factor of ws_normN from the surface
        z0 = np.ravel((Ak_matrix @ phi0 - d).reshape(-1, 3) * np.sqrt(ws_normN)[:, None])
        t0 = time.perf_counter() if TIMING else 0
        z_opt, z_history = self._FISTA(A=A_new, b=b_new, alpha=l1_reg, max_iter=max_iter, acceleration=acceleration, xi0=z0)
        if TIMING:
            print(f"    [solve_lasso] _FISTA: {time.perf_counter()-t0:.3f}s")
        # z_opt = z_opt.reshape(-1, 3) # / ws_normN[:, None])
        # Need to put back in the 1 / ws_normN dependence in K

        # Compute the history of values from the optimizer
        phi_history = []
        fB_history = []
        fK_history = []
        for i in range(len(z_history)):
            phi_history.append(Ak_inv @ (z_history[i] + d))
            fB_history.append(0.5 * np.linalg.norm(A_matrix @ phi_history[i] - b_e) ** 2 * nfp)
            fK_history.append(np.linalg.norm(z_history[i], ord=1))
            # fK_history.append(np.linalg.norm(Ak_matrix @ phi_history[i] - d, ord=1))

        # Remember, Lasso solved for z = A_k * phi_mn - b_k so need to convert back
        phi_mn_opt = Ak_inv @ (z_opt + d)
        self.current_potential.set_dofs(phi_mn_opt)
        f_B = 0.5 * np.linalg.norm(A_matrix @ phi_mn_opt - b_e) ** 2 * nfp
        f_K = np.linalg.norm(Ak_matrix @ phi_mn_opt - d, ord=1)
        self.ilambdas_l1.append(lam)
        self.dofs_l1.append(phi_mn_opt)
        # REGCOIL only uses 1 / 2 nfp of the winding surface
        self.current_potential_l1.append(np.copy(self.current_potential.Phi()[:self.nzeta_coil // nfp, :]))
        self.fBs_l1.append(f_B)
        self.fKs_l1.append(f_K)
        K2 = np.sum(self.current_potential.K() ** 2, axis=2)
        self.K2s_l1.append(K2)
        return phi_mn_opt, f_B, f_K, fB_history, fK_history

    def _FISTA(self, A, b, alpha=0.0, max_iter=1000, acceleration=True, xi0=None):
        """
        This function uses Nesterov's accelerated proximal
        gradient descent algorithm to solve the Lasso
        (L1-regularized) winding surface problem. This is
        usually called fast iterative soft-thresholding algorithm
        (FISTA). If acceleration = False, it will use the ISTA
        algorithm, which tends to converge much slower than FISTA
        but is a true descent algorithm, unlike FISTA. Here n = the
        number of plasma quadrature points and 
        m = the number of winding surface DOFs. The algorithm is:

        .. math::
            x_{k+1} = \text{prox}_{\alpha \| \cdot \|_1}(x_k + \frac{1}{L} (b - A x_k))
            x_0 = \text{prox}_{\alpha \| \cdot \|_1}(x_0)

        Args:
            A (shape (n, m)): The A matrix.
            b (shape (n,)): The b vector.
            alpha (float): The regularization parameter.
            max_iter (int): The maximum number of iterations.
            acceleration (bool): Whether to use acceleration.
            xi0 (shape (m,)): The initial guess.

        Returns:
            z_opt: The optimized z vector.
            z_history: The history of the z vector.
        """
        from scipy.sparse.linalg import svds

        # Tolerance for algorithm to consider itself converged
        tol = 1e-5

        # pre-compute/load some stuff so for loops (below) are faster
        t0 = time.perf_counter() if TIMING else 0
        AT = A.T
        ATb = AT @ b
        if TIMING:
            print(f"      [_FISTA] AT/ATb: {time.perf_counter()-t0:.3f}s")
        prox = self._prox_l1

        # L = largest eigenvalue of A^T A = (largest singular value of A)^2.
        # svds uses iterative methods (ARPACK) with only A@v and A.T@v matvecs,
        # avoiding O(n*m^2) cost of forming A^T A explicitly.
        t0 = time.perf_counter() if TIMING else 0
        try:
            sigma_max = svds(A, k=1, which='LM', return_singular_vectors=False)[0]
            L = sigma_max ** 2
        except Exception:
            ATA = AT @ A
            L = np.max(np.sum(np.abs(ATA), axis=1))  # Gershgorin upper bound
        if TIMING:
            print(f"      [_FISTA] svds (Lipschitz): {time.perf_counter()-t0:.3f}s")

        # initial step size should be just smaller than 1 / L
        # which for most of these problems L ~ 1e-13 or smaller
        # so the step size is enormous
        ti = 1.0 / L

        # initialize current potential to random values in [5e-4, 5e4]
        if xi0 is None:
            xi0 = (np.random.rand(A.shape[1]) - 0.5) * 1e5
        x_history = [xi0]
        if acceleration:  # FISTA algorithm
            # first iteration do ISTA
            x_prev = xi0
            x = prox(xi0 + ti * (ATb - AT @ (A @ xi0)), ti * alpha)
            for i in range(1, max_iter):
                vi = x + i / (i + 3) * (x - x_prev)
                x_prev = x
                # note l1 'threshold' is rescaled here
                x = prox(vi + ti * (ATb - AT @ (A @ vi)), ti * alpha)
                ti = (1 + np.sqrt(1 + 4 * ti ** 2)) / 2.0
                if (i % 100) == 0:
                    x_history.append(x)
                    if np.all(abs(x_history[-1] - x_history[-2]) / max(1e-10, np.mean(np.abs(x_history[-2]))) < tol):
                        if TIMING:
                            print(f"      [_FISTA] converged at iteration {i}")
                        break
            if TIMING:
                print(f"      [_FISTA] iterations: {i+1}")
        else:  # ISTA algorithm (forms ATA only when needed)
            alpha = ti * alpha  # ti does not vary in ISTA algorithm
            ATA = ti * (AT @ A)
            I_ATA = np.eye(ATA.shape[0]) - ATA
            ATb_scaled = ti * ATb
            for i in range(max_iter):
                x_history.append(prox(ATb_scaled + I_ATA @ x_history[i], alpha))
                if (i % 100) == 0:
                    if np.all(abs(x_history[i + 1] - x_history[i]) < tol):
                        break
        xi = x_history[-1]
        return xi, x_history

    def _prox_l1(self, x, threshold):
        """
        Proximal operator for L1 regularization,
        which is often called soft-thresholding.

        .. math::
            \text{prox}_{\lambda \| \cdot \|_1}(x) = \text{sign}(x) \max(0, |x| - \lambda)

        Args:
            x (shape (n,)): The x vector.
            threshold (float): The threshold for the L1 regularization.

        Returns:
            prox_x (shape (n,)): The proximal operator of the L1 regularization.
        """
        return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)
