import logging
from matplotlib import pyplot as plt
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from scipy.sparse import csr_matrix
import simsoptpp as sopp
import time

logger = logging.getLogger(__name__)


class PermanentMagnetOptimizer:
    r"""
        ``PermanentMagnetOptimizer`` is a class for solving the permanent
        magnet optimization problem for stellarators. The class
        takes as input two toroidal surfaces specified as SurfaceRZFourier
        objects, and initializes a set of points (in cylindrical coordinates)
        between these surfaces. It finishes initialization by pre-computing
        a number of quantities required for the optimization such as the
        geometric factor in the dipole part of the magnetic field and the 
        target Bfield that is a sum of the coil and plasma magnetic fields.
        It then provides functions for solving several variations of the
        optimization problem.

        Args:
            plasma_boundary:  SurfaceRZFourier object representing 
                              the plasma boundary surface.  
            rz_inner_surface: SurfaceRZFourier object representing 
                              the inner toroidal surface of the volume.
                              Defaults to the plasma boundary, extended
                              by the boundary normal vector projected on
                              to the (R, Z) plane to keep the phi = constant
                              cross-section the same as the plasma surface. 
            rz_outer_surface: SurfaceRZFourier object representing 
                              the outer toroidal surface of the volume.
                              Defaults to the inner surface, extended
                              by the boundary normal vector projected on
                              to the (R, Z) plane to keep the phi = constant
                              cross-section the same as the inner surface. 
            plasma_offset:    Offset to use for generating the inner toroidal surface.
            coil_offset:      Offset to use for generating the outer toroidal surface.
            B_plasma_surface: Magnetic field (coils and plasma) at the plasma
                              boundary. Typically this will be the optimized plasma
                              magnetic field from a stage-1 optimization, and the
                              optimized coils from a basic stage-2 optimization. 
                              This variable must be specified to run the permanent
                              magnet optimization.
            dr:               Radial and axial grid spacing in the permanent magnet manifold.
            filename:         Filename for the file containing the plasma boundary surface.
            FOCUS:            Flag to specify if the file containing the plasma boundary 
                              surface is in FOCUS format. Defaults to False, in which case the
                              file is assumed to be in VMEC format. 
    """

    def __init__(
        self, plasma_boundary, rz_inner_surface=None, 
        rz_outer_surface=None, plasma_offset=0.1, 
        coil_offset=0.1, B_plasma_surface=None, dr=0.1,
        filename=None, FOCUS=False,
    ):
        if plasma_offset <= 0 or coil_offset <= 0:
            raise ValueError('permanent magnets must be offset from the plasma')

        self.filename = filename
        self.FOCUS = FOCUS
        self.plasma_offset = plasma_offset
        self.coil_offset = coil_offset
        self.B_plasma_surface = B_plasma_surface
        self.dr = dr

        if not isinstance(plasma_boundary, SurfaceRZFourier):
            raise ValueError(
                'Plasma boundary must be specified as SurfaceRZFourier class.'
            )
        else:
            self.plasma_boundary = plasma_boundary
            self.phi = self.plasma_boundary.quadpoints_phi
            self.nphi = len(self.phi)
            self.theta = self.plasma_boundary.quadpoints_theta
            self.ntheta = len(self.theta)

        if dr <= 0:
            raise ValueError('dr grid spacing must be > 0')

        t1 = time.time()
        # If the inner surface is not specified, make default surface.
        if rz_inner_surface is None:
            print(
                "Inner toroidal surface not specified, defaulting to "
                "extending the plasma boundary shape using the normal "
                " vectors at the quadrature locations. The normal vectors "
                " are projected onto the RZ plane so that the new surface "
                " lies entirely on the original phi = constant cross-section. "
            )
            if self.filename is None:
                raise ValueError(
                    "Must specify a filename for loading in a toroidal surface"
                )
            self._set_inner_rz_surface()
        else:
            self.rz_inner_surface = rz_inner_surface
        if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
            raise ValueError("Inner surface is not SurfaceRZFourier object.")

        # If the outer surface is not specified, make default surface. 
        if rz_outer_surface is None:
            print(
                "Outer toroidal surface not specified, defaulting to "
                "extending the inner toroidal surface shape using the normal "
                " vectors at the quadrature locations. The normal vectors "
                " are projected onto the RZ plane so that the new surface "
                " lies entirely on the original phi = constant cross-section. "
            )
            if self.filename is None:
                raise ValueError(
                    "Must specify a filename for loading in a toroidal surface"
                )
            self._set_outer_rz_surface()
        else:
            self.rz_outer_surface = rz_outer_surface
        if not isinstance(self.rz_outer_surface, SurfaceRZFourier):
            raise ValueError("Outer surface is not SurfaceRZFourier object.")
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to get the SurfaceRZFourier objects done") 

        # check the inner and outer surface are same size
        # and defined at the same (theta, phi) coordinate locations
        if len(self.rz_inner_surface.quadpoints_theta) != len(self.rz_outer_surface.quadpoints_theta):
            raise ValueError(
                "Inner and outer toroidal surfaces have different number of "
                "poloidal quadrature points."
            )
        if len(self.rz_inner_surface.quadpoints_phi) != len(self.rz_outer_surface.quadpoints_phi):
            raise ValueError(
                "Inner and outer toroidal surfaces have different number of "
                "toroidal quadrature points."
            )
        if np.any(self.rz_inner_surface.quadpoints_theta != self.rz_outer_surface.quadpoints_theta):
            raise ValueError(
                "Inner and outer toroidal surfaces must be defined at the "
                "same poloidal quadrature points."
            )
        if np.any(self.rz_inner_surface.quadpoints_phi != self.rz_outer_surface.quadpoints_phi):
            raise ValueError(
                "Inner and outer toroidal surfaces must be defined at the "
                "same toroidal quadrature points."
            )

        RPhiZ_grid = self._setup_uniform_grid()

        # Have the uniform grid, now need to loop through and eliminate cells.
        t1 = time.time()
        self.final_RZ_grid, self.inds = sopp.make_final_surface(
            2 * np.pi * self.phi, self.normal_inner, self.normal_outer, 
            RPhiZ_grid, self.r_inner, self.r_outer, self.z_inner, self.z_outer
        )
        self.inds = np.array(self.inds, dtype=int)
        self.ndipoles = self.inds[-1]
        self.final_RZ_grid = self.final_RZ_grid[:self.ndipoles, :, :]
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to perform the C++ grid cell eliminations.")

        # Make flattened grids and compute the maximum allowable magnetic moment m_max
        t1 = time.time()
        self._make_flattened_grids()
        t2 = time.time()
        print("Grid setup took t = ", t2 - t1, " s")

        # Setup least-squares term for the impending optimization
        t1 = time.time()
        self._geo_setup()
        t2 = time.time()
        print("C++ geometric setup, t = ", t2 - t1, " s")

        # optionally plot the plasma boundary + inner/outer surfaces
        self._plot_surfaces()

    def _setup_uniform_grid(self):
        """ 
            Initializes a uniform grid in cylindrical coordinates and sets
            some important grid variables for later.
        """
        # Get (R, Z) coordinates of the three boundaries
        xyz_plasma = self.plasma_boundary.gamma()
        self.r_plasma = np.sqrt(xyz_plasma[:, :, 0] ** 2 + xyz_plasma[:, :, 1] ** 2)
        self.phi_plasma = np.arctan2(xyz_plasma[:, :, 1], xyz_plasma[:, :, 0])
        self.z_plasma = xyz_plasma[:, :, 2]
        xyz_inner = self.rz_inner_surface.gamma()
        self.r_inner = np.sqrt(xyz_inner[:, :, 0] ** 2 + xyz_inner[:, :, 1] ** 2)
        self.z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.rz_outer_surface.gamma()
        self.r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        self.z_outer = xyz_outer[:, :, 2]

        self.normal_inner = np.copy(self.rz_inner_surface.unitnormal())
        self.normal_outer = np.copy(self.rz_outer_surface.unitnormal())
        r_max = np.max(self.r_outer)
        r_min = np.min(self.r_outer)
        z_max = np.max(self.z_outer)
        z_min = np.min(self.z_outer)

        # Initialize uniform grid of curved, square bricks
        self.Delta_r = self.dr 
        Nr = int((r_max - r_min) / self.Delta_r)
        self.Nr = Nr
        self.Delta_z = self.Delta_r
        Nz = int((z_max - z_min) / self.Delta_z)
        self.Nz = Nz
        phi = 2 * np.pi * np.copy(self.rz_outer_surface.quadpoints_phi)
        print('Largest possible dipole dimension is = {0:.2f}'.format(
            (
                r_max - self.plasma_boundary.get_rc(0, 0)
            ) * (phi[1] - phi[0])
        )
        )
        print('dR = {0:.2f}'.format(self.Delta_r))
        print('dZ = {0:.2f}'.format(self.Delta_z))
        norm = self.plasma_boundary.unitnormal()
        norms = np.zeros(norm.shape)
        for i in range(self.nphi):
            rot_matrix = [[np.cos(phi[i]), np.sin(phi[i]), 0],
                          [-np.sin(phi[i]), np.cos(phi[i]), 0],
                          [0, 0, 1]]
            for j in range(self.ntheta):
                norms[i, j, :] = rot_matrix @ norm[i, j, :]
        print(
            'Approximate closest distance between plasma and inner surface = {0:.2f}'.format(
                np.min(np.sqrt(norms[:, :, 0] ** 2 + norms[:, :, 2] ** 2) * self.plasma_offset)
            )    
        )
        self.plasma_unitnormal_cylindrical = norms
        R = np.linspace(r_min, r_max, Nr)
        Z = np.linspace(z_min, z_max, Nz)

        # Make 3D mesh
        R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
        self.RPhiZ = np.transpose(np.array([R, Phi, Z]), [1, 2, 3, 0])
        RPhiZ_grid = np.transpose(self.RPhiZ, [0, 2, 1, 3])
        RPhiZ_grid = RPhiZ_grid.reshape(RPhiZ_grid.shape[0] * RPhiZ_grid.shape[1], RPhiZ_grid.shape[2], 3)
        return RPhiZ_grid

    def _make_flattened_grids(self):
        """
            Takes the final cylindrical grid and flattens it into 
            two flat grids in cylindrical and cartesian. Also computes
            the m_maxima factors needed for the constraints in the
            optimization.
        """
        phi = 2 * np.pi * self.phi
        B_max = 1.4
        mu0 = 4 * np.pi * 1e-7
        dipole_grid_r = np.zeros(self.ndipoles)
        dipole_grid_x = np.zeros(self.ndipoles)
        dipole_grid_y = np.zeros(self.ndipoles)
        dipole_grid_phi = np.zeros(self.ndipoles)
        dipole_grid_z = np.zeros(self.ndipoles)
        running_tally = 0
        for i in range(self.nphi):
            if i > 0:
                radii = self.final_RZ_grid[self.inds[i-1]:self.inds[i], i, 0]
                z_coords = self.final_RZ_grid[self.inds[i-1]:self.inds[i], i, 2]
            else:
                radii = self.final_RZ_grid[:self.inds[i], i, 0]
                z_coords = self.final_RZ_grid[:self.inds[i], i, 2]
            len_radii = len(radii)
            dipole_grid_r[running_tally:running_tally + len_radii] = radii
            dipole_grid_phi[running_tally:running_tally + len_radii] = phi[i]
            dipole_grid_x[running_tally:running_tally + len_radii] = radii * np.cos(phi[i])
            dipole_grid_y[running_tally:running_tally + len_radii] = radii * np.sin(phi[i])
            dipole_grid_z[running_tally:running_tally + len_radii] = z_coords
            running_tally += len_radii
        self.dipole_grid = np.array([dipole_grid_r, dipole_grid_phi, dipole_grid_z]).T
        self.dipole_grid_xyz = np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).T

        cell_vol = abs(dipole_grid_r - self.plasma_boundary.get_rc(0, 0)) * self.Delta_r * self.Delta_z * (phi[1] - phi[0])
        # cell_vol = dipole_grid_r * Delta_r * Delta_z * (phi[1] - phi[0])

        # FAMUS paper as typo that m_max = B_r / (mu0 * cell_vol) but it 
        # should be m_max = B_r * cell_vol / mu0  (just from units)
        self.m_maxima = B_max * cell_vol / mu0

    def _geo_setup(self):
        """
            Initialize 'b' vector in 0.5 * ||Am - b||^2 part of the optimization,
            corresponding to the normal component of the target fields. Note
            the factor of two in the least-squares term: 0.5 * m.T @ (A.T @ A) @ m - b.T @ m.
            Also initialize the A, A.T @ b, and A.T @ A matrices that are important in the
            optimization problem.
        """
        # set up 'b' vector 
        if self.B_plasma_surface.shape != (self.nphi, self.ntheta, 3):
            raise ValueError('Magnetic field surface data is incorrect shape.')
        Bs = self.B_plasma_surface
        dphi = (self.phi[1] - self.phi[0]) * 2 * np.pi
        dtheta = (self.theta[1] - self.theta[0]) * 2 * np.pi
        self.dphi = dphi
        self.dtheta = dtheta
        grid_fac = np.sqrt(dphi * dtheta)

        # minus sign below because ||Ax - b||^2 term but original
        # term is integral(B_P + B_C + B_M)
        self.b_obj = - np.sum(
            Bs * self.plasma_boundary.unitnormal(), axis=2
        ).reshape(self.nphi * self.ntheta) * grid_fac 

        # Compute geometric factor with the C++ routine
        self.A_obj, self.ATb, self.ATA = sopp.dipole_field_Bn(
            np.ascontiguousarray(self.plasma_boundary.gamma().reshape(self.nphi * self.ntheta, 3)), 
            np.ascontiguousarray(self.dipole_grid_xyz), 
            np.ascontiguousarray(self.plasma_boundary.unitnormal().reshape(self.nphi * self.ntheta, 3)), 
            self.plasma_boundary.nfp, int(self.plasma_boundary.stellsym), 
            np.ascontiguousarray(self.dipole_grid[:, 1]), 
            np.ascontiguousarray(self.b_obj)
        )
        # Rescale
        self.A_obj_expanded = self.A_obj * grid_fac
        self.A_obj = self.A_obj_expanded.reshape(self.nphi * self.ntheta, self.ndipoles * 3)
        self.ATb = np.ravel(self.ATb) * grid_fac  # only one factor here since b is already scaled!
        self.ATA = self.ATA * grid_fac ** 2  # grid_fac from each A matrix
        self.ATA_scale = np.linalg.norm(self.ATA, ord=2)

    def _plot_final_dipoles(self):
        """
            Makes cross-sections of the dipole grid and
            quiver plots of the sparse and nonsparse 
            dipole vectors after optimization.
        """
        for kk, dipoles in enumerate([self.m, self.m_proxy]):
            dipole_grid = self.dipole_grid
            plt.figure()
            ax = plt.axes(projection="3d")
            colors = []
            dipoles = dipoles.reshape(self.ndipoles, 3)
            for i in range(self.ndipoles):
                colors.append(np.sqrt(dipoles[i, 0] ** 2 + dipoles[i, 1] ** 2 + dipoles[i, 2] ** 2))
            sax = ax.scatter(dipole_grid[:, 0], dipole_grid[:, 1], dipole_grid[:, 2], c=colors)
            plt.colorbar(sax)
            plt.axis('off')
            plt.grid(None)

            plt.figure(figsize=(14, 14))
            for i, ind in enumerate([1, 5, 12, 15]):
                plt.subplot(2, 2, i + 1)
                plt.title(r'$\phi = ${0:.2f}$^o$'.format(360 * self.phi[ind]))
                r_plasma = np.hstack((self.r_plasma[ind, :], self.r_plasma[ind, 0]))
                z_plasma = np.hstack((self.z_plasma[ind, :], self.z_plasma[ind, 0]))
                r_inner = np.hstack((self.r_inner[ind, :], self.r_inner[ind, 0]))
                z_inner = np.hstack((self.z_inner[ind, :], self.z_inner[ind, 0]))
                r_outer = np.hstack((self.r_outer[ind, :], self.r_outer[ind, 0]))
                z_outer = np.hstack((self.z_outer[ind, :], self.z_outer[ind, 0]))

                plt.plot(r_plasma, z_plasma, label='Plasma surface', linewidth=2)
                plt.plot(r_inner, z_inner, label='Inner surface', linewidth=2)
                plt.plot(r_outer, z_outer, label='Outer surface', linewidth=2)

                running_tally = 0
                for k in range(ind):
                    if k > 0:
                        running_tally += len(self.final_RZ_grid[self.inds[k-1]:self.inds[k], k, 0])
                    else:
                        running_tally += len(self.final_RZ_grid[:self.inds[k], k, 0])
                colors = []
                dipoles_i = dipoles[running_tally:running_tally + len(self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 0]), :]
                for j in range(len(dipoles_i)):
                    colors.append(np.sqrt(dipoles_i[j, 0] ** 2 + dipoles_i[j, 1] ** 2 + dipoles_i[j, 2] ** 2))

                sax = plt.scatter(
                    self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 0],
                    self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 2],
                    c=colors,
                    label='PMs'
                )
                plt.colorbar(sax)
                plt.quiver(
                    self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 0],
                    self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 2],
                    dipoles_i[:, 0],
                    dipoles_i[:, 2],
                )
                plt.xlabel('R (m)')
                plt.ylabel('Z (m)')
                if i == 0:
                    plt.legend()
                plt.grid(True)
            if kk == 0:
                plt.savefig('grids_permanent_magnets.png')
            else:
                plt.savefig('grids_permanent_magnets_sparse.png')

    def _set_inner_rz_surface(self):
        """
            If the inner toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the 
            plasma boundary and shifts it by self.plasma_offset at constant
            theta value. 
        """
        if self.FOCUS:
            rz_inner_surface = SurfaceRZFourier.from_focus(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
        else:
            #rz_inner_surface = SurfaceRZFourier.from_vmec_input(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
            # make copy of plasma boundary
            mpol = self.plasma_boundary.mpol
            ntor = self.plasma_boundary.ntor
            nfp = self.plasma_boundary.nfp
            stellsym = self.plasma_boundary.stellsym
            range_surf = self.plasma_boundary.range 
            rz_inner_surface = SurfaceRZFourier(
                mpol=mpol, ntor=ntor, nfp=nfp,
                stellsym=stellsym, range=range_surf,
                quadpoints_theta=self.theta, 
                quadpoints_phi=self.phi
            )
            for i in range(mpol + 1):
                for j in range(-ntor, ntor + 1):
                    rz_inner_surface.set_rc(i, j, self.plasma_boundary.get_rc(i, j))
                    rz_inner_surface.set_rs(i, j, self.plasma_boundary.get_rs(i, j))
                    rz_inner_surface.set_zc(i, j, self.plasma_boundary.get_zc(i, j))
                    rz_inner_surface.set_zs(i, j, self.plasma_boundary.get_zs(i, j))

        # extend via the normal vector
        rz_inner_surface.extend_via_projected_normal(self.phi, self.plasma_offset)
        self.rz_inner_surface = rz_inner_surface

    def _set_outer_rz_surface(self):
        """
            If the outer toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the 
            inner toroidal surface and shifts it by self.coil_offset at constant
            theta value. 
        """
        if self.FOCUS:
            rz_outer_surface = SurfaceRZFourier.from_focus(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
        else:
            #rz_outer_surface = SurfaceRZFourier.from_vmec_input(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
            # make copy of inner boundary
            mpol = self.rz_inner_surface.mpol
            ntor = self.rz_inner_surface.ntor
            nfp = self.rz_inner_surface.nfp
            stellsym = self.rz_inner_surface.stellsym
            range_surf = self.rz_inner_surface.range 
            quadpoints_theta = self.rz_inner_surface.quadpoints_theta 
            quadpoints_phi = self.rz_inner_surface.quadpoints_phi 
            rz_outer_surface = SurfaceRZFourier(
                mpol=mpol, ntor=ntor, nfp=nfp,
                stellsym=stellsym, range=range_surf,
                quadpoints_theta=quadpoints_theta, 
                quadpoints_phi=quadpoints_phi
            ) 
            for i in range(mpol + 1):
                for j in range(-ntor, ntor + 1):
                    rz_outer_surface.set_rc(i, j, self.rz_inner_surface.get_rc(i, j))
                    rz_outer_surface.set_rs(i, j, self.rz_inner_surface.get_rs(i, j))
                    rz_outer_surface.set_zc(i, j, self.rz_inner_surface.get_zc(i, j))
                    rz_outer_surface.set_zs(i, j, self.rz_inner_surface.get_zs(i, j))

        # extend via the normal vector
        t1 = time.time()
        rz_outer_surface.extend_via_projected_normal(self.phi, self.coil_offset)
        t2 = time.time()
        print("Outer surface extension took t = ", t2 - t1)
        self.rz_outer_surface = rz_outer_surface

    def _plot_surfaces(self):
        """
            Simple plotting function for debugging the permanent
            magnet gridding procedure.
        """
        plt.figure(figsize=(14, 14))
        for i, ind in enumerate([0, 5, 12, 15]):
            plt.subplot(2, 2, i + 1)
            plt.title(r'$\phi = ${0:.2f}$^o$'.format(360 * self.phi[ind]))
            r_plasma = np.hstack((self.r_plasma[ind, :], self.r_plasma[ind, 0]))
            z_plasma = np.hstack((self.z_plasma[ind, :], self.z_plasma[ind, 0]))
            r_inner = np.hstack((self.r_inner[ind, :], self.r_inner[ind, 0]))
            z_inner = np.hstack((self.z_inner[ind, :], self.z_inner[ind, 0]))
            r_outer = np.hstack((self.r_outer[ind, :], self.r_outer[ind, 0]))
            z_outer = np.hstack((self.z_outer[ind, :], self.z_outer[ind, 0]))

            plt.plot(r_plasma, z_plasma, label='Plasma surface', linewidth=2)
            plt.plot(r_inner, z_inner, label='Inner surface', linewidth=2)
            plt.plot(r_outer, z_outer, label='Outer surface', linewidth=2)

            plt.scatter(
                self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 0], 
                self.final_RZ_grid[self.inds[ind-1]:self.inds[ind], ind, 2], 
                label='Final grid',
                c='k'
            )
            plt.scatter(np.ravel(self.RPhiZ[:, i, :, 0]), np.ravel(self.RPhiZ[:, i, :, 2]), c='k')
            if i == 0:
                plt.legend()
            plt.grid(True)
        plt.savefig('grids_permanent_magnets.png')

    def _prox_l0(self, m, reg_l0, nu):
        """Proximal operator for L0 regularization."""
        return m * (np.abs(m) > np.sqrt(2 * reg_l0 * nu))

    def _prox_l1(self, m, reg_l1, nu):
        """Proximal operator for L1 regularization."""
        return np.sign(m) * np.maximum(np.abs(m) - reg_l1 * nu, 0)

    def _projection_L2_balls(self, x, m_maxima):
        """
            Project the vector x onto a series of L2 balls in R3.
            Only used here for initializing a guess for the 
            permanent magnets before optimization begins.
        """
        N = len(x) // 3
        x_shaped = x.reshape(N, 3)
        denom = np.maximum(
            1,
            np.sqrt(x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2) / m_maxima 
        )
        return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)

    def _setup_initial_condition(self, m0):
        """
            If an initial guess for the dipole moments is specified,
            checks the initial condition lies in the allowed hypersurface.
            If an initial guess is not specified, defaults to initializing
            the permanent magnet dipole moments to the projection of the
            least-squares solution on the hypersurface, proj(pinv(A) * m).
        """
        # Initialize initial guess for the dipole strengths
        if m0 is not None:
            if len(m0) != self.ndipoles * 3:
                raise ValueError(
                    'Initial dipole guess is incorrect shape --'
                    ' guess must be 1D with shape (ndipoles * 3).'
                )
            m0_temp = self._projection_L2_balls(
                m0, 
                self.m_maxima
            )
            # check if m0 lies inside the hypersurface spanned by 
            # L2 balls, which we require it does
            if not np.allclose(m0, m0_temp):
                raise ValueError(
                    'Initial dipole guess must contain values '
                    'that are satisfy the maximum bound constraints.'
                )
        else:
            # Default initialized to proj(pinv(A) * b)
            m0 = self._projection_L2_balls(
                np.linalg.pinv(self.A_obj) @ self.b_obj, 
                self.m_maxima
            )
            #m0 = np.zeros(m0.shape)

        self.m0 = m0

    def _print_initial_opt(self):
        """
            Print out initial errors and the bulk optimization parameters 
            before the permanent magnets are optimized.
        """
        grid_fac = np.sqrt(self.dphi * self.dtheta)
        ave_Bn = np.mean(np.abs(self.b_obj)) * grid_fac
        Bmag = np.linalg.norm(self.B_plasma_surface, axis=-1, ord=2).reshape(self.nphi * self.ntheta)
        ave_BnB = np.mean(np.abs(self.b_obj) / Bmag) * grid_fac
        total_Bn = np.sum(np.abs(self.b_obj) ** 2)
        dipole_error = np.linalg.norm(self.A_obj.dot(self.m0), ord=2) ** 2
        total_error = np.linalg.norm(self.A_obj.dot(self.m0) - self.b_obj, ord=2) ** 2
        print('Number of phi quadrature points on plasma surface = ', self.nphi)
        print('Number of theta quadrature points on plasma surface = ', self.ntheta)
        print('<B * n> without the permanent magnets = {0:.4e}'.format(ave_Bn)) 
        print('<B * n / |B| > without the permanent magnets = {0:.4e}'.format(ave_BnB)) 
        print(r'$|b|_2^2 = |B * n|_2^2$ without the permanent magnets = {0:.4e}'.format(total_Bn))
        print(r'Initial $|Am_0|_2^2 = |B_M * n|_2^2$ without the coils/plasma = {0:.4e}'.format(dipole_error))
        print('Number of dipoles = ', self.ndipoles)
        print('Inner toroidal surface offset from plasma surface = ', self.plasma_offset)
        print('Outer toroidal surface offset from inner toroidal surface = ', self.coil_offset)
        print('Maximum dipole moment = ', np.max(self.m_maxima))
        print('Shape of A matrix = ', self.A_obj.shape)
        print('Shape of b vector = ', self.b_obj.shape)
        print('Initial error on plasma surface = {0:.4e}'.format(total_error))

    def _rescale_for_opt(self, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu):
        """ 
            Scale regularizers to the largest scale of ATA (~1e-6)
            to avoid regularization >> ||Am - b|| ** 2 term in the optimization.
            The prox operator uses reg_l0 * nu for the threshold so normalization 
            below allows reg_l0 and reg_l1 values to be exactly the thresholds 
            used in calculation of the prox. Then add contributions to ATA and
            ATb coming from extra loss terms such as L2 regularization and 
            relax-and-split. 
        """

        print('L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2))
        print('Shifted L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2_shifted))

        if reg_l0 < 0 or reg_l0 > 1:
            raise ValueError(
                'L0 regularization must be between 0 and 1. This '
                'value is automatically scaled to the largest of the '
                'dipole maximum values, so reg_l0 = 1 should basically '
                'truncate all the dipoles to zero. '
            )
        reg_l0 = reg_l0 * self.ATA_scale * np.max(self.m_maxima) ** 2 / (2 * nu)
        reg_l1 = reg_l1 * self.ATA_scale / nu
        nu = nu / self.ATA_scale

        reg_l2 = reg_l2  # * self.ATA_scale
        reg_l2_shifted = reg_l2_shifted  # * self.ATA_scale
        ATA = self.ATA + 2 * (reg_l2 + reg_l2_shifted) * np.eye(self.ATA.shape[0])

        # if using relax and split, add that contribution to ATA
        if reg_l0 > 0.0 or reg_l1 > 0.0: 
            ATA += np.eye(ATA.shape[0]) / nu

        # Add shifted L2 contribution to ATb
        ATb = self.ATb + reg_l2_shifted * np.ravel(np.outer(self.m_maxima, np.ones(3)))

        return ATA, ATb, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu

    def _optimize(self, m0=None, epsilon=1e-4, nu=1e100,
                  reg_l0=0, reg_l1=0, reg_l2=0, reg_l2_shifted=0, 
                  max_iter_MwPGP=50, max_iter_RS=4, verbose=True,
                  geometric_threshold=1e-50, 
                  ): 
        """ 
            Perform the permanent magnet optimization problem, 
            phrased as a relax-and-split formulation that 
            solves the convex and nonconvex parts separately. 
            This allows for speedy algorithms for both parts, 
            the imposition of convex equality and inequality
            constraints (including the required constraint on
            the strengths of the dipole moments). 

            Args:
                m0: Initial guess for the permanent magnet
                    dipole moments. Defaults to a 
                    starting guess that is proj(pinv(A) * b).
                    This vector must lie in the hypersurface
                    spanned by the L2 ball constraints. Note 
                    that if algorithm is being used properly,
                    the end result should be independent of 
                    the choice of initial condition.
                epsilon: Error tolerance for the convex 
                    part of the algorithm (MwPGP).
                nu: Hyperparameter used for the relax-and-split
                    least-squares. Set nu >> 1 to reduce the
                    importance of nonconvexity in the problem.
                reg_l0: Regularization value for the L0
                    nonconvex term in the optimization. This value 
                    is automatically scaled based on the max dipole
                    moment values, so that reg_l0 = 1 corresponds 
                    to reg_l0 = np.max(m_maxima). It follows that
                    users should choose reg_l0 in [0, 1]. 
                reg_l1: Regularization value for the L1
                    nonsmooth term in the optimization,
                reg_l2: Regularization value for any convex
                    regularizers in the optimization problem,
                    such as the often-used L2 norm.
                reg_l2_shifted: Regularization value for the L2
                    smooth and convex term in the optimization, 
                    shifted by the vector of maximum dipole magnitudes. 
                max_iter_MwPGP: Maximum iterations to perform during
                    a run of the convex part of the relax-and-split
                    algorithm (MwPGP). 
                max_iter_RS: Maximum iterations to perform of the 
                    overall relax-and-split algorithm. Therefore, 
                    also the number of times that MwPGP is called,
                    and the number of times a prox is computed.
                verbose: Prints out all the loss term errors separately.
                geometric_threshold: Threshold value for A.T * A matrix
                    appearing in the optimization. Any elements in |A.T * A|
                    below this value are truncated off. 
        """

        # Check initial condition is valid
        self._setup_initial_condition(m0)        

        # Rescale the hyperparameters and then add contributions to ATA and ATb
        ATA, ATb, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu = self._rescale_for_opt(
            reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu
        )

        # get optimal alpha value for the MwPGP algorithm
        alpha_max = 2.0 / np.linalg.norm(ATA, ord=2)
        alpha_max = alpha_max * (1 - 1e-5)

        # print initial errors and values before optimization
        self._print_initial_opt()

        # Auxiliary variable in relax-and-split is default
        # initialized to proj(pinv(A) * b)
        m_proxy = self._projection_L2_balls(
            np.linalg.pinv(self.A_obj) @ self.b_obj, 
            self.m_maxima
        )

        # convert to contiguous arrays for c++ code to work right
        ATA = np.ascontiguousarray(np.reshape(ATA, (self.ndipoles, 3, self.ndipoles, 3)))
        ATb = np.ascontiguousarray(np.reshape(ATb, (self.ndipoles, 3)))

        # Begin optimization
        err_RS = []
        if reg_l0 > 0.0 or reg_l1 > 0.0: 
            # Relax-and-split algorithm
            if reg_l0 > 0.0:
                prox = self._prox_l0
            elif reg_l1 > 0.0:
                prox = self._prox_l1
            m = self.m0
            for i in range(max_iter_RS):
                # update m with the CONVEX part of the algorithm
                MwPGP_hist, _, m_hist, m = sopp.MwPGP_algorithm(
                    A_obj=self.A_obj_expanded,
                    b_obj=self.b_obj,
                    ATA=ATA,
                    ATb=ATb,
                    m_proxy=np.ascontiguousarray(m_proxy.reshape(self.ndipoles, 3)),
                    m0=np.ascontiguousarray(m.reshape(self.ndipoles, 3)),
                    m_maxima=self.m_maxima,
                    alpha=alpha_max,
                    epsilon=epsilon,
                    max_iter=max_iter_MwPGP,
                    verbose=True,
                    reg_l0=reg_l0,
                    reg_l1=reg_l1,
                    reg_l2=reg_l2,
                    reg_l2_shifted=reg_l2_shifted,
                    nu=nu,
                )    
                m = np.ravel(m)
                err_RS.append(MwPGP_hist[-1])
                m_proxy = prox(m, reg_l0, nu)
                if np.linalg.norm(m - m_proxy) < epsilon:
                    print('Relax-and-split finished early, at iteration ', i)
        else:
            m0 = np.ascontiguousarray(self.m0.reshape(self.ndipoles, 3))
            # no nonconvex terms being used, so just need one round of the
            # convex algorithm called MwPGP
            MwPGP_hist, _, m_hist, m = sopp.MwPGP_algorithm(
                A_obj=self.A_obj_expanded,
                b_obj=self.b_obj,
                ATA=ATA,
                ATb=ATb,
                m_proxy=m0,
                m0=m0,
                m_maxima=self.m_maxima,
                alpha=alpha_max,
                epsilon=epsilon,
                max_iter=max_iter_MwPGP,
                verbose=True,
                reg_l0=reg_l0,
                reg_l1=reg_l1,
                reg_l2=reg_l2,
                reg_l2_shifted=reg_l2_shifted,
                nu=nu
            )    
            m = np.ravel(m)
            m_proxy = m

        # Compute metrics with permanent magnet results
        grid_fac = np.sqrt(self.dphi * self.dtheta)
        ave_Bn_proxy = np.mean(np.abs(self.A_obj.dot(m_proxy) - self.b_obj)) * grid_fac
        ave_Bn = np.mean(np.abs(self.A_obj.dot(m) - self.b_obj)) * grid_fac
        print(np.max(self.m_maxima), np.max(m_proxy))
        print('<B * n> with the optimized permanent magnets = {0:.8e}'.format(ave_Bn)) 
        print('<B * n> with the sparsified permanent magnets = {0:.8e}'.format(ave_Bn_proxy)) 

        # note m = m_proxy if not using relax-and-split 
        self.m = m
        self.m_proxy = m_proxy
        return MwPGP_hist, err_RS, m_hist, m_proxy
