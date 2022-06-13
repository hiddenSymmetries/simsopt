import logging
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
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
            is_premade_famus_grid:  Flag to use the pre-made dipole grid from the
                              half-Tesla NCSX configuration (or MUSE) to match with
                              the equivalent FAMUS run. 
                              If specified, rz_inner_surface, rz_outer_surface,
                              dr, plasma_offset, and coil_offset are all
                              ignored.
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
            Bn: Magnetic field (coils and plasma) at the plasma
                              boundary. Typically this will be the optimized plasma
                              magnetic field from a stage-1 optimization, and the
                              optimized coils from a basic stage-2 optimization. 
                              This variable must be specified to run the permanent
                              magnet optimization.
            dr:               Radial and axial grid spacing in the permanent magnet manifold.
            filename:         Filename for the file containing the plasma boundary surface.
            surface_flag:     Flag to specify if the format of the surface file. Defaults to VMEC. 
    """

    def __init__(
        self, plasma_boundary, is_premade_famus_grid=False,
        rz_inner_surface=None, 
        rz_outer_surface=None, plasma_offset=0.1, 
        coil_offset=0.2, Bn=None, dr=0.1,
        filename=None, surface_flag='vmec', out_dir='',
        cylindrical_flag=False, test_flag=False,
        pms_name=None,
    ):
        if plasma_offset <= 0 or coil_offset <= 0:
            raise ValueError('permanent magnets must be offset from the plasma')

        self.is_premade_famus_grid = is_premade_famus_grid
        if is_premade_famus_grid and pms_name is None:
            raise ValueError(
                'If using a premade FAMUS grid, you must specify the '
                'filename of the grid for loading via the pms_name parameter. '
            )
        self.pms_name = pms_name
        self.filename = filename
        self.surface_flag = surface_flag 
        self.out_dir = out_dir
        self.plasma_offset = plasma_offset
        self.coil_offset = coil_offset
        self.Bn = Bn
        self.dr = dr
        self.cylindrical_flag = cylindrical_flag
        self.test_flag = test_flag

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
        if rz_inner_surface is None and not is_premade_famus_grid:
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
        if not is_premade_famus_grid:
            if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
                raise ValueError("Inner surface is not SurfaceRZFourier object.")

        # If the outer surface is not specified, make default surface. 
        if rz_outer_surface is None and not is_premade_famus_grid:
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
        if not is_premade_famus_grid:
            if not isinstance(self.rz_outer_surface, SurfaceRZFourier):
                raise ValueError("Outer surface is not SurfaceRZFourier object.")
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to get the SurfaceRZFourier objects done") 

        if not is_premade_famus_grid:
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

            # Issues with the MUSE configuration when using self.normal_inner and outer. 
            # The issue is that the normal vectors on the inner and outer toroidal surfaces
            # look very different on MUSE than on the original plasma surface. To avoid this issue,
            # just use the nice plasma boundary normal vectors for the ray-tracing. This generates
            # small errors when resolution is low but works well when nphi, ntheta >~ 16 or so.

            if self.surface_flag == 'focus' or self.surface_flag == 'wout':
                self.final_RZ_grid, self.inds = sopp.make_final_surface(
                    2 * np.pi * self.phi, self.plasma_boundary.unitnormal(), self.plasma_boundary.unitnormal(),
                    # self.normal_inner, self.normal_outer, 
                    RPhiZ_grid, self.r_inner, self.r_outer, self.z_inner, self.z_outer
                )
            else:
                self.final_RZ_grid, self.inds = sopp.make_final_surface(
                    2 * np.pi * self.phi, self.normal_inner, self.normal_outer, 
                    RPhiZ_grid, self.r_inner, self.r_outer, self.z_inner, self.z_outer
                )
            self.inds = np.array(self.inds, dtype=int)
            for i in reversed(range(1, len(self.inds))):
                for j in range(0, i):
                    self.inds[i] += self.inds[j]
            print(self.inds)
            self.ndipoles = self.inds[-1]
            final_grid = []
            for i in range(self.final_RZ_grid.shape[0]):
                if not np.allclose(self.final_RZ_grid[i, :], 0.0):
                    final_grid.append(self.final_RZ_grid[i, :])
            self.final_RZ_grid = np.array(final_grid)
            print(self.final_RZ_grid.shape)
            t2 = time.time()
            print("Took t = ", t2 - t1, " s to perform the C++ grid cell eliminations.")
        else:
            ox, oy, oz, Ic = np.loadtxt('../../tests/test_files/' + self.pms_name, skiprows=3, usecols=[3, 4, 5, 6], delimiter=',', unpack=True)

            nonzero_inds = (Ic == 1.0)
            ox = ox[nonzero_inds]
            oy = oy[nonzero_inds]
            oz = oz[nonzero_inds]
            self.Ic_inds = nonzero_inds
            premade_dipole_grid = np.array([ox, oy, oz]).T
            self.premade_dipole_grid = premade_dipole_grid
            self.ndipoles = premade_dipole_grid.shape[0]

            # Not normalized to 1 like quadpoints_phi!
            self.pm_phi = np.arctan2(premade_dipole_grid[:, 1], premade_dipole_grid[:, 0])
            uniq_phi, counts_phi = np.unique(self.pm_phi.round(decimals=6), return_counts=True)
            self.pm_nphi = len(uniq_phi)
            self.pm_uniq_phi = uniq_phi
            self.inds = counts_phi
            for i in reversed(range(1, self.pm_nphi)):
                for j in range(0, i):
                    self.inds[i] += self.inds[j] 
            print(self.inds)
            self.final_RZ_grid = np.zeros((self.ndipoles, 3))
            self.final_RZ_grid[:, 0] = np.sqrt(premade_dipole_grid[:, 0] ** 2 + premade_dipole_grid[:, 1] ** 2)
            self.final_RZ_grid[:, 1] = self.pm_phi
            self.final_RZ_grid[:, 2] = premade_dipole_grid[:, 2] 
            self.dipole_grid_xyz = premade_dipole_grid

        xyz_plasma = self.plasma_boundary.gamma()
        self.r_plasma = np.sqrt(xyz_plasma[:, :, 0] ** 2 + xyz_plasma[:, :, 1] ** 2)
        self.z_plasma = xyz_plasma[:, :, 2]

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

        # Set initial condition for the dipoles to default IC
        self._setup_initial_condition()        

        # Print initial f_B metric using the initial guess
        total_error = np.linalg.norm((self.A_obj.dot(self.m0) - self.b_obj), ord=2) ** 2 / 2.0
        print('f_B (total with initial SIMSOPT guess) = ', total_error)

    def _setup_uniform_grid(self):
        """ 
            Initializes a uniform grid in cylindrical coordinates and sets
            some important grid variables for later.
        """
        # Get (R, Z) coordinates of the three boundaries
        xyz_inner = self.rz_inner_surface.gamma()
        self.r_inner = np.sqrt(xyz_inner[:, :, 0] ** 2 + xyz_inner[:, :, 1] ** 2)
        self.z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.rz_outer_surface.gamma()
        self.r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
        self.z_outer = xyz_outer[:, :, 2]

        self.normal_inner = np.copy(self.rz_inner_surface.unitnormal())
        self.normal_outer = np.copy(self.rz_outer_surface.unitnormal())

        # optional code below to debug if the normal vectors are behaving
        # which seems to be an issue with the MUSE inner/outer PM surfaces
        if False:
            ax = plt.figure().add_subplot(projection='3d')
            u = np.ravel(self.plasma_boundary.unitnormal()[:, :, 0]) 
            v = np.ravel(self.plasma_boundary.unitnormal()[:, :, 1])
            w = np.ravel(self.plasma_boundary.unitnormal()[:, :, 2])
            x = np.ravel(xyz_plasma[:, :, 0]) 
            y = np.ravel(xyz_plasma[:, :, 1]) 
            z = np.ravel(xyz_plasma[:, :, 2]) 
            ax.scatter(x, y, z, color='k')
            ax.quiver(x, y, z, u, v, w, length=0.025, normalize=True)

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
        phi = 2 * np.pi * np.copy(self.plasma_boundary.quadpoints_phi)
        print('dR = {0:.2f}'.format(self.Delta_r))
        print('dZ = {0:.2f}'.format(self.Delta_z))
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
        B_max = 1.465
        mu0 = 4 * np.pi * 1e-7
        dipole_grid_r = np.zeros(self.ndipoles)
        dipole_grid_x = np.zeros(self.ndipoles)
        dipole_grid_y = np.zeros(self.ndipoles)
        dipole_grid_phi = np.zeros(self.ndipoles)
        dipole_grid_z = np.zeros(self.ndipoles)
        running_tally = 0
        if self.is_premade_famus_grid:
            nphi = self.pm_nphi
        else:
            nphi = self.nphi
        for i in range(nphi):
            if i > 0:
                radii = self.final_RZ_grid[self.inds[i-1]:self.inds[i], 0]
                phi = self.final_RZ_grid[self.inds[i-1]:self.inds[i], 1]
                z_coords = self.final_RZ_grid[self.inds[i-1]:self.inds[i], 2]
            else:
                radii = self.final_RZ_grid[:self.inds[i], 0]
                phi = self.final_RZ_grid[:self.inds[i], 1]
                z_coords = self.final_RZ_grid[:self.inds[i], 2]
            len_radii = len(radii)
            dipole_grid_r[running_tally:running_tally + len_radii] = radii
            dipole_grid_phi[running_tally:running_tally + len_radii] = phi
            dipole_grid_x[running_tally:running_tally + len_radii] = radii * np.cos(phi)
            dipole_grid_y[running_tally:running_tally + len_radii] = radii * np.sin(phi)
            dipole_grid_z[running_tally:running_tally + len_radii] = z_coords
            running_tally += len_radii
        self.dipole_grid = np.array([dipole_grid_r, dipole_grid_phi, dipole_grid_z]).T

        if self.is_premade_famus_grid:
            M0s = np.loadtxt(
                '../../tests/test_files/' + self.pms_name, 
                skiprows=3, usecols=[7], delimiter=','
            )
            cell_vol = M0s * mu0 / B_max
            self.m_maxima = B_max * cell_vol[self.Ic_inds] / mu0
        else:
            cell_vol = dipole_grid_r * self.Delta_r * self.Delta_z * 2 * np.pi / (self.nphi * self.plasma_boundary.nfp * 2)
            #else:
            #    cell_vol = np.sqrt((dipole_grid_r - self.plasma_boundary.get_rc(0, 0)) ** 2 + dipole_grid_z ** 2) * self.Delta_r * self.Delta_z * 2 * np.pi / (self.nphi * self.plasma_boundary.nfp * 2)
            self.dipole_grid_xyz = np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).T
            self.m_maxima = B_max * cell_vol / mu0
        print('Total initial volume for magnet placement = ', np.sum(cell_vol) * self.plasma_boundary.nfp * 2, ' m^3')

        # FAMUS paper has a typo that m_max = B_r / (mu0 * cell_vol) but it 
        # should be m_max = B_r * cell_vol / mu0  (just from units)

    def _geo_setup(self):
        """
            Initialize 'b' vector in 0.5 * ||Am - b||^2 part of the optimization,
            corresponding to the normal component of the target fields. Note
            the factor of two in the least-squares term: 0.5 * m.T @ (A.T @ A) @ m - b.T @ m.
            Also initialize the A, A.T @ b, and A.T @ A matrices that are important in the
            optimization problem.
        """
        # set up 'b' vector 
        if self.Bn.shape != (self.nphi, self.ntheta):
            raise ValueError(
                'Normal magnetic field surface data is incorrect shape.'
            )
        dphi = (self.phi[1] - self.phi[0]) * 2 * np.pi
        dtheta = (self.theta[1] - self.theta[0]) * 2 * np.pi
        self.dphi = dphi
        self.dtheta = dtheta

        # minus sign below because ||Ax - b||^2 term but original
        # term is integral(B_P + B_C + B_M)^2
        self.b_obj = - self.Bn.reshape(self.nphi * self.ntheta)

        # Compute geometric factor with the C++ routine
        self.A_obj, self.ATb = sopp.dipole_field_Bn(
            np.ascontiguousarray(self.plasma_boundary.gamma().reshape(self.nphi * self.ntheta, 3)), 
            np.ascontiguousarray(self.dipole_grid_xyz), 
            np.ascontiguousarray(self.plasma_boundary.unitnormal().reshape(self.nphi * self.ntheta, 3)), 
            self.plasma_boundary.nfp, int(self.plasma_boundary.stellsym), 
            np.ascontiguousarray(self.dipole_grid[:, 1]), 
            np.ascontiguousarray(self.b_obj),
            self.cylindrical_flag  # If False use Cartesian coords. If True, cylindrical coords
        )
        # Rescale
        Ngrid = self.nphi * self.ntheta
        self.A_obj = self.A_obj.reshape(self.nphi * self.ntheta, self.ndipoles * 3)
        self.ATb = np.ravel(self.ATb)
        Nnorms = np.ravel(np.sqrt(np.sum(self.plasma_boundary.normal() ** 2, axis=-1)))
        for i in range(self.A_obj.shape[0]):
            self.A_obj[i, :] = self.A_obj[i, :] * np.sqrt(Nnorms[i] / Ngrid) 
        self.b_obj = self.b_obj * np.sqrt(Nnorms / Ngrid)
        self.ATb = self.A_obj.T @ self.b_obj

    def _set_inner_rz_surface(self):
        """
            If the inner toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the 
            plasma boundary and shifts it by self.plasma_offset at constant
            theta value. 
        """
        if self.surface_flag == 'focus':
            rz_inner_surface = SurfaceRZFourier.from_focus(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
        elif self.surface_flag == 'wout':
            rz_inner_surface = SurfaceRZFourier.from_wout(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
        else:
            rz_inner_surface = SurfaceRZFourier.from_vmec_input(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)

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
        if self.surface_flag == 'focus':
            rz_outer_surface = SurfaceRZFourier.from_focus(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
        elif self.surface_flag == 'wout':
            rz_outer_surface = SurfaceRZFourier.from_wout(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)
        else:
            rz_outer_surface = SurfaceRZFourier.from_vmec_input(self.filename, range=self.plasma_boundary.range, nphi=self.nphi, ntheta=self.ntheta)

        # extend via the normal vector
        t1 = time.time()
        rz_outer_surface.extend_via_projected_normal(self.phi, self.plasma_offset + self.coil_offset)
        t2 = time.time()
        print("Outer surface extension took t = ", t2 - t1)
        self.rz_outer_surface = rz_outer_surface

    def _prox_l0(self, m, reg_l0, nu):
        """
        Proximal operator for L0 regularization. 
        Note that the m values are normalized before hard
        thresholding to avoid only truncating the magnets on the 
        inner side of the configuration (which are often important!)
        """
        m_normalized = (np.abs(m).reshape(self.ndipoles, 3) / np.array([self.m_maxima, self.m_maxima, self.m_maxima]).T).reshape(self.ndipoles * 3)
        return m * (m_normalized > 2 * reg_l0 * nu)
        #return m * (m_normalized > np.sqrt(2 * reg_l0 * nu))

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
        denom_fac = np.sqrt(x_shaped[:, 0] ** 2 + x_shaped[:, 1] ** 2 + x_shaped[:, 2] ** 2) / m_maxima 
        denom = np.maximum(
            np.ones(len(denom_fac)),
            denom_fac, 
        )
        return np.divide(x_shaped, np.array([denom, denom, denom]).T).reshape(3 * N)

    def _setup_initial_condition(self, m0=None):
        """
            If an initial guess for the dipole moments is specified,
            checks the initial condition lies in the allowed hypersurface.
            If an initial guess is not specified, defaults to initializing
            the permanent magnet dipole moments to all zeros.
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
            self.m0 = m0
        elif not hasattr(self, 'm0'):
            # initialization to proj(pinv(A) * b) is usually a bad guess!
            #m0 = self._projection_L2_balls(
            #    np.linalg.pinv(self.A_obj) @ self.b_obj, 
            #    self.m_maxima
            #)
            m0 = np.zeros(self.ndipoles * 3)
            #m0 = np.random.rand(m0.shape)
            #m0 = np.ravel((np.random.rand(self.ndipoles, 3) - 0.5) * 2 * np.array([self.m_maxima, self.m_maxima, self.m_maxima]).T / np.sqrt(12))
            #m0 = np.ravel(np.array([self.m_maxima, self.m_maxima, self.m_maxima]).T) / np.sqrt(3)

            self.m0 = m0

    def _print_initial_opt(self):
        """
            Print out initial errors and the bulk optimization parameters 
            before the permanent magnets are optimized.
        """
        ave_Bn = np.mean(np.abs(self.b_obj))
        total_Bn = np.sum(np.abs(self.b_obj) ** 2)
        dipole_error = np.linalg.norm(self.A_obj.dot(self.m0), ord=2) ** 2
        total_error = np.linalg.norm(self.A_obj.dot(self.m0) - self.b_obj, ord=2) ** 2
        print('Number of phi quadrature points on plasma surface = ', self.nphi)
        print('Number of theta quadrature points on plasma surface = ', self.ntheta)
        print('<B * n> without the permanent magnets = {0:.4e}'.format(ave_Bn)) 
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
        # Compute singular values of A, use this to determine optimal step size
        # for the MwPGP algorithm, with alpha ~ 2 / ATA_scale
        S = np.linalg.svd(self.A_obj, full_matrices=False, compute_uv=False)
        self.ATA_scale = S[0] ** 2

        # Rescale L0, L1, and relax-and-split hyperparameters so that
        # these terms are normalized to the scale of the least-squares term.
        reg_l0 = reg_l0 / (2 * nu)
        #reg_l0 = reg_l0 * np.max(self.m_maxima) ** 2 / (2 * nu)
        #reg_l0 = reg_l0 * self.ATA_scale * np.max(self.m_maxima) ** 2 / (2 * nu)
        reg_l1 = reg_l1 * self.ATA_scale / nu
        #reg_l1 = reg_l1 * self.ATA_scale / nu
        #nu = nu / self.ATA_scale

        # Do not rescale L2 terms for now
        reg_l2 = reg_l2
        reg_l2_shifted = reg_l2_shifted

        # Update optimal algorithm step size if have extra loss terms
        self.ATA_scale = S[0] ** 2 + 2 * (reg_l2 + reg_l2_shifted) + 1.0 / nu
        #self.ATA_scale = S[0] ** 2 + 2 * (reg_l2 + reg_l2_shifted) + 1.0 / (nu * self.ATA_scale)

        # Add shifted L2 contribution to ATb
        ATb = self.ATb + reg_l2_shifted * np.ravel(np.outer(self.m_maxima, np.ones(3)))

        return ATb, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["curve"] = self.curve.as_dict()
        d["plasma_boundary"] = self.plasma_boundary
        d["rz_inner_surface"] = self.rz_inner_surface
        d["rz_outer_surface"] = self.rz_outer_surface
        d["plasma_offset"] = self.plasma_offset
        d["coil_offset"] = self.coil_offset
        d["Bn"] = self.Bn
        d["dr"] = self.dr
        d["filename"] = self.filename
        d["surface_flag"] = self.surface_flag
        d["out_dir"] = self.out_dir
        d["cylindrical_flag"] = self.cylindrical_flag
        d["test_flag"] = self.test_flag
        return d

    @classmethod
    def from_dict(cls, d):
        curve = MontyDecoder().process_decoded(d["curve"])
        return cls(
            d["plasma_boundary"],
            d["rz_inner_surface"],
            d["rz_outer_surface"],
            d["plasma_offset"],
            d["coil_offset"],
            d["Bn"],
            d["dr"],
            d["filename"],
            d["surface_flag"],
            d["out_dir"],
            d["cylindrical_flag"],
            d["test_flag"]
        )

    def _optimize(self, m0=None, epsilon=1e-4, nu=1e100,
                  reg_l0=0, reg_l1=0, reg_l2=0, reg_l2_shifted=0, 
                  max_iter_MwPGP=20, max_iter_RS=4, verbose=True,
                  min_fb=1.0e-20,
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
                    starting guess of all zeros.  
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
        """

        if (max_iter_MwPGP % 20) != 0:
            raise ValueError('Maximum iterations of MwPGP must be a multiple of 20 for now.')

        # reset m0 if desired
        self._setup_initial_condition(m0)        

        # Rescale the hyperparameters and then add contributions to ATA and ATb
        ATb, reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu = self._rescale_for_opt(
            reg_l0, reg_l1, reg_l2, reg_l2_shifted, nu
        )

        # get optimal alpha value for the MwPGP algorithm
        alpha_max = 2.0 / self.ATA_scale
        alpha_max = alpha_max * (1 - 1e-5)

        # print initial errors and values before optimization
        self._print_initial_opt()

        # Auxiliary variable in relax-and-split is initialized
        # to prox(m0), where m0 is the initial guess for m.
        m_proxy = self.m0
        m_proxy = self._prox_l0(m_proxy, reg_l0, nu)

        # change to row-major order for the C++ code
        self.A_obj = np.ascontiguousarray(self.A_obj)
        ATb = np.ascontiguousarray(np.reshape(ATb, (self.ndipoles, 3)))

        # Begin optimization
        err_RS = []
        m_history = []
        m_proxy_history = []
        if reg_l0 > 0.0 or reg_l1 > 0.0: 
            # Relax-and-split algorithm
            if reg_l0 > 0.0:
                prox = self._prox_l0
            elif reg_l1 > 0.0:
                prox = self._prox_l1
            m = self.m0
            for i in range(max_iter_RS):
                # update m with the CONVEX part of the algorithm
                MwPGP_hist, _, _, m = sopp.MwPGP_algorithm(
                    A_obj=self.A_obj,
                    b_obj=self.b_obj,
                    ATb=ATb,
                    m_proxy=np.ascontiguousarray(m_proxy.reshape(self.ndipoles, 3)),
                    m0=np.ascontiguousarray(m.reshape(self.ndipoles, 3)),  # note updated m is new guess
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
                    min_fb=min_fb,
                )  
                m_history.append(m)
                m = np.ravel(m)
                MwPGP_hist = MwPGP_hist[MwPGP_hist != 0]
                err_RS.append(MwPGP_hist[-1])

                # Solve the nonconvex optimization -- i.e. take a prox
                reg_l0_scaled = reg_l0  # * (1 + i / 100.0)  #### Scaling reg_l0 as algorithm progresses!
                print(i, reg_l0_scaled * 2 * nu)
                m_proxy = prox(m, reg_l0_scaled, nu)
                m_proxy_history.append(m_proxy)
                if np.linalg.norm(m - m_proxy) < epsilon:
                    print('Relax-and-split finished early, at iteration ', i)
        else:
            m0 = np.ascontiguousarray(self.m0.reshape(self.ndipoles, 3))
            # no nonconvex terms being used, so just need one round of the
            # convex algorithm called MwPGP
            MwPGP_hist, _, m_history, m = sopp.MwPGP_algorithm(
                A_obj=self.A_obj,
                b_obj=self.b_obj,
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
                nu=nu,
                min_fb=min_fb,
            )    
            m = np.ravel(m)
            m_proxy = m

        # note m = m_proxy if not using relax-and-split (i.e. problem is convex) 
        self.m = m
        self.m_proxy = m_proxy
        return err_RS, m_history, m_proxy_history
