import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import simsoptpp as sopp
import time
import warnings
from pyevtk.hl import pointsToVTK

__all__ = ['PermanentMagnetGrid']


class PermanentMagnetGrid:
    r"""
    ``PermanentMagnetGrid`` is a class for setting up the grid,
    plasma surface, and other objects needed to perform permanent
    magnet optimization for stellarators. The class
    takes as input two toroidal surfaces specified as SurfaceRZFourier
    objects, and initializes a set of points (in cylindrical coordinates)
    between these surfaces. If an existing FAMUS grid file called
    from FAMUS is desired, use from_famus() instead.
    It finishes initialization by pre-computing
    a number of quantities required for the optimization such as the
    geometric factor in the dipole part of the magnetic field and the
    target Bfield that is a sum of the coil and plasma magnetic fields.

    Args:
        plasma_boundary: SurfaceRZFourier class object 
            Representing the plasma boundary surface.
        rz_inner_surface: SurfaceRZFourier class object 
            Representing the inner toroidal surface of the volume.
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
        Bn: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
            Magnetic field (coils and plasma) at the plasma
            boundary. Typically this will be the optimized plasma
            magnetic field from a stage-1 optimization, and the
            optimized coils from a basic stage-2 optimization.
            This variable must be specified to run the permanent
            magnet optimization.
        dr: double
            Radial grid spacing in the permanent magnet manifold. Used only if 
            coordinate_flag = cylindrical, then dr is the radial size of the
            cylindrical bricks in the grid.
        dz: double
            Axial grid spacing in the permanent magnet manifold. Used only if
            coordinate_flag = cylindrical, then dz is the axial size of the
            cylindrical bricks in the grid.
        Nx: int
            Number of points in x to use in a cartesian grid, taken between the 
            inner and outer toroidal surfaces. Used only if the
            coordinate_flag = cartesian, then Nx is the x-size of the
            rectangular cubes in the grid.
        Ny: int
            Number of points in y to use in a cartesian grid, taken between the 
            inner and outer toroidal surfaces. Used only if the
            coordinate_flag = cartesian, then Ny is the y-size of the
            rectangular cubes in the grid.
        Nz: int
            Number of points in z to use in a cartesian grid, taken between the 
            inner and outer toroidal surfaces. Used only if the
            coordinate_flag = cartesian, then Nz is the z-size of the
            rectangular cubes in the grid.
        coordinate_flag: string
            Flag to specify the coordinate system used for the grid and optimization.
              This is primarily used to tell the optimizer which coordinate directions
              should be considered, "grid-aligned". Therefore, you can do weird stuff
              like cartesian grid-alignment on a simple toroidal grid, which might
              be self-defeating. However, if coordinate_flag='cartesian' a rectangular
              Cartesian grid is initialized using the Nx, Ny, and Nz parameters. If
              the coordinate_flag='cylindrical', a uniform cylindrical grid is initialized
              using the dr and dz parameters.
        pol_vectors: 3D numpy array, shape (Ndipoles, Ncoords, 3)
            Optional set of local coordinate systems for each dipole, 
            which specifies which directions should be considered grid-aligned.
            Ncoords can be > 3, as in the PM4Stell design. 
    """

    def __init__(
        self, plasma_boundary,
        rz_inner_surface,
        rz_outer_surface, Bn,
        dr=0.1, dz=0.1,
        Nx=10, Ny=10, Nz=10,
        coordinate_flag='cartesian',
        pol_vectors=None
    ):
        Bn = np.array(Bn)
        if len(Bn.shape) != 2: 
            raise ValueError(
                'Normal magnetic field surface data is incorrect shape.'
            )
        self.Bn = Bn
        self.dr = dr
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        self.Nx = Nx
        if dz is not None:
            self.dz = dz
        else:
            self.dz = dr
        if Ny is not None:
            self.Ny = Ny
        else:
            self.Ny = Nx
        if Nz is not None:
            self.Nz = Nz
        else:
            self.Nz = Nx

        if coordinate_flag not in ['cartesian', 'cylindrical', 'toroidal']:
            raise NotImplementedError(
                'Only cartesian, cylindrical, and toroidal (simple toroidal)'
                ' coordinate systems have been implemented so far.'
            )
        elif coordinate_flag == 'cartesian':
            warnings.warn(
                'Cartesian grid of rectangular cubes will be built, since '
                'coordinate_flag = "cartesian". However, such a grid can be '
                'appropriately reflected only for nfp = 2 and nfp = 4, '
                'unlike the uniform cylindrical grid, which has continuous '
                'rotational symmetry.'
            )
        self.coordinate_flag = coordinate_flag

        if not isinstance(plasma_boundary, SurfaceRZFourier):
            raise ValueError(
                'Plasma boundary must be specified as SurfaceRZFourier class.'
            )
        else:
            self.plasma_boundary = plasma_boundary
            self.R0 = plasma_boundary.get_rc(0, 0)
            self.phi = self.plasma_boundary.quadpoints_phi
            self.nphi = len(self.phi)
            self.theta = self.plasma_boundary.quadpoints_theta
            self.ntheta = len(self.theta)

        if dr <= 0:
            raise ValueError('dr grid spacing must be > 0')

        t1 = time.time()
        self.rz_inner_surface = rz_inner_surface
        if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
            raise ValueError("Inner surface is not SurfaceRZFourier object.")

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

        if pol_vectors is not None:
            pol_vectors = np.array(pol_vectors)
            if len(pol_vectors.shape) != 3:
                raise ValueError('pol vectors must be a 3D array.')
            elif pol_vectors.shape[2] != 3:
                raise ValueError('Third dimension of `pol_vectors` array '
                                 'must be 3')
            elif coordinate_flag != 'cartesian':
                raise ValueError('pol_vectors argument can only be used with coordinate_flag = cartesian currently')

        self.pol_vectors = pol_vectors
        self.famus_filename = None

    def _setup_uniform_grid(self):
        """
            Initializes a uniform grid in cartesian coordinates and sets
            some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the two boundaries
        xyz_inner = self.rz_inner_surface.gamma()
        self.xyz_inner = xyz_inner.reshape(-1, 3)
        self.x_inner = xyz_inner[:, :, 0]
        self.y_inner = xyz_inner[:, :, 1]
        self.z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.rz_outer_surface.gamma()
        self.xyz_outer = xyz_outer.reshape(-1, 3)
        self.x_outer = xyz_outer[:, :, 0]
        self.y_outer = xyz_outer[:, :, 1]
        self.z_outer = xyz_outer[:, :, 2]

        x_max = np.max(self.x_outer)
        x_min = np.min(self.x_outer)
        y_max = np.max(self.y_outer)
        y_min = np.min(self.y_outer)
        z_max = np.max(self.z_outer)
        z_min = np.min(self.z_outer)
        z_max = max(z_max, abs(z_min))

        # Make grid uniform in (X, Y)
        min_xy = min(x_min, y_min)
        max_xy = max(x_max, y_max)
        x_min = min_xy
        y_min = min_xy
        x_max = max_xy
        y_max = max_xy
        print(x_min, x_max, y_min, y_max, z_min, z_max)

        # Initialize uniform grid
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.dx = (x_max - x_min) / (Nx - 1)
        self.dy = (y_max - y_min) / (Ny - 1)
        self.dz = 2 * z_max / (Nz - 1)

        # Extra work below so that the stitching with the symmetries is done in
        # such a way that the reflected cells are still dx and dy away from
        # the old cells.
        #### Note that Cartesian cells can only do nfp = 2, 4, 6, ... 
        #### and correctly be rotated to have the right symmetries
        if (self.plasma_boundary.nfp % 2) == 0:
            X = np.linspace(self.dx / 2.0, (x_max - x_min) + self.dx / 2.0, Nx, endpoint=True)
            Y = np.linspace(self.dy / 2.0, (y_max - y_min) + self.dy / 2.0, Ny, endpoint=True)
        else:
            X = np.linspace(x_min, x_max, Nx, endpoint=True)
            Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        Z = np.linspace(-z_max, z_max, Nz, endpoint=True)
        print(Nx, Ny, Nz, X, Y, Z, self.dx, self.dy, self.dz)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)

        # Extra work for nfp = 4 to chop off half of the originally nfp = 2 uniform grid
        if self.plasma_boundary.nfp == 4:
            inds = []
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if X[i, j, k] < Y[i, j, k]:
                            inds.append(int(i * Ny * Nz + j * Nz + k))
            good_inds = np.setdiff1d(np.arange(Nx * Ny * Nz), inds)
            self.xyz_uniform = self.xyz_uniform[good_inds, :]
        contig = np.ascontiguousarray
        pointsToVTK(
            'uniform_grid',
            contig(self.xyz_uniform[:, 0]),
            contig(self.xyz_uniform[:, 1]),
            contig(self.xyz_uniform[:, 2])
        )

    def geo_setup_from_famus(self, famus_filename):
        """
        Function to initialize a SIMSOPT PermanentMagnetGrid from a 
        pre-existing FAMUS file defining a grid of magnets. For
        example, this function is used for the
        half-Tesla NCSX configuration (c09r000) and MUSE grids.
        If this flag is specified, rz_inner_surface,
        rz_outer_surface, dr
        are all ignored.


        Args
        ----------
        famus_filename : string
            Filename of a FAMUS grid file (a pre-made dipole grid).

        Returns
        -------
        None.

        """
        if famus_filename is not None:
            warnings.warn(
                'famus_filename variable is set, so a pre-defined grid will be used. '
                ' so that the following parameters are ignored: '
                'rz_inner_surface, rz_outer_surface, dr.'
            )
            if str(famus_filename)[-6:] != '.focus':
                raise ValueError('Famus filename must end in .focus')

        self.famus_filename = famus_filename
        ox, oy, oz, Ic, M0s = np.loadtxt(
            self.famus_filename,
            skiprows=3, usecols=[3, 4, 5, 6, 7], delimiter=',', unpack=True
        )

        # remove any dipoles where the diagnostic ports should be
        nonzero_inds = (Ic == 1.0)
        ox = ox[nonzero_inds]
        oy = oy[nonzero_inds]
        oz = oz[nonzero_inds]
        self.Ic_inds = nonzero_inds
        premade_dipole_grid = np.array([ox, oy, oz]).T
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
        self.dipole_grid_xyz = premade_dipole_grid

        B_max = 1.465  # value used in FAMUS runs for MUSE
        mu0 = 4 * np.pi * 1e-7
        cell_vol = M0s * mu0 / B_max
        self.m_maxima = B_max * cell_vol[self.Ic_inds] / mu0
        if self.pol_vectors is not None:
            if self.pol_vectors.shape[0] != self.ndipoles:
                raise ValueError('First dimension of `pol_vectors` array '
                                 'must equal the number of dipoles')
        self._optimization_setup()

    def _setup_uniform_rz_grid(self):
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

        r_max = np.max(self.r_outer)
        r_min = np.min(self.r_outer)
        z_max = np.max(self.z_outer)
        z_min = np.min(self.z_outer)

        # Initialize uniform grid of curved, square bricks
        Nr = int((r_max - r_min) / self.dr)
        self.Nr = Nr
        self.dz = self.dr
        Nz = int((z_max - z_min) / self.dz)
        self.Nz = Nz
        phi = 2 * np.pi * np.copy(self.plasma_boundary.quadpoints_phi)
        print('dR = {0:.2f}'.format(self.dr))
        print('dZ = {0:.2f}'.format(self.dz))
        R = np.linspace(r_min, r_max, Nr)
        Z = np.linspace(z_min, z_max, Nz)

        # Make 3D mesh
        R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
        self.RPhiZ = np.transpose(np.array([R, Phi, Z]), [1, 2, 3, 0])
        RPhiZ_grid = np.transpose(self.RPhiZ, [0, 2, 1, 3])
        RPhiZ_grid = RPhiZ_grid.reshape(RPhiZ_grid.shape[0] * RPhiZ_grid.shape[1], RPhiZ_grid.shape[2], 3)
        return RPhiZ_grid

    def geo_setup(self):
        """
            Initialize 'b' vector in 0.5 * ||Am - b||^2 part of the optimization,
            corresponding to the normal component of the target fields. Note
            the factor of two in the least-squares term: 0.5 * m.T @ (A.T @ A) @ m - b.T @ m.
            Also initialize the A, A.T @ b, and A.T @ A matrices that are important in the
            optimization problem.
        """
        # Have the uniform grid, now need to loop through and eliminate cells.
        t1 = time.time()

        contig = np.ascontiguousarray
        # Sometimes use 
        # self.plasma_boundary.unitnormal().reshape(-1, 3)
        normal_inner = self.rz_inner_surface.unitnormal().reshape(-1, 3)   
        normal_outer = self.rz_outer_surface.unitnormal().reshape(-1, 3)   
        B_max = 1.465  # value used in FAMUS runs for MUSE
        mu0 = 4 * np.pi * 1e-7

        # Make a uniform cylindrical grid, usually with dr=dz, i.e. "cylindrical bricks"
        if self.coordinate_flag == 'cylindrical':
            RPhiZ_grid = self._setup_uniform_rz_grid()
            self.final_grid, self.inds = sopp.define_a_uniform_cylindrical_grid_between_two_toroidal_surfaces(
                contig(2 * np.pi * self.phi), 
                contig(normal_inner), 
                contig(normal_outer),
                contig(RPhiZ_grid), 
                contig(self.r_inner), 
                contig(self.r_outer), 
                contig(self.z_inner), 
                contig(self.z_outer)
            )
            # define_a_uniform_cylindrical_grid_between_two_toroidal_surfaces only returns the inds to chop at
            # so need to loop through and chop the grid now
            self.inds = np.array(self.inds, dtype=int)
            for i in reversed(range(1, len(self.inds))):
                for j in range(0, i):
                    self.inds[i] += self.inds[j]
            self.ndipoles = self.inds[-1]
            final_grid = []
            for i in range(self.final_grid.shape[0]):
                if not np.allclose(self.final_grid[i, :], 0.0):
                    final_grid.append(self.final_grid[i, :])
            self.final_grid = np.array(final_grid)
            cell_vol = self.final_grid[:, 0] * self.dr * self.dz * 2 * np.pi / (self.nphi * self.plasma_boundary.nfp * 2)
            # alternatively, weight things roughly by the minor radius
            # cell_vol = np.sqrt((dipole_grid_r - self.plasma_boundary.get_rc(0, 0)) ** 2 + dipole_grid_z ** 2) * self.dr * self.dz * 2 * np.pi / (self.nphi * self.plasma_boundary.nfp * 2)

            self.m_maxima = B_max * cell_vol / mu0
            self.dipole_grid_xyz = np.zeros(self.final_grid.shape)
            self.dipole_grid_xyz[:, 0] = self.final_grid[:, 0] * np.cos(self.final_grid[:, 1])
            self.dipole_grid_xyz[:, 1] = self.final_grid[:, 0] * np.sin(self.final_grid[:, 1])
            self.dipole_grid_xyz[:, 2] = self.final_grid[:, 2]
            self.pm_phi = self.final_grid[:, 1]
        else:
            # If some other coordinate_flag, use a retangular Cartesian grid
            self._setup_uniform_grid()
            self.dipole_grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
                contig(normal_inner), 
                contig(normal_outer), 
                contig(self.xyz_uniform), 
                contig(self.xyz_inner), 
                contig(self.xyz_outer)
            )
            inds = np.ravel(np.logical_not(np.all(self.dipole_grid_xyz == 0.0, axis=-1)))
            self.dipole_grid_xyz = self.dipole_grid_xyz[inds, :]
            self.ndipoles = self.dipole_grid_xyz.shape[0]
            self.pm_phi = np.arctan2(self.dipole_grid_xyz[:, 1], self.dipole_grid_xyz[:, 0])
            cell_vol = self.dx * self.dy * self.dz 
            self.m_maxima = B_max * cell_vol / mu0 * np.ones(self.ndipoles)
            pointsToVTK(
                'dipole_grid',
                contig(self.dipole_grid_xyz[:, 0]),
                contig(self.dipole_grid_xyz[:, 1]),
                contig(self.dipole_grid_xyz[:, 2])
            )
        if self.pol_vectors is not None:
            if self.pol_vectors.shape[0] != self.ndipoles:
                raise ValueError('First dimension of `pol_vectors` array '
                                 'must equal the number of dipoles')

        t2 = time.time()
        print("Took t = ", t2 - t1, " s to perform the C++ grid cell eliminations.")
        self._optimization_setup()

    def _optimization_setup(self):

        if self.Bn.shape != (self.nphi, self.ntheta):
            raise ValueError(
                'Normal magnetic field surface data is incorrect shape.'
            )

        # minus sign below because ||Ax - b||^2 term but original
        # term is integral(B_P + B_C + B_M)^2
        self.b_obj = - self.Bn.reshape(self.nphi * self.ntheta)

        # Compute geometric factor with the C++ routine
        self.A_obj, self.ATb = sopp.dipole_field_Bn(
            np.ascontiguousarray(self.plasma_boundary.gamma().reshape(-1, 3)),
            np.ascontiguousarray(self.dipole_grid_xyz),
            np.ascontiguousarray(self.plasma_boundary.unitnormal().reshape(-1, 3)),
            self.plasma_boundary.nfp, int(self.plasma_boundary.stellsym),
            np.ascontiguousarray(self.b_obj),
            self.coordinate_flag,  # cartesian, cylindrical, or simple toroidal
            self.R0
        )

        # Rescale the A matrix so that 0.5 * ||Am - b||^2 = f_b,
        # where f_b is the metric for Bnormal on the plasma surface
        Ngrid = self.nphi * self.ntheta
        self.A_obj = self.A_obj.reshape(self.nphi * self.ntheta, self.ndipoles * 3)
        self.ATb = np.ravel(self.ATb)
        Nnorms = np.ravel(np.sqrt(np.sum(self.plasma_boundary.normal() ** 2, axis=-1)))
        for i in range(self.A_obj.shape[0]):
            self.A_obj[i, :] = self.A_obj[i, :] * np.sqrt(Nnorms[i] / Ngrid)
        self.b_obj = self.b_obj * np.sqrt(Nnorms / Ngrid)
        self.ATb = self.A_obj.T @ self.b_obj

        # Compute singular values of A, use this to determine optimal step size
        # for the MwPGP algorithm, with alpha ~ 2 / ATA_scale
        S = np.linalg.svd(self.A_obj, full_matrices=False, compute_uv=False)
        self.ATA_scale = S[0] ** 2

        # Set initial condition for the dipoles to default IC
        self.m0 = np.zeros(self.ndipoles * 3)

        # Set m to zeros
        self.m = self.m0

        # Print initial f_B metric using the initial guess
        total_error = np.linalg.norm((self.A_obj.dot(self.m0) - self.b_obj), ord=2) ** 2 / 2.0
        print('f_B (total with initial SIMSOPT guess) = ', total_error)

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
        print('Maximum dipole moment = ', np.max(self.m_maxima))
        print('Shape of A matrix = ', self.A_obj.shape)
        print('Shape of b vector = ', self.b_obj.shape)
        print('Initial error on plasma surface = {0:.4e}'.format(total_error))
