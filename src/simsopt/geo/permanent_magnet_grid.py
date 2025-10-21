from pathlib import Path
import warnings

import numpy as np
from pyevtk.hl import pointsToVTK

from .._core.descriptor import OneofStrings
from . import Surface
import simsoptpp as sopp

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
        plasma_boundary: Surface class object 
            Representing the plasma boundary surface. Gets converted
            into SurfaceRZFourier object for ease of use.
        Bn: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
            Magnetic field (coils and plasma) at the plasma
            boundary. Typically this will be the optimized plasma
            magnetic field from a stage-1 optimization, and the
            optimized coils from a basic stage-2 optimization.
            This variable must be specified to run the permanent
            magnet optimization.
        coordinate_flag: string
            Flag to specify the coordinate system used for the grid and optimization.
            This is primarily used to tell the optimizer which coordinate directions
            should be considered, "grid-aligned". Therefore, you can do weird stuff
            like cartesian grid-alignment on a simple toroidal grid, which might
            be self-defeating. However, if coordinate_flag='cartesian' a rectangular
            Cartesian grid is initialized using the Nx, Ny, and Nz parameters. If
            the coordinate_flag='cylindrical', a uniform cylindrical grid is initialized
            using the dr and dz parameters.
    """
    coordinate_flag = OneofStrings("cartesian", "cylindrical", "toroidal")

    def __init__(self, plasma_boundary: Surface, Bn, coordinate_flag='cartesian'):
        Bn = np.array(Bn)
        if len(Bn.shape) != 2:
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        self.Bn = Bn

        if coordinate_flag == 'cartesian':
            warnings.warn('Cartesian grid of rectangular cubes will be built, since '
                          'coordinate_flag = "cartesian". However, such a grid can be '
                          'appropriately reflected only for nfp = 2 and nfp = 4, '
                          'unlike the uniform cylindrical grid, which has continuous '
                          'rotational symmetry.')
        self.coordinate_flag = coordinate_flag
        self.plasma_boundary = plasma_boundary.to_RZFourier()
        self.R0 = self.plasma_boundary.get_rc(0, 0)
        self.nphi = len(self.plasma_boundary.quadpoints_phi)
        self.ntheta = len(self.plasma_boundary.quadpoints_theta)

    def _setup_uniform_grid(self):
        """
        Initializes a uniform grid in cartesian coordinates and sets
        some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the two boundaries
        self.xyz_inner = self.inner_toroidal_surface.gamma().reshape(-1, 3)
        self.xyz_outer = self.outer_toroidal_surface.gamma().reshape(-1, 3)
        if self.coordinate_flag != 'cylindrical':
            x_outer = self.xyz_outer[:, 0]
            y_outer = self.xyz_outer[:, 1]
            z_outer = self.xyz_outer[:, 2]

            x_max = np.max(x_outer)
            x_min = np.min(x_outer)
            y_max = np.max(y_outer)
            y_min = np.min(y_outer)
            z_max = np.max(z_outer)
            z_min = np.min(z_outer)
            z_max = max(z_max, abs(z_min))
            print(x_min, x_max, y_min, y_max, z_min, z_max)

            # Initialize uniform grid
            Nx = self.Nx
            Ny = self.Ny
            Nz = self.Nz
            self.dx = (x_max - x_min) / (Nx - 1)
            self.dy = (y_max - y_min) / (Ny - 1)
            self.dz = 2 * z_max / (Nz - 1)
            print(Nx, Ny, Nz, self.dx, self.dy, self.dz)

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
        else:
            # Get (R, Z) coordinates of the outer boundary
            rphiz_outer = np.array(
                [np.sqrt(self.xyz_outer[:, 0] ** 2 + self.xyz_outer[:, 1] ** 2),
                 np.arctan2(self.xyz_outer[:, 1], self.xyz_outer[:, 0]),
                 self.xyz_outer[:, 2]]
            ).T

            r_max = np.max(rphiz_outer[:, 0])
            r_min = np.min(rphiz_outer[:, 0])
            z_max = np.max(rphiz_outer[:, 2])
            z_min = np.min(rphiz_outer[:, 2])

            # Initialize uniform grid of curved, square bricks
            Nr = int((r_max - r_min) / self.dr)
            self.Nr = Nr
            self.dz = self.dr
            Nz = int((z_max - z_min) / self.dz)
            self.Nz = Nz
            phi = 2 * np.pi * np.copy(self.plasma_boundary.quadpoints_phi)
            R = np.linspace(r_min, r_max, Nr)
            Z = np.linspace(z_min, z_max, Nz)

            # Make 3D mesh
            R, Phi, Z = np.meshgrid(R, phi, Z, indexing='ij')
            X = R * np.cos(Phi)
            Y = R * np.sin(Phi)
            self.xyz_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(-1, 3)

        # Save uniform grid before we start chopping off parts.
        contig = np.ascontiguousarray
        pointsToVTK('uniform_grid', contig(self.xyz_uniform[:, 0]),
                    contig(self.xyz_uniform[:, 1]), contig(self.xyz_uniform[:, 2]))

    @classmethod
    def geo_setup_from_famus(cls, plasma_boundary, Bn, famus_filename, **kwargs):
        """
        Function to initialize a SIMSOPT PermanentMagnetGrid from a 
        pre-existing FAMUS file defining a grid of magnets. For
        example, this function is used for the
        half-Tesla NCSX configuration (c09r000) and MUSE grids.

        Args
        ----------
        famus_filename : string
            Filename of a FAMUS grid file (a pre-made dipole grid).
        kwargs: The following are valid keyword arguments.
            m_maxima: float or 1D numpy array, shape (Ndipoles)
                Optional array of maximal dipole magnitudes for the permanent
                magnets. If not provided, defaults to the common magnets
                used in the MUSE design, with strengths ~ 1 Tesla.
            pol_vectors: 3D numpy array, shape (Ndipoles, Ncoords, 3)
                Optional set of local coordinate systems for each dipole, 
                which specifies which directions should be considered grid-aligned.
                Ncoords can be > 3, as in the PM4Stell design.
            coordinate_flag: string
                Flag to specify the coordinate system used for the grid and optimization.
                This is primarily used to tell the optimizer which coordinate directions
                should be considered, "grid-aligned". Therefore, you can do weird stuff
                like cartesian grid-alignment on a simple toroidal grid, which might
                be self-defeating. However, if coordinate_flag='cartesian' a rectangular
                Cartesian grid is initialized using the Nx, Ny, and Nz parameters. If
                the coordinate_flag='cylindrical', a uniform cylindrical grid is initialized
                using the dr and dz parameters.
            downsample: int
                Optional integer for downsampling the FAMUS grid, since
                the MUSE and other grids can be very high resolution
                and this makes CI take a long while.
        Returns
        -------
        pm_grid: An initialized PermanentMagnetGrid class object.

        """
        coordinate_flag = kwargs.pop("coordinate_flag", "cartesian")
        downsample = kwargs.pop("downsample", 1)
        pol_vectors = kwargs.pop("pol_vectors", None)
        m_maxima = kwargs.pop("m_maxima", None)
        if str(famus_filename)[-6:] != '.focus':
            raise ValueError('Famus filename must end in .focus')

        pm_grid = cls(plasma_boundary, Bn, coordinate_flag)
        pm_grid.famus_filename = famus_filename
        ox, oy, oz, Ic, M0s = np.loadtxt(famus_filename, skiprows=3, usecols=[3, 4, 5, 6, 7],
                                         delimiter=',', unpack=True)

        # Downsample the resolution as needed
        inds_total = np.arange(len(ox))
        inds_downsampled = inds_total[::downsample]

        # also remove any dipoles where the diagnostic ports should be
        nonzero_inds = np.intersect1d(np.ravel(np.where(Ic == 1.0)), inds_downsampled)
        pm_grid.Ic_inds = nonzero_inds
        ox = ox[nonzero_inds]
        oy = oy[nonzero_inds]
        oz = oz[nonzero_inds]
        premade_dipole_grid = np.array([ox, oy, oz]).T
        pm_grid.ndipoles = premade_dipole_grid.shape[0]

        # Not normalized to 1 like quadpoints_phi!
        pm_grid.pm_phi = np.arctan2(premade_dipole_grid[:, 1], premade_dipole_grid[:, 0])
        uniq_phi, counts_phi = np.unique(pm_grid.pm_phi.round(decimals=6), return_counts=True)
        pm_grid.pm_nphi = len(uniq_phi)
        pm_grid.pm_uniq_phi = uniq_phi
        pm_grid.inds = counts_phi
        for i in reversed(range(1, pm_grid.pm_nphi)):
            for j in range(0, i):
                pm_grid.inds[i] += pm_grid.inds[j]
        pm_grid.dipole_grid_xyz = premade_dipole_grid

        if m_maxima is None:
            B_max = 1.465  # value used in FAMUS runs for MUSE
            mu0 = 4 * np.pi * 1e-7
            cell_vol = M0s * mu0 / B_max
            pm_grid.m_maxima = B_max * cell_vol[nonzero_inds] / mu0
        else:
            if isinstance(m_maxima, float):
                pm_grid.m_maxima = m_maxima * np.ones(pm_grid.ndipoles)
            else:
                pm_grid.m_maxima = m_maxima
            if len(pm_grid.m_maxima) != pm_grid.ndipoles:
                raise ValueError('m_maxima passed to geo_setup_from_famus but with '
                                 'the wrong shape, i.e. != number of dipoles.')

        if pol_vectors is not None:
            pol_vectors = np.array(pol_vectors)
            if len(pol_vectors.shape) != 3:
                raise ValueError('pol vectors must be a 3D array.')
            elif pol_vectors.shape[2] != 3:
                raise ValueError('Third dimension of `pol_vectors` array '
                                 'must be 3')
            elif pm_grid.coordinate_flag != 'cartesian':
                raise ValueError('pol_vectors argument can only be used with coordinate_flag = cartesian currently')
            elif pol_vectors.shape[0] != pm_grid.ndipoles:
                raise ValueError('First dimension of `pol_vectors` array '
                                 'must equal the number of dipoles')

        pm_grid.pol_vectors = pol_vectors
        pm_grid._optimization_setup()
        return pm_grid

    @classmethod
    def geo_setup_between_toroidal_surfaces(
        cls,
        plasma_boundary,
        Bn,
        inner_toroidal_surface: Surface,
        outer_toroidal_surface: Surface,
        **kwargs,
    ):
        """
        Function to initialize a SIMSOPT PermanentMagnetGrid from a 
        volume defined by two toroidal surfaces. These must be specified
        directly. Often a good choice is made by extending the plasma 
        boundary by its normal vectors.

        Args
        ----------
        inner_toroidal_surface: Surface class object 
            Representing the inner toroidal surface of the volume.
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        outer_toroidal_surface: Surface object representing
            the outer toroidal surface of the volume. Typically 
            want this to have same quadrature points as the inner
            surface for a functional grid setup. 
            Gets converted into SurfaceRZFourier object for 
            ease of use.
        kwargs: The following are valid keyword arguments.
            m_maxima: float or 1D numpy array, shape (Ndipoles)
                Optional array of maximal dipole magnitudes for the permanent
                magnets. If not provided, defaults to the common magnets
                used in the MUSE design, with strengths ~ 1 Tesla.
            pol_vectors: 3D numpy array, shape (Ndipoles, Ncoords, 3)
                Optional set of local coordinate systems for each dipole, 
                which specifies which directions should be considered grid-aligned.
                Ncoords can be > 3, as in the PM4Stell design.
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
        Returns
        -------
        pm_grid: An initialized PermanentMagnetGrid class object.

        """
        coordinate_flag = kwargs.pop("coordinate_flag", "cartesian")
        pol_vectors = kwargs.pop("pol_vectors", None)
        m_maxima = kwargs.pop("m_maxima", None)
        pm_grid = cls(plasma_boundary, Bn, coordinate_flag)
        Nx = kwargs.pop("Nx", 10)
        Ny = kwargs.pop("Ny", 10)
        Nz = kwargs.pop("Nz", 10)
        dr = kwargs.pop("dr", 0.1)
        dz = kwargs.pop("dz", 0.1)
        if Nx <= 0 or Ny <= 0 or Nz <= 0:
            raise ValueError('Nx, Ny, and Nz should be positive integers')
        if dr <= 0 or dz <= 0:
            raise ValueError('dr and dz should be positive floats')
        pm_grid.dr = dr
        pm_grid.dz = dz
        pm_grid.Nx = Nx
        pm_grid.Ny = Ny
        pm_grid.Nz = Nz
        pm_grid.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        pm_grid.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()
        warnings.warn(
            'Plasma boundary and inner and outer toroidal surfaces should '
            'all have the same "range" parameter in order for a permanent'
            ' magnet grid to be correctly initialized.'
        )

        # Have the uniform grid, now need to loop through and eliminate cells.
        contig = np.ascontiguousarray
        normal_inner = inner_toroidal_surface.unitnormal().reshape(-1, 3)
        normal_outer = outer_toroidal_surface.unitnormal().reshape(-1, 3)
        pm_grid._setup_uniform_grid()
        pm_grid.dipole_grid_xyz = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(normal_inner),
            contig(normal_outer),
            contig(pm_grid.xyz_uniform),
            contig(pm_grid.xyz_inner),
            contig(pm_grid.xyz_outer))
        inds = np.ravel(np.logical_not(np.all(pm_grid.dipole_grid_xyz == 0.0, axis=-1)))
        pm_grid.dipole_grid_xyz = pm_grid.dipole_grid_xyz[inds, :]
        pm_grid.ndipoles = pm_grid.dipole_grid_xyz.shape[0]
        pm_grid.pm_phi = np.arctan2(pm_grid.dipole_grid_xyz[:, 1], pm_grid.dipole_grid_xyz[:, 0])
        if coordinate_flag == 'cylindrical':
            cell_vol = np.sqrt(pm_grid.dipole_grid_xyz[:, 0] ** 2 + pm_grid.dipole_grid_xyz[:, 1] ** 2) * pm_grid.dr * pm_grid.dz * 2 * np.pi / (pm_grid.nphi * pm_grid.plasma_boundary.nfp * 2)
        else:
            cell_vol = pm_grid.dx * pm_grid.dy * pm_grid.dz * np.ones(pm_grid.ndipoles)
        pointsToVTK('dipole_grid',
                    contig(pm_grid.dipole_grid_xyz[:, 0]),
                    contig(pm_grid.dipole_grid_xyz[:, 1]),
                    contig(pm_grid.dipole_grid_xyz[:, 2]))
        if m_maxima is None:
            B_max = 1.465  # value used in FAMUS runs for MUSE
            mu0 = 4 * np.pi * 1e-7
            pm_grid.m_maxima = B_max * cell_vol / mu0
        else:
            if isinstance(m_maxima, float):
                pm_grid.m_maxima = m_maxima * np.ones(pm_grid.ndipoles)
            else:
                pm_grid.m_maxima = m_maxima
            if len(pm_grid.m_maxima) != pm_grid.ndipoles:
                raise ValueError(
                    'm_maxima passed to geo_setup_from_famus but with '
                    'the wrong shape, i.e. != number of dipoles.')
        if pol_vectors is not None:
            pol_vectors = np.array(pol_vectors)
            if len(pol_vectors.shape) != 3:
                raise ValueError('pol vectors must be a 3D array.')
            elif pol_vectors.shape[2] != 3:
                raise ValueError('Third dimension of `pol_vectors` array must be 3')
            elif pm_grid.coordinate_flag != 'cartesian':
                raise ValueError('pol_vectors argument can only be used with coordinate_flag = cartesian currently')
            elif pol_vectors.shape[0] != pm_grid.ndipoles:
                raise ValueError('First dimension of `pol_vectors` array '
                                 'must equal the number of dipoles')

        pm_grid.pol_vectors = pol_vectors
        pm_grid._optimization_setup()
        return pm_grid

    def _optimization_setup(self):

        if self.Bn.shape != (self.nphi, self.ntheta):
            raise ValueError('Normal magnetic field surface data is incorrect shape.')

        # minus sign below because ||Ax - b||^2 term but original
        # term is integral(B_P + B_C + B_M)^2
        self.b_obj = - self.Bn.reshape(self.nphi * self.ntheta)

        # Compute geometric factor with the C++ routine
        self.A_obj = sopp.dipole_field_Bn(
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

    def write_to_famus(self, out_dir=''):
        """
        Takes a PermanentMagnetGrid object and saves the geometry
        and optimization solution into a FAMUS input file.

        Args:
            out_dir: Path object for the output directory for saved files.
        """
        ndipoles = self.ndipoles
        m = self.m.reshape(ndipoles, 3)
        ox = self.dipole_grid_xyz[:, 0]
        oy = self.dipole_grid_xyz[:, 1]
        oz = self.dipole_grid_xyz[:, 2]

        # Transform the solution vector to the Cartesian basis if necessary
        if self.coordinate_flag == 'cartesian':
            mx = m[:, 0]
            my = m[:, 1]
            mz = m[:, 2]
        elif self.coordinate_flag == 'cylindrical':
            cos_ophi = np.cos(self.pm_phi)
            sin_ophi = np.sin(self.pm_phi)
            mx = m[:, 0] * cos_ophi - m[:, 1] * sin_ophi
            my = m[:, 0] * sin_ophi + m[:, 1] * cos_ophi
            mz = m[:, 2]
        elif self.coordinate_flag == 'toroidal':
            ormajor = np.sqrt(ox**2 + oy**2)
            otheta = np.arctan2(oz, ormajor - self.R0)
            cos_ophi = np.cos(self.pm_phi)
            sin_ophi = np.sin(self.pm_phi)
            cos_otheta = np.cos(otheta)
            sin_otheta = np.sin(otheta)
            mx = m[:, 0] * cos_ophi * cos_otheta \
                - m[:, 1] * sin_ophi              \
                - m[:, 2] * cos_ophi * sin_otheta
            my = m[:, 0] * sin_ophi * cos_otheta \
                + m[:, 1] * cos_ophi              \
                - m[:, 2] * sin_ophi * sin_otheta
            mz = m[:, 0] * sin_otheta \
                + m[:, 2] * cos_otheta

        m0 = self.m_maxima
        pho = np.sqrt(np.sum(m ** 2, axis=-1)) / m0
        Lc = 0

        mp = np.arctan2(my, mx)
        mt = np.arctan2(np.sqrt(mx ** 2 + my ** 2), mz)
        coilname = ["pm_{:010d}".format(i) for i in range(1, ndipoles + 1)]
        Ic = 1
        # symmetry = 2 for stellarator symmetry
        symmetry = int(self.plasma_boundary.stellsym) + 1
        filename = Path(out_dir) / 'SIMSOPT_dipole_solution.focus'

        with open(filename, "w") as wfile:
            wfile.write(" # Total number of dipoles,  momentq \n")
            wfile.write(f"{ndipoles:6d}  {1:4d}\n")     # .format(ndipoles, 1))
            wfile.write("#coiltype, symmetry,  coilname,  ox,  oy,  oz,  Ic,  M_0,  pho,  Lc,  mp,  mt \n")
            for i in range(ndipoles):
                wfile.write(f" 2, {symmetry:1d}, {coilname[i]}, {ox[i]:15.8E}, {oy[i]:15.8E}, "
                            f"{oz[i]:15.8E}, {Ic:2d}, {m0[i]:15.8E}, {pho[i]:15.8E}, {Lc:2d}, "
                            f"{mp[i]:15.8E}, {mt[i]:15.8E} \n")

    def rescale_for_opt(self, reg_l0, reg_l1, reg_l2, nu):
        """
        Scale regularizers to the largest scale of ATA (~1e-6)
        to avoid regularization >> ||Am - b|| ** 2 term in the optimization.
        The prox operator uses reg_l0 * nu for the threshold so normalization
        below allows reg_l0 and reg_l1 values to be exactly the thresholds
        used in calculation of the prox. Then add contributions to ATA and
        ATb coming from extra loss terms such as L2 regularization and
        relax-and-split. Currently does not rescale the L2 and nu
        hyperparameters, but users may want to play around with this.

        Args:
            reg_l0: L0 regularization.
            reg_l1: L1 regularization.
            reg_l2: L2 regularization.
            nu: nu hyperparameter in relax-and-split optimization.

        Returns:
            reg_l0: Rescaled L0 regularization.
            reg_l1: Rescaled L1 regularization.
            reg_l2: Rescaled L2 regularization.
            nu: Rescaled nu hyperparameter in relax-and-split optimization.
        """

        print('L2 regularization being used with coefficient = {0:.2e}'.format(reg_l2))

        if reg_l0 < 0 or reg_l0 > 1:
            raise ValueError(
                'L0 regularization must be between 0 and 1. This '
                'value is automatically scaled to the largest of the '
                'dipole maximum values, so reg_l0 = 1 should basically '
                'truncate all the dipoles to zero. ')

        # Rescale L0 and L1 so that the values used for thresholding
        # are only parametrized by the values of reg_l0 and reg_l1
        reg_l0 = reg_l0 / (2 * nu)
        reg_l1 = reg_l1 / nu

        # may want to rescale nu, otherwise just scan this value
        # nu = nu / self.ATA_scale

        # Do not rescale L2 term for now.
        reg_l2 = reg_l2

        # Update algorithm step size if we have extra smooth, convex loss terms
        self.ATA_scale += 2 * reg_l2 + 1.0 / nu

        return reg_l0, reg_l1, reg_l2, nu
