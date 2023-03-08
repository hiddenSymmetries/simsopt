import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
import simsoptpp as sopp
import time
import warnings

__all__ = ['PermanentMagnetGrid']


class PermanentMagnetGrid:
    r"""
        ``PermanentMagnetGrid`` is a class for setting up the grid,
        plasma surface, and other objects needed to perform permanent
        magnet optimization for stellarators. The class
        takes as input two toroidal surfaces specified as SurfaceRZFourier
        objects, and initializes a set of points (in cylindrical coordinates)
        between these surfaces. If an existing FAMUS grid file called
        from FAMUS is desired, set the value of famus_filename,
        instead of passing the surfaces.
        It finishes initialization by pre-computing
        a number of quantities required for the optimization such as the
        geometric factor in the dipole part of the magnetic field and the
        target Bfield that is a sum of the coil and plasma magnetic fields.
        It then provides rescaling in anticipation for optimization.

        Args:
            plasma_boundary:  SurfaceRZFourier object representing
                              the plasma boundary surface.
            famus_filename: Filename of a FAMUS grid file (a pre-made dipole grid).
                              The paper uses the
                              half-Tesla NCSX configuration (c09r000) and MUSE
                              to compare with the equivalent FAMUS run.
                              If this flag is specified, rz_inner_surface,
                              rz_outer_surface, dr, plasma_offset, and
                              coil_offset are all ignored.
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
            coordinate_flag:  Flag to specify the coordinate system used for the grid and optimization.
                              This is only used to tell the optimizer which coordinate directions
                              should be considered, "grid-aligned". Therefore, you can do weird stuff
                              like cartesian grid-alignment on a simple toroidal grid, which might
                              be self-defeating.
            pol_vectors:      Optional set of local coordinate systems for each dipole, 
                              shape (Ndipoles, Ncoords, 3)
                              which specifies which directions should be considered grid-aligned.
                              Ncoords can be > 3, as in the PM4Stell design.
    """

    def __init__(
        self, plasma_boundary,
        rz_inner_surface=None,
        rz_outer_surface=None, plasma_offset=0.1,
        coil_offset=0.2, Bn=None, dr=0.1,
        filename=None, surface_flag='vmec',
        coordinate_flag='cartesian',
        famus_filename=None,
        pol_vectors=None
    ):
        if plasma_offset <= 0 or coil_offset <= 0:
            raise ValueError('permanent magnets must be offset from the plasma')

        if famus_filename is not None:
            warnings.warn(
                'famus_filename variable is set, so a pre-defined grid will be used. '
                ' so that the following parameters are ignored: '
                'rz_inner_surface, rz_outer_surface, dr, plasma_offset, '
                'and coil_offset.'
            )
        self.famus_filename = famus_filename
        self.filename = filename
        self.surface_flag = surface_flag
        self.plasma_offset = plasma_offset
        self.coil_offset = coil_offset
        self.Bn = Bn
        self.dr = dr

        if coordinate_flag not in ['cartesian', 'cylindrical', 'toroidal']:
            raise NotImplementedError(
                'Only cartesian, cylindrical, and toroidal (simple toroidal)'
                ' coordinate systems have been implemented so far.'
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
        # If the inner surface is not specified, make default surface.
        if (rz_inner_surface is None) and (famus_filename is None):
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
        if famus_filename is None:
            if not isinstance(self.rz_inner_surface, SurfaceRZFourier):
                raise ValueError("Inner surface is not SurfaceRZFourier object.")

        # If the outer surface is not specified, make default surface.
        if (rz_outer_surface is None) and (famus_filename is None):
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
        if famus_filename is None:
            if not isinstance(self.rz_outer_surface, SurfaceRZFourier):
                raise ValueError("Outer surface is not SurfaceRZFourier object.")
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to get the SurfaceRZFourier objects done")

        if famus_filename is None:
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

            # make_final_surface only returns the inds to chop at
            # so need to loop through and chop the grid now
            self.inds = np.array(self.inds, dtype=int)
            for i in reversed(range(1, len(self.inds))):
                for j in range(0, i):
                    self.inds[i] += self.inds[j]
            self.ndipoles = self.inds[-1]
            final_grid = []
            for i in range(self.final_RZ_grid.shape[0]):
                if not np.allclose(self.final_RZ_grid[i, :], 0.0):
                    final_grid.append(self.final_RZ_grid[i, :])
            self.final_RZ_grid = np.array(final_grid)
            t2 = time.time()
            print("Took t = ", t2 - t1, " s to perform the C++ grid cell eliminations.")
        else:
            ox, oy, oz, Ic = np.loadtxt(
                '../../tests/test_files/' + self.famus_filename,
                skiprows=3, usecols=[3, 4, 5, 6], delimiter=',', unpack=True
            )

            # remove any dipoles where the diagnostic ports should be
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
            self.final_RZ_grid = np.zeros((self.ndipoles, 3))
            self.final_RZ_grid[:, 0] = np.sqrt(premade_dipole_grid[:, 0] ** 2 + premade_dipole_grid[:, 1] ** 2)
            self.final_RZ_grid[:, 1] = self.pm_phi
            self.final_RZ_grid[:, 2] = premade_dipole_grid[:, 2]
            self.dipole_grid_xyz = premade_dipole_grid

        xyz_plasma = self.plasma_boundary.gamma()
        self.r_plasma = np.sqrt(xyz_plasma[:, :, 0] ** 2 + xyz_plasma[:, :, 1] ** 2)
        self.z_plasma = xyz_plasma[:, :, 2]

        if pol_vectors is not None:
            if pol_vectors.shape[0] != self.ndipoles:
                raise ValueError('First dimension of `pol_vectors` array '
                                 'must equal the number of dipoles')
            elif pol_vectors.shape[2] != 3:
                raise ValueError('Third dimension of `pol_vectors` array '
                                 'must be 3')
            elif coordinate_flag != 'cartesian':
                raise ValueError('pol_vectors argument can only be used with coordinate_flag = cartesian currently')
            else:
                self.pol_vectors = pol_vectors

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
        self.m0 = np.zeros(self.ndipoles * 3)

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

        # get normal vectors of inner and outer PM surfaces
        self.normal_inner = np.copy(self.rz_inner_surface.unitnormal())
        self.normal_outer = np.copy(self.rz_outer_surface.unitnormal())

        # optional code below to debug if the normal vectors are behaving
        # which seems to be an issue with the MUSE inner/outer PM surfaces
        if False:
            from matplotlib import pyplot as plt
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
        B_max = 1.465  # value used in FAMUS runs for MUSE
        mu0 = 4 * np.pi * 1e-7
        dipole_grid_r = np.zeros(self.ndipoles)
        dipole_grid_x = np.zeros(self.ndipoles)
        dipole_grid_y = np.zeros(self.ndipoles)
        dipole_grid_phi = np.zeros(self.ndipoles)
        dipole_grid_z = np.zeros(self.ndipoles)
        running_tally = 0
        if self.famus_filename is not None:
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

        if self.famus_filename is not None:
            M0s = np.loadtxt(
                '../../tests/test_files/' + self.famus_filename,
                skiprows=3, usecols=[7], delimiter=','
            )
            cell_vol = M0s * mu0 / B_max
            self.m_maxima = B_max * cell_vol[self.Ic_inds] / mu0
        else:
            # defaults to cylindrical bricks, so volumes are weighted
            # by the cylindrical radius!
            cell_vol = dipole_grid_r * self.Delta_r * self.Delta_z * 2 * np.pi / (self.nphi * self.plasma_boundary.nfp * 2)
            # alternatively, weight things roughly by the minor radius
            #    cell_vol = np.sqrt((dipole_grid_r - self.plasma_boundary.get_rc(0, 0)) ** 2 + dipole_grid_z ** 2) * self.Delta_r * self.Delta_z * 2 * np.pi / (self.nphi * self.plasma_boundary.nfp * 2)
            self.dipole_grid_xyz = np.array([dipole_grid_x, dipole_grid_y, dipole_grid_z]).T
            self.m_maxima = B_max * cell_vol / mu0
        print(
            'Total initial volume for magnet placement = ',
            np.sum(cell_vol) * self.plasma_boundary.nfp * 2, ' m^3'
        )

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

    def _set_inner_rz_surface(self):
        """
            If the inner toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the
            plasma boundary and shifts it by self.plasma_offset at constant
            theta value.
        """
        if self.surface_flag == 'focus':
            rz_inner_surface = SurfaceRZFourier.from_focus(self.filename, range="half period", nphi=self.nphi, ntheta=self.ntheta)
        elif self.surface_flag == 'wout':
            rz_inner_surface = SurfaceRZFourier.from_wout(self.filename, range="half period", nphi=self.nphi, ntheta=self.ntheta)
        else:
            rz_inner_surface = SurfaceRZFourier.from_vmec_input(self.filename, range="half period", nphi=self.nphi, ntheta=self.ntheta)

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
            rz_outer_surface = SurfaceRZFourier.from_focus(self.filename, range="half period", nphi=self.nphi, ntheta=self.ntheta)
        elif self.surface_flag == 'wout':
            rz_outer_surface = SurfaceRZFourier.from_wout(self.filename, range="half period", nphi=self.nphi, ntheta=self.ntheta)
        else:
            rz_outer_surface = SurfaceRZFourier.from_vmec_input(self.filename, range="half period", nphi=self.nphi, ntheta=self.ntheta)

        # extend via the normal vector
        t1 = time.time()
        rz_outer_surface.extend_via_projected_normal(self.phi, self.plasma_offset + self.coil_offset)
        t2 = time.time()
        print("Outer surface extension took t = ", t2 - t1)
        self.rz_outer_surface = rz_outer_surface

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

    def as_dict(self) -> dict:
        d = {}
        d["@module"] = self.__class__.__module__
        d["@class"] = self.__class__.__name__
        d["plasma_boundary"] = self.plasma_boundary
        d["rz_inner_surface"] = self.rz_inner_surface
        d["rz_outer_surface"] = self.rz_outer_surface
        d["plasma_offset"] = self.plasma_offset
        d["coil_offset"] = self.coil_offset
        d["Bn"] = self.Bn
        d["dr"] = self.dr
        d["filename"] = self.filename
        d["surface_flag"] = self.surface_flag
        d["coordinate_flag"] = self.coordinate_flag
        return d

    @classmethod
    def from_dict(cls, d):
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
            d["coordinate_flag"],
        )
