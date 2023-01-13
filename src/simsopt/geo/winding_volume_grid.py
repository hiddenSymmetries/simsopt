import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from pyevtk.hl import pointsToVTK
import simsoptpp as sopp
import time
import warnings
from scipy.sparse import lil_matrix

__all__ = ['WindingVolumeGrid']


class WindingVolumeGrid:
    r"""
        ``WindingVolumeGrid`` is a class for setting up the grid,
        needed to perform finite-build coil optimization for stellarators
        using the method outlined in Kaptanoglu & Landreman 2023. This 
        class reuses some of the functionality used for the PermanentMagnetGrid
        class. It takes as input two toroidal surfaces specified as SurfaceRZFourier
        objects, and initializes a set of points 
        between these surfaces. If an existing FAMUS grid file called
        from FAMUS is desired, set the value of famus_filename,
        instead of passing the surfaces.

        Args:
            plasma_boundary:  SurfaceRZFourier object representing
                              the plasma boundary surface.
            famus_filename: Filename of a FAMUS grid file (a pre-made dipole grid).
                              If this flag is specified, rz_inner_surface,
                              rz_outer_surface, dx, dy, dz, plasma_offset, and
                              coil_offset are all ignored.
            rz_inner_surface: SurfaceRZFourier object representing
                              the inner toroidal surface of the volume.
                              Defaults to the plasma boundary, extended
                              by the boundary normal vector
            rz_outer_surface: SurfaceRZFourier object representing
                              the outer toroidal surface of the volume.
                              Defaults to the inner surface, extended
                              by the boundary normal vector
            plasma_offset:    Offset to use for generating the inner toroidal surface.
            coil_offset:      Offset to use for generating the outer toroidal surface.
            Bn:               Magnetic field (coils and plasma) at the plasma
                              boundary. Typically this will be the optimized plasma
                              magnetic field from a stage-1 optimization, and the
                              optimized coils from a basic stage-2 optimization.
            dx:               X-axis grid spacing in the winding volume grid. 
            dy:               Y-axis grid spacing in the winding volume grid. 
            dz:               Z-axis grid spacing in the winding volume grid. 
            filename:         Filename for the file containing the plasma boundary surface.
            surface_flag:     Flag to specify the format of the surface file. Defaults to VMEC.
            OUT_DIR:          Directory to save files in.
    """

    def __init__(
        self, plasma_boundary, Itarget_curve, Bn_Itarget, Itarget=1e6,
        rz_inner_surface=None,
        rz_outer_surface=None, plasma_offset=0.1,
        coil_offset=0.2, Bn=None,
        dx=0.02, dy=0.02, dz=0.02,
        filename=None, surface_flag='vmec',
        famus_filename=None, 
        OUT_DIR='', nx=2, ny=2, nz=2,
    ):
        if plasma_offset <= 0 or coil_offset <= 0:
            raise ValueError('permanent magnets must be offset from the plasma')

        if famus_filename is not None:
            warnings.warn(
                'famus_filename variable is set, so a pre-defined grid will be used. '
                ' so that the following parameters are ignored: '
                'rz_inner_surface, rz_outer_surface, dx, dy, dz, plasma_offset, '
                'and coil_offset.'
            )
        self.famus_filename = famus_filename
        self.filename = filename
        self.surface_flag = surface_flag
        self.plasma_offset = plasma_offset
        self.coil_offset = coil_offset
        self.Bn = Bn
        self.Bn_Itarget = Bn_Itarget
        self.Itarget = Itarget
        self.Itarget_curve = Itarget_curve
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.OUT_DIR = OUT_DIR
        self.n_functions = 11  # hard-coded for linear basis

        if not isinstance(plasma_boundary, SurfaceRZFourier):
            raise ValueError(
                'Plasma boundary must be specified as SurfaceRZFourier class.'
            )
        else:
            self.plasma_boundary = plasma_boundary
            self.R0 = plasma_boundary.get_rc(0, 0)
            # unlike plasma surface, use the whole phi and theta grids
            self.phi = self.plasma_boundary.quadpoints_phi
            self.nphi = len(self.phi)
            self.theta = self.plasma_boundary.quadpoints_theta
            self.ntheta = len(self.theta)

        if dx <= 0 or dy <= 0 or dz <= 0:
            raise ValueError('grid spacing must be > 0')

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

            self._setup_uniform_grid()

            # Have the uniform grid, now need to loop through and eliminate cells.
            t1 = time.time()

            contig = np.ascontiguousarray
            if self.surface_flag == 'focus' or self.surface_flag == 'wout':
                final_grid = sopp.make_winding_volume_grid(
                    # self.plasma_boundary.unitnormal().reshape(-1, 3), self.plasma_boundary.unitnormal().reshape(-1, 3),
                    contig(self.normal_inner.reshape(-1, 3)), contig(self.normal_outer.reshape(-1, 3)),
                    contig(self.XYZ_uniform), contig(self.xyz_inner), contig(self.xyz_outer),
                )
            else:
                final_grid = sopp.make_winding_volume_grid(
                    # self.plasma_boundary.unitnormal().reshape(-1, 3), self.plasma_boundary.unitnormal().reshape(-1, 3),
                    contig(self.normal_inner.reshape(-1, 3)), contig(self.normal_outer.reshape(-1, 3)),
                    contig(self.XYZ_uniform), contig(self.xyz_inner), contig(self.xyz_outer),
                )
            # remove all the grid elements that were omitted
            inds = np.ravel(np.logical_not(np.all(final_grid == 0.0, axis=-1)))
            self.XYZ_flat = final_grid[inds, :]
            print(self.XYZ_flat.shape, self.XYZ_flat)
            t2 = time.time()
            print("Took t = ", t2 - t1, " s to perform the total grid cell elimination process.")
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
            self.XYZ_flat = np.array([ox, oy, oz]).T
            self.N_grid = self.XYZ_flat.shape[0]
        self.N_grid = self.XYZ_flat.shape[0]
        self.alphas = np.random.rand(self.N_grid, self.n_functions)

        # Initialize Phi
        self._setup_polynomial_basis(nx=nx, ny=ny, nz=nz)

        # Initialize the factors for optimization
        self._construct_geo_factor()

    def _setup_uniform_grid(self):
        """
            Initializes a uniform grid in cartesian coordinates and sets
            some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the three boundaries
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

        # get normal vectors of inner and outer PM surfaces
        self.normal_inner = np.copy(self.rz_inner_surface.unitnormal())
        self.normal_outer = np.copy(self.rz_outer_surface.unitnormal())

        x_max = np.max(self.x_outer)
        x_min = np.min(self.x_outer)
        y_max = np.max(self.y_outer)
        y_min = np.min(self.y_outer)
        z_max = np.max(self.z_outer)
        z_min = np.min(self.z_outer)

        # Initialize uniform grid
        dx = self.dx
        dy = self.dy
        dz = self.dz
        Nx = int((x_max - x_min) / dx)
        Ny = int((y_max - y_min) / dy)
        Nz = int((z_max - z_min) / dz)
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        X = np.linspace(x_min, x_max, Nx)
        Y = np.linspace(y_min, y_max, Ny)
        Z = np.linspace(z_min, z_max, Nz)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.XYZ_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3) 

    def _set_inner_rz_surface(self):
        """
            If the inner toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the
            plasma boundary and shifts it by self.plasma_offset at constant
            theta value. Note that, for now, this makes the entire
            coil surface, not just 1 / 2 nfp of the surface.
        """
        range_surf = "full torus"
        nphi = self.nphi * 2 * self.plasma_boundary.nfp
        ntheta = self.ntheta
        f = self.filename
        if self.surface_flag == 'focus':
            rz_inner_surface = SurfaceRZFourier.from_focus(f, range=range_surf, nphi=nphi, ntheta=ntheta)
        elif self.surface_flag == 'wout':
            rz_inner_surface = SurfaceRZFourier.from_wout(f, range=range_surf, nphi=nphi, ntheta=ntheta)
        else:
            rz_inner_surface = SurfaceRZFourier.from_vmec_input(f, range=range_surf, nphi=nphi, ntheta=ntheta)

        # extend via the normal vector
        rz_inner_surface.extend_via_normal(self.plasma_offset)
        self.rz_inner_surface = rz_inner_surface

    def _set_outer_rz_surface(self):
        """
            If the outer toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the
            inner toroidal surface and shifts it by self.coil_offset at constant
            theta value.
        """
        range_surf = "full torus"
        nphi = self.nphi * 2 * self.plasma_boundary.nfp
        ntheta = self.ntheta
        f = self.filename
        if self.surface_flag == 'focus':
            rz_outer_surface = SurfaceRZFourier.from_focus(f, range=range_surf, nphi=nphi, ntheta=ntheta)
        elif self.surface_flag == 'wout':
            rz_outer_surface = SurfaceRZFourier.from_wout(f, range=range_surf, nphi=nphi, ntheta=ntheta)
        else:
            rz_outer_surface = SurfaceRZFourier.from_vmec_input(f, range=range_surf, nphi=nphi, ntheta=ntheta)

        # extend via the normal vector
        t1 = time.time()
        rz_outer_surface.extend_via_normal(self.plasma_offset + self.coil_offset)
        t2 = time.time()
        print("Outer surface extension took t = ", t2 - t1)
        self.rz_outer_surface = rz_outer_surface

    def _toVTK(self, vtkname):
        """
            Write dipole data into a VTK file (stolen from Caoxiang's CoilPy code).

        Args:
            vtkname (str): VTK filename, will be appended with .vts or .vtu.
            dim (tuple, optional): Dimension information if saved as structured grids. Defaults to (1).
        """

        # nfp = self.plasma_boundary.nfp
        # stellsym = self.plasma_boundary.stellsym
        n = self.N_grid 
        # if stellsym:
        #     stell_list = [1, -1]
        #     nsym = nfp * 2
        # else:
        #     stell_list = [1]
        #     nsym = nfp

        # get the coordinates
        ox = self.XYZ_flat[:, 0]
        oy = self.XYZ_flat[:, 1]
        oz = self.XYZ_flat[:, 2]

        Phi = self.Phi
        n_interp = Phi.shape[2] * Phi.shape[3] * Phi.shape[4]
        Jvec = np.zeros((n, n_interp, 3))
        Phi = Phi.reshape(self.n_functions, n, n_interp, 3)
        # Compute Jvec as average J over the integration points in a cell
        for i in range(3):
            for j in range(n_interp):
                Jvec[:, j, i] = np.sum(self.alphas.T * Phi[:, :, j, i], axis=0)
        Jvec = np.mean(Jvec, axis=-1)
        Jx = Jvec[:, 0]
        Jy = Jvec[:, 1]
        Jz = Jvec[:, 2]
        Jvec_normalization = 1.0 / np.sum(Jvec ** 2, axis=-1)

        # Save all the data to a vtk file which can be visualized nicely with ParaView
        contig = np.ascontiguousarray
        data = {"J": (contig(Jx), contig(Jy), contig(Jz)), 
                "J_normalized": 
                (contig(Jx / Jvec_normalization), 
                 contig(Jy / Jvec_normalization), 
                 contig(Jz / Jvec_normalization))
                } 
        pointsToVTK(
            vtkname, contig(ox), contig(oy), contig(oz), data=data
            #vtkname, 
            #contig(self.XYZ_flat[:, 0]),
            #contig(self.XYZ_flat[:, 1]),
            #contig(self.XYZ_flat[:, 2])
        )
        pointsToVTK(
            vtkname + '_uniform', 
            contig(self.XYZ_uniform[:, 0]),
            contig(self.XYZ_uniform[:, 1]),
            contig(self.XYZ_uniform[:, 2])
        )

    def _setup_polynomial_basis(self, nx=1, ny=1, nz=1):
        """
        Evaluate the basis of divergence-free polynomials
        at a given set of points. For now,
        the basis is hard-coded as a set of linear polynomials.

        returns:
            poly_basis, shape (num_basis_functions, N_grid, nx * ny * nz, 3)
                where N_grid is the number of cells in the winding volume,
                nx, ny, and nz, are the number of points to evaluate the 
                function WITHIN a cell, and num_basis_functions is hard-coded
                to 11 for now while we use a linear basis of polynomials. 
        """
        dx = self.dx
        dy = self.dy
        dz = self.dz
        n = self.N_grid
        x_leftpoints = self.XYZ_flat[:, 0]
        y_leftpoints = self.XYZ_flat[:, 1]
        z_leftpoints = self.XYZ_flat[:, 2]
        xrange = np.zeros((n, nx))
        yrange = np.zeros((n, ny))
        zrange = np.zeros((n, nz))
        for i in range(n):
            xrange[i, :] = np.linspace(
                x_leftpoints[i], 
                x_leftpoints[i] + dx,
                nx
            )
            yrange[i, :] = np.linspace(
                y_leftpoints[i], 
                y_leftpoints[i] + dy,
                ny
            )
            zrange[i, :] = np.linspace(
                z_leftpoints[i], 
                z_leftpoints[i] + dz,
                nz
            )
        Phi = np.zeros((self.n_functions, n, nx, ny, nz, 3)) 
        zeros = np.zeros(n)
        ones = np.ones(n)
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    Phi[0, :, i, j, k, :] = np.array([ones, zeros, zeros]).T
                    Phi[1, :, i, j, k, :] = np.array([zeros, ones, zeros]).T
                    Phi[2, :, i, j, k, :] = np.array([zeros, zeros, ones]).T
                    Phi[3, :, i, j, k, :] = np.array([xrange[:, i], zeros, zeros]).T
                    Phi[4, :, i, j, k, :] = np.array([yrange[:, j], zeros, zeros]).T
                    Phi[5, :, i, j, k, :] = np.array([zeros, zeros, xrange[:, i]]).T
                    Phi[6, :, i, j, k, :] = np.array([zeros, zeros, yrange[:, j]]).T
                    Phi[7, :, i, j, k, :] = np.array([zeros, zrange[:, k], zeros]).T
                    Phi[8, :, i, j, k, :] = np.array([zeros, xrange[:, i], zeros]).T
                    Phi[9, :, i, j, k, :] = np.array([xrange[:, i], -yrange[:, j], zeros]).T
                    Phi[10, :, i, j, k, :] = np.array([xrange[:, i], zeros, -zrange[:, k]]).T
        self.Phi = Phi

        # build up array of the integration points
        XYZ_integration = np.zeros((n, nx * ny * nz, 3))
        for i in range(n):
            X_n, Y_n, Z_n = np.meshgrid(
                xrange[i, :], yrange[i, :], zrange[i, :], 
                indexing='ij'
            )
            XYZ_integration[i, :, :] = np.transpose(np.array([X_n, Y_n, Z_n]), [1, 2, 3, 0]).reshape(nx * ny * nz, 3) 
        self.XYZ_integration = XYZ_integration

        return Phi

    def _construct_geo_factor(self):
        contig = np.ascontiguousarray

        # Compute Bnormal factor of the optimization problem
        points = contig(self.plasma_boundary.gamma().reshape(-1, 3))
        plasma_unitnormal = contig(self.plasma_boundary.unitnormal().reshape(-1, 3))
        coil_points = contig(self.XYZ_flat)
        integration_points = contig(self.XYZ_integration)
        print(points.shape, coil_points.shape, integration_points.shape, plasma_unitnormal.shape, self.Phi.shape)
        self.geo_factor = sopp.winding_volume_geo_factors(
            points, 
            integration_points, 
            plasma_unitnormal, 
            contig(self.Phi)
        )
        dphi = self.plasma_boundary.quadpoints_phi[1]
        dtheta = self.plasma_boundary.quadpoints_theta[1]
        plasma_normal = self.plasma_boundary.normal().reshape(-1, 3)
        normN = np.linalg.norm(plasma_normal, ord=2, axis=-1)
        B_matrix = (self.geo_factor * 1e-7 * np.sqrt(dphi * dtheta) * self.dx * self.dy * self.dz).reshape(self.geo_factor.shape[0], self.N_grid * self.n_functions)
        b_rhs = np.ravel(self.Bn * np.sqrt(dphi * dtheta))
        for i in range(B_matrix.shape[0]):
            B_matrix[i, :] *= np.sqrt(normN[i])
            b_rhs[i] *= np.sqrt(normN[i])

        self.B_matrix = B_matrix
        self.b_rhs = -b_rhs  # minus sign because ||B_{coil,N} + B_{0,N}||_2^2 -> ||A * alpha - b||_2^2

        curve_dl = self.Itarget_curve.gammadash().reshape(-1, 3)
        dphi_loop = np.sqrt(self.plasma_boundary.quadpoints_phi[1])
        self.Itarget_matrix = sopp.winding_volume_geo_factors(
            points, 
            integration_points, 
            contig(curve_dl), 
            contig(self.Phi)
        ).reshape(B_matrix.shape[0], self.N_grid * self.n_functions) * dphi_loop
        self.Itarget_rhs = self.Bn_Itarget * dphi_loop
        self.Itarget_matrix = np.sum(self.Itarget_matrix, axis=0)
        self.Itarget_rhs = np.sum(self.Itarget_rhs) + 4 * np.pi * 1e-7 * self.Itarget

        nx = self.Phi.shape[2]
        ny = self.Phi.shape[3]
        nz = self.Phi.shape[4]
        flux_factor, self.connection_list = sopp.winding_volume_flux_jumps(
            coil_points, 
            self.Phi, 
            self.dx, 
            self.dy, 
            self.dz
        )
        # Flux matrix is so large we need to use numpy sparse matrices here
        t1 = time.time()
        num_basis = self.n_functions
        flux_constraint_matrix = lil_matrix((6 * self.N_grid, self.N_grid * num_basis))
        for i in range(self.N_grid):
            flux_constraint_matrix[i * 6:(i + 1) * 6, i * num_basis:(i + 1) * num_basis] = flux_factor[:, i, :]
            for kk in range(6):
                for j in range(6):  # loop over the 6 neighbors
                    q = self.connection_list[i, j, kk]
                    if (q > 0):
                        flux_constraint_matrix[i * 6 + kk, q * num_basis:(q + 1) * num_basis] = flux_factor[kk, q, :]
        # Once matrix elements are set, convert to CSC for quicker matrix ops
        self.flux_constraint_matrix = flux_constraint_matrix.tocsc()
        t2 = time.time()
        print('Time to make the flux jump constraint matrix = ', t2 - t1, ' s')

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
        d["dx"] = self.dx
        d["dy"] = self.dy
        d["dz"] = self.dz
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
            d["dx"],
            d["dy"],
            d["dz"],
            d["filename"],
            d["surface_flag"],
        )
