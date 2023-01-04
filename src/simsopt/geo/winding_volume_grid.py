import numpy as np
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from pyevtk.hl import pointsToVTK
import simsoptpp as sopp
import time
import warnings

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
            Bn:               Magnetic field (coils and plasma) at the plasma
                              boundary. Typically this will be the optimized plasma
                              magnetic field from a stage-1 optimization, and the
                              optimized coils from a basic stage-2 optimization.
            dx:               X-axis grid spacing in the winding volume grid. 
            dy:               Y-axis grid spacing in the winding volume grid. 
            dz:               Z-axis grid spacing in the winding volume grid. 
            filename:         Filename for the file containing the plasma boundary surface.
            surface_flag:     Flag to specify the format of the surface file. Defaults to VMEC.
    """

    def __init__(
        self, plasma_boundary,
        rz_inner_surface=None,
        rz_outer_surface=None, plasma_offset=0.1,
        coil_offset=0.2, Bn=None, dx=0.02, dy=0.02, dz=0.02,
        filename=None, surface_flag='vmec',
        famus_filename=None,
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
        self.dx = dx
        self.dy = dy
        self.dz = dz

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

            RPhiZ_grid = self._setup_uniform_grid()

            # Have the uniform grid, now need to loop through and eliminate cells.
            t1 = time.time()

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
        Xgrid = self.final_RZ_grid[:, 0] * np.cos(self.final_RZ_grid[:, 1])
        Ygrid = self.final_RZ_grid[:, 0] * np.sin(self.final_RZ_grid[:, 1])
        Zgrid = self.final_RZ_grid[:, 2]
        self.XYZ_flat = np.array([Xgrid, Ygrid, Zgrid]).T
        self.N_grid = self.XYZ_flat.shape[0]
        # self.XYZ_flat = self.XYZ.reshape(-1, 3)

    def _setup_uniform_grid(self):
        """
            Initializes a uniform grid in cartesian coordinates and sets
            some important grid variables for later.
        """
        # Get (X, Y, Z) coordinates of the three boundaries
        xyz_inner = self.rz_inner_surface.gamma()
        self.r_inner = np.sqrt(xyz_inner[:, :, 0] ** 2 + xyz_inner[:, :, 1] ** 2)
        self.x_inner = xyz_inner[:, :, 0]
        self.y_inner = xyz_inner[:, :, 1]
        self.z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.rz_outer_surface.gamma()
        self.r_outer = np.sqrt(xyz_outer[:, :, 0] ** 2 + xyz_outer[:, :, 1] ** 2)
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
        # self.XYZ = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0])
        # self.XYZ_flat = self.XYZ.reshape(-1, 3)

        # Make 3D mesh in (R, phi, Z)
        Phi = np.arctan2(Y, X)
        R = np.sqrt(X ** 2 + Y ** 2)
        self.RPhiZ = np.transpose(np.array([R, Phi, Z]), [1, 2, 3, 0])
        RPhiZ_grid = np.transpose(self.RPhiZ, [0, 2, 1, 3])
        RPhiZ_grid = RPhiZ_grid.reshape(RPhiZ_grid.shape[0] * RPhiZ_grid.shape[1], RPhiZ_grid.shape[2], 3)
        return RPhiZ_grid

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

    def _toVTK(self, vtkname):
        """
            Write dipole data into a VTK file (stolen from Caoxiang's CoilPy code).

        Args:
            vtkname (str): VTK filename, will be appended with .vts or .vtu.
            dim (tuple, optional): Dimension information if saved as structured grids. Defaults to (1).
        """

        nfp = self.plasma_boundary.nfp
        stellsym = self.plasma_boundary.stellsym
        n = self.N_grid 
        if stellsym:
            stell_list = [1, -1]
            nsym = nfp * 2
        else:
            stell_list = [1]
            nsym = nfp

        # get the coordinates
        ox = np.ascontiguousarray(self.XYZ_flat[:, 0])
        oy = np.ascontiguousarray(self.XYZ_flat[:, 1])
        oz = np.ascontiguousarray(self.XYZ_flat[:, 2])

        # Initialize full grid variables
        ox_full = np.zeros(n * nsym)
        oy_full = np.zeros(n * nsym)
        oz_full = np.zeros(n * nsym)

        index = 0
        # Loop over stellarator and field-period symmetry contributions
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                ox_full[index:index + n] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                oy_full[index:index + n] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                oz_full[index:index + n] = oz * stell
                index += n

        # Save all the data to a vtk file which can be visualized nicely with ParaView
        # data = {"m": (mx, my, mz), "m_normalized": (mx_normalized, my_normalized, mz_normalized), "m_rphiz": (mr, mphi, mz), "m_rphiz_normalized": (mr_normalized, mphi_normalized, mz_normalized), "m_rphitheta": (mrminor, mphi, mtheta), "m_rphitheta_normalized": (mrminor_normalized, mphi_normalized, mtheta_normalized)}
        pointsToVTK(
            vtkname, ox_full, oy_full, oz_full,  # data=data
        )

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
