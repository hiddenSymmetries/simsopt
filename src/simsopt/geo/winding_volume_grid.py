import numpy as np
from simsopt.geo import Surface, SurfaceRZFourier
from pyevtk.hl import pointsToVTK, unstructuredGridToVTK
from pyevtk.vtk import VtkVoxel

import simsoptpp as sopp
import time
import warnings
from scipy.sparse import lil_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import vstack, hstack
from scipy.sparse import eye as sparse_eye
from scipy.sparse.linalg import inv as sparse_inv
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from .._core.json import GSONDecoder, GSONable
# from .._core.optimizable import Optimizable

__all__ = ['WindingVolumeGrid']

# class WindingVolumeGrid(Optimizable):


def _voxels_to_vtk(
    filename,
    points,
    cellData=None,
    pointData=None,
    fieldData=None,
):
    """
    Write a VTK file showing the voxels.

    Args:
        filename: Name of the file to write.
        points: Array of size ``(nvoxels, nquadpoints, 3)``
            Here, ``nvoxels`` is the number of voxels.
            The last array dimension corresponds to Cartesian coordinates.
            The max and min over the ``nquadpoints`` dimension will be used to
            define the max and min coordinates of each voxel.
        cellData: Data for each voxel to pass to ``pyevtk.hl.unstructuredGridToVTK``.
        pointData: Data for each voxel's vertices to pass to ``pyevtk.hl.unstructuredGridToVTK``.
        fieldData: Data for each voxel to pass to ``pyevtk.hl.unstructuredGridToVTK``.
    """

    # Some references I used while writing this function:
    # https://vtk.org/doc/nightly/html/classvtkVoxel.html
    # https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Testing/Baseline/Cxx/GeometricObjects/TestLinearCellDemo.png
    # https://github.com/pyscience-projects/pyevtk/blob/v1.2.0/pyevtk/hl.py
    # https://python.hotexamples.com/examples/vtk/-/vtkVoxel/python-vtkvoxel-function-examples.html

    assert points.ndim == 3
    nvoxels = points.shape[0]
    assert points.shape[2] == 3

    cell_types = np.empty(nvoxels, dtype="uint8")
    cell_types[:] = VtkVoxel.tid
    connectivity = np.arange(8 * nvoxels, dtype=np.int64)
    offsets = (np.arange(nvoxels, dtype=np.int64) + 1) * 8

    base_x = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    base_y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    base_z = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    x = np.zeros(8 * nvoxels)
    y = np.zeros(8 * nvoxels)
    z = np.zeros(8 * nvoxels)

    for j in range(nvoxels):
        x[8 * j: 8 * (j + 1)] = (
            np.min(points[j, :, 0])
            + (np.max(points[j, :, 0]) - np.min(points[j, :, 0])) * base_x
        )
        y[8 * j: 8 * (j + 1)] = (
            np.min(points[j, :, 1])
            + (np.max(points[j, :, 1]) - np.min(points[j, :, 1])) * base_y
        )
        z[8 * j: 8 * (j + 1)] = (
            np.min(points[j, :, 2])
            + (np.max(points[j, :, 2]) - np.min(points[j, :, 2])) * base_z
        )

    unstructuredGridToVTK(
        filename,
        x,
        y,
        z,
        connectivity,
        offsets,
        cell_types,
        cellData=cellData,
        pointData=pointData,
        fieldData=fieldData,
    )


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
            Nx:               Number of X points in winding volume grid. 
            Ny:               Number of Y points in winding volume grid. 
            Ny:               Number of Z points in winding volume grid. 
            filename:         Filename for the file containing the plasma boundary surface.
            surface_flag:     Flag to specify the format of the surface file. Defaults to VMEC.
            OUT_DIR:          Directory to save files in.
    """

    def __init__(
        self, plasma_boundary, Itarget_curve, Bn_Itarget, Itarget=1e6,
        rz_inner_surface=None,
        rz_outer_surface=None, plasma_offset=0.1,
        coil_offset=0.2, Bn=None,
        Nx=11, Ny=11, Nz=11,
        filename=None, surface_flag='vmec',
        famus_filename=None, 
        coil_range='half period',
        OUT_DIR='', nx=2, ny=2, nz=2,
        sparse_constraint_matrix=True,
    ):
        if plasma_offset <= 0 or coil_offset <= 0:
            raise ValueError('permanent magnets must be offset from the plasma')

        if famus_filename is not None:
            warnings.warn(
                'famus_filename variable is set, so a pre-defined grid will be used. '
                ' so that the following parameters are ignored: '
                'rz_inner_surface, rz_outer_surface, Nx, Ny, Nz, plasma_offset, '
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
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.OUT_DIR = OUT_DIR
        self.n_functions = 11  # hard-coded for linear basis
        self.sparse_constraint_matrix = sparse_constraint_matrix

        if not isinstance(plasma_boundary, SurfaceRZFourier):
            raise ValueError(
                'Plasma boundary must be specified as SurfaceRZFourier class.'
            )
        else:
            self.plasma_boundary = plasma_boundary
            self.coil_range = coil_range
            self.R0 = plasma_boundary.get_rc(0, 0)
            # unlike plasma surface, use the whole phi and theta grids
            self.phi = self.plasma_boundary.quadpoints_phi
            self.nphi = len(self.phi)
            self.theta = self.plasma_boundary.quadpoints_theta
            self.ntheta = len(self.theta)

        if Nx <= 0 or Ny <= 0 or Nz <= 0:
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

            t1 = time.time()
            self._setup_uniform_grid()
            t2 = time.time()
            print("Took t = ", t2 - t1, " s to setup the uniform grid.")

            # Have the uniform grid, now need to loop through and eliminate cells.
            t1 = time.time()

            contig = np.ascontiguousarray
            print(np.shape(self.XYZ_uniform))
            if self.surface_flag == 'focus' or self.surface_flag == 'wout':
                final_grid = sopp.make_winding_volume_grid(
                    #self.plasma_boundary.unitnormal().reshape(-1, 3), self.plasma_boundary.unitnormal().reshape(-1, 3),
                    contig(self.normal_inner.reshape(-1, 3)), contig(self.normal_outer.reshape(-1, 3)),
                    contig(self.XYZ_uniform), contig(self.xyz_inner), contig(self.xyz_outer),
                )
            else:
                final_grid = sopp.make_winding_volume_grid(
                    #self.plasma_boundary.unitnormal().reshape(-1, 3), self.plasma_boundary.unitnormal().reshape(-1, 3),
                    contig(self.normal_inner.reshape(-1, 3)), contig(self.normal_outer.reshape(-1, 3)),
                    contig(self.XYZ_uniform), contig(self.xyz_inner), contig(self.xyz_outer),
                )
            # remove all the grid elements that were omitted
            inds = np.ravel(np.logical_not(np.all(final_grid == 0.0, axis=-1)))
            self.XYZ_flat = final_grid[inds, :]
            print(self.XYZ_flat.shape)
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
        self.alphas = np.random.rand(self.n_functions * self.N_grid)
        self.nx = nx
        self.ny = ny
        self.nz = nz

        # Initialize Phi
        t1 = time.time()
        self._setup_polynomial_basis()
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to setup Phi.")

        # Initialize the factors for optimization
        t1 = time.time()
        self._construct_geo_factor()
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to setup geo factor.")

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
        z_max = max(z_max, abs(z_min))

        # Make grid uniform in (X, Y)
        min_xy = min(x_min, y_min)
        #min_xy = max(x_min, y_min)
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
        if self.coil_range != 'full torus':
            X = np.linspace(self.dx / 2.0, (x_max - x_min) + self.dx / 2.0, Nx, endpoint=True)
            Y = np.linspace(self.dy / 2.0, (y_max - y_min) + self.dy / 2.0, Ny, endpoint=True)
        else:
            X = np.linspace(x_min, x_max, Nx, endpoint=True)
            Y = np.linspace(y_min, y_max, Ny, endpoint=True)
        Z = np.linspace(-z_max, z_max, Nz, endpoint=True)
        print(Nx, Ny, Nz, X, Y, Z, self.dx, self.dy, self.dz)

        # Make 3D mesh
        X, Y, Z = np.meshgrid(X, Y, Z, indexing='ij')
        self.XYZ_uniform = np.transpose(np.array([X, Y, Z]), [1, 2, 3, 0]).reshape(Nx * Ny * Nz, 3)

        # Extra work for nfp = 4 to chop off half of the originally nfp = 2 uniform grid
        if self.plasma_boundary.nfp == 4:
            inds = []
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        if X[i, j, k] < Y[i, j, k]:
                            inds.append(int(i * Ny * Nz + j * Nz + k))
            good_inds = np.setdiff1d(np.arange(Nx * Ny * Nz), inds) 
            self.XYZ_uniform = self.XYZ_uniform[good_inds, :]

    def _set_inner_rz_surface(self):
        """
            If the inner toroidal surface was not specified, this function
            is called and simply takes each (r, z) quadrature point on the
            plasma boundary and shifts it by self.plasma_offset at constant
            theta value. Note that, for now, this makes the entire
            coil surface, not just 1 / 2 nfp of the surface.
        """
        range_surf = self.coil_range
        nphi = self.nphi
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
        range_surf = self.coil_range
        nphi = self.nphi
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

    def to_vtk_before_solve(self, vtkname):
        """
            Write voxel geometry data into a VTK file.

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
        ox = self.XYZ_flat[:, 0]
        oy = self.XYZ_flat[:, 1]
        oz = self.XYZ_flat[:, 2]

        # Initialize new grid and current vectors for all the dipoles
        # after we account for the symmetries below.
        ox_full = np.zeros(n * nsym)
        oy_full = np.zeros(n * nsym)
        oz_full = np.zeros(n * nsym)
        XYZ_integration_full = np.zeros((n * nsym, self.nx * self.ny * self.nz, 3))

        # loop through the dipoles and repeat for fp and stellarator symmetries
        index = 0

        # Loop over stellarator and field-period symmetry contributions
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                ox_full[index:index + n] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                oy_full[index:index + n] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                oz_full[index:index + n] = oz * stell
                XYZ_integration_full[index:index + n, :, 0] = self.XYZ_integration[:, :, 0] * np.cos(phi0) - self.XYZ_integration[:, :, 1] * np.sin(phi0) * stell
                XYZ_integration_full[index:index + n, :, 1] = self.XYZ_integration[:, :, 0] * np.sin(phi0) + self.XYZ_integration[:, :, 1] * np.cos(phi0) * stell
                XYZ_integration_full[index:index + n, :, 2] = self.XYZ_integration[:, :, 2] * stell

                index += n

        contig = np.ascontiguousarray
        pointsToVTK(
            vtkname, contig(ox_full), contig(oy_full), contig(oz_full)
        )
        pointsToVTK(
            vtkname + '_uniform',
            contig(self.XYZ_uniform[:, 0]),
            contig(self.XYZ_uniform[:, 1]),
            contig(self.XYZ_uniform[:, 2])
        )
        Phi = self.Phi.reshape(self.n_functions, self.N_grid, self.nx * self.ny * self.nz, 3)
        data = {
            "Phi0": (contig(Phi[0, :, :, 0].flatten()), contig(Phi[0, :, :, 1].flatten()), contig(Phi[0, :, :, 2].flatten())),
            "Phi1": (contig(Phi[1, :, :, 0].flatten()), contig(Phi[1, :, :, 1].flatten()), contig(Phi[1, :, :, 2].flatten())),
            "Phi2": (contig(Phi[2, :, :, 0].flatten()), contig(Phi[2, :, :, 1].flatten()), contig(Phi[2, :, :, 2].flatten())),
            "Phi3": (contig(Phi[3, :, :, 0].flatten()), contig(Phi[3, :, :, 1].flatten()), contig(Phi[3, :, :, 2].flatten())),
            "Phi4": (contig(Phi[4, :, :, 0].flatten()), contig(Phi[4, :, :, 1].flatten()), contig(Phi[4, :, :, 2].flatten())),
            "Phi5": (contig(Phi[5, :, :, 0].flatten()), contig(Phi[5, :, :, 1].flatten()), contig(Phi[5, :, :, 2].flatten())),
            "Phi6": (contig(Phi[6, :, :, 0].flatten()), contig(Phi[6, :, :, 1].flatten()), contig(Phi[6, :, :, 2].flatten())),
            "Phi7": (contig(Phi[7, :, :, 0].flatten()), contig(Phi[7, :, :, 1].flatten()), contig(Phi[7, :, :, 2].flatten())),
            "Phi8": (contig(Phi[8, :, :, 0].flatten()), contig(Phi[8, :, :, 1].flatten()), contig(Phi[8, :, :, 2].flatten())),
            "Phi9": (contig(Phi[9, :, :, 0].flatten()), contig(Phi[9, :, :, 1].flatten()), contig(Phi[9, :, :, 2].flatten())),
            "Phi10": (contig(Phi[10, :, :, 0].flatten()), contig(Phi[10, :, :, 1].flatten()), contig(Phi[10, :, :, 2].flatten()))
        }
        pointsToVTK(
            vtkname + '_quadrature',
            contig(self.XYZ_integration[:, :, 0].flatten()),
            contig(self.XYZ_integration[:, :, 1].flatten()),
            contig(self.XYZ_integration[:, :, 2].flatten()),
            data=data
        )
        _voxels_to_vtk(vtkname + '_voxels', self.XYZ_integration)
        _voxels_to_vtk(vtkname + '_voxels_full', XYZ_integration_full)
        print("Max quadrature x:", np.max(self.XYZ_integration[:, :, 0]))
        print("Min quadrature x:", np.min(self.XYZ_integration[:, :, 0]))
        print("Max quadrature z:", np.max(self.XYZ_integration[:, :, 2]))
        print("Min quadrature z:", np.min(self.XYZ_integration[:, :, 2]))

    def to_vtk_after_solve(self, vtkname):
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
        ox = self.XYZ_flat[:, 0]
        oy = self.XYZ_flat[:, 1]
        oz = self.XYZ_flat[:, 2]

        Phi = self.Phi
        n_interp = Phi.shape[2]
        Jvec = self.J
        Jvec_sparse = self.J_sparse
        alphas = self.alphas.reshape(n, self.n_functions).T
        Jvec_avg = np.mean(Jvec, axis=1)
        Jvec_sparse_avg = np.mean(Jvec_sparse, axis=1)
        Jvec = Jvec.reshape(n, self.nx, self.ny, self.nz, 3)
        Jvec_sparse = Jvec_sparse.reshape(n, self.nx, self.ny, self.nz, 3)

        dJx_dx = -(Jvec[:, 1:, :, :, 0] - Jvec[:, :-1, :, :, 0]) / self.dx
        dJy_dy = -(Jvec[:, :, 1:, :, 1] - Jvec[:, :, :-1, :, 1]) / self.dy
        dJz_dz = -(Jvec[:, :, :, 1:, 2] - Jvec[:, :, :, :-1, 2]) / self.dz
        divJ = dJx_dx[:, :, :-1, :-1] + dJy_dy[:, :-1, :, :-1] + dJz_dz[:, :-1, :-1, :]
        divJ = np.sum(np.sum(np.sum(divJ, axis=1), axis=1), axis=1)

        Jx = Jvec_avg[:, 0]
        Jy = Jvec_avg[:, 1]
        Jz = Jvec_avg[:, 2]

        Jx_sp = Jvec_sparse_avg[:, 0]
        Jy_sp = Jvec_sparse_avg[:, 1]
        Jz_sp = Jvec_sparse_avg[:, 2]

        # Initialize new grid and current vectors for all the dipoles
        # after we account for the symmetries below.
        ox_full = np.zeros(n * nsym)
        oy_full = np.zeros(n * nsym)
        oz_full = np.zeros(n * nsym)
        Jvec_full = np.zeros((n * nsym, 3))
        Jvec_full_sp = np.zeros((n * nsym, 3))
        Jvec_full_internal = np.zeros((n * nsym, self.nx, self.ny, self.nz, 3))

        # loop through the dipoles and repeat for fp and stellarator symmetries
        index = 0

        # Loop over stellarator and field-period symmetry contributions
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                ox_full[index:index + n] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                oy_full[index:index + n] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                oz_full[index:index + n] = oz * stell

                # get new dipole vectors by flipping the x component, then rotating by phi0
                Jvec_full_internal[index:index + n, :, :, :, 0] = Jvec[:, :, :, :, 0] * np.cos(phi0) * stell - Jvec[:, :, :, :, 1] * np.sin(phi0)
                Jvec_full_internal[index:index + n, :, :, :, 1] = Jvec[:, :, :, :, 0] * np.sin(phi0) * stell + Jvec[:, :, :, :, 1] * np.cos(phi0)
                Jvec_full_internal[index:index + n, :, :, :, 2] = Jvec[:, :, :, :, 2]

                Jvec_full[index:index + n, 0] = Jx * np.cos(phi0) * stell - Jy * np.sin(phi0)
                Jvec_full[index:index + n, 1] = Jx * np.sin(phi0) * stell + Jy * np.cos(phi0)
                Jvec_full[index:index + n, 2] = Jz

                Jvec_full_sp[index:index + n, 0] = Jx_sp * np.cos(phi0) * stell - Jy_sp * np.sin(phi0)
                Jvec_full_sp[index:index + n, 1] = Jx_sp * np.sin(phi0) * stell + Jy_sp * np.cos(phi0)
                Jvec_full_sp[index:index + n, 2] = Jz_sp

                index += n

        Jvec_normalization = 1.0 / np.sum(Jvec_full ** 2, axis=-1)
        # Jvec_normalization_sp = 1.0 / np.sum(Jvec_full_sp ** 2, axis=-1)

        # Save all the data to a vtk file which can be visualized nicely with ParaView
        Jx = Jvec_full[:, 0]
        Jy = Jvec_full[:, 1]
        Jz = Jvec_full[:, 2]

        Jx_sp = Jvec_full_sp[:, 0]
        Jy_sp = Jvec_full_sp[:, 1]
        Jz_sp = Jvec_full_sp[:, 2]

        Jvec_xmin = np.mean(np.mean(Jvec_full_internal[:, 0, :, :, :], axis=-2), axis=-2)
        Jvec_xmax = np.mean(np.mean(Jvec_full_internal[:, -1, :, :, :], axis=-2), axis=-2)
        Jvec_ymin = np.mean(np.mean(Jvec_full_internal[:, :, 0, :, :], axis=-2), axis=-2)
        Jvec_ymax = np.mean(np.mean(Jvec_full_internal[:, :, -1, :, :], axis=-2), axis=-2)
        Jvec_zmin = np.mean(np.mean(Jvec_full_internal[:, :, :, 0, :], axis=-2), axis=-2)
        Jvec_zmax = np.mean(np.mean(Jvec_full_internal[:, :, :, -1, :], axis=-2), axis=-2)

        contig = np.ascontiguousarray
        data = {"J": (contig(Jx), contig(Jy), contig(Jz)), 
                "J_normalized": 
                (contig(Jx / Jvec_normalization), 
                 contig(Jy / Jvec_normalization), 
                 contig(Jz / Jvec_normalization)),
                "J_sparse": (contig(Jx_sp), contig(Jy_sp), contig(Jz_sp)), 
                #"J_sparse_normalized": 
                #(contig(Jx_sp / Jvec_normalization_sp), 
                # contig(Jy_sp / Jvec_normalization_sp), 
                # contig(Jz_sp / Jvec_normalization_sp)),
                "Jvec_xmin": (contig(Jvec_xmin), contig(Jvec_xmin), contig(Jvec_xmin)), 
                "Jvec_xmax": (contig(Jvec_xmax), contig(Jvec_xmax), contig(Jvec_xmax)), 
                "Jvec_ymin": (contig(Jvec_ymin), contig(Jvec_ymin), contig(Jvec_ymin)), 
                "Jvec_ymax": (contig(Jvec_ymax), contig(Jvec_ymax), contig(Jvec_ymax)), 
                "Jvec_zmin": (contig(Jvec_zmin), contig(Jvec_zmin), contig(Jvec_zmin)), 
                "Jvec_zmax": (contig(Jvec_zmax), contig(Jvec_zmax), contig(Jvec_zmax)), 
                } 
        pointsToVTK(
            vtkname, contig(ox_full), contig(oy_full), contig(oz_full), data=data
        )
        data = {"divJ": (contig(divJ)), "num_constraints_per_cell": self.num_constraints_per_cell,
                "flux_factor0": np.sum(self.flux_factor[0, :, :] * alphas.T, axis=-1), 
                "flux_factor1": np.sum(self.flux_factor[1, :, :] * alphas.T, axis=-1), 
                "flux_factor2": np.sum(self.flux_factor[2, :, :] * alphas.T, axis=-1), 
                "flux_factor3": np.sum(self.flux_factor[3, :, :] * alphas.T, axis=-1), 
                "flux_factor4": np.sum(self.flux_factor[4, :, :] * alphas.T, axis=-1), 
                "flux_factor5": np.sum(self.flux_factor[5, :, :] * alphas.T, axis=-1),
                "index": np.arange(n)} 
        pointsToVTK(
            vtkname + '_divJ', contig(ox), contig(oy), contig(oz), data=data
        )
        pointsToVTK(
            vtkname + '_uniform', 
            contig(self.XYZ_uniform[:, 0]),
            contig(self.XYZ_uniform[:, 1]),
            contig(self.XYZ_uniform[:, 2])
        )

        fig = plt.figure(700, figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')

        # Make the grid
        colors = np.linalg.norm(Jvec_full_sp, axis=-1)
        inds = (colors > 1)
        # colors = colors / np.max(colors)
        #colors = plt.cm.jet(colors)
        colors = np.concatenate((colors, np.repeat(colors, 2))) 

        q = ax.quiver(ox_full[inds], oy_full[inds], oz_full[inds], Jvec_full_sp[inds, 0], Jvec_full_sp[inds, 1], Jvec_full_sp[inds, 2], length=0.4, normalize=True, cmap='Reds', norm=LogNorm(vmin=1, vmax=np.max(colors)))  # colors=colors[inds])
        inds = (colors > 1)
        q.set_array(colors[inds])
        fig.colorbar(q)
        plt.savefig(self.OUT_DIR + 'quiver_plot_sparse_solution.jpg')

    def _setup_polynomial_basis(self, shift=True):
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
        self.polynomial_shift = shift
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz
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
                nx,
                endpoint=True
            ) - dx / 2.0
            yrange[i, :] = np.linspace(
                y_leftpoints[i], 
                y_leftpoints[i] + dy,
                ny,
                endpoint=True
            ) - dy / 2.0
            zrange[i, :] = np.linspace(
                z_leftpoints[i], 
                z_leftpoints[i] + dz,
                nz,
                endpoint=True
            ) - dz / 2.0
        Phi = np.zeros((self.n_functions, n, nx, ny, nz, 3)) 
        zeros = np.zeros(n)
        ones = np.ones(n)
        if not shift:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        Phi[0, :, i, j, k, :] = np.array([ones, zeros, zeros]).T
                        Phi[1, :, i, j, k, :] = np.array([zeros, ones, zeros]).T
                        Phi[2, :, i, j, k, :] = np.array([zeros, zeros, ones]).T
                        Phi[3, :, i, j, k, :] = np.array([yrange[:, j], zeros, zeros]).T
                        Phi[4, :, i, j, k, :] = np.array([zrange[:, k], zeros, zeros]).T
                        Phi[5, :, i, j, k, :] = np.array([zeros, zeros, xrange[:, i]]).T
                        Phi[6, :, i, j, k, :] = np.array([zeros, zeros, yrange[:, j]]).T
                        Phi[7, :, i, j, k, :] = np.array([zeros, zrange[:, k], zeros]).T
                        Phi[8, :, i, j, k, :] = np.array([zeros, xrange[:, i], zeros]).T
                        Phi[9, :, i, j, k, :] = np.array([xrange[:, i], -yrange[:, j], zeros]).T
                        Phi[10, :, i, j, k, :] = np.array([xrange[:, i], zeros, -zrange[:, k]]).T
        # Shift all the basis functions but their midpoints in every cell, to sort of normalize them
        else:
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        Phi[0, :, i, j, k, :] = np.array([ones, zeros, zeros]).T
                        Phi[1, :, i, j, k, :] = np.array([zeros, ones, zeros]).T
                        Phi[2, :, i, j, k, :] = np.array([zeros, zeros, ones]).T
                        Phi[3, :, i, j, k, :] = np.array([yrange[:, j] - y_leftpoints, zeros, zeros]).T / np.cbrt(dx * dy * dz)
                        Phi[4, :, i, j, k, :] = np.array([zrange[:, k] - z_leftpoints, zeros, zeros]).T / np.cbrt(dx * dy * dz)
                        Phi[5, :, i, j, k, :] = np.array([zeros, zeros, xrange[:, i] - x_leftpoints]).T / np.cbrt(dx * dy * dz)
                        Phi[6, :, i, j, k, :] = np.array([zeros, zeros, yrange[:, j] - y_leftpoints]).T / np.cbrt(dx * dy * dz)
                        Phi[7, :, i, j, k, :] = np.array([zeros, zrange[:, k] - z_leftpoints, zeros]).T / np.cbrt(dx * dy * dz)
                        Phi[8, :, i, j, k, :] = np.array([zeros, xrange[:, i] - x_leftpoints, zeros]).T / np.cbrt(dx * dy * dz)
                        Phi[9, :, i, j, k, :] = np.array([xrange[:, i] - x_leftpoints, -yrange[:, j] + y_leftpoints, zeros]).T / np.cbrt(dx * dy * dz)
                        Phi[10, :, i, j, k, :] = np.array([xrange[:, i] - x_leftpoints, zeros, -zrange[:, k] + z_leftpoints]).T / np.cbrt(dx * dy * dz)

        for i in range(11):
            dJx_dx = -(Phi[i, :, 1:, :, :, 0] - Phi[i, :, :-1, :, :, 0]) / dx
            dJy_dy = -(Phi[i, :, :, 1:, :, 1] - Phi[i, :, :, :-1, :, 1]) / dy
            dJz_dz = -(Phi[i, :, :, :, 1:, 2] - Phi[i, :, :, :, :-1, 2]) / dz
            divJ = dJx_dx[:, :, :-1, :-1] + dJy_dy[:, :-1, :, :-1] + dJz_dz[:, :-1, :-1, :]
            divJ = np.sum(np.sum(np.sum(divJ, axis=1), axis=1), axis=1)
            assert np.allclose(divJ, 0.0)
        self.Phi = Phi.reshape(self.n_functions, n, nx * ny * nz, 3)

        # build up array of the integration points
        XYZ_integration = np.zeros((n, nx * ny * nz, 3))
        for i in range(n):
            X_n, Y_n, Z_n = np.meshgrid(
                xrange[i, :], yrange[i, :], zrange[i, :], 
                indexing='ij'
            )
            XYZ_integration[i, :, :] = np.transpose(np.array([X_n, Y_n, Z_n]), [1, 2, 3, 0]).reshape(nx * ny * nz, 3) 
        self.XYZ_integration = XYZ_integration

    def _construct_geo_factor(self):
        contig = np.ascontiguousarray

        # Compute Bnormal factor of the optimization problem
        points = contig(self.plasma_boundary.gamma().reshape(-1, 3))
        points_curve = contig(self.Itarget_curve.gamma().reshape(-1, 3))
        plasma_unitnormal = contig(self.plasma_boundary.unitnormal().reshape(-1, 3))
        coil_points = contig(self.XYZ_flat)
        integration_points = self.XYZ_integration

        # Begin computation of the coil fields with all the symmetries
        nfp = self.plasma_boundary.nfp
        stellsym = self.plasma_boundary.stellsym
        if self.coil_range == 'full torus':
            stell_list = [1]
            nfp = 1
            nsym = nfp
        elif stellsym:
            stell_list = [1, -1]
            nsym = nfp * 2

        # get the coordinates
        ox = integration_points[:, :, 0]
        oy = integration_points[:, :, 1]
        oz = integration_points[:, :, 2]

        Phi = self.Phi
        n = self.N_grid
        nphi = len(self.plasma_boundary.quadpoints_phi)
        ntheta = len(self.plasma_boundary.quadpoints_theta)
        N_quadrature_inv = 1.0 / (nphi * ntheta)
        plasma_normal = self.plasma_boundary.normal().reshape(-1, 3)
        normN = np.linalg.norm(plasma_normal, ord=2, axis=-1)
        num_basis = Phi.shape[0]
        n_interp = Phi.shape[2]
        nphi_loop_inv = 1.0 / len(self.Itarget_curve.quadpoints)
        curve_dl = self.Itarget_curve.gammadash().reshape(-1, 3)

        # Initialize new grid and current vectors for all the dipoles
        # after we account for the symmetries below.
        ox_full = np.zeros((n * nsym, n_interp))
        oy_full = np.zeros((n * nsym, n_interp))
        oz_full = np.zeros((n * nsym, n_interp))
        Phivec_full = np.zeros((num_basis, n * nsym, n_interp, 3))

        # loop through the dipoles and repeat for fp and stellarator symmetries
        index = 0
        contig = np.ascontiguousarray    

        # Loop over stellarator and field-period symmetry contributions
        t1 = time.time()
        num_points = points.shape[0]
        num_points_curve = points_curve.shape[0]
        geo_factor = np.zeros((num_points, n, num_basis))
        Itarget_matrix = np.zeros((num_points_curve, n, num_basis))
        for stell in stell_list:
            for fp in range(nfp):
                phi0 = (2 * np.pi / nfp) * fp

                # get new dipoles locations by flipping the y and z components, then rotating by phi0
                ox_full[index:index + n, :] = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                oy_full[index:index + n, :] = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                oz_full[index:index + n, :] = oz * stell

                # get new dipole vectors by flipping the x component, then rotating by phi0
                Phivec_full[:, index:index + n, :, 0] = Phi[:, :, :, 0] * np.cos(phi0) * stell - Phi[:, :, :, 1] * np.sin(phi0)
                Phivec_full[:, index:index + n, :, 1] = Phi[:, :, :, 0] * np.sin(phi0) * stell + Phi[:, :, :, 1] * np.cos(phi0)
                Phivec_full[:, index:index + n, :, 2] = Phi[:, :, :, 2]

                Phivec_transpose = np.transpose(Phivec_full[:, index:index + n, :, :], [1, 2, 0, 3])
                int_points = np.transpose(np.array([ox_full[index:index + n, :], oy_full[index:index + n, :], oz_full[index:index + n, :]]), [1, 2, 0])
                geo_factor += sopp.winding_volume_field_Bext_SIMD(
                    points, 
                    contig(int_points), 
                    #contig(Phivec_full[:, index:index + n, :, :])
                    contig(Phivec_transpose),
                    plasma_unitnormal
                )
                Itarget_matrix += sopp.winding_volume_field_Bext_SIMD(
                    points_curve, 
                    contig(int_points), 
                    #contig(Phivec_full[:, index:index + n, :, :])
                    contig(Phivec_transpose),
                    contig(curve_dl)
                )

                index += n

        self.geo_factor = geo_factor
        self.Itarget_matrix = Itarget_matrix
        t2 = time.time()
        print('Computing full coil surface grid took t = ', t2 - t1, ' s')
        t1 = time.time()
        Phivec_transpose = np.transpose(Phivec_full, [1, 2, 0, 3])
        int_points = np.transpose(np.array([ox_full, oy_full, oz_full]), [1, 2, 0])
        t2 = time.time()
        print('Computing B factor and I factor took t = ', t2 - t1, ' s')

        t1 = time.time()
        self.Phi_full = contig(Phivec_full)
        alphas = self.alphas.reshape(n, self.n_functions)
        self.J = np.zeros((n, n_interp, 3))
        for i in range(3):
            for j in range(n_interp):
                for k in range(self.n_functions):
                    self.J[:, j, i] += alphas[:, k] * Phi[k, :, j, i]
        self.integration_points_full = int_points  # contig(np.transpose(np.array([ox_full, oy_full, oz_full]), [1, 2, 0]))
        t2 = time.time()
        print('Computing J took t = ', t2 - t1, ' s')
        t1 = time.time()

        # Delta x from intra-cell integration is self.dx (uniform grid spacing) divided 
        # by self.nx (number of points within a cell)
        coil_integration_factor = self.dx * self.dy * self.dz / (self.nx * self.ny * self.nz)
        self.grid_scaling = coil_integration_factor
        B_matrix = (self.geo_factor * np.sqrt(N_quadrature_inv) * coil_integration_factor).reshape(
            self.geo_factor.shape[0], self.N_grid * self.n_functions
        )
        # print(B_matrix.shape, self.geo_factor.shape)
        b_rhs = np.ravel(self.Bn * np.sqrt(N_quadrature_inv))
        for i in range(B_matrix.shape[0]):
            B_matrix[i, :] *= np.sqrt(normN[i])
            b_rhs[i] *= np.sqrt(normN[i])

        self.B_matrix = B_matrix
        self.b_rhs = -b_rhs  # minus sign because ||B_{coil,N} + B_{0,N}||_2^2 -> ||A * alpha - b||_2^2
        self.Itarget_matrix = nphi_loop_inv * coil_integration_factor * self.Itarget_matrix.reshape(
            points_curve.shape[0], n * num_basis
        )
        self.Itarget_matrix = np.sum(self.Itarget_matrix, axis=0)
        self.Itarget_rhs = self.Bn_Itarget * nphi_loop_inv
        self.Itarget_rhs = np.sum(self.Itarget_rhs) + 4 * np.pi * 1e-7 * self.Itarget
        flux_factor, self.connection_list = sopp.winding_volume_flux_jumps(
            coil_points,
            contig(self.Phi.reshape(self.n_functions, self.N_grid, self.nx, self.ny, self.nz, 3)), 
            self.dx,
            self.dy, 
            self.dz
        )
        t2 = time.time()
        print('Computing the flux factors took t = ', t2 - t1, ' s')

        # need to normalize flux_factor by 1/nx ** 2 (assuming uniform grid)
        flux_factor *= 1 / (self.nx ** 2)
        t1 = time.time()

        # Find the coil boundary points at phi = pi / 2
        if nfp == 2:
            minus_x_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 1] < 0, axis=-1)))
            x0_indices = np.ravel(np.where(np.isclose(coil_points[:, 0], 0.0, atol=self.dx)))
            minus_x_indices = np.intersect1d(minus_x_indices, x0_indices)
        # Find the coil boundary points at phi = pi / 4
        elif nfp == 4:
            minus_x_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 1] < 0, axis=-1)))
            plus_y_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 2] < 0, axis=-1)))
            #x0_indices = np.ravel(np.where(np.isclose(coil_points[:, 0], 0.0, atol=self.dx)))
            minus_x_indices = np.intersect1d(minus_x_indices, plus_y_indices)
        z_flipped_inds_x = []
        x_inds = []
        for x_ind in minus_x_indices:
            ox = coil_points[x_ind, 0]
            oy = coil_points[x_ind, 1]
            oz = coil_points[x_ind, 2]
            z_flipped_point = np.array([ox, oy, -oz]).T
            z_flipped = np.ravel(np.where(np.all(np.isclose(coil_points, z_flipped_point), axis=-1)))
            if len(z_flipped) > 0:
                z_flipped_inds_x.append(z_flipped[0])
                x_inds.append(x_ind)
        self.x_inds = x_inds

        # Find the coil boundary points at phi = 0
        minus_y_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 3] < 0, axis=-1)))
        y0_indices = np.ravel(np.where(np.isclose(coil_points[:, 1], 0.0, atol=self.dy)))
        if len(y0_indices) == 0:
            y0_indices = np.ravel(np.where(np.isclose(coil_points[:, 1], 0.0, atol=self.dy * 2)))
        minus_y_indices = np.intersect1d(minus_y_indices, y0_indices)
        z_flipped_inds_y = []

        y_inds = []
        for y_ind in minus_y_indices:
            ox = coil_points[y_ind, 0]
            oy = coil_points[y_ind, 1]
            oz = coil_points[y_ind, 2]
            z_flipped_point = np.array([ox, oy, -oz]).T
            z_flipped = np.ravel(np.where(np.all(np.isclose(coil_points, z_flipped_point), axis=-1)))
            if len(z_flipped) > 0:
                z_flipped_inds_y.append(z_flipped[0])
                y_inds.append(y_ind)
        self.y_inds = y_inds 
        self.z_flip_x = z_flipped_inds_x
        self.z_flip_y = z_flipped_inds_y

        # count the number of constraints
        n_constraints = 0
        for i in range(self.N_grid):            
            # Loop through every cell and check the cell in + nx, + ny, or + nz direction
            if np.any(self.connection_list[i, :, 0] >= 0):
                n_constraints += 1
        # print('Number of constraints = ', n_constraints, ', 6N = ', 6 * self.N_grid)

            if np.any(self.connection_list[i, :, 2] >= 0):
                n_constraints += 1
        # print('Number of constraints = ', n_constraints, ', 6N = ', 6 * self.N_grid)

            if np.any(self.connection_list[i, :, 4] >= 0):
                n_constraints += 1

        # print('Number of constraints = ', n_constraints, ', 6N = ', 6 * self.N_grid)
            # Loop through cells and check if does not have a neighboring cell in the - nx, - ny, or - nz direction 
            if np.all(self.connection_list[i, :, 1] < 0):
                n_constraints += 1
            if np.all(self.connection_list[i, :, 3] < 0):
                n_constraints += 1
            if np.all(self.connection_list[i, :, 5] < 0):
                n_constraints += 1
        # print('Number of constraints = ', n_constraints, ', 6N = ', 6 * self.N_grid)

            # Loop through cells and check if does not have a neighboring cell in the +x, +y, or +z direction 
            if np.all(self.connection_list[i, :, 0] < 0):
                n_constraints += 1
            if np.all(self.connection_list[i, :, 2] < 0):
                n_constraints += 1
            if np.all(self.connection_list[i, :, 4] < 0):
                n_constraints += 1

        print('Number of constraints = ', n_constraints, ', 6N = ', 6 * self.N_grid)

        num_basis = self.n_functions
        N = self.N_grid * num_basis

        if self.sparse_constraint_matrix:
            flux_constraint_matrix = lil_matrix((n_constraints, N), dtype="double")
        else:
            flux_constraint_matrix = np.zeros((n_constraints, N))

        i_constraint = 0
        q = 0
        qq = 0
        print('Ngrid = ', self.N_grid)
        for i in range(self.N_grid):        
            ind = np.ravel(np.where(self.connection_list[i, :, 0] >= 0))
            k_ind = self.connection_list[i, ind, 0]
            ind = np.ravel(np.where(self.connection_list[i, :, 1] >= 0))
            k_ind = self.connection_list[i, ind, 1]
            ind = np.ravel(np.where(self.connection_list[i, :, 2] >= 0))
            k_ind = self.connection_list[i, ind, 2]
            ind = np.ravel(np.where(self.connection_list[i, :, 3] >= 0))
            k_ind = self.connection_list[i, ind, 3]
            ind = np.ravel(np.where(self.connection_list[i, :, 4] >= 0))
            k_ind = self.connection_list[i, ind, 4]
            ind = np.ravel(np.where(self.connection_list[i, :, 5] >= 0))
            k_ind = self.connection_list[i, ind, 5]
            # Loop through every cell and check the cell in + nx, + ny, or + nz direction
            if np.any(self.connection_list[i, :, 0] >= 0):
                ind = np.ravel(np.where(self.connection_list[i, :, 0] >= 0))[0]
                k_ind = self.connection_list[i, ind, 0]
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[0, i, :]
                flux_constraint_matrix[i_constraint, 
                                       k_ind * num_basis:(k_ind + 1) * num_basis
                                       ] = flux_factor[1, k_ind, :]
                i_constraint += 1
            if np.any(self.connection_list[i, :, 2] >= 0):
                ind = np.ravel(np.where(self.connection_list[i, :, 2] >= 0))[0]
                k_ind = self.connection_list[i, ind, 2]
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[2, i, :]
                flux_constraint_matrix[i_constraint, 
                                       k_ind * num_basis:(k_ind + 1) * num_basis
                                       ] = flux_factor[3, k_ind, :]
                i_constraint += 1
            if np.any(self.connection_list[i, :, 4] >= 0):
                ind = np.ravel(np.where(self.connection_list[i, :, 4] >= 0))[0]
                k_ind = self.connection_list[i, ind, 4]
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[4, i, :]
                flux_constraint_matrix[i_constraint, 
                                       k_ind * num_basis:(k_ind + 1) * num_basis
                                       ] = flux_factor[5, k_ind, :]
                i_constraint += 1

            # Loop through cells and check if does not have a cell in the - nx, - ny, or - nz direction 
            # Special case for Nfp = 2, -x direction and -y direction 
            # where the half-period boundary is stitched together
            if np.all(self.connection_list[i, :, 1] < 0):
                # ind = np.ravel(np.where(self.connection_list[i, :, 0] >= 0))
                if (i in x_inds) and (self.coil_range != 'full torus'):
                    flux_constraint_matrix[i_constraint, 
                                           i * num_basis:(i + 1) * num_basis
                                           ] = flux_factor[1, i, :]
                    k_ind = z_flipped_inds_x[q]
                    flux_constraint_matrix[i_constraint, 
                                           k_ind * num_basis:(k_ind + 1) * num_basis
                                           ] = -flux_factor[1, k_ind, :]
                    q = q + 1
                else:
                    # If not at the phi = pi / 2 boundary, zero out the -x flux
                    # If it is at phi = pi / 2 boundary, but there is no 
                    # adjacent cell at both +- x, just zero out the flux again
                    flux_constraint_matrix[i_constraint, 
                                           i * num_basis:(i + 1) * num_basis
                                           ] = flux_factor[1, i, :]
                i_constraint += 1
            if np.all(self.connection_list[i, :, 3] < 0):
                # ind = np.ravel(np.where(self.connection_list[i, :, 2] >= 0))
                if (i in y_inds) and (self.coil_range != 'full torus'):
                    flux_constraint_matrix[i_constraint, 
                                           i * num_basis:(i + 1) * num_basis
                                           ] = flux_factor[3, i, :]
                    # print(i, y_inds, qq, z_flipped_inds_y)
                    k_ind = z_flipped_inds_y[qq]
                    flux_constraint_matrix[i_constraint, 
                                           k_ind * num_basis:(k_ind + 1) * num_basis
                                           ] = -flux_factor[3, k_ind, :]  
                    qq = qq + 1
                else:
                    # If not at the phi = 0 boundary, zero out the -y flux
                    # If it is at phi = 0 boundary, but there is no 
                    # adjacent cell at both +- y, just zero out the flux again
                    flux_constraint_matrix[i_constraint, 
                                           i * num_basis:(i + 1) * num_basis
                                           ] = flux_factor[3, i, :]
                i_constraint += 1
            if np.all(self.connection_list[i, :, 5] < 0):
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[5, i, :]
                i_constraint += 1
            # Loop through cells and check if does not have a cell in the + nx, + ny, or + nz direction 
            if np.all(self.connection_list[i, :, 0] < 0):
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[0, i, :]
                i_constraint += 1
            if np.all(self.connection_list[i, :, 2] < 0):
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[2, i, :]
                i_constraint += 1
            if np.all(self.connection_list[i, :, 4] < 0):
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[4, i, :]
                i_constraint += 1

        connect_list_zeros = np.copy(self.connection_list)
        connect_list_zeros[connect_list_zeros >= 0] = 1
        connect_list_zeros[self.connection_list == -1] = 0
        self.num_constraints_per_cell = np.sum(np.sum(connect_list_zeros, axis=-1), axis=-1)
        self.flux_factor = flux_factor

        t2 = time.time()
        print('Time to make the flux jump constraint matrix = ', t2 - t1, ' s')
        t1 = time.time()

        if self.sparse_constraint_matrix:
            C = flux_constraint_matrix.tocsc()
            CT = C.transpose()
            CCT_inv = np.linalg.pinv((C @ CT).todense(), rcond=1e-8, hermitian=True)
            projection_onto_constraints = sparse_eye(N, format="csc", dtype="double") - CT @ CCT_inv @ C 
        else:
            C = flux_constraint_matrix
            CT = C.transpose()
            CCT_inv = np.linalg.pinv(C @ CT, rcond=1e-8, hermitian=True)
            projection_onto_constraints = np.eye(N) - CT @ CCT_inv @ C
            #sparse_eye(N, format="csc", dtype="double") - csc_matrix(CT @ CCT_inv @ C)

        t2 = time.time()
        print('Time to make CCT_inv = ', t2 - t1, ' s')

        # S = np.linalg.svd(C.todense(), compute_uv=False)
        # plt.semilogy(S, 'ro')
        # S2 = np.linalg.svd(projection_onto_constraints.todense(), compute_uv=False)
        # plt.semilogy(S2, 'bo')
        self.C = C    
        self.P = projection_onto_constraints

    def check_fluxes(self):
        """
            Post-optimization test function to check that the fluxes between
            adjacent cells match to a reasonable tolerance (i.e. there is 
            some flux mismatch in the Amps level, while the overall fluxes
            are in the MA level). If the constraints were applied correctly
            in the optimization, this function should run no problem.
        """
        Phi = self.Phi
        n = Phi.shape[1]
        n_interp = Phi.shape[2]
        alpha_opt = self.alphas.reshape(n, self.n_functions)
        # integration_points = self.integration_points
        Nx = self.nx
        Ny = self.ny
        Nz = self.nz
        Jvec = self.J.reshape(n, Nx, Ny, Nz, 3)

        # Compute all the fluxes (unnormalized!)
        flux = np.zeros((n, 6))
        for i in range(n):
            for j in range(6):
                nx = 0
                ny = 0
                nz = 0
                if j == 0:
                    nx = 1
                    x_ind = Nx - 1
                elif j == 1:
                    nx = -1
                    x_ind = 0
                elif j == 2:
                    ny = 1
                    y_ind = Ny - 1
                elif j == 3:
                    ny = -1
                    y_ind = 0
                elif j == 4:
                    nz = 1
                    z_ind = Nz - 1
                elif j == 5:
                    nz = -1
                    z_ind = 0
                if j < 2:
                    for l in range(Ny):
                        for m in range(Nz):
                            flux[i, j] += nx * Jvec[i, x_ind, l, m, 0] + ny * Jvec[i, x_ind, l, m, 1] + nz * Jvec[i, x_ind, l, m, 2]
                elif j < 4:
                    for l in range(Nx):
                        for m in range(Nz):
                            flux[i, j] += nx * Jvec[i, l, y_ind, m, 0] + ny * Jvec[i, l, y_ind, m, 1] + nz * Jvec[i, l, y_ind, m, 2]
                else:
                    for l in range(Nx):
                        for m in range(Ny):
                            flux[i, j] += nx * Jvec[i, l, m, z_ind, 0] + ny * Jvec[i, l, m, z_ind, 1] + nz * Jvec[i, l, m, z_ind, 2]

        # Compare fluxes across adjacent cells
        flux_max = 5000  # np.max(abs(flux)) / 1e4
        q = 0
        qq = 0
        for i in range(n):
            for j in range(6):
                #print(i, j, flux[i, j])
                if (i in self.x_inds) and (j == 1) and (self.coil_range != 'full torus'):
                    k_ind = self.z_flip_x[q]
                    assert np.isclose(flux[i, 1], flux[k_ind, 1], atol=flux_max, rtol=1e-3)
                    q += 1
                elif (i in self.y_inds) and (j == 3) and (self.coil_range != 'full torus'):
                    k_ind = self.z_flip_y[qq]
                    assert np.isclose(flux[i, 3], flux[k_ind, 3], atol=flux_max, rtol=1e-3)
                    qq += 1
                elif np.any(self.connection_list[i, :, j] >= 0):
                    inds = np.ravel(np.where(self.connection_list[i, :, j] >= 0))
                    for ind in inds:
                        k_ind = self.connection_list[i, ind, j]
                        if (j % 2) == 0:
                            j_ind = j + 1
                        else:
                            j_ind = j - 1
                        if not np.isclose(flux[i, j], -flux[k_ind, j_ind], atol=flux_max, rtol=1e-1):
                            print(i, j, k_ind, j_ind, ind, flux[i, j], flux[k_ind, j_ind])
                        assert np.isclose(flux[i, j], -flux[k_ind, j_ind], atol=flux_max, rtol=1e-1)
                else:
                    if not np.isclose(flux[i, j], 0.0, atol=flux_max):
                        print(i, j, flux[i, j])
                    assert np.isclose(flux[i, j], 0.0, atol=flux_max)

