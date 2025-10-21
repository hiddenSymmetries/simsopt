import numpy as np
from simsopt.geo import Surface
from simsopt.solve.current_voxels_optimization import compute_J
from simsopt.util.permanent_magnet_helper_functions import make_curve_at_theta0 
from pyevtk.hl import pointsToVTK, unstructuredGridToVTK
from pyevtk.vtk import VtkVoxel

import simsoptpp as sopp
import time
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

__all__ = ['CurrentVoxelsGrid']

contig = np.ascontiguousarray


def _voxels_to_vtk(
    filename,
    points,
    cellData=None,
    pointData=None,
    fieldData=None,
):
    """
    Write a VTK file showing the individual voxel cubes.

    Args:
    -------
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
    # Some references Matt L. used while writing this function:
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


class CurrentVoxelsGrid:
    r"""
    ``CurrentVoxelsGrid`` is a class for setting up the grid,
    needed to perform finite-build coil optimization for stellarators
    using the method outlined in Kaptanoglu & Langlois & Landreman 2023. This 
    class reuses some of the functionality used for the PermanentMagnetGrid
    class. It takes as input two toroidal surfaces specified as SurfaceRZFourier
    objects, and initializes a set of points 
    between these surfaces.

    Args:
    -----------
    plasma_boundary: Surface class object 
        Representing the plasma boundary surface. Gets converted
        into SurfaceRZFourier object for ease of use.
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
        Bn: 2D numpy array, shape (ntheta_quadpoints, nphi_quadpoints)
            Magnetic field (coils and plasma) at the plasma
            boundary. Typically this will be the optimized plasma
            magnetic field from a stage-1 optimization, and the
            optimized coils from a basic stage-2 optimization.
            Defaults to all zeros.
        Bn_Itarget: 1D numpy array, shape (ngamma_quadpoints)
            Magnetic field (coils and plasma) along the toroidal
            curve (gamma) used for avoiding the trivial solution.
            Defaults to all zeros.
        Itarget: float 
            Target current I in mega-amperes to flow through the toroidal
            curve (gamma) used for avoiding the trivial solution. Defaults
            to 0.5 MA, which is approximately suitable for a 1-meter device
            with average B = 0.1 Tesla on axis. 
        Itarget_curve: Curve class object
            Initialized Curve object representing the toroidal
            curve (gamma) used for avoiding the trivial solution. Defaults
            to using the plasma boundary at theta = 0.
        Nx: Number of X points in current voxels grid. Defaults to 10. 
        Ny: Number of Y points in current voxels grid. Defaults to 10.
        Ny: Number of Z points in current voxels grid. Defaults to 10.
        nx: Number of X points in each individual voxel (for integrating Biot Savart). Defaults to 6.
        ny: Number of Y points in each individual voxel (for integrating Biot Savart). Defaults to 6. 
        nz: Number of Z points in each individual voxel (for integrating Biot Savart). Defaults to 6. 
    """

    def __init__(
        self, plasma_boundary: Surface,
        inner_toroidal_surface: Surface, 
        outer_toroidal_surface: Surface,
        **kwargs,
    ):
        self.plasma_boundary = plasma_boundary.to_RZFourier()
        self.inner_toroidal_surface = inner_toroidal_surface.to_RZFourier()
        self.outer_toroidal_surface = outer_toroidal_surface.to_RZFourier()
        Nx = kwargs.pop("Nx", 10)
        Ny = kwargs.pop("Ny", 10)
        Nz = kwargs.pop("Nz", 10)
        nx = kwargs.pop("nx", 6)
        ny = kwargs.pop("ny", 6)
        nz = kwargs.pop("nz", 6)
        if Nx <= 0 or Ny <= 0 or Nz <= 0 or nx <= 0 or ny <= 0 or nz <= 0:
            raise ValueError('Nx, Ny, Nz, nx, ny, nz should be positive integers')

        Bn = kwargs.pop("Bn", np.zeros(self.plasma_boundary.gamma().shape[:2]))
        Bn = np.array(Bn)
        if (len(Bn.shape) != 2) or (Bn.shape != self.plasma_boundary.gamma().shape[:2]): 
            raise ValueError('Normal magnetic field surface data is incorrect shape.')
        self.Bn = Bn
        self.Itarget = kwargs.pop("Itarget", 0.5e6)
        self.Itarget_curve = kwargs.pop("Itarget_curve", None)
        if self.Itarget_curve is None:
            numquadpoints = len(self.plasma_boundary.quadpoints_phi)  # * plasma_boundary.nfp
            self.Itarget_curve = make_curve_at_theta0(self.plasma_boundary, numquadpoints)

        Bn_Itarget = kwargs.pop("Bn_Itarget", np.zeros(self.Itarget_curve.gamma().shape[0]))
        Bn_Itarget = np.array(Bn_Itarget)
        if (len(Bn_Itarget.shape) != 1) or (Bn_Itarget.shape[0] != self.Itarget_curve.gamma().shape[0]): 
            raise ValueError('Normal magnetic field curve data is incorrect shape.')
        self.Bn_Itarget = Bn_Itarget
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.n_functions = 5  # hard-coded for linear basis

        if self.plasma_boundary.nfp == 2 or self.plasma_boundary.nfp == 4:
            if self.plasma_boundary.stellsym:
                self.coil_range = 'half period'
            else:
                self.coil_range = 'full period'
        else:
            self.coil_range = 'full torus'
        self.phi = self.plasma_boundary.quadpoints_phi
        self.nphi = len(self.phi)
        self.theta = self.plasma_boundary.quadpoints_theta
        self.ntheta = len(self.theta)

        t1 = time.time()
        self._setup_uniform_grid()
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to setup the uniform grid.")

        # Have the uniform grid, now need to loop through and eliminate cells.
        t1 = time.time()
        final_grid = sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            contig(self.normal_inner.reshape(-1, 3)), contig(self.normal_outer.reshape(-1, 3)),
            contig(self.XYZ_uniform), contig(self.xyz_inner), contig(self.xyz_outer),
        )
        # remove all the grid elements that were omitted
        inds = np.ravel(np.logical_not(np.all(final_grid == 0.0, axis=-1)))
        self.XYZ_flat = final_grid[inds, :]
        t2 = time.time()
        print("Took t = ", t2 - t1, " s to perform the total grid cell elimination process.")

        self.N_grid = self.XYZ_flat.shape[0]
        self.alphas = np.random.rand(self.n_functions * self.N_grid)

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
        # Get (X, Y, Z) coordinates of the inner and outer toroidal boundaries
        xyz_inner = self.inner_toroidal_surface.gamma()
        self.xyz_inner = xyz_inner.reshape(-1, 3)
        self.x_inner = xyz_inner[:, :, 0]
        self.y_inner = xyz_inner[:, :, 1]
        self.z_inner = xyz_inner[:, :, 2]
        xyz_outer = self.outer_toroidal_surface.gamma()
        self.xyz_outer = xyz_outer.reshape(-1, 3)
        self.x_outer = xyz_outer[:, :, 0]
        self.y_outer = xyz_outer[:, :, 1]
        self.z_outer = xyz_outer[:, :, 2]

        # get normal vectors of inner and outer PM surfaces
        self.normal_inner = np.copy(self.inner_toroidal_surface.unitnormal())
        self.normal_outer = np.copy(self.outer_toroidal_surface.unitnormal())

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

    def to_vtk_before_solve(self, vtkname, plot_quadrature_points=False):
        """
        Write voxel geometry data into a VTK file.

        Args:
        -------
        vtkname (str): VTK filename, will be appended with .vts or .vtu.
        dim (tuple, optional): Dimension information if saved as structured grids. Defaults to (1).
        plot_quadrature_points (bool, optional): Boolean flag to plot the full quadrature grid. 
          Defaults to False because this can make enormous vtk files. 
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

        self.XYZ_integration_full = XYZ_integration_full
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
        }
        if plot_quadrature_points:
            pointsToVTK(
                vtkname + '_quadrature',
                contig(self.XYZ_integration[:, :, 0].flatten()),
                contig(self.XYZ_integration[:, :, 1].flatten()),
                contig(self.XYZ_integration[:, :, 2].flatten()),
                data=data
            )
            pointsToVTK(
                vtkname + '_quadrature_full',
                contig(XYZ_integration_full[:, :, 0].flatten()),
                contig(XYZ_integration_full[:, :, 1].flatten()),
                contig(XYZ_integration_full[:, :, 2].flatten()),
            )
        _voxels_to_vtk(vtkname + '_voxels', self.XYZ_integration)
        _voxels_to_vtk(vtkname + '_voxels_full', XYZ_integration_full)
        print("Max quadrature x:", np.max(self.XYZ_integration[:, :, 0]))
        print("Min quadrature x:", np.min(self.XYZ_integration[:, :, 0]))
        print("Max quadrature z:", np.max(self.XYZ_integration[:, :, 2]))
        print("Min quadrature z:", np.min(self.XYZ_integration[:, :, 2]))

    def to_vtk_after_solve(self, vtkname, plot_quadrature_points=False):
        """
        Write dipole data into a VTK file (stolen from Caoxiang's CoilPy code).

        Args:
        -------
        vtkname (str): VTK filename, will be appended with .vts or .vtu.
        dim (tuple, optional): Dimension information if saved as structured grids. Defaults to (1).
        plot_quadrature_points (bool, optional): Boolean flag to plot the full quadrature grid. 
          Defaults to False because this can make enormous vtk files.
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

        # Phi = self.Phi
        # n_interp = Phi.shape[2]
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
        print('divJ max = ', np.max(np.abs(divJ)), divJ.shape)

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

        Jvec = Jvec_full_internal.reshape(n * nsym, self.nx, self.ny, self.nz, 3)
        dJx_dx = -(Jvec[:, 1:, :, :, 0] - Jvec[:, :-1, :, :, 0]) / self.dx
        dJy_dy = -(Jvec[:, :, 1:, :, 1] - Jvec[:, :, :-1, :, 1]) / self.dy
        dJz_dz = -(Jvec[:, :, :, 1:, 2] - Jvec[:, :, :, :-1, 2]) / self.dz
        divJ_total = dJx_dx[:, :, :-1, :-1] + dJy_dy[:, :-1, :, :-1] + dJz_dz[:, :-1, :-1, :]
        divJ_total = np.sum(np.sum(np.sum(divJ_total, axis=1), axis=1), axis=1)

        data = {"J": (contig(Jx), contig(Jy), contig(Jz)), 
                "J_normalized": 
                (contig(Jx / Jvec_normalization), 
                 contig(Jy / Jvec_normalization), 
                 contig(Jz / Jvec_normalization)),
                "J_sparse": (contig(Jx_sp), contig(Jy_sp), contig(Jz_sp)), 
                "Jvec_xmin": (contig(Jvec_xmin[:, 0]), contig(Jvec_xmin[:, 1]), contig(Jvec_xmin[:, 2])), 
                "Jvec_xmax": (contig(Jvec_xmax[:, 0]), contig(Jvec_xmax[:, 1]), contig(Jvec_xmax[:, 2])), 
                "Jvec_ymin": (contig(Jvec_ymin[:, 0]), contig(Jvec_ymin[:, 1]), contig(Jvec_ymin[:, 2])), 
                "Jvec_ymax": (contig(Jvec_ymax[:, 0]), contig(Jvec_ymax[:, 1]), contig(Jvec_ymax[:, 2])), 
                "Jvec_zmin": (contig(Jvec_zmin[:, 0]), contig(Jvec_zmin[:, 1]), contig(Jvec_zmin[:, 2])), 
                "Jvec_zmax": (contig(Jvec_zmax[:, 0]), contig(Jvec_zmax[:, 1]), contig(Jvec_zmax[:, 2])), 
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
        Jvec_full_internal = np.reshape(Jvec_full_internal, (n * nsym, self.nx * self.ny * self.nz, 3))
        if plot_quadrature_points: 
            data = {
                "Jvec": (contig(Jvec_full_internal[:, :, 0].flatten()), 
                         contig(Jvec_full_internal[:, :, 1].flatten()),
                         contig(Jvec_full_internal[:, :, 2].flatten())
                         )
            }
            pointsToVTK(
                vtkname + '_quadrature_full_solution',
                contig(self.XYZ_integration_full[:, :, 0].flatten()),
                contig(self.XYZ_integration_full[:, :, 1].flatten()),
                contig(self.XYZ_integration_full[:, :, 2].flatten()),
                data=data
            )

        # Make a rough quiver plot for immediate user viewing of the solution
        fig = plt.figure(700, figsize=(12, 12))
        ax = fig.add_subplot(projection='3d')
        colors = np.linalg.norm(Jvec_full_sp, axis=-1)
        inds = (colors > 1)
        colors = np.concatenate((colors, np.repeat(colors, 2))) 
        q = ax.quiver(ox_full[inds], oy_full[inds], oz_full[inds], 
                      Jvec_full_sp[inds, 0], Jvec_full_sp[inds, 1], Jvec_full_sp[inds, 2], 
                      length=0.4, normalize=True, cmap='Reds', norm=LogNorm(vmin=1, vmax=np.max(colors)))
        inds = (colors > 1)
        q.set_array(colors[inds])
        fig.colorbar(q)

    def _setup_polynomial_basis(self):
        """
        Evaluate the basis of divergence-free polynomials
        at a given set of points. For now,
        the basis is hard-coded as a set of linear polynomials.
        The basis, Phi, is shape (num_basis_functions, N_grid, nx * ny * nz, 3)
        where N_grid is the number of cells in the current voxels,
        nx, ny, and nz, are the number of points to evaluate the 
        function WITHIN a cell, and num_basis_functions is hard-coded
        to 5 for now while we use a linear basis of polynomials that conserve
        local flux jumps across cell interfaces. 
        """
        dx = self.dx
        dy = self.dy
        dz = self.dz
        nx = self.nx
        ny = self.ny
        nz = self.nz
        n = self.N_grid

        # Define a uniform grid, with xrange, yrange, zrange
        # defined as the local coordinates in each grid (i.e.
        # the coordinates - the cell midpoint)
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
        zeros = np.zeros((n, nx, ny, nz))
        ones = np.ones((n, nx, ny, nz))
        # Shift all the basis functions by their midpoints in every cell
        Phi[0, :, :, :, :, :] = np.transpose(np.array([ones, zeros, zeros]), axes=[1, 2, 3, 4, 0])
        Phi[1, :, :, :, :, :] = np.transpose(np.array([zeros, ones, zeros]), axes=[1, 2, 3, 4, 0])
        Phi[2, :, :, :, :, :] = np.transpose(np.array([zeros, zeros, ones]), axes=[1, 2, 3, 4, 0])
        xrange_extend = (xrange - x_leftpoints[:, np.newaxis])[:, :, np.newaxis, np.newaxis] * ones
        yrange_extend = (yrange - y_leftpoints[:, np.newaxis])[:, np.newaxis, :, np.newaxis] * ones
        zrange_extend = (zrange - z_leftpoints[:, np.newaxis])[:, np.newaxis, np.newaxis, :] * ones
        Phi[3, :, :, :, :, :] = np.transpose(np.array([xrange_extend, -yrange_extend, zeros]), axes=[1, 2, 3, 4, 0]) / np.cbrt(dx * dy * dz)
        Phi[4, :, :, :, :, :] = np.transpose(np.array([xrange_extend, zeros, -zrange_extend]), axes=[1, 2, 3, 4, 0]) / np.cbrt(dx * dy * dz)

        # Flatten the indices corresponding to the integration points in each voxel
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
        """
        Main geometrical setup function. The grid of voxels is now defined so this 
        function goes through and calculations the various inductance and optimization
        related matrices we need. Calculates the hardest part of this problem -- 
        getting the constraint matrix correct with the symmetries.
        """

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
            # nsym = nfp
        elif stellsym:
            stell_list = [1, -1]
            # nsym = nfp * 2

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
        # n_interp = Phi.shape[2]
        nphi_loop_inv = 1.0 / len(self.Itarget_curve.quadpoints)
        curve_dl = self.Itarget_curve.gammadash().reshape(-1, 3)

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
                ox_full = ox * np.cos(phi0) - oy * np.sin(phi0) * stell
                oy_full = ox * np.sin(phi0) + oy * np.cos(phi0) * stell
                oz_full = oz * stell

                # get new dipole vectors by flipping the x component, then rotating by phi0
                Phivec_x = Phi[:, :, :, 0] * np.cos(phi0) * stell - Phi[:, :, :, 1] * np.sin(phi0)
                Phivec_y = Phi[:, :, :, 0] * np.sin(phi0) * stell + Phi[:, :, :, 1] * np.cos(phi0)
                Phivec_z = Phi[:, :, :, 2]

                int_points = np.transpose(np.array([ox_full, oy_full, oz_full]), [1, 2, 0])
                Phivec_transpose = np.transpose(np.array([Phivec_x, Phivec_y, Phivec_z]), [2, 3, 1, 0])

                geo_factor += sopp.current_voxels_field_Bext(
                    points, 
                    contig(int_points), 
                    contig(Phivec_transpose),
                    plasma_unitnormal
                )
                Itarget_matrix += sopp.current_voxels_field_Bext(
                    points_curve, 
                    contig(int_points), 
                    contig(Phivec_transpose),
                    contig(curve_dl)
                )
        t2 = time.time()
        print('Computing full coil surface grid took t = ', t2 - t1, ' s')

        t1 = time.time()
        alphas = self.alphas.reshape(n, self.n_functions)
        compute_J(self, alphas, alphas)
        t2 = time.time()
        print('Computing J took t = ', t2 - t1, ' s')
        t1 = time.time()

        # Delta x from intra-cell integration is self.dx (uniform grid spacing) divided 
        # by self.nx (number of points within a cell)
        coil_integration_factor = self.dx * self.dy * self.dz / (self.nx * self.ny * self.nz)
        self.grid_scaling = coil_integration_factor
        self.B_matrix = (geo_factor * np.sqrt(N_quadrature_inv) * coil_integration_factor).reshape(
            geo_factor.shape[0], self.N_grid * self.n_functions
        )
        b_rhs = np.ravel(self.Bn * np.sqrt(N_quadrature_inv))
        for i in range(self.B_matrix.shape[0]):
            self.B_matrix[i, :] *= np.sqrt(normN[i])
            b_rhs[i] *= np.sqrt(normN[i])

        self.b_rhs = -b_rhs  # minus sign because ||B_{coil,N} + B_{0,N}||_2^2 -> ||A * alpha - b||_2^2

        self.Itarget_matrix = nphi_loop_inv * coil_integration_factor * Itarget_matrix.reshape(
            points_curve.shape[0], n * num_basis
        )
        self.Itarget_matrix = np.sum(self.Itarget_matrix, axis=0)
        self.Itarget_rhs = self.Bn_Itarget * nphi_loop_inv
        self.Itarget_rhs = np.sum(self.Itarget_rhs) + 4 * np.pi * 1e-7 * self.Itarget
        flux_factor, self.connection_list = sopp.current_voxels_flux_jumps(
            coil_points,
            contig(self.Phi.reshape(self.n_functions, self.N_grid, self.nx, self.ny, self.nz, 3)), 
            self.dx,
            self.dy, 
            self.dz
        )
        t2 = time.time()
        print('Computing the flux factors took t = ', t2 - t1, ' s')

        # need to normalize flux_factor by 1/nx ** 2 (assuming uniform grid!)
        flux_factor *= 1 / (self.nx ** 2)
        t1 = time.time()

        # Find the coil boundary points at phi = pi / 2
        minus_x_indices = []
        if nfp == 2:
            minus_x_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 1] < 0, axis=-1)))
            x0_indices = np.ravel(np.where(np.isclose(coil_points[:, 0], 0.0, atol=self.dx)))
            minus_x_indices = np.intersect1d(minus_x_indices, x0_indices)
        # Find the coil boundary points at phi = pi / 4
        elif nfp == 4:
            minus_x_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 1] < 0, axis=-1)))
            plus_y_indices = np.ravel(np.where(np.all(self.connection_list[:, :, 2] < 0, axis=-1)))
            minus_x_indices = np.intersect1d(minus_x_indices, plus_y_indices)

        z_flipped_inds_x = []
        x_inds = []
        # The points to stitch on the "left" are rotated then flipped
        for x_ind in minus_x_indices:
            stell = -1
            ox = coil_points[x_ind, 0]
            oy = coil_points[x_ind, 1]
            oz = coil_points[x_ind, 2] * stell
            z_flipped_point = np.array([ox, oy, oz]).T
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

            if np.any(self.connection_list[i, :, 2] >= 0):
                n_constraints += 1

            if np.any(self.connection_list[i, :, 4] >= 0):
                n_constraints += 1

            # Loop through cells and check if does not have a neighboring cell in the - nx, - ny, or - nz direction 
            if np.all(self.connection_list[i, :, 1] < 0):
                n_constraints += 1
            if np.all(self.connection_list[i, :, 3] < 0):
                n_constraints += 1
            if np.all(self.connection_list[i, :, 5] < 0):
                n_constraints += 1

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

        flux_constraint_matrix = lil_matrix((n_constraints, N), dtype="double")
        i_constraint = 0
        q = 0
        qq = 0
        qqq = 0
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
            # Special case for Nfp = 4, +y direction
            # where the half-period boundary is stitched together
            if np.all(self.connection_list[i, :, 0] < 0):
                flux_constraint_matrix[i_constraint, 
                                       i * num_basis:(i + 1) * num_basis
                                       ] = flux_factor[0, i, :]
                i_constraint += 1
            if np.all(self.connection_list[i, :, 2] < 0):
                if (i in x_inds) and (self.coil_range != 'full torus') and nfp != 2:
                    flux_constraint_matrix[i_constraint, 
                                           i * num_basis:(i + 1) * num_basis
                                           ] = flux_factor[2, i, :]
                    k_ind = z_flipped_inds_x[qqq]
                    flux_constraint_matrix[i_constraint, 
                                           k_ind * num_basis:(k_ind + 1) * num_basis
                                           ] = -flux_factor[2, k_ind, :]
                    qqq = qqq + 1
                else:
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
        self.C = flux_constraint_matrix.tocsc()

        t2 = time.time()
        print('Time to make the flux jump constraint matrix = ', t2 - t1, ' s')

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
        # n_interp = Phi.shape[2]
        # alpha_opt = self.alphas.reshape(n, self.n_functions)
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
        flux_max = 5000  # flag any disagreements in the few kilo-Amperes range
        q = 0
        qq = 0
        for i in range(n):
            for j in range(6):
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

